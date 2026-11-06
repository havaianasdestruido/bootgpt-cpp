"""Check reference arithmetic against the bundled real-mode inference code."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

try:
    import numpy as np
    import unicorn
    from unicorn import x86_const as x86
except ImportError:
    np = unicorn = None

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(np is not None and unicorn and shutil.which('nasm'),
                     'requires numpy, unicorn and nasm')
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('evaluation', ROOT / 'tools/evaluate_generation.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.model = module.Model()
        with tempfile.TemporaryDirectory() as tmp:
            boot, listing = Path(tmp) / 'boot.bin', Path(tmp) / 'boot.lst'
            subprocess.run(['nasm', '-f', 'bin', '-l', str(listing), 'boot.asm', '-o', str(boot)],
                           cwd=ROOT, check=True)
            assert boot.read_bytes() == (ROOT / 'model/boot.bin').read_bytes()
            lines = listing.read_text().splitlines()
            idx = next(i for i, line in enumerate(lines) if re.search(r'\binfer:$', line))
            entry = next(re.search(r'^\s*\d+\s+([0-9A-F]{8})\s+', line)
                         for line in lines[idx + 1:]
                         if re.search(r'^\s*\d+\s+([0-9A-F]{8})\s+', line))
            cls.entry = 0x7c00 + int(entry.group(1), 16)
            idx = next(i for i, line in enumerate(lines) if re.search(r'\bmainloop:$', line))
            entry = next(re.search(r'^\s*\d+\s+([0-9A-F]{8})\s+', line)
                         for line in lines[idx + 1:]
                         if re.search(r'^\s*\d+\s+([0-9A-F]{8})\s+', line))
            cls.mainloop = 0x7c00 + int(entry.group(1), 16)

    def test_full_boot_loop_updates_context(self):
        m = self.model
        cpu = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_16)
        cpu.mem_map(0, 0x20000)
        image = (ROOT / 'model/disk.img').read_bytes()
        cpu.mem_write(0x7c00, image[:512])
        cpu.reg_write(x86.UC_X86_REG_DL, 0x80)
        seed = 0xcafe
        ctx = list(m.p['seed_ctx'])
        rnd = seed | 1  # the normal (non-DBGPORT) boot path forces an odd seed
        steps = 0

        def interrupt(uc, no, _):
            if no == 0x13:
                uc.mem_write(0x8000, image[512:])
                uc.reg_write(x86.UC_X86_REG_EFLAGS, uc.reg_read(x86.UC_X86_REG_EFLAGS) & ~1)
            elif no == 0x1a:
                uc.reg_write(x86.UC_X86_REG_DX, seed)
            elif no == 0x16:
                uc.reg_write(x86.UC_X86_REG_EFLAGS, uc.reg_read(x86.UC_X86_REG_EFLAGS) | 0x40)
            else:
                self.assertIn(no, (0x10, 0x15))

        def instruction(uc, address, size, _):
            nonlocal ctx, rnd, steps
            if address != self.mainloop:
                return
            self.assertEqual(bytes(uc.mem_read(0x7600, m.p['C'])), bytes(ctx))
            self.assertEqual(int.from_bytes(uc.mem_read(0x7b80, 2), 'little'), rnd)
            if steps == 270:
                uc.emu_stop()
                return
            token = m.generate(ctx, rnd, 1)
            ctx = ctx[1:] + [m.vocab.index(token)]
            for _ in range(m.p['V']):
                rnd = (rnd * 25173 + 13849) & 0xffff
            steps += 1

        cpu.hook_add(unicorn.UC_HOOK_INTR, interrupt)
        cpu.hook_add(unicorn.UC_HOOK_CODE, instruction, begin=self.mainloop, end=self.mainloop)
        cpu.emu_start(0x7c00, 0, count=100000000)
        self.assertEqual(steps, 270)

    def test_logits_and_sample_match_real_instructions(self):
        m = self.model
        contexts = [m.p['seed_ctx'], m.context(' ' * 32),
                    m.context('int sum(int n) {\n    int s = 0;\n'),
                    m.context('int f(int x) {\n    if (x > 0) {\n')]
        for ctx in contexts:
            for seed in (1, 0x1234, 0xcafe):
                with self.subTest(ctx=ctx, seed=seed):
                    cpu = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_16)
                    cpu.mem_map(0, 0x20000)
                    cpu.mem_write(0x7c00, (ROOT / 'model/boot.bin').read_bytes())
                    cpu.mem_write(0x8000, (ROOT / 'model/weights.bin').read_bytes())
                    cpu.mem_write(0x7600, bytes(ctx))
                    cpu.mem_write(0x7b80, seed.to_bytes(2, 'little'))
                    cpu.reg_write(x86.UC_X86_REG_FS, 0x0800)
                    cpu.reg_write(x86.UC_X86_REG_SP, 0x7000)
                    cpu.mem_write(0x7000, b'\x00\x20')
                    cpu.emu_start(self.entry, 0x2000, count=1000000)
                    self.assertEqual(cpu.reg_read(x86.UC_X86_REG_IP), 0x2000)
                    _, hidden, logits = m.forward(ctx)
                    actual_hidden = np.frombuffer(bytes(cpu.mem_read(0x7700, m.p['H'])), dtype='i1')
                    actual_logits = np.frombuffer(bytes(cpu.mem_read(0x7800, m.p['V'] * 2)), dtype='<i2')
                    np.testing.assert_array_equal(actual_hidden, hidden)
                    np.testing.assert_array_equal(actual_logits, logits)
                    choice = cpu.reg_read(x86.UC_X86_REG_BX)
                    self.assertEqual(m.vocab[choice], m.generate(ctx, seed, 1))
