"""Regression coverage for softmax sampling and finite-window space loops."""
import importlib.util
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from sampling import lcg, sample, weights_for

try:
    import numpy
    import unicorn
    from unicorn import x86_const as x86
except ImportError:
    numpy = unicorn = None

ROOT = Path(__file__).resolve().parent.parent


class SamplingTests(unittest.TestCase):
    def test_quantized_weights_approximate_softmax(self):
        rng = random.Random(14)
        for _ in range(200):
            logits = [rng.randint(-3000, 3000) for _ in range(95)]
            weights = weights_for(logits)
            maximum = max(logits)
            exact = [math.exp((x - maximum) / 256) for x in logits]
            total, exact_total = sum(weights), sum(exact)
            tv = sum(abs(w / total - p / exact_total) for w, p in zip(weights, exact)) / 2
            self.assertLess(tv, 0.012)

    def test_rng_range_reduction_is_reproducible_and_never_out_of_bounds(self):
        for logits in ([32767, -32768], [-32768] * 256, [0] * 256,
                       [0, -240, -1233, -2500]):
            for seed in (0, 1, 0x1234, 0xffff):
                token, rnd = sample(logits, seed)
                self.assertEqual((token, rnd), sample(logits, seed))
                self.assertGreaterEqual(token, 0)
                self.assertLess(token, len(logits))
                first = lcg(seed)
                expected = first
                for _ in range(1 if first >> 8 < 64 else len(logits)):
                    expected = lcg(expected)
                self.assertEqual(rnd, expected)


@unittest.skipUnless(numpy is not None, 'requires numpy')
class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('evaluation', ROOT / 'tools/evaluate_generation.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.module = module
        cls.model = module.Model()

    def test_space_context_has_real_alternatives(self):
        m = self.model
        logits = m.forward(m.context(' ' * m.p['C']))[2]
        w = weights_for(logits)
        space = m.vocab.index(' ')
        self.assertGreater(sum(w) - w[space], 0)
        self.assertGreater((sum(w) - w[space]) / sum(w), .07)
        self.assertLess((sum(w) - w[space]) / sum(w), .10)
        # The original 240-point gate allowed exactly one token here.
        self.assertEqual(sum(int(x >= max(logits) - 240) for x in logits), 1)

    def test_all_space_state_escapes_for_every_rng_state(self):
        m = self.model
        logits = m.forward(m.context(' ' * m.p['C']))[2]
        weights = weights_for(logits)
        space = m.vocab.index(' ')
        self.assertEqual(sum(int(x >= max(logits) - 240) for x in logits), 1)
        self.assertEqual(int(logits.argmax()), space)
        prefix = sum(weights[:space])
        total = sum(weights)
        # The legacy branch always chooses space here and consumes V draws.
        # Precompute the V-fold LCG transition to inspect all 65,536 states.
        advance = list(range(65536))
        for _ in range(m.p['V']):
            advance = [lcg(x) for x in advance]
        next_state = [-1] * 65536
        for state in range(65536):
            branch = lcg(state)
            if branch >> 8 < 64:
                rnd = lcg(branch)
                threshold = (rnd * total) >> 16
                if prefix <= threshold < prefix + weights[space]:
                    next_state[state] = rnd
            else:
                next_state[state] = advance[branch]
        marks = [0] * 65536
        for start in range(65536):
            if marks[start]:
                continue
            path = []
            state = start
            while state != -1 and marks[state] == 0:
                marks[state] = 1
                path.append(state)
                state = next_state[state]
            self.assertTrue(state == -1 or marks[state] == 2,
                            'an all-space cycle survives for some RNG state')
            for state in path:
                marks[state] = 2

    def test_fixed_prompt_suite_improves_repetition(self):
        m = self.model
        cases = [(m.p['seed_ctx'], 0x1235),
                 (m.context('int add(int a, int b) { return '), 1),
                 (m.context('int sum(int n) {\n    int s = 0;\n'), 2),
                 (m.p['seed_ctx'], 0xcaff),
                 (m.context('int f(int x) {\n    if (x > 0) {\n'), 3)]
        for ctx, seed in cases:
            with self.subTest(seed=seed, ctx=ctx):
                old = m.generate(ctx, seed, 1500, legacy=True)
                new = m.generate(ctx, seed, 1500)
                self.assertEqual(new, m.generate(ctx, seed, 1500))
                self.assertEqual(len(new), 1500)
                self.assertTrue(set(new) <= set(m.vocab))
                before, after = self.module.metrics(old), self.module.metrics(new)
                self.assertLess(after['longest_space_run'], before['longest_space_run'])
                self.assertLess(after['repeated_4gram_fraction'], before['repeated_4gram_fraction'])
                self.assertLess(after['longest_space_run'], 200)
                self.assertGreater(after['newlines'], 0)


@unittest.skipUnless(unicorn and shutil.which('nasm'), 'requires unicorn and nasm')
class SamplerInstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'syntax.bin'
            subprocess.run(['nasm', '-f', 'bin', 'syntax.asm', '-o', str(output)], cwd=ROOT, check=True)
            cls.code = output.read_bytes()

    def test_assembly_matches_reference_at_int16_extremes(self):
        rng = random.Random(47)
        cases = [[32767, -32768], [-32768] * 256, [0] * 256,
                 [0, -240, -1233, -2500],
                 [rng.randint(-32768, 32767) for _ in range(95)]]
        for logits in cases:
            for seed in (0, 1, 0x1234, 0xffff):
                with self.subTest(length=len(logits), seed=seed):
                    cpu = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_16)
                    cpu.mem_map(0, 0x20000)
                    # Different code base than the bundled image checks relocation.
                    cpu.mem_write(0x8000, self.code)
                    cpu.mem_write(0x7800, b''.join(x.to_bytes(2, 'little', signed=True) for x in logits))
                    cpu.mem_write(0x7b80, seed.to_bytes(2, 'little'))
                    cpu.reg_write(x86.UC_X86_REG_DX, max(logits) & 0xffff)
                    cpu.reg_write(x86.UC_X86_REG_BP, len(logits))
                    cpu.reg_write(x86.UC_X86_REG_SP, 0x7000)
                    cpu.mem_write(0x7000, b'\x00\x20')
                    cpu.emu_start(0x8002, 0x2000, count=100000)
                    self.assertEqual(cpu.reg_read(x86.UC_X86_REG_IP), 0x2000)
                    actual = bytes(cpu.mem_read(0x7200, 2 * len(logits)))
                    expected = b''.join(w.to_bytes(2, 'little') for w in weights_for(logits))
                    if lcg(seed) >> 8 < 64:
                        self.assertEqual(actual, expected)
                    choice, newrnd = sample(logits, seed)
                    self.assertEqual(cpu.reg_read(x86.UC_X86_REG_BX), choice)
                    self.assertEqual(int.from_bytes(cpu.mem_read(0x7b80, 2), 'little'), newrnd)
