"""Exercise the actual 16-bit highlighter, intercepting BIOS teletype calls.

Requires NASM and the optional unicorn package. The tests execute instructions,
not a Python reimplementation of the lexer.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

try:
    import unicorn
    from unicorn import x86_const as x86
except ImportError:
    unicorn = None

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(unicorn and shutil.which('nasm'), 'requires unicorn and nasm')
class HighlighterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'syntax.bin'
            subprocess.run(['nasm', '-f', 'bin', 'syntax.asm', '-o', str(out)],
                           cwd=ROOT, check=True)
            cls.code = out.read_bytes()

    def render(self, text):
        cpu = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_16)
        cpu.mem_map(0, 0x20000)
        cpu.mem_write(0x8000, self.code)
        cpu.reg_write(x86.UC_X86_REG_CS, 0)
        cpu.reg_write(x86.UC_X86_REG_DS, 0)
        cpu.reg_write(x86.UC_X86_REG_SS, 0)
        output = []

        def bios(uc, no, _):
            self.assertEqual(no, 0x10)
            ax = uc.reg_read(x86.UC_X86_REG_AX)
            if ax >> 8 == 0x0e:
                output.append(chr(ax & 0xff))
            else:
                self.assertEqual(ax >> 8, 0x09)
        cpu.hook_add(unicorn.UC_HOOK_INTR, bios)
        for char in text:
            cpu.reg_write(x86.UC_X86_REG_SP, 0x7000)
            cpu.mem_write(0x7000, b'\x00\x20')  # return to 0000:2000
            cpu.reg_write(x86.UC_X86_REG_AX, ord(char))
            cpu.emu_start(0x8000, 0x2000, count=100000)
            self.assertEqual(cpu.reg_read(x86.UC_X86_REG_IP), 0x2000)
        return ''.join(output)

    def test_long_identifiers_and_following_tokens_are_not_dropped(self):
        for length in (30, 31, 32, 33, 64, 128):
            with self.subTest(length=length):
                text = 'struct ' + 'A' * length + '_9; int next;\r\n'
                self.assertEqual(self.render(text), text)

    def test_delimiters_and_quoted_content(self):
        text = 'int ' + 'a' * 65 + '(int x) { return x / 2; }\r\n'
        text += '"hello world"; // comment\r\n'
        self.assertEqual(self.render(text), text)
