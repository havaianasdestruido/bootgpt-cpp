#!/usr/bin/env python3
"""Reassemble the runtime around the already-trained, committed model.

This is not retraining: it preserves the complete weights/seed/strings prefix,
all generated constants and the existing sector count. It refuses to grow the
blob because that would require regenerating metadata with train.py.
"""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / 'model'


def main():
    inc = (MODEL / 'params.inc').read_text()
    entries = dict(re.findall(r'^(\w+)\s+equ\s+(\S+)', inc, re.M))
    offset = int(entries['SYNTAXADDR'], 0) - 0x8000
    size = int(entries['SECTORS'], 0) * 512
    weights = (MODEL / 'weights.bin').read_bytes()
    if len(weights) != size or not 0 <= offset < size:
        raise ValueError('inconsistent model layout')
    with tempfile.TemporaryDirectory() as tmp:
        syntax = Path(tmp) / 'syntax.bin'
        boot = Path(tmp) / 'boot.bin'
        subprocess.run(['nasm', '-f', 'bin', 'syntax.asm', '-o', str(syntax)],
                       cwd=ROOT, check=True)
        subprocess.run(['nasm', '-f', 'bin', 'boot.asm', '-o', str(boot)],
                       cwd=ROOT, check=True)
        code, boot_data = syntax.read_bytes(), boot.read_bytes()
    if len(boot_data) != 512 or boot_data[-2:] != b'\x55\xaa':
        raise ValueError('invalid boot sector')
    if offset + len(code) > size:
        raise ValueError('highlighter exceeds existing sector budget; retrain/rebuild')
    weights = weights[:offset] + code + bytes(size - offset - len(code))
    (MODEL / 'weights.bin').write_bytes(weights)
    (MODEL / 'boot.bin').write_bytes(boot_data)
    (MODEL / 'disk.img').write_bytes(boot_data + weights)
    print(f'boot: {len(boot_data)} bytes; weights: {len(weights)} bytes; '
          f'highlighter: {len(code)} bytes at {offset:#x}')


if __name__ == '__main__':
    main()
