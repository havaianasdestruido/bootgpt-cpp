#!/usr/bin/env bash
# build the neural C++20 boot sector
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

nasm -f bin syntax.asm -o syntax.bin # position-independent VGA highlighter
"$PYTHON" corpus_gen.py             # ~4.0 MB of synthetic C++20 translation units
"$PYTHON" train.py                  # train, quantize, emit params.inc + weights.bin
nasm -f bin boot.asm -o boot.bin    # 512-byte stage 1 (asserted below)
BOOT_BYTES=$(wc -c < boot.bin | tr -d ' ')
[ "$BOOT_BYTES" -eq 512 ] || { echo "BOOT SECTOR SIZE $BOOT_BYTES != 512"; exit 1; }
cat boot.bin weights.bin > disk.img
DISK_BYTES=$(wc -c < disk.img | tr -d ' ')
echo "disk.img ready: ${DISK_BYTES} bytes (512-byte boot sector + int8 C++20 weights)"
