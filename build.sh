#!/usr/bin/env bash
# build the neural C++20 boot sector
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

nasm -f bin syntax.asm -o model/syntax.bin # position-independent VGA highlighter
"$PYTHON" corpus_gen.py             # mixed real and synthetic C++ corpus
"$PYTHON" train.py                  # train and quantize into model/
nasm -f bin boot.asm -o model/boot.bin    # 512-byte stage 1 (asserted below)
BOOT_BYTES=$(wc -c < model/boot.bin | tr -d ' ')
[ "$BOOT_BYTES" -eq 512 ] || { echo "BOOT SECTOR SIZE $BOOT_BYTES != 512"; exit 1; }
cat model/boot.bin model/weights.bin > model/disk.img
DISK_BYTES=$(wc -c < model/disk.img | tr -d ' ')
echo "model/disk.img ready: ${DISK_BYTES} bytes (512-byte boot sector + int8 C++20 weights)"
