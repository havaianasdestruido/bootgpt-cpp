#!/usr/bin/env bash
# boot the C++20 dream machine in a QEMU window (any key reboots)
set -euo pipefail
cd "$(dirname "$0")"
qemu-system-i386 -drive format=raw,file=model/disk.img,if=ide -boot c
