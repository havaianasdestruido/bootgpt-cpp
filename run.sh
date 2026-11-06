#!/usr/bin/env bash
# boot the C++20 dream machine in a QEMU window (any key reboots)
set -euo pipefail
qemu-system-i386 -drive format=raw,file=disk.img,if=ide -boot c
