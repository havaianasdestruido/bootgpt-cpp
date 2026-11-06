# bootgpt-cpp

A neural network that boots your PC and dreams C++20.

The MBR is 512 bytes of hand-written 16-bit x86 assembly (`445` bytes of code
before zero padding + `0xAA55`). It loads a `28.5 KB` `int8`/`int16` language
model from the next 57 sectors (`CHS 0/0/2`) and runs character-level inference
in 16-bit real mode on a 386+. No OS, no C runtime, no allocator. The machine
wakes up and streams indented, warning-free C++20 systems code: `#include`
headers, namespaces, `constexpr` constants, `enum class`, templates, structs,
classes, methods, and free functions.

![demo](bootgpt_demo.gif)

With 32 characters of context (`88`-char C++ vocabulary, `E=6` embeddings,
`H=100` hidden units, integer softclip, and Min-P logit-margin sampling), here
is actual integer output from the boot-sector model:

```cpp
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace net {

constexpr size_t kPageSize = 8192;
constexpr size_t kCacheLine = 32;
constexpr uint32_t kMagic = 0xCAFE0001;
constexpr size_t kMaxSlots = 512;

enum class Status : uint8_t {
    Ok = 0,
    Empty = 1,
    Full = 2,
    Invalid = 3,
    Timeout = 4,
};

struct PacketHdr {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t body_bytes;
};

inline bool check_hdr(const PacketHdr* hdr, size_t wire_len) {
    if (!hdr || wire_len < sizeof(PacketHdr)) {
        return hdr != nullptr && wire_len >= sizeof(PacketHdr);
    }
    if (hdr->magic == 0 || hdr->version == 0) {
        return 0;
    }
    return sizeof(PacketHdr) + hdr->body_bytes <= wire_len;
}
```

## Run it

```bash
# macOS: brew install nasm qemu
# Ubuntu/Debian: sudo apt install nasm qemu-system-x86
python3 -m venv .venv && .venv/bin/pip install numpy
./build.sh
./run.sh
```

`disk.img` (`29,696` bytes) is a standard raw BIOS boot image. `dd` it to a USB
stick or attach it as `-drive format=raw,file=disk.img,if=ide` in QEMU / Bochs /
v86 — any keypress interrupts generation and reboots via `int 19h` with a fresh
timer-seeded LCG state.

## Pieces

- **`corpus_gen.py`** generates ~4.0 MB of modern C++20 systems translation
  units (4-space indentation, K&R braces, `<= 76` columns for 80x25 VGA) with
  canonical declaration ordering and disjoint 30-gram class idioms so a 32-char
  sliding window tracks brace depth and member scope cleanly.
- **`train.py`** trains the `V=88, C=32, E=6, H=100` character-level MLP with
  annealed distant-context dropout, applies 99.8th-percentile power-of-two
  quantization followed by 5,000 steps of Quantization-Aware Training (Straight-Through
  Estimator on the exact `int8`/`int16` grid + integer softclip), emits
  `weights.bin` and `params.inc`, and verifies exact integer parity via `BootSim`.
- **`boot.asm`** is the entire bare-metal runtime inside the 510-byte MBR code
  budget: `int 13h` to load weights into `0x0800:0000`, `int 10h` teletype
  output, integer softclip `trunc(64*p / (4096 + |p|))` via `cdq`/`idiv`, and
  exact Min-P logit-margin gating (`L[v] >= L_max - MARGIN`) with PIT-seeded
  16-bit LCG noise (`int 1Ah`).
- **`record.py`** records `bootgpt_demo.mp4` and `bootgpt_demo.gif` from
  headless QEMU via the monitor socket.

## Prior art

The boot sector genre is old and glorious. Oscar Toledo put
[chess](https://github.com/nanochess/Toledo-Atomchess), a whole
[OS](https://github.com/nanochess/bootOS) and even a
[BASIC interpreter](https://github.com/nanochess/bootBASIC) in there,
other people did a [C compiler](https://github.com/xorvoid/sectorc),
all inside 512 bytes.

Closest neighbors include the original [bootgpt](https://x.com/krlntn/status/2116514628457185757)
(which hallucinated Linux `dmesg` boot logs), [sectorllm](https://github.com/rdmsr/sectorllm)
(which boots a multi-layer Llama-2 transformer from ~1.2 KB of real-mode asm),
and [sectorlisp-nn](https://github.com/woodrush/sectorlisp-nn). `bootgpt-cpp`
keeps the entire inference engine inside the 510-byte MBR itself and trains a
compact int8 network to hallucinate clean, indented C++20 code on bare metal.
