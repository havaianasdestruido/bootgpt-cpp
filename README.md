# bootgpt-cpp

A neural network that boots your PC and dreams C++20.

The MBR is 512 bytes of hand-written 16-bit x86 assembly (`460` bytes of code
before zero padding + `0xAA55`). It loads an `int8`/`int16` character-level
model and a position-independent syntax highlighter from the following sectors,
then runs inference in 16-bit real mode on a 386+. No OS, C runtime, or
allocator is needed. Its colorized output resembles C++ systems code; it is not
guaranteed to compile.

![demo](bootgpt_demo.gif)

With 32 characters of context (`88`-char C++ vocabulary, `E=6` embeddings,
`H=100` hidden units, integer softclip, and Min-P logit-margin sampling), here
is actual integer output from an earlier boot-sector model (before the
latest corpus expansion):

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

The trained boot image is included in `model/`; no training or source download
is needed to try it. Install QEMU (`brew install qemu` on macOS or
`sudo apt install qemu-system-x86` on Ubuntu/Debian), then run:

```bash
./run.sh
```

`model/disk.img` is a standard raw BIOS boot image. Attach it with
`-drive format=raw,file=model/disk.img,if=ide` in QEMU / Bochs / v86, or
write it to a USB stick. Any keypress interrupts generation and reboots via
`int 19h` with a fresh timer-seeded LCG state.

### Retrain and rebuild

```bash
# macOS: brew install nasm qemu
# Ubuntu/Debian: sudo apt install nasm qemu-system-x86
python3 -m venv .venv && .venv/bin/pip install numpy
python3 tools/fetch_sources.py  # download the pinned training files once
./build.sh
```

The build overwrites the trained artifacts in `model/`. `model.npz` contains
floating-point weights before quantization-aware training; `weights.bin` is the
final quantized model and syntax highlighter used by `disk.img`. See
`model/README.md` for the current artifacts and their training details.

## Pieces

- **`corpus_gen.py`** mixes 100 synthetic C++20 translation units with 41
  comment-stripped upstream source files from Chromium, Firefox (gecko-dev),
  Thunderbird (comm-central), GCC, and LLVM/Clang. Each upstream file appears
  once, and units are shuffled deterministically; upstream styles and longer
  lines are retained. Comments are removed, tabs and form feeds normalized,
  and runs of empty lines compacted for BIOS output.
  The sources require their original projects and are not standalone programs.
- **`train.py`** derives the character vocabulary from the corpus (currently
  95 characters), checks the boot-sector weight budget, and trains the
  `C=32, E=6, H=100` character-level MLP. It applies quantization and 5,000
  steps of Quantization-Aware Training, emits `model/weights.bin` and
  `model/params.inc`,
  and simulates integer inference via `BootSim`. Run `./build.sh` to retrain;
  changing the corpus alone does not update a previously built model. A larger
  and more varied corpus does not guarantee that this small model will generate
  correct or non-repetitive code.
- **`boot.asm`** is the entire bare-metal runtime inside the 510-byte MBR code
  budget: `int 13h` to load weights into `0x0800:0000`, `int 10h` colored
  output, integer softclip `trunc(64*p / (4096 + |p|))` via `cdq`/`idiv`, and
  exact Min-P logit-margin gating (`L[v] >= L_max - MARGIN`) with PIT-seeded
  16-bit LCG noise (`int 1Ah`).
- **`syntax.asm`** is a position-independent streaming C++ highlighter loaded
  alongside the model. It maps the GitHub-inspired palette to standard VGA
  attributes: gray text, light-red keywords, magenta functions, cyan constants
  and strings, yellow built-ins, dark-gray comments, and green entities on black.
- **`record.py`** records `bootgpt_demo.mp4` and `bootgpt_demo.gif` from
  headless QEMU via the monitor socket.

## Upstream source and licensing

`upstream_sources.json` identifies the pinned repository, commit, path,
original header notices, and hashes for each training file. The `.txt` files
contain code with comments removed; license notices are kept separately and
are not fed to the model. Complete license texts are in `licenses/upstream/`.
Chromium excerpts are BSD-3-Clause, Firefox and Thunderbird excerpts MPL-2.0,
LLVM/Clang excerpts Apache-2.0 with LLVM exceptions, and GCC excerpts GPL-3.0
or later. **The repository's MIT license does not replace these upstream
licenses**; redistribution of the files or the combined corpus must comply
with their applicable terms, including the GCC GPL. See the pinned originals
in the manifest for the full source and context. `tools/strip_cpp_comments.py`
contains the preparation logic. The generated `source_code/` directory is
Git-ignored: `tools/fetch_sources.py` downloads the pinned originals, verifies
their hashes, strips comments, and verifies the resulting files. Run it once
before building or testing; subsequent builds are offline. If GitHub rate-limits
the download, rerun the command later; already verified files are reused. You
can also use `--from-dir PATH` with originals arranged as `PROJECT/UPSTREAM_PATH`.

## Tests

Run `python3 -m unittest discover -s tests -v` to check the bundled model, corpus constraints,
source hashes, comment stripping, and deterministic mixing; also compile sample
synthetic translation units and exercise their C++ examples. The C++ tests
require `g++` with C++20 support.

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
