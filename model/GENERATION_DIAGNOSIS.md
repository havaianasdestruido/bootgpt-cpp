# Generation investigation (incomplete)

The long-running repetition problem is **not yet fixed**. Do not interpret the
highlighter correction as a generation-quality fix. The model is a 32-character
MLP, with position-specific first-layer weights, not a transformer; it has no
attention cache, positional embedding table, EOS token, or prompt interface in
the boot image. It generates until a keypress. The character vocabulary is
stored at the start of `weights.bin` and the seed is in that same blob.

## Reproduce the absorbing state

Install NumPy and run `python3 tools/evaluate_generation.py`. Add `--trace` for
JSON records with context and chosen token IDs, position, and integer logits and
hidden/preactivation ranges. The tool reads the bundled final integer weights;
it is not a substitute for booting the image. Its extra prompts are reference
experiments, left-padded with spaces and truncated to the most recent 32
characters, not an interface provided by the BIOS image.

For the original model and this branch, the same weights and decoding settings
produce these 1,500-character simulations. Repeated 4-gram fraction is
`1 - (unique 4-grams / all 4-grams)`; spaces count as characters. Exact seed,
prompts, and settings are in the script.

| Case | Seed (hex) | Repeated 4-gram fraction | Longest space run | Newlines |
|---|---:|---:|---:|---:|
| boot-seed-loop | 1235 | 0.9045 | 1282 | 10 |
| short-completion | 0001 | 0.7956 | 1009 | 35 |
| multiline-function | 0002 | 0.9586 | 1426 | 3 |
| long-generation | caff | 0.9078 | 1325 | 8 |
| balanced-delimiters | 0003 | 0.9993 | 1500 | 0 |

The 0x1235 case first enters the final run at output index 218 (zero-based).
After 32 spaces the context is all spaces: the space logit is 1840, and the
next highest logit is 607. The Min-P margin is 240, so space is the **only**
allowed token; the context stays all spaces forever. This is a consequence of
both the model's predictions and the hard cutoff. The context cannot distinguish
32 from 33 or more consecutive spaces. The training corpus does contain long
space runs (389 of at least 32), so those contexts are not literally absent.
The pre-QAT float checkpoint also favors space there (7.222 versus 2.636 for
its runner-up). It is a different checkpoint, not the float version of the
final QAT weights; a numerical equality comparison would be misleading.

Removing or substantially widening the cutoff did not establish a safe fix:
with margin 1280 the 0xcafe sample no longer begins with coherent includes and
contains visibly more garbled code. A lower 4-gram repetition score alone does
not establish improved C++ quality. A correct solution still requires a
validated decoding/training approach that can escape the absorbing state while
maintaining output quality. Small context and imperfect training also limit C++
syntax and long-range consistency. Model output is not guaranteed to compile.

## Separately fixed: display loses long identifiers

`syntax.asm` formerly stopped accepting identifier bytes after its 31-byte
buffer filled, silently discarding all later characters until a delimiter.
This produced a displayed stream that did not match generated tokens. It now
flushes the prefix and streams the remainder with the same color. The change
does not change model predictions, which is why the table is identical before
and after. `tests/test_highlighter.py` executes the assembled highlighter under
Unicorn, including identifiers on both sides of the boundary. On the original,
`struct ` + 40 `A`s + `_9;` displayed only 31 `A`s before the semicolon.

`tests/test_inference.py` executes the real boot-sector instructions under
Unicorn with BIOS interrupts stubbed. For four contexts and three seeds, hidden
activations, logits, and token selection match the independent integer
reference exactly; a separate 270-token full boot-loop test checks the context
and RNG on each iteration. The latter crosses the all-space failure point for the boot seed and
PIT value 0xcafe (the normal boot code forces it odd, to 0xcaff).
This is not exhaustive proof of inference correctness
or a hardware/BIOS test. During investigation the image was also booted in
headless QEMU's x86_64 system emulator with an IDE raw disk and VGA output was
observed. The exact `qemu-system-i386` command remains untested here because
that binary was unavailable.

Full pre-QAT training and QAT were not rerun. There is no saved post-QAT float
checkpoint, so quantization-only numerical error cannot be isolated by comparing
the final binary to `model.npz`. At four example contexts the mean absolute
logit difference against the *pre-QAT* checkpoint ranged from 0.2166 to 0.4578
(logit units), and an argmax differed for one. This includes QAT changes and is
not a quantization tolerance test. For the three C++ function prompt cases above, compiling the entire prompt plus
1,500 generated characters with `g++ -std=c++20 -fsyntax-only` failed in all
three cases; the generated samples may also end mid-statement, so this is not a
perfect measure of C++ quality. No syntax-validity improvement is claimed.
