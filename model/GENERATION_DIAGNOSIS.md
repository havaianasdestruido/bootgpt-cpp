# Generation investigation and validation

The included character MLP uses 32 context characters and position-specific
first-layer weights. It is not a transformer: there is no attention mask, cache,
or positional embedding table. The boot image has no prompt interface or EOS;
it generates until a keypress. Vocabulary and seed are in `weights.bin`.

## Root cause and correction

With the original weights and sampler, 32 spaces yield a space logit of 1840
and a runner-up of 607. The 240-point Min-P cutoff admits **only** space; once
32 spaces have been generated, the context is identical forever and the sampler
cannot exit. The corpus contains 389 space runs of at least 32; among the 3,807
next-character targets following 32 spaces, 3,418 are spaces (89.8%). Thus the
model's strong space preference reflects the finite-context training data, not
an incorrectly shifted context. Its Q8 logits imply 91.5% space under softmax;
the pre-QAT float checkpoint also strongly prefers space. It is a different
checkpoint, not a dequantized copy of the final QAT model.

The sampler now uses the original Min-P/noise behavior on three quarters of
branch draws and approximate categorical softmax on one quarter. This keeps much
of the original output style while allowing alternatives the cutoff excluded.
Separate LCG updates are used for branch selection and sampling; they are
deterministic and not statistically independent. No repetition penalty,
context-specific rule, or output truncation is used. The position-independent
sampler is in the loaded runtime extension alongside the highlighter. The
softmax branch uses a 16-entry Q14 lookup for fractional powers of two and
integer multiply-high sampling. It approximates temperature-one softmax; very
small weights can still quantize to zero. The combined distribution is a
mixture, **not** exact softmax. The exact sector budget is now full.

For this model's all-space context, an exhaustive test of all 65,536 possible
RNG states finds no all-space cycle; the longest continuous all-space trajectory
before a non-space is selected is 225 draws. This does not prove the absence of
all other loops. The highlighter was also corrected: identifiers longer than
31 characters were silently truncated in the displayed stream. It now flushes
the prefix and streams the rest.

## Reproduce the comparison

With NumPy installed and the corpus sources downloaded (see the main README):

```sh
python3 corpus_gen.py
python3 tools/evaluate_generation.py --legacy --corpus-file corpus.txt --syntax-check > before.jsonl
python3 tools/evaluate_generation.py          --corpus-file corpus.txt --syntax-check > after.jsonl
```

Use `--trace` for JSON lines with each context's IDs, selected token ID,
position, logits, and intermediate ranges. The simulator reads the committed
integer weights. Extra prompts are reference experiments left-padded with spaces
and truncated to 32 characters; they are not an interface offered by the image.
The decoder is the variable being compared; weights, contexts, seeds and lengths
are held constant. Legacy decoding uses the original margin and noise values.

For 1,500 generated characters per case, the following are repeated 4-gram
fraction (`1 - unique / total`) and longest consecutive space run:

| Case (seed) | Repetition before → after | Space run before → after | Newlines before → after |
|---|---:|---:|---:|
| boot seed (1235) | 0.9045 → 0.4402 | 1282 → 56 | 10 → 73 |
| short completion (0001) | 0.7956 → 0.4983 | 1009 → 94 | 35 → 59 |
| multiline function (0002) | 0.9586 → 0.4021 | 1426 → 16 | 3 → 58 |
| long generation (caff) | 0.9078 → 0.4723 | 1325 → 162 | 8 → 64 |
| balanced delimiters (0003) | 0.9993 → 0.4776 | 1500 → 61 | 0 → 70 |

In another 20 boot-seed trials (odd seeds 1 through 39), longest runs were
281–1479 before and 16–158 after; repeated 4-gram fraction fell in every case.
The fixed suite is exactly reproducible. The first boot-seed sample begins
with a run of recognizable includes, but subsequent output is often incoherent.

These repetition improvements do **not** establish improved C++ correctness.
All three function-prompt completions fail `g++ -std=c++20 -fsyntax-only`
both before and after; the fixed-length samples can also end mid-statement.
As a rough continuity proxy, the fraction of non-whitespace 6-grams in the first
300 output characters found in the training corpus was 0.710 → 0.708 for the
boot seed, 0.626 → 0.449 for short completion, 0.631 → 0.351 for multiline,
and 0.473 → 0.580 for long generation. The balanced case had no non-whitespace
6-grams before and 0.381 after. Counts differ because the samples differ; this
is a crude, training-corpus-based diagnostic, not a quality score. It exposes
some early-output regressions. The small model does not reliably produce
compilable or globally coherent C++ and may still repeat for other contexts.

## Numerical and runtime checks

On 2,000 deterministically chosen corpus contexts, the softmax branch's Q14
weight distribution had mean total-variation distance 0.00290 and maximum
0.00699 from floating-point softmax of the **same integer logits**. The original
cutoff excluded the actual next corpus character in 490 contexts; the Q14
softmax grid assigned zero weight to that target in 2. This checks the sampler
approximation, not model quantization accuracy or generated code validity.
`tests/test_sampling.py` compares the extension's actual 16-bit instructions to
the independent sampler for both branches and extreme int16 logits, and checks
the 65,536-state all-space graph. `tests/test_inference.py` compares real
boot-sector hidden activations, logits and choices with the integer reference,
and checks 270 consecutive context/RNG updates with BIOS interrupts stubbed.
`tests/test_highlighter.py` executes the assembled highlighter at the identifier
buffer boundary. These tests do not prove every context or hardware BIOS.

The image was booted and VGA text observed in headless QEMU's x86_64 system
emulator with the specified IDE raw-disk setup. The exact
`qemu-system-i386` invocation remains untested because that executable was
unavailable. Full retraining/QAT was not run. There is no post-QAT floating
checkpoint, so comparing final integer logits with `model.npz` cannot isolate
quantization error. At four example contexts, mean absolute logit differences
against that pre-QAT checkpoint ranged from 0.2166 to 0.4578 logit units;
one of four argmaxes differed. The differences include QAT changes.
