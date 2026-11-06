# Trained model

These files are the result of training on the mixed corpus described in the
main README, with seed 7, 50,000 training steps and 5,000 quantization-aware
training steps. The recorded training log contains loss metrics and samples;
the samples are often invalid or repetitive C++.

- `disk.img`: bootable disk image containing the boot sector and final weights.
- `boot.bin`: assembled 512-byte boot sector.
- `weights.bin`: quantized weights, vocabulary, seed, runtime strings and the
  position-independent VGA syntax highlighter.
- `params.inc`: assembly constants generated for those weights.
- `params.json`: model dimensions, quantization settings and seed indices.
- `model.npz`: floating-point weights **before** quantization-aware training.
  It is not the source of the final quantized weights without retraining.
- `training.log`: output from the full training and integer simulation.

Use `./run.sh` from the repository root, or attach `disk.img` as a raw disk in
a BIOS-capable emulator. Run `./build.sh` from the repository root to regenerate
these files (except the saved training log). Regeneration overwrites the current
artifacts. The upstream training sources and their licenses are documented in
the main README and `../upstream_sources.json`.

The highlighter can be rebuilt without retraining via
`python3 tools/rebuild_runtime.py`; see `GENERATION_DIAGNOSIS.md` for the
remaining generation-loop bug and instruction-level validation. The committed
`weights.bin` includes the corrected highlighter, while its trained-weight
prefix and the model parameters remain unchanged.
