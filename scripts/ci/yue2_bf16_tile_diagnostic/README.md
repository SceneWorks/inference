# YuE2 BF16 VAE tile diagnostic

This is a read-only numerical diagnostic for the frozen M3 engine, not an acceptance test or a changed tolerance. It uses the pinned 75-frame standard VAE fixture with core 16 and halo 16. The default waveform path loads the verified standard VAE with the same Full parts setting as the failed proof (encoder execution is not selected), and performs two BF16 full/tiled pairs plus one FP32 control pair. It does not generate a song or fetch weights.

The repository stores `Cargo.toml.in`, `Cargo.lock.snapshot`, and `src/main.rs`. The manifest includes M3's vendored Candle kernel patch, and the lock was seeded from M3's `Cargo.lock`, then pruned without changing any dependency tuple. The controller must materialize these in a fresh directory **outside both checkouts**, replacing only `__ENGINE_ROOT__` with the absolute path of the clean engine checkout at `4127a675fc8575555e029e01b7f6867488880a8f`. Copy the lock unchanged and use `--locked --offline`. The binary independently verifies that engine SHA and clean state.

Runtime interface:

```text
YUE2_ENGINE_ROOT=<exact-engine-checkout>
YUE2_HF_HUB=<offline-pinned-snapshot-root>
CUDA_VISIBLE_DEVICES=0
yue2-bf16-tile-diagnostic [--diagnostic waveform|first_conv|first_conv_math] --reference-dir <verified-fixture-directory> --output-dir <fresh-absolute-directory>
```

`report.json` records source/fixture/decoder identity, raw and clamped tiled-versus-full residuals (full is the SNR denominator), repeatability, per-seam and interior maxima, argmax location, and the original 1/64 comparison. The same output directory contains per-run little-endian interleaved stereo F32 sample arrays; each file has a byte size and SHA-256 in the report. The directory must not exist before invocation. Keep the directory as run-owned evidence. A numerical bound miss is data in the report, not a process failure; setup, integrity, and execution faults fail the process.

`waveform` is the default and retains schema version 1. The explicit `first_conv` selector runs only `decoder.layers.0` Conv7 on the same 75-frame latent, once in BF16 and once in F32. It folds verified FP32 weight-norm tensors before casting resident operands, then saves full and five halo-window input, pre-bias, and post-bias tensors. Its schema version 2 report names the selector, source/fixture/component identities, requested operator geometry and resident dtypes, and each aligned core's first differing coordinate and maximum error. All 36 tensors are flattened BCT F32LE files with dtype, shape, byte count and SHA-256; the controller independently recomputes the aligned comparisons. The prior waveform 1/64 failure is retained as an observation, never applied as a Conv7 pass threshold.

`first_conv_math` preserves those default BF16/F32 captures under a separate schema version 3. Only when BF16 aligned input cores are byte-identical **and** a pre-bias core has a finite positive numerical residual does it compare another BF16 Conv7 arm with cuBLAS reduced-precision reductions disallowed. Signed-zero bit differences alone do not trigger it. The harness reads the raw handle's default mode, sets/readbacks mode 16, synchronizes, then restores/readbacks mode 0; the run-owned `math-mode-events.jsonl` records every status and readback. The report retains 36 default arrays plus 18 flagged arrays when applicable, independent cross-arm comparisons, or an explicit `not_applicable` reason. It is controlled diagnostic evidence, not a waveform acceptance result or a causal conclusion.
