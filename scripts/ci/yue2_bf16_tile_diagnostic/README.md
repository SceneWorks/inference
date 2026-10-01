# YuE2 BF16 VAE tile diagnostic

This is a read-only numerical diagnostic for the frozen M3 engine, not an acceptance test or a changed tolerance. It uses the pinned 75-frame standard VAE fixture with core 16 and halo 16; loads the verified standard VAE with the same Full parts setting as the failed proof (encoder execution is not selected); and performs two BF16 full/tiled pairs plus one FP32 control pair. It does not generate a song or fetch weights.

The repository stores `Cargo.toml.in`, `Cargo.lock.snapshot`, and `src/main.rs`. The manifest includes M3's vendored Candle kernel patch, and the lock was seeded from M3's `Cargo.lock`, then pruned without changing any dependency tuple. The controller must materialize these in a fresh directory **outside both checkouts**, replacing only `__ENGINE_ROOT__` with the absolute path of the clean engine checkout at `4127a675fc8575555e029e01b7f6867488880a8f`. Copy the lock unchanged and use `--locked --offline`. The binary independently verifies that engine SHA and clean state.

Runtime interface:

```text
YUE2_ENGINE_ROOT=<exact-engine-checkout>
YUE2_HF_HUB=<offline-pinned-snapshot-root>
CUDA_VISIBLE_DEVICES=0
yue2-bf16-tile-diagnostic --reference-dir <verified-fixture-directory> --output-dir <fresh-absolute-directory>
```

`report.json` records source/fixture/decoder identity, raw and clamped tiled-versus-full residuals (full is the SNR denominator), repeatability, per-seam and interior maxima, argmax location, and the original 1/64 comparison. The same output directory contains per-run little-endian interleaved stereo F32 sample arrays; each file has a byte size and SHA-256 in the report. The directory must not exist before invocation. Keep the directory as run-owned evidence. A numerical bound miss is data in the report, not a process failure; setup, integrity, and execution faults fail the process.
