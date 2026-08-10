# Cross-family MLX performance gate (sc-18321)

This local-only harness is the acceptance gate for shared MLX optimizations. It runs fixed,
real-weight workloads through the worker-adjacent `runtime-macos` catalog for:

- Wan 2.2 TI2V-5B video, including long-sequence denoise and the heavy causal VAE decode;
- Qwen-Image image DiT at 512, 1024, and a decode-bound 2048 square;
- SDXL UNet at the same three image geometries.

The committed matrix fixes provider, prompt, seed, tier, geometry, frames, steps, one warmup, and
three measured repetitions. The default campaign runs a baseline, each P1/P3/P4/P5/P9 toggle in
isolation, and all toggles together. Each case/variant runs in a fresh child process so MLX's
process-global allocator state cannot contaminate another row.

## Inputs

Copy `mlx-perf-artifacts.example.json` outside the checkout and replace each illustrative path with
the exact installed tier directory selected by the matrix (Wan/Qwen q4 and SDXL bf16). The validator
requires:

- exactly the three matrix artifact keys, repositories, and tiers;
- a 40-character lowercase resolved revision;
- an existing absolute directory whose final component is the selected tier and whose parent is that
  exact resolved revision;
- every provider-specific load sentinel (config, tokenizer, transformer/UNet, encoder, and VAE
  files) to be present before any run starts.

The revision is an assertion about the local bytes, not something the harness guesses from a
mutable `refs/main`. Use a SceneWorks install receipt or the immutable HF snapshot directory name.
The harness never downloads weights.

Validate the matrix and artifact bindings without loading MLX weights:

```sh
cargo run --release --locked -p runtime-macos --no-default-features --features perf-bench \
  --bin mlx-perf-bench -- validate --artifacts /absolute/path/mlx-perf-artifacts.json
```

## Run

Start from a clean inference commit, an idle Apple-Silicon Mac, and an empty output directory outside
the checkout. One local command produces per-run JSON, a schema-validated `summary.json`, and timing
plus phase-peak tables:

```sh
cargo run --release --locked -p runtime-macos --no-default-features --features perf-bench \
  --bin mlx-perf-bench -- run --artifacts /absolute/path/mlx-perf-artifacts.json \
  --output-dir /absolute/path/empty-results-directory
```

Before optimization stories are wired, an explicit baseline-only campaign is available:

```sh
cargo run --release --locked -p runtime-macos --no-default-features --features perf-bench \
  --bin mlx-perf-bench -- run --artifacts /absolute/path/mlx-perf-artifacts.json \
  --output-dir /absolute/path/empty-baseline-directory --variants baseline
```

Do not interpret a baseline-only run as acceptance of any optimization. Nonbaseline variants are
fail-closed: selecting a toggle merely exposes the request to provider code; every measured request
must also emit a positive `diagnostics::record_toggle(..., ToggleDisposition::Applied)` receipt.
An unavailable, fallback, or unacknowledged implementation aborts the child before it can publish a
comparison.

## Evidence semantics

Every measured request must emit exactly the configured number of `Progress::Step` events, one
`Progress::Decoding` event, nonempty output with the expected geometry/frame count, and a SHA-256
digest identical across repetitions. Cold load timing, exact inference and mlx-rs revisions, Rust,
macOS, hardware, Metal device, compile/cache/fallback diagnostics, and per-phase allocator samples
are recorded in JSON.

`loadActivePeakBytes` and `loadCacheBytesAfterLoad` are observations, not validity gates: providers
that defer materialization until generation can legitimately report zero for both while still
recording positive cold-load wall time. Request phases must always report nonzero samples and active
peaks.

MLX exposes an active-memory high-water mark but no cache high-water mark. At each progress and phase
boundary the runner therefore records `get_peak_memory()` as the active peak and samples
`get_cache_memory()` separately, retaining the maximum observed cache sample and the boundary value.
It resets the active peak at every phase transition.

The current generator contract emits its first `Progress::Step` after denoise step 1. That event is
the only provider-neutral encode-to-denoise seam, so the reported encode interval includes the first
step and steady denoise throughput uses the remaining `steps - 1` intervals. `Progress::Decoding` is
the exact denoise-to-decode seam. This limitation is explicit in the JSON interpretation; the runner
does not pretend it observed a finer boundary than the production contract exposes.
