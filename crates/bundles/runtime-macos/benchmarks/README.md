# Cross-family MLX performance gate (sc-18321)

This local-only harness is the acceptance gate for the shared MLX optimization train. It exercises
the production `runtime-macos` catalog with nine fixed, real-weight workloads:

- Wan 2.2 TI2V-5B q4 video at three sequence/geometry points;
- Qwen-Image q4 at 512, 1024, and decode-bound 2048 squares;
- SDXL bf16 at the same three image geometries.

The matrix fixes provider, prompt, seed, exact artifact revision and content inventory, tier,
geometry, frames, steps, one warmup, and three measured repetitions. Its required-all campaign is
baseline, each P1/P3/P4/P5/P9 toggle independently, and all five together. Every case/variant runs
in a fresh child process so process-global MLX allocator state cannot leak between comparison rows.

## Exact inputs and executable provenance

Copy `mlx-perf-artifacts.example.json` outside the checkout and replace its illustrative paths with
the installed tier directories. A binding is accepted only when its key, repository, 40-character
revision, tier, path shape, deterministic recursive file count, byte count, and content inventory
match the committed matrix. The inventory algorithm hashes each sorted relative path, file size,
and file-content SHA-256; an old sentinel fixture or a mutable `refs/main` assertion cannot satisfy
it. The harness never downloads weights.

`validate` reads every artifact byte to establish those inventories (without constructing MLX
models):

```sh
cargo run --release --locked -p runtime-macos --no-default-features --features perf-bench \
  --bin mlx-perf-bench -- validate --artifacts /absolute/path/mlx-perf-artifacts.json
```

The executable embeds the inference HEAD, dirty state, and pinned mlx-rs revision at build time.
Every command that can create or accept evidence compares those receipts with the runtime checkout
and lockfile. Build from a clean committed checkout; changing or committing source requires a fresh
binary. A runtime-only `git rev-parse` cannot substitute for executable provenance.

## Frozen campaigns and execution

Use an idle Apple-Silicon Mac and an empty absolute output directory outside the checkout:

```sh
cargo run --release --locked -p runtime-macos --no-default-features --features perf-bench \
  --bin mlx-perf-bench -- run --artifacts /absolute/path/mlx-perf-artifacts.json \
  --output-dir /absolute/path/empty-results-directory
```

Before starting any child, the parent writes `campaign.json`. It contains canonical copies and
hashes of the full matrix and artifact manifest, exact artifact inventories, selected variants,
build and host identities, and provider-owned toggle capability declarations. Its campaign ID binds
all of that state. Children receive only this frozen envelope plus their case/variant identity; they
rehash the selected artifact before loading and reject changed, mixed, stale, wrong-build, or
wrong-host state.

The default required-all campaign fails at preflight until every benchmark provider explicitly
declares every requested capability. Declaration is only availability, never proof that a path ran.
Every warmup and measured request must independently emit exactly one aggregated terminal
`ToggleDisposition::Applied` record for each requested toggle, no unrequested terminal record, and
no fallback or unavailable outcome. Baseline forbids all toggle terminal records.

A baseline-only campaign remains available while optimization call sites are being integrated:

```sh
cargo run --release --locked -p runtime-macos --no-default-features --features perf-bench \
  --bin mlx-perf-bench -- run --artifacts /absolute/path/mlx-perf-artifacts.json \
  --output-dir /absolute/path/empty-baseline-directory --variants baseline
```

Baseline-only and partial campaigns are diagnostic runs, not acceptance evidence. `summary.json`
sets `acceptanceComplete` only for the exact required-all selection. Merging this infrastructure
does not complete sc-18321: the story stays open until P1/P3/P4/P5/P9 are integrated and the full
nine-case required-all comparison is successfully captured and reviewed.

## Evidence semantics

Providers emit explicit `DenoiseStart` and `DecodeStart` diagnostics at production code boundaries.
The runner does not infer phases from UI progress. It requires exactly ordered boundaries; exact
`Progress::Step` receipts `1..N` inside the denoise interval; one later `Progress::Decoding`; no
post-decode steps; and stage durations whose encode + denoise + decode intervals cover the measured
request. Denoise throughput uses all configured steps.

Each phase owns a 50 ms background allocator probe with immediate, periodic, and final samples.
Every tick reads live active and cache bytes as one pair. JSON preserves:

- MLX's native active-memory high-water mark;
- independently observed sampled active and cache maxima;
- the maximum same-tick `active + cache` footprint;
- the active/cache witness pair from the exact tick that established that footprint;
- interval, sample-count, span, and maximum-gap coverage receipts.

The summary reports stage durations plus active, cache, and paired-footprint values separately. It
chooses the binding phase from the median same-sample footprint and never adds independent active and
cache maxima that may not have coexisted. These remain allocator-local host diagnostics, not a
portable process-footprint or target-device admission estimate.

Every measured output is nonempty, has the exact requested geometry/frame count, and carries a
SHA-256 over the produced bytes. Digests must be stable across repetitions and exactly equal to the
case's baseline digest across every selected variant.

Validate an existing result directory without rewriting it:

```sh
cargo run --release --locked -p runtime-macos --no-default-features --features perf-bench \
  --bin mlx-perf-bench -- validate-results --results-dir /absolute/path/results-directory
```

Legacy v1 directories have no frozen campaign and are rejected as unbound evidence.
