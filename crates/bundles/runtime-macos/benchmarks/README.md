# Cross-family MLX performance gate (sc-18321)

This local-only harness is the acceptance gate for the shared MLX optimization train. It exercises
the production `runtime-macos` catalog with nine fixed, real-weight workloads:

- Wan 2.2 TI2V-5B q4 video at three sequence/geometry points;
- Qwen-Image q4 at 512, 1024, and decode-bound 2048 squares;
- SDXL bf16 at the same three image geometries.

The matrix fixes provider, prompt, seed, exact artifact revision and content inventory, tier,
geometry, frames, steps, one warmup, and three measured repetitions. Its required-all campaign is
baseline, a fixed-tile decode control, each P1/P3/P4/P5/P9 toggle independently, and all five
together. Every case/variant runs in a fresh child process so process-global MLX allocator state
cannot leak between comparison rows. P5 uses the same fixed tile geometry as its toggle-free
control, which isolates accumulator mechanics from the separate P9 admission-policy change.

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

The executable embeds the inference HEAD, dirty state, pinned mlx-rs revision, Cargo profile and
optimization level, debug-assertion state, target triple, sorted Cargo and target features,
`RUSTFLAGS`, and full rustc version at build time. Its own SHA-256 is added at runtime. Every command
that can create or accept evidence compares that complete receipt (including the exact executable
bytes) with the frozen campaign, runtime checkout, and lockfile. Acceptance requires the documented
`--release --locked --no-default-features --features perf-bench` build for
`aarch64-apple-darwin`, with no custom rustflags. Build from a clean committed checkout; changing or
committing source requires a fresh binary. A runtime-only `git rev-parse` cannot substitute for
executable provenance.

## Frozen campaigns and execution

Use an idle Apple-Silicon Mac and an empty absolute output directory outside the checkout:

```sh
cargo run --release --locked -p runtime-macos --no-default-features --features perf-bench \
  --bin mlx-perf-bench -- run --artifacts /absolute/path/mlx-perf-artifacts.json \
  --output-dir /absolute/path/empty-results-directory
```

Before starting any child, the parent creates one private snapshot of each exact artifact and then
writes `campaign.json`. It contains canonical copies and hashes of the full matrix and artifact
manifest, exact artifact inventories, selected variants, build and host identities, and
provider-owned toggle capability declarations. Its campaign ID binds all of that state. Snapshot
files are cloned copy-on-write from already-open file descriptors when the local filesystem supports
it, with a safe full-copy fallback from those same descriptors. File symlinks are materialized as
independent regular files; directory symlinks and special entries are refused. Every completed tree
must match its frozen inventory, use independent file identities, and accept macOS's user-immutable
seal before any child starts. Children load only from the sealed private path, verify its content and
path identities before loading and immediately before publishing each run record, and bind that
snapshot equivalence into the record. The parent keeps each snapshot alive across all rows that use
it, then verifies and removes every snapshot before it can publish `summary.json`. Changed, mixed,
stale, wrong-build, wrong-host, unsealable, or uncleanable state is refused. Each private tree also
has a durable harness-owned lease. Every benchmark child inherits that same locked lease, so parent
termination cannot make a tree look stale while a child is still loading or publishing a record.
The lease is fully staged and synced before atomic publication, binds the exact root device and
inode, and is paired with a full raw-path cleanup identity manifest published before sealing. At the
next `run`, the harness leaves live leases untouched and scavenges only unlocked stale trees whose
ownership and cleanup identities validate. Cleanup walks already-open directory descriptors in
post-order and revalidates device, inode, and link count immediately before each flag, mode, or
unlink mutation; malformed, foreign, symlinked, multiply-linked, or rebound state is left untouched
and fails closed. This recovers trees left by signals, aborts, OOM kills, or other parent termination
without clearing flags through an unrelated path.

The default required-all campaign fails at preflight until every benchmark provider explicitly
declares every requested capability. Declaration is only availability, never proof that a path ran.
Every warmup and measured request must independently emit exactly one aggregated terminal
`ToggleDisposition::Applied` record for each toggle that can execute under the request's actual
physical decode path, no unrequested terminal record, and no fallback or unavailable outcome. The
one conditional is `all_on`: a dense-path P9 receipt forbids P5's terminal receipt because its tiled
accumulator did not execute, while a tiled-path receipt requires P5 `Applied`. The physical path is
separate from P9's semantic decision because Wan can preserve its pre-existing production policy
(`unchanged`) while that policy auto-tiles the request. The dedicated fixed-tile P5 row always
requires its own `Applied` receipt. Baseline and the fixed-tile control forbid all toggle terminal
records.

A baseline-only campaign remains available while optimization call sites are being integrated:

```sh
cargo run --release --locked -p runtime-macos --no-default-features --features perf-bench \
  --bin mlx-perf-bench -- run --artifacts /absolute/path/mlx-perf-artifacts.json \
  --output-dir /absolute/path/empty-baseline-directory --variants baseline
```

Baseline-only, partial, and custom-matrix campaigns are diagnostic runs, not acceptance evidence. A
canonical required-all campaign is refused unless it uses the documented acceptance build.
`summary.json` sets `acceptanceComplete` only for the exact committed matrix, exact required-all
selection, and documented acceptance build. `--matrix` remains useful for diagnosis, but changing
even a prompt, seed, step count, geometry, artifact, or variant contract cannot produce acceptance
evidence. Merging this infrastructure does not complete sc-18321: the story stays open until
P1/P3/P4/P5/P9 are integrated and the full nine-case required-all comparison is successfully
captured and reviewed.

## Evidence semantics

Providers emit explicit `DenoiseStart` and `DecodeStart` diagnostics at production code boundaries.
The runner does not infer phases from UI progress. It requires exactly ordered boundaries; exact
`Progress::Step` receipts `1..N` inside the denoise interval; one later `Progress::Decoding`; no
post-decode steps; and stage durations whose encode + denoise + decode intervals cover the measured
request. Denoise throughput uses all configured steps.

Each phase owns a 10 ms background allocator probe with immediate, periodic, and final samples.
Validation rejects a phase whose observed inter-sample gap exceeds 30 ms, so the tens-of-
milliseconds P5 allocation transients cannot be hidden behind the former coarse cadence.
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
SHA-256 over the produced bytes. Digests must be stable across repetitions. P1/P3/P4 compare exactly
with baseline; P5 compares exactly with the fixed-tile control; and `all_on` compares exactly with
P9. P9 itself may differ from baseline only because its quality-admitted tiled policy is permitted
to change output bytes. P9 and `all_on` must each emit exactly one stable `decode_policy` decision:
`unchanged` stays byte-identical to baseline, while `geometry_tiled` may drift only when it carries
the lower-hex SHA-256 identity of the production evidence that admitted tiling. Each receipt also
records the physical `dense`/`tiled` decoder path; `geometry_tiled` requires `tiled`, while
`unchanged` may be either. P9 and `all_on` must match on the decision, physical path, and evidence
identity. The harness does not invent a permissive image-distance threshold; all outputs still must
be byte-stable across repetitions and emit their required applied-toggle diagnostics.

Validate an existing result directory without rewriting it:

```sh
cargo run --release --locked -p runtime-macos --no-default-features --features perf-bench \
  --bin mlx-perf-bench -- validate-results --results-dir /absolute/path/results-directory
```

Legacy v1 directories have no frozen campaign and are rejected as unbound evidence.
Run records without the real stored `summary.json` are also rejected: that file is the parent's
finalization marker and is published only after every private snapshot was reverified and removed.
