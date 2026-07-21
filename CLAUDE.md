# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SceneWorks Inference is a history-preserving Cargo **workspace** (one lockfile; its member count is
enforced by `scripts/check-workspace.py`)
assembled from five former repositories: `core-llm`, `mlx-llm`, `candle-llm`, `mlx-gen`, and
`candle-gen`. It holds backend-neutral inference contracts, the MLX and Candle engines, media/LLM
provider families, weights-free conformance suites, and named platform runtime bundles. Rust
toolchain is pinned to **1.96.0** via `rust-toolchain.toml`.

## Build & test

There is **no universal `--workspace --all-features` build** — MLX, CUDA, and CPU are mutually
exclusive platform targets, not additive features. Build/test per platform lane instead:

```sh
# Contracts + conformance testkits (any platform)
cargo test --locked -p core-llm -p core-llm-testkit -p sceneworks-gen-core -p sceneworks-gen-core-testkit

# Candle CPU lane (Linux/any)
cargo clippy --locked -p candle-llm -p 'candle-gen*' -p 'candle-audio*' -p runtime-cpu --all-targets -- -D warnings
cargo test --locked --lib -j 1 -p candle-llm -p 'candle-gen*' -p 'candle-audio*' -p runtime-cpu   # -j 1 avoids lld OOM
cargo test --locked -j 1 -p candle-llm --test conformance

# MLX + Candle Metal lane (macOS only)
cargo test --locked --lib --tests -p mlx-llm -p mlx-llm-server -p mlx-gen -p 'mlx-gen-*' -p runtime-macos

# Candle CUDA lane (Windows self-hosted; needs vcvars64 + CUDA_COMPUTE_CAP)
# CUDA 12.9's nvcc rejects MSVC 14.51+ — use the VS2022 BuildTools vcvars64, not a VS18 one.
cargo clippy --locked -p candle-llm -p 'candle-gen*' -p 'candle-audio*' -p runtime-cuda --all-targets --features cuda -- -D warnings
cargo test --locked --lib --tests -p candle-llm -p 'candle-gen*' -p 'candle-audio*' -p runtime-cuda --features cuda

# LLM-only composition profile of any bundle (no media provider graph compiled)
cargo test --locked --no-default-features --lib -p runtime-cpu   # or runtime-macos / runtime-cuda
# LLM + Candle audio lane, no media graph (the sc-12835 additive `audio` feature)
cargo test --locked --no-default-features --features audio --lib -p runtime-cpu
```

Run a **single test**: `cargo test --locked -p <crate> <test_name>`
(e.g. `cargo test --locked -p mlx-llm --test conformance real_model_passes_core_llm_conformance -- --ignored`).
Real-weight tests are `#[ignore]`d and gated behind snapshot env vars — see `.github/workflows/real-weights.yml`.

`cargo test` runs **single-threaded** by default here (`.cargo/config.toml` forces
`RUST_TEST_THREADS=1` with `force = true`): MLX's shared Metal device is not thread-safe and
parallel tests SIGSEGV. Do not remove or override this.

### Repository gates (Python 3, no deps)

```sh
./scripts/check-workspace.py                          # graph invariants (see below)
python3 -m unittest discover -s scripts/tests -v      # tooling unit tests
python3 scripts/check_docs.py                         # local doc-link check
cargo deny --locked check advisories bans licenses sources   # supply-chain policy (deny.toml)
```

`check-workspace.py` is the enforcement point for the architecture: it asserts the
`EXPECTED_MEMBER_COUNT` of path members, one root `Cargo.lock`, one `[workspace]` manifest, that all internal deps are path edges
(not SHA pins), that pinned MLX/Candle backend revisions are unchanged, that `inventory` is absent
from the resolved graph, and that the intentional `tokenizers` 0.21/0.22 split is preserved. If you
add or remove a crate, update `EXPECTED_MEMBER_COUNT`.

It also enforces the **epic-13657 self-fetch boundary**: no network/HTTP client
(`hf-hub`, `reqwest`, `ureq`, `curl`, `git2`, `hyper`, …) may resolve in the graph, and no Rust of
any member — src, tests, examples, testkits — may reference an HF download cache (`HF_HOME`,
`HF_HUB_CACHE`, `.cache/huggingface`, `hf_hub`, `Api::new`) or re-introduce a deleted production
env side channel (`PERTH_SNAPSHOT`, `MOSS_XY_TOKENIZER_SNAPSHOT`, `LTX_GEMMA_DIR`, …). Inference
**receives every model component as a caller-provisioned local path** (`WeightsSource::Dir`/`File`);
fetching and cache placement are the consumer's job, and user-supplied models at arbitrary paths
must load. `deny.toml` bans the same network clients for defense in depth. Explicit passed-in-path
test env vars (`MLX_LLM_TEST_MODEL`, per-crate `*_SNAPSHOT`/`*_SNAPSHOT_DIR`) stay allowed — the
lint targets cache-location *derivation*, not passed-in paths.

## Architecture — explicit composition (the core invariant)

Read `docs/architecture/inference-rearchitecture.md` before changing composition. Dependency
direction is strictly one-way:

```
backend-neutral contracts  →  MLX/Candle engines  →  provider-family crates
  →  platform catalog crates  →  application composition root (one immutable ProviderRegistry)
```

- `crates/contracts/` — `core-llm` (LLM contracts) and `sceneworks-gen-core` (media contracts),
  plus their `*-testkit` conformance suites. **These stay tensor-neutral** — never depend on MLX or
  Candle tensor types here.
- `crates/llm/` — `mlx-llm` (+ `server`) and `candle-llm` engines.
- `crates/media/` — `mlx-gen` / `candle-gen` engines and provider families, plus the
  `mlx-gen-catalog` / `candle-gen-catalog` composition roots.
- `crates/audio/` — the Candle-native audio family (`candle-audio` commons + the
  `candle-audio-catalog` composition root). Candle on **every** platform — on `runtime-macos` it
  rides the catalog's dedicated audio lane beside the mlx media graph, the one sanctioned
  cross-backend seam (`docs/architecture/audio-backend-strategy.md`).
- `crates/bundles/` — `runtime-macos`, `runtime-cuda`, `runtime-cpu` (each assembles one media +
  one LLM + one snapshot-preparer registry, plus the candle audio lane) validated by
  `runtime-catalog`.

**Provider registration is explicit, not linker-discovered.** There is deliberately no
process-global provider state. Provider crates publish named registration constants; family crates
expose `register_providers(builder) -> builder` functions; catalog crates call those in stable order
and expose `provider_registry()`. A provider crate existing in the repo does **not** mean it ships —
catalog inclusion is a deliberate edit. `check-workspace.py` forbids the `inventory` crate; **never
reintroduce `inventory` submissions or `force_link` anchors** as a shortcut. Adding a provider means:
export its registration → add its family to the right platform catalog → update that catalog's
ordered surface test.

Bundles carry a `default = ["media", "audio"]` feature set. SceneWorks consumes the full bundle;
LLM-only products (ChatWorks) build with `--no-default-features` to get the same explicit
LLM/preparer catalog without the media graph, optionally adding `--features audio` for the Candle
audio lane without the image/video graph. Both consume a **named `runtime-*` bundle**, never
assembled backend crates.

## Compatibility boundaries (treat as breaking changes)

Per `CONTRIBUTING.md`: do not rename public crates, provider IDs, serialized fields, or weight keys
as incidental cleanup. MLX and Candle catalog surfaces are allowed to differ when that reflects a
real implementation difference — pin the difference explicitly in the catalog surface tests rather
than papering over it.

## Platform build notes

`.cargo/config.toml` sets `MACOSX_DEPLOYMENT_TARGET = "26.2"` (unforced) so MLX compiles the correct
Apple matrix-unit "NAX" Metal kernels; hosted macOS CI overrides it lower (SDK 15 can't build 26.2)
and therefore cannot exercise the NAX fast path — a self-hosted macOS 26.2 runner is required for
that. The extensive comments there are load-bearing; read them before touching deployment target or
`RUST_TEST_THREADS`.

## CI & releases

CI (`.github/workflows/ci.yml`) selects lanes from changed paths via `scripts/ci/select_lanes.py` —
new/unclassified top-level paths fail safe to all lanes. Releases are immutable calendar-versioned
`runtime-YYYY.MM.patch[-rc.N]` tags built by `scripts/release/build_release.py` (source archive +
`runtime-manifest.json` + SPDX SBOM + `SHA256SUMS`), verified by `verify_release.py`. Tags are never
moved or reused. Record migration/compatibility evidence and release-boundary decisions under
`docs/migration/`.
