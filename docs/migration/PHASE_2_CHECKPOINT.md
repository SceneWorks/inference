# Phase 2 Workspace Normalization Checkpoint

> **Status:** Complete and published. The normalization commits are part of the
> canonical `SceneWorks/inference` history.

## Result

The five imported source trees now live at their ownership paths beneath
`crates/`. The repository has one active Cargo workspace, one root lockfile, one
root Rust toolchain pin, and one root Cargo configuration.

The source relocation was committed separately as `9e857956`. It moved 2,303
files with zero insertions or deletions. The committed subtree IDs still match
the Phase 1 import baselines:

| Ownership path | Tree ID |
|---|---|
| `crates/contracts/core-llm` | `e115397e83414217c3384698318522f6f4e8593a` |
| `crates/llm/mlx-llm` | `4e19df11b186c864879ed871b55e55eeb147335b` |
| `crates/llm/candle-llm` | `40ec531d2d6735930227a1e84175e8c894726add` |
| `crates/media/mlx-gen` | `a8cf8de2713df6ff9a426ee40018fedfb64b3809` |
| `crates/media/candle-gen` | `b610bdf1565ceb810e7b938122473cca8b785b83` |

## Workspace invariants

- 67 packages are workspace members: 2 contract, 3 LLM, 33 MLX media, and
  29 Candle media packages.
- `core-llm`, `core-llm-testkit`, `mlx-llm`, `candle-llm`,
  `sceneworks-gen-core`, and `sceneworks-gen-core-testkit` all resolve from local
  paths. No active internal Git dependency remains.
- `pmetal-mlx-rs` resolves once at
  `38e1cc1730a11b1e40c2c8ecda01606763a12d51`.
- `candle-core` resolves once at
  `c1e6756a89faefa888ea57b056394a0619925b87`.
- The existing tokenizer compatibility split is preserved: MLX/contracts use
  0.21 and Candle media uses 0.22. No tokenizer upgrade is part of this phase.
- The vendored Candle CUDA kernel patch remains rooted at
  `crates/media/candle-gen/vendor/candle-kernels`.
- The former Candle root workspace manifest is retained as the inactive
  `Cargo.legacy-workspace.toml` for pin-rationale review during migration. Cargo
  does not discover it as an active workspace.

## Root configuration

Member-local Cargo configuration is not discovered when commands run from a
monorepo root. The MLX safety settings were therefore hoisted to
`.cargo/config.toml`: tests are single-threaded and local macOS builds default to
the 26.2 deployment target required by the NAX kernels. CI may continue to
override the deployment target. The shared Rust 1.96.0 pin was already present at
the repository root; redundant member pins were removed.

## Validation

The following passed from the repository root:

- `cargo metadata --no-deps --format-version 1`
- `cargo generate-lockfile`
- `cargo metadata --offline --locked --format-version 1`
- contract/testkit `cargo check --locked`
- Candle roots: `cargo check --locked -p candle-llm -p candle-gen`
- MLX roots: `cargo check --locked -p mlx-llm -p mlx-gen`
- all Candle packages: `cargo check --locked -p candle-llm -p 'candle-gen*'`
- all MLX packages: `cargo check --locked -p mlx-llm -p mlx-llm-server -p mlx-gen -p 'mlx-gen-*'`
- all contract, Candle, and MLX package selectors pass Clippy with `-D warnings`
- all default Candle package tests pass; weight/GPU-dependent cases remain ignored by their existing gates
- all default MLX package tests pass; real-weight and performance cases remain ignored by their existing gates
- all Candle packages check with `--features metal` on the local macOS host
- contract/testkit library tests: 357 passed, 0 failed, 3 ignored
- `git diff --check`

The all-Candle check exposed one additive contract delta between the imported
MLX head and Candle's recorded gen-core pin: `Capabilities` had gained
`supports_sequential_offload`. All 24 Candle descriptors now declare the bit;
FLUX, FLUX.2, and Qwen Image advertise `true` because they already implement the
sequential policy, while unwired providers explicitly advertise `false`.
The same contract commit added `Progress::Loading`; Candle's exhaustive example
callbacks now display that phase so every `--all-targets` build remains compatible.

This checkpoint validates the normalized dependency seams, checks, Clippy, and
default tests for every workspace package on the local macOS host. The local
Candle Metal compile lane also passes. Hosted platform CI and the self-hosted
Windows/CUDA lane remain separate gates before provider relocation.

## Next migration slice

1. Run the root-owned graph-skew and workspace-structure gate in CI.
2. Run the root CI platform lanes and repair any host-specific failures.
3. Validate the manual CUDA lane on its self-hosted Windows runner.
4. Only then begin model-first provider relocation and compatibility shims.

The first item is implemented by `scripts/check-workspace.py`; it asserts the
member count, path-only internal edges, single workspace/lockfile, exact backend
Git revisions, and intentional tokenizer split from Cargo metadata.

The root `.github/workflows/ci.yml` owns the consolidated CI definition. It
partitions backend-neutral contracts, Candle CPU, macOS MLX/Metal, and manual
self-hosted Windows/CUDA work so `--workspace` cannot pull an unsupported backend
onto the wrong host. Historical member-local workflow files were removed.
