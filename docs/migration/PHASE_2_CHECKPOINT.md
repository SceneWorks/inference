# Phase 2 Workspace Normalization Checkpoint

> **Status:** Complete locally; remote publication remains deferred.

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
- contract/testkit library tests: 357 passed, 0 failed, 3 ignored
- `git diff --check`

This checkpoint validates the normalized dependency seams and representative
backend roots. It does not claim all 67 provider packages or platform-specific
Metal/CUDA feature matrices have been tested; those lanes follow before provider
relocation.

## Next migration slice

1. Add root-owned graph-skew and workspace-structure gates.
2. Port CI into platform lanes that invoke the root workspace explicitly.
3. Run all-package metadata/check coverage on the supported host matrix.
4. Only then begin model-first provider relocation and compatibility shims.
