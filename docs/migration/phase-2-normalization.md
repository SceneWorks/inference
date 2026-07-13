# Phase 2 Workspace Normalization Matrix

> **Status:** Normalization implemented locally and validated; see
> `PHASE_2_CHECKPOINT.md`.

## Final Phase 2 ownership paths

```text
crates/contracts/core-llm/   core-llm tree
crates/contracts/gen-core/   generative-media contract
crates/contracts/gen-core-testkit/ generative-media conformance suite
crates/llm/mlx-llm/         mlx-llm tree
crates/llm/candle-llm/      candle-llm tree
crates/media/mlx-gen/       mlx-gen tree
crates/media/candle-gen/    candle-gen tree
```

Moving each root intact preserves all existing relative provider paths. The later
model-first relocation remains a separate phase.

## Imported package inventory

| Source workspace | Packages |
|---|---:|
| `core-llm` | 2 |
| `mlx-llm` | 2 |
| `candle-llm` | 1 |
| `mlx-gen` | 33 |
| `candle-gen` | 29 at the recorded baseline |
| **Total** | **67** |

The earlier static manifest count included excluded/vendor/spike manifests; Cargo
metadata's 67-package set is authoritative for initial workspace membership.

## Dependency normalization decisions

| Dependency | Current forms | Phase 2 treatment |
|---|---|---|
| `sceneworks-gen-core` | MLX path; Candle Git SHA | One root workspace path under the neutral contracts directory. |
| `sceneworks-gen-core-testkit` | MLX path; Candle Git SHA | One root workspace path. |
| `core-llm` | Separate branch-based Git dependencies | Path to relocated contract package. |
| `mlx-llm` | Separate rev Git dependencies | Path to relocated MLX LLM package. |
| `candle-llm` | Two revs in SceneWorks graph | Path to relocated Candle LLM package. |
| `mlx-rs` / `mlx-sys` | Personal-fork URL at one SHA | Keep one root workspace dependency at the existing SHA until canonical tag work. |
| `inventory` | Version 0.3 in all workspaces | One root dependency; retained until explicit-registry phase. |
| `thiserror` | Version 2 | One root dependency. |
| `serde_json` | Version 1 with differing features | Root inheritance stays at base version 1; direct `preserve_order` users retain their explicit feature to avoid changing other packages' map semantics. |
| `tokenizers` | MLX/core 0.21; Candle media 0.22 | Preserve both versions mechanically: root inheritance remains 0.22 for Candle's 14 users; MLX's three inherited users become explicit 0.21 declarations. Contract already declares 0.21 directly. Do not upgrade during migration. |
| Candle crates | One upstream Candle Git SHA | Keep shared root declarations at the exact current SHA. |

## Mechanical sequence

1. Commit the `imports/` to ownership-path moves with no file edits.
2. Verify the moved trees still equal the imported baseline trees.
3. Remove nested `[workspace]` declarations while preserving `[package]`, profiles,
   and dependency tables.
4. Build the root member list from the 67 Cargo-metadata packages.
5. Merge inherited dependency declarations into the root.
6. Convert internal Git dependencies to paths.
7. Preserve the tokenizer 0.21/0.22 split explicitly.
8. Generate one lockfile.
9. Run contract and Candle CPU metadata/check/test lanes before MLX/CUDA work.

The byte-preserving move is commit `9e857956`. Workspace normalization is kept as
the following independently reviewable commit.

## Guardrails

- No provider ID, package name, public Rust path, feature name, or serialized type
  changes.
- No dependency major/minor upgrades to make the workspace merge easier.
- No registry refactor in Phase 2.
- No model-first provider moves until the normalized graph passes platform checks.
- The two tokenizer versions are intentional during normalization.
