# Phase 5 Consumer Cutover Closeout

Date: 2026-07-14

> **Status:** Canonical release and SceneWorks cutover complete. ChatWorks PR
> [#31](https://github.com/SceneWorks/ChatWorks/pull/31) is release-pinned, validated, and pending
> product-owner review. Phase 5 closes when that product commit lands.

## Authority decision

`SceneWorks/inference` is the canonical inference source, integration boundary, and release train.
Once the final ChatWorks cutover lands, new contract, MLX, Candle, media-provider, LLM-provider,
catalog, and runtime-bundle work originates here and is fixed forward here.

The five imported repositories remain frozen at the reconciled heads below as rollback and history
sources. This decision does not archive them, change their visibility, disable workflows, or delete
branches. Those administrative actions require separate approval.

## Canonical release boundary

Both products select named bundles from one immutable release:

```text
repository = https://github.com/SceneWorks/inference
tag        = runtime-2026.07.2
commit     = 27d7908de401ce9b270d7e53e87f717fee151b23
```

The release contains 74 path-owned workspace packages and one lockfile. Active workspace metadata
contains no internal legacy Git dependency. Provider composition is explicit; active runtime code
contains no `inventory` submission or force-link discovery seam.

The published release has a 365,512,487-byte source archive, a 455-package SPDX document, a clean
runtime manifest, and verified checksums. Offline verification built an external consumer against
the extracted neutral contracts. Full hosted/self-hosted CI, NAX, Windows CUDA, and all four MLX /
Candle-CUDA real-weight jobs passed at the exact release commit.

## Consumer state

- SceneWorks PR [#1512](https://github.com/SceneWorks/SceneWorks/pull/1512) merged as
  `8ad8c00178670b4d08c1171561c420aeb3fb5166`. PR and post-merge web, parity, Windows/CUDA,
  macOS/NAX, Docker/API, and Windows NSIS/MSI packaging gates passed. Manual Linux/NVIDIA server
  run [29338546768](https://github.com/SceneWorks/SceneWorks/actions/runs/29338546768) also passed
  strict Candle Clippy, release server build, and artifact upload.
- ChatWorks PR [#31](https://github.com/SceneWorks/ChatWorks/pull/31) is clean and mergeable at
  `bdcca7d443fdb59aa99511210a7399ac95d5d5b3`. Its pin gate, strict Clippy, 79 Rust tests, locked
  native build, frontend lint, and production web build passed.
- Resolved Cargo metadata contains exactly one `core-llm`, one `sceneworks-gen-core`, one canonical
  inference source identity, and no legacy inference source in each product graph.

The product-side rationale, old release set, private-source credential boundary, detailed evidence,
and rollback procedure live in SceneWorks'
[`PHASE_5_CUTOVER.md`](https://github.com/SceneWorks/SceneWorks/blob/main/documents/rearchitecture/PHASE_5_CUTOVER.md).

## Reconciliation and published refs

The current legacy remote heads equal their final recorded cutoffs:

| Repository | Final reconciled head |
|---|---|
| `core-llm` | `54cbac806304e823470ce3ded08f78589acdbb62` |
| `mlx-llm` | `af5d0e83f4afb921241b7e965076e22f23c107fb` |
| `candle-llm` | `482ba5e5e99770967dd0c912f7f87b31c6f08576` |
| `mlx-gen` | `45428fa9727c569f3f3723c7343c96b0944f9007` |
| `candle-gen` | `ef84441c82222df8f63701e761c0834c1699ceb0` |

There is no unrecorded post-import delta. The canonical remote retains:

- annotated `migration-baseline/{core-llm,mlx-llm,candle-llm,mlx-gen,candle-gen}` tags;
- annotated `history/candle-gen-tracking-main`;
- MLX/Candle product-cutoff and post-cutover history branches; and
- immutable `runtime-2026.07.0`, `runtime-2026.07.1`, and `runtime-2026.07.2` release tags and
  candidate tags.

The sequential-residency work that landed after `runtime-2026.07.1` is mapped and validated in
[`RUNTIME_2026_07_2_CHECKPOINT.md`](RUNTIME_2026_07_2_CHECKPOINT.md) and
[`post-cutover-reconciliation-epic-10834.toml`](post-cutover-reconciliation-epic-10834.toml).

## Rollback proof

Rollback was exercised in disposable worktrees on 2026-07-14:

- reverting the SceneWorks cutover merge applied without conflict on current `main`; the restored
  legacy graph fetched and the skew gate resolved one `sceneworks-gen-core` at final MLX cutoff
  `45428fa9727c569f3f3723c7343c96b0944f9007`;
- reversing the two ChatWorks cutover commits reproduced its pre-cutover `main` exactly; the legacy
  three-source pin gate and `cargo check --workspace --locked` passed.

Rollback is therefore a product commit, not a mutable inference tag. Preserve the migration
checkpoints, revert the affected product integration, validate the old graph, then fix forward here
and issue a new runtime tag before retrying the cutover.

## Deferred scope

Provider-family relocation, product-repository consolidation, shared UI work, legacy-repository
archival, workflow disabling, and visibility changes are outside Phase 5. None is required for the
canonical inference migration to be complete.
