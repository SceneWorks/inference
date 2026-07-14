# Documentation

Documentation for the SceneWorks inference workspace.

## Start here

- **[Getting Started](guide/getting-started.md)** — depend on a runtime bundle, build its
  validated registries, and load/run media generators, LLMs, and snapshot preparers. The
  consumer *how-to*.
- **[Model Catalog Reference](reference/model-catalog.md)** — every shipped provider id, per
  platform, with the MLX/Candle deltas. Built from the committed exact-surface tests.

## Architecture

- **[Inference Rearchitecture Rationale](architecture/inference-rearchitecture.md)** — why
  the repositories were consolidated, why provider discovery is explicit, the alternatives
  considered, the accepted tradeoffs, and the invariants future changes must preserve. The
  *why* behind everything above.

## Migration records

The [`migration/`](migration/) directory records how this repository was assembled from the
former `core-llm`, `mlx-llm`, `candle-llm`, `mlx-gen`, and `candle-gen` histories — source
SHAs, filtered-history commit maps, tree-equivalence checks, and per-phase checkpoints. See
[`migration/README.md`](migration/README.md) for the index.

## Crate-level docs

Each layer has a README next to its source:

| Layer | Crates |
| --- | --- |
| Contracts | [`gen-core`](../crates/contracts/gen-core/README.md), [`gen-core-testkit`](../crates/contracts/gen-core-testkit/README.md), [`core-llm`](../crates/contracts/core-llm/README.md) |
| Bundles | [`runtime-catalog`](../crates/bundles/runtime-catalog/README.md), [`runtime-macos`](../crates/bundles/runtime-macos/README.md), [`runtime-cuda`](../crates/bundles/runtime-cuda/README.md), [`runtime-cpu`](../crates/bundles/runtime-cpu/README.md) |
| LLM engines | [`mlx-llm`](../crates/llm/mlx-llm/README.md), [`candle-llm`](../crates/llm/candle-llm/README.md) |
| Media engines | [`mlx-gen`](../crates/media/mlx-gen/README.md) (+ [`mlx-gen-catalog`](../crates/media/mlx-gen/mlx-gen-catalog/README.md)), [`candle-gen`](../crates/media/candle-gen/README.md) (+ [`candle-gen-catalog`](../crates/media/candle-gen/candle-gen-catalog/README.md)) |

For MLX media internals, see
[`mlx-gen/ARCHITECTURE.md`](../crates/media/mlx-gen/ARCHITECTURE.md) and
[`mlx-gen/docs/MODEL_ARCHITECTURE.md`](../crates/media/mlx-gen/docs/MODEL_ARCHITECTURE.md).

## Release

- [`release/README.md`](../release/README.md) — the immutable, calendar-versioned release
  train (`runtime-YYYY.MM.patch` tags), release gates, and bundle contents.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) and [`SECURITY.md`](../SECURITY.md) — contribution
  boundaries and security reporting.
