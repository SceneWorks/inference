# Documentation

Documentation for the SceneWorks inference workspace.

## Start here

- **[Getting Started](guide/getting-started.md)** — depend on a runtime bundle, build its
  validated registries, and load/run media generators, LLMs, and snapshot preparers. The
  consumer *how-to*.
- **[Model Catalog Reference](reference/model-catalog.md)** — every shipped provider id, per
  platform, with the MLX/Candle deltas. Built from the committed exact-surface tests.
- **[Mochi 1 tier strategy](reference/mochi-1-tier-strategy.md)** — the accepted quantization
  packaging decision for the native Mochi 1 port (pre-quantized per-tier artifacts, default q4).
- **[MiniMax-H3 withheld upstream components](reference/minimax-h3-withheld-upstream-components.md)**
  — what `H3-Context-IR`, `H3-Regenerate-2K`, sparse-attention inference and the `<d>` dialogue
  markers cost this port, what the crates do instead, and what would change if upstream publishes.

## Architecture

- **[Inference Rearchitecture Rationale](architecture/inference-rearchitecture.md)** — why
  the repositories were consolidated, why provider discovery is explicit, the alternatives
  considered, the accepted tradeoffs, and the invariants future changes must preserve. The
  *why* behind everything above.
- **[Audio Backend Strategy](architecture/audio-backend-strategy.md)** — why audio generation
  is Candle-native on every platform (no ONNX/third backend), and how the runtime catalog's
  dedicated audio section carries a `candle` audio lane inside the mlx macOS bundle.

## Licensing

The [`licensing/`](licensing/) directory holds the primary-source evidence behind the model-weight
licence surface — the canonical text URL, the verbatim upstream identifier, and a quoted operative
clause for every term assigned to a licence family.

- **[Licence family evidence pack (sc-16662)](licensing/sc-16662-licence-family-evidence.md)** —
  the licence families, quote-checkable, with an explicit unresolved list. **Draft, unsigned.** It
  records facts, not legal conclusions.
- **[Media checkpoint census (sc-16665)](licensing/sc-16665-media-checkpoint-census.md)** — which
  upstream checkpoints every registered media provider actually loads, read from the code. Records
  component identity only: it assigns no families and asserts no licences, and marks every component
  whose upstream the repository does not state as UNDETERMINED.
- **[Checkpoint licence evidence (sc-16665)](licensing/sc-16665-checkpoint-licence-evidence.md)** —
  the primary-source licence read for each of those checkpoints, and the sign-off document behind
  `license::components`. **Draft, unsigned.** Its
  [known holes](licensing/sc-16665-checkpoint-licence-evidence.md#known-holes--the-rows-sc-16665-deliberately-did-not-write)
  section is the authoritative list of checkpoints that were left without a row rather than given a
  guessed one.

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
