# SceneWorks Inference

Unified source repository for SceneWorks' backend-neutral inference contracts,
MLX and Candle engines, model providers, conformance suites, and platform runtime
bundles.

## Migration status

This repository is being assembled from the existing `core-llm`, `mlx-llm`,
`candle-llm`, `mlx-gen`, and `candle-gen` histories. During Phase 1, imported
sources remain under `imports/` and must stay behaviorally identical to their
recorded source SHAs. Workspace normalization begins only after imported tree and
commit-map verification passes.

The current authoritative migration plan and release-set baseline live in the
SceneWorks repository under `documents/rearchitecture/`.

## Provisional layout

```text
imports/      Byte-preserving filtered-history imports; temporary layout
docs/         Migration maps, architecture, compatibility, and release records
crates/       Target location for normalized contracts/backends/providers/bundles
```

No package in `imports/` is published from this repository during Phase 1.

