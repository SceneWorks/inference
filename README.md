# SceneWorks Inference

Unified source repository for SceneWorks' backend-neutral inference contracts,
MLX and Candle engines, model providers, conformance suites, and platform runtime
bundles.

## Migration status

This repository was assembled from the existing `core-llm`, `mlx-llm`,
`candle-llm`, `mlx-gen`, and `candle-gen` histories. Phase 1 preserved and
verified those histories; Phase 2 has moved all 67 packages into their ownership
paths and normalized them under one Cargo workspace and lockfile.

The current authoritative migration plan and release-set baseline live in the
SceneWorks repository under `documents/rearchitecture/`.

## Layout

```text
crates/contracts/  Backend-neutral contracts and conformance suites
crates/llm/        MLX and Candle LLM engines
crates/media/      MLX and Candle media engines and provider families
docs/              Migration maps, architecture, compatibility, and release records
```

See [`docs/migration/PHASE_2_CHECKPOINT.md`](docs/migration/PHASE_2_CHECKPOINT.md)
for the normalization invariants and validation record.
