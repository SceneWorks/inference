# SceneWorks Inference

Unified source repository for SceneWorks' backend-neutral inference contracts,
MLX and Candle engines, model providers, conformance suites, and platform runtime
bundles.

## Migration status

This repository was assembled from the existing `core-llm`, `mlx-llm`,
`candle-llm`, `mlx-gen`, and `candle-gen` histories. Phase 1 preserved and
verified those histories; Phase 2 moved the imported packages into their ownership
paths and normalized them under one Cargo workspace and lockfile. Phase 3 now
provides dependency-aware CI selection, supply-chain policy, immutable real-weight
fixture pins, and deterministic source/SBOM release tooling.
Phase 4 adds validated `runtime-macos`, `runtime-cuda`, and `runtime-cpu` composition boundaries for
explicit media, LLM, and snapshot-preparation catalogs; the former link-time provider registries
have been removed. Phase 5 is the immutable release and final product-pin validation.

The current authoritative migration plan and release-set baseline live in the
SceneWorks repository under `documents/rearchitecture/`.

## Layout

```text
crates/contracts/  Backend-neutral contracts and conformance suites
crates/bundles/    Named supported platform compositions and catalog validation
crates/llm/        MLX and Candle LLM engines
crates/media/      MLX and Candle media engines and provider families
docs/              Migration maps, architecture, compatibility, and release records
```

See [`docs/migration/PHASE_2_CHECKPOINT.md`](docs/migration/PHASE_2_CHECKPOINT.md)
for normalization invariants and
[`docs/migration/PHASE_3_CHECKPOINT.md`](docs/migration/PHASE_3_CHECKPOINT.md) for
the local release-train validation record, and
[`docs/migration/PHASE_4_CHECKPOINT.md`](docs/migration/PHASE_4_CHECKPOINT.md) for
the bundle composition and remaining platform gates.

The architectural rationale—including why the repositories were consolidated,
why provider discovery is explicit, the alternatives considered, and the
tradeoffs accepted—is recorded in
[`docs/architecture/inference-rearchitecture.md`](docs/architecture/inference-rearchitecture.md).

Contribution boundaries and security reporting are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).

Validate the normalized graph with:

```sh
./scripts/check-workspace.py
```
