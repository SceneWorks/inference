# SceneWorks Inference

Unified source repository for SceneWorks' backend-neutral inference contracts,
MLX and Candle engines, model providers, conformance suites, and platform runtime
bundles.

## Migration status

This repository was assembled from the existing `core-llm`, `mlx-llm`,
`candle-llm`, `mlx-gen`, and `candle-gen` histories. Phase 1 preserved and
verified those histories; Phase 2 moved the imported packages into their ownership
paths and normalized them under one Cargo workspace and lockfile. Phase 3
provides dependency-aware CI selection, supply-chain policy, immutable real-weight
fixture pins, and deterministic source/SBOM release tooling.
Phase 4 adds validated `runtime-macos`, `runtime-cuda`, and `runtime-cpu` composition boundaries for
explicit media, LLM, and snapshot-preparation catalogs; the former link-time provider registries
have been removed. Phase 5 published immutable release `runtime-2026.07.0` and cut the SceneWorks
and ChatWorks lockfiles over to its exact commit. Local product validation is complete; hosted
SceneWorks validation requires its scoped credential for this private repository.

After the cutover, the legacy MLX generation repository received one further epic
(10834, sequential component residency). That post-cutover drift was reconciled into
the monorepo and published as `runtime-2026.07.2`; see
[`docs/migration/RUNTIME_2026_07_2_CHECKPOINT.md`](docs/migration/RUNTIME_2026_07_2_CHECKPOINT.md).
Legacy development is halted pending the durable decision on whether the monorepo is
the sole live development line.

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
the published release-train evidence, and
[`docs/migration/PHASE_4_CHECKPOINT.md`](docs/migration/PHASE_4_CHECKPOINT.md) for
the explicit bundle composition and platform results.

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
