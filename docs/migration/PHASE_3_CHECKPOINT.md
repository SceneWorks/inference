# Phase 3 Release Train Checkpoint

> **Status:** Complete. Immutable release `runtime-2026.07.0`, hosted Linux/macOS,
> self-hosted CUDA/NAX, real-weight execution, tagged artifacts, SBOM, and external-consumer
> validation are published.

## Result

The unified repository now owns its CI selection, dependency policy, real-weight
fixture identities, and deterministic release artifacts. A change is classified
once and expanded through downstream ownership before jobs are scheduled:

- contract changes exercise contracts plus both backend/platform families;
- Candle changes exercise Linux CPU, macOS Metal, and manual Windows CUDA;
- MLX changes exercise macOS MLX/Metal;
- root graph/toolchain/workflow changes fail safe to every lane;
- documentation and release-policy changes can use their focused gates.

`scripts/ci/select_lanes.py` is covered by repository-owned unit tests. Unknown
top-level paths deliberately select every lane until they receive an explicit
classification.

## Release artifacts

Runtime tags use `runtime-YYYY.MM.patch[-rc.N]`. The release builder derives all
identity from a committed revision and emits:

- a deterministic `tar.gz` source archive with a single versioned root;
- `runtime-manifest.json` with 74 workspace packages, the Rust toolchain, lockfile
  hash, and exact Candle/MLX Git revisions;
- an SPDX 2.3 JSON SBOM containing all 455 cross-platform lockfile packages and
  1,608 describe/dependency relationships;
- `SHA256SUMS` covering the manifest, source archive, and SBOM.

The clean dry run `runtime-2026.07.0-rc.0` at
`4e2d4b48764908e562705c48afa4901d9dc67534` produced a 364,316,569-byte source
archive. Its verification safely extracted the archive and compiled an external
Cargo consumer against the archived `core-llm` and `sceneworks-gen-core` paths.
The dry-run tag was an artifact label only; no Git tag was created.

The subsequent clean metadata/SBOM release gate `runtime-2026.07.0-rc.1` at
`05584c6b13e08f653fe9896422ca22dd618ede98` passed with the current 74-package
workspace and 455-package cross-platform lockfile. It likewise created no Git tag.

## Supply-chain policy

`deny.toml` restricts dependencies to the crates.io registry and the two approved
Git sources: the pinned Candle repository and the separate `mlx-rs` fork. The
license allowlist passes the complete configured platform graph.

The advisory gate contains two narrow baseline exceptions:

- `RUSTSEC-2025-0119`: unmaintained `number_prefix` through
  `hf-hub 0.4.3`/`indicatif`, with no safe compatible upgrade;
- `RUSTSEC-2024-0436`: unmaintained `paste` through the pinned Candle/GEMM graph,
  also with no safe in-place upgrade.

Both are maintenance notices rather than known vulnerabilities. Their comments
name the upstream refresh that must remove each exception.

## Real-weight fixture policy

Nightly/manual real-weight jobs use persistent runner caches and reject a moving
branch identity. A missing or incomplete snapshot is materialized on demand from
the exact repository revision before verification; revision drift is rejected
rather than overwritten. The initial representative models are fixed to:

| Profile | Repository | Revision |
|---|---|---|
| small LLM | `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| thinking LLM | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| media | `Tongyi-MAI/Z-Image-Turbo` | `f332072aa78be7aecdf3ee76d5c247082da564a6` |

The materializer uses pinned `huggingface_hub` 1.20.1. The verifier accepts the
standard Hugging Face `snapshots/<revision>` layout or a materialized snapshot
carrying `.sceneworks-model-revision`, and checks required files before any
expensive test begins. Model bytes stay outside the Actions workspace and are
reused by later self-hosted runs.

The repository variables are the runner contract: `MLX_LLM_TEST_MODEL`,
`MLX_LLM_QWEN3_MODEL`, and `ZIMAGE_SNAPSHOT` locate persistent macOS caches;
`CANDLE_LLM_TEST_MODEL`, `CANDLE_LLM_QWEN3_MODEL`, and `Z_IMAGE_SNAPSHOT` locate
the Windows caches. Self-hosted Windows jobs bootstrap the repository's pinned
Rust toolchain through a commit-pinned setup action into the Actions service
account, so service restarts do not depend on an interactive user's `PATH`.

## Local validation

The following pass from the repository root:

- 20 repository-tooling unit tests;
- dependency-aware lane-selection examples for contracts, MLX, Candle, docs, and
  unknown paths;
- documentation local-link validation;
- YAML parsing for both workflow definitions;
- `cargo deny --locked check advisories bans licenses sources`;
- deterministic manifest/SBOM generation and checksum verification;
- safe source-archive extraction and external-consumer compilation;
- the Phase 2 workspace gate and `git diff --check`.

GitHub Actions run [`29272581805`](https://github.com/SceneWorks/inference/actions/runs/29272581805)
passed the complete hosted matrix at `05584c6b13e08f653fe9896422ca22dd618ede98`: repository
documentation, contracts, Linux Candle CPU, macOS MLX/Candle Metal, dependency policy, workspace
invariants, and release metadata/SBOM.

## Publication outcome

- Exact-commit CI run
  [`29284987010`](https://github.com/SceneWorks/inference/actions/runs/29284987010) passed hosted
  Linux/macOS plus the self-hosted NAX and Windows/CUDA suites at
  `48cc2d87e14de0189ac4f7763fddc0a8581c2e68`.
- Real-weight run
  [`29285222380`](https://github.com/SceneWorks/inference/actions/runs/29285222380) passed MLX and
  Candle LLM/media execution against all three pinned snapshots.
- Tag CI run
  [`29293208430`](https://github.com/SceneWorks/inference/actions/runs/29293208430) passed the hosted
  matrix, deterministic tagged archive, checksums, SPDX SBOM, and extracted external consumer.
- GitHub Release
  [`runtime-2026.07.0`](https://github.com/SceneWorks/inference/releases/tag/runtime-2026.07.0)
  contains the four exact CI-produced assets. Downloaded `SHA256SUMS` verification passes, and the
  tag dereferences to the exact commit above.

Product cutover is recorded in the SceneWorks Phase 5 checkpoint. Provider-registry refactoring is
complete in the explicit runtime bundles; no link-time inference inventory remains.
