# Phase 4 Explicit Composition Checkpoint

> **Status:** Complete. Named runtime bundles, explicit media/LLM composition, consumer cutover,
> inventory removal, hosted validation, self-hosted CUDA/NAX, and real-weight execution pass at
> immutable release `runtime-2026.07.0`.

## Result

The backend-neutral generative-media contract and testkit now live at
`crates/contracts/gen-core` and `crates/contracts/gen-core-testkit`, beside `core-llm`. Package and
Rust crate names did not change, so the move changes ownership without changing public type
identity.

The supported build products are ordinary Cargo packages under `crates/bundles/`:

| Bundle | Backend | Composition | Supported targets |
|---|---|---|---|
| `runtime-macos` | MLX | complete MLX media catalog, MLX LLM catalog, MLX snapshot preparer | `aarch64-apple-darwin` |
| `runtime-cuda` | Candle CUDA | complete Candle media catalog, Candle LLM catalog, Candle snapshot preparer | x86-64 Linux and Windows with CUDA |
| `runtime-cpu` | Candle CPU | complete Candle media catalog, Candle LLM catalog, Candle snapshot preparer | x86-64 Linux/Windows and Apple-silicon macOS |

`runtime-catalog` is a tensor-neutral composition layer shared by the three bundles. Construction
rejects media descriptor violations, backend mismatches, duplicate provider/preparer identities,
and empty LLM identities. A constructed catalog exposes the exact media, LLM, and preparer
registries plus a stable JSON-compatible snapshot for release and consumer compatibility checks.

The MLX and Candle platform catalogs also own and re-export provider-specific utility packages that
do not implement the general `Generator` contract. SAM2/SAM3, face, depth, InstantID, PiD, and the
Candle PuLID utility therefore have explicit bundle ownership instead of remaining product-side
dependency exceptions.

## Explicit LLM composition

`mlx-llm` and `candle-llm` now expose ordinary builder functions for both provider loading and
snapshot preparation. Snapshot preparation has the same immutable registry model as model-first
loading; bundle callers do not depend on which crates the linker happens to retain.

SceneWorks and ChatWorks now construct named runtime catalogs. The old process-global `core-llm`
functions, linker submissions, and `inventory` dependency have been removed. Backend-scoped
convenience loaders remain, but each constructs a deterministic registry solely from that backend's
ordinary registration constants.

## CI ownership

Dependency-aware lane selection treats the shared runtime catalog as affecting every platform and
each named bundle as affecting only its platform. The Linux CPU, hosted macOS, and self-hosted CUDA
jobs now compile and test their corresponding bundle package in addition to the backend packages.
Root manifest or lockfile changes continue to select every lane.

## Local validation

The following passed from the repository root:

- workspace structure and graph gate with 74 path members;
- repository tooling and bundle lane-selection tests;
- all `core-llm`, `candle-llm`, and `mlx-llm` library tests;
- `runtime-cpu` and `runtime-macos` catalog smoke tests and doc tests;
- the exact manual NAX command on macOS 26.5.2 / arm64 / Metal 32023.883: 174 `mlx-llm`
  tests passed with one slow case ignored, plus complete and LLM-only `runtime-macos` catalog tests;
- real-weight MLX LLM conformance with the pinned SmolLM2 135M snapshot at revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac` and Qwen3 0.6B snapshot at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`;
- real-weight Z-Image-Turbo generator conformance with the pinned snapshot at revision
  `f332072aa78be7aecdf3ee76d5c247082da564a6`, including progress, typed cancellation,
  pre-cancellation, and seed-determinism checks;
- Clippy with `-D warnings` for the CPU and macOS bundle graphs;
- the complete pre-bundle Candle CPU lane and contract suites recorded in the post-import
  reconciliation checkpoint.

`runtime-cuda` resolves in the unified Cargo graph and passed on the self-hosted Windows/CUDA runner.
The self-hosted NAX runner passed the macOS/MLX configuration, and the four real-weight jobs passed
the pinned LLM/media conformance cases. The exact-commit and real-weight evidence is linked from the
Phase 3 checkpoint.

## Rollback

The contract relocation and bundle introduction are separate commits. Either can be reverted
without reverting the imported histories or workspace normalization. The last pre-removal commit is
the source-level compatibility rollback point; the release itself has no hidden link-time fallback.

## Phase 4 exit outcome

1. The checked-in workflow passed on the macOS NAX runner.
2. `runtime-cuda` built and tested on the self-hosted Windows/CUDA runner.
3. Immutable release `runtime-2026.07.0` was published at exact commit
   `48cc2d87e14de0189ac4f7763fddc0a8581c2e68`.
4. SceneWorks and ChatWorks resolve their selected bundles from that tag and exact commit; their
   local product gates pass. Hosted SceneWorks validation requires its scoped private-repository
   read secret and does not weaken the release identity.
