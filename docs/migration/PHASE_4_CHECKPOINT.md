# Phase 4 Explicit Composition Checkpoint

> **Status:** Named runtime bundles and explicit media/LLM composition are implemented and locally
> validated. Consumer cutover, hosted CUDA/NAX execution, and removal of the LLM compatibility
> inventory remain open exit gates.

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

## Explicit LLM composition

`mlx-llm` and `candle-llm` now expose ordinary builder functions for both provider loading and
snapshot preparation. Snapshot preparation has the same immutable registry model as model-first
loading; bundle callers do not depend on which crates the linker happens to retain.

The old process-global `core-llm` functions remain as compatibility adapters while SceneWorks and
ChatWorks are cut over. They are no longer the supported composition boundary. Their
`inventory` submissions can be removed after both consumers construct a named runtime catalog and
no rollback path depends on the old calls.

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
- Clippy with `-D warnings` for the CPU and macOS bundle graphs;
- the complete pre-bundle Candle CPU lane and contract suites recorded in the post-import
  reconciliation checkpoint.

`runtime-cuda` resolves in the unified Cargo graph but is not compiled on this host because its
manifest deliberately enables the CUDA backend. The self-hosted CUDA lane is the compile/test
authority for that named configuration.

## Rollback

The contract relocation and bundle introduction are separate commits. Either can be reverted
without reverting the imported histories or workspace normalization. Until consumer cutover, the
global LLM compatibility functions retain the prior routing behavior.

## Remaining Phase 4 exit gates

1. Run `runtime-macos` on the hosted macOS lane and the macOS 26.2+ NAX runner.
2. Build and test `runtime-cuda` on the self-hosted CUDA runner.
3. Cut SceneWorks and ChatWorks to named runtime catalogs and compare their provider snapshots.
4. Remove the LLM `inventory` compatibility layer after the immutable release rollback point no
   longer requires it.

