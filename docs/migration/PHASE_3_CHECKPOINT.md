# Phase 3 Release Train Checkpoint

> **Status:** Implemented and validated locally; hosted platform execution,
> repository publication, and the tagged release-candidate exit gate remain
> deferred.

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
- `runtime-manifest.json` with 67 workspace packages, the Rust toolchain, lockfile
  hash, and exact Candle/MLX Git revisions;
- an SPDX 2.3 JSON SBOM containing all 449 cross-platform lockfile packages and
  1,576 describe/dependency relationships;
- `SHA256SUMS` covering the manifest, source archive, and SBOM.

The clean dry run `runtime-2026.07.0-rc.0` at
`4e2d4b48764908e562705c48afa4901d9dc67534` produced a 364,316,569-byte source
archive. Its verification safely extracted the archive and compiled an external
Cargo consumer against the archived `core-llm` and `sceneworks-gen-core` paths.
The dry-run tag was an artifact label only; no Git tag was created.

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

Nightly/manual real-weight jobs use runner-provisioned snapshots and reject a
moving branch identity. The initial representative models are fixed to:

| Profile | Repository | Revision |
|---|---|---|
| small LLM | `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| thinking LLM | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| media | `Tongyi-MAI/Z-Image-Turbo` | `f332072aa78be7aecdf3ee76d5c247082da564a6` |

The verifier accepts the standard Hugging Face `snapshots/<revision>` layout or
a materialized snapshot carrying `.sceneworks-model-revision`, and checks required
files before any expensive test begins.

## Local validation

The following pass from the repository root:

- 13 repository-tooling unit tests;
- dependency-aware lane-selection examples for contracts, MLX, Candle, docs, and
  unknown paths;
- documentation local-link validation;
- YAML parsing for both workflow definitions;
- `cargo deny --locked check advisories bans licenses sources`;
- deterministic manifest/SBOM generation and checksum verification;
- safe source-archive extraction and external-consumer compilation;
- the Phase 2 workspace gate and `git diff --check`.

## Remaining Phase 3 exit gates

1. Publish the repository and run the hosted Linux/macOS lanes.
2. Configure the self-hosted Windows/CUDA and real-weight runner variables, then
   execute those matrices against the pinned snapshots.
3. Create an immutable release-candidate tag only after those jobs pass.
4. Rebuild and upload the tagged source/SBOM bundle, then verify its hashes.

Product cutover and provider-registry refactoring remain later phases; this phase
does not change any consumer dependency.
