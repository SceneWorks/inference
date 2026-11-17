# FLUX.1 migration comparison baseline (SW-6932)

Engine `2b09938e33370915d4f853389961709b8bbf8bf9` and SceneWorks
`5a28d413dd9f579cc9607d44f252a045d025bd8e` are the feature branch points.
The latter pins that exact engine. Use both identifiers for subsequent comparisons.

## Source census

[Fact audit](facts.md) records the §10.2 grep recipes and reviewed production
encodings for all 26 categories. [LOC census](loc.json) records counts per area.
To replay with two local repository paths:

```sh
python3 scripts/sga_loc_census.py \
  --inference "$ENGINE" --inference-rev 2b09938e33370915d4f853389961709b8bbf8bf9 \
  --sceneworks "$APP" --sceneworks-rev 5a28d413dd9f579cc9607d44f252a045d025bd8e
```

There are 1,926,085 source and 542,943 separately identifiable test lines across
the recorded engine areas, and 982,679 and 448,595 across the app areas. This
counts physical lines including blanks/comments in tracked first-party source
files. Co-located Rust unit tests remain with their production file. Data,
manifests, Markdown, vendored code and build outputs are excluded. Standalone
test paths, fixtures and filenames use the same rule on both sides. These are
informational counts; the historical estimates in §8 covered fewer directories.

## Paired iOS release binaries

Use the `catalog-size` profile of the already-registered dispatch-only
[workflow](../../../.github/workflows/ios-simulator-sana.yml) on a branch
containing this measurement harness. It checks out the requested source as a
second immutable tree, builds two arm64 device executables on the same hosted
macOS runner, and uploads one JSON receipt after both succeed.

```sh
gh workflow run ios-simulator-sana.yml --repo SceneWorks/inference --ref "$FEATURE_BRANCH" \
  -f mode=catalog-size -f source_sha=2b09938e33370915d4f853389961709b8bbf8bf9
```

The exact consumer calls the public `runtime_ios::catalog().snapshot()` and links
the resulting provider registries into a Rust release executable. The measurement
[script](../../../scripts/ios/catalog_size.py) binds the body, target, feature
flags, checkout, compiler, SDK, architecture, Mach-O load command, byte count and
digest. It does not launch either binary. For a later comparison use the same harness
and tool versions, compare each partition with its own pre-feature receipt. PERF-2's
upper bound is 1.02 times the comparable baseline for that partition. The measured
artifact is the linked executable; it excludes app packaging, metallib and model
assets. The device smoke harness has its own capture/feature graph, and its manifest
enables runtime-ios default features even with the smoke media flag off.

The unmodified base has an empty media registry in the LLM-only graph. With
`runtime-ios/media` it composes the seven image generators and Iris restore in
`crates/media/mlx-gen/mlx-gen-ios-catalog/src/lib.rs:134`. The expectations in
`crates/bundles/runtime-ios/src/lib.rs:1655–1692` also verify that trainer and
unrelated IDs stay absent. Normal-edge locked `cargo tree --target aarch64-apple-ios
-p runtime-ios -e normal` runs with the two feature sets show no Candle/CUDA
dependency. Media does reach the Wan crate through Anima's Qwen VAE and Qwen's
single-frame decoder, and shared MLX imports gen-core-train. LLM-only excludes
both packages. Registry membership and linked production dependencies are
separate observations.

For resident memory, preserve the existing [physical-device receipt](../ios-process-limit-2026-10-26/receipt.md)
at its own identity. Its restored-on primitive on iPhone17,2, OS build 23G90,
source `e49f5345bd254936f20c15c5ea84e76656faf01f` measured 14,336,872 bytes
before and 92,406,656 bytes resident with the same entitlement. Its kernel
lifetime peak was 196,035,480 bytes. This historical primitive has a different
source and scope from the binary-size baseline; ENG-13's admission replay
uses matched device evidence for changed footprint or headroom behavior.

## Prior art

Reviewed inference `34222b1de142b9323b711dad5b430cb55224aa52` (sc-20414,
47 commits unique to the branch, 4,295 unique to the current base). Paths
below refer to that revision:

| Source | Disposition for the runner |
|---|---|
| `crates/contracts/gen-core/src/sampling/pass_executor.rs:70–102` | Preserve domain-separated boundary RNG for a newly introduced draw and the f32 fresh-schedule length/terminal-node math for an actual fresh pass. Require bounded schedule length before allocating (the old division may saturate for tiny positive denoise). An inert split keeps the route's draws and its existing schedule. |
| `pass_executor.rs:209–324` | Reuse cancellation before boundaries and inside evaluations. Keep public progress monotone by the legacy route's outer steps; evaluations are internal. Fresh solver creation, unconditional re-noise after the first pass, a single latent and host-owned CFG cannot represent typed continuing phases or multiple streams. |
| `pass_executor.rs:289–298,730–818` | Preserve the primary error and partial-begin cleanup on request-local faults. For a poisoned provider/process skip further GPU work, release host accounting, and retain the original error. Do not restore mutable shared adapter state on the next request. |
| `crates/contracts/gen-core-testkit/src/denoise_passes.rs:221–729` | Port the behavioral cases for terminal schedule, one-pass equality, noise law (FLOW vs VE), isolation, reset, evaluation accounting and cancellation to the new owner-local mock. Reset and re-noise apply only at a declared incompatible boundary. Continue with compatible history requires its own trace. |
| `crates/contracts/gen-core/src/denoise_passes.rs:433–435,775–834` | Its wire schema and mutual refusal of phases/pass chains remain compatibility inputs for ENG-16. The shared option system owns eventual recipe persistence. |

Two other stale branches have different owners. SceneWorks sc-19048
`15d113ec2d7b34765ac3faa5ac2048207d508ce0` contains application calibration,
base admission and decision inventory scripts; no worker/admission replacement
is needed for the engine baseline. Reconcile those sources at the first admission
edit against delivered SW-F327. Inference sc-25327
`1a038fc68f2bdddabe9b5cce774e3b1cdac72685` contains H3/SeedVR2 upscale
prototypes and reference fixtures, outside FLUX.1. Salvage their operation/oracle
evidence if a later H3 transition touches that route.

## Branch policy and Tier 1 route

`scripts/ci/feature_epic_policy.py` already recognizes
`story/sw-6932-feature-sw-f331-runner-core-flux1` and
`feature/sw-f331-runner-core-flux1`. The existing policy tests cover canonical
slug, unique remote ref, mismatch refusal and feature-to-main topology.

The proof workflow is dispatch-only and its native controller already routes
`runner=nax` to `[self-hosted, macOS, ARM64, rw-llm]`, then checks the actual
runner name is `nax-macos-2`. Candle CUDA uses `[self-hosted, windows, cuda,
real-weights]`. On November 7 the accessible organization group had one matching
secondary Mac and two matching CUDA listeners. Refresh the registration scopes,
group access, physical device assignments, holds and availability before a live
run. The shared dev Mac is outside the MLX route.

[MLX planner receipt](mlx-route-plan.json) and [CUDA planner receipt](cuda-route-plan.json)
come from the controller's `--plan` entry on the exact engine SHA. Each selects
one registered FLUX.1 case and validates a finite 20-minute deadline, 24 GiB
child host footprint cap and 8 GiB free reserve. The CUDA example binds a
historically observed UUID; it requires live device matching before dispatch.
Reproduce on a clean checkout of the pinned engine:

```sh
export MODEL_PROOF_REQUEST="$(jq -c .request "$SAVED_PLAN")"
python3.12 -I scripts/ci/model_inference_proofs.py --plan --backend "$BACKEND" \
  --request-env MODEL_PROOF_REQUEST --expected-sha 2b09938e33370915d4f853389961709b8bbf8bf9
```

Neither planner executed a learned operation. They establish routing and
explicit host guards. The registered example cases have different scopes from
the batch's Tier 1 numerical comparison. For the migrated batch, predeclare
the FLUX.1 route/seed/geometry/steps/dtype, shape-faithful random fixture with
truncated depth, effective adapters and matched allocator/workspace. Build
and run the base/head in the same held job, compare output bytes and intermediate
traces at equal outer steps for AP13 class (a), and save the actual device
and nonzero body receipts. Use the feature's finite proof budget, keep ordinary
CI model-free, and reuse unchanged evidence.
