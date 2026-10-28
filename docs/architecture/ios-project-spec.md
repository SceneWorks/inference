# iOS runtime project specification

Reconciled 2026-10-26. This describes the feature branch, not a main release or a
completed device campaign. Requirements and acceptance authority:
[epic 16774](https://app.shortcut.com/trefry/epic/16774).
Architecture rationale: [strategy](ios-strategy.md). Operation:
[device runbook](../ios-device-runbook.md).

## Components and boundaries

| Component | Responsibility |
|---|---|
| `crates/bundles/runtime-ios` | MLX LLM/preparer registries, narrow media composition, platform allocator/cache limits and pre-load admission |
| `crates/media/mlx-gen/mlx-gen-ios-catalog` | Explicit iOS provider registration, provider-owned memory selections, artifact applicability |
| `ios-host/smoke` | Explicit root-workspace package sharing the SDK lock and compiled dependency units; static diagnostic bridge |
| `ios-host/App` and `ios-host/project.yml` | Generated Xcode host, launch-memory observation and persistent diagnostic reports |
| `scripts/ios/run_smoke.sh` | Build, verify metallib, package, install, launch, reject stale outputs and pull reports |
| `scripts/ios/push_model.sh` | Move caller-provisioned snapshot trees into the selected app container; no downloads |
| `scripts/ios/bundle_metallib.py` | Package the target metallib and reject an incorrect platform |
| `.github/workflows/ci.yml`, `scripts/ci/select_lanes.py` | Select and execute the iOS cross-build lane |

`runtime-ios` has a default `media` feature; disabling it gives the LLM/preparer
composition without the image catalog. It has no audio registry. Model training
is not part of the iOS bundle. Anima Turbo is registered without Base/Aesthetic;
transitive Qwen-Image/Wan decoder code does not register those generators.
Do not replace the narrow catalog with the macOS catch-all catalog.

## Build and packaging contract

Use the committed `rust-toolchain.toml`, lockfiles and workspace MLX pin. Targets
are `aarch64-apple-ios` and `aarch64-apple-ios-sim`; the deployment floor is iOS
18.0, not the OS version of the measurement phone. `.cargo/config.toml` carries
the target settings. Do not leak an iOS deployment environment into macOS builds.

The CI selection key is `ios_build`; its device/simulator matrix builds
`runtime-ios` and asserts `air64-apple-ios18.0.0` or
`air64-apple-ios18.0.0-simulator` respectively in the metallib. A macOS library
is not an acceptable successful cross-build. The host packages `mlx.metallib`
next to the app executable; a build-tree or host-cache path is not usable inside
the app sandbox. Simulator compilation is packaging/build coverage, not target
memory, speed, thermal or hardware-correctness evidence. Ordinary macOS/CPU/CUDA
lanes remain in place; this epic does not replace them with iOS CI.

## Public loading contract

Every registered image model and mode loads through ONE admission seam
(`mlx_gen_ios_catalog::ios_admission`), exposed as one admitted facade per surface:

- `runtime_ios::media::load_sana`, `load_sdxl`, `load_anima_turbo` and `load_iris` take ordered
  `IosCandidate`s (built with `media::build_candidate` from the provider's own contract) and return
  the admitted index plus a `BoundGenerator`. `load_iris_restore` returns a `BoundTransform`,
  `load_iris_depth` a `BoundDepth`. Each requires `bound_mlx_to_platform_limits()` first, then
  keys the request (hardware, provider, route, tier, path-free layout digest, mode, geometry,
  reference count, execution levers, token window — never prompt, seed or install path), prices
  it from the single reviewed ANCHOR table
  `crates/media/mlx-gen/mlx-gen-ios-catalog/evidence/ios_device_evidence.json` (schema 3: per
  family a resident and a fully optimized anchor; at an anchor's own key its measured growth, else
  an applicable certified transfer or explicitly approved engineering estimate
  over the providers' actual source pricing), runs the provider's registered `standard_memory_strategy_safety_check` with that demand
  plus the anchor's measured repeat-spread margin against the observed entitled OS process ceiling and live headroom,
  and only then loads, inside the admission's measured envelope (`ios_quantum`). The bound wrapper rechecks the exact
  request (geometry, tile, mode, conditioning, prompt, execution), live capacity and source pins
  before every provider call and preserves cancellation.
- Refusals are typed (`media::RefusalKind`: capacity, no_evidence, mode, strategy, artifact). The
  table is empty, so every product load currently refuses `no_evidence`. Modes a provider does
  not serve are refused `mode`, never dropped.
- `runtime_ios::load_admitted(estimate_mib, closure)` rejects absent estimates, estimates over the
  observed process ceiling, or over current OS headroom before invoking its closure.
- `runtime_ios::catalog()` and raw catalog registries are composition/inspection surfaces, not
  admission authority. Raw LLM loading is likewise unadmitted.
- `runtime_ios::media::capture_permit(candidate)` is the explicitly labelled DEVICE CAPTURE path:
  it needs no evidence row (that would be circular) and is granted only where every quantum of
  the model checks live OS headroom before constructing its graph (`mlx_gen::ios_quantum`, a
  physical device). A quantum that does not fit refuses before allocating; that typed refusal
  (`ios_quantum::take_refusal`) is the "next size refused before jetsam" evidence. Capture output
  never changes admission: `scripts/ios/campaign.py evidence` prints candidate anchor rows for
  review.
  There are no unadmitted image or depth loaders.

## Evidence and measurement contract

The 8 GB iPhone 16 Pro Max is the actual target. The 2026-10-26 A/B on iPhone17,2,
iOS 26.6.2 (23G90), observed current OS process limits
of 6144 MiB entitled and 3376 MiB off (available memory plus current footprint, three pairs
each; restored-on confirmed). The runtime uses the entitled limit as a ceiling, not a safe
render budget; [original receipts](../evidence/ios-process-limit-2026-10-26/README.md)
remain diagnostic model-independent evidence. A validated safe working-set envelope still needs device captures; neither launch headroom nor
old active+cache observations can supply it.

The 2026-10-26 approved contract permits a practical scoped engineering estimate
for real file-backed catalog/model paths: optimized measured growth times an
upward-only maximum source phase-pressure ratio, repeat-spread margin floored
at 32 MiB, and live headroom checks. Record HOST/driver and source-preparation
assumptions explicitly; this is not a mathematical iOS physical-charge bound.
Preserve arbitrary public Array inputs without claiming this phone-fit
certification. The runbook specifies endpoint consistency and provenance rules.

SANA profile identity is header-only: component inventory, tensor names, shapes,
storage dtypes and payload sizes. It is not a content/quality hash. Binding is
about applicability, not revision currency: moving a pin is not grounds to
invalidate a capture or alter runtime behavior. A valid memory record also
identifies hardware, provider, geometry and execution strategy. Packed Gemma
q4/q8 embedding loading/conversion and source-bound admission landed in
[#1272](https://github.com/SceneWorks/inference/pull/1272), merge `23c7e3953`.
The failed target attempt's actual dense embedding was 1,179,648,000 bytes;
its directory tier name did not establish whole-pipeline packed storage.

For each campaign row preserve source SHA, model/artifact identity and component
precision, phone/OS, entitlement, geometry, selected strategy, elapsed time,
MLX peak, kernel footprint and minimum headroom, result/refusal, output image,
and matching termination diagnostics if relevant. A breadcrumb identifies the
last completed phase, not why a process died. The runbook distinguishes existing
Mac engineering evidence from the still-unfilled target campaign.

## Execution and ownership

Keep MLX model work serialized on its owning thread; a Rust callback is not a
product Swift concurrency API. Preserve fresh-build/report guards and explicit
admitted-versus-diagnostic labels in the harness. Do not run another MLX workload
concurrently with a device investigation's host-side measurement.

The remaining terminal sequence is owned by sc-26270: finish the general model
set, review E1–E8, integrate inference through its protected main PR, perform the
single permanent SceneWorks pin bump, collect the target endpoint evidence and
derived claims, then integrate SceneWorks. Supra2 iOS registration remains the
user-approved non-blocking sc-26269 follow-up. No pin or main merge is performed
by this docs story. Product FFI, Swift package and app provisioning remain separately scoped.
