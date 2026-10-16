# iOS runtime project specification

Updated 2026-10-16. This describes the feature branch, not a main release or a
completed device campaign. Requirements and acceptance authority:
[epic 16774](https://app.shortcut.com/trefry/epic/16774).
Architecture rationale: [strategy](ios-strategy.md). Operation:
[device runbook](../ios-device-runbook.md).

## Components and boundaries

| Component | Responsibility |
|---|---|
| `crates/bundles/runtime-ios` | MLX LLM/preparer registries, narrow media composition, platform allocator/cache limits and pre-load admission |
| `crates/media/mlx-gen/mlx-gen-ios-catalog` | Explicit iOS provider registration, provider-owned memory selections, artifact applicability |
| `ios-host/smoke` | Separate Cargo workspace, static diagnostic bridge into the runtime bundle |
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

On the merged feature base `a54ad6c3a`:

- `runtime_ios::media::load_sana(sana_id, spec, image_edge, tile_edge)` is the
  admitted SANA facade. `admit_sana_snapshot` checks actual component layout,
  provider and hardware applicability before any model load. It presently
  refuses because no applicable target capture exists.
- `runtime_ios::load_admitted(estimate_mib, closure)` rejects absent estimates,
  estimates over the provisional budget, or over current OS headroom before
  invoking its closure. Supplying a number does not itself establish evidence.
- `runtime_ios::catalog()` and raw catalog registries are composition/inspection
  surfaces, not admission authority. Raw LLM loading is likewise unadmitted.
- `load_sana_unadmitted_for_diagnostics` is explicit diagnostic access only.
  The harness labels evidence capture **UNADMITTED**, including if it renders.

Iris registration [#1277](https://github.com/SceneWorks/inference/pull/1277) and
SDXL registration [#1278](https://github.com/SceneWorks/inference/pull/1278) were
open at this documentation snapshot. Their final merged public facades must be
checked before documenting consumer calls. In particular, preserve the safe
SANA/SDXL facade boundary: do not recommend raw registry loads as admitted.
Iris depth is a distinct callable surface, with explicitly unadmitted diagnostic
loading, not an image-registry mode or a measured-fit claim. No Iris or SDXL
phone fit is established by these registration changes. All conditioning/mode
routes must remain represented even when admission refuses them.

## Evidence and measurement contract

The 8 GB iPhone 16 Pro Max is the actual target. Entitled/unentitled launch
headroom (6142/3374 MiB, three readings each, iOS 26.6.2) is diagnostic only.
The existing budget constants are provisional and unchanged. A validated safe
working-set envelope still needs device captures; neither launch headroom nor
old active+cache observations can supply it.

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

The remaining terminal sequence is owned by sc-26270: complete registrations
(including Supra2 after epic 24212), collect the target model × tier × geometry
campaign, review E1–E7 once, integrate inference, perform the one SceneWorks pin
bump, and integrate SceneWorks. No pin or main merge is performed by this docs
story. Product FFI, Swift package and app provisioning remain separately scoped.
