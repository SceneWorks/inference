# iOS strategy — MLX on iPhone

Status: implementation on `feature/sc-16774-ios-runtime`, not an announcement of
availability on main. Updated 2026-10-16 for [epic 16774](https://app.shortcut.com/trefry/epic/16774).
See the [project specification](ios-project-spec.md), [device runbook](../ios-device-runbook.md),
and [remaining work](../ios-epics.md).

## Decision

Use MLX/Metal through the existing Rust inference workspace, with `runtime-ios`
as the platform bundle and `mlx-gen-ios-catalog` as its explicit media composition.
Reuse the backend-neutral contracts and `mlx-llm`; do not build a parallel engine
or a second history. The workspace's existing MLX pin supplies iOS SDK selection
and `DEP_MLX_METALLIB`. There is no additional iOS-specific fork pin or accepted
"fork of a fork" maintenance cost.

The reviewed workspace pin is `michaeltrefry/mlx-rs` at
`fe22998bf73e5e785d7659beb2f24d624b785bda`, as declared in `Cargo.toml`.
The old build-feasibility question has been replaced by cross-build CI and actual
device evidence. Use the current manifest, not a historical branch's toolchain.

MLX is appropriate because the product needs caller-provisioned model weights
and on-device image generation as well as text. The media samplers and model
implementations already use MLX arrays. A Core ML/ANE backend would require its
own conversion/runtime integration, not merely another bundle. Apple's
Foundation Models does not substitute for loading those model snapshots.
Neither alternative is an implemented capability of this epic. MLX uses the GPU,
not the ANE; battery, thermal behavior and UI contention remain device questions,
not architectural promises.

## Composition is not a fit verdict

The target is **iPhone 16 Pro Max, 8 GB, iOS 26.6.2**. The complete epic media set
is SANA, Anima Turbo, Iris 3B, SDXL/Illustrious-XL v1/v2/RealVisXL/RealVisXL
Lightning, and Supra2-IMG. Register all of each model's modes; refuse unsupported
memory requests rather than deleting modes. Registration, cross-compilation,
Mac rendering and a successful target-phone render are four different claims.
The runbook lists their current status separately.

The narrow media catalog avoids registering the desktop's entire video/audio
surface. Shared decoder dependencies may still compile without their generators
being registered. Training/heavy optional dependencies are excluded by the iOS
composition. The LLM registry is reused, but raw LLM loads have no target-bound
footprint estimate and are not admitted product loads.

## Memory contract

There is **no validated safe target-phone rendering cap yet**. Entitlement A/B
launches reported 6142 MiB available with increased-memory-limit and 3374 MiB
without it, three identical readings each. `os_proc_available_memory()` reports
remaining headroom at that instant. Those launch observations are not safe peak
working-set limits and do not authorize changing the provisional runtime budget.
`IOS_APP_MEMORY_CAP_MIB` / `IOS_MEMORY_BUDGET_MIB` remain provisional; do not
interpret the constants as measurements.

Product loading needs both the provider's registered memory-strategy safety check
and applicable device/artifact evidence. They solve different questions. SANA's
header-only profile binds component membership, keys, shapes and storage dtypes;
a directory called q4 does not establish that its embedding or every other
component is packed. `runtime_ios::media::load_sana` checks the actual snapshot
before loading. No target-bound SANA evidence is installed, so its present
product result is refusal. The explicitly named diagnostic loader is not a
fallback product path.

Call `bound_mlx_to_platform_limits()` before allocating models and serialize MLX
work on the owning execution thread. Allocator/cache limits reduce growth; they
are not proof against termination. Record MLX live peak, active+cache, kernel
process `phys_footprint`, and minimum available headroom separately. Historical
SANA observations on a 12 GB iPhone 17 Pro Max measured MLX active+cache, not the
target's kernel process footprint, and lack an artifact-layout binding. They
remain planning data only. The target-phone 1024/tile192 attempt terminated;
its cause is unconfirmed, not proven jetsam. No target image success is recorded.

## Delivery boundary

This epic supplies the Rust bundle, catalog, build lane and measurement harness.
The smoke host's C entry point is a diagnostic bridge, not the product FFI or a
Swift package. App-side provisioning, the product app, SenseNova and Anima
Base/Aesthetic are outside this epic. Supra2 implementation belongs to epic
24212, but its eventual iOS registration remains required here, not silently
removed from scope.

Final device campaign, feature-end review, the single SceneWorks pin bump and
feature-to-main integration remain sc-26270. Until those results exist, neither
this strategy nor a passing build asserts a production-safe iPhone model set.
