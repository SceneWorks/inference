//! Supported iOS runtime: the narrow MLX image catalog plus the MLX LLM and snapshot-preparer
//! catalogs.
//!
//! This is the iPhone/iPad product boundary. It composes the same `mlx-llm` engine
//! [`runtime-macos`](../runtime-macos/README.md) does — MLX runs on iOS, so the LLM lane needs no
//! iOS-specific backend — but its media registry is `mlx-gen-ios-catalog`, the reviewable subset
//! of image generators that fit a phone, and it ships **no audio registry**. See `Cargo.toml` for
//! why both are composition decisions rather than missing features.
//!
//! # Packaging
//!
//! An iOS app must carry MLX's Metal kernel library. Inside the app sandbox only two links of
//! MLX's resolver chain are reachable: `$PMETAL_METALLIB_PATH`, or `mlx.metallib` sitting next to
//! the executable. The host-side links are unavailable — `~/.cache/pmetal/lib` is not readable in
//! the sandbox, and the compiled-in `METAL_PATH` points into the cargo target directory, which is
//! not shipped. `pmetal-mlx-sys` publishes the built metallib's path as `DEP_MLX_METALLIB`.
//!
//! # Admission (epic 16774, E4/E5/E7)
//!
//! [`load_admitted`] is the generic estimate/cap guard: it refuses, before the loader runs, an
//! absent estimate or one above [`IOS_APP_MEMORY_CAP_MIB`] or live process headroom. Every image
//! model and mode loads through ONE admitted facade per surface in [`media`] — `load_sana`,
//! `load_sdxl`, `load_anima_turbo`, `load_iris`, `load_iris_restore`, `load_iris_depth` — each of
//! which requires the platform limits to be bound, admits through the catalog's single
//! evidence-table seam (the provider's registered `standard_memory_strategy_safety_check`, a
//! reviewed device row, the cap and the live budget), refuses with a typed [`media::RefusalKind`],
//! and returns a request-bound wrapper that rechecks the exact work, live capacity and source pins
//! before every provider call. Other raw load surfaces are not admitted:
//!
//! * [`llm`] re-exports `mlx-llm` whole, and its loaders (`load_for_model`, …) do **not** admit.
//!   MLX LLMs carry no footprint estimate yet, so the admitted answer for an LLM load today is a
//!   refusal (`admit_estimate_mib(None)`); a caller loading one anyway is running unadmitted and
//!   must say so.
//! * [`catalog`]'s registries are a composition/snapshot surface (the shared
//!   [`RuntimeCatalog`] contract) and load unadmitted. A product loads through
//!   the model's admitted facade, not through them.
//! * `media` deliberately does not re-export the catalog's `provider_registry`.
//! * `media::capture_permit` is the explicitly labelled evidence-collection path: no evidence
//!   row, but only where every quantum is bounded by live OS headroom before construction (a
//!   physical device). It is never an admission and never changes one.

use std::sync::atomic::{AtomicBool, Ordering};

/// The iOS image lane. The catalog's specs, constants, and footprint table, plus the admitted SANA
/// loader — but not its `provider_registry`, whose `load` is unadmitted (see the crate docs).
///
/// The product media facade must not expose an unrestricted registry loader:
/// ```compile_fail
/// let _ = runtime_ios::media::provider_registry();
/// ```
///
/// Nor, in a product build, the device-capture path (only the `device-capture` feature, which
/// only `ios-host/smoke` enables, compiles it). Without the feature these must not compile; with
/// it, they must (so the doctest is meaningful both ways):
#[cfg_attr(not(feature = "device-capture"), doc = "```compile_fail")]
#[cfg_attr(feature = "device-capture", doc = "```")]
/// let _ = runtime_ios::media::capture_permit;
/// let _ = runtime_ios::media::capture_capacity;
/// ```
#[cfg_attr(not(feature = "device-capture"), doc = "```compile_fail")]
#[cfg_attr(feature = "device-capture", doc = "```")]
/// let _ = mlx_gen_ios_catalog::capture_permit;
/// ```
#[cfg(feature = "media")]
pub mod media {
    pub use mlx_gen_ios_catalog::admit;
    #[cfg(feature = "device-capture")]
    use mlx_gen_ios_catalog::capture_permit as catalog_capture_permit;
    pub use mlx_gen_ios_catalog::control_required;
    pub use mlx_gen_ios_catalog::{
        anchor, anima, anima_turbo_ios_selection, anima_turbo_load_spec,
        anima_turbo_memory_selection, anima_turbo_planning_decision, build_candidate,
        capacity_budget, evidence, evidence_key, ios_memory_budget, ios_quantum, pricing,
        sana_decode_plan, sana_decode_tile_edge, sana_footprint_bound, sana_generation_memory,
        sana_load_spec, sana_snapshot_profile, sdxl, zimage, Admission, AdmissionBasis,
        AdmissionError, AdmissionResult, BoundDepth, BoundGenerator, BoundTransform,
        FootprintBound, IosCandidate, IosWork, Refusal, RefusalKind, SanaDeviceFootprint,
        SanaSnapshotProfile, ANIMA_TURBO_ID, CONTROL_REQUIRED_ROUTES, IOS_IMAGE_EDGES,
        IOS_MEMORY_BUDGET_MIB, IOS_MLX_CACHE_LIMIT_CAP_MIB, IOS_OFFLOAD_POLICY, IRIS_DEPTH_ID,
        IRIS_RESTORE_ID, SANA_DEVICE_FOOTPRINTS,
    };

    use crate::gen_core::{Error, ProviderRegistry, Result};

    fn require_limits_bound() -> AdmissionResult<()> {
        if super::PLATFORM_LIMITS_BOUND.load(std::sync::atomic::Ordering::Acquire) {
            Ok(())
        } else {
            Err(AdmissionError::Failed(Error::Unsupported(
                "iOS admission requires bound_mlx_to_platform_limits first".into(),
            )))
        }
    }

    /// The capacity check at admission, before load and before every bound call (E4).
    ///
    /// `required_bytes` is the admitted work's own growth plus its margin
    /// ([`Admission::required_bytes`]): the anchor-measured (or anchor-law derived) growth over the
    /// capture's starting footprint, plus the anchor's measured repeat spread. Nothing this process
    /// already holds is credited as reusable. On a recheck only the growth realized SINCE ADMISSION
    /// is credited, `remaining = required - (F_now - F_admission)`, so a loaded generator's second
    /// render is not charged its weights twice while an unrelated resident never makes room.
    /// `remaining` must fit live headroom; the projected absolute footprint must fit the single cap.
    /// The anchor's measured margin replaces the 512 MiB formula reserve here: a kernel
    /// `phys_footprint` anchor already contains the driver and allocator overhead that reserve
    /// guessed at, and stacking it on a measurement would make the low end inaccurate.
    pub(crate) fn incremental_capacity(
        required_bytes: u64,
        now: super::ProcessMemory,
        at_admission: super::ProcessMemory,
    ) -> Result<()> {
        let realized = match (now.footprint, at_admission.footprint) {
            (Some(current), Some(admitted)) => current.saturating_sub(admitted),
            // No kernel reading: the headroom consumed since admission.
            _ => match (at_admission.available, now.available) {
                (Some(then), Some(current)) => then.saturating_sub(current),
                _ => 0,
            },
        };
        let remaining = required_bytes.saturating_sub(realized);
        let held = now
            .footprint
            .or_else(|| at_admission.footprint.map(|f| f.saturating_add(realized)))
            .unwrap_or(0);
        let projected = held.saturating_add(remaining);
        let projected_mib = u32::try_from(projected.div_ceil(1 << 20))
            .map_err(|_| Error::Unsupported("iOS footprint exceeds runtime range".into()))?;
        super::admit_estimate_mib(Some(projected_mib))
            .inspect_err(|_| record_capacity_refusal())?;
        let Some(available) = now.available else {
            // Off-iOS / no live reading: nothing to refuse on beyond the cap.
            return Ok(());
        };
        if remaining > available {
            record_capacity_refusal();
            return Err(Error::Unsupported(format!(
                "iOS admission: {} MiB still to come (demand + margin {} MiB, {} MiB realized \
                 since admission) exceeds the {} MiB of per-process headroom left",
                remaining >> 20,
                required_bytes >> 20,
                realized >> 20,
                available >> 20
            )));
        }
        Ok(())
    }

    static CAPACITY_REFUSED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    fn record_capacity_refusal() {
        CAPACITY_REFUSED.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Take (and clear) whether a bound call's live capacity recheck refused since the last take:
    /// errors from a bound generator cross provider seams as messages, so the refusal is ALSO
    /// recorded here, typed, for callers (the campaign) that must keep its `capacity` kind.
    pub fn take_capacity_refusal() -> bool {
        CAPACITY_REFUSED.swap(false, std::sync::atomic::Ordering::AcqRel)
    }

    /// The per-load capacity check: readings at admission are captured once.
    fn capacity_from(at_admission: super::ProcessMemory) -> impl Fn(u64) -> Result<()> + 'static {
        move |bytes| incremental_capacity(bytes, super::process_memory(), at_admission)
    }

    /// Admit the caller's highest-preference candidate for one of `providers` (the facade's own
    /// surface; a candidate for another provider is a typed `mode` refusal).
    fn admitted(
        providers: &[&str],
        candidates: &[IosCandidate],
    ) -> AdmissionResult<(ProviderRegistry, Admission, super::ProcessMemory)> {
        require_limits_bound()?;
        if let Some(c) = candidates
            .iter()
            .find(|c| !providers.contains(&c.provider.as_str()))
        {
            return Err(AdmissionError::refused(
                RefusalKind::Mode,
                format!(
                    "{} is not served by this facade ({providers:?})",
                    c.provider
                ),
            ));
        }
        let registry = mlx_gen_ios_catalog::provider_registry()?;
        // The captured footprint is an absolute process peak: what this process already holds is
        // committed (subtracted from the prediction exactly once), and the total is the smaller of
        // the fixed cap and what the process could reach now.
        let now = super::process_memory();
        let mut budget = mlx_gen_ios_catalog::capacity_budget(
            u64::from(super::IOS_APP_MEMORY_CAP_MIB) << 20,
            now.footprint.unwrap_or(0),
            now.available,
        );
        // No blanket reserve: the predicted peak is the anchor's demand plus its MEASURED margin.
        budget.reserved_headroom_bytes = 0;
        let admission = admit(&registry, candidates, &super::device_machine(), budget)?;
        // The cap/headroom check the generic guard runs, BEFORE any weight payload.
        incremental_capacity(admission.required_bytes(), now, now)
            .map_err(|e| AdmissionError::refused(RefusalKind::Capacity, e.to_string()))?;
        Ok((registry, admission, now))
    }

    fn generator(
        providers: &[&str],
        candidates: &[IosCandidate],
    ) -> AdmissionResult<(usize, BoundGenerator)> {
        let (registry, admission, at) = admitted(providers, candidates)?;
        let index = admission.index();
        Ok((
            index,
            admission.load_generator(&registry, capacity_from(at))?,
        ))
    }

    /// SANA (`sana_1600m`, `sana_sprint_1600m`): text-to-image and image-to-image. Returns the
    /// admitted candidate's index and a wrapper bound to its exact request (geometry, decode tile,
    /// mode, prompt, execution), which rechecks live capacity before every call.
    pub fn load_sana(candidates: &[IosCandidate]) -> AdmissionResult<(usize, BoundGenerator)> {
        generator(&["sana_1600m", "sana_sprint_1600m"], candidates)
    }

    /// The SDXL family (every checkpoint route): text, image-to-image, edit/inpaint, control and
    /// IP-adapter.
    pub fn load_sdxl(candidates: &[IosCandidate]) -> AdmissionResult<(usize, BoundGenerator)> {
        generator(&[sdxl::SDXL_ID], candidates)
    }

    /// Anima Turbo. The host-planned selection ([`anima_turbo_ios_selection`]) is what to offer;
    /// only a reviewed device row admits it.
    pub fn load_anima_turbo(
        candidates: &[IosCandidate],
    ) -> AdmissionResult<(usize, BoundGenerator)> {
        generator(&[ANIMA_TURBO_ID], candidates)
    }

    /// Z-Image Turbo: `z_image_turbo` (text, image-to-image, LoRA/LoKr) and its Fun-Controlnet-Union
    /// route `z_image_turbo_control` (pose / canny / depth, optional img2img init), at the stored
    /// Q4 / Q8 / bf16 tier ([`zimage::z_image_turbo_load_spec`], [`zimage::check_artifact`]).
    pub fn load_z_image_turbo(
        candidates: &[IosCandidate],
    ) -> AdmissionResult<(usize, BoundGenerator)> {
        generator(
            &[zimage::Z_IMAGE_TURBO_ID, zimage::Z_IMAGE_TURBO_CONTROL_ID],
            candidates,
        )
    }

    /// Iris 3B generation, including prompt batches.
    pub fn load_iris(candidates: &[IosCandidate]) -> AdmissionResult<(usize, BoundGenerator)> {
        generator(&["iris_3b"], candidates)
    }

    /// The Iris 3B restoration transform.
    pub fn load_iris_restore(
        candidates: &[IosCandidate],
    ) -> AdmissionResult<(usize, BoundTransform)> {
        let (registry, admission, at) = admitted(&[IRIS_RESTORE_ID], candidates)?;
        let index = admission.index();
        Ok((
            index,
            admission.load_restorer(&registry, capacity_from(at))?,
        ))
    }

    /// The Iris 3B depth task.
    pub fn load_iris_depth(candidates: &[IosCandidate]) -> AdmissionResult<(usize, BoundDepth)> {
        let (_, admission, at) = admitted(&[IRIS_DEPTH_ID], candidates)?;
        let index = admission.index();
        Ok((index, admission.load_depth(capacity_from(at))?))
    }

    /// Explicitly labelled DEVICE CAPTURE permit for one candidate of any registered model/mode.
    /// Compiled only with the non-default `device-capture` feature (the device harness).
    /// Not admission: no evidence row is consulted or created. Granted only where every quantum is
    /// bounded by live OS headroom before its graph is constructed, and (on iOS) once the platform
    /// limits are bound. Load with the permit's `load_generator`/`load_restorer`/`load_depth`; a
    /// quantum refusal is then readable as `ios_quantum::take_refusal()`.
    #[cfg(feature = "device-capture")]
    pub fn capture_permit(
        candidate: &IosCandidate,
    ) -> AdmissionResult<(ProviderRegistry, Admission)> {
        if cfg!(target_os = "ios") {
            require_limits_bound()?;
        }
        let registry = mlx_gen_ios_catalog::provider_registry()?;
        let permit = catalog_capture_permit(&registry, candidate, &super::device_machine())?;
        Ok((registry, permit))
    }

    /// The bounded-execution check a capture load and every capture call run: the quanta are only
    /// bounded while `ios_quantum` is active.
    #[cfg(feature = "device-capture")]
    pub fn capture_capacity(_: u64) -> Result<()> {
        if ios_quantum::enabled() {
            Ok(())
        } else {
            Err(Error::Unsupported(
                "iOS capture requires bounded per-quantum execution".into(),
            ))
        }
    }
}
/// Callable Iris depth types. Loads go through `media::load_iris_depth` (admitted) or
/// `media::capture_permit` (bounded device capture); there is no unadmitted depth loader.
#[cfg(feature = "media")]
pub use mlx_gen_ios_catalog::iris_depth;
pub use mlx_llm as llm;
pub use runtime_catalog::{core_llm, gen_core, RuntimeCatalog, RuntimeCatalogSnapshot};

/// Platform label for this bundle; matches [`RuntimeCatalog::platform`].
pub const PLATFORM: &str = "ios";
/// The single tensor backend every media, LLM, and snapshot-preparer provider in this bundle uses.
pub const BACKEND: &str = "mlx";
/// Target triples this bundle is supported on.
///
/// The simulator triple is a supported **build** target, not a supported runtime: it has no Apple
/// Neural Engine and its Metal implementation differs from a device's, so performance work and any
/// kernel-correctness claim belong on real hardware.
pub const SUPPORTED_TARGET_TRIPLES: &[&str] = &["aarch64-apple-ios", "aarch64-apple-ios-sim"];
/// Native (non-Cargo) prerequisites required to build and run this bundle.
///
/// The iOS 18 floor is not arbitrary: MLX's Metal kernels are compiled against a deployment
/// target that maps to a Metal version (iOS 16 → Metal 300, 17 → 310, 18 → 320), and MLX's own
/// macOS floor is Metal 310. Building below 18.0 would also decouple the `fence` kernel (built at
/// Metal ≥ 320) from its runtime guard (`__builtin_available(macOS 15, iOS 18, *)`). See
/// `.cargo/config.toml`.
pub const NATIVE_PREREQUISITES: &[&str] = &["iOS 18.0+", "Xcode 16+ with the Metal toolchain"];

/// The bundle's media registry: the narrow iOS image catalog under `media`, empty without it.
fn media_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    #[cfg(feature = "media")]
    {
        mlx_gen_ios_catalog::provider_registry()
    }

    #[cfg(not(feature = "media"))]
    {
        gen_core::ProviderRegistryBuilder::new().build()
    }
}

/// What this bundle's LLM backend can serve on this host before any model is loaded (sc-24139).
/// MLX has no CUDA device features: `Quantize::Nvfp4` and `LoadSpec::cuda_graphs` are unavailable.
pub fn text_backend_capabilities() -> core_llm::BackendCapabilities {
    core_llm::BackendCapabilities::without_cuda("mlx", "metal")
}

/// What [`bound_mlx_to_platform_limits`] set, for a caller that wants to log or assert it.
///
/// All figures MiB. `previous_*` are what MLX had chosen for itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlxLimits {
    /// `os_proc_available_memory()` at the time of the call.
    ///
    /// **This is headroom REMAINING, not the total cap** — it shrinks as the process grows, which is
    /// why the call belongs at startup. See [`MlxLimits::called_late`].
    pub os_available_mib: f64,
    /// MLX's live allocation when the call was made. Near zero in a freshly started process.
    pub active_at_call_mib: f64,
    pub previous_memory_limit_mib: f64,
    pub previous_cache_limit_mib: f64,
    pub memory_limit_mib: f64,
    pub cache_limit_mib: f64,
}

impl MlxLimits {
    /// Whether the call actually lowered anything. False if MLX's own sizing was already tighter
    /// than the platform budget.
    pub fn changed(&self) -> bool {
        self.memory_limit_mib < self.previous_memory_limit_mib
            || self.cache_limit_mib < self.previous_cache_limit_mib
    }

    /// The call happened after a model was already resident, so the budget it derived is too small.
    ///
    /// `os_proc_available_memory()` reports headroom *remaining*. Called at startup it is
    /// effectively the cap; called with 3 GB of weights loaded it reports what is left, so a naive
    /// 85% of it can land **below the memory already live**. [`bound_mlx_to_platform_limits`]
    /// clamps rather than setting such a limit, and this reports that it did.
    pub fn called_late(&self) -> bool {
        self.active_at_call_mib > self.os_available_mib * 0.1
    }
}

/// The ceiling on MLX's buffer cache that [`platform_limits_mib`] applies, MiB.
///
/// With `media`, this IS `mlx_gen_ios_catalog::IOS_MLX_CACHE_LIMIT_CAP_MIB`: the catalog's inferred
/// SANA footprint bounds assume exactly this cap, so the image build takes the constant from there
/// rather than restating it. An LLM-only build does not compile the catalog (that is the point of
/// disabling `media`) and has no inferred bound to agree with, so it carries the value itself.
#[cfg(feature = "media")]
pub const MLX_CACHE_LIMIT_CAP_MIB: u32 = mlx_gen_ios_catalog::IOS_MLX_CACHE_LIMIT_CAP_MIB;
/// The ceiling on MLX's buffer cache that [`platform_limits_mib`] applies, MiB.
#[cfg(not(feature = "media"))]
pub const MLX_CACHE_LIMIT_CAP_MIB: u32 = 1024;

/// The iOS per-app memory cap admission refuses against, MiB (epic 16774, E4).
///
/// **PROVISIONAL — UNMEASURED.** With `media` this IS `mlx_gen_ios_catalog::IOS_MEMORY_BUDGET_MIB`
/// (the one cap constant; see its docs): 4096 is an *assumed* 8 GB-device cap carried from the
/// `claude/ios-strategy` branch. The iPhone 16 Pro Max launch readings were 6142 MiB entitled
/// and 3374 MiB unentitled (three identical readings each). Those are diagnostics, not a validated
/// safe rendering envelope: DO NOT replace this provisional cap with a launch reading.
#[cfg(feature = "media")]
pub const IOS_APP_MEMORY_CAP_MIB: u32 = mlx_gen_ios_catalog::IOS_MEMORY_BUDGET_MIB;
/// The iOS per-app memory cap admission refuses against, MiB. **PROVISIONAL — UNMEASURED.**
///
/// An LLM-only build does not compile the catalog, so it carries the value itself; a test asserts it
/// equals the catalog's constant, so the two cannot drift.
#[cfg(not(feature = "media"))]
pub const IOS_APP_MEMORY_CAP_MIB: u32 = 4096;

/// Admit a request whose footprint estimate is `estimate_mib`, against [`IOS_APP_MEMORY_CAP_MIB`].
///
/// Returns the admitted estimate. Refuses with [`gen_core::Error::Unsupported`] when the estimate
/// exceeds the cap (an exact fit is admitted), and when there is **no** estimate: an unmeasured fit
/// is not a fit, and on iOS the alternative to refusing is a jetsam kill, not backpressure.
pub fn admit_estimate_mib(estimate_mib: Option<u32>) -> gen_core::Result<u32> {
    match estimate_mib {
        None => Err(gen_core::Error::Unsupported(format!(
            "iOS admission: request has no footprint estimate; refusing rather than risk a jetsam \
             kill against the {IOS_APP_MEMORY_CAP_MIB} MiB per-app cap"
        ))),
        Some(estimate) if estimate > IOS_APP_MEMORY_CAP_MIB => {
            Err(gen_core::Error::Unsupported(format!(
                "iOS admission: estimated footprint {estimate} MiB exceeds the \
                 {IOS_APP_MEMORY_CAP_MIB} MiB per-app cap"
            )))
        }
        Some(estimate) => Ok(estimate),
    }
}

/// Refuse an admitted `estimate_mib` that exceeds the per-process headroom left **now**.
///
/// The cap check alone ignores memory already resident: headroom does not recover across renders
/// in one process (measured 4664 -> 1223 MiB across a single SANA render), so a request under the
/// cap can still be over what is left. `available_mib` is `os_proc_available_memory()`; `None`
/// (off-iOS, or the simulator reporting no budget) means there is no live reading to refuse on.
fn admit_within_headroom(estimate_mib: u32, available_mib: Option<f64>) -> gen_core::Result<u32> {
    match available_mib {
        Some(available) if f64::from(estimate_mib) > available => {
            Err(gen_core::Error::Unsupported(format!(
                "iOS admission: estimated footprint {estimate_mib} MiB exceeds the {available:.0} \
                 MiB of per-process headroom left"
            )))
        }
        _ => Ok(estimate_mib),
    }
}

/// Run `load` only if `estimate_mib` is admitted: present, within [`IOS_APP_MEMORY_CAP_MIB`], and
/// within the per-process headroom iOS reports right now. **The only admitted load entry point.**
///
/// The refusal happens **before** `load` is called, so a refused request never touches its weights.
pub fn load_admitted<T>(
    estimate_mib: Option<u32>,
    load: impl FnOnce() -> gen_core::Result<T>,
) -> gen_core::Result<T> {
    let estimate = admit_estimate_mib(estimate_mib)?;
    admit_within_headroom(estimate, os_proc_available_mib())?;
    load()
}

/// Set once [`bound_mlx_to_platform_limits`] has capped MLX's buffer cache in this process — the
/// precondition for every admitted image load (a captured footprint was measured with the cache
/// capped; an uncapped process could exceed it).
static PLATFORM_LIMITS_BOUND: AtomicBool = AtomicBool::new(false);

/// Hardware identity is independent of total RAM and launch-time available memory.
#[cfg(feature = "media")]
pub fn device_machine() -> String {
    #[cfg(all(target_os = "ios", not(target_abi = "sim")))]
    {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                oldp: *mut std::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut std::ffi::c_void,
                newlen: usize,
            ) -> i32;
        }
        let mut buffer = [0u8; 128];
        let mut size = buffer.len();
        let status = unsafe {
            sysctlbyname(
                c"hw.machine".as_ptr(),
                buffer.as_mut_ptr().cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if status == 0 && size > 1 && size <= buffer.len() {
            if let Ok(value) = std::str::from_utf8(&buffer[..size - 1]) {
                return value.to_owned();
            }
        }
        "unknown-ios-device".into()
    }
    #[cfg(not(all(target_os = "ios", not(target_abi = "sim"))))]
    {
        "non-device-host".into()
    }
}

/// The `(memory_limit, cache_limit)` pair, MiB, for a process with `available_mib` of per-process
/// headroom and `active_mib` already live.
///
/// * **Cache** — `min(25% of budget, `[`MLX_CACHE_LIMIT_CAP_MIB`]`)` (1 GiB): large enough that tile-to-tile buffer reuse still
///   hits, small enough to leave room beside a multi-GB transient.
/// * **Live allocation** — 85% of the budget, so MLX applies backpressure *before* jetsam applies a
///   kill — but never below 110% of what is already live, because a ceiling under the resident set
///   protects nothing and stalls the process against a limit it has already passed.
pub fn platform_limits_mib(available_mib: f64, active_mib: f64) -> (f64, f64) {
    let cache = (available_mib / 4.0).min(f64::from(MLX_CACHE_LIMIT_CAP_MIB));
    let memory = (available_mib * 0.85).max(active_mib * 1.1);
    (memory, cache)
}

/// Bind MLX's memory and buffer-cache limits to the **per-process** budget iOS enforces.
///
/// Call once at app startup, before loading any model. Returns what it set, or `None` where there
/// is no per-process cap to bind to (any non-iOS target, or iOS reporting no budget) — in which case
/// MLX is not touched at all, because its own sizing from the recommended working set is already
/// correct there and capping it would slow a Mac to imitate a constraint it does not have.
///
/// # Why this is necessary, and why it is not automatic
///
/// MLX sizes its memory and buffer-cache limits from the system's *recommended working set*. An
/// iOS app is bounded instead by a per-process jetsam limit far below device RAM, and nothing tells
/// MLX so. Measured on an iPhone 17 Pro Max (12 GB): `memory_limit=11109 MiB`,
/// `cache_limit=11109 MiB`, `os_proc_available_memory=6014 MiB` — MLX applies backpressure at a
/// threshold the process can never reach, and its cache returns nothing to the OS. A Z-Image decode
/// traced on device was killed with 4 MiB of headroom while holding ~3.9 GB of *reclaimable* cache;
/// binding the limits let the identical request finish with 2146 MiB still free. Jetsam counts
/// `phys_footprint`, which includes the cache that `get_active_memory`/`get_peak_memory` exclude.
///
/// **Not called from [`catalog`].** Building a catalog is a pure composition step, and this mutates
/// process-global allocator state. An app opts in explicitly, once, where it can log the result.
pub fn bound_mlx_to_platform_limits() -> Option<MlxLimits> {
    let available_mib = os_proc_available_mib()?;
    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);

    // MLX exposes setters that return the prior value and no getter for the cache limit, so the
    // prior cache limit is read by setting and restoring.
    let previous_memory_limit = mlx_rs::memory::get_memory_limit();
    let previous_cache_limit = {
        let prev = mlx_rs::memory::set_cache_limit(0);
        mlx_rs::memory::set_cache_limit(prev);
        prev
    };
    let active_at_call_mib = mib(mlx_rs::memory::get_active_memory());

    let (memory_mib, cache_mib) = platform_limits_mib(available_mib, active_at_call_mib);
    let to_bytes = |m: f64| (m * 1024.0 * 1024.0) as usize;
    mlx_rs::memory::set_memory_limit(to_bytes(memory_mib));
    mlx_rs::memory::set_cache_limit(to_bytes(cache_mib));
    // The cache is now capped at `MLX_CACHE_LIMIT_CAP_MIB` (or tighter), so inferred footprint
    // bounds hold from here on.
    PLATFORM_LIMITS_BOUND.store(true, Ordering::Release);

    Some(MlxLimits {
        os_available_mib: available_mib,
        active_at_call_mib,
        previous_memory_limit_mib: mib(previous_memory_limit),
        previous_cache_limit_mib: mib(previous_cache_limit),
        memory_limit_mib: memory_mib,
        cache_limit_mib: cache_mib,
    })
}

/// One reading of this process's memory: the kernel `phys_footprint` and the live per-process
/// headroom (`os_proc_available_memory`), bytes. Both `None` off-iOS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProcessMemory {
    pub footprint: Option<u64>,
    pub available: Option<u64>,
}

/// Read [`ProcessMemory`] now.
pub fn process_memory() -> ProcessMemory {
    ProcessMemory {
        footprint: phys_footprint_bytes(),
        available: os_proc_available_mib().map(|mib| (mib * 1048576.0).floor() as u64),
    }
}

/// `TASK_VM_INFO.phys_footprint` (mach/task_info.h, `#pragma pack(4)`: words 36-37, present from
/// REV1). The reply below REV1 or a failed call is `None`.
fn phys_footprint_bytes() -> Option<u64> {
    #[cfg(target_os = "ios")]
    {
        const TASK_VM_INFO: i32 = 22;
        const PHYS_FOOTPRINT_WORD: usize = 36;
        let mut words = [0u32; 128];
        let mut count = words.len() as u32;
        #[allow(deprecated)]
        let status = unsafe {
            libc::task_info(
                libc::mach_task_self(),
                TASK_VM_INFO as u32,
                words.as_mut_ptr().cast(),
                &mut count,
            )
        };
        (status == 0 && count as usize >= PHYS_FOOTPRINT_WORD + 2).then(|| {
            u64::from(words[PHYS_FOOTPRINT_WORD])
                | (u64::from(words[PHYS_FOOTPRINT_WORD + 1]) << 32)
        })
    }
    #[cfg(not(target_os = "ios"))]
    {
        None
    }
}

/// `os_proc_available_memory()` in MiB, or `None` where it does not exist or reports no budget.
fn os_proc_available_mib() -> Option<f64> {
    #[cfg(target_os = "ios")]
    {
        extern "C" {
            fn os_proc_available_memory() -> usize;
        }
        // SAFETY: no arguments, no pointers; returns 0 when unavailable.
        let bytes = unsafe { os_proc_available_memory() };
        (bytes > 0).then(|| bytes as f64 / (1024.0 * 1024.0))
    }
    #[cfg(not(target_os = "ios"))]
    {
        None
    }
}

/// Build the complete validated iOS runtime composition.
///
/// No audio lane: that lane is the one sanctioned cross-backend seam (sc-12835) and is not
/// extended to a new platform incidentally.
pub fn catalog() -> runtime_catalog::Result<RuntimeCatalog> {
    RuntimeCatalog::try_new(
        PLATFORM,
        BACKEND,
        media_registry(),
        mlx_llm::catalog_text_registry(),
        mlx_llm::snapshot_preparer_registry(),
    )
}

#[cfg(test)]
mod tests {
    /// Off-iOS there is no per-process cap, so binding must be a no-op that says so — and must not
    /// touch MLX's allocator at all (a regression that lowered MLX's cache on a Mac would slow every
    /// host render invisibly).
    #[cfg(not(target_os = "ios"))]
    #[test]
    fn binding_limits_is_a_no_op_without_a_per_process_cap() {
        assert_eq!(super::bound_mlx_to_platform_limits(), None);
    }

    /// The limit arithmetic: backpressure below the budget, a 1 GiB cache ceiling, and a late call
    /// clamped above what is already live rather than under it.
    #[test]
    fn platform_limits_sit_below_the_budget_and_never_under_the_live_set() {
        let (memory, cache) = super::platform_limits_mib(6014.0, 10.0);
        assert!((memory - 6014.0 * 0.85).abs() < 1e-9);
        assert_eq!(cache, 1024.0, "cache must be capped at 1 GiB");
        assert_eq!(cache, f64::from(super::MLX_CACHE_LIMIT_CAP_MIB));
        assert!(memory < 6014.0, "backpressure must precede a jetsam kill");

        let (memory, cache) = super::platform_limits_mib(2000.0, 10.0);
        assert_eq!(cache, 500.0, "a small budget caches a quarter of it");
        assert!((memory - 1700.0).abs() < 1e-9);

        // Late call: 3000 MiB already live, 1000 MiB headroom left.
        let (memory, _) = super::platform_limits_mib(1000.0, 3000.0);
        assert!((memory - 3300.0).abs() < 1e-9, "clamped to 110% of live");
        let late = super::MlxLimits {
            os_available_mib: 1000.0,
            active_at_call_mib: 3000.0,
            previous_memory_limit_mib: 11109.0,
            previous_cache_limit_mib: 11109.0,
            memory_limit_mib: memory,
            cache_limit_mib: 250.0,
        };
        assert!(late.called_late());
        assert!(late.changed());
    }

    /// E4: a request whose estimate exceeds the cap is refused BEFORE its loader runs; an exact fit
    /// and anything under it load; an absent estimate is refused, also before loading.
    #[test]
    fn admission_refuses_over_cap_estimates_before_any_weight_load() {
        use std::cell::Cell;
        let cap = super::IOS_APP_MEMORY_CAP_MIB;
        let loads = Cell::new(0u32);
        let load = || {
            loads.set(loads.get() + 1);
            Ok("loaded")
        };

        let refused = super::load_admitted(Some(cap + 1), load);
        assert!(
            matches!(refused, Err(super::gen_core::Error::Unsupported(_))),
            "an estimate over the cap must be refused: {refused:?}"
        );
        assert_eq!(loads.get(), 0, "refusal must precede the weight load");

        assert!(matches!(
            super::load_admitted(None, load),
            Err(super::gen_core::Error::Unsupported(_))
        ));
        assert_eq!(
            loads.get(),
            0,
            "an unestimated request must not load either"
        );

        assert_eq!(super::load_admitted(Some(cap), load).unwrap(), "loaded");
        assert_eq!(super::load_admitted(Some(cap - 1), load).unwrap(), "loaded");
        assert_eq!(loads.get(), 2, "admitted requests load exactly once each");
    }

    /// Already-resident memory counts: an estimate under the cap but over the headroom left now is
    /// refused; an exact fit and no live reading (off-iOS) are admitted.
    #[test]
    fn admission_refuses_estimates_over_the_live_headroom() {
        assert!(matches!(
            super::admit_within_headroom(1300, Some(1223.0)),
            Err(super::gen_core::Error::Unsupported(_))
        ));
        assert_eq!(
            super::admit_within_headroom(1223, Some(1223.0)).unwrap(),
            1223
        );
        assert_eq!(super::admit_within_headroom(4096, None).unwrap(), 4096);
    }

    /// There is ONE cap constant: the LLM-only literal must equal the catalog's.
    #[test]
    fn the_app_memory_cap_is_the_catalogs_single_constant() {
        assert_eq!(
            super::IOS_APP_MEMORY_CAP_MIB,
            mlx_gen_ios_catalog::IOS_MEMORY_BUDGET_MIB
        );
    }

    /// Tiny weights-free (header-only) fixtures for every admitted facade's surface.
    #[cfg(feature = "media")]
    pub(crate) mod fixtures {
        use crate::gen_core::{GenerationRequest, Image, MemoryStrategy, WeightsSource};
        use crate::media::{build_candidate, IosCandidate, IosWork};
        use std::path::{Path, PathBuf};

        pub fn tensor(path: &Path, key: &str) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let header = format!(
                "{{\"{key}\":{{\"dtype\":\"F32\",\"shape\":[8,8],\"data_offsets\":[0,256]}}}}"
            );
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header.as_bytes());
            bytes.resize(bytes.len() + 256, 0);
            std::fs::write(path, bytes).unwrap();
        }

        fn tokenizer(dir: &Path) {
            std::fs::create_dir_all(dir).unwrap();
            let mut vocab = String::from("{");
            for (i, c) in ('a'..='z').enumerate() {
                vocab += &format!("\"{c}\":{},\"{c}</w>\":{},", 2 * i, 2 * i + 1);
            }
            vocab += "\"<|startoftext|>\":49406,\"<|endoftext|>\":49407}";
            std::fs::write(dir.join("vocab.json"), vocab).unwrap();
            std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
        }

        pub fn iris_fixtures() -> PathBuf {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../media/mlx-gen/mlx-gen-iris/tests/fixtures")
        }

        fn copy_tree(from: &Path, to: &Path) {
            std::fs::create_dir_all(to).unwrap();
            for entry in std::fs::read_dir(from).unwrap() {
                let entry = entry.unwrap();
                let target = to.join(entry.file_name());
                if entry.path().is_dir() {
                    copy_tree(&entry.path(), &target);
                } else {
                    std::fs::copy(entry.path(), target).unwrap();
                }
            }
        }

        fn request(edge: u32) -> GenerationRequest {
            GenerationRequest {
                prompt: "p".into(),
                width: edge,
                height: edge,
                count: 1,
                ..Default::default()
            }
        }

        fn image(edge: u32) -> Image {
            Image {
                width: edge,
                height: edge,
                pixels: (0..edge * edge * 3).map(|i| (i % 251) as u8).collect(),
            }
        }

        /// One candidate per admitted facade, keyed by provider id.
        pub fn candidates(root: &Path) -> Vec<IosCandidate> {
            let registry = mlx_gen_ios_catalog::provider_registry().unwrap();
            let sana = root.join("sana");
            tensor(&sana.join("transformer/a.safetensors"), "proj.weight");
            tensor(
                &sana.join("text_encoder/gemma-2-2b-it.safetensors"),
                "model.embed_tokens.weight",
            );
            tensor(&sana.join("vae/a.safetensors"), "conv.weight");
            let sdxl = root.join("sdxl");
            for c in ["unet", "vae", "text_encoder", "text_encoder_2"] {
                tensor(&sdxl.join(c).join("model.safetensors"), "x");
            }
            tokenizer(&sdxl.join("tokenizer"));
            tokenizer(&sdxl.join("tokenizer_2"));
            let anima = root.join("anima");
            for c in ["diffusion_models", "text_encoders", "vae"] {
                tensor(
                    &anima.join("split_files").join(c).join("f.safetensors"),
                    "x",
                );
            }
            let zimage = root.join("zimage");
            for component in ["transformer", "vae"] {
                tensor(&zimage.join(component).join("model.safetensors"), "x");
                std::fs::write(
                    zimage.join(component).join("config.json"),
                    r#"{"quantization":{"bits":4,"group_size":64}}"#,
                )
                .unwrap();
            }
            gen_core_testkit::write_encoder_contract_fixture_with_quant(
                &zimage.join("text_encoder"),
                mlx_gen_ios_catalog::zimage::ENCODER_CONTRACT,
                Some(4),
            )
            .unwrap();
            tensor(&root.join("zimage-control.safetensors"), "x");
            let zimage_spec = |control: bool| {
                let mut spec = crate::media::zimage::z_image_turbo_load_spec(
                    WeightsSource::Dir(zimage.clone()),
                    control.then(|| WeightsSource::File(root.join("zimage-control.safetensors"))),
                );
                spec.quantize = Some(crate::gen_core::Quant::Q4);
                spec
            };
            let iris = root.join("iris");
            copy_tree(&iris_fixtures().join("tiny-snapshot/iris"), &iris);
            let mut iris_spec = crate::gen_core::LoadSpec::new(WeightsSource::Dir(iris));
            iris_spec.components.insert(
                "text_encoder".into(),
                WeightsSource::Dir(iris_fixtures().join("tiny-snapshot/text_encoder")),
            );
            let restore = crate::gen_core::LoadSpec::new(WeightsSource::Dir(
                iris_fixtures().join("tiny-snapshot/upscaler"),
            ));
            let depth = crate::gen_core::LoadSpec::new(WeightsSource::Dir(
                iris_fixtures().join("tiny-depth"),
            ));
            let gen = |provider: &str, spec, strategy| {
                build_candidate(
                    &registry,
                    provider,
                    spec,
                    None,
                    strategy,
                    IosWork::Generate(request(512)),
                )
                .unwrap_or_else(|e| panic!("{provider}: {e}"))
            };
            vec![
                gen(
                    "sana_1600m",
                    crate::media::sana_load_spec(WeightsSource::Dir(sana)),
                    MemoryStrategy::BoundedDecode,
                ),
                gen(
                    "sdxl",
                    crate::media::sdxl::sdxl_load_spec("sdxl", WeightsSource::Dir(sdxl)).unwrap(),
                    MemoryStrategy::BoundedDecode,
                ),
                gen(
                    "anima_turbo",
                    crate::media::anima_turbo_load_spec(WeightsSource::Dir(anima)),
                    MemoryStrategy::BoundedDecode,
                ),
                gen("iris_3b", iris_spec, MemoryStrategy::Resident),
                gen(
                    "z_image_turbo",
                    zimage_spec(false),
                    MemoryStrategy::BoundedTransformerResidency,
                ),
                build_candidate(
                    &registry,
                    "z_image_turbo_control",
                    zimage_spec(true),
                    None,
                    MemoryStrategy::BoundedTransformerResidency,
                    IosWork::Generate(GenerationRequest {
                        conditioning: vec![crate::gen_core::Conditioning::Control {
                            image: image(512),
                            kind: crate::gen_core::ControlKind::Pose,
                            scale: Some(1.0),
                        }],
                        ..request(512)
                    }),
                )
                .unwrap(),
                build_candidate(
                    &registry,
                    crate::media::IRIS_RESTORE_ID,
                    restore,
                    None,
                    MemoryStrategy::Resident,
                    IosWork::Restore(crate::gen_core::TransformRequest {
                        image: image(16),
                        ..Default::default()
                    }),
                )
                .unwrap(),
                build_candidate(
                    &registry,
                    crate::media::IRIS_DEPTH_ID,
                    depth,
                    None,
                    MemoryStrategy::Resident,
                    IosWork::Depth(crate::iris_depth::DepthRequest {
                        image: image(16),
                        ..Default::default()
                    }),
                )
                .unwrap(),
            ]
        }
    }

    /// Findings 3/4: every image surface has an admitted facade; each requires the platform limits
    /// bound (a typed failure, before evidence), then refuses with the typed `no_evidence` kind
    /// while the reviewed table has no row — before any weight load. A candidate offered to the
    /// wrong facade is a typed `mode` refusal, never silently routed elsewhere.
    #[cfg(feature = "media")]
    #[test]
    fn every_admitted_facade_requires_bound_limits_and_refuses_without_evidence() {
        use crate::media::{self, AdmissionError, RefusalKind};
        let root = tempfile::tempdir().unwrap();
        let candidates = fixtures::candidates(root.path());
        type Facade = fn(&[media::IosCandidate]) -> Result<usize, AdmissionError>;
        let facades: [(&str, Facade); 8] = [
            ("sana_1600m", |c| media::load_sana(c).map(|r| r.0)),
            ("sdxl", |c| media::load_sdxl(c).map(|r| r.0)),
            ("anima_turbo", |c| media::load_anima_turbo(c).map(|r| r.0)),
            ("iris_3b", |c| media::load_iris(c).map(|r| r.0)),
            ("z_image_turbo", |c| {
                media::load_z_image_turbo(c).map(|r| r.0)
            }),
            ("z_image_turbo_control", |c| {
                media::load_z_image_turbo(c).map(|r| r.0)
            }),
            ("iris_3b_restore", |c| {
                media::load_iris_restore(c).map(|r| r.0)
            }),
            ("iris_3b_depth", |c| media::load_iris_depth(c).map(|r| r.0)),
        ];
        let kind = |r: Result<usize, AdmissionError>| match r {
            Err(AdmissionError::Refused(r)) => Some(r.kind),
            Err(AdmissionError::Failed(_)) => None,
            Ok(_) => panic!("admitted without evidence"),
        };
        super::PLATFORM_LIMITS_BOUND.store(false, super::Ordering::Release);
        for (id, facade) in facades.iter() {
            let candidate = candidates.iter().find(|c| c.provider == *id).unwrap();
            assert_eq!(
                kind(facade(std::slice::from_ref(candidate))),
                None,
                "{id}: limits must be bound first"
            );
        }
        super::PLATFORM_LIMITS_BOUND.store(true, super::Ordering::Release);
        let by_id = |id: &str| candidates.iter().find(|c| c.provider == id).unwrap();
        for (id, facade) in facades.iter() {
            let candidate = by_id(id);
            assert_eq!(
                kind(facade(std::slice::from_ref(candidate))),
                Some(RefusalKind::NoEvidence),
                "{id}"
            );
            let other = candidates
                .iter()
                .find(|c| !c.provider.starts_with(&id[..4]))
                .unwrap();
            assert_eq!(
                kind(facade(std::slice::from_ref(other))),
                Some(RefusalKind::Mode),
                "{id} accepted a {} candidate",
                other.provider
            );
        }
        super::PLATFORM_LIMITS_BOUND.store(false, super::Ordering::Release);
    }

    /// The per-call recheck charges only the demand still to come: weights loaded since admission
    /// are in the process footprint and already absent from live headroom, so they are not counted
    /// twice. First and second renders on a loaded generator fit; a genuinely short headroom and an
    /// over-cap footprint still refuse. The charge is `demand + measured margin`, with no 512 MiB
    /// formula reserve stacked on the measurement.
    #[cfg(feature = "media")]
    #[test]
    fn the_loaded_generator_recheck_charges_only_incremental_demand() {
        use super::media::incremental_capacity;
        use super::ProcessMemory;
        const MIB: u64 = 1 << 20;
        // The anchor: 3368 MiB of growth over its capture baseline + its 32 MiB margin.
        let required = 3400 * MIB;
        let read = |footprint: u64, available: u64| ProcessMemory {
            footprint: Some(footprint * MIB),
            available: Some(available * MIB),
        };
        // Admission needs exactly demand + margin of live headroom: 3400 fits, 3399 does not.
        let exact = read(200, 3400);
        incremental_capacity(required, exact, exact).unwrap();
        let short = read(200, 3399);
        assert!(incremental_capacity(required, short, short).is_err());
        let fresh = read(200, 5900);
        incremental_capacity(required, fresh, fresh).unwrap();
        // An unrelated resident (SANA ~2 GB still loaded): 1800 MiB available. The growth is 3400:
        // refused (a product's residents are never credited).
        let busy = read(2200, 1800);
        let _ = super::media::take_capacity_refusal();
        assert!(incremental_capacity(required, busy, busy).is_err());
        assert!(
            super::media::take_capacity_refusal(),
            "typed capacity record"
        );
        // First render after loading 2000 MiB of weights: 1400 still to come <= 3900.
        incremental_capacity(required, read(2200, 3900), fresh).unwrap();
        // Second render on the same generator: the whole demand already realized.
        incremental_capacity(required, read(3600, 2500), fresh).unwrap();
        // The boundary is the remaining growth itself: exactly 1400 fits, one MiB less not.
        incremental_capacity(required, read(2200, 1400), fresh).unwrap();
        assert!(incremental_capacity(required, read(2200, 1399), fresh).is_err());
        // Kernel footprint unavailable: realized growth derived from the headroom consumed.
        let blind = ProcessMemory {
            footprint: None,
            available: Some(3900 * MIB),
        };
        incremental_capacity(required, blind, fresh).unwrap();
        // Over the single cap: 3000 held + 1400 to come > 4096.
        assert!(incremental_capacity(required, read(3000, 9000), read(1600, 9000)).is_err());
        // Off-device (no live reading): only the cap applies.
        incremental_capacity(required, ProcessMemory::default(), ProcessMemory::default()).unwrap();
    }

    /// The capture permit is not an admission and is unavailable off-device: without bounded
    /// quanta it is a typed failure before any load.
    #[cfg(all(feature = "device-capture", not(target_os = "ios")))]
    #[test]
    fn capture_permit_requires_bounded_quanta() {
        let root = tempfile::tempdir().unwrap();
        for candidate in fixtures::candidates(root.path()) {
            assert!(matches!(
                crate::media::capture_permit(&candidate),
                Err(crate::media::AdmissionError::Failed(_))
            ));
        }
        assert!(crate::media::capture_capacity(0).is_err());
    }

    #[cfg(feature = "media")]
    #[test]
    fn media_facade_preserves_anima_turbo_surface() {
        let spec = super::media::anima_turbo_load_spec(super::gen_core::WeightsSource::Dir(
            "/no-weights".into(),
        ));
        assert_eq!(
            spec.offload_policy,
            super::gen_core::OffloadPolicy::Sequential
        );
        assert_eq!(
            spec.load_shape,
            super::gen_core::LoadShape::DeferredMaterialization
        );
        assert_eq!(super::media::ANIMA_TURBO_ID, "anima_turbo");
        assert_eq!(
            super::media::ios_memory_budget().total_bytes,
            u64::from(super::IOS_APP_MEMORY_CAP_MIB) << 20
        );
        // Planning stays reachable; admission is the evidence facade (`load_anima_turbo`).
        let _ = super::media::anima_turbo_planning_decision;
        let _ = super::media::anima_turbo_ios_selection;
        let _ = super::media::anima_turbo_memory_selection;
        let _ = super::media::load_anima_turbo;
        assert!(!super::media::anima::ANIMA_HOST_REQUEST_PEAKS.is_empty());
    }

    /// The bundle's exact, ordered surface. This is the reviewable source of truth for what ships
    /// on iOS: adding a provider means editing this test deliberately.
    #[test]
    fn catalog_surface_is_explicit_and_machine_readable() {
        let snapshot = super::catalog().unwrap().snapshot();

        assert_eq!(snapshot.platform, "ios");
        assert_eq!(snapshot.backend, "mlx");

        // The media surface is SANA + Anima Turbo + SDXL + Iris + Z-Image Turbo under `media`, empty
        // without it. An
        // incidental `mlx-gen-catalog` dependency — every provider including video — fails here
        // rather than quietly compiling into an iPhone app.
        #[cfg(feature = "media")]
        assert_eq!(
            snapshot.generator_ids,
            [
                "sana_1600m",
                "sana_sprint_1600m",
                "anima_turbo",
                "sdxl",
                "iris_3b",
                "z_image_turbo",
                "z_image_turbo_control"
            ],
            "the iOS image surface changed; registration is not a device fitting claim"
        );
        #[cfg(not(feature = "media"))]
        assert!(snapshot.generator_ids.is_empty());

        // The same evidence-gated `mlx-llm` catalog runtime-macos ships, on the same backend.
        assert_eq!(
            snapshot.text_llm_ids,
            [
                "mlx-llama",
                "mlx-joycaption",
                "mlx-starvector-1b",
                "mlx-starvector-8b",
            ]
        );
        assert_eq!(snapshot.snapshot_preparer_backends, ["mlx"]);

        // Iris restoration is a transform, not a generator; keep the mode registered on iOS.
        #[cfg(feature = "media")]
        assert_eq!(snapshot.transform_ids, ["iris_3b_restore"]);
        #[cfg(not(feature = "media"))]
        assert!(snapshot.transform_ids.is_empty());
        // Never, on either profile: no training or unrelated full-catalog routes.
        assert!(snapshot.trainer_ids.is_empty());
        assert!(snapshot.captioner_ids.is_empty());
        assert!(snapshot.image_embedder_ids.is_empty());
        assert!(snapshot.text_embedder_ids.is_empty());

        // No audio lane is declared.
        assert!(snapshot.audio_backend.is_none());
        assert!(snapshot.audio_generator_ids.is_empty());
    }

    /// The threading contract is enforced by the **type system**: a loaded provider holds MLX
    /// `Array`s and MLX's default Metal device is not thread-safe, so `Box<dyn TextLlm>` must stay
    /// `!Send`. On iOS a Swift host hopping a provider across queues would race MLX's device and
    /// crash intermittently. `impls_send` is resolved by autoref specialization, so this reports
    /// what the type system decided rather than what a comment claims.
    #[test]
    // The explicit `&` on each call is the whole mechanism: it lets method resolution fall through
    // to the trait impl when `T: !Send`. Removing the borrows would make the check vacuous.
    #[allow(clippy::needless_borrow)]
    fn provider_is_not_send_so_cross_thread_use_cannot_compile() {
        struct Wrap<T>(std::marker::PhantomData<T>);
        trait NotSend {
            fn impls_send(&self) -> bool {
                false
            }
        }
        impl<T> NotSend for &Wrap<T> {}
        impl<T: Send> Wrap<T> {
            fn impls_send(&self) -> bool {
                true
            }
        }

        let provider = Wrap::<Box<dyn super::core_llm::TextLlm>>(std::marker::PhantomData);
        assert!(
            !(&provider).impls_send(),
            "Box<dyn TextLlm> became Send: a host could now move a provider across threads, \
             which races MLX's non-thread-safe Metal device"
        );

        // Control: a plainly-Send type reports true, so a false negative above would be caught.
        let snapshot = Wrap::<super::RuntimeCatalogSnapshot>(std::marker::PhantomData);
        assert!((&snapshot).impls_send(), "control type should be Send");
    }

    /// The declared triples are the iOS device and simulator triples.
    #[test]
    fn supported_triples_are_ios_only() {
        assert_eq!(
            super::SUPPORTED_TARGET_TRIPLES,
            ["aarch64-apple-ios", "aarch64-apple-ios-sim"]
        );
    }
}

// This bundle-only consumer imports no provider crate. Keep the CPU subprocess/Metal guard so
// the miniature callable task test cannot contend with the host GPU.
#[cfg(all(test, feature = "device-capture", target_os = "macos"))]
#[path = "../../../media/mlx-gen/mlx-gen-iris/tests/common/metal_guard.rs"]
mod iris_cpu_guard;

#[cfg(all(test, feature = "device-capture", target_os = "macos"))]
#[test]
fn bundle_only_consumer_executes_iris_depth() {
    use crate::{gen_core, iris_depth};
    use iris_depth::IrisDepthEstimator;
    const CHILD: &str = "SW_IOS_DEPTH_CPU_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "bundle_only_consumer_executes_iris_depth",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "{}",
            String::from_utf8_lossy(&child.stderr)
        );
        assert!(String::from_utf8_lossy(&child.stdout).contains("iris-depth-body-complete"));
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    iris_cpu_guard::forbid_metal();
    assert!(iris_cpu_guard::metal_forbidden());
    // The bundle's ONLY depth load paths are admitted or a bounded capture; this drives the capture
    // path end to end on CPU, with the bounded-quantum fixture standing in for a device's headroom.
    crate::media::ios_quantum::with_cpu_diagnostic_headroom(1 << 34, || {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../media/mlx-gen/mlx-gen-iris/tests/fixtures/tiny-depth");
        let spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(path));
        let req = iris_depth::DepthRequest {
            image: gen_core::Image {
                width: 16,
                height: 16,
                pixels: vec![127; 16 * 16 * 3],
            },
            resolution: iris_depth::DepthResolution::Native,
            ..Default::default()
        };
        let registry = mlx_gen_ios_catalog::provider_registry().unwrap();
        let candidate = crate::media::build_candidate(
            &registry,
            crate::media::IRIS_DEPTH_ID,
            spec,
            None,
            gen_core::MemoryStrategy::Resident,
            crate::media::IosWork::Depth(req.clone()),
        )
        .unwrap();
        let (_, permit) = crate::media::capture_permit(&candidate).unwrap();
        assert!(permit.is_capture());
        let model = permit.load_depth(crate::media::capture_capacity).unwrap();
        let output = model.estimate(&req, &mut |_| {}).unwrap();
        assert_eq!((output.map.width, output.map.height), (16, 16));
        assert_eq!(output.map.values.len(), 256);
        assert!(output.map.values.iter().all(|v| v.is_finite()));
        let mut other = req.clone();
        other.image.pixels[0] = 0;
        assert!(
            model.estimate(&other, &mut |_| {}).is_err(),
            "the capture is bound to its exact work"
        );
        println!("iris-depth-body-complete");
        Ok(())
    })
    .unwrap();
}
