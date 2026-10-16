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
//! # Admission (epic 16774, E4)
//!
//! [`load_admitted`] is the **only** admitted load entry point: it refuses, before the loader runs,
//! a request whose footprint estimate is absent, exceeds [`IOS_APP_MEMORY_CAP_MIB`], or exceeds the
//! per-process headroom iOS reports at that moment. `media::load_sana` (with `media`) is SANA's
//! admitted load, built on it. Nothing else here is admitted:
//!
//! * [`llm`] re-exports `mlx-llm` whole, and its loaders (`load_for_model`, …) do **not** admit.
//!   MLX LLMs carry no footprint estimate yet, so the admitted answer for an LLM load today is a
//!   refusal (`admit_estimate_mib(None)`); a caller loading one anyway is running unadmitted and
//!   must say so.
//! * [`catalog`]'s registries are a composition/snapshot surface (the shared
//!   [`RuntimeCatalog`] contract) and load unadmitted. A product loads through
//!   [`load_admitted`], not through them.
//! * `media` deliberately does not re-export the catalog's `provider_registry`, so the
//!   unadmitted registry load is not this bundle's default image surface. The one unadmitted image
//!   loader, `media::load_sana_unadmitted_for_diagnostics`, is named for what it is.

use std::sync::atomic::{AtomicBool, Ordering};

/// The iOS image lane. The catalog's specs, constants, and footprint table, plus the admitted SANA
/// loader — but not its `provider_registry`, whose `load` is unadmitted (see the crate docs).
#[cfg(feature = "media")]
pub mod media {
    pub use mlx_gen_ios_catalog::{
        sana_decode_plan, sana_decode_tile_edge, sana_footprint_bound, sana_generation_memory,
        sana_load_spec, sana_snapshot_profile, FootprintBound, SanaDeviceFootprint,
        SanaSnapshotProfile, IOS_IMAGE_EDGES, IOS_MEMORY_BUDGET_MIB, IOS_MLX_CACHE_LIMIT_CAP_MIB,
        IOS_OFFLOAD_POLICY, SANA_DEVICE_FOOTPRINTS,
    };

    /// Load SANA generator `sana_id` for a render of `image_edge`² decoded at `tile_edge`, through
    /// [`crate::admit_sana_snapshot`] and [`crate::load_admitted`]: a refused render never touches its
    /// weights.
    pub fn load_sana(
        sana_id: &str,
        spec: &crate::gen_core::LoadSpec,
        image_edge: u32,
        tile_edge: u32,
    ) -> crate::gen_core::Result<Box<dyn crate::gen_core::Generator>> {
        let estimate = crate::admit_sana_snapshot(sana_id, spec, image_edge, tile_edge)?;
        crate::load_admitted(Some(estimate), || {
            mlx_gen_ios_catalog::provider_registry()?.load(sana_id, spec)
        })
    }

    /// Load SANA **without** admission. For a diagnostic run whose purpose is to execute a render
    /// admission would refuse (the harness's untiled experiment); never a product path.
    pub fn load_sana_unadmitted_for_diagnostics(
        sana_id: &str,
        spec: &crate::gen_core::LoadSpec,
    ) -> crate::gen_core::Result<Box<dyn crate::gen_core::Generator>> {
        mlx_gen_ios_catalog::provider_registry()?.load(sana_id, spec)
    }
}
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
/// precondition for every [`mlx_gen_ios_catalog::FootprintBound::Inferred`] bound.
static PLATFORM_LIMITS_BOUND: AtomicBool = AtomicBool::new(false);

/// Legacy geometry-only entry point, now fail-closed: dimensions alone do not identify a
/// snapshot, component precision, hardware, or applicable footprint evidence.
/// Use [`admit_sana_snapshot`] or the load-bound [`media::load_sana`] entry point.
///
/// Checks the cap and the live per-process headroom, like [`load_admitted`]. An **inferred** bound
/// (`live_peak + IOS_MLX_CACHE_LIMIT_CAP_MIB`) is valid only once [`bound_mlx_to_platform_limits`]
/// has capped the cache, so it is refused until then — off-device and on the simulator that is
/// always, because there is no per-process cap to bind to. Both kinds require artifact/device binding.
#[cfg(feature = "media")]
pub fn admit_sana_render(_image_edge: u32, _tile_edge: u32) -> gen_core::Result<u32> {
    Err(gen_core::Error::Unsupported(
        "iOS SANA admission requires the actual snapshot and hardware; use admit_sana_snapshot"
            .into(),
    ))
}

/// Admit using the actual component tensor layout, stored precision and hardware identity.
/// Historical geometry-only observations cannot authorize an arbitrary q4-named directory.
#[cfg(feature = "media")]
pub fn admit_sana_snapshot(
    sana_id: &str,
    spec: &gen_core::LoadSpec,
    image_edge: u32,
    tile_edge: u32,
) -> gen_core::Result<u32> {
    let bound = mlx_gen_ios_catalog::sana_snapshot_footprint_bound(
        sana_id,
        spec,
        &device_machine(),
        image_edge,
        tile_edge,
    )?;
    let estimate = sana_bound_estimate(bound, PLATFORM_LIMITS_BOUND.load(Ordering::Acquire))?;
    admit_within_headroom(estimate, os_proc_available_mib())
}

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

/// [`admit_sana_snapshot`]'s cap decision with the binding state passed in, so both states are
/// testable without mutating the process-global flag.
#[cfg(feature = "media")]
fn sana_bound_estimate(
    bound: mlx_gen_ios_catalog::FootprintBound,
    limits_bound: bool,
) -> gen_core::Result<u32> {
    use mlx_gen_ios_catalog::FootprintBound;
    match bound {
        FootprintBound::Inferred(mib) if !limits_bound => {
            Err(gen_core::Error::Unsupported(format!(
                "iOS admission: render has only an inferred bound ({mib} MiB), \
                 which holds only after bound_mlx_to_platform_limits capped MLX's cache; it has not \
                 in this process"
            )))
        }
        bound => admit_estimate_mib(Some(bound.mib())),
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

    #[cfg(feature = "media")]
    #[test]
    fn source_aware_admission_precedes_even_a_deferred_sequential_load() {
        let root = tempfile::tempdir().unwrap();
        let spec = mlx_gen_ios_catalog::sana_load_spec(super::gen_core::WeightsSource::Dir(
            root.path().to_path_buf(),
        ));
        assert!(super::admit_sana_snapshot("sana_1600m", &spec, 1024, 192).is_err());
        assert!(super::media::load_sana("sana_1600m", &spec, 1024, 192).is_err());
        assert!(super::admit_sana_render(1024, 192).is_err());
    }

    /// SANA admission reads the catalog's bound and compares it to the cap (whatever value the
    /// cap is filled in with); a pair with no bound (1024²/256) is refused.
    #[cfg(feature = "media")]
    #[test]
    fn sana_admission_uses_the_catalog_footprint_bound() {
        for (image, tile) in [(1024, 192), (512, 192), (1024, 128), (512, 256)] {
            let bound = mlx_gen_ios_catalog::sana_footprint_bound(image, tile)
                .expect("bounded pair")
                .mib();
            assert_eq!(
                super::sana_bound_estimate(
                    mlx_gen_ios_catalog::sana_footprint_bound(image, tile).unwrap(),
                    true
                )
                .ok(),
                (bound <= super::IOS_APP_MEMORY_CAP_MIB).then_some(bound),
                "{image}²/{tile}: bound {bound} MiB vs cap {}",
                super::IOS_APP_MEMORY_CAP_MIB
            );
        }
        assert!(super::admit_sana_render(1024, 256).is_err());
    }

    /// An inferred bound assumes the capped cache, so it is refused until the limits are bound; a
    /// measured bound is admitted either way. Off-iOS nothing ever binds, so the public entry point
    /// is the unbound case.
    #[cfg(feature = "media")]
    #[test]
    fn inferred_sana_bounds_are_refused_until_the_cache_is_capped() {
        use mlx_gen_ios_catalog::{sana_footprint_bound, FootprintBound};
        let inferred = (512, 192);
        let measured = (1024, 192);
        assert!(matches!(
            sana_footprint_bound(inferred.0, inferred.1),
            Some(FootprintBound::Inferred(_))
        ));
        assert!(matches!(
            sana_footprint_bound(measured.0, measured.1),
            Some(FootprintBound::Measured(_))
        ));

        assert!(super::sana_bound_estimate(
            sana_footprint_bound(inferred.0, inferred.1).unwrap(),
            false
        )
        .is_err());
        assert!(super::sana_bound_estimate(
            sana_footprint_bound(inferred.0, inferred.1).unwrap(),
            true
        )
        .is_ok());
        assert!(super::sana_bound_estimate(
            sana_footprint_bound(measured.0, measured.1).unwrap(),
            false
        )
        .is_ok());

        #[cfg(not(target_os = "ios"))]
        {
            assert!(super::admit_sana_render(inferred.0, inferred.1).is_err());
            assert!(
                super::admit_sana_render(measured.0, measured.1).is_err(),
                "geometry alone is not artifact evidence"
            );
        }
    }

    /// The bundle's exact, ordered surface. This is the reviewable source of truth for what ships
    /// on iOS: adding a provider means editing this test deliberately.
    #[test]
    fn catalog_surface_is_explicit_and_machine_readable() {
        let snapshot = super::catalog().unwrap().snapshot();

        assert_eq!(snapshot.platform, "ios");
        assert_eq!(snapshot.backend, "mlx");

        // The media surface is exactly SANA under `media`, empty without it. An incidental
        // `mlx-gen-catalog` dependency — every provider including video — fails here rather than
        // quietly compiling into an iPhone app.
        #[cfg(feature = "media")]
        assert_eq!(
            snapshot.generator_ids,
            ["sana_1600m", "sana_sprint_1600m"],
            "the iOS image surface changed; confirm the new provider fits the measured iOS cap \
             before updating this"
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

        // Never, on either profile: these belong to the full media graph, not to a phone.
        assert!(snapshot.transform_ids.is_empty());
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
