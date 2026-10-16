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

#[cfg(feature = "media")]
pub use mlx_gen_ios_catalog as media;
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

    /// The bundle's exact, ordered surface. This is the reviewable source of truth for what ships
    /// on iOS: adding a provider means editing this test deliberately.
    #[test]
    fn catalog_surface_is_explicit_and_machine_readable() {
        let snapshot = super::catalog().unwrap().snapshot();

        assert_eq!(snapshot.platform, "ios");
        assert_eq!(snapshot.backend, "mlx");

        // The media surface is exactly SANA + Anima Turbo under `media`, empty without it. An
        // incidental `mlx-gen-catalog` dependency — every provider including video — fails here
        // rather than quietly compiling into an iPhone app.
        #[cfg(feature = "media")]
        assert_eq!(
            snapshot.generator_ids,
            ["sana_1600m", "sana_sprint_1600m", "anima_turbo"],
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
