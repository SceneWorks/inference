//! Supported iOS runtime: the explicit MLX LLM and snapshot-preparer catalogs.
//!
//! This is the iPhone/iPad product boundary. It composes the same `mlx-llm` engine
//! [`runtime_macos`](../runtime-macos/README.md) does — MLX runs on iOS, so the LLM lane needs no
//! iOS-specific backend — but ships **no media and no audio registry**. See `Cargo.toml` for why
//! that is a composition decision rather than a missing feature.
//!
//! # Packaging
//!
//! An iOS app must carry MLX's Metal kernel library. Inside the app sandbox only two links of
//! MLX's resolver chain are reachable: `$PMETAL_METALLIB_PATH`, or `mlx.metallib` sitting next to
//! the executable. The host-side links are unavailable — `~/.cache/pmetal/lib` is not readable in
//! the sandbox, and the compiled-in `METAL_PATH` points into the cargo target directory, which is
//! not shipped. Use `scripts/ios/bundle_metallib.py` as an Xcode "Run Script" phase; it reads the
//! path `pmetal-mlx-sys` publishes as `DEP_MLX_METALLIB` and refuses to copy a metallib built for
//! the wrong platform.

#[cfg(feature = "media")]
pub use mlx_gen_ios_catalog as media;
pub use mlx_llm as llm;
pub use runtime_catalog::{core_llm, gen_core, RuntimeCatalog, RuntimeCatalogSnapshot};

/// Platform label for this bundle; matches [`RuntimeCatalog::platform`].
pub const PLATFORM: &str = "ios";
/// The single tensor backend every LLM and snapshot-preparer provider in this bundle uses.
pub const BACKEND: &str = "mlx";
/// Target triples this bundle is supported on.
///
/// The simulator triple is included because it is a supported **build** target (the nightly CI
/// tier exercises it), not because the simulator is a supported runtime: it has no Apple Neural
/// Engine and its Metal implementation differs from a device's, so performance work and any
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
///
/// Deliberately `mlx-gen-ios-catalog` rather than `mlx-gen-catalog` — see this crate's
/// `Cargo.toml` and the catalog's own docs for why a phone does not compile the 32-provider graph.
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

/// What [`bound_mlx_to_platform_limits`] changed, for a caller that wants to log or assert it.
///
/// All figures MiB. `previous_*` are what MLX had chosen for itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlxLimits {
    /// `os_proc_available_memory()` at the time of the call — the real budget. `None` off-iOS.
    pub os_available_mib: Option<f64>,
    pub previous_memory_limit_mib: f64,
    pub previous_cache_limit_mib: f64,
    pub memory_limit_mib: f64,
    pub cache_limit_mib: f64,
}

impl MlxLimits {
    /// Whether the call actually lowered anything. False off-iOS, and false if MLX's own sizing was
    /// already tighter than the platform budget.
    pub fn changed(&self) -> bool {
        self.memory_limit_mib < self.previous_memory_limit_mib
            || self.cache_limit_mib < self.previous_cache_limit_mib
    }
}

/// Bind MLX's memory and buffer-cache limits to the **per-process** budget iOS enforces.
///
/// Call once at app startup, before loading any model. Returns what changed.
///
/// # Why this is necessary, and why it is not automatic
///
/// MLX sizes its memory limit and its buffer-cache limit from the system's *recommended working
/// set*. On a Mac that is the right denominator. An iOS app is bounded instead by a per-process
/// jetsam limit far below device RAM, and nothing tells MLX so. Measured on an iPhone 17 Pro Max
/// (12 GB):
///
/// ```text
/// memory_limit=11109 MiB, cache_limit=11109 MiB, os_proc_available_memory=6014 MiB
/// ```
///
/// MLX applies backpressure at ~11 GB — a threshold the process can never reach, because jetsam
/// kills it at ~6.1 GB first. The cache limit is therefore effectively infinite and MLX returns
/// nothing to the OS. A Z-Image decode traced on device held `active + cache` at **6068 MiB**
/// across every sample, the cache absorbing exactly what active released, and the app was killed
/// with 4 MiB of headroom while sitting on ~3.9 GB of *reclaimable* memory. The decode itself was
/// bounded and correct throughout — 24 of 25 tiles at a 2901 MiB plateau, against 3102 MiB for the
/// same render on a Mac. Binding the limits let the identical request finish: 25/25 tiles, 2146 MiB
/// still free at the tightest point.
///
/// This is a property of MLX on iOS, not of any one model. It also means **`get_peak_memory` alone
/// cannot predict an iOS kill**: jetsam counts `phys_footprint`, which includes the cache that
/// `get_active_memory`/`get_peak_memory` both exclude.
///
/// **Not called from [`catalog`].** Building a catalog is a pure composition step, and this mutates
/// process-global allocator state — the kind of hidden side effect this architecture keeps out of
/// composition roots. An app opts in explicitly, once, where it can also log the result.
///
/// # The two limits, and why both
///
/// * **Cache** — `min(25% of budget, 1 GiB)`. Large enough that tile-to-tile buffer reuse still
///   hits (a tiled decode recycles same-shaped buffers), small enough to leave room beside a
///   multi-GB transient under a ~6 GB cap.
/// * **Live allocation** — 85% of the budget, so MLX applies backpressure *before* jetsam applies a
///   kill. Backpressure is recoverable; a kill is not.
///
/// A limit rather than periodic `clear_cache` because clearing is reactive and unsynchronized: it
/// frees whatever happens to be cached when it runs, not necessarily before the allocation that
/// overruns. A limit is enforced by the allocator at every allocation.
pub fn bound_mlx_to_platform_limits() -> MlxLimits {
    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);

    #[cfg(target_os = "ios")]
    let os_available_mib = {
        extern "C" {
            fn os_proc_available_memory() -> usize;
        }
        // SAFETY: no arguments, no pointers; returns 0 when unavailable.
        let bytes = unsafe { os_proc_available_memory() };
        (bytes > 0).then(|| mib(bytes))
    };
    // Not an iOS process: there is no per-process cap, so there is no smaller truth to tell MLX and
    // its own sizing is already correct. Deliberately not mirrored for symmetry — capping the cache
    // on a Mac would slow it to resemble a constraint it does not have.
    #[cfg(not(target_os = "ios"))]
    let os_available_mib: Option<f64> = None;

    // Read the current limits by setting and restoring: MLX exposes setters that return the prior
    // value, and no getter for the cache limit.
    let previous_memory_limit = mlx_rs::memory::get_memory_limit();
    let previous_cache_limit = {
        let prev = mlx_rs::memory::set_cache_limit(0);
        mlx_rs::memory::set_cache_limit(prev);
        prev
    };

    let Some(available) = os_available_mib else {
        return MlxLimits {
            os_available_mib: None,
            previous_memory_limit_mib: mib(previous_memory_limit),
            previous_cache_limit_mib: mib(previous_cache_limit),
            memory_limit_mib: mib(previous_memory_limit),
            cache_limit_mib: mib(previous_cache_limit),
        };
    };

    let cache_mib = (available / 4.0).min(1024.0);
    let memory_mib = available * 0.85;
    let to_bytes = |m: f64| (m * 1024.0 * 1024.0) as usize;
    mlx_rs::memory::set_memory_limit(to_bytes(memory_mib));
    mlx_rs::memory::set_cache_limit(to_bytes(cache_mib));

    MlxLimits {
        os_available_mib: Some(available),
        previous_memory_limit_mib: mib(previous_memory_limit),
        previous_cache_limit_mib: mib(previous_cache_limit),
        memory_limit_mib: memory_mib,
        cache_limit_mib: cache_mib,
    }
}

/// Build the complete validated iOS runtime composition.
///
/// No audio lane on either profile: that lane is the one sanctioned cross-backend seam
/// (sc-12835) and is not extended to a new platform incidentally.
pub fn catalog() -> runtime_catalog::Result<RuntimeCatalog> {
    RuntimeCatalog::try_new(
        PLATFORM,
        BACKEND,
        media_registry(),
        mlx_llm::text_registry(),
        mlx_llm::snapshot_preparer_registry(),
    )
}

#[cfg(test)]
mod tests {
    /// Off-iOS the call must be a **no-op that says so**, not a silent one.
    ///
    /// This runs on the macOS lane, which is the only place it can run — the behaviour it guards is
    /// on the other branch. What it pins is that a host has no per-process budget to bind to, so the
    /// limits are reported unchanged and `changed()` is false. A regression that made this lower
    /// MLX's cache on a Mac would slow every host render to imitate a constraint macOS does not
    /// impose, and would do it invisibly.
    #[test]
    fn binding_limits_is_a_no_op_without_a_per_process_cap() {
        let limits = super::bound_mlx_to_platform_limits();

        #[cfg(not(target_os = "ios"))]
        {
            assert_eq!(limits.os_available_mib, None, "a Mac has no per-app cap to report");
            assert_eq!(limits.memory_limit_mib, limits.previous_memory_limit_mib);
            assert_eq!(limits.cache_limit_mib, limits.previous_cache_limit_mib);
            assert!(!limits.changed(), "must not touch MLX's sizing where it is already correct");
        }
        #[cfg(target_os = "ios")]
        {
            let available = limits.os_available_mib.expect("iOS reports a per-process budget");
            assert!(limits.cache_limit_mib <= 1024.0, "cache must be capped at 1 GiB");
            assert!(
                limits.memory_limit_mib < available,
                "the live-allocation limit must sit BELOW the budget so backpressure precedes a \
                 jetsam kill, got {} against {available}",
                limits.memory_limit_mib
            );
        }
    }

    /// The bundle's exact, ordered surface. This is the reviewable source of truth for what ships
    /// on iOS: adding a provider means editing this test deliberately, never discovering the
    /// change after the fact.
    #[test]
    fn catalog_surface_is_explicit_and_machine_readable() {
        let snapshot = super::catalog().unwrap().snapshot();

        assert_eq!(snapshot.platform, "ios");
        assert_eq!(snapshot.backend, "mlx");

        // Ordered, and identical to runtime-macos's LLM surface — the same `mlx-llm` catalog on
        // the same backend. A divergence here means the two platforms silently drifted.
        assert_eq!(snapshot.text_llm_ids, ["mlx-llama", "mlx-joycaption"]);
        assert_eq!(snapshot.snapshot_preparer_backends, ["mlx"]);

        // The media surface is feature-gated and NARROW by design. Under `media` it is exactly
        // the iOS image catalog; without it, empty. Either way an incidental `mlx-gen-catalog`
        // dependency — 32 providers including video — fails here rather than quietly compiling
        // into an iPhone app.
        #[cfg(feature = "media")]
        assert_eq!(
            snapshot.generator_ids,
            ["sana_1600m", "sana_sprint_1600m"],
            "the iOS image surface changed; confirm the new provider fits an 8 GB device's cap \
             (mlx-llm's memory_budget example) before updating this"
        );
        #[cfg(not(feature = "media"))]
        assert!(snapshot.generator_ids.is_empty());

        // Never, on either profile: these belong to the full media graph, not to a phone.
        assert!(snapshot.transform_ids.is_empty());
        assert!(snapshot.trainer_ids.is_empty());
        assert!(snapshot.captioner_ids.is_empty());
        assert!(snapshot.image_embedder_ids.is_empty());
        assert!(snapshot.text_embedder_ids.is_empty());

        // No audio lane is declared: the cross-backend seam (sc-12835) is a deliberate exception
        // for the platforms that ship it, and is not extended to iOS by default.
        assert!(snapshot.audio_backend.is_none());
        assert!(snapshot.audio_generator_ids.is_empty());
    }

    /// The threading contract is enforced by the **type system**, not by documentation.
    ///
    /// A loaded provider holds MLX `Array`s and MLX's default Metal device is not thread-safe, so
    /// a provider must be driven from one thread (or behind a mutex). On macOS that reads as a
    /// test-harness detail — `.cargo/config.toml` forces `RUST_TEST_THREADS=1`, so the problem
    /// stays hidden. **On iOS it is host-app correctness**: a Swift caller that hops a provider
    /// onto a `DispatchQueue` or into a `Task` would race MLX's device and crash intermittently,
    /// and intermittent is the expensive kind.
    ///
    /// Asserted here rather than assumed: `Box<dyn TextLlm>` is **not `Send`**, so that mistake
    /// does not compile. The check is a runtime read of a compile-time fact — `impls_send` is
    /// resolved by autoref specialization, so it reports what the type system actually decided
    /// rather than what this comment claims. If a future change made the trait object `Send`, the
    /// unsafe pattern would silently become legal and this test fails first.
    #[test]
    // The explicit `&` on each call is the whole mechanism: it is what lets method resolution
    // fall through to the trait impl when `T: !Send`. Removing the borrows, as
    // clippy::needless_borrow suggests, would make both cases resolve identically and the check
    // vacuous -- it would pass no matter what.
    #[allow(clippy::needless_borrow)]
    fn provider_is_not_send_so_cross_thread_use_cannot_compile() {
        // Autoref specialization: the inherent method on `Wrap<T>` wins only when `T: Send`,
        // otherwise the blanket trait method on `&Wrap<T>` is selected. That makes "is this type
        // Send?" observable at runtime without a `trybuild` fixture.
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

    /// The declared triples are the ones the toolchain and CI actually build.
    #[test]
    fn supported_triples_are_ios_only() {
        assert_eq!(
            super::SUPPORTED_TARGET_TRIPLES,
            ["aarch64-apple-ios", "aarch64-apple-ios-sim"]
        );
        for triple in super::SUPPORTED_TARGET_TRIPLES {
            assert!(
                triple.contains("apple-ios"),
                "{triple} is not an iOS target"
            );
        }
    }
}
