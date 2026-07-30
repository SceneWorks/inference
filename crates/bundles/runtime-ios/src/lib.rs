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

/// Build the complete validated iOS runtime composition.
///
/// No media registry and no audio lane: an empty [`gen_core::ProviderRegistry`] is passed for
/// media, mirroring how the other bundles compose under `--no-default-features`.
pub fn catalog() -> runtime_catalog::Result<RuntimeCatalog> {
    RuntimeCatalog::try_new(
        PLATFORM,
        BACKEND,
        gen_core::ProviderRegistryBuilder::new().build(),
        mlx_llm::text_registry(),
        mlx_llm::snapshot_preparer_registry(),
    )
}

#[cfg(test)]
mod tests {
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

        // Deliberately empty (Cargo.toml explains why). These assertions are the guard: the iOS
        // media story is a narrow, purpose-built registry in E5, so an incidental
        // `mlx-gen-catalog` dependency — 32 providers including video — must fail here rather
        // than quietly compile into an iPhone app.
        assert!(snapshot.generator_ids.is_empty());
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
