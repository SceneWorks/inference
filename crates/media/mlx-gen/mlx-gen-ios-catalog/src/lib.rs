//! The iOS media composition root: a **narrow** image-generation catalog for a phone.
//!
//! This exists instead of reusing `mlx-gen-catalog`, and the difference is the whole point. That
//! catalog composes every MLX provider crate — including the video families (`wan`, `ltx`,
//! `mochi`, `svd`) — none of which is validated on iOS or shaped for a device with a hard per-app
//! memory cap. Depending on it to obtain one small generator would compile a graph the platform
//! cannot run.
//!
//! So this catalog is a deliberate, reviewable subset of candidate generators. Registration is
//! not a claim of device fit: artifact-bound product admission is separate. Adding
//! one is an edit here *and* to this crate's ordered surface test, exactly as the platform
//! catalogs work — a provider crate existing in the workspace does not mean it ships.
//!
//! # Admission (epic 16774, E4/E5/E7)
//!
//! Every registered model and mode is admitted through ONE seam, [`ios_admission`]: the provider's
//! registered safety check fed by a row of the single reviewed device-evidence table
//! ([`evidence`], `evidence/ios_device_evidence.json`), fail-closed with a typed [`RefusalKind`].
//! The table is filled only from explicitly labelled device captures (`capture_permit`, feature `device-capture`), which are
//! safe without evidence because every quantum of every registered model checks live OS headroom
//! before constructing its graph (`mlx_gen::ios_quantum`). Host peaks and the historical iPhone 17
//! SANA observations below are PLANNING data (tile choice, what to capture first), never admission.
//!
//! # What ships
//!
//! **SANA** (`mlx-gen-sana`), base and sprint — text conditioning reused from `mlx-gen-pid`'s
//! Gemma-2-2b-it caption encoder (epic 8485 / sc-8488) rather than duplicated. Its memory ladder
//! is the provider's own contract, admitted through `standard_memory_strategy_safety_check`
//! exactly as on macOS; this crate only chooses the iOS **defaults** inside that published
//! domain: the [`IOS_OFFLOAD_POLICY`] load and the decode tile edge [`sana_decode_tile_edge`]
//! selects from the device measurements (and bounds inferred from them).
//!
//! **Anima Turbo** (`mlx-gen-anima`, `anima_turbo` only — not Base or Aesthetic), compiled without
//! the crate's `training` feature, so no trainer and no `mlx-gen-perceptual` reaches the bundle. Its
//! ladder is the provider's contract too; [`anima`] PLANS the iOS selection — staged residency with
//! a tiled decode — from host peaks; admission is the shared evidence seam. Its Qwen-Image VAE brings `mlx-gen-qwen-image`, and with it `mlx-gen-wan` (Qwen's
//! optional single-frame decoder), as compiled code only: no Qwen-Image or Wan generator is
//! registered here.
//!
//! **SDXL family** (`sdxl`, Illustrious v1/v2, RealVisXL and Lightning checkpoints): one complete
//! generator, no trainer even under feature unification. Evidence is keyed per route; Mac engine
//! measurements are never rows.
//!
//! **Iris 3B** (`iris_3b` generation incl. prompt batches, the `iris_3b_restore` transform and the
//! `iris_3b_depth` task), inference only.

mod admission;
pub use admission::{sana_snapshot_profile, SanaSnapshotProfile};
mod bound;
pub mod evidence;
pub mod ios_admission;
mod layout;
mod refusal;
pub use bound::{BoundDepth, BoundGenerator, BoundTransform};
/// The labelled bounded device-capture permit; compiled only for the device harness.
#[cfg(feature = "device-capture")]
pub use ios_admission::capture_permit;
pub use ios_admission::{
    admit, build_candidate, capacity_budget, evidence_key, execution_key, tier_label, token_window,
    validate_actual, work_binding, work_mode, Admission, IosCandidate, IosWork, IRIS_DEPTH_ID,
    IRIS_RESTORE_ID,
};
pub use layout::{artifact_layout_digest, hex};
/// The bounded-quantum mechanism (live headroom before every graph): a capture reads
/// `take_refusal()` from it, and host fixtures use its CPU diagnostic headroom.
pub use mlx_gen::ios_quantum;
pub use refusal::{AdmissionError, AdmissionResult, Refusal, RefusalKind};

use mlx_gen::gen_core::{
    Error, GenerationMemory, LoadSpec, OffloadPolicy, ProviderRegistry, ProviderRegistryBuilder,
    Result, WeightsSource,
};
use mlx_gen_sana::memory_strategy::DECODE_TILE_EDGES;
use mlx_gen_sana::pipeline::{resolved_decode_plan, DecodeTilingPlan, DECODE_OVERLAP};

// Iris generation, depth and restoration are inference-only registrations, not device-fit claims.

pub mod anima;
pub mod iris_depth;
pub mod sdxl;
pub use anima::{
    anima_turbo_ios_selection, anima_turbo_load_spec, anima_turbo_memory_selection,
    anima_turbo_planning_decision, ios_memory_budget, ANIMA_TURBO_ID,
};

/// Add the iOS image generators to an explicit media registry builder, in catalog order.
///
/// Order is part of the surface: [`provider_registry`]'s ids are asserted in this order by the
/// crate's surface test, so a reordering is a deliberate edit rather than an accident.
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    // Turbo alone, through the provider's own per-variant seam: its generator and complete memory
    // registration, never a trainer — whatever features another crate unifies onto `mlx-gen-anima`.
    mlx_gen_iris::register_inference_providers(mlx_gen_sdxl::register_generator(
        mlx_gen_anima::register_turbo_generator(mlx_gen_sana::register_providers(registry)),
    ))
}

/// Build the complete, explicit iOS media provider catalog.
pub fn provider_registry() -> Result<ProviderRegistry> {
    register_providers(ProviderRegistryBuilder::new()).build()
}

/// The load policy an iOS SANA load uses: phases are shed between conditioning, denoise and
/// decode, and the provider's bounded DC-AE decode is on by default. An untiled DC-AE decode at
/// 1024² was measured at 9177 MiB on the host and killed the app on device.
pub const IOS_OFFLOAD_POLICY: OffloadPolicy = OffloadPolicy::Sequential;

/// The per-app memory cap of the 8 GB target device (iPhone 16 Pro Max), in MiB — the **single**
/// cap constant: the iOS default tile choice here and `runtime-ios`'s admission
/// (`IOS_APP_MEMORY_CAP_MIB`, which is this constant) both read it (epic 16774, E4).
///
/// **PROVISIONAL — UNMEASURED.** 4096 is the `claude/ios-strategy` branch's *assumed* 8 GB cap,
/// not a validated rendering ceiling. Entitled launch headroom on iPhone 16 Pro Max was
/// 6142 MiB (three launches), versus 3374 MiB without the entitlement (three launches). These
/// are diagnostics, NOT a safe peak working-set budget. Do not substitute a launch reading.
pub const IOS_MEMORY_BUDGET_MIB: u32 = 4096;

/// The square output edges (px) every iOS default must fit at.
pub const IOS_IMAGE_EDGES: &[u32] = &[512, 1024];

/// The ceiling `runtime-ios`'s `bound_mlx_to_platform_limits` puts on MLX's buffer (reuse) cache,
/// in MiB — `platform_limits_mib` caches `min(25% of budget, this)`. Defined here, the lower crate,
/// because [`sana_footprint_bound`] needs it; `runtime-ios` applies this constant rather than a copy.
pub const IOS_MLX_CACHE_LIMIT_CAP_MIB: u32 = 1024;

/// One on-device measurement of a SANA render.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SanaDeviceFootprint {
    /// Square output edge, px.
    pub image_edge: u32,
    /// DC-AE decode tile edge, px.
    pub tile_edge: u32,
    /// Historical sampled MLX active + reuse-cache peak, MiB. This is NOT a measurement of
    /// kernel `phys_footprint` and does not include every process/driver allocation.
    pub footprint_mib: u32,
    /// MLX's **live** peak (`get_peak_memory`, excludes the reuse cache), MiB.
    pub mlx_live_peak_mib: u32,
}

/// SANA (`sana_1600m`, q4, Sequential) renders measured on device (iPhone 17 Pro Max).
///
/// Unbound historical planning observations, NOT applicable product-admission evidence.
/// Source: the device table in `mlx-gen-sana/src/pipeline.rs`'s `DECODE_TILE_EDGE` notes on the
/// `claude/ios-strategy` branch (the 1024² rows), and `docs/ios-device-runbook.md` there (the 512²
/// row). **Label conflict:** the runbook labels its 2733 / 2860 MiB SANA 1024² run "tile128"
/// (lines 122 and 133), while the pipeline.rs table — which measured 192 and 128 side by side, 128
/// at 2839 / 2904 MiB — attributes 2733 / 2860 to tile 192. The pipeline.rs table is followed here.
///
/// | render | MLX live peak | footprint |
/// |---|---:|---:|
/// | 1024², tile 192 | 2733 MiB | 2860 MiB |
/// | 1024², tile 128 | 2839 MiB | 2904 MiB |
/// | 512², tile 256 | 3093 MiB | **4113 MiB** — over a 4 GiB budget |
pub const SANA_DEVICE_FOOTPRINTS: &[SanaDeviceFootprint] = &[
    SanaDeviceFootprint {
        image_edge: 1024,
        tile_edge: 192,
        footprint_mib: 2860,
        mlx_live_peak_mib: 2733,
    },
    SanaDeviceFootprint {
        image_edge: 1024,
        tile_edge: 128,
        footprint_mib: 2904,
        mlx_live_peak_mib: 2839,
    },
    SanaDeviceFootprint {
        image_edge: 512,
        tile_edge: 256,
        footprint_mib: 4113,
        mlx_live_peak_mib: 3093,
    },
];

/// A footprint bound for one SANA render, MiB, and where it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FootprintBound {
    /// A historical device observation of this geometry; artifact/device applicability is separate.
    Measured(u32),
    /// Not measured: derived by [`sana_footprint_bound`] from a measured live peak plus the iOS
    /// cache ceiling. Holds only in a process where `runtime-ios`'s `bound_mlx_to_platform_limits`
    /// has run, because only then is the cache capped at [`IOS_MLX_CACHE_LIMIT_CAP_MIB`].
    Inferred(u32),
}

impl FootprintBound {
    /// The bound in MiB, whatever its provenance.
    pub fn mib(self) -> u32 {
        match self {
            Self::Measured(mib) | Self::Inferred(mib) => mib,
        }
    }
}

/// Historical planning bound for `image_edge`² decoded at `tile_edge`. NOT product admission:
/// the source snapshot identity is missing and the capture device was an iPhone 17 Pro Max.
/// Admission is [`ios_admission::admit`] against the reviewed evidence table.
///
/// An exact device record is [`FootprintBound::Measured`]. Otherwise the bound is
/// [`FootprintBound::Inferred`] from the smallest record at the **same tile edge and a larger
/// image**, and only from its *live* peak: SANA's conditioning phase is geometry-independent (the
/// CHI prompt pads to a fixed 300 caption slots), its denoise phase grows with the token count
/// `(edge/32)²`, and its tiled decode transient is set by the tile's area, not the image's (host
/// sweeps in `mlx-gen-sana`'s `memory_strategy` and `DECODE_TILE_EDGE` notes) — so no phase of the
/// smaller render allocates more live memory than the larger one at the same tile.
///
/// That argument says nothing about the **reuse cache**, which `phys_footprint` also counts and
/// which does not shrink with image size (512²/256 carries ~1 GiB of cache over its live peak;
/// 1024²/192 ~130 MiB). So the larger render's *footprint* is not a bound; its live peak plus the
/// most cache the process may hold is: `live_peak + IOS_MLX_CACHE_LIMIT_CAP_MIB`, valid only after
/// `bound_mlx_to_platform_limits` capped the cache. Nothing bounds a render from a *different*
/// tile edge — a smaller image at a larger tile can cost more — so such a pair is `None`.
pub fn sana_footprint_bound(image_edge: u32, tile_edge: u32) -> Option<FootprintBound> {
    let same_tile = SANA_DEVICE_FOOTPRINTS
        .iter()
        .filter(|record| record.tile_edge == tile_edge);
    if let Some(exact) = same_tile
        .clone()
        .find(|record| record.image_edge == image_edge)
    {
        return Some(FootprintBound::Measured(exact.footprint_mib));
    }
    same_tile
        .filter(|record| record.image_edge > image_edge)
        .min_by_key(|record| record.image_edge)
        .map(|record| {
            FootprintBound::Inferred(record.mlx_live_peak_mib + IOS_MLX_CACHE_LIMIT_CAP_MIB)
        })
}

/// The iOS default DC-AE tile edge: the **largest** edge in SANA's published decode ladder whose
/// footprint bound ([`sana_footprint_bound`], measured or inferred) fits [`IOS_MEMORY_BUDGET_MIB`]
/// at every one of [`IOS_IMAGE_EDGES`]. An inferred bound assumes the app called `runtime-ios`'s
/// `bound_mlx_to_platform_limits` at startup.
///
/// Largest, because a larger tile is the better image (fewer seams) and the cheaper decode (fewer
/// dispatches); it is only rejected for memory. An edge without a bound at some iOS size is not
/// selected — an unmeasured fit is not a fit.
pub fn sana_decode_tile_edge() -> Result<u32> {
    DECODE_TILE_EDGES
        .iter()
        .copied()
        .find(|&tile| {
            IOS_IMAGE_EDGES.iter().all(|&image| {
                sana_footprint_bound(image, tile)
                    .is_some_and(|bound| bound.mib() <= IOS_MEMORY_BUDGET_MIB)
            })
        })
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "no published SANA decode tile edge {DECODE_TILE_EDGES:?} has an iOS \
                 footprint bound within {IOS_MEMORY_BUDGET_MIB} MiB at every size {IOS_IMAGE_EDGES:?}"
            ))
        })
}

/// The request-scoped memory selection an iOS SANA render carries: bounded decode at
/// [`sana_decode_tile_edge`] with the provider's published overlap. Set it on
/// `GenerationRequest::memory` so the decode geometry is pinned by the request rather than
/// inherited from whatever the provider's macOS default happens to be.
pub fn sana_generation_memory() -> Result<GenerationMemory> {
    Ok(GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: Some(sana_decode_tile_edge()?),
        decode_overlap: Some(DECODE_OVERLAP as u32),
        ..Default::default()
    })
}

/// The decode plan SANA's pipeline actually resolves for an iOS render (an
/// [`IOS_OFFLOAD_POLICY`] load carrying [`sana_generation_memory`]), read back from the provider's
/// own resolver rather than restated, so it cannot drift from what runs.
pub fn sana_decode_plan() -> Result<DecodeTilingPlan> {
    resolved_decode_plan(
        Some(sana_generation_memory()?),
        IOS_OFFLOAD_POLICY == OffloadPolicy::Sequential,
    )
    .ok_or_else(|| Error::Unsupported("SANA resolved a whole-image decode on iOS".to_owned()))
}

/// A SANA load spec with the iOS defaults applied.
pub fn sana_load_spec(source: WeightsSource) -> LoadSpec {
    LoadSpec::new(source).with_offload_policy(IOS_OFFLOAD_POLICY)
}

#[cfg(test)]
mod tests {
    use mlx_gen::gen_core::{
        MemoryNumericTier, MemorySelection, MemoryStrategy, MemoryStrategyParameters, WeightsSource,
    };
    use mlx_gen_sana::pipeline::DecodeTilingSource;

    /// The exact, ordered surface this catalog ships on iOS.
    ///
    /// This is the reviewable source of truth for what a phone compiles. The negative assertions
    /// matter as much as the positive one: an incidental `mlx-gen-catalog` dependency, or a video
    /// family added by reflex, fails here rather than being discovered as a jetsam kill on a
    /// device.
    #[test]
    fn catalog_surface_is_explicit_and_narrow() {
        let registry = super::provider_registry().expect("iOS media catalog builds");
        let ids: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(
            ids,
            [
                "sana_1600m",
                "sana_sprint_1600m",
                "anima_turbo",
                "sdxl",
                "iris_3b"
            ],
            "the iOS media surface changed -- if deliberate, update this test and confirm the new \
             provider is admission-gated; registration is not a device fitting claim"
        );
        assert_eq!(registry.trainers().len(), 0, "iOS ships no trainers");
        assert!(registry
            .transforms()
            .any(|r| (r.descriptor)().id == "iris_3b_restore"));
        let memory: Vec<_> = registry
            .memory_strategy_registrations()
            .map(|r| r.provider_id)
            .collect();
        for id in ["iris_3b", "iris_3b_depth", "iris_3b_restore"] {
            assert!(memory.contains(&id), "missing Iris mode {id}: {memory:?}");
        }
        assert_eq!(registry.captioners().len(), 0, "iOS ships no captioners");
    }

    #[test]
    fn sdxl_is_reachable_with_the_full_generator_surface_and_no_trainer() {
        let registry = super::provider_registry().unwrap();
        let descriptor = registry
            .generators()
            .map(|r| (r.descriptor)())
            .find(|d| d.id == "sdxl")
            .expect("SDXL must actually be registered, not just linked");
        let caps = descriptor.capabilities;
        assert!(!caps.mac_only, "SDXL is supported on iOS");
        assert!(caps.supports_lora && caps.supports_lokr);
        for kind in [
            mlx_gen::gen_core::ConditioningKind::Reference,
            mlx_gen::gen_core::ConditioningKind::Mask,
            mlx_gen::gen_core::ConditioningKind::Control,
        ] {
            assert!(
                caps.conditioning.contains(&kind),
                "mode {kind:?} was dropped"
            );
        }
        for sampler in [
            "euler_ancestral",
            "lightning",
            "lcm",
            "hyper",
            "euler",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "ddim",
        ] {
            assert!(
                caps.samplers.contains(&sampler),
                "sampler {sampler} was dropped"
            );
        }
        use mlx_gen::gen_core::{Conditioning, ControlKind, GenerationRequest, Image, Quant};
        let image = || Image {
            width: 1,
            height: 1,
            pixels: vec![128; 3],
        };
        for conditioning in [
            vec![],
            vec![Conditioning::Reference {
                image: image(),
                strength: Some(0.5),
            }],
            vec![
                Conditioning::Reference {
                    image: image(),
                    strength: Some(0.5),
                },
                Conditioning::Mask { image: image() },
            ],
            vec![Conditioning::Control {
                image: image(),
                kind: ControlKind::Other("tile".into()),
                scale: Some(1.0),
            }],
        ] {
            let request = GenerationRequest {
                width: 512,
                height: 512,
                count: 1,
                conditioning,
                ..Default::default()
            };
            caps.validate_request("sdxl", &request)
                .expect("registered route must remain selectable");
        }
        assert_eq!(caps.supported_quants, &[Quant::Q4, Quant::Q8]);
        for route in super::sdxl::SDXL_FAMILY_ROUTES {
            let spec =
                super::sdxl::sdxl_load_spec(route, WeightsSource::Dir("/nonexistent".into()))
                    .unwrap();
            assert_eq!(spec.resolved_route.as_deref(), Some(*route));
        }
        assert!(registry
            .memory_strategy_registrations()
            .any(|r| r.provider_id == "sdxl"));
        assert_eq!(
            registry.trainers().len(),
            0,
            "feature unification must not register a trainer"
        );
    }

    /// The shared weights-free behavior oracle over this catalog's own registrations — Anima
    /// Turbo's contract, fixture, safety check and request scope arrive through
    /// `register_turbo_generator` exactly as `mlx-gen-catalog` registers them, on the iOS load.
    #[test]
    fn memory_registrations_pass_the_weights_free_behavior_oracle() {
        let registry = super::provider_registry().expect("iOS media catalog builds");
        let spec = super::anima_turbo_load_spec(WeightsSource::Dir("/nonexistent".into()));
        gen_core_testkit::memory_strategy::memory_strategy_registry_conformance(&registry, &spec);
        let adopted: Vec<_> = registry
            .memory_strategy_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        assert!(adopted.contains(&super::ANIMA_TURBO_ID), "{adopted:?}");
    }

    /// Every descriptor is on the `mlx` backend, which `runtime-catalog` enforces again when this
    /// registry is composed into the bundle. Asserted here too so the failure names this catalog.
    #[test]
    fn every_generator_is_mlx() {
        let registry = super::provider_registry().expect("iOS media catalog builds");
        for registration in registry.generators() {
            let descriptor = (registration.descriptor)();
            assert_eq!(
                descriptor.backend, "mlx",
                "generator {} is on backend {}",
                descriptor.id, descriptor.backend
            );
        }
    }

    /// The selected iOS plan: tile 192. At 1024² its bound is measured (2860 MiB); 512² at tile
    /// 192 was never measured, so its bound is INFERRED — 1024²/192's live peak (2733) plus the
    /// 1 GiB iOS cache ceiling = 3757 MiB — and both fit 4 GiB. The larger 256 is excluded by its
    /// measured 4113 MiB at 512², and 512/384 have no bound at all.
    #[test]
    fn ios_sana_plan_selects_the_largest_tile_that_fits_4_gib_at_512_and_1024() {
        use super::FootprintBound::{Inferred, Measured};

        assert_eq!(super::IOS_MEMORY_BUDGET_MIB, 4096);
        assert_eq!(super::IOS_IMAGE_EDGES, [512, 1024]);

        let plan = super::sana_decode_plan().expect("an iOS SANA plan is selected");
        assert_eq!(plan.edge, 192, "iOS SANA must default to tile 192");
        assert_eq!(plan.overlap, mlx_gen_sana::pipeline::DECODE_OVERLAP);
        assert_eq!(
            plan.source,
            DecodeTilingSource::Request,
            "the iOS request must carry the geometry, not inherit a load default"
        );

        for image in [512, 1024] {
            let bound = super::sana_footprint_bound(image, plan.edge as u32)
                .unwrap_or_else(|| panic!("no footprint bound for {image}² at tile 192"));
            assert!(
                bound.mib() <= 4096,
                "{image}² at tile {} is bounded at {bound:?}, over 4 GiB",
                plan.edge
            );
        }
        assert_eq!(super::sana_footprint_bound(1024, 192), Some(Measured(2860)));
        // Inferred, not measured: live peak of the larger render plus the cache ceiling. NOT the
        // larger render's footprint (2860), which under-counts the smaller render's cache.
        assert_eq!(super::IOS_MLX_CACHE_LIMIT_CAP_MIB, 1024);
        assert_eq!(super::sana_footprint_bound(512, 192), Some(Inferred(3757)));

        // Every published edge larger than the selection is excluded for a stated reason.
        assert_eq!(super::sana_footprint_bound(512, 256), Some(Measured(4113)));
        for larger in mlx_gen_sana::memory_strategy::DECODE_TILE_EDGES
            .iter()
            .filter(|&&edge| edge > plan.edge as u32)
        {
            assert!(
                [512, 1024]
                    .iter()
                    .any(|&image| super::sana_footprint_bound(image, *larger)
                        .is_none_or(|bound| bound.mib() > 4096)),
                "tile {larger} fits 4 GiB at every iOS size and should have been selected"
            );
        }

        // A larger image never bounds a smaller one across tile edges.
        assert_eq!(super::sana_footprint_bound(512, 128), Some(Inferred(3863)));
        assert_eq!(super::sana_footprint_bound(1024, 256), None);
    }

    /// The selected default is inside SANA's published bounded-decode domain, so the provider's
    /// contract (the one `standard_memory_strategy_safety_check` admits against) accepts it on an
    /// iOS load for both shipped ids.
    #[test]
    fn ios_sana_selection_is_inside_the_published_contract() {
        let registry = super::provider_registry().expect("iOS media catalog builds");
        let spec = super::sana_load_spec(WeightsSource::Dir(
            "/nonexistent/ios-sana-contract-fixture".into(),
        ));
        assert_eq!(spec.offload_policy, super::IOS_OFFLOAD_POLICY);
        let memory = super::sana_generation_memory().expect("iOS SANA memory selection");
        for provider in ["sana_1600m", "sana_sprint_1600m"] {
            let contract = registry
                .memory_strategy_contract(provider, &spec)
                .expect("contract resolves weights-free")
                .expect("SANA exposes a memory contract");
            let selection = MemorySelection {
                strategy: MemoryStrategy::BoundedDecode,
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: memory.decode_tile_edge,
                    decode_overlap: memory.decode_overlap,
                    ..Default::default()
                },
                tier: MemoryNumericTier {
                    precision: spec.precision,
                    quant: spec.quantize,
                    component_precision_floors: &[],
                },
            };
            contract
                .validate_selection(&selection)
                .unwrap_or_else(|error| panic!("{provider}: iOS selection rejected: {error}"));
        }
    }
}
