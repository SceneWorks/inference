//! The iOS media composition root: a **narrow** image-generation catalog for a phone.
//!
//! This exists instead of reusing `mlx-gen-catalog`, and the difference is the whole point. That
//! catalog composes every MLX provider crate — including the video families (`wan`, `ltx`,
//! `mochi`, `svd`) — none of which is validated on iOS or shaped for a device with a hard per-app
//! memory cap. Depending on it to obtain one small generator would compile a graph the platform
//! cannot run.
//!
//! So this catalog is a deliberate, reviewable subset: the generators that actually fit. Adding
//! one is an edit here *and* to this crate's ordered surface test, exactly as the platform
//! catalogs work — a provider crate existing in the workspace does not mean it ships.
//!
//! # What ships
//!
//! **SANA** (`mlx-gen-sana`), base and sprint — text conditioning reused from `mlx-gen-pid`'s
//! Gemma-2-2b-it caption encoder (epic 8485 / sc-8488) rather than duplicated. Its memory ladder
//! is the provider's own contract, admitted through `standard_memory_strategy_safety_check`
//! exactly as on macOS; this crate only chooses the iOS **defaults** inside that published
//! domain: the [`IOS_OFFLOAD_POLICY`] load and the decode tile edge [`sana_decode_tile_edge`]
//! selects from the recorded device footprints.

use mlx_gen::gen_core::{
    Error, GenerationMemory, LoadSpec, OffloadPolicy, ProviderRegistry, ProviderRegistryBuilder,
    Result, WeightsSource,
};
use mlx_gen_sana::memory_strategy::DECODE_TILE_EDGES;
use mlx_gen_sana::pipeline::{resolved_decode_plan, DecodeTilingPlan, DECODE_OVERLAP};

/// Add the iOS image generators to an explicit media registry builder, in catalog order.
///
/// Order is part of the surface: [`provider_registry`]'s ids are asserted in this order by the
/// crate's surface test, so a reordering is a deliberate edit rather than an accident.
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    mlx_gen_sana::register_providers(registry)
}

/// Build the complete, explicit iOS media provider catalog.
pub fn provider_registry() -> Result<ProviderRegistry> {
    register_providers(ProviderRegistryBuilder::new()).build()
}

/// The load policy an iOS SANA load uses: phases are shed between conditioning, denoise and
/// decode, and the provider's bounded DC-AE decode is on by default. An untiled DC-AE decode at
/// 1024² was measured at 9177 MiB on the host and killed the app on device.
pub const IOS_OFFLOAD_POLICY: OffloadPolicy = OffloadPolicy::Sequential;

/// The per-app memory budget iOS defaults are chosen against, in MiB: an 8 GB device's assumed
/// cap. The measured cap of the target device replaces this assumption at admission time
/// (epic 16774, E4); the default tile choice only has to fit under it.
pub const IOS_MEMORY_BUDGET_MIB: u32 = 4096;

/// The square output edges (px) every iOS default must fit at.
pub const IOS_IMAGE_EDGES: &[u32] = &[512, 1024];

/// One on-device measurement of a SANA render: the process **footprint** (`phys_footprint`, what
/// jetsam charges — MLX active allocation *plus* its reuse cache), not MLX's live peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SanaDeviceFootprint {
    /// Square output edge, px.
    pub image_edge: u32,
    /// DC-AE decode tile edge, px.
    pub tile_edge: u32,
    /// Peak process footprint, MiB.
    pub footprint_mib: u32,
}

/// SANA (`sana_1600m`, q4, Sequential) footprints recorded on device (iPhone 17 Pro Max) by the
/// `claude/ios-strategy` device harness (`docs/ios-device-runbook.md` there, and
/// `mlx-gen-sana`'s `DECODE_TILE_EDGE` notes):
///
/// | render | footprint |
/// |---|---:|
/// | 1024², tile 192 | 2860 MiB |
/// | 1024², tile 128 | 2904 MiB |
/// | 512², tile 256 | **4113 MiB** — over a 4 GiB budget |
pub const SANA_DEVICE_FOOTPRINTS: &[SanaDeviceFootprint] = &[
    SanaDeviceFootprint {
        image_edge: 1024,
        tile_edge: 192,
        footprint_mib: 2860,
    },
    SanaDeviceFootprint {
        image_edge: 1024,
        tile_edge: 128,
        footprint_mib: 2904,
    },
    SanaDeviceFootprint {
        image_edge: 512,
        tile_edge: 256,
        footprint_mib: 4113,
    },
];

/// The recorded footprint bound for a render of `image_edge`² decoded at `tile_edge`, in MiB.
///
/// An exact record wins. Otherwise the smallest record at the **same tile edge and a larger
/// image** bounds it: SANA's conditioning phase is geometry-independent (the CHI prompt pads to a
/// fixed 300 caption slots), its denoise phase grows with the token count `(edge/32)²`, and its
/// tiled decode transient is set by the tile's area, not the image's (host sweeps in
/// `mlx-gen-sana`'s `memory_strategy` and `DECODE_TILE_EDGE` notes). So no phase of the smaller
/// render exceeds the same phase of the larger one at the same tile. Nothing bounds a render from
/// a *different* tile edge — a smaller image at a larger tile can cost more (512²/256 above is
/// over budget while 1024²/192 is not) — so such a pair has no bound and is `None`.
pub fn sana_footprint_bound_mib(image_edge: u32, tile_edge: u32) -> Option<u32> {
    let same_tile = SANA_DEVICE_FOOTPRINTS
        .iter()
        .filter(|record| record.tile_edge == tile_edge);
    if let Some(exact) = same_tile
        .clone()
        .find(|record| record.image_edge == image_edge)
    {
        return Some(exact.footprint_mib);
    }
    same_tile
        .filter(|record| record.image_edge > image_edge)
        .min_by_key(|record| record.image_edge)
        .map(|record| record.footprint_mib)
}

/// The iOS default DC-AE tile edge: the **largest** edge in SANA's published decode ladder whose
/// recorded footprint bound fits [`IOS_MEMORY_BUDGET_MIB`] at every one of [`IOS_IMAGE_EDGES`].
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
                sana_footprint_bound_mib(image, tile)
                    .is_some_and(|footprint| footprint <= IOS_MEMORY_BUDGET_MIB)
            })
        })
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "no published SANA decode tile edge {DECODE_TILE_EDGES:?} has a recorded iOS \
                 footprint within {IOS_MEMORY_BUDGET_MIB} MiB at every size {IOS_IMAGE_EDGES:?}"
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
            ["sana_1600m", "sana_sprint_1600m"],
            "the iOS media surface changed -- if deliberate, update this test and confirm the new \
             provider fits the measured iOS cap"
        );
        assert_eq!(registry.trainers().len(), 0, "iOS ships no trainers");
        assert_eq!(registry.captioners().len(), 0, "iOS ships no captioners");
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

    /// The selected iOS plan: tile 192, because 192's recorded footprint (2860 MiB at 1024², which
    /// bounds 512² at the same tile) fits 4 GiB at both sizes, while the larger 256 is excluded by
    /// its 4113 MiB record at 512² and 512/384 have no device record at all.
    #[test]
    fn ios_sana_plan_selects_the_largest_tile_that_fits_4_gib_at_512_and_1024() {
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
            let footprint = super::sana_footprint_bound_mib(image, plan.edge as u32)
                .unwrap_or_else(|| panic!("no recorded footprint bounds {image}² at tile 192"));
            assert!(
                footprint <= 4096,
                "{image}² at tile {} records {footprint} MiB, over 4 GiB",
                plan.edge
            );
        }
        assert_eq!(super::sana_footprint_bound_mib(1024, 192), Some(2860));
        assert_eq!(super::sana_footprint_bound_mib(512, 192), Some(2860));

        // Every published edge larger than the selection is excluded for a recorded reason.
        assert_eq!(super::sana_footprint_bound_mib(512, 256), Some(4113));
        for larger in mlx_gen_sana::memory_strategy::DECODE_TILE_EDGES
            .iter()
            .filter(|&&edge| edge > plan.edge as u32)
        {
            assert!(
                [512, 1024]
                    .iter()
                    .any(|&image| super::sana_footprint_bound_mib(image, *larger)
                        .is_none_or(|footprint| footprint > 4096)),
                "tile {larger} fits 4 GiB at every iOS size and should have been selected"
            );
        }

        // A larger image never bounds a smaller one across tile edges.
        assert_eq!(super::sana_footprint_bound_mib(512, 128), Some(2904));
        assert_eq!(super::sana_footprint_bound_mib(1024, 256), None);
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
