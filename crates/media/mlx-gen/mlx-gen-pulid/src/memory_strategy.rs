//! Shared image-memory ladder for the bespoke PuLID-FLUX MLX route.
//!
//! ## Envelope vs. structure in the request route gate (sc-20569 twin)
//!
//! The route gate carries two kinds of clause and they must not be confused — the same distinction
//! SC-20569 established for mlx-gen-sensenova (PR #699) and PR #700 mirrored into mlx-gen-flux. A
//! **structural** clause (the loaded mode/reference/overlay match, PiD requested without a loaded
//! PiD overlay, request phases) states what THIS loaded route can do at all — no authority changes
//! that — and stays fail-closed unconditionally. The **envelope** clause (the calibrated-geometry
//! block) states what the `2026-08-04` PuLID-FLUX Q4 campaign MEASURED: 1024x1024, batch 1, one
//! frame. That is a statement about which evidence exists, not about what the engine can render.
//!
//! The PuLID descriptor accepts `256..=2048` per side, and the SceneWorks image lane deliberately
//! does NOT enforce a model's advertised `limits.resolutions` (sc-12384): `ImageRequest` only
//! sanity-clamps `width`/`height` to `256..=4096` and hands them to the engine as-is, so any
//! API/MCP caller may name an off-cell but engine-legal geometry for a `pulid_flux_dev`
//! `character_image` job. This clause fired on ANY optimized-rung selection regardless of
//! authority, so such a request was hard-refused instead of being admitted on the caller's
//! estimate. Only a context that also claims measured evidence
//! (`optimization_authority == Calibrated`) is held to the measured cell; an `Estimated` claim —
//! exactly what `AdmissionPath::Legacy` in the SceneWorks fit gate carries for an out-of-envelope
//! request — must degrade to that legacy/estimated admission instead of refusing.
//!
//! Every clause inside the route gate therefore builds a `CoreError::Msg`, not
//! `CoreError::Unsupported`. `standard_memory_strategy_safety_check` stringifies a route-gate error
//! into `MemorySafetyDecision::Reject` immediately (`error.to_string()`), so nothing downstream can
//! read the type, while `begin_with` types the surviving string as `Unsupported` once at the
//! request boundary. Building `Unsupported` inside the closure too doubled the prefix into
//! `unsupported: unsupported: …` — the same secondary defect SC-20569 fixed in mlx-gen-sensenova.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, MemoryBehaviorFixture,
    MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryMode, MemoryNumericTier,
    MemoryOptimizationAuthority, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, Result as CoreResult,
};
use mlx_gen::{LoadSpec, WeightsSource};
use sha2::{Digest, Sha256};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "pulid-flux-dev-q4-mlx-shared-ladder-2026-08-04-v1";
/// Source-owned weights-free behavior identity. Version route/fixture semantics here without
/// restamping the real-weight calibration above.
const STATIC_CALIBRATION: &str = "pulid-flux-static-registry-behavior-v2";
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen_flux::memory_strategy::ATTENTION_CHUNK_SIZE;
pub const DECODE_TILE_EDGES: &[u32] = mlx_gen_flux::memory_strategy::DECODE_TILE_EDGES;
pub const DECODE_OVERLAP: u32 = mlx_gen_flux::memory_strategy::DECODE_OVERLAP;

const EXPECTED: &[&str] = &[
    "92c41c3af322b02e58e1b32842e4601e08c8f16ec1fe80089dbe957df510f51d",
    "69417ca25d214f5e93704475f3e2a7e235fd1f1bf4a5acdf3f39ec69b51513cd",
    "7b40147a85771139e70a8d9fe6be27ffcf32f4c911770ef24b5b05c29f534eda",
    "9deff2fef8fe1b3e357a99c01f28cc478dd8acbeab0d3749d252f6d69990ee39",
    "c6b6f092ab266b13593ae0fbbc21c4546877577f6b1d2ae707f25d62ee767cac",
];

fn path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::File(path) | WeightsSource::Dir(path) => path,
    }
}

fn identity_paths(spec: &LoadSpec) -> CoreResult<Vec<PathBuf>> {
    let identity = spec.identity.as_ref().ok_or_else(|| {
        CoreError::Unsupported("pulid_flux: exact contract requires identity weights".into())
    })?;
    let encoder = identity
        .encoder
        .as_ref()
        .ok_or_else(|| CoreError::Unsupported("pulid_flux: missing identity encoder".into()))?;
    let eva = identity
        .eva
        .as_ref()
        .ok_or_else(|| CoreError::Unsupported("pulid_flux: missing EVA tower".into()))?;
    let face = identity
        .face_dir
        .as_ref()
        .ok_or_else(|| CoreError::Unsupported("pulid_flux: missing face_dir".into()))?;
    Ok(vec![
        path(encoder).into(),
        path(eva).into(),
        path(face).join("scrfd_10g.safetensors"),
        path(face).join("arcface_iresnet100.safetensors"),
        path(face).join("bisenet_parsing.safetensors"),
    ])
}

fn hash(path: &Path) -> CoreResult<String> {
    let mut file = File::open(path).map_err(|error| {
        CoreError::Unsupported(format!("pulid_flux: open {}: {error}", path.display()))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|error| {
            CoreError::Unsupported(format!("pulid_flux: hash {}: {error}", path.display()))
        })?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Clone)]
pub(crate) struct IdentityArtifactInventory {
    files: Vec<(PathBuf, PathBuf, String)>,
    bytes: u64,
    exact_calibration: bool,
}

impl IdentityArtifactInventory {
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn is_exact_calibration(&self) -> bool {
        self.exact_calibration
    }

    pub(crate) fn ensure_unchanged(&self) -> CoreResult<()> {
        for (path, admitted_path, expected) in &self.files {
            let current_path = path.canonicalize().map_err(|error| {
                CoreError::Unsupported(format!(
                    "pulid_flux: resolve {} after admission: {error}",
                    path.display()
                ))
            })?;
            if current_path != *admitted_path
                || !admitted_path.is_file()
                || hash(admitted_path)? != *expected
            {
                return Err(CoreError::Unsupported(format!(
                    "pulid_flux: pinned identity artifact changed after admission: {}",
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

pub(crate) fn admitted_identity_inventory(
    spec: &LoadSpec,
) -> CoreResult<IdentityArtifactInventory> {
    let mut bytes = 0_u64;
    let mut exact_calibration = true;
    let files = identity_paths(spec)?
        .into_iter()
        .zip(EXPECTED.iter().copied())
        .map(|(path, expected)| {
            let admitted_path = path.canonicalize().map_err(|error| {
                CoreError::Unsupported(format!(
                    "pulid_flux: resolve {} during admission: {error}",
                    path.display()
                ))
            })?;
            let admitted_hash = hash(&admitted_path)?;
            Ok((path, admitted_path, admitted_hash, expected))
        })
        .collect::<CoreResult<Vec<_>>>()?;
    for (_, admitted_path, admitted_hash, expected) in &files {
        exact_calibration &= admitted_hash == expected;
        bytes = bytes.saturating_add(
            admitted_path
                .metadata()
                .map_err(|error| CoreError::Unsupported(error.to_string()))?
                .len(),
        );
    }
    let inventory = IdentityArtifactInventory {
        files: files
            .into_iter()
            .map(|(path, admitted_path, admitted_hash, _)| (path, admitted_path, admitted_hash))
            .collect(),
        bytes,
        exact_calibration,
    };
    inventory.ensure_unchanged()?;
    Ok(inventory)
}

pub(crate) fn backbone_spec(spec: &LoadSpec) -> LoadSpec {
    let mut base = spec.clone();
    base.identity = None;
    base
}

fn adapt(
    mut contract: MemoryProviderContract,
    spec: &LoadSpec,
    identity_bytes: Option<u64>,
    weights_free: bool,
) -> MemoryProviderContract {
    contract.provider_id = crate::pulid_flux::MODEL_ID.into();
    contract.asset_facts.base_bytes = contract
        .asset_facts
        .base_bytes
        .saturating_add(identity_bytes.unwrap_or(0));
    contract.calibration = if weights_free {
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION,
            spec.load_shape,
        ))
    } else if identity_bytes.is_some() && contract.calibration.is_some() {
        Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        ))
    } else {
        None
    };
    contract
}

pub fn contract_for(spec: &LoadSpec) -> CoreResult<MemoryProviderContract> {
    let inventory = admitted_identity_inventory(spec)?;
    contract_with_inventory(spec, &inventory)
}

pub(crate) fn contract_with_inventory(
    spec: &LoadSpec,
    inventory: &IdentityArtifactInventory,
) -> CoreResult<MemoryProviderContract> {
    let base = backbone_spec(spec);
    let contract =
        mlx_gen_flux::memory_strategy::memory_strategy_contract(mlx_gen_flux::FLUX1_DEV_ID, &base)?;
    Ok(adapt(
        contract,
        spec,
        inventory.is_exact_calibration().then(|| inventory.bytes()),
        false,
    ))
}

pub fn weights_free_contract(spec: &LoadSpec) -> CoreResult<MemoryProviderContract> {
    let base = backbone_spec(spec);
    let contract = mlx_gen_flux::memory_strategy::weights_free_memory_strategy_contract(
        mlx_gen_flux::FLUX1_DEV_ID,
        &base,
    )?;
    Ok(adapt(contract, spec, Some(0), true))
}

pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route = || {
        if context.mode != MemoryMode::ImageToImage
            || context.geometry.reference_count != 1
            || context.use_pid
            || context.has_phases
            || context.overlay.as_deref() != Some("identity")
        {
            return Err(CoreError::Msg("pulid_flux: memory route requires one identity reference, overlay=identity, native VAE decode, and no request phases".into()));
        }
        // sc-20569 twin: `claims_measured_evidence` is the same envelope/structure discriminator the
        // SenseNova and FLUX route gates use. An optimized rung admitted behind a legacy/estimated
        // authority has NOT claimed to be graded against the measured cell, so it must degrade to
        // admission instead of refusing; only a `Calibrated` claim off the measured cell fails
        // closed, because admitting THAT would grade a request against evidence captured at a
        // different geometry.
        let claims_measured_evidence =
            context.optimization_authority == MemoryOptimizationAuthority::Calibrated;
        if contract
            .calibration
            .as_ref()
            .is_some_and(|id| id.fingerprint == MEMORY_CALIBRATION_FINGERPRINT)
            && context.selection.strategy.is_optimized()
            && claims_measured_evidence
            && (context.geometry.width != 1024
                || context.geometry.height != 1024
                || context.geometry.batch != 1
                || context.geometry.frames != 1)
        {
            return Err(CoreError::Msg("pulid_flux: calibrated geometry is exactly 1024x1024, batch 1, one frame/reference".into()));
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        }),
        Some(&route),
    )
}

pub fn registered_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: MemoryMode::ImageToImage,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: Some("identity".into()),
        },
    )?;
    let mut fixture = MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free PuLID memory behavior".into();
    Ok(vec![fixture])
}

pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_with(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub fn begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_with(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_with(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    // The one place a refused route becomes a TYPED error. `MemorySafetyDecision::Reject` carries an
    // already-rendered string, so the route gate deliberately hands back plain reasons (see
    // `safety_check`) and the `unsupported: ` prefix is applied exactly once, here.
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        crate::pulid_flux::MODEL_ID,
        context.geometry,
        contract.generation_memory(&context.selection),
        false,
        57,
        |_pid, edge, overlap| {
            if DECODE_TILE_EDGES.contains(&edge) && overlap == DECODE_OVERLAP {
                Ok(())
            } else {
                Err(CoreError::Unsupported(format!(
                    "pulid_flux: decode tile {edge}/{overlap} is outside the production domain"
                )))
            }
        },
    )?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten();
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{IdentityWeights, LoadShape, OffloadPolicy, Quant};

    fn spec() -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_quant(Quant::Q4)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        spec.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File("/encoder".into())),
            eva: Some(WeightsSource::File("/eva".into())),
            face_dir: Some(WeightsSource::Dir("/face".into())),
        });
        spec
    }

    /// AC (SC-22662): PuLID-FLUX owns no denoiser — it borrows FLUX.1-dev's contract and rewrites
    /// the provider id — so the FLUX DiT's axes must survive that adaptation intact, and the adapted
    /// contract must pass the shared facts conformance check.
    #[test]
    fn architecture_facts_ride_through_from_the_flux_backbone() {
        let spec = spec();
        let contract = weights_free_contract(&spec).unwrap();
        let backbone = mlx_gen_flux::memory_strategy::weights_free_memory_strategy_contract(
            mlx_gen_flux::FLUX1_DEV_ID,
            &backbone_spec(&spec),
        )
        .unwrap();
        assert_eq!(
            contract.architecture_facts, backbone.architecture_facts,
            "adapting the contract must not drop the backbone's declared axes"
        );
        assert_eq!(
            contract.architecture_facts,
            mlx_gen::gen_core::MemoryArchitectureFacts {
                attention_heads: Some(24),
                head_dim: Some(128),
                // 19 joint + 38 single FLUX.1 blocks.
                transformer_blocks: Some(57),
                patch_size: Some(2),
                latent_channels: Some(16),
                vae_spatial_scale: Some(8),
                vae_temporal_scale: None,
                activation_dtype_width: Some(4),
            }
        );
        assert!(contract.architecture_facts.has_snapshot_read_axis());
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);
    }

    #[test]
    fn bespoke_contract_declares_every_rung_and_identity_route() {
        let spec = spec();
        let contract = weights_free_contract(&spec).unwrap();
        assert_eq!(contract.provider_id, crate::pulid_flux::MODEL_ID);
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            "pulid-flux-static-registry-behavior-v2"
        );
        assert_eq!(
            MEMORY_CALIBRATION_FINGERPRINT, "pulid-flux-dev-q4-mlx-shared-ladder-2026-08-04-v1",
            "source-owned route correction must not restamp measured evidence"
        );
        assert!(contract.conformance_errors().is_empty());
        for strategy in MemoryStrategy::ALL {
            assert!(matches!(
                contract.capability(strategy).unwrap().support,
                mlx_gen::gen_core::MemoryStrategySupport::Implemented
            ));
        }
        let fixture = registered_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0);
        assert_eq!(
            safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Accept
        );
        assert!(!fixture.context.has_phases);
        assert!(fixture.request.phases.is_none());
        assert_eq!(fixture.request.width, fixture.context.geometry.width);
        assert_eq!(fixture.request.height, fixture.context.geometry.height);
        assert_eq!(fixture.request.count, fixture.context.geometry.batch);
        assert_eq!(fixture.request.use_pid, fixture.context.use_pid);
        assert!(matches!(
            fixture.request.conditioning.as_slice(),
            [mlx_gen::Conditioning::Reference { .. }]
        ));
        crate::pulid_flux::descriptor()
            .capabilities
            .validate_request(crate::pulid_flux::MODEL_ID, &fixture.request)
            .unwrap();

        let mut wrong = fixture.context.clone();
        wrong.overlay = None;
        assert!(matches!(
            safety_check(&spec, &contract, &wrong),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut phases = fixture.context;
        phases.has_phases = true;
        assert!(matches!(
            safety_check(&spec, &contract, &phases),
            MemorySafetyDecision::Reject { .. }
        ));
        assert!(registered_begin_request(&spec, &contract, &phases).is_err());
    }

    /// Engine-legal, off-measured-cell geometries a real `pulid_flux_dev` request can name. The
    /// SceneWorks manifest advertises only `1024x1024` in `limits.resolutions`, but the image lane
    /// deliberately does NOT enforce that list (sc-12384): `ImageRequest` sanity-clamps
    /// `width`/`height` to `256..=4096` and hands them to the engine, and this descriptor accepts
    /// `256..=2048` per side. So every pair below reaches the route gate from an API/MCP caller.
    const OFF_CELL_GEOMETRIES: [(u32, u32); 4] =
        [(768, 768), (1024, 1024), (1280, 720), (720, 1280)];

    /// `pulid_flux_dev`'s manifest `limits.count`. SceneWorks pins the provider-facing `batch` to
    /// one forward pass today — `pulid_memory_inputs` hard-sets `count: 1` and `request_batch`
    /// returns 1 (sc-16194), because a multi-image job is a sequential loop — but the gate is a
    /// provider-owned seam and any caller may set it, so the degrade is proven across the full
    /// advertised count axis rather than at the worker's current pin.
    const MANIFEST_COUNTS: [u32; 3] = [1, 2, 4];

    fn calibrated_contract(spec: &LoadSpec) -> MemoryProviderContract {
        let mut contract = weights_free_contract(spec).unwrap();
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        ));
        contract
    }

    fn identity_route() -> MemoryBehaviorRoute {
        MemoryBehaviorRoute {
            mode: MemoryMode::ImageToImage,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: Some("identity".into()),
        }
    }

    fn route_context(
        spec: &LoadSpec,
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
    ) -> MemoryRunContext {
        mlx_gen::gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            MemoryNumericTier {
                precision: spec.precision,
                quant: spec.quantize,
                component_precision_floors: &[],
            },
            identity_route(),
        )
        .unwrap()
    }

    /// sc-20569 twin (mlx-gen-pulid): every engine-legal geometry/count off the measured
    /// 1024x1024/batch-1/frame-1 cell used to be refused unconditionally by the calibrated-geometry
    /// envelope clause whenever an optimized rung was selected, regardless of authority. A context
    /// that does NOT claim measured evidence — `AdmissionPath::Legacy` in the SceneWorks fit gate,
    /// whether it synthesized an estimate ladder or froze to the resident baseline — must be
    /// ADMITTED at every such cell, and must be able to open a request scope.
    #[test]
    fn off_cell_geometry_and_count_degrade_to_legacy_admission() {
        let spec = spec();
        let contract = calibrated_contract(&spec);

        let mut admitted = 0;
        for (width, height) in OFF_CELL_GEOMETRIES {
            for batch in MANIFEST_COUNTS {
                for (authority, strategy) in [
                    (
                        MemoryOptimizationAuthority::Estimated,
                        MemoryStrategy::BoundedAttention,
                    ),
                    (
                        MemoryOptimizationAuthority::Estimated,
                        MemoryStrategy::BoundedTransformerResidency,
                    ),
                    (
                        MemoryOptimizationAuthority::Resident,
                        MemoryStrategy::Resident,
                    ),
                ] {
                    let label =
                        format!("{width}x{height} batch {batch} {authority:?}/{strategy:?}");
                    let mut context = route_context(&spec, &contract, strategy);
                    context.optimization_authority = authority;
                    context.geometry.width = width;
                    context.geometry.height = height;
                    context.geometry.batch = batch;
                    assert_eq!(
                        safety_check(&spec, &contract, &context),
                        MemorySafetyDecision::Accept,
                        "{label}"
                    );
                    assert!(
                        registered_begin_request(&spec, &contract, &context)
                            .unwrap_or_else(|error| panic!("{label}: {error}"))
                            .is_some(),
                        "{label}"
                    );
                    admitted += 1;
                }
            }
        }
        assert_eq!(
            admitted,
            OFF_CELL_GEOMETRIES.len() * MANIFEST_COUNTS.len() * 3,
            "four geometries x three counts x three legacy dispositions"
        );
    }

    /// A `Calibrated` claim off the measured cell must STILL fail closed — admitting it would grade
    /// the request against evidence captured at a different geometry. This is the other half of the
    /// conditioning: dropping `claims_measured_evidence` makes the degrade test above pass on its
    /// own, so this test is what pins the clause to the authority rather than deleting it.
    #[test]
    fn a_measured_claim_off_the_campaign_cell_still_refuses() {
        let spec = spec();
        let contract = calibrated_contract(&spec);
        for (width, height, batch, frames) in [
            (768, 768, 1, 1),
            (1280, 720, 1, 1),
            (1024, 1024, 2, 1),
            (1024, 1024, 1, 2),
        ] {
            let mut context = route_context(
                &spec,
                &contract,
                MemoryStrategy::BoundedTransformerResidency,
            );
            assert_eq!(
                context.optimization_authority,
                MemoryOptimizationAuthority::Calibrated,
                "the behavior context must claim measured evidence by default"
            );
            context.geometry.width = width;
            context.geometry.height = height;
            context.geometry.batch = batch;
            context.geometry.frames = frames;
            let label = format!("{width}x{height} batch {batch} frames {frames}");
            assert!(
                matches!(
                    safety_check(&spec, &contract, &context),
                    MemorySafetyDecision::Reject { reason }
                        if reason.contains("calibrated geometry is exactly 1024x1024")
                ),
                "{label}: a measured claim off the campaign cell must fail closed"
            );
            assert!(
                registered_begin_request(&spec, &contract, &context).is_err(),
                "{label}"
            );
        }
    }

    /// The degrade must not weaken the STRUCTURAL refusals. No amount of estimate authority makes a
    /// clean PuLID load answer a text-to-image request, run without its identity reference, drop the
    /// `identity` overlay, admit PiD it never loaded, or run a multi-phase trajectory. Each axis is
    /// mutated on its own so every guard is asked its own question.
    #[test]
    fn structural_refusals_are_not_weakened_by_the_legacy_degrade() {
        let spec = spec();
        let contract = calibrated_contract(&spec);
        let mut legacy = route_context(&spec, &contract, MemoryStrategy::BoundedAttention);
        legacy.optimization_authority = MemoryOptimizationAuthority::Estimated;
        legacy.geometry.width = 768;
        legacy.geometry.height = 768;
        // The unmutated legacy context is admitted, so each rejection below is attributable to the
        // one axis it mutates rather than to the off-cell geometry it carries.
        assert_eq!(
            safety_check(&spec, &contract, &legacy),
            MemorySafetyDecision::Accept
        );

        let mut wrong_mode = legacy.clone();
        wrong_mode.mode = MemoryMode::TextToImage;
        let mut no_reference = legacy.clone();
        no_reference.geometry.reference_count = 0;
        no_reference.has_reference = false;
        let mut no_overlay = legacy.clone();
        no_overlay.overlay = None;
        let mut use_pid = legacy.clone();
        use_pid.use_pid = true;
        let mut has_phases = legacy.clone();
        has_phases.has_phases = true;
        for (label, context) in [
            ("wrong_mode", wrong_mode),
            ("no_reference", no_reference),
            ("no_overlay", no_overlay),
            ("use_pid", use_pid),
            ("has_phases", has_phases),
        ] {
            assert!(
                matches!(
                    safety_check(&spec, &contract, &context),
                    MemorySafetyDecision::Reject { reason }
                        if reason.contains("memory route requires one identity reference")
                ),
                "{label}: structural refusal must survive the legacy degrade"
            );
            assert!(
                registered_begin_request(&spec, &contract, &context).is_err(),
                "{label}"
            );
        }
    }

    /// sc-20569 twin secondary: `begin_with` used to double-stringify a route-gate refusal —
    /// `standard_memory_strategy_safety_check` renders the route gate's error into
    /// `MemorySafetyDecision::Reject` once (`error.to_string()`), and `begin_with` typed that
    /// already-rendered string as `Unsupported` again, printing
    /// `unsupported: unsupported: pulid_flux: …`. Exactly one prefix must reach the caller.
    #[test]
    fn a_route_refusal_surfaces_the_unsupported_prefix_exactly_once() {
        let spec = spec();
        let contract = calibrated_contract(&spec);
        let mut context = route_context(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        );
        context.geometry.width = 768;
        context.geometry.height = 768;
        let error = match registered_begin_request(&spec, &contract, &context) {
            Ok(_) => panic!("a measured claim off the campaign cell must still refuse"),
            Err(error) => error.to_string(),
        };
        assert_eq!(
            error.matches("unsupported: ").count(),
            1,
            "the rendered refusal must carry one `unsupported: ` prefix: {error}"
        );
        assert!(
            error.starts_with(&format!("unsupported: {}: ", crate::pulid_flux::MODEL_ID)),
            "{error}"
        );
    }

    #[test]
    fn registry_publishes_exact_twelve_surface_ladder() {
        use mlx_gen::gen_core::{
            LoadShape, MemoryContractSurfaceTier, MemoryStrategySupport, OffloadPolicy,
        };
        use std::collections::BTreeSet;

        let registry = crate::provider_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        assert_eq!(surfaces.len(), 12);
        let selectors: BTreeSet<_> = surfaces
            .iter()
            .map(|surface| surface.selector.id())
            .collect();
        let expected: BTreeSet<_> = [
            "bf16:resident:eager",
            "bf16:resident:deferred",
            "bf16:sequential:eager",
            "bf16:sequential:deferred",
            "q4:resident:eager",
            "q4:resident:deferred",
            "q4:sequential:eager",
            "q4:sequential:deferred",
            "q8:resident:eager",
            "q8:resident:deferred",
            "q8:sequential:eager",
            "q8:sequential:deferred",
        ]
        .into_iter()
        .collect();
        assert_eq!(selectors, expected);

        let mut resident = 0;
        let mut staged = 0;
        let mut bounded_transformer = 0;
        for surface in surfaces {
            assert_eq!(surface.contract.provider_id, crate::pulid_flux::MODEL_ID);
            assert!(matches!(
                surface.resolved_artifact_tier(),
                MemoryContractSurfaceTier::Bf16
                    | MemoryContractSurfaceTier::Q4
                    | MemoryContractSurfaceTier::Q8
            ));
            let sequential = surface.selector.offload_policy == OffloadPolicy::Sequential;
            let deferred = surface.selector.load_shape == LoadShape::DeferredMaterialization;
            let implemented = |strategy| {
                surface.contract.capability(strategy).unwrap().support
                    == MemoryStrategySupport::Implemented
            };
            resident += usize::from(implemented(MemoryStrategy::Resident));
            staged += usize::from(implemented(MemoryStrategy::StagedResidency));
            bounded_transformer +=
                usize::from(implemented(MemoryStrategy::BoundedTransformerResidency));
            assert!(implemented(MemoryStrategy::Resident));
            assert_eq!(implemented(MemoryStrategy::StagedResidency), sequential);
            assert_eq!(
                implemented(MemoryStrategy::BoundedTransformerResidency),
                sequential && deferred
            );
        }
        assert_eq!(resident, 12);
        assert_eq!(staged, 6);
        assert_eq!(bounded_transformer, 3);
    }

    #[test]
    fn noncanonical_identity_is_admitted_uncalibrated_and_pinned() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let face = root.join("face");
        std::fs::create_dir_all(&face).unwrap();
        let encoder = root.join("encoder.safetensors");
        let eva = root.join("eva.safetensors");
        std::fs::write(&encoder, b"compatible noncanonical encoder").unwrap();
        std::fs::write(&eva, b"compatible noncanonical eva").unwrap();
        for name in [
            "scrfd_10g.safetensors",
            "arcface_iresnet100.safetensors",
            "bisenet_parsing.safetensors",
        ] {
            std::fs::write(face.join(name), name.as_bytes()).unwrap();
        }

        let mut spec = spec();
        spec.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File(encoder.clone())),
            eva: Some(WeightsSource::File(eva)),
            face_dir: Some(WeightsSource::Dir(face)),
        });
        let inventory = admitted_identity_inventory(&spec).unwrap();
        assert!(!inventory.is_exact_calibration());
        inventory.ensure_unchanged().unwrap();

        std::fs::write(&encoder, b"mutated noncanonical encoder").unwrap();
        assert!(inventory.ensure_unchanged().is_err());
    }
}
