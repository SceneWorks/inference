//! Request-scoped memory contract for the bespoke InstantID Candle/CUDA route.
//!
//! This contract deliberately does not inherit SDXL or PuLID evidence. InstantID's IdentityNet,
//! face IP adapter, optional OpenPose branch, adapters, PiD decoder, and restoration pass form one
//! exact composition identity and must be admitted together.

use std::path::Path;

use candle_gen::gen_core::{
    self, AdapterResidencyMode, LoadShape, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, MemoryWindowMaterialization, Precision, WeightsSource,
};

use crate::model::InstantIdPaths;

pub const PROVIDER_ID: &str = "instantid";
/// Revision of the shared InstantID request/evidence schema. Backend is an independent identity
/// axis, so encoding Candle here would split otherwise identical cross-backend evidence semantics.
pub const REQUEST_EVIDENCE_REVISION: &str = "instantid-request-contract-v1";
/// Identity of the *executable memory semantics* this contract calibrates — deliberately a separate
/// string from [`REQUEST_EVIDENCE_REVISION`], which versions the request/evidence schema. They are
/// two independent axes: sharing one literal made a bump to either silently restate the other.
const CALIBRATION_FINGERPRINT: &str = "instantid-candle-staged-conditioning-v1";

/// InstantID materializes fp16 weights ([`crate::model`]'s `DTYPE`), so each float element of every
/// component costs two bytes once loaded.
const FLOAT_WIDTH: u64 = 2;

fn source_path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path,
    }
}

/// Bytes one resolved component occupies once loaded: float tensors at the compute width, integer
/// tensors at their stored width. Header-only — no tensor data is materialized.
fn component_bytes(path: &Path) -> gen_core::Result<u64> {
    gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .try_fold(0_u64, |sum, header| {
            let bytes = if header.is_float() {
                header.materialized_bytes(FLOAT_WIDTH)?
            } else {
                header.data_bytes
            };
            sum.checked_add(bytes).ok_or_else(|| {
                gen_core::Error::Msg("instantid: component byte sum overflow".into())
            })
        })
}

/// Bytes the [`candle_gen_face`] SCRFD + ArcFace stack occupies once `candle_gen_face::load_on`
/// materializes it.
///
/// That loader reads exactly the two files [`candle_gen_face::ANALYSIS_STACK_FILES`] names and
/// coerces every tensor to **f32**, so the stack is priced at 4 bytes per float element and NOT at
/// [`FLOAT_WIDTH`] — it is not part of the fp16 diffusion graph. Pricing the whole dir instead would
/// also charge a `bisenet_parsing.safetensors` that shares the layout but that InstantID never loads.
fn face_stack_bytes(dir: &Path) -> gen_core::Result<u64> {
    const FACE_WIDTH: u64 = 4;
    candle_gen_face::ANALYSIS_STACK_FILES
        .iter()
        .try_fold(0_u64, |sum, file| {
            let bytes = gen_core::weightsmeta::materialized_path_bytes(dir.join(file), FACE_WIDTH)?;
            sum.checked_add(bytes).ok_or_else(|| {
                gen_core::Error::Msg("instantid: face stack byte sum overflow".into())
            })
        })
}

/// Load-exact component bytes for the exact InstantID composition.
///
/// IdentityNet, the face IP-Adapter, the optional OpenPose ControlNet and the optional
/// face-analysis stack are auxiliary networks resident alongside the SDXL base, so they are
/// declared in `overlay_bytes` (the aggregate this contract's
/// [`MemoryFormulaVariable::OverlayBytes`] makes load-bearing) rather than folded into the three
/// base-model fields. User LoRA/LoKr adapters are folded onto the UNet at load and therefore add no
/// resident bytes of their own.
///
/// OpenPose and the face stack are priced **because the served generator holds them**:
/// `InstantId::with_openpose` loads a second full SDXL ControlNet through the same
/// `load_sdxl_controlnet` as IdentityNet, `with_face` loads SCRFD + ArcFace, and a staged reload
/// re-materializes both. Before epic SC-22657 neither had a field on [`InstantIdPaths`] at all, so
/// this function could not have charged them even in principle — an under-price of two whole
/// networks, which is exactly the defect class E1 excludes. A composition that attaches neither
/// leaves both `None` and pays nothing.
pub fn asset_facts(paths: &InstantIdPaths) -> gen_core::Result<MemoryAssetFacts> {
    let conditioning = component_bytes(&paths.sdxl_base.join("text_encoder"))?
        .saturating_add(component_bytes(&paths.sdxl_base.join("text_encoder_2"))?);
    let transformer = component_bytes(&paths.sdxl_base.join("unet"))?;
    let decoder = component_bytes(source_path(paths.sdxl.vae_fp16_fix()))?;
    let overlay = component_bytes(source_path(&paths.identitynet))?
        .saturating_add(component_bytes(&paths.ip_adapter)?)
        .saturating_add(
            gen_core::adapter_stack_resident_bytes(&paths.adapters, AdapterResidencyMode::Folded)
                .ok_or_else(|| {
                gen_core::Error::Unsupported(
                    "instantid: every resident adapter must have an exact non-zero size".into(),
                )
            })?,
        );
    let overlay = match &paths.openpose {
        Some(source) => overlay.saturating_add(component_bytes(source_path(source))?),
        None => overlay,
    };
    let overlay = match &paths.face_dir {
        Some(dir) => overlay.saturating_add(face_stack_bytes(dir)?),
        None => overlay,
    };
    Ok(MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: overlay,
    })
}

/// The executable contract for a real InstantID composition: identical to [`provider_contract`]
/// except that its declared [`MemoryFormulaVariable::AssetBytes`] / [`MemoryFormulaVariable::OverlayBytes`]
/// inputs are the exact on-disk component inventory rather than zero placeholders.
pub fn provider_contract_for_paths(
    paths: &InstantIdPaths,
) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = provider_contract();
    contract.asset_facts = asset_facts(paths)?;
    contract.architecture_facts = architecture_facts(paths);
    Ok(contract)
}

/// Architecture axes for a resolved InstantID composition (epic SC-22657, E2).
///
/// InstantID's denoiser is the **vendored SDXL UNet** and its decoder the staged
/// `madebyollin/sdxl-vae-fp16-fix` `AutoencoderKL`: it loads both through [`candle_gen_sdxl`]
/// (`model.rs` → `candle_gen_sdxl::load_instantid_unet`), so the axes come from that crate's own
/// [`candle_gen_sdxl::sdxl_unet_family_architecture_facts`] rather than from duplicated constants
/// or from a snapshot `config.json` the vendored stack never reads. The IdentityNet and face
/// IP-Adapter are auxiliary networks bolted onto that same UNet; they change none of these axes.
///
/// That helper declines the four axes a UNet denoiser structurally lacks — a per-stage head count
/// (5/10/20) is not a uniform `attention_heads`; a down/mid/up trunk is not a `transformer_blocks`
/// stack; the UNet consumes the latent grid unpatchified; and the image `AutoencoderKL` has no
/// temporal axis. The activation width is [`crate::model::DTYPE`] (fp16), what InstantID
/// materializes at.
///
/// `paths.sdxl_base` is the proof of a materialized snapshot: [`provider_contract`] is the
/// weights-free form, has resolved nothing, and keeps every axis `None`.
fn architecture_facts(paths: &InstantIdPaths) -> gen_core::MemoryArchitectureFacts {
    if !paths.sdxl_base.is_dir() {
        return gen_core::MemoryArchitectureFacts::default();
    }
    candle_gen_sdxl::sdxl_unet_family_architecture_facts(crate::model::DTYPE)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstantIdRoute {
    Identity,
    Angle,
    Pose,
}

impl InstantIdRoute {
    pub const fn as_key(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Angle => "angle",
            Self::Pose => "pose-openpose",
        }
    }
}

/// Immutable identity of everything that can change InstantID residency or execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstantIdMemoryIdentity {
    pub route: InstantIdRoute,
    pub adapter_count: usize,
    pub use_pid: bool,
    pub face_restore: bool,
    pub artifact_fingerprint: String,
}

impl InstantIdMemoryIdentity {
    pub fn overlay_key(&self) -> String {
        format!(
            "instantid-v1/{}/a{}/p{}/r{}/{}",
            self.route.as_key(),
            self.adapter_count,
            u8::from(self.use_pid),
            u8::from(self.face_restore),
            self.artifact_fingerprint
        )
    }
}

pub fn provider_contract() -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        PROVIDER_ID,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: false,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        CALIBRATION_FINGERPRINT,
        LoadShape::EagerMaterialization,
    ));
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::OverlayBytes,
        ],
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedDecode
            | MemoryStrategy::BoundedAttention
            | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
    }
    contract
}

pub const fn resolved_numeric_tier() -> MemoryNumericTier {
    MemoryNumericTier {
        // `Bf16` is gen-core's dense-default sentinel; InstantID materializes fp16 weights.
        precision: Precision::Bf16,
        quant: None,
        component_precision_floors: &[],
    }
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    identity: &InstantIdMemoryIdentity,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route = || {
        if context.mode != MemoryMode::Other("character_image".into())
            || !context.has_reference
            || context.geometry.reference_count != 1
            || context.geometry.batch != 1
            || context.geometry.frames != 1
            || context.use_pid != identity.use_pid
            || context.has_phases != identity.face_restore
            || context.overlay.as_deref() != Some(identity.overlay_key().as_str())
            || context.evidence_revision != REQUEST_EVIDENCE_REVISION
        {
            return Err(gen_core::Error::Msg(format!(
                "{PROVIDER_ID}: context does not match the exact character_image composition"
            )));
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(resolved_numeric_tier()),
        Some(&route),
    )
}

pub fn validate_context(
    contract: &MemoryProviderContract,
    identity: &InstantIdMemoryIdentity,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    match safety_check(contract, identity, context) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_gen::gen_core::{MemoryBehaviorRoute, MemoryStrategy};
    use std::path::PathBuf;

    #[test]
    fn only_resident_and_staged_are_selectable() {
        let contract = provider_contract();
        for capability in contract.strategies {
            assert_eq!(
                capability.support,
                if matches!(
                    capability.strategy,
                    MemoryStrategy::Resident | MemoryStrategy::StagedResidency
                ) {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                }
            );
        }
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn composition_identity_distinguishes_every_overlay_axis() {
        let base = InstantIdMemoryIdentity {
            route: InstantIdRoute::Identity,
            adapter_count: 0,
            use_pid: false,
            face_restore: false,
            artifact_fingerprint: "a".into(),
        };
        let mut variants = vec![];
        let mut value = base.clone();
        value.route = InstantIdRoute::Angle;
        variants.push(value);
        let mut value = base.clone();
        value.adapter_count = 1;
        variants.push(value);
        let mut value = base.clone();
        value.use_pid = true;
        variants.push(value);
        let mut value = base.clone();
        value.face_restore = true;
        variants.push(value);
        let mut value = base.clone();
        value.artifact_fingerprint = "b".into();
        variants.push(value);
        assert!(variants
            .iter()
            .all(|variant| variant.overlay_key() != base.overlay_key()));
    }

    #[test]
    fn exact_route_context_accepts_and_crossed_evidence_fails_closed() {
        let contract = provider_contract();
        let identity = InstantIdMemoryIdentity {
            route: InstantIdRoute::Pose,
            adapter_count: 2,
            use_pid: true,
            face_restore: true,
            artifact_fingerprint: "artifacts-a".into(),
        };
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            resolved_numeric_tier(),
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("character_image".into()),
                reference_count: 1,
                use_pid: true,
                has_phases: true,
                overlay: Some(identity.overlay_key()),
            },
        )
        .unwrap();
        context.evidence_revision = REQUEST_EVIDENCE_REVISION.into();
        assert_eq!(
            safety_check(&contract, &identity, &context),
            MemorySafetyDecision::Accept
        );
        context.evidence_revision = "borrowed-sdxl-evidence".into();
        assert!(matches!(
            safety_check(&contract, &identity, &context),
            MemorySafetyDecision::Reject { .. }
        ));
        context.evidence_revision = REQUEST_EVIDENCE_REVISION.into();
        context.overlay = Some("instantid-v1/identity/a2/p1/r1/artifacts-a".into());
        assert!(matches!(
            safety_check(&contract, &identity, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    /// Executable memory semantics and the request/evidence schema are independent axes. Sharing one
    /// literal made a bump to either silently restate the other.
    #[test]
    fn calibration_identity_is_not_the_request_evidence_revision() {
        let fingerprint = provider_contract().calibration.unwrap().fingerprint;
        assert_ne!(fingerprint, REQUEST_EVIDENCE_REVISION);
        assert!(!fingerprint.is_empty());
    }

    /// Write a synthetic composition whose components have deliberately distinct element counts, so
    /// a swapped assignment cannot pass. Returns the paths plus the exact per-field byte totals
    /// derived from the tensors actually written.
    pub(crate) fn priced_paths(temp: &tempfile::TempDir) -> (InstantIdPaths, u64, u64, u64, u64) {
        use candle_gen::candle_core::{DType, Device, Tensor};
        use std::collections::HashMap;

        let root = temp.path().join("instantid_priced");
        let write = |relative: &str, rows: usize, columns: usize| -> u64 {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut tensors = HashMap::new();
            tensors.insert(
                "x.weight".to_string(),
                Tensor::zeros((rows, columns), DType::F32, &Device::Cpu).unwrap(),
            );
            candle_gen::candle_core::safetensors::save(&tensors, &path).unwrap();
            // Written F32, materialized fp16: two bytes per element, derived from the shape written
            // here rather than pinned to a literal.
            (rows as u64) * (columns as u64) * FLOAT_WIDTH
        };
        let conditioning = write("sdxl/text_encoder/model.safetensors", 16, 8)
            + write("sdxl/text_encoder_2/model.safetensors", 12, 8);
        let transformer = write("sdxl/unet/diffusion_pytorch_model.safetensors", 64, 32);
        let decoder = write("vae/diffusion_pytorch_model.safetensors", 4, 2);
        let overlay = write("identitynet/diffusion_pytorch_model.safetensors", 24, 16)
            + write("ip-adapter.safetensors", 6, 4);
        let paths = InstantIdPaths {
            sdxl_base: root.join("sdxl"),
            identitynet: WeightsSource::Dir(root.join("identitynet")),
            ip_adapter: root.join("ip-adapter.safetensors"),
            adapters: Vec::new(),
            sdxl: crate::model::SdxlComponents::for_test(
                WeightsSource::Dir(root.join("sdxl")),
                WeightsSource::Dir(root.join("sdxl")),
                WeightsSource::File(root.join("vae/diffusion_pytorch_model.safetensors")),
            ),
            openpose: None,
            face_dir: None,
        };
        (paths, conditioning, transformer, decoder, overlay)
    }

    /// The two auxiliary networks, written as a synthetic tree beside a [`priced_paths`] fixture.
    /// Returns the OpenPose source, the face dir, and the exact bytes each must add to the overlay.
    pub(crate) fn priced_extras(temp: &tempfile::TempDir) -> (WeightsSource, PathBuf, u64, u64) {
        use candle_gen::candle_core::{DType, Device, Tensor};
        use std::collections::HashMap;

        let extra = temp.path().join("instantid_extras");
        let write = |path: PathBuf, rows: usize, columns: usize| {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let tensors = HashMap::from([(
                "x.weight".to_owned(),
                Tensor::zeros((rows, columns), DType::F32, &Device::Cpu).unwrap(),
            )]);
            candle_gen::candle_core::safetensors::save(&tensors, &path).unwrap();
        };
        write(
            extra.join("openpose/diffusion_pytorch_model.safetensors"),
            10,
            8,
        );
        let face = extra.join("face");
        write(face.join(candle_gen_face::ANALYSIS_STACK_FILES[0]), 5, 4);
        write(face.join(candle_gen_face::ANALYSIS_STACK_FILES[1]), 7, 4);
        // A file the shared face layout may hold but `load_on` never reads: never charged.
        write(face.join("bisenet_parsing.safetensors"), 100, 100);
        (
            WeightsSource::Dir(extra.join("openpose")),
            face,
            10 * 8 * FLOAT_WIDTH, // fp16, like every other diffusion component
            (5 * 4 + 7 * 4) * 4,  // f32, the width `candle_gen_face::load_on` coerces to
        )
    }

    /// AC (epic SC-22657, E1): the OpenPose ControlNet and the face-analysis stack are networks the
    /// served generator holds resident on the priced route, so attaching them must move
    /// `overlay_bytes` STRICTLY up — by the OpenPose ControlNet at the fp16 compute width and by the
    /// SCRFD + ArcFace pair at the f32 width `candle_gen_face::load_on` coerces them to.
    ///
    /// Before this they had no field on `InstantIdPaths` at all, so the pose route was admitted with
    /// two whole networks charged nowhere.
    #[test]
    fn openpose_and_the_face_stack_are_priced_into_the_overlay() {
        let temp = tempfile::tempdir().unwrap();
        let (bare, _, _, _, overlay) = priced_paths(&temp);
        let (openpose, face, openpose_bytes, face_bytes) = priced_extras(&temp);

        // Bare composition first — nothing attached, nothing extra charged.
        assert_eq!(asset_facts(&bare).unwrap().overlay_bytes, overlay);

        let mut posed = bare.clone();
        posed.openpose = Some(openpose.clone());
        posed.face_dir = Some(face.clone());
        let priced = asset_facts(&posed).unwrap();
        assert_eq!(priced.overlay_bytes, overlay + openpose_bytes + face_bytes);
        assert!(
            priced.overlay_bytes > overlay,
            "attaching two resident networks must raise the declared overlay"
        );
        // The base fields describe the SDXL trunk only; overlays never leak into them.
        assert_eq!(priced.base_bytes, asset_facts(&bare).unwrap().base_bytes);

        // Each axis moves the price on its own, so one covering for the other cannot pass.
        let mut only_pose = bare.clone();
        only_pose.openpose = Some(openpose);
        assert_eq!(
            asset_facts(&only_pose).unwrap().overlay_bytes,
            overlay + openpose_bytes
        );
        let mut only_face = bare.clone();
        only_face.face_dir = Some(face);
        assert_eq!(
            asset_facts(&only_face).unwrap().overlay_bytes,
            overlay + face_bytes
        );
    }

    /// AC (epic SC-22657, E2): a resolved InstantID composition publishes the SDXL UNet + VAE axes
    /// it actually loads, declines the four a UNet denoiser structurally lacks, and the
    /// weights-free contract publishes none.
    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        let temp = tempfile::tempdir().unwrap();
        let (paths, ..) = priced_paths(&temp);
        let contract = provider_contract_for_paths(&paths).unwrap();
        assert_eq!(
            contract.architecture_facts,
            gen_core::MemoryArchitectureFacts {
                // `sdxl_unet_config()` heads are per stage (5/10/20): no uniform head count exists.
                attention_heads: None,
                // Every stage's `out_channels / attention_head_dim` is 64 (320/5, 640/10, 1280/20).
                head_dim: Some(64),
                // A UNet down/mid/up trunk is not a uniform transformer-block stack.
                transformer_blocks: None,
                // The UNet consumes the latent grid directly; nothing is patchified.
                patch_size: None,
                // `sdxl_vae_config().latent_channels` (the staged `sdxl-vae-fp16-fix`).
                latent_channels: Some(4),
                // `block_out_channels` `[128,256,512,512]` = 4 stages => 3 halvings => x8.
                vae_spatial_scale: Some(8),
                // The SDXL `AutoencoderKL` is an image VAE: no temporal axis exists to declare.
                vae_temporal_scale: None,
                // `model::DTYPE` is fp16, the width InstantID materializes at.
                activation_dtype_width: Some(2),
            }
        );
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);

        // The weights-free contract has resolved no snapshot, so no axis is knowable there.
        assert!(provider_contract().architecture_facts.is_empty());
        let mut unresolved = paths.clone();
        unresolved.sdxl_base = "/__sceneworks_memory_contract_surface__".into();
        assert!(architecture_facts(&unresolved).is_empty());
    }

    #[test]
    fn the_contract_prices_its_declared_asset_and_overlay_bytes_from_disk() {
        let temp = tempfile::tempdir().unwrap();
        let (paths, conditioning, transformer, decoder, overlay) = priced_paths(&temp);
        let contract = provider_contract_for_paths(&paths).unwrap();

        assert!(contract.formula.uses(MemoryFormulaVariable::AssetBytes));
        assert!(contract.formula.uses(MemoryFormulaVariable::OverlayBytes));
        assert_eq!(contract.asset_facts.conditioning_bytes, conditioning);
        assert_eq!(contract.asset_facts.transformer_bytes, transformer);
        assert_eq!(contract.asset_facts.decoder_bytes, decoder);
        assert_eq!(contract.asset_facts.overlay_bytes, overlay);
        assert_eq!(
            contract.asset_facts.base_bytes,
            conditioning + transformer + decoder
        );
        assert_ne!(
            contract.asset_facts,
            MemoryAssetFacts::default(),
            "declared AssetBytes/OverlayBytes variables must not be pinned at zero"
        );
    }
}
