//! MLX LTX-2.3 video memory-provider contract (SC-19109).
//!
//! The declaration follows the SC-18813 source survey rather than projecting the image ladder onto
//! video:
//!
//! - rung 1 is the shipped, unconditional Gemma -> AvDiT -> decode phase staging;
//! - rung 2 is the shipped budgeted LTX VAE tiler;
//! - rung 3 is missing because inference attention remains one monolithic SDPA call;
//! - rung 4 is missing because the 48-block AvDiT has no block-window materialization path.
//!
//! `MemoryStrategy::Resident` is gen-core's mandatory protocol baseline: it means "apply no new
//! request controls", not "keep every model component co-resident". LTX's historical baseline
//! already performs rung-1 phase staging, so `ResidentRequestMemory::PreserveLoadDefaults` leaves
//! that physical path unchanged while an explicit `StagedResidency` selection emits
//! `stage_residency=true`. The current contract vocabulary cannot separately say "unconditional"
//! versus "selectable"; SC-18816 owns that descriptor-level tri-state. Until then, resident
//! calibration must measure the already-staged baseline rather than synthesize a co-resident peak.
//!
//! The production factory sizes the exact component files this loader can materialize. The separate
//! weights-free factory exists only for registry conformance and deliberately injects zero asset
//! facts; production registry resolution never consults it.

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
use mlx_gen::gen_core::{
    self, AdapterKind, AdapterResidencyMode, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryRequestScope, MemoryResidentComponent, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategyPrerequisite,
    MemoryStrategySupport, Quant, ResidentRequestMemory, WeightsSource,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, Result};

use crate::config::{LtxConfig, SplitModel};
use crate::gemma::GemmaQuant;

/// Contract/execution identity introduced by SC-19109. The earlier SC-18808 capture predates this
/// shared-contract carrier and therefore cannot be reused as if its calibration semantics matched.
pub const CALIBRATION_FINGERPRINT: &str = "sc-19109-ltx-2-3-mlx-memory-ladder-v1";
const STATIC_CALIBRATION_FINGERPRINT: &str = "sc-19109-ltx-2-3-mlx-registry-behavior-v1";

pub const DECODE_OVERLAP: u32 = 64;

fn decode_tile_edges() -> Vec<u32> {
    crate::pipeline::LTX_VAE_SPATIAL_PX
        .iter()
        .map(|&edge| edge as u32)
        .collect()
}

fn checked_sum(label: &str, values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            mlx_gen::Error::Msg(format!(
                "ltx_2_3: {label} safetensors byte total overflows u64"
            ))
        })
    })
}

fn required_projected_safetensors_bytes(
    path: &std::path::Path,
    label: &str,
    projection: ResidentProjection,
) -> Result<u64> {
    let bytes = projected_safetensors_bytes(path, |_| projection).map_err(mlx_gen::Error::from)?;
    if bytes == 0 {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_3: {label} has no projected resident safetensors bytes at {}",
            path.display()
        )));
    }
    Ok(bytes)
}

struct AssetDeclaration {
    facts: MemoryAssetFacts,
    resident_components: Vec<MemoryResidentComponent>,
}

fn adapters_have_load_exact_additive_accounting(spec: &LoadSpec) -> Result<bool> {
    for adapter in &spec.adapters {
        if adapter.kind == AdapterKind::Lokr {
            return Ok(false);
        }
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&adapter.path)
            .map_err(mlx_gen::Error::from)?;
        let reconstructs_dense_delta = headers.iter().any(|tensor| {
            gen_core::weightsmeta::LOKR_TP_SUFFIXES
                .iter()
                .chain(gen_core::weightsmeta::LOHA_TP_SUFFIXES.iter())
                .any(|suffix| tensor.name.ends_with(suffix))
        });
        if reconstructs_dense_delta {
            return Ok(false);
        }
    }
    Ok(true)
}

fn production_asset_declaration(
    spec: &LoadSpec,
    gemma_dir: &std::path::Path,
) -> Result<AssetDeclaration> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::Error::Msg(
            "ltx_2_3 memory contract requires the split model directory used by the loader".into(),
        ));
    };

    // The calibrated route rejects prompt enhancement, so its conditioning phase is exactly the
    // canonical dense Gemma snapshot plus the LTX connector. Both loaders materialize every tensor
    // as bf16 irrespective of the stored payload width. A provisioned uncensored enhancer remains a
    // supported generator feature but is not materialized by this route.
    let conditioning_bytes = checked_sum(
        "conditioning",
        [
            required_projected_safetensors_bytes(
                gemma_dir,
                "Gemma text encoder",
                ResidentProjection::Bfloat16,
            )?,
            required_projected_safetensors_bytes(
                &root.join("connector.safetensors"),
                "connector",
                ResidentProjection::Bfloat16,
            )?,
        ],
    )?;

    // The upsampler runs between the two denoise passes while the AvDiT is live, so it belongs to
    // the denoise-phase base footprint rather than to decode.
    let transformer_bytes = checked_sum(
        "denoise",
        [
            required_projected_safetensors_bytes(
                &root.join("transformer.safetensors"),
                "AudioVideo transformer",
                ResidentProjection::Stored,
            )?,
            required_projected_safetensors_bytes(
                &root.join("upsampler.safetensors"),
                "latent upsampler",
                ResidentProjection::Stored,
            )?,
        ],
    )?;

    // Pure T2V never materializes vae_encoder.safetensors: the provider retains only its path and
    // opens it lazily for image-conditioned routes, which this calibrated contract rejects.
    let decoder_bytes = checked_sum(
        "decode",
        [
            required_projected_safetensors_bytes(
                &root.join("vae_decoder.safetensors"),
                "video VAE decoder",
                ResidentProjection::Float32,
            )?,
            required_projected_safetensors_bytes(
                &root.join("audio_vae.safetensors"),
                "audio VAE decoder",
                ResidentProjection::Float32,
            )?,
            required_projected_safetensors_bytes(
                &root.join("vocoder.safetensors"),
                "vocoder",
                ResidentProjection::Float32,
            )?,
        ],
    )?;

    let overlay_bytes = match gen_core::adapter_stack_resident_bytes(
        &spec.adapters,
        AdapterResidencyMode::Additive,
    ) {
        Some(bytes) => bytes,
        None => {
            return Err(mlx_gen::Error::Msg(
                "ltx_2_3: every additive adapter must have a non-zero load-exact safetensors size"
                    .into(),
            ));
        }
    };
    let resident_components = (overlay_bytes > 0)
        .then(|| MemoryResidentComponent {
            id: "adapter_stack".to_owned(),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: overlay_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        })
        .into_iter()
        .collect();

    let base_bytes = checked_sum(
        "base model",
        [conditioning_bytes, transformer_bytes, decoder_bytes],
    )?;
    Ok(AssetDeclaration {
        facts: MemoryAssetFacts {
            base_bytes,
            conditioning_bytes,
            transformer_bytes,
            decoder_bytes,
            overlay_bytes,
        },
        resident_components,
    })
}

fn quant_from_split(split: &SplitModel) -> Result<Option<Quant>> {
    if !split.quantized {
        return Ok(None);
    }
    match split.bits {
        4 => Ok(Some(Quant::Q4)),
        8 => Ok(Some(Quant::Q8)),
        bits => Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: split_model.json declares unsupported {bits}-bit transformer weights"
        ))),
    }
}

pub(crate) fn numeric_tier_from_split(
    spec: &LoadSpec,
    split: &SplitModel,
) -> Result<MemoryNumericTier> {
    let quant = quant_from_split(split)?;
    if let Some(requested) = spec.quantize {
        if quant != Some(requested) {
            return Err(mlx_gen::Error::Unsupported(format!(
                "ltx_2_3: requested {requested:?} does not match the checkpoint tier {quant:?}"
            )));
        }
    }
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    })
}

pub fn resolved_numeric_tier(spec: &LoadSpec) -> Result<MemoryNumericTier> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::Error::Msg(
            "ltx_2_3 numeric tier requires a split model directory".into(),
        ));
    };
    numeric_tier_from_split(spec, &SplitModel::from_model_dir(root)?)
}

pub(crate) fn route_overlay(spec: &LoadSpec) -> Option<String> {
    let mut axes = Vec::new();
    if !spec.adapters.is_empty() {
        axes.push("adapters");
    }
    if spec.components.contains_key("uncensored_enhancer") {
        axes.push("uncensored-enhancer");
    }
    (!axes.is_empty()).then(|| axes.join("-"))
}

fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident
                | MemoryStrategy::StagedResidency
                | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
                MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                    MemoryStrategySupport::Missing
                }
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: decode_tile_edges(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

fn build_contract(
    spec: &LoadSpec,
    asset_declaration: AssetDeclaration,
    calibration_fingerprint: &str,
) -> Result<MemoryProviderContract> {
    if spec.load_shape != LoadShape::EagerMaterialization {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3 has no deferred/block-window loader; use EagerMaterialization".into(),
        ));
    }
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::FrameCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::OverlayBytes,
        MemoryFormulaVariable::DecodeTileArea,
    ];
    let formula = if asset_declaration.resident_components.is_empty() {
        MemoryFormulaKind::PhaseEnvelope {
            phases: phases.clone(),
            variables,
        }
    } else {
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: phases.clone(),
            variables,
            resident_components: asset_declaration.resident_components,
        }
    };
    Ok(MemoryProviderContract {
        provider_id: crate::MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: strategies(),
        // LTX declares no decode-quality geometry policy table, so this route carries no semantic
        // decode authority — the fail-closed default every non-declaring provider contract uses.
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        // LTX stages Gemma before the AvDiT for every render. A selected decode rung therefore
        // co-engages rung 1 even though the shared cost-order default intentionally does not.
        additional_prerequisites: vec![(
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        )],
        default_engagement_exclusions: Vec::new(),
        // Gen-core requires Resident as the no-new-control baseline. It does not assert literal
        // co-residency: preserving defaults keeps LTX's historical always-staged phase order and
        // automatic decode guard. SC-18816 will make that unconditional staging separately visible.
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula,
        calibration: Some(MemoryCalibrationIdentity::new(
            calibration_fingerprint,
            spec.load_shape,
        )),
        asset_facts: asset_declaration.facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

pub(crate) fn contract_for_loaded(
    spec: &LoadSpec,
    split: &SplitModel,
    gemma_dir: &std::path::Path,
    gemma_quant: Option<GemmaQuant>,
) -> Result<Option<(MemoryProviderContract, MemoryNumericTier, Option<String>)>> {
    // The first production evidence campaign is calibrated against the canonical dense-bf16 Gemma
    // route. Quantized Gemma remains a supported generator input, but must fail open until it has a
    // separately identifiable and measured contract rather than borrowing the canonical evidence.
    if gemma_quant.is_some() || !adapters_have_load_exact_additive_accounting(spec)? {
        return Ok(None);
    }
    let tier = numeric_tier_from_split(spec, split)?;
    let contract = build_contract(
        spec,
        production_asset_declaration(spec, gemma_dir)?,
        CALIBRATION_FINGERPRINT,
    )?;
    Ok(Some((contract, tier, route_overlay(spec))))
}

pub fn memory_strategy_contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::Error::Msg(
            "ltx_2_3 memory contract requires a split model directory".into(),
        ));
    };
    let split = SplitModel::from_model_dir(root)?;
    let _tier = numeric_tier_from_split(spec, &split)?;
    let gemma_dir = crate::model::resolve_gemma_dir(spec.text_encoder.as_ref())?;
    if let Some(quant) = crate::model::resolve_gemma_quant(&gemma_dir)? {
        return Err(mlx_gen::Error::Unsupported(format!(
            "ltx_2_3: the calibrated memory contract requires canonical dense-bf16 Gemma; \
             the requested Gemma snapshot declares {}-bit group-{} quantization",
            quant.bits, quant.group
        )));
    }
    if !adapters_have_load_exact_additive_accounting(spec)? {
        return Err(mlx_gen::Error::Unsupported(
            "ltx_2_3: calibrated memory admission supports additive LoRA factors but not \
             LoKr/LoHa routes that reconstruct dense bf16 deltas"
                .into(),
        ));
    }
    build_contract(
        spec,
        production_asset_declaration(spec, &gemma_dir)?,
        CALIBRATION_FINGERPRINT,
    )
}

pub(crate) fn weights_free_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    build_contract(
        spec,
        AssetDeclaration {
            facts: MemoryAssetFacts::default(),
            resident_components: Vec::new(),
        },
        STATIC_CALIBRATION_FINGERPRINT,
    )
    .map_err(Into::into)
}

/// LTX witnesses the shared MLX tiers under both offload policies, but only the eager half of the
/// materialization axis. `build_contract` rejects `DeferredMaterialization` outright — LTX has no
/// deferred/block-window loader — so publishing the deferred selectors would advertise a load
/// surface no contract can be built for, and the registry conformance walk (which constructs every
/// published selector) would fail the whole MLX catalog. The witness set is the provider's own
/// finite inventory, not the shared default.
pub(crate) fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| surface.selector.load_shape == LoadShape::EagerMaterialization)
        .collect()
}

fn fixture_contract(contract: &MemoryProviderContract) -> bool {
    contract
        .calibration
        .as_ref()
        .is_some_and(|identity| identity.fingerprint == STATIC_CALIBRATION_FINGERPRINT)
}

fn registered_tier(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> Result<MemoryNumericTier> {
    if fixture_contract(contract) {
        return Ok(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        });
    }
    resolved_numeric_tier(spec)
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    let edge = edge.ok_or_else(|| {
        gen_core::Error::Unsupported("ltx_2_3: bounded decode requires a tile edge".into())
    })?;
    let overlap = overlap.ok_or_else(|| {
        gen_core::Error::Unsupported("ltx_2_3: bounded decode requires a tile overlap".into())
    })?;
    let edges = decode_tile_edges();
    if !edges.contains(&edge) {
        return Err(gen_core::Error::Unsupported(format!(
            "ltx_2_3: decode tile edge {edge} is outside the production domain {edges:?}"
        )));
    }
    if overlap != DECODE_OVERLAP {
        return Err(gen_core::Error::Unsupported(format!(
            "ltx_2_3: decode overlap {overlap} is not the production overlap {DECODE_OVERLAP}"
        )));
    }
    Ok(())
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.mode.as_key() != "text_to_video"
            || context.geometry.reference_count != 0
            || context.has_reference
            || context.use_pid
        {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: calibrated memory route is reference-free text_to_video without PiD"
                    .into(),
            ));
        }
        if context.overlay.as_deref() != expected_overlay {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_3: memory overlay {:?} does not match the loaded route {:?}",
                context.overlay, expected_overlay
            )));
        }
        let geometry = context.geometry;
        if geometry.batch != 1
            || !(crate::model::MIN_SIZE..=crate::model::MAX_SIZE).contains(&geometry.width)
            || !(crate::model::MIN_SIZE..=crate::model::MAX_SIZE).contains(&geometry.height)
            || !geometry.width.is_multiple_of(crate::SIZE_MULTIPLE)
            || !geometry.height.is_multiple_of(crate::SIZE_MULTIPLE)
            || !(1..=crate::model::MAX_FRAMES).contains(&geometry.frames)
            || !(geometry.frames - 1).is_multiple_of(8)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_3: unsupported memory geometry {}x{}x{} frames={}",
                geometry.width, geometry.height, geometry.batch, geometry.frames
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            validate_decode(
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap,
            )?;
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier),
        Some(&route_gate),
    )
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match registered_tier(spec, contract) {
        Ok(tier) => safety_check(contract, tier, route_overlay(spec).as_deref(), context),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn registered_valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let tier = registered_tier(spec, contract).map_err(gen_core::Error::from)?;
    let context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("text_to_video".into()),
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: route_overlay(spec),
        },
    )?;
    Ok(vec![MemoryBehaviorFixture::new(context)])
}

fn begin_with_cleanup(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    transformer_blocks: usize,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, loaded_tier, expected_overlay, context)
    {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        crate::MODEL_ID,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        transformer_blocks,
        |_use_pid, edge, overlap| validate_decode(Some(edge), Some(overlap)),
    )?;
    Ok(Some(Box::new(LtxMemoryRequestScope {
        inner: mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    })))
}

/// LTX's calibrated evidence is deliberately narrower than the generator's complete capability
/// surface. Bind the admitted context to the actual request before the shared core installs any
/// selected controls, including axes that [`GenerationRequest::image_reference_count`] does not
/// represent (temporal conditioning and prompt enhancement).
struct LtxMemoryRequestScope {
    inner: mlx_gen::request_scope::MlxRequestScopeCore,
}

impl LtxMemoryRequestScope {
    fn validate_request(request: &GenerationRequest) -> gen_core::Result<()> {
        if !request.conditioning.is_empty() {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: calibrated memory route requires empty conditioning".into(),
            ));
        }
        if request.enhance_prompt || request.use_uncensored_enhancer {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: calibrated memory route does not include prompt enhancement controls"
                    .into(),
            ));
        }
        if request.video_mode.is_some() {
            return Err(gen_core::Error::Unsupported(
                "ltx_2_3: calibrated memory route does not include video_mode variants".into(),
            ));
        }
        let fps = request.fps.unwrap_or(24);
        if !(24..=30).contains(&fps) {
            return Err(gen_core::Error::Unsupported(format!(
                "ltx_2_3: calibrated memory route requires fps in 24..=30, got {fps}"
            )));
        }
        Ok(())
    }
}

impl MemoryRequestScope for LtxMemoryRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        Self::validate_request(request)?;
        self.inner.configure_request(request)
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.enter_phase(phase)
    }

    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.leave_phase(phase)
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.inner.configure_decode(tile_edge, overlap, geometry)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.inner.configure_attention(chunk_size)
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> gen_core::Result<()> {
        self.inner
            .materialize_transformer_window(first_block, block_count)
    }

    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.inner.finish(outcome)
    }
}

pub(crate) fn begin_request(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    expected_overlay: Option<&str>,
    transformer_blocks: usize,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_with_cleanup(
        contract,
        loaded_tier,
        expected_overlay,
        transformer_blocks,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let tier = registered_tier(spec, contract).map_err(gen_core::Error::from)?;
    begin_with_cleanup(
        contract,
        tier,
        route_overlay(spec).as_deref(),
        LtxConfig::video_only_defaults().num_layers as usize,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn decode_tiling(request: &GenerationRequest) -> Result<Option<TilingConfig>> {
    let Some(memory) = request.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    let edge = memory.decode_tile_edge.ok_or_else(|| {
        mlx_gen::Error::Unsupported(
            "ltx_2_3: selected bounded decode is missing decode_tile_edge".into(),
        )
    })?;
    let overlap = memory.decode_overlap.ok_or_else(|| {
        mlx_gen::Error::Unsupported(
            "ltx_2_3: selected bounded decode is missing decode_overlap".into(),
        )
    })?;
    validate_decode(Some(edge), Some(overlap)).map_err(mlx_gen::Error::from)?;
    crate::pipeline::selected_tiling_budgeted_ltx(
        request.height as i32,
        request.width as i32,
        request.frames.unwrap_or(1) as i32,
        edge as i32,
        overlap as i32,
    )
    .map(Some)
}

pub(crate) const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: crate::MODEL_ID,
    contract: |spec| memory_strategy_contract(spec).map_err(Into::into),
    safety_check: registered_safety_check,
};

pub(crate) const MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        Conditioning, GenerationMemory, Image, MemoryStrategyParameters, Precision, ReplacementMode,
    };

    fn fixture_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-fixture".into())).with_quant(Quant::Q8)
    }

    /// `ProviderRegistry::memory_contract_surfaces` constructs a contract for **every** selector the
    /// fixture publishes and fails the entire MLX catalog when one errors, so the published witness
    /// set must be exactly the set this provider can build. Asserting it here localizes the failure
    /// to LTX instead of surfacing it as eight red `mlx-gen-catalog` tests.
    #[test]
    fn every_published_contract_surface_builds_and_no_deferred_surface_is_published() {
        let surfaces = memory_contract_surface_specs();
        assert_eq!(
            surfaces.len(),
            gen_core::mlx_memory_contract_surface_specs().len() / 2,
            "the witness set is the eager half of the shared MLX surface"
        );
        for surface in &surfaces {
            assert_eq!(
                surface.selector.load_shape,
                LoadShape::EagerMaterialization,
                "{} has no deferred/block-window loader",
                surface.selector.id()
            );
            weights_free_memory_strategy_contract(&surface.spec).unwrap_or_else(|error| {
                panic!("surface {} must build: {error}", surface.selector.id())
            });
        }
        assert!(
            gen_core::mlx_memory_contract_surface_specs()
                .into_iter()
                .filter(|surface| surface.selector.load_shape == LoadShape::DeferredMaterialization)
                .all(|surface| weights_free_memory_strategy_contract(&surface.spec).is_err()),
            "a deferred surface that now builds must be published, not filtered out"
        );
    }

    #[test]
    fn survey_rungs_and_video_formula_are_declared_without_a_default_standin() {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        assert!(matches!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        ));
        assert!(matches!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        ));
        for missing in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert!(matches!(
                contract.capability(missing).unwrap().support,
                MemoryStrategySupport::Missing
            ));
        }
        assert!(contract.formula.uses(MemoryFormulaVariable::FrameCount));
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedDecode),
            [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
            ]
        );

        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q8),
            component_precision_floors: &[],
        };
        let resident = contract
            .representative_selection(MemoryStrategy::Resident, tier, false)
            .unwrap();
        let staged = contract
            .representative_selection(MemoryStrategy::StagedResidency, tier, false)
            .unwrap();
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::Resident),
            [MemoryStrategy::Resident]
        );
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::StagedResidency),
            [MemoryStrategy::Resident, MemoryStrategy::StagedResidency]
        );
        assert_eq!(contract.generation_memory(&resident), None);
        assert_eq!(
            contract.generation_memory(&staged),
            Some(GenerationMemory {
                stage_residency: true,
                ..Default::default()
            })
        );
    }

    #[test]
    fn selected_decode_parameters_reach_the_real_tiling_carrier_and_mutations_fail() {
        let mut request = GenerationRequest {
            // Keep the carrier assertion below the smallest supported CI host's live MLX budget.
            // Long-clip temporal-budget behavior is covered by the injected-budget pipeline test;
            // this test owns only request-memory propagation and domain validation.
            width: 256,
            height: 256,
            frames: Some(25),
            count: 1,
            memory: Some(GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                decode_tile_edge: Some(256),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }),
            ..Default::default()
        };
        let tiling = decode_tiling(&request).unwrap().unwrap();
        let spatial = tiling.spatial.unwrap();
        assert_eq!((spatial.tile_px, spatial.overlap_px), (256, 64));

        request.memory.as_mut().unwrap().decode_tile_edge = Some(255);
        assert!(decode_tiling(&request).is_err());
        request.memory.as_mut().unwrap().decode_tile_edge = None;
        assert!(decode_tiling(&request).is_err());

        let contract = weights_free_memory_strategy_contract(&fixture_spec()).unwrap();
        let mut selection = contract
            .representative_selection(
                MemoryStrategy::BoundedDecode,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q8),
                    component_precision_floors: &[],
                },
                false,
            )
            .unwrap();
        selection.parameters = MemoryStrategyParameters {
            decode_tile_edge: Some(511),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        };
        assert!(contract.validate_selection(&selection).is_err());
    }

    fn calibrated_t2v_scope_and_request() -> (Box<dyn MemoryRequestScope>, GenerationRequest) {
        let spec = fixture_spec();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        let fixture = registered_valid_fixtures(&spec, &contract, MemoryStrategy::StagedResidency)
            .unwrap()
            .pop()
            .unwrap();
        let scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        (scope, fixture.request)
    }

    #[test]
    fn calibrated_scope_binds_empty_t2v_request_and_fps_envelope_before_installing_controls() {
        let (mut scope, request) = calibrated_t2v_scope_and_request();
        for fps in [None, Some(24), Some(25), Some(30)] {
            let mut accepted = GenerationRequest {
                fps,
                ..request.clone()
            };
            scope.configure_request(&mut accepted).unwrap();
            assert!(accepted.memory.unwrap().stage_residency);
        }

        let image = Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        };
        for conditioning in [
            Conditioning::Keyframe {
                image: image.clone(),
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::VideoClip {
                frames: vec![image.clone()],
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::ControlClip {
                frames: vec![image.clone()],
                mask: vec![image],
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::FaceOnly,
            },
        ] {
            let mut temporal_conditioning = GenerationRequest {
                conditioning: vec![conditioning],
                ..request.clone()
            };
            assert_eq!(temporal_conditioning.image_reference_count(), 0);
            let error = scope
                .configure_request(&mut temporal_conditioning)
                .unwrap_err()
                .to_string();
            assert!(error.contains("requires empty conditioning"));
            assert_eq!(temporal_conditioning.memory, None);
        }

        let mut enhanced = GenerationRequest {
            enhance_prompt: true,
            ..request.clone()
        };
        let error = scope
            .configure_request(&mut enhanced)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not include prompt enhancement controls"));
        assert_eq!(enhanced.memory, None);

        let mut uncensored = GenerationRequest {
            use_uncensored_enhancer: true,
            ..request.clone()
        };
        let error = scope
            .configure_request(&mut uncensored)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not include prompt enhancement controls"));
        assert_eq!(uncensored.memory, None);

        for video_mode in ["no_audio", "video_only"] {
            let mut variant = GenerationRequest {
                video_mode: Some(video_mode.into()),
                ..request.clone()
            };
            let error = scope
                .configure_request(&mut variant)
                .unwrap_err()
                .to_string();
            assert!(error.contains("does not include video_mode variants"));
            assert_eq!(variant.memory, None);
        }

        for fps in [Some(0), Some(23), Some(31)] {
            let mut outside_envelope = GenerationRequest {
                fps,
                ..request.clone()
            };
            let error = scope
                .configure_request(&mut outside_envelope)
                .unwrap_err()
                .to_string();
            assert!(error.contains("requires fps in 24..=30"));
            assert_eq!(outside_envelope.memory, None);
        }

        // Provider-route rejection happens before the shared carrier mutates the request, and it
        // leaves the inner scope active so the caller can report the terminal error and run cleanup.
        scope
            .finish(MemoryRunOutcome::Error {
                message: "request route rejected".into(),
            })
            .unwrap();
        let mut after_finish = request;
        assert!(scope.configure_request(&mut after_finish).is_err());
    }
}
