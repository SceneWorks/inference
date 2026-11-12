//! The **`minimax_h3_image`** memory contract on candle/CUDA (SW-7091, ladder SW-7095).
//!
//! | rung | strategy | status |
//! |---|---|---|
//! | 0 | [`MemoryStrategy::Resident`] | **implemented** as the baseline — the render never exceeds it: the generator always stages (as H3's does), so a resident admission over-provisions rather than under-provisions |
//! | 1 | [`MemoryStrategy::StagedResidency`] | **implemented** — `crate::image_model` runs the vision tower (edit), the conditioner, then the reference VAE encode (edit), dropping and synchronizing each; maps the DiT, evicts `adaln_proj` after the precompute, denoises, drops it, then maps the VAE |
//! | 2 | [`MemoryStrategy::BoundedDecode`] | **implemented** — the 2-D VAE's shipped 256 / 64 tiled decode ([`crate::image_vae::bounded_decode_parameters`]) |
//! | 3 | [`MemoryStrategy::BoundedAttention`] | **implemented** — H3's attention runs FlashAttention by default with a query-chunked materializing fallback; the selected chunk bounds that fallback (and the text encoder's masked attention and FFN chunks) through the request scope's active context ([`crate::nn::attention_budget`]) |
//! | 4 | [`MemoryStrategy::BoundedTransformerResidency`] | **implemented on a deferred load of a supported source** — the text encoder and the DiT stay on the host in runtime format and run through a one-block device window, with the DiT's root modules (projections, refiner, the six `pdd_proj_out` heads) on the device and each block's LoRA factors travelling with it (`crate::block_stream`) |
//!
//! Admission prices every request phase by phase, host and GPU separately ([`admission`]): a rung
//! whose GPU estimate does not fit the request's budget is refused, so the ladder moves on to the
//! streamed rung instead of rejecting the request at the resident peak. The figures are
//! header-derived **estimates**; no calibration is claimed (E13).

use std::path::Path;

use candle_gen::gen_core::{
    self, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryFixturePhases, MemoryFixtureRoute, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryProfileFixtures, MemoryProfileGates,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, MemoryWindowMaterialization, ProviderMemoryProfile,
    ResidentRequestMemory, WeightsSource,
};

use crate::image_model::IMAGE_MODEL_ID;
use crate::image_vae::{IMAGE_DECODE_OVERLAP, IMAGE_DECODE_TILE_EDGE, IMAGE_VAE_COMPONENT};

pub mod admission;

/// Content fingerprint of this contract: the four-rung ladder priced by header-derived estimates.
pub const IMAGE_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "minimax-h3-image-candle-estimated-ladder-2026-11-10-v2";

/// The bounded-attention domain, in score elements per materializing query chunk. Both points are
/// at or under the CUDA default (32 Mi), and the smaller still holds one query row of the longest
/// sequence the admission geometry produces (a 20 MP canvas with 16 references priced at 2048²,
/// ~110k keys × 56 heads ≈ 6.2 Mi), so neither point refuses a legal request.
pub const IMAGE_ATTENTION_CHUNK_SIZES: [u32; 2] = [8 << 20, 32 << 20];

/// Rung 4's window domain: one block (one encoder layer) at a time — `crate::block_stream`'s
/// window over this same block stack and encoder, and the MLX twin's value.
pub const IMAGE_TRANSFORMER_WINDOW_SIZE: u32 = 1;

/// Width the conditioner is held at (bf16). The DiT is loaded at its stored widths (bf16 block
/// store, float32 top-level heads) and priced there.
const BF16_WIDTH: u64 = 2;

/// Width the VAE decode half is held at: `crate::image_model` loads it with `DType::F32`, which
/// casts every float tensor to float32 whatever its stored dtype.
///
/// The route wires no `LoadSpec::precision` override — the generator hardcodes the DiT's
/// `DType::BF16` and the VAE's `DType::F32` — so an `Fp32` spec is refused by [`contract_for`]
/// and by the loader ([`crate::image_model::refuse_precision_override`]) rather than priced and
/// rendered as a `Bf16` one.
const VAE_F32_WIDTH: u64 = 4;

/// Bytes the resolved tensors of one component occupy once loaded. Float tensors price at
/// `float_width` (or their stored width when `None`); `keep` selects the tensors the loader opens.
fn component_bytes(
    path: &Path,
    float_width: Option<u64>,
    keep: &dyn Fn(&str) -> bool,
) -> gen_core::Result<u64> {
    gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .filter(|header| keep(&header.name))
        .try_fold(0_u64, |sum, header| {
            let bytes = match float_width {
                Some(width) if header.is_float() => header.materialized_bytes(width)?,
                _ => header.data_bytes,
            };
            sum.checked_add(bytes).ok_or_else(|| {
                gen_core::Error::Msg(format!("{IMAGE_MODEL_ID}: component byte sum overflow"))
            })
        })
}

/// The load-exact inventory: the conditioner's layer-50 tap (the `lm_prefixes` the render maps)
/// at bf16, the DiT at its stored mixed bf16/f32 widths, and the VAE's decode half at the f32 it
/// is cast to.
fn asset_facts(dit_dir: &Path, root: &Path, te_dir: &Path) -> gen_core::Result<MemoryAssetFacts> {
    let te_cfg = crate::text_encoder::MiniMaxH3TeConfig::from_component_dir(te_dir)
        .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
    let prefixes = crate::text_encoder::lm_prefixes(crate::text_encoder::LM_PREFIX, &te_cfg);
    let conditioning = component_bytes(te_dir, Some(BF16_WIDTH), &|name: &str| {
        prefixes.iter().any(|p| name.starts_with(p.as_str()))
    })?;
    let transformer = component_bytes(dit_dir, None, &|_| true)?;
    let decoder = component_bytes(
        &root.join(IMAGE_VAE_COMPONENT),
        Some(VAE_F32_WIDTH),
        &|name: &str| name.starts_with("decoder.") || name.starts_with("post_quant_conv."),
    )?;
    Ok(MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: 0,
    })
}

/// The installed LoRA stack's resident bytes (factors plus prepared caches) on the resolved DiT,
/// header-only — SW-7096's adapter pricing over the image transformer's target surface.
fn overlay_bytes(
    dit_dir: &Path,
    cfg: &crate::dit::MiniMaxH3DitConfig,
    allocation: &gen_core_h3::h3_memory::NativeComponentAllocation,
    spec: &LoadSpec,
) -> gen_core::Result<u64> {
    if spec.adapters.is_empty() {
        return Ok(0);
    }
    Ok(crate::memory_strategy::adapter_profile_for(
        dit_dir,
        &crate::adapters::image_adapter_target_paths(cfg),
        &spec.adapters,
        &allocation.projections,
    )?
    .resident_bytes)
}

/// Architecture axes read from the snapshot's own `transformer_diffusers/` and `vae_diffusers/`
/// configs. Gated on a materialized root, as every Candle provider's are.
fn architecture_facts_from_snapshot(
    _provider_id: &str,
    root: &Path,
    spec: &LoadSpec,
) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    // The staged tier's directory when one is staged, else the snapshot's own.
    let dit = af::config_file(&crate::image_model::resolve_dit_dir(root, spec).join("config.json"));
    let vae = af::component_config(root, IMAGE_VAE_COMPONENT);
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::axis_of(dit.as_ref(), &["num_attention_heads"]),
        head_dim: af::axis_of(dit.as_ref(), &["attention_head_dim"]),
        transformer_blocks: af::axis_of(dit.as_ref(), &["num_layers"]),
        patch_size: af::axis_of(dit.as_ref(), &["patch_size"]),
        latent_channels: af::axis_of(dit.as_ref(), &["in_channels"]),
        vae_spatial_scale: vae.as_ref().map(|_| crate::config::VAE_RATIO as u32),
        // A still image: one latent frame, no temporal compression on this route.
        vae_temporal_scale: None,
        activation_dtype_width: u32::try_from(BF16_WIDTH).ok(),
    }
}

/// The rung capabilities. Rung 4 is implemented only when `streamable` (a deferred load of a
/// source whose blocks migrate byte-for-byte); every other rung always is.
fn strategies(streamable: bool) -> Vec<gen_core::MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| {
            let mut parameters = MemoryParameterRanges::default();
            let support = match strategy {
                MemoryStrategy::BoundedTransformerResidency if !streamable => {
                    MemoryStrategySupport::Missing
                }
                _ => MemoryStrategySupport::Implemented,
            };
            match strategy {
                MemoryStrategy::BoundedDecode => {
                    parameters = crate::image_vae::bounded_decode_parameters();
                }
                MemoryStrategy::BoundedAttention => {
                    parameters.attention_chunk_sizes = IMAGE_ATTENTION_CHUNK_SIZES.to_vec();
                }
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    // `crate::block_stream` runs a one-block window over BOTH the text encoder
                    // and the DiT.
                    parameters.transformer_window_sizes = vec![IMAGE_TRANSFORMER_WINDOW_SIZE];
                    parameters.transformer_window_components =
                        vec![gen_core::TransformerComponent::Both];
                }
                _ => {}
            }
            gen_core::MemoryStrategyCapability {
                strategy,
                support,
                parameters,
            }
        })
        .collect()
}

/// The weights-free contract — the registry conformance surface. A deferred-materialization spec
/// declares rung 4: the fixture has no source to inspect, as H3 video's does not.
pub fn weights_free_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    build_contract(
        spec,
        spec.load_shape == LoadShape::DeferredMaterialization,
        0,
    )
}

fn build_contract(
    spec: &LoadSpec,
    streamable: bool,
    overlay: u64,
) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = MemoryProviderContract::compatibility_default(
        IMAGE_MODEL_ID,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: streamable,
            host_to_device_block_materialization: streamable,
            // A window copies the host's runtime-format block bytes; it never repacks.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.architecture_facts =
        gen_core::standard_architecture_facts(&MEMORY_PROFILE, IMAGE_MODEL_ID, spec);
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        IMAGE_MEMORY_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ));
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: true,
        transformer_window_materialization: streamable,
    };
    let mut variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
    ];
    if streamable {
        variables.push(MemoryFormulaVariable::TransformerWindowSize);
    }
    // The LoRA stack as a typed auxiliary component, declared only when one is configured so an
    // ordinary load's formula keeps its shape (H3 video's spelling, sc-18650).
    let mut resident_components = Vec::new();
    if overlay > 0 {
        variables.push(MemoryFormulaVariable::OverlayBytes);
        resident_components.push(gen_core::MemoryResidentComponent {
            id: crate::memory_strategy::ADAPTER_STACK_COMPONENT_ID.to_owned(),
            kind: gen_core::MemoryComponentKind::AdapterStack,
            resident_bytes: overlay,
            bounded_by: None,
            residency: gen_core::MemoryComponentResidency::WholeRender,
        });
    }
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables,
        resident_components,
    };
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;
    contract.strategies = strategies(streamable);
    // Staging is structural (three successive components). The streamable DiT's trunk and per-block
    // bytes are added by `contract_for`, which reads them; the weights-free surface knows none.
    contract.phase_facts = Some(gen_core::MemoryPhaseFacts::staged(
        gen_core::StagedWeightSchedule::ThreeStage,
    ));
    Ok(contract)
}

/// The executable contract for a real load: the declaration for this load shape and source, plus
/// the on-disk inventory and the configured LoRA stack.
pub fn contract_for(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    crate::image_model::refuse_precision_override(spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Msg(format!(
            "{IMAGE_MODEL_ID}: memory facts require a snapshot directory"
        )));
    };
    let te_dir = crate::image_model::resolve_text_encoder_dir(root, spec);
    let dit_dir = crate::image_model::resolve_dit_dir(root, spec);
    let streamable = spec.load_shape == LoadShape::DeferredMaterialization
        && crate::image_model::image_streaming_supported(spec)
            .map_err(|e| gen_core::Error::Unsupported(e.to_string()))?;
    let cfg = crate::image_model::image_dit_config_at(&dit_dir)
        .map_err(|e| gen_core::Error::Unsupported(e.to_string()))?;
    let allocation = crate::memory_strategy::dit_allocation(&dit_dir, false)?;
    let overlay = overlay_bytes(&dit_dir, &cfg, &allocation, spec)?;
    let mut contract = build_contract(spec, streamable, overlay)?;
    contract.asset_facts = asset_facts(&dit_dir, root, &te_dir)?;
    contract.asset_facts.overlay_bytes = overlay;
    if allocation.resolved {
        if let Some(phase_facts) = &mut contract.phase_facts {
            phase_facts.transformer_stream = Some(admission::transformer_stream(
                &allocation.resident_by_name,
                cfg.num_layers,
            )?);
        }
    }
    Ok(contract)
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    gen_core::validate_bounded_decode_params(
        IMAGE_MODEL_ID,
        &[IMAGE_DECODE_TILE_EDGE],
        IMAGE_DECODE_OVERLAP,
        edge,
        overlap,
    )
}

const MEMORY_PROFILE: ProviderMemoryProfile = ProviderMemoryProfile {
    gates: MemoryProfileGates::within(route_gate),
    fixtures: MemoryProfileFixtures::routes(&[
        MemoryFixtureRoute::TEXT_TO_IMAGE.with_phases(MemoryFixturePhases::Always)
    ])
    .implemented_only(),
    architecture: Some(gen_core::MemoryArchitectureSource::Snapshot(
        architecture_facts_from_snapshot,
    )),
    ..ProviderMemoryProfile::STANDARD
};

fn route_gate(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _loaded_tier: MemoryNumericTier,
) -> gen_core::Result<()> {
    let id = contract.provider_id.as_str();
    let references = context.geometry.reference_count;
    let max_references = gen_core::h3_image::MAX_REFERENCES as u32;
    match context.mode {
        MemoryMode::TextToImage if references == 0 => {}
        // SW-7092's edit: 1..=16 references, each through the vision tower and the VAE encode.
        MemoryMode::Edit if (1..=max_references).contains(&references) => {}
        _ => {
            return Err(gen_core::Error::Unsupported(format!(
                "{id}: the candle route is text-to-image, or an edit of 1..={max_references} \
                 references; got {:?} with {references}",
                context.mode
            )));
        }
    }
    let g = context.geometry;
    if g.width == 0 || g.height == 0 || g.batch != 1 || g.frames > 1 {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: one still image of positive size per request; got {}x{} x{} frames={}",
            g.width, g.height, g.batch, g.frames
        )));
    }
    if context.use_pid {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: there is no PiD decoder on this route"
        )));
    }
    if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
        validate_decode(
            context.selection.parameters.decode_tile_edge,
            context.selection.parameters.decode_overlap,
        )?;
    }
    chunk_gate(
        contract,
        context,
        &admission::ImageRequestShape::with_reference_count(g.width, g.height, references),
    )?;
    admission_canvas(g.width, g.height)?;
    Ok(())
}

/// The materializing fallback cannot split one query row: a selected attention chunk below one
/// row of the longest attention of `shape` (every DiT head over its packed sequence) fails at
/// runtime, so it is refused here.
fn chunk_gate(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    shape: &admission::ImageRequestShape,
) -> gen_core::Result<()> {
    if let Some(elements) = context.selection.parameters.attention_chunk_size {
        let rows = admission::ImageRequestRows::of(shape);
        let heads = contract.architecture_facts.attention_heads.map_or(
            crate::dit::MiniMaxH3DitConfig::image().num_attention_heads as u64,
            u64::from,
        );
        if rows
            .sequence_rows()
            .checked_mul(heads)
            .is_none_or(|row| row > u64::from(elements))
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: one materializing attention row ({heads} heads x {} keys) exceeds the \
                 selected {elements}-element chunk",
                contract.provider_id,
                rows.sequence_rows()
            )));
        }
    }
    Ok(())
}

/// The canvas a `width x height` request renders on, which is what its resources are sized from.
///
/// Admission takes the **requested** size, exactly as the generator does: any positive size
/// ([`gen_core::h3_image::validate_size`]). The generator renders on
/// [`gen_core::h3_image::generation_canvas`] at the default minimum area and resizes back, so the
/// latent grid, the packed sequence and the decode all follow the canvas, not the request. The
/// admitted geometry stays the requested size, so the request scope's equality check binds the
/// request the caller actually sends.
pub fn admission_canvas(width: u32, height: u32) -> gen_core::Result<(u32, u32)> {
    use gen_core::h3_image::{
        generation_canvas, latent_grid, validate_size, MIN_GENERATION_AREA_DEFAULT,
    };
    validate_size(width, height).map_err(gen_core::Error::from)?;
    let canvas = generation_canvas(width, height, MIN_GENERATION_AREA_DEFAULT);
    latent_grid(canvas.0, canvas.1)?;
    Ok(canvas)
}

/// The admission estimate (SW-7095): the selected rung's per-phase GPU peak for the admitted
/// geometry must fit the request's budget — the device's free memory as the caller measured it.
/// A rung whose estimate does not fit is refused, so the ladder moves to the next rung (rung 4
/// streams what rung 1 cannot hold) instead of admitting a render that would run out of memory.
/// Without readable source headers (the weights-free surface) there is nothing to price.
fn estimate_gate(
    facts: Option<&admission::ImageMemoryFacts>,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    let g = context.geometry;
    estimate_shape(
        facts,
        contract,
        context,
        &admission::ImageRequestShape::with_reference_count(g.width, g.height, g.reference_count),
    )
}

/// [`estimate_gate`] at an explicit request shape.
fn estimate_shape(
    facts: Option<&admission::ImageMemoryFacts>,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    shape: &admission::ImageRequestShape,
) -> gen_core::Result<()> {
    let Some(facts) = facts else {
        return Ok(());
    };
    let strategy = context.selection.strategy;
    // A rung the contract does not run here is the shared pipeline's refusal, not an estimate.
    if strategy.engages(MemoryStrategy::BoundedTransformerResidency)
        && !contract.engages(strategy, MemoryStrategy::BoundedTransformerResidency)
    {
        return Ok(());
    }
    let plan = admission::estimate(contract, facts, shape, &context.selection)?;
    let budget = context.budget.effective_bytes();
    if plan.peak_device_bytes() > budget {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: {plan} exceeds the {:.2} GB ({:.2} GiB) GPU budget",
            contract.provider_id,
            admission::decimal_gb(budget),
            admission::binary_gib(budget)
        )));
    }
    Ok(())
}

/// The source facts the admission estimate prices, when `spec` resolves a snapshot whose headers
/// can be read; `None` for the weights-free surface and for a source with no CUDA block-window
/// inventory (which then admits on the shared checks alone).
pub fn facts_for(spec: &LoadSpec) -> Option<admission::ImageMemoryFacts> {
    gen_core::resolved_snapshot_root(spec)?;
    admission::ImageMemoryFacts::resolve(spec).ok()
}

/// The admitted request re-priced at its **exact** geometry (SW-7092 edit), run by the generator
/// before any work. Admission sees only the reference count (`MemoryGeometry` carries no
/// reference sizes) and prices each reference as a 2048² VAE copy, but `reference_vae_size` caps
/// the short edge only: a 4096x2048 reference keeps 8192 rows where that bound priced 4096. So
/// the chunk-row check and the per-phase estimate run again against the admitted selection and
/// budget with every reference's own size, and a request whose real sequence does not fit is
/// refused instead of running out of memory.
pub(crate) fn admit_exact_request(
    facts: Option<&admission::ImageMemoryFacts>,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    req: &gen_core::GenerationRequest,
) -> gen_core::Result<()> {
    let shape = admission::ImageRequestShape::from_request(req)?;
    chunk_gate(contract, context, &shape)?;
    estimate_shape(facts, contract, context, &shape)
}

/// The loaded generator's safety check: the shared handshake, this route's gate, then the
/// per-phase estimate against the budget. The weights-free registration
/// ([`IMAGE_MEMORY_REGISTRATION`]) runs the first two only — it holds no source to price.
pub(crate) fn safety_check_with(
    facts: Option<&admission::ImageMemoryFacts>,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match gen_core::standard_registered_safety_check(&MEMORY_PROFILE, spec, contract, context) {
        MemorySafetyDecision::Accept => match estimate_gate(facts, contract, context) {
            Ok(()) => MemorySafetyDecision::Accept,
            Err(error) => MemorySafetyDecision::rejected(error),
        },
        reject => reject,
    }
}

/// The executable admission seam the loaded generator calls — the same
/// [`IMAGE_MEMORY_BEHAVIOR`] routing, through the profile's standard request handshake, after the
/// per-phase estimate.
pub(crate) fn registered_begin_request(
    facts: Option<&admission::ImageMemoryFacts>,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    estimate_gate(facts, contract, context)?;
    gen_core::standard_begin_request(
        &MEMORY_PROFILE,
        IMAGE_MODEL_ID,
        spec,
        contract,
        context,
        registered_scope,
    )
}

fn registered_scope(
    provider_id: &'static str,
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _loaded_tier: MemoryNumericTier,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if crate::memory_strategy::active_context().is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: a memory request is already active on this thread"
        )));
    }
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider_id,
        candle_gen::default_device()?,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::dit::MiniMaxH3DitConfig::image().num_layers,
        move |use_pid, edge, overlap| {
            if use_pid {
                return Err(gen_core::Error::Unsupported(format!(
                    "{provider_id}: there is no PiD decoder on this route"
                )));
            }
            validate_decode(Some(edge), Some(overlap))
        },
    )?;
    config.attention_chunk_size = context.selection.parameters.attention_chunk_size;
    config.transformer_window = context.selection.parameters.transformer_window_size;
    // The selected context is this thread's attention budget for the life of the scope.
    Ok(Some(crate::memory_strategy::activate_request_scope(
        config, context,
    )))
}

/// The memory-strategy registration paired with the `minimax_h3_image` generator.
pub const IMAGE_MEMORY_REGISTRATION: candle_gen::gen_core::MemoryRegistration = candle_gen::gen_core::profile_memory_registration!(
    MEMORY_PROFILE,
    IMAGE_MODEL_ID,
    contract_for
);

/// The weights-free contract fixture, dense tier only: the quantized tiers add no strategy or
/// formula, only smaller asset bytes read from the staged directories.
pub const IMAGE_MEMORY_CONTRACT_FIXTURE: candle_gen::gen_core::MemoryContractFixtureRegistration =
    candle_gen::gen_core::MemoryContractFixtureRegistration {
        provider_id: IMAGE_MODEL_ID,
        contract: weights_free_contract,
        surface_specs: memory_contract_surface_specs,
    };

/// The executable behavior seam every implemented optimized rung must have.
pub const IMAGE_MEMORY_BEHAVIOR: candle_gen::gen_core::MemoryBehaviorRegistration = candle_gen::gen_core::profile_memory_behavior_registration!(
    MEMORY_PROFILE,
    IMAGE_MODEL_ID,
    registered_scope
);

fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|s| s.selector.tier == gen_core::MemoryContractSurfaceTier::Bf16)
        .collect()
}

#[cfg(test)]
mod tests;
