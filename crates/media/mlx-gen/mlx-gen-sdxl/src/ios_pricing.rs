//! SDXL-family source-owned counted/logical iOS pricing (epic16774).
//!
//! Component constructors and activation walks use header facts, never native shape models.
//! Directory and collision-equivalent Fused sources retain their distinct load recipes;
//! normalized Fused headers come from the full actual remapper/IR, not a synthetic directory.
//! The full source-context replay gate is preserved separately from these pricing mechanisms.
//!
//! Logical/source requested work is not native backing size or physical process growth.
//! Fused source/view transfer remains explicitly unbound. General non-equivalent Fused aliases,
//! configured adapters, PiD/sealed coverage and complete source-host/physical proofs remain
//! mandatory same-story work; this module does not certify their completion.

use crate::config::{ClipTextConfig, UNetConfig, VaeConfig};
use crate::ip_adapter::pricing::{IpImageEncoderFacts, ResamplerFacts, VisionFacts};
use crate::ip_adapter::ResamplerConfig;
use crate::quantum::{self as q, Backend, Quanta};
use crate::sampling::{
    ReferenceImageFacts, ResolvedSampling, SamplingRecipe, SamplingRoute, SamplingWorkFacts,
};
use crate::text_encoder::ClipPrice;
use crate::unet::pricing::{
    tensor as price_tensor, tensor_bytes as price_bytes, ControlFacts, FactView, UNetFacts,
};
use crate::vae::pricing::AutoencoderFacts;
use crate::vision_encoder::VisionConfig;
use mlx_gen::adapters::facts::TensorFacts as PriceTensor;
use mlx_gen::gen_core::{CfgBatching, GenerationMemory, LoadSpec, MemoryMode, WeightsSource};
use mlx_gen::ios_pricing::{
    IosAnchorCandidate, IosAnchorWitness, IosMemoryCoordinate, IosMemoryDomain, IosPhase,
    IosPricing, IosPricingRequest, IosWindowBound, IosWindowPrice, IosWorkBasis,
};
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;
use std::path::{Path, PathBuf};

/// Every quantum label this crate checks (`ios_quantum::{run, before, before_load*}`) and the phase
/// it runs in. A device capture attributes its kernel peak through [`ios_quantum_phase`].
const PHASES: &[(&str, IosPhase)] = &[
    (q::TEXT_ENCODER_1_LOAD, IosPhase::Load),
    (q::TEXT_ENCODER_2_LOAD, IosPhase::Load),
    (q::UNET_LOAD, IosPhase::Load),
    (q::VAE_LOAD, IosPhase::Load),
    (q::VAE_FP16_FIX_LOAD, IosPhase::Load),
    (q::CONTROLNET_LOAD, IosPhase::Load),
    (q::IP_IMAGE_ENCODER_LOAD, IosPhase::Load),
    (q::IP_ADAPTER_LOAD, IosPhase::Load),
    (q::LDM_TEXT_LOAD, IosPhase::Load),
    (q::LDM_HEAVY_LOAD, IosPhase::Load),
    (q::ADAPTER_LOAD, IosPhase::Load),
    (q::ADAPTER_MERGE, IosPhase::Load),
    (q::LOAD_QUANTIZE, IosPhase::Load),
    (q::CLIP_EMBEDDINGS, IosPhase::Conditioning),
    (q::CLIP_LAYER, IosPhase::Conditioning),
    (q::CLIP_POOLED, IosPhase::Conditioning),
    (q::VISION_EMBEDDINGS, IosPhase::Conditioning),
    (q::VISION_LAYER, IosPhase::Conditioning),
    (q::RESAMPLER_PROJ_IN, IosPhase::Conditioning),
    (q::RESAMPLER_LAYER, IosPhase::Conditioning),
    (q::RESAMPLER_PROJ_OUT, IosPhase::Conditioning),
    (q::IP_CFG_TOKENS, IosPhase::Conditioning),
    (REQUEST_INPUTS, IosPhase::Load),
    (q::VAE_ENCODE_IN, IosPhase::Encode),
    (q::VAE_ENCODE_RESNET, IosPhase::Encode),
    (q::VAE_ENCODE_ATTENTION, IosPhase::Encode),
    (q::VAE_DOWNSAMPLE, IosPhase::Encode),
    (q::VAE_ENCODE_TILE, IosPhase::Encode),
    (q::VAE_ENCODE_TILED_STAGE, IosPhase::Encode),
    (q::VAE_ENCODE_OUT, IosPhase::Encode),
    (q::CONTROL_COND_CONV, IosPhase::Encode),
    (q::UNET_STEM, IosPhase::Denoise),
    (q::UNET_RESNET, IosPhase::Denoise),
    (q::UNET_DOWNSAMPLE, IosPhase::Denoise),
    (q::UNET_UPSAMPLE, IosPhase::Denoise),
    (q::TRANSFORMER_PROJ_IN, IosPhase::Denoise),
    (q::TRANSFORMER_BLOCK, IosPhase::Denoise),
    (q::TRANSFORMER_PROJ_OUT, IosPhase::Denoise),
    (q::CONTROL_DOWN_INJECTION, IosPhase::Denoise),
    (q::CONTROL_MID_INJECTION, IosPhase::Denoise),
    (q::UNET_CONV_OUT, IosPhase::Denoise),
    (q::CONTROL_STEM, IosPhase::Denoise),
    (q::CONTROL_ZERO_CONV, IosPhase::Denoise),
    (q::SAMPLER_STEP, IosPhase::Denoise),
    // The shared sampler-step guard a few-step acceleration sampler checks inside SDXL's denoise.
    (mlx_gen::sampler::SAMPLER_STEP_QUANTUM, IosPhase::Denoise),
    (q::VAE_DECODE_IN, IosPhase::Decode),
    (q::VAE_RESNET, IosPhase::Decode),
    (q::VAE_ATTENTION, IosPhase::Decode),
    (q::VAE_UPSAMPLE, IosPhase::Decode),
    (q::VAE_DECODE_OUT, IosPhase::Decode),
    (q::VAE_TILE, IosPhase::Decode),
    (q::VAE_TILED_STAGE, IosPhase::Decode),
    (q::DECODED_TO_IMAGE, IosPhase::Decode),
];

/// The priced item of the request's input tensors (init / control / IP image), created after the
/// last component load and before the first execution quantum — a capture attributes them to the
/// load phase when no conditioning quantum runs in between (a staged request). Pricing only: no
/// guard checks it.
pub(crate) const REQUEST_INPUTS: &str = "SDXL request inputs";

/// Modes requiring their own iOS anchor at `tier` (`bf16`, `q8`, `q4`).
/// The CPU additivity proof executes every advertised mode at both endpoints for all three
/// tiers. This compatibility list is not a physical-transfer certificate: correlated admission
/// additionally requires the source-owned bounder to cover every actual request window.
/// Unknown tiers retain the fail-closed exception set; no unmeasured tier is inferred.
pub fn ios_non_additive_modes(tier: &str) -> &'static [&'static str] {
    match tier {
        "bf16" | "q8" | "q4" => &[],
        _ => &["image_to_image", "edit", "control", "ip_adapter"],
    }
}

/// The phase a quantum label of this crate runs in: the longest label constant `label` starts with
/// (so `SDXL VAE tiled stage` is not read as `SDXL VAE tile`). `None` for a label this crate never
/// checks (e.g. the shared tiled-convolution core's synchronization).
pub fn ios_quantum_phase(label: &str) -> Option<IosPhase> {
    PHASES
        .iter()
        .filter(|(constant, _)| label.starts_with(constant))
        .max_by_key(|(constant, _)| constant.len())
        .map(|(_, phase)| *phase)
}

/// Path-free configured overlay graph facts from SDXL's actual load selectors. This describes
/// logical layouts and install branches only; it supplies neither a native backing certificate nor
/// a ProcessGrowth bridge. No model, GenerationRequest, image, schedule or Array is constructed.
pub fn configured_overlay_semantic_facts(spec: &LoadSpec) -> Result<serde_json::Value> {
    use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
    use serde_json::json;
    if spec.pid.is_some() {
        return Err(Error::Unsupported(
            "sdxl ios pricing: the PiD decoder overlay is not priced".into(),
        ));
    }
    if spec.decode_geometry_policy_authoritative {
        return Err(Error::Unsupported(
            "sdxl ios pricing: a sealed geometry decode policy is not priced".into(),
        ));
    }
    // This is OVERLAY identity, not a substitute for the caller's base artifact/architecture/
    // tier validation. SDXL has no intrinsic default overlay. Unknown/named configuration stays
    // on the normalizer path; any configured adapter/control/IP/VAE overlay still needs real targets.
    if spec.adapters.is_empty()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.components.is_empty()
        && spec.identity.is_none()
        && spec.text_encoder.is_none()
        && spec.decode_geometry_policies.is_empty()
        && spec.decode_quality_runtime_identity.is_none()
    {
        return Ok(serde_json::Value::Null);
    }
    let (form, unet_headers) = match &spec.weights {
        WeightsSource::Dir(root) => {
            if let Some(quant) = spec.quantize {
                mlx_gen::quant::needs_load_time_quant(
                    root,
                    "unet",
                    quant.bits(),
                    crate::model::descriptor().id,
                )?;
            }
            let file = crate::loader::resolve_unet_weight_file(root, crate::model::DTYPE)?;
            (
                "diffusers",
                safetensors_path_tensor_headers(file).map_err(|e| Error::Msg(e.to_string()))?,
            )
        }
        WeightsSource::File(file) => {
            // The actual native loader opens the entire fused file before component remapping.
            for h in safetensors_path_tensor_headers(file).map_err(|e| Error::Msg(e.to_string()))? {
                crate::quant::header_dtype(h.dtype)?;
            }
            (
                "fused_ldm",
                crate::ldm::component_headers(
                    file,
                    mlx_gen::gen_core::sdxl_ldm::SdxlComponent::Unet,
                )?,
            )
        }
    };
    let base_layouts = configured_component_layouts(&unet_headers)?;
    let adapter_facts = crate::adapters::configured_adapter_semantic_facts(spec, &unet_headers)?;
    let cfg = UNetConfig::sdxl_base();
    let mut controls = Vec::new();
    for source in spec.control.iter().chain(&spec.extra_controls) {
        let path = match source {
            WeightsSource::File(p) | WeightsSource::Dir(p) => p,
        };
        let headers =
            safetensors_path_tensor_headers(path).map_err(|e| Error::Msg(e.to_string()))?;
        let (_, complete, convs) = crate::unet::adapter_target_paths_from_headers(&cfg, &headers);
        // ControlNet::from_weights uses the SAME down BlockSpec selector, then the same mid
        // constructors. It has no up stack or conv_out; its own embedding and zero heads are fixed.
        let linears: Vec<_> = complete
            .into_iter()
            .filter(|p| !p.starts_with("up_blocks."))
            .map(|p| projection_semantic_fact(&headers, &p))
            .collect::<Result<_>>()?;
        let convolution_paths: Vec<_> = convs
            .into_iter()
            .filter(|p| !p.starts_with("up_blocks.") && p != "conv_out")
            .collect();
        controls.push(json!({"constructor":"sdxl_control_encoder_copy","layers_per_block":cfg.layers_per_block,
            "block_out_channels":cfg.block_out_channels,"heads":cfg.num_attention_heads,
            "transformer_layers":cfg.transformer_layers_per_block,"down_block_types":cfg.down_block_types,
            "projection_sources":linears,"convolution_targets":convolution_paths,
            "load_quantization_after_construction":spec.quantize.map(|q|q.bits()),
            "loaded_source_layouts":configured_component_layouts(&headers)?}));
    }
    let ip = match &spec.ip_adapter {
        Some(WeightsSource::File(_)) => {
            return Err(Error::Unsupported(
                "sdxl ip_adapter expects an h94/IP-Adapter snapshot directory".into(),
            ))
        }
        Some(WeightsSource::Dir(dir)) => {
            let (encoder_file, weights_file) = crate::loader::resolve_ip_adapter_files(dir)?;
            let encoder = safetensors_path_tensor_headers(encoder_file)
                .map_err(|e| Error::Msg(e.to_string()))?;
            let weights = safetensors_path_tensor_headers(weights_file)
                .map_err(|e| Error::Msg(e.to_string()))?;
            let indices =
                crate::ip_adapter::ip_kv_indices(weights.iter().map(|h| h.name.as_str()))?;
            let (_, complete, _) =
                crate::unet::adapter_target_paths_from_headers(&cfg, &unet_headers);
            let slots: Vec<_> = complete
                .into_iter()
                .filter(|p| p.ends_with(".attn2.to_k"))
                .collect();
            if indices.len() != slots.len() {
                return Err(Error::Msg(
                    "ip_adapter: K/V pair count does not match actual cross-attention walk".into(),
                ));
            }
            let mut pairs = Vec::new();
            for (index, slot) in indices.into_iter().zip(slots) {
                let lookup = |role: &str| {
                    weights
                        .iter()
                        .find(|h| h.name == format!("ip_adapter.{index}.to_{role}_ip.weight"))
                        .ok_or_else(|| {
                            Error::MissingTensor(format!("ip_adapter.{index}.to_{role}_ip.weight"))
                        })
                };
                let k = lookup("k")?;
                let v = lookup("v")?;
                pairs.push(json!({"target":slot,"k_shape":k.shape,"v_shape":v.shape,
                    "k_dtype":format!("{:?}",crate::quant::loaded_header_dtype(&weights,k)?),
                    "v_dtype":format!("{:?}",crate::quant::loaded_header_dtype(&weights,v)?),
                    "installed":"dense_kv_before_unet_quantization"}));
            }
            let vision = VisionConfig::vit_h_14();
            let resampler = ResamplerConfig::plus_sdxl_vit_h();
            json!({"pairs":pairs,"vision":{"hidden":vision.hidden,"layers":vision.num_layers,"heads":vision.num_heads,
                "patch":vision.patch,"image_size":vision.image_size,"channels":vision.num_channels,"quick_gelu":vision.quick_gelu},
                "resampler":{"dim":resampler.dim,"depth":resampler.depth,"heads":resampler.heads,
                    "head_dim":resampler.dim_head,"queries":resampler.num_queries,"embed_dim":resampler.embed_dim,"output_dim":resampler.output_dim},
                "encoder_loaded_source_layouts":configured_component_layouts(&encoder)?,
                "resampler_and_kv_loaded_source_layouts":configured_component_layouts(&weights)?})
        }
        None => serde_json::Value::Null,
    };
    let vae_overlay=spec.components.get(crate::loader::VAE_FP16_FIX_COMPONENT).map(|source| {
        let file=crate::loader::resolve_vae_fp16_fix_file(source)?;
        let headers=safetensors_path_tensor_headers(file).map_err(|e|Error::Msg(e.to_string()))?;
        Ok::<_,Error>(json!({"installed":"fp16_fix_vae","loaded_source_layouts":configured_component_layouts(&headers)?}))
    }).transpose()?;
    Ok(
        json!({"provider":"sdxl","load_form":form,"load_quantization_after_adapters_and_ip":spec.quantize.map(|q|q.bits()),
        "base_loaded_source_layouts":base_layouts,"adapter_graph":adapter_facts,"controls":controls,"ip_adapter":ip,"vae_overlay":vae_overlay}),
    )
}

fn projection_semantic_fact(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    path: &str,
) -> Result<serde_json::Value> {
    let projection = crate::quant::linear_from_headers(headers, path)?;
    let rows = if path.ends_with(".ff.net.0.proj") {
        Some(crate::quant::geglu_half_ranges(projection.shape[0])?)
    } else {
        None
    };
    Ok(
        serde_json::json!({"target":path,"logical_shape":projection.shape,"source_packed_bits":projection.source_packed_bits,
        "weight_dtype":format!("{:?}",projection.weight_dtype),"scales_dtype":projection.scales_dtype.map(|d|format!("{d:?}")),
        "affine_biases_dtype":projection.affine_biases_dtype.map(|d|format!("{d:?}")),"geglu_rows":rows,
        "codes_shape":projection.packed_codes_shape,"scales_shape":projection.packed_scales_shape,"biases_shape":projection.packed_biases_shape}),
    )
}

fn configured_component_layouts(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
) -> Result<serde_json::Value> {
    let mut layouts = std::collections::BTreeMap::new();
    for h in headers {
        layouts.insert(
            h.name.clone(),
            serde_json::json!({"shape":h.shape,
            "loaded_dtype":format!("{:?}",crate::quant::loaded_header_dtype(headers,h)?)}),
        );
    }
    Ok(serde_json::json!(layouts))
}

/// The request's counted per-phase price on the device (Metal kernel transients), header-only.
pub fn ios_pricing(req: &IosPricingRequest<'_>) -> Result<IosPricing> {
    ios_pricing_on(req, Backend::Metal)
}

/// Reprice a captured sampling recipe without constructing a request or its image payloads.
/// `work` records original conditioning roles, before the provider's init/IP interpretation.
/// A live generation/transform/depth context is rejected rather than silently overridden.
/// Like [`ios_pricing`], this is a raw counted ledger, not physical/domain certification.
/// Full source-context replay seam. Transport coherence is checked here; this does NOT certify
/// tokenizer/fixed-vocab host allocations or source-request layouts as physical process growth.
/// Native preprocessing/constructor validation remains at its existing runtime chronology.
pub fn ios_pricing_with_source(
    req: &IosPricingRequest<'_>,
    recipe: &SamplingRecipe,
    source: &crate::sampling::SamplingSourceFacts,
) -> Result<IosPricing> {
    let regime = Regime::from_recipe(req, recipe, source.work())?;
    source.validate_roles()?;
    // Return the existing RAW COUNTED NATIVE ledger only. Complete source facts are transported
    // here, not converted to an invented tokenizer/image-host or physical-process certificate.
    price_regime(req, &regime, Backend::Metal)
}

pub fn ios_pricing_with_recipe(
    req: &IosPricingRequest<'_>,
    recipe: &SamplingRecipe,
    work: SamplingWorkFacts,
) -> Result<IosPricing> {
    let regime = Regime::from_recipe(req, recipe, work)?;
    price_regime(req, &regime, Backend::Metal)
}

/// [`ios_pricing`] with `backend`'s kernel transients (the CPU proof fixture prices its own).
pub(crate) fn ios_pricing_on(req: &IosPricingRequest<'_>, backend: Backend) -> Result<IosPricing> {
    let regime = Regime::from_request(req)?;
    price_regime(req, &regime, backend)
}

fn price_regime(
    req: &IosPricingRequest<'_>,
    regime: &Regime,
    backend: Backend,
) -> Result<IosPricing> {
    let inputs = Inputs::from_spec(req.spec, regime)?;
    q::with_backend(backend, || {
        let mut priced = Priced::build_configured(&inputs, req.spec)?;
        priced.price_configured(req.spec, &inputs.unet, inputs.quant_bits, regime)
    })
}

/// Source-owned NATIVE-ALLOCATOR matched-window certificates for the correlated anchor law.
/// The typed domain forbids using these as process-growth authority. The law must check
/// coverage of ALL requested windows; an unreturned class is not permission to ignore that work.
/// In particular, a new Encode phase is never certified from a logical VAE scratch sum here.
pub fn ios_bounder(
    anchor_request: &IosPricingRequest<'_>,
    anchor_pricing: &IosPricing,
    request: &IosPricingRequest<'_>,
    request_pricing: &IosPricing,
) -> Result<Vec<IosWindowBound>> {
    let anchor = Regime::from_request(anchor_request)?;
    let requested = Regime::from_request(request)?;
    let anchor_inputs = Inputs::from_spec(anchor_request.spec, &anchor)?;
    let request_inputs = Inputs::from_spec(request.spec, &requested)?;
    if anchor_inputs.fused.is_some() || request_inputs.fused.is_some() {
        return Err(Error::Unsupported("sdxl fused source/view native backing transfer remains unbound; logical parents are not physical certificates".into()));
    }
    if anchor_request.execution != request.execution
        || anchor_request.spec.load_shape != request.spec.load_shape
        || anchor_request.spec.offload_policy != request.spec.offload_policy
        || anchor_inputs.unet != request_inputs.unet
        || anchor_inputs.te1 != request_inputs.te1
        || anchor_inputs.te2 != request_inputs.te2
        || anchor_inputs.vae != request_inputs.vae
        || anchor_inputs.quant_bits != request_inputs.quant_bits
        || !matches!(anchor_inputs.quant_bits, Some(4 | 8))
        || anchor_inputs.streamable != request_inputs.streamable
        || !anchor_inputs.controls.is_empty()
        || !request_inputs.controls.is_empty()
        || anchor_inputs.ip.is_some()
        || request_inputs.ip.is_some()
    {
        return Err(Error::Unsupported(
            "sdxl common-forward transfer needs the same source component graph".into(),
        ));
    }
    q::with_backend(Backend::Metal, || {
        let expected_anchor = Priced::build(&anchor_inputs)?.price(&anchor)?.0;
        let expected_request = Priced::build(&request_inputs)?.price(&requested)?.0;
        let matches_source = |expected: &IosPricing, supplied: &IosPricing| {
            expected.phases == supplied.phases
                && expected.windows == supplied.windows
                && expected.absent_phases == supplied.absent_phases
        };
        if !matches_source(&expected_anchor, anchor_pricing)
            || !matches_source(&expected_request, request_pricing)
        {
            return Err(Error::Unsupported(
                "sdxl transfer pricing is not the source-generated profile".into(),
            ));
        }
        bound_ancestral_forward_windows(
            &anchor,
            anchor_pricing,
            &requested,
            request_pricing,
            anchor_inputs.arch.latent_scale,
        )
    })
}

/// The body cancellation is source-specific: forward_eps supplies the SAME F16 input shape to
/// the SAME packed U-Net operations at the SAME first/steady epoch and CFG half. The separately
/// owned state/cast/init inputs are backed by the native lifetime proof, not by guard-upper
/// subtraction. Class strings locate those already-proved points; equality is not proof authority.
pub(crate) fn bound_ancestral_forward_windows(
    anchor: &Regime,
    anchor_pricing: &IosPricing,
    request: &Regime,
    request_pricing: &IosPricing,
    latent_scale: i32,
) -> Result<Vec<IosWindowBound>> {
    if anchor.sampling.route() != SamplingRoute::Ancestral
        || request.sampling.route() != SamplingRoute::Ancestral
        || !matches!(anchor.route, Route::Text | Route::Reference | Route::Mask)
        || !matches!(request.route, Route::Text | Route::Reference | Route::Mask)
        || anchor.width != request.width
        || anchor.height != request.height
        || anchor.context_tokens != request.context_tokens
        || anchor.sampling.allocation_class() != request.sampling.allocation_class()
        || anchor.staged != request.staged
        || anchor.count != request.count
    {
        return Err(Error::Unsupported(
            "sdxl forward transfer has no source-matched execution domain".into(),
        ));
    }
    let mut bounds = Vec::new();
    for requested in request_pricing
        .windows
        .iter()
        .filter(|window| window.owner_domain == "sdxl/ancestral/forward-input-native-v1")
    {
        let Some(class) = requested.profile_class() else {
            continue;
        };
        let mut candidates = Vec::new();
        for at_anchor in anchor_pricing.windows.iter().filter(|window| {
            window.owner_domain == "sdxl/ancestral/forward-input-native-v1"
                && window.profile_class() == Some(class)
        }) {
            // This checks that the generated ledger still describes the identical common body.
            // It does NOT turn either scratch upper into a physical anchor lower bound.
            if requested.label != at_anchor.label
                || requested.basis != at_anchor.basis
                || requested.scratch_bytes != at_anchor.scratch_bytes
                || requested.resident_bytes != at_anchor.resident_bytes
            {
                continue;
            }
            let witnesses =
                ancestral_forward_witnesses(anchor, anchor_pricing, at_anchor, latent_scale)?;
            if witnesses.is_empty() {
                continue;
            }
            // The raw window ledger is not an allocation certificate. Price each unique
            // source-owned backing with the ACTUAL allocator operand before taking its delta.
            // Keep every observed allocator correspondence: the public law takes MAX, not cheapest.
            for witness in witnesses {
                let page = witnessed_cache_zero_page(anchor_pricing, &witness)?;
                let elements = (anchor.height as u64 / latent_scale as u64)
                    .checked_mul(anchor.width as u64 / latent_scale as u64)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or_else(|| {
                        Error::Unsupported("sdxl native input extent overflow".into())
                    })?;
                let anchor_owned =
                    fresh_ancestral_forward_inputs(anchor, at_anchor, elements, page)?;
                let request_owned =
                    fresh_ancestral_forward_inputs(request, requested, elements, page)?;
                candidates.push(IosAnchorCandidate {
                    anchor_class: class.into(),
                    delta_growth_upper: i128::from(request_owned) - i128::from(anchor_owned),
                    physical_anchor_lower_bound: None,
                    witnesses: vec![witness],
                });
            }
        }
        if !candidates.is_empty() {
            bounds.push(IosWindowBound::Transfer {
                domain: IosMemoryDomain::NativeAllocatorGrowth,
                coordinate: IosMemoryCoordinate::BeforeModelLoad,
                request_class: class.into(),
                anchor_candidates: candidates,
            });
        }
    }
    Ok(bounds)
}

fn witnessed_cache_zero_page(pricing: &IosPricing, witness: &IosAnchorWitness) -> Result<u64> {
    let invalid = || {
        Error::Unsupported("sdxl native allocation lacks an actual checked cache-zero basis".into())
    };
    let capture = pricing
        .observations
        .get(witness.capture as usize)
        .ok_or_else(invalid)?;
    let check = capture
        .checks
        .get(witness.check as usize)
        .ok_or_else(invalid)?;
    let fields = check
        .context
        .iter()
        .find_map(|&index| {
            capture
                .metadata
                .get(index as usize)
                .filter(|mark| mark.provider == "mlx" && mark.kind == "allocation-scope")
                .map(|mark| &mark.fields)
        })
        .ok_or_else(invalid)?;
    if fields.get("setter_succeeded") != Some(&1)
        || fields.get("cache_query_succeeded") != Some(&1)
        || fields.get("cache_limit_requested_bytes") != Some(&0)
        || fields.get("cache_bytes") != Some(&0)
        || fields.get("restoring") != Some(&0)
    {
        return Err(invalid());
    }
    fields
        .get("page_bytes")
        .copied()
        .filter(|page| page.is_power_of_two())
        .ok_or_else(invalid)
}

fn fresh_ancestral_forward_inputs(
    regime: &Regime,
    window: &IosWindowPrice,
    elements: u64,
    page: u64,
) -> Result<u64> {
    let invalid = || {
        Error::Unsupported("sdxl native owner allocation provenance/extent is unverified".into())
    };
    let epoch = window
        .profile_class()
        .and_then(|class| class.rsplit(':').nth(2))
        .ok_or_else(invalid)?;
    let state = match epoch {
        "first" => regime.initial_state_dtype(),
        "steady" => Dtype::Float32,
        _ => return Err(invalid()),
    };
    // These sampler states are full-span outputs, not arbitrary caller-provided views. The
    // generation's outer checked Scope encloses their construction and existing eval barriers.
    // Init/noise/mask are separate retained owners; never round their combined logical sum.
    let mut owned = regime
        .forward_owners(state)
        .fresh_native_bytes(
            elements,
            elements,
            crate::model::DTYPE,
            regime.model_rows() as u64,
            page,
        )
        .ok_or_else(invalid)?;
    let mut add = |bytes| -> Result<()> {
        owned = owned
            .checked_add(
                mlx_gen::ios_quantum::fresh_metal_allocation_bytes(bytes, page)
                    .ok_or_else(invalid)?,
            )
            .ok_or_else(invalid)?;
        Ok(())
    };
    if regime.has_init {
        add(elements
            .checked_mul(q::dtype_bytes(crate::model::DTYPE))
            .ok_or_else(invalid)?)?;
    }
    if regime.route == Route::Mask {
        add(elements
            .checked_mul(q::dtype_bytes(Dtype::Float32))
            .ok_or_else(invalid)?)?;
        add((elements / 4)
            .checked_mul(q::dtype_bytes(Dtype::Float32))
            .ok_or_else(invalid)?)?;
    }
    Ok(owned)
}

fn ancestral_forward_witnesses(
    regime: &Regime,
    pricing: &IosPricing,
    window: &IosWindowPrice,
    latent_scale: i32,
) -> Result<Vec<IosAnchorWitness>> {
    let malformed = || Error::Unsupported("sdxl malformed source forward window".into());
    let mut parts = window.profile_class().ok_or_else(malformed)?.rsplit(':');
    let ordinal: usize = parts
        .next()
        .and_then(|s| s.strip_prefix('q'))
        .and_then(|s| s.parse().ok())
        .ok_or_else(malformed)?;
    let half: u64 = parts
        .next()
        .and_then(|s| s.strip_prefix("cfg"))
        .and_then(|s| s.parse().ok())
        .ok_or_else(malformed)?;
    let first = match parts.next() {
        Some("first") => true,
        Some("steady") => false,
        _ => return Err(malformed()),
    };
    if latent_scale <= 0 {
        return Err(malformed());
    }
    let expected_steps = regime
        .sampling
        .ancestral_window(
            crate::config::DiffusionConfig::sdxl_base().num_train_steps as f32,
            regime
                .sampling
                .effective_init_strength(regime.has_init, regime.route == Route::Mask),
        )?
        .0 as u64;
    let sequential = regime.sampling.guidance() > 1.0 && !regime.cfg_batching.is_batched();
    let expected_half = if sequential { half + 1 } else { 0 };
    let latent_h = u64::try_from(regime.height / latent_scale).map_err(|_| malformed())?;
    let latent_w = u64::try_from(regime.width / latent_scale).map_err(|_| malformed())?;
    let expected_elements = (regime.model_rows() as u64)
        .checked_mul(latent_h)
        .and_then(|n| n.checked_mul(latent_w))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(malformed)?;
    let state = if first {
        regime.initial_state_dtype()
    } else {
        Dtype::Float32
    };
    let mut witnesses = Vec::new();
    for (capture_index, capture) in pricing.observations.iter().enumerate() {
        if capture.domain != IosMemoryDomain::NativeAllocatorGrowth
            || capture.coordinate != IosMemoryCoordinate::BeforeModelLoad
        {
            continue;
        }
        let mut ordinals = std::collections::BTreeMap::<u32, usize>::new();
        for (check_index, check) in capture.checks.iter().enumerate() {
            if check.phase != Some(IosPhase::Denoise)
                || check.label == q::SAMPLER_STEP
                || check.label == mlx_gen::sampler::SAMPLER_STEP_QUANTUM
            {
                continue;
            }
            let metadata = |kind: &str| {
                check.context.iter().find_map(|&index| {
                    capture
                        .metadata
                        .get(index as usize)
                        .filter(|m| m.provider == "sdxl" && m.kind == kind)
                        .map(|m| (index, &m.fields))
                })
            };
            let (
                Some((_, epoch)),
                Some((half_index, cfg_half)),
                Some((_, input)),
                Some((cast_index, cast)),
            ) = (
                metadata("discrete-epoch"),
                metadata("forward-cfg-half"),
                metadata("discrete-state"),
                metadata("forward-input-after-cast-pending"),
            )
            else {
                continue;
            };
            let current = ordinals.entry(half_index).or_default();
            let current_ordinal = *current;
            *current += 1;
            let Some((barrier_index, ready)) = metadata("forward-input-materialized") else {
                continue;
            };
            let Some(&evaluation) = epoch.get("evaluation") else {
                continue;
            };
            if current_ordinal != ordinal
                || barrier_index <= half_index
                || barrier_index <= cast_index
                || ready.get("dtype")
                    != Some(&mlx_gen::sampler::sampling_dtype_code(crate::model::DTYPE))
                || ready.get("rows") != Some(&(regime.model_rows() as u64))
                || ready.get("elements") != Some(&expected_elements)
                || cast.get("elements") != Some(&expected_elements)
                || check.label != window.label
                || (evaluation == 0) != first
                || evaluation >= expected_steps
                || epoch.get("steps") != Some(&expected_steps)
                || cfg_half.get("half") != Some(&expected_half)
                || input.get("dtype") != Some(&mlx_gen::sampler::sampling_dtype_code(state))
                || input.get("evaluation") != Some(&evaluation)
                || cast.get("dtype")
                    != Some(&mlx_gen::sampler::sampling_dtype_code(crate::model::DTYPE))
                || cast.get("dim0") != Some(&(regime.model_rows() as u64))
                || cast.get("dim1") != Some(&((regime.height / latent_scale) as u64))
                || cast.get("dim2") != Some(&((regime.width / latent_scale) as u64))
                || cast.get("dim3") != Some(&4)
                || cast.get("rank") != Some(&4)
            {
                continue;
            }
            witnesses.push(IosAnchorWitness {
                capture: u32::try_from(capture_index).map_err(|_| malformed())?,
                check: u32::try_from(check_index).map_err(|_| malformed())?,
            });
        }
    }
    Ok(witnesses)
}

/// The SDXL-family architecture every route loads (`model::load_heavy` and the text loader pin
/// these configs); a tiny proof fixture substitutes its own.
#[derive(Clone)]
pub(crate) struct Architecture {
    pub unet: UNetConfig,
    pub vae: VaeConfig,
    pub te1: ClipTextConfig,
    pub te2: ClipTextConfig,
    pub vision: VisionConfig,
    pub resampler: ResamplerConfig,
    /// CLIP's per-window token count (77).
    pub text_tokens: i32,
    /// The text memory's width (`cross_attention_dim`, 2048).
    pub context_dim: i32,
    /// Output pixels per latent cell along each axis (the VAE's spatial scale, 8).
    pub latent_scale: i32,
    /// Control-image pixels per latent cell (the ControlNet conditioning embedding's stride, 8).
    pub control_scale: i32,
}

impl Architecture {
    pub(crate) fn sdxl() -> Self {
        Self {
            unet: UNetConfig::sdxl_base(),
            vae: VaeConfig::sdxl_base(),
            te1: ClipTextConfig::sdxl_te1(),
            te2: ClipTextConfig::sdxl_te2(),
            vision: VisionConfig::vit_h_14(),
            resampler: ResamplerConfig::plus_sdxl_vit_h(),
            text_tokens: 77,
            context_dim: 2048,
            latent_scale: 8,
            control_scale: 8,
        }
    }
}

/// The memory route of a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    Text,
    /// img2img: an init-image encode, then the denoise from the f32 encoded latent.
    Reference,
    /// Inpaint: the img2img encode plus the per-step mask blend.
    Mask,
    /// ControlNet: the conditioning embedding, then the branch beside every U-Net forward.
    Control,
    /// IP-Adapter: the image prompt encode, then the decoupled cross-attention.
    IpAdapter,
}

impl Route {
    pub(crate) fn from_mode(mode: &MemoryMode) -> Result<Self> {
        Ok(match mode {
            MemoryMode::TextToImage => Self::Text,
            MemoryMode::ImageToImage => Self::Reference,
            MemoryMode::Edit => Self::Mask,
            MemoryMode::Other(other) if other == "control" => Self::Control,
            MemoryMode::Other(other) if other == "ip_adapter" => Self::IpAdapter,
            MemoryMode::Other(other) => {
                return Err(Error::Unsupported(format!(
                    "sdxl ios pricing: no memory route `{other}`"
                )))
            }
        })
    }
}

/// The execution regime one request runs under.
#[derive(Clone)]
pub(crate) struct Regime {
    pub route: Route,
    /// Actual shared dispatch semantics, resolved from the post-caller-policy request.
    pub sampling: ResolvedSampling,
    /// Output images are generated serially; this is NOT the U-Net forward batch.
    pub count: u32,
    /// A true VAE init (including ControlNet+reference), not an IP image prompt.
    pub has_init: bool,
    pub width: i32,
    pub height: i32,
    /// Rung 1: shed the text encoders before the heavy load, the IP image encoder after the
    /// image prompt, and the U-Net / ControlNet / IP state before the decode.
    pub staged: bool,
    pub cfg_batching: CfgBatching,
    /// Rung 2's decode tiling (the init encode tiles at the same edge).
    pub tiling: Option<mlx_gen::tiling::TilingConfig>,
    /// The text memory's token count (CLIP windows x 77).
    pub context_tokens: i32,
    /// Raw IP scalar follows actual Reference override/global/default semantics, not init clamp.
    pub ip_scale: f32,
    /// Exact actual source facts, never the output/CLIP crop geometry. None only outside IP or
    /// legacy non-executed contexts; a live/captured IP price requires Some supplied actual facts.
    pub reference_image: Option<ReferenceImageFacts>,
}

impl Regime {
    pub(crate) fn from_request(req: &IosPricingRequest<'_>) -> Result<Self> {
        if req.transform.is_some() || req.depth.is_some() {
            return Err(Error::Unsupported(
                "sdxl pricing requires generation context, not transform/depth".into(),
            ));
        }
        let sampling = ResolvedSampling::for_pricing(req.generation)?;
        let generation = req
            .generation
            .expect("shared resolver rejected missing generation");
        if generation.width != req.geometry.width
            || generation.height != req.geometry.height
            || generation.count != req.geometry.batch
        {
            return Err(Error::Unsupported(
                "sdxl pricing generation and admitted geometry/count differ".into(),
            ));
        }
        let work = SamplingWorkFacts::from_request(generation);
        // Same source extraction order as generate; metadata only, no media allocation/validation.
        crate::model::reference_inputs(generation)?;
        crate::model::mask_input(generation)?;
        if work
            .reference_count
            .checked_add(work.mask_count)
            .and_then(|count| count.checked_add(work.control_count))
            != Some(req.geometry.reference_count as usize)
        {
            return Err(Error::Unsupported(
                "sdxl pricing generation and admitted aggregate image counts differ".into(),
            ));
        }
        // Control Some(x), including0, is multiplied after as_dtype(x.dtype()) with no branch
        // skip. Its value stays in actual full-work binding; count/shape/dtype determine this cost.
        Self::from_resolved(req, sampling, generation.count, work)
    }

    pub(crate) fn from_recipe(
        req: &IosPricingRequest<'_>,
        recipe: &SamplingRecipe,
        work: SamplingWorkFacts,
    ) -> Result<Self> {
        if req.generation.is_some() || req.transform.is_some() || req.depth.is_some() {
            return Err(Error::Unsupported(
                "sdxl captured recipe pricing requires payload-free work context".into(),
            ));
        }
        if work
            .reference_count
            .checked_add(work.mask_count)
            .and_then(|count| count.checked_add(work.control_count))
            != Some(req.geometry.reference_count as usize)
        {
            return Err(Error::Unsupported(
                "sdxl captured recipe and geometry aggregate image counts differ".into(),
            ));
        }
        let sampling = ResolvedSampling::resolve_recipe(recipe, work)?;
        Self::from_resolved(req, sampling, req.geometry.batch, work)
    }

    fn from_resolved(
        req: &IosPricingRequest<'_>,
        sampling: ResolvedSampling,
        count: u32,
        work: SamplingWorkFacts,
    ) -> Result<Self> {
        if count == 0 || count > crate::model::descriptor().capabilities.max_count {
            return Err(Error::Unsupported(
                "sdxl pricing count is outside the declared serial-output domain".into(),
            ));
        }
        crate::model::validate_image_roles(
            sampling.sampler_name(),
            sampling.route().is_acceleration(),
            sampling.route().is_curated(),
            work.reference_count != 0,
            work.mask_count != 0,
        )?;
        let has_reference = work.reference_count != 0;
        let ip = crate::model::uses_ip_reference(
            req.spec.ip_adapter.is_some(),
            has_reference,
            work.mask_count != 0,
            work.control_count,
            sampling.route().is_acceleration(),
        );
        let route = if work.mask_count != 0 {
            Route::Mask
        } else if work.control_count != 0 {
            Route::Control
        } else if ip {
            Route::IpAdapter
        } else if has_reference && !sampling.route().is_acceleration() {
            Route::Reference
        } else {
            Route::Text
        };
        // Catalog/memory mode is not a model::generate dispatch input. Preserve the existing
        // unknown-mode refusal, but a supported label must not override actual mixed roles.
        Route::from_mode(req.mode)?;
        let loaded_controls =
            usize::from(req.spec.control.is_some()) + req.spec.extra_controls.len();
        if !crate::model::control_count_matches_loaded(work.control_count, loaded_controls) {
            return Err(Error::Unsupported(
                "sdxl pricing Control conditioning count differs from loaded branches".into(),
            ));
        }
        crate::model::validate_control_roles(
            sampling.sampler_name(),
            req.generation
                .map_or(Some(sampling.guidance()), |generation| generation.guidance),
            work.control_count,
            loaded_controls,
            work.mask_count != 0,
        )?;
        if route == Route::IpAdapter && work.reference_image.is_none() {
            return Err(Error::Unsupported("sdxl IP pricing needs actual ReferenceImageFacts; missing legacy source facts are not crop geometry".into()));
        }
        let has_init = has_reference && !ip && !sampling.route().is_acceleration();
        if !req.geometry.width.is_multiple_of(8) || !req.geometry.height.is_multiple_of(8) {
            return Err(Error::Unsupported(format!(
                "sdxl ios pricing: {}x{} is not a multiple of the latent stride",
                req.geometry.width, req.geometry.height
            )));
        }
        let memory: &GenerationMemory = req.execution;
        let cfg_batching = sampling.cfg_batching();
        if cfg_batching != memory.cfg_batching.unwrap_or(CfgBatching::Batched) {
            return Err(Error::Unsupported(
                "sdxl pricing request and selected CFG batching differ".into(),
            ));
        }
        let tiling = if memory.tile_vae_decode {
            let (edge, overlap) = crate::memory_strategy::resolve_decode_geometry(
                false,
                memory.decode_tile_edge,
                memory.decode_overlap,
            );
            Some(mlx_gen::tiling::TilingConfig::spatial_only(
                edge.unwrap_or(crate::memory_strategy::DECODE_TILE_EDGE) as i32,
                overlap.unwrap_or(crate::memory_strategy::DECODE_OVERLAP) as i32,
            ))
        } else {
            None
        };
        if memory.chunk_attention {
            return Err(Error::Unsupported(
                "sdxl ios pricing: bounded attention is not selectable on this provider".into(),
            ));
        }
        if memory.stream_transformer_blocks && !crate::memory_strategy::streamable(req.spec) {
            return Err(Error::Unsupported(
                "sdxl ios pricing: bounded transformer residency needs a streamable load".into(),
            ));
        }
        let windows = (req.token_window as i32).max(77);
        let ip_scale = sampling.effective_ip_scale();
        Ok(Self {
            route,
            sampling,
            ip_scale,
            reference_image: work.reference_image,
            count,
            has_init,
            width: req.geometry.width as i32,
            height: req.geometry.height as i32,
            staged: memory.stage_residency,
            cfg_batching,
            tiling,
            context_tokens: windows,
        })
    }

    pub(crate) fn conditioning_rows(&self) -> i32 {
        if self.sampling.guidance() > 1.0 {
            2
        } else {
            1
        }
    }

    pub(crate) fn model_rows(&self) -> i32 {
        self.sampling.denoise_rows() as i32
    }

    pub(crate) fn initial_state_dtype(&self) -> Dtype {
        if self.sampling.route() == SamplingRoute::Ancestral && !self.has_init {
            crate::model::DTYPE
        } else {
            Dtype::Float32
        }
    }

    /// Persistent owners after the real model-input materialization barrier. Lightning's local
    /// scaled F32 intermediate is a PENDING graph obligation, not zero working set; it is not an
    /// additional persistent owner after its returned half input has materialized. Curated x_in
    /// is separately Rust-owned across the callback, even after the half cast materializes.
    pub(crate) fn forward_owners(&self, state: Dtype) -> q::ForwardInputOwnership {
        q::ForwardInputOwnership {
            state,
            scaled: self
                .sampling
                .retains_scaled_input_in_callback()
                .then_some(Dtype::Float32),
            cast_before_cfg: self.sampling.route() != SamplingRoute::Ancestral,
        }
    }

    pub(crate) fn initial_fixed_inputs(&self, latent_elements: u64) -> u64 {
        let init = if self.has_init {
            latent_elements.saturating_mul(q::dtype_bytes(crate::model::DTYPE))
        } else {
            0
        };
        let inpaint = if self.route == Route::Mask {
            latent_elements
                .saturating_mul(q::dtype_bytes(Dtype::Float32))
                .saturating_add(
                    (latent_elements / 4).saturating_mul(q::dtype_bytes(Dtype::Float32)),
                )
        } else {
            0
        };
        let enclosing_noise = if self.sampling.retains_initial_noise() {
            latent_elements.saturating_mul(q::dtype_bytes(Dtype::Float32))
        } else {
            0
        };
        init.saturating_add(inpaint).saturating_add(enclosing_noise)
    }
}

/// Fused source facts and deferred optional sources. Optional source resolution stays at the
/// native heavy-loader point, after both CLIP constructors/pack and the U-Net constructor.
pub(crate) struct FusedInputs {
    source: crate::ldm::pricing::FusedHeaderSource,
    vae_fix: Option<WeightsSource>,
    ip_source: Option<WeightsSource>,
}

/// The files and load facts one request's components come from.
pub(crate) struct Inputs {
    pub fused: Option<FusedInputs>,
    pub arch: Architecture,
    pub te1: PathBuf,
    pub te2: PathBuf,
    pub unet: PathBuf,
    /// `(file, dtype)` of the VAE the load runs (the staged fp16-fix at f16, else the in-snapshot
    /// one at f32).
    pub vae: (PathBuf, Dtype),
    pub controls: Vec<PathBuf>,
    /// `(image encoder file, IP weights file)`.
    pub ip: Option<(PathBuf, PathBuf)>,
    pub quant_bits: Option<i32>,
    /// Actual post-load CLIP pair policy. Production mirrors `spec.quantize`; explicit-config
    /// legacy TE8 fixtures leave this None because they never exercised a groupable CLIP pack.
    pub text_quant_bits: Option<i32>,
    /// Whether the load arms the rung-4 block stream ([`crate::memory_strategy::streamable`]).
    pub streamable: bool,
}

impl Inputs {
    /// Test access to the SAME served file/policy resolver. Tests may subsequently replace only
    /// this private pricing architecture with their existing explicit tiny configs; this does not
    /// inject configuration into Sdxl::generate or bypass its real request/model validation.
    #[cfg(test)]
    pub(crate) fn from_spec_for_test(spec: &LoadSpec, regime: &Regime) -> Result<Self> {
        Self::from_spec(spec, regime)
    }

    fn from_spec(spec: &LoadSpec, regime: &Regime) -> Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root,
            WeightsSource::File(file) => {
                if spec.precision != mlx_gen::gen_core::Precision::Bf16 {
                    return Err(Error::Msg(
                        "sdxl: precision override is not wired; the dense path runs fp16".into(),
                    ));
                }
                file
            }
        };
        if !spec.adapters.is_empty() {
            return Err(Error::Unsupported(
                "sdxl ios pricing: adapter (LoRA/LoKr/LoHa) installs are not priced".into(),
            ));
        }
        if spec.pid.is_some() {
            return Err(Error::Unsupported(
                "sdxl ios pricing: the PiD decoder overlay is not priced".into(),
            ));
        }
        if spec.decode_geometry_policy_authoritative {
            return Err(Error::Unsupported(
                "sdxl ios pricing: a sealed geometry decode policy is not priced".into(),
            ));
        }
        if regime.route == Route::Control
            && spec.control.is_none()
            && spec.extra_controls.is_empty()
        {
            return Err(Error::Unsupported(
                "sdxl ios pricing: a control request needs a loaded ControlNet".into(),
            ));
        }
        if regime.route == Route::IpAdapter && spec.ip_adapter.is_none() {
            return Err(Error::Unsupported(
                "sdxl ios pricing: an IP-Adapter request needs a loaded IP-Adapter".into(),
            ));
        }
        let fused = match &spec.weights {
            WeightsSource::Dir(_) => None,
            WeightsSource::File(file) => Some(FusedInputs {
                source: crate::ldm::pricing::FusedHeaderSource::read(file)?,
                vae_fix: spec
                    .components
                    .get(crate::loader::VAE_FP16_FIX_COMPONENT)
                    .cloned(),
                ip_source: spec.ip_adapter.clone(),
            }),
        };
        let dtype = crate::model::DTYPE;
        let resolve = |sub: &str, stem: &str| {
            if fused.is_some() {
                Ok(root.clone())
            } else {
                crate::loader::resolve_weight_file(root, sub, stem, dtype)
            }
        };
        let vae = if fused.is_some() {
            (root.clone(), Dtype::Float32)
        } else {
            match spec.components.get(crate::loader::VAE_FP16_FIX_COMPONENT) {
                Some(source) => (
                    crate::loader::resolve_vae_fp16_fix_file(source)?,
                    crate::loader::VAE_FP16_FIX_DTYPE,
                ),
                None => (
                    crate::loader::resolve_vae_weight_file(root)?,
                    Dtype::Float32,
                ),
            }
        };
        let controls = spec
            .control
            .iter()
            .chain(&spec.extra_controls)
            .map(|source| match source {
                WeightsSource::File(path) | WeightsSource::Dir(path) => path.clone(),
            })
            .collect();
        let ip = if fused.is_some() {
            None
        } else {
            match &spec.ip_adapter {
                Some(WeightsSource::Dir(dir)) => {
                    Some(crate::loader::resolve_ip_adapter_files(dir)?)
                }
                Some(WeightsSource::File(_)) => {
                    return Err(Error::Unsupported(
                        "sdxl ip_adapter expects an h94/IP-Adapter snapshot directory".into(),
                    ))
                }
                None => None,
            }
        };
        Ok(Self {
            te1: resolve("text_encoder", "model")?,
            te2: resolve("text_encoder_2", "model")?,
            unet: resolve("unet", "diffusion_pytorch_model")?,
            fused,
            arch: Architecture::sdxl(),
            vae,
            controls,
            ip,
            quant_bits: spec.quantize.map(|quant| quant.bits()),
            text_quant_bits: spec.quantize.map(|quant| quant.bits()),
            streamable: crate::memory_strategy::streamable(spec),
        })
    }
}

/// `(resident bytes after the load, the load's floor scratch)` of one component.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Footprint {
    pub resident: u64,
    pub floor: u64,
}

/// One complete source-logical pack point, never a scratch maximum paired with a different
/// stage's retained maximum. Pre-pack roots remain charged; no physical release is inferred.
pub(crate) struct LoadPackPoint {
    pub before_resident: u64,
    pub scratch: u64,
}

fn collect_pack_effect(
    path: &str,
    linear: &q::LinearWork,
    bits: i32,
    retained: &mut u64,
    outputs: &mut u64,
    points: &mut Vec<LoadPackPoint>,
    effects: &mut q::LinearEffects,
) -> Result<()> {
    let mut work = linear.clone();
    if work.bits.is_none() {
        let scratch = work.pack(bits)?;
        points.push(LoadPackPoint {
            before_resident: *retained,
            scratch,
        });
        // Only the base changes. Residual factors/deltas stay in their original owner set.
        let base_outputs = work
            .weights
            .iter()
            .chain(work.bias.iter())
            .fold(0u64, |n, a| {
                n.saturating_add(mlx_gen::ios_quantum::allocated_bytes(a.bytes()))
            });
        *retained = retained.saturating_add(base_outputs);
        *outputs = outputs.saturating_add(base_outputs);
    }
    if effects.0.insert(path.to_owned(), work).is_some() {
        return Err(Error::Msg(format!(
            "sdxl duplicate actual pack-walk target: {path}"
        )));
    }
    Ok(())
}

/// Every served component here is metadata; native execution remains in the actual loaders.
/// All footprints are header-only. The heavy pack collector
/// below is metadata-only and never quantizes/evaluates those graphs to discover pricing.
pub(crate) struct Priced {
    fused: Option<FusedLoadFacts>,
    te: (ClipPrice, ClipPrice),
    unet: UNetFacts,
    vae: AutoencoderFacts,
    controls: Vec<ControlFacts>,
    ip: Option<IpImageEncoderFacts>,
    /// Pre-pack read/cast floors and declared owners, including TE1 while TE2 is loaded.
    pub te_bytes: [Footprint; 2],
    /// Pure-header post-pack declared owners and per-projection coincident raw budgets.
    /// These are not physical lower/backing certificates; pre-pack/lazy read parents stay above.
    pub te_pack: [crate::text_encoder::TextPackFacts; 2],
    pub unet_bytes: Footprint,
    pub vae_bytes: Footprint,
    /// Per branch: `(load footprint at the compute dtype, resident after the tier quantization)`.
    pub control_bytes: Vec<(Footprint, u64)>,
    pub ip_encoder_bytes: Footprint,
    /// The IP weights file's load footprint (Resampler + K/V pairs at the compute dtype).
    pub ip_adapter_bytes: Footprint,
    /// The Resampler's resident bytes (shed with the image encoder after the image prompt).
    pub ip_resampler_bytes: u64,
    /// The installed K/V pairs' resident bytes (quantized with the U-Net at a packed tier).
    pub ip_kv_bytes: u64,
    /// Each actual heavy projection pack with its coincident declared logical owners. Source/cast
    /// roots are conservatively retained; these points grant no native backing-release credit.
    pub load_quantize: Vec<LoadPackPoint>,
    pub(crate) unet_effects: Option<q::LinearEffects>,
    pub(crate) control_effects: Vec<Option<q::LinearEffects>>,
    unet_retained: u64,
    adapter_load: AdapterLoadLedger,
    arch: Architecture,
}

fn footprint(files: &[PathBuf], dtype: Dtype, keep: impl Fn(&str) -> bool) -> Result<Footprint> {
    let (resident, floor) = crate::loader::cast_load_parts(files, dtype, true, keep)?;
    Ok(Footprint { resident, floor })
}

/// Source-parent terms stay logical and unclassified by the Directory physical bounder.
/// No source/view logical product establishes a native backing size or retirement time.
struct FusedLoadFacts {
    text_floor: u64,
    heavy_floor: u64,
    text_parents: u64,
    heavy_parents: u64,
    external_vae: bool,
    native_backing: FusedNativeBacking,
}
#[derive(Clone, Copy)]
enum FusedNativeBacking {
    Unbound,
}
fn fused_header_footprint(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    dtype: Dtype,
    all_cast: bool,
) -> Result<Footprint> {
    let mut resident = 0u64;
    for header in headers {
        let loaded = if all_cast {
            dtype
        } else {
            crate::quant::loaded_header_dtype(headers, header)?
        };
        let bytes = header
            .materialized_bytes(q::dtype_bytes(loaded))
            .map_err(|e| Error::Msg(e.to_string()))?;
        resident = resident.saturating_add(mlx_gen::ios_quantum::allocated_bytes(bytes));
    }
    Ok(Footprint { resident, floor: 0 })
}

/// The sinusoid tables a ControlNet constructor builds beside its file's tensors (the timestep and
/// `text_time` tables, `unet::embeddings`), in the small-buffer page they open.
fn control_tables(cfg: &crate::config::UNetConfig) -> u64 {
    use crate::unet::SinusoidalPositionalEncoding as Sinusoid;
    mlx_gen::ios_quantum::PAGE_BYTES
        + Sinusoid::timestep_table_bytes(cfg.block_out_channels[0])
        + Sinusoid::timestep_table_bytes(cfg.addition_time_embed_dim.unwrap_or(256))
}

/// Only the premature heavy metadata pack is deferred. Text finalization, IP construction,
/// native constructors and all final post-install pack validation keep their actual order.
pub(crate) fn initial_heavy_pack(
    configured: Option<&LoadSpec>,
    pack: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if configured.is_some_and(|spec| !spec.adapters.is_empty()) {
        Ok(())
    } else {
        pack()
    }
}

impl Priced {
    pub(crate) fn build(inputs: &Inputs) -> Result<Self> {
        Self::build_with_pack_timing(inputs, None)
    }

    pub(crate) fn build_configured(inputs: &Inputs, spec: &LoadSpec) -> Result<Self> {
        Self::build_with_pack_timing(inputs, Some(spec))
    }

    fn build_with_pack_timing(inputs: &Inputs, configured: Option<&LoadSpec>) -> Result<Self> {
        let dtype = crate::model::DTYPE;
        let arch = inputs.arch.clone();
        let one = |p: &Path| vec![p.to_path_buf()];
        let headers = |file: &Path| {
            mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(file)
                .map_err(|e| Error::Msg(e.to_string()))
        };
        use mlx_gen::gen_core::sdxl_ldm::SdxlComponent as C;
        let component_headers = |file: &Path, component: C| match &inputs.fused {
            Some(fused) => Ok(fused.source.headers(component)),
            None => headers(file),
        };
        let te1_headers = component_headers(&inputs.te1, C::ClipL)?;
        let mut te1 = ClipPrice::from_headers(&te1_headers, "text_model", &arch.te1)?;
        let te2_headers = component_headers(&inputs.te2, C::ClipBigG)?;
        let mut te2 = ClipPrice::from_headers(&te2_headers, "text_model", &arch.te2)?;
        let te_bytes = if inputs.fused.is_some() {
            [
                fused_header_footprint(&te1_headers, dtype, false)?,
                fused_header_footprint(&te2_headers, dtype, false)?,
            ]
        } else {
            [
                footprint(&one(&inputs.te1), dtype, |_| true)?,
                footprint(&one(&inputs.te2), dtype, |_| true)?,
            ]
        };
        let te_pack = [
            crate::text_encoder::text_pack_facts(&te1_headers, &arch.te1, inputs.text_quant_bits)?,
            crate::text_encoder::text_pack_facts(&te2_headers, &arch.te2, inputs.text_quant_bits)?,
        ];
        if let Some(bits) = inputs.text_quant_bits {
            // Both header constructors precede the actual pair's ordered base-only metadata pack.
            // No native CLIP/Weights/Array, quantizer interception or readiness query is involved.
            te1.quantize(bits)?;
            te2.quantize(bits)?;
        }

        let unet_headers = component_headers(&inputs.unet, C::Unet)?;
        let mut unet = UNetFacts::from_headers(&unet_headers, &arch.unet)?;
        let unet_bytes = if inputs.fused.is_some() {
            let mut load = fused_header_footprint(&unet_headers, dtype, false)?;
            use crate::unet::SinusoidalPositionalEncoding as Sinusoid;
            load.resident = load
                .resident
                .saturating_add(Sinusoid::timestep_table_bytes(
                    arch.unet.block_out_channels[0],
                ))
                .saturating_add(Sinusoid::timestep_table_bytes(
                    arch.unet.addition_time_embed_dim.unwrap_or(256),
                ));
            load
        } else {
            let (resident, floor) = crate::loader::unet_load_parts(
                &one(&inputs.unet),
                dtype,
                true,
                inputs.streamable,
                &arch.unet,
            )?;
            Footprint { resident, floor }
        };

        // Fused heavy constructs U-Net, then resolves/materializes its VAE, BEFORE adapters.
        // An external fix replaces the fused VAE unread; never run a fused VAE constructor first.
        let external_vae = inputs
            .fused
            .as_ref()
            .and_then(|fused| fused.vae_fix.as_ref());
        let selected_vae = external_vae
            .map(crate::loader::resolve_vae_fp16_fix_file)
            .transpose()?;
        let vae_file = selected_vae.as_deref().unwrap_or(&inputs.vae.0);
        let vae_dtype = if external_vae.is_some() {
            crate::loader::VAE_FP16_FIX_DTYPE
        } else {
            inputs.vae.1
        };
        let vae_headers = if inputs.fused.is_some() && external_vae.is_none() {
            component_headers(vae_file, C::Vae)?
        } else {
            headers(vae_file)?
        };
        let vae = AutoencoderFacts::from_headers(&vae_headers, &arch.vae, vae_dtype)?;
        let vae_bytes = if inputs.fused.is_some() && external_vae.is_none() {
            fused_header_footprint(&vae_headers, vae_dtype, true)?
        } else {
            footprint(&one(vae_file), vae_dtype, |_| true)?
        };

        // All loads precede the actual ordered heavy pack walk.
        let mut load_quantize = Vec::new();
        let mut controls = Vec::new();
        let mut control_bytes = Vec::new();
        for file in &inputs.controls {
            let control = ControlFacts::from_headers(&headers(file)?, &arch.unet)?;
            // The guard checks the header floor; the constructor's tables stay resident beside it.
            let mut load = footprint(&one(file), dtype, |_| true)?;
            load.resident += control_tables(&arch.unet);
            let resident = load.resident;
            controls.push(control);
            control_bytes.push((load, resident));
        }

        let (mut ip, mut ip_encoder_bytes, mut ip_adapter_bytes) =
            (None, Footprint::default(), Footprint::default());
        let (mut ip_resampler_bytes, mut ip_kv_bytes) = (0, 0);
        let fused_ip = match inputs
            .fused
            .as_ref()
            .and_then(|fused| fused.ip_source.as_ref())
        {
            Some(WeightsSource::Dir(path)) => Some(crate::loader::resolve_ip_adapter_files(path)?),
            Some(WeightsSource::File(_)) => {
                return Err(Error::Msg(
                    "sdxl ip_adapter expects an h94/IP-Adapter snapshot directory".into(),
                ))
            }
            None => None,
        };
        if let Some((encoder_file, ip_file)) = fused_ip.as_ref().or(inputs.ip.as_ref()) {
            let encoder_headers = headers(encoder_file)?;
            let encoder = VisionFacts::from_weights(
                &crate::unet::pricing::HeaderWeights::new(&encoder_headers)?,
                &arch.vision,
            )?;
            let ip_headers = headers(ip_file)?;
            let resampler = ResamplerFacts::from_weights(
                &crate::unet::pricing::HeaderWeights::new(&ip_headers)?,
                "image_proj",
                &arch.resampler,
            )?;
            unet.install_ip_adapter(crate::unet::pricing::ip_pairs_from_headers(&ip_headers)?)?;
            ip = Some(IpImageEncoderFacts::with_image_size(
                encoder,
                resampler,
                arch.vision.image_size as usize,
            ));
            ip_encoder_bytes = footprint(&one(encoder_file), dtype, |_| true)?;
            ip_adapter_bytes = footprint(&one(ip_file), dtype, |_| true)?;
            ip_resampler_bytes =
                footprint(&one(ip_file), dtype, |name| name.starts_with("image_proj."))?.resident;
            // Raw IP source owners remain beside later packed base outputs; no release credit.
            ip_kv_bytes =
                footprint(&one(ip_file), dtype, |name| name.starts_with("ip_adapter."))?.resident;
        }
        let mut unet_effects = None;
        let mut control_effects = Vec::new();
        let mut unet_retained = unet_bytes.resident;
        if let Some(bits) = inputs.quant_bits {
            initial_heavy_pack(configured, || {
                let mut retained = unet_bytes
                    .resident
                    .saturating_add(vae_bytes.resident)
                    .saturating_add(
                        control_bytes
                            .iter()
                            .map(|(load, _)| load.resident)
                            .sum::<u64>(),
                    )
                    .saturating_add(ip_encoder_bytes.resident)
                    .saturating_add(ip_adapter_bytes.resident);
                let mut effects = q::LinearEffects::default();
                let mut outputs = 0;
                // This is the ACTUAL ordered module walk, with a pure metadata visitor. The borrowed
                // constructor facts stay untouched: no quantize_linear, Array creation, eval or read.
                unet.quantize_using(bits, &mut |path, linear, bits| {
                    collect_pack_effect(
                        path,
                        linear,
                        bits,
                        &mut retained,
                        &mut outputs,
                        &mut load_quantize,
                        &mut effects,
                    )
                })?;
                unet_retained = unet_retained.saturating_add(outputs);
                unet_effects = Some(effects);
                for (index, control) in controls.iter_mut().enumerate() {
                    let mut effects = q::LinearEffects::default();
                    let mut outputs = 0;
                    control.quantize_using(bits, &mut |path, linear, bits| {
                        collect_pack_effect(
                            path,
                            linear,
                            bits,
                            &mut retained,
                            &mut outputs,
                            &mut load_quantize,
                            &mut effects,
                        )
                    })?;
                    control_bytes[index].1 =
                        control_bytes[index].0.resident.saturating_add(outputs);
                    control_effects.push(Some(effects));
                }
                Ok(())
            })?;
        }
        // A deferred configured walk keeps RAW owners until its selected post-install pack.
        // Ordinary/no-adapter builds already filled this vector, so resize is a no-op there.
        control_effects.resize_with(controls.len(), || None);
        if inputs.streamable {
            unet.arm_block_streams(&unet_headers, dtype, inputs.quant_bits)?;
        }
        let fused = inputs
            .fused
            .as_ref()
            .map(|fused| {
                let heavy_components = if fused.vae_fix.is_some() {
                    &[C::Unet][..]
                } else {
                    &[C::Unet, C::Vae][..]
                };
                Ok::<_, Error>(FusedLoadFacts {
                    text_floor: crate::ldm::component_load_floor_from_headers(
                        &fused.source.raw,
                        &[C::ClipL, C::ClipBigG],
                    )?,
                    heavy_floor: crate::ldm::component_load_floor_from_headers(
                        &fused.source.raw,
                        heavy_components,
                    )?,
                    text_parents: fused.source.raw_parent_bytes(&[C::ClipL, C::ClipBigG]),
                    // During U-Net construction even a later-replaced VAE map still exists.
                    heavy_parents: fused.source.raw_parent_bytes(&[C::Unet, C::Vae]),
                    external_vae: fused.vae_fix.is_some(),
                    native_backing: FusedNativeBacking::Unbound,
                })
            })
            .transpose()?;
        Ok(Self {
            fused,
            te: (te1, te2),
            unet,
            vae,
            controls,
            ip,
            te_bytes,
            te_pack,
            unet_bytes,
            vae_bytes,
            control_bytes,
            ip_encoder_bytes,
            ip_adapter_bytes,
            ip_resampler_bytes,
            ip_kv_bytes,
            load_quantize,
            unet_effects,
            control_effects,
            unet_retained,
            adapter_load: AdapterLoadLedger::default(),
            arch,
        })
    }

    /// Consume a complete selected world at the real adapter boundary. Each world is replayed
    /// through selected dispatch, the load ledger, actual target storage and the SAME ordered base
    /// pack walk, then through the whole forward before any envelope is taken. The native pricing
    /// model facts are never adapted/packed to discover values. Remaining IP graphs stay separate.
    pub(crate) fn price_configured(
        &mut self,
        spec: &LoadSpec,
        unet_file: &Path,
        bits: Option<i32>,
        regime: &Regime,
    ) -> Result<IosPricing> {
        self.price_configured_observing(spec, unet_file, bits, regime, |_, _, _, _| Ok(()))
    }

    /// Borrowed selected-world view at the actual consumer boundary. The observer runs after the
    /// whole forward price, before state restoration/enveloping, and cannot change target facts or
    /// retain native owners. Production supplies a no-op; CPU fixtures query native owners apart.
    pub(crate) fn price_configured_observing(
        &mut self,
        spec: &LoadSpec,
        unet_file: &Path,
        bits: Option<i32>,
        regime: &Regime,
        mut observe: impl FnMut(
            &crate::adapters::AdapterOutcome<'_>,
            &Self,
            &AdapterLoadLedger,
            &IosPricing,
        ) -> Result<()>,
    ) -> Result<IosPricing> {
        if spec.adapters.is_empty() {
            return Ok(self.price(regime)?.0);
        }
        let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(unet_file)
            .map_err(|e| Error::Msg(e.to_string()))?;
        let config = self.arch.unet.clone();
        let mut envelope: std::collections::BTreeMap<
            (IosPhase, Option<String>, String, String, usize),
            IosWindowPrice,
        > = std::collections::BTreeMap::new();
        let mut accepted = false;
        let mut first_refusal = None;
        let mut absent = None;
        crate::adapters::visit_configured_adapter_outcomes(spec, &headers, &config, |outcome| {
            let ledger =
                AdapterLoadLedger::from_work_checked(outcome.work, self.unet_bytes.resident)?;
            let pricing = if let Some(error) = &outcome.refusal {
                first_refusal.get_or_insert_with(|| error.clone());
                // Dir installs precede the VAE/control/IP loads. A source refusal never fabricates
                // those loads, a successful target result, a post-install pack or a forward branch.
                let warm = if regime.staged { 0 } else { self.text() };
                let mut prefix = IosPricing::default();
                prefix.take(
                    IosPhase::Load,
                    0,
                    self.te_bytes[0].floor,
                    q::TEXT_ENCODER_1_LOAD,
                );
                prefix.take(
                    IosPhase::Load,
                    self.te_bytes[0].resident,
                    self.te_bytes[1].floor,
                    q::TEXT_ENCODER_2_LOAD,
                );
                for (encoder, facts) in self.te_pack.iter().enumerate() {
                    let other = if encoder == 0 {
                        self.te_bytes[1].resident
                    } else {
                        self.te_pack[0].retained
                    };
                    for point in &facts.points {
                        prefix.take(
                            IosPhase::Load,
                            other.saturating_add(point.before_resident),
                            point.scratch,
                            q::LOAD_QUANTIZE,
                        );
                    }
                }
                prefix.take(IosPhase::Load, warm, self.unet_bytes.floor, q::UNET_LOAD);
                for point in ledger.quanta {
                    prefix.take(
                        point.phase,
                        warm.saturating_add(point.resident),
                        point.scratch,
                        point.label,
                    );
                }
                prefix
            } else {
                let saved_effects = self.unet_effects.take();
                let saved_controls = std::mem::take(&mut self.control_effects);
                let saved_packs = std::mem::take(&mut self.load_quantize);
                let saved_adapter = std::mem::replace(&mut self.adapter_load, ledger);
                let saved_retained = self.unet_retained;
                let saved_control_resident: Vec<_> =
                    self.control_bytes.iter().map(|(_, r)| *r).collect();
                let selected = (|| -> Result<IosPricing> {
                    use crate::adapters::{AdapterInstallStorage as S, AdapterWork};
                    let mut effects = q::LinearEffects::default();
                    self.unet
                        .quantize_using(bits.unwrap_or(4), &mut |path, linear, _| {
                            if effects.0.insert(path.to_owned(), linear.clone()).is_some() {
                                return Err(Error::Msg(format!(
                                    "sdxl duplicate adapter base target: {path}"
                                )));
                            }
                            Ok(())
                        })?;
                    for item in outcome.work {
                        let AdapterWork::Install(install) = item else {
                            continue;
                        };
                        let path: Vec<_> = install.target.split('.').collect();
                        if install.storage == S::DenseConv {
                            self.unet.adaptable_conv_mut(&path).ok_or_else(|| {
                                Error::Msg(format!(
                                    "sdxl missing actual selected convolution: {}",
                                    install.target
                                ))
                            })?;
                            let weight = install.installed.first().ok_or_else(|| {
                                Error::Msg(
                                    "sdxl selected convolution lacks its declared OHWI result"
                                        .into(),
                                )
                            })?;
                            effects.1.insert(install.target.clone(), weight.clone());
                        } else {
                            if !effects.0.contains_key(&install.target) {
                                // Dense shortcut is deliberately absent from the base pack walk.
                                let native = self.unet.adaptable_mut(&path).ok_or_else(|| {
                                    Error::Msg(format!(
                                        "sdxl missing actual selected projection: {}",
                                        install.target
                                    ))
                                })?;
                                effects.0.insert(install.target.clone(), native.clone());
                            }
                            effects
                                .0
                                .get_mut(&install.target)
                                .expect("inserted selected target")
                                .install(install)?;
                        }
                    }
                    let adapter_retained = self.adapter_load.retained_upper();
                    self.unet_retained = self.unet_bytes.resident.saturating_add(adapter_retained);
                    let mut retained = self
                        .unet_retained
                        .saturating_add(self.vae_bytes.resident)
                        .saturating_add(
                            self.control_bytes
                                .iter()
                                .map(|(load, _)| load.resident)
                                .sum::<u64>(),
                        )
                        .saturating_add(self.ip_encoder_bytes.resident)
                        .saturating_add(self.ip_adapter_bytes.resident);
                    if let Some(bits) = bits {
                        self.unet.quantize_using(bits, &mut |path, _, bits| {
                            let projection = effects.0.get_mut(path).ok_or_else(|| {
                                Error::Msg(format!(
                                    "sdxl selected world missing ordered pack target: {path}"
                                ))
                            })?;
                            if projection.bits.is_none() {
                                let scratch = projection.pack(bits)?;
                                self.load_quantize.push(LoadPackPoint {
                                    before_resident: retained,
                                    scratch,
                                });
                                // Whole declared projection, including unchanged residual owners,
                                // remains an upper. No upper subtraction or backing release credit.
                                let outputs = projection.declared_bytes();
                                retained = retained.saturating_add(outputs);
                                self.unet_retained = self.unet_retained.saturating_add(outputs);
                            }
                            Ok(())
                        })?;
                        for (index, control) in self.controls.iter_mut().enumerate() {
                            let mut control_effects = q::LinearEffects::default();
                            let mut outputs = 0;
                            control.quantize_using(bits, &mut |path, linear, bits| {
                                collect_pack_effect(
                                    path,
                                    linear,
                                    bits,
                                    &mut retained,
                                    &mut outputs,
                                    &mut self.load_quantize,
                                    &mut control_effects,
                                )
                            })?;
                            self.control_bytes[index].1 =
                                self.control_bytes[index].0.resident.saturating_add(outputs);
                            self.control_effects.push(Some(control_effects));
                        }
                    } else {
                        self.control_effects
                            .resize_with(self.controls.len(), || None);
                    }
                    self.unet_effects = Some(effects);
                    let candidate = self.price(regime)?.0;
                    observe(&outcome, self, &self.adapter_load, &candidate)?;
                    Ok(candidate)
                })();
                self.unet_effects = saved_effects;
                self.control_effects = saved_controls;
                self.load_quantize = saved_packs;
                self.adapter_load = saved_adapter;
                self.unet_retained = saved_retained;
                for ((_, resident), saved) in
                    self.control_bytes.iter_mut().zip(saved_control_resident)
                {
                    *resident = saved;
                }
                let selected = selected?;
                accepted = true;
                absent.get_or_insert_with(|| selected.absent_phases.clone());
                selected
            };
            // Keep a whole coincident candidate, never max(resident) + max(scratch) from different
            // worlds. Ordinals distinguish repeated same-label epochs without retaining world lists.
            let mut ordinals = std::collections::BTreeMap::new();
            for window in pricing.windows {
                let identity = (
                    window.phase,
                    window.class.clone(),
                    window.label.clone(),
                    window.owner_domain.clone(),
                );
                let ordinal = ordinals.entry(identity.clone()).or_insert(0usize);
                let key = (identity.0, identity.1, identity.2, identity.3, *ordinal);
                *ordinal += 1;
                match envelope.entry(key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(window);
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        if window.logical_peak_bytes() > entry.get().logical_peak_bytes() {
                            entry.insert(window);
                        }
                    }
                }
            }
            Ok(())
        })?;
        if !accepted {
            return Err(Error::Msg(
                first_refusal.unwrap_or_else(|| "sdxl no complete adapter world".into()),
            ));
        }
        let mut pricing = IosPricing::default();
        for (_, window) in envelope {
            pricing.take_window(window);
        }
        pricing.absent_phases = absent.unwrap_or_default();
        Ok(pricing)
    }

    /// The heavy bundle's resident bytes in the denoise (`shed_ip_encoder`: lever d) and the
    /// load order's running totals.
    fn heavy(&self, shed_ip_encoder: bool) -> u64 {
        let controls: u64 = self.control_bytes.iter().map(|(_, r)| r).sum();
        let ip = if self.ip.is_some() {
            let encoder = if shed_ip_encoder {
                0
            } else {
                self.ip_encoder_bytes.resident + self.ip_resampler_bytes
            };
            encoder + self.ip_kv_bytes
        } else {
            0
        };
        self.unet_retained + self.vae_bytes.resident + controls + ip
    }

    fn text(&self) -> u64 {
        self.te_pack[0].retained + self.te_pack[1].retained
    }

    /// The actual encoder pair's header-only conditioning flow after its ordered metadata finalize.
    /// No fixed tier/compute dtype substitutes for packed projection promotion.
    pub(crate) fn price_text_conditioning(&self, rows: i32, out: &mut Quanta) -> (Dtype, Dtype) {
        let (h1, _) = self.te.0.price_forward(rows, self.arch.text_tokens, out);
        let (h2, pooled) = self.te.1.price_forward(rows, self.arch.text_tokens, out);
        (Dtype::from_promoting_types(h1, h2), pooled)
    }

    /// Actual fused warm/text/heavy load recipe. There are no Directory component-load guards
    /// on from-weights constructors. Source parents are logical, never a backing-size credit.
    fn fused_load_points(&self, regime: &Regime) -> Option<Vec<PricedQuantum>> {
        let fused = self.fused.as_ref()?;
        let FusedNativeBacking::Unbound = fused.native_backing;
        let mut points = Vec::new();
        let mut push = |resident, label, scratch| {
            points.push(PricedQuantum {
                phase: IosPhase::Load,
                resident,
                label,
                scratch,
            })
        };
        let warm_text = if regime.staged { 0 } else { self.text() };
        let all_parents = fused.text_parents.saturating_add(fused.heavy_parents);
        let text_parents = if regime.staged {
            fused.text_parents
        } else {
            all_parents
        };
        push(0, q::LDM_TEXT_LOAD, fused.text_floor);
        // Warm retains raw U-Net/VAE maps during both CLIP constructors and pair pack.
        // SequentialText discards sibling maps before those constructors. Both constructors
        // still precede TE1→TE2 pack; raw source-parent terms grant no release/donation credit.
        for (encoder, facts) in self.te_pack.iter().enumerate() {
            let other = if encoder == 0 {
                self.te_bytes[1].resident
            } else {
                self.te_pack[0].retained
            };
            for point in &facts.points {
                push(
                    text_parents
                        .saturating_add(other)
                        .saturating_add(point.before_resident),
                    q::LOAD_QUANTIZE,
                    point.scratch,
                );
            }
        }
        let conditioning = if regime.staged {
            let mut unused = Quanta::new();
            let (context, pooled) =
                self.price_text_conditioning(regime.conditioning_rows(), &mut unused);
            mlx_gen::ios_quantum::tensor_bytes(
                q::dtype_bytes(context),
                &[
                    regime.conditioning_rows() as u64,
                    regime.context_tokens as u64,
                    self.arch.context_dim as u64,
                ],
            )
            .saturating_add(mlx_gen::ios_quantum::tensor_bytes(
                q::dtype_bytes(pooled),
                &[
                    regime.conditioning_rows() as u64,
                    self.te.1.pooled_width() as u64,
                ],
            ))
        } else {
            0
        };
        // SequentialHeavy reopens/re-splits the same file; the warm arm uses its existing maps.
        // Dropping the staged text handles/evaluating conditioning is not a native backing
        // retirement certificate. Keep its source-parent term at the heavy-load boundary.
        push(
            warm_text
                .saturating_add(conditioning)
                .saturating_add(if regime.staged {
                    fused.text_parents
                } else {
                    all_parents
                }),
            q::LDM_HEAVY_LOAD,
            fused.heavy_floor,
        );
        let base = warm_text
            .saturating_add(conditioning)
            .saturating_add(all_parents);
        let mut loaded = base.saturating_add(self.unet_bytes.resident);
        if fused.external_vae {
            push(loaded, q::VAE_FP16_FIX_LOAD, self.vae_bytes.floor);
        }
        loaded = loaded.saturating_add(self.vae_bytes.resident);
        // FusedHeavy resolves/materializes VAE/fix BEFORE adapters, unlike Directory.
        for point in &self.adapter_load.quanta {
            push(
                base.saturating_add(self.vae_bytes.resident)
                    .saturating_add(point.resident),
                point.label,
                point.scratch,
            );
        }
        loaded = loaded.saturating_add(self.adapter_load.retained_upper());
        for (load, _) in &self.control_bytes {
            push(loaded, q::CONTROLNET_LOAD, load.floor);
            loaded = loaded.saturating_add(load.resident);
        }
        if self.ip.is_some() {
            push(
                loaded,
                q::IP_IMAGE_ENCODER_LOAD,
                self.ip_encoder_bytes.floor,
            );
            loaded = loaded.saturating_add(self.ip_encoder_bytes.resident);
            push(loaded, q::IP_ADAPTER_LOAD, self.ip_adapter_bytes.floor);
        }
        for point in &self.load_quantize {
            push(
                base.saturating_add(point.before_resident),
                q::LOAD_QUANTIZE,
                point.scratch,
            );
        }
        Some(points)
    }

    /// Price `regime`: the per-phase pricing and every `(phase, resident, label, scratch)` quantum.
    pub(crate) fn price(&self, regime: &Regime) -> Result<(IosPricing, Vec<PricedQuantum>)> {
        let mut items: Vec<PricedQuantum> = Vec::new();
        let mut take = |phase: IosPhase, resident: u64, quanta: Quanta| {
            for (label, scratch) in quanta {
                items.push(PricedQuantum {
                    phase,
                    resident,
                    label,
                    scratch,
                });
            }
        };
        let staged = regime.staged;
        let warm_text = if staged { 0 } else { self.text() };

        if let Some(points) = self.fused_load_points(regime) {
            for point in points {
                take(
                    point.phase,
                    point.resident,
                    vec![(point.label, point.scratch)],
                );
            }
        } else {
            // Load floors in load order: the text encoders (staged: into an empty process), then the
            // heavy bundle — the U-Net, the VAE, the ControlNets (each at the compute dtype until the
            // bundle's tier quantization), the IP image encoder and IP weights.
            take(
                IosPhase::Load,
                0,
                vec![(q::TEXT_ENCODER_1_LOAD, self.te_bytes[0].floor)],
            );
            take(
                IosPhase::Load,
                self.te_bytes[0].resident,
                vec![(q::TEXT_ENCODER_2_LOAD, self.te_bytes[1].floor)],
            );
            // Runtime packs only AFTER both CLIP loads. At each point retain the other encoder at
            // its actual current stage (TE2 still unquantized during TE1, finalized TE1 during TE2).
            // Do not use pre-pack/final maxima interchangeably: tiny page-rounded packs can grow.
            for (encoder, facts) in self.te_pack.iter().enumerate() {
                let other = if encoder == 0 {
                    self.te_bytes[1].resident
                } else {
                    self.te_pack[0].retained
                };
                for point in &facts.points {
                    take(
                        IosPhase::Load,
                        other + point.before_resident,
                        vec![(q::LOAD_QUANTIZE, point.scratch)],
                    );
                }
            }
            let mut loaded = warm_text;
            take(
                IosPhase::Load,
                loaded,
                vec![(q::UNET_LOAD, self.unet_bytes.floor)],
            );
            loaded += self.unet_bytes.resident;
            for point in &self.adapter_load.quanta {
                take(
                    IosPhase::Load,
                    warm_text.saturating_add(point.resident),
                    vec![(point.label, point.scratch)],
                );
            }
            loaded = loaded.saturating_add(self.adapter_load.retained_upper());
            let vae_label = if self.vae.dtype() == Dtype::Float32 {
                q::VAE_LOAD
            } else {
                q::VAE_FP16_FIX_LOAD
            };
            take(
                IosPhase::Load,
                loaded,
                vec![(vae_label, self.vae_bytes.floor)],
            );
            loaded += self.vae_bytes.resident;
            for (load, _) in &self.control_bytes {
                take(
                    IosPhase::Load,
                    loaded,
                    vec![(q::CONTROLNET_LOAD, load.floor)],
                );
                loaded += load.resident;
            }
            if self.ip.is_some() {
                take(
                    IosPhase::Load,
                    loaded,
                    vec![(q::IP_IMAGE_ENCODER_LOAD, self.ip_encoder_bytes.floor)],
                );
                loaded += self.ip_encoder_bytes.resident;
                take(
                    IosPhase::Load,
                    loaded,
                    vec![(q::IP_ADAPTER_LOAD, self.ip_adapter_bytes.floor)],
                );
            }

            for point in &self.load_quantize {
                take(
                    IosPhase::Load,
                    warm_text.saturating_add(point.before_resident),
                    vec![(q::LOAD_QUANTIZE, point.scratch)],
                );
            }
        }
        let heavy = self.heavy(false);
        let (h, w) = (regime.height, regime.width);
        let (lh, lw) = (h / self.arch.latent_scale, w / self.arch.latent_scale);
        let (ch, cw) = (lh * self.arch.control_scale, lw * self.arch.control_scale);
        // The request's input tensors, live from their upload (after the loads) through the quanta
        // that consume them: the f32 init image of an img2img / inpaint encode; each control image
        // and its CFG pair (`preprocess_control_image`, then the batched copy the lazy pair keeps
        // the original for); the IP image prompt's f32 pixels.
        let f32_bytes = |shape: &[i32]| -> u64 {
            mlx_gen::ios_quantum::tensor_bytes(
                4,
                &shape.iter().map(|d| (*d).max(0) as u64).collect::<Vec<_>>(),
            )
        };
        let init_image = if regime.has_init {
            f32_bytes(&[1, h, w, 3])
        } else {
            0
        };
        let control_images = if regime.route == Route::Control {
            f32_bytes(&[1 + regime.conditioning_rows(), ch, cw, 3])
        } else {
            0
        };
        let ip_pixels = match (&self.ip, regime.route) {
            (Some(_), Route::IpAdapter) => {
                let size = self.arch.vision.image_size;
                f32_bytes(&[1, size, size, 3])
            }
            _ => 0,
        };
        let inputs = init_image + control_images + ip_pixels;
        if inputs > 0 {
            take(
                IosPhase::Load,
                heavy + warm_text + inputs,
                vec![(REQUEST_INPUTS, 0)],
            );
        }
        // Conditioning: both CLIP towers over the guided pair's two token rows, one window at a
        // time (each window is its own forward).
        let rows = regime.conditioning_rows();
        let mut text = Quanta::new();
        let (conditioning_dtype, pooled) = self.price_text_conditioning(rows, &mut text);
        let text_resident = self.text() + if staged { 0 } else { heavy };
        take(IosPhase::Conditioning, text_resident, text);

        if let (Some(ip), Route::IpAdapter) = (&self.ip, regime.route) {
            let source = regime.reference_image.ok_or_else(|| {
                Error::Unsupported("sdxl IP preprocessing has no actual source facts".into())
            })?;
            let geometry = ip.preprocess_geometry(source)?;
            take(
                IosPhase::Conditioning,
                heavy + warm_text + ip_pixels + geometry.source_buffer_bytes,
                vec![(IP_PREPROCESS_HOST, geometry.host_requested_upper()?)],
            );
        }
        let ip_tokens = match (&self.ip, regime.route) {
            (Some(ip), Route::IpAdapter) => {
                let mut quanta = Quanta::new();
                let tokens = ip.price_tokens(&mut quanta)?;
                if rows > 1 {
                    quanta.push((q::IP_CFG_TOKENS, q::ip_cfg_tokens_bytes_metadata(&tokens)));
                }
                take(
                    IosPhase::Conditioning,
                    heavy + warm_text + ip_pixels,
                    quanta,
                );
                Some(crate::ip_adapter::pricing::cfg_pair(&tokens, rows > 1)?)
            }
            _ => None,
        };

        // Encode: the img2img / inpaint init latent, or each ControlNet's conditioning embedding.
        let mut cond_embeds = Vec::new();
        if regime.has_init {
            let mut quanta = Quanta::new();
            let tile = regime
                .tiling
                .as_ref()
                .and_then(|cfg| cfg.spatial.as_ref().map(|s| s.tile_px))
                .unwrap_or(crate::vae::IOS_ENCODE_TILE_PX);
            self.vae.price_encode(
                &price_tensor(&[1, h, w, 3], Dtype::Float32)?,
                tile,
                &mut quanta,
            )?;
            take(IosPhase::Encode, heavy + warm_text + init_image, quanta);
        }
        if regime.route == Route::Control {
            let mut quanta = Quanta::new();
            for control in &self.controls {
                cond_embeds.push(control.price_embed_cond(
                    &price_tensor(&[rows, ch, cw, 3], Dtype::Float32)?,
                    &mut quanta,
                )?);
            }
            // Conservatively: one branch's control image pair (a branch's images are released once
            // its embedding exists) and every branch's embedding beside each embedding quantum.
            let embeds: u64 = cond_embeds.iter().map(price_bytes).sum();
            take(
                IosPhase::Encode,
                heavy + warm_text + control_images + embeds,
                quanta,
            );
        }

        // Denoise: one guided step (two sequential batch-1 halves, or one batched pair), the
        // ControlNet branches beside the U-Net, the IP tokens in the cross-attention.
        // add_noise(_with) uses f32 noise; Euler retains that promoted sampler state.
        // The per-forward cast in price_step is a different tensor, not a state conversion.
        let latent = regime.initial_state_dtype();
        let sequential = regime.sampling.guidance() > 1.0 && !regime.cfg_batching.is_batched();
        let batch = regime.model_rows();
        // Bespoke execution has at most two dtype epochs: its first state and the promoted
        // steady suffix. Resolve their existence without allocating the actual O(steps) schedule.
        // Other solver histories are not certified by this ancestral ownership profile.
        let ancestral_steps = if regime.sampling.route() == SamplingRoute::Ancestral {
            Some(
                regime
                    .sampling
                    .ancestral_window(
                        crate::config::DiffusionConfig::sdxl_base().num_train_steps as f32,
                        regime
                            .sampling
                            .effective_init_strength(regime.has_init, regime.route == Route::Mask),
                    )?
                    .0,
            )
        } else {
            None
        };
        // Acceleration policies always have one first callback; more than one requested step
        // also has a steady callback. ORIGINAL_STEPS=50 is LCM/Hyper's origin table, NOT a
        // requested-step cap: longer requests repeat timestep indices and remain supported.
        // Their state remains F32, so this projection is bounded independently of requested N;
        // it does not instantiate Lightning's O(N) coefficient table during header pricing.
        let acceleration = regime.sampling.route().is_acceleration();
        let acceleration_families = if acceleration {
            let cfg = crate::config::DiffusionConfig::sdxl_base();
            let alpha = mlx_gen::AlphaSchedule::scaled_linear(
                cfg.num_train_steps,
                cfg.beta_start,
                cfg.beta_end,
            );
            Some(regime.sampling.bounded_acceleration_families(&alpha)?)
        } else {
            None
        };
        let discrete_steps =
            ancestral_steps.or_else(|| acceleration.then_some(regime.sampling.steps()));
        let profiled_discrete = discrete_steps.is_some();
        let profiled_curated = regime.sampling.route().is_curated();
        let unclassified_control_ip = matches!(regime.route, Route::Control | Route::IpAdapter);
        let epoch_family = if acceleration {
            "acceleration"
        } else {
            "ancestral"
        };
        let mut denoise_windows = Vec::new();
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED
            .with_cfg_batching(regime.cfg_batching)
            .with_cfg_row(sequential.then_some(0));
        let encoder_x = price_tensor(
            &[rows, regime.context_tokens, self.arch.context_dim],
            conditioning_dtype,
        )?;
        let text_emb = price_tensor(&[rows, self.te.1.pooled_width()], pooled)?;
        // Step 1 runs the U-Net on the init latent's dtype; every later step on the sampler's
        // output, which the step promotes to the U-Net output's dtype (a packed tier's f32) — so
        // the denoise prices every distinct latent dtype the loop feeds the U-Net.
        let mut quanta = Quanta::new();
        let mut step_dtype = latent;
        let mut epoch = 0;
        loop {
            if discrete_steps == Some(0) {
                break;
            }
            let mut forward = Quanta::new();
            let (eps, coincident_control) = self.price_step(
                regime,
                price_tensor(&[batch, lh, lw, 4], step_dtype)?,
                &encoder_x,
                &text_emb,
                &cond_embeds,
                ip_tokens.as_ref(),
                plan,
                &mut forward,
            )?;
            let next = Dtype::from_promoting_types(step_dtype, eps);
            if let Some(steps) = discrete_steps.filter(|_| profiled_discrete) {
                let elements = (lh as u64) * (lw as u64) * 4;
                let owners = regime.forward_owners(step_dtype).bytes(
                    elements,
                    crate::model::DTYPE,
                    batch as u64,
                );
                let fixed = regime.initial_fixed_inputs(elements);
                let addons = cond_embeds.iter().map(price_bytes).sum::<u64>()
                    + ip_tokens.as_ref().map_or(0, price_bytes);
                let completed = elements * q::dtype_bytes(next) * u64::from(regime.count - 1);
                // Both conditioning tensors and model::generate's f32 [rows,6] time_ids remain
                // owned in the enclosing generation scope (pipeline::text_time_ids).
                let conditioning = price_bytes(&encoder_x)
                    + price_bytes(&text_emb)
                    + rows as u64 * 6 * q::dtype_bytes(Dtype::Float32);
                let halves = if sequential { 2 } else { 1 };
                let shed_ip = crate::model::sheds_ip_image_encoder(staged)
                    && regime.route == Route::IpAdapter;
                let structural = self.heavy(shed_ip) + warm_text;
                for half in 0..halves {
                    for (ordinal, &(label, scratch)) in forward.iter().enumerate() {
                        // Only the stem's guard includes the original model-input tensor. Later
                        // guards price their local activation, while forward_eps still borrows
                        // the caller's original input. Do not count that input twice at the stem.
                        let overlap = if label == q::UNET_STEM || label == q::CONTROL_STEM {
                            elements * q::dtype_bytes(crate::model::DTYPE) * batch as u64
                        } else {
                            0
                        };
                        let first_eps = if half == 1 {
                            elements * q::dtype_bytes(eps)
                        } else {
                            0
                        };
                        denoise_windows.push(IosWindowPrice {
                            phase: IosPhase::Denoise,
                            // Required raw fallback: Control/IP never acquire physical classes
                            // merely because coincident logical ownership is now explicit.
                            class: (!unclassified_control_ip).then(|| {
                                format!(
                                    "{}:{epoch_family}:{}:cfg{half}:q{ordinal}",
                                    regime.sampling.allocation_class(),
                                    if epoch == 0 { "first" } else { "steady" }
                                )
                            }),
                            label: label.into(),
                            resident_bytes: structural,
                            held_bytes: owners.saturating_sub(overlap)
                                + fixed
                                + addons
                                + coincident_control[ordinal]
                                + completed
                                + conditioning
                                + first_eps,
                            scratch_bytes: scratch,
                            basis: Some(IosWorkBasis {
                                forward_rows: batch as u64,
                                token_count: regime.context_tokens as u64,
                                model_width: regime.width as u32,
                                model_height: regime.height as u32,
                                ..IosWorkBasis::default()
                            }),
                            owner_domain: if unclassified_control_ip {
                                "sdxl/control-ip/coincident-owners-logical-v1"
                            } else if acceleration {
                                // Lightning's pending scale parent and all acceleration update
                                // branches still need their own native backing proof.
                                "sdxl/acceleration/forward-input-logical-v1"
                            } else if label == q::UNET_STEM {
                                "sdxl/ancestral/forward-input-pending-v1"
                            } else {
                                "sdxl/ancestral/forward-input-native-v1"
                            }
                            .into(),
                        });
                    }
                }
                // denoise_core retains the current latent through the lazy CFG/update/blend
                // quantum. Per-forward CFG concatenations/casts and ControlNet residuals have
                // left their Rust scopes; do not carry those independent forward maxima here.
                // The unchanged guard is a raw scratch upper, never a subtractable physical lower.
                let update = IosWindowPrice {
                    phase: IosPhase::Denoise,
                    class: (!unclassified_control_ip).then(|| {
                        format!(
                            "{}:{epoch_family}:{}:sampler-update",
                            regime.sampling.allocation_class(),
                            if epoch == 0 { "first" } else { "steady" }
                        )
                    }),
                    label: q::SAMPLER_STEP.into(),
                    resident_bytes: structural,
                    held_bytes: elements * q::dtype_bytes(step_dtype)
                        + fixed
                        + conditioning
                        + completed
                        + cond_embeds.iter().map(price_bytes).sum::<u64>()
                        + ip_tokens.as_ref().map_or(0, price_bytes),
                    scratch_bytes: q::sampler_step_bytes_metadata(
                        (lh as u64) * (lw as u64) * 4,
                        regime.route == Route::Mask,
                    ),
                    basis: Some(IosWorkBasis {
                        forward_rows: 1,
                        token_count: regime.context_tokens as u64,
                        model_width: regime.width as u32,
                        model_height: regime.height as u32,
                        ..IosWorkBasis::default()
                    }),
                    owner_domain: format!("sdxl/{epoch_family}/sampler-update-logical-v1"),
                };
                if let Some(families) = &acceleration_families {
                    let mut represented = false;
                    for family in families.iter().filter(|f| f.first_step == (epoch == 0)) {
                        let mut candidate = update.clone();
                        candidate.class = (!unclassified_control_ip).then(|| {
                            format!(
                                "{}:acceleration:{}:sampler-update",
                                regime.sampling.allocation_class(),
                                family.family()
                            )
                        });
                        candidate.owner_domain =
                            "sdxl/acceleration/sampler-update-logical-v2".into();
                        denoise_windows.push(candidate);
                        represented = true;
                    }
                    if !represented {
                        return Err(Error::Unsupported(
                            "sdxl actual acceleration epoch has no bounded mechanism family".into(),
                        ));
                    }
                } else {
                    denoise_windows.push(update);
                }
                // The second epoch is a source-proved fixed dtype suffix, not an enumeration of
                // the requested schedule. A one/zero-step request must not acquire this suffix.
                quanta.extend(forward);
                step_dtype = next;
                epoch += 1;
                if epoch == 2 || steps == 1 {
                    break;
                }
            } else {
                if profiled_curated {
                    let (solver, cfg_pp) = match regime.sampling.route() {
                        SamplingRoute::Curated(solver) => (solver, false),
                        SamplingRoute::CfgPp(solver) => (solver, true),
                        _ => unreachable!("curated profile uses its resolved factory"),
                    };
                    let cfg = crate::config::DiffusionConfig::sdxl_base();
                    let alpha = mlx_gen::AlphaSchedule::scaled_linear(
                        cfg.num_train_steps,
                        cfg.beta_start,
                        cfg.beta_end,
                    );
                    let ms = mlx_gen::DiscreteModelSampling::sdxl(&alpha);
                    let elements = (lh as u64) * (lw as u64) * 4;
                    let candidates =
                        mlx_gen::sampler::solver_memory::solver_callback_windows_for_model(
                            solver,
                            cfg_pp,
                            &ms,
                            elements,
                            q::dtype_bytes(step_dtype),
                            q::dtype_bytes(eps),
                        )?;
                    let fixed = regime.initial_fixed_inputs(elements);
                    let conditioning = price_bytes(&encoder_x)
                        + price_bytes(&text_emb)
                        + rows as u64 * 6 * q::dtype_bytes(Dtype::Float32);
                    let completed =
                        elements * q::dtype_bytes(Dtype::Float32) * u64::from(regime.count - 1);
                    let shed_ip = crate::model::sheds_ip_image_encoder(staged)
                        && regime.route == Route::IpAdapter;
                    let structural = self.heavy(shed_ip) + warm_text;
                    let addons = cond_embeds.iter().map(price_bytes).sum::<u64>()
                        + ip_tokens.as_ref().map_or(0, price_bytes);
                    // This is the real VE/CFG++ factory's bounded POSSIBLE family set, not a
                    // claim that a representative sigma schedule was observed for this request.
                    // Current x, scaled input, history and secondary-callback outer nodes remain
                    // separate coincident operands; never subtract their independent maxima.
                    for candidate in candidates {
                        let family = candidate.family();
                        let scaled = if candidate.scaled_input_alias {
                            0
                        } else {
                            candidate.scaled_input_bytes
                        };
                        // Curated callbacks cast F32 x_in BEFORE CFG concatenation. The shared
                        // solver retains x_in across the callback; its F16 boundary is distinct.
                        let cast = elements * q::dtype_bytes(crate::model::DTYPE);
                        let cfg_pair = if batch > 1 { cast * batch as u64 } else { 0 };
                        let owners = candidate.callback_outer_bytes
                            + candidate.state_bytes
                            + scaled
                            + cast
                            + cfg_pair;
                        for half in 0..if sequential { 2 } else { 1 } {
                            for (ordinal, &(label, scratch)) in forward.iter().enumerate() {
                                let overlap = if label == q::UNET_STEM || label == q::CONTROL_STEM {
                                    cast * batch as u64
                                } else {
                                    0
                                };
                                denoise_windows.push(IosWindowPrice {
                                    phase: IosPhase::Denoise,
                                    class: (!unclassified_control_ip).then(|| {
                                        format!(
                                            "{}:curated:{family}:cfg{half}:q{ordinal}",
                                            regime.sampling.allocation_class()
                                        )
                                    }),
                                    label: label.into(),
                                    resident_bytes: structural,
                                    held_bytes: owners.saturating_sub(overlap)
                                        + fixed
                                        + conditioning
                                        + addons
                                        + coincident_control[ordinal]
                                        + completed
                                        + if half == 1 {
                                            elements * q::dtype_bytes(eps)
                                        } else {
                                            0
                                        },
                                    scratch_bytes: scratch,
                                    basis: Some(IosWorkBasis {
                                        forward_rows: batch as u64,
                                        token_count: regime.context_tokens as u64,
                                        model_width: regime.width as u32,
                                        model_height: regime.height as u32,
                                        ..IosWorkBasis::default()
                                    }),
                                    owner_domain: if unclassified_control_ip {
                                        "sdxl/control-ip/coincident-owners-logical-v1"
                                    } else {
                                        "sdxl/curated/forward-input-logical-v1"
                                    }
                                    .into(),
                                });
                            }
                        }
                        denoise_windows.push(IosWindowPrice {
                            phase: IosPhase::Denoise,
                            class: (!unclassified_control_ip).then(|| {
                                format!(
                                    "{}:curated:{family}:sampler-update",
                                    regime.sampling.allocation_class()
                                )
                            }),
                            label: mlx_gen::sampler::SAMPLER_STEP_QUANTUM.into(),
                            resident_bytes: structural,
                            held_bytes: candidate.callback_outer_bytes
                                + candidate.state_bytes
                                + fixed
                                + conditioning
                                + addons
                                + completed,
                            scratch_bytes: mlx_gen::ios_quantum::sampler_step_bytes(elements),
                            basis: Some(IosWorkBasis {
                                forward_rows: 1,
                                token_count: regime.context_tokens as u64,
                                model_width: regime.width as u32,
                                model_height: regime.height as u32,
                                ..IosWorkBasis::default()
                            }),
                            owner_domain: "sdxl/curated/sampler-update-logical-v1".into(),
                        });
                    }
                }
                quanta.extend(forward);
                if next == step_dtype {
                    break;
                }
                step_dtype = next;
            }
        }
        let latents = price_tensor(&[1, lh, lw, 4], step_dtype)?;
        if regime.sampling.route().is_curated() {
            quanta.push((
                mlx_gen::sampler::SAMPLER_STEP_QUANTUM,
                mlx_gen::ios_quantum::sampler_step_bytes(latents.size() as u64),
            ));
        } else if ancestral_steps != Some(0) {
            // Acceleration and bespoke ancestral both run the actual SDXL denoise_core guard.
            quanta.push((
                q::SAMPLER_STEP,
                q::sampler_step_bytes_metadata(latents.elements(), regime.route == Route::Mask),
            ));
        }
        let shed_ip =
            crate::model::sheds_ip_image_encoder(staged) && regime.route == Route::IpAdapter;
        // A real init owner is F16, independently of the F32 noised sampler state. The mask
        // path additionally retains its F32 noise/mask. ControlNet+reference also owns that init;
        // an IP image prompt does not. These are owners, not a fitted phase correction.
        let elements = (lh as u64) * (lw as u64) * 4;
        let held = regime.initial_fixed_inputs(elements)
            + cond_embeds.iter().map(price_bytes).sum::<u64>()
            + ip_tokens.as_ref().map_or(0, price_bytes);
        take(
            IosPhase::Denoise,
            self.heavy(shed_ip) + warm_text + held,
            quanta,
        );

        // Decode: the staged decode holds the VAE alone; a warm one decodes beside everything.
        let mut quanta = Quanta::new();
        let decoded = self
            .vae
            .price_decode(&latents, regime.tiling.as_ref(), &mut quanta)?;
        quanta.push((
            q::DECODED_TO_IMAGE,
            q::decoded_to_image_bytes_from_metadata(decoded.elements()),
        ));
        let decode_resident = if staged {
            self.vae_bytes.resident
        } else {
            heavy + warm_text
        };
        // The decode closure retains the caller's sampler latent. to_compute aliases it only
        // at the same dtype (or for an f32 VAE); otherwise its compute copy is live beside it.
        let caller = price_bytes(&latents);
        let cast = self.vae.decode_input_retained_bytes(&latents) - caller;
        for (label, scratch) in quanta {
            // The first guard pays the not-yet-materialized cast as scratch. Later decoder
            // guards see it already resident. VAE return drops it before pixel conversion.
            let retained_cast = if label == q::VAE_DECODE_IN || label == q::DECODED_TO_IMAGE {
                0
            } else {
                cast
            };
            take(
                IosPhase::Decode,
                // `finals` owns EVERY completed serial latent throughout the decode loop; the
                // borrowed/clone caller handle aliases its entry, not a second latent allocation.
                decode_resident + caller * u64::from(regime.count) + retained_cast,
                vec![(label, scratch)],
            );
        }

        if let Some(fused) = &self.fused {
            for item in &mut items {
                if item.phase != IosPhase::Load || item.label == REQUEST_INPUTS {
                    let text_only = staged
                        && matches!(
                            item.label,
                            q::CLIP_EMBEDDINGS | q::CLIP_LAYER | q::CLIP_POOLED
                        );
                    let parents = fused.text_parents.saturating_add(if text_only {
                        0
                    } else {
                        fused.heavy_parents
                    });
                    item.resident = item.resident.saturating_add(parents);
                }
            }
            for window in &mut denoise_windows {
                window.resident_bytes = window
                    .resident_bytes
                    .saturating_add(fused.text_parents)
                    .saturating_add(fused.heavy_parents);
                window.class = None;
                window.owner_domain = format!(
                    "sdxl/fused/source-parents-logical-v1/{}",
                    window.owner_domain
                );
            }
        }
        let mut pricing = IosPricing::default();
        for item in &items {
            if item.label == IP_PREPROCESS_HOST {
                pricing.take_window(IosWindowPrice {
                    phase: item.phase,
                    class: None,
                    label: item.label.into(),
                    resident_bytes: item.resident,
                    held_bytes: 0,
                    scratch_bytes: item.scratch,
                    basis: None,
                    owner_domain: "sdxl/ip/preprocess-host-requested-logical-v1".into(),
                });
                continue;
            }
            if (profiled_discrete || profiled_curated) && item.phase == IosPhase::Denoise {
                // Forward AND update candidates are recorded at their own coincident epochs,
                // not the largest input owner combined with another epoch's binding operation.
                // Update/pending windows remain unverified physical domains; their presence is
                // mandatory coverage, not permission to certify from a raw scratch upper.
                continue;
            }
            if self.fused.is_some() {
                pricing.take_window(IosWindowPrice {
                    phase: item.phase,
                    class: None,
                    label: item.label.into(),
                    resident_bytes: item.resident,
                    held_bytes: 0,
                    scratch_bytes: item.scratch,
                    basis: None,
                    owner_domain: "sdxl/fused/source-parents-logical-v1".into(),
                });
            } else {
                pricing.take(item.phase, item.resident, item.scratch, item.label);
            }
        }
        for window in denoise_windows {
            pricing.take_window(window);
        }
        if ancestral_steps == Some(0) {
            pricing.phase_not_run(
                IosPhase::Denoise,
                "actual ancestral strength schedule has zero steps",
            );
        }
        if !regime.has_init && regime.route != Route::Control {
            pricing.phase_not_run(IosPhase::Encode, "no init or active ControlNet encode");
        }
        Ok((pricing, items))
    }

    /// One guided denoise forward's quanta over the shape-only latent `x` (the ControlNet branches
    /// beside the U-Net, the IP tokens in the cross-attention). Returns the U-Net output's dtype and
    /// the bytes of the ControlNet residuals the U-Net forward holds.
    #[allow(clippy::too_many_arguments)]
    fn price_step(
        &self,
        regime: &Regime,
        x: PriceTensor,
        encoder_x: &PriceTensor,
        text_emb: &PriceTensor,
        cond_embeds: &[PriceTensor],
        ip_tokens: Option<&PriceTensor>,
        plan: crate::plan::SdxlForwardPlan<'_>,
        quanta: &mut Quanta,
    ) -> Result<(Dtype, Vec<u64>)> {
        let first_quantum = quanta.len();
        let x = price_tensor(x.shape(), crate::model::DTYPE)?;
        let batch = x.shape()[0];
        let mut control: Option<(Vec<PriceTensor>, PriceTensor)> = None;
        let mut pending_control = q::ControlMergeWork::default();
        let mut owners = Vec::new();
        let (mut raw_down, mut raw_mid, mut selected_rows) = (0u64, 0u64, 0u64);
        if regime.route == Route::Control {
            for (branch_index, (branch, embed)) in self.controls.iter().zip(cond_embeds).enumerate()
            {
                let s = embed.shape();
                let row = price_tensor(&[batch, s[1], s[2], s[3]], embed.dtype())?;
                let selected = if plan.cfg_row.is_some() && s[0] == 2 {
                    price_bytes(&row)
                } else {
                    0
                };
                let first_owner = owners.len();
                let effects = self.control_effects[branch_index]
                    .as_ref()
                    .map_or_else(q::EffectScope::native, q::EffectScope::root);
                let (down, mid) = branch.price_forward_with_effects(
                    &x,
                    &row,
                    encoder_x,
                    text_emb,
                    plan,
                    quanta,
                    Some(&mut owners),
                    &effects,
                )?;
                for (index, held) in owners[first_owner..].iter_mut().enumerate() {
                    // Prior branch payloads are kept by the unevaluated sum DAG. The new branch's
                    // final residual stack does NOT yet exist at its early-forward quanta.
                    *held +=
                        raw_down + raw_mid + selected_rows + if index == 0 { 0 } else { selected };
                }
                selected_rows += selected;
                raw_down += down.iter().map(price_bytes).sum::<u64>();
                raw_mid += price_bytes(&mid);
                control = Some(match control {
                    None => (down, mid),
                    Some((prev_down, prev_mid)) => {
                        if prev_down.len() != down.len() {
                            return Err(Error::Msg(
                                "controlnet residual sum: branch down counts differ".into(),
                            ));
                        }
                        let merged_down = prev_down
                            .iter()
                            .zip(&down)
                            .map(|(a, b)| {
                                // Same elementwise broadcast/result-type semantics, over metadata only;
                                // no native add/zero graph is built to discover its output.
                                let added = crate::unet::pricing::add(a, b)?;
                                price_tensor(added.shape(), added.dtype())
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let added = crate::unet::pricing::add(&prev_mid, &mid)?;
                        let merged_mid = price_tensor(added.shape(), added.dtype())?;
                        pending_control.record_outputs_metadata(&merged_down, &merged_mid)?;
                        (merged_down, merged_mid)
                    }
                });
            }
        }
        // `pipeline::forward_eps` casts the combined residuals at the iOS U-Net boundary.
        if let Some((down, mid)) = &mut control {
            for residual in down {
                *residual = price_tensor(residual.shape(), crate::model::DTYPE)?;
            }
            *mid = price_tensor(mid.shape(), crate::model::DTYPE)?;
        }
        let (cast_down, cast_mid) = control.as_ref().map_or((0, 0), |(down, mid)| {
            (down.iter().map(price_bytes).sum::<u64>(), price_bytes(mid))
        });
        let unet_first = quanta.len();
        let effects = self
            .unet_effects
            .as_ref()
            .map_or_else(q::EffectScope::native, q::EffectScope::root);
        let eps = self.unet.price_forward_with_effects(
            &x,
            encoder_x,
            text_emb,
            control.as_ref(),
            ip_tokens.map(|t| (t, regime.ip_scale)),
            plan,
            pending_control,
            quanta,
            &effects,
        )?;
        let (mut down_complete, mut mid_complete) = (false, false);
        for (label, _) in &quanta[unet_first..] {
            // Only the ACTUAL existing injection finish may replace raw source parents with the
            // materialized cast's logical owner. This remains unclassified, not backing equality:
            // inherited storage/cache oversizing and native/process domains are not inferred.
            owners.push(
                selected_rows
                    + if down_complete { cast_down } else { raw_down }
                    + if mid_complete { cast_mid } else { raw_mid },
            );
            if *label == q::CONTROL_DOWN_INJECTION {
                down_complete = true;
            }
            if *label == q::CONTROL_MID_INJECTION {
                mid_complete = true;
            }
        }
        if owners.len() != quanta.len() - first_quantum {
            return Err(Error::Unsupported(
                "sdxl control coincident-owner ledger is incomplete".into(),
            ));
        }
        Ok((eps, owners))
    }
}

/// The numeric interpretation of the selected actual install work. Runtime guarding and the
/// adapter load ledger use this exact budget, not an independent adapter-kind formula.
pub(crate) fn adapter_install_quantum(
    work: &crate::adapters::AdapterInstallWork,
    resident: u64,
) -> PricedQuantum {
    PricedQuantum {
        phase: IosPhase::Load,
        resident: resident.saturating_add(work.pre_guard),
        label: q::ADAPTER_MERGE,
        scratch: work.scratch,
    }
}

/// Logical upper ledger over a factual adapter producer run. This does not certify backing or
/// make the served header-only exporter complete. FileScopeEnd never subtracts the raw payload:
/// evaluated outputs, lazy parents and streamed/captured clones have distinct ownership lifetimes.
#[derive(Default)]
pub(crate) struct AdapterLoadLedger {
    pub quanta: Vec<PricedQuantum>,
    pub raw_parent_upper: u64,
    pub installed_upper: u64,
    pub preparation_upper: u64,
    pub residuals: Vec<mlx_gen::adapters::facts::AdapterFacts>,
}

impl AdapterLoadLedger {
    #[cfg(test)]
    pub(crate) fn from_work(work: &[crate::adapters::AdapterWork], resident: u64) -> Self {
        Self::from_work_checked(work, resident)
            .expect("actual adapter producer dispatch is coherent")
    }

    pub(crate) fn retained_upper(&self) -> u64 {
        self.raw_parent_upper
            .saturating_add(self.preparation_upper)
            .saturating_add(self.installed_upper)
    }

    pub(crate) fn from_work_checked(
        work: &[crate::adapters::AdapterWork],
        resident: u64,
    ) -> Result<Self> {
        use crate::adapters::{AdapterInstallStorage as S, AdapterWork, InstallKind as K};
        let mut result = Self::default();
        let mut selected = None;
        let mut pending: Option<&crate::adapters::AdapterInstallWork> = None;
        for item in work {
            let current = resident.saturating_add(result.retained_upper());
            match item {
                AdapterWork::FileMaterialized(payloads) => {
                    let (raw, load) = crate::adapters::adapter_file_payload_budget(payloads);
                    result.quanta.push(PricedQuantum {
                        phase: IosPhase::Load,
                        resident: current,
                        label: q::ADAPTER_LOAD,
                        scratch: load,
                    });
                    result.raw_parent_upper = result.raw_parent_upper.saturating_add(raw);
                }
                AdapterWork::Selected(kind) => {
                    selected = Some(*kind);
                }
                AdapterWork::AlphaRead { key, scratch } => {
                    if selected.is_none() || key.is_empty() {
                        return Err(Error::Msg(
                            "sdxl alpha work lacks actual selected parser/key".into(),
                        ));
                    }
                    result.quanta.push(PricedQuantum {
                        phase: IosPhase::Load,
                        resident: current,
                        label: q::ADAPTER_LOAD,
                        scratch: *scratch,
                    });
                }
                AdapterWork::Preparation(parents) => {
                    result.preparation_upper =
                        parents.iter().fold(result.preparation_upper, |n, p| {
                            n.saturating_add(mlx_gen::ios_quantum::allocated_bytes(p.bytes()))
                        });
                    result.quanta.push(PricedQuantum {
                        phase: IosPhase::Load,
                        resident: resident.saturating_add(result.retained_upper()),
                        label: q::ADAPTER_MERGE,
                        scratch: 0,
                    });
                }
                AdapterWork::GuardAttempt(install) | AdapterWork::Install(install) => {
                    let allowed = match selected {
                        Some(K::Lora) => matches!(
                            install.storage,
                            S::DenseLinear | S::DenseConv | S::PackedLora | S::PackedDelta
                        ),
                        Some(K::Lokr) => {
                            matches!(install.storage, S::DenseLinear | S::StructuredLokr)
                        }
                        Some(K::ThirdPartyLokr | K::ThirdPartyLoha) => {
                            matches!(install.storage, S::DenseLinear | S::PackedDelta)
                        }
                        None => {
                            return Err(Error::Msg(
                                "sdxl install has no selected file dispatch".into(),
                            ))
                        }
                    };
                    if !allowed {
                        return Err(Error::Msg(
                            "sdxl install storage disagrees with selected file dispatch".into(),
                        ));
                    }
                    if matches!(item, AdapterWork::GuardAttempt(_)) {
                        result
                            .quanta
                            .push(adapter_install_quantum(install, current));
                        pending = Some(install);
                        continue;
                    }
                    if let Some(attempt) = pending.take() {
                        if attempt.target != install.target || attempt.storage != install.storage {
                            return Err(Error::Msg(
                                "sdxl successful declaration disagrees with guard attempt".into(),
                            ));
                        }
                    } else {
                        // Actual producer witnesses record success only; their original event shape
                        // remains protected. Header outcomes carry a distinct preceding attempt.
                        result
                            .quanta
                            .push(adapter_install_quantum(install, current));
                    }
                    // An upper may overlap earlier base/raw/view payload. Never subtract one
                    // logical upper from another to manufacture a physical lower/credit.
                    result.installed_upper = install
                        .installed
                        .iter()
                        .fold(result.installed_upper, |n, p| n.saturating_add(p.bytes()));
                    if let Some(residual) = install.residual() {
                        result.residuals.push(residual);
                    }
                }
                AdapterWork::FileScopeEnd => {
                    selected = None;
                    pending = None;
                }
            }
        }
        Ok(result)
    }
}

/// One priced quantum: the phase it runs in, the component bytes resident around it, its label and
/// its counted scratch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PricedQuantum {
    pub phase: IosPhase,
    pub resident: u64,
    pub label: &'static str,
    pub scratch: u64,
}
