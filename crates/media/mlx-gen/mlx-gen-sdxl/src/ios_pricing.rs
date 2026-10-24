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

mod heavy_alias;
mod native_ledger;
mod text_alias;

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
    // Pricing-only host work priced inside the IP route's conditioning (no guard checks it).
    (IP_PREPROCESS_HOST, IosPhase::Conditioning),
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

/// Pricing-only Rust host preprocessing work; not a native quantum or physical bound.
const IP_PREPROCESS_HOST: &str = "SDXL IP preprocess host logical";

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
    let (form, unet_headers, alternatives) = match &spec.weights {
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
                Vec::new(),
            )
        }
        WeightsSource::File(file) => {
            // The actual native loader opens the entire fused file before component remapping.
            for h in safetensors_path_tensor_headers(file).map_err(|e| Error::Msg(e.to_string()))? {
                crate::quant::header_dtype(h.dtype)?;
            }
            let source = crate::ldm::component_header_source(file)?;
            use mlx_gen::gen_core::sdxl_ldm::SdxlComponent as C;
            // The identity is a function of the source, not of one native order: every
            // order-dependent UNet group lists each realizable winner's layouts and adapter facts.
            let mut groups = Vec::new();
            for (group, alias) in source.aliases.iter().enumerate() {
                if alias.component != C::Unet {
                    continue;
                }
                let mut assignments = Vec::new();
                for assignment in 0..alias.assignments.len() {
                    let world = source.headers_with(C::Unet, &[(group, assignment)]);
                    let mut layouts = std::collections::BTreeMap::new();
                    for target in &alias.targets {
                        let h = world
                            .iter()
                            .find(|h| &h.name == target)
                            .ok_or_else(|| Error::MissingTensor(target.clone()))?;
                        layouts.insert(target.clone(), json!({"shape":h.shape,
                            "loaded_dtype":format!("{:?}",crate::quant::loaded_header_dtype(&world,h)?)}));
                    }
                    let adapters = if spec.adapters.is_empty() {
                        serde_json::Value::Null
                    } else {
                        crate::adapters::configured_adapter_semantic_facts(spec, &world)?
                    };
                    assignments.push(json!({"layouts":layouts,"adapter_graph":adapters}));
                }
                groups.push(json!({"targets":alias.targets,"assignments":assignments}));
            }
            ("fused_ldm", source.headers(C::Unet), groups)
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
    let mut identity = json!({"provider":"sdxl","load_form":form,"load_quantization_after_adapters_and_ip":spec.quantize.map(|q|q.bits()),
        "base_loaded_source_layouts":base_layouts,"adapter_graph":adapter_facts,"controls":controls,"ip_adapter":ip,"vae_overlay":vae_overlay});
    // Present only for order-dependent fused UNet sources: alias-free identities are unchanged.
    if !alternatives.is_empty() {
        identity["order_dependent_unet_aliases"] = json!(alternatives);
    }
    Ok(identity)
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
    let mut inputs = Inputs::from_spec(req.spec, regime)?;
    q::with_backend(backend, || {
        price_inputs(&mut inputs, regime, Some(req.spec))
    })
}

/// The exact pricer over every witness world of the source's order-dependent aliases (text
/// [`text_alias`], UNet/VAE [`heavy_alias`]), kept as one whole-candidate window envelope. A
/// source without aliases has the single base world and returns its ledger unchanged.
fn price_inputs(
    inputs: &mut Inputs,
    regime: &Regime,
    configured: Option<&LoadSpec>,
) -> Result<IosPricing> {
    let worlds = text_alias::witness_worlds(inputs, regime, configured)?;
    let mut pricing = price_worlds(inputs, regime, configured, worlds)?;
    native_ledger::classify_windows(&mut pricing, regime)?;
    Ok(pricing)
}

/// The unchanged exact pricer over the currently selected source world.
fn price_world(
    inputs: &Inputs,
    regime: &Regime,
    configured: Option<&LoadSpec>,
) -> Result<IosPricing> {
    match configured {
        Some(spec) => {
            let mut priced = Priced::build_configured(inputs, spec)?;
            priced.price_configured(spec, &inputs.unet, inputs.quant_bits, regime)
        }
        None => Ok(Priced::build(inputs)?.price(regime)?.0),
    }
}

fn price_worlds(
    inputs: &mut Inputs,
    regime: &Regime,
    configured: Option<&LoadSpec>,
    worlds: Vec<text_alias::Picks>,
) -> Result<IosPricing> {
    if worlds.len() == 1 && worlds[0].is_empty() {
        return price_world(inputs, regime, configured);
    }
    let mut envelope = WindowEnvelope::new();
    let mut absent = None;
    for world in worlds {
        inputs.fused.as_mut().expect("alias worlds are fused").world = world;
        let pricing = price_world(inputs, regime, configured)?;
        absent.get_or_insert_with(|| pricing.absent_phases.clone());
        merge_envelope(&mut envelope, pricing);
    }
    inputs.fused.as_mut().expect("alias worlds are fused").world = Vec::new();
    let mut pricing = IosPricing::default();
    for (_, window) in envelope {
        pricing.take_window(window);
    }
    pricing.absent_phases = absent.unwrap_or_default();
    Ok(pricing)
}

#[cfg(test)]
pub(crate) use native_ledger::{
    certified_request_bounds_for_test, resolve_bound_side_for_test, source_pricing_for_test,
};

/// Test oracle: the same envelope over EVERY realizable alias world (the full product of each
/// group's realizable assignments), or over the production witness worlds.
#[cfg(test)]
pub(crate) fn alias_envelope_for_test(
    inputs: &mut Inputs,
    regime: &Regime,
    configured: Option<&LoadSpec>,
    every_world: bool,
) -> Result<IosPricing> {
    let worlds = if every_world {
        let aliases = &inputs
            .fused
            .as_ref()
            .expect("fused alias fixture")
            .source
            .aliases;
        let mut worlds: Vec<text_alias::Picks> = vec![Vec::new()];
        for (group, alias) in aliases.iter().enumerate() {
            worlds = worlds
                .into_iter()
                .flat_map(|world| {
                    (0..alias.assignments.len()).map(move |assignment| {
                        let mut world = world.clone();
                        world.push((group, assignment));
                        world
                    })
                })
                .collect();
        }
        worlds
    } else {
        text_alias::witness_worlds(inputs, regime, configured)?
    };
    price_worlds(inputs, regime, configured, worlds)
}

/// Test access: the unchanged exact pricer over one selected alias world (CPU kernels).
#[cfg(test)]
pub(crate) fn price_world_for_test(
    inputs: &mut Inputs,
    regime: &Regime,
    world: text_alias::Picks,
) -> IosPricing {
    inputs.fused.as_mut().expect("fused alias fixture").world = world;
    let pricing = q::with_backend(Backend::Cpu, || price_world(inputs, regime, None)).unwrap();
    inputs.fused.as_mut().expect("fused alias fixture").world = Vec::new();
    pricing
}

type WindowEnvelope =
    std::collections::BTreeMap<(IosPhase, Option<String>, String, String, usize), IosWindowPrice>;

/// Keep a whole coincident candidate per window identity, never max(resident) + max(scratch)
/// from different worlds. Ordinals distinguish repeated same-label epochs.
fn merge_envelope(envelope: &mut WindowEnvelope, pricing: IosPricing) {
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
}

/// How a transfer certificate treats Metal command-buffer captures (completed-handler-held input,
/// sibling and encoder-temporary `Data`) that MLX `eval` does not retire on return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdxlRetirementPremise {
    /// Default. A capture is retired only by proven completed-handler/`Data` destruction. No
    /// immutable MLX source establishes that point (`mlx_synchronize` waits for command
    /// completion, not for handler destruction), and every certified forward window follows
    /// earlier command buffers, so SDXL transfer certificates are refused.
    ProvenOnly,
    /// Named platform premise, disabled by default: a command buffer's captures are retired once
    /// `mlx_synchronize` (`waitUntilCompleted`) returns. It may be enabled only with explicit
    /// approval of this physical/platform premise alongside activity 26569; certificates carry
    /// its identifier.
    MetalSynchronizeRetirement,
}

impl SdxlRetirementPremise {
    /// The transfer refusal of [`Self::ProvenOnly`]; the named premise admits certification.
    pub(crate) fn admit(self) -> Result<()> {
        match self {
            Self::ProvenOnly => Err(Error::Unsupported(format!(
                "sdxl transfer certificate refused under {}: every certified forward window \
                 follows earlier command buffers whose captured Metal buffers retire only by \
                 completed-handler/Data destruction, which no source proves; the {} premise is \
                 disabled without approval",
                self.id(),
                Self::MetalSynchronizeRetirement.id()
            ))),
            Self::MetalSynchronizeRetirement => Ok(()),
        }
    }

    pub const fn id(self) -> &'static str {
        match self {
            Self::ProvenOnly => "sdxl/metal-capture-retirement/proven-only-v1",
            Self::MetalSynchronizeRetirement => {
                "sdxl/metal-capture-retirement/wait-until-completed-premise-v1"
            }
        }
    }
}

/// Transfer certificates together with the retirement premise every one of them assumes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdxlPremisedBounds {
    pub premise: &'static str,
    pub bounds: Vec<IosWindowBound>,
}

/// Source-owned NATIVE-ALLOCATOR matched-window certificates for the correlated anchor law, under
/// the default [`SdxlRetirementPremise::ProvenOnly`]: cumulative native growth receives no retirement credit.
pub fn ios_bounder(
    anchor_request: &IosPricingRequest<'_>,
    anchor_pricing: &IosPricing,
    request: &IosPricingRequest<'_>,
    request_pricing: &IosPricing,
) -> Result<Vec<IosWindowBound>> {
    ios_bounder_with_premise(
        SdxlRetirementPremise::ProvenOnly,
        anchor_request,
        anchor_pricing,
        request,
        request_pricing,
    )
    .map(|premised| premised.bounds)
}

/// [`ios_bounder`] under an explicit retirement premise. The typed domain forbids using these as
/// process-growth authority. The law must check coverage of ALL requested windows; an unreturned
/// class is not permission to ignore that work. A new Encode phase is never certified from a
/// logical VAE scratch sum here.
pub fn ios_bounder_with_premise(
    premise: SdxlRetirementPremise,
    anchor_request: &IosPricingRequest<'_>,
    anchor_pricing: &IosPricing,
    request: &IosPricingRequest<'_>,
    request_pricing: &IosPricing,
) -> Result<SdxlPremisedBounds> {
    if premise == SdxlRetirementPremise::ProvenOnly {
        return proven_only_bounds(
            anchor_request,
            None,
            anchor_pricing,
            request,
            request_pricing,
        );
    }
    let anchor = Regime::from_request(anchor_request)?;
    let requested = Regime::from_request(request)?;
    let mut anchor_inputs = Inputs::from_spec(anchor_request.spec, &anchor)?;
    let mut request_inputs = Inputs::from_spec(request.spec, &requested)?;
    if anchor_inputs.fused.is_some() != request_inputs.fused.is_some() {
        return Err(Error::Unsupported(
            "sdxl common-forward transfer needs the same source component graph".into(),
        ));
    }
    // Fused native backing follows the MLX ACTIVE-ALLOCATOR counter law only (audit
    // sc26270-fused-native-backing-source-authority-v2): anchor and request split the same file,
    // so equal-order backing cancels and the counter delta adds the full source-order variance.
    // NativeAllocatorGrowth only; task footprint/physical mapping stays the activity-26569 premise.
    let order_variance = request_inputs.fused_counter_order_variance()?;
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
        let expected_anchor = price_inputs(&mut anchor_inputs, &anchor, None)?;
        let expected_request = price_inputs(&mut request_inputs, &requested, None)?;
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
        premise.admit()?;
        let states = [
            Priced::build(&anchor_inputs)?.ancestral_states(&anchor)?,
            Priced::build(&request_inputs)?.ancestral_states(&requested)?,
        ];
        let mut bounds = bound_ancestral_forward_windows(
            &anchor,
            anchor_pricing,
            &requested,
            request_pricing,
            anchor_inputs.arch.latent_scale,
            states,
            &mut request_inputs,
        )?;
        for bound in &mut bounds {
            if let IosWindowBound::Transfer {
                anchor_candidates, ..
            } = bound
            {
                for candidate in anchor_candidates {
                    candidate.delta_growth_upper = candidate
                        .delta_growth_upper
                        .saturating_add(i128::from(order_variance));
                }
            }
        }
        Ok(SdxlPremisedBounds {
            premise: premise.id(),
            bounds,
        })
    })
}

/// Typed payload-free catalog anchor facts. Counts are actual input roles, not the aggregate
/// geometry reference count. Source facts are mandatory for source-dependent IP geometry.
pub struct SdxlReplay<'a> {
    pub token: &'a str,
    pub true_reference_count: usize,
    pub mask_count: usize,
    pub control_count: usize,
    pub source: Option<&'a crate::sampling::SamplingSourceFacts>,
}

/// Full native request certificate from either an actual-request anchor or its typed replay.
/// An actual GenerationRequest takes precedence; no request or image facts are synthesized.
pub fn ios_bounder_with_replay(
    anchor_request: &IosPricingRequest<'_>,
    anchor_replay: Option<SdxlReplay<'_>>,
    anchor_pricing: &IosPricing,
    request: &IosPricingRequest<'_>,
    request_pricing: &IosPricing,
) -> Result<Vec<IosWindowBound>> {
    proven_only_bounds(
        anchor_request,
        anchor_replay,
        anchor_pricing,
        request,
        request_pricing,
    )
    .map(|premised| premised.bounds)
}

fn proven_only_bounds(
    anchor_request: &IosPricingRequest<'_>,
    anchor_replay: Option<SdxlReplay<'_>>,
    anchor_pricing: &IosPricing,
    request: &IosPricingRequest<'_>,
    request_pricing: &IosPricing,
) -> Result<SdxlPremisedBounds> {
    let bounds = match anchor_replay {
        None => native_ledger::certified_request_bounds(
            anchor_request,
            anchor_pricing,
            request,
            request_pricing,
        )?,
        replay => native_ledger::certified_request_bounds_with_replay(
            anchor_request,
            replay,
            anchor_pricing,
            request,
            request_pricing,
        )?,
    };
    Ok(SdxlPremisedBounds {
        premise: SdxlRetirementPremise::ProvenOnly.id(),
        bounds,
    })
}

/// Body cancellation requires the SAME actual input dtype/shape and CFG-half/quantum.
/// A promoted first request can match an anchor's promoted steady epoch, never its F16 body.
/// Separately owned state/init use native lifetime proof, not guard-upper subtraction.
/// Different initialization prefixes additionally retain a complete source-derived native upper;
/// matching a promoted first input to a steady anchor never grants free first-use retirement.
pub(crate) fn bound_ancestral_forward_windows(
    anchor: &Regime,
    anchor_pricing: &IosPricing,
    request: &Regime,
    request_pricing: &IosPricing,
    latent_scale: i32,
    states: [(Dtype, Dtype); 2],
    request_inputs: &mut Inputs,
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
    let uncancelled = forward_prefix_differs(
        (anchor.route, anchor.has_init, states[0].0),
        (request.route, request.has_init, states[1].0),
    );
    let mut prefix_by_page = std::collections::BTreeMap::new();
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
        let requested_body = ancestral_body_coordinate(class, states[1])?;
        for at_anchor in anchor_pricing
            .windows
            .iter()
            .filter(|window| window.owner_domain == "sdxl/ancestral/forward-input-native-v1")
        {
            let Some(anchor_class) = at_anchor.profile_class() else {
                continue;
            };
            if ancestral_body_coordinate(anchor_class, states[0])? != requested_body {
                continue;
            }
            // This checks that the generated ledger still describes the identical common body.
            // It does NOT turn either scratch upper into a physical anchor lower bound.
            if requested.label != at_anchor.label
                || requested.basis != at_anchor.basis
                || requested.scratch_bytes != at_anchor.scratch_bytes
                || requested.resident_bytes != at_anchor.resident_bytes
            {
                continue;
            }
            let witnesses = ancestral_forward_witnesses(
                anchor,
                anchor_pricing,
                at_anchor,
                latent_scale,
                states[0],
            )?;
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
                // Anchor: certainly-live owners at their exact fresh length (a cached buffer is
                // never shorter). Request: every owner at the buffer-cache reuse upper, since a
                // foreign cache-limit change can refill the cache inside the generation scope.
                let anchor_owned = ancestral_forward_owners(
                    anchor,
                    at_anchor,
                    elements,
                    page,
                    states[0],
                    mlx_gen::ios_quantum::fresh_metal_allocation_bytes,
                )?;
                let request_owned = ancestral_forward_owners(
                    request,
                    requested,
                    elements,
                    page,
                    states[1],
                    q::metal_reuse_upper_bytes,
                )?;
                let prefix = if uncancelled {
                    if let Some(&upper) = prefix_by_page.get(&page) {
                        upper
                    } else {
                        let upper =
                            uncancelled_forward_prefix_upper(request_inputs, request, page)?;
                        prefix_by_page.insert(page, upper);
                        upper
                    }
                } else {
                    0
                };
                candidates.push(IosAnchorCandidate {
                    anchor_class: anchor_class.into(),
                    delta_growth_upper: i128::from(prefix) + i128::from(request_owned)
                        - i128::from(anchor_owned),
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

/// Equal forward dtypes do not prove equal initialization/first-use native owners.
pub(crate) fn forward_prefix_differs(
    anchor: (Route, bool, Dtype),
    request: (Route, bool, Dtype),
) -> bool {
    anchor != request
}

pub(crate) fn uncancelled_forward_prefix_upper(
    inputs: &mut Inputs,
    request: &Regime,
    page: u64,
) -> Result<u64> {
    native_ledger::uncancelled_forward_prefix_upper(inputs, request, page)
}

#[test]
fn equal_forward_dtype_does_not_cancel_a_different_initialization_prefix() {
    let text = (Route::Text, false, Dtype::Float16);
    assert!(!forward_prefix_differs(text, text));
    assert!(forward_prefix_differs(
        text,
        (Route::Reference, true, Dtype::Float32)
    ));
    assert!(
        forward_prefix_differs(
            (Route::Text, false, Dtype::Float32),
            (Route::Reference, true, Dtype::Float32)
        ),
        "even equal first dtypes do not cancel VAE/noising work"
    );
    assert!(
        forward_prefix_differs(
            (Route::Reference, true, Dtype::Float32),
            (Route::Mask, true, Dtype::Float32)
        ),
        "retained mask/noise work is a different source prefix"
    );
}

/// Native body identity. On the ancestral route state dtype IS forward dtype.
fn ancestral_body_coordinate(
    class: &str,
    (first, steady): (Dtype, Dtype),
) -> Result<(&str, &str, &str, Dtype)> {
    let invalid = || Error::Unsupported("sdxl malformed source forward body coordinate".into());
    let mut parts = class.rsplitn(4, ':');
    let quantum = parts
        .next()
        .filter(|s| s.starts_with('q'))
        .ok_or_else(invalid)?;
    let half = parts
        .next()
        .filter(|s| s.starts_with("cfg"))
        .ok_or_else(invalid)?;
    let dtype = match parts.next() {
        Some("first") => first,
        Some("steady") => steady,
        _ => return Err(invalid()),
    };
    let recipe = parts.next().filter(|s| !s.is_empty()).ok_or_else(invalid)?;
    Ok((recipe, half, quantum, dtype))
}

#[test]
fn ancestral_body_matching_preserves_precision_and_cfg_coordinates() {
    let mixed = (Dtype::Float16, Dtype::Float32);
    let promoted = (Dtype::Float32, Dtype::Float32);
    let first = "recipe:ancestral:first:cfg0:q3";
    let steady = "recipe:ancestral:steady:cfg0:q3";
    let request = ancestral_body_coordinate(first, promoted).unwrap();
    assert_eq!(
        request,
        ancestral_body_coordinate(steady, mixed).unwrap(),
        "promoted first body matches actual promoted steady anchor"
    );
    assert_ne!(
        request,
        ancestral_body_coordinate(first, mixed).unwrap(),
        "F16 and F32 bodies cannot cancel even at the same named epoch"
    );
    for other in [
        "recipe:ancestral:steady:cfg1:q3",
        "recipe:ancestral:steady:cfg0:q4",
        "other:ancestral:steady:cfg0:q3",
    ] {
        assert_ne!(request, ancestral_body_coordinate(other, mixed).unwrap());
    }
    assert!(ancestral_body_coordinate("recipe:unknown:cfg0:q3", mixed).is_err());
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

/// The forward-input owners alive through a certified (non-stem) ancestral forward window, each
/// charged by `charge`. Under the synchronize-retirement premise the stem quantum's own pending
/// graph (the first epoch's prior/noise draw, CFG concatenation and cast) is retired with the
/// stem command buffer, leaving only these Rust-held outputs.
fn ancestral_forward_owners(
    regime: &Regime,
    window: &IosWindowPrice,
    elements: u64,
    page: u64,
    (first, steady): (Dtype, Dtype),
    charge: fn(u64, u64) -> Option<u64>,
) -> Result<u64> {
    let invalid = || {
        Error::Unsupported("sdxl native owner allocation provenance/extent is unverified".into())
    };
    let epoch = window
        .profile_class()
        .and_then(|class| class.rsplit(':').nth(2))
        .ok_or_else(invalid)?;
    let state = match epoch {
        "first" => first,
        "steady" => steady,
        _ => return Err(invalid()),
    };
    // These sampler states are full-span outputs, not arbitrary caller-provided views.
    // Init/noise/mask are separate retained owners; never round their combined logical sum.
    let mut owned = regime
        .forward_owners(state)
        .native_bytes(
            elements,
            elements,
            crate::model::DTYPE,
            regime.model_rows() as u64,
            page,
            charge,
        )
        .ok_or_else(invalid)?;
    let mut add = |bytes| -> Result<()> {
        owned = owned
            .checked_add(charge(bytes, page).ok_or_else(invalid)?)
            .ok_or_else(invalid)?;
        Ok(())
    };
    if regime.has_init {
        add(elements
            .checked_mul(q::dtype_bytes(Dtype::Float32))
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
    (first_state, steady_state): (Dtype, Dtype),
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
    let state = if first { first_state } else { steady_state };
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
                || ready.get("dtype") != Some(&mlx_gen::sampler::sampling_dtype_code(state))
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
                || cast.get("dtype") != Some(&mlx_gen::sampler::sampling_dtype_code(state))
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
            latent_elements.saturating_mul(q::dtype_bytes(Dtype::Float32))
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
    /// The selected world of `source.aliases`; empty selects assignment 0 of every group.
    world: text_alias::Picks,
}

impl Inputs {
    /// MLX active-counter variance (upper minus lower) of the fused load over every realizable
    /// native source order, summed over the text and heavy splits; zero for Directory inputs.
    /// Counter-scoped: not task footprint, retirement or source-parent subtraction.
    fn fused_counter_order_variance(&self) -> Result<u64> {
        use mlx_gen::gen_core::sdxl_ldm::SdxlComponent as C;
        let Some(fused) = &self.fused else {
            return Ok(0);
        };
        let vae_dtype = self.vae.1;
        let loaded =
            |component: C, header: &mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader| {
                match component {
                    C::Vae => Ok(vae_dtype),
                    _ => crate::quant::loaded_dtype_in_component(
                        fused.source.component_packed(component),
                        header,
                    ),
                }
            };
        let heavy: &[C] = if fused.vae_fix.is_some() {
            &[C::Unet]
        } else {
            &[C::Unet, C::Vae]
        };
        let mut variance = 0u64;
        for components in [&[C::ClipL, C::ClipBigG][..], heavy] {
            let backing = fused.source.active_counter_backing(components, &loaded)?;
            variance = variance.saturating_add(backing.upper.saturating_sub(backing.lower));
        }
        Ok(variance)
    }
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
                world: Vec::new(),
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

fn take(items: &mut Vec<PricedQuantum>, phase: IosPhase, resident: u64, quanta: Quanta) {
    for (label, scratch) in quanta {
        items.push(PricedQuantum {
            phase,
            resident,
            label,
            scratch,
        });
    }
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
        // The native pack guard is admitted before the quantize op validates the base.
        points.push(LoadPackPoint {
            before_resident: *retained,
            scratch: work.pack_scratch(bits),
        });
        work.pack(bits)?;
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
/// The native shape failure that ends a build-stage pack walk; `price` then emits exactly the
/// admitted native prefix through the failing pack guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeFailure {
    TextPack,
    HeavyPack,
}

pub(crate) struct Priced {
    configured_unet_headers: Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    native_failure: Option<NativeFailure>,
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

/// Source-parent terms stay logical. Native backing is the separate counter-scoped
/// [`Inputs::fused_counter_order_variance`] law; no logical product is a retirement time.
struct FusedLoadFacts {
    text_floor: u64,
    heavy_floor: u64,
    text_parents: u64,
    heavy_parents: u64,
    external_vae: bool,
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
    /// Exact normalized source graph used by adapter target selection in this alias world.
    pub(crate) fn configured_source_headers(
        &self,
        unet_file: &Path,
    ) -> Result<Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>> {
        if self.fused.is_some() {
            Ok(self.configured_unet_headers.clone())
        } else {
            mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(unet_file)
                .map_err(|e| Error::Msg(e.to_string()))
        }
    }

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
            Some(fused) => Ok(fused.source.headers_with(component, &fused.world)),
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
        let mut native_failure = None;
        // TE1 then TE2 pack in order; a native quantize shape failure stops the pair there.
        let (first_pack, first_failure) =
            crate::text_encoder::text_pack_prefix(&te1_headers, &arch.te1, inputs.text_quant_bits)?;
        let (second_pack, second_failure) = if first_failure.is_none() {
            crate::text_encoder::text_pack_prefix(&te2_headers, &arch.te2, inputs.text_quant_bits)?
        } else {
            // TE2 never packs after TE1's failing pack: its unpacked retained owners only.
            let unpacked = crate::text_encoder::text_pack_facts(&te2_headers, &arch.te2, None)?;
            (unpacked, None)
        };
        let te_pack = [first_pack, second_pack];
        if first_failure.is_some() || second_failure.is_some() {
            native_failure = Some(NativeFailure::TextPack);
        } else if let Some(bits) = inputs.text_quant_bits {
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
                let packed = unet.quantize_using(bits, &mut |path, linear, bits| {
                    collect_pack_effect(
                        path,
                        linear,
                        bits,
                        &mut retained,
                        &mut outputs,
                        &mut load_quantize,
                        &mut effects,
                    )
                });
                if let Err(error) = packed {
                    if !q::is_native_shape_failure(&error) {
                        return Err(error);
                    }
                    // Native fails inside this pack walk: later control packs never run.
                    native_failure = Some(NativeFailure::HeavyPack);
                    return Ok(());
                }
                unet_retained = unet_retained.saturating_add(outputs);
                unet_effects = Some(effects);
                for (index, control) in controls.iter_mut().enumerate() {
                    let mut effects = q::LinearEffects::default();
                    let mut outputs = 0;
                    let packed = control.quantize_using(bits, &mut |path, linear, bits| {
                        collect_pack_effect(
                            path,
                            linear,
                            bits,
                            &mut retained,
                            &mut outputs,
                            &mut load_quantize,
                            &mut effects,
                        )
                    });
                    if let Err(error) = packed {
                        if !q::is_native_shape_failure(&error) {
                            return Err(error);
                        }
                        native_failure = Some(NativeFailure::HeavyPack);
                        return Ok(());
                    }
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
                })
            })
            .transpose()?;
        Ok(Self {
            configured_unet_headers: unet_headers,
            native_failure,
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
        let headers = self.configured_source_headers(unet_file)?;
        let config = self.arch.unet.clone();
        let mut envelope = WindowEnvelope::new();
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
                let saved_failure = self.native_failure;
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
                        let packed = self.unet.quantize_using(bits, &mut |path, _, bits| {
                            let projection = effects.0.get_mut(path).ok_or_else(|| {
                                Error::Msg(format!(
                                    "sdxl selected world missing ordered pack target: {path}"
                                ))
                            })?;
                            if projection.bits.is_none() {
                                // The native pack guard precedes the quantize op's validation.
                                self.load_quantize.push(LoadPackPoint {
                                    before_resident: retained,
                                    scratch: projection.pack_scratch(bits),
                                });
                                projection.pack(bits)?;
                                // Whole declared projection, including unchanged residual owners,
                                // remains an upper. No upper subtraction or backing release credit.
                                let outputs = projection.declared_bytes();
                                retained = retained.saturating_add(outputs);
                                self.unet_retained = self.unet_retained.saturating_add(outputs);
                            }
                            Ok(())
                        });
                        // A native quantize shape failure ends this world inside the heavy pack:
                        // its admitted prefix is priced, no later control pack or forward.
                        if let Err(error) = packed {
                            if !q::is_native_shape_failure(&error) {
                                return Err(error);
                            }
                            self.native_failure = Some(NativeFailure::HeavyPack);
                            self.unet_effects = Some(effects);
                            let candidate = self.price(regime)?.0;
                            observe(&outcome, self, &self.adapter_load, &candidate)?;
                            return Ok(candidate);
                        }
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
                self.native_failure = saved_failure;
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
            merge_envelope(&mut envelope, pricing);
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
    /// Both towers in native order. A native shape failure returns `Err` with the admitted
    /// quanta (through the failing layer's guard) already in `out`.
    /// The ancestral sampler state dtypes of the first and steady epochs. `EulerSampler::step`
    /// computes in the U-Net output's dtype and joins it with x_t, and `price_step` casts its input
    /// to the model dtype first, so the steady state is the first state joined with eps.
    pub(crate) fn ancestral_states(&self, regime: &Regime) -> Result<(Dtype, Dtype)> {
        let rows = regime.conditioning_rows();
        let (conditioning, pooled) = self.price_text_conditioning(rows, &mut Quanta::new())?;
        let encoder_x = price_tensor(
            &[rows, regime.context_tokens, self.arch.context_dim],
            conditioning,
        )?;
        let text_emb = price_tensor(&[rows, self.te.1.pooled_width()], pooled)?;
        let sequential = regime.sampling.guidance() > 1.0 && !regime.cfg_batching.is_batched();
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED
            .with_cfg_batching(regime.cfg_batching)
            .with_cfg_row(sequential.then_some(0));
        let first = regime.initial_state_dtype();
        let (lh, lw) = (
            regime.height / self.arch.latent_scale,
            regime.width / self.arch.latent_scale,
        );
        let eps = self.price_step(
            regime,
            price_tensor(&[regime.model_rows(), lh, lw, 4], first)?,
            &encoder_x,
            &text_emb,
            &[],
            None,
            plan,
            &mut Quanta::new(),
            &mut Vec::new(),
        )?;
        Ok((first, Dtype::from_promoting_types(first, eps)))
    }

    pub(crate) fn price_text_conditioning(
        &self,
        rows: i32,
        out: &mut Quanta,
    ) -> Result<(Dtype, Dtype)> {
        let (h1, _) = self
            .te
            .0
            .price_forward_with_hidden(rows, self.arch.text_tokens, out)?;
        let (h2, pooled) = self
            .te
            .1
            .price_forward_with_hidden(rows, self.arch.text_tokens, out)?;
        // Native pipeline::text_conditioning concatenates the two penultimate states on -1.
        if h1.shape.len() != h2.shape.len()
            || h1.shape[..h1.shape.len() - 1] != h2.shape[..h2.shape.len() - 1]
        {
            return Err(q::native_shape_failure(
                "CLIP penultimate states do not concatenate",
            ));
        }
        Ok((Dtype::from_promoting_types(h1.dtype, h2.dtype), pooled))
    }

    /// Actual fused warm/text/heavy load recipe. There are no Directory component-load guards
    /// on from-weights constructors. Source parents are logical, never a backing-size credit.
    fn fused_load_points(&self, regime: &Regime) -> Option<Vec<PricedQuantum>> {
        let fused = self.fused.as_ref()?;
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
        // A natively failing staged conditioning never reaches the heavy load; `price` drops these
        // heavy points in that world, so its unknown conditioning bytes are never read.
        let conditioned = if regime.staged {
            self.price_text_conditioning(regime.conditioning_rows(), &mut Quanta::new())
                .ok()
        } else {
            None
        };
        let conditioning = if let Some((context, pooled)) = conditioned {
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
        let staged = regime.staged;
        let warm_text = if staged { 0 } else { self.text() };

        let text_stage_end;
        if let Some(points) = self.fused_load_points(regime) {
            // Native chronology: the text load and text packs precede the heavy load guard.
            text_stage_end = points
                .iter()
                .position(|point| point.label == q::LDM_HEAVY_LOAD)
                .unwrap_or(points.len());
            for point in points {
                take(
                    &mut items,
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
                &mut items,
                IosPhase::Load,
                0,
                vec![(q::TEXT_ENCODER_1_LOAD, self.te_bytes[0].floor)],
            );
            take(
                &mut items,
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
                        &mut items,
                        IosPhase::Load,
                        other + point.before_resident,
                        vec![(q::LOAD_QUANTIZE, point.scratch)],
                    );
                }
            }
            text_stage_end = items.len();
            let mut loaded = warm_text;
            take(
                &mut items,
                IosPhase::Load,
                loaded,
                vec![(q::UNET_LOAD, self.unet_bytes.floor)],
            );
            loaded += self.unet_bytes.resident;
            for point in &self.adapter_load.quanta {
                take(
                    &mut items,
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
                &mut items,
                IosPhase::Load,
                loaded,
                vec![(vae_label, self.vae_bytes.floor)],
            );
            loaded += self.vae_bytes.resident;
            for (load, _) in &self.control_bytes {
                take(
                    &mut items,
                    IosPhase::Load,
                    loaded,
                    vec![(q::CONTROLNET_LOAD, load.floor)],
                );
                loaded += load.resident;
            }
            if self.ip.is_some() {
                take(
                    &mut items,
                    IosPhase::Load,
                    loaded,
                    vec![(q::IP_IMAGE_ENCODER_LOAD, self.ip_encoder_bytes.floor)],
                );
                loaded += self.ip_encoder_bytes.resident;
                take(
                    &mut items,
                    IosPhase::Load,
                    loaded,
                    vec![(q::IP_ADAPTER_LOAD, self.ip_adapter_bytes.floor)],
                );
            }

            for point in &self.load_quantize {
                take(
                    &mut items,
                    IosPhase::Load,
                    warm_text.saturating_add(point.before_resident),
                    vec![(q::LOAD_QUANTIZE, point.scratch)],
                );
            }
        }
        let loads_end = items.len();
        if let Some(failure) = self.native_failure {
            // A build-stage pack failure: the admitted native prefix ends at its pack guard.
            if failure == NativeFailure::TextPack {
                items.truncate(text_stage_end);
            } else if staged {
                // SequentialText ran the conditioning before the heavy load that failed.
                let mut text = Quanta::new();
                let conditioned =
                    self.price_text_conditioning(regime.conditioning_rows(), &mut text);
                take(&mut items, IosPhase::Conditioning, self.text(), text);
                if let Err(error) = conditioned {
                    if !q::is_native_shape_failure(&error) {
                        return Err(error);
                    }
                    let mut kept = items[..text_stage_end].to_vec();
                    kept.extend(items[loads_end..].iter().cloned());
                    items = kept;
                }
            }
            return Ok(self.finish(regime, items, Vec::new(), false, false));
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
                &mut items,
                IosPhase::Load,
                heavy + warm_text + inputs,
                vec![(REQUEST_INPUTS, 0)],
            );
        }
        // Conditioning: both CLIP towers over the guided pair's two token rows, one window at a
        // time (each window is its own forward).
        let rows = regime.conditioning_rows();
        let mut text = Quanta::new();
        let conditioned = self.price_text_conditioning(rows, &mut text);
        let text_resident = self.text() + if staged { 0 } else { heavy };
        take(&mut items, IosPhase::Conditioning, text_resident, text);
        let (conditioning_dtype, pooled) = match conditioned {
            Ok(dtypes) => dtypes,
            Err(error) if q::is_native_shape_failure(&error) => {
                if staged {
                    // SequentialText: the failing conditioning precedes every heavy load.
                    let mut kept = items[..text_stage_end].to_vec();
                    kept.extend(
                        items[loads_end..]
                            .iter()
                            .filter(|item| item.phase == IosPhase::Conditioning)
                            .cloned(),
                    );
                    items = kept;
                }
                return Ok(self.finish(regime, items, Vec::new(), false, false));
            }
            Err(error) => return Err(error),
        };

        if let (Some(ip), Route::IpAdapter) = (&self.ip, regime.route) {
            let source = regime.reference_image.ok_or_else(|| {
                Error::Unsupported("sdxl IP preprocessing has no actual source facts".into())
            })?;
            let geometry = ip.preprocess_geometry(source)?;
            take(
                &mut items,
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
                    &mut items,
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
            let encoded = self.vae.price_encode(
                &price_tensor(&[1, h, w, 3], Dtype::Float32)?,
                tile,
                &mut quanta,
            );
            take(
                &mut items,
                IosPhase::Encode,
                heavy + warm_text + init_image,
                quanta,
            );
            if let Err(error) = encoded {
                if !q::is_native_shape_failure(&error) {
                    return Err(error);
                }
                // The init encode failed natively after its admitted tile quanta.
                return Ok(self.finish(regime, items, Vec::new(), false, false));
            }
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
                &mut items,
                IosPhase::Encode,
                heavy + warm_text + control_images + embeds,
                quanta,
            );
        }

        // Denoise: one guided step (two sequential batch-1 halves, or one batched pair), the
        // ControlNet branches beside the U-Net, the IP tokens in the cross-attention.
        // add_noise(_with) uses f32 noise; Euler retains that promoted sampler state.
        // Ancestral forwards preserve that promoted dtype; curated callbacks retain their
        // historical pre-CFG compute cast independently of sampler state.
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
        let mut native_failed = false;
        loop {
            if discrete_steps == Some(0) {
                break;
            }
            let mut forward = Quanta::new();
            let mut coincident_control = Vec::new();
            let stepped = self.price_step(
                regime,
                price_tensor(&[batch, lh, lw, 4], step_dtype)?,
                &encoder_x,
                &text_emb,
                &cond_embeds,
                ip_tokens.as_ref(),
                plan,
                &mut forward,
                &mut coincident_control,
            );
            // A native shape failure inside the first forward's first half ends the world after
            // that half's admitted quanta: no second half, sampler update, later epoch or decode.
            let (eps, failed) = match stepped {
                Ok(eps) => (eps, false),
                Err(error)
                    if q::is_native_shape_failure(&error)
                        && coincident_control.len() == forward.len() =>
                {
                    (step_dtype, true)
                }
                Err(error) => return Err(error),
            };
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
                // A failing first forward has no completed earlier serial latent.
                let completed = if failed {
                    0
                } else {
                    elements * q::dtype_bytes(next) * u64::from(regime.count - 1)
                };
                // Both conditioning tensors and model::generate's f32 [rows,6] time_ids remain
                // owned in the enclosing generation scope (pipeline::text_time_ids).
                let conditioning = price_bytes(&encoder_x)
                    + price_bytes(&text_emb)
                    + rows as u64 * 6 * q::dtype_bytes(Dtype::Float32);
                let halves = if sequential && !failed { 2 } else { 1 };
                let shed_ip = crate::model::sheds_ip_image_encoder(staged)
                    && regime.route == Route::IpAdapter;
                let structural = self.heavy(shed_ip) + warm_text;
                for half in 0..halves {
                    for (ordinal, &(label, scratch)) in forward.iter().enumerate() {
                        // Only the stem's guard includes the original model-input tensor. Later
                        // guards price their local activation, while forward_eps still borrows
                        // the caller's original input. Do not count that input twice at the stem.
                        let overlap = if label == q::UNET_STEM || label == q::CONTROL_STEM {
                            elements
                                * q::dtype_bytes(
                                    if regime.sampling.route() == SamplingRoute::Ancestral {
                                        step_dtype
                                    } else {
                                        crate::model::DTYPE
                                    },
                                )
                                * batch as u64
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
                if failed {
                    quanta.extend(forward);
                    native_failed = true;
                    break;
                }
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
                    let completed = if failed {
                        0
                    } else {
                        elements * q::dtype_bytes(Dtype::Float32) * u64::from(regime.count - 1)
                    };
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
                        for half in 0..if sequential && !failed { 2 } else { 1 } {
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
                        if failed {
                            continue;
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
                if failed {
                    native_failed = true;
                    break;
                }
                if next == step_dtype {
                    break;
                }
                step_dtype = next;
            }
        }
        let latents = price_tensor(&[1, lh, lw, 4], step_dtype)?;
        // A failing forward never reached the sampler update.
        if !native_failed {
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
        }
        let shed_ip =
            crate::model::sheds_ip_image_encoder(staged) && regime.route == Route::IpAdapter;
        // A real init owner is the VAE's F32 mean, independently of the noised sampler state. The mask
        // path additionally retains its F32 noise/mask. ControlNet+reference also owns that init;
        // an IP image prompt does not. These are owners, not a fitted phase correction.
        let elements = (lh as u64) * (lw as u64) * 4;
        let held = regime.initial_fixed_inputs(elements)
            + cond_embeds.iter().map(price_bytes).sum::<u64>()
            + ip_tokens.as_ref().map_or(0, price_bytes);
        take(
            &mut items,
            IosPhase::Denoise,
            self.heavy(shed_ip) + warm_text + held,
            quanta,
        );
        if native_failed {
            return Ok(self.finish(
                regime,
                items,
                denoise_windows,
                profiled_discrete || profiled_curated,
                ancestral_steps == Some(0),
            ));
        }

        // Decode: the staged decode holds the VAE alone; a warm one decodes beside everything.
        let mut quanta = Quanta::new();
        let decoded = self
            .vae
            .price_decode(&latents, regime.tiling.as_ref(), &mut quanta);
        if let Err(error) = &decoded {
            if !q::is_native_shape_failure(error) {
                return Err(decoded.expect_err("checked decode error"));
            }
        }
        // A native decode shape failure keeps the decode quanta admitted before it.
        if let Ok(decoded) = &decoded {
            quanta.push((
                q::DECODED_TO_IMAGE,
                q::decoded_to_image_bytes_from_metadata(decoded.elements()),
            ));
        }
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
                &mut items,
                IosPhase::Decode,
                // `finals` owns EVERY completed serial latent throughout the decode loop; the
                // borrowed/clone caller handle aliases its entry, not a second latent allocation.
                decode_resident + caller * u64::from(regime.count) + retained_cast,
                vec![(label, scratch)],
            );
        }

        Ok(self.finish(
            regime,
            items,
            denoise_windows,
            profiled_discrete || profiled_curated,
            ancestral_steps == Some(0),
        ))
    }

    /// Fused source-parent residency, the window/summary assembly and the regime's not-run phases,
    /// shared by complete ledgers and native-failure prefixes.
    fn finish(
        &self,
        regime: &Regime,
        mut items: Vec<PricedQuantum>,
        mut denoise_windows: Vec<IosWindowPrice>,
        profiled: bool,
        ancestral_zero: bool,
    ) -> (IosPricing, Vec<PricedQuantum>) {
        let staged = regime.staged;
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
            if (profiled) && item.phase == IosPhase::Denoise {
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
        if ancestral_zero {
            pricing.phase_not_run(
                IosPhase::Denoise,
                "actual ancestral strength schedule has zero steps",
            );
        }
        if !regime.has_init && regime.route != Route::Control {
            pricing.phase_not_run(IosPhase::Encode, "no init or active ControlNet encode");
        }
        (pricing, items)
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
        owners: &mut Vec<u64>,
    ) -> Result<Dtype> {
        let first_quantum = quanta.len();
        // Curated/acceleration callbacks keep their historical explicit pre-forward cast;
        // ancestral callbacks pass their actual (possibly promoted F32) sampler operand.
        let input_dtype = if regime.sampling.route() == SamplingRoute::Ancestral {
            x.dtype()
        } else {
            crate::model::DTYPE
        };
        let x = price_tensor(x.shape(), input_dtype)?;
        let batch = x.shape()[0];
        let mut control: Option<(Vec<PriceTensor>, PriceTensor)> = None;
        let mut pending_control = q::ControlMergeWork::default();
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
                // ControlNet branches reuse U-Net module paths; only the U-Net/VAE walk traces.
                let (down, mid) = q::without_module_trace(|| {
                    branch.price_forward_with_effects(
                        &x,
                        crate::unet::pricing::ControlConditioning {
                            cond_embed: &row,
                            encoder_x,
                            text_emb,
                        },
                        plan,
                        quanta,
                        Some(&mut *owners),
                        &effects,
                    )
                })?;
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
        // Combined ControlNet residuals retain their native promoted dtype, exactly as on host.
        let (merged_down_bytes, merged_mid_bytes) =
            control.as_ref().map_or((0, 0), |(down, mid)| {
                (down.iter().map(price_bytes).sum::<u64>(), price_bytes(mid))
            });
        let unet_first = quanta.len();
        let effects = self
            .unet_effects
            .as_ref()
            .map_or_else(q::EffectScope::native, q::EffectScope::root);
        let forward = self.unet.price_forward_with_effects(
            &x,
            crate::unet::pricing::UNetConditioning {
                encoder_x,
                text_emb,
                control: control.as_ref(),
                ip: ip_tokens.map(|t| (t, regime.ip_scale)),
            },
            plan,
            pending_control,
            quanta,
            &effects,
        );
        // Owners for every admitted U-Net quantum, including a failing forward's prefix.
        let (mut down_complete, mut mid_complete) = (false, false);
        for (label, _) in &quanta[unet_first..] {
            // Only the ACTUAL existing injection finish may replace raw source parents with the
            // materialized sum's logical owner. This remains unclassified, not backing equality:
            // inherited storage/cache oversizing and native/process domains are not inferred.
            owners.push(
                selected_rows
                    + if down_complete {
                        merged_down_bytes
                    } else {
                        raw_down
                    }
                    + if mid_complete {
                        merged_mid_bytes
                    } else {
                        raw_mid
                    },
            );
            if *label == q::CONTROL_DOWN_INJECTION {
                down_complete = true;
            }
            if *label == q::CONTROL_MID_INJECTION {
                mid_complete = true;
            }
        }
        let eps = forward?;
        if owners.len() != quanta.len() - first_quantum {
            return Err(Error::Unsupported(
                "sdxl control coincident-owner ledger is incomplete".into(),
            ));
        }
        Ok(eps)
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
