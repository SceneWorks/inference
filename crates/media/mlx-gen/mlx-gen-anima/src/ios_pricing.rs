//! Header-only, per-phase COUNTED pricing of one iOS `anima_turbo` request (epic 16774): the
//! [`mlx_gen::ios_pricing`] interface for this crate.
//!
//! Every price is the bytes the production guard of that quantum checks, computed by the SAME
//! formula functions the guard calls (the Qwen3 tower's, the conditioner's, the DiT's, the
//! Qwen-Image VAE's `decode_quanta`), over modules built from the snapshot's safetensors HEADERS:
//! each tensor is a shape-only `zeros` placeholder that is never evaluated, so no tensor payload is
//! read and nothing is allocated. Resident bytes are the components (and the small carried tensors)
//! held through each phase under the request's regime — staging, decode tile, attention chunk,
//! DiT window — resolved by the same `memory_strategy` resolvers `generate` uses.
//!
//! Phases follow this crate's own phase boundaries (`model.rs`'s `calibration_fault` points and the
//! staged schedule): the Qwen3 tower and the conditioner-input materialization are
//! [`IosPhase::Conditioning`]; the `AnimaTextConditioner` (bundled in the DiT checkpoint, run inside
//! the heavy phase after `MemoryPhase::Denoise`'s fault point), the DiT and the latent
//! materialization are [`IosPhase::Denoise`]; the Qwen-Image VAE decode is [`IosPhase::Decode`].
//! Anima is text-to-image only: there is no [`IosPhase::Encode`].

use std::collections::HashMap;

use mlx_gen::gen_core::weightsmeta::{Dtype as StoredDtype, SafetensorsTensorHeader};
use mlx_gen::gen_core::{
    adapter_stack_resident_bytes, AdapterResidencyMode, GenerationMemory, LoadSpec, MemoryMode,
};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosPricingRequest};
use mlx_gen::ios_quantum::{dtype_bytes, sampler_step_bytes, tensor_bytes, Backend};
use mlx_gen::sampler::SAMPLER_STEP_QUANTUM;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_qwen_image::vae::blocks::{
    DECODE_ATTENTION_QUANTUM, DECODE_RESNET_QUANTUM, DECODE_UPSAMPLE_QUANTUM,
};
use mlx_gen_qwen_image::vae::{
    DECODE_FINAL_NORMALIZE_QUANTUM, DECODE_HEAD_QUANTUM, DECODE_OUTPUT_QUANTUM,
    DECODE_TILE_FOLD_QUANTUM,
};
use mlx_rs::Dtype;

use crate::conditioner::{
    AnimaTextConditioner, CONDITIONER_BLOCK_QUANTUM, CONDITIONER_EMBED_QUANTUM,
    CONDITIONER_OUTPUT_QUANTUM,
};
use crate::config::{ConditionerConfig, DitConfig, Qwen3Config, Variant};
use crate::loader::{
    CONDITIONING_LOAD_QUANTUM, HEAVY_LOAD_QUANTUM, PREFIX_ANCHOR, TEXT_LOAD_QUANTUM,
    VAE_LOAD_QUANTUM,
};
use crate::model::{CONDITIONING_INPUTS_QUANTUM, LATENTS_QUANTUM};
use crate::text_encoder::{AnimaQwen3, TE_EMBED_QUANTUM, TE_LAYER_QUANTUM, TE_NORM_QUANTUM};
use crate::transformer::{CosmosDiT, DIT_BLOCK_QUANTUM, DIT_EMBED_QUANTUM, DIT_OUTPUT_QUANTUM};
use crate::vae::QwenVae;

/// Width of every RoPE table this family builds (the Qwen3/conditioner `TextRope` and the Cosmos
/// RoPE are f32 host tables). The rotated stream — and everything after it in a Qwen3 layer, a
/// conditioner block and a DiT block — runs at `max(input width, this)`
/// (`text_encoder::layer_element_bytes`, `transformer::block_element_bytes`).
const ROPE_TABLE_BYTES: u64 = 4;

/// The facts a price is computed from: the three files' tensor headers, the model configs, and the
/// load/route facts. [`snapshot_facts`] reads them for a real load; a CPU proof builds them over a
/// tiny fixture's files, and the reachable-sizes prediction over a tier-projected inventory.
#[derive(Clone, Debug)]
pub(crate) struct AnimaPricingFacts {
    /// The DiT checkpoint (Cosmos DiT + the bundled `llm_adapter` conditioner).
    pub(crate) dit: Vec<SafetensorsTensorHeader>,
    pub(crate) text_encoder: Vec<SafetensorsTensorHeader>,
    pub(crate) vae: Vec<SafetensorsTensorHeader>,
    pub(crate) configs: (DitConfig, ConditionerConfig, Qwen3Config),
    /// Whether the variant runs a CFG pair (an uncond prompt): never for Turbo.
    pub(crate) uses_cfg: bool,
    /// The resident LoRA/LoKr stack ([`adapter_stack_resident_bytes`], additive).
    pub(crate) adapter_bytes: u64,
    /// Whether the load can window the DiT (`memory_strategy::streamable`).
    pub(crate) streamable_load: bool,
}

/// The request geometry a price is computed at.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AnimaPricingWork {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Images per request (`GenerationRequest::count`).
    pub(crate) count: u32,
    /// Encoded Qwen3 tokens (the conditioner's source sequence).
    pub(crate) qwen_tokens: u32,
    /// Encoded T5 query tokens (the conditioner's target sequence).
    pub(crate) t5_tokens: u32,
    /// The kernel backend the guards price SDPA scores for (`Backend::current()` where they run).
    pub(crate) backend: Backend,
}

/// One priced quantum: what its guard checks (`scratch_bytes`, under `label`) and what is resident
/// beside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PricedQuantum {
    pub(crate) phase: IosPhase,
    pub(crate) label: &'static str,
    pub(crate) resident_bytes: u64,
    pub(crate) scratch_bytes: u64,
}

/// The [`mlx_gen::ios_pricing::IosPricer`] for `anima_turbo` (and the other Anima variants, which
/// share the architecture): read the load's headers and price the request's phases.
pub fn ios_pricing(req: &IosPricingRequest<'_>) -> Result<IosPricing> {
    let variant = Variant::from_id(req.provider)
        .ok_or_else(|| Error::Unsupported(format!("{}: not an Anima provider", req.provider)))?;
    if !matches!(req.mode, MemoryMode::TextToImage) || req.geometry.reference_count != 0 {
        return Err(Error::Unsupported(format!(
            "{}: anima is text-to-image only; got {:?} with {} references",
            req.provider, req.mode, req.geometry.reference_count
        )));
    }
    let facts = snapshot_facts(req.spec, variant)?;
    // Anima tokenizes `padding="longest"` up to the 512-token window: a request-dependent window
    // prices at it, an unstated one at the window's maximum.
    let max = crate::tokenizer::MAX_LEN as u32;
    let tokens = match req.token_window {
        0 => max,
        window => window.min(max),
    };
    let work = AnimaPricingWork {
        width: req.geometry.width,
        height: req.geometry.height,
        count: req.geometry.batch.max(1),
        qwen_tokens: tokens,
        t5_tokens: tokens,
        // A physical iOS device runs the Metal kernels.
        backend: Backend::Metal,
    };
    Ok(fold(&price_quanta(&facts, work, req.execution)?))
}

/// The phase every quantum label this route emits runs in — this crate's own labels, the shared
/// curated-solver step (`mlx_gen::sampler`) its denoise runs, and the Qwen-Image VAE decode labels
/// (`mlx-gen-qwen-image`) its decode runs. `None` for a label it never emits.
pub fn ios_quantum_phase(label: &str) -> Option<IosPhase> {
    Some(match label {
        TEXT_LOAD_QUANTUM | HEAVY_LOAD_QUANTUM | VAE_LOAD_QUANTUM | CONDITIONING_LOAD_QUANTUM => {
            IosPhase::Load
        }
        TE_EMBED_QUANTUM | TE_LAYER_QUANTUM | TE_NORM_QUANTUM | CONDITIONING_INPUTS_QUANTUM => {
            IosPhase::Conditioning
        }
        CONDITIONER_EMBED_QUANTUM
        | CONDITIONER_BLOCK_QUANTUM
        | CONDITIONER_OUTPUT_QUANTUM
        | DIT_EMBED_QUANTUM
        | DIT_BLOCK_QUANTUM
        | DIT_OUTPUT_QUANTUM
        | LATENTS_QUANTUM
        | SAMPLER_STEP_QUANTUM => IosPhase::Denoise,
        DECODE_HEAD_QUANTUM
        | DECODE_RESNET_QUANTUM
        | DECODE_ATTENTION_QUANTUM
        | DECODE_UPSAMPLE_QUANTUM
        | DECODE_OUTPUT_QUANTUM
        | DECODE_TILE_FOLD_QUANTUM
        | DECODE_FINAL_NORMALIZE_QUANTUM => IosPhase::Decode,
        _ => return None,
    })
}

/// Read a load's pricing facts: the headers of the three files the loader reads (resolved exactly
/// as it resolves them) and the model configs it builds.
pub(crate) fn snapshot_facts(spec: &LoadSpec, variant: Variant) -> Result<AnimaPricingFacts> {
    let root = crate::loader::resolve_split_files(&spec.weights)?;
    let headers = |path: std::path::PathBuf| {
        mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&path)
            .map_err(|e| Error::Msg(format!("{}: {e}", path.display())))
    };
    Ok(AnimaPricingFacts {
        dit: headers(root.join("diffusion_models").join(variant.dit_filename()))?,
        text_encoder: headers(root.join(crate::loader::TEXT_ENCODER_FILE))?,
        vae: headers(root.join(crate::loader::VAE_FILE))?,
        configs: crate::loader::model_configs(),
        uses_cfg: variant.uses_cfg(),
        adapter_bytes: adapter_stack_resident_bytes(&spec.adapters, AdapterResidencyMode::Additive)
            .ok_or_else(|| {
                Error::Unsupported("anima: an adapter source could not be sized".to_owned())
            })?,
        streamable_load: crate::memory_strategy::streamable(spec),
    })
}

/// Fold priced quanta into the per-phase price (each phase's worst resident + scratch quantum).
pub(crate) fn fold(quanta: &[PricedQuantum]) -> IosPricing {
    let mut pricing = IosPricing::default();
    for q in quanta {
        pricing.take(q.phase, q.resident_bytes, q.scratch_bytes, q.label);
    }
    pricing
}

/// The MLX dtype of a stored tensor.
fn mlx_dtype(dtype: StoredDtype) -> Result<Dtype> {
    Ok(match dtype {
        StoredDtype::BOOL => Dtype::Bool,
        StoredDtype::U8 => Dtype::Uint8,
        StoredDtype::I8 => Dtype::Int8,
        StoredDtype::U16 => Dtype::Uint16,
        StoredDtype::I16 => Dtype::Int16,
        StoredDtype::F16 => Dtype::Float16,
        StoredDtype::BF16 => Dtype::Bfloat16,
        StoredDtype::U32 => Dtype::Uint32,
        StoredDtype::I32 => Dtype::Int32,
        StoredDtype::F32 => Dtype::Float32,
        StoredDtype::U64 => Dtype::Uint64,
        StoredDtype::I64 => Dtype::Int64,
        StoredDtype::F64 => Dtype::Float64,
        other => {
            return Err(Error::Unsupported(format!(
                "anima pricing: no MLX dtype for stored {other:?}"
            )))
        }
    })
}

/// Shape-only placeholders for every header: lazy `zeros`, never evaluated (nothing is allocated
/// and no payload is read), so a module built over them answers its quanta's formulas.
fn placeholders(headers: &[SafetensorsTensorHeader]) -> Result<Weights> {
    let mut map = HashMap::with_capacity(headers.len());
    for h in headers {
        let shape = h
            .shape
            .iter()
            .map(|&d| i32::try_from(d))
            .collect::<std::result::Result<Vec<i32>, _>>()
            .map_err(|_| Error::Unsupported(format!("{}: dimension overflow", h.name)))?;
        map.insert(
            h.name.clone(),
            mlx_rs::ops::zeros_dtype(&shape, mlx_dtype(h.dtype)?)?,
        );
    }
    Ok(Weights::from_map(map))
}

fn elements(h: &SafetensorsTensorHeader) -> Vec<u64> {
    h.shape.iter().map(|&d| d as u64).collect()
}

fn stored(headers: &[SafetensorsTensorHeader], keep: impl Fn(&str) -> bool) -> u64 {
    headers
        .iter()
        .filter(|h| keep(&h.name))
        .fold(0u64, |a, h| a.saturating_add(h.data_bytes))
}

/// Every quantum the request runs, in production order, with what is resident beside it.
pub(crate) fn price_quanta(
    facts: &AnimaPricingFacts,
    work: AnimaPricingWork,
    execution: &GenerationMemory,
) -> Result<Vec<PricedQuantum>> {
    let (dit_cfg, cond_cfg, qwen_cfg) = facts.configs;
    let memory = Some(*execution);
    // The request's levers, resolved exactly as `Anima::generate_impl` resolves them.
    let staged = crate::memory_strategy::stage_residency_for(memory, false);
    let attention = crate::memory_strategy::attention_plan_for(memory);
    let window = crate::memory_strategy::transformer_window_size_for(memory)?;
    let tiling = crate::memory_strategy::decode_tiling_for(memory)?;
    if window.is_some() && !(facts.streamable_load && staged) {
        return Err(Error::Unsupported(
            "anima: bounded transformer residency needs a DeferredMaterialization load and staged \
             residency in the same request"
                .to_owned(),
        ));
    }
    let windowed = window.is_some();

    // ---- Modules over shape-only placeholders (the loaders' own construction).
    let prefix = facts
        .dit
        .iter()
        .find_map(|h| h.name.strip_suffix(PREFIX_ANCHOR))
        .ok_or_else(|| Error::Msg("anima: DiT root prefix not found in headers".into()))?
        .to_string();
    let blocks_prefix = format!("{prefix}.blocks.");
    let dit_w = placeholders(&facts.dit)?;
    let dit = CosmosDiT::from_weights(&dit_w, &prefix, dit_cfg)?;
    let cond =
        AnimaTextConditioner::from_weights(&dit_w, &format!("{prefix}.llm_adapter"), cond_cfg)?;
    let te = AnimaQwen3::from_weights(&placeholders(&facts.text_encoder)?, "model", &qwen_cfg)?;
    let (vae, _) = crate::vae::vae_from_source(&placeholders(&facts.vae)?)?;

    // ---- Byte facts.
    let te_bytes = stored(&facts.text_encoder, |_| true);
    // The heavy load floor (`loader::load_heavy_phase_with_stream`): a windowed load reads only the
    // non-block DiT tensors and the conditioner; the block bodies are read per window.
    let heavy_floor = if windowed {
        stored(&facts.dit, |n| !n.starts_with(&blocks_prefix))
    } else {
        stored(&facts.dit, |_| true)
    };
    let block_read = |i: usize| {
        let p = format!("{blocks_prefix}{i}.");
        stored(&facts.dit, |n| n.starts_with(&p))
    };
    let vae_width = dtype_bytes(crate::vae::VAE_DTYPE);
    let vae_resident: u64 = facts
        .vae
        .iter()
        .map(|h| tensor_bytes(vae_width, &elements(h)))
        .sum();
    // `ios_quantum::before_load_cast(.., VAE_DTYPE width)`: every stored tensor plus its cast copy.
    let vae_cast_floor = stored(&facts.vae, |_| true).saturating_add(vae_resident);
    let latent_shape = crate::pipeline::latent_shape(work.width, work.height);
    let latent: [u64; 5] = latent_shape.map(|d| d.max(0) as u64);
    let latent_elements: u64 = latent.iter().product();
    let latent_bytes = tensor_bytes(4, &latent);
    let (pt, ph, pw) = dit_cfg.patch_size;
    let image_tokens = (latent[2] / pt as u64) * (latent[3] / ph as u64) * (latent[4] / pw as u64);
    let (s, st) = (work.qwen_tokens as u64, work.t5_tokens as u64);
    let (s_i, st_i) = (work.qwen_tokens as i32, work.t5_tokens as i32);
    let source = qwen_cfg.hidden_size.max(0) as u64;
    let prompts = 1 + u64::from(facts.uses_cfg);
    let count = u64::from(work.count.max(1));
    let te_width = dtype_bytes(te.compute_dtype());
    // The tower's rotated stream; its final norm — and so the conditioner inputs, which the
    // conditioner runs in (`pipeline::conditioner_forward`) — come out at this width.
    let te_stream = te_width.max(ROPE_TABLE_BYTES);
    let cond_stream = te_stream.max(ROPE_TABLE_BYTES);
    let dit_width = dtype_bytes(crate::pipeline::DIT_DTYPE);
    let dit_stream = dit_width.max(ROPE_TABLE_BYTES);
    // Carried between phases: the conditioner inputs (masked Qwen3 states + T5 ids), the
    // conditioner outputs, every image's latent.
    let cond_inputs =
        prompts * (tensor_bytes(te_stream, &[1, s, source]) + tensor_bytes(4, &[1, st]));
    let cond_outputs = prompts
        * tensor_bytes(
            te_stream,
            &[
                1,
                cond_cfg.min_sequence_length as u64,
                cond_cfg.target_dim as u64,
            ],
        );
    let latents = count * latent_bytes;
    let tiled = tiling.as_ref().is_some_and(|cfg| {
        cfg.needs_tiling(
            mlx_gen::tiling::VaeTiling::QWEN_IMAGE,
            latent_shape[2],
            latent_shape[3],
            latent_shape[4],
        )
    });
    // Carried through the decode tail: the mid-block output it reads (the decoder `conv_in` width at
    // latent extent) and, tiled, the full-size RGB slab and output accumulators.
    let mid_channels = facts
        .vae
        .iter()
        .find(|h| crate::vae::convert_vae_key(&h.name) == "decoder.conv_in.weight")
        .and_then(|h| h.shape.first().copied())
        .unwrap_or(0) as u64;
    let decode_carried = tensor_bytes(
        vae_width,
        &[1, mid_channels, latent[2], latent[3], latent[4]],
    ) + if tiled {
        2 * tensor_bytes(vae_width, &[1, 3, latent[2], 8 * latent[3], 8 * latent[4]])
    } else {
        0
    };

    // ---- Residency per phase (staged sheds the tower before the heavy load and the DiT +
    // conditioner before the decode; a warm request loads the tower, then the heavy bundle).
    let heavy_resident = heavy_floor + facts.adapter_bytes + vae_resident;
    let (text_load, heavy_load, vae_load, conditioning, denoise, decode) = if staged {
        (
            0,
            cond_inputs,
            cond_inputs + heavy_floor,
            te_bytes,
            heavy_resident + cond_inputs + cond_outputs + latents,
            vae_resident + latents + decode_carried,
        )
    } else {
        let warm = te_bytes + heavy_resident;
        (
            0,
            te_bytes,
            te_bytes + heavy_floor,
            warm,
            warm + cond_inputs + cond_outputs + latents,
            warm + cond_inputs + cond_outputs + latents + decode_carried,
        )
    };

    let mut out = Vec::new();
    let mut push = |phase, label, resident_bytes, scratch_bytes| {
        out.push(PricedQuantum {
            phase,
            label,
            resident_bytes,
            scratch_bytes,
        })
    };
    let max = |it: &mut dyn Iterator<Item = u64>| it.max().unwrap_or(0);
    use IosPhase::{Conditioning, Decode, Denoise, Load};

    // Loads, in the order the residency drives them.
    push(Load, TEXT_LOAD_QUANTUM, text_load, te_bytes);
    // Conditioning: the Qwen3 tower over the prompt(s), then (staged) the inputs' materialization.
    for _ in 0..prompts {
        push(
            Conditioning,
            TE_EMBED_QUANTUM,
            conditioning,
            te.embed_body_bytes(1, s_i),
        );
        push(
            Conditioning,
            TE_LAYER_QUANTUM,
            conditioning,
            max(&mut (0..te.num_layers()).map(|i| te.layer(i, 1, s_i, te_stream, work.backend).1)),
        );
        push(
            Conditioning,
            TE_NORM_QUANTUM,
            conditioning,
            te.norm_bytes(1, s_i, te_stream),
        );
    }
    if staged {
        push(
            Conditioning,
            CONDITIONING_INPUTS_QUANTUM,
            conditioning,
            prompts * AnimaTextConditioner::mask_multiply_bytes_for(s * source),
        );
    }
    push(Load, HEAVY_LOAD_QUANTUM, heavy_load, heavy_floor);
    push(Load, VAE_LOAD_QUANTUM, vae_load, vae_cast_floor);
    // Denoise: the conditioner per prompt, then every step's DiT forward(s).
    push(
        Denoise,
        CONDITIONER_EMBED_QUANTUM,
        denoise,
        cond.embed_body_bytes(1, st_i, s_i, te_stream)
            .saturating_add(AnimaTextConditioner::mask_multiply_bytes_for(s * source)),
    );
    push(
        Denoise,
        CONDITIONER_BLOCK_QUANTUM,
        denoise,
        max(&mut cond
            .blocks()
            .iter()
            .map(|b| b.quantum_scratch(1, st_i, s_i, cond_stream, work.backend))),
    );
    push(
        Denoise,
        CONDITIONER_OUTPUT_QUANTUM,
        denoise,
        cond.output_quantum_scratch(1, st_i, cond_stream),
    );
    push(
        Denoise,
        DIT_EMBED_QUANTUM,
        denoise,
        dit.embed_quantum_scratch_for(&latent_shape, crate::pipeline::DIT_DTYPE),
    );
    let context = cond_cfg.min_sequence_length as i32;
    push(
        Denoise,
        DIT_BLOCK_QUANTUM,
        denoise,
        max(&mut dit.blocks().iter().enumerate().map(|(i, b)| {
            b.quantum_scratch(
                1,
                image_tokens as i32,
                context,
                attention,
                dit_stream,
                if windowed { block_read(i) } else { 0 },
                work.backend,
            )
        })),
    );
    push(
        Denoise,
        DIT_OUTPUT_QUANTUM,
        denoise,
        dit.output_quantum_scratch(1, image_tokens as i32, dit_stream),
    );
    // The curated solver's step boundary (`mlx_gen::sampler`): one latent per step.
    push(
        Denoise,
        SAMPLER_STEP_QUANTUM,
        denoise,
        sampler_step_bytes(latent_elements),
    );
    if staged {
        push(
            Denoise,
            LATENTS_QUANTUM,
            denoise,
            count * sampler_step_bytes(latent_elements),
        );
    }
    // Decode: every VAE quantum (the mid-block attention query-chunked, as on iOS), per image.
    for (label, scratch) in vae.decode_quanta(
        latent,
        tiling.as_ref(),
        QwenVae::ios_decode_attention_plan(),
        vae_width,
    ) {
        push(Decode, label, decode, scratch);
    }
    Ok(out)
}
