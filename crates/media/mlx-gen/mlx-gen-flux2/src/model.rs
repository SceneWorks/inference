//! FLUX.2 provider registration + the generation path, shared across the klein and **dev** variants.
//!
//! `load()` assembles the tokenizer, text encoder, MMDiT transformer, and 32-ch VAE from a snapshot
//! directory — klein uses the Qwen3 loaders, dev (sc-2365) the Mistral3 `*_dev` loaders (which load
//! a pre-quantized Q4 snapshot packed, sc-5917); `spec.quantize` (Q4/Q8, sc-2643) then quantizes the
//! dense parts in place (a no-op for already-packed dev weights). `generate()` runs the flow-match
//! denoise loop, then BN-denormalizes + 2×2-unpatchifies + VAE-decodes. Guidance is variant-typed:
//! distilled klein runs CFG-free (1.0 = single forward; a base variant would CFG dual-forward when
//! `guidance > 1`); guidance-distilled **dev** feeds its scale as an embedded scalar into the
//! transformer's guidance embedder (single forward, default ~4.0 over ~28 steps — NOT true-CFG).
//! txt2img (`flux2_klein_9b`, `flux2_dev`) and the single-/multi-reference edit variants share this
//! path.
//!
//! Activations run f32 (matmul(f32, bf16)→f32): dodges the dense 16-bit Metal GEMM bug and is the
//! quality target. Pixel-parity with the fork's bf16 render is therefore not the gate (see the
//! e2e test) — component f32 parity + visual correctness is.

use mlx_gen::array::scalar;
use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, run_flow_sampler_with_latent_hook, CancelFlag, Error, GenerationOutput,
    GenerationRequest, Generator, LatentDecoder, LoadSpec, ModelDescriptor, OffloadPolicy,
    Precision, Progress, Residency, Result, TimestepConvention, WeightsSource,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use mlx_rs::ops::{add, concatenate_axis, multiply, pad, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use std::path::Path;

use crate::caption_upsample;
use crate::chunk::MemoryConfig;
use crate::config::{Flux2Variant, SIZE_MULTIPLE};
use crate::kv_cache::{CacheMode, Flux2KvCache};
use crate::pipeline::{
    add_noise_by_interpolation, create_noise, init_time_step, pack_latents, patchify_latents,
    prepare_grid_ids, prepare_text_ids, preprocess_ref_image, schedule_with,
};
use crate::text_encoder::Qwen3TextEncoder;
use crate::transformer::{Flux2ForwardInputs, Flux2Transformer};
use crate::vae::Flux2Vae;
use crate::vision::{Mistral3Projector, PixtralVisionTower};
use crate::{loader, Flux2Config};

/// PiD latent-space tag for the FLUX.2 family (epic 7840, sc-7847): the FLUX.2 `AutoencoderKLFlux2`
/// 32-ch / 2×2-patchified / BatchNorm latent. `flux2`, `flux2-klein-4b`, and `flux2-klein-9b` all
/// resolve to the same student + checkpoint in [`mlx_gen_pid::registry`]; Lens and Ideogram 4 reuse
/// this same space.
pub const PID_BACKBONE: &str = "flux2";

/// Joint DiT sequence length (txt + target + reference tokens) above which the gated activation
/// levers (sc-6266) engage. Sits between a single-reference 1024² edit (~8.7K tokens, fits the 96 GB
/// budget) and a 2-reference one (~12.8K tokens, ~104 GB un-bounded, sc-6124) so only the over-budget
/// multi-reference / high-resolution edits take the bounded-memory path; every shipped path (T2I,
/// single-reference edit, strict pose, LoRA) stays on the byte-identical [`MemoryConfig::OFF`].
const LONG_SEQ_TOKEN_THRESHOLD: usize = 10_000;

/// Per-reference stride on the RoPE time axis (the fork's `prepare_reference_image_conditioning`):
/// reference `i` is tagged at `t = REFERENCE_TIME_STRIDE * (i + 1)` (10, 20, 30, …) so each edit
/// reference occupies its own time band, distinct from the target's `t = 0`. The stride must exceed a
/// single reference's t-extent (1, since each ref is one packed grid at a fixed t) to avoid two refs
/// colliding on the same time index; at the [`MAX_EDIT_REFERENCES`] cap (8) the band tops out at
/// `t = 80`, well inside the RoPE t-axis range. Named so the invariant is explicit rather than a bare
/// `10 + 10*i`.
const REFERENCE_TIME_STRIDE: i32 = 10;

/// F-027: hard cap on the number of edit references a single request may supply. Each reference adds
/// ~4096 joint-DiT tokens (sc-6124 measured ~104 GB peak with 2 UNBOUNDED refs at 1024² → quadratic
/// SDPA + OOM from request input), and the RoPE time band tops out at `REFERENCE_TIME_STRIDE * 8 = 80`
/// (the [`REFERENCE_TIME_STRIDE`] invariant). This constant is the ONLY enforcement of that invariant:
/// `Capabilities::max_count` bounds `req.count` (the output batch size), NOT the reference count, so
/// the shared capability floor never caps references — previously `collect_edit_references` flattened
/// every Reference/MultiReference with no bound. Checked both in `validate_request` (up-front worker
/// rejection) and `collect_edit_references` (the generate path). Request-input OOM is the repo's
/// historical highest-severity class.
const MAX_EDIT_REFERENCES: usize = 8;

/// Sanitize model-generated text for a single-line, machine-parsed log record (the worker consumes
/// the `ENHANCED_PROMPT:` / `ENHANCER_FALLBACK:` prefix): replace every control/whitespace char (incl.
/// embedded newlines that would split the record or forge a second prefix line) with a space, collapse
/// runs, and length-cap to 512 chars. Only the logged copy is touched — never the prompt itself.
fn sanitize_log_text(s: &str) -> String {
    const CAP: usize = 512;
    let collapsed: String = s
        .chars()
        .map(|c| {
            if c.is_control() || c.is_whitespace() {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.chars().count() > CAP {
        let truncated: String = collapsed.chars().take(CAP).collect();
        format!("{truncated}…")
    } else {
        collapsed
    }
}

/// Walk the request conditioning for reference images (`Reference` + `MultiReference`), flattened in
/// conditioning order then image order (the fork's flat `image_paths`). Shared by the edit and
/// caption-upsample paths; the empty-check is the caller's (edit requires ≥1, upsample's T2I path
/// tolerates none) (F-013/L-dedup).
fn collect_reference_images(req: &GenerationRequest) -> Vec<&mlx_gen::media::Image> {
    let mut refs: Vec<&mlx_gen::media::Image> = Vec::new();
    for c in &req.conditioning {
        match c {
            mlx_gen::Conditioning::Reference { image, .. } => refs.push(image),
            mlx_gen::Conditioning::MultiReference { images } => refs.extend(images.iter()),
            _ => {}
        }
    }
    refs
}

pub fn descriptor_klein_9b() -> ModelDescriptor {
    Flux2Variant::Klein9b.descriptor()
}

pub fn descriptor_klein_9b_edit() -> ModelDescriptor {
    Flux2Variant::Klein9bEdit.descriptor()
}

pub fn descriptor_klein_9b_kv_edit() -> ModelDescriptor {
    Flux2Variant::Klein9bKvEdit.descriptor()
}

pub fn load_klein_9b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9b, spec)
}

pub fn load_klein_9b_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9bEdit, spec)
}

pub fn load_klein_9b_kv_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9bKvEdit, spec)
}

pub fn descriptor_dev() -> ModelDescriptor {
    Flux2Variant::Dev.descriptor()
}

pub fn descriptor_dev_edit() -> ModelDescriptor {
    Flux2Variant::DevEdit.descriptor()
}

/// FLUX.2-dev txt2img (sc-2365): the guidance-distilled 32B flagship. Loads the dev snapshot
/// (Mistral3 TE + dev DiT, pre-quantized Q4 per sc-5917) and runs the embedded-guidance denoise
/// (single forward, default guidance ~4.0 over ~28 steps — NOT true-CFG).
pub fn load_dev(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Dev, spec)
}

/// FLUX.2-dev image-conditioned edit (sc-5919): single + multi reference. Loads the same dev
/// snapshot as [`load_dev`] and runs the shared edit conditioning path — reference images are
/// VAE-encoded, packed, and concatenated to the DiT image stream (the klein edit mechanism, faithful
/// to the diffusers `Flux2Pipeline`; the prompt embeds stay text-only). Embedded-guidance denoise.
pub fn load_dev_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::DevEdit, spec)
}

fn load_variant(variant: Flux2Variant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // Precision + snapshot-dir guard up front for BOTH policies (fail fast).
    let root = resolve_root(variant, spec)?;
    // Publish the tier the transformer actually realizes. A packed Dev turnkey selected without a
    // request is still Q4/Q8; a dense turnkey with a request is folded at load time.
    let memory_numeric_tier = effective_memory_numeric_tier(variant, spec, root, variant.id())?;
    // F-181: a `Sequential` + `spec.quantize` load over a dense snapshot re-quantizes the whole model
    // on every generate; `Resident` quantizes once. Warn for that combination only.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
            mlx_gen::residency::warn_sequential_requantize(variant.id(), q.bits());
        }
    }
    // The dev checkpoint has a different tokenizer (Mistral3, not Qwen3) than klein.
    let text_encoder_source = variant.encoder_contract().source_for_load(spec, root)?;
    let tokenizer = if variant.is_dev() {
        loader::load_validated_tokenizer_dev(&text_encoder_source)?
    } else {
        loader::load_validated_tokenizer(&text_encoder_source)?
    };
    Ok(Box::new(Flux2 {
        descriptor: variant.descriptor(),
        variant,
        config: variant.config(),
        memory_strategy: crate::memory_strategy::contract_for_variant(variant, spec)?,
        memory_numeric_tier: Some(memory_numeric_tier),
        loaded_spec: spec.clone(),
        tokenizer: Some(tokenizer),
        residency: build_residency_with_source(variant, spec, text_encoder_source)?,
    }))
}

/// Precision guard (only the default precision is wired) + snapshot-dir resolution (rejecting a
/// single-file source), shared by [`load_variant`] and [`build_residency`]'s per-phase loaders.
fn resolve_root(variant: Flux2Variant, spec: &LoadSpec) -> Result<&Path> {
    if spec.precision != Precision::Bf16 {
        // The dense path loads at the on-disk dtype and runs f32 activations; an explicit fp32
        // precision override isn't a separate wired mode. Q4/Q8 (sc-2643) go through `spec.quantize`.
        return Err(Error::Msg(format!(
            "{}: only the default precision is wired; drop the precision override (Q4/Q8 = spec.quantize)",
            variant.id()
        )));
    }
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(format!(
            "{} expects a FLUX.2 snapshot directory (tokenizer/ text_encoder/ \
             transformer/ vae/), not a single .safetensors file",
            variant.id()
        ))),
    }
}

/// The phase-A text components dropped first under `Sequential`: the text encoder (klein Qwen3 / dev
/// Mistral-3) plus — dev-only — the Pixtral vision tower + Mistral3 projector that caption upsampling
/// (sc-6030) runs in the SAME phase (it uses the text encoder's LM head + the vision tower). All parsed
/// from the shared `text_encoder/` shard set; klein has no vision tower / projector (`None`).
pub(crate) struct Flux2TextOwned {
    pub(crate) text_encoder: Qwen3TextEncoder,
    /// FLUX.2-dev caption upsampling (sc-6030): the Pixtral vision tower + Mistral3 projector that
    /// encode reference images for the image-conditioned prompt rewrite. `None` for klein — caption
    /// upsampling is dev-only and gated on `enhance_prompt` — and for the control variant (it does not
    /// caption-upsample).
    pub(crate) vision_tower: Option<PixtralVisionTower>,
    pub(crate) projector: Option<Mistral3Projector>,
}

/// The heavy render-phase components (the MMDiT transformer, the VAE, and the optional PiD decoder) —
/// everything but the text encoder / vision tower. Owned by the `Resident` components or by a
/// `Sequential` generate.
pub(crate) struct Flux2HeavyOwned {
    transformer: Flux2Transformer,
    vae: Flux2Vae,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7847): loaded when the request
    /// carries `LoadSpec::pid` AND this generate uses it (`req.use_pid`). `Some` → decode the packed
    /// BN-normalized latent through the `flux2` PiD student (4× SR) instead of the VAE.
    pid: Option<PidEngine>,
}

/// Load the text encoder (+ dev vision tower + projector) and quantize the encoder — the phase-A
/// components dropped first under `Sequential`. The dev checkpoint parses the ~45 GB `text_encoder/`
/// shard set ONCE (F-112) into the Mistral tower + Pixtral vision tower + projector; klein has neither.
/// Q4/Q8 quantizes the **text encoder** here (the transformer + VAE quant lives in [`load_flux2_heavy`]);
/// the vision tower + projector stay full precision, matching the pre-seam `load` (sc-2604). Factored so
/// the `Resident` and `Sequential` paths build byte-identical encoders.
fn load_flux2_text(
    variant: Flux2Variant,
    text_encoder_source: &mlx_gen::gen_core::ValidatedEncoderSource,
    multimodal_encoder_source: &mlx_gen::gen_core::ValidatedEncoderSource,
    text_encoder_load_time_quant_bits: Option<i32>,
) -> Result<Flux2TextOwned> {
    text_encoder_source.read_unchanged(|source| {
        let (mut text_encoder, vision_tower, projector) = if variant.is_dev() {
            let (encoder, vision_tower, projector) =
                multimodal_encoder_source.read_unchanged(|multimodal| {
                    loader::load_dev_text_encoder_group_from_sources(source, multimodal)
                })?;
            (encoder, Some(vision_tower), Some(projector))
        } else {
            (loader::load_text_encoder_from_source(source)?, None, None)
        };
        if let Some(bits) = text_encoder_load_time_quant_bits {
            text_encoder.quantize(bits)?;
        }
        Ok(Flux2TextOwned {
            text_encoder,
            vision_tower,
            projector,
        })
    })
}

/// Load the heavy render-phase components — the MMDiT transformer (+ Q4/Q8 + LoRA/LoKr residuals), the
/// VAE (+ Q4/Q8), and the optional PiD overlay — everything but the text encoder. Factored so the
/// `Sequential` path loads these AFTER the encoder is dropped (bounding peak to `max(TE, DiT+VAE)`).
/// Q4/Q8 quantizes the transformer + VAE here — the fork's whole-model `nn.quantize` (group_size 64):
/// the text encoder is quantized in [`load_flux2_text`], and the VAE's quantized surface is just its two
/// mid-block attentions. Quantize-then-adapters order matches the pre-seam `load`; the components are
/// independent of the text encoder (separate weight files, deterministic RNG-free quant), so the
/// `Resident` composition is byte-identical.
fn load_flux2_heavy(
    variant: Flux2Variant,
    spec: &LoadSpec,
    load_pid: bool,
) -> Result<Flux2HeavyOwned> {
    let root = resolve_root(variant, spec)?;
    let mut transformer = if variant.is_dev() {
        loader::load_transformer_dev(root)?
    } else {
        loader::load_transformer(root)?
    };
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // LoRA/LoKr (sc-2646): applied AFTER quantization, as forward-time residuals over the
    // (possibly quantized) transformer — fork-faithful, transformer-only. No-op when empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_flux2_adapters(&mut transformer, &spec.adapters)?;
    }
    if matches!(variant, Flux2Variant::Klein9b | Flux2Variant::Klein9bEdit)
        && spec.load_shape == mlx_gen::LoadShape::DeferredMaterialization
        && spec.offload_policy == OffloadPolicy::Sequential
        && spec.quantize.is_none()
        && spec.adapters.is_empty()
    {
        let inventory = crate::artifact_inventory::KleinArtifactInventory::verify(spec)?
            .ok_or_else(|| Error::Unsupported(
                "flux2 Klein deferred materialization requires the exact calibrated BF16 HF artifact"
                    .to_owned(),
            ))?;
        let quant = crate::loader::read_component_quant(&root.join("transformer"))?;
        transformer = transformer.with_block_stream(inventory, variant.config(), quant);
        transformer.finalize_block_stream()?;
    }
    // PiD decoder overlay (epic 7840, sc-7847): load the `flux2` student + Gemma caption encoder once
    // when the spec carries it AND this generate uses it (`load_pid`, F-177). The student is shared
    // across the whole FLUX.2 family (klein + dev).
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    Ok(Flux2HeavyOwned {
        transformer,
        vae,
        pid,
    })
}

/// The policy→[`Residency`] dispatch every FLUX.2 variant shares (sc-10840), routed through the single
/// [`Residency::from_policy`] seam. `Resident` eager-loads the text encoder + heavy bundle now (any PiD
/// overlay loaded once, reused); `Sequential` captures the two per-phase loaders and loads nothing now,
/// deferring each to [`Residency::run`]. Both use the same [`load_flux2_text`] / [`load_flux2_heavy`],
/// so the `Resident` composition is byte-identical to the pre-seam one.
#[cfg(test)]
fn build_residency(
    variant: Flux2Variant,
    spec: &LoadSpec,
) -> Result<Residency<Flux2TextOwned, Flux2HeavyOwned>> {
    let root = resolve_root(variant, spec)?;
    let text_encoder_source = variant.encoder_contract().source_for_load(spec, root)?;
    build_residency_with_source(variant, spec, text_encoder_source)
}

fn build_residency_with_source(
    variant: Flux2Variant,
    spec: &LoadSpec,
    text_encoder_source: mlx_gen::gen_core::ValidatedEncoderSource,
) -> Result<Residency<Flux2TextOwned, Flux2HeavyOwned>> {
    let root = resolve_root(variant, spec)?;
    // Klein artifacts intentionally keep Qwen3 dense in every Q4/Q8 tier. Dev applies the
    // effective transformer tier to its language tower, including an already-packed snapshot.
    let effective_quant_bits = variant
        .is_dev()
        .then(|| effective_base_quant(spec, root, variant.id()))
        .transpose()?
        .flatten()
        .map(mlx_gen::gen_core::Quant::bits);
    let text_encoder_load_time_quant_bits =
        text_encoder_source.load_time_quant_bits(effective_quant_bits, variant.id())?;
    let multimodal_encoder_source = if variant.is_dev() {
        let source = variant
            .encoder_contract()
            .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
        source.validate_vision(
            &crate::config::DEV_VISION_ENCODER_CONTRACT,
            &crate::config::DEV_ENCODER_CONTRACT,
        )?;
        source
    } else {
        text_encoder_source.clone()
    };
    build_residency_from_admitted_sources(
        variant,
        spec,
        text_encoder_source,
        multimodal_encoder_source,
        text_encoder_load_time_quant_bits,
    )
}

fn build_residency_from_admitted_sources(
    variant: Flux2Variant,
    spec: &LoadSpec,
    text_encoder_source: mlx_gen::gen_core::ValidatedEncoderSource,
    multimodal_encoder_source: mlx_gen::gen_core::ValidatedEncoderSource,
    text_encoder_load_time_quant_bits: Option<i32>,
) -> Result<Residency<Flux2TextOwned, Flux2HeavyOwned>> {
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || {
            load_flux2_text(
                variant,
                &text_encoder_source,
                &multimodal_encoder_source,
                text_encoder_load_time_quant_bits,
            )
        },
        move |use_pid| load_flux2_heavy(variant, &spec_heavy, use_pid),
    )
}

pub(crate) fn effective_base_quant(
    spec: &LoadSpec,
    root: &Path,
    provider_id: &str,
) -> Result<Option<mlx_gen::gen_core::Quant>> {
    if let Some(requested) = spec.quantize {
        mlx_gen::quant::needs_load_time_quant(root, "transformer", requested.bits(), provider_id)?;
    }
    match mlx_gen::quant::packed_quant_bits(root, "transformer")? {
        Some(4) => Ok(Some(mlx_gen::gen_core::Quant::Q4)),
        Some(8) => Ok(Some(mlx_gen::gen_core::Quant::Q8)),
        Some(bits) => Err(Error::Unsupported(format!(
            "{provider_id}: transformer declares unsupported packed quantization width {bits}"
        ))),
        None => Ok(spec.quantize),
    }
}

/// Resolve the exact numeric tier a loaded FLUX.2 generator publishes for memory admission.
pub(crate) fn effective_memory_numeric_tier(
    variant: Flux2Variant,
    spec: &LoadSpec,
    root: &Path,
    provider_id: &str,
) -> Result<mlx_gen::gen_core::MemoryNumericTier> {
    let quant = if variant.is_dev() {
        effective_base_quant(spec, root, provider_id)?
    } else {
        spec.quantize
    };
    Ok(mlx_gen::gen_core::MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    })
}

/// Registry-side counterpart of [`effective_memory_numeric_tier`] for Dev-family routes.
pub(crate) fn effective_dev_memory_numeric_tier(
    spec: &LoadSpec,
    provider_id: &str,
) -> Result<mlx_gen::gen_core::MemoryNumericTier> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root.as_path(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{provider_id} expects a FLUX.2 snapshot directory, not a single .safetensors file"
            )))
        }
    };
    effective_memory_numeric_tier(Flux2Variant::Dev, spec, root, provider_id)
}

/// The FLUX.2 generator (klein + dev, ±edit/kv-edit).
pub struct Flux2 {
    descriptor: ModelDescriptor,
    variant: Flux2Variant,
    config: Flux2Config,
    memory_strategy: Option<mlx_gen::gen_core::MemoryProviderContract>,
    /// Exact load-time tier used by provider-owned route validation. Test-only instances have no
    /// load artifact, so their tier remains unknown.
    memory_numeric_tier: Option<mlx_gen::gen_core::MemoryNumericTier>,
    loaded_spec: LoadSpec,
    /// The (small, always-warm) tokenizer. `None` only for the weightless `new_for_tests` instances;
    /// the production load path always populates it.
    tokenizer: Option<TextTokenizer>,
    /// Component-residency strategy (sc-10840), selected from [`LoadSpec::offload_policy`]. `Resident`
    /// (default) holds the text encoder (+ dev vision tower/projector) + DiT + VAE warm; `Sequential`
    /// holds only the per-phase loader closures and re-loads per generation in phase order (encode →
    /// **drop the text encoder** → denoise/decode). Weightless test instances hold loader closures that
    /// error if invoked (the validation tests never `run` the residency). The [`Residency`] seam owns
    /// the eval/drop/clear discipline, the stage-boundary cancel checks, and the error-safe cache flush.
    residency: Residency<Flux2TextOwned, Flux2HeavyOwned>,
}

impl Flux2 {
    /// Construct a weightless instance for validation tests (no tokenizer, loader closures that error).
    pub fn new_for_tests(variant: Flux2Variant) -> Self {
        let loaded_spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        Self {
            descriptor: variant.descriptor(),
            variant,
            config: variant.config(),
            memory_strategy: match variant {
                Flux2Variant::Dev => {
                    Some(crate::memory_strategy::registered_dev_t2i_contract(&loaded_spec).unwrap())
                }
                Flux2Variant::DevEdit => {
                    Some(crate::memory_strategy::registered_dev_contract(&loaded_spec).unwrap())
                }
                _ => None,
            },
            memory_numeric_tier: None,
            loaded_spec,
            tokenizer: None,
            residency: Residency::sequential(
                || {
                    Err(Error::Msg(
                        "flux2: text encoder not loadable in a test-only instance".into(),
                    ))
                },
                |_use_pid| {
                    Err(Error::Msg(
                        "flux2: heavy bundle not loadable in a test-only instance".into(),
                    ))
                },
            ),
        }
    }

    /// Encode a prompt → `(prompt_embeds [1,512,joint], text_ids [1,512,4])`.
    fn encode(
        &self,
        tokenizer: &TextTokenizer,
        te: &Qwen3TextEncoder,
        prompt: &str,
    ) -> Result<(Array, Array)> {
        let tok = tokenizer.tokenize(prompt)?;
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tok);
        let embeds = te.prompt_embeds(&input_ids, &attention_mask)?;
        let ids = prepare_text_ids(embeds.shape()[1] as usize);
        Ok((embeds, ids))
    }

    /// Edit reference conditioning for **N** images (the fork's `prepare_reference_image_conditioning`):
    /// each image → resize → VAE-encode → crop-to-even → 2×2 patchify → BN-normalize → pack, tagged
    /// with grid ids at `t = 10 + 10·i` (the per-reference time offset), then all refs concatenated
    /// on the sequence axis. Returns `(image_latents [1, Σseq_ref, 128], image_latent_ids
    /// [1, Σseq_ref, 4])`. A single reference (N = 1) reduces to the original `t = 10` path. The
    /// FLUX.2 text encoder is a dense Qwen3 LLM with no vision input, so the prompt embeds are
    /// independent of the references — multi-image conditioning flows ONLY through these tokens.
    fn encode_references(
        &self,
        vae: &Flux2Vae,
        images: &[&mlx_gen::media::Image],
        width: u32,
        height: u32,
        cancel: &CancelFlag,
    ) -> Result<(Array, Array)> {
        let mut packed: Vec<Array> = Vec::with_capacity(images.len());
        let mut ids: Vec<Array> = Vec::with_capacity(images.len());
        for (i, image) in images.iter().enumerate() {
            // F-037: honor a cancel between the (up to 8) per-reference VAE encodes. `eval` on the
            // just-packed latent forces this reference's encode so the next iteration's check observes
            // real progress (no lazy-eval false green).
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let pre = preprocess_ref_image(image, width, height)?; // NHWC [1,H,W,3]
            let enc = vae.encode_mean(&pre)?; // NHWC [1,H/8,W/8,32]
            let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for the pipeline helpers
            let enc = crop_to_even(&enc)?;
            let patchified = patchify_latents(&enc)?; // [1,128,h,w]
            let normed = vae.bn_normalize_nchw(&patchified)?;
            let sh = patchified.shape();
            let packed_ref = pack_latents(&normed)?; // [1, seq_ref, 128]
            eval([&packed_ref])?;
            packed.push(packed_ref);
            ids.push(prepare_grid_ids(
                sh[2] as usize,
                sh[3] as usize,
                REFERENCE_TIME_STRIDE * (i as i32 + 1),
            ));
        }
        let packed_refs: Vec<&Array> = packed.iter().collect();
        let id_refs: Vec<&Array> = ids.iter().collect();
        Ok((
            concatenate_axis(&packed_refs, 1)?,
            concatenate_axis(&id_refs, 1)?,
        ))
    }

    /// Collect the ordered edit reference images from the request: a single `Reference`, a
    /// `MultiReference { images }` (N images, sc-2645), or several `Reference`s — flattened in
    /// conditioning order then image order (the fork passes a flat `image_paths` list). At least
    /// one reference is required (the empty-check is the edit caller's; the upsample T2I path uses the
    /// shared [`collect_reference_images`] walk directly and tolerates none).
    fn collect_edit_references<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Vec<&'a mlx_gen::media::Image>> {
        let refs = collect_reference_images(req);
        if refs.is_empty() {
            return Err(Error::Msg(format!(
                "{}: edit requires at least one reference image",
                self.descriptor.id
            )));
        }
        // F-027: cap the reference count — each ref adds ~4096 joint-DiT tokens (quadratic SDPA + a
        // request-input OOM class); see [`MAX_EDIT_REFERENCES`]. Also enforced in `validate_request`
        // so `validate()` rejects up front; this generate-path check covers direct callers.
        if refs.len() > MAX_EDIT_REFERENCES {
            return Err(Error::Msg(format!(
                "{}: edit supports at most {MAX_EDIT_REFERENCES} reference images (got {}); \
                 each adds ~4096 joint-DiT tokens",
                self.descriptor.id,
                refs.len()
            )));
        }
        Ok(refs)
    }

    /// FLUX.2-dev caption upsampling (sc-6030): rewrite the prompt with the Mistral3 multimodal LLM
    /// before encoding (the diffusers `upsample_prompt`), gated on `req.enhance_prompt` — the
    /// LTX-2.3 prompt-enhancement contract field (sc-2845), reused here for the image-aware analog.
    /// Returns the rewritten prompt, or the original `req.prompt` when the gate is off, the variant
    /// isn't dev, or on **any** upsampler failure / empty output (reference-faithful fallback, like
    /// `generate_av.py`'s try/except). Logs the LTX `ENHANCED_PROMPT:` / `ENHANCER_FALLBACK:` tokens.
    ///
    /// Runs in the residency seam's phase-A (with the text encoder + vision tower), so it takes them
    /// as arguments rather than reaching a resident field (sc-10840).
    fn maybe_upsample(
        &self,
        tokenizer: &TextTokenizer,
        text: &Flux2TextOwned,
        req: &GenerationRequest,
    ) -> String {
        if !req.enhance_prompt || !self.variant.is_dev() {
            return req.prompt.clone();
        }
        match self.run_upsample(tokenizer, text, req) {
            Ok(p) if !p.trim().is_empty() => {
                // The log record is machine-parsed on the `ENHANCED_PROMPT:` prefix; sanitize the
                // model-generated text so an embedded newline can't split the record or forge a
                // second prefix line (the returned `p` itself is unchanged) (L-log-injection).
                eprintln!("ENHANCED_PROMPT:{}", sanitize_log_text(&p));
                p
            }
            Ok(_) => {
                eprintln!("ENHANCER_FALLBACK:EmptyOutput:caption upsampler returned empty output");
                req.prompt.clone()
            }
            Err(e) => {
                eprintln!("ENHANCER_FALLBACK:{}", sanitize_log_text(&e.to_string()));
                req.prompt.clone()
            }
        }
    }

    /// Run the dev caption upsampler: the Mistral3 multimodal `generate()` over the prompt plus any
    /// reference images (through the Pixtral tower). Errors surface to
    /// [`maybe_upsample`](Self::maybe_upsample)'s fallback.
    fn run_upsample(
        &self,
        tokenizer: &TextTokenizer,
        text: &Flux2TextOwned,
        req: &GenerationRequest,
    ) -> Result<String> {
        let id = self.descriptor.id;
        let not_loaded = |what: &str| Error::Msg(format!("{id}: {what} is not loaded"));
        let vision = text
            .vision_tower
            .as_ref()
            .ok_or_else(|| not_loaded("vision tower"))?;
        let projector = text
            .projector
            .as_ref()
            .ok_or_else(|| not_loaded("projector"))?;
        let refs = collect_reference_images(req);
        let temperature = req
            .enhance_temperature
            .unwrap_or(caption_upsample::DEFAULT_TEMPERATURE);
        // Clamp the requested decode length to a hard ceiling (F-012): each step is a full ~32B forward
        // over a growing KV cache, so an unclamped `enhance_max_tokens` is an effectively unbounded job.
        let max_new_tokens = caption_upsample::clamp_max_new_tokens(req.enhance_max_tokens);
        let seed = req.seed.unwrap_or_else(default_seed);
        caption_upsample::upsample_prompt(
            tokenizer,
            &text.text_encoder,
            vision,
            projector,
            &req.prompt,
            &refs,
            temperature,
            max_new_tokens,
            seed,
            &req.cancel,
        )
    }

    /// Extract the single img2img init image + its strength from the txt2img request. The
    /// per-reference strength wins over `req.strength`. txt2img img2img conditions on exactly one
    /// init image, so more than one `Reference` is an error (multi-reference is the edit variant +
    /// `MultiReference`, sc-2645). Returns `None` for pure txt2img.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a mlx_gen::media::Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let mlx_gen::Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(format!(
                        "{}: multiple reference images are not supported (single img2img init only)",
                        self.descriptor.id
                    )));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }

    /// img2img init conditioning: resize → VAE-encode → NCHW → crop-to-even → center-crop/pad to the
    /// target latent grid → 2×2 patchify → BN-normalize → pack. Returns the **clean** packed latents
    /// `[1, lat_h·lat_w, 128]` (seed-independent — blended with the per-seed noise in `generate`).
    /// Mirrors the fork's `_prepare_img2img_latents` (minus the noise blend); same encode chain as
    /// `encode_reference`, plus the `_match_latent_spatial_size` step and the txt2img grid ids.
    fn encode_init_latents(
        &self,
        vae: &Flux2Vae,
        image: &mlx_gen::media::Image,
        width: u32,
        height: u32,
    ) -> Result<Array> {
        let pre = preprocess_ref_image(image, width, height)?; // NHWC [1,H,W,3]
        let enc = vae.encode_mean(&pre)?; // NHWC [1,H/8,W/8,32]
        let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for the pipeline helpers
        let enc = crop_to_even(&enc)?;
        // Target the denoise latent grid: `latent_h·2 × latent_w·2 = H/8 × W/8`. A no-op at the
        // standard multiple-of-16 sizes (encoded H/8 already equals the target).
        let enc = match_latent_spatial_size(&enc, (height / 8) as i32, (width / 8) as i32)?;
        let patchified = patchify_latents(&enc)?; // [1,128,h,w]
        let normed = vae.bn_normalize_nchw(&patchified)?;
        pack_latents(&normed) // [1, lat_h·lat_w, 128]
    }
}

/// Crop a NCHW latent's spatial dims down to even (the fork's `crop_to_even_spatial`), so the 2×2
/// patchify divides cleanly. A no-op at the standard multiple-of-16 sizes.
pub(crate) fn crop_to_even(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let mut x = x.clone();
    if sh[2] % 2 != 0 {
        let idx = Array::from_slice(&(0..sh[2] - 1).collect::<Vec<i32>>(), &[sh[2] - 1]);
        x = x.take_axis(&idx, 2)?;
    }
    if sh[3] % 2 != 0 {
        let idx = Array::from_slice(&(0..sh[3] - 1).collect::<Vec<i32>>(), &[sh[3] - 1]);
        x = x.take_axis(&idx, 3)?;
    }
    Ok(x)
}

/// Center-crop or symmetric-pad a NCHW latent's spatial dims to `(target_h, target_w)` — the fork's
/// `_match_latent_spatial_size`. A no-op at the standard multiple-of-16 sizes (the VAE-encoded H/8
/// already equals the `latent_h·2` target); guards odd / mismatched user images.
pub(crate) fn match_latent_spatial_size(x: &Array, target_h: i32, target_w: i32) -> Result<Array> {
    let mut x = x.clone();
    let (h, w) = (x.shape()[2], x.shape()[3]);
    if h != target_h {
        if h > target_h {
            let off = (h - target_h) / 2;
            let idx = Array::from_slice(&(off..off + target_h).collect::<Vec<i32>>(), &[target_h]);
            x = x.take_axis(&idx, 2)?;
        } else {
            let before = (target_h - h) / 2;
            let after = (target_h - h) - before;
            x = pad(
                &x,
                &[(0, 0), (0, 0), (before, after), (0, 0)][..],
                None,
                None,
            )?;
        }
    }
    if w != target_w {
        if w > target_w {
            let off = (w - target_w) / 2;
            let idx = Array::from_slice(&(off..off + target_w).collect::<Vec<i32>>(), &[target_w]);
            x = x.take_axis(&idx, 3)?;
        } else {
            let before = (target_w - w) / 2;
            let after = (target_w - w) - before;
            x = pad(
                &x,
                &[(0, 0), (0, 0), (0, 0), (before, after)][..],
                None,
                None,
            )?;
        }
    }
    Ok(x)
}

impl Generator for Flux2 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_request(
            &self.descriptor,
            self.variant.is_edit(),
            self.variant.is_kv(),
            req,
        )
        .map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        self.memory_strategy.as_ref().map_or_else(
            || mlx_gen::gen_core::MemorySafetyDecision::Reject {
                reason: format!("{} has no memory-strategy contract", self.descriptor.id),
            },
            |contract| {
                let Some(expected_tier) = self.memory_numeric_tier else {
                    return mlx_gen::gen_core::MemorySafetyDecision::Reject {
                        reason: format!(
                            "{} has no loaded numeric tier for memory admission",
                            self.descriptor.id
                        ),
                    };
                };
                match self.variant {
                    Flux2Variant::Dev => crate::memory_strategy::dev_t2i_safety_check(
                        contract,
                        context,
                        expected_tier,
                    ),
                    Flux2Variant::DevEdit => {
                        crate::memory_strategy::safety_check(contract, context, expected_tier)
                    }
                    Flux2Variant::Klein9b | Flux2Variant::Klein9bEdit => {
                        crate::memory_strategy::klein_safety_check(
                            &self.loaded_spec,
                            contract,
                            context,
                        )
                    }
                    _ => mlx_gen::gen_core::MemorySafetyDecision::Reject {
                        reason: format!(
                            "{} has no memory-safety implementation for its registered contract",
                            self.descriptor.id
                        ),
                    },
                }
            },
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return Ok(None);
        };
        if matches!(self.variant, Flux2Variant::Dev | Flux2Variant::DevEdit) {
            return Ok(None);
        }
        crate::memory_strategy::begin_klein_request(&self.loaded_spec, contract, context)
    }
}

/// Resolve the classifier-free negative branch for a request.
///
/// All Klein variants share this path (txt2img, edit, and KV edit): an explicit guidance scale above
/// one enables the second forward and must encode the caller's negative prompt verbatim. An unset
/// prompt preserves the historical single-space unconditional condition. Dev uses embedded guidance
/// and therefore never creates this branch.
fn cfg_negative_prompt(
    variant: Flux2Variant,
    guidance: f32,
    req: &GenerationRequest,
) -> Option<&str> {
    if !variant.uses_embedded_guidance() && guidance > 1.0 {
        Some(req.negative_prompt.as_deref().unwrap_or(" "))
    } else {
        None
    }
}

/// Apply [`cfg_negative_prompt`] to the text-encoding seam. This stays generic so the request-to-text
/// handoff (including exact empty/unset semantics) is testable without loading a tokenizer or weights.
fn encode_cfg_negative_with<T, E>(
    variant: Flux2Variant,
    guidance: f32,
    req: &GenerationRequest,
    mut encode: impl FnMut(&str) -> std::result::Result<T, E>,
) -> std::result::Result<Option<T>, E> {
    cfg_negative_prompt(variant, guidance, req)
        .map(&mut encode)
        .transpose()
}

impl Flux2 {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::Msg(format!("{}: model is not loaded", self.descriptor.id)))?;
        // F-037: bail before the pre-denoise conditioning stage (up to 8 reference VAE encodes at
        // 2048², the TE encode, and the img2img init encode — all ahead of the first denoise step).
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or(self.variant.default_steps()) as usize;
        let guidance = req.guidance.unwrap_or(self.variant.default_guidance());
        // dev is guidance-DISTILLED: the scale is an embedded scalar fed into the transformer's
        // guidance embedder (single forward), NOT a true-CFG dual-forward over a negative prompt.
        let embedded_guidance = self.variant.uses_embedded_guidance().then_some(guidance);

        // Staged residency lifecycle (sc-10840): under `Sequential` the seam loads the text encoder
        // (+ dev vision tower/projector), runs any caption upsample + the prompt encode, materializes,
        // then DROPS them + `clear_cache()` before the DiT/VAE load below — the peak-bounding win. Under
        // `Resident` it borrows the warm encoder and runs the identical encode/denoise/decode with no
        // eval/clear. The edit reference conditioning that must PERSIST through denoise is VAE-encoded in
        // the heavy phase (after the TE drop), byte-identical to the resident order (a deterministic,
        // TE-independent VAE encode — same hoist argument as the img2img init latents).
        self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            // ── Phase A: (dev) caption upsample → prompt encode (+ base true-CFG negative).
            |text: &Flux2TextOwned| {
                // FLUX.2-dev caption upsampling (sc-6030): optionally rewrite the prompt with the
                // Mistral3 multimodal LLM (using any reference images) before encoding, gated on
                // `enhance_prompt`. A no-op (returns `req.prompt`) for klein, gate off, or any failure.
                let prompt = self.maybe_upsample(tokenizer, text, req);
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let (prompt_embeds, text_ids) =
                    self.encode(tokenizer, &text.text_encoder, &prompt)?;
                // Classifier-free dual-forward only for the non-embedded-guidance path at guidance >1;
                // dev routes its scale through the embedded guidance embedder instead, so it never takes
                // a negative pass, and distilled klein runs at guidance 1.0 (also no negative).
                let negative = encode_cfg_negative_with(self.variant, guidance, req, |negative| {
                    self.encode(tokenizer, &text.text_encoder, negative)
                })?;
                Ok((prompt_embeds, text_ids, negative))
            },
            // Materialize the (TE-dependent) embeds while the encoder is still alive (Sequential only) —
            // MLX is lazy, so an un-evaluated embed keeps the encoder referenced and the drop would free
            // nothing. `text_ids` are host-derived position ids (TE-independent), so evaling the embeds
            // is sufficient.
            |encoded| {
                let Some((prompt_embeds, _text_ids, negative)) = encoded else {
                    return Ok(());
                };
                match negative {
                    Some((neg_embeds, _)) => eval([prompt_embeds, neg_embeds])?,
                    None => eval([prompt_embeds])?,
                }
                Ok(())
            },
            // ── Phase B: reference/img2img conditioning (VAE) + denoise + decode from the heavy bundle.
            |heavy, (prompt_embeds, text_ids, negative), on_progress| {
                let transformer = &heavy.transformer;
                let vae = &heavy.vae;

                // Edit: build the reference-image conditioning from one `Reference` or one
                // `MultiReference` (sc-2645). The transformer sees the joint sequence
                // `[txt, target, ref0, ref1, …]`; its output keeps the leading `target_seq` image tokens.
                let reference = if self.variant.is_edit() {
                    let images = self.collect_edit_references(req)?;
                    Some(self.encode_references(
                        vae,
                        &images,
                        req.width,
                        req.height,
                        &req.cancel,
                    )?)
                } else {
                    None
                };

                // img2img (txt2img variant): a single `Reference` init image seeds the latents via the
                // noise blend at `sigmas[start_step]`, with the denoise loop starting at `start_step`
                // (= the fork's `_prepare_img2img_latents` + `Config.init_time_step`). The edit variant
                // consumes its `Reference` above (token concat), so img2img is txt2img-only.
                let img2img = if self.variant.is_edit() {
                    None
                } else {
                    self.resolve_reference(req)?
                };
                let start_step = match &img2img {
                    Some((_, strength)) => init_time_step(steps, *strength),
                    None => 0,
                };

                let sched = schedule_with(steps, req.width, req.height, req.scheduler.as_deref())?;
                let lat_h = (req.height / 16) as usize;
                let lat_w = (req.width / 16) as usize;
                let latent_ids = prepare_grid_ids(lat_h, lat_w, 0);
                let in_channels = self.config.in_channels as i32;

                // The img2img clean init latents are seed-independent — encode once, blend with per-seed
                // noise below. `None` for pure txt2img (or strength ≤ 0, where `start_step == 0`).
                let clean_init = match &img2img {
                    Some((image, _)) if start_step > 0 => {
                        Some(self.encode_init_latents(vae, image, req.width, req.height)?)
                    }
                    _ => None,
                };

                // sc-6266: a multi-reference edit concatenates each reference's latent tokens onto the
                // joint `[txt, target, ref…]` DiT sequence, making the denoise activation-bound — a
                // 2-reference 1024² edit peaks ~104 GB, over the 96 GB budget (sc-6124). Above the
                // single-reference ceiling, bound the per-step activation high-water with
                // `eval_per_block` (bit-exact, so the edit's pixels are unchanged). Shorter sequences
                // (T2I, single-reference edit, pose) stay on `MemoryConfig::OFF` → the shipped forward is
                // byte-identical. Env-overridable (`MemoryConfig::from_env`) so a deployment can tune
                // chunking without a recompile.
                let total_seq = prompt_embeds.shape()[1] as usize
                    + lat_h * lat_w
                    + reference
                        .as_ref()
                        .map(|(r, _)| r.shape()[1] as usize)
                        .unwrap_or(0);
                let mem = MemoryConfig::from_env(if total_seq > LONG_SEQ_TOKEN_THRESHOLD {
                    MemoryConfig::LONG_SEQ
                } else {
                    MemoryConfig::OFF
                });

                // For an edit, the transformer's image input/ids are `[target, ref]` (or `[target]` only
                // on a cached KV step); its output keeps the image stream, of which we take the leading
                // `target_seq` tokens. txt2img has no ref, so the concat + slice are no-ops.
                // `include_ref=false` drops the reference tokens (the 9b-kv cached step); `cache` threads
                // the per-seed KV cache through the transformer.
                let run = |latents: &Array,
                           embeds: &Array,
                           ids: &Array,
                           ts: f32,
                           include_ref: bool,
                           cache: Option<&Flux2KvCache>|
                 -> Result<Array> {
                    let target_seq = latents.shape()[1];
                    let (hidden, img_ids) = match (&reference, include_ref) {
                        (Some((ref_lat, ref_ids)), true) => (
                            concatenate_axis(&[latents, ref_lat], 1)?,
                            concatenate_axis(&[&latent_ids, ref_ids], 1)?,
                        ),
                        _ => (latents.clone(), latent_ids.clone()),
                    };
                    let out = transformer.forward_with_mem(
                        &Flux2ForwardInputs {
                            hidden_states: &hidden,
                            encoder_hidden_states: embeds,
                            img_ids: &img_ids,
                            txt_ids: ids,
                            timestep: ts,
                            guidance: embedded_guidance,
                        },
                        cache,
                        &mem,
                        if self.variant.is_dev() {
                            mlx_gen::attention::AttentionPlan::UNBOUNDED
                        } else {
                            crate::memory_strategy::attention_plan(req)
                        },
                        if self.variant.is_dev() {
                            None
                        } else {
                            crate::memory_strategy::transformer_window(req)?
                                .map(|size| (size, &req.cancel))
                        },
                    )?;
                    let idx =
                        Array::from_slice(&(0..target_seq).collect::<Vec<i32>>(), &[target_seq]);
                    Ok(out.take_axis(&idx, 1)?)
                };

                // 9b-kv edit: cache reference K/V on step 0, reuse on steps 1+ (the ~2.4× speedup). The
                // edit path always has a reference, so `num_ref > 0`.
                let kv_enabled = self.variant.is_kv() && reference.is_some();
                let num_ref = reference
                    .as_ref()
                    .map(|(r, _)| r.shape()[1] as usize)
                    .unwrap_or(0);

                // sc-2963 (rollout of sc-2957): run the MMDiT's fusable elementwise glue (adaLN affine,
                // SwiGLU, gated residual, RoPE rotation) through `mx.compile`. Under MLX 0.32 bf16 is
                // exact to eager and f32 stays within the established ULP contract; compile_parity.rs
                // gates the composed forward. Scoped to this render by the RAII guard (F-007): the
                // render thread's prior setting is restored on drop.
                let _compile_glue = crate::transformer::CompileGlueGuard::enable();

                // PiD decode overlay (epic 7840, sc-7847) + `from_ldm` early-stop (sc-8048): when
                // `req.use_pid` is set and the model was loaded with `LoadSpec::pid`, mint a
                // per-generation decoder that super-resolves the packed latent 4× in place of the VAE.
                // Errors if requested-but-not-loaded; `None` (the default) → the byte-exact VAE path. One
                // decoder serves the whole count loop. When `pid_capture_sigma` asks for an early exit,
                // stop the denoise at the achieved-σ step and hand PiD the partially-denoised packed
                // latent (FLUX.2 is `vp_frame=false`, so the schedule σ *is* the degrade σ — flow-match
                // identity); the packed-128ch BN seam below is unchanged. `start_step` is the img2img
                // noise-blend offset (0 for txt2img), matching the schedule slice fed to the solver.
                let (capture_sigma, keep) =
                    flow_capture_for_request(req, &sched.sigmas, start_step);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid.as_ref(),
                    req,
                    base_seed,
                    self.descriptor.id,
                    capture_sigma,
                )?;

                // Image-guidance CFG on the reference condition (sc-8273/sc-8278): the identity-strength
                // lever, off by default so the shipped render is byte-identical. On the klein/dev EDIT
                // path identity rides ENTIRELY on the concatenated reference tokens; a strong prompt
                // drowns them (sc-8234). With `s > 1` the denoise extrapolates the with-reference
                // velocity against the reference-dropped (image-unconditional) velocity:
                // `v = v_img0 + s·(v_ref − v_img0)`, reusing the `include_ref=false` forward (non-kv
                // path). `req.image_guidance` wins; `FLUX2_IMG_GUIDANCE` is a debug fallback only (F-087).
                // Scoped to the non-kv edit path; the kv variant rejects `image_guidance` in
                // `validate_impl`. The env debug override skips `validate_request`, so filter a
                // non-finite value here (F-036).
                let img_guidance_debug = std::env::var("FLUX2_IMG_GUIDANCE")
                    .ok()
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .filter(|v| v.is_finite());
                let img_guidance: Option<f32> = req
                    .image_guidance
                    .or(img_guidance_debug)
                    .filter(|s| *s > 1.0 && reference.is_some());

                let sampler_name = req.sampler.as_deref();
                let mut images = Vec::with_capacity(req.count as usize);
                for i in 0..req.count {
                    let seed = base_seed.wrapping_add(i as u64);
                    let noise = create_noise(seed, req.width, req.height, self.config.in_channels)?;
                    // img2img: `(1-σ)·clean + σ·noise` at `σ = sigmas[start_step]`; txt2img: pure noise.
                    let latents = match &clean_init {
                        Some(clean) => {
                            add_noise_by_interpolation(clean, &noise, sched.sigmas[start_step])?
                        }
                        None => noise,
                    };
                    // Fresh cache per seed — the cached reference K/V depend on the step-0 target latents.
                    let cache = kv_enabled.then(|| {
                        Flux2KvCache::new(
                            self.config.num_double_layers,
                            self.config.num_single_layers,
                        )
                    });
                    // The curated unified-framework solver owns the loop (epic 7114 P3). KV step role:
                    // the first executed forward extracts the reference K/V (the full `[txt, target,
                    // ref]` pass); later forwards run `[txt, target]` and splice the cached ref K/V back
                    // in. "First executed forward" is tracked by `extracted` so a multi-eval solver still
                    // extracts once; the single-eval Euler default is byte-identical to the prior loop.
                    // FLUX.2 feeds `sigma · 1000` as the transformer timestep (Sigma convention).
                    let mut extracted = false;
                    let predict = |latents: &Array, sigma: f32| -> Result<Array> {
                        let ts = sigma * 1000.0;
                        let (include_ref, cache_ref) = match &cache {
                            Some(c) => {
                                let mode = if extracted {
                                    CacheMode::Cached
                                } else {
                                    CacheMode::Extract
                                };
                                c.configure(mode, num_ref);
                                extracted = true;
                                (mode == CacheMode::Extract, Some(c))
                            }
                            None => (true, None),
                        };
                        let v = run(
                            latents,
                            &prompt_embeds,
                            &text_ids,
                            ts,
                            include_ref,
                            cache_ref,
                        )?;
                        // sc-8273 spike: image-guidance CFG (non-kv edit only — the ref-dropped forward
                        // must run without a KV cache in play). Recompute this step with the reference
                        // tokens dropped, then extrapolate toward the with-reference prediction.
                        let v = match img_guidance {
                            Some(s) if include_ref && cache_ref.is_none() => {
                                let v_img0 =
                                    run(latents, &prompt_embeds, &text_ids, ts, false, None)?;
                                add(&v_img0, &multiply(&subtract(&v, &v_img0)?, scalar(s))?)?
                            }
                            _ => v,
                        };
                        match &negative {
                            Some((neg_embeds, neg_ids)) => {
                                // CFG with the cache mirrors the fork: the same cache feeds both forwards
                                // (the negative extract overwrites the positive's slots). Distilled klein
                                // runs guidance 1.0 → no negative pass, so this is the base path.
                                let vn =
                                    run(latents, neg_embeds, neg_ids, ts, include_ref, cache_ref)?;
                                // noise = neg + guidance·(pos − neg)
                                Ok(add(&vn, &multiply(&subtract(&v, &vn)?, scalar(guidance))?)?)
                            }
                            None => Ok(v),
                        }
                    };
                    // Cancellation, the per-step `eval` (sc-5522 / sc-5399), and progress live in
                    // `run_flow_sampler`. img2img slices the schedule from `start_step`.
                    let denoise_sigmas = &sched.sigmas[start_step..keep];
                    let previews = mlx_gen::preview::PreviewCounter::new(denoise_sigmas);
                    let final_latents = run_flow_sampler_with_latent_hook(
                        sampler_name,
                        TimestepConvention::Sigma,
                        denoise_sigmas,
                        latents,
                        seed,
                        &req.cancel,
                        on_progress,
                        |latents, sigma| {
                            crate::preview::emit_flux_preview(
                                &req.preview,
                                &previews,
                                denoise_sigmas,
                                sigma,
                                latents,
                                lat_h as i32,
                                lat_w as i32,
                                vae,
                            );
                        },
                        predict,
                    )?;
                    on_progress(Progress::Decoding);
                    let packed =
                        final_latents.reshape(&[1, lat_h as i32, lat_w as i32, in_channels])?;
                    let nchw = match &pid_decoder {
                        // PiD: `packed` (NHWC [1,h,w,128]) is already the BN-normalized packed latent the
                        // student trained on — the exact tensor `decode_packed_latents` BN-de-normalizes
                        // (sc-7847). Hand it over as NCHW [1,128,h,w]; the student returns [1,3,4H,4W].
                        Some(d) => {
                            mlx_gen::ensure_decoder_layout(
                                self.descriptor.denoiser_output_latent_space,
                                d,
                            )?;
                            d.decode(&packed.transpose_axes(&[0, 3, 1, 2])?)?
                        }
                        // Native VAE: BN-de-normalize + 2×2-unpatchify + decode → NHWC [1,H,W,3] → NCHW.
                        None => match (!self.variant.is_dev())
                            .then(|| crate::memory_strategy::decode_tiling(req))
                            .transpose()?
                            .flatten()
                        {
                            Some(tiling) => vae
                                .decode_packed_latents_tiled(&packed, &tiling, Some(&req.cancel))?
                                .transpose_axes(&[0, 3, 1, 2])?,
                            None => vae
                                .decode_packed_latents(&packed)?
                                .transpose_axes(&[0, 3, 1, 2])?,
                        },
                    };
                    images.push(decoded_to_image(&nchw)?);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

pub(crate) fn validate_request(
    desc: &ModelDescriptor,
    is_edit: bool,
    is_kv: bool,
    req: &GenerationRequest,
) -> Result<()> {
    // Empty-prompt first so it wins over the shared floor for a bare default request.
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{}: prompt is required", desc.id)));
    }
    // The shared capability floor (count, size range, negative/guidance/true_cfg, sampler, scheduler,
    // conditioning) — the same check chroma delegates to (F-100; this dedups flux2's near-verbatim
    // copy and adds the previously-missing scheduler validation).
    desc.capabilities.validate_request(desc.id, req)?;
    // FLUX.2-specific: latent dims must be a multiple of 16 (VAE 8× × patch 2).
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "{}: width and height must be multiples of {SIZE_MULTIPLE}, got {}x{}",
            desc.id, req.width, req.height
        )));
    }
    // F-088: the edit variants consume at least one reference image (single `Reference`,
    // `MultiReference`, or several `Reference`s). Previously this was enforced only inside
    // `generate` (`collect_edit_references`); surface it at validate time — mirroring
    // `Flux2DevControl`'s `require_control_present` — so an editor rejects a reference-less request up
    // front instead of after loading and starting the run.
    if is_edit {
        let n_refs = collect_reference_images(req).len();
        if n_refs == 0 {
            return Err(Error::Msg(format!(
                "{}: edit requires at least one reference image",
                desc.id
            )));
        }
        // F-027: mirror `collect_edit_references`' reference cap at validate time (same F-088
        // rationale) so a worker's `validate()` rejects an over-cap request — e.g. 50 refs, a
        // quadratic-SDPA OOM — up front instead of after loading and starting the run.
        if n_refs > MAX_EDIT_REFERENCES {
            return Err(Error::Msg(format!(
                "{}: edit supports at most {MAX_EDIT_REFERENCES} reference images (got {n_refs}); \
                 each adds ~4096 joint-DiT tokens",
                desc.id
            )));
        }
        let reference_strength = req.conditioning.iter().any(|c| {
            matches!(
                c,
                mlx_gen::Conditioning::Reference {
                    strength: Some(_),
                    ..
                }
            )
        });
        if req.strength.is_some() || reference_strength {
            let img2img_id = if desc.id == crate::config::FLUX2_DEV_EDIT_ID {
                crate::config::FLUX2_DEV_ID
            } else {
                crate::config::FLUX2_KLEIN_9B_ID
            };
            return Err(Error::Msg(format!(
                "{}: strength is not supported for edit conditioning; use {img2img_id} for img2img strength",
                desc.id
            )));
        }
    }
    // image_guidance (reference true-CFG) is honored ONLY on the non-kv EDIT path — it needs the
    // uncached `include_ref=false` forward against a present reference. Everywhere else a set value is
    // silently ignored, so reject it up front instead of letting the request appear to succeed while
    // doing nothing (F-036 extends the F-087 kv-only rejection to txt2img and the control variant,
    // which both call this with `is_edit = false`). Non-finite `image_guidance` is already rejected by
    // the shared floor's central finiteness guard above (F-001).
    if req.image_guidance.is_some() && (!is_edit || is_kv) {
        let why = if is_kv {
            "the kv-edit variant's cached reference path can't run the dropped-ref CFG"
        } else {
            "image_guidance applies only to the non-kv edit path (a reference true-CFG lever)"
        };
        return Err(Error::Msg(format!(
            "{}: image_guidance is not supported here — {why}; it would be silently ignored",
            desc.id
        )));
    }
    Ok(())
}

// The registration constants bridge the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
/// Load-exact conditioning footprint for the staged-residency split. The selected source contributes
/// only its materialized language tower. Dev and DevEdit additionally retain the builtin Pixtral +
/// projector surface for caption upsampling; Klein and dev-control do not.
pub(crate) fn component_footprint_for(
    variant: Flux2Variant,
    provider_id: &str,
    include_builtin_multimodal: bool,
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root.as_path(),
        WeightsSource::File(_) => {
            return Err(mlx_gen::gen_core::Error::Msg(
                "FLUX.2 component footprint requires a snapshot directory".into(),
            ))
        }
    };
    let language_contract = variant.encoder_contract();
    let selected = language_contract.source_for_load(spec, root)?;
    // Dev's selected language tower follows the transformer tier exactly, including a pre-packed
    // base selected without `LoadSpec::quantize`. Klein intentionally keeps Qwen dense. Resolve and
    // validate that policy here, at the registry footprint consumed by the estimated fit fallback,
    // so admission cannot underprice a dense alternate or accept a packed mismatch the loader rejects.
    let expected_language_bits = if variant.is_dev() {
        effective_base_quant(spec, root, provider_id)?.map(mlx_gen::Quant::bits)
    } else {
        // Klein's published tiers deliberately keep Qwen exactly as stored. Do not inherit the
        // transformer request or reinterpret an existing encoder pack in this Dev-only fallback.
        None
    };
    // Always run the selected source through the contract's packed-policy gate. Klein's deliberate
    // `None` rejects packed Dir/File/complete-snapshot encoders instead of admitting a surface the
    // concrete loader rejects.
    let language_load_time_quant_bits =
        selected.load_time_quant_bits(expected_language_bits, provider_id)?;
    let language = selected.materialized_language_tensor_headers(&language_contract)?;
    let language_bytes =
        mlx_gen::asset_facts::projected_tensor_headers_bytes(&language, |tensor| {
            if let Some(bits) = language_load_time_quant_bits.filter(|_| {
                tensor
                    .name
                    .strip_suffix(".weight")
                    .is_some_and(crate::convert::is_te_quant_target)
            }) {
                mlx_gen::asset_facts::ResidentProjection::GroupQuantized {
                    bits,
                    group_size: language_contract
                        .packing
                        .expect("Dev's validated packed language policy has a packing contract")
                        .group_size,
                }
            } else {
                mlx_gen::asset_facts::ResidentProjection::Stored
            }
        })?;
    let mut multimodal_bytes = 0;
    if include_builtin_multimodal {
        let builtin = crate::config::DEV_ENCODER_CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
        let multimodal = builtin.materialized_vision_tensor_headers(
            &crate::config::DEV_VISION_ENCODER_CONTRACT,
            &crate::config::DEV_ENCODER_CONTRACT,
        )?;
        multimodal_bytes =
            mlx_gen::asset_facts::projected_tensor_headers_bytes(&multimodal, |_| {
                mlx_gen::asset_facts::ResidentProjection::Stored
            })?;
    }
    let conditioning_bytes = language_bytes
        .checked_add(multimodal_bytes)
        .ok_or_else(|| {
            mlx_gen::gen_core::Error::Msg(format!(
                "{}: selected language plus builtin multimodal resident byte overflow",
                provider_id
            ))
        })?;
    let mut footprint = mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )?;
    footprint.text_encoder = conditioning_bytes;
    Ok(footprint)
}

pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    component_footprint_for(
        Flux2Variant::Klein9b,
        crate::config::FLUX2_KLEIN_9B_ID,
        false,
        spec,
    )
}

pub(crate) fn klein_edit_component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    component_footprint_for(
        Flux2Variant::Klein9bEdit,
        crate::config::FLUX2_KLEIN_9B_EDIT_ID,
        false,
        spec,
    )
}

pub(crate) fn klein_kv_edit_component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    component_footprint_for(
        Flux2Variant::Klein9bKvEdit,
        crate::config::FLUX2_KLEIN_9B_KV_EDIT_ID,
        false,
        spec,
    )
}

pub(crate) fn dev_component_footprint_for(
    provider_id: &str,
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    component_footprint_for(Flux2Variant::Dev, provider_id, true, spec)
}

pub(crate) fn dev_component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    dev_component_footprint_for(crate::config::FLUX2_DEV_ID, spec)
}

pub(crate) fn dev_edit_component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    dev_component_footprint_for(crate::config::FLUX2_DEV_EDIT_ID, spec)
}

pub(crate) fn dev_control_component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    component_footprint_for(
        Flux2Variant::Dev,
        crate::config::FLUX2_DEV_CONTROL_ID,
        false,
        spec,
    )
}

mlx_gen::register_generators! {
    pub(crate) const KLEIN_REGISTRATION = descriptor_klein_9b => load_klein_9b;
    footprint = component_footprint
}
mlx_gen::register_generators! {
    pub(crate) const KLEIN_EDIT_REGISTRATION = descriptor_klein_9b_edit => load_klein_9b_edit;
    footprint = klein_edit_component_footprint
}
mlx_gen::register_generators! {
    pub(crate) const KLEIN_KV_EDIT_REGISTRATION =
        descriptor_klein_9b_kv_edit => load_klein_9b_kv_edit;
    footprint = klein_kv_edit_component_footprint
}
mlx_gen::register_generators! {
    pub(crate) const DEV_REGISTRATION = descriptor_dev => load_dev;
    footprint = dev_component_footprint
}
mlx_gen::register_generators! {
    pub(crate) const DEV_EDIT_REGISTRATION = descriptor_dev_edit => load_dev_edit;
    footprint = dev_edit_component_footprint
}

pub(crate) const DEV_EDIT_MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: crate::config::FLUX2_DEV_EDIT_ID,
        contract: crate::memory_strategy::registered_dev_contract,
        safety_check: crate::memory_strategy::registered_dev_safety_check,
    };

pub(crate) const DEV_MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: crate::config::FLUX2_DEV_ID,
        contract: crate::memory_strategy::registered_dev_t2i_contract,
        safety_check: crate::memory_strategy::registered_dev_t2i_safety_check,
    };

pub(crate) const KLEIN_MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: crate::config::FLUX2_KLEIN_9B_ID,
        contract: crate::memory_strategy::klein_contract,
        safety_check: crate::memory_strategy::registered_klein_safety_check,
    };

pub(crate) const KLEIN_EDIT_MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: crate::config::FLUX2_KLEIN_9B_EDIT_ID,
        contract: |spec| {
            crate::memory_strategy::klein_contract_for(crate::config::FLUX2_KLEIN_9B_EDIT_ID, spec)
        },
        safety_check: crate::memory_strategy::registered_klein_safety_check,
    };

pub(crate) const KLEIN_MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: crate::config::FLUX2_KLEIN_9B_ID,
        valid_fixtures: crate::memory_strategy::registered_klein_fixture,
        begin_request: crate::memory_strategy::registered_klein_begin_request,
    };

pub(crate) const KLEIN_EDIT_MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: crate::config::FLUX2_KLEIN_9B_EDIT_ID,
        valid_fixtures: crate::memory_strategy::registered_klein_fixture,
        begin_request: crate::memory_strategy::registered_klein_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_GUIDANCE_DEV, DEFAULT_STEPS_DEV, FLUX2_DEV_EDIT_ID, FLUX2_DEV_ID,
        FLUX2_KLEIN_9B_EDIT_ID, FLUX2_KLEIN_9B_ID,
    };
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryMode, MemoryNumericTier,
        MemoryRunContext, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    };
    use mlx_gen::media::Image;
    use mlx_gen::{Conditioning, Precision, Quant};

    /// L-log-injection: sanitize collapses embedded newlines/control chars (no second prefix line) and
    /// length-caps, so a model-generated rewrite can't break the machine-parsed `ENHANCED_PROMPT:` record.
    #[test]
    fn sanitize_log_text_collapses_and_caps() {
        let dirty = "a\nb\tc\r\nENHANCED_PROMPT:forged";
        let clean = sanitize_log_text(dirty);
        assert!(!clean.contains('\n') && !clean.contains('\t') && !clean.contains('\r'));
        assert_eq!(clean, "a b c ENHANCED_PROMPT:forged"); // newlines → spaces, but on ONE line
        let long = "x".repeat(1000);
        let capped = sanitize_log_text(&long);
        assert!(
            capped.chars().count() <= 513,
            "capped to ~512 chars + ellipsis"
        );
        assert!(capped.ends_with('…'));
        assert_eq!(sanitize_log_text("   "), "");
    }

    #[test]
    fn validates_basic_txt2img_request() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "a hummingbird".into(),
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn klein_variants_advertise_the_negative_branch_they_render() {
        for variant in [
            Flux2Variant::Klein9b,
            Flux2Variant::Klein9bEdit,
            Flux2Variant::Klein9bKvEdit,
        ] {
            let descriptor = variant.descriptor();
            assert!(
                descriptor.capabilities.supports_negative_prompt,
                "{} runs a negative forward at guidance > 1",
                variant.id()
            );
            assert!(descriptor.capabilities.supports_guidance);
            assert!(
                !descriptor.capabilities.supports_true_cfg,
                "{} does not consume the separate request.true_cfg knob",
                variant.id()
            );
            descriptor
                .capabilities
                .validate_request(
                    variant.id(),
                    &GenerationRequest {
                        prompt: "a portrait".into(),
                        guidance: Some(2.5),
                        negative_prompt: Some("watermark, blur".into()),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }

    #[test]
    fn klein_cfg_branch_encodes_the_supplied_negative_verbatim() {
        let supplied = GenerationRequest {
            negative_prompt: Some("watermark, blur".into()),
            ..Default::default()
        };
        let empty = GenerationRequest {
            negative_prompt: Some(String::new()),
            ..Default::default()
        };
        let unset = GenerationRequest::default();

        for variant in [
            Flux2Variant::Klein9b,
            Flux2Variant::Klein9bEdit,
            Flux2Variant::Klein9bKvEdit,
        ] {
            assert_eq!(
                cfg_negative_prompt(variant, 2.0, &supplied),
                Some("watermark, blur"),
                "{} must not replace the user's negative prompt with a hardcoded blank",
                variant.id()
            );
            assert_eq!(
                cfg_negative_prompt(variant, 2.0, &empty),
                Some(""),
                "{} must preserve an explicitly empty negative prompt",
                variant.id()
            );
            assert_eq!(
                cfg_negative_prompt(variant, 2.0, &unset),
                Some(" "),
                "{} preserves the historical unconditional prompt when unset",
                variant.id()
            );
            assert_eq!(cfg_negative_prompt(variant, 1.0, &supplied), None);

            let mut encoded = Vec::new();
            let branch = encode_cfg_negative_with(variant, 2.0, &supplied, |prompt| {
                encoded.push(prompt.to_owned());
                Ok::<_, ()>(prompt.len())
            })
            .unwrap();
            assert_eq!(encoded, ["watermark, blur"]);
            assert_eq!(
                branch,
                Some("watermark, blur".len()),
                "{} must create a conditional branch from the supplied text",
                variant.id()
            );
        }

        for variant in [Flux2Variant::Dev, Flux2Variant::DevEdit] {
            assert_eq!(
                cfg_negative_prompt(variant, 4.0, &supplied),
                None,
                "{} uses embedded guidance and must remain single-forward",
                variant.id()
            );
            let mut called = false;
            let branch = encode_cfg_negative_with(variant, 4.0, &supplied, |_| {
                called = true;
                Ok::<_, ()>(())
            })
            .unwrap();
            assert_eq!(branch, None);
            assert!(
                !called,
                "{} must not invoke the negative encoder",
                variant.id()
            );
        }
    }

    // ---- sc-2365 FLUX.2-dev T2I wiring ---------------------------------------------------------

    #[test]
    fn dev_descriptor_registered_with_t2i_caps() {
        // The dev variant is registered (loadable by id) with the dev id + txt2img/img2img caps.
        assert_eq!(descriptor_dev().id, FLUX2_DEV_ID);
        let d = descriptor_dev();
        assert!(d.capabilities.supports_guidance, "dev consumes guidance");
        assert!(
            !d.capabilities.supports_negative_prompt && !d.capabilities.supports_true_cfg,
            "dev is guidance-distilled, not true-CFG"
        );
        assert!(d.capabilities.mac_only);
        // A single Reference (img2img init), like klein txt2img — no edit conditioning.
        assert_eq!(
            d.capabilities.conditioning,
            vec![mlx_gen::ConditioningKind::Reference]
        );
    }

    #[test]
    fn dev_uses_embedded_guidance_with_dev_defaults() {
        assert!(Flux2Variant::Dev.uses_embedded_guidance());
        assert!(!Flux2Variant::Klein9b.uses_embedded_guidance());
        assert_eq!(Flux2Variant::Dev.default_steps(), DEFAULT_STEPS_DEV);
        assert_eq!(Flux2Variant::Dev.default_guidance(), DEFAULT_GUIDANCE_DEV);
    }

    #[test]
    fn dev_validates_basic_txt2img_request() {
        let model = Flux2::new_for_tests(Flux2Variant::Dev);
        let req = GenerationRequest {
            prompt: "a red fox in fresh snow".into(),
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    fn dev_memory_context(
        model: &Flux2,
        mode: MemoryMode,
        reference_count: u32,
    ) -> MemoryRunContext {
        let contract = model
            .memory_strategy_contract()
            .expect("test model memory contract");
        let calibration = contract.calibration.as_ref().expect("calibration identity");
        let tier = model.memory_numeric_tier.expect("loaded numeric tier");
        MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier,
            },
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: calibration.load_shape,
            mode,
            has_reference: reference_count > 0,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 768,
                height: 768,
                batch: 1,
                frames: 1,
                reference_count,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 96 * 1024 * 1024 * 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 80 * 1024 * 1024 * 1024,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "sc-18218-loaded-generator-test".to_owned(),
        }
    }

    #[test]
    fn dev_loaded_generator_uses_the_t2i_contract_not_the_edit_contract() {
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let mut t2i = Flux2::new_for_tests(Flux2Variant::Dev);
        t2i.memory_numeric_tier = Some(tier);
        assert_eq!(
            t2i.memory_strategy_contract().unwrap().provider_id,
            FLUX2_DEV_ID
        );
        let t2i_context = dev_memory_context(&t2i, MemoryMode::TextToImage, 0);
        assert_eq!(
            t2i.memory_strategy_safety_check(&t2i_context),
            MemorySafetyDecision::Accept
        );

        let edit_shaped = dev_memory_context(&t2i, MemoryMode::Edit, 2);
        let MemorySafetyDecision::Reject { reason } =
            t2i.memory_strategy_safety_check(&edit_shaped)
        else {
            panic!("T2I generator must reject the edit route");
        };
        assert!(reason.contains("reference-free text-to-image"), "{reason}");

        let mut edit = Flux2::new_for_tests(Flux2Variant::DevEdit);
        edit.memory_numeric_tier = Some(tier);
        assert_eq!(
            edit.memory_strategy_contract().unwrap().provider_id,
            FLUX2_DEV_EDIT_ID
        );
        let edit_context = dev_memory_context(&edit, MemoryMode::Edit, 2);
        assert_eq!(
            edit.memory_strategy_safety_check(&edit_context),
            MemorySafetyDecision::Accept,
            "the existing edit route must retain its own safety path"
        );
    }

    #[test]
    fn dev_loaded_generator_rejects_stale_evidence_and_a_wrong_tier() {
        let q4 = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let mut model = Flux2::new_for_tests(Flux2Variant::Dev);
        model.memory_numeric_tier = Some(q4);
        let exact = dev_memory_context(&model, MemoryMode::TextToImage, 0);

        let mut stale = exact.clone();
        stale.calibration_fingerprint = "stale".to_owned();
        let MemorySafetyDecision::Reject { reason } = model.memory_strategy_safety_check(&stale)
        else {
            panic!("loaded T2I generator must reject stale evidence");
        };
        assert!(
            reason.contains("calibration handshake mismatch"),
            "{reason}"
        );

        let mut wrong_tier = exact;
        wrong_tier.selection.tier.quant = Some(Quant::Q8);
        let MemorySafetyDecision::Reject { reason } =
            model.memory_strategy_safety_check(&wrong_tier)
        else {
            panic!("loaded T2I generator must reject a tier mismatch");
        };
        assert!(reason.contains("does not match loaded tier"), "{reason}");
    }

    fn dev_tier_spec(
        root: &Path,
        variant: Flux2Variant,
        packed_bits: Option<i32>,
        requested: Option<Quant>,
    ) -> LoadSpec {
        let mut spec = validation_complete_snapshot_spec(root, variant, OffloadPolicy::Sequential);
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(
            root.join("transformer/config.json"),
            packed_bits.map_or_else(
                || "{}".to_owned(),
                |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            ),
        )
        .unwrap();
        spec.quantize = requested;
        spec
    }

    fn public_dev_context(
        generator: &dyn Generator,
        mode: MemoryMode,
        reference_count: u32,
        tier: MemoryNumericTier,
    ) -> MemoryRunContext {
        let contract = generator
            .memory_strategy_contract()
            .expect("loaded Dev generator memory contract");
        let calibration = contract.calibration.as_ref().expect("Dev calibration");
        MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier,
            },
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: calibration.load_shape,
            mode,
            has_reference: reference_count > 0,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 768,
                height: 768,
                batch: 1,
                frames: 1,
                reference_count,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 96 * 1024 * 1024 * 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "effective-dev-tier-public-context".to_owned(),
        }
    }

    #[test]
    fn dev_loaded_and_registered_safety_use_the_effective_transformer_tier() {
        let registry = crate::provider_registry().unwrap();
        for (quant, bits) in [(Quant::Q4, 4), (Quant::Q8, 8)] {
            for prepacked in [false, true] {
                for (variant, provider_id, mode, references) in [
                    (Flux2Variant::Dev, FLUX2_DEV_ID, MemoryMode::TextToImage, 0),
                    (
                        Flux2Variant::DevEdit,
                        FLUX2_DEV_EDIT_ID,
                        MemoryMode::Edit,
                        2,
                    ),
                ] {
                    let fixture = tempfile::tempdir().unwrap();
                    let spec = dev_tier_spec(
                        fixture.path(),
                        variant,
                        prepacked.then_some(bits),
                        (!prepacked).then_some(quant),
                    );
                    let generator = match variant {
                        Flux2Variant::Dev => load_dev(&spec),
                        Flux2Variant::DevEdit => load_dev_edit(&spec),
                        _ => unreachable!(),
                    }
                    .unwrap_or_else(|error| {
                        panic!("Q{bits} prepacked={prepacked} {provider_id}: {error}")
                    });
                    assert_eq!(
                        generator.memory_strategy_contract().unwrap().provider_id,
                        provider_id
                    );
                    let tier = MemoryNumericTier {
                        precision: Precision::Bf16,
                        quant: Some(quant),
                        component_precision_floors: &[],
                    };
                    let context = public_dev_context(generator.as_ref(), mode, references, tier);
                    assert_eq!(
                        generator.memory_strategy_safety_check(&context),
                        MemorySafetyDecision::Accept,
                        "loaded Q{bits} prepacked={prepacked} {provider_id}"
                    );

                    let registration = registry
                        .memory_strategy_registrations()
                        .find(|registration| registration.provider_id == provider_id)
                        .unwrap();
                    let contract = (registration.contract)(&spec).unwrap();
                    assert_eq!(
                        (registration.safety_check)(&spec, &contract, &context),
                        MemorySafetyDecision::Accept,
                        "registered Q{bits} prepacked={prepacked} {provider_id}"
                    );

                    let mut wrong_tier = context;
                    wrong_tier.selection.tier.quant = Some(if quant == Quant::Q4 {
                        Quant::Q8
                    } else {
                        Quant::Q4
                    });
                    for decision in [
                        generator.memory_strategy_safety_check(&wrong_tier),
                        (registration.safety_check)(&spec, &contract, &wrong_tier),
                    ] {
                        let MemorySafetyDecision::Reject { reason } = decision else {
                            panic!("wrong Q{bits} public tier must reject for {provider_id}")
                        };
                        assert!(reason.contains("does not match loaded tier"), "{reason}");
                    }
                }
            }
        }
    }

    #[test]
    fn dev_load_and_registered_safety_reject_requested_vs_packed_tier_mismatches() {
        let registry = crate::provider_registry().unwrap();
        for (stored_bits, requested) in [(4, Quant::Q8), (8, Quant::Q4)] {
            for (variant, provider_id, mode, references) in [
                (Flux2Variant::Dev, FLUX2_DEV_ID, MemoryMode::TextToImage, 0),
                (
                    Flux2Variant::DevEdit,
                    FLUX2_DEV_EDIT_ID,
                    MemoryMode::Edit,
                    2,
                ),
            ] {
                let fixture = tempfile::tempdir().unwrap();
                let spec =
                    dev_tier_spec(fixture.path(), variant, Some(stored_bits), Some(requested));
                let load_error = match variant {
                    Flux2Variant::Dev => load_dev(&spec),
                    Flux2Variant::DevEdit => load_dev_edit(&spec),
                    _ => unreachable!(),
                }
                .err()
                .expect("packed/requested mismatch must reject load")
                .to_string();
                assert!(
                    load_error.contains(provider_id) && load_error.contains("pre-quantized"),
                    "{load_error}"
                );

                let registration = registry
                    .memory_strategy_registrations()
                    .find(|registration| registration.provider_id == provider_id)
                    .unwrap();
                let contract = (registration.contract)(&spec).unwrap();
                let requested_tier = MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(requested),
                    component_precision_floors: &[],
                };
                let test = Flux2::new_for_tests(variant);
                let context = public_dev_context(&test, mode, references, requested_tier);
                let MemorySafetyDecision::Reject { reason } =
                    (registration.safety_check)(&spec, &contract, &context)
                else {
                    panic!("registered mismatch must reject")
                };
                assert!(
                    reason.contains(provider_id) && reason.contains("pre-quantized"),
                    "{reason}"
                );
            }
        }
    }

    #[test]
    fn rejects_empty_prompt() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest::default();
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("prompt is required"));
    }

    #[test]
    fn rejects_unsupported_scheduler() {
        // F-100: flux2 delegated to the shared floor now validates the scheduler (was silently
        // accepted). epic 7114 scheduler axis: the curated names (e.g. "karras") + the "flow_match_euler"
        // native alias now pass; a genuinely unknown name is still rejected.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let err = model
            .validate(&GenerationRequest {
                prompt: "x".into(),
                scheduler: Some("not_a_real_scheduler".into()),
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported scheduler"), "got: {err}");
        for ok in ["flow_match_euler", "karras", "sgm_uniform"] {
            model
                .validate(&GenerationRequest {
                    prompt: "x".into(),
                    scheduler: Some(ok.into()),
                    ..Default::default()
                })
                .unwrap_or_else(|e| panic!("{ok} should validate: {e}"));
        }
    }

    #[test]
    fn rejects_non_multiple_of_16() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            width: 1023,
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("multiples of 16"));

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised FLUX.2 bucket
        // to. Pin the value and mutation-check that a size which is a multiple of 8 (the VAE scale) but
        // not SIZE_MULTIPLE (16) is still rejected with the stride error, and an on-stride size passes.
        assert_eq!(SIZE_MULTIPLE, 16);
        let off_stride = model
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1000, // 125×8 — a multiple of 8 but not SIZE_MULTIPLE
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiples of 16"),
            "expected the stride error, got: {off_stride}"
        );
        model
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024, // 64×16 — on-stride
                ..Default::default()
            })
            .unwrap();
    }

    #[test]
    fn txt2img_accepts_reference_conditioning() {
        // A `Reference` on the txt2img variant is an img2img init image (sc-2644).
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: Some(0.6),
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn txt2img_rejects_multiple_references() {
        // img2img conditions on exactly one init image; the resolver rejects more than one.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![
                Conditioning::Reference {
                    image: Image::default(),
                    strength: Some(0.6),
                },
                Conditioning::Reference {
                    image: Image::default(),
                    strength: Some(0.6),
                },
            ],
            ..Default::default()
        };
        let err = model.resolve_reference(&req).unwrap_err().to_string();
        assert!(err.contains("multiple reference images"));
    }

    #[test]
    fn edit_accepts_single_reference() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
        assert_eq!(model.collect_edit_references(&req).unwrap().len(), 1);
    }

    #[test]
    fn all_edit_variants_reject_reference_and_request_strength() {
        for (variant, img2img_id) in [
            (Flux2Variant::Klein9bEdit, FLUX2_KLEIN_9B_ID),
            (Flux2Variant::Klein9bKvEdit, FLUX2_KLEIN_9B_ID),
            (Flux2Variant::DevEdit, FLUX2_DEV_ID),
        ] {
            let model = Flux2::new_for_tests(variant);
            let reference_strength = GenerationRequest {
                prompt: "edit it".into(),
                conditioning: vec![Conditioning::Reference {
                    image: Image::default(),
                    strength: Some(0.5),
                }],
                ..Default::default()
            };
            let reference_err = model.validate(&reference_strength).unwrap_err().to_string();
            assert!(reference_err.contains(img2img_id), "got: {reference_err}");

            let request_strength = GenerationRequest {
                strength: Some(0.5),
                conditioning: vec![Conditioning::Reference {
                    image: Image::default(),
                    strength: None,
                }],
                ..reference_strength
            };
            let request_err = model.validate(&request_strength).unwrap_err().to_string();
            assert!(request_err.contains(img2img_id), "got: {request_err}");
        }
    }

    #[test]
    fn edit_accepts_multi_reference() {
        // sc-2645: N reference images via `MultiReference`, flattened in order.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "combine these".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(), Image::default(), Image::default()],
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
        assert_eq!(model.collect_edit_references(&req).unwrap().len(), 3);
    }

    #[test]
    fn edit_without_reference_errors() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            ..Default::default()
        };
        let err = model.collect_edit_references(&req).unwrap_err().to_string();
        assert!(err.contains("at least one reference image"));
    }

    #[test]
    fn edit_without_reference_rejected_at_validate() {
        // F-088: the edit variants require a reference — enforce it at `validate` (not only inside
        // `generate` via `collect_edit_references`), mirroring `Flux2DevControl::require_control_present`.
        // A non-edit variant (Klein9b txt2img) must still validate a reference-less request.
        let edit = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            ..Default::default()
        };
        let err = edit.validate(&req).unwrap_err().to_string();
        assert!(
            err.contains("at least one reference image"),
            "edit validate should reject a reference-less request, got: {err}"
        );
        // The txt2img variant tolerates no reference.
        let t2i = Flux2::new_for_tests(Flux2Variant::Klein9b);
        assert!(t2i.validate(&req).is_ok());
    }

    #[test]
    fn edit_over_cap_references_rejected() {
        // F-027: more than `MAX_EDIT_REFERENCES` refs is rejected at BOTH `validate` (up-front worker
        // rejection) and the generate-path collector; exactly the cap is accepted.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let over = GenerationRequest {
            prompt: "combine these".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(); MAX_EDIT_REFERENCES + 1],
            }],
            ..Default::default()
        };
        let err = model.validate(&over).unwrap_err().to_string();
        assert!(
            err.contains("at most 8 reference images (got 9)"),
            "validate should reject an over-cap request, got: {err}"
        );
        let err = model
            .collect_edit_references(&over)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("at most 8 reference images (got 9)"),
            "generate-path collector should reject an over-cap request, got: {err}"
        );
        // At the cap: both paths accept.
        let at_cap = GenerationRequest {
            prompt: "combine these".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(); MAX_EDIT_REFERENCES],
            }],
            ..Default::default()
        };
        model.validate(&at_cap).unwrap();
        assert_eq!(
            model.collect_edit_references(&at_cap).unwrap().len(),
            MAX_EDIT_REFERENCES
        );
    }

    #[test]
    fn image_guidance_rejected_where_not_honored() {
        // F-036: `image_guidance` is honored ONLY on the non-kv EDIT path. A set value on txt2img or
        // the kv-edit variant is silently ignored, so validate rejects it up front; the non-kv edit
        // path accepts it (with a reference present).
        let t2i = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let t2i_req = GenerationRequest {
            prompt: "a fox".into(),
            image_guidance: Some(2.0),
            ..Default::default()
        };
        let err = t2i.validate(&t2i_req).unwrap_err().to_string();
        assert!(
            err.contains("image_guidance is not supported here"),
            "txt2img must reject image_guidance, got: {err}"
        );

        // Non-kv edit with a reference + image_guidance validates.
        let edit = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let edit_req = GenerationRequest {
            prompt: "make it night".into(),
            image_guidance: Some(2.5),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        assert!(
            edit.validate(&edit_req).is_ok(),
            "non-kv edit should honor image_guidance"
        );

        // A non-finite image_guidance is rejected by the shared floor's central finiteness guard
        // (F-001), even on the honoring edit path.
        let nan_req = GenerationRequest {
            image_guidance: Some(f32::NAN),
            ..edit_req
        };
        let err = edit.validate(&nan_req).unwrap_err().to_string();
        assert!(
            err.contains("image_guidance") && err.contains("finite"),
            "non-finite image_guidance must be rejected, got: {err}"
        );
    }

    // ---- sc-5919 FLUX.2-dev edit (DiT-concat reference conditioning) ---------------------------

    #[test]
    fn dev_edit_registered_with_edit_caps() {
        // Registered (loadable by id) with the dev-edit id + the klein edit conditioning surface.
        assert_eq!(descriptor_dev_edit().id, FLUX2_DEV_EDIT_ID);
        let caps = descriptor_dev_edit().capabilities;
        assert_eq!(
            caps.conditioning,
            vec![
                mlx_gen::ConditioningKind::Reference,
                mlx_gen::ConditioningKind::MultiReference,
            ]
        );
        // Embedded guidance (no negative/true-CFG), no KV cache, mac-only.
        assert!(
            caps.supports_guidance && !caps.supports_negative_prompt && !caps.supports_true_cfg
        );
        assert!(!caps.supports_kv_cache && caps.mac_only);
    }

    #[test]
    fn dev_edit_accepts_single_and_multi_reference() {
        let model = Flux2::new_for_tests(Flux2Variant::DevEdit);
        // Single `Reference`.
        let single = GenerationRequest {
            prompt: "make it a watercolor".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        model.validate(&single).unwrap();
        assert_eq!(model.collect_edit_references(&single).unwrap().len(), 1);
        // `MultiReference` (N images).
        let multi = GenerationRequest {
            prompt: "combine these".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(), Image::default()],
            }],
            ..Default::default()
        };
        model.validate(&multi).unwrap();
        assert_eq!(model.collect_edit_references(&multi).unwrap().len(), 2);
    }

    #[test]
    fn dev_edit_without_reference_errors() {
        let model = Flux2::new_for_tests(Flux2Variant::DevEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            ..Default::default()
        };
        let err = model.collect_edit_references(&req).unwrap_err().to_string();
        assert!(err.contains("at least one reference image"));
    }

    #[test]
    fn txt2img_rejects_multi_reference() {
        // Multi-image editing belongs to the edit variant, not txt2img.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(), Image::default()],
            }],
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("conditioning"));
    }

    #[test]
    fn generate_without_weights_errors_not_loaded() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let mut progress = |_p: Progress| {};
        let err = model.generate(&req, &mut progress).unwrap_err().to_string();
        assert!(err.contains("not loaded"));
    }

    #[test]
    fn ids_match_expected() {
        assert_eq!(descriptor_klein_9b().id, FLUX2_KLEIN_9B_ID);
        assert_eq!(descriptor_klein_9b_edit().id, FLUX2_KLEIN_9B_EDIT_ID);
    }

    // ── sc-10840: weight-free, default-run proof that FLUX.2's dispatch HONORS `offload_policy`.
    // `build_residency` at a validation-complete sparse snapshot: `Sequential` admits the encoder
    // contract but defers payload materialization (`is_sequential`); `Resident` immediately enters
    // the unchanged payload bracket, proven without materializing the sparse production-size file.
    // Runs for a klein (Qwen3)
    // and a dev (Mistral-3 group) variant, so both text-loader arms are exercised. The real-weight A/B
    // is deferred (weights not on disk).
    fn validation_complete_snapshot_spec(
        root: &Path,
        variant: Flux2Variant,
        policy: OffloadPolicy,
    ) -> LoadSpec {
        if variant.is_dev() {
            gen_core_testkit::write_multimodal_encoder_contract_fixture(
                &root.join("text_encoder"),
                variant.encoder_contract(),
                crate::config::DEV_VISION_ENCODER_CONTRACT,
            )
            .unwrap();
        } else {
            gen_core_testkit::write_encoder_contract_fixture(
                &root.join("text_encoder"),
                variant.encoder_contract(),
            )
            .unwrap();
        }
        LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        for variant in [Flux2Variant::Klein9b, Flux2Variant::Dev] {
            for shape in [
                mlx_gen::LoadShape::EagerMaterialization,
                mlx_gen::LoadShape::DeferredMaterialization,
            ] {
                let fixture = tempfile::tempdir().unwrap();
                let mut spec = validation_complete_snapshot_spec(
                    fixture.path(),
                    variant,
                    OffloadPolicy::Sequential,
                );
                spec.load_shape = shape;
                let res = build_residency(variant, &spec)
                    .expect("Sequential must admit every consumed encoder surface before deferring payload loads");
                assert!(
                    res.is_sequential(),
                    "{} {shape:?}: Sequential policy must build a deferred residency",
                    variant.id()
                );
            }
        }
    }

    fn rewrite_tensor_shape(root: &Path, tensor: &str, first_dimension: usize) {
        use std::io::{Read as _, Seek as _, SeekFrom, Write as _};

        let path = root.join("text_encoder/model.safetensors");
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut encoded_len = [0_u8; 8];
        file.read_exact(&mut encoded_len).unwrap();
        let header_len = u64::from_le_bytes(encoded_len) as usize;
        let mut encoded = vec![0_u8; header_len];
        file.read_exact(&mut encoded).unwrap();
        let mut header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&encoded).unwrap();
        let entry = header.get_mut(tensor).unwrap();
        let old_first = entry["shape"][0].as_u64().unwrap();
        let row_elements = entry["shape"].as_array().unwrap()[1..]
            .iter()
            .map(|dimension| dimension.as_u64().unwrap())
            .product::<u64>();
        let old_end = entry["data_offsets"][1].as_u64().unwrap();
        let added_bytes = (first_dimension as u64 - old_first) * row_elements * 2;
        entry["shape"][0] = serde_json::json!(first_dimension);
        entry["data_offsets"][1] = serde_json::json!(old_end + added_bytes);
        let payload_len = header
            .values()
            .filter_map(|entry| entry["data_offsets"][1].as_u64())
            .max()
            .unwrap();
        let encoded = serde_json::to_vec(&header).unwrap();
        file.set_len(0).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.set_len(8 + encoded.len() as u64 + payload_len)
            .unwrap();
    }

    fn append_sparse_f16_tensor_to(path: &Path, name: &str, shape: &[usize]) {
        use std::io::{Read as _, Seek as _, SeekFrom, Write as _};

        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let mut encoded_len = [0_u8; 8];
        file.read_exact(&mut encoded_len).unwrap();
        let mut encoded = vec![0_u8; u64::from_le_bytes(encoded_len) as usize];
        file.read_exact(&mut encoded).unwrap();
        let mut header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&encoded).unwrap();
        let start = header
            .values()
            .filter_map(|entry| entry["data_offsets"][1].as_u64())
            .max()
            .unwrap();
        let bytes = shape
            .iter()
            .try_fold(2_u64, |bytes, dimension| {
                bytes.checked_mul(*dimension as u64)
            })
            .unwrap();
        let end = start.checked_add(bytes).unwrap();
        assert!(header
            .insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": "F16",
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            )
            .is_none());
        let encoded = serde_json::to_vec(&header).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.set_len(8 + encoded.len() as u64 + end).unwrap();
    }

    fn append_sparse_f16_tensor(root: &Path, name: &str, shape: &[usize]) {
        append_sparse_f16_tensor_to(&root.join("text_encoder/model.safetensors"), name, shape);
    }

    #[test]
    fn dev_multimodal_contract_fails_closed_for_deferred_routes_and_public_loaders() {
        // One sparse fixture represents a logical production-size multimodal checkpoint. Reuse it
        // across the route/load-shape matrix: recreating the same ~45 GiB logical sparse file for
        // every case is unnecessary and can trip hosted-runner resource accounting even though no
        // payload is ever materialized.
        let fixture = tempfile::tempdir().unwrap();
        let base_spec = validation_complete_snapshot_spec(
            fixture.path(),
            Flux2Variant::Dev,
            OffloadPolicy::Sequential,
        );
        let config_path = fixture.path().join("text_encoder/config.json");
        let valid_config = std::fs::read(&config_path).unwrap();
        let mut invalid_config: serde_json::Value = serde_json::from_slice(&valid_config).unwrap();
        invalid_config["vision_config"]["num_hidden_layers"] = serde_json::json!(23);
        std::fs::write(&config_path, serde_json::to_vec(&invalid_config).unwrap()).unwrap();

        for variant in [Flux2Variant::Dev, Flux2Variant::DevEdit] {
            for shape in [
                mlx_gen::LoadShape::EagerMaterialization,
                mlx_gen::LoadShape::DeferredMaterialization,
            ] {
                let mut spec = base_spec.clone();
                spec.load_shape = shape;
                let error = build_residency(variant, &spec)
                    .err()
                    .expect("deferred construction must still load-admit Pixtral config")
                    .to_string();
                assert!(error.contains("vision_config.num_hidden_layers"), "{error}");
            }
        }

        std::fs::write(&config_path, valid_config).unwrap();
        rewrite_tensor_shape(
            fixture.path(),
            "multi_modal_projector.linear_2.weight",
            5121,
        );
        for error in [
            crate::loader::load_vision_tower_dev(fixture.path())
                .err()
                .expect("vision loader must validate the paired projector")
                .to_string(),
            crate::loader::load_multimodal_projector_dev(fixture.path())
                .err()
                .expect("projector loader must validate its exact header")
                .to_string(),
            crate::loader::load_dev_text_encoder_group(fixture.path())
                .err()
                .expect("group loader must validate the whole multimodal source")
                .to_string(),
        ] {
            assert!(error.contains("vision_tensor_shape"), "{error}");
            assert!(
                error.contains("multi_modal_projector.linear_2.weight"),
                "{error}"
            );
        }
    }

    #[test]
    fn dev_registry_footprints_dedup_builtin_multimodal_and_ignore_override_visuals() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = crate::provider_registry().unwrap();
        let base_spec = validation_complete_snapshot_spec(
            tmp.path(),
            Flux2Variant::Dev,
            OffloadPolicy::Sequential,
        );
        let footprint =
            |id: &str, spec: &LoadSpec| registry.footprint(id, spec).unwrap().unwrap().text_encoder;
        let dev = footprint(crate::config::FLUX2_DEV_ID, &base_spec);
        let edit = footprint(crate::config::FLUX2_DEV_EDIT_ID, &base_spec);
        let control = footprint(crate::config::FLUX2_DEV_CONTROL_ID, &base_spec);
        assert_eq!(dev, edit);
        assert!(dev > control, "Dev routes add Pixtral + projector once");

        let language_only = tmp.path().join("alternate-language");
        gen_core_testkit::write_encoder_contract_fixture(
            &language_only,
            crate::config::DEV_ENCODER_CONTRACT,
        )
        .unwrap();
        let complete = tmp.path().join("alternate-complete");
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &complete.join("text_encoder"),
            crate::config::DEV_ENCODER_CONTRACT,
            crate::config::DEV_VISION_ENCODER_CONTRACT,
        )
        .unwrap();
        let language_spec = base_spec
            .clone()
            .with_text_encoder(WeightsSource::Dir(language_only));
        let complete_spec = base_spec
            .clone()
            .with_text_encoder(WeightsSource::Dir(complete));
        for id in [
            crate::config::FLUX2_DEV_ID,
            crate::config::FLUX2_DEV_EDIT_ID,
            crate::config::FLUX2_DEV_CONTROL_ID,
        ] {
            assert_eq!(
                footprint(id, &language_spec),
                footprint(id, &complete_spec),
                "{id}: alternate visual/projector tensors are not consumed"
            );
        }
        assert_eq!(
            footprint(crate::config::FLUX2_DEV_ID, &language_spec)
                - footprint(crate::config::FLUX2_DEV_CONTROL_ID, &language_spec),
            dev - control,
            "builtin Pixtral + projector must be counted exactly once"
        );

        let estimated = mlx_gen::PerComponentBytes::from_spec_subdirs(
            &base_spec,
            &["text_encoder"],
            &["transformer"],
            &["vae"],
        )
        .expect("the generic estimated-fallback path remains available");
        assert!(estimated.text_encoder > 0);
    }

    #[derive(Clone, Copy, Debug)]
    enum DevFootprintSelection {
        Builtin,
        ComponentDir,
        ComponentFile,
        CompleteSnapshot,
    }

    fn write_tiny_component(path: &Path) {
        std::fs::create_dir_all(path).unwrap();
        let header = br#"{"probe":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 2]);
        std::fs::write(path.join("model.safetensors"), bytes).unwrap();
    }

    fn dev_footprint_spec(
        fixture: &Path,
        selection: DevFootprintSelection,
        base_quant: Option<i32>,
        requested_quant: Option<Quant>,
        selected_quant: Option<i32>,
    ) -> LoadSpec {
        let base = fixture.join("base");
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &base.join("text_encoder"),
            crate::config::DEV_ENCODER_CONTRACT,
            crate::config::DEV_VISION_ENCODER_CONTRACT,
        )
        .unwrap();
        write_tiny_component(&base.join("transformer"));
        write_tiny_component(&base.join("vae"));
        std::fs::write(
            base.join("transformer/config.json"),
            base_quant.map_or_else(
                || "{}".to_owned(),
                |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            ),
        )
        .unwrap();

        let selected = fixture.join(format!("selected-{selection:?}"));
        let mut spec = LoadSpec::new(WeightsSource::Dir(base));
        spec.quantize = requested_quant;
        spec.text_encoder = match selection {
            DevFootprintSelection::Builtin => None,
            DevFootprintSelection::ComponentDir => {
                gen_core_testkit::write_encoder_contract_fixture_with_quant(
                    &selected,
                    crate::config::DEV_ENCODER_CONTRACT,
                    selected_quant,
                )
                .unwrap();
                Some(WeightsSource::Dir(selected))
            }
            DevFootprintSelection::ComponentFile => {
                gen_core_testkit::write_encoder_contract_fixture_with_quant(
                    &selected,
                    crate::config::DEV_ENCODER_CONTRACT,
                    selected_quant,
                )
                .unwrap();
                Some(WeightsSource::File(selected.join("model.safetensors")))
            }
            DevFootprintSelection::CompleteSnapshot => {
                gen_core_testkit::write_encoder_contract_fixture_with_quant(
                    &selected.join("text_encoder"),
                    crate::config::DEV_ENCODER_CONTRACT,
                    selected_quant,
                )
                .unwrap();
                Some(WeightsSource::Dir(selected))
            }
        };
        spec
    }

    fn klein_footprint_spec(
        fixture: &Path,
        selection: DevFootprintSelection,
        base_quant: Option<i32>,
        requested_quant: Option<Quant>,
        selected_quant: Option<i32>,
    ) -> LoadSpec {
        let base = fixture.join("base");
        let builtin_quant = matches!(selection, DevFootprintSelection::Builtin)
            .then_some(selected_quant)
            .flatten();
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            &base.join("text_encoder"),
            crate::config::KLEIN_ENCODER_CONTRACT,
            builtin_quant,
        )
        .unwrap();
        write_tiny_component(&base.join("transformer"));
        write_tiny_component(&base.join("vae"));
        std::fs::write(
            base.join("transformer/config.json"),
            base_quant.map_or_else(
                || "{}".to_owned(),
                |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            ),
        )
        .unwrap();

        let selected = fixture.join(format!("selected-{selection:?}"));
        let mut spec = LoadSpec::new(WeightsSource::Dir(base));
        spec.quantize = requested_quant;
        spec.text_encoder = match selection {
            DevFootprintSelection::Builtin => None,
            DevFootprintSelection::ComponentDir => {
                gen_core_testkit::write_encoder_contract_fixture_with_quant(
                    &selected,
                    crate::config::KLEIN_ENCODER_CONTRACT,
                    selected_quant,
                )
                .unwrap();
                Some(WeightsSource::Dir(selected))
            }
            DevFootprintSelection::ComponentFile => {
                gen_core_testkit::write_encoder_contract_fixture_with_quant(
                    &selected,
                    crate::config::KLEIN_ENCODER_CONTRACT,
                    selected_quant,
                )
                .unwrap();
                Some(WeightsSource::File(selected.join("model.safetensors")))
            }
            DevFootprintSelection::CompleteSnapshot => {
                gen_core_testkit::write_encoder_contract_fixture_with_quant(
                    &selected.join("text_encoder"),
                    crate::config::KLEIN_ENCODER_CONTRACT,
                    selected_quant,
                )
                .unwrap();
                Some(WeightsSource::Dir(selected))
            }
        };
        spec
    }

    fn expected_stored_language_bytes(
        spec: &LoadSpec,
        contract: mlx_gen::gen_core::EncoderContract,
        provider_id: &str,
    ) -> u64 {
        let root = mlx_gen::require_base_snapshot(spec, provider_id).unwrap();
        let selected = contract.source_for_load(spec, root).unwrap();
        selected.load_time_quant_bits(None, provider_id).unwrap();
        let headers = selected
            .materialized_language_tensor_headers(&contract)
            .unwrap();
        mlx_gen::asset_facts::projected_tensor_headers_bytes(&headers, |_| {
            mlx_gen::asset_facts::ResidentProjection::Stored
        })
        .unwrap()
    }

    fn expected_dev_language_bytes(spec: &LoadSpec, bits: Option<i32>, provider_id: &str) -> u64 {
        let root = mlx_gen::require_base_snapshot(spec, provider_id).unwrap();
        let selected = crate::config::DEV_ENCODER_CONTRACT
            .source_for_load(spec, root)
            .unwrap();
        let action = selected.load_time_quant_bits(bits, provider_id).unwrap();
        let headers = selected
            .materialized_language_tensor_headers(&crate::config::DEV_ENCODER_CONTRACT)
            .unwrap();
        mlx_gen::asset_facts::projected_tensor_headers_bytes(&headers, |tensor| {
            if let Some(bits) = action.filter(|_| {
                tensor
                    .name
                    .strip_suffix(".weight")
                    .is_some_and(crate::convert::is_te_quant_target)
            }) {
                mlx_gen::asset_facts::ResidentProjection::GroupQuantized {
                    bits,
                    group_size: 64,
                }
            } else {
                mlx_gen::asset_facts::ResidentProjection::Stored
            }
        })
        .unwrap()
    }

    fn expected_dev_multimodal_bytes(spec: &LoadSpec) -> u64 {
        let root = mlx_gen::require_base_snapshot(spec, FLUX2_DEV_ID).unwrap();
        let builtin = crate::config::DEV_ENCODER_CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)
            .unwrap();
        let headers = builtin
            .materialized_vision_tensor_headers(
                &crate::config::DEV_VISION_ENCODER_CONTRACT,
                &crate::config::DEV_ENCODER_CONTRACT,
            )
            .unwrap();
        mlx_gen::asset_facts::projected_tensor_headers_bytes(&headers, |_| {
            mlx_gen::asset_facts::ResidentProjection::Stored
        })
        .unwrap()
    }

    #[test]
    fn dev_estimated_fallback_prices_effective_language_tier_for_every_route_and_selector() {
        let registry = crate::provider_registry().unwrap();
        for (quant, bits) in [(Quant::Q4, 4), (Quant::Q8, 8)] {
            for prepacked_base in [false, true] {
                for selection in [
                    DevFootprintSelection::Builtin,
                    DevFootprintSelection::ComponentDir,
                    DevFootprintSelection::ComponentFile,
                    DevFootprintSelection::CompleteSnapshot,
                ] {
                    let tmp = tempfile::tempdir().unwrap();
                    let spec = dev_footprint_spec(
                        tmp.path(),
                        selection,
                        prepacked_base.then_some(bits),
                        (!prepacked_base).then_some(quant),
                        None,
                    );
                    let language = expected_dev_language_bytes(&spec, Some(bits), FLUX2_DEV_ID);
                    let multimodal = expected_dev_multimodal_bytes(&spec);
                    let generic_stored = mlx_gen::PerComponentBytes::from_spec_subdirs(
                        &spec,
                        &["text_encoder"],
                        &["transformer"],
                        &["vae"],
                    )
                    .unwrap();
                    for id in [
                        FLUX2_DEV_ID,
                        FLUX2_DEV_EDIT_ID,
                        crate::config::FLUX2_DEV_CONTROL_ID,
                    ] {
                        let footprint = registry.footprint(id, &spec).unwrap().unwrap();
                        let expected = if id == crate::config::FLUX2_DEV_CONTROL_ID {
                            language
                        } else {
                            language + multimodal
                        };
                        assert_eq!(
                            footprint.text_encoder, expected,
                            "Q{bits} prepacked={prepacked_base} {selection:?} {id}"
                        );
                        assert_eq!(footprint.dit, generic_stored.dit);
                        assert_eq!(footprint.vae, generic_stored.vae);
                    }
                    assert!(
                        generic_stored.text_encoder > language,
                        "the registry estimate must replace raw shards with the effective language projection"
                    );
                }
            }
        }
    }

    #[test]
    fn dev_estimated_fallback_preserves_matching_packs_rejects_mismatches_and_keeps_klein_stored() {
        let registry = crate::provider_registry().unwrap();
        for bits in [4, 8] {
            for selection in [
                DevFootprintSelection::ComponentDir,
                DevFootprintSelection::ComponentFile,
                DevFootprintSelection::CompleteSnapshot,
            ] {
                let matching_tmp = tempfile::tempdir().unwrap();
                let matching = dev_footprint_spec(
                    matching_tmp.path(),
                    selection,
                    Some(bits),
                    None,
                    Some(bits),
                );
                for id in [
                    FLUX2_DEV_ID,
                    FLUX2_DEV_EDIT_ID,
                    crate::config::FLUX2_DEV_CONTROL_ID,
                ] {
                    assert!(
                        registry.footprint(id, &matching).unwrap().is_some(),
                        "Q{bits} {selection:?} {id}"
                    );
                }

                let mismatch_tmp = tempfile::tempdir().unwrap();
                let mismatch = dev_footprint_spec(
                    mismatch_tmp.path(),
                    selection,
                    Some(bits),
                    None,
                    Some(if bits == 4 { 8 } else { 4 }),
                );
                for id in [
                    FLUX2_DEV_ID,
                    FLUX2_DEV_EDIT_ID,
                    crate::config::FLUX2_DEV_CONTROL_ID,
                ] {
                    let error = registry.footprint(id, &mismatch).unwrap_err().to_string();
                    assert!(
                        error.contains("pre-quantized") && error.contains("model policy"),
                        "Q{bits} {selection:?} {id}: {error}"
                    );
                    assert!(
                        error.contains(id),
                        "route-specific fallback error branding for {id}: {error}"
                    );
                }
            }

            let klein_tmp = tempfile::tempdir().unwrap();
            gen_core_testkit::write_encoder_contract_fixture_with_quant(
                &klein_tmp.path().join("text_encoder"),
                crate::config::KLEIN_ENCODER_CONTRACT,
                None,
            )
            .unwrap();
            write_tiny_component(&klein_tmp.path().join("transformer"));
            write_tiny_component(&klein_tmp.path().join("vae"));
            let mut klein = LoadSpec::new(WeightsSource::Dir(klein_tmp.path().to_path_buf()));
            let stored = registry
                .footprint(FLUX2_KLEIN_9B_ID, &klein)
                .unwrap()
                .unwrap();
            klein.quantize = Some(if bits == 4 { Quant::Q4 } else { Quant::Q8 });
            assert_eq!(
                registry
                    .footprint(FLUX2_KLEIN_9B_ID, &klein)
                    .unwrap()
                    .unwrap(),
                stored,
                "Klein keeps its selected Qwen bytes exactly Stored"
            );
        }
    }

    #[test]
    fn klein_registry_footprint_keeps_dense_language_stored_and_rejects_every_packed_selector() {
        let registry = crate::provider_registry().unwrap();
        let routes = [
            FLUX2_KLEIN_9B_ID,
            FLUX2_KLEIN_9B_EDIT_ID,
            crate::config::FLUX2_KLEIN_9B_KV_EDIT_ID,
        ];
        let selections = [
            DevFootprintSelection::Builtin,
            DevFootprintSelection::ComponentDir,
            DevFootprintSelection::ComponentFile,
            DevFootprintSelection::CompleteSnapshot,
        ];

        for base_quant in [None, Some(4), Some(8)] {
            for requested in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                for selection in selections {
                    let dense_tmp = tempfile::tempdir().unwrap();
                    let dense = klein_footprint_spec(
                        dense_tmp.path(),
                        selection,
                        base_quant,
                        requested,
                        None,
                    );
                    let expected = expected_stored_language_bytes(
                        &dense,
                        crate::config::KLEIN_ENCODER_CONTRACT,
                        FLUX2_KLEIN_9B_ID,
                    );
                    for route in routes {
                        assert_eq!(
                            registry
                                .footprint(route, &dense)
                                .unwrap()
                                .unwrap()
                                .text_encoder,
                            expected,
                            "dense {selection:?} base={base_quant:?} request={requested:?} {route}"
                        );
                    }

                    for packed_bits in [4, 8] {
                        let packed_tmp = tempfile::tempdir().unwrap();
                        let packed = klein_footprint_spec(
                            packed_tmp.path(),
                            selection,
                            base_quant,
                            requested,
                            Some(packed_bits),
                        );
                        for route in routes {
                            let error = registry.footprint(route, &packed).unwrap_err().to_string();
                            assert!(
                                error.contains(route)
                                    && error.contains("pre-quantized")
                                    && error.contains("model policy"),
                                "packed Q{packed_bits} {selection:?} base={base_quant:?} request={requested:?} {route}: {error}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn registry_footprint_excludes_unrelated_loaded_layer_namespace_tensors() {
        let fixture = tempfile::tempdir().unwrap();
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            &fixture.path().join("text_encoder"),
            crate::config::KLEIN_ENCODER_CONTRACT,
            None,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(fixture.path().to_path_buf()));
        let footprint = || {
            crate::provider_registry()
                .unwrap()
                .footprint(crate::config::FLUX2_KLEIN_9B_ID, &spec)
                .unwrap()
                .unwrap()
                .text_encoder
        };
        let baseline = footprint();
        append_sparse_f16_tensor(
            fixture.path(),
            "model.layers.0.unused_projection.weight",
            &[257],
        );
        assert_eq!(
            footprint(),
            baseline,
            "a valid but unconsumed tensor sharing a loaded-layer prefix must not affect staged-fit bytes"
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum DevEncoderSelection {
        DefaultDir,
        OverrideDir,
        OverrideFile,
    }

    fn dev_encoder_spec_with_sidecars(
        fixture: &Path,
        bits: i32,
        selection: DevEncoderSelection,
        sidecars: &[&str],
    ) -> LoadSpec {
        let base = fixture.join("base");
        let selected = fixture.join("selected");
        let selected_root = match selection {
            DevEncoderSelection::DefaultDir => base.join("text_encoder"),
            DevEncoderSelection::OverrideDir | DevEncoderSelection::OverrideFile => {
                gen_core_testkit::write_encoder_contract_fixture(
                    &base.join("text_encoder"),
                    crate::config::DEV_ENCODER_CONTRACT,
                )
                .unwrap();
                selected.clone()
            }
        };
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            &selected_root,
            crate::config::DEV_ENCODER_CONTRACT,
            Some(bits),
        )
        .unwrap();
        for sidecar in sidecars {
            append_sparse_f16_tensor_to(
                &selected_root.join("model.safetensors"),
                &format!("language_model.lm_head.{sidecar}"),
                &[1],
            );
        }
        let mut spec = LoadSpec::new(WeightsSource::Dir(base));
        spec.text_encoder = match selection {
            DevEncoderSelection::DefaultDir => None,
            DevEncoderSelection::OverrideDir => Some(WeightsSource::Dir(selected_root)),
            DevEncoderSelection::OverrideFile => {
                Some(WeightsSource::File(selected_root.join("model.safetensors")))
            }
        };
        spec
    }

    #[test]
    fn packed_dev_rejects_lm_head_sidecars_on_every_selection_surface() {
        for bits in [4, 8] {
            for selection in [
                DevEncoderSelection::DefaultDir,
                DevEncoderSelection::OverrideDir,
                DevEncoderSelection::OverrideFile,
            ] {
                for sidecars in [&["scales"][..], &["biases"][..], &["scales", "biases"][..]] {
                    let fixture = tempfile::tempdir().unwrap();
                    let spec =
                        dev_encoder_spec_with_sidecars(fixture.path(), bits, selection, sidecars);
                    let base = mlx_gen::require_base_snapshot(&spec, FLUX2_DEV_ID).unwrap();
                    let error = crate::config::DEV_ENCODER_CONTRACT
                        .source_for_load(&spec, base)
                        .expect_err("Dev's dense LM head must reject every packed sidecar")
                        .to_string();
                    assert!(
                        error.contains("language_model.lm_head")
                            && (error.contains("packed_surface")
                                || error.contains("packed_components")),
                        "Q{bits} {selection:?} {sidecars:?}: {error}"
                    );
                }
            }
        }
    }

    #[test]
    fn packed_dev_materialized_surface_keeps_the_dense_lm_head_only() {
        for bits in [4, 8] {
            let fixture = tempfile::tempdir().unwrap();
            let spec = dev_encoder_spec_with_sidecars(
                fixture.path(),
                bits,
                DevEncoderSelection::OverrideFile,
                &[],
            );
            let base = mlx_gen::require_base_snapshot(&spec, FLUX2_DEV_ID).unwrap();
            let selected = crate::config::DEV_ENCODER_CONTRACT
                .source_for_load(&spec, base)
                .unwrap();
            let names = selected
                .materialized_language_tensor_headers(&crate::config::DEV_ENCODER_CONTRACT)
                .unwrap()
                .into_iter()
                .map(|header| header.name)
                .collect::<std::collections::BTreeSet<_>>();
            assert!(names.contains("language_model.lm_head.weight"));
            assert!(!names.contains("language_model.lm_head.scales"));
            assert!(!names.contains("language_model.lm_head.biases"));

            let mut expected_matrix_bases = std::collections::BTreeSet::from([
                "language_model.model.embed_tokens".to_owned(),
                "language_model.lm_head".to_owned(),
            ]);
            for layer in 0..crate::config::DEV_ENCODER_CONTRACT.loaded_hidden_layers {
                for suffix in [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ] {
                    expected_matrix_bases
                        .insert(format!("language_model.model.layers.{layer}.{suffix}"));
                }
            }
            let actual_matrix_bases = names
                .iter()
                .filter_map(|name| {
                    name.strip_suffix(".weight")
                        .or_else(|| name.strip_suffix(".scales"))
                        .or_else(|| name.strip_suffix(".biases"))
                })
                .filter(|base| {
                    base.ends_with("embed_tokens")
                        || base.ends_with("lm_head")
                        || [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ]
                        .iter()
                        .any(|suffix| base.ends_with(suffix))
                })
                .map(str::to_owned)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(actual_matrix_bases, expected_matrix_bases);

            let runtime = [
                include_str!("text_encoder/attention.rs"),
                include_str!("text_encoder/mlp.rs"),
                include_str!("text_encoder/encoder.rs"),
                include_str!("text_encoder/mod.rs"),
            ]
            .join("\n");
            for suffix in [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
                "embed_tokens",
                "lm_head.weight",
            ] {
                assert!(
                    runtime.contains(suffix),
                    "contract matrix surface has no matching runtime constructor for {suffix}"
                );
            }
        }
    }

    #[test]
    fn build_residency_resident_enters_payload_bracket_after_admission() {
        for variant in [Flux2Variant::Klein9b, Flux2Variant::Dev] {
            let fixture = tempfile::tempdir().unwrap();
            let spec =
                validation_complete_snapshot_spec(fixture.path(), variant, OffloadPolicy::Resident);
            let root = resolve_root(variant, &spec).unwrap();
            let text_encoder_source = variant
                .encoder_contract()
                .source_for_load(&spec, root)
                .unwrap();
            let effective_quant_bits = variant
                .is_dev()
                .then(|| effective_base_quant(&spec, root, variant.id()))
                .transpose()
                .unwrap()
                .flatten()
                .map(mlx_gen::gen_core::Quant::bits);
            let text_encoder_load_time_quant_bits = text_encoder_source
                .load_time_quant_bits(effective_quant_bits, variant.id())
                .unwrap();
            let multimodal_encoder_source = if variant.is_dev() {
                let source = variant
                    .encoder_contract()
                    .validate_source_against_base(
                        &WeightsSource::Dir(root.join("text_encoder")),
                        root,
                    )
                    .unwrap();
                source
                    .validate_vision(
                        &crate::config::DEV_VISION_ENCODER_CONTRACT,
                        &crate::config::DEV_ENCODER_CONTRACT,
                    )
                    .unwrap();
                source
            } else {
                text_encoder_source.clone()
            };

            // The contract fixtures are sparse production-shape files and must never be
            // materialized. Mutate the admitted shard inventory instead: Sequential leaves this
            // check deferred, while Resident must invoke the real text-loader closure immediately.
            std::fs::write(
                root.join("text_encoder/added-after-admission.safetensors"),
                [],
            )
            .unwrap();
            let err = build_residency_from_admitted_sources(
                variant,
                &spec,
                text_encoder_source,
                multimodal_encoder_source,
                text_encoder_load_time_quant_bits,
            )
            .err()
            .expect("Resident must immediately enter the admitted payload-load bracket");
            let msg = err.to_string();
            assert!(
                msg.contains("shard inventory changed after validation"),
                "{}: expected the eager payload bracket to detect the post-admission mutation: {msg}",
                variant.id()
            );
        }
    }
}
