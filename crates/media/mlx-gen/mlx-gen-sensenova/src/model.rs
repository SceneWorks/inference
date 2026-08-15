//! sc-3194: the `Generator` registration that wires SenseNova-U1's image modes into the mlx-gen
//! provider registry as **`sensenova_u1_8b`**.
//!
//! The `Generator` contract emits images, so the image-producing modes route through one
//! [`Generator::generate`] dispatch: **T2I** (no conditioning → [`T2iModel::generate`]) and
//! **image-edit / Character Studio** (a [`Conditioning::Reference`]/[`Conditioning::MultiReference`] →
//! [`T2iModel::it2i_generate`]). The generic request maps as: `guidance` → text `cfg_scale`,
//! `true_cfg` → image `img_cfg_scale` (edit ≈ 1.0, character ≈ 1.5), `scheduler_shift` →
//! `timestep_shift`, `steps`/`seed`/`width`/`height` as given.
//!
//! VQA ([`T2iModel::vqa`], text out) and interleave / Document Studio
//! ([`T2iModel::interleave_gen`], text + images) cannot be expressed by [`GenerationOutput`]
//! (`Images`/`Video` only), so they are consumed by the SceneWorks worker through those public
//! [`T2iModel`] methods directly — the registry path here covers exactly the image-generation
//! surface. `spec.quantize` (Q4/Q8) quantizes the backbone decoder stack (sc-3193).
//!
//! A **second** id, `sensenova_u1_8b_fast` (sc-3192), shares this loader: its [`load_fast`] merges
//! the 8-step distill LoRA into the dense generation path before any quantization, and its
//! generator applies the distilled defaults (`cfg_scale=1.0`, `num_steps=8`). Registering it under
//! a distinct id makes the merge part of the model cache key — the worker caches by id, so the
//! merged variant can never be served for the base id (and vice versa) even though they share the
//! same on-disk base weights. User-supplied LoRAs stay rejected for both ids (`supports_lora=false`,
//! matching the torch adapter); the distill LoRA is a curated property of the fast variant, not a
//! user adapter.

use mlx_rs::ops::divide;
use mlx_rs::Array;

use mlx_gen::gen_core::reject_unknown_components;
use mlx_gen::image::{decoded_to_image, resize_bicubic_u8};
use mlx_gen::{
    default_seed, gen_core, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress,
    Quant, Result, SizeFloor, WeightsSource,
};

use crate::config::NeoChatConfig;
use crate::distill::{resolve_distill_lora, DISTILL_MERGED_MARKER};
use crate::loader::{check_coverage, load_raw};
use crate::t2i::{smart_resize, StepReporter, T2iModel, T2iOptions};
use crate::text::load_tokenizer;
use mlx_gen::weights::Weights;

pub const MODEL_ID: &str = "sensenova_u1_8b";
/// The 8-step distilled variant (sc-3192): same base weights with the distill LoRA merged in.
pub const MODEL_ID_FAST: &str = "sensenova_u1_8b_fast";

const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 4.0;
/// Distilled defaults (`docs/base_vs_distill.md`): 8 NFE at CFG 1.0 (guidance off).
const DEFAULT_STEPS_FAST: u32 = 8;
const DEFAULT_GUIDANCE_FAST: f32 = 1.0;
const DEFAULT_TIMESTEP_SHIFT: f32 = 3.0;
/// Cell = patch·merge: every side must be a multiple of this (the patchify grid). Exposed as the
/// pinned-engine stride SceneWorks ties each advertised SenseNova image bucket to (sc-12612).
/// `validate_dims_and_steps` enforces exactly this value, so the const cannot drift from the check.
pub const CELL: u32 = 32;
/// Source-image preprocessing bounds (the reference `it2i_generate` `load_image_native`).
const REF_MIN_PIXELS: i64 = 512 * 512;
const REF_MAX_PIXELS: i64 = 2048 * 2048;

/// Resolve the request's image-guidance scale for the reference-conditioned it2i path. Keeping this
/// seam explicit prevents `true_cfg` from being advertised while accidentally dropping it before
/// [`T2iModel::it2i_generate`].
fn image_cfg_scale(req: &GenerationRequest) -> f32 {
    req.true_cfg.unwrap_or(1.0)
}

pub fn descriptor() -> ModelDescriptor {
    descriptor_for(MODEL_ID)
}

/// The descriptor for the 8-step distilled variant. Identical capabilities to the base — only the
/// id and the generation defaults (applied in `SenseNova::options`) differ.
pub fn descriptor_fast() -> ModelDescriptor {
    descriptor_for(MODEL_ID_FAST)
}

fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id,
        family: "sensenova-u1",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // `guidance` → text cfg_scale; `true_cfg` → image cfg (it2i edit≈1.0 / character≈1.5).
            supports_guidance: true,
            supports_true_cfg: true,
            // Reference image(s) → it2i edit / Character Studio reference. No control/depth/mask.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ],
            supports_lora: false,
            supports_lokr: false,
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            // Bespoke-by-architecture (epic 7114, sc-7120, task 7185): SenseNova-U1 is NOT routed through
            // the unified curated-sampler framework. Its denoise threads each step through an
            // autoregressive backbone with a per-step-mutated `KvCache` (`predict_v` appends to the cache;
            // `cache.offset()` feeds the RoPE/position build), shared between the cond/uncond passes. The
            // multistep/2nd-order solvers (heun = 2 evals/step; dpmpp_2m / uni_pc reuse prior-step state)
            // would append to the cache multiple times per step and desync the AR positions → corrupt
            // output. It is also x0-prediction over a clean-fraction timestep grid (not noise-fraction σ).
            // Its native shifted-Euler loop is its only valid sampler. See `t2i::denoise`/`it2i_denoise`.
            samplers: Vec::new(),
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            // The backbone uses a KV cache for the AR prefix + denoise.
            supports_kv_cache: true,
            // Flow-match schedule uses a timestep shift (mapped from scheduler_shift).
            requires_sigma_shift: true,
            // Not wired onto the shared `Residency` seam (F-176); Sequential is a no-op fallback.
            supports_sequential_offload: false,
            unconditionally_engages_staged_residency: false,
            supports_preview: false,
            supports_prompt_enhancement: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
        },
    }
}

/// A loaded SenseNova-U1 generator: the unified [`T2iModel`] + tokenizer + cached descriptor.
pub struct SenseNova {
    descriptor: ModelDescriptor,
    tokenizer: mlx_gen::tokenizer::TextTokenizer,
    model: T2iModel,
    /// The 8-step distilled variant — selects the distilled generation defaults (8 NFE, CFG 1.0).
    fast: bool,
    memory_strategy: gen_core::MemoryProviderContract,
    loaded_quant: Option<Quant>,
    pinned_artifact: Option<crate::memory_strategy::PinnedArtifact>,
}

fn guard_pinned_artifact<T>(
    artifact: Option<&crate::memory_strategy::PinnedArtifact>,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if let Some(artifact) = artifact {
        artifact
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
    }
    let result = operation();
    if let Some(artifact) = artifact {
        // Always run the post-check, including after a generation error. A mutation error takes
        // precedence so no output (or misleading earlier failure) can escape a changed artifact.
        artifact
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
    }
    result
}

impl SenseNova {
    /// The unified model, for the worker paths the `Generator` contract can't express (VQA text,
    /// interleave text+images): call [`T2iModel::vqa`] / [`T2iModel::interleave_gen`] directly.
    pub fn model(&self) -> &T2iModel {
        &self.model
    }

    /// The tokenizer (shared by every mode).
    pub fn tokenizer(&self) -> &mlx_gen::tokenizer::TextTokenizer {
        &self.tokenizer
    }
}

/// Construct the base [`SenseNova`] (`sensenova_u1_8b`) from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `sensenova/SenseNova-U1-8B-MoT` snapshot. Weights load dense
/// at their on-disk dtype (bf16); `spec.quantize` (Q4/Q8) then quantizes the backbone decoder stack
/// (sc-3193).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_inner(spec, false)
}

/// Construct the 8-step distilled [`SenseNova`] (`sensenova_u1_8b_fast`, sc-3192), plus the distilled
/// generation defaults. Two tier shapes load here:
/// - a **dense base snapshot** (no [`DISTILL_MERGED_MARKER`]): the distill LoRA is resolved by
///   [`resolve_distill_lora`] from the caller-supplied `distill_lora` component (else the co-located
///   snapshot file — no env / HF-cache fallback, sc-13664) and merged into the dense generation path
///   **before** any quantization — the original sc-3192 path; and
/// - a **pre-merged turnkey** (sc-8775: the packed q4/q8 or dense bf16 fast tiers built by
///   [`crate::convert::prequantize_fast_turnkey`], marked with [`DISTILL_MERGED_MARKER`]): the merge
///   is already baked into the on-disk weights, so the loader skips it (a packed tier cannot re-merge
///   — its base is quantized). Either way the distilled 8-NFE / CFG-1.0 defaults apply.
pub fn load_fast(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_inner(spec, true)
}

fn load_inner(spec: &LoadSpec, fast: bool) -> Result<Box<dyn Generator>> {
    let id = if fast { MODEL_ID_FAST } else { MODEL_ID };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only dense bf16 is wired (drop the precision override)"
        )));
    }
    // User-supplied LoRAs are unsupported on both ids (the distill LoRA is merged internally by the
    // fast loader, not stacked via `spec.adapters`).
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: user-supplied adapters are not supported (supports_lora=false)"
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{id} expects a snapshot directory, not a single .safetensors file"
            )))
        }
    };
    // Named-component contract (sc-13658/sc-13664): the fast variant reads the `distill_lora`
    // component; the base id reads none. Reject any unrecognized component key up front.
    let known: &[&str] = if fast { &["distill_lora"] } else { &[] };
    reject_unknown_components(spec, known, id)?;
    // Resolve (and existence-check) the distill LoRA path for a dense-base fast tier *before* the heavy
    // config/weights I/O, so a missing LoRA fails fast at load with an actionable, `distill_lora`-named
    // error (sc-13664: from the caller-supplied component or the co-located snapshot file — never an
    // env side-channel or the HF cache). A **pre-merged** turnkey tier (`DISTILL_MERGED_MARKER`
    // present, sc-8775) bakes the merge in and needs no LoRA, so it is exempt — which is exactly why the
    // fast descriptor does NOT declare `required_components: &["distill_lora"]` (that would be a false
    // universal requirement that broke the pre-merged tier).
    let distill_lora_path = if fast && !root.join(DISTILL_MERGED_MARKER).exists() {
        Some(resolve_distill_lora(
            spec.components.get("distill_lora"),
            root,
        )?)
    } else {
        None
    };
    let cfg = NeoChatConfig::from_dir(root)?;
    // A single-file artifact is verified once and then becomes the source of truth for the initial
    // load, calibration identity, and every deferred Gen window. Sharded/multi-file layouts
    // retain the historical eager directory loader but truthfully do not advertise rung 4.
    let pinned_artifact = crate::memory_strategy::verified_artifact(spec);
    let weights = if let Some(artifact) = pinned_artifact.as_ref() {
        artifact.open_weights()?
    } else {
        load_raw(root)?
    };
    // F-137: diff the checkpoint against the canonical key set before building modules (the loader
    // module doc promised this validation). Missing keys still fail via `require` with the exact
    // name during `from_weights`; this additionally rejects extra/renamed tensors that would
    // otherwise load silently with whatever subset matches.
    check_coverage(weights.keys(), &cfg).require_no_unexpected(id)?;
    let deferred_gen = distill_lora_path.is_none()
        && crate::memory_strategy::can_stream_gen_with_artifact(id, spec, pinned_artifact.as_ref());
    let mut model = if deferred_gen {
        T2iModel::from_weights_deferred(
            &weights,
            &cfg,
            pinned_artifact
                .clone()
                .expect("streamable artifact is pinned"),
            spec.quantize,
        )?
    } else {
        // A fast dense-base load needs a runtime LoRA merge into all generation blocks. Keep that
        // exact path eager; the contract refuses rung 4 until the artifact is a pre-merged turnkey.
        T2iModel::from_weights(&weights, &cfg)?
    };
    // The fast variant merges the 8-step distill LoRA into the dense generation path — UNLESS the
    // tier is a **pre-merged** turnkey (sc-8775: the packed/dense fast tiers bake the merge in at
    // convert time and drop `DISTILL_MERGED_MARKER`). A pre-merged tier must NOT re-merge: for a
    // packed tier `from_weights` already built quantized Linears, and `merge_dense_delta` errors on a
    // quantized base — so the marker is what lets the distilled fast defaults ride a packed tier. When
    // there is no marker (a dense base snapshot), merge at load as before. The merge MUST precede
    // quantization; assert full coverage (`7·layers` gen-path projections + the 2 FM-head Linears) so
    // a stale/mismatched LoRA fails loudly rather than silently merging a subset. (Pointing the fast
    // id at a *packed base* tier — no marker — stays a loud `merge_dense_delta` "base is quantized"
    // error, never a silent double-merge or half-load.)
    if let Some(lora_path) = distill_lora_path {
        let lora = Weights::from_file(&lora_path)?;
        let applied = model.merge_distill_lora(&lora)?;
        let expected = cfg.llm.num_hidden_layers * 7 + 2;
        if applied != expected {
            return Err(Error::Msg(format!(
                "{id}: distill LoRA merged {applied} targets, expected {expected} \
                 (7·{} gen-path linears + 2 fm_head) — wrong LoRA file?",
                cfg.llm.num_hidden_layers
            )));
        }
    }
    // Q4/Q8 quantize the backbone decoder stack after the dense load (sc-3193). For the fast variant
    // the distill LoRA is already merged, so quantization sees the distilled weights.
    if let Some(q) = spec.quantize {
        model.quantize(q.bits())?;
    }
    let tokenizer = load_tokenizer(root)?;
    Ok(Box::new(SenseNova {
        descriptor: descriptor_for(id),
        tokenizer,
        model,
        fast,
        memory_strategy: crate::memory_strategy::memory_strategy_contract_with_artifact(
            id,
            spec,
            pinned_artifact.as_ref(),
        )?,
        loaded_quant: spec.quantize,
        pinned_artifact,
    }))
}

impl SenseNova {
    /// Collect the reference images (`Reference` + `MultiReference`) as preprocessed
    /// `[3,H,W]`-in-`[0,1]` arrays for [`T2iModel::it2i_generate`]. Empty ⇒ T2I.
    fn references(&self, req: &GenerationRequest) -> Result<Vec<Array>> {
        let mut out = Vec::new();
        for c in &req.conditioning {
            match c {
                Conditioning::Reference { image, .. } => out.push(image_to_chw01(image)?),
                Conditioning::MultiReference { images } => {
                    for image in images {
                        out.push(image_to_chw01(image)?);
                    }
                }
                _ => {}
            }
        }
        Ok(out)
    }

    fn options(&self, req: &GenerationRequest, seed: u64) -> T2iOptions {
        // The distilled variant defaults to 8 NFE at CFG 1.0; the base to 50 NFE at CFG 4.0. An
        // explicit request value always wins.
        let (def_steps, def_guidance) = if self.fast {
            (DEFAULT_STEPS_FAST, DEFAULT_GUIDANCE_FAST)
        } else {
            (DEFAULT_STEPS, DEFAULT_GUIDANCE)
        };
        T2iOptions {
            cfg_scale: req.guidance.unwrap_or(def_guidance),
            img_cfg_scale: image_cfg_scale(req),
            num_steps: req.steps.unwrap_or(def_steps) as usize,
            timestep_shift: req.scheduler_shift.unwrap_or(DEFAULT_TIMESTEP_SHIFT),
            seed,
            attention_score_budget: req
                .memory
                .filter(|memory| memory.chunk_attention)
                .and_then(|memory| memory.attention_chunk_size),
            transformer_window_size: req
                .memory
                .filter(|memory| memory.stream_transformer_blocks)
                .and_then(|memory| memory.transformer_window_size),
            calibration_stream_fault: req.memory.is_some_and(|memory| {
                memory.calibration_fault_harness_authorized
                    && memory.calibration_error_phase == Some(gen_core::MemoryPhase::Denoise)
            }),
            ..Default::default()
        }
    }
}

impl Generator for SenseNova {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Use the descriptor's own id so the base and `_fast` variants attribute rejections to the
        // right model (F-143).
        let id = self.descriptor.id;
        self.descriptor.capabilities.validate_request(id, req)?;
        validate_dims_and_steps(id, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        guard_pinned_artifact(self.pinned_artifact.as_ref(), || {
            self.generate_impl(req, on_progress)
        })
        .map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.memory_strategy, self.loaded_quant, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        crate::memory_strategy::begin_request(
            self.descriptor.id,
            &self.memory_strategy,
            self.loaded_quant,
            context,
            mlx_gen::request_scope::MlxScopeCleanup::Device,
        )
    }
}

impl SenseNova {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the
    /// family helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`]
    /// (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let references = self.references(req)?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let (w, h) = (req.width as i32, req.height as i32);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            // Check the worker's cancel flag between images too (a 50-step 8B run is multi-minute;
            // the per-step check lives in the denoise loop via the StepReporter). F-128.
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let opts = self.options(req, base_seed.wrapping_add(i as u64));
            // Thread cancellation + per-step progress into the denoise loop. Progress now reports the
            // denoise step (Kolors/SDXL semantics), not the image index as the old single tick did.
            let reporter = StepReporter::new(&req.cancel, on_progress);
            let out = if references.is_empty() {
                self.model.generate(
                    &self.tokenizer,
                    &req.prompt,
                    w,
                    h,
                    &opts,
                    None,
                    Some(reporter),
                )?
            } else {
                self.model.it2i_generate(
                    &self.tokenizer,
                    &req.prompt,
                    &references,
                    w,
                    h,
                    &opts,
                    None,
                    Some(reporter),
                )?
            };
            images.push(decoded_to_image(&out.image)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Request-boundary checks beyond the capability surface: reference-only image guidance, 32-pixel
/// alignment per side, and a positive step count. Factored out so it can be unit-tested without
/// loaded weights. `id` is the rejecting model's descriptor id (base or `_fast`) so the error
/// attributes to the right variant (F-143).
fn validate_dims_and_steps(id: &str, req: &GenerationRequest) -> Result<()> {
    if req.true_cfg.is_some()
        && !req
            .conditioning
            .iter()
            .any(|conditioning| match conditioning {
                Conditioning::Reference { .. } => true,
                Conditioning::MultiReference { images } => !images.is_empty(),
                _ => false,
            })
    {
        return Err(Error::Unsupported(format!(
            "{id}: true_cfg is image guidance and requires Reference or non-empty MultiReference \
             conditioning"
        )));
    }
    if !req.width.is_multiple_of(CELL) || !req.height.is_multiple_of(CELL) {
        return Err(Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {CELL} per side",
            req.width, req.height
        )));
    }
    // `steps == 0` builds an empty denoise trajectory, so `generate`/`it2i_generate`/`interleave_gen`
    // panic on `.last().expect("at least one step")` (F-125). Reject it at the boundary; `None` falls
    // back to the variant default.
    if req.steps == Some(0) {
        return Err(Error::Msg(format!("{id}: steps must be >= 1")));
    }
    Ok(())
}

/// Decode an [`Image`] (RGB8 HWC) to a `[3,H,W]` f32 tensor in `[0,1]`, smart-resized to a
/// 32-aligned bucket within `[512², 2048²]` pixels (the reference `load_image_native`).
fn image_to_chw01(img: &Image) -> Result<Array> {
    let (in_w, in_h) = (img.width as i32, img.height as i32);
    let (out_h, out_w) = smart_resize(in_h, in_w, CELL as i32, REF_MIN_PIXELS, REF_MAX_PIXELS);
    // Resize (bicubic, PIL-faithful) → f32 HWC in [0,255].
    let hwc = resize_bicubic_u8(
        &img.pixels,
        in_h as usize,
        in_w as usize,
        out_h as usize,
        out_w as usize,
    )?;
    let hwc = Array::from_slice(&hwc, &[out_h, out_w, 3]);
    let chw = hwc.transpose_axes(&[2, 0, 1])?; // HWC → CHW
    divide(&chw, Array::from_f32(255.0)).map_err(Error::from)
}

// The registration constants bridge the crate's rich `Result` into backend-neutral
// `gen_core::Result`. The 8-step
// distilled variant (sc-3192) registers under `descriptor_fast`. `impl Generator` stays
// hand-written because `validate` attributes rejections to the per-variant descriptor id (F-143).
/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate. SenseNova-U1 (NEO-Unify) is a single
/// FLAT sharded checkpoint: the text, vision, and generation paths are INTERLEAVED in the same tensors
/// (the dual-path `_mot_gen` layout — see [`crate::loader`]), so there is NO separable text encoder to
/// stage. Report the whole checkpoint as the heavy component (`text_encoder = 0`) — an honest
/// "`Sequential` residency buys nothing here" (staged peak == resident peak) rather than a fabricated
/// split. sensenova's descriptor sets `supports_sequential_offload: false` — the capability bit the
/// worker's fit-gate keys on — so the sequential path is never taken; this only makes that explicit.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let root = match &spec.weights {
        mlx_gen::WeightsSource::Dir(p) => p.clone(),
        mlx_gen::WeightsSource::File(_) => {
            return Err(mlx_gen::gen_core::Error::Msg(
                "sensenova footprint requires a snapshot directory".to_owned(),
            ))
        }
    };
    Ok(mlx_gen::PerComponentBytes {
        text_encoder: 0,
        dit: mlx_gen::safetensors_dir_bytes(&root),
        vae: 0,
    })
}

mlx_gen::register_generators! {
    pub(crate) const QUALITY_REGISTRATION = descriptor => load;
    footprint = component_footprint
}

pub const QUALITY_MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID,
        contract: |spec| crate::memory_strategy::memory_strategy_contract(MODEL_ID, spec),
        safety_check: crate::memory_strategy::registered_safety_check,
    };

pub const FAST_MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID_FAST,
        contract: |spec| crate::memory_strategy::memory_strategy_contract(MODEL_ID_FAST, spec),
        safety_check: crate::memory_strategy::registered_safety_check,
    };

pub const QUALITY_MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(MODEL_ID, spec, contract, context)
        },
    };

pub const FAST_MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID_FAST,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(MODEL_ID_FAST, spec, contract, context)
        },
    };
mlx_gen::register_generators! {
    pub(crate) const FAST_REGISTRATION = descriptor_fast => load_fast;
    footprint = component_footprint
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_guard_reports_post_materialization_mutation_even_after_operation_error() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let path = root.join("model.safetensors");
        std::fs::write(&path, [0_u8; 8]).unwrap();
        let artifact = crate::memory_strategy::PinnedArtifact::verify_file(&path).unwrap();
        let result: Result<()> = guard_pinned_artifact(Some(&artifact), || {
            let replacement = root.join("replacement.safetensors");
            std::fs::write(&replacement, [1_u8; 8]).unwrap();
            std::fs::rename(replacement, &path).unwrap();
            Err(Error::Msg("earlier generation failure".to_owned()))
        });
        let error = result.unwrap_err().to_string();
        assert!(error.contains("replaced or mutated"), "got: {error}");
        assert!(
            !error.contains("earlier generation failure"),
            "got: {error}"
        );
    }

    #[test]
    fn descriptor_is_sensenova() {
        let d = descriptor();
        assert_eq!(d.id, "sensenova_u1_8b");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_true_cfg);
    }

    #[test]
    fn true_cfg_is_reference_image_guidance_not_text_cfg() {
        let request = GenerationRequest {
            prompt: "turn this into a watercolor".into(),
            true_cfg: Some(1.75),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        assert_eq!(
            image_cfg_scale(&request),
            1.75,
            "request.true_cfg must reach T2iOptions.img_cfg_scale verbatim"
        );
        assert!(validate_dims_and_steps(MODEL_ID, &request).is_ok());

        let no_reference = GenerationRequest {
            conditioning: vec![],
            ..request.clone()
        };
        let error = validate_dims_and_steps(MODEL_ID, &no_reference)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("requires Reference"),
            "true_cfg on txt2img would otherwise be silently ignored: {error}"
        );

        let empty_multi = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference { images: vec![] }],
            ..request
        };
        assert!(
            validate_dims_and_steps(MODEL_ID, &empty_multi).is_err(),
            "an empty MultiReference does not create the image-guidance branch"
        );
        assert_eq!(
            image_cfg_scale(&GenerationRequest::default()),
            1.0,
            "unset true_cfg preserves the reference path's neutral image-guidance scale"
        );
    }

    #[test]
    fn descriptor_fast_differs_only_in_id() {
        let base = descriptor();
        let fast = descriptor_fast();
        assert_eq!(fast.id, "sensenova_u1_8b_fast");
        assert_ne!(fast.id, base.id);
        // Same capability surface as the base — only the id (and the generation defaults) differ.
        assert_eq!(fast.family, base.family);
        assert_eq!(fast.modality, base.modality);
        assert_eq!(
            fast.capabilities.supports_guidance,
            base.capabilities.supports_guidance
        );
        assert_eq!(
            fast.capabilities.supports_true_cfg,
            base.capabilities.supports_true_cfg
        );
        assert!(fast.capabilities.accepts(ConditioningKind::Reference));
        assert!(fast.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(!fast.capabilities.supports_lora);
        assert_eq!(fast.capabilities.max_size, base.capabilities.max_size);
    }

    #[test]
    fn registered_in_registry() {
        // The explicit family catalog contains both ids.
        let ids: Vec<&str> = crate::provider_registry()
            .unwrap()
            .generators()
            .copied()
            .map(|r| (r.descriptor)().id)
            .collect();
        assert!(ids.contains(&MODEL_ID), "{MODEL_ID} not registered");
        assert!(
            ids.contains(&MODEL_ID_FAST),
            "{MODEL_ID_FAST} not registered"
        );
    }

    #[test]
    fn load_rejects_single_file() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        assert!(load(&spec).is_err());
        // The fast loader rejects a single file the same way (before touching the LoRA).
        assert!(load_fast(&spec).is_err());
    }

    /// sc-13664: the fast loader resolves its distill LoRA from caller-supplied paths only (the
    /// `distill_lora` component, else the co-located snapshot file) — no `$SENSENOVA_DISTILL_LORA`, no
    /// HF-cache scan. A dense-base fast load with neither staged fails **at load** (fail-fast, before
    /// the config/weights I/O) with an actionable error naming the component. Weights-free:
    /// `/nonexistent` has no marker and no co-located LoRA.
    #[test]
    fn fast_load_requires_distill_lora_path() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load_fast(&spec).err().expect("err").to_string();
        assert!(err.contains("distill_lora"), "got: {err}");
        assert!(!err.contains("SENSENOVA_DISTILL_LORA"), "got: {err}");
    }

    /// sc-13664: a staged-but-nonexistent `distill_lora` component errors at load naming the component
    /// (not a bare later I/O error); a `Dir` staged where a `.safetensors` file is required is rejected.
    #[test]
    fn fast_load_validates_distill_lora_component() {
        let missing = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_component(
            "distill_lora",
            WeightsSource::File("/nope/lora.safetensors".into()),
        );
        let err = load_fast(&missing).err().expect("err").to_string();
        assert!(err.contains("distill_lora"), "got: {err}");
        assert!(err.contains("does not exist"), "got: {err}");

        let as_dir = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_component("distill_lora", WeightsSource::Dir("/some/dir".into()));
        let err = load_fast(&as_dir).err().expect("err").to_string();
        assert!(err.contains("distill_lora"), "got: {err}");
    }

    /// sc-13658/sc-13664: an unrecognized component key is rejected at load (typed `Unsupported`).
    #[test]
    fn fast_load_rejects_unknown_component() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_component(
            "bogus_component",
            WeightsSource::File("/x.safetensors".into()),
        );
        assert!(matches!(
            load_fast(&spec).err().expect("err"),
            Error::Unsupported(_)
        ));
    }

    #[test]
    fn both_loaders_reject_user_adapters() {
        // `supports_lora=false` on both ids; the distill LoRA is merged internally by `load_fast`,
        // never supplied via `spec.adapters`.
        let mut spec = LoadSpec::new(WeightsSource::Dir("/tmp/does-not-exist".into()));
        spec.adapters = vec![mlx_gen::AdapterSpec::new(
            "/tmp/some.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        )];
        let msg = |r: Result<Box<dyn Generator>>| match r {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error rejecting adapters"),
        };
        assert!(msg(load(&spec)).contains("adapters"));
        assert!(msg(load_fast(&spec)).contains("adapters"));
    }

    #[test]
    fn validate_rejects_unaligned_size() {
        let d = descriptor();
        let req = GenerationRequest {
            width: 300,
            height: 256,
            ..Default::default()
        };
        // Capability floor passes (in range) but the 32-alignment check rejects 300.
        assert!(d.capabilities.validate_request(MODEL_ID, &req).is_ok());
        assert!(!300u32.is_multiple_of(CELL));
        let err = validate_dims_and_steps(MODEL_ID, &req)
            .unwrap_err()
            .to_string();
        assert!(err.contains("multiple of"), "got: {err}");
        // F-143: the rejecting model's own id is in the message, so the fast variant attributes the
        // error to `sensenova_u1_8b_fast`, not the hardcoded base id.
        let fast_err = validate_dims_and_steps(MODEL_ID_FAST, &req)
            .unwrap_err()
            .to_string();
        assert!(
            fast_err.contains(MODEL_ID_FAST),
            "fast id should appear: {fast_err}"
        );

        // sc-12612: `CELL` is the pinned stride SceneWorks ties every advertised SenseNova bucket to.
        // Pin the value and mutation-check that a size which is a multiple of 16 but not CELL (32) is
        // still rejected with the stride error, and an on-stride size passes.
        assert_eq!(CELL, 32);
        let off_stride = validate_dims_and_steps(
            MODEL_ID,
            &GenerationRequest {
                width: 1040, // 65×16 — a multiple of 16 but not CELL
                height: 512,
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            off_stride.contains("multiple of 32"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(validate_dims_and_steps(
            MODEL_ID,
            &GenerationRequest {
                width: 1024, // 32×32 — on-stride
                height: 512,
                ..Default::default()
            },
        )
        .is_ok());
    }

    #[test]
    fn validate_rejects_zero_steps() {
        // F-125: `steps == 0` builds an empty denoise trajectory → `.expect("at least one step")`
        // panic. Reject at the boundary; `None` and any positive count pass.
        let bad = GenerationRequest {
            width: 512,
            height: 512,
            steps: Some(0),
            ..Default::default()
        };
        let err = validate_dims_and_steps(MODEL_ID, &bad)
            .unwrap_err()
            .to_string();
        assert!(err.contains("steps must be >= 1"), "got: {err}");

        for steps in [None, Some(1), Some(50)] {
            let ok = GenerationRequest {
                width: 512,
                height: 512,
                steps,
                ..Default::default()
            };
            assert!(
                validate_dims_and_steps(MODEL_ID, &ok).is_ok(),
                "steps={steps:?}"
            );
        }
    }
}
