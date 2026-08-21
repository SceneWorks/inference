//! `Krea` — the [`mlx_gen::Generator`] implementation for Krea 2 Turbo, plus its [`descriptor`] /
//! [`load`] entry points and explicit registrations exposed through the family catalog.
//!
//! **Status (P1 complete):** the provider crate + `krea_2_turbo` registration + architecture-validated
//! [`load`] + offline Q4/Q8 converter ([`crate::convert`]) landed in sc-7567; the DiT forward in
//! sc-7568 ([`crate::transformer`]); the Qwen3-VL-4B text encoder in sc-7569 ([`crate::text_encoder`]);
//! the VAE + rectified-flow sampler in sc-7570 ([`crate::vae`] / [`crate::schedule`]); and the
//! end-to-end Turbo t2i [`crate::pipeline`] in sc-7571. [`Krea::generate`] now renders real images
//! (CFG-free, few-step) through the assembled tokenizer → TE → DiT → VAE pipeline.

use mlx_gen::img2img::init_time_step;
use mlx_gen::media::Image;
use mlx_gen::{
    advanced_pass_scheduler_names, curated_sampler_names, curated_scheduler_names, default_seed,
    AdapterSpec, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, LatentDecoder, LoadSpec, Modality, ModelDescriptor, Precision,
    Progress, Quant, Residency, Result, SizeFloor, WeightsSource, BASE_SNAPSHOT_COMPONENT,
    VAE_COMPONENT,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use mlx_gen_qwen_image::pipeline::PID_BACKBONE;
use mlx_gen_wan::OwnedWanSingleFrameDecoder;

use mlx_rs::Array;
use std::path::Path;

use crate::multiphase::{self, ResolvedPhase};
use crate::pipeline::{
    base_schedule, maybe_apply_style_gain, turbo_schedule, EditPlan, Img2ImgPlan, KreaHeavy,
    KreaText, T2iPlan, TurboOptions,
};

/// Registry id for the Krea 2 Turbo text-to-image variant. Matches the SceneWorks worker's
/// `payload.model` and the manifest `engine_id` (sc-7572).
pub const KREA_2_TURBO_ID: &str = "krea_2_turbo";

/// Qwen3-VL-4B conditioning architecture shared by every Krea 2 route.
pub const TOKENIZER_CONTRACT: mlx_gen::gen_core::EncoderTokenizerContract =
    mlx_gen::gen_core::EncoderTokenizerContract {
        family: "qwen3_vl",
        binding: mlx_gen::gen_core::EncoderTokenizerBinding::RetainBase,
        artifact_candidates: &["tokenizer/tokenizer.json"],
        required_tokens: &[
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_endoftext",
                literal: "<|endoftext|>",
                id: 151_643,
                config_field: Some("bos_token_id"),
            },
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_im_start",
                literal: "<|im_start|>",
                id: 151_644,
                config_field: None,
            },
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_im_end",
                literal: "<|im_end|>",
                id: 151_645,
                config_field: Some("eos_token_id"),
            },
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_vision_start",
                literal: "<|vision_start|>",
                id: 151_652,
                config_field: Some("vision_start_token_id"),
            },
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_vision_end",
                literal: "<|vision_end|>",
                id: 151_653,
                config_field: Some("vision_end_token_id"),
            },
            mlx_gen::gen_core::EncoderRequiredToken {
                role: "qwen_image_pad",
                literal: "<|image_pad|>",
                id: 151_655,
                config_field: Some("image_token_id"),
            },
        ],
    };
pub const PROMPT_EXECUTIONS: &[mlx_gen::gen_core::EncoderPromptExecutionContract] = &[
    mlx_gen::gen_core::EncoderPromptExecutionContract {
        purpose: "krea_t2i",
        template: mlx_gen::gen_core::EncoderPromptTemplate::KreaQwen3Vl,
        add_special_tokens: false,
        length: mlx_gen::gen_core::EncoderPromptLengthPolicy::Unbounded,
        padding: mlx_gen::gen_core::EncoderPromptPadding::None,
        prefix_trim: 34,
    },
    mlx_gen::gen_core::EncoderPromptExecutionContract {
        purpose: "krea_edit",
        template: mlx_gen::gen_core::EncoderPromptTemplate::KreaQwen3VlEdit,
        add_special_tokens: false,
        length: mlx_gen::gen_core::EncoderPromptLengthPolicy::Unbounded,
        padding: mlx_gen::gen_core::EncoderPromptPadding::None,
        prefix_trim: 34,
    },
];

pub const ENCODER_CONTRACT: mlx_gen::gen_core::EncoderContract =
    mlx_gen::gen_core::EncoderContract {
        architecture: "qwen3_vl_text",
        hidden_size: 2560,
        intermediate_size: 9728,
        num_hidden_layers: 36,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 151_936,
        output_width: 2560,
        loaded_hidden_layers: 35,
        requires_final_norm: false,
        requires_lm_head: false,
        hidden_activation: "silu",
        attention_dropout: mlx_gen::gen_core::EncoderConfigFloat::new(0.0),
        rms_norm_eps: mlx_gen::gen_core::EncoderConfigFloat::new(1e-6),
        qk_norm_eps: Some(mlx_gen::gen_core::EncoderConfigFloat::new(1e-6)),
        rope_theta: mlx_gen::gen_core::EncoderConfigFloat::new(5_000_000.0),
        max_position_embeddings: 262_144,
        attention_bias: mlx_gen::gen_core::EncoderConfigBool::Required(false),
        tie_word_embeddings: mlx_gen::gen_core::EncoderConfigBool::Required(true),
        tokenizer: TOKENIZER_CONTRACT,
        prompt_executions: PROMPT_EXECUTIONS,
        bos_token_id: Some(151_643),
        eos_token_id: Some(151_645),
        image_token_id: Some(151_655),
        vision_start_token_id: Some(151_652),
        vision_end_token_id: Some(151_653),
        mrope_section: &[24, 20, 20],
        mrope_interleaved: Some(true),
        selected_hidden_layers: &[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
        packing: Some(mlx_gen::gen_core::EncoderPackingContract {
            group_size: 64,
            pack_embedding: false,
            pack_lm_head: false,
            supports_file: true,
        }),
        dense_storage_dtype_probe: None,
    };

pub const VISION_ENCODER_CONTRACT: mlx_gen::gen_core::VisionEncoderContract =
    mlx_gen::gen_core::VisionEncoderContract {
        architecture: mlx_gen::gen_core::VisionEncoderArchitecture::Qwen3Vl,
        hidden_size: 1024,
        intermediate_size: 4096,
        num_hidden_layers: 24,
        num_attention_heads: 16,
        output_width: 2560,
        hidden_activation: "gelu_pytorch_tanh",
        rope_theta: mlx_gen::gen_core::EncoderConfigFloat::new(10_000.0),
        normalization_eps: mlx_gen::gen_core::EncoderConfigFloat::new(1e-6),
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        in_channels: 3,
        num_position_embeddings: Some(2304),
        deepstack_visual_indexes: &[5, 11, 17],
        window_size: None,
        full_attention_block_indexes: &[],
    };

/// Max images per request (the image-model standard, shared with the other MLX families).
const MAX_COUNT: u32 = 8;
/// Resolution bounds (W/H). Turbo renders up to 2048²; the catalog/worker gate the UI options tighter.
const RES_MIN: u32 = 256;
/// `pub(crate)` so the pose-control lane's load-time branch-quant gate (sc-11748) can size its
/// worst-case-resolution estimate against the largest render the model can serve.
pub(crate) const RES_MAX: u32 = 2048;
/// patch_size(2)·vae_downsample(8) = 16 — patchify requires W/H divisible by this. Exposed as the
/// pinned-engine stride SceneWorks ties each advertised Krea image bucket to (sc-12612), mirroring
/// `wan::config::SIZE_MULTIPLE_14B`. `validate_request` enforces exactly this value, so the const
/// cannot drift from the check.
pub const RES_MULTIPLE: u32 = 16;

/// Turbo defaults: the TDM-distilled few-step student renders CFG-free at 8 steps (reference
/// `is_distilled` + `guidance_scale 0`). Consumed by `generate` (`req.steps.unwrap_or(DEFAULT_STEPS)`);
/// the manifest `default_steps` mirrors this (sc-7572).
const DEFAULT_STEPS: u32 = 8;

/// Registry id for the undistilled **Raw** text-to-image variant (epic 9992). The SAME string as the
/// Krea LoRA *trainer* base ([`crate::training::KREA_2_RAW_TRAINER_ID`]) — Path 1 makes one id both the
/// training base and a first-class generator; the trainer + generator live in separate registries so
/// the shared id never collides. Matches the SceneWorks worker's `payload.model` + manifest `engine_id`.
pub const KREA_2_RAW_ID: &str = "krea_2_raw";

/// Raw defaults (the reference `sampling.py` Raw preset per the sc-7566 spike): full-CFG at 52 steps,
/// guidance 3.5, resolution-dynamic mu. Consumed by `generate_impl`
/// (`req.steps.unwrap_or(DEFAULT_RAW_STEPS)` / `req.guidance.unwrap_or(DEFAULT_RAW_GUIDANCE)`); the
/// manifest `default_steps` / `defaults.guidanceScale` mirror these (sc-9999 / sc-10003).
const DEFAULT_RAW_STEPS: u32 = 52;
const DEFAULT_RAW_GUIDANCE: f32 = 3.5;
/// img2img reference fidelity default when neither the `Reference`'s own `strength` nor the
/// request-level `strength` is set — the full-range slider's midpoint (epic 8588 A2/A3).
const DEFAULT_IMG2IMG_STRENGTH: f32 = 0.5;

/// Registry id for the **image-edit** variant (epic 10871). The Kontext-style edit surface shares the
/// undistilled Raw pipeline (full-CFG, denoise-from-noise) but routes a single `Reference` — the SOURCE
/// image — through [`crate::pipeline::KreaPipeline::generate_edit_with_progress`] (in-context VAE tokens + Qwen3-VL
/// grounding) instead of the img2img latent-init. A DISTINCT engine id (the Qwen-Image-Edit /
/// FLUX.2-Klein-Edit pattern) is what disambiguates edit from img2img: the SAME source `Reference` means
/// "edit" or "img2img" purely by which generator the worker loaded. The community `krea2_identity_edit`
/// LoRA rides `spec.adapters`.
pub const KREA_2_EDIT_ID: &str = "krea_2_edit";

/// Registry id for the **CFG-free Turbo image-edit** variant (sc-11640, follow-on to epic 10871). Same
/// Kontext edit surface as [`KREA_2_EDIT_ID`] — a source image (or scene+person pair) drives the dual
/// conditioning (in-context VAE tokens + Qwen3-VL grounding) through
/// [`crate::pipeline::KreaHeavy::render_edit`] — but on the **distilled Turbo** checkpoint: the
/// few-step `turbo_schedule` run **CFG-free** (`guidance = 0`, a single conditional forward, no cond/uncond
/// split), the fast-path alternative to the ~52-step full-CFG Raw edit. The `krea2_identity_edit` LoRA
/// (trained on the Raw DiT, family-compatible with Turbo) folds in via `spec.adapters` exactly as on
/// Raw. A DISTINCT id so the worker's edit lane can select the fast tier by model, the same way
/// `krea_2_edit` disambiguates edit from img2img.
pub const KREA_2_TURBO_EDIT_ID: &str = "krea_2_turbo_edit";

/// Krea 2 Turbo identity + capabilities — constructible without loading weights (registry
/// introspection / capability advertisement). Distilled few-step text-to-image: **CFG-free** (the TDM
/// distillation baked the guided velocity into the weights, so no unconditional branch / `guidance`),
/// no user negative prompt, no img2img/control conditioning on the Turbo checkpoint.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: Some(ENCODER_CONTRACT),
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::QWEN_KREA_Z16_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: KREA_2_TURBO_ID,
        family: "krea_2",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // CFG-free distilled student (like Ideogram Turbo / Boogu Turbo / SDXL-Lightning).
            supports_guidance: false,
            supports_true_cfg: false,
            // Reference-image conditioning = img2img latent-init (epic 8588 slice A, sc-10135): a single
            // `Conditioning::Reference { image, strength }` seeds the denoise from the VAE-encoded
            // reference (see [`generate_impl`] → `generate_turbo_img2img_with_progress`). Turbo only; the
            // Raw descriptor clears this (no Raw img2img entrypoint yet).
            conditioning: vec![ConditioningKind::Reference],
            // LoRA/LoKr trained on the undistilled Raw DiT (sc-7577) apply at Turbo inference via the
            // shared `apply_adapters_strict_with_diff_patch` seam onto the `Krea2Transformer` adapter
            // host (sc-7911; the seam also folds a ComfyUI `.diff`/`.diff_b` diff-patch, sc-13825).
            // Family-match cross-apply, no base-model gating (the Lens / Z-Image precedent).
            supports_lora: true,
            supports_lokr: true,
            // Rectified-flow v-param over the unified curated-sampler framework (epic 7114). The
            // distilled-coherent sampler subset is narrowed by the real-weight survey at e2e (sc-7571,
            // the Boogu Turbo precedent); the scaffold advertises the full curated menu as a starting
            // point. The native distilled loop stays the byte-exact default (`req.sampler == None`).
            samplers: curated_sampler_names(),
            // Krea 2 is epic 20414's first acceptance target, so it is the one family validated for
            // the GATED advanced multi-pass schedules (`linear_quadratic` / `bong_tangent`,
            // sc-20416) on top of the curated eight — the candle twin advertises the same pair.
            // Every route here resolves its schedule through `mlx_gen::resolve_flow_schedule` ->
            // `gen_core::sampling::schedule_sigmas`, so the advertisement and the honoring are the
            // same code path. Families that have NOT been validated keep the bare
            // `curated_scheduler_names()` menu — that is the gate.
            schedulers: [
                curated_scheduler_names(),
                advanced_pass_scheduler_names(),
                // `flow_match` is the honored NATIVE alias (resolves to the byte-exact native
                // exponential-mu schedule through the N3 fallback in `resolve_flow_schedule`).
                // Advertised since sc-20418 because the chained denoise-pass resolution ladder
                // needs an explicit, menu-valid id for "the model's default schedule" — a resolved
                // plan naming it must replay through validation. The candle twin advertises it too.
                vec!["flow_match"],
            ]
            .concat(),
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: true,
            // The turnkey ships pre-packed Q8/Q4 ([`crate::convert::assemble_quantized_snapshot`]);
            // load-time quantize over a dense bf16 build is a no-op on an already-packed snapshot.
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // Wired onto the shared `Residency` seam; honors Sequential offload (F-176).
            supports_sequential_offload: true,
            unconditionally_engages_staged_residency: false,
            supports_preview: true,
            supports_prompt_enhancement: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            // Chained denoise passes (epic 20414, sc-20418): Krea 2 is the first wired family —
            // Turbo here and Raw (which derives from this descriptor). The executor is the shared
            // `gen_core::sampling::pass_executor`, driven by `KreaHeavy::render_denoise_passes`;
            // the lockstep rule applies (never advertise ahead of the wiring), and the edit
            // variants explicitly opt back OUT below (grounded edit conditioning is not in the
            // t2i-from-noise v1 surface). The candle twin advertises identically.
            supports_denoise_passes: true,
            // What a pass may actually name here, beyond the on/off bit above (sc-20425). The candle
            // twin declares the same pair.
            //
            // `flow_match` is the advertised NATIVE scheduler alias — the byte-exact native
            // exponential-mu schedule the resolution ladder bottoms out on, honored by
            // `KreaPassHost::build_schedule` through the family's own resolver and by nothing in the
            // curated registry. Declaring it keeps a resolved plan naming it replayable through
            // validation, while an *undeclared* native id stays a typed rejection.
            //
            // Per-pass adapter overrides are real here: `prepare_denoise_passes` builds one
            // job-local DiT clone per pass over that pass's re-scaled stack, so "adapter off for
            // pass 1, on for pass 2" genuinely renders that way. That is the exception, not the
            // rule — see `DenoisePassSurface::per_pass_adapters`.
            denoise_pass_surface: mlx_gen::gen_core::DenoisePassSurface {
                native_schedulers: &["flow_match"],
                per_pass_adapters: true,
            },
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
            execution: Default::default(),
            approximation: Default::default(),
        },
    }
}

/// Krea 2 **Raw** identity + capabilities — the undistilled 12B DiT run with **true classifier-free
/// guidance** (two DiT forwards/step: cond vs uncond) at 52 steps, unlike the CFG-free distilled Turbo.
/// Same architecture / snapshot layout as Turbo (only the DiT weights differ, distilled vs base), so it
/// shares `load_variant` + the whole [`crate::pipeline::KreaPipeline`]. Exposes a real guidance scale AND a user
/// negative prompt — the reference `sample()` accepts `negative_prompts` (richer than Boogu's base,
/// which fixes the uncond to the empty prompt). NOT guidance-distilled, so `supports_true_cfg` stays
/// false: there is no separate embedded-guidance axis to layer a `true_cfg_scale` over — the two-forward
/// CFG IS the guidance (the Boogu-base precedent). Derived from [`descriptor`] so the shared surface
/// (family/backend/samplers/quants/size/LoRA) stays in lockstep.
pub fn raw_descriptor() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = KREA_2_RAW_ID;
    d.capabilities.supports_negative_prompt = true;
    d.capabilities.supports_guidance = true;
    d.capabilities.supports_true_cfg = false;
    // img2img reference latent-init (epic 8588 slice A, sc-10224): Raw advertises `Reference` just like
    // Turbo, but routes to the CFG entrypoint `generate_base_img2img_with_progress` (honoring guidance +
    // negative prompt), NOT the CFG-free Turbo one. Inherited from `descriptor()` — same single-Reference
    // surface — so this is a no-op re-affirmation kept explicit for the reader.
    d.capabilities.conditioning = vec![ConditioningKind::Reference];
    d
}

/// Krea 2 **image-edit** identity + capabilities (epic 10871). Same full-CFG surface as
/// [`raw_descriptor`] — an edit denoises from noise under true CFG, honoring guidance + a negative
/// prompt — but with the distinct [`KREA_2_EDIT_ID`] so the worker's edit lane can select it. Carries the
/// single-`Reference` (source) conditioning + LoRA/LoKr (the `krea2_identity_edit` edit LoRA). Derived
/// from [`raw_descriptor`] so the shared surface (family/backend/samplers/quants/size/CFG) stays in
/// lockstep; only the id (→ the `generate_impl` edit branch) differs.
pub fn edit_descriptor() -> ModelDescriptor {
    let mut d = raw_descriptor();
    d.id = KREA_2_EDIT_ID;
    // Edit accepts a single source (`Reference`) OR a scene+person pair (`MultiReference`, epic 10871
    // P1.3 — scene = image 1, person = image 2, fixed order). The img2img Raw/Turbo descriptors stay
    // single-`Reference`; only the edit surface advertises `MultiReference`, so `validate_request`
    // accepts a two-source edit here while still rejecting it on the img2img path.
    d.capabilities.conditioning = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    // Chained denoise passes are wired for the t2i-from-noise Turbo/Raw variants only (sc-20418):
    // the grounded edit conditioning path is out of the v1 surface, so the derived edit descriptor
    // must not inherit the advertisement. BOTH halves are cleared — a surface without the capability
    // is rejected by the descriptor conformance sweep (sc-20425), because half-inherited is exactly
    // how `model_control`'s descriptor came to advertise a chain it cannot run.
    d.capabilities.supports_denoise_passes = false;
    d.capabilities.denoise_pass_surface = mlx_gen::gen_core::DenoisePassSurface::NONE;
    d
}

/// Krea 2 **CFG-free Turbo image-edit** identity + capabilities (sc-11640). Same Kontext edit
/// conditioning surface as [`edit_descriptor`] (single `Reference` source OR a scene+person
/// `MultiReference`, + the `krea2_identity_edit` LoRA) but derived from the distilled Turbo
/// [`descriptor`] rather than Raw: **CFG-free** (`supports_guidance = false`, no user negative prompt),
/// so the edit runs a single conditional forward on the few-step `turbo_schedule`. Only the id (→ the
/// `generate_impl` `is_turbo_edit` branch: `turbo_schedule` / 8-step default / `guidance = 0`) and the
/// widened `conditioning` differ from [`descriptor`].
pub fn turbo_edit_descriptor() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = KREA_2_TURBO_EDIT_ID;
    // Same edit conditioning surface as `edit_descriptor` — a single source `Reference` or one
    // scene+person `MultiReference`. The Turbo img2img descriptor stays single-`Reference`; only the
    // edit surfaces advertise `MultiReference`.
    d.capabilities.conditioning = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    // Chained denoise passes stay off the edit surfaces (sc-20418) — see `edit_descriptor`. Both
    // halves, for the reason recorded there (sc-20425).
    d.capabilities.supports_denoise_passes = false;
    d.capabilities.denoise_pass_surface = mlx_gen::gen_core::DenoisePassSurface::NONE;
    d
}

/// A loaded Krea 2 generator (Turbo, Raw, or edit): the cached descriptor + a component-residency
/// strategy. The variant is read back off `descriptor.id` at generate time (Turbo = CFG-free distilled;
/// Raw = full-CFG undistilled; edit = the Raw pipeline routed to the Kontext edit entrypoint).
pub struct Krea {
    descriptor: ModelDescriptor,
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    precision: Precision,
    quant: Option<Quant>,
    streamable_transformer: bool,
    /// The constructor-time pin for an imported File route. Sequential/lazy loader closures retain a
    /// clone of this same identity; keeping it on the generator makes that lifetime explicit.
    _native_dit: Option<mlx_gen::PinnedWeightsFile>,
    /// Component-residency strategy (epic 10834 Phase 1, sc-11101; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`]. `Resident` (default) holds the Qwen3-VL-4B
    /// text phase + DiT + VAE warm for the whole job and across jobs; `Sequential` holds only the
    /// per-phase loader closures and re-loads per generation in phase order (encode → **drop the text
    /// phase** → denoise/decode), bounding peak unified memory to `max(text, DiT+VAE)` instead of the
    /// sum (the Qwen3-VL-4B text phase is the dropped ~4B component; the single-stream DiT is 12B). The
    /// [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel checks, and
    /// the error-safe cache flush.
    residency: Residency<KreaText, KreaHeavyOwned>,
    /// The LoRA/LoKr adapters this model was loaded with (`LoadSpec::adapters`), retained so the
    /// multi-phase render (epic 13879, sc-13884) can install each phase's named subset on that phase's
    /// job-local DiT clone. A phase references these by index (bounds-checked against `adapters.len()`),
    /// with an optional per-phase weight override. Empty ⇒ a base (adapter-free) model, so only
    /// base-only phases are valid. The adapters are ALSO baked into the resident DiT at load (unchanged
    /// single-phase behavior); the multi-phase path clears + re-installs the phase subset on its clone,
    /// so the phase's adapter set is authoritative regardless of what was baked.
    adapters: Vec<AdapterSpec>,
    /// `true` if any load-time adapter is a ComfyUI/lightx2v **diff-patch** (`.diff`/`.diff_b`), detected
    /// from the adapter file headers at load (sc-13884). A diff-patch delta folds IRREVERSIBLY into the
    /// dense base at load (`W += δ`); every job-local DiT clone inherits that mutated base, and
    /// `clear_adapters` (which only drops low-rank residual stacks) cannot undo it — so a "base-only"
    /// phase would silently carry the diff-patch. Multi-phase is therefore rejected loudly on such a
    /// model (low-rank LoRA/LoKr — including the turbo LoRA — toggle cleanly and are unaffected).
    has_diff_patch: bool,
    /// Caller-prepared identities retained for any adapter files reopened by multi-phase generation.
    /// Primary/PiD deferred loaders keep their exact tokens too; retaining the complete spec makes
    /// every later path lookup use the same cache-identity contract.
    file_pin_spec: LoadSpec,
}

/// The heavy render-phase components (the single-stream DiT + VAE, via [`KreaHeavy`], plus the optional
/// PiD decoder) — everything but the text phase. Owned by the `Resident` components or by a
/// `Sequential` generate.
pub(crate) struct KreaHeavyOwned {
    heavy: KreaHeavy,
    /// Optional PiD super-resolving decoder (epic 7840, sc-7845), loaded when `spec.pid` is set; Krea
    /// reuses the Qwen-Image latent space, so it shares the `qwenimage` PiD student. `req.use_pid`
    /// routes decode through it instead of the VAE. `None` for the plain VAE path.
    pid: Option<PidEngine>,
    /// Experimental load-time VAE override. This is mutually exclusive with PiD and exists only
    /// for the explicitly compatible z16 Krea variants advertised by gen-core.
    alternate_decoder: Option<OwnedWanSingleFrameDecoder>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode dispatch runs identically
/// whether they are held resident or were just loaded by the `Sequential` path.
struct KreaHeavyRef<'a> {
    heavy: &'a KreaHeavy,
    pid: Option<&'a PidEngine>,
    alternate_decoder: Option<&'a OwnedWanSingleFrameDecoder>,
}

impl KreaHeavyOwned {
    fn as_ref(&self) -> KreaHeavyRef<'_> {
        KreaHeavyRef {
            heavy: &self.heavy,
            pid: self.pid.as_ref(),
            alternate_decoder: self.alternate_decoder.as_ref(),
        }
    }
}

/// The pre-encoded DiT text context(s) a `generate` renders from (sc-11101): the conditional context
/// always, plus the unconditional one for true-CFG (`krea_2_raw` / `krea_2_edit` with `guidance > 0`).
/// Produced once by [`Krea::encode`] (Turbo/Raw = plain text encode; edit = Qwen3-VL grounded encode)
/// so a `Sequential` job can drop the text phase before the DiT loads.
struct KreaContexts {
    pos: Array,
    neg: Option<Array>,
}

/// Load a Krea generator from a [`LoadSpec`]. `spec.weights` must be a [`WeightsSource::Dir`] pointing
/// at a Krea 2 snapshot (`transformer/ text_encoder/ vae/ tokenizer/`). Parses + validates the DiT
/// config against the spike architecture (catches a wrong/truncated snapshot at load); a precision
/// override is rejected rather than silently ignored. Raw-trained LoRA/LoKr adapters in `spec.adapters`
/// are installed onto the DiT (sc-7911).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, descriptor())
}

/// Load the undistilled **Raw** generator (`krea_2_raw`, epic 9992). Identical snapshot assembly to
/// [`load`] — the Raw + Turbo turnkeys share the exact architecture / weight layout (only distilled-vs-
/// base DiT weights differ), so one loader serves both — but stores the CFG-capable [`raw_descriptor`]
/// so `generate` runs the full-CFG path.
pub fn load_raw(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, raw_descriptor())
}

/// Load the **image-edit** generator (`krea_2_edit`, epic 10871). Identical snapshot assembly to
/// [`load_raw`] — edit shares the Raw pipeline (the source is in-context conditioning, not a distinct
/// model) — but stores the [`edit_descriptor`] so `generate` routes a source `Reference` to the Kontext
/// edit entrypoint. The snapshot MUST carry the Qwen3-VL vision tower (`text_encoder/` `visual.*`) for
/// the grounded half of the dual conditioning; the turnkey keeps it dense ([`crate::convert`]).
pub fn load_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, edit_descriptor())
}

/// Load the **CFG-free Turbo image-edit** generator (`krea_2_turbo_edit`, sc-11640). Identical snapshot
/// assembly to [`load_edit`] — same dual-conditioning edit surface — but `spec.weights` must point at a
/// **Turbo** (distilled) snapshot and the stored [`turbo_edit_descriptor`] makes `generate` route the
/// source(s) to the edit entrypoint on the **few-step CFG-free** schedule (`turbo_schedule`, single
/// conditional forward). The snapshot MUST carry the Qwen3-VL vision tower (`text_encoder/` `visual.*`)
/// for the grounded conditioning — the Turbo turnkey shares Raw's dense text encoder, so it does.
pub fn load_turbo_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(spec, turbo_edit_descriptor())
}

/// Build a Krea generator from a **community single-file** DiT checkpoint (epic 14015, sc-14017 S0b) —
/// the out-of-registry entrypoint the SceneWorks worker (S0c) calls, mirroring candle z-image's
/// `load_from_comfyui_components`. `dit_file` is a ComfyUI-exported dense bf16 Krea 2 DiT with
/// native-mmdit keys under `model.diffusion_model.` (e.g. `kreamania_variant5.safetensors`, DiT-only);
/// `base_snapshot_dir` is a **resident turnkey** snapshot dir (`transformer/ text_encoder/ vae/
/// tokenizer/`) supplying the shared text-encoder, VAE, tokenizer, and the DiT architecture config the
/// single file omits.
///
/// The DiT is read from the single file, key-remapped native→diffusers, coverage/shape-validated, and
/// assembled ([`crate::loader::load_transformer_from_native_file`] — fail-closed on any unmapped on-disk
/// key or missing module weight); the text-encoder / VAE / tokenizer load from `base_snapshot_dir` exactly
/// as [`load`] does from a full snapshot. The result is a warm-`Resident` generator that renders through
/// the same pipeline as a snapshot load. `descriptor` selects the surface (Turbo `descriptor()` is the
/// natural default — variant5 is a distilled-Turbo dense merge; `edit_descriptor()` for the edit lane).
///
/// `adapters` are Raw-trained LoRA/LoKr adapters (`LoadSpec::adapters`) installed onto the single-file DiT
/// via [`KreaHeavy::apply_adapters`] BEFORE the residency is finalized — the same load→apply order the
/// snapshot path uses (`load_krea_heavy`) — so the community edit lane's `krea2_identity_edit` adapter
/// (sc-14119) rides this entrypoint exactly as it does a snapshot edit load. Application is fail-closed (a
/// typed error on any adapter target that matches no module, never a silent drop). The t2i/img2img callers
/// pass `&[]`, whose dense-bf16 native-key load is byte-identical to before this parameter existed.
///
/// Supports dense bf16 and descriptor-validated, non-rotated int8-per-row single files. This legacy
/// signature is now only a `LoadSpec` construction shim; registry and direct callers share exactly the
/// same validation, adapter folding, pinned-file streaming, and generator assembly path.
pub fn load_from_native_dit_file(
    dit_file: impl AsRef<Path>,
    base_snapshot_dir: impl AsRef<Path>,
    adapters: &[AdapterSpec],
    descriptor: ModelDescriptor,
) -> Result<Box<dyn Generator>> {
    let dit_file = dit_file.as_ref();
    let base_snapshot_dir = base_snapshot_dir.as_ref();
    let mut spec = LoadSpec::new(WeightsSource::File(dit_file.to_path_buf()))
        .with_component(
            BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(base_snapshot_dir.to_path_buf()),
        )
        .with_adapters(adapters.to_vec());
    spec.prepare_file_sources()?;
    load_variant(&spec, descriptor)
}

/// The concrete-[`Krea`] assembly behind [`load_from_native_dit_file`] (which boxes the result). Returning
/// the concrete type lets the real-weight harness assert the installed `adapters` / `has_diff_patch`
/// fields — a `Box<dyn Generator>` could not be inspected. See [`load_from_native_dit_file`] for the full
/// contract; this carries the adapter-fold ordering.
#[cfg(test)]
pub(crate) fn build_native_krea(
    dit_file: impl AsRef<Path>,
    base_snapshot_dir: impl AsRef<Path>,
    adapters: &[AdapterSpec],
    descriptor: ModelDescriptor,
) -> Result<Krea> {
    let dit_file = dit_file.as_ref();
    let base_snapshot_dir = base_snapshot_dir.as_ref();
    let mut spec = LoadSpec::new(WeightsSource::File(dit_file.to_path_buf()))
        .with_component(
            BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(base_snapshot_dir.to_path_buf()),
        )
        .with_adapters(adapters.to_vec());
    spec.prepare_file_sources()?;
    build_native_krea_from_spec(&spec, descriptor)
}

/// Shared loader behind [`load`] / [`load_raw`] / [`load_edit`]: build the residency from a snapshot
/// dir. `Resident` (default) assembles every component now and holds it warm; `Sequential` keeps only
/// the [`LoadSpec`] and re-loads per generate in phase order (encode → drop the text phase →
/// denoise/decode) to bound peak memory to `max(text, DiT+VAE)`. Both use the same per-phase loaders
/// ([`load_krea_text`] / [`load_krea_heavy`]), so the components are byte-identical. `descriptor`
/// selects the variant (Turbo vs Raw vs edit) the returned [`Krea`] renders.
fn load_variant(spec: &LoadSpec, descriptor: ModelDescriptor) -> Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    validate_base_krea_load_axes(spec, descriptor.id)?;
    mlx_gen_wan::validate_selected_single_frame_decoder(spec, &descriptor)?;
    if matches!(spec.weights, WeightsSource::File(_)) {
        return Ok(Box::new(build_native_krea_from_spec(spec, descriptor)?));
    }
    let (memory_strategy, load_plan) =
        crate::block_memory_strategy::memory_strategy_contract_with_plan(descriptor.id, spec)?;
    let residency = build_residency(spec, descriptor.id, load_plan)?;
    Ok(Box::new(Krea {
        descriptor,
        memory_strategy,
        precision: spec.precision,
        quant: load_plan.effective_quant,
        streamable_transformer: load_plan.streamable_transformer,
        _native_dit: None,
        residency,
        adapters: spec.adapters.clone(),
        has_diff_patch: adapters_have_diff_patch_for_spec(spec)?,
        file_pin_spec: spec.clone(),
    }))
}

pub(crate) fn validate_base_krea_load_axes(spec: &LoadSpec, provider_id: &str) -> Result<()> {
    let allowed_components: &[&str] = if matches!(spec.weights, WeightsSource::File(_)) {
        &[BASE_SNAPSHOT_COMPONENT, VAE_COMPONENT]
    } else {
        &[VAE_COMPONENT]
    };
    mlx_gen::gen_core::reject_unknown_components(spec, allowed_components, provider_id)?;
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(Error::Unsupported(format!(
            "{provider_id}: the base Krea provider does not accept control/IP-adapter overlays"
        )));
    }
    if spec.identity.is_some() {
        return Err(Error::Unsupported(format!(
            "{provider_id}: the base Krea provider does not accept identity fields"
        )));
    }
    Ok(())
}

pub(crate) fn validate_native_krea_spec(spec: &LoadSpec, provider_id: &str) -> Result<()> {
    validate_base_krea_load_axes(spec, provider_id)?;
    mlx_gen_wan::validate_selected_single_frame_decoder(spec, &descriptor_for_id(provider_id))?;
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{}: only the default dense precision is wired (drop the precision override)",
            provider_id
        )));
    }
    let _ = mlx_gen::gen_core::require_base_snapshot(spec, provider_id)?;
    Ok(())
}

fn build_native_krea_from_spec(spec: &LoadSpec, descriptor: ModelDescriptor) -> Result<Krea> {
    validate_native_krea_spec(spec, descriptor.id)?;
    let base = mlx_gen::gen_core::require_base_snapshot(spec, descriptor.id)?;
    let WeightsSource::File(_) = &spec.weights else {
        unreachable!("native builder is called only for File weights")
    };
    let native_dit = spec
        .weights_file_pin()?
        .expect("File weights must resolve to a pin");
    let mut pinned_spec = spec.clone();
    pinned_spec.weights = WeightsSource::File(native_dit.loader_path().to_path_buf());
    // Physical execution eligibility is intentionally independent from public evidence. An explicit
    // Sequential + Deferred File request can use the retained pin to reopen one transformer block at
    // a time, while the contract below continues to report rung 4 as Missing until File-specific
    // measurements are promoted.
    let reopenable = native_file_streamable(spec)?;
    let memory_strategy = native_dit.read_unchanged(|_| {
        crate::block_memory_strategy::native_memory_strategy_contract_from_spec(
            descriptor.id,
            &pinned_spec,
            base,
            false,
        )
        .map_err(Error::from)
    })?;
    let text_base = base.to_path_buf();
    let text_encoder_source = ENCODER_CONTRACT.source_for_load(spec, base)?;
    let expected_text_encoder_bits = native_text_encoder_expected_quant_bits(base)?;
    let text_encoder_load_time_quant_bits =
        text_encoder_source.load_time_quant_bits(expected_text_encoder_bits, descriptor.id)?;
    let heavy_base = base.to_path_buf();
    let heavy_dit = native_dit.clone();
    let heavy_spec = spec.clone();
    let heavy_id = descriptor.id;
    let residency = Residency::from_policy(
        spec.offload_policy,
        move || {
            load_krea_text_resolved(
                &text_base,
                &text_encoder_source,
                text_encoder_load_time_quant_bits,
            )
        },
        move |load_pid| {
            load_native_krea_heavy(
                &heavy_spec,
                &heavy_base,
                &heavy_dit,
                reopenable,
                load_pid,
                heavy_id,
            )
        },
    )?;
    Ok(Krea {
        descriptor,
        memory_strategy,
        precision: spec.precision,
        quant: spec.quantize,
        streamable_transformer: reopenable,
        _native_dit: Some(native_dit),
        residency,
        adapters: spec.adapters.clone(),
        has_diff_patch: adapters_have_diff_patch_for_spec(spec)?,
        file_pin_spec: spec.clone(),
    })
}

/// Whether a registered primary-File provider can physically execute bounded transformer residency.
///
/// This is an execution predicate, not an evidence claim: File contracts deliberately keep rung 4
/// `Missing`. Low-rank adapters are replayed by [`crate::block_stream::KreaBlockStream`], while a
/// dense diff-patch is excluded because it irreversibly mutates the eager base and cannot be rebuilt
/// from a pristine per-window reopen. Header inspection runs beneath the caller-prepared File tokens.
pub(crate) fn native_file_streamable(spec: &LoadSpec) -> Result<bool> {
    if !matches!(spec.weights, WeightsSource::File(_)) {
        return Ok(false);
    }
    Ok(matches!(
        spec.offload_policy,
        mlx_gen::gen_core::OffloadPolicy::Sequential
    ) && matches!(
        spec.load_shape,
        mlx_gen::gen_core::LoadShape::DeferredMaterialization
    ) && spec.quantize.is_none()
        && !adapters_have_diff_patch_for_spec(spec)?)
}

fn load_native_krea_heavy(
    spec: &LoadSpec,
    base: &Path,
    dit_file: &mlx_gen::PinnedWeightsFile,
    streamable: bool,
    load_pid: bool,
    id: &'static str,
) -> Result<KreaHeavyOwned> {
    let cfg = crate::config::Krea2Config::from_snapshot(base)?;
    let dit = if let Some(quant) = spec.quantize {
        crate::loader::load_transformer_from_pinned_native_file_bounded(dit_file, &cfg, |dit| {
            if !spec.adapters.is_empty() {
                spec.read_files_unchanged(
                    spec.adapters.iter().map(|adapter| &adapter.path),
                    || dit.apply_adapters_strict(&spec.adapters, true),
                )?;
            }
            dit.quantize(quant.bits())
        })?
    } else {
        crate::loader::load_transformer_from_pinned_native_file_with_stream(
            dit_file, &cfg, streamable,
        )?
    };
    let vae = crate::vae::load_vae(base)?;
    let mut heavy = KreaHeavy::from_parts(dit, vae);
    if spec.quantize.is_none() && !spec.adapters.is_empty() {
        spec.read_files_unchanged(spec.adapters.iter().map(|adapter| &adapter.path), || {
            heavy.apply_adapters(&spec.adapters)
        })?;
    }
    let pid = load_pid
        .then(|| load_prepared_pid(spec))
        .transpose()?
        .flatten();
    let alternate_decoder =
        mlx_gen_wan::load_selected_single_frame_decoder(spec, &descriptor_for_id(id))?;
    Ok(KreaHeavyOwned {
        heavy,
        pid,
        alternate_decoder,
    })
}

fn descriptor_for_id(id: &str) -> ModelDescriptor {
    match id {
        KREA_2_TURBO_ID => descriptor(),
        KREA_2_RAW_ID => raw_descriptor(),
        KREA_2_EDIT_ID => edit_descriptor(),
        KREA_2_TURBO_EDIT_ID => turbo_edit_descriptor(),
        _ => unreachable!("Krea loader called with unregistered descriptor id {id}"),
    }
}

/// Detect whether any load-time adapter is a ComfyUI/lightx2v **diff-patch** (`.diff`/`.diff_b`), read
/// from each adapter file's safetensors HEADER only (no tensor load) — the sc-13884 multi-phase guard's
/// input. Best-effort: a header we cannot read yields `false` here, but the same file is read for real
/// by the load-time [`KreaHeavy::apply_adapters`], which surfaces the genuine error loudly — so an
/// unreadable file never silently slips a diff-patch through into a wrong multi-phase render.
pub(crate) fn adapters_have_diff_patch(specs: &[AdapterSpec]) -> bool {
    specs.iter().any(|spec| {
        mlx_gen::gen_core::weightsmeta::CheckpointMeta::from_file(&spec.path)
            .map(|meta| mlx_gen::adapters::loader::has_diff_patch_key_names(meta.keys()))
            .unwrap_or(false)
    })
}

/// Prepared-token wrapper for [`adapters_have_diff_patch`]. The header classification influences
/// multi-phase safety, so it must inspect the same adapter identities the cache key and merge use.
pub(crate) fn adapters_have_diff_patch_for_spec(spec: &LoadSpec) -> Result<bool> {
    spec.read_prepared_files_unchanged(|| Ok(adapters_have_diff_patch(&spec.adapters)))
}

/// Load PiD beneath the same caller-prepared File tokens used for request cache identity.
///
/// Today the checkpoint is normally a File and Gemma is normally a Dir, but guarding both declared
/// sources keeps this correct for any accepted File-shaped compatibility input. The outer pre/post
/// checks also span the provider's lazy tensor materialization.
fn load_prepared_pid(spec: &LoadSpec) -> Result<Option<PidEngine>> {
    let Some(pid) = &spec.pid else {
        return Ok(None);
    };
    let file_paths = [&pid.checkpoint, &pid.gemma]
        .into_iter()
        .filter_map(|source| match source {
            WeightsSource::File(path) => Some(path.as_path()),
            WeightsSource::Dir(_) => None,
        });
    spec.read_files_unchanged(file_paths, || PidEngine::from_spec(pid, PID_BACKBONE))
        .map(Some)
}

pub(crate) fn resolve_transformer_window(
    req: &GenerationRequest,
    streamable: bool,
) -> Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    let component = memory.transformer_window_component.unwrap_or_default();
    if component != mlx_gen::gen_core::TransformerComponent::Dit {
        return Err(Error::Unsupported(format!(
            "krea: rung 4 implements the DiT component only; requested {component:?}"
        )));
    }
    if !streamable {
        return Err(Error::Unsupported(
            "krea: bounded transformer residency requires a Sequential, deferred-materialization, \
             re-openable snapshot load without a dense diff-patch adapter"
                .to_owned(),
        ));
    }
    let window = memory
        .transformer_window_size
        .unwrap_or(crate::block_memory_strategy::TRANSFORMER_WINDOW_SIZE);
    if window != crate::block_memory_strategy::TRANSFORMER_WINDOW_SIZE {
        return Err(Error::Unsupported(format!(
            "krea: transformer_window_size={window} is outside the measured domain {:?}",
            [crate::block_memory_strategy::TRANSFORMER_WINDOW_SIZE]
        )));
    }
    Ok(Some(window as usize))
}

/// The policy→[`Residency`] dispatch every Krea variant shares (sc-11101; routed through the single
/// [`Residency::from_policy`] seam in sc-11126, F-180), so no variant re-derives the
/// `match offload_policy`. `Resident` eager-loads the text phase + heavy bundle now (the heavy loader
/// with `use_pid = true` so any PiD overlay is loaded once and reused); `Sequential` captures the two
/// per-phase loaders and loads nothing now, deferring each to [`Residency::run`]. Both go through the
/// same [`load_krea_text`] / [`load_krea_heavy`], so the `Resident` composition is byte-identical to
/// the pre-seam one. The up-front [`resolve_root`] fails fast (precision + single-file rejection) for
/// BOTH policies. The deferral is weight-free-testable: under `Sequential` this touches no component
/// weights, so a dispatch that mapped `Sequential → Resident` (ignoring `offload_policy`) would
/// eager-load here and fail the "Sequential defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
    id: &'static str,
    load_plan: ResolvedLoadPlan,
) -> Result<Residency<KreaText, KreaHeavyOwned>> {
    // Up-front fail-fast for both policies (precision override + single-file rejection).
    let root = resolve_root(spec, id)?;
    let text_encoder_source = ENCODER_CONTRACT.source_for_load(spec, root)?;
    let text_encoder_load_time_quant_bits =
        text_encoder_source.load_time_quant_bits(load_plan.effective_quant.map(Quant::bits), id)?;
    let text_root = root.to_path_buf();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || {
            load_krea_text_resolved(
                &text_root,
                &text_encoder_source,
                text_encoder_load_time_quant_bits,
            )
        },
        move |use_pid| {
            load_krea_heavy(
                &spec_heavy,
                resolve_root(&spec_heavy, id)?,
                use_pid,
                load_plan,
                id,
            )
        },
    )
}

/// Precision guard (only dense bf16 is wired) + snapshot-dir resolution (rejecting a single-file
/// source), shared by [`load_krea_text`] / [`load_krea_heavy`] and the `Sequential` per-phase loaders
/// (sc-11101).
fn resolve_root<'a>(spec: &'a LoadSpec, id: &str) -> Result<&'a Path> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only the default dense precision is wired (drop the precision override)"
        )));
    }
    mlx_gen::gen_core::require_base_snapshot(spec, id).map_err(Into::into)
}

/// Resolve the load-time quantize for a component (F-076). Returns `Some(bits)` to quantize the dense
/// base in place, or `None` when there is no quant OR the turnkey is already packed at the requested
/// bits (`quantize()` would be a no-op). Errors on a packed-vs-requested mismatch so e.g. Q4 over a Q8
/// turnkey never silently serves Q8. Shared by the text + heavy loaders (the marker in
/// `transformer/config.json` is model-wide), so both phases decide identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedLoadPlan {
    pub(crate) load_time_quant_bits: Option<i32>,
    pub(crate) effective_quant: Option<Quant>,
    pub(crate) streamable_transformer: bool,
}

pub(crate) fn resolve_load_plan(
    spec: &LoadSpec,
    root: &Path,
    id: &str,
) -> Result<ResolvedLoadPlan> {
    resolve_load_plan_for_component(
        spec,
        root,
        id,
        matches!(spec.weights, WeightsSource::File(_)),
    )
}

/// Resolve the quantization carried by `root`. For an imported primary File, the base snapshot's
/// text tower is only a companion component: it may be prepacked even though the File DiT carries
/// its own numeric format. Callers loading that text-only component set `primary_file_rules=false`;
/// primary-DiT admission retains the mismatch refusal.
pub(crate) fn resolve_load_plan_for_component(
    spec: &LoadSpec,
    root: &Path,
    id: &str,
    primary_file_rules: bool,
) -> Result<ResolvedLoadPlan> {
    // Parse the marker even without a quantization override. Contract construction, admission, and
    // loading must all reject the same malformed/unreadable packed snapshot instead of letting rung 4
    // advertise a source the generator cannot subsequently load.
    let packed_bits = mlx_gen::quant::packed_quant_bits(root, "transformer")?;
    let requested_bits = match spec.quantize {
        Some(quant @ (Quant::Q4 | Quant::Q8)) => Some(quant.bits()),
        Some(quant) => {
            return Err(Error::Unsupported(format!(
                "{id}: unsupported MLX quantization tier {quant:?}; expected Q4 or Q8"
            )))
        }
        None => None,
    };
    if let (true, Some(packed), None) = (primary_file_rules, packed_bits, requested_bits) {
        return Err(Error::Msg(format!(
            "{id}: imported single-file weights have no quant request, but the companion snapshot is pre-quantized Q{}; request the matching tier or stage a dense companion snapshot",
            packed
        )));
    }
    if let (Some(packed), Some(requested)) = (packed_bits, requested_bits) {
        if packed != requested {
            return Err(Error::Msg(format!(
                "{id}: transformer/ is a pre-quantized Q{packed} turnkey but Q{requested} was \
                 requested; quantize is a no-op on packed weights so the request would silently \
                 serve Q{packed}. Point at a Q{requested} snapshot (or a dense one)."
            )));
        }
    }
    let load_time_quant_bits = packed_bits.is_none().then_some(requested_bits).flatten();
    let effective_bits = packed_bits.or(load_time_quant_bits);
    let effective_quant = effective_bits
        .map(|bits| {
            crate::memory::tier_from_bits(bits).ok_or_else(|| {
                Error::Unsupported(format!(
                    "{id}: transformer declares unsupported packed quantization width {bits}"
                ))
            })
        })
        .transpose()?;
    Ok(ResolvedLoadPlan {
        load_time_quant_bits,
        effective_quant,
        streamable_transformer: false,
    })
}

pub(crate) fn load_time_quant_bits(spec: &LoadSpec, root: &Path, id: &str) -> Result<Option<i32>> {
    Ok(resolve_load_plan(spec, root, id)?.load_time_quant_bits)
}

/// The base DiT's **effective** quant bits for the pose-control branch gate (sc-11748): the tier the base
/// actually runs at, whether packed AT LOAD (a dense snapshot + `spec.quantize`) or ALREADY packed on
/// disk (a Q4/Q8 turnkey). Distinct from [`load_time_quant_bits`], which returns `None` for a pre-packed
/// turnkey (there is nothing to quantize *at load*) — but a pre-packed base still has a tier the pose
/// branch should match. `None` ⇒ a dense bf16 base (no tier). Surfaces the same packed-vs-requested
/// mismatch error as [`load_time_quant_bits`].
pub(crate) fn effective_base_quant_bits(
    spec: &LoadSpec,
    root: &Path,
    id: &str,
) -> Result<Option<i32>> {
    Ok(resolve_load_plan(spec, root, id)?
        .effective_quant
        .map(Quant::bits))
}

/// Resolve the tier the base transformer actually uses. Unlike `LoadSpec::quantize`, this observes a
/// pre-packed turnkey's on-disk marker and therefore remains correct when the worker selects Q4/Q8 by
/// choosing a tier-specific snapshot without requesting an in-place quantization pass.
pub(crate) fn effective_base_quant_tier(spec: &LoadSpec, id: &str) -> Result<Option<Quant>> {
    let root = resolve_root(spec, id)?;
    Ok(resolve_load_plan(spec, root, id)?.effective_quant)
}

/// Project the exact language tensors retained from a validated encoder source using the same
/// effective numeric policy as [`load_krea_text_resolved`]. A dense alternate inherits the base
/// transformer's Q4/Q8 tier, a matching packed alternate keeps its stored affine triples, and a
/// packed mismatch fails before memory admission can authorize a load the runtime rejects.
pub(crate) fn selected_language_resident_bytes(
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
    expected_bits: Option<i32>,
    provider_id: &str,
) -> mlx_gen::gen_core::Result<u64> {
    let load_time_quant_bits = source.load_time_quant_bits(expected_bits, provider_id)?;
    let headers = source.materialized_language_tensor_headers(&ENCODER_CONTRACT)?;
    mlx_gen::asset_facts::projected_tensor_headers_bytes(&headers, |tensor| {
        if let Some(bits) = load_time_quant_bits
            .filter(|_| crate::convert::is_text_encoder_quant_target(&tensor.name))
        {
            mlx_gen::asset_facts::ResidentProjection::GroupQuantized {
                bits,
                group_size: crate::quant::GROUP_SIZE as usize,
            }
        } else {
            mlx_gen::asset_facts::ResidentProjection::Stored
        }
    })
}

/// The text-encoder policy retained by an imported native-DiT composition. Native Krea files are
/// materialized dense independently of the borrowed snapshot, while the runtime deliberately keeps
/// the borrowed snapshot's language tier. Keep that File-specific policy in one seam shared by load
/// and admission rather than attempting to infer it from the imported DiT's storage descriptor.
pub(crate) fn native_text_encoder_expected_quant_bits(
    base_snapshot_dir: &Path,
) -> mlx_gen::gen_core::Result<Option<i32>> {
    mlx_gen::gen_core::text_encoder_packed_quant_bits(&WeightsSource::Dir(
        base_snapshot_dir.join("text_encoder"),
    ))
}

/// Load the Krea text phase (tokenizer + Qwen3-VL-4B condition encoder + vision tower) — the component
/// dropped first under `Sequential`. Applies the optional (F-076-guarded) text-encoder quantize; the
/// VAE + vision tower stay dense (the monolithic `KreaPipeline::quantize` quantized `te` + `dit`, not
/// the VAE/vision), so the `Resident` and `Sequential` paths build byte-identical text phases.
pub(crate) fn load_krea_text(spec: &LoadSpec, root: &Path, id: &str) -> Result<KreaText> {
    let plan = resolve_load_plan(spec, root, id)?;
    let source = ENCODER_CONTRACT.source_for_load(spec, root)?;
    let bits = source.load_time_quant_bits(plan.effective_quant.map(Quant::bits), id)?;
    load_krea_text_resolved(root, &source, bits)
}

pub(crate) fn load_krea_text_resolved(
    root: &Path,
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
    load_time_quant_bits: Option<i32>,
) -> Result<KreaText> {
    let mut text = KreaText::from_snapshot_with_text_encoder(root, source)?;
    if let Some(bits) = load_time_quant_bits {
        text.quantize(bits)?;
        text.materialize_weights()?;
    }
    Ok(text)
}

/// Load the Krea heavy render phase (single-stream DiT + VAE + the optional PiD overlay) — everything
/// but the text phase. Install Raw-trained LoRA/LoKr adapters onto the DiT BEFORE the optional quantize,
/// so the residual stacks over the (possibly already-packed) base (the Lens load→apply→quantize order);
/// the shared seam errors (never silently drops) on an adapter target that matches no module. Factored
/// so `Sequential` loads these AFTER the text phase is dropped (bounding peak to `max(text, DiT+VAE)`).
fn load_krea_heavy(
    spec: &LoadSpec,
    root: &Path,
    load_pid: bool,
    load_plan: ResolvedLoadPlan,
    id: &'static str,
) -> Result<KreaHeavyOwned> {
    let mut heavy = KreaHeavy::from_snapshot_with_stream(root, load_plan.streamable_transformer)?;
    if !spec.adapters.is_empty() {
        spec.read_files_unchanged(spec.adapters.iter().map(|adapter| &adapter.path), || {
            heavy.apply_adapters(&spec.adapters)
        })?;
    }
    if let Some(bits) = load_plan.load_time_quant_bits {
        heavy.quantize(bits)?;
    }
    // Optional PiD decoder overlay (sc-7845): Krea reuses the Qwen-Image latent space, so it loads the
    // same `qwenimage` student + Gemma-2 caption encoder when `spec.pid` is set AND this generate uses
    // it (`load_pid`, F-177) — Resident passes `true` (loaded once, reused), Sequential passes
    // `req.use_pid` so a non-PiD generate skips the student + its Gemma-2 caption encoder entirely.
    let pid = load_pid
        .then(|| load_prepared_pid(spec))
        .transpose()?
        .flatten();
    let alternate_decoder =
        mlx_gen_wan::load_selected_single_frame_decoder(spec, &descriptor_for_id(id))?;
    Ok(KreaHeavyOwned {
        heavy,
        pid,
        alternate_decoder,
    })
}

impl Generator for Krea {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        crate::block_memory_strategy::safety_check(
            &self.memory_strategy,
            self.precision,
            self.quant,
            context,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        crate::block_memory_strategy::begin_request(
            self.descriptor.id,
            &self.memory_strategy,
            self.precision,
            self.quant,
            context,
        )
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_request(&self.descriptor, req)?;
        ensure_multiphase_allowed_for(self.descriptor.id, self.has_diff_patch, req)?;
        // Chained denoise passes (epic 20414, sc-20418): t2i-from-noise Turbo/Raw only. The shared
        // floor above already gated capability, ids, ranges, and the phases mutual exclusion.
        validate_denoise_pass_surface(self.descriptor.id, &self.descriptor.capabilities, req)?;
        ensure_denoise_pass_adapters_allowed(self.descriptor.id, self.has_diff_patch, req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl Krea {
    /// Text-encode the prompt (and, for true CFG, the negative) per the residency (sc-11101). `Resident`
    /// borrows the warm Qwen3-VL-4B text phase (byte-identical to the pre-sc-11101 per-image re-encode —
    /// the encode is deterministic, the per-image variation comes from the seed inside `render`);
    /// `Sequential` loads the text phase, encodes, then the seam materializes + DROPS it + `clear_cache()`
    /// so its ~4 GB frees before the DiT/VAE load. `is_edit` uses the Qwen3-VL grounded encode over the
    /// source image; `is_raw`/Turbo the plain text encode. The unconditional context is built only when
    /// `guidance > 0` (reference `cfg = guidance > 0`; Turbo is CFG-free → always `None`). Called by the
    /// shared residency seam's encode closure with the phase-A `text` component.
    #[allow(clippy::too_many_arguments)]
    fn encode_contexts(
        &self,
        text: &KreaText,
        req: &GenerationRequest,
        is_raw: bool,
        is_edit: bool,
        guidance: f32,
        negative: &str,
        edit_sources: &[&Image],
    ) -> Result<KreaContexts> {
        if is_edit {
            if edit_sources.is_empty() {
                return Err(Error::Msg(format!(
                    "{}: edit requires a source image",
                    self.descriptor.id
                )));
            }
            // Ground on ALL edit sources (scene + person), not just the first (F-071); run the vision
            // tower ONCE and reuse it for both the positive and (CFG) negative grounded encode (F-073).
            let gv = text.run_vision(edit_sources)?;
            // The optional "text style" tap-reweight gain (sc-12009) applies to the POSITIVE grounded
            // context — the grounded encode returns the SAME `[b, n_tok, 12, hidden]` tap structure the
            // plain encode does, so `apply_tap_weights` is shape-safe. The CFG-negative grounded context
            // is left untouched so the knob steers only the conditional prediction (mirrors the plain
            // Raw branch below); `None`/g≈1 is a no-op.
            let pos = maybe_apply_style_gain(
                text.encode_grounded_from_vision(&gv, &req.prompt)?,
                req.text_style_gain,
            )?;
            let neg = if guidance > 0.0 {
                Some(text.encode_grounded_from_vision(&gv, negative)?)
            } else {
                None
            };
            Ok(KreaContexts { pos, neg })
        } else if is_raw {
            // POSITIVE context carries the Krea "text style" tap-reweight gain (sc-11878); the
            // CFG-negative context is encoded WITHOUT it so the knob steers only the conditional
            // prediction (mirrors candle-gen-krea `encode_prompt_context`). `None`/g≈1 is a no-op.
            let pos = maybe_apply_style_gain(text.encode(&req.prompt)?, req.text_style_gain)?;
            let neg = if guidance > 0.0 {
                Some(text.encode(negative)?)
            } else {
                None
            };
            Ok(KreaContexts { pos, neg })
        } else {
            // Turbo (CFG-free) t2i/img2img: single conditional context, gain applied (no negative).
            Ok(KreaContexts {
                pos: maybe_apply_style_gain(text.encode(&req.prompt)?, req.text_style_gain)?,
                neg: None,
            })
        }
    }

    /// The rich-`Result` body behind [`Generator::generate`] — kept on the crate's own
    /// [`mlx_gen::Error`] so `?` lifts `mlx_rs` device exceptions transparently; the trait wrapper
    /// bridges the tail into [`gen_core::Error`]. Renders `req.count` images, one per seed (`seed + n`,
    /// mirroring the reference per-prompt seeding), through the residency (encode → drop text phase under
    /// `Sequential` → load heavy → per-image render → free heavy).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;
        // Loud-reject a multi-phase request on a diff-patch model (sc-13884): its baked `.diff` delta
        // can't be toggled off per phase, so it would silently corrupt "base-only" phases.
        ensure_multiphase_allowed_for(self.descriptor.id, self.has_diff_patch, req)?;
        // Chained denoise passes (sc-20418): the same floor `validate` applies, re-checked here for
        // callers that reach `generate` directly.
        validate_denoise_pass_surface(self.descriptor.id, &self.descriptor.capabilities, req)?;
        ensure_denoise_pass_adapters_allowed(self.descriptor.id, self.has_diff_patch, req)?;
        let transformer_window_size = resolve_transformer_window(req, self.streamable_transformer)?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        // Variant read back off the descriptor id: Raw = full-CFG undistilled (52-step, dynamic-mu);
        // Turbo = CFG-free distilled (8-step, fixed mu). One `Krea` struct, two render paths. The edit
        // variant (epic 10871) shares Raw's full-CFG sampler — an edit denoises from noise under true CFG
        // — so `is_raw` (the full-CFG path selector for schedule/steps/guidance) covers edit too; only the
        // per-image entrypoint below differs (`is_edit` → the Kontext edit path, not img2img/t2i).
        // The distilled CFG-free Turbo edit (sc-11640): routes to the SAME Kontext edit entrypoint as
        // `krea_2_edit`, but on the few-step `turbo_schedule` at `guidance = 0` (single conditional
        // forward). So it is an edit (`is_edit`) but NOT a full-CFG variant (`is_raw`).
        let is_turbo_edit = self.descriptor.id == KREA_2_TURBO_EDIT_ID;
        let is_edit = self.descriptor.id == KREA_2_EDIT_ID || is_turbo_edit;
        // `is_raw` gates the full-CFG sampler (52-step, dynamic-mu `base_schedule`, guidance) — Raw and
        // the full-CFG `krea_2_edit`, but NOT `krea_2_turbo_edit` (distilled few-step, CFG-free).
        let is_raw = self.descriptor.id == KREA_2_RAW_ID || (is_edit && !is_turbo_edit);
        let steps = req.steps.unwrap_or(if is_raw {
            DEFAULT_RAW_STEPS
        } else {
            DEFAULT_STEPS
        }) as usize;
        // Decode seam (sc-7845) + `from_ldm` early-stop (sc-7993): resolve the achieved degrade σ + the
        // truncation `keep` from the (seed-independent) schedule; the PiD decoder itself is built below
        // (it needs the heavy phase's PiD engine). Raw/edit use the resolution-dynamic schedule.
        let sigmas = if is_raw {
            base_schedule(steps, req.width, req.height, req.scheduler.as_deref())
        } else {
            turbo_schedule(steps, req.scheduler.as_deref())
        };
        // Edit extracts its own ordered source list (a single `Reference` or a `MultiReference`
        // scene+person pair, epic 10871 P1.3); img2img/t2i use the single-`Reference` helper. Kept
        // separate so an edit's `MultiReference` never trips `single_reference`'s "exactly one
        // Reference" img2img guard, and an img2img job still can't smuggle in two references.
        let edit_sources = if is_edit {
            edit_references(req)?
        } else {
            Vec::new()
        };
        let reference = if is_edit {
            None
        } else {
            single_reference(self.descriptor.id, req)?
        };
        // img2img resolves the PiD `from_ldm` capture against the SLICED window it actually denoises
        // (sc-10121). An img2img job seeds the denoise at `start = init_time_step(strength)` and runs
        // `sigmas[start..]`, so the capture index and its degrade σ MUST be resolved with that `start`
        // or the decoder's σ desyncs from the truncated latent. `flow_capture_for_request`'s
        // `start_step` does exactly that: it drops a capture whose ceiling would land at/before `start`
        // (no benefit) and otherwise returns a `keep` (into the full schedule) whose `sigmas[keep-1]`
        // matches the `render_*_img2img_from` truncation `sigmas[start..keep]` — so `capture_sigma`
        // always names the σ of the latent actually handed to PiD. t2i / edit / control denoise the
        // whole schedule from pure noise → `start = 0` (the reference-less default).
        let img2img_strength = reference.map(|(_, ref_strength)| {
            ref_strength
                .or(req.strength)
                .unwrap_or(DEFAULT_IMG2IMG_STRENGTH)
        });
        let start_step = img2img_strength
            .map(|s| init_time_step(steps, Some(s)).min(sigmas.len().saturating_sub(1)))
            .unwrap_or(0);
        let (capture_sigma, keep) = flow_capture_for_request(req, &sigmas, start_step);
        // Raw CFG knobs: guidance defaults to the reference Raw preset, an empty/absent negative → ""
        // (reference `negative_prompts = [""] * n`). Inert on the Turbo (CFG-free) t2i/img2img path.
        // FORCED to 0 for `krea_2_turbo_edit` so the edit runs a single conditional forward — its
        // descriptor advertises no guidance, so `req.guidance` is already rejected upstream; this just
        // pins the CFG-free default that makes `encode` skip the unconditional grounded context.
        let guidance = if is_turbo_edit {
            0.0
        } else {
            req.guidance.unwrap_or(DEFAULT_RAW_GUIDANCE)
        };
        let negative = req.negative_prompt.clone().unwrap_or_default();

        // Multi-phase denoise (epic 13879, sc-13884): resolve the phase list ONCE up-front so both the
        // text-encode phase (whether the unconditional context is needed) and the render phase drive
        // from the same plan. `validate_phases` has already gated this to the Raw t2i variant (no
        // reference/PiD, non-empty, ≥1-step); here we resolve the contiguous schedule slices, per-phase
        // guidance, and per-phase adapter sets (indices bounds-checked against `self.adapters`). `None`
        // ⇒ the ordinary single-phase render below.
        let mp_resolved: Option<Vec<ResolvedPhase>> = match req.phases.as_deref() {
            Some(list) if !list.is_empty() => {
                let default_guidance = req.guidance.unwrap_or(DEFAULT_RAW_GUIDANCE);
                Some(multiphase::resolve_phases(
                    list,
                    default_guidance,
                    self.adapters.len(),
                    self.descriptor.id,
                )?)
            }
            _ => None,
        };
        // The multi-phase schedule is ONE global Raw schedule for the TOTAL step budget (the sum of the
        // phases' steps); every phase runs a contiguous slice of it — the crux that keeps the sigma
        // trajectory continuous across boundaries (no per-phase recompute / reset).
        let mp_sigmas = mp_resolved.as_ref().map(|resolved| {
            let total = resolved.last().map(|p| p.slice.end).unwrap_or(0);
            base_schedule(total, req.width, req.height, req.scheduler.as_deref())
        });
        // Chained denoise passes (epic 20414, sc-20418): resolve the EXPLICIT plan once up-front —
        // adapter-index bounds against the real loaded count, the encode decision (whether any pass
        // uses CFG), and the per-image render all drive from the same resolution. `None` ⇒ every
        // existing path below is byte-untouched (including the RAW `phases` executor, with which
        // this is mutually exclusive at the shared floor).
        let dp_defaults = krea_denoise_defaults(is_raw);
        let dp_ctx = self
            .descriptor
            .capabilities
            .denoise_pass_context(Some(self.adapters.len()));
        let dp_resolve = |seed: u64| -> Result<mlx_gen::gen_core::ResolvedDenoisePlan> {
            req.resolve_denoise_plan(seed, &dp_defaults, &dp_ctx)
                .map_err(mlx_gen::gen_core::Error::from)
                .map_err(Error::from)
        };
        let dp_plan: Option<mlx_gen::gen_core::ResolvedDenoisePlan> = match &req.denoise_passes {
            Some(_) => Some(dp_resolve(base_seed)?),
            None => None,
        };
        let dp_need_neg = is_raw
            && dp_plan
                .as_ref()
                .is_some_and(|plan| plan.passes.iter().any(|p| p.guidance.unwrap_or(0.0) > 0.0));

        // The text encode builds the unconditional context iff ANY phase runs CFG (guidance > 0). A
        // single positive value suffices for `encode_contexts`' neg-gate (it consults `guidance` only as
        // `guidance > 0`, never in the combine — the per-phase guidance drives the actual CFG combine).
        // The chained denoise-pass plan gates the same way: the negative context is encoded iff any
        // pass uses CFG.
        let encode_guidance = match &mp_resolved {
            Some(resolved) if multiphase::any_phase_uses_cfg(resolved) => 1.0,
            Some(_) => 0.0,
            None if dp_plan.is_some() => {
                if dp_need_neg {
                    1.0
                } else {
                    0.0
                }
            }
            None => guidance,
        };

        // Phase A: prompt → context(s) (sc-11101; sc-11125). Under `Sequential` the shared seam loads
        // the Qwen3-VL-4B text phase, encodes, materializes, then DROPS it + `clear_cache()` so its
        // ~4 GB frees before the DiT/VAE load below — the peak-bounding win. Under `Resident` it borrows
        // the warm text phase. Edit grounds on ALL source images (scene + person), F-071.
        self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            |text: &KreaText| {
                self.encode_contexts(
                    text,
                    req,
                    is_raw,
                    is_edit,
                    encode_guidance,
                    &negative,
                    &edit_sources,
                )
            },
            // Materialize pos (+neg) while the text phase is still alive (Sequential only) — MLX is
            // lazy, so an un-evaluated context keeps the encoder referenced and the drop frees nothing.
            |ctx: Option<&KreaContexts>| {
                let Some(ctx) = ctx else { return Ok(()) };
                match &ctx.neg {
                    Some(neg) => mlx_rs::transforms::eval([&ctx.pos, neg])?,
                    None => mlx_rs::transforms::eval([&ctx.pos])?,
                }
                Ok(())
            },
            // Phase B: heavy render components (DiT + VAE + PiD). The render dispatch below runs
            // identically for both residencies.
            |heavy_owned, ctx, on_progress| {
                let heavy = heavy_owned.as_ref();

                // PiD decode overlay (sc-7845): one decoder serves the whole count loop (same prompt → same
                // caption). Errors if `req.use_pid` but the model wasn't loaded with `LoadSpec::pid`; `None`
                // → the native VAE. Resolved against the heavy phase's PiD engine.
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid,
                    req,
                    base_seed,
                    self.descriptor.id,
                    capture_sigma,
                )?;
                let decoder = pid_decoder
                    .as_ref()
                    .map(|decoder| decoder as &dyn LatentDecoder)
                    .or_else(|| {
                        heavy
                            .alternate_decoder
                            .map(|decoder| decoder as &dyn LatentDecoder)
                    });

                // Chained denoise passes (epic 20414, sc-20418): drive the resolved plan through
                // the SHARED gen-core pass executor (fresh schedule + solver state per pass,
                // deterministic latent-space boundary re-noise on the pass seed, chain-wide
                // effective-model-eval progress, per-eval cancellation) over per-pass prepared
                // state — the provider's own DiT when no pass overrides adapter weights, else one
                // job-local clone per pass. ONE decode per image, after the whole chain. Returns
                // before the multi-phase and single-pass dispatches.
                if let Some(base_plan) = dp_plan.as_ref() {
                    let plans = self.file_pin_spec.read_files_unchanged(
                        self.adapters.iter().map(|adapter| &adapter.path),
                        || {
                            heavy.heavy.prepare_denoise_passes(
                                &base_plan.passes,
                                &self.adapters,
                                is_raw,
                                &ctx.pos,
                                ctx.neg.as_ref(),
                                req.width,
                                req.height,
                            )
                        },
                    )?;
                    let mut images = Vec::with_capacity(req.count as usize);
                    let mut first_execution: Option<mlx_gen::gen_core::DenoisePlanExecution> = None;
                    for n in 0..req.count {
                        let seed = base_seed.wrapping_add(n as u64);
                        // Image n re-resolves the plan at its own job seed (`seed + n`, the batch
                        // convention), so its pass seeds are domain-separated per image exactly as
                        // a standalone render at that seed would be.
                        let plan = if n == 0 {
                            base_plan.clone()
                        } else {
                            dp_resolve(seed)?
                        };
                        let opts = TurboOptions {
                            width: req.width,
                            height: req.height,
                            steps,
                            seed,
                            sampler: req.sampler.clone(),
                            scheduler: req.scheduler.clone(),
                            transformer_window_size,
                            memory: req.memory.unwrap_or_default(),
                        };
                        let (image, execution) = heavy.heavy.render_denoise_passes(
                            &plans,
                            &plan,
                            is_raw,
                            &opts,
                            decoder,
                            &req.cancel,
                            &req.preview,
                            on_progress,
                        )?;
                        if first_execution.is_none() {
                            first_execution = Some(execution);
                        }
                        images.push(image);
                    }
                    // The execution record (requested + resolved per-pass values + effective
                    // evaluation accounting), emitted exactly once per generation.
                    if let Some(execution) = first_execution {
                        req.denoise_pass_report
                            .emit(execution.with_requested(req.denoise_passes.as_deref()));
                    }
                    return Ok(GenerationOutput::Images(images));
                }

                // Multi-phase render (epic 13879, sc-13884): drive the resolved phases over the ONE
                // global schedule — per-phase guidance selecting the true-CFG (two-forward) or CFG-off
                // (single-forward) body, AND per-phase adapters toggled on that phase's own job-local
                // DiT clone — sharing the latent + sigma trajectory across every boundary. The per-phase
                // plans (clone + adapters + prep) are built ONCE here (`prep_neg` present per phase iff
                // that phase uses CFG, backed by the `encode_guidance` neg-context gate) and reused
                // across the count loop (one image per seed). Returns before the single-phase dispatch.
                if let (Some(resolved), Some(full)) = (mp_resolved.as_ref(), mp_sigmas.as_ref()) {
                    let plans = self.file_pin_spec.read_files_unchanged(
                        self.adapters.iter().map(|adapter| &adapter.path),
                        || {
                            heavy.heavy.prepare_multiphase(
                                resolved,
                                &self.adapters,
                                &ctx.pos,
                                ctx.neg.as_ref(),
                                req.width,
                                req.height,
                            )
                        },
                    )?;
                    let mut images = Vec::with_capacity(req.count as usize);
                    for n in 0..req.count {
                        let opts = TurboOptions {
                            width: req.width,
                            height: req.height,
                            steps: full.len().saturating_sub(1),
                            seed: base_seed.wrapping_add(n as u64),
                            sampler: req.sampler.clone(),
                            scheduler: req.scheduler.clone(),
                            transformer_window_size,
                            memory: req.memory.unwrap_or_default(),
                        };
                        images.push(heavy.heavy.render_multiphase(
                            &plans,
                            full,
                            &opts,
                            decoder,
                            &req.cancel,
                            &req.preview,
                            on_progress,
                        )?);
                    }
                    return Ok(GenerationOutput::Images(images));
                }

                // Hoist the count-invariant work OUT of the per-image loop (F-073): the reference/pose
                // VAE encodes and the step-invariant text-fusion + host-RoPE prep depend only on the
                // (shared) context + target geometry, NOT the per-seed noise. Build the plan ONCE here;
                // each seed below reuses it via the `render_*_from` seam (byte-identical to the pre-hoist
                // per-seed build — the prep only ever read the latent *shape*). An 8-count two-source
                // edit thus does 2 VAE encodes + 2 preps total, not 16 + 16.
                let plan = if is_edit {
                    // Kontext-style edit (epic 10871): the source image(s) are kept as in-context
                    // conditioning (VAE tokens + the Qwen3-VL grounding baked into `ctx`) — NOT a noised
                    // img2img init. `edit_sources` is the ordered slice the pipeline VAE-encodes at
                    // successive RoPE frames; the `krea2_identity_edit` LoRA in `spec.adapters` steers it.
                    KreaRenderPlan::Edit(heavy.heavy.prepare_edit_plan(
                        &ctx.pos,
                        ctx.neg.as_ref(),
                        &edit_sources,
                        req.width,
                        req.height,
                    )?)
                } else if let Some((init, _)) = reference {
                    // Reference fidelity strength — resolved ONCE above (sc-10121) so the `start` the
                    // capture `keep` was resolved against equals the one `render_*_img2img_from`
                    // truncates on. `Some` here exactly when a reference is present.
                    let strength = img2img_strength
                        .expect("img2img_strength is Some whenever a reference is present");
                    let img2img = heavy.heavy.prepare_img2img(
                        &ctx.pos,
                        ctx.neg.as_ref(),
                        init,
                        req.width,
                        req.height,
                    )?;
                    // img2img dispatch splits by variant: Raw takes the true-CFG entrypoint (guidance +
                    // negative prompt honored, sc-10224); Turbo the CFG-free distilled one (sc-10135).
                    if is_raw {
                        KreaRenderPlan::Img2ImgRaw {
                            plan: img2img,
                            strength,
                        }
                    } else {
                        KreaRenderPlan::Img2ImgTurbo {
                            plan: img2img,
                            strength,
                        }
                    }
                } else if is_raw {
                    KreaRenderPlan::BaseCfg(heavy.heavy.prepare_t2i(
                        &ctx.pos,
                        ctx.neg.as_ref(),
                        req.width,
                        req.height,
                    )?)
                } else {
                    // Turbo t2i is CFG-free (`ctx.neg` is always `None` here).
                    KreaRenderPlan::Turbo(
                        heavy
                            .heavy
                            .prepare_t2i(&ctx.pos, None, req.width, req.height)?,
                    )
                };

                let mut images = Vec::with_capacity(req.count as usize);
                for n in 0..req.count {
                    let opts = TurboOptions {
                        width: req.width,
                        height: req.height,
                        steps,
                        seed: base_seed.wrapping_add(n as u64),
                        sampler: req.sampler.clone(),
                        scheduler: req.scheduler.clone(),
                        transformer_window_size,
                        memory: req.memory.unwrap_or_default(),
                    };
                    // The one render body per path (sc-11101): the same `KreaHeavy::render_*_from` for
                    // both residencies, so a Sequential job (text phase already dropped) is byte-identical
                    // to Resident.
                    let img = match &plan {
                        KreaRenderPlan::Edit(p) => heavy.heavy.render_edit_from(
                            p,
                            guidance,
                            // Distilled Turbo edit → few-step `turbo_schedule`; Raw edit → dynamic-mu
                            // `base_schedule`. Matches the capture-σ `sigmas` selector above (`is_raw`).
                            is_turbo_edit,
                            &opts,
                            decoder,
                            // Honor the PiD `from_ldm` early-stop on the edit path (F-069): `keep`
                            // truncates the schedule so the decoder built at `capture_sigma` receives the
                            // partially-denoised latent it expects, instead of the σ=0 clean one.
                            keep,
                            &req.cancel,
                            &req.preview,
                            on_progress,
                        )?,
                        KreaRenderPlan::Img2ImgRaw { plan, strength } => {
                            heavy.heavy.render_base_img2img_from(
                                plan,
                                guidance,
                                *strength,
                                &opts,
                                decoder,
                                // from_ldm early-stop (sc-10121): `keep` truncates the img2img-sliced
                                // schedule so the decoder built at `capture_sigma` gets the matching
                                // partially-denoised latent; `sigmas.len()` (no capture) runs the tail.
                                keep,
                                &req.cancel,
                                &req.preview,
                                on_progress,
                            )?
                        }
                        KreaRenderPlan::Img2ImgTurbo { plan, strength } => {
                            heavy.heavy.render_turbo_img2img_from(
                                plan,
                                *strength,
                                &opts,
                                decoder,
                                // from_ldm early-stop (sc-10121): see the Raw arm above.
                                keep,
                                &req.cancel,
                                &req.preview,
                                on_progress,
                            )?
                        }
                        KreaRenderPlan::BaseCfg(p) => heavy.heavy.render_base_from(
                            p,
                            guidance,
                            &opts,
                            decoder,
                            keep,
                            &req.cancel,
                            &req.preview,
                            on_progress,
                        )?,
                        KreaRenderPlan::Turbo(p) => heavy.heavy.render_turbo_from(
                            p,
                            &opts,
                            decoder,
                            keep,
                            &req.cancel,
                            &req.preview,
                            on_progress,
                        )?,
                    };
                    images.push(img);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded weights.
/// Layers Krea's model-specific constraints (non-empty prompt, size multiple-of-16, steps ≥ 1) on top
/// of the shared [`Capabilities::validate_request`] floor (count/size range, negative/guidance/true_cfg
/// flags, conditioning kinds).
/// # Callable without weights, and deliberately `pub`
///
/// Every rule in this function reads only `desc` and `req` — there is no `&self`,
/// no loaded generator and no tensor. It was `pub(crate)`, which made it
/// unreachable to a caller that wants to type-check a request *before* paying for
/// a load, so such a caller had no option but to re-implement these rules and
/// maintain a copy that drifts.
///
/// That is not hypothetical: SceneWorks' Aether Studio mirrors this function by
/// hand for exactly that reason, and keeps a test that runs the engine's own
/// [`Capabilities::validate_request`] over the same corpus to detect the drift it
/// cannot prevent. Making this `pub` lets that mirror be deleted rather than
/// maintained.
pub fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    let id = desc.id;
    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    desc.capabilities.validate_request(id, req)?;
    let img2img_id = match id {
        KREA_2_EDIT_ID => Some(KREA_2_RAW_ID),
        KREA_2_TURBO_EDIT_ID => Some(KREA_2_TURBO_ID),
        _ => None,
    };
    if let Some(img2img_id) = img2img_id {
        let reference_strength = req.conditioning.iter().any(|c| {
            matches!(
                c,
                Conditioning::Reference {
                    strength: Some(_),
                    ..
                }
            )
        });
        if req.strength.is_some() || reference_strength {
            return Err(Error::Msg(format!(
                "{id}: strength is not supported for edit conditioning; use {img2img_id} for img2img strength"
            )));
        }
    }
    if req.steps == Some(0) {
        return Err(Error::Msg(format!("{id}: steps must be >= 1")));
    }
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    validate_phases(id, req)?;
    Ok(())
}

/// Multi-phase request validation (epic 13879, sc-13884). Multi-phase is the **Raw t2i** variant only:
/// an ordered phase list run from pure noise over ONE global schedule, each phase a contiguous step
/// slice with its own guidance (per-phase CFG on/off) AND its own adapter set (per-phase LoRA/LoKr
/// toggling — the "N steps Raw + M steps Raw+turbo-LoRA" workflow). Rejects, loudly:
/// - phases on any non-Raw variant (Turbo is CFG-free single-phase; edit/control are out of scope);
/// - phases combined with reference/edit conditioning or the PiD decoder (t2i-from-noise only in v1);
/// - an empty phase list or a 0-step phase (a malformed trajectory).
///
/// Per-phase **adapter index bounds** (against the model's loaded adapter set) are checked at
/// `generate` time by [`crate::multiphase::resolve_phases`] (the count isn't on the descriptor). A
/// no-op when `req.phases` is `None` (the ordinary single-phase render).
fn validate_phases(id: &str, req: &GenerationRequest) -> Result<()> {
    let Some(phases) = req.phases.as_ref() else {
        return Ok(());
    };
    if id != KREA_2_RAW_ID {
        return Err(Error::Msg(format!(
            "{id}: multi-phase denoise (phases) is supported on {KREA_2_RAW_ID} only"
        )));
    }
    if phases.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: phases must contain at least one phase"
        )));
    }
    if !req.conditioning.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: multi-phase denoise renders from pure noise — reference/edit conditioning is not \
             supported (sc-13884 v1)"
        )));
    }
    if req.use_pid {
        return Err(Error::Msg(format!(
            "{id}: multi-phase denoise does not support the PiD decoder yet (sc-13884 follow-on)"
        )));
    }
    for (i, ph) in phases.iter().enumerate() {
        if ph.steps == 0 {
            return Err(Error::Msg(format!(
                "{id}: phase {i} must run at least one step"
            )));
        }
    }
    Ok(())
}

/// Reject a multi-phase (`phases`) request on a model loaded with a **diff-patch** adapter (sc-13884).
/// The per-phase adapter toggle clears + re-installs low-rank residuals on a job-local DiT clone, but a
/// `.diff`/`.diff_b` diff-patch delta folds irreversibly into the dense base at load (`W += δ`) — every
/// clone inherits it and `clear_adapters` cannot undo it, so a "base-only" phase would silently carry
/// the diff-patch (a wrong render, no error). Turn that silent-wrong into a loud reject. Low-rank
/// LoRA/LoKr adapters — including the epic's rank-64 turbo LoRA — toggle cleanly and are allowed.
/// Factored as a free fn (id + the load-time flag) so the reject is unit-testable without a loaded
/// model. A no-op when `req.phases` is `None` or the model has no diff-patch adapter.
fn ensure_multiphase_allowed_for(
    id: &str,
    has_diff_patch: bool,
    req: &GenerationRequest,
) -> Result<()> {
    if req.phases.is_some() && has_diff_patch {
        return Err(Error::Msg(format!(
            "{id}: multi-phase denoise is not supported on a model loaded with a diff-patch \
             (.diff/.diff_b) adapter — a diff-patch folds irreversibly into the base weights at load \
             and cannot be toggled off for a base-only phase; load a low-rank LoRA/LoKr adapter for \
             multi-phase"
        )));
    }
    Ok(())
}

/// Chained denoise-pass request validation (epic 20414, sc-20418) — the Krea-specific floor on top
/// of the shared `validate_denoise_passes` gate (which already checked capability, arity, ranges,
/// sampler/scheduler ids against the advertised menus, and the `phases` mutual exclusion). The
/// candle twin (`candle-gen-krea::validate_denoise_pass_surface`) applies the same rules.
///
/// Rejects, loudly:
/// - passes combined with reference/edit conditioning or the PiD decoder (t2i-from-noise only in v1
///   — the multiphase precedent);
/// - a guidance-bearing pass on a CFG-free variant (Turbo has no guidance axis, so a per-pass
///   guidance would be silently ignored — the class of silent-wrong this floor exists to close);
/// - higher-rung per-generation memory adaptation (anything beyond `stage_residency`) — the chain
///   host runs the unbounded attention/decode paths, so a selected lever must never be silently
///   dropped (the candle twin rejects the same combination through its memory-rung branch).
fn validate_denoise_pass_surface(
    id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> Result<()> {
    let Some(passes) = req.denoise_passes.as_ref() else {
        return Ok(());
    };
    if !req.conditioning.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: chained denoise passes render from pure noise — reference/edit conditioning is \
             not supported (sc-20418 v1)"
        )));
    }
    if req.use_pid {
        return Err(Error::Msg(format!(
            "{id}: chained denoise passes do not support the PiD decoder yet (sc-20418 follow-on)"
        )));
    }
    if !caps.supports_guidance {
        if let Some(index) = passes.iter().position(|p| p.guidance.is_some()) {
            return Err(Error::Msg(format!(
                "{id}: pass {index} sets guidance, but this variant is CFG-free (no guidance \
                 axis) — use {KREA_2_RAW_ID} for per-pass guidance"
            )));
        }
    }
    let has_higher_rung_controls = req.memory.is_some_and(|memory| {
        mlx_gen::gen_core::GenerationMemory {
            stage_residency: false,
            ..memory
        } != mlx_gen::gen_core::GenerationMemory::default()
    });
    if has_higher_rung_controls {
        return Err(Error::Unsupported(format!(
            "{id}: chained denoise passes do not support per-generation memory adaptation beyond \
             staged residency (sc-20418 v1)"
        )));
    }
    Ok(())
}

/// Reject a chained denoise-pass request carrying **adapter weight overrides** on a model loaded
/// with a diff-patch adapter (the sc-13884 rule, applied to the pass mechanism): the per-pass DiT
/// clone clears the baked adapter stack and re-installs low-rank residuals, but a `.diff` fold
/// lives in the shared dense base and cannot be re-scaled per pass — the override chain would
/// silently keep it at full strength. Chains WITHOUT overrides are fine — they run on the
/// provider's own (folded) DiT untouched.
fn ensure_denoise_pass_adapters_allowed(
    id: &str,
    has_diff_patch: bool,
    req: &GenerationRequest,
) -> Result<()> {
    let overrides = req.denoise_passes.as_ref().is_some_and(|passes| {
        passes
            .iter()
            .any(|p| p.adapters.iter().any(|a| a.weight.is_some()))
    });
    if overrides && has_diff_patch {
        return Err(Error::Msg(format!(
            "{id}: per-pass adapter weight overrides are not supported on a model loaded with a \
             diff-patch (.diff/.diff_b) adapter — the fold is irreversible, so a pass-local \
             re-adaptation would silently drop it; run the chain without adapter overrides, or \
             load low-rank LoRA/LoKr adapters"
        )));
    }
    Ok(())
}

/// The Krea model defaults the chained denoise-pass resolution ladder bottoms out on (sc-20418):
/// the byte-exact single-pass behaviour, expressed as explicit menu-valid ids. `euler` is the
/// solver the drivers fall back to when no sampler is named; `flow_match` is the advertised native
/// alias that resolves to the family's own schedule. Mirrors the candle twin exactly.
fn krea_denoise_defaults(is_raw: bool) -> mlx_gen::gen_core::DenoiseDefaults {
    if is_raw {
        mlx_gen::gen_core::DenoiseDefaults::new(DEFAULT_RAW_STEPS, "euler", "flow_match")
            .with_guidance(DEFAULT_RAW_GUIDANCE)
    } else {
        mlx_gen::gen_core::DenoiseDefaults::new(DEFAULT_STEPS, "euler", "flow_match")
    }
}

/// Extract the single reference image + its optional `strength` for img2img (epic 8588 slice A), or
/// `None` for plain txt2img. Krea conditions on exactly one reference image; `MultiReference` or more
/// than one `Reference` errors. Both variants advertise `Reference` (Turbo → CFG-free img2img sc-10135,
/// Raw → CFG img2img sc-10224), so this is reached on either path; the `generate_impl` dispatch then
/// picks the matching entrypoint by `is_raw`. Mirrors the FLUX single-reference idiom.
fn single_reference<'a>(
    id: &str,
    req: &'a GenerationRequest,
) -> Result<Option<(&'a Image, Option<f32>)>> {
    match req.conditioning.as_slice() {
        [] => Ok(None),
        [Conditioning::Reference { image, strength }] => Ok(Some((image, *strength))),
        // F-076: name the actual variant (`krea_2_raw` reaches this too), not a hardcoded `krea_2_turbo`.
        _ => Err(Error::Msg(format!(
            "{id}: img2img supports exactly one Reference image"
        ))),
    }
}

/// The per-request, count-invariant render plan (F-073), built ONCE from the shared context + target
/// geometry before the count loop and reused for every seed via the `KreaHeavy::render_*_from` seam. It
/// carries the hoisted heavy work (reference/pose VAE encodes + the step-invariant DiT prep(s)); only
/// the per-seed noise varies across the loop. The variant mirrors `generate_impl`'s path dispatch.
enum KreaRenderPlan {
    Edit(EditPlan),
    Img2ImgRaw { plan: Img2ImgPlan, strength: f32 },
    Img2ImgTurbo { plan: Img2ImgPlan, strength: f32 },
    BaseCfg(T2iPlan),
    Turbo(T2iPlan),
}

/// The most reference images a Krea edit accepts (epic 10871 P1.3): scene = image 1, person = image 2.
/// The edit LoRA was trained on this fixed pair order — swapping degrades identity — and the ComfyUI-
/// Krea2Edit node caps at two. Mirrors candle-gen-krea's `MAX_EDIT_REFERENCES`.
const MAX_EDIT_REFERENCES: usize = 2;

/// The ordered source image(s) for a Krea edit (epic 10871): one `Conditioning::Reference` (the common
/// single-source edit) or one `Conditioning::MultiReference` (scene, then person — the fixed P1.3
/// order). At least one is required; at most [`MAX_EDIT_REFERENCES`]. Distinct from [`single_reference`]
/// (img2img), which rejects `MultiReference` and any count > 1 — the edit surface is the only one that
/// advertises `MultiReference` ([`edit_descriptor`]). The returned slice is passed straight to
/// [`crate::pipeline::KreaPipeline::generate_edit_with_progress`], which VAE-encodes each at
/// successive RoPE frames.
fn edit_references(req: &GenerationRequest) -> Result<Vec<&Image>> {
    let sources: Vec<&Image> = match req.conditioning.as_slice() {
        [Conditioning::Reference { image, .. }] => vec![image],
        [Conditioning::MultiReference { images }] => images.iter().collect(),
        [] => {
            return Err(Error::Msg(format!(
                "{KREA_2_EDIT_ID}: edit requires a source image (a Reference or a MultiReference)"
            )))
        }
        _ => {
            return Err(Error::Msg(format!(
                "{KREA_2_EDIT_ID}: edit expects a single Reference or one MultiReference of sources"
            )))
        }
    };
    if sources.is_empty() {
        return Err(Error::Msg(format!(
            "{KREA_2_EDIT_ID}: edit requires at least one source image"
        )));
    }
    if sources.len() > MAX_EDIT_REFERENCES {
        return Err(Error::Msg(format!(
            "{KREA_2_EDIT_ID}: at most {MAX_EDIT_REFERENCES} references are supported \
             (scene = image 1, person = image 2)"
        )));
    }
    Ok(sources)
}

// The registration constants bridge the crate's rich `Result` into backend-neutral
// `gen_core::Result`. Four variants register
// here — `krea_2_turbo` (distilled t2i, CFG-free), `krea_2_raw` (undistilled t2i, full-CFG; epic 9992),
// `krea_2_edit` (the Raw pipeline routed to the Kontext edit entrypoint; epic 10871), and
// `krea_2_turbo_edit` (that edit surface on the distilled few-step CFG-free schedule; sc-11640).
/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate's staged-residency split — the
/// Route-exact conditioning plus the DiT (`transformer/`) and Qwen-Image VAE (`vae/`). Every route
/// materializes the selected Qwen3 language tower; edit/turbo-edit also materialize the checkpoint-
/// coupled builtin vision side. The control checkpoint itself is folded by the worker.
pub(crate) fn component_footprint_for(
    provider_id: &str,
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    if provider_id != crate::model_control::KREA_2_TURBO_CONTROL_ID {
        validate_base_krea_load_axes(spec, provider_id)
            .map_err(|error| mlx_gen::gen_core::Error::Msg(error.to_string()))?;
    }
    let base = mlx_gen::require_base_snapshot(spec, "krea_2 imported provider")?;
    let expected_language_bits = match &spec.weights {
        WeightsSource::Dir(root) => resolve_load_plan(spec, root, provider_id)?
            .effective_quant
            .map(Quant::bits),
        WeightsSource::File(_) => {
            if provider_id == crate::model_control::KREA_2_TURBO_CONTROL_ID {
                crate::model_control::validate_control_spec(spec)
                    .map_err(|error| mlx_gen::gen_core::Error::Msg(error.to_string()))?;
            } else {
                validate_native_krea_spec(spec, provider_id)
                    .map_err(|error| mlx_gen::gen_core::Error::Msg(error.to_string()))?;
            }
            native_text_encoder_expected_quant_bits(base)?
        }
    };
    let selected = ENCODER_CONTRACT.source_for_load(spec, base)?;
    let language_bytes =
        selected_language_resident_bytes(&selected, expected_language_bits, provider_id)?;
    let mut vision_bytes = 0;
    if provider_id == KREA_2_EDIT_ID || provider_id == KREA_2_TURBO_EDIT_ID {
        let builtin = ENCODER_CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(base.join("text_encoder")), base)?;
        let vision = builtin
            .materialized_vision_tensor_headers(&VISION_ENCODER_CONTRACT, &ENCODER_CONTRACT)?;
        vision_bytes = mlx_gen::asset_facts::projected_tensor_headers_bytes(&vision, |_| {
            mlx_gen::asset_facts::ResidentProjection::Stored
        })?;
    }
    let text_encoder = language_bytes.checked_add(vision_bytes).ok_or_else(|| {
        mlx_gen::gen_core::Error::Msg(format!(
            "{provider_id}: selected language plus builtin vision resident byte overflow"
        ))
    })?;
    match &spec.weights {
        WeightsSource::Dir(_) => {
            let mut footprint = mlx_gen::PerComponentBytes::from_spec_subdirs(
                spec,
                &["text_encoder"],
                &["transformer"],
                &["vae"],
            )?;
            footprint.text_encoder = text_encoder;
            Ok(footprint)
        }
        WeightsSource::File(dit) => Ok(mlx_gen::PerComponentBytes {
            text_encoder,
            dit: mlx_gen::safetensors_path_bytes(dit),
            vae: mlx_gen::safetensors_path_bytes(base.join("vae")),
        }),
    }
}

pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    component_footprint_for(KREA_2_TURBO_ID, spec)
}

pub(crate) fn raw_component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    component_footprint_for(KREA_2_RAW_ID, spec)
}

pub(crate) fn edit_component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    component_footprint_for(KREA_2_EDIT_ID, spec)
}

pub(crate) fn turbo_edit_component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    component_footprint_for(KREA_2_TURBO_EDIT_ID, spec)
}

pub(crate) fn control_component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    component_footprint_for(crate::model_control::KREA_2_TURBO_CONTROL_ID, spec)
}

mlx_gen::register_generators! {
    pub(crate) const TURBO_REGISTRATION = descriptor => load;
    footprint = component_footprint
}

macro_rules! memory_registration {
    ($name:ident, $behavior:ident, $provider_id:expr) => {
        pub const $name: mlx_gen::gen_core::MemoryRegistration =
            mlx_gen::gen_core::MemoryRegistration {
                provider_id: $provider_id,
                contract: |spec| {
                    crate::block_memory_strategy::memory_strategy_contract($provider_id, spec)
                },
                safety_check: crate::block_memory_strategy::registered_safety_check,
            };
        pub const $behavior: mlx_gen::gen_core::MemoryBehaviorRegistration =
            mlx_gen::gen_core::MemoryBehaviorRegistration {
                provider_id: $provider_id,
                valid_fixtures: crate::block_memory_strategy::registered_valid_fixture,
                begin_request: |spec, contract, context| {
                    crate::block_memory_strategy::registered_begin_request(
                        $provider_id,
                        spec,
                        contract,
                        context,
                    )
                },
            };
    };
}

memory_registration!(
    TURBO_MEMORY_REGISTRATION,
    TURBO_MEMORY_BEHAVIOR,
    KREA_2_TURBO_ID
);
memory_registration!(RAW_MEMORY_REGISTRATION, RAW_MEMORY_BEHAVIOR, KREA_2_RAW_ID);
memory_registration!(
    EDIT_MEMORY_REGISTRATION,
    EDIT_MEMORY_BEHAVIOR,
    KREA_2_EDIT_ID
);
memory_registration!(
    TURBO_EDIT_MEMORY_REGISTRATION,
    TURBO_EDIT_MEMORY_BEHAVIOR,
    KREA_2_TURBO_EDIT_ID
);
mlx_gen::register_generators! {
    pub(crate) const RAW_REGISTRATION = raw_descriptor => load_raw;
    footprint = raw_component_footprint
}
mlx_gen::register_generators! {
    pub(crate) const EDIT_REGISTRATION = edit_descriptor => load_edit;
    footprint = edit_component_footprint
}
mlx_gen::register_generators! {
    pub(crate) const TURBO_EDIT_REGISTRATION = turbo_edit_descriptor => load_turbo_edit;
    footprint = turbo_edit_component_footprint
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{AdapterKind, AdapterSpec, OffloadPolicy};
    use std::path::PathBuf;

    fn write_minimal_safetensors(path: &Path) {
        write_named_safetensors(path, "probe");
    }

    fn write_named_safetensors(path: &Path, tensor: &str) {
        let mut header =
            format!(r#"{{"{tensor}":{{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}}}"#)
                .into_bytes();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 2]);
        std::fs::write(path, bytes).expect("write minimal safetensors");
    }

    fn footprint_snapshot(tmp: &tempfile::TempDir) -> PathBuf {
        let root = tmp.path().join("footprint-base");
        for component in ["transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_minimal_safetensors(&dir.join("model.safetensors"));
        }
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &root.join("text_encoder"),
            ENCODER_CONTRACT,
            VISION_ENCODER_CONTRACT,
        )
        .unwrap();
        root
    }

    #[test]
    fn registry_footprints_price_language_only_except_edit_builtin_vision() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = crate::provider_registry().unwrap();
        let root = footprint_snapshot(&tmp);
        let base_spec = LoadSpec::new(WeightsSource::Dir(root));
        let footprint =
            |id: &str, spec: &LoadSpec| registry.footprint(id, spec).unwrap().unwrap().text_encoder;
        let t2i = footprint(KREA_2_TURBO_ID, &base_spec);
        assert_eq!(footprint(KREA_2_RAW_ID, &base_spec), t2i);
        assert_eq!(
            footprint(crate::model_control::KREA_2_TURBO_CONTROL_ID, &base_spec),
            t2i
        );
        let edit = footprint(KREA_2_EDIT_ID, &base_spec);
        assert_eq!(footprint(KREA_2_TURBO_EDIT_ID, &base_spec), edit);
        assert!(edit > t2i);

        let language_only = tmp.path().join("alternate-language");
        gen_core_testkit::write_encoder_contract_fixture(&language_only, ENCODER_CONTRACT).unwrap();
        let complete = tmp.path().join("alternate-complete");
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &complete.join("text_encoder"),
            ENCODER_CONTRACT,
            VISION_ENCODER_CONTRACT,
        )
        .unwrap();
        let language_spec = base_spec
            .clone()
            .with_text_encoder(WeightsSource::Dir(language_only));
        let complete_spec = base_spec
            .clone()
            .with_text_encoder(WeightsSource::Dir(complete));
        for id in [
            KREA_2_TURBO_ID,
            KREA_2_RAW_ID,
            KREA_2_EDIT_ID,
            KREA_2_TURBO_EDIT_ID,
            crate::model_control::KREA_2_TURBO_CONTROL_ID,
        ] {
            assert_eq!(
                footprint(id, &language_spec),
                footprint(id, &complete_spec),
                "{id}: selected visual tensors are ignored and must not be priced"
            );
        }
        assert_eq!(
            footprint(KREA_2_EDIT_ID, &language_spec) - footprint(KREA_2_TURBO_ID, &language_spec),
            edit - t2i,
            "edit adds the builtin vision side exactly once"
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum FootprintEncoderSelection {
        Builtin,
        ComponentDir,
        ComponentFile,
        CompleteSnapshot,
    }

    fn selected_footprint_spec(
        base_spec: &LoadSpec,
        fixture: &Path,
        selection: FootprintEncoderSelection,
        packed_bits: Option<i32>,
    ) -> LoadSpec {
        let mut spec = base_spec.clone();
        let selected = fixture.join(format!("selected-{selection:?}"));
        spec.text_encoder = match selection {
            FootprintEncoderSelection::Builtin => None,
            FootprintEncoderSelection::ComponentDir => {
                gen_core_testkit::write_encoder_contract_fixture_with_quant(
                    &selected,
                    ENCODER_CONTRACT,
                    packed_bits,
                )
                .unwrap();
                Some(WeightsSource::Dir(selected))
            }
            FootprintEncoderSelection::ComponentFile => {
                gen_core_testkit::write_encoder_contract_fixture_with_quant(
                    &selected,
                    ENCODER_CONTRACT,
                    packed_bits,
                )
                .unwrap();
                Some(WeightsSource::File(selected.join("model.safetensors")))
            }
            FootprintEncoderSelection::CompleteSnapshot => {
                gen_core_testkit::write_encoder_contract_fixture_with_quant(
                    &selected.join("text_encoder"),
                    ENCODER_CONTRACT,
                    packed_bits,
                )
                .unwrap();
                Some(WeightsSource::Dir(selected))
            }
        };
        spec
    }

    fn expected_language_bytes(spec: &LoadSpec, bits: Option<i32>, provider_id: &str) -> u64 {
        let base = mlx_gen::require_base_snapshot(spec, provider_id).unwrap();
        let selected = ENCODER_CONTRACT.source_for_load(spec, base).unwrap();
        let action = selected.load_time_quant_bits(bits, provider_id).unwrap();
        let headers = selected
            .materialized_language_tensor_headers(&ENCODER_CONTRACT)
            .unwrap();
        mlx_gen::asset_facts::projected_tensor_headers_bytes(&headers, |tensor| {
            if let Some(bits) =
                action.filter(|_| crate::convert::is_text_encoder_quant_target(&tensor.name))
            {
                mlx_gen::asset_facts::ResidentProjection::GroupQuantized {
                    bits,
                    group_size: crate::quant::GROUP_SIZE as usize,
                }
            } else {
                mlx_gen::asset_facts::ResidentProjection::Stored
            }
        })
        .unwrap()
    }

    fn expected_builtin_vision_bytes(root: &Path) -> u64 {
        let builtin = ENCODER_CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)
            .unwrap();
        let headers = builtin
            .materialized_vision_tensor_headers(&VISION_ENCODER_CONTRACT, &ENCODER_CONTRACT)
            .unwrap();
        mlx_gen::asset_facts::projected_tensor_headers_bytes(&headers, |_| {
            mlx_gen::asset_facts::ResidentProjection::Stored
        })
        .unwrap()
    }

    #[test]
    fn all_registered_dir_routes_price_selected_language_at_the_effective_base_tier() {
        let registry = crate::provider_registry().unwrap();
        let routes = [
            KREA_2_TURBO_ID,
            KREA_2_RAW_ID,
            KREA_2_EDIT_ID,
            KREA_2_TURBO_EDIT_ID,
            crate::model_control::KREA_2_TURBO_CONTROL_ID,
        ];
        for (quant, bits) in [(Quant::Q4, 4), (Quant::Q8, 8)] {
            for requested in [false, true] {
                for selection in [
                    FootprintEncoderSelection::Builtin,
                    FootprintEncoderSelection::ComponentDir,
                    FootprintEncoderSelection::ComponentFile,
                    FootprintEncoderSelection::CompleteSnapshot,
                ] {
                    let tmp = tempfile::tempdir().unwrap();
                    let root = footprint_snapshot(&tmp);
                    std::fs::write(
                        root.join("transformer/config.json"),
                        if requested {
                            "{}".to_owned()
                        } else {
                            format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#)
                        },
                    )
                    .unwrap();
                    let control = tmp.path().join("control");
                    std::fs::create_dir_all(&control).unwrap();
                    write_minimal_safetensors(&control.join("model.safetensors"));
                    let mut base_spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
                    if requested {
                        base_spec.quantize = Some(quant);
                    }
                    let spec = selected_footprint_spec(
                        &base_spec,
                        &tmp.path().join(format!("{bits}-{requested}")),
                        selection,
                        None,
                    );
                    let language = expected_language_bytes(&spec, Some(bits), KREA_2_TURBO_ID);
                    let vision = expected_builtin_vision_bytes(&root);

                    for id in routes {
                        let route_spec = if id == crate::model_control::KREA_2_TURBO_CONTROL_ID {
                            spec.clone()
                                .with_control(WeightsSource::Dir(control.clone()))
                        } else {
                            spec.clone()
                        };
                        let footprint = registry
                            .footprint(id, &route_spec)
                            .unwrap_or_else(|error| {
                                panic!("Q{bits} requested={requested} {selection:?} {id}: {error}")
                            })
                            .expect("registered Krea route exposes a footprint");
                        let expected_conditioning =
                            if matches!(id, KREA_2_EDIT_ID | KREA_2_TURBO_EDIT_ID) {
                                language + vision
                            } else {
                                language
                            };
                        assert_eq!(
                            footprint.text_encoder, expected_conditioning,
                            "Q{bits} requested={requested} {selection:?} {id}"
                        );
                        let contract = registry
                            .memory_strategy_contract(id, &route_spec)
                            .unwrap_or_else(|error| {
                                panic!("Q{bits} requested={requested} {selection:?} {id}: {error}")
                            })
                            .expect("registered Krea route exposes a memory contract");
                        assert_eq!(
                            contract.asset_facts.conditioning_bytes,
                            expected_conditioning
                        );
                        assert!(contract.asset_facts.transformer_bytes > 0);
                        assert!(contract.asset_facts.transformer_bytes <= footprint.dit);
                        assert!(contract.asset_facts.decoder_bytes > 0);
                        assert!(contract.asset_facts.decoder_bytes <= footprint.vae);
                        assert_eq!(
                            contract.asset_facts.base_bytes,
                            contract.asset_facts.conditioning_bytes
                                + contract.asset_facts.transformer_bytes
                                + contract.asset_facts.decoder_bytes
                        );
                        if id == crate::model_control::KREA_2_TURBO_CONTROL_ID {
                            assert!(contract.asset_facts.overlay_bytes > 0);
                        } else {
                            assert_eq!(contract.asset_facts.overlay_bytes, 0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn all_registered_dir_routes_preserve_matching_packs_and_reject_mismatches() {
        let registry = crate::provider_registry().unwrap();
        let routes = [
            KREA_2_TURBO_ID,
            KREA_2_RAW_ID,
            KREA_2_EDIT_ID,
            KREA_2_TURBO_EDIT_ID,
            crate::model_control::KREA_2_TURBO_CONTROL_ID,
        ];
        for bits in [4, 8] {
            for selection in [
                FootprintEncoderSelection::ComponentDir,
                FootprintEncoderSelection::ComponentFile,
                FootprintEncoderSelection::CompleteSnapshot,
            ] {
                let tmp = tempfile::tempdir().unwrap();
                let root = footprint_snapshot(&tmp);
                std::fs::write(
                    root.join("transformer/config.json"),
                    format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
                )
                .unwrap();
                let control = tmp.path().join("control");
                std::fs::create_dir_all(&control).unwrap();
                write_minimal_safetensors(&control.join("model.safetensors"));
                let base_spec = LoadSpec::new(WeightsSource::Dir(root));
                let matching = selected_footprint_spec(
                    &base_spec,
                    &tmp.path().join("matching"),
                    selection,
                    Some(bits),
                );
                for id in routes {
                    let route_spec = if id == crate::model_control::KREA_2_TURBO_CONTROL_ID {
                        matching
                            .clone()
                            .with_control(WeightsSource::Dir(control.clone()))
                    } else {
                        matching.clone()
                    };
                    assert!(
                        registry.footprint(id, &route_spec).unwrap().is_some(),
                        "Q{bits} {selection:?} {id}"
                    );
                    assert!(registry
                        .memory_strategy_contract(id, &route_spec)
                        .unwrap()
                        .is_some());
                }

                let mismatching = selected_footprint_spec(
                    &base_spec,
                    &tmp.path().join("mismatching"),
                    selection,
                    Some(if bits == 4 { 8 } else { 4 }),
                );
                for id in routes {
                    let route_spec = if id == crate::model_control::KREA_2_TURBO_CONTROL_ID {
                        mismatching
                            .clone()
                            .with_control(WeightsSource::Dir(control.clone()))
                    } else {
                        mismatching.clone()
                    };
                    let footprint_error =
                        registry.footprint(id, &route_spec).unwrap_err().to_string();
                    assert!(
                        footprint_error.contains("pre-quantized")
                            && footprint_error.contains("model policy"),
                        "Q{bits} {selection:?} {id}: {footprint_error}"
                    );
                    assert!(footprint_error.contains(id), "{id}: {footprint_error}");
                    let contract_error = registry
                        .memory_strategy_contract(id, &route_spec)
                        .unwrap_err()
                        .to_string();
                    assert!(
                        contract_error.contains("pre-quantized")
                            && contract_error.contains("model policy"),
                        "Q{bits} {selection:?} {id}: {contract_error}"
                    );
                    assert!(contract_error.contains(id), "{id}: {contract_error}");
                }
            }
        }
    }

    fn complete_native_file_spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let base = tmp.path().join("base");
        for component in ["text_encoder", "vae"] {
            let dir = base.join(component);
            std::fs::create_dir_all(&dir).expect("create base component");
            write_minimal_safetensors(&dir.join("model.safetensors"));
        }
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &base.join("text_encoder"),
            ENCODER_CONTRACT,
            VISION_ENCODER_CONTRACT,
        )
        .expect("validation-complete text encoder fixture");
        let dit = tmp.path().join("imported-krea.safetensors");
        write_minimal_safetensors(&dit);
        LoadSpec::new(WeightsSource::File(dit))
            .with_component(BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
    }

    #[test]
    fn native_file_base_and_pose_routes_follow_the_borrowed_encoder_tier() {
        let registry = crate::provider_registry().unwrap();
        for bits in [4, 8] {
            for selection in [
                FootprintEncoderSelection::ComponentDir,
                FootprintEncoderSelection::ComponentFile,
                FootprintEncoderSelection::CompleteSnapshot,
            ] {
                let tmp = tempfile::tempdir().unwrap();
                let base_spec = complete_native_file_spec(&tmp);
                let base = mlx_gen::require_base_snapshot(&base_spec, KREA_2_TURBO_ID).unwrap();
                gen_core_testkit::write_multimodal_encoder_contract_fixture_with_quant(
                    &base.join("text_encoder"),
                    ENCODER_CONTRACT,
                    VISION_ENCODER_CONTRACT,
                    Some(bits),
                )
                .unwrap();
                let control = tmp.path().join("native-pose.safetensors");
                write_minimal_safetensors(&control);
                let dense =
                    selected_footprint_spec(&base_spec, &tmp.path().join("dense"), selection, None);
                let expected = expected_language_bytes(&dense, Some(bits), KREA_2_TURBO_ID);
                for id in [
                    KREA_2_TURBO_ID,
                    KREA_2_RAW_ID,
                    KREA_2_EDIT_ID,
                    KREA_2_TURBO_EDIT_ID,
                    crate::model_control::KREA_2_TURBO_CONTROL_ID,
                ] {
                    let route_spec = if id == crate::model_control::KREA_2_TURBO_CONTROL_ID {
                        dense
                            .clone()
                            .with_control(WeightsSource::File(control.clone()))
                    } else {
                        dense.clone()
                    };
                    let footprint = registry.footprint(id, &route_spec).unwrap().unwrap();
                    let expected_conditioning =
                        if matches!(id, KREA_2_EDIT_ID | KREA_2_TURBO_EDIT_ID) {
                            expected + expected_builtin_vision_bytes(base)
                        } else {
                            expected
                        };
                    assert_eq!(
                        footprint.text_encoder, expected_conditioning,
                        "Q{bits} {selection:?} {id}"
                    );
                    let contract = registry
                        .memory_strategy_contract(id, &route_spec)
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        contract.asset_facts.conditioning_bytes,
                        expected_conditioning
                    );
                    assert!(contract.asset_facts.transformer_bytes > 0);
                    assert!(contract.asset_facts.decoder_bytes > 0);
                    assert_eq!(
                        contract.asset_facts.base_bytes,
                        contract.asset_facts.conditioning_bytes
                            + contract.asset_facts.transformer_bytes
                            + contract.asset_facts.decoder_bytes
                    );
                    assert_eq!(
                        contract.asset_facts.overlay_bytes > 0,
                        id == crate::model_control::KREA_2_TURBO_CONTROL_ID
                    );
                }

                let matching = selected_footprint_spec(
                    &base_spec,
                    &tmp.path().join("matching"),
                    selection,
                    Some(bits),
                );
                for id in [
                    KREA_2_TURBO_ID,
                    KREA_2_RAW_ID,
                    KREA_2_EDIT_ID,
                    KREA_2_TURBO_EDIT_ID,
                    crate::model_control::KREA_2_TURBO_CONTROL_ID,
                ] {
                    let route_spec = if id == crate::model_control::KREA_2_TURBO_CONTROL_ID {
                        matching
                            .clone()
                            .with_control(WeightsSource::File(control.clone()))
                    } else {
                        matching.clone()
                    };
                    let expected_conditioning =
                        if matches!(id, KREA_2_EDIT_ID | KREA_2_TURBO_EDIT_ID) {
                            expected + expected_builtin_vision_bytes(base)
                        } else {
                            expected
                        };
                    assert_eq!(
                        registry
                            .footprint(id, &route_spec)
                            .unwrap()
                            .unwrap()
                            .text_encoder,
                        expected_conditioning,
                        "Q{bits} matching {selection:?} {id}"
                    );
                    assert_eq!(
                        registry
                            .memory_strategy_contract(id, &route_spec)
                            .unwrap()
                            .unwrap()
                            .asset_facts
                            .conditioning_bytes,
                        expected_conditioning
                    );
                }
                let mismatch = selected_footprint_spec(
                    &base_spec,
                    &tmp.path().join("mismatch"),
                    selection,
                    Some(if bits == 4 { 8 } else { 4 }),
                );
                for id in [
                    KREA_2_TURBO_ID,
                    KREA_2_RAW_ID,
                    KREA_2_EDIT_ID,
                    KREA_2_TURBO_EDIT_ID,
                    crate::model_control::KREA_2_TURBO_CONTROL_ID,
                ] {
                    let route_spec = if id == crate::model_control::KREA_2_TURBO_CONTROL_ID {
                        mismatch
                            .clone()
                            .with_control(WeightsSource::File(control.clone()))
                    } else {
                        mismatch.clone()
                    };
                    let footprint_error =
                        registry.footprint(id, &route_spec).unwrap_err().to_string();
                    assert!(footprint_error.contains(id), "{id}: {footprint_error}");
                    let contract_error = registry
                        .memory_strategy_contract(id, &route_spec)
                        .unwrap_err()
                        .to_string();
                    assert!(contract_error.contains(id), "{id}: {contract_error}");
                }
            }
        }
    }

    fn incomplete_native_file_fixture(tmp: &tempfile::TempDir) -> (PathBuf, PathBuf) {
        let base = tmp.path().join("incomplete-base");
        std::fs::create_dir_all(base.join("transformer")).expect("create transformer config dir");
        std::fs::write(base.join("transformer/config.json"), "{}")
            .expect("write parseable transformer config");
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &base.join("text_encoder"),
            ENCODER_CONTRACT,
            VISION_ENCODER_CONTRACT,
        )
        .expect("validation-complete text encoder fixture");
        let dit = tmp.path().join("native-dit.safetensors");
        write_minimal_safetensors(&dit);
        (dit, base)
    }

    #[test]
    fn sequential_native_generator_retains_the_constructor_pin() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = complete_native_file_spec(&tmp);
        spec.load_shape = mlx_gen::LoadShape::DeferredMaterialization;
        let native = match &spec.weights {
            WeightsSource::File(path) => path.clone(),
            WeightsSource::Dir(_) => unreachable!(),
        };
        spec.prepare_file_sources().unwrap();
        let prepared = spec
            .weights_file_pin()
            .unwrap()
            .expect("prepared primary token");
        let model = build_native_krea_from_spec(&spec, descriptor()).unwrap();
        assert!(
            model.streamable_transformer,
            "an explicit Sequential + Deferred File load must arm the physical stream"
        );
        assert_eq!(
            model
                .memory_strategy
                .capability(mlx_gen::gen_core::MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            mlx_gen::gen_core::MemoryStrategySupport::Missing,
            "physical File streaming must not inherit the Dir evidence cell"
        );
        let pin = model
            ._native_dit
            .expect("native generator must retain its pin");
        assert_eq!(pin, prepared, "provider must retain the cache-key token");
        assert_eq!(pin.loader_path(), std::path::absolute(&native).unwrap());

        std::fs::write(&native, b"replacement after construction").unwrap();
        let error = pin
            .ensure_unchanged()
            .expect_err("sequential reopen must reject a replacement")
            .to_string();
        assert!(error.contains("changed after load"), "{error}");
    }

    #[test]
    fn native_file_streaming_requires_the_explicit_shape_and_excludes_diff_patches() {
        let tmp = tempfile::tempdir().unwrap();
        let mut eligible = complete_native_file_spec(&tmp);
        eligible.load_shape = mlx_gen::LoadShape::DeferredMaterialization;
        assert!(native_file_streamable(&eligible).unwrap());

        let mut resident = eligible.clone();
        resident.offload_policy = mlx_gen::OffloadPolicy::Resident;
        assert!(!native_file_streamable(&resident).unwrap());
        let mut eager = eligible.clone();
        eager.load_shape = mlx_gen::LoadShape::EagerMaterialization;
        assert!(!native_file_streamable(&eager).unwrap());
        assert!(!native_file_streamable(&eligible.clone().with_quant(mlx_gen::Quant::Q4)).unwrap());

        let lora = tmp.path().join("adapter.safetensors");
        write_named_safetensors(
            &lora,
            "transformer.transformer_blocks.0.attn.to_q.lora_down.weight",
        );
        let mut residual =
            eligible
                .clone()
                .with_adapters(vec![AdapterSpec::new(lora, 1.0, AdapterKind::Lora)]);
        residual.prepare_file_sources().unwrap();
        assert!(
            native_file_streamable(&residual).unwrap(),
            "MLX block streams capture and replay forward-time low-rank adapters"
        );

        let diff = tmp.path().join("diff-patch.safetensors");
        write_named_safetensors(&diff, "diffusion_model.transformer_blocks.0.attn.to_q.diff");
        let mut patched =
            eligible.with_adapters(vec![AdapterSpec::new(diff, 1.0, AdapterKind::Lora)]);
        patched.prepare_file_sources().unwrap();
        assert!(
            !native_file_streamable(&patched).unwrap(),
            "an irreversible diff-patch cannot be replayed from pristine window reopens"
        );
    }

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    fn tiny_image() -> Image {
        Image {
            width: 16,
            height: 16,
            pixels: vec![0u8; 16 * 16 * 3],
        }
    }

    /// A 1024² request carrying `refs` `Reference` conditionings (each with `strength`).
    fn ref_req(refs: usize, strength: Option<f32>) -> GenerationRequest {
        let mut r = req(1024, 1024);
        r.conditioning = (0..refs)
            .map(|_| Conditioning::Reference {
                image: tiny_image(),
                strength,
            })
            .collect();
        r
    }

    #[test]
    fn both_variants_advertise_reference_conditioning() {
        // img2img is now on BOTH variants: Turbo → CFG-free (sc-10135), Raw → CFG (sc-10224).
        assert_eq!(
            descriptor().capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        assert_eq!(
            raw_descriptor().capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
    }

    /// sc-12612: `RES_MULTIPLE` is the pinned stride SceneWorks ties every advertised Krea bucket to.
    /// Pin the value and mutation-check that an off-stride (multiple of 8 not 16) in-range size is
    /// rejected with the stride error, and an on-stride size passes.
    #[test]
    fn size_multiple_is_the_pinned_stride() {
        assert_eq!(RES_MULTIPLE, 16);
        let off = validate_request(&descriptor(), &req(1000, 1024))
            .unwrap_err()
            .to_string();
        assert!(off.contains("multiple of 16"), "got: {off}");
        assert!(validate_request(&descriptor(), &req(1024, 1024)).is_ok());
    }

    #[test]
    fn validate_reference_accepted_on_both_variants() {
        // A single Reference (img2img) validates on Turbo AND Raw (sc-10224). The conditioning-floor
        // checks the KIND is allowed; the exactly-one-Reference count is enforced later by
        // `single_reference` (see `single_reference_extracts_one_or_errors`).
        assert!(validate_request(&descriptor(), &ref_req(1, Some(0.5))).is_ok());
        assert!(validate_request(&raw_descriptor(), &ref_req(1, Some(0.5))).is_ok());
    }

    #[test]
    fn img2img_capture_window_agrees_with_decoder_sigma() {
        // The sc-10121 core invariant, as pure host math (synthetic schedule + `init_time_step` +
        // `flow_capture_for_request`, no weights): an img2img `from_ldm` capture is resolved against the
        // SLICED window the denoise actually runs (`start = init_time_step(strength)`), so the decoder's
        // degrade σ (`capture_sigma`) is EXACTLY the last σ of `full[start..keep]` — never desynced from
        // the truncated latent — and a ceiling that would stop at/before `start` collapses to the clean
        // σ=0 tail instead of a negative/empty window.
        let full: [f32; 9] = [1.0, 0.9, 0.78, 0.64, 0.5, 0.36, 0.22, 0.1, 0.0];
        let steps = full.len() - 1; // 8-step flow-match schedule (len = steps + 1).
        let start_of = |strength: f32| init_time_step(steps, Some(strength)).min(full.len() - 1);
        let capture_req = |ceiling: f32| GenerationRequest {
            use_pid: true,
            pid_capture_sigma: Some(ceiling),
            ..Default::default()
        };

        // strength 0.5 → start = floor(8·0.5) = 4 (σ_start = 0.5); ceiling 0.25 → first σ ≤ 0.25 is
        // full[6] = 0.22 → an ACTIVE capture that still denoises ≥ 1 img2img step.
        let start = start_of(0.5);
        assert_eq!(start, 4);
        let (capture_sigma, keep) = flow_capture_for_request(&capture_req(0.25), &full, start);
        assert!(keep < full.len(), "expected an active early stop");
        assert!(keep > start, "must denoise at least one img2img step");
        // No σ desync: the decoder's σ is exactly the sliced window's terminal σ.
        let window = &full[start..keep];
        assert_eq!(*window.last().unwrap(), capture_sigma);
        assert_eq!(capture_sigma, full[keep - 1]);

        // strength 0.9 → start = 7 (σ_start = 0.1), already below the 0.25 ceiling, so the capture would
        // stop at/before the img2img start → NO benefit → collapse to the clean σ=0 full tail.
        let late = start_of(0.9);
        assert_eq!(late, 7);
        let (late_sigma, late_keep) = flow_capture_for_request(&capture_req(0.25), &full, late);
        assert_eq!(
            late_keep,
            full.len(),
            "no-benefit capture runs the clean tail"
        );
        assert_eq!(late_sigma, 0.0);

        // The reference-less t2i path (start = 0) is unaffected — the same ceiling still resolves.
        let (t2i_sigma, t2i_keep) = flow_capture_for_request(&capture_req(0.25), &full, 0);
        assert!(t2i_keep < full.len());
        assert_eq!(t2i_sigma, full[t2i_keep - 1]);
    }

    #[test]
    fn single_reference_extracts_one_or_errors() {
        // No conditioning → plain txt2img.
        assert!(single_reference(KREA_2_TURBO_ID, &req(1024, 1024))
            .unwrap()
            .is_none());
        // Exactly one → the image + its strength.
        let r1 = ref_req(1, Some(0.4));
        let one = single_reference(KREA_2_TURBO_ID, &r1).unwrap();
        assert_eq!(one.map(|(_, s)| s), Some(Some(0.4)));
        // More than one → error (Krea conditions on a single reference).
        assert!(single_reference(KREA_2_TURBO_ID, &ref_req(2, None)).is_err());
    }

    /// F-076: the img2img single-reference error names the ACTUAL descriptor id — `krea_2_raw` reaches
    /// this path too (both variants advertise `Reference`), so a hardcoded `krea_2_turbo` misled Raw
    /// img2img diagnostics.
    #[test]
    fn single_reference_error_uses_the_descriptor_id() {
        let err = single_reference(KREA_2_RAW_ID, &ref_req(2, None))
            .unwrap_err()
            .to_string();
        assert!(err.contains(KREA_2_RAW_ID), "{err}");
        assert!(!err.contains(KREA_2_TURBO_ID), "{err}");
    }

    #[test]
    fn edit_references_takes_one_reference_or_a_scene_person_pair() {
        // A single `Reference` → one source (the common single-image edit).
        assert_eq!(edit_references(&ref_req(1, None)).unwrap().len(), 1);

        // A `MultiReference` → the ordered source list (scene, then person; P1.3).
        let mut two = req(1024, 1024);
        two.conditioning = vec![Conditioning::MultiReference {
            images: vec![tiny_image(), tiny_image()],
        }];
        assert_eq!(edit_references(&two).unwrap().len(), 2);

        // Empty conditioning → error (an edit needs a source).
        assert!(edit_references(&req(1024, 1024)).is_err());

        // Past the scene/person cap → error naming the fixed order.
        let mut three = req(1024, 1024);
        three.conditioning = vec![Conditioning::MultiReference {
            images: vec![tiny_image(), tiny_image(), tiny_image()],
        }];
        let err = edit_references(&three).unwrap_err().to_string();
        assert!(err.contains("scene") && err.contains("person"), "{err}");

        // Two separate `Reference`s (not a `MultiReference`) → error: an edit takes one Reference or
        // one MultiReference, never a bare list.
        assert!(edit_references(&ref_req(2, None)).is_err());
    }

    #[test]
    fn descriptor_is_krea_2_turbo() {
        let d = descriptor();
        assert_eq!(d.id, "krea_2_turbo");
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // CFG-free distilled Turbo: no guidance, no negative prompt. img2img reference conditioning
        // (sc-10135) IS advertised (the only conditioning surface).
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        // Raw-trained LoRA/LoKr apply at Turbo inference (sc-7911).
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(DEFAULT_STEPS, 8);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn validate_accepts_in_surface() {
        assert!(validate_request(&descriptor(), &req(1024, 1024)).is_ok());
        assert!(validate_request(&descriptor(), &req(2048, 2048)).is_ok());
    }

    #[test]
    fn validate_rejects_empty_prompt_and_bad_size() {
        assert!(validate_request(&descriptor(), &GenerationRequest::default()).is_err());
        for (w, h) in [(1000, 1000), (257, 256)] {
            let e = validate_request(&descriptor(), &req(w, h))
                .unwrap_err()
                .to_string();
            assert!(e.contains("multiple of 16"), "{w}x{h} got: {e}");
        }
        assert!(validate_request(&descriptor(), &req(128, 128)).is_err()); // below min
        assert!(validate_request(&descriptor(), &req(2064, 256)).is_err()); // above max
    }

    #[test]
    fn validate_rejects_guidance_and_negative_prompt() {
        // Turbo is CFG-free: the capability floor rejects a guidance override and a negative prompt.
        assert!(validate_request(
            &descriptor(),
            &GenerationRequest {
                guidance: Some(3.5),
                ..req(512, 512)
            }
        )
        .is_err());
        assert!(validate_request(
            &descriptor(),
            &GenerationRequest {
                negative_prompt: Some("x".into()),
                ..req(512, 512)
            }
        )
        .is_err());
    }

    // ---- Chained denoise passes (epic 20414, sc-20418) -------------------------------------------

    /// The t2i variants advertise the capability and the native scheduler alias the resolution
    /// ladder bottoms out on; the edit variants explicitly do NOT — mirroring the candle twin.
    #[test]
    fn denoise_pass_advertisement_covers_t2i_variants_only() {
        assert!(descriptor().capabilities.supports_denoise_passes);
        assert!(raw_descriptor().capabilities.supports_denoise_passes);
        assert!(!edit_descriptor().capabilities.supports_denoise_passes);
        assert!(!turbo_edit_descriptor().capabilities.supports_denoise_passes);
        for d in [descriptor(), raw_descriptor()] {
            assert!(d.capabilities.schedulers.contains(&"flow_match"));
            assert!(d.capabilities.samplers.contains(&"euler"));
        }
        // The model defaults the ladder resolves to are menu-valid ids with the family's real
        // default steps/guidance.
        let raw = krea_denoise_defaults(true);
        assert_eq!(
            (raw.steps, raw.sampler.as_str(), raw.scheduler.as_str()),
            (DEFAULT_RAW_STEPS, "euler", "flow_match")
        );
        assert_eq!(raw.guidance, Some(DEFAULT_RAW_GUIDANCE));
        let turbo = krea_denoise_defaults(false);
        assert_eq!(turbo.steps, DEFAULT_STEPS);
        assert_eq!(turbo.guidance, None);
    }

    /// The Krea-specific denoise-pass floor: t2i only, no PiD, no guidance on the CFG-free Turbo,
    /// no higher-rung memory adaptation — plus the shared floor's mutual exclusion with `phases`
    /// and the diff-patch override reject. Weights-free (no loaded model).
    #[test]
    fn denoise_pass_surface_rejections_are_typed_and_specific() {
        let pass = mlx_gen::gen_core::DenoisePass {
            steps: Some(2),
            ..Default::default()
        };
        let base = GenerationRequest {
            denoise_passes: Some(vec![pass.clone()]),
            ..req(512, 512)
        };
        let caps = descriptor().capabilities;

        validate_denoise_pass_surface(KREA_2_TURBO_ID, &caps, &base).unwrap();
        validate_request(&descriptor(), &base).unwrap();

        // Reference conditioning is out of the v1 surface.
        let with_ref = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: tiny_image(),
                strength: None,
            }],
            ..base.clone()
        };
        let err = validate_denoise_pass_surface(KREA_2_TURBO_ID, &caps, &with_ref).unwrap_err();
        assert!(err.to_string().contains("pure noise"), "{err}");

        // PiD decode is a follow-on.
        let with_pid = GenerationRequest {
            use_pid: true,
            ..base.clone()
        };
        let err = validate_denoise_pass_surface(KREA_2_TURBO_ID, &caps, &with_pid).unwrap_err();
        assert!(err.to_string().contains("PiD"), "{err}");

        // A guidance-bearing pass on the CFG-free Turbo is a loud reject, not a silent ignore; the
        // same pass is fine on Raw (which has the guidance axis).
        let guided = GenerationRequest {
            denoise_passes: Some(vec![mlx_gen::gen_core::DenoisePass {
                guidance: Some(2.5),
                ..pass.clone()
            }]),
            ..base.clone()
        };
        let err = validate_denoise_pass_surface(KREA_2_TURBO_ID, &caps, &guided).unwrap_err();
        assert!(err.to_string().contains("CFG-free"), "{err}");
        validate_denoise_pass_surface(KREA_2_RAW_ID, &raw_descriptor().capabilities, &guided)
            .unwrap();

        // Higher-rung memory adaptation is rejected on the chain path (typed Unsupported).
        let with_memory = GenerationRequest {
            memory: Some(mlx_gen::gen_core::GenerationMemory {
                chunk_attention: true,
                ..Default::default()
            }),
            ..base.clone()
        };
        let err = validate_denoise_pass_surface(KREA_2_TURBO_ID, &caps, &with_memory).unwrap_err();
        assert!(err.to_string().contains("memory adaptation"), "{err}");
        // ... but staged residency alone is fine (the residency seam owns it).
        let with_staged = GenerationRequest {
            memory: Some(mlx_gen::gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            ..base.clone()
        };
        validate_denoise_pass_surface(KREA_2_TURBO_ID, &caps, &with_staged).unwrap();

        // AC6 guard: `phases` and `denoisePasses` stay mutually exclusive at the shared floor, so
        // a phases request can never reach the pass executor (and vice versa).
        let both = GenerationRequest {
            phases: Some(vec![mlx_gen::gen_core::GenerationPhase {
                steps: 4,
                guidance: None,
                adapters: vec![],
            }]),
            ..base.clone()
        };
        let err = validate_request(&raw_descriptor(), &both).unwrap_err();
        assert!(err.to_string().contains("mutually exclusive"), "{err}");

        // Diff-patch models reject pass-local weight OVERRIDES but accept override-free chains.
        let with_override = GenerationRequest {
            denoise_passes: Some(vec![mlx_gen::gen_core::DenoisePass {
                adapters: vec![mlx_gen::gen_core::PhaseAdapter {
                    adapter: 0,
                    weight: Some(0.5),
                }],
                ..pass.clone()
            }]),
            ..base.clone()
        };
        let err = ensure_denoise_pass_adapters_allowed(KREA_2_TURBO_ID, true, &with_override)
            .unwrap_err();
        assert!(err.to_string().contains("diff-patch"), "{err}");
        ensure_denoise_pass_adapters_allowed(KREA_2_TURBO_ID, true, &base).unwrap();
        ensure_denoise_pass_adapters_allowed(KREA_2_TURBO_ID, false, &with_override).unwrap();
    }

    #[test]
    fn load_accepts_complete_single_file_spec() {
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load(&file).err().expect("error").to_string();
        assert!(e.contains(BASE_SNAPSHOT_COMPONENT), "got: {e}");
        let tmp = tempfile::tempdir().expect("temp dir");
        let complete = complete_native_file_spec(&tmp);
        assert!(
            load(&complete).is_ok(),
            "complete File spec is registry-loadable"
        );
    }

    #[test]
    fn load_accepts_adapter_spec_without_rejecting() {
        // sc-7911: adapters are no longer rejected at the door; a LoadSpec carrying an adapter
        // resolves the snapshot first, so a missing snapshot — not an "unsupported adapters" error —
        // is what surfaces (the real install runs in the #[ignore] real-weight harness).
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into())).with_adapters(vec![
                AdapterSpec::new(
                    std::path::PathBuf::from("/nonexistent-krea/adapter.safetensors"),
                    1.0,
                    AdapterKind::Lora,
                ),
            ]);
        let e = load(&spec).err().expect("error").to_string();
        assert!(
            !e.to_lowercase().contains("not yet supported")
                && !e.to_lowercase().contains("not supported"),
            "adapters must be accepted, got: {e}"
        );
    }

    #[test]
    fn native_load_empty_adapters_preserves_load() {
        // sc-14119: an empty adapter slice keeps the native single-file load behaving as it always has —
        // it still runs the fail-closed base inventory first (and, with an incomplete base, fails there),
        // so the new parameter is inert for the t2i/img2img callers that pass `&[]`.
        let tmp = tempfile::tempdir().unwrap();
        let (dit, base) = incomplete_native_file_fixture(&tmp);
        let e = load_from_native_dit_file(&dit, &base, &[], descriptor())
            .err()
            .expect("missing base snapshot → err")
            .to_string();
        assert!(
            e.contains("native base VAE asset facts"),
            "expected the missing-base inventory error, got: {e}"
        );
    }

    #[test]
    fn native_load_accepts_adapters_without_early_rejection() {
        // sc-14119: a non-empty adapter slice is threaded through the native loader (parity with the
        // snapshot `load` path) and must NOT be rejected at the door. With an incomplete base the load still
        // fails first at the fail-closed base inventory — the adapter fold
        // (`KreaHeavy::apply_adapters`) is weights-gated and exercised in the #[ignore] real-weight
        // harness below.
        let tmp = tempfile::tempdir().unwrap();
        let (dit, base) = incomplete_native_file_fixture(&tmp);
        let adapter = tmp.path().join("krea2_identity_edit.safetensors");
        write_minimal_safetensors(&adapter);
        let adapters = vec![AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)];
        let e = load_from_native_dit_file(&dit, &base, &adapters, edit_descriptor())
            .err()
            .expect("missing base snapshot → err")
            .to_string();
        assert!(
            !e.to_lowercase().contains("not yet supported")
                && !e.to_lowercase().contains("not supported"),
            "adapters must be accepted by the native loader, got: {e}"
        );
        assert!(
            e.contains("native base VAE asset facts"),
            "expected the missing-base inventory error, got: {e}"
        );
    }

    #[test]
    fn prepared_adapter_header_classification_rejects_replacement() {
        let tmp = tempfile::tempdir().unwrap();
        let adapter = tmp.path().join("adapter.safetensors");
        write_minimal_safetensors(&adapter);
        let mut spec =
            LoadSpec::new(WeightsSource::Dir(tmp.path().join("base"))).with_adapters(vec![
                AdapterSpec::new(adapter.clone(), 1.0, AdapterKind::Lora),
            ]);
        spec.prepare_file_sources().unwrap();

        std::fs::write(&adapter, b"replacement adapter bytes").unwrap();
        let error = adapters_have_diff_patch_for_spec(&spec)
            .expect_err("header classification must consume the prepared adapter token")
            .to_string();
        assert!(error.contains("changed after load"), "got: {error}");
    }

    #[test]
    fn native_load_valid_config_reaches_fail_closed_base_asset_sizing() {
        let root_tmp = tempfile::tempdir().unwrap();
        let (dit, root) = incomplete_native_file_fixture(&root_tmp);

        let e = load_from_native_dit_file(&dit, &root, &[], descriptor())
            .err()
            .expect("missing required base components must fail")
            .to_string();
        assert!(
            e.contains("native base VAE asset facts"),
            "expected the fail-closed base asset-sizing stage, got: {e}"
        );
        assert!(!e.contains("config.json"), "config was valid, got: {e}");
    }

    /// Real-weight harness for the native single-file + adapter fold (sc-14119): the discriminating check
    /// the GPU-free tests can't run (the fold needs a real DiT/base). Set `KREA_NATIVE_DIT` to a ComfyUI
    /// single-file Krea 2 DiT, `KREA_TURBO_DIR` to a resident turnkey snapshot root, and
    /// `KREA_EDIT_ADAPTER` to a `krea2_identity_edit` LoRA. Asserts the passed adapter is folded onto the
    /// returned generator (`apply_adapters` invoked) and the `adapters` / `has_diff_patch` fields reflect
    /// it, rather than the old hardcoded empty/false.
    #[test]
    #[ignore = "needs real weights: set KREA_NATIVE_DIT, KREA_TURBO_DIR, KREA_EDIT_ADAPTER"]
    fn native_load_folds_edit_adapter() {
        let dit = std::env::var("KREA_NATIVE_DIT").expect("set KREA_NATIVE_DIT");
        let base = std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR");
        let adapter = std::env::var("KREA_EDIT_ADAPTER").expect("set KREA_EDIT_ADAPTER");
        let adapters = vec![AdapterSpec::new(
            std::path::PathBuf::from(adapter),
            1.0,
            AdapterKind::Lora,
        )];
        let krea = build_native_krea(&dit, &base, &adapters, edit_descriptor())
            .expect("native single-file load + adapter fold");
        // `apply_adapters` ran (a target mismatch would have errored above) and the field is populated
        // from the passed adapters — not the hardcoded empty vec the entrypoint used before sc-14119.
        assert_eq!(
            krea.adapters.len(),
            1,
            "the passed adapter must be retained"
        );
        assert_eq!(
            krea.has_diff_patch,
            adapters_have_diff_patch(&adapters),
            "has_diff_patch must be computed from the passed adapters, not hardcoded"
        );
    }

    #[test]
    fn load_accepts_quant_spec_but_fails_on_missing_weights() {
        for q in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into())).with_quant(q);
            let e = load(&spec).err().expect("error").to_string();
            // The quant is accepted (not the failure); the missing snapshot (the pipeline assembly
            // hits the absent tokenizer/config first) is.
            assert!(
                !e.contains("not supported"),
                "quant should be accepted: {e}"
            );
            assert!(
                e.contains("No such file")
                    || e.contains("config.json")
                    || e.contains("tokenizer")
                    || e.contains("read"),
                "expected a missing-snapshot error, got: {e}"
            );
        }
    }

    #[test]
    fn reachable_via_registry_by_id() {
        assert!(
            crate::provider_registry()
                .unwrap()
                .generators()
                .copied()
                .any(|r| (r.descriptor)().id == KREA_2_TURBO_ID),
            "id {KREA_2_TURBO_ID} not registered"
        );
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into()));
        let e = crate::provider_registry()
            .unwrap()
            .load(KREA_2_TURBO_ID, &spec)
            .err()
            .expect("missing weights → err")
            .to_string();
        assert!(
            !e.contains("no generator registered"),
            "id not resolved: {e}"
        );
    }

    // --- Raw (undistilled, full-CFG) variant — epic 9992 ---

    #[test]
    fn raw_descriptor_is_krea_2_raw_and_cfg_capable() {
        let d = raw_descriptor();
        assert_eq!(d.id, "krea_2_raw");
        // The generator id MUST equal the LoRA-trainer base id (Path 1: one id, both roles).
        assert_eq!(KREA_2_RAW_ID, crate::training::KREA_2_RAW_TRAINER_ID);
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // Undistilled base: real CFG guidance + a user negative prompt (unlike Turbo / Boogu base).
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        // Not guidance-distilled: no separate embedded-guidance axis, so no true_cfg toggle.
        assert!(!d.capabilities.supports_true_cfg);
        // Shared surface stays in lockstep with Turbo (derived from `descriptor()`).
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(d.capabilities.samplers, descriptor().capabilities.samplers);
        assert!(d.capabilities.mac_only);
        assert_eq!(DEFAULT_RAW_STEPS, 52);
        assert_eq!(DEFAULT_RAW_GUIDANCE, 3.5);
    }

    #[test]
    fn raw_validate_accepts_guidance_and_negative_prompt() {
        // The CFG floor that rejects these on Turbo must ACCEPT them on Raw.
        assert!(validate_request(
            &raw_descriptor(),
            &GenerationRequest {
                guidance: Some(3.5),
                negative_prompt: Some("blurry, lowres".into()),
                ..req(1024, 1024)
            }
        )
        .is_ok());
    }

    #[test]
    fn raw_reachable_via_registry_by_id() {
        assert!(
            crate::provider_registry()
                .unwrap()
                .generators()
                .copied()
                .any(|r| (r.descriptor)().id == KREA_2_RAW_ID),
            "id {KREA_2_RAW_ID} not registered"
        );
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load_raw(&file).err().expect("error").to_string();
        assert!(e.contains(BASE_SNAPSHOT_COMPONENT), "got: {e}");
        let tmp = tempfile::tempdir().expect("temp dir");
        let complete = complete_native_file_spec(&tmp);
        assert!(load_raw(&complete).is_ok());
    }

    // --- Image-edit variant (Kontext-style) — epic 10871 ---

    #[test]
    fn edit_descriptor_is_krea_2_edit_and_cfg_capable() {
        let d = edit_descriptor();
        assert_eq!(d.id, "krea_2_edit");
        assert_eq!(d.id, KREA_2_EDIT_ID);
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // Edit shares Raw's full-CFG surface (guidance + negative prompt; an edit denoises from noise
        // under true CFG), derived from `raw_descriptor()` — so it stays in lockstep with Raw.
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        // The source rides a single `Reference`, or a scene+person pair rides a `MultiReference`
        // (epic 10871 P1.3); the `krea2_identity_edit` LoRA rides `spec.adapters`.
        assert_eq!(
            d.capabilities.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
    }

    #[test]
    fn edit_validate_accepts_reference_with_guidance_and_negative() {
        // An edit job: a source Reference + full-CFG knobs must pass the capability floor.
        let mut r = ref_req(1, None);
        r.guidance = Some(3.5);
        r.negative_prompt = Some("blurry, lowres".into());
        assert!(validate_request(&edit_descriptor(), &r).is_ok());
    }

    #[test]
    fn edit_variants_reject_reference_and_request_strength() {
        for (desc, img2img_id) in [
            (edit_descriptor(), KREA_2_RAW_ID),
            (turbo_edit_descriptor(), KREA_2_TURBO_ID),
        ] {
            let reference_err = validate_request(&desc, &ref_req(1, Some(0.5)))
                .unwrap_err()
                .to_string();
            assert!(reference_err.contains(img2img_id), "got: {reference_err}");

            let mut request_strength = ref_req(1, None);
            request_strength.strength = Some(0.5);
            let request_err = validate_request(&desc, &request_strength)
                .unwrap_err()
                .to_string();
            assert!(request_err.contains(img2img_id), "got: {request_err}");
        }
    }

    #[test]
    fn edit_reachable_via_registry_by_id() {
        assert!(
            crate::provider_registry()
                .unwrap()
                .generators()
                .copied()
                .any(|r| (r.descriptor)().id == KREA_2_EDIT_ID),
            "id {KREA_2_EDIT_ID} not registered"
        );
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load_edit(&file).err().expect("error").to_string();
        assert!(e.contains(BASE_SNAPSHOT_COMPONENT), "got: {e}");
        let tmp = tempfile::tempdir().expect("temp dir");
        let complete = complete_native_file_spec(&tmp);
        assert!(load_edit(&complete).is_ok());
    }

    // --- CFG-free Turbo image-edit variant — sc-11640 ---

    #[test]
    fn turbo_edit_descriptor_is_krea_2_turbo_edit_and_cfg_free() {
        let d = turbo_edit_descriptor();
        assert_eq!(d.id, "krea_2_turbo_edit");
        assert_eq!(d.id, KREA_2_TURBO_EDIT_ID);
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // Derived from the distilled Turbo descriptor: CFG-free (no guidance, no user negative prompt),
        // UNLIKE the full-CFG `krea_2_edit`. This is the recipe difference the spike validates.
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        // Same edit conditioning surface as `edit_descriptor` — a single `Reference` or a scene+person
        // `MultiReference`; the `krea2_identity_edit` LoRA rides `spec.adapters`.
        assert_eq!(
            d.capabilities.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        // Same curated sampler/scheduler menu + size bounds as Turbo t2i (shared `descriptor()` base).
        assert_eq!(d.capabilities.samplers, descriptor().capabilities.samplers);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn turbo_edit_rejects_guidance_and_negative_prompt() {
        // The CFG-free floor (like Turbo t2i) rejects both — the edit runs a single conditional forward.
        let mut r = ref_req(1, None);
        r.guidance = Some(3.5);
        assert!(validate_request(&turbo_edit_descriptor(), &r).is_err());
        let mut r = ref_req(1, None);
        r.negative_prompt = Some("blurry".into());
        assert!(validate_request(&turbo_edit_descriptor(), &r).is_err());
        // A source Reference with NO CFG knobs passes the capability floor.
        assert!(validate_request(&turbo_edit_descriptor(), &ref_req(1, None)).is_ok());
    }

    #[test]
    fn turbo_edit_reachable_via_registry_by_id() {
        assert!(
            crate::provider_registry()
                .unwrap()
                .generators()
                .copied()
                .any(|r| (r.descriptor)().id == KREA_2_TURBO_EDIT_ID),
            "id {KREA_2_TURBO_EDIT_ID} not registered"
        );
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load_turbo_edit(&file).err().expect("error").to_string();
        assert!(e.contains(BASE_SNAPSHOT_COMPONENT), "got: {e}");
        let tmp = tempfile::tempdir().expect("temp dir");
        let complete = complete_native_file_spec(&tmp);
        assert!(load_turbo_edit(&complete).is_ok());
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that Krea's dispatch HONORS
    // `offload_policy` — not a smoke test. `build_residency` points at a non-existent snapshot dir
    // (a *directory* source, so the up-front `resolve_root` precision/single-file guard passes). The
    // discriminator is the deferral:
    //   * `Sequential` must capture the two loaders and touch NO component weights → `Ok`, and the
    //     built residency is `Sequential` (`is_sequential()`).
    //   * `Resident` must eager-load the text encoder from that non-existent dir → `Err`.
    // A dispatch that ignored `offload_policy` and always built `Resident` (the F-172 bug class) would
    // eager-load under a `Sequential` request and turn the first assertion's `Ok` into an `Err` —
    // this test would fail. That is exactly the ignore-`offload_policy` regression the smoke tests miss.
    fn validation_complete_snapshot_spec(root: &Path, policy: OffloadPolicy) -> LoadSpec {
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            ENCODER_CONTRACT,
        )
        .expect("validation-complete text encoder fixture");
        LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        // Sequential defers every heavy/text load, so a missing snapshot dir is NOT touched here.
        let fixture = tempfile::tempdir().expect("snapshot fixture");
        let spec = validation_complete_snapshot_spec(fixture.path(), OffloadPolicy::Sequential);
        let plan = resolve_load_plan(
            &spec,
            resolve_root(&spec, KREA_2_TURBO_ID).unwrap(),
            KREA_2_TURBO_ID,
        )
        .unwrap();
        let res = build_residency(&spec, KREA_2_TURBO_ID, plan)
            .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential residency (the deferred state machine)"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        // Resident eager-loads the text encoder now, so the missing snapshot dir surfaces as an error
        // at construction — the flip side that proves the Sequential test's `Ok` came from deferral.
        let fixture = tempfile::tempdir().expect("snapshot fixture");
        let spec = validation_complete_snapshot_spec(fixture.path(), OffloadPolicy::Resident);
        let plan = resolve_load_plan(
            &spec,
            resolve_root(&spec, KREA_2_TURBO_ID).unwrap(),
            KREA_2_TURBO_ID,
        )
        .unwrap();
        let err = build_residency(&spec, KREA_2_TURBO_ID, plan)
            .err()
            .expect("Resident must eager-load and fail on a missing snapshot dir");
        // A load/IO error, not the precision/single-file guard (which a Dir source passes).
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("precision override"),
            "expected an eager-load failure, got the up-front guard: {msg}"
        );
    }

    #[test]
    fn shared_quant_guard_drives_load_time_and_effective_tiers() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();

        let mut q8 = LoadSpec::new(WeightsSource::Dir(root.clone()));
        q8.quantize = Some(Quant::Q8);
        assert_eq!(
            load_time_quant_bits(&q8, &root, KREA_2_TURBO_ID).unwrap(),
            None
        );
        assert_eq!(
            effective_base_quant_bits(&q8, &root, KREA_2_TURBO_ID).unwrap(),
            Some(8)
        );

        q8.quantize = Some(Quant::Q4);
        let error = load_time_quant_bits(&q8, &root, KREA_2_TURBO_ID)
            .unwrap_err()
            .to_string();
        assert!(error.contains(KREA_2_TURBO_ID), "{error}");
        assert!(error.contains("Q8") && error.contains("Q4"), "{error}");

        std::fs::write(root.join("transformer/config.json"), "{").unwrap();
        assert!(effective_base_quant_bits(&q8, &root, KREA_2_TURBO_ID).is_err());
    }

    // ── Multi-phase request validation (epic 13879, sc-13884) ────────────────────────────────────

    fn phase(steps: u32, guidance: Option<f32>) -> mlx_gen::GenerationPhase {
        mlx_gen::GenerationPhase {
            steps,
            guidance,
            adapters: vec![],
        }
    }

    fn phase_req(phases: Vec<mlx_gen::GenerationPhase>) -> GenerationRequest {
        GenerationRequest {
            phases: Some(phases),
            ..req(1024, 1024)
        }
    }

    /// The canonical Raw multi-phase split (CFG-on then CFG-off, base-only) validates on `krea_2_raw`.
    #[test]
    fn multiphase_base_only_guidance_split_validates_on_raw() {
        let r = phase_req(vec![phase(20, Some(3.5)), phase(8, Some(0.0))]);
        assert!(validate_request(&raw_descriptor(), &r).is_ok());
    }

    /// Multi-phase is Raw-only in v1: Turbo/edit reject it (Turbo is CFG-free single-phase).
    #[test]
    fn multiphase_rejected_on_non_raw_variants() {
        let r = phase_req(vec![phase(8, None)]);
        for desc in [descriptor(), turbo_edit_descriptor(), edit_descriptor()] {
            let err = validate_request(&desc, &r).unwrap_err().to_string();
            assert!(
                err.contains("supported on krea_2_raw"),
                "{}: {err}",
                desc.id
            );
        }
    }

    /// An empty phase list and a 0-step phase are malformed trajectories.
    #[test]
    fn multiphase_rejects_empty_list_and_zero_step_phase() {
        let empty = validate_request(&raw_descriptor(), &phase_req(vec![]))
            .unwrap_err()
            .to_string();
        assert!(empty.contains("at least one phase"), "{empty}");
        let zero = validate_request(
            &raw_descriptor(),
            &phase_req(vec![phase(4, None), phase(0, None)]),
        )
        .unwrap_err()
        .to_string();
        assert!(
            zero.contains("phase 1 must run at least one step"),
            "{zero}"
        );
    }

    /// Per-phase adapters are WIRED (sc-13884): the flagship "N steps Raw (CFG on) + M steps
    /// Raw+turbo-LoRA (CFG off)" request VALIDATES (no longer rejected). The adapter-index bounds check
    /// runs at `generate` time (`resolve_phases`), not here — see the multiphase resolver tests.
    #[test]
    fn multiphase_accepts_per_phase_adapters() {
        let with_adapter = mlx_gen::GenerationPhase {
            steps: 8,
            guidance: Some(0.0),
            adapters: vec![mlx_gen::PhaseAdapter {
                adapter: 0,
                weight: Some(1.0),
            }],
        };
        assert!(validate_request(
            &raw_descriptor(),
            &phase_req(vec![phase(20, Some(3.5)), with_adapter])
        )
        .is_ok());
    }

    /// sc-13884 diff-patch guard: a multi-phase request on a model that had a `.diff`/`.diff_b`
    /// diff-patch folded at load is REJECTED loudly (the baked delta can't be toggled off per phase);
    /// a multi-phase request on a low-rank-only model (the epic's turbo-LoRA case) is ACCEPTED; and a
    /// single-phase request on a diff-patch model is unaffected.
    #[test]
    fn multiphase_rejected_on_diff_patch_model_but_allowed_low_rank() {
        let mp = phase_req(vec![phase(20, Some(3.5)), phase(8, Some(0.0))]);

        // diff-patch folded at load → multi-phase rejected with the diff-patch error.
        let err = ensure_multiphase_allowed_for(KREA_2_RAW_ID, true, &mp)
            .unwrap_err()
            .to_string();
        assert!(err.contains("diff-patch"), "{err}");
        assert!(err.contains(".diff/.diff_b"), "{err}");

        // Low-rank-only model (no diff-patch) → the same multi-phase request is allowed.
        assert!(ensure_multiphase_allowed_for(KREA_2_RAW_ID, false, &mp).is_ok());

        // A single-phase request on a diff-patch model is unaffected (the guard is phases-only).
        assert!(ensure_multiphase_allowed_for(KREA_2_RAW_ID, true, &req(1024, 1024)).is_ok());
    }

    /// Multi-phase renders from pure noise — reference/edit conditioning and the PiD decoder are
    /// rejected (t2i-from-noise only in v1).
    #[test]
    fn multiphase_rejects_reference_and_pid() {
        let mut with_ref = phase_req(vec![phase(8, None)]);
        with_ref.conditioning = vec![Conditioning::Reference {
            image: tiny_image(),
            strength: None,
        }];
        assert!(validate_request(&raw_descriptor(), &with_ref)
            .unwrap_err()
            .to_string()
            .contains("renders from pure noise"));

        let mut with_pid = phase_req(vec![phase(8, None)]);
        with_pid.use_pid = true;
        assert!(validate_request(&raw_descriptor(), &with_pid)
            .unwrap_err()
            .to_string()
            .contains("PiD decoder"));
    }

    /// `phases: None` is the ordinary single-phase render — validation is unchanged.
    #[test]
    fn single_phase_request_is_unaffected() {
        assert!(validate_request(&raw_descriptor(), &req(1024, 1024)).is_ok());
        assert_eq!(req(1024, 1024).phases, None);
    }
}
