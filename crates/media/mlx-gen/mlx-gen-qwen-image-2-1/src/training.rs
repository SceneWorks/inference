//! LoRA/LoKr **training** on the Qwen-Image 2.1 DiT, in pure Rust on mlx-rs (sc-24159) — the MLX
//! trainer for `qwen_image_2_1`: text-to-image on captioned datasets, and (sc-24161)
//! **instruction-edit** on edit-pair datasets ([`TrainingItem::edit_pair`]).
//!
//! [`QwenImage21Trainer`] realizes the core [`Trainer`] contract the way every MLX family trainer
//! does (`mlx-gen-krea`, `mlx-gen-mage`): the DiT is a hand-rolled `&self` forward over raw
//! `Array`s, so the trainable factors live OUTSIDE the model in a [`LoraParams`] map, are
//! re-injected each step into the target [`AdaptableLinear`](mlx_gen::adapters::AdaptableLinear)s
//! through the shared [`mlx_gen::train::lora`] seam, and are stepped with `keyed_value_and_grad` +
//! the core [`TrainOptimizer`] + `clip_grad_norm`. The install mirrors the inference reload
//! op-for-op on the dense training tier, so the trained adapter loads back through the sc-24156 adapter host
//! ([`crate::apply_qwen_image_2_1_adapters`]) with no key conversion.
//!
//! ## What is Qwen-Image-2.1-specific
//! - **Flow-match velocity target = `noise − x0`**, DiT timestep = the noise fraction `t` itself
//!   ([`QwenImage21Transformer::forward`] takes the raw sigma and scales it by 1000 inside), and
//!   `x_t = (1 − t)·x0 + t·noise`. The pipeline's Euler step is `x ← x + v·(σ_{i+1} − σ_i)`, so the
//!   DiT's velocity is `dx/dσ = noise − x0` — no sign flip.
//! - **Latents** are the RGBA autoencoder's posterior **mode** of the centre-cropped, resized image
//!   (an RGB image is the opaque RGBA case: a constant `+1` alpha plane, exactly the reference
//!   route's `img.convert("RGBA")`), normalised `(z − mean)/std` and packed unpatched to
//!   `[1, (edge/16)², z_dim]` — the very tensor the DiT denoises in.
//! - **Caption features** are the Qwen3-VL language tower's last-layer hidden states for the T2I
//!   prompt template with the system prefix dropped ([`QwenImage21TextEncoder::encode_prompt`]) —
//!   the same conditioning the render path feeds the DiT.
//! - **Targets** default to every per-block Linear ([`BLOCK_ADAPTER_TARGETS`] on each of the
//!   transformer blocks); an explicit `lora_target_modules` suffix-matches anywhere on the
//!   sc-24156 host, including the
//!   [`GLOBAL_ADAPTER_TARGETS`](crate::GLOBAL_ADAPTER_TARGETS).
//!
//! ## Memory lifecycle (staged; never more than one heavy component at once)
//! 1. **Preflight** ([`training_footprint`] / [`check_training_footprint`]) — before any weight is
//!    read, the run's peak is derived from *this snapshot's own* configs and safetensors headers
//!    (never borrowed numbers) and refused with an actionable message if it exceeds the device.
//! 2. **Captions** — the Qwen3-VL tower (language half only; plus the vision half for an edit
//!    run) encodes every caption and preview prompt ONCE, then is dropped.
//! 3. **Latents** — the VAE encodes every image ONCE; the encoder half is then dropped (the decoder
//!    stays only when preview samples are requested).
//! 4. **Train** — the dense DiT loads last. With `gradient_checkpointing` each block runs inside an
//!    `mlx::checkpoint` segment ([`QwenImage21Transformer::forward_checkpointed`]) for LoRA and
//!    LoKr alike.
//!
//! ## Edit mode (sc-24161)
//! A dataset whose items carry [`TrainingItem::reference_image_paths`] trains an **edit** adapter.
//! The shared gen-core floor ([`gen_core::train::validate_edit_request`]) caps each item at
//! [`MAX_REFERENCE_IMAGES`] — the render path's own cap, advertised through
//! [`TrainerDescriptor::max_reference_images`] — and refuses mixed datasets. Each item's ordered
//! references go through the **render path's own** assembly: host preprocessing
//! ([`prepare_conditioning_references`]), the image-conditioned Qwen3-VL template with vision tokens
//! ([`QwenImage21TextEncoder::encode_conditioning`]), the joint layout + text rows
//! ([`joint_branch`]) and the reference VAE latents ([`encode_references`]) — all cached ONCE,
//! like the target latents. Each step feeds the DiT the same image stream the denoise loop does
//! ([`joint_images`]: references in order, then the noised target) over the same layout, so the
//! positional ids and RoPE offsets are the render path's; the DiT returns the target block's
//! velocity only, so the loss covers target tokens only. The preflight prices the reference
//! latents, the longer joint sequence and the vision tower. Saved edit adapters additionally carry
//! [`EDIT_ADAPTER_MARKER`].
//!
//! The trainer is **dense-bf16 only**: a pre-quantized (Q4/Q8) tier or a quantize request is a
//! typed refusal — QLoRA is a non-goal.
//!
//! Every saved adapter (final and intermediate) carries its provenance and licence in the
//! safetensors `__metadata__` ([`ADAPTER_PROVENANCE`]): `family = qwen-image-2-1`,
//! `baseModel = qwen_image_2_1`, and the Qwen Research License the base weights are under.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
use mlx_gen::gen_core::{self, BucketSchedule};
use mlx_gen::img2img::preprocess_init_image;
use mlx_gen::tiling::TilingConfig;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_edges, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, adapter_optimizer_update, average_grads, build_lokr_targets,
    build_lora_targets, factorization, LoraParams, TrainAdapter,
};
use mlx_gen::train::loss::{prepared_subject_mask_weight, reduce_loss};
use mlx_gen::train::perceptual::{
    combine_step_loss, AuxAlternation, Parameterization, PerceptualPath, StepPlan,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::train::subject_mask::{CropBox, PreparedSubjectMask};
use mlx_gen::train::tae::TinyDecoderSpec;
use mlx_gen::{
    CancelFlag, Error, Image, LoadSpec, Modality, NetworkType, Precision, Progress, Result,
    RgbaImage, TrainOptimizer, Trainer, TrainerDescriptor, TrainingConfig, TrainingItem,
    TrainingOutput, TrainingProgress, TrainingRequest,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::{
    SchedulerConfig, TextEncoderConfig, TransformerConfig, VaeConfig, VisionConfig,
    IMAGE_TOKENS_PER_SLOT, MAX_REFERENCE_IMAGES, VAE_SCALE_FACTOR,
};
use crate::loader;
use crate::memory_strategy::derived::{MLX_EVAL_SLACK_BYTES, VAE_PIPELINED_DECODE_MAPS};
use crate::model::{FAMILY, MODEL_ID};
use crate::pipeline::{
    create_noise, decode_rgb, denoise, encode_references, joint_branch, joint_images, joint_layout,
    missing_vision_tower, pack_latents, prepare_conditioning_references, DenoiseInputs,
    JointBranch, ReferenceConditioning, DECODE_OVERLAP, DECODE_TILE_EDGE,
};
use crate::quant::{installed_tier, Tier};
use crate::reference::{
    calculate_dimensions, prepare_references, reference_fit, PreparedReference,
};
use crate::text_encoder::{
    image_pad_token_id, prompt_template, prompt_template_ti2i, system_prompt_drop_count,
    QwenImage21TextEncoder,
};
use crate::transformer::{
    BlockTrainables, CheckpointedTrainables, JointLayout, QwenImage21Transformer,
    BLOCK_ADAPTER_TARGETS,
};
use crate::vae::QwenImage21Vae;
use crate::{UPSTREAM_LICENSE, UPSTREAM_LICENSE_NOTICE};

/// Registry id of the Qwen-Image 2.1 trainer — the generator id of the same base model (the
/// [`TrainerDescriptor::id`] convention), so a trained adapter's `baseModel` names the route it
/// applies to. Pinned equal to [`crate::MODEL_ID`] by a test.
pub const TRAINER_ID: &str = "qwen_image_2_1";

/// The LoKr delta-reconstruction dtype — what the shared inference loader reconstructs LyCORIS
/// deltas at (bf16), so a trained LoKr round-trips through the sc-24156 apply path.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence.
const SAMPLE_PROMPT_CAP: usize = 4;

/// Provenance + licence stamped into every saved adapter's `__metadata__` (final and
/// intermediate, LoRA and LoKr), alongside the shared `networkType`/`rank`/`alpha` reload contract.
///
/// * `family` / `baseModel` are the SceneWorks-native pair its `detect_metadata_family` reads
///   first (the candle Krea / MLX Mage trainers stamp the same keys); `family` is already the
///   canonical `qwen-image-2-1` token, so a re-imported adapter lands in the 2.1 LoRA pool rather
///   than the 2512 `qwen-image` one its tensor names would otherwise resemble.
/// * `ss_base_model_version` is the kohya spelling of the same base id, for tools that only read
///   that convention.
/// * `license` / `modelspec.license` / `licenseNotice` record the Qwen Research License the base
///   weights are distributed under — a fine-tune of them inherits it (epic requirement E12).
pub const ADAPTER_PROVENANCE: [(&str, &str); 6] = [
    ("family", FAMILY),
    ("baseModel", MODEL_ID),
    ("ss_base_model_version", MODEL_ID),
    ("license", UPSTREAM_LICENSE),
    ("modelspec.license", UPSTREAM_LICENSE),
    ("licenseNotice", UPSTREAM_LICENSE_NOTICE),
];

/// The extra `__metadata__` entry an **edit** adapter (one trained on an edit-pair dataset,
/// sc-24161) carries on top of [`ADAPTER_PROVENANCE`], so the product layer can tell an
/// instruction-edit adapter from a text-to-image one without inspecting tensors. Text-to-image
/// adapters do not carry the key at all (their metadata is unchanged by sc-24161). The adapter
/// itself is the same PEFT/LyCORIS file either way and loads through the same host.
pub const EDIT_ADAPTER_MARKER: (&str, &str) = ("trainingMode", "edit");

/// Recognized `timestep_type` values [`sample_sigma`] branches on (plus the `sigmoid` default).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values [`sample_sigma`] branches on (plus the neutral default).
const TIMESTEP_BIASES: [&str; 9] = [
    "balanced",
    "none",
    "neutral",
    "high",
    "high_noise",
    "favor_high_noise",
    "low",
    "low_noise",
    "favor_low_noise",
];
/// Recognized `loss_type` values — `mae`/`l1` → MAE, `mse`/`l2` → the MSE default.
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// The refusal every quantized-base path raises (typed [`Error::Unsupported`]). QLoRA is a
/// non-goal: the trainer runs over the dense bf16 base only.
fn quantized_base_refusal(what: &str) -> Error {
    Error::Unsupported(format!(
        "{TRAINER_ID} trainer: {what}, but LoRA/LoKr training runs over the dense BF16 base only \
         (training over a packed Q4/Q8 tier — QLoRA — is not supported); install the BF16 tier \
         to train"
    ))
}

// ── preflight: the training footprint, derived from the snapshot's own facts ─────────────────────

/// Width of one f32 element — the caption/latent caches, the trainable factors and the VAE.
const F32_WIDTH: u64 = 4;

/// f32 buffers every trainable factor element carries whatever the optimizer: the factor itself,
/// its gradient and the gradient-accumulation buffer. The optimizer's own state comes on top
/// ([`optimizer_state_per_param`]).
const TRAINABLE_BASE_BUFFERS: u64 = 3;

/// Width of one element of a materialised LoKr delta: `install_training_lokr` reconstructs every
/// target's dense `[out, in]` delta at the trainer's LoKr dtype (bf16).
const LOKR_DELTA_WIDTH: u64 = 2;

/// f32 optimizer-state elements [`TrainOptimizer`] keeps per trainable element: AdamW/Adam two
/// (`m`, `v`), Rose none (stateless), Prodigy four (`exp_avg`, `exp_avg_sq`, `s`, `p0`). Names
/// normalise the way the optimizer picker does (case, `-`/`_`, the `…opt` aliases).
pub fn optimizer_state_per_param(optimizer: &str) -> u64 {
    let name: String = optimizer
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| *c != '-' && *c != '_')
        .collect();
    match name.as_str() {
        "rose" | "roseopt" => 0,
        "prodigy" | "prodigyopt" => 4,
        _ => 2,
    }
}

/// `[S, inner]`-shaped tensors the dense backward retains per block for the **attention** half,
/// counted off [`crate::transformer`]'s block forward: the LayerNorm input and the modulated `h`,
/// the q/k/v projections, q/k after the per-head RMS norm and after RoPE (whose f32 rotation
/// planes count double at bf16 width), and the SDPA output. A **structural count, not a
/// measurement** — like every number in [`crate::memory_strategy::derived`].
pub const ATTENTION_SAVED_HIDDEN: u64 = crate::training_memory::ATTENTION_SAVED_HIDDEN;
/// `[S, inner]` tensors the **feed-forward** half retains regardless of its width: the residual
/// after attention, the modulated `h`, and the FFN output.
pub const FFN_SAVED_HIDDEN_FIXED: u64 = crate::training_memory::FFN_SAVED_HIDDEN_FIXED;
/// `[S, inner·mlp_ratio]` tensors the SwiGLU retains: `gate(h)`, `silu(gate(h))`, `proj(h)` and
/// their product — each `mlp_ratio` hidden-widths wide.
pub const FFN_SAVED_PER_MLP_RATIO: u64 = crate::training_memory::FFN_SAVED_PER_MLP_RATIO;
/// Score buffers in the unfused Metal softmax VJP (see `training_memory`).
pub const ATTENTION_BACKWARD_SCORE_MATRICES: u64 =
    crate::training_memory::SOFTMAX_VJP_SCORE_BUFFERS;
/// `[L, text_hidden]` f32 tensors live in one Qwen3 decoder layer while a caption encodes (the
/// residual, the normed input, q/k/v, the attention output and the SwiGLU's three wide halves,
/// rounded up). Captions are short; this term never decides a refusal on its own.
const TEXT_ENCODER_LIVE_HIDDEN: u64 = 16;

/// The facts the training footprint is derived from — read off **this snapshot**: its three
/// component configs and its safetensors **headers** (no tensor data is read). Nothing here is a
/// number borrowed from 2512, Krea or the production table (epic requirement E13): the miniature
/// parity snapshot derives its own small footprint and the release derives the real one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FootprintFacts {
    /// Elements of every tensor under `transformer/` (the dense DiT, loaded whole).
    pub dit_elements: u64,
    /// Bytes of the Qwen3 language tower as loaded (`model.language_model.*` at its stored dtype;
    /// the `lm_head` and the vision tower are never read by the trainer).
    pub text_encoder_bytes: u64,
    /// Bytes of the VAE's encoder half (`encoder.*` + `quant_conv.*`), dropped after caching.
    pub vae_encoder_bytes: u64,
    /// Bytes of the VAE's decoder half, kept only for preview samples.
    pub vae_decoder_bytes: u64,
    pub num_layers: u64,
    /// `num_attention_heads · attention_head_dim`.
    pub inner: u64,
    pub heads: u64,
    pub mlp_ratio: u64,
    /// Latent channels (`in_channels` of the DiT = `z_dim` of the VAE).
    pub latent_channels: u64,
    /// The text tower's hidden width (the caption feature width).
    pub text_hidden: u64,
    /// The VAE encoder's full-resolution channel count (`base_dim`).
    pub vae_encode_channels: u64,
    /// The VAE decoder's full-resolution channel count (`decoder_base_dim`).
    pub vae_decode_channels: u64,
    /// Pixels one latent token covers per side (`scale_factor_spatial`).
    pub pixels_per_token: u64,
    /// Bytes of the Qwen3-VL **vision** tower (`model.visual.*`) — loaded only by an edit run,
    /// which encodes its references as vision context (sc-24161). `0` when the snapshot ships none.
    pub vision_tower_bytes: u64,
    /// The vision tower's hidden width (`vision_config.hidden_size`); `0` without a tower.
    pub vision_hidden: u64,
}

impl FootprintFacts {
    /// Derive the facts for the snapshot at `root` from its configs and safetensors headers.
    pub fn from_snapshot(root: &Path) -> Result<Self> {
        let dit_cfg = TransformerConfig::from_json_file(&root.join("transformer/config.json"))?;
        let te_cfg = TextEncoderConfig::from_json_file(&root.join("text_encoder/config.json"))?;
        let vae_cfg = VaeConfig::from_json_file(&root.join("vae/config.json"))?;

        let mut dit_elements = 0u64;
        for header in safetensors_path_tensor_headers(root.join("transformer"))? {
            dit_elements += header.element_count()?;
        }
        let text_encoder_headers = safetensors_path_tensor_headers(root.join("text_encoder"))?;
        let tower_bytes = |prefix: &str| -> u64 {
            let prefix = format!("{prefix}.");
            text_encoder_headers
                .iter()
                .filter(|h| h.name.starts_with(&prefix))
                .map(|h| h.data_bytes)
                .sum()
        };
        let text_encoder_bytes = tower_bytes(loader::TEXT_ENCODER_PREFIX);
        let vision_tower_bytes = tower_bytes(loader::VISION_TOWER_PREFIX);
        let vision_hidden = loader::load_vision_config(root)?
            .map_or(0, |vision| vision.tower.hidden_size.max(0) as u64);
        let (mut vae_encoder_bytes, mut vae_decoder_bytes) = (0u64, 0u64);
        for header in safetensors_path_tensor_headers(root.join("vae"))? {
            if header.name.starts_with("encoder.") || header.name.starts_with("quant_conv.") {
                vae_encoder_bytes += header.data_bytes;
            } else {
                vae_decoder_bytes += header.data_bytes;
            }
        }
        Ok(Self {
            dit_elements,
            text_encoder_bytes,
            vae_encoder_bytes,
            vae_decoder_bytes,
            num_layers: dit_cfg.num_layers as u64,
            inner: dit_cfg.inner_dim() as u64,
            heads: dit_cfg.num_attention_heads as u64,
            mlp_ratio: dit_cfg.mlp_ratio as u64,
            latent_channels: dit_cfg.in_channels as u64,
            text_hidden: te_cfg.hidden_size as u64,
            vae_encode_channels: vae_cfg.base_dim as u64,
            vae_decode_channels: vae_cfg.decoder_base_dim as u64,
            pixels_per_token: vae_cfg.scale_factor_spatial as u64,
            vision_tower_bytes,
            vision_hidden,
        })
    }
}

/// The shape of one training run, as far as memory is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingShape {
    /// Bucketed square training edge, in pixels — the **largest** resolution bucket's edge when
    /// the run trains several (sc-2127, epic 2123 E7).
    pub edge: u32,
    /// Latent tokens of the largest **target**. `0` means the square `edge × edge` target every
    /// text-to-image item trains at; an edit run (sc-24161) keeps each target's aspect ratio
    /// (`edit_target_size`), so it prices its largest target explicitly.
    pub target_tokens: u64,
    /// Target latent tokens cached across the whole dataset — every item at every resolution
    /// bucket (sc-2127). `0` means one target per item at [`target_tokens`](Self::target_tokens)
    /// (the single-bucket cache).
    pub target_cache_tokens: u64,
    /// The longest conditioning sequence (caption or preview prompt), in **text** tokens — the
    /// vision slots of an edit prompt are counted by [`reference_tokens`](Self::reference_tokens).
    pub caption_tokens: u64,
    /// Edit runs (sc-24161): the largest per-item sum of reference latent tokens — the condition
    /// blocks the joint sequence carries ahead of the target. `0` for text-to-image.
    pub reference_tokens: u64,
    /// Edit runs: the latent tokens of the single largest reference (its VAE-encode and ViT
    /// transients). `0` for text-to-image.
    pub largest_reference_tokens: u64,
    /// Edit runs: reference latent tokens cached across the whole dataset. `0` for text-to-image.
    pub reference_cache_tokens: u64,
    /// Score elements (per head) of the block-causal **prefix** attention calls for the costliest
    /// prompt — `Σ (end − start)·end` over its layout's prefix segments ([`prefix_score_elements`]):
    /// each prefix segment's rows attend to every key up to that segment's end (sc-24162 review).
    /// `0` falls back to the whole prefix squared (`(caption_tokens + reference_tokens)²`), which
    /// is exact for a text-only prefix and an upper bound otherwise.
    pub prefix_scores: u64,
    /// Largest single prefix SDPA call, per head. Zero uses the whole prefix as an upper bound.
    pub largest_prefix_call: u64,
    /// Dataset items (each caches one caption feature, and one target latent per resolution
    /// bucket — [`target_cache_tokens`](Self::target_cache_tokens)).
    pub items: u64,
    /// Bytes per element of the DiT compute dtype (2 for bf16, 4 for f32).
    pub compute_width: u64,
    /// Elements of the trainable adapter factors.
    pub trainable_params: u64,
    /// f32 optimizer-state elements per trainable element ([`optimizer_state_per_param`]).
    pub optimizer_state_per_param: u64,
    /// `Σ out·in` over the targets when training **LoKr** (0 for LoRA): every LoKr install
    /// materialises a dense `[out, in]` bf16 delta per target.
    pub lokr_delta_elements: u64,
    /// Whether the blocks run gradient-checkpointed.
    pub checkpointed: bool,
    /// Whether preview samples are rendered (keeps the VAE decoder resident).
    pub sampling: bool,
    /// Epic 2123 E7: the training-time auxiliary models the enabled perceptual losses add
    /// (the TAEQI2.1 decoder + Depth-Anything-V2 resident weights, one differentiable decode +
    /// forward/backward each, and every cached reference — `mlx_gen_perceptual::perceptual_footprint`),
    /// resident through the train phase on the dense and checkpointed paths alike. `0` when no
    /// aux loss is enabled.
    pub aux_model_bytes: u64,
}

/// The derived peak of each stage of a run, in bytes. The stages never overlap (each drops its
/// heavy component before the next loads), so the run's peak is the largest of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingFootprint {
    /// Qwen3 tower resident + one caption's encode transient.
    pub caption_phase: u64,
    /// The VAE resident + one image's encode transient + the caption and latent caches.
    pub latent_phase: u64,
    /// The dense DiT + trainable state + caches + the step's activation working set.
    pub train_phase: u64,
}

impl TrainingFootprint {
    /// The run's peak: the largest stage.
    pub fn peak(&self) -> u64 {
        self.caption_phase
            .max(self.latent_phase)
            .max(self.train_phase)
    }
}

/// Derive the training footprint of `shape` on the snapshot described by `facts`.
///
/// The DiT working set follows this crate's own block structure: a dense step retains every
/// block's [`ATTENTION_SAVED_HIDDEN`] + [`FFN_SAVED_HIDDEN_FIXED`] + `mlp_ratio ·`
/// [`FFN_SAVED_PER_MLP_RATIO`] `[S, inner]` tensors for the backward; a gradient-checkpointed step
/// retains each block's input plus its forward and backward recompute working sets. Both add the unfused attention
/// backward ([`ATTENTION_BACKWARD_SCORE_MATRICES`] score matrices of the block-causal calls:
/// `heads·(T·S + Σ (end − start)·end)` elements — the target rows attend to every key, each prefix
/// segment's rows to the keys up to its end, [`TrainingShape::prefix_scores`]) and MLX's pipelined
/// evaluation: input/output buffers, wide f32 SwiGLU intermediates and score matrices held by
/// the ten in-flight Metal command buffers plus the buffer being scheduled. See `training_memory`
/// for the pinned upstream source and buffer threshold. Adds [`MLX_EVAL_SLACK_BYTES`].
///
/// LoKr adds its materialised deltas (`lokr_delta_elements` at bf16): a dense step keeps every
/// target's delta live for the backward, a checkpointed step only one block's (rebuilt once in the
/// forward and once in the recompute), and a preview install materialises all of them in either
/// mode.
pub fn training_footprint(facts: &FootprintFacts, shape: &TrainingShape) -> TrainingFootprint {
    let w = shape.compute_width;
    let side = shape.edge as u64 / facts.pixels_per_token.max(1);
    let image_tokens = if shape.target_tokens > 0 {
        shape.target_tokens
    } else {
        side * side
    };
    // The joint sequence: text, every condition block (edit), the target.
    let seq = image_tokens + shape.caption_tokens + shape.reference_tokens;
    let token_pixels = facts.pixels_per_token * facts.pixels_per_token;
    // The largest single VAE encode: the target, or (edit) a reference fitted to the vision
    // resolution, which can exceed the training edge.
    let pixels = (shape.edge as u64 * shape.edge as u64)
        .max(image_tokens * token_pixels)
        .max(shape.largest_reference_tokens * token_pixels);

    // Caches (f32): caption features (shared by an item's bucket entries) and packed latents
    // (targets at every bucket + edit references).
    let caption_cache = shape.items * shape.caption_tokens * facts.text_hidden * F32_WIDTH;
    let target_cache = if shape.target_cache_tokens > 0 {
        shape.target_cache_tokens
    } else {
        shape.items * image_tokens
    };
    let latent_cache =
        (target_cache + shape.reference_cache_tokens) * facts.latent_channels * F32_WIDTH;

    // 1. captions: the language tower + one caption's per-layer live set (f32 stream). An edit
    //    caption also runs the vision tower (resident, plus one reference's patch stream) and its
    //    language sequence carries one vision slot per merged 2×2 latent group.
    let edit = shape.reference_tokens > 0;
    let vision_slots = shape.reference_tokens / IMAGE_TOKENS_PER_SLOT as u64;
    let vision = if edit {
        facts.vision_tower_bytes
            + TEXT_ENCODER_LIVE_HIDDEN
                * shape.largest_reference_tokens
                * facts.vision_hidden
                * F32_WIDTH
    } else {
        0
    };
    let caption_phase = facts.text_encoder_bytes
        + vision
        + TEXT_ENCODER_LIVE_HIDDEN
            * (shape.caption_tokens + vision_slots)
            * facts.text_hidden
            * F32_WIDTH
        + MLX_EVAL_SLACK_BYTES;

    // 2. latents: the whole VAE + one image's pipelined full-resolution encode maps (f32).
    let vae_encode = VAE_PIPELINED_DECODE_MAPS * facts.vae_encode_channels * pixels * F32_WIDTH;
    let latent_phase = facts.vae_encoder_bytes
        + facts.vae_decoder_bytes
        + vae_encode
        + caption_cache
        + latent_cache
        + MLX_EVAL_SLACK_BYTES;

    // 3. train: dense DiT at the compute width + trainable state + caches + the step.
    let prefix = shape.caption_tokens + shape.reference_tokens;
    let prefix_scores = if shape.prefix_scores > 0 {
        shape.prefix_scores
    } else {
        prefix * prefix
    };
    let largest_prefix = if shape.largest_prefix_call > 0 {
        shape.largest_prefix_call
    } else {
        prefix * prefix
    };
    let lokr_deltas = shape.lokr_delta_elements * LOKR_DELTA_WIDTH;
    let step = crate::training_memory::step_bytes(crate::training_memory::StepShape {
        sequence: seq,
        inner: facts.inner,
        layers: facts.num_layers,
        heads: facts.heads,
        mlp_ratio: facts.mlp_ratio,
        width: w,
        target_scores: image_tokens * seq,
        prefix_scores,
        largest_prefix_call: largest_prefix,
        lokr_delta_elements: shape.lokr_delta_elements,
        checkpointed: shape.checkpointed,
    });
    // A preview render runs between steps (the step's activations are released by then), so its
    // decode transient competes with the step rather than adding to it.
    let (decoder_resident, preview) = if shape.sampling {
        let tile = if crate::memory_strategy::default_decode_is_bounded(shape.edge, shape.edge) {
            DECODE_TILE_EDGE as u64
        } else {
            shape.edge as u64
        };
        (
            facts.vae_decoder_bytes,
            VAE_PIPELINED_DECODE_MAPS * facts.vae_decode_channels * tile * tile * F32_WIDTH
                + lokr_deltas,
        )
    } else {
        (0, 0)
    };
    let train_phase = facts.dit_elements * w
        + shape.aux_model_bytes
        + decoder_resident
        + shape.trainable_params
            * (TRAINABLE_BASE_BUFFERS + shape.optimizer_state_per_param)
            * F32_WIDTH
        + caption_cache
        + latent_cache
        + step.max(preview)
        + MLX_EVAL_SLACK_BYTES;

    TrainingFootprint {
        caption_phase,
        latent_phase,
        train_phase,
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Refuse a run whose derived peak exceeds `budget_bytes`, with an actionable message naming the
/// stage that does not fit and the levers that help. Pure (no device query), so it is unit-tested.
pub fn check_training_footprint(
    facts: &FootprintFacts,
    shape: &TrainingShape,
    budget_bytes: u64,
) -> Result<()> {
    let fp = training_footprint(facts, shape);
    let peak = fp.peak();
    if peak <= budget_bytes {
        return Ok(());
    }
    let stage = if peak == fp.train_phase {
        "the training step (dense DiT + adapter state + activations)"
    } else if peak == fp.latent_phase {
        "latent caching (VAE encode)"
    } else {
        "caption caching (Qwen3-VL text tower)"
    };
    let mut advice = Vec::new();
    if !shape.checkpointed {
        let with_ckpt = TrainingShape {
            checkpointed: true,
            ..*shape
        };
        let ckpt_peak = training_footprint(facts, &with_ckpt).peak();
        advice.push(format!(
            "enable Gradient Checkpointing (recomputes each block in the backward; derived peak \
             ~{:.1} GiB{})",
            gib(ckpt_peak),
            if ckpt_peak <= budget_bytes {
                ", which fits"
            } else {
                ", still over"
            }
        ));
    }
    advice.push(format!(
        "lower the training resolution (now {} px)",
        shape.edge
    ));
    if shape.sampling {
        advice.push("turn off preview samples (frees the VAE decoder)".to_string());
    }
    advice.push("lower the rank".to_string());
    Err(Error::Msg(format!(
        "{TRAINER_ID} trainer: this run's derived peak memory is ~{:.1} GiB in {stage}, which \
         exceeds this device's ~{:.1} GiB training budget (captions ~{:.1} / latents ~{:.1} / \
         train ~{:.1} GiB). Refusing before step 1 rather than letting the OS kill the worker \
         mid-run. To fit: {}.",
        gib(peak),
        gib(budget_bytes),
        gib(fp.caption_phase),
        gib(fp.latent_phase),
        gib(fp.train_phase),
        advice.join("; ")
    )))
}

/// This device's training budget: MLX's own memory limit × [`mlx_gen::memory::SAFE_FRAC`] — the
/// engine-wide "safe peak" convention (15 % headroom for the OS, other processes and MLX
/// backpressure).
fn device_budget_bytes() -> u64 {
    (mlx_rs::memory::get_memory_limit() as f64 * mlx_gen::memory::SAFE_FRAC) as u64
}

/// Caps MLX's freed-buffer pool (`set_cache_limit`) for one training run and restores the
/// previous cap on drop. Installed as `min(previous, requested)`, like
/// [`crate::memory_strategy::AllocatorBounds`]: a run only ever tightens what a harness set.
struct TrainingPoolBound {
    previous: usize,
    effective: usize,
}

impl TrainingPoolBound {
    fn enter(limit_bytes: u64) -> Self {
        let limit = usize::try_from(limit_bytes).unwrap_or(usize::MAX);
        let previous = mlx_rs::memory::set_cache_limit(limit);
        if previous < limit {
            mlx_rs::memory::set_cache_limit(previous);
        }
        Self {
            previous,
            effective: previous.min(limit),
        }
    }
}

/// Explicit operator-owned diagnostic output; unset for normal training. These
/// Foreground snapshots at existing serialized phase boundaries. MLX's scalar
/// counters are not an atomic snapshot; no extra array evaluation or Metal sync
/// is introduced for sampling. Background physical samples can only be aligned
/// approximately by timestamp. No allocator limit or peak accounting is changed.
fn training_memory_trace(
    stage: &str,
    preflight: Option<TrainingFootprint>,
    cache_limit: Option<usize>,
) {
    use std::io::Write;
    let Some(directory) = std::env::var_os("QWEN_IMAGE_2_1_TRAINING_DIAGNOSTICS_OUT") else {
        return;
    };
    let directory = PathBuf::from(directory);
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let event = serde_json::json!({
        "unixMillis": time, "stage": stage,
        "counterScope": "foreground_non_atomic_snapshot_no_added_eval_or_sync",
        "activeBytes": mlx_rs::memory::get_active_memory(),
        "cacheBytes": mlx_rs::memory::get_cache_memory(),
        "peakActiveBytes": mlx_rs::memory::get_peak_memory(),
        "mlxMemoryLimitBytes": mlx_rs::memory::get_memory_limit(),
        "configuredCacheLimitBytes": if preflight.is_none() { cache_limit } else { None },
    });
    let write = || -> std::io::Result<()> {
        if let Some(fp) = preflight {
            std::fs::write(
                directory.join("training-preflight.json"),
                serde_json::json!({
                    "peakBytes": fp.peak(), "captionBytes": fp.caption_phase,
                    "latentBytes": fp.latent_phase, "trainBytes": fp.train_phase,
                    "requestedCacheLimitBytes": cache_limit,
                })
                .to_string(),
            )?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(directory.join("training-stages.jsonl"))?;
        writeln!(file, "{event}")
    };
    if let Err(error) = write() {
        eprintln!("training memory diagnostic write failed: {error}");
    }
}

impl Drop for TrainingPoolBound {
    fn drop(&mut self) {
        mlx_rs::memory::set_cache_limit(self.previous);
    }
}

/// `(trainable elements, LoKr delta elements)` the targets get under `cfg` — exact, from the
/// host's base shapes (the probe half; no weight is read). Trainable: LoRA `rank·(in + out)`; LoKr
/// `w1` plus a full or low-rank `w2` by PEFT's `use_w2` rule, exactly as [`build_lokr_targets`]
/// sizes them. LoKr deltas: `Σ out·in` (each install materialises the dense delta); 0 for LoRA.
fn adapter_elements(
    host: &mut QwenImage21Transformer,
    target_paths: &[String],
    cfg: &TrainingConfig,
) -> Result<(u64, u64)> {
    let rank = cfg.rank as i64;
    let mut total = 0u64;
    let mut deltas = 0u64;
    for path in target_paths {
        let segs: Vec<&str> = path.split('.').collect();
        let facts = host.adaptable_facts(&segs).ok_or_else(|| {
            Error::Msg(format!(
                "{TRAINER_ID} trainer: target {path} does not resolve on the DiT"
            ))
        })?;
        let (out_f, in_f) = (facts.base_shape[0] as i64, facts.base_shape[1] as i64);
        let elements = match cfg.network_type {
            NetworkType::Lora => rank * (in_f + out_f),
            NetworkType::Lokr => {
                let (out_a, out_b) = factorization(out_f as i32, cfg.decompose_factor);
                let (in_a, in_b) = factorization(in_f as i32, cfg.decompose_factor);
                let (out_a, out_b, in_a, in_b) =
                    (out_a as i64, out_b as i64, in_a as i64, in_b as i64);
                let w2 = if (rank as f32) < (out_b.max(in_b) as f32) / 2.0 {
                    rank * (out_b + in_b)
                } else {
                    out_b * in_b
                };
                out_a * in_a + w2
            }
        };
        total += elements.max(0) as u64;
        if cfg.network_type == NetworkType::Lokr {
            deltas += (out_f * in_f).max(0) as u64;
        }
    }
    Ok((total, deltas))
}

// ── the trainer ──────────────────────────────────────────────────────────────────────────────────

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: TRAINER_ID,
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        // LoRA/LoKr adapters only — no control-branch training path.
        supports_control: false,
        // Adapter-only: the shared `validate_full_finetune_request` floor rejects a full tune.
        supports_full_finetune: false,
        // Instruction-edit datasets (sc-24161), capped at the render path's own reference limit —
        // the one constant `collect_references`/`validate_reference_count` enforce.
        max_reference_images: MAX_REFERENCE_IMAGES as u32,
        // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
        // update.
        // sc-2127 (epic 2123): honors `resolution_buckets` — every item (captioned or edit pair)
        // is cached once per bucket edge and the loop walks a `BucketSchedule`.
        // sc-24828 (epic 2123): subject-masked loss on the target latent tokens, both modes.
        // sc-24830 (epic 2123): depth anchoring — the shared decoded-x0 perceptual path (TAEQI2.1
        // decode of the target block's flow x0 estimate → Depth-Anything-V2 → cached round-trip
        // reference), text-to-image and edit, dense and gradient-checkpointed.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: true,
            subject_mask_loss: true,
            depth_anchoring: true,
            // sc-24832: the body losses ride the same builder arms as depth anchoring
            // (decoded-x0 pixel losses through this trainer's x0 decoder).
            body_proportion_loss: true,
            body_shape_loss: true,
            normal_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// The production [`Trainer`] for `qwen_image_2_1`: a frozen dense-bf16 base that caches a
/// captioned (or edit-pair, sc-24161) dataset to Qwen3-VL conditioning + VAE latents (staged, one
/// heavy component at a time), then runs the functional-autograd LoRA/LoKr loop with the core
/// runtime glue (LR schedule, gradient accumulation, checkpoints + resume, cancel, previews,
/// progress bands).
pub struct QwenImage21Trainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    tokenizer: TextTokenizer,
    /// Tokens of the system-role prefix the conditioning drops.
    drop_count: usize,
    scheduler: SchedulerConfig,
    /// The snapshot's Qwen3-VL vision geometry, when it ships a vision tower — required by an edit
    /// run (its references are vision context, exactly as on the render path).
    vision: Option<VisionConfig>,
    /// The DiT as an **unevaluated** graph — no weight is read. Resolves target modules and their
    /// base shapes for `validate` and the preflight; the train loop loads the real DiT itself.
    probe: QwenImage21Transformer,
    facts: FootprintFacts,
    /// Test seam: replaces [`device_budget_bytes`] so the preflight refusal is exercisable on any
    /// machine.
    memory_budget_override: Option<u64>,
}

/// Construct the trainer from a `Qwen/Qwen-Image-2.1` snapshot directory (the dense BF16 tier).
/// Reads only the tokenizer, configs and safetensors headers: the heavy components load staged
/// inside [`Trainer::train`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(QwenImage21Trainer::load(spec)?))
}

// The trainer registration constant bridges the crate's rich `Result` into gen-core's.
mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

impl QwenImage21Trainer {
    /// [`load_trainer`] without the box.
    pub fn load(spec: &LoadSpec) -> Result<Self> {
        gen_core::reject_unknown_components(spec, &[], TRAINER_ID)?;
        if let Some(quant) = spec.quantize {
            return Err(quantized_base_refusal(&format!(
                "a {quant:?} load was requested"
            )));
        }
        if spec.precision != Precision::Bf16 {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: the compute dtype comes from TrainingConfig::train_dtype; \
                 drop the LoadSpec precision override"
            )));
        }
        if spec.text_encoder.is_some() {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: the Qwen3-VL text encoder is loaded from the snapshot's \
                 own text_encoder/; LoadSpec::text_encoder substitution is not supported"
            )));
        }
        if !spec.adapters.is_empty() {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: training starts from the bare base; LoadSpec adapters \
                 would not be part of the trained adapter, so they are refused rather than \
                 silently dropped"
            )));
        }
        // Shared with the candle twin (sc-24163): one refusal, one message.
        gen_core::train::refuse_trainer_load_overlays(&format!("{TRAINER_ID} trainer"), spec)?;
        let root = loader::snapshot_root(&spec.weights)?.to_path_buf();
        let tier = installed_tier(&root)?;
        if tier != Tier::Bf16 {
            return Err(quantized_base_refusal(&format!(
                "{} is the pre-quantized {} tier",
                root.display(),
                tier.dir_name()
            )));
        }
        let probe = loader::load_transformer_lazy(&root)?;
        if probe.is_quantized() {
            return Err(quantized_base_refusal(&format!(
                "{}/transformer holds packed weights",
                root.display()
            )));
        }
        let tokenizer = loader::load_tokenizer(&root)?;
        let drop_count = system_prompt_drop_count(&tokenizer)?;
        let scheduler = loader::load_scheduler_config(&root)?;
        let vision = loader::load_vision_config(&root)?;
        let facts = FootprintFacts::from_snapshot(&root)?;
        Ok(Self {
            descriptor: trainer_descriptor(),
            root,
            tokenizer,
            drop_count,
            scheduler,
            vision,
            probe,
            facts,
            memory_budget_override: None,
        })
    }

    /// The snapshot facts the preflight derives from.
    pub fn footprint_facts(&self) -> &FootprintFacts {
        &self.facts
    }

    /// Replace the device budget the preflight compares against (bytes). A test seam — the
    /// production budget is MLX's memory limit × `SAFE_FRAC`.
    pub fn set_memory_budget_override(&mut self, bytes: Option<u64>) {
        self.memory_budget_override = bytes;
    }
}

/// Normalize a free-form config string the way the trainer's own parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation, unit-testable without loaded weights. Rejects an
/// empty dataset, zero rank, zero steps (a 0-step run would write a no-op identity adapter), an
/// unsupported optimizer, an unrecognized `timestep_type`/`timestep_bias`/`loss_type`, a control
/// image on any item, and `model_options` that select reference / control conditioning — the same
/// inputs, with the same messages, the candle twin refuses (sc-24163).
fn validate_request(req: &TrainingRequest) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err(format!("{TRAINER_ID} trainer: dataset is empty").into());
    }
    if cfg.rank == 0 {
        return Err(format!("{TRAINER_ID} trainer: rank must be > 0").into());
    }
    if cfg.steps == 0 {
        return Err(format!("{TRAINER_ID} trainer: steps must be > 0").into());
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(format!(
            "{TRAINER_ID} trainer: optimizer '{}' is not available on MLX training (supported: \
             adamw, adam, rose, prodigy)",
            cfg.optimizer
        )
        .into());
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&cfg.timestep_type).as_str()) {
        return Err(format!(
            "{TRAINER_ID} trainer: timestep_type '{}' is not recognized (supported: {})",
            cfg.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
        return Err(format!(
            "{TRAINER_ID} trainer: timestep_bias '{}' is not recognized (supported: {})",
            cfg.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
        return Err(format!(
            "{TRAINER_ID} trainer: loss_type '{}' is not recognized (supported: {})",
            cfg.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    if req
        .items
        .iter()
        .any(|item| item.control_image_path.is_some())
    {
        return Err(format!(
            "{TRAINER_ID} trainer: control images are not part of Qwen-Image 2.1 LoRA/LoKr training"
        )
        .into());
    }
    // The worker forwards its whole `advanced` map as `model_options` and native trainers parse
    // only the keys they own, so an unknown key is ignored — but a key that would switch this run
    // into a workflow it does not read from there (reference / control conditioning) is refused
    // rather than silently trained without it; its *off* state (null, `[]`, `""`, `{}`, `false`,
    // `"none"`) selects nothing. The list, the off-value rule and the message are gen-core's, shared
    // with the candle twin, so the two backends cannot drift.
    gen_core::train::refuse_reference_control_model_options(&format!("{TRAINER_ID} trainer"), req)?;
    Ok(())
}

/// `bf16`/`bfloat16` → bf16 mixed precision (the default); anything else → f32 (the contract's
/// "unrecognized values mean f32").
fn resolve_compute_dtype(train_dtype: &str) -> Dtype {
    let t = train_dtype.trim();
    if t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16") {
        Dtype::Bfloat16
    } else {
        Dtype::Float32
    }
}

/// Resolve the config's target-module *suffixes* to full dotted paths on the sc-24156 host. The
/// DEFAULT (empty `lora_target_modules`) is every [`BLOCK_ADAPTER_TARGETS`] Linear of every
/// transformer block; an explicit list suffix-matches anywhere — PEFT's `target_modules` rule —
/// including the [`GLOBAL_ADAPTER_TARGETS`](crate::GLOBAL_ADAPTER_TARGETS).
fn resolve_target_paths(transformer: &QwenImage21Transformer, cfg: &TrainingConfig) -> Vec<String> {
    let default = cfg.lora_target_modules.is_empty();
    let suffixes: Vec<String> = if default {
        BLOCK_ADAPTER_TARGETS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    AdaptableHost::adaptable_paths(transformer)
        .into_iter()
        .filter(|path| {
            (!default || path.starts_with("transformer_blocks."))
                && suffixes
                    .iter()
                    .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// Split the trainable factors by transformer block for the gradient-checkpointed forward: per
/// block, a [`TrainAdapter`] restricted to that block's targets (built against the whole DiT, so
/// its keys are exactly the run's `{path}.<factor>` keys) and the factor keys it reads.
fn block_trainables(
    transformer: &mut QwenImage21Transformer,
    target_paths: &[String],
    params: &LoraParams,
    cfg: &TrainingConfig,
) -> Result<Vec<BlockTrainables>> {
    let mut out = Vec::with_capacity(transformer.num_blocks());
    for index in 0..transformer.num_blocks() {
        let prefix = format!("transformer_blocks.{index}.");
        let paths: Vec<String> = target_paths
            .iter()
            .filter(|p| p.starts_with(&prefix))
            .cloned()
            .collect();
        if paths.is_empty() {
            out.push(BlockTrainables {
                adapter: None,
                keys: Vec::new(),
            });
            continue;
        }
        // Only the target bookkeeping is used; the freshly initialised factors are discarded (the
        // live ones come from `params`, under the same keys).
        let adapter = match cfg.network_type {
            NetworkType::Lora => TrainAdapter::Lora {
                targets: build_lora_targets(transformer, &paths, cfg.rank as i32, 0)?.0,
            },
            NetworkType::Lokr => TrainAdapter::Lokr {
                targets: build_lokr_targets(
                    transformer,
                    &paths,
                    cfg.rank as i32,
                    cfg.decompose_factor,
                    0,
                )?
                .0,
            },
        };
        let mut keys: Vec<Rc<str>> = params
            .keys()
            .filter(|key| {
                paths.iter().any(|path| {
                    key.strip_prefix(path.as_str())
                        .and_then(|rest| rest.strip_prefix('.'))
                        .is_some_and(|factor| !factor.contains('.'))
                })
            })
            .cloned()
            .collect();
        keys.sort();
        out.push(BlockTrainables {
            adapter: Some(Rc::new(adapter)),
            keys,
        });
    }
    Ok(out)
}

/// Decode an image file (PNG/JPEG) into the core RGB8 [`Image`].
fn decode_image(path: &Path) -> Result<Image> {
    let dynimg = image::open(path)
        .map_err(|e| Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// Decode a reference image file as RGBA8 — upstream's `img.convert("RGBA")`: an RGB file is the
/// opaque case (`A = 255`, exactly what the render path's `Conditioning::Reference` widening
/// produces), a file with alpha keeps it (the `Conditioning::ReferenceRgba` case).
fn decode_reference(path: &Path) -> Result<RgbaImage> {
    let dynimg = image::open(path)
        .map_err(|e| Error::Msg(format!("decode reference image {}: {e}", path.display())))?;
    let rgba = dynimg.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());
    Ok(RgbaImage {
        width,
        height,
        pixels: rgba.into_raw(),
    })
}

/// An item's ordered references, decoded — empty for a captioned item.
fn decode_references(item: &TrainingItem) -> Result<Vec<RgbaImage>> {
    item.reference_image_paths
        .iter()
        .map(|path| decode_reference(path.as_path()))
        .collect()
}

/// The token budget one edit prompt contributes, from the tokenizer and the reference **headers**
/// alone (no pixel is decoded, no weight is read): `(text tokens, Σ reference latent tokens,
/// largest single reference's latent tokens)`. Text tokens are the image-conditioned template's
/// tokens minus the system prefix and the `count` `<|image_pad|>` placeholders — exactly the rows
/// [`joint_branch`] keeps; each reference's latent tokens follow the render path's own fit
/// ([`reference_fit`], which also **refuses** a reference whose fit the Qwen3-VL processor would
/// rebind — the exact predicate `prepare_reference` applies — so a bad reference fails here,
/// before any weight loads, rather than after the tower is resident).
fn edit_prompt_tokens(
    tokenizer: &TextTokenizer,
    drop: usize,
    vision: &VisionConfig,
    prompt: &str,
    reference_paths: &[PathBuf],
) -> Result<(u64, u64, u64)> {
    let count = reference_paths.len();
    let tokens = tokenizer.tokenize_preformatted(&prompt_template_ti2i(prompt, count))?;
    let text = tokens.ids.len().saturating_sub(drop + count) as u64;
    let (mut sum, mut largest) = (0u64, 0u64);
    for (index, path) in reference_paths.iter().enumerate() {
        let size = image::image_dimensions(path).map_err(|e| {
            Error::Msg(format!(
                "{TRAINER_ID} trainer: read reference image {}: {e}",
                path.display()
            ))
        })?;
        let (rw, rh) = reference_fit(size, index, vision)?;
        let tokens = (rw / VAE_SCALE_FACTOR) as u64 * (rh / VAE_SCALE_FACTOR) as u64;
        sum += tokens;
        largest = largest.max(tokens);
    }
    Ok((text, sum, largest))
}

/// The exact joint layout an edit prompt assembles at a `(width, height)` target, from the
/// tokenizer and the reference **headers** alone — what [`joint_branch`] builds from the encoded
/// conditioning, without loading a weight: the image-conditioned template's ids with each
/// `<|image_pad|>` placeholder expanded to its reference's vision slots (one per merged 2×2 latent
/// group, [`reference_fit`]'s grid), the system prefix dropped, each slot run one condition block,
/// the target block last. The preflight prices its block-causal prefix
/// ([`prefix_score_elements`]); a test pins it to the branch the real encoder assembles. The candle
/// twin's function of the same name (sc-24162).
fn edit_prompt_layout(
    tokenizer: &TextTokenizer,
    drop: usize,
    vision: &VisionConfig,
    prompt: &str,
    reference_paths: &[PathBuf],
    (width, height): (u32, u32),
) -> Result<JointLayout> {
    use crate::transformer::Segment as S;
    let image_token = image_pad_token_id(tokenizer)?;
    let ids = tokenizer
        .tokenize_preformatted(&prompt_template_ti2i(prompt, reference_paths.len()))?
        .ids;
    let mut grids: Vec<(usize, usize)> = Vec::with_capacity(reference_paths.len());
    for (index, path) in reference_paths.iter().enumerate() {
        let size = image::image_dimensions(path).map_err(|e| {
            Error::Msg(format!(
                "{TRAINER_ID} trainer: read reference image {}: {e}",
                path.display()
            ))
        })?;
        let (rw, rh) = reference_fit(size, index, vision)?;
        grids.push((
            (rh / VAE_SCALE_FACTOR) as usize,
            (rw / VAE_SCALE_FACTOR) as usize,
        ));
    }
    // `None` = a text token, `Some(k)` = a vision slot of reference `k`.
    let mut expanded: Vec<Option<usize>> = Vec::with_capacity(ids.len());
    let mut next = 0usize;
    for &id in &ids {
        if id == image_token {
            let (h, w) = *grids.get(next).ok_or_else(|| {
                Error::Msg(format!(
                    "{TRAINER_ID} trainer: the image-conditioned template carries more \
                     placeholders than the {} reference images",
                    grids.len()
                ))
            })?;
            expanded.extend(std::iter::repeat_n(
                Some(next),
                h * w / IMAGE_TOKENS_PER_SLOT,
            ));
            next += 1;
        } else {
            expanded.push(None);
        }
    }
    if next != grids.len() || expanded.len() <= drop {
        return Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: the image-conditioned template placed {next} of {} reference \
             images in a {}-token prompt (system prefix {drop})",
            grids.len(),
            expanded.len()
        )));
    }
    let mut segments: Vec<S> = Vec::with_capacity(2 * grids.len() + 2);
    let (mut text, mut current) = (0usize, None);
    for slot in &expanded[drop..] {
        match *slot {
            None => {
                text += 1;
                current = None;
            }
            Some(k) if current != Some(k) => {
                if text > 0 {
                    segments.push(S::Text { len: text });
                    text = 0;
                }
                let (h, w) = grids[k];
                segments.push(S::Image {
                    height: h,
                    width: w,
                });
                current = Some(k);
            }
            Some(_) => {}
        }
    }
    if text > 0 {
        segments.push(S::Text { len: text });
    }
    segments.push(S::Image {
        height: (height / VAE_SCALE_FACTOR) as usize,
        width: (width / VAE_SCALE_FACTOR) as usize,
    });
    Ok(JointLayout { segments })
}

/// One prompt the preflight prices: `(text, ordered reference paths, target (width, height),
/// whether its reference latents are cached for the whole run)`.
type PreflightPrompt<'a> = (&'a str, &'a [PathBuf], (u32, u32), bool);

/// Largest per-head block-causal prefix call: `max((end − start)·end)`. The target call
/// is priced separately from its target-row count and total key count.
pub fn largest_prefix_score_call(layout: &JointLayout) -> u64 {
    layout
        .prefix_segments()
        .iter()
        .map(|&(start, end, _)| ((end - start) * end) as u64)
        .max()
        .unwrap_or(0)
}

/// Sum of all prefix calls, per head.
pub fn prefix_score_elements(layout: &JointLayout) -> u64 {
    layout
        .prefix_segments()
        .iter()
        .map(|&(start, end, _)| ((end - start) * end) as u64)
        .sum()
}

/// The size an item's **target** trains at. A captioned (text-to-image) item is the centre-cropped
/// `edge × edge` square, as before. An edit pair's target keeps its **aspect ratio** — the render
/// path's own fit, [`calculate_dimensions`]`(edge², w/h)` on the 32-px grid — because its references
/// keep theirs: cropping the target square while the references stay whole would teach the adapter
/// "zoom into the middle" and break the spatial correspondence the edit is about (sc-24161).
/// Reads only the image header.
fn edit_target_size(item: &TrainingItem, edge: u32) -> Result<(u32, u32)> {
    if !item.is_edit_pair() {
        return Ok((edge, edge));
    }
    let (w, h) = image::image_dimensions(&item.image_path).map_err(|e| {
        Error::Msg(format!(
            "{TRAINER_ID} trainer: read target image {}: {e}",
            item.image_path.display()
        ))
    })?;
    if w == 0 || h == 0 {
        return Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: target image {} is {w}x{h}",
            item.image_path.display()
        )));
    }
    Ok(calculate_dimensions(
        f64::from(edge) * f64::from(edge),
        f64::from(w) / f64::from(h),
    ))
}

/// Latent tokens of a `(width, height)` target.
fn target_tokens((width, height): (u32, u32)) -> u64 {
    (width / VAE_SCALE_FACTOR) as u64 * (height / VAE_SCALE_FACTOR) as u64
}

/// The target-latent sizing of a bucketed run (sc-2127): the latent tokens of the single largest
/// target (every item at its [`edit_target_size`] for every bucket edge — the largest edge decides
/// it, epic 2123 E7) and the target tokens the latent cache holds across the dataset (every item
/// once per edge). With one edge the cache total is the per-item sum the single-bucket cache held.
/// Reads only image headers.
fn bucketed_target_tokens(items: &[TrainingItem], edges: &[u32]) -> Result<(u64, u64)> {
    let (mut largest, mut cached) = (0u64, 0u64);
    for item in items {
        for &edge in edges {
            let tokens = target_tokens(edit_target_size(item, edge)?);
            largest = largest.max(tokens);
            cached += tokens;
        }
    }
    Ok((largest, cached))
}

/// One dataset item's text side, exactly as phase 1 caches it: its ordered references
/// host-preprocessed by [`prepare_conditioning_references`], the conditioning encoded **once**,
/// then one [`JointBranch`] per bucket edge in `edges` order (sc-2127) at the item's target size
/// for that edge ([`edit_target_size`]). The branches share one evaluated text-row array; only the
/// layout (the target block's grid) differs per edge. With one edge this is exactly
/// [`encode_branch`]. Returns the branches and the prepared references (the first item's also
/// condition edit previews).
fn item_branches(
    encoder: &QwenImage21TextEncoder,
    tokenizer: &TextTokenizer,
    drop: usize,
    item: &TrainingItem,
    edges: &[u32],
) -> Result<(Vec<JointBranch>, Vec<PreparedReference>)> {
    let references = prepare_conditioning_references(encoder, &decode_references(item)?)?;
    let conditioning = encoder.encode_conditioning(tokenizer, &item.caption, drop, &references)?;
    let mut branches: Vec<JointBranch> = Vec::with_capacity(edges.len());
    for &edge in edges {
        let (width, height) = edit_target_size(item, edge)?;
        let branch = match branches.first() {
            None => {
                let branch = joint_branch(&conditioning, &references, width, height)?;
                eval([&branch.text])?;
                branch
            }
            Some(first) => JointBranch {
                text: first.text.clone(),
                layout: joint_layout(&conditioning.image_pad_mask, &references, width, height)?,
            },
        };
        branches.push(branch);
    }
    Ok((branches, references))
}

/// An item's packed target latents, one per bucket edge in `edges` order (sc-2127), each at
/// [`edit_target_size`] for that edge: a captioned item's centre-cropped square (unchanged), an
/// edit pair's whole picture at its aspect-preserving fit. The image is decoded once.
///
/// Each latent comes with (subject-masked loss, sc-24828) its own latent loss-weight map: the
/// item's mask is read and checked **once**, then cropped with the same rule as the image
/// (`center_square` / `full`), area-resampled onto **that bucket's** latent grid and packed by the
/// same [`pack_latents`], so it lines up element-for-element with that bucket's packed target.
/// `None` when the technique is off (no file read).
fn encode_item_targets(
    vae: &QwenImage21Vae,
    item: &TrainingItem,
    edges: &[u32],
    mask: Option<&gen_core::SubjectMaskLoss>,
) -> Result<Vec<(Array, Option<Array>)>> {
    let label = format!("{TRAINER_ID} trainer");
    let mask = PreparedSubjectMask::load_if_enabled(&label, item, mask)?;
    let decoded = decode_image(&item.image_path)?;
    let (image, crop): (Image, fn(u32, u32) -> CropBox) = if item.is_edit_pair() {
        (decoded, CropBox::full)
    } else {
        (center_crop_square(&decoded), CropBox::center_square)
    };
    edges
        .iter()
        .map(|&edge| {
            let (width, height) = edit_target_size(item, edge)?;
            let z = encode_latents_unpacked(vae, &image, width, height)?;
            let weight = prepared_subject_mask_weight(&label, mask.as_ref(), crop, z.shape())?
                .map(|w| pack_latents(&w))
                .transpose()?;
            Ok((pack_latents(&z)?, weight))
        })
        .collect()
}

/// One prompt's [`JointBranch`] through the render path's own assembly — the tower's
/// [`QwenImage21TextEncoder::encode_conditioning`] (the image-conditioned template with vision
/// tokens when `references` is non-empty, the text-to-image template otherwise) then
/// [`joint_branch`] at the target's `(width, height)` — evaluated so it survives the tower's drop.
fn encode_branch(
    encoder: &QwenImage21TextEncoder,
    tokenizer: &TextTokenizer,
    drop: usize,
    prompt: &str,
    references: &[PreparedReference],
    (width, height): (u32, u32),
) -> Result<JointBranch> {
    let conditioning = encoder.encode_conditioning(tokenizer, prompt, drop, references)?;
    let branch = joint_branch(&conditioning, references, width, height)?;
    eval([&branch.text])?;
    Ok(branch)
}

/// An edit item's reference latents, in reference order, through the render path's own host
/// preprocessing ([`prepare_references`]) and VAE encode ([`encode_references`]). Re-prepared from
/// disk (host work only) so the caption phase keeps no pixel buffers alive. Empty for a captioned
/// item, or when `vision` is `None` (a text-to-image run).
fn encode_item_references(
    vae: &QwenImage21Vae,
    vision: Option<&VisionConfig>,
    item: &TrainingItem,
) -> Result<Vec<Array>> {
    match vision {
        Some(vision) if item.is_edit_pair() => {
            encode_references(vae, &prepare_references(&decode_references(item)?, vision)?)
        }
        _ => Ok(Vec::new()),
    }
}

/// One cached training example: the target latent and the joint-sequence pieces the step feeds
/// the DiT — the conditioning rows and layout of [`joint_branch`] and (edit) the ordered reference
/// latents of [`encode_references`].
struct CachedItem {
    /// Packed target latent `[1, h·w, C]` (f32).
    x0: Array,
    /// Packed subject-mask loss weight, the shape of `x0` (sc-24828) — `None` when off.
    mask_weight: Option<Array>,
    /// The branch's text rows `[1, text_len, hidden]` (f32).
    text: Array,
    /// The joint layout, target block last.
    layout: JointLayout,
    /// Packed reference latents, in reference order (empty for text-to-image).
    references: Vec<Array>,
}

/// Encode an image into the denoiser-space latent the DiT trains on, before the pack: resize to
/// `width × height` + `[−1, 1]` NCHW, widen to opaque RGBA (a constant `+1` alpha plane) when the
/// VAE takes four channels, take the posterior **mode** and normalise `(z − mean)/std` →
/// `[1, z_dim, height/16, width/16]`. [`pack_latents`] flattens it unpatched to
/// `[1, (height/16)·(width/16), z_dim]` (see [`encode_item_targets`]).
fn encode_latents_unpacked(
    vae: &QwenImage21Vae,
    image: &Image,
    width: u32,
    height: u32,
) -> Result<Array> {
    let rgb = preprocess_init_image(image, width, height)?; // [1, 3, height, width]
    let input = if vae.config().in_channels == 4 {
        let alpha = Array::ones::<f32>(&[1, 1, height as i32, width as i32])?;
        concatenate_axis(&[&rgb, &alpha], 1)?
    } else {
        rgb
    };
    let mode = vae.encode_mode(&input)?;
    vae.normalize(&mode)
}

/// Tokens a caption contributes to the joint sequence (template rendered, system prefix dropped)
/// — the tokenizer alone, so the preflight knows the exact sequence before any weight loads.
fn caption_tokens(tokenizer: &TextTokenizer, drop: usize, caption: &str) -> Result<u64> {
    let tokens = tokenizer.tokenize_preformatted(&prompt_template(caption))?;
    Ok(tokens.ids.len().saturating_sub(drop) as u64)
}

/// `(x_t, target)` for one sample at flow-match `t`: `x_t = (1−t)·x0 + t·noise`,
/// `target = noise − x0` (the velocity the DiT predicts; see the module docs).
fn build_batch(x0: &Array, noise: &Array, t: f32) -> Result<(Array, Array)> {
    let one_minus = Array::from_slice(&[1.0 - t], &[1]);
    let s = Array::from_slice(&[t], &[1]);
    let x_t = add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?;
    Ok((x_t, target))
}

/// Sample a normalized flow-match timestep `t ∈ [1e-3, 1−1e-3]` — the SceneWorks
/// `sample_training_timestep` the sibling trainers port: `sigmoid(randn)` by default, `uniform` for
/// linear, `(uniform + sigmoid(randn))/2` for weighted; bias `high` → `√t`, `low` → `t²`.
/// Deterministic in `seed`.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let t = match normalize_cfg(timestep_type).as_str() {
        "linear" | "uniform" => {
            random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>()
        }
        "weighted" => {
            let k2 = random::key(seed ^ 0x9E37_79B9)?;
            let base = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>();
            let center = sigmoid(random::normal::<f32>(&[1], None, None, Some(&k2))?.item::<f32>());
            (base + center) / 2.0
        }
        _ => sigmoid(random::normal::<f32>(&[1], None, None, Some(&k1))?.item::<f32>()),
    };
    let t = match normalize_cfg(timestep_bias).as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    Ok(t.clamp(1e-3, 1.0 - 1e-3))
}

/// The per-step inputs of one loss evaluation.
struct StepInputs<'a> {
    x0: &'a Array,
    /// The conditioning text rows ([`JointBranch::text`]).
    text: &'a Array,
    /// The joint layout ([`JointBranch::layout`]): text and condition blocks, target last.
    layout: &'a JointLayout,
    /// Packed reference latents in reference order — empty for text-to-image.
    references: &'a [Array],
    noise: &'a Array,
    t: f32,
    /// Subject-mask loss weight, the shape of `x0` (sc-24828) — `None` ⇒ the plain mean.
    mask_weight: Option<&'a Array>,
}

/// The fixed install/loss parameters of a run.
struct LossSpec<'a> {
    adapter: &'a TrainAdapter,
    alpha: f32,
    rank: f32,
    mae: bool,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    /// `Some` switches the forward to the per-block gradient-checkpointed path.
    checkpoint: Option<&'a [BlockTrainables]>,
}

/// One forward+backward over the trainable factors: install `params` (LoRA or LoKr), run the DiT
/// over the joint sequence — the denoise loop's own image stream ([`joint_images`]: references in
/// order, then the noised target) on the step's layout — regress the **target block's** velocity
/// onto `noise − x0`, return `(loss, grads)`. The DiT returns the target block only, so the
/// condition tokens never enter the loss. The graph runs at `spec.dtype`; the noising math, the
/// loss and the grads stay f32 (master weights).
#[cfg(test)]
fn compute_loss_grads(
    transformer: &mut QwenImage21Transformer,
    params: &LoraParams,
    spec: &LossSpec<'_>,
    step: &StepInputs<'_>,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) = compute_step_loss_grads(transformer, params, spec, step, None)?;
    Ok((losses.total, grads))
}

/// The per-step loss breakdown [`compute_step_loss_grads`] returns (epic 2123 E8).
#[derive(Clone, Copy, Debug, PartialEq)]
struct StepLosses {
    /// The differentiated step loss.
    total: f32,
    /// The diffusion (velocity-regression) term, `None` on an aux-only step (it contributed zero).
    diffusion: Option<f32>,
    /// The weighted aux-loss term, `None` when no aux loss contributed this step.
    aux: Option<f32>,
}

/// One aux-loss step's view of the trainer's [`PerceptualPath`].
struct AuxStep<'a> {
    path: &'a PerceptualPath,
    plan: &'a StepPlan,
    /// The step's reference key — its (item, bucket) cache entry.
    entry: usize,
}

/// The target block's latent grid `(h, w)` — the layout's last segment (validated non-empty image
/// by the DiT; refused here otherwise).
fn target_grid(layout: &JointLayout) -> Result<(i32, i32)> {
    match layout.segments.last() {
        Some(crate::transformer::Segment::Image { height, width }) => {
            Ok((*height as i32, *width as i32))
        }
        _ => Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: the joint layout must end with the target image block"
        ))),
    }
}

/// A packed target latent `[1, h·w, z_dim]` back to the decoder's NCHW `[1, z_dim, h, w]` — the
/// inverse of [`pack_latents`] (unpatched: one token per latent cell). The values stay in the
/// normalised `(z − mean)/std` space: TAEQI2.1 "consumes / produces normalized latents directly"
/// (its published diffusers wrapper sets `latents_mean = 0`, `latents_std = 1`).
fn unpack_to_decoder_layout(packed: &Array, layout: &JointLayout) -> Result<Array> {
    let (h, w) = target_grid(layout)?;
    let sh = packed.shape();
    if sh.len() != 3 || sh[1] != h * w {
        return Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: expected a packed [B, {}, C] target latent, got {sh:?}",
            h * w
        )));
    }
    Ok(packed
        .transpose_axes(&[0, 2, 1])?
        .reshape(&[sh[0], sh[2], h, w])?)
}

/// [`compute_loss_grads`] with the step's perceptual plan (epic 2123 E8): on an aux-only step the
/// diffusion term is not computed (it contributes zero) and the loss is the weighted perceptual term
/// on the **target block's** x0 estimate `x0 = x_t − t·v` (the DiT returns the target block's
/// velocity, regressed onto `noise − x0` with `x_t = (1−t)·x0 + t·noise`), unpacked to TAEQI2.1's
/// `[1, 64, h, w]` layout ([`unpack_to_decoder_layout`]). The same closure serves text-to-image and
/// edit (the reference blocks condition the forward; only the target is decoded) and the dense and
/// gradient-checkpointed forwards. With `aux = None` (or a diffusion-only plan with no aux loss)
/// the traced graph is exactly the pre-epic-2123 one.
fn compute_step_loss_grads(
    transformer: &mut QwenImage21Transformer,
    params: &LoraParams,
    spec: &LossSpec<'_>,
    step: &StepInputs<'_>,
    aux: Option<AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    let (x_t_f32, target) = build_batch(step.x0, step.noise, step.t)?;
    let mask_weight = step.mask_weight.cloned();
    let x_t = x_t_f32.as_dtype(spec.dtype)?;
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    let context = step.text.clone();
    let (t, layout, references) = (step.t, step.layout, step.references);
    let LossSpec {
        adapter,
        alpha,
        rank,
        mae,
        lora_dtype,
        checkpoint: ckpt_blocks,
        ..
    } = *spec;
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // NEVER check the cancel flag in here (it returns `MlxResult`); cancellation is the
        // caller's job at the step boundary. Install ALL targets so the dense path (and any
        // global target on the checkpointed path) trains through ordinary autograd; on the
        // checkpointed path each block's adapters are rebuilt inside its segment from the
        // explicit-input factors.
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        let images = joint_images(references, &x_t);
        let v = match ckpt_blocks {
            Some(blocks) => transformer.forward_checkpointed_joint(
                &context,
                &images,
                t,
                layout,
                &CheckpointedTrainables {
                    params: &p,
                    blocks,
                    alpha,
                    rank,
                    lora_dtype,
                    lokr_dtype: LOKR_DTYPE,
                },
            ),
            None => transformer.forward_joint(&context, &images, t, layout),
        }
        .map_err(|e| Exception::custom(e.to_string()))?;
        let diffusion = if diffusion_on {
            let diff = subtract(&v, &target)?;
            // MSE / MAE, subject-mask weighted when on (sc-24828) — reduces to a 0-d scalar (grad
            // needs a scalar cotangent).
            Some(reduce_loss(&diff, mask_weight.as_ref(), mae)?)
        } else {
            None
        };
        let aux_term = match &aux {
            Some(a) if aux_on => {
                // Target-block x0 estimate in f32 (x0 = x_t − t·v), unpacked for TAEQI2.1.
                let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma: t }
                    .recover_x0(&x_t_f32, &v.as_dtype(Dtype::Float32)?)
                    .and_then(|x| unpack_to_decoder_layout(&x, layout))
                    .map_err(|e| Exception::custom(e.to_string()))?;
                a.path
                    .aux_loss(a.plan, a.entry, &x0_hat)
                    .map_err(|e| Exception::custom(e.to_string()))?
                    .map(|t| t.weighted)
            }
            _ => None,
        };
        // Only the first output is differentiated; the other two are reported terms.
        let zero = || Array::from_f32(0.0);
        let d_out = diffusion.clone().unwrap_or_else(zero);
        let a_out = aux_term.clone().unwrap_or_else(zero);
        let total =
            combine_step_loss(diffusion, aux_term).map_err(|e| Exception::custom(e.to_string()))?;
        Ok(vec![total, d_out, a_out])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (val, grads) = vg(params.clone(), 0)?;
    let losses = StepLosses {
        total: val[0].item::<f32>(),
        diffusion: diffusion_on.then(|| val[1].item::<f32>()),
        aux: aux_on.then(|| val[2].item::<f32>()),
    };
    Ok((losses, grads))
}

/// Qwen-Image 2.1's x0 decoder for the shared aux-loss builder (epic 2123 E8): TAEQI2.1
/// (`madebyollin/taeqi2_1`, the 64-channel, 16× RGBA latent API — `F16Decoder`).
fn taeqi2_1_decoder() -> mlx_gen_perceptual::DecoderSpec {
    mlx_gen_perceptual::DecoderSpec::Tiny {
        name: "TAEQI2.1",
        config: TinyDecoderSpec::taeqi2_1(),
    }
}

/// Build the epic-2123 perceptual path through the shared builder: `None` when no aux loss is
/// enabled (nothing loads; every step is the plain diffusion step).
fn load_perceptual_path(cfg: &TrainingConfig) -> Result<Option<PerceptualPath>> {
    mlx_gen_perceptual::build_perceptual_path(
        cfg,
        &mlx_gen_perceptual::AuxLossContext {
            label: "qwen_image_2_1 trainer",
            decoder: taeqi2_1_decoder(),
            latent_lpips: None,
        },
    )
}

/// The aux-model bytes ([`TrainingShape::aux_model_bytes`], epic 2123 E7) of `cfg` for a run whose
/// largest decoded target is the square `edge` or (edit) `target_tokens` latent cells, with
/// `entries` cached references: TAEQI2.1 + every enabled loss, sized at the square of equal area.
/// `0` when no aux loss is enabled.
fn perceptual_footprint_bytes(
    cfg: &TrainingConfig,
    edge: u32,
    target_tokens: u64,
    entries: usize,
) -> u64 {
    let token_pixels = (VAE_SCALE_FACTOR * VAE_SCALE_FACTOR) as u64;
    let pixels = (edge as u64 * edge as u64).max(target_tokens * token_pixels);
    let side = (pixels as f64).sqrt().ceil() as u32;
    mlx_gen_perceptual::perceptual_footprint(
        cfg,
        &taeqi2_1_decoder(),
        mlx_gen_perceptual::AuxGeometry::image(side, entries),
    )
}

/// Compute every cache entry's perceptual reference once (its clean packed target latent, unpacked
/// to the decoder layout at the entry's own target grid), keyed per (item, bucket) entry.
fn prepare_perceptual_references(path: &mut PerceptualPath, cache: &[CachedItem]) -> Result<()> {
    for (i, item) in cache.iter().enumerate() {
        path.ensure_reference(i, &unpack_to_decoder_layout(&item.x0, &item.layout)?)?;
    }
    Ok(())
}

/// One training micro-step on the 1-based `step`: pick the step's cached (item, bucket) entry,
/// sample its `t` and noise (seeded, exactly as before epic 2123), plan the step's loss terms through
/// the perceptual path (when one is configured: the alternation key comes from the item's own update
/// count, and an aux-only step trains at `t` remapped into the loss window), and run
/// [`compute_step_loss_grads`]. With no perceptual path every step is the plain diffusion step,
/// bit-identical to the pre-epic-2123 loop.
#[allow(clippy::too_many_arguments)]
fn run_train_step(
    transformer: &mut QwenImage21Transformer,
    params: &LoraParams,
    spec: &LossSpec<'_>,
    cfg: &TrainingConfig,
    cache: &[CachedItem],
    schedule: &BucketSchedule,
    perceptual: Option<(&mut PerceptualPath, &mut AuxAlternation)>,
    step: u32,
) -> Result<(StepLosses, LoraParams)> {
    let k = (step - 1) as usize;
    let (item_index, _bucket) = schedule.sample(k);
    let entry = schedule.cache_index(k);
    let item = &cache[entry];
    let mut t = sample_sigma(
        &cfg.timestep_type,
        &cfg.timestep_bias,
        cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
    )?;
    let noise = random::normal::<f32>(
        item.x0.shape(),
        None,
        None,
        Some(&random::key(
            cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
        )?),
    )?;
    let plan;
    let aux = match perceptual {
        Some((path, alternation)) => {
            // Normally a no-op (references were computed once, before the loop).
            path.ensure_reference(entry, &unpack_to_decoder_layout(&item.x0, &item.layout)?)?;
            plan = path.plan(alternation.key(step, item_index), entry, t)?;
            t = plan.noise_level;
            let path: &PerceptualPath = path;
            Some(AuxStep {
                path,
                plan: &plan,
                entry,
            })
        }
        None => None,
    };
    compute_step_loss_grads(
        transformer,
        params,
        spec,
        &StepInputs {
            x0: &item.x0,
            text: &item.text,
            layout: &item.layout,
            references: &item.references,
            noise: &noise,
            t,
            mask_weight: item.mask_weight.as_ref(),
        },
        aux,
    )
}

/// A resume must continue the same **training mode** (sc-24161): the checkpoint written at the
/// snapshot's step carries [`EDIT_ADAPTER_MARKER`] iff it was an edit run. Same-shaped factors from
/// a text-to-image run would otherwise silently continue as an edit adapter (or vice versa).
/// `checkpoint_mode` is that checkpoint's `trainingMode` metadata value, if any.
fn check_resumed_mode(checkpoint_mode: Option<&str>, edit: bool) -> Result<()> {
    let resumed_edit = checkpoint_mode == Some(EDIT_ADAPTER_MARKER.1);
    if resumed_edit != edit {
        let name = |edit: bool| {
            if edit {
                "an instruction-edit"
            } else {
                "a text-to-image"
            }
        };
        return Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: the resume snapshot is from {} run but this is {} run; \
             start a fresh run or resume with the original dataset",
            name(resumed_edit),
            name(edit)
        )));
    }
    Ok(())
}

/// A resumed snapshot must hold exactly this run's factors (same keys, same shapes): a run whose
/// rank, network type or targets changed since the snapshot cannot continue from it.
fn check_resumed_params(built: &LoraParams, resumed: &LoraParams) -> Result<()> {
    let mismatch = built.len() != resumed.len()
        || built
            .iter()
            .any(|(key, value)| resumed.get(key).is_none_or(|r| r.shape() != value.shape()));
    if mismatch {
        return Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: the resume snapshot's factors do not match this run's adapter \
             (rank, network type or target modules changed since it was written); start a fresh \
             run or resume with the original settings"
        )));
    }
    Ok(())
}

/// Render a preview from the in-progress (already-installed, concrete — so the render is
/// graph-free) adapter through the crate's own render path: seeded packed noise →
/// resolution-shifted flow-match Euler denoise (true CFG against the empty prompt when
/// `guidance > 1`) → RGBA decode composited over white. An edit run (sc-24161) previews an
/// **edit**: the prompt branches were assembled with the first dataset item's references and its
/// cached reference latents condition the denoise, exactly as a reference render does. A
/// best-effort nicety: failures are logged by the caller, never fatal.
#[allow(clippy::too_many_arguments)]
fn render_sample(
    dit: &QwenImage21Transformer,
    vae: &QwenImage21Vae,
    scheduler: &SchedulerConfig,
    pos: &JointBranch,
    neg: Option<&JointBranch>,
    reference_latents: &[Array],
    seed: u64,
    edge: u32,
    steps: usize,
    guidance: f32,
    cancel: &CancelFlag,
) -> Result<Image> {
    // The terminal-sigma stretch is undefined at one step.
    let steps = steps.max(2);
    let sigmas =
        crate::scheduler::sigmas(scheduler, steps, crate::scheduler::image_tokens(edge, edge))?;
    let latents = create_noise(seed, edge, edge, dit.config().in_channels)?;
    let negative = if guidance > 1.0 { neg } else { None };
    let references = (!reference_latents.is_empty()).then(|| ReferenceConditioning {
        latents: reference_latents,
        layout: &pos.layout,
        negative_layout: negative.map(|neg| &neg.layout),
    });
    let latents = denoise(
        DenoiseInputs {
            transformer: dit,
            sigmas: &sigmas,
            latents,
            prompt_embeds: &pos.text,
            negative_embeds: negative.map(|neg| &neg.text),
            true_cfg_scale: guidance,
            width: edge,
            height: edge,
            sampler: None,
            seed,
            cancel,
            references,
        },
        &mut |_: Progress| {},
    )?;
    let tiling = crate::memory_strategy::default_decode_is_bounded(edge, edge)
        .then(|| TilingConfig::spatial_only(DECODE_TILE_EDGE as i32, DECODE_OVERLAP as i32));
    decode_rgb(vae, &latents, edge, edge, tiling.as_ref(), Some(cancel))
}

impl Trainer for QwenImage21Trainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared floors: a control-branch or full-base-fine-tune request on this adapter-only
        // trainer is a typed `Unsupported`, never a silently trained plain adapter.
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
        // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        // Instruction-edit datasets (sc-24161): the shared floor caps references at this
        // descriptor's `max_reference_images` (the render path's own cap) and refuses mixed or
        // hybrid datasets; the snapshot must also carry the vision tower the references go through.
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        if req.items.iter().any(TrainingItem::is_edit_pair) && self.vision.is_none() {
            return Err(missing_vision_tower().into());
        }
        validate_request(req)?;
        if resolve_target_paths(&self.probe, &req.config).is_empty() {
            return Err(format!(
                "{TRAINER_ID} trainer: lora_target_modules {:?} matched no adaptable module on the \
                 Qwen-Image 2.1 DiT (default: every block's {})",
                req.config.lora_target_modules,
                BLOCK_ADAPTER_TARGETS.join("/")
            )
            .into());
        }
        Ok(())
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        // The full `validate` — the shared control / full-fine-tune / edit floors included — so a
        // caller that skips it cannot slip a control or full-tune request through as a plain
        // text-to-image run (the candle twin does the same).
        self.validate(req)?;
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl QwenImage21Trainer {
    /// The rich-`Result` body behind [`Trainer::train`].
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        validate_request(req)?;
        // Edit mode is a different training input, so its floor is re-checked here rather than
        // trusted to a prior `validate` call.
        gen_core::train::validate_edit_request(&self.descriptor, req)?;
        let edit = req.items.iter().any(TrainingItem::is_edit_pair);
        let vision = if edit {
            Some(self.vision.clone().ok_or_else(missing_vision_tower)?)
        } else {
            None
        };
        let cfg = &req.config;
        let target_paths = resolve_target_paths(&self.probe, cfg);
        if target_paths.is_empty() {
            return Err(format!(
                "{TRAINER_ID} trainer: lora_target_modules {:?} matched no adaptable module on the \
                 Qwen-Image 2.1 DiT",
                cfg.lora_target_modules
            )
            .into());
        }
        let compute_dtype = resolve_compute_dtype(&cfg.train_dtype);
        // bf16 mixed precision: the folded LoRA residual joins the bf16 activation stream (else
        // every adapted Linear silently re-promotes the chain to f32). f32 → no cast.
        let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
        let checkpointed = cfg.gradient_checkpointing;
        let sampling_requested = cfg.sample_every > 0 && !cfg.sample_prompts.is_empty();
        let sample_prompts: Vec<String> = if sampling_requested {
            cfg.sample_prompts
                .iter()
                .take(SAMPLE_PROMPT_CAP)
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        on_progress(TrainingProgress::Preparing);
        // sc-2127 — one training edge per resolution bucket (just `[resolution]` when buckets are
        // off). Every item trains at each edge; the preflight and the previews size for the
        // largest (epic 2123 E7).
        let edges = bucket_edges(cfg);
        let edge = edges.iter().copied().max().unwrap_or(0);

        // --- preflight: the derived peak against this device, before any weight is read ---
        // Edit previews condition on the first item's references (see `render_sample`).
        let sample_reference_paths: &[PathBuf] = if edit {
            &req.items[0].reference_image_paths
        } else {
            &[]
        };
        let mut longest = 0u64;
        let (mut reference_tokens, mut largest_reference_tokens, mut reference_cache_tokens) =
            (0u64, 0u64, 0u64);
        let (largest_target_tokens, target_cache_tokens) =
            bucketed_target_tokens(&req.items, &edges)?;
        let mut prefix_scores = 0u64;
        let mut largest_prefix_call = 0u64;
        let mut prompts: Vec<PreflightPrompt<'_>> = Vec::new();
        for item in &req.items {
            prompts.push((
                item.caption.as_str(),
                item.reference_image_paths.as_slice(),
                edit_target_size(item, edge)?,
                true,
            ));
        }
        // Previews render square at `edge²`, the positive prompts and the empty negative alike.
        prompts.extend(
            sample_prompts
                .iter()
                .map(|prompt| (prompt.as_str(), sample_reference_paths, (edge, edge), false)),
        );
        if sampling_requested {
            prompts.push(("", sample_reference_paths, (edge, edge), false));
        }
        for (text, reference_paths, size, cached) in prompts {
            let tokens = match vision.as_ref() {
                Some(vision) => {
                    let (text_tokens, sum, largest) = edit_prompt_tokens(
                        &self.tokenizer,
                        self.drop_count,
                        vision,
                        text,
                        reference_paths,
                    )?;
                    reference_tokens = reference_tokens.max(sum);
                    largest_reference_tokens = largest_reference_tokens.max(largest);
                    if cached {
                        reference_cache_tokens += sum;
                    }
                    let layout = edit_prompt_layout(
                        &self.tokenizer,
                        self.drop_count,
                        vision,
                        text,
                        reference_paths,
                        size,
                    )?;
                    prefix_scores = prefix_scores.max(prefix_score_elements(&layout));
                    largest_prefix_call =
                        largest_prefix_call.max(largest_prefix_score_call(&layout));
                    text_tokens
                }
                None => {
                    let tokens = caption_tokens(&self.tokenizer, self.drop_count, text)?;
                    prefix_scores = prefix_scores.max(tokens * tokens);
                    largest_prefix_call = largest_prefix_call.max(tokens * tokens);
                    tokens
                }
            };
            longest = longest.max(tokens);
        }
        let (trainable_params, lokr_delta_elements) =
            adapter_elements(&mut self.probe, &target_paths, cfg)?;
        let shape = TrainingShape {
            edge,
            target_tokens: largest_target_tokens,
            target_cache_tokens,
            caption_tokens: longest,
            reference_tokens,
            largest_reference_tokens,
            reference_cache_tokens,
            prefix_scores,
            largest_prefix_call,
            items: req.items.len() as u64,
            compute_width: if compute_dtype == Dtype::Float32 {
                4
            } else {
                2
            },
            trainable_params,
            optimizer_state_per_param: optimizer_state_per_param(&cfg.optimizer),
            lokr_delta_elements,
            checkpointed,
            sampling: sampling_requested,
            aux_model_bytes: perceptual_footprint_bytes(
                cfg,
                edge,
                largest_target_tokens,
                req.items.len() * edges.len(),
            ),
        };
        let budget = self
            .memory_budget_override
            .unwrap_or_else(device_budget_bytes);
        if std::env::var("QWEN_IMAGE_2_1_TRAINING_DIAGNOSTICS").as_deref() == Ok("1") {
            eprintln!(
                "training-shape {shape:?} facts {:#?} footprint {:?}",
                self.facts,
                training_footprint(&self.facts, &shape)
            );
        }
        let derived = training_footprint(&self.facts, &shape);
        training_memory_trace(
            "preflight",
            Some(derived),
            Some(
                usize::try_from(
                    derived
                        .train_phase
                        .saturating_sub(self.facts.dit_elements * shape.compute_width)
                        .max(1 << 30),
                )
                .unwrap_or(usize::MAX),
            ),
        );
        check_training_footprint(&self.facts, &shape, budget)?;
        // Bound MLX's freed-buffer pool for the rest of the run to the derived working set above
        // the DiT. Unbounded, the allocator pools every freed buffer up to ~0.95 x the device's
        // recommended working set, so the process footprint the OS sees climbs to that pool line
        // whatever the run needs (the first real-weight edit run crossed the evidence lane's
        // 100 GB phys_footprint ceiling). The render path bounds the pool per request the same way
        // (`memory_strategy::AllocatorBounds`, sc-24114); training bounds ONLY the cache, never
        // MLX's memory limit, so a figure the derived model rounds can never throttle a step.
        let _pool = TrainingPoolBound::enter(
            derived
                .train_phase
                .saturating_sub(self.facts.dit_elements * shape.compute_width)
                .max(1 << 30),
        );
        training_memory_trace("pool_bound", None, Some(_pool.effective));

        // --- resume admission, before any model loads ---
        // The run's identity (sc-24163): its training config and its dataset fingerprint (item
        // order, captions, image / ordered reference paths and contents). Every resume bundle
        // records it, and a resume whose config, dataset or training mode changed is refused HERE —
        // from the snapshot's metadata alone, right after the preflight, as the candle twin does —
        // so a refused resume never pays the text-encoder / VAE caching pass.
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string();
        let fingerprint = checkpoint::request_fingerprint(req)?;
        let identity = checkpoint::ResumeIdentity {
            config: cfg,
            request_fingerprint: &fingerprint,
        };
        let resume_from = if cfg.resume {
            checkpoint::find_latest_resume(&req.output_dir, &stem)
        } else {
            None
        };
        if let Some((snapshot, step)) = &resume_from {
            let meta = gen_core::weightsmeta::safetensors_file_metadata(
                req.output_dir.join(checkpoint_filename(&stem, *step)),
            )?;
            check_resumed_mode(meta.get(EDIT_ADAPTER_MARKER.0).map(String::as_str), edit)?;
            checkpoint::check_resume_fingerprints(
                &gen_core::weightsmeta::safetensors_file_metadata(snapshot)?
                    .into_iter()
                    .collect::<std::collections::HashMap<_, _>>(),
                cfg,
                &fingerprint,
            )?;
        }

        // --- 1. captions: the Qwen3-VL tower, encoded ONCE, then dropped ---
        // Every prompt goes through the render path's own assembly: references preprocessed by
        // `prepare_conditioning_references`, the (image-conditioned, for edit) template encoded by
        // `encode_conditioning`, then `joint_branch` → the text rows + joint layout the DiT sees.
        // A captioned item has no references, so this is the text-to-image conditioning exactly.
        // An edit run loads the vision tower too (its references are vision context).
        on_progress(TrainingProgress::LoadingModel);
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let (branches, sample_caps, sample_neg) = {
            let encoder: QwenImage21TextEncoder =
                loader::load_text_encoder_from(&self.root.join("text_encoder"), vision.as_ref())?;
            // Previews render square (`render_sample`), so their branches assemble at `edge²`.
            let encode = |prompt: &str, references: &[PreparedReference]| {
                encode_branch(
                    &encoder,
                    &self.tokenizer,
                    self.drop_count,
                    prompt,
                    references,
                    (edge, edge),
                )
            };
            // One `Vec` of per-edge branches per item (sc-2127), in `edges` order.
            let mut branches: Vec<Vec<JointBranch>> = Vec::with_capacity(req.items.len());
            let mut sample_references = Vec::new();
            for (i, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let (item_edges, references) =
                    item_branches(&encoder, &self.tokenizer, self.drop_count, item, &edges)?;
                branches.push(item_edges);
                if i == 0 && edit && sampling_requested {
                    sample_references = references;
                }
            }
            let mut sample_caps: Vec<(String, JointBranch)> =
                Vec::with_capacity(sample_prompts.len());
            for prompt in &sample_prompts {
                sample_caps.push((prompt.clone(), encode(prompt, &sample_references)?));
            }
            let sample_neg = if sample_caps.is_empty() {
                None
            } else {
                Some(encode("", &sample_references)?)
            };
            (branches, sample_caps, sample_neg)
            // `encoder` drops here: every caption is cached, the tower is idle from now on.
        };
        training_memory_trace(
            "caption_tower_dropped_before_clear",
            None,
            Some(_pool.effective),
        );
        mlx_rs::memory::clear_cache();
        training_memory_trace("caption_cache_cleared", None, Some(_pool.effective));

        // --- 2. latents: the VAE encodes each image ONCE; the encoder half is then dropped ---
        let mut vae = loader::load_vae(&self.root)?;
        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127). An item's bucket entries
        // share its text rows and reference latents (refcounted clones).
        let mut cache: Vec<CachedItem> = Vec::with_capacity(req.items.len() * edges.len());
        for (i, (item, per_edge)) in req.items.iter().zip(branches).enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let targets = encode_item_targets(&vae, item, &edges, cfg.subject_mask_loss.as_ref())?;
            let references = encode_item_references(&vae, vision.as_ref(), item)?;
            eval(
                targets
                    .iter()
                    .flat_map(|(x0, w)| std::iter::once(x0).chain(w.iter()))
                    .chain(references.iter()),
            )?;
            for ((x0, mask_weight), branch) in targets.into_iter().zip(per_edge) {
                cache.push(CachedItem {
                    x0,
                    mask_weight,
                    text: branch.text,
                    layout: branch.layout,
                    references: references.clone(),
                });
            }
        }
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            return Err(format!("{TRAINER_ID} trainer: no usable dataset items").into());
        }
        vae.drop_encoder();
        let vae: Option<QwenImage21Vae> = (!sample_caps.is_empty()).then_some(vae);
        // Epic 2123 depth anchoring: load the frozen TAEQI2.1 decoder + aux models (small — they
        // stay resident through training; priced in the preflight's train phase) and compute each
        // (item, bucket) entry's reference ONCE, before the loop. A missing checkpoint fails here,
        // before the DiT loads.
        let mut perceptual = load_perceptual_path(cfg)?;
        if let Some(path) = perceptual.as_mut() {
            prepare_perceptual_references(path, &cache)?;
        }
        training_memory_trace(
            "vae_encoder_dropped_before_clear",
            None,
            Some(_pool.effective),
        );
        mlx_rs::memory::clear_cache();
        training_memory_trace("latent_cache_cleared", None, Some(_pool.effective));
        // Cancelled during caching: nothing has trained, so write nothing (and skip the DiT load).
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 3. the dense DiT, at the training compute dtype ---
        let mut transformer = loader::load_transformer(&self.root)?;
        if transformer.is_quantized() {
            return Err(quantized_base_refusal("the loaded DiT is packed"));
        }
        if transformer.compute_dtype() != compute_dtype {
            transformer.cast_weights(compute_dtype)?;
        }
        training_memory_trace("dit_loaded", None, Some(_pool.effective));

        // --- adapter targets + params (LoRA or LoKr) + optimizer ---
        let rank = cfg.rank as f32;
        let alpha = cfg.alpha;
        let (adapter, mut params) = match cfg.network_type {
            NetworkType::Lora => {
                let (targets, params) =
                    build_lora_targets(&mut transformer, &target_paths, cfg.rank as i32, cfg.seed)?;
                (TrainAdapter::Lora { targets }, params)
            }
            NetworkType::Lokr => {
                let (targets, params) = build_lokr_targets(
                    &mut transformer,
                    &target_paths,
                    cfg.rank as i32,
                    cfg.decompose_factor,
                    cfg.seed,
                )?;
                (TrainAdapter::Lokr { targets }, params)
            }
        };
        let blocks = if checkpointed {
            Some(block_trainables(
                &mut transformer,
                &target_paths,
                &params,
                cfg,
            )?)
        } else {
            None
        };
        let mae = matches!(normalize_cfg(&cfg.loss_type).as_str(), "mae" | "l1");
        // AdamW with wd = 0 is Adam, so the one optimizer covers both choices.
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut opt = TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;
        let accum = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);

        // --- resume: continue from the snapshot admitted above (identity + mode already checked;
        // `load_resume_with_identity` re-checks the identity against the bundle it restores) ---
        let mut update_idx: u32 = 0;
        let mut start_step: u32 = 0;
        if let Some((snapshot, _)) = &resume_from {
            let (loaded, meta) =
                checkpoint::load_resume_with_identity(snapshot, &mut opt, identity)?;
            check_resumed_params(&params, &loaded)?;
            params = loaded;
            start_step = meta.step;
            update_idx = meta.update_idx;
        }

        // Provenance + licence on every saved adapter; an edit adapter is also marked as one.
        let provenance: Vec<(&str, &str)> = ADAPTER_PROVENANCE
            .iter()
            .copied()
            .chain(edit.then_some(EDIT_ADAPTER_MARKER))
            .collect();

        let spec = LossSpec {
            adapter: &adapter,
            alpha,
            rank,
            mae,
            dtype: compute_dtype,
            lora_dtype,
            checkpoint: blocks.as_deref(),
        };

        // --- train loop ---
        // sc-2127: which cached (item, bucket) entry each step trains on (round-robin over items
        // for a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
        let schedule =
            BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
        // Epic 2123 E8: per-image, per-update alternation keys for the perceptual losses, keyed on
        // the real dataset item (not the (item, bucket) entry). A resumed run replays the skipped
        // prefix so the phase matches.
        let mut alternation = perceptual
            .as_ref()
            .map(|_| AuxAlternation::new(cache.len() / edges.len(), accum));
        if let Some(alt) = alternation.as_mut() {
            for step in 1..=start_step {
                alt.key(step, schedule.sample((step - 1) as usize).0);
            }
        }
        let mut accumulated: Option<LoraParams> = None;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            training_memory_trace(&format!("step_{step}_begin"), None, Some(_pool.effective));
            let (losses, grads) = run_train_step(
                &mut transformer,
                &params,
                &spec,
                cfg,
                &cache,
                &schedule,
                perceptual.as_mut().zip(alternation.as_mut()),
                step,
            )?;
            last_loss = losses.total;
            training_memory_trace(
                &format!("step_{step}_gradients_returned"),
                None,
                Some(_pool.effective),
            );
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                // The final update can fire with fewer than `accum` grads; divide by the actual
                // in-window count.
                let window = if step % accum == 0 {
                    accum
                } else {
                    step % accum
                };
                let avg = average_grads(
                    accumulated
                        .take()
                        .expect("an update fires only after accumulation"),
                    window,
                )?;
                // Epic 2123 (sc-24827): clip → gradient noise → step → weight noise.
                adapter_optimizer_update(&mut opt, &mut params, &avg, cfg, update_idx, cfg.seed)?;
                update_idx += 1;
            }

            training_memory_trace(
                &format!("step_{step}_before_progress"),
                None,
                Some(_pool.effective),
            );
            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                adapter.save_with_meta(
                    &params,
                    alpha,
                    rank,
                    cfg.decompose_factor,
                    "",
                    &provenance,
                    &ckpt,
                )?;
                // The resume bundle holds the factors + optimizer state, not the in-flight
                // gradient-accumulation buffer, so it is written only on an optimizer-update
                // boundary: resuming from it is then exact. A `save_every` that lands inside an
                // accumulation window still writes the adapter checkpoint above, and the run
                // resumes from the latest boundary snapshot instead of silently dropping the
                // partial window's gradients.
                if step % accum == 0 {
                    checkpoint::save_resume_with_identity(
                        &req.output_dir,
                        &stem,
                        step,
                        update_idx,
                        &opt,
                        &params,
                        identity,
                    )?;
                }
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // Periodic best-effort previews. The current factors install as CONCRETE adapters
            // (evaluated arrays outside any trace), so the render builds no autograd graph; the
            // next step's traced `loss_fn` re-installs them. A failure never aborts training.
            if let (Some(vae), true) = (vae.as_ref(), step % cfg.sample_every.max(1) == 0) {
                adapter.install_as(
                    &mut transformer,
                    &params,
                    alpha,
                    rank,
                    lora_dtype,
                    LOKR_DTYPE,
                )?;
                let total = sample_caps.len() as u32;
                // Edit previews condition on the first item's cached reference latents.
                let sample_reference_latents: &[Array] =
                    if edit { &cache[0].references } else { &[] };
                for (i, (prompt, pos)) in sample_caps.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match render_sample(
                        &transformer,
                        vae,
                        &self.scheduler,
                        pos,
                        sample_neg.as_ref(),
                        sample_reference_latents,
                        sample_seed,
                        edge,
                        cfg.sample_steps as usize,
                        cfg.sample_guidance_scale,
                        &req.cancel,
                    ) {
                        Ok(image) => on_progress(TrainingProgress::Sample {
                            step,
                            index: i as u32 + 1,
                            total,
                            prompt: prompt.clone(),
                            image,
                        }),
                        Err(Error::Canceled) => break,
                        Err(e) => eprintln!(
                            "[sc-24159] {TRAINER_ID} preview sample failed at step {step} (prompt \
                             {}): {e} — skipping this preview, training continues",
                            i + 1
                        ),
                    }
                }
                mlx_rs::memory::clear_cache();
            }
        }

        // Cancelled before a single step completed: the factors are still the no-op init. Surface
        // the cancellation rather than writing a valid-looking identity adapter.
        if steps_run == 0 {
            return Err(Error::Canceled);
        }

        // --- save the final adapter (PEFT keys + reload contract + provenance/licence) ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        adapter.save_with_meta(
            &params,
            alpha,
            rank,
            cfg.decompose_factor,
            "",
            &provenance,
            &adapter_path,
        )?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/lokr_rounding.rs"]
mod lokr_rounding;

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;
    use mlx_rs::optimizers::clip_grad_norm;

    use crate::transformer::Segment;

    fn tiny_snapshot() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot")
    }

    /// The single-bucket branch of [`item_branches`] (the pre-bucket `item_branch`).
    fn item_branch(
        encoder: &QwenImage21TextEncoder,
        tokenizer: &TextTokenizer,
        drop: usize,
        item: &TrainingItem,
        edge: u32,
    ) -> Result<(JointBranch, Vec<PreparedReference>)> {
        let (mut branches, references) = item_branches(encoder, tokenizer, drop, item, &[edge])?;
        Ok((branches.remove(0), references))
    }

    /// The single-bucket latent of [`encode_item_targets`] (the pre-bucket `encode_item_target`).
    fn encode_item_target(vae: &QwenImage21Vae, item: &TrainingItem, edge: u32) -> Result<Array> {
        Ok(encode_item_targets(vae, item, &[edge], None)?.remove(0).0)
    }

    fn base_config() -> TrainingConfig {
        TrainingConfig {
            rank: 4,
            steps: 10,
            ..Default::default()
        }
    }

    fn req_with(config: TrainingConfig) -> TrainingRequest {
        TrainingRequest {
            items: vec![TrainingItem::captioned(
                PathBuf::from("/nonexistent/x.png"),
                "a swatch".into(),
            )],
            config,
            output_dir: PathBuf::from("/nonexistent/qwen21_unused"),
            file_name: "lora.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    #[test]
    fn descriptor_is_the_qwen_image_2_1_route() {
        let d = trainer_descriptor();
        assert_eq!(d.id, "qwen_image_2_1");
        assert_eq!(TRAINER_ID, crate::MODEL_ID);
        assert_eq!(d.family, "qwen-image-2-1");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.supports_lora && d.supports_lokr);
        assert!(!d.supports_control && !d.supports_full_finetune);
        // sc-2127: multi-resolution buckets are honored (the shared floor would refuse them
        // otherwise).
        assert!(d.techniques.resolution_buckets);
        // sc-24161: edit-capable, capped at the render path's own reference limit (one constant).
        assert_eq!(d.max_reference_images as usize, MAX_REFERENCE_IMAGES);
        assert_eq!(MAX_REFERENCE_IMAGES, 10);
    }

    #[test]
    fn reachable_via_the_trainer_registry_by_id() {
        assert!(
            crate::provider_registry()
                .unwrap()
                .trainers()
                .any(|r| (r.descriptor)().id == TRAINER_ID),
            "trainer id {TRAINER_ID} not registered"
        );
    }

    #[test]
    fn provenance_names_the_family_base_and_research_licence() {
        let meta: std::collections::BTreeMap<&str, &str> =
            ADAPTER_PROVENANCE.iter().copied().collect();
        assert_eq!(meta["family"], "qwen-image-2-1");
        assert_eq!(meta["baseModel"], "qwen_image_2_1");
        assert_eq!(meta["ss_base_model_version"], "qwen_image_2_1");
        assert!(meta["license"].contains("Qwen Research License"));
        assert!(meta["licenseNotice"].contains("Qwen RESEARCH LICENSE AGREEMENT"));
        // The reload-contract keys are written by the shared saver; the stamp must never shadow
        // them (the saver writes them last, but a collision here would still be a bug).
        for key in ["networkType", "rank", "alpha", "decomposeFactor"] {
            assert!(!meta.contains_key(key), "{key} is a reload-contract key");
        }
    }

    #[test]
    fn validate_rejects_empty_dataset_and_zero_rank_or_steps() {
        let mut r = req_with(base_config());
        r.items.clear();
        let err = validate_request(&r).unwrap_err().to_string();
        assert!(err.contains("dataset is empty"), "{err}");
        let err = validate_request(&req_with(TrainingConfig {
            rank: 0,
            ..base_config()
        }))
        .unwrap_err()
        .to_string();
        assert!(err.contains("rank"), "{err}");
        let err = validate_request(&req_with(TrainingConfig {
            steps: 0,
            ..base_config()
        }))
        .unwrap_err()
        .to_string();
        assert!(err.contains("steps"), "{err}");
    }

    #[test]
    fn validate_rejects_unrecognized_optimizer_timestep_and_loss() {
        for cfg in [
            TrainingConfig {
                optimizer: "nope".into(),
                ..base_config()
            },
            TrainingConfig {
                timestep_type: "bogus".into(),
                ..base_config()
            },
            TrainingConfig {
                timestep_bias: "sideways".into(),
                ..base_config()
            },
            TrainingConfig {
                loss_type: "huber".into(),
                ..base_config()
            },
        ] {
            assert!(validate_request(&req_with(cfg)).is_err());
        }
        assert!(validate_request(&req_with(TrainingConfig {
            timestep_type: "Weighted".into(),
            timestep_bias: "high-noise".into(),
            loss_type: "L1".into(),
            optimizer: "adamw".into(),
            gradient_checkpointing: true,
            ..base_config()
        }))
        .is_ok());
    }

    #[test]
    fn build_batch_is_the_flow_match_velocity_with_no_sign_flip() {
        let x0 = Array::from_slice(&[2.0f32, 4.0, 6.0], &[1, 3, 1]);
        let noise = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3, 1]);
        let (x_t, target) = build_batch(&x0, &noise, 0.25).unwrap();
        assert_eq!(target.as_slice::<f32>(), &[-1.0, -3.0, -5.0]);
        for (got, want) in x_t.as_slice::<f32>().iter().zip([1.75f32, 3.25, 4.75]) {
            assert!((got - want).abs() < 1e-6, "x_t {got} != {want}");
        }
    }

    #[test]
    fn sample_sigma_is_deterministic_and_in_range() {
        for kind in ["sigmoid", "linear", "weighted"] {
            for bias in ["balanced", "high", "low"] {
                let a = sample_sigma(kind, bias, 42).unwrap();
                assert_eq!(a, sample_sigma(kind, bias, 42).unwrap());
                assert!((1e-3..=1.0 - 1e-3).contains(&a), "{kind}/{bias} t={a}");
            }
        }
        assert!(
            sample_sigma("sigmoid", "high", 7).unwrap()
                > sample_sigma("sigmoid", "low", 7).unwrap()
        );
    }

    #[test]
    fn compute_dtype_follows_train_dtype() {
        assert_eq!(resolve_compute_dtype("bf16"), Dtype::Bfloat16);
        assert_eq!(resolve_compute_dtype(" BFloat16 "), Dtype::Bfloat16);
        assert_eq!(resolve_compute_dtype("f32"), Dtype::Float32);
        assert_eq!(resolve_compute_dtype("fp16?"), Dtype::Float32);
    }

    /// The default surface is every [`BLOCK_ADAPTER_TARGETS`] Linear of every block (and no
    /// global); an explicit list reaches the globals too.
    #[test]
    fn default_targets_are_every_block_linear_and_explicit_ones_reach_the_globals() {
        let probe = loader::load_transformer_lazy(&tiny_snapshot()).unwrap();
        let paths = resolve_target_paths(&probe, &TrainingConfig::default());
        assert_eq!(
            paths.len(),
            probe.num_blocks() * BLOCK_ADAPTER_TARGETS.len()
        );
        for i in 0..probe.num_blocks() {
            for local in BLOCK_ADAPTER_TARGETS {
                let want = format!("transformer_blocks.{i}.{local}");
                assert!(paths.contains(&want), "missing default target {want}");
            }
        }
        let explicit = resolve_target_paths(
            &probe,
            &TrainingConfig {
                lora_target_modules: vec!["proj_out".into(), "modulation.1".into()],
                ..Default::default()
            },
        );
        assert_eq!(explicit, ["modulation.1", "proj_out"]);
        assert!(crate::GLOBAL_ADAPTER_TARGETS.contains(&"proj_out"));
        let none = resolve_target_paths(
            &probe,
            &TrainingConfig {
                lora_target_modules: vec!["no_such_module".into()],
                ..Default::default()
            },
        );
        assert!(none.is_empty());
    }

    // ── preflight ────────────────────────────────────────────────────────────────────────────

    /// The released snapshot's facts, spelled from the frozen header counts this crate already
    /// records (`memory_strategy::derived`) and the production configs — 2.1's own numbers.
    fn production_facts() -> FootprintFacts {
        use crate::memory_strategy::derived as d;
        let dit = TransformerConfig::production();
        FootprintFacts {
            dit_elements: d::DIT_LINEAR_PARAMS + d::DIT_DENSE_BYTES / d::BF16_WIDTH,
            text_encoder_bytes: d::resident_weights(Tier::Bf16).conditioning,
            vae_encoder_bytes: d::VAE_BYTES / 2,
            vae_decoder_bytes: d::VAE_BYTES / 2,
            num_layers: dit.num_layers as u64,
            inner: dit.inner_dim() as u64,
            heads: dit.num_attention_heads as u64,
            mlp_ratio: dit.mlp_ratio as u64,
            latent_channels: dit.in_channels as u64,
            text_hidden: 4096,
            vae_encode_channels: d::VAE_ENCODE_FULL_RES_CHANNELS,
            vae_decode_channels: d::VAE_FULL_RES_CHANNELS,
            pixels_per_token: d::PIXELS_PER_TOKEN,
            vision_tower_bytes: 0,
            vision_hidden: 1152,
        }
    }

    fn shape(edge: u32, checkpointed: bool) -> TrainingShape {
        TrainingShape {
            edge,
            target_tokens: 0,
            target_cache_tokens: 0,
            caption_tokens: 64,
            reference_tokens: 0,
            largest_reference_tokens: 0,
            reference_cache_tokens: 0,
            prefix_scores: 0,
            largest_prefix_call: 0,
            items: 20,
            compute_width: 2,
            trainable_params: 40_000_000,
            optimizer_state_per_param: 2,
            lokr_delta_elements: 0,
            checkpointed,
            sampling: false,
            aux_model_bytes: 0,
        }
    }

    #[test]
    fn footprint_grows_with_resolution_and_checkpointing_shrinks_it() {
        let facts = production_facts();
        for checkpointed in [false, true] {
            let a = training_footprint(&facts, &shape(512, checkpointed)).peak();
            let b = training_footprint(&facts, &shape(1024, checkpointed)).peak();
            let c = training_footprint(&facts, &shape(1536, checkpointed)).peak();
            assert!(a < b && b < c, "{a} {b} {c}");
        }
        for edge in [512, 1024, 1536] {
            let dense = training_footprint(&facts, &shape(edge, false)).train_phase;
            let ckpt = training_footprint(&facts, &shape(edge, true)).train_phase;
            assert!(ckpt < dense, "checkpointing must lower the step at {edge}");
        }
        // The dense bf16 DiT alone is the floor of the train stage.
        let fp = training_footprint(&facts, &shape(512, true));
        assert!(fp.train_phase > facts.dit_elements * 2);
        // f32 training costs more than bf16.
        let f32_shape = TrainingShape {
            compute_width: 4,
            ..shape(1024, true)
        };
        assert!(
            training_footprint(&facts, &f32_shape).peak()
                > training_footprint(&facts, &shape(1024, true)).peak()
        );
    }

    #[test]
    fn preflight_refuses_over_budget_with_actionable_advice_and_passes_under() {
        let facts = production_facts();
        let dense = shape(1024, false);
        let ckpt_peak = training_footprint(&facts, &shape(1024, true)).peak();
        let dense_peak = training_footprint(&facts, &dense).peak();
        // A budget between the two: dense is refused, and the message says checkpointing fits.
        let budget = (ckpt_peak + dense_peak) / 2;
        let err = check_training_footprint(&facts, &dense, budget)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Gradient Checkpointing"), "{err}");
        assert!(err.contains("which fits"), "{err}");
        assert!(err.contains("1024 px"), "{err}");
        assert!(err.contains("before step 1"), "{err}");
        assert!(check_training_footprint(&facts, &shape(1024, true), budget).is_ok());
        // A tiny budget refuses even the checkpointed run, and says it is still over.
        let err = check_training_footprint(&facts, &dense, 1 << 30)
            .unwrap_err()
            .to_string();
        assert!(err.contains("still over"), "{err}");
    }

    /// `Σ out·in` over the default targets of the released DiT — what a LoKr install materialises.
    fn production_lokr_delta_elements() -> u64 {
        let dit = TransformerConfig::production();
        let inner = dit.inner_dim() as u64;
        let ffn = inner * dit.mlp_ratio as u64;
        // to_q/to_k/to_v/to_out.0 are inner→inner; gate_layer/proj inner→ffn; out ffn→inner.
        dit.num_layers as u64 * (4 * inner * inner + 3 * inner * ffn)
    }

    /// LoKr materialises a dense bf16 delta per target, so at the same rank it must derive a
    /// larger footprint than LoRA: by the whole delta set on a dense step, by one block's worth
    /// (twice: forward + recompute) on a checkpointed one, and by the whole set again in the
    /// preview install in both modes.
    ///
    /// *Mutation that reds this:* dropping `step_deltas` from the step, or `lokr_deltas` from
    /// the preview arm.
    #[test]
    fn lokr_deltas_raise_the_footprint_above_lora_at_the_same_rank() {
        let facts = production_facts();
        let deltas = production_lokr_delta_elements();
        for checkpointed in [false, true] {
            for sampling in [false, true] {
                let lora = TrainingShape {
                    sampling,
                    ..shape(1024, checkpointed)
                };
                let lokr = TrainingShape {
                    lokr_delta_elements: deltas,
                    ..lora
                };
                let (a, b) = (
                    training_footprint(&facts, &lora).train_phase,
                    training_footprint(&facts, &lokr).train_phase,
                );
                assert!(
                    b > a,
                    "LoKr must cost more than LoRA (checkpointed={checkpointed}, \
                     sampling={sampling}): {b} <= {a}"
                );
            }
        }
        // A dense LoKr step holds every delta: ~the bf16 size of the targeted Linears.
        let lora = training_footprint(&facts, &shape(1024, false)).train_phase;
        let lokr = training_footprint(
            &facts,
            &TrainingShape {
                lokr_delta_elements: deltas,
                ..shape(1024, false)
            },
        )
        .train_phase;
        assert_eq!(lokr - lora, deltas * LOKR_DELTA_WIDTH);
    }

    /// The optimizer's own state is sized per optimizer: Prodigy keeps four f32 buffers per
    /// element, AdamW two, Rose none.
    #[test]
    fn optimizer_state_is_sized_per_optimizer() {
        assert_eq!(optimizer_state_per_param("adamw"), 2);
        assert_eq!(optimizer_state_per_param("Adam"), 2);
        assert_eq!(optimizer_state_per_param("adamw8bit"), 2);
        assert_eq!(optimizer_state_per_param("rose"), 0);
        assert_eq!(optimizer_state_per_param("prodigy"), 4);
        assert_eq!(optimizer_state_per_param("Prodigy-Opt"), 4);
        let facts = production_facts();
        let with = |state| {
            training_footprint(
                &facts,
                &TrainingShape {
                    optimizer_state_per_param: state,
                    ..shape(1024, true)
                },
            )
            .train_phase
        };
        assert!(with(0) < with(2) && with(2) < with(4));
        assert_eq!(with(4) - with(2), 2 * 40_000_000 * F32_WIDTH);
    }

    /// Latents accumulate while the VAE is resident, so the latent stage carries their cache.
    #[test]
    fn the_latent_stage_carries_the_latent_cache() {
        let facts = production_facts();
        let few = training_footprint(&facts, &shape(1024, true)).latent_phase;
        let many = training_footprint(
            &facts,
            &TrainingShape {
                items: 1020,
                ..shape(1024, true)
            },
        )
        .latent_phase;
        // 1000 more items: their captions AND their packed latents.
        let per_item = 64 * facts.text_hidden * F32_WIDTH
            + (1024 / 16) * (1024 / 16) * facts.latent_channels * F32_WIDTH;
        assert_eq!(many - few, 1000 * per_item);
    }

    /// sc-2127 (epic 2123 E7): a bucketed run's preflight prices its **largest** target (the
    /// largest edge decides it) and a latent cache that holds every item at every bucket; one
    /// bucket is exactly the single-bucket cache.
    ///
    /// *Mutations that red this:* `bucketed_target_tokens` sizing `largest` from the first edge,
    /// or caching only one edge per item; `training_footprint` ignoring `target_cache_tokens`.
    #[test]
    fn bucketed_preflight_sizes_for_the_largest_edge_and_caches_every_bucket() {
        let items: Vec<TrainingItem> = (0..3)
            .map(|i| {
                TrainingItem::captioned(PathBuf::from(format!("/nonexistent/{i}.png")), "x".into())
            })
            .collect();
        let (t512, t1024) = ((512 / 16) * (512 / 16), (1024 / 16) * (1024 / 16));
        assert_eq!(
            bucketed_target_tokens(&items, &[512, 1024]).unwrap(),
            (t1024, 3 * (t512 + t1024))
        );
        assert_eq!(
            bucketed_target_tokens(&items, &[1024, 512]).unwrap(),
            (t1024, 3 * (t512 + t1024)),
            "order-independent"
        );
        assert_eq!(
            bucketed_target_tokens(&items, &[1024]).unwrap(),
            (t1024, 3 * t1024)
        );

        // An edit pair's aspect-preserving target is priced per edge too.
        let tmp = tempfile::tempdir().unwrap();
        let wide = write_png(tmp.path(), "wide.png", 256, 128, 1);
        let reference = write_png(tmp.path(), "ref.png", 64, 64, 2);
        let pair = [TrainingItem::edit_pair(wide, "w".into(), vec![reference])];
        let (small, large) = (
            target_tokens(edit_target_size(&pair[0], 64).unwrap()),
            target_tokens(edit_target_size(&pair[0], 128).unwrap()),
        );
        assert_eq!(large, (192 / 16) * (96 / 16));
        assert!(small < large);
        assert_eq!(
            bucketed_target_tokens(&pair, &[64, 128]).unwrap(),
            (large, small + large)
        );

        // The footprint: buckets [512, 1024] price the 1024 step plus the 512 latents cached on
        // top; an explicit single-bucket cache equals the `0` default.
        let facts = production_facts();
        let at_1024 = training_footprint(&facts, &shape(1024, true));
        let at_512 = training_footprint(&facts, &shape(512, true));
        let (largest, cached) = bucketed_target_tokens(
            &(0..20)
                .map(|i| TrainingItem::captioned(PathBuf::from(format!("{i}.png")), "x".into()))
                .collect::<Vec<_>>(),
            &[512, 1024],
        )
        .unwrap();
        let bucketed = training_footprint(
            &facts,
            &TrainingShape {
                target_tokens: largest,
                target_cache_tokens: cached,
                ..shape(1024, true)
            },
        );
        let extra = 20 * t512 * facts.latent_channels * F32_WIDTH;
        assert_eq!(bucketed.latent_phase - at_1024.latent_phase, extra);
        assert_eq!(bucketed.train_phase - at_1024.train_phase, extra);
        assert!(bucketed.peak() > at_512.peak());
        assert_eq!(
            training_footprint(
                &facts,
                &TrainingShape {
                    target_cache_tokens: 20 * t1024,
                    ..shape(1024, true)
                }
            ),
            at_1024
        );
    }

    /// sc-2127: each item caches one branch and one target latent per bucket edge, in edge order —
    /// the conditioning encoded once (shared text rows), the target block / latent sized for that
    /// edge — and each bucket entry is bit-identical to a single-bucket run at that edge. Covers
    /// a captioned item (square) and an edit pair (aspect-preserving per edge).
    ///
    /// *Mutations that red this:* `item_branches` building every layout at the first edge;
    /// `encode_item_targets` encoding every latent at one edge.
    #[test]
    fn an_item_caches_one_branch_and_latent_per_bucket_edge() {
        let root = tiny_snapshot();
        let tmp = tempfile::tempdir().unwrap();
        let square = write_png(tmp.path(), "sq.png", 160, 160, 3);
        let wide = write_png(tmp.path(), "wide.png", 256, 128, 5);
        let reference = write_png(tmp.path(), "ref.png", 64, 64, 9);
        let tokenizer = loader::load_tokenizer(&root).unwrap();
        let drop = system_prompt_drop_count(&tokenizer).unwrap();
        let encoder = loader::load_text_encoder(&root).unwrap();
        let vae = loader::load_vae(&root).unwrap();
        let edges = [64u32, 128];
        for item in [
            TrainingItem::captioned(square, "a swatch".into()),
            TrainingItem::edit_pair(wide, "widen it".into(), vec![reference]),
        ] {
            let (branches, _) = item_branches(&encoder, &tokenizer, drop, &item, &edges).unwrap();
            let targets: Vec<Array> = encode_item_targets(&vae, &item, &edges, None)
                .unwrap()
                .into_iter()
                .map(|(x0, _)| x0)
                .collect();
            assert_eq!((branches.len(), targets.len()), (2, 2));
            for (b, &edge) in edges.iter().enumerate() {
                let (w, h) = edit_target_size(&item, edge).unwrap();
                let (gw, gh) = ((w / 16) as usize, (h / 16) as usize);
                assert_eq!(
                    branches[b].layout.segments.last(),
                    Some(&Segment::Image {
                        height: gh,
                        width: gw
                    }),
                    "bucket {b} (edge {edge}) target block"
                );
                assert_eq!(targets[b].shape()[1] as usize, gw * gh, "bucket {b} latent");
                let (single, _) = item_branch(&encoder, &tokenizer, drop, &item, edge).unwrap();
                assert_eq!(branches[b].layout, single.layout);
                assert_bit_equal("text rows", &branches[b].text, &single.text);
                assert_bit_equal(
                    "target latent",
                    &targets[b],
                    &encode_item_target(&vae, &item, edge).unwrap(),
                );
            }
            assert_ne!(
                branches[0].layout, branches[1].layout,
                "the two buckets train different grids"
            );
        }
    }

    /// sc-24828 × sc-2127: with subject-masked loss on and two buckets, every bucket's packed
    /// weight is resampled onto **that bucket's** latent grid — the shape of that bucket's packed
    /// target — and the masked-out (right-half) tokens are zero while the subject (left-half)
    /// tokens carry the subject weight. Covers a captioned item (centre square) and an edit pair
    /// (whole picture).
    ///
    /// *Mutations that red this:* `encode_item_targets` resampling the mask once at the first
    /// bucket's grid and reusing it for every bucket.
    #[test]
    fn subject_mask_weight_is_resampled_per_bucket() {
        let root = tiny_snapshot();
        let tmp = tempfile::tempdir().unwrap();
        let left_half_mask = |name: &str, width: u32, height: u32| {
            let img = image::GrayImage::from_fn(width, height, |x, _| {
                image::Luma([if x < width / 2 { 255 } else { 0 }])
            });
            let path = tmp.path().join(name);
            img.save(&path).unwrap();
            path
        };
        let mut captioned = TrainingItem::captioned(
            write_png(tmp.path(), "sq.png", 160, 160, 3),
            "a swatch".into(),
        );
        captioned.subject_mask_path = Some(left_half_mask("sq_mask.png", 160, 160));
        let mut edit = TrainingItem::edit_pair(
            write_png(tmp.path(), "wide.png", 256, 128, 5),
            "widen it".into(),
            vec![write_png(tmp.path(), "ref.png", 64, 64, 9)],
        );
        edit.subject_mask_path = Some(left_half_mask("wide_mask.png", 256, 128));
        let cfg = gen_core::SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        };
        let vae = loader::load_vae(&root).unwrap();
        let edges = [64u32, 128];
        for item in [captioned, edit] {
            let targets = encode_item_targets(&vae, &item, &edges, Some(&cfg)).unwrap();
            assert_eq!(targets.len(), 2);
            for (b, (&edge, (x0, weight))) in edges.iter().zip(&targets).enumerate() {
                let weight = weight.as_ref().expect("mask loss on ⇒ a weight");
                assert_eq!(weight.shape(), x0.shape(), "bucket {b} (edge {edge})");
                let (w, _) = edit_target_size(&item, edge).unwrap();
                let gw = (w / 16) as usize;
                let channels = x0.shape()[2] as usize;
                let flat = weight.reshape(&[-1]).unwrap();
                let flat = flat.as_slice::<f32>();
                for (token, cells) in flat.chunks(channels).enumerate() {
                    let gx = token % gw;
                    let want = if (gx + 1) * 2 <= gw {
                        1.0
                    } else if gx * 2 >= gw {
                        0.0
                    } else {
                        continue;
                    };
                    assert!(
                        cells.iter().all(|&v| (v - want).abs() < 1e-5),
                        "bucket {b} (edge {edge}) token {token}: {cells:?} want {want}"
                    );
                }
            }
        }
    }

    /// The preflight's adapter sizing on the tiny DiT: LoRA has no deltas; LoKr's deltas are
    /// exactly `Σ out·in` over the default targets.
    #[test]
    fn adapter_elements_size_lora_and_lokr_off_the_host() {
        let mut probe = loader::load_transformer_lazy(&tiny_snapshot()).unwrap();
        let lora_cfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let paths = resolve_target_paths(&probe, &lora_cfg);
        let mut want_deltas = 0u64;
        let mut want_lora = 0u64;
        for path in &paths {
            let segs: Vec<&str> = path.split('.').collect();
            let shape = probe.adaptable_facts(&segs).unwrap().base_shape;
            want_deltas += (shape[0] * shape[1]) as u64;
            want_lora += 4 * (shape[0] + shape[1]) as u64;
        }
        assert_eq!(
            adapter_elements(&mut probe, &paths, &lora_cfg).unwrap(),
            (want_lora, 0)
        );
        let lokr_cfg = TrainingConfig {
            network_type: NetworkType::Lokr,
            ..lora_cfg
        };
        let (trainable, deltas) = adapter_elements(&mut probe, &paths, &lokr_cfg).unwrap();
        assert_eq!(deltas, want_deltas);
        assert!(trainable > 0);
    }

    /// The facts come from the snapshot itself: the miniature snapshot derives its own (tiny)
    /// footprint rather than the release's.
    #[test]
    fn facts_are_read_off_the_snapshot_itself() {
        let facts = FootprintFacts::from_snapshot(&tiny_snapshot()).unwrap();
        let cfg =
            TransformerConfig::from_json_file(&tiny_snapshot().join("transformer/config.json"))
                .unwrap();
        assert_eq!(facts.num_layers, cfg.num_layers as u64);
        assert_eq!(facts.inner, cfg.inner_dim() as u64);
        assert_eq!(facts.latent_channels, cfg.in_channels as u64);
        assert!(facts.dit_elements > 0 && facts.text_encoder_bytes > 0);
        assert!(facts.vae_encoder_bytes > 0 && facts.vae_decoder_bytes > 0);
        let fp = training_footprint(&facts, &shape(64, false));
        assert!(
            fp.peak() < 4 << 30,
            "the tiny snapshot must derive a tiny footprint, got {} bytes",
            fp.peak()
        );
    }

    #[test]
    fn preflight_refusal_stops_the_run_before_any_weight_loads() {
        let mut trainer =
            QwenImage21Trainer::load(&LoadSpec::new(WeightsSource::Dir(tiny_snapshot()))).unwrap();
        trainer.set_memory_budget_override(Some(1 << 20));
        let out = tempfile::tempdir().unwrap();
        let mut req = req_with(TrainingConfig {
            resolution: 64,
            steps: 2,
            ..base_config()
        });
        req.output_dir = out.path().to_path_buf();
        let mut events = Vec::new();
        let err = trainer
            .train_impl(&req, &mut |p| events.push(format!("{p:?}")))
            .unwrap_err()
            .to_string();
        assert!(err.contains("exceeds"), "{err}");
        assert!(err.contains("Gradient Checkpointing"), "{err}");
        assert!(
            !events.iter().any(|e| e.starts_with("Caching")
                || e.starts_with("LoadingModel")
                || e.starts_with("Training")),
            "the refusal must precede every load: {events:?}"
        );
        assert_eq!(std::fs::read_dir(out.path()).unwrap().count(), 0);
    }

    // ── the step itself, on the tiny DiT ─────────────────────────────────────────────────────

    fn tiny_dit() -> QwenImage21Transformer {
        loader::load_transformer(&tiny_snapshot()).unwrap()
    }

    fn randn(shape: &[i32], seed: u64) -> Array {
        random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap())).unwrap()
    }

    /// Fixed synthetic batch at the DiT's own geometry: a 4×4 latent grid, 5 caption rows.
    fn fixed_batch(dit: &QwenImage21Transformer) -> (Array, Array, Array) {
        let c = dit.config();
        let x0 = randn(&[1, 16, c.in_channels as i32], 1);
        let ctx = randn(&[1, 5, c.context_in_dim as i32], 2);
        let noise = randn(&[1, 16, c.in_channels as i32], 3);
        (x0, ctx, noise)
    }

    /// The text-to-image layout of [`fixed_batch`]: 5 text rows, a 4×4 target.
    fn t2i_layout() -> JointLayout {
        JointLayout::text_to_image(5, 4, 4)
    }

    /// A fixed synthetic **edit** batch: two references (2×2 and 2×4 latent grids) interleaved with
    /// text exactly as `joint_layout` interleaves an image-conditioned prompt, and the 4×4 target
    /// last.
    struct EditBatch {
        x0: Array,
        text: Array,
        noise: Array,
        references: Vec<Array>,
        layout: JointLayout,
    }

    fn fixed_edit_batch(dit: &QwenImage21Transformer) -> EditBatch {
        let c = dit.config();
        let channels = c.in_channels as i32;
        let layout = JointLayout {
            segments: vec![
                Segment::Text { len: 3 },
                Segment::Image {
                    height: 2,
                    width: 2,
                },
                Segment::Text { len: 1 },
                Segment::Image {
                    height: 2,
                    width: 4,
                },
                Segment::Text { len: 4 },
                Segment::Image {
                    height: 4,
                    width: 4,
                },
            ],
        };
        EditBatch {
            x0: randn(&[1, 16, channels], 1),
            text: randn(&[1, 8, c.context_in_dim as i32], 2),
            noise: randn(&[1, 16, channels], 3),
            references: vec![randn(&[1, 4, channels], 4), randn(&[1, 8, channels], 5)],
            layout,
        }
    }

    struct Built {
        adapter: TrainAdapter,
        params: LoraParams,
        cfg: TrainingConfig,
        paths: Vec<String>,
    }

    fn build(dit: &mut QwenImage21Transformer, network: NetworkType) -> Built {
        let cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            network_type: network,
            decompose_factor: -1,
            ..Default::default()
        };
        let paths = resolve_target_paths(dit, &cfg);
        let (adapter, params) = match network {
            NetworkType::Lora => {
                let (targets, params) = build_lora_targets(dit, &paths, 4, 11).unwrap();
                (TrainAdapter::Lora { targets }, params)
            }
            NetworkType::Lokr => {
                let (targets, params) = build_lokr_targets(dit, &paths, 4, -1, 11).unwrap();
                (TrainAdapter::Lokr { targets }, params)
            }
        };
        Built {
            adapter,
            params,
            cfg,
            paths,
        }
    }

    /// AC: the loss decreases over N steps — a fixed batch (fixed `t`, fixed noise) overfits under
    /// the real step (install → DiT → MSE → grads → clip → AdamW).
    #[test]
    fn loss_decreases_over_steps_on_a_fixed_batch() {
        let mut dit = tiny_dit();
        let Built {
            adapter,
            mut params,
            ..
        } = build(&mut dit, NetworkType::Lora);
        let (x0, ctx, noise) = fixed_batch(&dit);
        let spec = LossSpec {
            adapter: &adapter,
            alpha: 4.0,
            rank: 4.0,
            mae: false,
            dtype: Dtype::Float32,
            lora_dtype: None,
            checkpoint: None,
        };
        let mut opt = TrainOptimizer::from_config("adamw", 1e-2, 0.0).unwrap();
        let mut losses = Vec::new();
        for _ in 0..30 {
            let (loss, grads) = compute_loss_grads(
                &mut dit,
                &params,
                &spec,
                &StepInputs {
                    x0: &x0,
                    text: &ctx,
                    layout: &t2i_layout(),
                    references: &[],
                    noise: &noise,
                    t: 0.5,
                    mask_weight: None,
                },
            )
            .unwrap();
            assert!(loss.is_finite(), "non-finite loss {loss}");
            losses.push(loss);
            let (clipped, _) = clip_grad_norm(&grads, 1.0).unwrap();
            let clipped: LoraParams = clipped
                .into_iter()
                .map(|(k, v)| (k, v.into_owned()))
                .collect();
            opt.step(&mut params, &clipped).unwrap();
            eval(params.values()).unwrap();
        }
        let (first, last) = (losses[0], *losses.last().unwrap());
        eprintln!("[sc-24159] fixed-batch loss {first:.5} -> {last:.5}");
        assert!(
            last < 0.9 * first,
            "the loss must fall on a fixed batch: {losses:?}"
        );
    }

    fn max_rel_diff(a: &LoraParams, b: &LoraParams) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut max_rel = 0f32;
        for (k, ga) in a {
            let gb = b.get(k).expect("same keys");
            let num = ga.subtract(gb).unwrap().abs().unwrap().max(None).unwrap();
            let den = ga.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        max_rel
    }

    /// The gradient-checkpointed block forward is the dense forward: same loss, same grads, for
    /// LoRA and LoKr alike (non-zero factors, so every factor's gradient is exercised).
    #[test]
    fn checkpointed_grads_match_dense_for_lora_and_lokr() {
        for network in [NetworkType::Lora, NetworkType::Lokr] {
            let mut dit = tiny_dit();
            let Built {
                adapter,
                params,
                cfg,
                paths,
            } = build(&mut dit, network);
            // Perturb every factor off its no-op init so all gradients are non-trivial.
            let params: LoraParams = params
                .into_iter()
                .enumerate()
                .map(|(i, (k, v))| {
                    let p =
                        multiply(randn(v.shape(), 100 + i as u64), Array::from_f32(0.05)).unwrap();
                    (k, p)
                })
                .collect();
            let blocks = block_trainables(&mut dit, &paths, &params, &cfg).unwrap();
            assert!(blocks
                .iter()
                .all(|b| b.adapter.is_some() && !b.keys.is_empty()));
            let (x0, ctx, noise) = fixed_batch(&dit);
            let layout = t2i_layout();
            let step = StepInputs {
                x0: &x0,
                text: &ctx,
                layout: &layout,
                references: &[],
                noise: &noise,
                t: 0.4,
                mask_weight: None,
            };
            let dense_spec = LossSpec {
                adapter: &adapter,
                alpha: 4.0,
                rank: 4.0,
                mae: false,
                dtype: Dtype::Float32,
                lora_dtype: None,
                checkpoint: None,
            };
            let (l_dense, g_dense) =
                compute_loss_grads(&mut dit, &params, &dense_spec, &step).unwrap();
            let ckpt_spec = LossSpec {
                checkpoint: Some(blocks.as_slice()),
                ..dense_spec
            };
            let (l_ckpt, g_ckpt) =
                compute_loss_grads(&mut dit, &params, &ckpt_spec, &step).unwrap();
            eval(g_dense.values().chain(g_ckpt.values())).unwrap();
            assert!(
                (l_dense - l_ckpt).abs() <= 1e-5 * l_dense.abs().max(1.0),
                "{network:?}: loss {l_dense} vs {l_ckpt}"
            );
            let rel = max_rel_diff(&g_dense, &g_ckpt);
            eprintln!("[sc-24159] {network:?} checkpointed-vs-dense grad max rel diff {rel:.2e}");
            assert!(rel < 1e-3, "{network:?}: max rel grad diff {rel:.2e}");
        }
    }

    #[test]
    fn a_resume_snapshot_from_another_shape_is_refused() {
        let mut dit = tiny_dit();
        let Built { params, .. } = build(&mut dit, NetworkType::Lora);
        assert!(check_resumed_params(&params, &params).is_ok());
        let mut fewer = params.clone();
        let key = fewer.keys().next().unwrap().clone();
        fewer.remove(&key);
        let err = check_resumed_params(&params, &fewer)
            .unwrap_err()
            .to_string();
        assert!(err.contains("resume snapshot"), "{err}");
    }

    // ── edit mode (sc-24161) ─────────────────────────────────────────────────────────────────

    /// AC: the reference cap comes from the descriptor (the render path's `MAX_REFERENCE_IMAGES`)
    /// — exactly ten references validate, eleven are refused naming the cap — and a mixed dataset
    /// is refused, all before any file is read.
    #[test]
    fn validate_caps_edit_references_at_the_render_paths_limit() {
        let trainer =
            QwenImage21Trainer::load(&LoadSpec::new(WeightsSource::Dir(tiny_snapshot()))).unwrap();
        let edit_req = |count: usize| {
            let mut req = req_with(base_config());
            req.items = vec![TrainingItem::edit_pair(
                PathBuf::from("/nonexistent/target.png"),
                "make it blue".into(),
                (0..count)
                    .map(|i| PathBuf::from(format!("/nonexistent/ref{i}.png")))
                    .collect(),
            )];
            req
        };
        assert!(trainer.validate(&edit_req(1)).is_ok());
        assert!(trainer.validate(&edit_req(MAX_REFERENCE_IMAGES)).is_ok());
        let err = trainer
            .validate(&edit_req(MAX_REFERENCE_IMAGES + 1))
            .unwrap_err()
            .to_string();
        assert!(err.contains("at most 10"), "{err}");

        let mut mixed = edit_req(1);
        mixed.items.push(TrainingItem::captioned(
            PathBuf::from("/nonexistent/x.png"),
            "a swatch".into(),
        ));
        let err = trainer.validate(&mixed).unwrap_err().to_string();
        assert!(err.contains("item 1 has none"), "{err}");
    }

    /// A deterministic `width × height` RGB PNG under `dir`.
    fn write_png(dir: &Path, name: &str, width: u32, height: u32, phase: u32) -> PathBuf {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([
                ((x * 5 + phase) % 256) as u8,
                ((y * 3 + 2 * phase) % 256) as u8,
                (((x + y) * 7 + phase) % 256) as u8,
            ])
        });
        let path = dir.join(name);
        img.save(&path).unwrap();
        path
    }

    fn assert_bit_equal(what: &str, got: &Array, want: &Array) {
        assert_eq!(got.shape(), want.shape(), "{what}: shape");
        assert!(
            got.all_close(want, Some(0.0), Some(0.0), None)
                .unwrap()
                .item::<bool>(),
            "{what}: the trainer's tensor differs from the render path's"
        );
    }

    /// A deterministic RGBA PNG whose alpha is a horizontal ramp (non-constant), plus the same
    /// pixels as the render path's `RgbaImage`.
    fn write_rgba_png(dir: &Path, name: &str, width: u32, height: u32) -> (PathBuf, RgbaImage) {
        let img = image::RgbaImage::from_fn(width, height, |x, y| {
            image::Rgba([
                ((x * 7) % 256) as u8,
                ((y * 11) % 256) as u8,
                ((x + y) % 256) as u8,
                (x * 255 / width.max(2)) as u8,
            ])
        });
        let path = dir.join(name);
        img.save(&path).unwrap();
        let rgba = RgbaImage {
            width,
            height,
            pixels: img.into_raw(),
        };
        (path, rgba)
    }

    /// AC (E11): the edit trainer assembles **the same joint sequence as the render path**. One
    /// edit item (a square RGB, a 4:1 RGB and a transparent RGBA reference, in that order) goes
    /// through the trainer's own caching functions (`item_branch`, `encode_item_references` — the
    /// exact calls phase 1/2 make), and the same pictures + prompt go through the code the render
    /// path RUNS (`model::assemble_reference_branches`, then `encode_references`), not a
    /// re-implementation of it. Layout, positional ids (hence RoPE offsets), text rows and every
    /// reference latent must be bit-identical, the RGBA reference's alpha must survive to its
    /// latent, and the layout must carry the references as distinct blocks in dataset order.
    ///
    /// *Mutations that red this:* `decode_reference` using `to_rgb8` (drops the alpha → the RGBA
    /// reference latent differs); `decode_references` reversing/sorting the paths (layout and
    /// latents differ); `item_branch` assembling at a different size than the render request.
    #[test]
    fn edit_training_assembles_the_render_paths_joint_sequence() {
        use mlx_gen::{Conditioning, GenerationRequest};

        let root = tiny_snapshot();
        let tmp = tempfile::tempdir().unwrap();
        let square = write_png(tmp.path(), "square.png", 64, 64, 0);
        let wide = write_png(tmp.path(), "wide.png", 256, 64, 90);
        let (layer, layer_rgba) = write_rgba_png(tmp.path(), "layer.png", 64, 64);
        assert!(!layer_rgba.is_opaque(), "the fixture must carry real alpha");
        let target = write_png(tmp.path(), "target.png", 64, 64, 33);
        let prompt = "put the second image's colours onto the first, under the third";
        let edge = 64u32;

        let tokenizer = loader::load_tokenizer(&root).unwrap();
        let drop = system_prompt_drop_count(&tokenizer).unwrap();
        let encoder = loader::load_text_encoder(&root).unwrap();
        let vision = loader::load_vision_config(&root).unwrap().unwrap();
        let vae = loader::load_vae(&root).unwrap();

        // The trainer's path, from the dataset item.
        let item = TrainingItem::edit_pair(
            target.clone(),
            prompt.into(),
            vec![square.clone(), wide.clone(), layer.clone()],
        );
        let (trained, _) = item_branch(&encoder, &tokenizer, drop, &item, edge).unwrap();
        let trained_refs = encode_item_references(&vae, Some(&vision), &item).unwrap();

        // The render path's own assembly, from a request carrying the same pictures.
        let rgb = |path: &Path| {
            let img = image::open(path).unwrap().to_rgb8();
            Image {
                width: img.width(),
                height: img.height(),
                pixels: img.into_raw(),
            }
        };
        let req = GenerationRequest {
            prompt: prompt.into(),
            width: edge,
            height: edge,
            conditioning: vec![
                Conditioning::Reference {
                    image: rgb(&square),
                    strength: None,
                },
                Conditioning::Reference {
                    image: rgb(&wide),
                    strength: None,
                },
                Conditioning::ReferenceRgba {
                    image: layer_rgba,
                    strength: None,
                },
            ],
            ..Default::default()
        };
        let rendered =
            crate::model::assemble_reference_branches(&encoder, &tokenizer, &req, drop, false)
                .unwrap();
        assert!(rendered.neg.is_none());
        let rendered_refs = encode_references(&vae, &rendered.references).unwrap();

        assert_eq!(trained.layout, rendered.pos.layout, "joint layout");
        // sc-24162: the preflight's header-only layout is the one the encoder really assembles,
        // so the attention it prices is this sequence's.
        assert_eq!(
            edit_prompt_layout(
                &tokenizer,
                drop,
                &vision,
                prompt,
                &item.reference_image_paths,
                edit_target_size(&item, edge).unwrap(),
            )
            .unwrap(),
            trained.layout,
            "the preflight's header-only layout"
        );
        assert_eq!(
            trained.layout.position_ids(),
            rendered.pos.layout.position_ids(),
            "positional ids (RoPE offsets)"
        );
        assert_bit_equal("text rows", &trained.text, &rendered.pos.text);
        assert_eq!(trained_refs.len(), 3);
        assert_eq!(rendered_refs.len(), 3);
        for (i, (got, want)) in trained_refs.iter().zip(&rendered_refs).enumerate() {
            assert_bit_equal(&format!("reference {i} latents"), got, want);
        }

        // The RGBA reference's alpha reaches its latent: the same picture made opaque encodes
        // differently.
        let opaque = TrainingItem::edit_pair(target.clone(), prompt.into(), vec![square.clone()]);
        let mut flattened = decode_references(&opaque).unwrap();
        flattened[0] = decode_reference(&layer).unwrap();
        for px in flattened[0].pixels.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let flat_latent = encode_references(
            &vae,
            &crate::reference::prepare_references(&flattened, &vision).unwrap(),
        )
        .unwrap();
        assert!(
            !flat_latent[0]
                .all_close(&trained_refs[2], Some(0.0), Some(0.0), None)
                .unwrap()
                .item::<bool>(),
            "the transparent reference's alpha must reach the VAE encode"
        );

        // The layout: three distinct reference blocks in dataset order (4×4, the 4:1 2×8, 4×4)
        // and the 4×4 target last.
        let blocks: Vec<(usize, usize)> = trained
            .layout
            .segments
            .iter()
            .filter_map(|s| match *s {
                Segment::Image { height, width } => Some((height, width)),
                Segment::Text { .. } => None,
            })
            .collect();
        assert_eq!(blocks, [(4, 4), (2, 8), (4, 4), (4, 4)]);

        // Order is semantic: the swapped item assembles a different sequence.
        let swapped = TrainingItem::edit_pair(target, prompt.into(), vec![wide, square, layer]);
        let (swapped, _) = item_branch(&encoder, &tokenizer, drop, &swapped, edge).unwrap();
        assert_ne!(
            swapped.layout, trained.layout,
            "reference order must be kept"
        );
    }

    /// Major (review): an edit target keeps its aspect ratio — a 2:1 target trains as a 2:1 target
    /// block, not a centre-cropped square — so it stays spatially aligned with its (whole)
    /// references. Text-to-image items are unchanged (square).
    ///
    /// *Mutation that reds this:* `edit_target_size` returning `(edge, edge)` for edit pairs, or
    /// `encode_item_target` centre-cropping an edit target.
    #[test]
    fn a_wide_edit_target_trains_as_a_wide_target_block() {
        let root = tiny_snapshot();
        let tmp = tempfile::tempdir().unwrap();
        let target = write_png(tmp.path(), "wide_target.png", 256, 128, 5);
        let reference = write_png(tmp.path(), "ref.png", 64, 64, 9);
        let edge = 128u32;
        let item = TrainingItem::edit_pair(target.clone(), "widen it".into(), vec![reference]);

        let size = edit_target_size(&item, edge).unwrap();
        assert_eq!(
            size,
            (192, 96),
            "calculate_dimensions(128², 2) on the 32-px grid"
        );
        let captioned = TrainingItem::captioned(target, "a swatch".into());
        assert_eq!(edit_target_size(&captioned, edge).unwrap(), (edge, edge));

        let tokenizer = loader::load_tokenizer(&root).unwrap();
        let drop = system_prompt_drop_count(&tokenizer).unwrap();
        let encoder = loader::load_text_encoder(&root).unwrap();
        let (branch, _) = item_branch(&encoder, &tokenizer, drop, &item, edge).unwrap();
        let h = 96 / 16;
        assert_eq!(
            branch.layout.segments.last(),
            Some(&Segment::Image {
                height: h,
                width: 2 * h
            }),
            "the target block is {h}×{}",
            2 * h
        );

        let vae = loader::load_vae(&root).unwrap();
        let x0 = encode_item_target(&vae, &item, edge).unwrap();
        assert_eq!(
            x0.shape()[1] as usize,
            h * 2 * h,
            "the target latent covers the whole 2:1 picture"
        );
        let square = encode_item_target(&vae, &captioned, edge).unwrap();
        assert_eq!(square.shape()[1], (128 / 16) * (128 / 16));
    }

    /// Review: a reference the Qwen3-VL processor would rebind (its fit leaves the pixel budget)
    /// is refused by the header-only preflight with the render path's own `Unsupported` — before
    /// the text encoder or vision tower load.
    ///
    /// *Mutation that reds this:* `edit_prompt_tokens` using `reference_target_size` without the
    /// `smart_resize` check (the refusal then only comes from `prepare_reference`, after
    /// `LoadingModel`).
    #[test]
    fn a_rebinding_reference_is_refused_before_any_weight_loads() {
        let tmp = tempfile::tempdir().unwrap();
        // 16:1 → the tiny 64-px fit is 256×32 = 8192 px, over the tiny processor's 4096-px cap.
        let thin = write_png(tmp.path(), "thin.png", 512, 32, 1);
        let target = write_png(tmp.path(), "target.png", 64, 64, 2);
        let vision = loader::load_vision_config(&tiny_snapshot())
            .unwrap()
            .unwrap();
        let tokenizer = loader::load_tokenizer(&tiny_snapshot()).unwrap();
        let drop = system_prompt_drop_count(&tokenizer).unwrap();
        match edit_prompt_tokens(
            &tokenizer,
            drop,
            &vision,
            "edit",
            std::slice::from_ref(&thin),
        ) {
            Err(Error::Unsupported(message)) => {
                assert!(message.contains("smart_resize"), "{message}")
            }
            other => panic!("expected the render path's Unsupported, got {other:?}"),
        }

        let mut trainer =
            QwenImage21Trainer::load(&LoadSpec::new(WeightsSource::Dir(tiny_snapshot()))).unwrap();
        let out = tempfile::tempdir().unwrap();
        let mut req = req_with(TrainingConfig {
            resolution: 64,
            steps: 2,
            ..base_config()
        });
        req.items = vec![TrainingItem::edit_pair(target, "edit".into(), vec![thin])];
        req.output_dir = out.path().to_path_buf();
        let mut events = Vec::new();
        let err = trainer
            .train_impl(&req, &mut |p| events.push(format!("{p:?}")))
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
        assert!(
            !events.iter().any(|e| e.starts_with("LoadingModel")),
            "the refusal must precede every load: {events:?}"
        );
    }

    /// Scope (review): a resume continues the same training mode — an edit run never resumes a
    /// text-to-image snapshot (or vice versa) just because the factor shapes match.
    ///
    /// *Mutation that reds this:* dropping the `check_resumed_mode` call / making it compare
    /// nothing.
    #[test]
    fn a_resume_across_training_modes_is_refused() {
        assert!(check_resumed_mode(None, false).is_ok());
        assert!(check_resumed_mode(Some("edit"), true).is_ok());
        let err = check_resumed_mode(None, true).unwrap_err().to_string();
        assert!(
            err.contains("from a text-to-image run but this is an instruction-edit run"),
            "{err}"
        );
        let err = check_resumed_mode(Some("edit"), false)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("from an instruction-edit run but this is a text-to-image run"),
            "{err}"
        );
    }

    /// AC: edit training learns the **conditional** velocity, not one memorised sample — with a
    /// fresh seeded noise draw and timestep every step (as the train loop samples them), the
    /// windowed-average loss falls; and the references genuinely condition the forward:
    /// different reference latents give a different loss at the same factors.
    ///
    /// *Mutations that red this:* the step ignoring `references` (the two losses coincide);
    /// regressing onto the wrong target or not stepping the factors (the window does not fall).
    #[test]
    fn edit_loss_decreases_over_steps_and_depends_on_the_references() {
        let mut dit = tiny_dit();
        let Built {
            adapter,
            mut params,
            ..
        } = build(&mut dit, NetworkType::Lora);
        let EditBatch {
            x0,
            text,
            noise,
            references,
            layout,
        } = fixed_edit_batch(&dit);
        let spec = LossSpec {
            adapter: &adapter,
            alpha: 4.0,
            rank: 4.0,
            mae: false,
            dtype: Dtype::Float32,
            lora_dtype: None,
            checkpoint: None,
        };

        let other: Vec<Array> = references
            .iter()
            .enumerate()
            .map(|(i, r)| randn(r.shape(), 50 + i as u64))
            .collect();
        let (with_refs, _) = compute_loss_grads(
            &mut dit,
            &params,
            &spec,
            &StepInputs {
                x0: &x0,
                text: &text,
                layout: &layout,
                references: &references,
                noise: &noise,
                t: 0.5,
                mask_weight: None,
            },
        )
        .unwrap();
        let (with_other, _) = compute_loss_grads(
            &mut dit,
            &params,
            &spec,
            &StepInputs {
                x0: &x0,
                text: &text,
                layout: &layout,
                references: &other,
                noise: &noise,
                t: 0.5,
                mask_weight: None,
            },
        )
        .unwrap();
        assert!(
            (with_refs - with_other).abs() > 1e-7,
            "the reference latents must enter the forward: {with_refs} vs {with_other}"
        );

        let mut opt = TrainOptimizer::from_config("adamw", 1e-2, 0.0).unwrap();
        let mut losses = Vec::new();
        for step in 0..120u64 {
            let noise = randn(x0.shape(), 1_000 + step);
            let t = sample_sigma("sigmoid", "balanced", 77 + step).unwrap();
            let (loss, grads) = compute_loss_grads(
                &mut dit,
                &params,
                &spec,
                &StepInputs {
                    x0: &x0,
                    text: &text,
                    layout: &layout,
                    references: &references,
                    noise: &noise,
                    t,
                    mask_weight: None,
                },
            )
            .unwrap();
            assert!(loss.is_finite(), "non-finite loss {loss}");
            losses.push(loss);
            let (clipped, _) = clip_grad_norm(&grads, 1.0).unwrap();
            let clipped: LoraParams = clipped
                .into_iter()
                .map(|(k, v)| (k, v.into_owned()))
                .collect();
            opt.step(&mut params, &clipped).unwrap();
            eval(params.values()).unwrap();
        }
        let window = |w: &[f32]| w.iter().sum::<f32>() / w.len() as f32;
        let (first, last) = (window(&losses[..20]), window(&losses[losses.len() - 20..]));
        eprintln!("[sc-24161] varying-noise edit loss window {first:.5} -> {last:.5}");
        assert!(
            last < 0.9 * first,
            "the windowed edit loss must fall under varying noise/t: {losses:?}"
        );
    }

    /// The gradient-checkpointed joint forward is the dense joint forward on an **edit** layout
    /// too: same loss, same grads.
    #[test]
    fn checkpointed_edit_grads_match_dense() {
        let mut dit = tiny_dit();
        let Built {
            adapter,
            params,
            cfg,
            paths,
        } = build(&mut dit, NetworkType::Lora);
        let params: LoraParams = params
            .into_iter()
            .enumerate()
            .map(|(i, (k, v))| {
                let p = multiply(randn(v.shape(), 200 + i as u64), Array::from_f32(0.05)).unwrap();
                (k, p)
            })
            .collect();
        let blocks = block_trainables(&mut dit, &paths, &params, &cfg).unwrap();
        let EditBatch {
            x0,
            text,
            noise,
            references,
            layout,
        } = fixed_edit_batch(&dit);
        let step = StepInputs {
            x0: &x0,
            text: &text,
            layout: &layout,
            references: &references,
            noise: &noise,
            t: 0.4,
            mask_weight: None,
        };
        let dense_spec = LossSpec {
            adapter: &adapter,
            alpha: 4.0,
            rank: 4.0,
            mae: false,
            dtype: Dtype::Float32,
            lora_dtype: None,
            checkpoint: None,
        };
        let (l_dense, g_dense) = compute_loss_grads(&mut dit, &params, &dense_spec, &step).unwrap();
        let ckpt_spec = LossSpec {
            checkpoint: Some(blocks.as_slice()),
            ..dense_spec
        };
        let (l_ckpt, g_ckpt) = compute_loss_grads(&mut dit, &params, &ckpt_spec, &step).unwrap();
        eval(g_dense.values().chain(g_ckpt.values())).unwrap();
        assert!(
            (l_dense - l_ckpt).abs() <= 1e-5 * l_dense.abs().max(1.0),
            "loss {l_dense} vs {l_ckpt}"
        );
        let rel = max_rel_diff(&g_dense, &g_ckpt);
        assert!(rel < 1e-3, "max rel grad diff {rel:.2e}");
    }

    /// sc-24828: the packed subject-mask weight lines up token-for-token with the packed latent —
    /// [`pack_latents`] of a `[1, C, h, w]` weight whose value encodes `(y, x)` puts `y·10 + x` on
    /// every channel of token `y·w + x`, exactly the token the target latent's cell lands in.
    #[test]
    fn packed_subject_weight_lines_up_with_packed_latent_tokens() {
        let (c, h, w) = (3i32, 2usize, 3usize);
        let values: Vec<f32> = (0..h * w).map(|i| ((i / w) * 10 + i % w) as f32).collect();
        let shape = [1, c, h as i32, w as i32];
        let weight = mlx_gen::train::loss::subject_mask_weight(&values, h, w, &shape).unwrap();
        let packed = pack_latents(&weight).unwrap();
        assert_eq!(packed.shape(), &[1, (h * w) as i32, c]);
        // Flatten row-major (the pack is a strided view; the reshape copies it out in order).
        let packed = packed.reshape(&[-1]).unwrap();
        let packed = packed.as_slice::<f32>().to_vec();
        for token in 0..h * w {
            for ch in 0..c as usize {
                let want = ((token / w) * 10 + token % w) as f32;
                assert_eq!(
                    packed[token * c as usize + ch],
                    want,
                    "token {token} ch {ch}"
                );
            }
        }
    }

    /// sc-24828: the subject-mask weight reaches the loss on every backward path — text-to-image
    /// and edit layouts, dense and gradient-checkpointed. An all-ones map is the plain loss, an
    /// all-background map zeroes the loss and every adapter grad, a half map lies strictly between
    /// and agrees across paths.
    #[test]
    fn subject_mask_weight_reaches_both_backward_paths() {
        let mut dit = tiny_dit();
        let Built {
            adapter,
            params,
            cfg,
            paths,
        } = build(&mut dit, NetworkType::Lora);
        let params: LoraParams = params
            .into_iter()
            .enumerate()
            .map(|(i, (k, v))| {
                let p = multiply(randn(v.shape(), 300 + i as u64), Array::from_f32(0.05)).unwrap();
                (k, p)
            })
            .collect();
        let blocks = block_trainables(&mut dit, &paths, &params, &cfg).unwrap();
        let channels = dit.config().in_channels as i32;
        // The packed weight exactly as `encode_item_targets` builds it: a [1, C, 4, 4] map packed.
        let map = |w: &[f32]| {
            let unpacked =
                mlx_gen::train::loss::subject_mask_weight(w, 4, 4, &[1, channels, 4, 4]).unwrap();
            pack_latents(&unpacked).unwrap()
        };
        let ones = map(&[1.0; 16]);
        let zeros = map(&[0.0; 16]);
        let half: Vec<f32> = (0..16).map(|i| if i % 4 < 2 { 1.0 } else { 0.0 }).collect();
        let half = map(&half);
        let dense_spec = LossSpec {
            adapter: &adapter,
            alpha: 4.0,
            rank: 4.0,
            mae: false,
            dtype: Dtype::Float32,
            lora_dtype: None,
            checkpoint: None,
        };
        let ckpt_spec = LossSpec {
            checkpoint: Some(blocks.as_slice()),
            ..dense_spec
        };
        let (x0, ctx, noise) = fixed_batch(&dit);
        let t2i = t2i_layout();
        let edit = fixed_edit_batch(&dit);
        let no_refs: Vec<Array> = Vec::new();
        for (mode, x0, text, noise, layout, references) in [
            ("t2i", &x0, &ctx, &noise, &t2i, &no_refs),
            (
                "edit",
                &edit.x0,
                &edit.text,
                &edit.noise,
                &edit.layout,
                &edit.references,
            ),
        ] {
            let mut run = |weight: Option<&Array>, spec: &LossSpec<'_>| {
                let step = StepInputs {
                    x0,
                    text,
                    layout,
                    references,
                    noise,
                    t: 0.4,
                    mask_weight: weight,
                };
                let (loss, grads) = compute_loss_grads(&mut dit, &params, spec, &step).unwrap();
                eval(grads.values()).unwrap();
                (loss, grads)
            };
            let (plain, _) = run(None, &dense_spec);
            let (with_ones, _) = run(Some(&ones), &dense_spec);
            assert!(
                (with_ones - plain).abs() <= 1e-6 * plain.abs().max(1.0),
                "{mode}: all-ones map {with_ones} vs plain {plain}"
            );
            for (path, spec) in [("dense", &dense_spec), ("checkpointed", &ckpt_spec)] {
                let (loss, grads) = run(Some(&zeros), spec);
                assert_eq!(
                    loss, 0.0,
                    "{mode}/{path}: an all-background map must zero the loss"
                );
                assert!(!grads.is_empty(), "{mode}/{path}: no adapter grads");
                for (k, g) in &grads {
                    assert!(
                        g.as_slice::<f32>().iter().all(|x| *x == 0.0),
                        "{mode}/{path}: nonzero grad on {k}"
                    );
                }
            }
            let (dense, _) = run(Some(&half), &dense_spec);
            let (ckpt, _) = run(Some(&half), &ckpt_spec);
            assert!(
                dense > 0.0 && dense < plain,
                "{mode}: half {dense} vs plain {plain}"
            );
            assert!(
                (dense - ckpt).abs() <= 1e-5 * dense.abs().max(1.0),
                "{mode}: dense {dense} vs checkpointed {ckpt}"
            );
        }
    }

    /// A changed render alone cannot identify a save/reload defect. Compare the *trained*
    /// in-memory velocity with the saved file installed on a fresh base, for both adapter kinds,
    /// both modalities and the actual bf16 production compute path, before any sampling metric.
    #[test]
    fn trained_factors_preserve_velocity_through_save_and_reload() {
        let mut failures = Vec::new();
        for network in [NetworkType::Lora, NetworkType::Lokr] {
            for edit in [false, true] {
                let mut dit = tiny_dit();
                dit.cast_weights(Dtype::Bfloat16).unwrap();
                let Built {
                    adapter,
                    mut params,
                    mut cfg,
                    paths,
                } = build(&mut dit, network);
                cfg.alpha = 2.5; // Exercise a non-unit alpha/rank fold.
                let blocks = block_trainables(&mut dit, &paths, &params, &cfg).unwrap();
                let mut batch = fixed_edit_batch(&dit);
                if !edit {
                    let (x0, text, noise) = fixed_batch(&dit);
                    batch = EditBatch {
                        x0,
                        text,
                        noise,
                        references: vec![],
                        layout: t2i_layout(),
                    };
                }
                let spec = LossSpec {
                    adapter: &adapter,
                    alpha: cfg.alpha,
                    rank: cfg.rank as f32,
                    mae: false,
                    dtype: Dtype::Bfloat16,
                    lora_dtype: Some(Dtype::Bfloat16),
                    checkpoint: Some(&blocks),
                };
                let mut opt = TrainOptimizer::from_config("adamw", 1e-3, 0.0).unwrap();
                for _ in 0..3 {
                    let (_, grads) = compute_loss_grads(
                        &mut dit,
                        &params,
                        &spec,
                        &StepInputs {
                            x0: &batch.x0,
                            text: &batch.text,
                            noise: &batch.noise,
                            t: 0.5,
                            mask_weight: None,
                            references: &batch.references,
                            layout: &batch.layout,
                        },
                    )
                    .unwrap();
                    opt.step(&mut params, &grads).unwrap();
                    eval(params.values()).unwrap();
                }
                adapter
                    .install_as(
                        &mut dit,
                        &params,
                        cfg.alpha,
                        cfg.rank as f32,
                        Some(Dtype::Bfloat16),
                        LOKR_DTYPE,
                    )
                    .unwrap();
                let (x_t, _) = build_batch(&batch.x0, &batch.noise, 0.5).unwrap();
                let x_t = x_t.as_dtype(Dtype::Bfloat16).unwrap();
                let images = joint_images(&batch.references, &x_t);
                let before = dit
                    .forward_joint(&batch.text, &images, 0.5, &batch.layout)
                    .unwrap();
                eval([&before]).unwrap();
                let dir = tempfile::tempdir().unwrap();
                let file = dir.path().join("adapter.safetensors");
                let provenance: Vec<_> = ADAPTER_PROVENANCE
                    .into_iter()
                    .chain(edit.then_some(EDIT_ADAPTER_MARKER))
                    .collect();
                adapter
                    .save_with_meta(
                        &params,
                        cfg.alpha,
                        cfg.rank as f32,
                        cfg.decompose_factor,
                        "",
                        &provenance,
                        &file,
                    )
                    .unwrap();
                let mut fresh = tiny_dit();
                fresh.cast_weights(Dtype::Bfloat16).unwrap();
                let kind = if network == NetworkType::Lora {
                    mlx_gen::runtime::AdapterKind::Lora
                } else {
                    mlx_gen::runtime::AdapterKind::Lokr
                };
                let report = crate::apply_qwen_image_2_1_adapters(
                    &mut fresh,
                    &[mlx_gen::runtime::AdapterSpec::new(file.clone(), 1.0, kind)],
                )
                .unwrap();
                assert!(report.unmatched_paths.is_empty());
                assert_eq!(report.applied, paths.len());
                let after = fresh
                    .forward_joint(&batch.text, &images, 0.5, &batch.layout)
                    .unwrap();
                eval([&after]).unwrap();
                let error = subtract(&after, &before)
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max(None)
                    .unwrap()
                    .item::<f32>();
                let peak = before
                    .as_dtype(Dtype::Float32)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max(None)
                    .unwrap()
                    .item::<f32>();
                println!("FIDELITY {network:?} edit={edit} dense error={error} peak={peak}");
                if !error.is_finite() || !peak.is_finite() || error > 1e-6 * peak.max(1.0) {
                    failures.push(format!("{network:?} edit={edit} dense: {error} vs {peak}"));
                }
                // Compare reload with the same trained factors over an identical packed base.
                // The direct in-memory reference MUST use the packed representation too:
                // materialized bf16 kron and two bf16 contractions round at different places.
                // Test that representation error separately per linear against an f64 oracle;
                // serialization still has the strict dense velocity bound on every tier.
                for bits in [4, 8] {
                    let mut memory = tiny_dit();
                    memory.cast_weights(Dtype::Bfloat16).unwrap();
                    memory.quantize(bits).unwrap();
                    adapter
                        .install_as(
                            &mut memory,
                            &params,
                            cfg.alpha,
                            cfg.rank as f32,
                            Some(Dtype::Bfloat16),
                            LOKR_DTYPE,
                        )
                        .unwrap();
                    if network == NetworkType::Lokr {
                        let materialized = memory
                            .forward_joint(&batch.text, &images, 0.5, &batch.layout)
                            .unwrap();
                        eval([&materialized]).unwrap();
                        install_packed_lokr_reference(
                            &mut memory,
                            &params,
                            &paths,
                            cfg.alpha / cfg.rank as f32,
                            &mut failures,
                            &format!("edit={edit} q{bits}"),
                        );
                        let structured = memory
                            .forward_joint(&batch.text, &images, 0.5, &batch.layout)
                            .unwrap();
                        let representation_error = subtract(&materialized, &structured)
                            .unwrap()
                            .as_dtype(Dtype::Float32)
                            .unwrap()
                            .abs()
                            .unwrap()
                            .max(None)
                            .unwrap()
                            .item::<f32>();
                        // Diagnostic only: a nonlinear model does not inherit a single
                        // residual's componentwise rounding budget as a global peak ratio.
                        println!("REPRESENTATION Lokr edit={edit} q{bits} velocity_error={representation_error}");
                    }
                    let want = memory
                        .forward_joint(&batch.text, &images, 0.5, &batch.layout)
                        .unwrap();
                    eval([&want]).unwrap();
                    let mut reload = tiny_dit();
                    reload.cast_weights(Dtype::Bfloat16).unwrap();
                    reload.quantize(bits).unwrap();
                    crate::apply_qwen_image_2_1_adapters(
                        &mut reload,
                        &[mlx_gen::runtime::AdapterSpec::new(file.clone(), 1.0, kind)],
                    )
                    .unwrap();
                    let got = reload
                        .forward_joint(&batch.text, &images, 0.5, &batch.layout)
                        .unwrap();
                    eval([&got]).unwrap();
                    let error = subtract(&want, &got)
                        .unwrap()
                        .as_dtype(Dtype::Float32)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .max(None)
                        .unwrap()
                        .item::<f32>();
                    let peak = want
                        .as_dtype(Dtype::Float32)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .max(None)
                        .unwrap()
                        .item::<f32>();
                    println!("FIDELITY {network:?} edit={edit} q{bits} error={error} peak={peak}");
                    if !error.is_finite() || !peak.is_finite() || error > 1e-6 * peak.max(1.0) {
                        failures.push(format!(
                            "{network:?} edit={edit} q{bits}: {error} vs {peak}"
                        ));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    // This test reference neither reads the serialized file nor calls the loader/builder.
    // Construct the small factors directly from the trained parameter keys and metadata.
    fn install_packed_lokr_reference(
        host: &mut QwenImage21Transformer,
        params: &LoraParams,
        paths: &[String],
        scale: f32,
        failures: &mut Vec<String>,
        cell: &str,
    ) {
        use mlx_gen::adapters::{Adapter, LokrFactors};
        use mlx_rs::ops::matmul;

        let mut drop_detected = false;
        let mut scale_detected = false;
        for path in paths {
            let get = |suffix: &str| params.get(format!("{path}.{suffix}").as_str());
            let w1 = get("lokr_w1").unwrap();
            let w2 = match get("lokr_w2") {
                Some(w2) => w2.clone(),
                None => matmul(get("lokr_w2_a").unwrap(), get("lokr_w2_b").unwrap()).unwrap(),
            };
            let (a, c) = (w1.shape()[0], w1.shape()[1]);
            let (b, d) = (w2.shape()[0], w2.shape()[1]);
            let factors = LokrFactors {
                w1: w1.as_dtype(Dtype::Bfloat16).unwrap(),
                w2: multiply(&w2, Array::from_f32(scale))
                    .unwrap()
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap(),
                a,
                b,
                c,
                d,
                scale,
            };
            let x: Vec<_> = (0..3 * c * d)
                .map(|i| ((i * 7 % 31) as f32 - 15.0) / 16.0)
                .collect();
            let x = Array::from_slice(&x, &[3, c * d])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let f32_values = |value: &Array| {
                let value = value.as_dtype(Dtype::Float32).unwrap();
                eval([&value]).unwrap();
                value.as_slice::<f32>().to_vec()
            };
            let (want, products) = lokr_rounding::oracle(
                &f32_values(w1),
                &f32_values(&w2),
                &f32_values(&x),
                [a as usize, b as usize, c as usize, d as usize],
                scale,
            );
            let lin = host
                .adaptable_mut(&path.split('.').collect::<Vec<_>>())
                .unwrap();
            let dense = f32_values(&lin.adapters()[0].residual(&x).unwrap());
            let structured = f32_values(&factors.residual(&x).unwrap());
            for (representation, got, fraction) in [
                (
                    "dense",
                    &dense,
                    lokr_rounding::rounding_fraction(c as usize, d as usize, false),
                ),
                (
                    "structured",
                    &structured,
                    lokr_rounding::rounding_fraction(c as usize, d as usize, true),
                ),
            ] {
                if !lokr_rounding::within_bound(got, &want, &products, fraction) {
                    failures.push(format!(
                        "{cell} {path} {representation}: outside componentwise bf16 bound"
                    ));
                }
            }
            let fraction = lokr_rounding::rounding_fraction(c as usize, d as usize, true);
            // A zero residual or forgotten alpha/rank must fail the SAME oracle assertion.
            let mut wrong_scale = factors.clone();
            wrong_scale.w2 = multiply(&wrong_scale.w2, Array::from_f32(scale.recip())).unwrap();
            drop_detected |=
                !lokr_rounding::within_bound(&vec![0.0; want.len()], &want, &products, fraction);
            scale_detected |= !lokr_rounding::within_bound(
                &f32_values(&wrong_scale.residual(&x).unwrap()),
                &want,
                &products,
                fraction,
            );
            lin.set_adapters(vec![Adapter::LokrStructured { factors }]);
        }
        println!("ORACLE {cell} paths={} drop_mutant_detected={drop_detected} scale_mutant_detected={scale_detected}", paths.len());
        if !drop_detected || !scale_detected {
            failures.push(format!(
                "{cell}: LoKr residual oracle failed to discriminate drop/scale mutants"
            ));
        }
    }

    /// AC: the preflight prices an edit run's references — their cached latents (latent and train
    /// stages), the longer joint sequence (train stage), and the vision tower plus the vision
    /// slots of the image-conditioned prompt (caption stage).
    ///
    /// *Mutation that reds this:* dropping `shape.reference_tokens` from `seq`, or
    /// `reference_cache_tokens` from the latent cache, or the vision term from the caption stage.
    #[test]
    fn the_preflight_prices_edit_references() {
        let facts = FootprintFacts {
            vision_tower_bytes: 1 << 30,
            ..production_facts()
        };
        let t2i = shape(1024, true);
        let edit = TrainingShape {
            reference_tokens: 2 * 4096,
            largest_reference_tokens: 4096,
            reference_cache_tokens: 20 * 2 * 4096,
            ..t2i
        };
        let (a, b) = (
            training_footprint(&facts, &t2i),
            training_footprint(&facts, &edit),
        );
        assert!(b.caption_phase >= a.caption_phase + facts.vision_tower_bytes);
        assert_eq!(
            b.latent_phase - a.latent_phase,
            edit.reference_cache_tokens * facts.latent_channels * F32_WIDTH,
            "the latent stage carries exactly the cached reference latents on top"
        );
        // The train stage carries the reference cache AND a longer joint sequence.
        let cache_only = TrainingShape {
            reference_tokens: 0,
            largest_reference_tokens: 0,
            ..edit
        };
        let cache_train = training_footprint(&facts, &cache_only).train_phase;
        assert_eq!(
            cache_train - a.train_phase,
            edit.reference_cache_tokens * facts.latent_channels * F32_WIDTH
        );
        assert!(
            b.train_phase > cache_train,
            "the reference blocks lengthen the joint sequence the step attends over"
        );

        // The largest (aspect-preserving) edit target is priced, not the square edge: a target
        // with twice the edge² tokens raises the latent cache and the joint sequence.
        // *Mutation that reds this:* `training_footprint` ignoring `target_tokens`.
        let square = (1024 / 16) * (1024 / 16);
        let wide = TrainingShape {
            target_tokens: 2 * square,
            ..t2i
        };
        let explicit_square = TrainingShape {
            target_tokens: square,
            ..t2i
        };
        assert_eq!(
            training_footprint(&facts, &explicit_square),
            a,
            "target_tokens = edge² is the square default"
        );
        let c = training_footprint(&facts, &wide);
        assert!(c.latent_phase > a.latent_phase && c.train_phase > a.train_phase);
    }

    /// sc-24162 review: the block-causal attention is priced exactly — the target rows against
    /// every key plus `Σ (end − start)·end` over the prefix segments — not as `S²`. On the fixture
    /// edit layout (text 3 / 2×2 block / text 1 / 2×4 block / text 4 / target) the prefix segments
    /// are `[0,3) [3,7) [7,8) [8,16) [16,20)`, so the prefix calls materialise
    /// `3·3 + 4·7 + 1·8 + 8·16 + 4·20 = 253` scores per head, not `20² = 400`; the text-to-image
    /// prefix is `L²`. The footprint moves by exactly those elements, and the fallback
    /// (`prefix_scores = 0`) is the whole prefix squared.
    ///
    /// *Mutations that red this:* `prefix_score_elements` summing `(end − start)²`;
    /// `training_footprint` ignoring `prefix_scores` (the delta collapses to 0); pricing the
    /// attention as `S²` again (the fallback no longer equals `T·S + P²`).
    #[test]
    fn the_prefix_attention_is_priced_exactly() {
        let edit = JointLayout {
            segments: vec![
                Segment::Text { len: 3 },
                Segment::Image {
                    height: 2,
                    width: 2,
                },
                Segment::Text { len: 1 },
                Segment::Image {
                    height: 2,
                    width: 4,
                },
                Segment::Text { len: 4 },
                Segment::Image {
                    height: 4,
                    width: 4,
                },
            ],
        };
        assert_eq!(prefix_score_elements(&edit), 253);
        assert_eq!(largest_prefix_score_call(&edit), 128);
        assert_eq!(
            prefix_score_elements(&JointLayout::text_to_image(5, 4, 4)),
            25
        );

        let facts = production_facts();
        let base = TrainingShape {
            reference_tokens: 12,
            caption_tokens: 8,
            ..shape(1024, true)
        };
        let at = |prefix_scores| {
            training_footprint(
                &facts,
                &TrainingShape {
                    prefix_scores,
                    ..base
                },
            )
            .train_phase
        };
        let per_element = ATTENTION_BACKWARD_SCORE_MATRICES * facts.heads * base.compute_width;
        assert_eq!(at(400) - at(253), (400 - 253) * per_element);
        assert_eq!(at(0), at(400), "the fallback is the whole prefix squared");
        // The target rows' `T·S` is priced on top: an `S²` attention term would exceed it by
        // `(S − T)·S − P²` — `P·T` — elements.
        let target = (1024 / 16) * (1024 / 16);
        let seq = target + 20;
        let squared = TrainingShape {
            prefix_scores: seq * seq - target * seq,
            largest_prefix_call: 0,
            ..base
        };
        assert_eq!(
            training_footprint(&facts, &squared).train_phase - at(400),
            (seq * seq - target * seq - 400) * per_element
        );
    }

    // ── sc-24163: the candle twin's refusals, with the candle twin's messages ────────────────

    /// A control image on any item is refused (the candle twin's message), and `model_options`
    /// that select reference / control conditioning are refused through gen-core's one rule —
    /// the defaults-shaped *off* values the worker sends pass, the same keys turned on do not.
    ///
    /// *Mutation that reds this:* dropping the control-image check or the
    /// `refuse_reference_control_model_options` call from `validate_request`.
    #[test]
    fn validate_refuses_control_items_and_reference_control_model_options_like_candle() {
        let base = req_with(base_config());
        validate_request(&base).unwrap();

        let mut control = base.clone();
        control.items[0].control_image_path = Some(PathBuf::from("/nonexistent/c.png"));
        let err = validate_request(&control).unwrap_err().to_string();
        assert_eq!(
            err,
            "qwen_image_2_1 trainer: control images are not part of Qwen-Image 2.1 LoRA/LoKr \
             training"
        );

        let advanced = serde_json::json!({
            "mixedPrecision": "bf16",
            "cacheLatents": true,
            "networkType": "lora",
            "controlType": null,
            "control_type": "none",
            "references": [],
            "referenceImages": "",
            "reference_images": "  ",
            "referenceImagePaths": {},
            "controlImage": false,
            "control_image": "None",
        });
        let mut off = base.clone();
        off.config.model_options = advanced.as_object().unwrap().clone();
        off.items[0].model_options = advanced.as_object().unwrap().clone();
        validate_request(&off).expect("off-state reference/control keys select nothing");

        for (key, on) in [
            ("references", serde_json::json!(["/r.png"])),
            ("referenceImages", serde_json::json!("/r.png")),
            ("controlImage", serde_json::json!(true)),
            ("controlType", serde_json::json!("canny")),
            ("referenceImagePaths", serde_json::json!({ "0": "/r.png" })),
        ] {
            let mut on_req = off.clone();
            on_req.items[0].model_options.insert(key.into(), on.clone());
            let err = validate_request(&on_req).unwrap_err().to_string();
            assert!(err.contains(&format!("model_options `{key}`")), "{err}");
            // The candle twin calls the same gen-core refusal with the same label, so the text
            // is identical on both backends.
            let shared = gen_core::train::refuse_reference_control_model_options(
                "qwen_image_2_1 trainer",
                &on_req,
            )
            .unwrap_err()
            .to_string();
            assert_eq!(err, shared);
            let mut on_config = off.clone();
            on_config.config.model_options.insert(key.into(), on);
            assert!(validate_request(&on_config).is_err(), "{key} on the config");
        }
    }

    /// `load` refuses a control / extra-control / IP-adapter / identity overlay with the candle
    /// twin's typed `Unsupported`, before any weight is read.
    ///
    /// *Mutation that reds this:* dropping the `refuse_trainer_load_overlays` call from `load`.
    #[test]
    fn load_refuses_control_ip_adapter_and_identity_overlays_like_candle() {
        let dense = LoadSpec::new(WeightsSource::Dir(tiny_snapshot()));
        QwenImage21Trainer::load(&dense).expect("the bare snapshot loads");
        let other = || WeightsSource::Dir(PathBuf::from("/nonexistent/overlay"));
        let mut identity = dense.clone();
        identity.identity = Some(Default::default());
        for spec in [
            dense.clone().with_control(other()),
            dense.clone().with_extra_control(other()),
            dense.clone().with_ip_adapter(other()),
            identity,
        ] {
            match QwenImage21Trainer::load(&spec).err() {
                Some(Error::Unsupported(message)) => assert_eq!(
                    message,
                    "qwen_image_2_1 trainer: control / IP-adapter / identity overlays are not \
                     part of text-to-image LoRA/LoKr training"
                ),
                other => panic!("an overlay must be a typed Unsupported, got {other:?}"),
            }
        }
    }
}

/// Epic 2123 (sc-24827) MLX call-site test — the REAL `train_impl` loop on the 1 MB tiny snapshot
/// (tiny text encoder / VAE / DiT, one 64-px image, seconds): the loop invokes the shared adapter
/// update's noise on real optimizer updates only.
#[cfg(test)]
mod adapter_noise_loop_tests {
    use super::*;
    use mlx_gen::train::lora::apply_weight_noise;
    use mlx_gen::WeightsSource;

    fn tiny_snapshot() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot")
    }

    /// Train `steps` micro-steps at `accum` with the given noise knobs; return the saved adapter's
    /// factors keyed like the trainer's `LoraParams` (`{path}.lora_a` / `.lora_b`).
    fn train(weight_sigma: f32, grad_eta: f32, steps: u32, accum: u32) -> LoraParams {
        let data = tempfile::tempdir().unwrap();
        let img = image::RgbImage::from_fn(64, 64, |x, y| {
            image::Rgb([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8])
        });
        let img_path = data.path().join("a.png");
        img.save(&img_path).unwrap();
        let out = tempfile::tempdir().unwrap();
        let mut trainer =
            QwenImage21Trainer::load(&LoadSpec::new(WeightsSource::Dir(tiny_snapshot()))).unwrap();
        let req = TrainingRequest {
            items: vec![TrainingItem::captioned(img_path, "a swatch".into())],
            config: TrainingConfig {
                rank: 4,
                alpha: 4.0,
                steps,
                gradient_accumulation: accum,
                resolution: 64,
                seed: 11,
                learning_rate: 1e-2,
                weight_noise_sigma: weight_sigma,
                gradient_noise_eta: grad_eta,
                ..Default::default()
            },
            output_dir: out.path().to_path_buf(),
            file_name: "lora.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        };
        let result = trainer.train_impl(&req, &mut |_| {}).unwrap();
        assert_eq!(result.steps, steps);
        let saved = Array::load_safetensors(&result.adapter_path).unwrap();
        let params: LoraParams = saved
            .into_iter()
            .filter_map(|(k, v)| {
                let k = k
                    .strip_suffix(".lora_A.weight")
                    .map(|p| format!("{p}.lora_a"))
                    .or_else(|| {
                        k.strip_suffix(".lora_B.weight")
                            .map(|p| format!("{p}.lora_b"))
                    })?;
                Some((Rc::from(k.as_str()), v))
            })
            .collect();
        assert!(!params.is_empty(), "the adapter holds LoRA factors");
        params
    }

    fn host(a: &Array) -> Vec<f32> {
        let a = a.as_dtype(Dtype::Float32).unwrap();
        eval([&a]).unwrap();
        a.as_slice::<f32>().to_vec()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The compared runs are SEPARATE Metal executions of the training loop, which agree only to a
    /// few ulps (Metal reductions are not bit-deterministic across runs). Compare within
    /// `1e-6 + 1e-5·max|want|`; the guarded mutations (noise on a micro-step, a wrong update
    /// index, dropped or unseeded noise) move the adapter by orders of magnitude more.
    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let tol = 1e-6 + 1e-5 * scale;
        let diff = max_abs_diff(got, want);
        assert!(diff <= tol, "{what}: max |diff| {diff} > tolerance {tol}");
    }

    /// One real update (steps = accum = 2): the weight-noised run's adapter equals (within
    /// [`assert_close`]) the clean run's adapter + `apply_weight_noise(.., update 0)`. Noise on micro-step 1 (or on every
    /// micro-step) would change micro-step 2's gradient and add a second draw; a wrong update index
    /// draws different noise; a loop that skips the shared update leaves the adapter clean.
    ///
    /// *Mutation that reds this:* restoring the inline clip → step → eval at the Qwen update site
    /// (noise dropped), or firing the update every micro-step.
    #[test]
    fn train_loop_applies_weight_noise_once_per_real_update() {
        let sigma = 0.05f32;
        let mut clean = train(0.0, 0.0, 2, 2);
        let noisy = train(sigma, 0.0, 2, 2);
        apply_weight_noise(&mut clean, sigma, 11, 0).unwrap();
        assert_eq!(clean.len(), noisy.len());
        for (k, v) in &clean {
            assert_close(&host(&noisy[k]), &host(v), k);
        }
    }

    /// Gradient noise reaches the loop's optimizer step and reproduces under the job seed.
    #[test]
    fn train_loop_applies_gradient_noise_reproducibly() {
        let clean = train(0.0, 0.0, 2, 1);
        let a = train(0.0, 0.05, 2, 1);
        let b = train(0.0, 0.05, 2, 1);
        let mut effect = 0.0f32;
        for (k, v) in &a {
            assert_close(&host(&b[k]), &host(v), &format!("{k} reproducible"));
            effect = effect.max(max_abs_diff(&host(v), &host(&clean[k])));
        }
        assert!(
            effect > 1e-3,
            "gradient noise must reach the trained adapter (max |diff| {effect})"
        );
    }
}

/// sc-24830 (epic 2123 depth anchoring) — the Qwen-Image 2.1 step seam ([`run_train_step`] /
/// [`compute_step_loss_grads`]) on the tiny-snapshot DiT (2 blocks, 8 latent channels), text-to-image
/// and edit, dense and gradient-checkpointed, with a random-init **tiny `F16Decoder`** (the TAEQI2.1
/// structure: pixel-shuffle head, RGBA → RGB, 16×) and a random-init tiny Depth-Anything-V2. Seconds;
/// no weights downloaded.
#[cfg(test)]
mod depth_anchoring_tests {
    use super::*;
    use crate::transformer::Segment;
    use mlx_gen::train::perceptual::{AuxLoss, AuxLossSchedule};
    use mlx_gen::train::tae::{synthetic_tiny_decoder_weights, TinyDecoder};
    use mlx_gen_depth::anchor::{synthetic_weights, tiny_config, DepthAnchorLoss};
    use mlx_gen_depth::DepthAnythingV2;

    fn tiny_snapshot() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot")
    }

    fn tiny_dit() -> QwenImage21Transformer {
        loader::load_transformer(&tiny_snapshot()).unwrap()
    }

    fn aux_sched() -> AuxLossSchedule {
        AuxLossSchedule {
            weight: 0.1,
            t_min: 0.0,
            t_max: 1.0,
            every_n: 2,
        }
    }

    /// A tiny TAEQI2.1-structured decoder (`F16Decoder`: widening stages, RGBA head, 2×2 pixel
    /// shuffle) for the tiny DiT's 8 latent channels, + a tiny DA2.
    fn path_with(schedule: AuxLossSchedule) -> PerceptualPath {
        let spec = TinyDecoderSpec {
            latent_channels: 8,
            stage_channels: [16, 8, 8, 8],
            ..TinyDecoderSpec::taeqi2_1()
        };
        let decoder =
            TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&spec, 11).unwrap(), spec)
                .unwrap();
        let da2 = tiny_config();
        PerceptualPath::new(
            Some(Box::new(decoder)),
            vec![AuxLoss {
                schedule,
                loss: Box::new(DepthAnchorLoss::new(
                    DepthAnythingV2::from_weights(&synthetic_weights(&da2, 12).unwrap(), da2)
                        .unwrap(),
                )),
            }],
        )
        .unwrap()
    }

    fn cfg() -> TrainingConfig {
        let mut cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            seed: 7,
            ..Default::default()
        };
        cfg.depth_anchoring.schedule = aux_sched();
        cfg
    }

    fn randn(shape: &[i32], seed: u64) -> Array {
        let a =
            random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap())).unwrap();
        eval([&a]).unwrap();
        a
    }

    /// A text-to-image cache entry: a packed `[1, h·w, 8]` target, 5 caption rows.
    fn t2i_entry(h: usize, w: usize, k: u64) -> CachedItem {
        CachedItem {
            x0: randn(&[1, (h * w) as i32, 8], k),
            mask_weight: None,
            text: randn(&[1, 5, 32], 100 + k),
            layout: JointLayout::text_to_image(5, h, w),
            references: Vec::new(),
        }
    }

    /// An edit cache entry: two references (2×2, 2×4) interleaved with text, the 4×4 target last.
    fn edit_entry(k: u64) -> CachedItem {
        CachedItem {
            x0: randn(&[1, 16, 8], k),
            mask_weight: None,
            text: randn(&[1, 8, 32], 100 + k),
            layout: JointLayout {
                segments: vec![
                    Segment::Text { len: 3 },
                    Segment::Image {
                        height: 2,
                        width: 2,
                    },
                    Segment::Text { len: 1 },
                    Segment::Image {
                        height: 2,
                        width: 4,
                    },
                    Segment::Text { len: 4 },
                    Segment::Image {
                        height: 4,
                        width: 4,
                    },
                ],
            },
            references: vec![randn(&[1, 4, 8], 200 + k), randn(&[1, 8, 8], 300 + k)],
        }
    }

    struct Fixture {
        dit: QwenImage21Transformer,
        adapter: TrainAdapter,
        params: LoraParams,
        blocks: Vec<BlockTrainables>,
    }

    fn fixture(cfg: &TrainingConfig) -> Fixture {
        let mut dit = tiny_dit();
        let paths = resolve_target_paths(&dit, cfg);
        let (targets, params) =
            build_lora_targets(&mut dit, &paths, cfg.rank as i32, cfg.seed).unwrap();
        let blocks = block_trainables(&mut dit, &paths, &params, cfg).unwrap();
        Fixture {
            dit,
            adapter: TrainAdapter::Lora { targets },
            params,
            blocks,
        }
    }

    fn single_bucket(n: usize) -> BucketSchedule {
        BucketSchedule::new(
            n,
            &[gen_core::ResolutionBucket {
                resolution: 64,
                repeats: 1,
            }],
            7,
        )
    }

    fn step(
        f: &mut Fixture,
        cfg: &TrainingConfig,
        cache: &[CachedItem],
        schedule: &BucketSchedule,
        path: Option<(&mut PerceptualPath, &mut AuxAlternation)>,
        n: u32,
        ckpt: bool,
    ) -> (StepLosses, LoraParams) {
        let spec = LossSpec {
            adapter: &f.adapter,
            alpha: cfg.alpha,
            rank: cfg.rank as f32,
            mae: false,
            dtype: Dtype::Float32,
            lora_dtype: None,
            checkpoint: ckpt.then_some(f.blocks.as_slice()),
        };
        let (l, g) =
            run_train_step(&mut f.dit, &f.params, &spec, cfg, cache, schedule, path, n).unwrap();
        eval(g.values()).unwrap();
        (l, g)
    }

    fn lora_b_sum(g: &LoraParams) -> f32 {
        let v: Vec<f32> = g
            .iter()
            .filter(|(k, _)| k.ends_with(".lora_b"))
            .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
            .collect();
        assert!(!v.is_empty());
        v.iter().sum()
    }

    /// `unpack_to_decoder_layout` inverts `pack_latents` at the layout's (non-square) target grid.
    /// Mutation: reshape to `[B, C, w, h]` ⇒ red.
    #[test]
    fn unpack_inverts_the_target_packing() {
        let grid = randn(&[1, 8, 4, 2], 9);
        let packed = pack_latents(&grid).unwrap();
        let back = unpack_to_decoder_layout(&packed, &JointLayout::text_to_image(5, 4, 2)).unwrap();
        assert_eq!(back.shape(), grid.shape());
        let err = back
            .subtract(&grid)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert_eq!(err, 0.0);
        assert!(unpack_to_decoder_layout(&packed, &JointLayout::text_to_image(5, 4, 4)).is_err());
    }

    /// AC (a)/(b) on every loss path the trainer runs while depth is on — text-to-image and edit,
    /// dense and gradient-checkpointed: a depth step trains the LoRA through the depth term alone
    /// (no diffusion term, total == aux, non-zero finite LoRA-B grad); a diffusion step has no depth
    /// term. Mutation: force `diffusion_on = true` ⇒ red.
    #[test]
    fn depth_step_trains_the_lora_on_every_path() {
        for edit in [false, true] {
            for ckpt in [false, true] {
                let cfg = cfg();
                let mut f = fixture(&cfg);
                let cache = vec![if edit {
                    edit_entry(1)
                } else {
                    t2i_entry(4, 4, 1)
                }];
                let schedule = single_bucket(1);
                let mut p = path_with(aux_sched());
                prepare_perceptual_references(&mut p, &cache).unwrap();
                let mut alt = AuxAlternation::new(1, 1);
                let tag = format!("edit={edit} ckpt={ckpt}");
                let (diff, _) = step(
                    &mut f,
                    &cfg,
                    &cache,
                    &schedule,
                    Some((&mut p, &mut alt)),
                    1,
                    ckpt,
                );
                assert_eq!(diff.aux, None, "{tag}");
                assert_eq!(Some(diff.total), diff.diffusion, "{tag}");
                let (depth, g) = step(
                    &mut f,
                    &cfg,
                    &cache,
                    &schedule,
                    Some((&mut p, &mut alt)),
                    2,
                    ckpt,
                );
                assert_eq!(depth.diffusion, None, "{tag}");
                let aux = depth.aux.expect("depth term");
                assert!(aux > 0.0 && aux.is_finite(), "{tag}: {aux}");
                assert_eq!(depth.total, aux, "{tag}");
                let gb = lora_b_sum(&g);
                assert!(gb > 0.0 && gb.is_finite(), "{tag}: LoRA-B |Σ| = {gb}");
            }
        }
    }

    /// The aux step trains at `t` remapped into `[0.6, 0.8]`, and its depth term is the depth loss
    /// of the explicitly recovered target-block `x_t − t·v`, unpacked (edit layout: the references
    /// condition the forward, only the target is decoded). Mutations: keep the sampled `t` ⇒ red;
    /// recover with the opposite flow sign ⇒ red.
    #[test]
    fn aux_step_trains_at_the_remapped_noise_level() {
        let mut cfg = cfg();
        let window = AuxLossSchedule {
            t_min: 0.6,
            t_max: 0.8,
            ..aux_sched()
        };
        cfg.depth_anchoring.schedule = window;
        let mut f = fixture(&cfg);
        let cache = vec![edit_entry(1)];
        let schedule = single_bucket(1);
        let mut p = path_with(window);
        prepare_perceptual_references(&mut p, &cache).unwrap();
        let mut alt = AuxAlternation::new(1, 1);
        step(
            &mut f,
            &cfg,
            &cache,
            &schedule,
            Some((&mut p, &mut alt)),
            1,
            false,
        );
        let (depth, _) = step(
            &mut f,
            &cfg,
            &cache,
            &schedule,
            Some((&mut p, &mut alt)),
            2,
            false,
        );
        let raw = sample_sigma(
            &cfg.timestep_type,
            &cfg.timestep_bias,
            cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(2),
        )
        .unwrap();
        let plan = p.plan(2, 0, raw).unwrap();
        let t = plan.noise_level;
        assert!((0.6..=0.8).contains(&t) && t != raw, "{t} vs {raw}");
        let item = &cache[0];
        let noise = random::normal::<f32>(
            item.x0.shape(),
            None,
            None,
            Some(&random::key(cfg.seed.wrapping_add(2).wrapping_mul(2) + 1).unwrap()),
        )
        .unwrap();
        f.adapter
            .install_as(&mut f.dit, &f.params, 4.0, 4.0, None, LOKR_DTYPE)
            .unwrap();
        let (x_t, _) = build_batch(&item.x0, &noise, t).unwrap();
        let v = f
            .dit
            .forward_joint(
                &item.text,
                &joint_images(&item.references, &x_t),
                t,
                &item.layout,
            )
            .unwrap();
        let x0_hat = subtract(&x_t, multiply(&v, Array::from_f32(t)).unwrap()).unwrap();
        let want = p
            .aux_loss(
                &plan,
                0,
                &unpack_to_decoder_layout(&x0_hat, &item.layout).unwrap(),
            )
            .unwrap()
            .unwrap()
            .weighted
            .item::<f32>();
        let got = depth.aux.expect("depth step");
        assert!(
            (want - got).abs() <= 1e-5 * want.abs().max(1.0),
            "{want} vs {got}"
        );
    }

    /// Per-image alternation (round-robin N = 2) and per-entry references across two buckets with
    /// different target grids (every step recomputed against its SCHEDULED entry). Mutations: key
    /// the plan on the global step ⇒ red; pass the item as the `AuxStep` entry ⇒ red.
    #[test]
    fn alternation_is_per_image_with_per_entry_references() {
        let cfg = cfg();
        let mut f = fixture(&cfg);
        let cache = vec![t2i_entry(4, 4, 1), t2i_entry(4, 4, 2)];
        let schedule = single_bucket(2);
        let mut p = path_with(aux_sched());
        prepare_perceptual_references(&mut p, &cache).unwrap();
        let mut alt = AuxAlternation::new(2, 1);
        let kinds: Vec<(usize, bool)> = (1..=4)
            .map(|n| {
                let (l, _) = step(
                    &mut f,
                    &cfg,
                    &cache,
                    &schedule,
                    Some((&mut p, &mut alt)),
                    n,
                    false,
                );
                (schedule.sample((n - 1) as usize).0, l.aux.is_some())
            })
            .collect();
        for image in 0..2 {
            let mine: Vec<bool> = kinds.iter().filter(|k| k.0 == image).map(|k| k.1).collect();
            assert!(mine.contains(&true) && mine.contains(&false), "{kinds:?}");
        }

        // Two buckets, item-major: 4×4 and 2×4 target grids.
        let cache = vec![
            t2i_entry(4, 4, 10),
            t2i_entry(2, 4, 11),
            t2i_entry(4, 4, 12),
            t2i_entry(2, 4, 13),
        ];
        let schedule = BucketSchedule::new(
            2,
            &[
                gen_core::ResolutionBucket {
                    resolution: 64,
                    repeats: 1,
                },
                gen_core::ResolutionBucket {
                    resolution: 32,
                    repeats: 1,
                },
            ],
            7,
        );
        let mut p = path_with(aux_sched());
        prepare_perceptual_references(&mut p, &cache).unwrap();
        let mut alt = AuxAlternation::new(2, 1);
        let mut replay = AuxAlternation::new(2, 1);
        let mut depth_off_item = false;
        for n in 1..=2 * schedule.epoch_len() as u32 {
            let (l, _) = step(
                &mut f,
                &cfg,
                &cache,
                &schedule,
                Some((&mut p, &mut alt)),
                n,
                false,
            );
            let k = (n - 1) as usize;
            let (item, entry) = (schedule.sample(k).0, schedule.cache_index(k));
            let raw = sample_sigma(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(n as u64),
            )
            .unwrap();
            let plan = p.plan(replay.key(n, item), entry, raw).unwrap();
            let c = &cache[entry];
            let noise = random::normal::<f32>(
                c.x0.shape(),
                None,
                None,
                Some(&random::key(cfg.seed.wrapping_add(n as u64).wrapping_mul(2) + 1).unwrap()),
            )
            .unwrap();
            let spec = LossSpec {
                adapter: &f.adapter,
                alpha: cfg.alpha,
                rank: cfg.rank as f32,
                mae: false,
                dtype: Dtype::Float32,
                lora_dtype: None,
                checkpoint: None,
            };
            let (expected, _) = compute_step_loss_grads(
                &mut f.dit,
                &f.params,
                &spec,
                &StepInputs {
                    x0: &c.x0,
                    text: &c.text,
                    layout: &c.layout,
                    references: &c.references,
                    noise: &noise,
                    t: plan.noise_level,
                    mask_weight: None,
                },
                Some(AuxStep {
                    path: &p,
                    plan: &plan,
                    entry,
                }),
            )
            .unwrap();
            assert_eq!(l, expected, "step {n} (item {item}, entry {entry})");
            depth_off_item |= l.aux.is_some() && entry != item;
        }
        assert!(depth_off_item, "no depth step on an entry != its item");
        assert_eq!(p.reference_computations(), cache.len());
    }

    /// E1: depth off ⇒ nothing loaded, no aux bytes, and the step is bit-identical to the
    /// pre-epic-2123 closure (reproduced verbatim); a diffusion-only step of an enabled path is
    /// bit-identical too. Mutation: flip MAE/MSE in the diffusion term ⇒ red.
    #[test]
    fn everything_off_is_bit_identical_to_the_legacy_step() {
        assert!(load_perceptual_path(&TrainingConfig::default())
            .unwrap()
            .is_none());
        assert_eq!(
            perceptual_footprint_bytes(&TrainingConfig::default(), 1024, 0, 10),
            0
        );
        let off_cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            seed: 7,
            ..Default::default()
        };
        let mut f = fixture(&off_cfg);
        let cache = vec![edit_entry(1), edit_entry(2)];
        let schedule = single_bucket(2);
        let (off, g_off) = step(&mut f, &off_cfg, &cache, &schedule, None, 1, false);
        assert_eq!(off.aux, None);
        let item = &cache[0];
        let t = sample_sigma(
            &off_cfg.timestep_type,
            &off_cfg.timestep_bias,
            off_cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(1),
        )
        .unwrap();
        let noise = random::normal::<f32>(
            item.x0.shape(),
            None,
            None,
            Some(&random::key(off_cfg.seed.wrapping_add(1).wrapping_mul(2) + 1).unwrap()),
        )
        .unwrap();
        let (x_t, target) = build_batch(&item.x0, &noise, t).unwrap();
        let (text, layout, references) = (item.text.clone(), &item.layout, &item.references);
        let dit = &mut f.dit;
        let adapter = &f.adapter;
        let legacy = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
            adapter.install_as(dit, &p, 4.0, 4.0, None, LOKR_DTYPE)?;
            let images = joint_images(references, &x_t);
            let v = dit
                .forward_joint(&text, &images, t, layout)
                .map_err(|e| Exception::custom(e.to_string()))?;
            Ok(vec![reduce_loss(&subtract(&v, &target)?, None, false)?])
        };
        let (val, g_legacy) = keyed_value_and_grad(legacy)(f.params.clone(), 0).unwrap();
        eval(g_legacy.values()).unwrap();
        assert_eq!(off.total, val[0].item::<f32>());
        let bits =
            |a: &Array| -> Vec<u32> { a.as_slice::<f32>().iter().map(|x| x.to_bits()).collect() };
        for (k, v) in &g_legacy {
            assert_eq!(bits(v), bits(&g_off[k]), "{k}");
        }
        let on_cfg = cfg();
        let mut f2 = fixture(&on_cfg);
        let mut p = path_with(aux_sched());
        prepare_perceptual_references(&mut p, &cache).unwrap();
        let mut alt = AuxAlternation::new(2, 1);
        let (on1, g_on1) = step(
            &mut f2,
            &on_cfg,
            &cache,
            &schedule,
            Some((&mut p, &mut alt)),
            1,
            false,
        );
        let mut f3 = fixture(&on_cfg);
        let (none1, g_none1) = step(&mut f3, &on_cfg, &cache, &schedule, None, 1, false);
        assert_eq!(on1, none1);
        for (k, v) in &g_none1 {
            assert_eq!(bits(v), bits(&g_on1[k]), "{k}");
        }
    }

    /// E7: the aux bytes (TAEQI2.1 + DA2, more for Large, sized at the largest target) land in the
    /// estimator's train phase on the dense AND checkpointed shapes, and the preflight refuses at a
    /// synthetic budget between the base peak and base + aux. Mutations: drop
    /// `+ shape.aux_model_bytes` from `train_phase` ⇒ red; price the edit target at `edge` only
    /// (ignore `target_tokens`) ⇒ the wide-target figure stops growing ⇒ red.
    #[test]
    fn memory_estimate_includes_the_aux_models() {
        let mut on = TrainingConfig::default();
        on.depth_anchoring.schedule = aux_sched();
        let small = perceptual_footprint_bytes(&on, 1024, 0, 10);
        on.depth_anchoring.model_size = gen_core::train::DepthModelSize::Large;
        let large = perceptual_footprint_bytes(&on, 1024, 0, 10);
        assert!(
            small > 0 && large > small + 1_000_000_000,
            "{small} / {large}"
        );
        // An edit target wider than the square edge (more latent cells) prices a larger decode.
        assert!(perceptual_footprint_bytes(&on, 1024, 4 * 64 * 64, 10) > large);

        let facts = FootprintFacts::from_snapshot(&tiny_snapshot()).unwrap();
        for checkpointed in [false, true] {
            let base = TrainingShape {
                edge: 64, // the train phase (with aux) is the peak
                target_tokens: 0,
                target_cache_tokens: 0,
                caption_tokens: 64,
                reference_tokens: 0,
                largest_reference_tokens: 0,
                reference_cache_tokens: 0,
                prefix_scores: 0,
                largest_prefix_call: 0,
                items: 4,
                compute_width: 2,
                trainable_params: 1_000_000,
                optimizer_state_per_param: 2,
                lokr_delta_elements: 0,
                checkpointed,
                sampling: false,
                aux_model_bytes: 0,
            };
            let with_aux = TrainingShape {
                aux_model_bytes: large,
                ..base
            };
            let (fp0, fp1) = (
                training_footprint(&facts, &base),
                training_footprint(&facts, &with_aux),
            );
            assert_eq!(
                fp1.train_phase - fp0.train_phase,
                large,
                "checkpointed={checkpointed}"
            );
            let between = fp0.peak() + large / 2;
            assert!(check_training_footprint(&facts, &base, between).is_ok());
            assert!(
                check_training_footprint(&facts, &with_aux, between).is_err(),
                "checkpointed={checkpointed}"
            );
        }
    }

    /// E3: Qwen-Image 2.1 declares depth anchoring.
    #[test]
    fn descriptor_declares_depth_anchoring() {
        assert!(trainer_descriptor().techniques.depth_anchoring);
    }

    /// A missing decoder is a named error (TAEQI2.1).
    #[test]
    fn missing_aux_weights_are_named() {
        let tmp = tempfile::tempdir().unwrap();
        let mut c = cfg();
        c.perceptual_decoder_dir = Some(tmp.path().join("no-taeqi2_1"));
        c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
        let err = load_perceptual_path(&c)
            .err()
            .expect("must fail")
            .to_string();
        assert!(err.contains("TAEQI2.1"), "{err}");
    }
}
