//! LoRA/LoKr **training** on the Qwen-Image 2.1 DiT on candle (sc-24160) — the candle/CUDA
//! text-to-image trainer for `qwen_image_2_1`, the twin of `mlx-gen-qwen-image-2-1`'s trainer
//! (sc-24159).
//!
//! [`QwenImage21Trainer`] realizes the backend-neutral [`Trainer`] contract over the shared
//! [`candle_gen::train`] harness: the trainable factors are `Var`-backed residuals attached to the
//! DiT's own [`AdaptLinear`] projections through the [`AdaptLoraHost`] seam — the **same dotted
//! module paths** [`QwenImage21Transformer::visit_adaptable_mut`] exposes to the inference adapter
//! host (sc-24157) — so a trained adapter loads back through [`crate::adapters::install`] strictly,
//! with no key conversion, and carries exactly the keys the MLX trainer writes.
//!
//! ## What is Qwen-Image-2.1-specific (identical to the MLX twin)
//! - **Flow-match velocity target = `noise − x0`**, `x_t = (1 − t)·x0 + t·noise`, and the DiT gets
//!   the raw noise fraction `t` ([`QwenImage21Transformer::forward_train`] scales it by 1000 inside,
//!   exactly as the render path's `TimestepConvention::Sigma` does). The pipeline's Euler step is
//!   `x ← x + v·(σ_{i+1} − σ_i)`, so there is no sign flip.
//! - **Latents** are the RGBA autoencoder's posterior **mode** of the centre-cropped, resized image
//!   (an RGB image is the opaque RGBA case: a constant `+1` alpha plane), normalised
//!   `(z − mean)/std` and packed unpatched to `[1, (edge/16)², z_dim]`.
//! - **Caption features** are the Qwen3-VL language tower's T2I conditioning
//!   ([`QwenImage21TextEncoder::encode_prompt`]) — the render path's conditioning.
//! - **Targets** default to every per-block Linear ([`BLOCK_ADAPTER_TARGETS`] on each block); an
//!   explicit `lora_target_modules` suffix-matches anywhere, including the
//!   [`GLOBAL_ADAPTER_TARGETS`](crate::transformer::GLOBAL_ADAPTER_TARGETS).
//! - **LoKr** trains PEFT's factor surface (`w1` + full `w2` or low-rank `w2_a·w2_b` by the
//!   `use_w2` rule) — [`build_adapt_lokr_targets_peft`] — so the keys match the MLX trainer's.
//!
//! ## Memory lifecycle (staged; never more than one heavy component at once)
//! 1. **Preflight** ([`training_footprint`] / [`check_training_footprint`]) — before any weight is
//!    read, the run's peak is derived from *this snapshot's own* configs and safetensors headers and
//!    refused with an actionable message if it exceeds the device budget.
//! 2. **Captions** — the Qwen3-VL language tower encodes every caption and preview prompt ONCE,
//!    then is dropped.
//! 3. **Latents** — the VAE encodes every image ONCE; its encoder half is then dropped
//!    ([`QwenImage21Vae::drop_encoder`]; the decoder stays only when previews are requested).
//! 4. **Train** — the dense DiT loads last, at the training compute dtype. With
//!    `gradient_checkpointing` every block is one segment of the shared segmented-VJP
//!    ([`checkpointed_backward_with_input_grad`]): the block activations are recomputed in the
//!    backward instead of retained, for LoRA and LoKr alike, and the retained pre-block forward
//!    (global projections) is stitched in through the recovered boundary cotangent.
//!
//! The DiT forward used for training is [`QwenImage21Transformer::forward_train`]: the render
//! forward with the attention core's fused, backward-less kernels swapped for composable ones
//! ([`crate::transformer::Ops`]); the render path itself is untouched.
//!
//! ## Deliberate differences from the MLX twin
//! - **LoKr is priced as the Kronecker vec-trick, not a dense delta.** candle's trainable LoKr never
//!   materialises the `[out, in]` delta (the training residual and the inference install both apply
//!   `w1·X·w2ᵀ`), so the preflight prices the vec-trick's `[S·in_a, out_b]` intermediate per
//!   adapted projection instead of MLX's dense bf16 delta.
//! - **The dense backward retains every block's attention probabilities.** candle is eager: the
//!   composable attention's score tensors stay alive in the graph until the backward, so a dense
//!   step prices all `num_layers` blocks' score matrices; a checkpointed step prices one.
//! - **CPU trains in f32.** candle's CPU backend has no BF16 matmul, so on a CPU device the compute
//!   dtype is f32 whatever `train_dtype` says (the loader's `compute_dtype_on` convention); a GPU
//!   honours `train_dtype` (bf16 by default).
//!
//! The trainer is **dense-bf16 only**: a pre-quantized (Q4/Q8) tier or a quantize request is a typed
//! refusal. Every saved adapter (final and intermediate) carries [`ADAPTER_PROVENANCE`] in its
//! safetensors `__metadata__`: `family = qwen-image-2-1`, `baseModel = qwen_image_2_1`, and the
//! Qwen Research License the base weights are under.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_gen::gen_core::tiling::TilingConfig;
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{
    self, CancelFlag, Image, LoadSpec, Modality, NetworkType, Precision, Progress,
};
use candle_gen::quant::AdaptLinear;
use candle_gen::train::checkpoint::{
    checkpoint_filename, file_stem, find_latest_resume, load_resume, save_resume,
};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match::{
    self, apply_update, create_output_dir, effective_weight_decay, request_fingerprint,
    sample_seed, save_adapter, validate_flow_match_request, velocity_loss,
};
use candle_gen::train::gradient_checkpoint::{checkpointed_backward_with_input_grad, Segment};
use candle_gen::train::lora::{
    build_adapt_lokr_targets_peft, build_adapt_lora_targets, factorization, AdaptLoraHost, LoraSet,
};
use candle_gen::train::optim::{accumulate_grads, TrainOptimizer};
use candle_gen::train::schedule::schedule_updates;
use candle_gen::{CandleError as Error, Result};

use crate::config::{SchedulerConfig, TextEncoderConfig, TransformerConfig, VaeConfig};
use crate::loader;
use crate::pipeline::{
    create_noise, decode_rgb, denoise, latent_grid, pack_latents, DenoiseInputs, DECODE_OVERLAP,
    DECODE_TILE_EDGE,
};
use crate::quant::{installed_tier, Tier};
use crate::text_encoder::{prompt_template, system_prompt_drop_count, QwenImage21TextEncoder};
use crate::transformer::{QwenImage21Transformer, BLOCK_ADAPTER_TARGETS};
use crate::vae::QwenImage21Vae;
use crate::{FAMILY, MODEL_ID, UPSTREAM_LICENSE, UPSTREAM_LICENSE_NOTICE};

/// Registry id of the Qwen-Image 2.1 trainer — the generator id of the same base model (the
/// [`TrainerDescriptor::id`] convention), so a trained adapter's `baseModel` names the route it
/// applies to. Identical to the MLX twin's.
pub const TRAINER_ID: &str = MODEL_ID;

/// Error-message prefix.
const LABEL: &str = "qwen_image_2_1 trainer";

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence.
const SAMPLE_PROMPT_CAP: usize = 4;

/// Provenance + licence stamped into every saved adapter's `__metadata__` (final and
/// intermediate, LoRA and LoKr), alongside the shared `networkType`/`rank`/`alpha` reload contract —
/// the same six pairs the MLX trainer stamps.
///
/// * `family` / `baseModel` are the SceneWorks-native pair `detect_metadata_family` reads first;
///   `family` is the canonical `qwen-image-2-1` token, so a re-imported adapter lands in the 2.1
///   LoRA pool rather than the 2512 `qwen-image` one its tensor names would otherwise resemble.
/// * `ss_base_model_version` is the kohya spelling of the same base id.
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

fn provenance_meta() -> HashMap<String, String> {
    ADAPTER_PROVENANCE
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

/// The refusal every quantized-base path raises (typed `Unsupported`). QLoRA is a non-goal: the
/// trainer runs over the dense bf16 base only. Same wording as the MLX twin.
fn quantized_base_refusal(what: &str) -> Error {
    Error::Unsupported(format!(
        "{TRAINER_ID} trainer: {what}, but LoRA/LoKr training runs over the dense BF16 base only \
         (training over a packed Q4/Q8 tier — QLoRA — is not supported); install the BF16 tier \
         to train"
    ))
}

// ── preflight: the training footprint, derived from the snapshot's own facts ─────────────────────

/// Width of one f32 element — the caption/latent caches, the trainable factors, the f32 islands of
/// the forward.
const F32_WIDTH: u64 = 4;

/// f32 buffers every trainable factor element carries whatever the optimizer: the factor `Var`,
/// its gradient and the gradient-accumulation buffer. The optimizer's own state comes on top
/// ([`optimizer_state_per_param`]).
const TRAINABLE_BASE_BUFFERS: u64 = 3;

/// f32 optimizer-state elements [`TrainOptimizer`] keeps per trainable element: AdamW/Adam two
/// (`m`, `v`), Rose none, Prodigy four (`exp_avg`, `exp_avg_sq`, `s`, `p0`) — read off
/// `candle_gen::train::optim`'s own state structs. Names normalise the way the optimizer picker
/// does ([`candle_gen::train::optim::normalize`]).
pub fn optimizer_state_per_param(optimizer: &str) -> u64 {
    match candle_gen::train::optim::normalize(optimizer).as_str() {
        "rose" => 0,
        "prodigy" => 4,
        _ => 2,
    }
}

/// `[S, inner]` tensors at the **compute** width a dense backward retains per block — a
/// **structural count** of [`QwenImage21Transformer::forward_train`]'s block (every block Linear
/// adapted, the default): the LayerNorm cast-backs (2) and the two `x·(1+scale)` modulations (2·2);
/// q/k/v/out projections with their adapter residual (base output, residual, scaled residual, sum:
/// 4 each = 16); the per-head contiguous q/k/v (3); the q/k RMSNorm and RoPE cast-backs (4); the
/// per-segment SDPA outputs and their concatenation (2) plus the head merge (1); and the two gated
/// residual adds (`tanh(gate)`, product, sum: 3·2). Not a measurement — like every number here.
pub const BLOCK_SAVED_HIDDEN: u64 = 42;
/// `[S, inner]` **f32** tensors a block retains: the two f32 LayerNorms (upcast, centred, squared,
/// normalised: 4·2), the q/k composable RMSNorms (4·2) and the q/k composable RoPE (upcast, the
/// four half-width products ≈ 2 full, the concatenation: 5·2).
pub const BLOCK_SAVED_HIDDEN_F32: u64 = 26;
/// `[S, inner·mlp_ratio]` compute-width tensors the SwiGLU retains: the adapted `gate_layer` and
/// `proj` (4 each), `silu(gate)` and the product.
pub const BLOCK_SAVED_PER_MLP_RATIO: u64 = 10;
/// `[heads, Sq, Sk]` attention-score tensors the composable attention retains at the **compute**
/// width: the raw scores, the scaled scores and the probabilities cast back.
pub const SCORE_COMPUTE_TENSORS: u64 = 3;
/// `[heads, Sq, Sk]` score tensors it retains at **f32** — the softmax island: the upcast, the
/// max-shifted, the exponential and the normalised probabilities.
pub const SCORE_F32_TENSORS: u64 = 4;
/// f32 `[heads, Sq, Sk]` gradients the attention backward holds at once (the probabilities' and the
/// scores' cotangents).
const BACKWARD_SCORE_GRADS: u64 = 2;
/// f32 `[S, inner]` gradients in flight at once during a block's backward.
const BACKWARD_HIDDEN_GRADS: u64 = 4;
/// `[S, inner]` compute-width tensors the retained pre-block forward holds (the text/image
/// projections and their joint concatenation, adapter residuals included).
const PRELUDE_SAVED_HIDDEN: u64 = 8;
/// `[L, text_hidden]` f32 tensors live in one Qwen3 decoder layer while a caption encodes (the
/// residual, the normed input, q/k/v, the attention output and the SwiGLU halves, rounded up), plus
/// [`TEXT_ENCODER_SCORE_MATRICES`] `[heads, L, L]` f32 scores. Captions are short; this stage never
/// decides a refusal on a real run.
const TEXT_ENCODER_LIVE_HIDDEN: u64 = 16;
const TEXT_ENCODER_SCORE_MATRICES: u64 = 3;
/// Full-resolution `[base_dim, H, W]` maps live at once in the VAE encoder's first residual block
/// (input, the f32 channel-norm's upcast / square / normalised, the activation, the conv output) —
/// f32 width, conservative.
const VAE_LIVE_MAPS: u64 = 6;
/// `[S, inner]` compute-width tensors one **inference** (preview) block holds at once — no graph is
/// retained, so this is a single block's working set.
const PREVIEW_LIVE_HIDDEN: u64 = 16;

/// The facts the training footprint is derived from — read off **this snapshot**: its component
/// configs and its safetensors **headers** (no tensor data is read). Nothing here is a number
/// borrowed from 2512, Krea or the production table (epic requirement E13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FootprintFacts {
    /// Elements of every tensor under `transformer/` (the dense DiT, loaded whole at the training
    /// compute dtype).
    pub dit_elements: u64,
    /// Bytes of the Qwen3 language tower as the trainer materialises it (`model.language_model.*`
    /// at the component width; the `lm_head` and the vision tower are never read).
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
    /// The text tower's attention heads.
    pub text_heads: u64,
    /// The VAE encoder's full-resolution channel count (`base_dim`).
    pub vae_encode_channels: u64,
    /// The VAE decoder's full-resolution channel count (`decoder_base_dim`).
    pub vae_decode_channels: u64,
    /// Pixels one latent token covers per side (`scale_factor_spatial`).
    pub pixels_per_token: u64,
}

impl FootprintFacts {
    /// Derive the facts for the snapshot at `root`, pricing the text encoder and the VAE at
    /// `component_width` bytes per float element (what [`loader::compute_dtype_on`] materialises
    /// them at on the training device).
    pub fn from_snapshot(root: &Path, component_width: u64) -> Result<Self> {
        let dit_cfg = TransformerConfig::from_json_file(&root.join("transformer/config.json"))?;
        let te_cfg = TextEncoderConfig::from_json_file(&root.join("text_encoder/config.json"))?;
        let vae_cfg = VaeConfig::from_json_file(&root.join("vae/config.json"))?;
        let headers = |sub: &str| {
            gen_core::weightsmeta::safetensors_path_tensor_headers(root.join(sub))
                .map_err(|e| Error::Msg(format!("{LABEL}: {sub}/ headers: {e}")))
        };
        let bytes = |h: &gen_core::weightsmeta::SafetensorsTensorHeader| -> Result<u64> {
            if h.is_float() {
                h.materialized_bytes(component_width)
                    .map_err(|e| Error::Msg(format!("{LABEL}: {e}")))
            } else {
                Ok(h.data_bytes)
            }
        };

        let mut dit_elements = 0u64;
        for header in headers("transformer")? {
            dit_elements += header
                .element_count()
                .map_err(|e| Error::Msg(format!("{LABEL}: {e}")))?;
        }
        let prefix = format!("{}.", loader::TEXT_ENCODER_PREFIX);
        let mut text_encoder_bytes = 0u64;
        for header in headers("text_encoder")? {
            if header.name.starts_with(&prefix) {
                text_encoder_bytes += bytes(&header)?;
            }
        }
        let (mut vae_encoder_bytes, mut vae_decoder_bytes) = (0u64, 0u64);
        for header in headers("vae")? {
            if header.name.starts_with("encoder.") || header.name.starts_with("quant_conv.") {
                vae_encoder_bytes += bytes(&header)?;
            } else {
                vae_decoder_bytes += bytes(&header)?;
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
            text_heads: te_cfg.num_attention_heads as u64,
            vae_encode_channels: vae_cfg.base_dim as u64,
            vae_decode_channels: vae_cfg.decoder_base_dim as u64,
            pixels_per_token: vae_cfg.scale_factor_spatial as u64,
        })
    }
}

/// The adapter half of a run's shape, sized exactly from the targets' base shapes (no weight read).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdapterFootprint {
    /// Elements of the trainable factors.
    pub trainable_params: u64,
    /// LoKr only (0 for LoRA): per joint token, `Σ in_a·out_b` over the targets inside the most
    /// expensive block — the Kronecker vec-trick's `[S·in_a, out_b]` intermediate.
    pub lokr_block_per_token: u64,
    /// LoKr only: the same sum over every block target.
    pub lokr_blocks_per_token: u64,
    /// LoKr only: the same sum over the global (pre-block / head) targets.
    pub lokr_global_per_token: u64,
}

/// The shape of one training run, as far as memory is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingShape {
    /// Bucketed square training edge, in pixels.
    pub edge: u32,
    /// The longest conditioning sequence (caption or preview prompt), in tokens.
    pub caption_tokens: u64,
    /// Dataset items (each caches one caption feature and one latent).
    pub items: u64,
    /// Bytes per element of the DiT compute dtype (2 for bf16, 4 for f32). The text encoder's
    /// and the VAE's widths are already folded into [`FootprintFacts`]' byte counts.
    pub compute_width: u64,
    pub adapter: AdapterFootprint,
    /// f32 optimizer-state elements per trainable element ([`optimizer_state_per_param`]).
    pub optimizer_state_per_param: u64,
    /// Whether the blocks run gradient-checkpointed.
    pub checkpointed: bool,
    /// Whether preview samples are rendered (keeps the VAE decoder resident).
    pub sampling: bool,
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
/// The DiT working set follows [`QwenImage21Transformer::forward_train`]'s own block structure:
/// one block retains [`BLOCK_SAVED_HIDDEN`] compute-width + [`BLOCK_SAVED_HIDDEN_F32`] f32 `[S,
/// inner]` tensors, `mlp_ratio ·` [`BLOCK_SAVED_PER_MLP_RATIO`] SwiGLU-wide ones, and its attention
/// scores — `heads·(T·S + L²)` elements for the text-to-image layout (the target rows attend to
/// every key, the causal text rows to the prefix), each held [`SCORE_COMPUTE_TENSORS`] times at the
/// compute width and [`SCORE_F32_TENSORS`] times at f32 — plus, for LoKr, the vec-trick
/// intermediates. A
/// **dense** step retains that for every block (candle is eager: the graph holds it until the
/// backward); a **checkpointed** step retains each block's `[S, inner]` input plus ONE block's set
/// (the recompute) and the boundary copy the segmented VJP makes. Both add the backward's in-flight
/// gradients and the retained pre-block forward.
pub fn training_footprint(facts: &FootprintFacts, shape: &TrainingShape) -> TrainingFootprint {
    let w = shape.compute_width;
    let side = shape.edge as u64 / facts.pixels_per_token.max(1);
    let image_tokens = side * side;
    let text = shape.caption_tokens;
    let seq = image_tokens + text;
    let pixels = shape.edge as u64 * shape.edge as u64;

    // Caches (f32): caption features and packed latents, per item.
    let caption_cache = shape.items * text * facts.text_hidden * F32_WIDTH;
    let latent_cache = shape.items * image_tokens * facts.latent_channels * F32_WIDTH;

    // 1. captions: the language tower + one caption's per-layer live set.
    let caption_phase = facts.text_encoder_bytes
        + TEXT_ENCODER_LIVE_HIDDEN * text * facts.text_hidden * F32_WIDTH
        + TEXT_ENCODER_SCORE_MATRICES * facts.text_heads * text * text * F32_WIDTH;

    // 2. latents: the whole VAE + one image's full-resolution encode maps, with the caches growing.
    let latent_phase = facts.vae_encoder_bytes
        + facts.vae_decoder_bytes
        + VAE_LIVE_MAPS * facts.vae_encode_channels * pixels * F32_WIDTH
        + caption_cache
        + latent_cache;

    // 3. train: the DiT at the compute width + trainable state + caches + the step.
    let hidden = seq * facts.inner * w;
    let hidden_f32 = seq * facts.inner * F32_WIDTH;
    let score_elements = facts.heads * (image_tokens * seq + text * text);
    let scores = score_elements * (SCORE_COMPUTE_TENSORS * w + SCORE_F32_TENSORS * F32_WIDTH);
    let block_hidden = BLOCK_SAVED_HIDDEN * hidden
        + BLOCK_SAVED_HIDDEN_F32 * hidden_f32
        + BLOCK_SAVED_PER_MLP_RATIO * facts.mlp_ratio * hidden;
    let lokr = |per_token: u64| seq * per_token * w;
    let prelude = PRELUDE_SAVED_HIDDEN * hidden + lokr(shape.adapter.lokr_global_per_token);
    let retained = if shape.checkpointed {
        facts.num_layers * hidden
            + hidden
            + block_hidden
            + scores
            + lokr(shape.adapter.lokr_block_per_token)
    } else {
        facts.num_layers * (block_hidden + scores) + lokr(shape.adapter.lokr_blocks_per_token)
    };
    let backward =
        BACKWARD_SCORE_GRADS * score_elements * F32_WIDTH + BACKWARD_HIDDEN_GRADS * hidden_f32;
    let step = prelude + retained + backward;
    // A preview render runs between steps (the step's activations are released by then), so its
    // denoise + decode transient competes with the step rather than adding to it.
    let (decoder_resident, preview) = if shape.sampling {
        let tile = if shape.edge > DECODE_TILE_EDGE {
            DECODE_TILE_EDGE as u64
        } else {
            shape.edge as u64
        };
        let decode = VAE_LIVE_MAPS * facts.vae_decode_channels * tile * tile * F32_WIDTH;
        let chunk = score_elements.min(candle_gen::ATTN_SCORES_BUDGET as u64);
        let denoise = PREVIEW_LIVE_HIDDEN * hidden + 3 * chunk * F32_WIDTH;
        (facts.vae_decoder_bytes, decode.max(denoise))
    } else {
        (0, 0)
    };
    let train_phase = facts.dit_elements * w
        + decoder_resident
        + shape.adapter.trainable_params
            * (TRAINABLE_BASE_BUFFERS + shape.optimizer_state_per_param)
            * F32_WIDTH
        + caption_cache
        + latent_cache
        + step.max(preview);

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
         train ~{:.1} GiB). Refusing before step 1 rather than failing mid-run. To fit: {}.",
        gib(peak),
        gib(budget_bytes),
        gib(fp.caption_phase),
        gib(fp.latent_phase),
        gib(fp.train_phase),
        advice.join("; ")
    )))
}

/// The engine-wide "safe peak" fraction of a device's effective free memory (15 % headroom for the
/// driver, other processes and allocator fragmentation) — the same `0.85` the candle decode tilers
/// budget with.
const SAFE_FRAC: f64 = 0.85;

/// This device's training budget: on a CUDA device, the rendered GPU's effective free memory
/// ([`candle_gen::gpu::rendered_effective_free_gib`] — driver-free plus this process's reusable
/// pool, read from a trusted `nvidia-smi`) × [`SAFE_FRAC`]. The preflight runs before the trainer
/// allocates anything, so free memory is what the run can actually have. A CPU device has no
/// device-memory budget to compare against (`u64::MAX`); a CUDA device whose free memory cannot be
/// read logs that the preflight is blind and does not refuse (a candle CUDA OOM is a catchable
/// error, not a process kill).
fn device_budget_bytes(device: &Device) -> u64 {
    if device.is_cpu() {
        return u64::MAX;
    }
    match candle_gen::gpu::rendered_effective_free_gib() {
        Some(free) => (free * SAFE_FRAC * 1024.0 * 1024.0 * 1024.0) as u64,
        None => {
            eprintln!(
                "[sc-24160] {LABEL}: could not read the device's free memory (no trusted \
                 nvidia-smi); the memory preflight cannot refuse this run"
            );
            u64::MAX
        }
    }
}

/// The exact trainable / vec-trick sizing the targets get under `cfg`, from the weight-free
/// projection table. LoRA: `rank·(in + out)`. LoKr: `w1` plus a full or low-rank `w2` by PEFT's
/// `use_w2` rule (exactly as [`build_adapt_lokr_targets_peft`] sizes them), and the per-token
/// vec-trick intermediate `in_a·out_b`.
fn adapter_footprint(
    targets: &[(String, (usize, usize))],
    cfg: &TrainingConfig,
) -> AdapterFootprint {
    let rank = cfg.rank as u64;
    let mut fp = AdapterFootprint::default();
    let mut per_block: HashMap<&str, u64> = HashMap::new();
    for (path, (out_f, in_f)) in targets {
        let (out_f, in_f) = (*out_f, *in_f);
        match cfg.network_type {
            NetworkType::Lora => fp.trainable_params += rank * (in_f + out_f) as u64,
            NetworkType::Lokr => {
                let (out_a, out_b) = factorization(out_f, cfg.decompose_factor);
                let (in_a, in_b) = factorization(in_f, cfg.decompose_factor);
                let w2 = if (cfg.rank as f32) < (out_b.max(in_b) as f32) / 2.0 {
                    rank * (out_b + in_b) as u64
                } else {
                    (out_b * in_b) as u64
                };
                fp.trainable_params += (out_a * in_a) as u64 + w2;
                let intermediate = (in_a * out_b) as u64;
                match block_of(path) {
                    Some(block) => {
                        fp.lokr_blocks_per_token += intermediate;
                        *per_block.entry(block).or_default() += intermediate;
                    }
                    None => fp.lokr_global_per_token += intermediate,
                }
            }
        }
    }
    fp.lokr_block_per_token = per_block.values().copied().max().unwrap_or(0);
    fp
}

/// `transformer_blocks.{i}` for a block target path, `None` for a global one.
fn block_of(path: &str) -> Option<&str> {
    let rest = path.strip_prefix("transformer_blocks.")?;
    let end = rest.find('.')?;
    Some(&path[.."transformer_blocks.".len() + end])
}

// ── the trainer ──────────────────────────────────────────────────────────────────────────────────

/// Identity + capabilities of the candle Qwen-Image 2.1 trainer: LoRA + LoKr, `backend = "candle"`.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: TRAINER_ID,
        family: FAMILY,
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        // Text-to-image LoRA/LoKr only — no control-branch training path.
        supports_control: false,
        // Adapter-only: the shared `validate_full_finetune_request` floor rejects a full tune.
        supports_full_finetune: false,
    }
}

/// The production [`Trainer`] for `qwen_image_2_1` on candle: a frozen dense base that caches a
/// captioned dataset to Qwen3-VL caption features + VAE latents (staged, one heavy component at a
/// time), then runs the LoRA/LoKr loop with the shared runtime glue (LR schedule, gradient
/// accumulation, checkpoints + resume, cancel, previews, progress).
pub struct QwenImage21Trainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
    tokenizer: TextTokenizer,
    /// Tokens of the system-role prefix the conditioning drops.
    drop_count: usize,
    scheduler: SchedulerConfig,
    dit_cfg: TransformerConfig,
    facts: FootprintFacts,
    /// Test seam: replaces [`device_budget_bytes`] so the preflight refusal is exercisable on any
    /// machine.
    memory_budget_override: Option<u64>,
}

/// Construct the trainer from a `Qwen/Qwen-Image-2.1` snapshot directory (the dense BF16 tier) on
/// the default candle device. Reads only the tokenizer, configs and safetensors headers: the heavy
/// components load staged inside [`Trainer::train`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(QwenImage21Trainer::load(spec)?))
}

// The trainer registration constant bridges the crate's rich `Result` into gen-core's.
candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

impl QwenImage21Trainer {
    /// [`load_trainer`] without the box, on [`candle_gen::default_device`].
    pub fn load(spec: &LoadSpec) -> Result<Self> {
        Self::load_on(spec, candle_gen::default_device()?)
    }

    /// [`Self::load`] onto an explicit `device`.
    pub fn load_on(spec: &LoadSpec, device: Device) -> Result<Self> {
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
        if spec.control.is_some()
            || !spec.extra_controls.is_empty()
            || spec.ip_adapter.is_some()
            || spec.identity.is_some()
        {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: control / IP-adapter / identity overlays are not part of \
                 text-to-image LoRA/LoKr training"
            )));
        }
        let root = loader::snapshot_root(&spec.weights)?.to_path_buf();
        let tier = installed_tier(&root)?;
        if tier != Tier::Bf16 {
            return Err(quantized_base_refusal(&format!(
                "{} is the pre-quantized {} tier",
                root.display(),
                tier.dir_name()
            )));
        }
        let tokenizer = loader::load_tokenizer(&root)?;
        let drop_count = system_prompt_drop_count(&tokenizer)?;
        let scheduler = loader::load_scheduler_config(&root)?;
        let dit_cfg = TransformerConfig::from_json_file(&root.join("transformer/config.json"))?;
        let component_width = loader::compute_dtype_on(&device).size_in_bytes() as u64;
        let facts = FootprintFacts::from_snapshot(&root, component_width)?;
        Ok(Self {
            descriptor: trainer_descriptor(),
            root,
            device,
            tokenizer,
            drop_count,
            scheduler,
            dit_cfg,
            facts,
            memory_budget_override: None,
        })
    }

    /// The snapshot facts the preflight derives from.
    pub fn footprint_facts(&self) -> &FootprintFacts {
        &self.facts
    }

    /// Replace the device budget the preflight compares against (bytes). A test seam — the
    /// production budget is the CUDA device's effective free memory × 0.85 (no device budget on a
    /// CPU device).
    pub fn set_memory_budget_override(&mut self, bytes: Option<u64>) {
        self.memory_budget_override = bytes;
    }

    /// The DiT compute dtype for a run on this trainer's device: `train_dtype` on a GPU (bf16 by
    /// default), f32 on a CPU device whatever it says (candle's CPU backend has no BF16 matmul).
    fn compute_dtype(&self, cfg: &TrainingConfig) -> DType {
        if self.device.is_cpu() {
            DType::F32
        } else {
            flow_match::parse_compute_dtype(&cfg.train_dtype)
        }
    }

    /// `(path, [out, in])` for every resolved target of `cfg`.
    fn targets(&self, cfg: &TrainingConfig) -> Vec<(String, (usize, usize))> {
        resolve_targets(&self.dit_cfg, cfg)
    }
}

/// Capability-free training-request validation, unit-testable without loaded weights: the shared
/// flow-match floor (empty dataset, zero rank/steps, unsupported optimizer, unrecognised
/// timestep/loss knobs) plus this text-to-image trainer's own refusals of inputs it would otherwise
/// silently ignore.
fn validate_request(req: &TrainingRequest) -> Result<()> {
    validate_flow_match_request(req, LABEL)?;
    if req
        .items
        .iter()
        .any(|item| item.control_image_path.is_some())
    {
        return Err(Error::Msg(format!(
            "{LABEL}: control images are not part of text-to-image LoRA/LoKr training"
        )));
    }
    if req.items.iter().any(|item| !item.model_options.is_empty())
        || !req.config.model_options.is_empty()
    {
        return Err(Error::Msg(format!(
            "{LABEL}: model_options are not consumed by the text-to-image trainer; refusing \
             rather than silently training without them"
        )));
    }
    Ok(())
}

/// Resolve the config's target-module *suffixes* to `(dotted path, [out, in])` on the DiT's
/// weight-free projection table. The DEFAULT (empty `lora_target_modules`) is every
/// [`BLOCK_ADAPTER_TARGETS`] Linear of every block; an explicit list suffix-matches anywhere — PEFT's
/// `target_modules` rule — including the globals. Same rule as the MLX twin.
fn resolve_targets(
    dit_cfg: &TransformerConfig,
    cfg: &TrainingConfig,
) -> Vec<(String, (usize, usize))> {
    let default = cfg.lora_target_modules.is_empty();
    let suffixes: Vec<String> = if default {
        BLOCK_ADAPTER_TARGETS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    QwenImage21Transformer::adaptable_projections(dit_cfg)
        .into_iter()
        .filter(|(path, _)| {
            (!default || path.starts_with("transformer_blocks."))
                && suffixes
                    .iter()
                    .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

impl AdaptLoraHost for QwenImage21Transformer {
    fn visit_adapt_lora_mut(
        &mut self,
        visitor: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        self.visit_adaptable_mut(&mut |path, linear| {
            visitor(path, linear).map_err(|e| candle_core::Error::Msg(e.to_string()))
        })?;
        Ok(())
    }
}

/// Install the run's trainable adapters on `dit`: exactly the resolved `paths` (full dotted paths,
/// so the PEFT suffix rule matches each one alone).
fn install_adapters(
    dit: &mut QwenImage21Transformer,
    paths: &[String],
    cfg: &TrainingConfig,
    device: &Device,
) -> Result<LoraSet> {
    match cfg.network_type {
        NetworkType::Lora => {
            build_adapt_lora_targets(dit, paths, cfg.rank, cfg.alpha, cfg.seed, device)
        }
        NetworkType::Lokr => build_adapt_lokr_targets_peft(
            dit,
            paths,
            cfg.rank,
            cfg.alpha,
            cfg.decompose_factor,
            cfg.seed,
            device,
        ),
    }
}

/// Freeze (`true`) or thaw (`false`) every trainable residual on the DiT — the graph-free preview
/// seam ([`AdaptLinear::set_trainable_frozen`]).
fn set_frozen(dit: &mut QwenImage21Transformer, frozen: bool) -> Result<()> {
    dit.visit_adaptable_mut(&mut |_, linear| {
        linear.set_trainable_frozen(frozen);
        Ok(())
    })?;
    Ok(())
}

/// Tokens a caption contributes to the joint sequence (template rendered, system prefix dropped) —
/// the tokenizer alone, so the preflight knows the exact sequence before any weight loads.
fn caption_tokens(tokenizer: &TextTokenizer, drop: usize, caption: &str) -> Result<u64> {
    let tokens = tokenizer.tokenize_preformatted(&prompt_template(caption))?;
    Ok(tokens.ids.len().saturating_sub(drop) as u64)
}

/// Encode a dataset image into the packed denoiser-space latent the DiT trains on: centre-crop +
/// resize to `[1, 3, edge, edge]` in `[−1, 1]`, widen to opaque RGBA (a constant `+1` alpha plane)
/// when the VAE takes four channels, take the posterior **mode**, normalise `(z − mean)/std`, and
/// flatten unpatched to `[1, (edge/16)², z_dim]` (f32).
fn encode_latents(vae: &QwenImage21Vae, path: &Path, edge: u32, device: &Device) -> Result<Tensor> {
    let rgb = load_image_tensor(path, edge, device)?; // [1, 3, edge, edge]
    let input = if vae.config().in_channels == 4 {
        let alpha = Tensor::ones((1, 1, edge as usize, edge as usize), DType::F32, device)?;
        Tensor::cat(&[&rgb, &alpha], 1)?
    } else {
        rgb
    };
    let mode = vae.encode_mode(&input)?;
    Ok(pack_latents(&vae.normalize(&mode)?)?.detach())
}

/// One cached dataset sample: the packed f32 latent and the f32 caption features.
struct Cached {
    x0: Tensor,
    context: Tensor,
}

/// `candle_core::Error` from the crate error, for the checkpoint segments' closures.
fn to_core(e: Error) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// One micro-step's forward+backward over the trainable factors: build `x_t` at flow-match `t`,
/// predict the velocity through [`QwenImage21Transformer::forward_train`] (raw `t`, no sign flip),
/// regress it onto `noise − x0` in f32, and return `(loss, grads)` keyed by `vars`.
///
/// `checkpoint` selects the gradient-checkpointed backward: the retained pre-block forward
/// ([`QwenImage21Transformer::train_prelude`] — the global projections, so a global target trains
/// through ordinary autograd), every block as one segment of the shared segmented VJP
/// ([`checkpointed_backward_with_input_grad`]) carrying `[x, modulation rows, norm_out rows]` across
/// the boundaries, and the head + loss as the final segment. The recovered boundary cotangent is
/// then stitched back through the retained pre-block forward. Numerically the dense grads (the
/// `checkpointed_grads_match_dense_for_lora_and_lokr` gate).
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &QwenImage21Transformer,
    vars: &[Var],
    x0: &Tensor,
    context: &Tensor,
    noise: &Tensor,
    t: f32,
    grid: (usize, usize),
    mae: bool,
    checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let (h, w) = grid;
    let (x_t, target) = flow_match::build_batch(x0, noise, t as f64)?;
    let x_t = x_t.to_dtype(dit.compute_dtype())?;
    if !checkpoint {
        let v = dit.forward_train(&x_t, context, t, h, w)?;
        let loss = velocity_loss(&v, &target, mae)?;
        let value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        return Ok((value, loss.backward()?));
    }

    let prelude = dit.train_prelude(&x_t, context, t, h, w)?;
    let geometry = &prelude.geometry;
    let retained = [
        prelude.x.clone(),
        prelude.modulation.clone(),
        prelude.out_rows.clone(),
    ];
    let inputs: Vec<Tensor> = retained.iter().map(Tensor::detach).collect();
    let mut segments: Vec<Segment> = Vec::with_capacity(dit.num_blocks() + 1);
    for index in 0..dit.num_blocks() {
        segments.push(Box::new(move |state: &[Tensor]| {
            let x = dit
                .train_block(index, &state[0], &state[1], geometry)
                .map_err(to_core)?;
            Ok(vec![x, state[1].clone(), state[2].clone()])
        }));
    }
    let target_ref = &target;
    segments.push(Box::new(move |state: &[Tensor]| {
        let v = dit
            .train_head(&state[0], &state[2], geometry)
            .map_err(to_core)?;
        Ok(vec![velocity_loss(&v, target_ref, mae)?])
    }));
    let (loss, mut grads, cotangents) =
        checkpointed_backward_with_input_grad(&segments, &inputs, vars)?;
    drop(segments);

    // Continue the chain rule into the retained pre-block forward: `s = Σ ⟨retainedₖ, cotₖ⟩`, then
    // `s.backward()` delivers the global adapters' grads (the cotangents are detached constants).
    // With only block targets nothing there is tracked and this adds nothing.
    let mut surrogate: Option<Tensor> = None;
    for (r, c) in retained.iter().zip(&cotangents) {
        let term = (r.to_dtype(DType::F32)? * c.detach().to_dtype(DType::F32)?)?.sum_all()?;
        surrogate = Some(match surrogate {
            None => term,
            Some(s) => (s + term)?,
        });
    }
    if let Some(surrogate) = surrogate {
        let prelude_grads = surrogate.backward()?;
        for var in vars {
            if let Some(g) = prelude_grads.get(var.as_tensor()) {
                let merged = match grads.get(var.as_tensor()) {
                    Some(prev) => (prev + g)?,
                    None => g.clone(),
                };
                grads.insert(var.as_tensor(), merged);
            }
        }
    }
    Ok((loss, grads))
}

/// Render a preview from the in-progress (frozen — graph-free) adapter through the crate's own
/// render path: seeded packed noise → resolution-shifted flow-match Euler denoise (true CFG
/// against the empty prompt when `guidance > 1`) → RGBA decode composited over white, tiled above
/// [`DECODE_TILE_EDGE`]. A best-effort nicety: failures are logged by the caller, never fatal.
#[allow(clippy::too_many_arguments)]
fn render_sample(
    dit: &QwenImage21Transformer,
    vae: &QwenImage21Vae,
    scheduler: &SchedulerConfig,
    ctx_pos: &Tensor,
    ctx_neg: Option<&Tensor>,
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
    let latents = create_noise(seed, edge, edge, dit.config().in_channels, dit.device())?;
    let negative = if guidance > 1.0 { ctx_neg } else { None };
    let latents = denoise(
        DenoiseInputs {
            transformer: dit,
            sigmas: &sigmas,
            latents,
            prompt_embeds: ctx_pos,
            negative_embeds: negative,
            true_cfg_scale: guidance,
            width: edge,
            height: edge,
            sampler: None,
            seed,
            cancel,
            references: None,
        },
        &mut |_: Progress| {},
    )?;
    let tiling = (edge > DECODE_TILE_EDGE)
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
        validate_request(req)?;
        if self.targets(&req.config).is_empty() {
            return Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: lora_target_modules {:?} matched no adaptable module on the \
                 Qwen-Image 2.1 DiT (default: every block's {})",
                req.config.lora_target_modules,
                BLOCK_ADAPTER_TARGETS.join("/")
            ))
            .into());
        }
        Ok(())
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
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
        let cfg = &req.config;
        let device = self.device.clone();
        let targets = self.targets(cfg);
        if targets.is_empty() {
            return Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: lora_target_modules {:?} matched no adaptable module on the \
                 Qwen-Image 2.1 DiT",
                cfg.lora_target_modules
            )));
        }
        let target_paths: Vec<String> = targets.iter().map(|(p, _)| p.clone()).collect();
        let compute_dtype = self.compute_dtype(cfg);
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
        let edge = bucket_resolution(cfg.resolution);
        let grid = latent_grid(edge, edge);

        // --- preflight: the derived peak against this device, before any weight is read ---
        let mut longest = 0u64;
        for text in req
            .items
            .iter()
            .map(|item| item.caption.as_str())
            .chain(sample_prompts.iter().map(String::as_str))
            .chain(sampling_requested.then_some(""))
        {
            longest = longest.max(caption_tokens(&self.tokenizer, self.drop_count, text)?);
        }
        let shape = TrainingShape {
            edge,
            caption_tokens: longest,
            items: req.items.len() as u64,
            compute_width: compute_dtype.size_in_bytes() as u64,
            adapter: adapter_footprint(&targets, cfg),
            optimizer_state_per_param: optimizer_state_per_param(&cfg.optimizer),
            checkpointed,
            sampling: sampling_requested,
        };
        let budget = self
            .memory_budget_override
            .unwrap_or_else(|| device_budget_bytes(&device));
        check_training_footprint(&self.facts, &shape, budget)?;
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Every input that selects the cached data, so a resume against a changed dataset is
        // refused rather than continued on different data.
        let fingerprint = request_fingerprint(req)?;

        // --- 1. captions: the Qwen3-VL language tower, encoded ONCE, then dropped ---
        on_progress(TrainingProgress::LoadingModel);
        let (captions, sample_caps, sample_neg) = {
            let encoder: QwenImage21TextEncoder =
                loader::load_text_encoder_from(&self.root.join("text_encoder"), &device, None)?;
            let encode = |text: &str| -> Result<Tensor> {
                Ok(encoder
                    .encode_prompt(&self.tokenizer, text, self.drop_count)?
                    .to_dtype(DType::F32)?
                    .detach())
            };
            let mut captions: Vec<Tensor> = Vec::with_capacity(req.items.len());
            for item in &req.items {
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                captions.push(encode(&item.caption)?);
            }
            let mut sample_caps: Vec<(String, Tensor)> = Vec::with_capacity(sample_prompts.len());
            for prompt in &sample_prompts {
                sample_caps.push((prompt.clone(), encode(prompt)?));
            }
            let sample_neg = if sample_caps.is_empty() {
                None
            } else {
                Some(encode("")?)
            };
            (captions, sample_caps, sample_neg)
            // `encoder` drops here: every caption is cached, the tower is idle from now on.
        };

        // --- 2. latents: the VAE encodes each image ONCE; the encoder half is then dropped ---
        let mut vae = loader::load_vae(&self.root, &device)?;
        let total = req.items.len() as u32;
        let mut cache: Vec<Cached> = Vec::with_capacity(req.items.len());
        for (i, (item, context)) in req.items.iter().zip(captions).enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let x0 = encode_latents(&vae, &item.image_path, edge, &device)?;
            cache.push(Cached { x0, context });
        }
        // Cancelled during caching: nothing has trained, so write nothing (and skip the DiT load).
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if cache.is_empty() {
            return Err(Error::Msg(format!("{LABEL}: no usable dataset items")));
        }
        vae.drop_encoder();
        let vae: Option<QwenImage21Vae> = (!sample_caps.is_empty()).then_some(vae);

        // --- 3. the dense DiT, at the training compute dtype ---
        let vb = candle_gen::loader::component_vb(
            &self.root,
            "transformer",
            compute_dtype,
            &device,
            loader::LABEL,
        )?;
        let mut dit = QwenImage21Transformer::new(&self.dit_cfg, vb)?;
        let mut packed = false;
        dit.visit_adaptable_mut(&mut |_, linear| {
            packed |= linear.is_quantized();
            Ok(())
        })?;
        if packed {
            return Err(quantized_base_refusal(
                "the loaded DiT holds packed weights",
            ));
        }

        // --- adapter targets (LoRA or LoKr) + optimizer ---
        let set = install_adapters(&mut dit, &target_paths, cfg, &device)?;
        let mut opt = TrainOptimizer::from_config(
            &cfg.optimizer,
            set.vars.clone(),
            cfg.learning_rate,
            effective_weight_decay(cfg),
        )?;
        let accum = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let stem = file_stem(&req.file_name).to_string();
        let meta = provenance_meta();
        let mae = flow_match::is_mae(cfg);

        // --- resume: continue from the latest snapshot of THIS adapter in output_dir, if any ---
        // `load_resume` refuses a snapshot whose factor surface (rank, network type, targets),
        // training config or dataset fingerprint differs from this run's.
        let mut start_step = 0u32;
        let mut update_idx = 0u32;
        if cfg.resume {
            if let Some((snapshot, _)) = find_latest_resume(&req.output_dir, &stem) {
                let restored = load_resume(&snapshot, &mut opt, &set, cfg, &fingerprint)?;
                if restored.step > cfg.steps {
                    return Err(Error::Msg(format!(
                        "{LABEL}: resume snapshot step {} exceeds requested total steps {}",
                        restored.step, cfg.steps
                    )));
                }
                start_step = restored.step;
                update_idx = restored.update_idx;
            }
        }

        // --- train loop ---
        let mut accumulated: Option<GradStore> = None;
        let mut pending = 0u32;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        // A checkpoint cadence may fall mid-accumulation; its resume bundle is deferred to the next
        // completed optimizer-update boundary, so resuming from it is always exact.
        let mut resume_due = false;
        for step in start_step.saturating_add(1)..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let sample = &cache[((step - 1) as usize) % cache.len()];
            let t = flow_match::sample_unit_timestep(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                flow_match::timestep_seed(cfg.seed, step),
            );
            let noise = flow_match::sample_noise(
                sample.x0.dims(),
                flow_match::noise_seed(cfg.seed, step),
                &device,
            )?;
            let (loss, grads) = compute_loss_grads(
                &dit,
                &set.vars,
                &sample.x0,
                &sample.context,
                &noise,
                t,
                grid,
                mae,
                checkpointed,
            )?;
            last_loss = loss;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads, &set.vars)?;
            pending += 1;
            if step.is_multiple_of(accum) {
                apply_update(
                    &mut opt,
                    &mut accumulated,
                    &set,
                    pending,
                    cfg,
                    update_idx,
                    total_updates,
                    warmup_updates,
                )?;
                pending = 0;
                update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            // Periodic best-effort previews from the in-progress adapter. The trainable residuals
            // are frozen to detached views so the multi-step denoise builds no autograd graph, and
            // thawed afterwards whatever happened, so training resumes with live leaves.
            if let (Some(vae), true) = (vae.as_ref(), step % cfg.sample_every.max(1) == 0) {
                set_frozen(&mut dit, true)?;
                let total = sample_caps.len() as u32;
                for (i, (prompt, ctx_pos)) in sample_caps.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    match render_sample(
                        &dit,
                        vae,
                        &self.scheduler,
                        ctx_pos,
                        sample_neg.as_ref(),
                        sample_seed(cfg.seed, step, i),
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
                            "[sc-24160] {LABEL}: preview sample failed at step {step} (prompt {}): \
                             {e} — skipping this preview, training continues",
                            i + 1
                        ),
                    }
                }
                set_frozen(&mut dit, false)?;
            }

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                create_output_dir(&req.output_dir)?;
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                save_adapter(&set, &meta, &ckpt)?;
                resume_due = true;
                on_progress(TrainingProgress::Checkpoint { step });
            }
            if resume_due && pending == 0 {
                save_resume(
                    &req.output_dir,
                    &stem,
                    step,
                    update_idx,
                    &opt,
                    &set,
                    cfg,
                    &fingerprint,
                )?;
                resume_due = false;
            }
        }

        // Cancelled before a single step completed: the factors are still the no-op init. Surface
        // the cancellation rather than writing a valid-looking identity adapter.
        if steps_run == 0 {
            return Err(Error::Canceled);
        }
        // Flush a pending (sub-`accum`) window as a true mean of the grads it holds.
        if accumulated.is_some() {
            apply_update(
                &mut opt,
                &mut accumulated,
                &set,
                pending,
                cfg,
                update_idx,
                total_updates,
                warmup_updates,
            )?;
            if resume_due {
                save_resume(
                    &req.output_dir,
                    &stem,
                    steps_run,
                    update_idx + 1,
                    &opt,
                    &set,
                    cfg,
                    &fingerprint,
                )?;
            }
        }

        // --- save the final adapter (bare dotted keys + reload contract + provenance/licence) ---
        on_progress(TrainingProgress::Saving);
        create_output_dir(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_adapter(&set, &meta, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::train::TrainingItem;
    use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant, WeightsSource};

    fn tiny_snapshot() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures/tiny-snapshot")
    }

    /// A per-process scratch directory, removed on drop.
    fn scratch(tag: &str) -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(&format!(
                "qwen21_candle_train_{}_{tag}_",
                std::process::id()
            ))
            .tempdir()
            .expect("scratch dir")
    }

    fn trainer() -> QwenImage21Trainer {
        QwenImage21Trainer::load_on(
            &LoadSpec::new(WeightsSource::Dir(tiny_snapshot())),
            Device::Cpu,
        )
        .expect("the tiny snapshot loads as a trainer")
    }

    fn base_config() -> TrainingConfig {
        TrainingConfig {
            rank: 4,
            alpha: 4.0,
            steps: 4,
            resolution: 32,
            save_every: 0,
            learning_rate: 1e-2,
            seed: 7,
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

    /// `n` small, distinct PNGs (non-square, so the centre crop runs) in `dir`.
    fn dataset(dir: &Path, n: usize) -> Vec<TrainingItem> {
        (0..n)
            .map(|i| {
                let path = dir.join(format!("img{i}.png"));
                image::RgbImage::from_fn(40, 36, |x, y| {
                    image::Rgb([
                        (x * 6 + i as u32 * 40) as u8,
                        (y * 7) as u8,
                        ((x + y) * 3 + i as u32 * 90) as u8,
                    ])
                })
                .save(&path)
                .expect("write fixture png");
                TrainingItem::captioned(path, format!("a swatch number {i}"))
            })
            .collect()
    }

    fn request(dir: &Path, items: Vec<TrainingItem>, config: TrainingConfig) -> TrainingRequest {
        TrainingRequest {
            items,
            config,
            output_dir: dir.join("out"),
            file_name: "adapter.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    // ── identity, provenance, validation ─────────────────────────────────────────────────────

    #[test]
    fn descriptor_is_the_qwen_image_2_1_route() {
        let d = trainer_descriptor();
        assert_eq!(d.id, "qwen_image_2_1");
        assert_eq!(TRAINER_ID, crate::MODEL_ID);
        assert_eq!(d.family, "qwen-image-2-1");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.supports_lora && d.supports_lokr);
        assert!(!d.supports_control && !d.supports_full_finetune);
    }

    #[test]
    fn reachable_via_the_trainer_registry_by_id() {
        let registry = crate::provider_registry().unwrap();
        assert!(
            registry
                .trainers()
                .any(|r| (r.descriptor)().id == TRAINER_ID),
            "trainer id {TRAINER_ID} not registered"
        );
        let t = registry
            .load_trainer(
                TRAINER_ID,
                &LoadSpec::new(WeightsSource::Dir(tiny_snapshot())),
            )
            .expect("the registered loader builds the trainer");
        assert_eq!(t.descriptor().backend, "candle");
    }

    #[test]
    fn provenance_names_the_family_base_and_research_licence() {
        let meta: std::collections::BTreeMap<&str, &str> =
            ADAPTER_PROVENANCE.iter().copied().collect();
        assert_eq!(meta["family"], "qwen-image-2-1");
        assert_eq!(meta["baseModel"], "qwen_image_2_1");
        assert_eq!(meta["ss_base_model_version"], "qwen_image_2_1");
        assert!(meta["license"].contains("Qwen Research License"));
        assert_eq!(meta["modelspec.license"], meta["license"]);
        assert!(meta["licenseNotice"].contains("Qwen RESEARCH LICENSE AGREEMENT"));
        for key in ["networkType", "rank", "alpha", "decomposeFactor"] {
            assert!(!meta.contains_key(key), "{key} is a reload-contract key");
        }
    }

    #[test]
    fn validate_rejects_bad_requests_and_accepts_normalised_spellings() {
        let t = trainer();
        let base = req_with(base_config());
        assert!(t.validate(&base).is_ok());
        let bad = |mutate: &dyn Fn(&mut TrainingRequest), want: &str| {
            let mut r = base.clone();
            mutate(&mut r);
            let err = t.validate(&r).unwrap_err().to_string();
            assert!(err.contains(want), "{want}: {err}");
        };
        bad(&|r| r.items.clear(), "dataset is empty");
        bad(&|r| r.config.rank = 0, "rank");
        bad(&|r| r.config.steps = 0, "steps");
        bad(&|r| r.config.optimizer = "lion".into(), "optimizer");
        bad(
            &|r| r.config.timestep_type = "bogus".into(),
            "timestep_type",
        );
        bad(
            &|r| r.config.timestep_bias = "sideways".into(),
            "timestep_bias",
        );
        bad(&|r| r.config.loss_type = "huber".into(), "loss_type");
        bad(
            &|r| r.items[0].control_image_path = Some("/c.png".into()),
            "control images",
        );
        bad(
            &|r| {
                r.config
                    .model_options
                    .insert("references".into(), serde_json::json!([]));
            },
            "model_options",
        );
        bad(
            &|r| r.config.lora_target_modules = vec!["no_such_module".into()],
            "matched no adaptable module",
        );
        let mut full = base.clone();
        full.config.full_finetune = true;
        assert!(t.validate(&full).is_err(), "adapter-only trainer");
        let mut ok = base.clone();
        ok.config.timestep_type = "Weighted".into();
        ok.config.timestep_bias = "high-noise".into();
        ok.config.loss_type = "L1".into();
        ok.config.gradient_checkpointing = true;
        assert!(t.validate(&ok).is_ok());
    }

    /// The default surface is every [`BLOCK_ADAPTER_TARGETS`] Linear of every block (and no
    /// global); an explicit list reaches the globals too.
    #[test]
    fn default_targets_are_every_block_linear_and_explicit_ones_reach_the_globals() {
        let cfg =
            TransformerConfig::from_json_file(&tiny_snapshot().join("transformer/config.json"))
                .unwrap();
        let paths: Vec<String> = resolve_targets(&cfg, &TrainingConfig::default())
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        assert_eq!(paths.len(), cfg.num_layers * BLOCK_ADAPTER_TARGETS.len());
        for i in 0..cfg.num_layers {
            for local in BLOCK_ADAPTER_TARGETS {
                let want = format!("transformer_blocks.{i}.{local}");
                assert!(paths.contains(&want), "missing default target {want}");
            }
        }
        let explicit: Vec<String> = resolve_targets(
            &cfg,
            &TrainingConfig {
                lora_target_modules: vec!["proj_out".into(), "modulation.1".into()],
                ..Default::default()
            },
        )
        .into_iter()
        .map(|(p, _)| p)
        .collect();
        assert_eq!(explicit, ["modulation.1", "proj_out"]);
        assert!(crate::GLOBAL_ADAPTER_TARGETS.contains(&"proj_out"));
    }

    // ── the refusals of the load spec ───────────────────────────────────────────────────────────

    fn copy_dir(src: &Path, dst: &Path) {
        std::fs::create_dir_all(dst).unwrap();
        for entry in std::fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            let to = dst.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &to);
            } else {
                std::fs::copy(entry.path(), to).unwrap();
            }
        }
    }

    /// A complete packed q8 snapshot: the committed packed transformer + text encoder with the
    /// converter's `quantization` marker, everything else copied from the dense tiny snapshot.
    fn q8_snapshot(dir: &Path) -> PathBuf {
        let out = dir.join("q8");
        copy_dir(&tiny_snapshot(), &out);
        let packed = tiny_snapshot().join("../tiers/q8");
        for component in ["transformer", "text_encoder"] {
            for entry in std::fs::read_dir(out.join(component)).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().is_some_and(|e| e == "safetensors") {
                    std::fs::remove_file(path).unwrap();
                }
            }
            std::fs::copy(
                packed.join(component).join("model.safetensors"),
                out.join(component).join("model.safetensors"),
            )
            .unwrap();
            let config_path = out.join(component).join("config.json");
            let mut config: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
            config["quantization"] =
                serde_json::json!({ "bits": 8, "group_size": crate::GROUP_SIZE });
            std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
        }
        out
    }

    fn load_err(spec: &LoadSpec) -> Error {
        match QwenImage21Trainer::load_on(spec, Device::Cpu) {
            Ok(_) => panic!("the trainer must refuse this spec"),
            Err(e) => e,
        }
    }

    /// The trainer is dense-BF16 only: a quantize request and a pre-quantized tier on disk are
    /// both the typed `Unsupported` naming the way out.
    #[test]
    fn a_quantized_base_is_refused_with_install_the_bf16_tier() {
        let spec = LoadSpec::new(WeightsSource::Dir(tiny_snapshot()));
        for err in [
            load_err(&spec.clone().with_quant(Quant::Q8)),
            load_err(&spec.clone().with_quant(Quant::Q4)),
        ] {
            assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
            assert!(err.to_string().contains("install the BF16 tier"), "{err}");
        }
        let dir = scratch("q8");
        let packed = LoadSpec::new(WeightsSource::Dir(q8_snapshot(dir.path())));
        let err = load_err(&packed);
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("pre-quantized q8 tier"), "{msg}");
        assert!(msg.contains("install the BF16 tier"), "{msg}");
    }

    #[test]
    fn precision_override_encoder_substitution_and_load_adapters_are_refused() {
        let spec = LoadSpec::new(WeightsSource::Dir(tiny_snapshot()));
        let mut fp32 = spec.clone();
        fp32.precision = Precision::Fp32;
        assert!(load_err(&fp32).to_string().contains("precision override"));
        let te = spec
            .clone()
            .with_text_encoder(WeightsSource::Dir("/elsewhere".into()));
        assert!(load_err(&te).to_string().contains("text_encoder"));
        let adapted = spec.clone().with_adapters(vec![AdapterSpec::new(
            "/a.safetensors".into(),
            1.0,
            AdapterKind::Lora,
        )]);
        assert!(load_err(&adapted).to_string().contains("bare base"));
        let file = LoadSpec::new(WeightsSource::File("/x.safetensors".into()));
        assert!(load_err(&file).to_string().contains("snapshot directory"));
    }

    // ── preflight ────────────────────────────────────────────────────────────────────────────

    /// The released snapshot's facts, spelled from the production configs (2.1's own geometry):
    /// the DiT's element count is the projection table plus its norm vectors; the text tower and
    /// VAE byte counts are their parameter arithmetic at bf16.
    fn production_facts() -> FootprintFacts {
        let dit = TransformerConfig::production();
        let te = TextEncoderConfig::production();
        let projections: u64 = QwenImage21Transformer::adaptable_projections(&dit)
            .iter()
            .map(|(_, (o, i))| (*o * *i) as u64)
            .sum();
        let norms = (dit.num_layers * 2 * dit.attention_head_dim + dit.context_in_dim) as u64;
        let attn = te.hidden_size * te.num_attention_heads * te.head_dim * 2
            + te.hidden_size * te.num_key_value_heads * te.head_dim * 2;
        let te_params = te.vocab_size * te.hidden_size
            + te.num_hidden_layers * (attn + 3 * te.hidden_size * te.intermediate_size);
        FootprintFacts {
            dit_elements: projections + norms,
            text_encoder_bytes: te_params as u64 * 2,
            vae_encoder_bytes: 300 << 20,
            vae_decoder_bytes: 400 << 20,
            num_layers: dit.num_layers as u64,
            inner: dit.inner_dim() as u64,
            heads: dit.num_attention_heads as u64,
            mlp_ratio: dit.mlp_ratio as u64,
            latent_channels: dit.in_channels as u64,
            text_hidden: te.hidden_size as u64,
            text_heads: te.num_attention_heads as u64,
            vae_encode_channels: 160,
            vae_decode_channels: 144,
            pixels_per_token: 16,
        }
    }

    fn production_adapter(network: NetworkType) -> AdapterFootprint {
        let dit = TransformerConfig::production();
        let cfg = TrainingConfig {
            rank: 16,
            network_type: network,
            ..Default::default()
        };
        adapter_footprint(&resolve_targets(&dit, &cfg), &cfg)
    }

    fn shape(edge: u32, checkpointed: bool) -> TrainingShape {
        TrainingShape {
            edge,
            caption_tokens: 64,
            items: 20,
            compute_width: 2,
            adapter: production_adapter(NetworkType::Lora),
            optimizer_state_per_param: 2,
            checkpointed,
            sampling: false,
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
        let fp = training_footprint(&facts, &shape(512, true));
        assert!(
            fp.train_phase > facts.dit_elements * 2,
            "the bf16 DiT is the floor"
        );
        let f32_shape = TrainingShape {
            compute_width: 4,
            ..shape(1024, true)
        };
        assert!(
            training_footprint(&facts, &f32_shape).peak()
                > training_footprint(&facts, &shape(1024, true)).peak()
        );
    }

    /// The structural consequence the SceneWorks routing relies on (it requires checkpointing on
    /// for this kernel): on a 96 GB card (~81.6 GiB at the safe fraction) a 1024² run fits
    /// checkpointed and does not fit dense — candle's eager dense backward retains every block's
    /// attention probabilities.
    #[test]
    fn at_1024_a_96gb_card_fits_the_checkpointed_run_but_not_the_dense_one() {
        let facts = production_facts();
        let budget = (96.0 * 1e9 * SAFE_FRAC) as u64;
        let ckpt = training_footprint(&facts, &shape(1024, true)).peak();
        let dense = training_footprint(&facts, &shape(1024, false)).peak();
        eprintln!(
            "[sc-24160] 1024² bf16 rank-16 LoRA: checkpointed ~{:.1} GiB, dense ~{:.1} GiB",
            gib(ckpt),
            gib(dense)
        );
        assert!(
            ckpt <= budget && dense > budget,
            "{ckpt} / {dense} vs {budget}"
        );
    }

    #[test]
    fn preflight_refuses_over_budget_with_actionable_advice_and_passes_under() {
        let facts = production_facts();
        let dense = shape(1024, false);
        let ckpt_peak = training_footprint(&facts, &shape(1024, true)).peak();
        let dense_peak = training_footprint(&facts, &dense).peak();
        let budget = (ckpt_peak + dense_peak) / 2;
        let err = check_training_footprint(&facts, &dense, budget)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Gradient Checkpointing"), "{err}");
        assert!(err.contains("which fits"), "{err}");
        assert!(err.contains("1024 px"), "{err}");
        assert!(err.contains("before step 1"), "{err}");
        assert!(check_training_footprint(&facts, &shape(1024, true), budget).is_ok());
        let err = check_training_footprint(&facts, &dense, 1 << 30)
            .unwrap_err()
            .to_string();
        assert!(err.contains("still over"), "{err}");
    }

    /// LoKr's vec-trick intermediates cost more than LoRA at the same rank, by exactly the priced
    /// `[S·in_a, out_b]` term: all blocks' on a dense step, the costliest block's checkpointed.
    ///
    /// *Mutation that reds this:* dropping the `lokr(..)` terms from `retained`.
    #[test]
    fn lokr_intermediates_raise_the_footprint_above_lora() {
        let facts = production_facts();
        let lokr_adapter = production_adapter(NetworkType::Lokr);
        assert!(lokr_adapter.lokr_blocks_per_token > 0);
        assert_eq!(
            lokr_adapter.lokr_blocks_per_token,
            facts.num_layers * lokr_adapter.lokr_block_per_token,
            "every block has the same targets"
        );
        assert_eq!(lokr_adapter.lokr_global_per_token, 0, "default: no globals");
        let seq = (1024 / 16) * (1024 / 16) + 64;
        for checkpointed in [false, true] {
            let lora = shape(1024, checkpointed);
            let lokr = TrainingShape {
                adapter: AdapterFootprint {
                    // Hold the trainable count equal, to isolate the intermediate's term.
                    trainable_params: lora.adapter.trainable_params,
                    ..lokr_adapter
                },
                ..lora
            };
            let delta = training_footprint(&facts, &lokr).train_phase
                - training_footprint(&facts, &lora).train_phase;
            let per_token = if checkpointed {
                lokr_adapter.lokr_block_per_token
            } else {
                lokr_adapter.lokr_blocks_per_token
            };
            assert_eq!(delta, seq * per_token * 2, "checkpointed={checkpointed}");
        }
    }

    #[test]
    fn optimizer_state_is_sized_per_optimizer() {
        assert_eq!(optimizer_state_per_param("adamw"), 2);
        assert_eq!(optimizer_state_per_param("Adam"), 2);
        assert_eq!(optimizer_state_per_param("adamw8bit"), 2);
        assert_eq!(optimizer_state_per_param("rose"), 0);
        assert_eq!(optimizer_state_per_param("prodigy"), 4);
        assert_eq!(optimizer_state_per_param("Prodigy-Opt"), 4);
        let facts = production_facts();
        let base = shape(1024, true);
        let with = |state| {
            training_footprint(
                &facts,
                &TrainingShape {
                    optimizer_state_per_param: state,
                    ..base
                },
            )
            .train_phase
        };
        assert!(with(0) < with(2) && with(2) < with(4));
        assert_eq!(
            with(4) - with(2),
            2 * base.adapter.trainable_params * F32_WIDTH
        );
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
        let per_item = 64 * facts.text_hidden * F32_WIDTH
            + (1024 / 16) * (1024 / 16) * facts.latent_channels * F32_WIDTH;
        assert_eq!(many - few, 1000 * per_item);
    }

    /// The preflight's adapter sizing on the tiny DiT, against the factors the install really
    /// makes: LoRA `rank·(in+out)`; LoKr the PEFT `use_w2` surface.
    #[test]
    fn adapter_footprint_sizes_exactly_what_the_install_trains() {
        let dir_cfg = tiny_snapshot().join("transformer/config.json");
        let dit_cfg = TransformerConfig::from_json_file(&dir_cfg).unwrap();
        for (network, rank) in [
            (NetworkType::Lora, 4),
            (NetworkType::Lokr, 2),
            (NetworkType::Lokr, 4),
        ] {
            let cfg = TrainingConfig {
                rank,
                network_type: network,
                ..base_config()
            };
            let targets = resolve_targets(&dit_cfg, &cfg);
            let paths: Vec<String> = targets.iter().map(|(p, _)| p.clone()).collect();
            let mut dit = loader::load_transformer(&tiny_snapshot(), &Device::Cpu).unwrap();
            let set = install_adapters(&mut dit, &paths, &cfg, &Device::Cpu).unwrap();
            let installed: u64 = set.vars.iter().map(|v| v.elem_count() as u64).sum();
            assert_eq!(
                adapter_footprint(&targets, &cfg).trainable_params,
                installed,
                "{network:?} rank {rank}"
            );
        }
    }

    /// The facts come from the snapshot itself: the miniature snapshot derives its own (tiny)
    /// footprint rather than the release's.
    #[test]
    fn facts_are_read_off_the_snapshot_itself() {
        let facts = FootprintFacts::from_snapshot(&tiny_snapshot(), 4).unwrap();
        let cfg =
            TransformerConfig::from_json_file(&tiny_snapshot().join("transformer/config.json"))
                .unwrap();
        assert_eq!(facts.num_layers, cfg.num_layers as u64);
        assert_eq!(facts.inner, cfg.inner_dim() as u64);
        assert_eq!(facts.latent_channels, cfg.in_channels as u64);
        assert!(facts.dit_elements > 0 && facts.text_encoder_bytes > 0);
        assert!(facts.vae_encoder_bytes > 0 && facts.vae_decoder_bytes > 0);
        // Pricing at half the width halves every float component.
        let half = FootprintFacts::from_snapshot(&tiny_snapshot(), 2).unwrap();
        assert_eq!(half.text_encoder_bytes * 2, facts.text_encoder_bytes);
        assert_eq!(half.dit_elements, facts.dit_elements);
        let fp = training_footprint(
            &facts,
            &TrainingShape {
                edge: 64,
                ..shape(64, false)
            },
        );
        assert!(
            fp.peak() < 4 << 30,
            "tiny snapshot, tiny footprint: {}",
            fp.peak()
        );
    }

    /// AC: the refusal precedes every load and every step, and writes nothing.
    #[test]
    fn preflight_refusal_stops_the_run_before_any_weight_loads() {
        let mut t = trainer();
        t.set_memory_budget_override(Some(1 << 10));
        let out = scratch("preflight");
        let mut req = req_with(TrainingConfig {
            steps: 2,
            ..base_config()
        });
        req.output_dir = out.path().to_path_buf();
        let mut events = Vec::new();
        let err = t
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

    fn randn(shape: &[usize], seed: u64) -> Tensor {
        flow_match::sample_noise(shape, seed, &Device::Cpu).unwrap()
    }

    /// The tiny DiT's geometry: a 4×4 latent grid (16 tokens) and 5 caption rows.
    fn fixed_batch(dit: &QwenImage21Transformer) -> (Tensor, Tensor, Tensor) {
        let c = dit.config();
        (
            randn(&[1, 16, c.in_channels], 1),
            randn(&[1, 5, c.context_in_dim], 2),
            randn(&[1, 16, c.in_channels], 3),
        )
    }

    fn dit_at(dtype: DType) -> QwenImage21Transformer {
        let cfg =
            TransformerConfig::from_json_file(&tiny_snapshot().join("transformer/config.json"))
                .unwrap();
        let vb = candle_gen::loader::component_vb(
            &tiny_snapshot(),
            "transformer",
            dtype,
            &Device::Cpu,
            loader::LABEL,
        )
        .unwrap();
        QwenImage21Transformer::new(&cfg, vb).unwrap()
    }

    fn install(
        dit: &mut QwenImage21Transformer,
        network: NetworkType,
        rank: u32,
        modules: Vec<String>,
    ) -> LoraSet {
        let cfg = TrainingConfig {
            rank,
            network_type: network,
            lora_target_modules: modules,
            ..base_config()
        };
        let paths: Vec<String> = resolve_targets(dit.config(), &cfg)
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        install_adapters(dit, &paths, &cfg, &Device::Cpu).unwrap()
    }

    /// Move every factor off its no-op init so all of their gradients are non-trivial.
    fn perturb(set: &LoraSet) {
        for (i, v) in set.vars.iter().enumerate() {
            let p = (randn(v.dims(), 100 + i as u64) * 0.05).unwrap();
            v.set(&p).unwrap();
        }
    }

    /// AC: the loss decreases on a fixed batch under the real step (forward_train → MSE → grads →
    /// clip → AdamW), for LoRA and LoKr, dense and checkpointed.
    #[test]
    fn loss_decreases_over_steps_on_a_fixed_batch() {
        for (network, checkpoint) in [
            (NetworkType::Lora, false),
            (NetworkType::Lora, true),
            (NetworkType::Lokr, false),
            (NetworkType::Lokr, true),
        ] {
            let mut dit = dit_at(DType::F32);
            let set = install(&mut dit, network, 4, vec![]);
            let (x0, ctx, noise) = fixed_batch(&dit);
            let mut opt =
                TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
            let mut losses = Vec::new();
            for _ in 0..25 {
                let (loss, mut grads) = compute_loss_grads(
                    &dit,
                    &set.vars,
                    &x0,
                    &ctx,
                    &noise,
                    0.5,
                    (4, 4),
                    false,
                    checkpoint,
                )
                .unwrap();
                assert!(loss.is_finite(), "non-finite loss {loss}");
                losses.push(loss);
                candle_gen::train::optim::clip_grad_norm(&mut grads, &set.vars, 1.0).unwrap();
                opt.step(&grads).unwrap();
            }
            let (first, last) = (losses[0], *losses.last().unwrap());
            eprintln!(
                "[sc-24160] {network:?} ckpt={checkpoint} fixed-batch loss {first:.5} -> {last:.5}"
            );
            assert!(last < 0.9 * first, "{network:?}/{checkpoint}: {losses:?}");
        }
    }

    fn grad_values(g: &GradStore, v: &Var) -> Vec<f32> {
        g.get(v.as_tensor())
            .expect("every factor has a gradient")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// AC: the gradient-checkpointed backward is the dense one — same loss, same grads, for LoRA
    /// (and LoKr at a full and a low-rank `w2`), over the default block surface AND a surface
    /// including global targets (stitched through the retained pre-block forward).
    #[test]
    fn checkpointed_grads_match_dense_for_lora_and_lokr() {
        let with_globals: Vec<String> = [
            "attn.to_q",
            "img_mlp.out",
            "img_in",
            "txt_in.in_layer",
            "modulation.1",
            "norm_out.linear",
            "proj_out",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        for (network, rank) in [
            (NetworkType::Lora, 4),
            (NetworkType::Lokr, 4),
            (NetworkType::Lokr, 2),
        ] {
            for modules in [vec![], with_globals.clone()] {
                let mut dit = dit_at(DType::F32);
                let set = install(&mut dit, network, rank, modules.clone());
                perturb(&set);
                let (x0, ctx, noise) = fixed_batch(&dit);
                let run = |checkpoint| {
                    compute_loss_grads(
                        &dit,
                        &set.vars,
                        &x0,
                        &ctx,
                        &noise,
                        0.4,
                        (4, 4),
                        false,
                        checkpoint,
                    )
                    .unwrap()
                };
                let (l_dense, g_dense) = run(false);
                let (l_ckpt, g_ckpt) = run(true);
                assert!(
                    (l_dense - l_ckpt).abs() <= 1e-5 * l_dense.abs().max(1.0),
                    "{network:?}: loss {l_dense} vs {l_ckpt}"
                );
                let mut max_rel = 0f32;
                for v in &set.vars {
                    let (a, b) = (grad_values(&g_dense, v), grad_values(&g_ckpt, v));
                    let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
                    for (x, y) in a.iter().zip(&b) {
                        max_rel = max_rel.max((x - y).abs() / scale);
                    }
                    assert!(a.iter().any(|x| *x != 0.0), "a factor got a zero gradient");
                }
                eprintln!(
                    "[sc-24160] {network:?} r{rank} globals={} ckpt-vs-dense max rel {max_rel:.2e}",
                    !modules.is_empty()
                );
                assert!(max_rel < 1e-4, "{network:?} r{rank}: {max_rel:.2e}");
            }
        }
    }

    /// The bf16-GPU vs f32-CPU trap, on the CPU lane: candle's CPU backend has no BF16 matmul, so
    /// F16 stands in for the half-precision compute dtype. A DiT held at half precision against f32
    /// factors, f32 caches, f32 noise and an f32 loss must step (dense and checkpointed, LoRA and
    /// LoKr) and render a frozen preview without a dtype mismatch — the class of failure an f32-only
    /// CPU test cannot see.
    #[test]
    fn a_half_precision_dit_steps_and_previews_without_dtype_mismatch() {
        let vae = loader::load_vae(&tiny_snapshot(), &Device::Cpu).unwrap();
        let scheduler = loader::load_scheduler_config(&tiny_snapshot()).unwrap();
        for network in [NetworkType::Lora, NetworkType::Lokr] {
            let mut dit = dit_at(DType::F16);
            let set = install(&mut dit, network, 2, vec![]);
            perturb(&set);
            let (x0, ctx, noise) = fixed_batch(&dit);
            for checkpoint in [false, true] {
                let (loss, grads) = compute_loss_grads(
                    &dit,
                    &set.vars,
                    &x0,
                    &ctx,
                    &noise,
                    0.3,
                    (4, 4),
                    true,
                    checkpoint,
                )
                .unwrap();
                assert!(loss.is_finite(), "{network:?}/{checkpoint}: {loss}");
                for v in &set.vars {
                    let g = grads.get(v.as_tensor()).expect("gradient");
                    assert_eq!(g.dtype(), DType::F32, "master-weight grads stay f32");
                }
            }
            set_frozen(&mut dit, true).unwrap();
            let image = render_sample(
                &dit,
                &vae,
                &scheduler,
                &ctx,
                Some(&ctx),
                3,
                64,
                2,
                2.0,
                &CancelFlag::new(),
            )
            .unwrap();
            assert_eq!((image.width, image.height), (64, 64));
            set_frozen(&mut dit, false).unwrap();
        }
    }

    /// The training forward (composable attention ops) computes the function the render path runs
    /// (fused kernels): the adapter is optimised against what inference will execute. Checked bare
    /// and adapted, and the checkpoint pieces (`train_prelude` → `train_block`… → `train_head`)
    /// compose to the same velocity.
    #[test]
    fn the_training_forward_is_the_render_forward() {
        let mut dit = dit_at(DType::F32);
        let (x0, ctx, _) = fixed_batch(&dit);
        let max_diff = |a: &Tensor, b: &Tensor| {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        for adapted in [false, true] {
            if adapted {
                let set = install(&mut dit, NetworkType::Lora, 4, vec![]);
                perturb(&set);
            }
            let render = dit.forward(&x0, &ctx, 0.6, 4, 4).unwrap();
            let train = dit.forward_train(&x0, &ctx, 0.6, 4, 4).unwrap();
            let prelude = dit.train_prelude(&x0, &ctx, 0.6, 4, 4).unwrap();
            let mut x = prelude.x.clone();
            for index in 0..dit.num_blocks() {
                x = dit
                    .train_block(index, &x, &prelude.modulation, &prelude.geometry)
                    .unwrap();
            }
            let pieces = dit
                .train_head(&x, &prelude.out_rows, &prelude.geometry)
                .unwrap();
            let (d_train, d_pieces) = (max_diff(&render, &train), max_diff(&train, &pieces));
            eprintln!("[sc-24160] adapted={adapted}: render vs train {d_train:.2e}, pieces {d_pieces:.2e}");
            assert!(d_train < 1e-4, "adapted={adapted}: {d_train}");
            assert_eq!(
                d_pieces, 0.0,
                "the pieces are the training forward, op for op"
            );
        }
    }

    /// A frozen preview builds no graph (no gradient reaches a factor through it), and thawing
    /// restores the live leaves.
    #[test]
    fn freezing_detaches_the_factors_and_thawing_restores_them() {
        let mut dit = dit_at(DType::F32);
        let set = install(&mut dit, NetworkType::Lora, 4, vec![]);
        perturb(&set);
        let (x0, ctx, _) = fixed_batch(&dit);
        let velocity_sum = |dit: &QwenImage21Transformer| {
            dit.forward_train(&x0, &ctx, 0.5, 4, 4)
                .unwrap()
                .sum_all()
                .unwrap()
        };
        set_frozen(&mut dit, true).unwrap();
        let frozen = velocity_sum(&dit);
        let grads = frozen.backward().unwrap();
        assert!(
            set.vars.iter().all(|v| grads.get(v.as_tensor()).is_none()),
            "a frozen forward must not reach the factors"
        );
        set_frozen(&mut dit, false).unwrap();
        let live = velocity_sum(&dit);
        assert_eq!(
            frozen.to_scalar::<f32>().unwrap(),
            live.to_scalar::<f32>().unwrap(),
            "freezing changes the graph, never the values"
        );
        let grads = live.backward().unwrap();
        assert!(set.vars.iter().all(|v| grads.get(v.as_tensor()).is_some()));
    }

    // ── the whole run, through `Trainer::train` ─────────────────────────────────────────────

    fn tensors(path: &Path) -> HashMap<String, Tensor> {
        candle_core::safetensors::load(path, &Device::Cpu).unwrap()
    }

    fn metadata(path: &Path) -> std::collections::BTreeMap<String, String> {
        gen_core::weightsmeta::safetensors_file_metadata(path).unwrap()
    }

    /// The exact key → shape layout the MLX trainer writes for `cfg` on the tiny DiT (bare dotted
    /// paths; LoRA `lora_A.weight [r,in]` / `lora_B.weight [out,r]` / `alpha [1]`; LoKr `lokr_w1` and
    /// `lokr_w2` or `lokr_w2_a`/`lokr_w2_b` by PEFT's `use_w2` rule) — spelled here independently
    /// of the candle installer.
    fn mlx_layout(cfg: &TrainingConfig) -> std::collections::BTreeMap<String, Vec<usize>> {
        let dit_cfg =
            TransformerConfig::from_json_file(&tiny_snapshot().join("transformer/config.json"))
                .unwrap();
        let r = cfg.rank as usize;
        let mut out = std::collections::BTreeMap::new();
        for (path, (o, i)) in resolve_targets(&dit_cfg, cfg) {
            match cfg.network_type {
                NetworkType::Lora => {
                    out.insert(format!("{path}.lora_A.weight"), vec![r, i]);
                    out.insert(format!("{path}.lora_B.weight"), vec![o, r]);
                    out.insert(format!("{path}.alpha"), vec![1]);
                }
                NetworkType::Lokr => {
                    let (oa, ob) = factorization(o, cfg.decompose_factor);
                    let (ia, ib) = factorization(i, cfg.decompose_factor);
                    out.insert(format!("{path}.lokr_w1"), vec![oa, ia]);
                    if (r as f32) < (ob.max(ib) as f32) / 2.0 {
                        out.insert(format!("{path}.lokr_w2_a"), vec![ob, r]);
                        out.insert(format!("{path}.lokr_w2_b"), vec![r, ib]);
                    } else {
                        out.insert(format!("{path}.lokr_w2"), vec![ob, ib]);
                    }
                }
            }
        }
        out
    }

    /// AC: a full `Trainer::train` writes an adapter whose keys are the MLX trainer's layout, whose
    /// metadata carries the reload contract + provenance + licence, which loads back strictly
    /// through the sc-24157 host with no conversion and changes the DiT's velocity — for LoRA and
    /// LoKr (low-rank and full `w2`), with previews rendered from the in-progress adapter on the way.
    #[test]
    fn trained_adapters_match_the_mlx_layout_reload_strictly_and_change_the_velocity() {
        for (network, rank, kind) in [
            (NetworkType::Lora, 4, AdapterKind::Lora),
            (NetworkType::Lokr, 2, AdapterKind::Lokr),
            (NetworkType::Lokr, 4, AdapterKind::Lokr),
        ] {
            let dir = scratch("reload");
            let cfg = TrainingConfig {
                rank,
                network_type: network,
                gradient_checkpointing: true,
                sample_every: 2,
                sample_prompts: vec!["a preview swatch".into()],
                sample_steps: 2,
                ..base_config()
            };
            let req = request(dir.path(), dataset(dir.path(), 2), cfg.clone());
            let mut t = trainer();
            let mut events = Vec::new();
            let out = t
                .train(&req, &mut |p| events.push(p))
                .unwrap_or_else(|e| panic!("{network:?} r{rank}: {e}"));
            assert_eq!(out.steps, 4);
            assert!(out.final_loss.is_finite());
            let samples = events
                .iter()
                .filter(|e| matches!(e, TrainingProgress::Sample { .. }))
                .count();
            assert_eq!(samples, 2, "one preview per cadence (steps 2 and 4)");

            // Keys + shapes are exactly the MLX trainer's.
            let saved = tensors(&out.adapter_path);
            let layout: std::collections::BTreeMap<String, Vec<usize>> = saved
                .iter()
                .map(|(k, v)| (k.clone(), v.dims().to_vec()))
                .collect();
            assert_eq!(layout, mlx_layout(&cfg), "{network:?} r{rank}");

            // Metadata: the reload contract + provenance + licence.
            let meta = metadata(&out.adapter_path);
            assert_eq!(meta["networkType"], network_type_name(network));
            assert_eq!(meta["rank"], rank.to_string());
            assert_eq!(meta["alpha"], "4");
            if network == NetworkType::Lokr {
                assert_eq!(meta["decomposeFactor"], "-1");
            }
            for (k, v) in ADAPTER_PROVENANCE {
                assert_eq!(meta[k], v, "{k}");
            }

            // Strict reload through the inference host, and a changed velocity.
            let specs = [AdapterSpec::new(out.adapter_path.clone(), 1.0, kind)];
            crate::adapters::preflight(&tiny_snapshot(), &specs, Tier::Bf16)
                .expect("the weight-free admission accepts the trained adapter");
            let base = loader::load_transformer(&tiny_snapshot(), &Device::Cpu).unwrap();
            let mut adapted = loader::load_transformer(&tiny_snapshot(), &Device::Cpu).unwrap();
            let report =
                crate::adapters::install(&mut adapted, &specs, Tier::Bf16, &Device::Cpu).unwrap();
            assert_eq!(
                report.residuals,
                saved
                    .keys()
                    .filter(|k| k.ends_with(".lokr_w1") || k.ends_with(".lora_A.weight"))
                    .count(),
                "every trained projection carries a residual"
            );
            let (x0, ctx, _) = fixed_batch(&base);
            let delta = (base.forward(&x0, &ctx, 0.5, 4, 4).unwrap()
                - adapted.forward(&x0, &ctx, 0.5, 4, 4).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
            eprintln!(
                "[sc-24160] {network:?} r{rank}: reloaded adapter moves velocity by {delta:.3e}"
            );
            assert!(
                delta > 1e-6,
                "{network:?} r{rank}: the adapter must change the output"
            );
        }
    }

    fn network_type_name(network: NetworkType) -> &'static str {
        match network {
            NetworkType::Lora => "lora",
            NetworkType::Lokr => "lokr",
        }
    }

    /// AC: checkpoint + resume equals the straight run. A run interrupted after step 2 (its
    /// `save_every = 2` resume bundle written on the update boundary) and resumed to step 4 lands on
    /// the same factors as an uninterrupted 4-step run — with and without gradient accumulation.
    #[test]
    fn resume_round_trip_equals_the_straight_run() {
        for accum in [1u32, 2] {
            let dir = scratch("resume");
            let items = dataset(dir.path(), 2);
            let cfg = TrainingConfig {
                gradient_accumulation: accum,
                ..base_config()
            };

            // A: straight through.
            let straight = {
                let mut req = request(dir.path(), items.clone(), cfg.clone());
                req.output_dir = dir.path().join("straight");
                trainer().train(&req, &mut |_| {}).unwrap()
            };

            // B: interrupted right after step 2, with a checkpoint cadence of 2.
            let interrupted_dir = dir.path().join("resumed");
            let mut req = request(
                dir.path(),
                items.clone(),
                TrainingConfig {
                    save_every: 2,
                    ..cfg.clone()
                },
            );
            req.output_dir = interrupted_dir.clone();
            let cancel = req.cancel.clone();
            let partial = trainer()
                .train(&req, &mut |p| {
                    if matches!(p, TrainingProgress::Training { step: 2, .. }) {
                        cancel.cancel();
                    }
                })
                .unwrap();
            assert_eq!(partial.steps, 2);
            assert!(find_latest_resume(&interrupted_dir, "adapter").is_some());

            // C: resume to the end.
            let mut req = request(
                dir.path(),
                items.clone(),
                TrainingConfig {
                    save_every: 2,
                    resume: true,
                    ..cfg.clone()
                },
            );
            req.output_dir = interrupted_dir.clone();
            let mut trained_steps = Vec::new();
            let resumed = trainer()
                .train(&req, &mut |p| {
                    if let TrainingProgress::Training { step, .. } = p {
                        trained_steps.push(step);
                    }
                })
                .unwrap();
            assert_eq!(
                trained_steps,
                [3, 4],
                "the resumed run continues from step 3"
            );
            assert_eq!(resumed.steps, 4);

            let (a, b) = (
                tensors(&straight.adapter_path),
                tensors(&resumed.adapter_path),
            );
            assert_eq!(a.len(), b.len());
            for (key, ta) in &a {
                let diff = (ta - &b[key])
                    .unwrap()
                    .abs()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .max(0)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(diff <= 1e-6, "accum {accum}: {key} differs by {diff}");
            }

            // A mismatched resume (a different rank) is refused, never silently continued.
            let mut req = request(
                dir.path(),
                items,
                TrainingConfig {
                    rank: 8,
                    save_every: 2,
                    resume: true,
                    ..cfg.clone()
                },
            );
            req.output_dir = interrupted_dir;
            assert!(trainer().train(&req, &mut |_| {}).is_err());
        }
    }

    /// A cancel before the first step writes nothing and surfaces the typed cancellation.
    #[test]
    fn a_cancel_before_any_step_writes_no_adapter() {
        let dir = scratch("cancel");
        let req = request(dir.path(), dataset(dir.path(), 1), base_config());
        req.cancel.cancel();
        let err = trainer().train(&req, &mut |_| {}).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("cancel"), "{err}");
        assert!(!req.output_dir.join(&req.file_name).exists());
    }
}
