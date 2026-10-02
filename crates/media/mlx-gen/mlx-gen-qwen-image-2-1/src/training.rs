//! LoRA/LoKr **training** on the Qwen-Image 2.1 DiT, in pure Rust on mlx-rs (sc-24159) — the MLX
//! text-to-image trainer for `qwen_image_2_1`.
//!
//! [`QwenImage21Trainer`] realizes the core [`Trainer`] contract the way every MLX family trainer
//! does (`mlx-gen-krea`, `mlx-gen-mage`): the DiT is a hand-rolled `&self` forward over raw
//! `Array`s, so the trainable factors live OUTSIDE the model in a [`LoraParams`] map, are
//! re-injected each step into the target [`AdaptableLinear`](mlx_gen::adapters::AdaptableLinear)s
//! through the shared [`mlx_gen::train::lora`] seam, and are stepped with `keyed_value_and_grad` +
//! the core [`TrainOptimizer`] + `clip_grad_norm`. The install mirrors the inference reload
//! op-for-op, so the trained adapter loads back through the sc-24156 adapter host
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
//! 2. **Captions** — the Qwen3-VL tower (language half only) encodes every caption and preview
//!    prompt ONCE, then is dropped.
//! 3. **Latents** — the VAE encodes every image ONCE; the encoder half is then dropped (the decoder
//!    stays only when preview samples are requested).
//! 4. **Train** — the dense DiT loads last. With `gradient_checkpointing` each block runs inside an
//!    `mlx::checkpoint` segment ([`QwenImage21Transformer::forward_checkpointed`]) for LoRA and
//!    LoKr alike.
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
use mlx_gen::gen_core;
use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
use mlx_gen::img2img::preprocess_init_image;
use mlx_gen::tiling::TilingConfig;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, factorization,
    LoraParams, TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    CancelFlag, Error, Image, LoadSpec, Modality, NetworkType, Precision, Progress, Result,
    TrainOptimizer, Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::{SchedulerConfig, TextEncoderConfig, TransformerConfig, VaeConfig};
use crate::loader;
use crate::memory_strategy::derived::{
    MLX_EVAL_SLACK_BYTES, MLX_MAX_ACTIVE_TASKS, VAE_PIPELINED_DECODE_MAPS,
};
use crate::model::{FAMILY, MODEL_ID};
use crate::pipeline::{
    create_noise, decode_rgb, denoise, latent_grid, pack_latents, DenoiseInputs, DECODE_OVERLAP,
    DECODE_TILE_EDGE,
};
use crate::quant::{installed_tier, Tier};
use crate::text_encoder::{prompt_template, system_prompt_drop_count, QwenImage21TextEncoder};
use crate::transformer::{
    BlockTrainables, CheckpointedTrainables, QwenImage21Transformer, BLOCK_ADAPTER_TARGETS,
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

/// Bytes each trainable factor element costs across a step: the f32 factor itself, its f32
/// gradient, the gradient-accumulation buffer, and the two AdamW moments.
const TRAINABLE_BYTES_PER_PARAM: u64 = 5 * F32_WIDTH;

/// `[S, inner]`-shaped tensors the dense backward retains per block for the **attention** half,
/// counted off [`crate::transformer`]'s block forward: the LayerNorm input and the modulated `h`,
/// the q/k/v projections, q/k after the per-head RMS norm and after RoPE (whose f32 rotation
/// planes count double at bf16 width), and the SDPA output. A **structural count, not a
/// measurement** — like every number in [`crate::memory_strategy::derived`].
pub const ATTENTION_SAVED_HIDDEN: u64 = 14;
/// `[S, inner]` tensors the **feed-forward** half retains regardless of its width: the residual
/// after attention, the modulated `h`, and the FFN output.
pub const FFN_SAVED_HIDDEN_FIXED: u64 = 3;
/// `[S, inner·mlp_ratio]` tensors the SwiGLU retains: `gate(h)`, `silu(gate(h))`, `proj(h)` and
/// their product — each `mlp_ratio` hidden-widths wide.
pub const FFN_SAVED_PER_MLP_RATIO: u64 = 4;
/// `[heads, S, S]` matrices one block's attention backward holds at once (the softmax output, its
/// cotangent and the score cotangent of the SDPA fallback VJP), at the compute width. One block
/// at a time: the backward walks the blocks in reverse.
pub const ATTENTION_BACKWARD_SCORE_MATRICES: u64 = 3;
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
        let text_encoder_bytes: u64 = safetensors_path_tensor_headers(root.join("text_encoder"))?
            .iter()
            .filter(|h| {
                h.name
                    .starts_with(&format!("{}.", loader::TEXT_ENCODER_PREFIX))
            })
            .map(|h| h.data_bytes)
            .sum();
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
        })
    }
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
    /// Bytes per element of the DiT compute dtype (2 for bf16, 4 for f32).
    pub compute_width: u64,
    /// Elements of the trainable adapter factors.
    pub trainable_params: u64,
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
    /// The VAE resident + one image's encode transient + the caption cache.
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
/// retains only each block's input plus ONE block's recompute set. Both add one block's attention
/// backward ([`ATTENTION_BACKWARD_SCORE_MATRICES`] `[heads, S, S]` matrices) and MLX's pipelined
/// evaluation ([`MLX_MAX_ACTIVE_TASKS`] + 1 in-flight `[S, inner]` outputs, the same term the
/// render path's derived model carries), plus [`MLX_EVAL_SLACK_BYTES`].
pub fn training_footprint(facts: &FootprintFacts, shape: &TrainingShape) -> TrainingFootprint {
    let w = shape.compute_width;
    let side = shape.edge as u64 / facts.pixels_per_token.max(1);
    let image_tokens = side * side;
    let seq = image_tokens + shape.caption_tokens;
    let pixels = shape.edge as u64 * shape.edge as u64;

    // Caches (f32): caption features and packed latents, per item.
    let caption_cache = shape.items * shape.caption_tokens * facts.text_hidden * F32_WIDTH;
    let latent_cache = shape.items * image_tokens * facts.latent_channels * F32_WIDTH;

    // 1. captions: the language tower + one caption's per-layer live set (f32 stream).
    let caption_phase = facts.text_encoder_bytes
        + TEXT_ENCODER_LIVE_HIDDEN * shape.caption_tokens * facts.text_hidden * F32_WIDTH
        + MLX_EVAL_SLACK_BYTES;

    // 2. latents: the whole VAE + one image's pipelined full-resolution encode maps (f32).
    let vae_encode = VAE_PIPELINED_DECODE_MAPS * facts.vae_encode_channels * pixels * F32_WIDTH;
    let latent_phase = facts.vae_encoder_bytes
        + facts.vae_decoder_bytes
        + vae_encode
        + caption_cache
        + MLX_EVAL_SLACK_BYTES;

    // 3. train: dense DiT at the compute width + trainable state + caches + the step.
    let hidden = seq * facts.inner * w;
    let block_saved = (ATTENTION_SAVED_HIDDEN
        + FFN_SAVED_HIDDEN_FIXED
        + FFN_SAVED_PER_MLP_RATIO * facts.mlp_ratio)
        * hidden;
    let retained = if shape.checkpointed {
        facts.num_layers * hidden + block_saved
    } else {
        facts.num_layers * block_saved
    };
    let attention_backward = ATTENTION_BACKWARD_SCORE_MATRICES * facts.heads * seq * seq * w;
    let pipelined = (MLX_MAX_ACTIVE_TASKS + 1) * hidden;
    let step = retained + attention_backward + pipelined;
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
            VAE_PIPELINED_DECODE_MAPS * facts.vae_decode_channels * tile * tile * F32_WIDTH,
        )
    } else {
        (0, 0)
    };
    let train_phase = facts.dit_elements * w
        + decoder_resident
        + shape.trainable_params * TRAINABLE_BYTES_PER_PARAM
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

/// Elements of the trainable factors `target_paths` get under `cfg` — exact, from the host's base
/// shapes (the probe half; no weight is read): LoRA `rank·(in + out)`; LoKr `w1` plus a full or
/// low-rank `w2` by PEFT's `use_w2` rule, exactly as [`build_lokr_targets`] sizes them.
fn trainable_param_count(
    host: &mut QwenImage21Transformer,
    target_paths: &[String],
    cfg: &TrainingConfig,
) -> Result<u64> {
    let rank = cfg.rank as i64;
    let mut total = 0u64;
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
    }
    Ok(total)
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
        // Text-to-image LoRA/LoKr only — no control-branch training path.
        supports_control: false,
        // Adapter-only: the shared `validate_full_finetune_request` floor rejects a full tune.
        supports_full_finetune: false,
    }
}

/// The production [`Trainer`] for `qwen_image_2_1`: a frozen dense-bf16 base that caches a
/// captioned dataset to Qwen3-VL caption features + VAE latents (staged, one heavy component at a
/// time), then runs the functional-autograd LoRA/LoKr loop with the core runtime glue (LR
/// schedule, gradient accumulation, checkpoints + resume, cancel, previews, progress bands).
pub struct QwenImage21Trainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    tokenizer: TextTokenizer,
    /// Tokens of the system-role prefix the conditioning drops.
    drop_count: usize,
    scheduler: SchedulerConfig,
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
        let facts = FootprintFacts::from_snapshot(&root)?;
        Ok(Self {
            descriptor: trainer_descriptor(),
            root,
            tokenizer,
            drop_count,
            scheduler,
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
/// unsupported optimizer, and an unrecognized `timestep_type`/`timestep_bias`/`loss_type`.
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

/// Encode a centre-cropped square image into the packed denoiser-space latent the DiT trains on:
/// resize + `[−1, 1]` NCHW, widen to opaque RGBA (a constant `+1` alpha plane) when the VAE takes
/// four channels, take the posterior **mode**, normalise `(z − mean)/std`, and flatten unpatched to
/// `[1, (edge/16)², z_dim]`.
fn encode_latents(vae: &QwenImage21Vae, image: &Image, edge: u32) -> Result<Array> {
    let rgb = preprocess_init_image(image, edge, edge)?; // [1, 3, edge, edge]
    let input = if vae.config().in_channels == 4 {
        let alpha = Array::ones::<f32>(&[1, 1, edge as i32, edge as i32])?;
        concatenate_axis(&[&rgb, &alpha], 1)?
    } else {
        rgb
    };
    let mode = vae.encode_mode(&input)?;
    pack_latents(&vae.normalize(&mode)?)
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
    context: &'a Array,
    noise: &'a Array,
    t: f32,
    /// The target's latent grid `(h, w)`.
    grid: (usize, usize),
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

/// One forward+backward over the trainable factors: install `params` (LoRA or LoKr), run the DiT,
/// regress its velocity onto `noise − x0`, return `(loss, grads)`. The DiT graph runs at
/// `spec.dtype`; the noising math, the loss and the grads stay f32 (master weights).
fn compute_loss_grads(
    transformer: &mut QwenImage21Transformer,
    params: &LoraParams,
    spec: &LossSpec<'_>,
    step: &StepInputs<'_>,
) -> Result<(f32, LoraParams)> {
    let (x_t, target) = build_batch(step.x0, step.noise, step.t)?;
    let x_t = x_t.as_dtype(spec.dtype)?;
    let context = step.context.clone();
    let (t, (h, w)) = (step.t, step.grid);
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
        let v = match ckpt_blocks {
            Some(blocks) => transformer.forward_checkpointed(
                &x_t,
                &context,
                t,
                h,
                w,
                &CheckpointedTrainables {
                    params: &p,
                    blocks,
                    alpha,
                    rank,
                    lora_dtype,
                    lokr_dtype: LOKR_DTYPE,
                },
            ),
            None => transformer.forward(&x_t, &context, t, h, w),
        }
        .map_err(|e| Exception::custom(e.to_string()))?;
        let diff = subtract(&v, &target)?;
        // `mean(None)` reduces to a 0-d scalar (grad needs a scalar cotangent).
        let loss = if mae {
            diff.abs()?.mean(None)?
        } else {
            diff.square()?.mean(None)?
        };
        Ok(vec![loss])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (val, grads) = vg(params.clone(), 0)?;
    Ok((val[0].item::<f32>(), grads))
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
/// `guidance > 1`) → RGBA decode composited over white. A best-effort nicety: failures are logged
/// by the caller, never fatal.
#[allow(clippy::too_many_arguments)]
fn render_sample(
    dit: &QwenImage21Transformer,
    vae: &QwenImage21Vae,
    scheduler: &SchedulerConfig,
    ctx_pos: &Array,
    ctx_neg: Option<&Array>,
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
        let edge = bucket_resolution(cfg.resolution);
        let (grid_h, grid_w) = latent_grid(edge, edge);
        let grid = (grid_h as usize, grid_w as usize);

        // --- preflight: the derived peak against this device, before any weight is read ---
        let mut longest = 0u64;
        for text in req
            .items
            .iter()
            .map(|item| item.caption.as_str())
            .chain(sample_prompts.iter().map(String::as_str))
        {
            longest = longest.max(caption_tokens(&self.tokenizer, self.drop_count, text)?);
        }
        let shape = TrainingShape {
            edge,
            caption_tokens: longest,
            items: req.items.len() as u64,
            compute_width: if compute_dtype == Dtype::Float32 {
                4
            } else {
                2
            },
            trainable_params: trainable_param_count(&mut self.probe, &target_paths, cfg)?,
            checkpointed,
            sampling: sampling_requested,
        };
        let budget = self
            .memory_budget_override
            .unwrap_or_else(device_budget_bytes);
        check_training_footprint(&self.facts, &shape, budget)?;

        // --- 1. captions: the Qwen3-VL language tower, encoded ONCE, then dropped ---
        on_progress(TrainingProgress::LoadingModel);
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let (captions, sample_caps, sample_neg) = {
            let encoder: QwenImage21TextEncoder =
                loader::load_text_encoder_from(&self.root.join("text_encoder"), None)?;
            let mut captions: Vec<Array> = Vec::with_capacity(req.items.len());
            for item in &req.items {
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let ctx = encoder.encode_prompt(&self.tokenizer, &item.caption, self.drop_count)?;
                eval([&ctx])?;
                captions.push(ctx);
            }
            let mut sample_caps: Vec<(String, Array)> = Vec::with_capacity(sample_prompts.len());
            for prompt in &sample_prompts {
                let ctx = encoder.encode_prompt(&self.tokenizer, prompt, self.drop_count)?;
                eval([&ctx])?;
                sample_caps.push((prompt.clone(), ctx));
            }
            let sample_neg = if sample_caps.is_empty() {
                None
            } else {
                let neg = encoder.encode_prompt(&self.tokenizer, "", self.drop_count)?;
                eval([&neg])?;
                Some(neg)
            };
            (captions, sample_caps, sample_neg)
            // `encoder` drops here: every caption is cached, the tower is idle from now on.
        };
        mlx_rs::memory::clear_cache();

        // --- 2. latents: the VAE encodes each image ONCE; the encoder half is then dropped ---
        let mut vae = loader::load_vae(&self.root)?;
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Array)> = Vec::with_capacity(req.items.len());
        for (i, (item, context)) in req.items.iter().zip(&captions).enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            let x0 = encode_latents(&vae, &img, edge)?;
            eval([&x0])?;
            cache.push((x0, context.clone()));
        }
        drop(captions);
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            return Err(format!("{TRAINER_ID} trainer: no usable dataset items").into());
        }
        vae.drop_encoder();
        let vae: Option<QwenImage21Vae> = (!sample_caps.is_empty()).then_some(vae);
        mlx_rs::memory::clear_cache();
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
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string();

        // --- resume: continue from the latest snapshot of THIS adapter in output_dir, if any ---
        let mut update_idx: u32 = 0;
        let mut start_step: u32 = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
                check_resumed_params(&params, &loaded)?;
                params = loaded;
                start_step = meta.step;
                update_idx = meta.update_idx;
            }
        }

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
        let mut accumulated: Option<LoraParams> = None;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, context) = &cache[((step - 1) as usize) % cache.len()];
            let t = sample_sigma(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
            )?;
            let noise = random::normal::<f32>(
                x0.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let (loss, grads) = compute_loss_grads(
                &mut transformer,
                &params,
                &spec,
                &StepInputs {
                    x0,
                    context,
                    noise: &noise,
                    t,
                    grid,
                },
            )?;
            last_loss = loss;
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
                let (clipped, _norm) = clip_grad_norm(&avg, 1.0)?;
                let clipped: LoraParams = clipped
                    .into_iter()
                    .map(|(k, v)| (k, v.into_owned()))
                    .collect();
                opt.step(&mut params, &clipped)?;
                eval(params.values())?;
                update_idx += 1;
            }

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
                    &ADAPTER_PROVENANCE,
                    &ckpt,
                )?;
                checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
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
                for (i, (prompt, ctx_pos)) in sample_caps.iter().enumerate() {
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
                        ctx_pos,
                        sample_neg.as_ref(),
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
            &ADAPTER_PROVENANCE,
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
mod tests {
    use super::*;
    use mlx_gen::{TrainingItem, WeightsSource};

    fn tiny_snapshot() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot")
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
        }
    }

    fn shape(edge: u32, checkpointed: bool) -> TrainingShape {
        TrainingShape {
            edge,
            caption_tokens: 64,
            items: 20,
            compute_width: 2,
            trainable_params: 40_000_000,
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
                    context: &ctx,
                    noise: &noise,
                    t: 0.5,
                    grid: (4, 4),
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
                        multiply(&randn(v.shape(), 100 + i as u64), Array::from_f32(0.05)).unwrap();
                    (k, p)
                })
                .collect();
            let blocks = block_trainables(&mut dit, &paths, &params, &cfg).unwrap();
            assert!(blocks
                .iter()
                .all(|b| b.adapter.is_some() && !b.keys.is_empty()));
            let (x0, ctx, noise) = fixed_batch(&dit);
            let step = StepInputs {
                x0: &x0,
                context: &ctx,
                noise: &noise,
                t: 0.4,
                grid: (4, 4),
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
}
