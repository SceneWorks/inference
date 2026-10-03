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
//! The DiT forward used for training is [`QwenImage21Transformer::forward_train_joint`]: the render
//! forward ([`QwenImage21Transformer::forward_joint`]) with the attention core's fused,
//! backward-less kernels swapped for composable ones ([`crate::transformer::Ops`]); the render path
//! itself is untouched.
//!
//! ## Edit mode (sc-24162; the MLX twin's sc-24161)
//! A dataset whose items carry [`TrainingItem::reference_image_paths`] trains an **edit** adapter.
//! The shared gen-core floor ([`gen_core::train::validate_edit_request`]) caps each item at
//! [`MAX_REFERENCE_IMAGES`] — the render path's own cap (the constant
//! [`crate::reference::validate_reference_count`] enforces), advertised through
//! [`TrainerDescriptor::max_reference_images`] — and refuses mixed datasets. Each item's ordered
//! references go through the **render path's own** assembly: host preprocessing
//! ([`prepare_conditioning_references`]), the image-conditioned Qwen3-VL template with vision tokens
//! ([`QwenImage21TextEncoder::encode_conditioning`]), the joint layout + text rows
//! ([`joint_branch`]) and the reference VAE latents ([`encode_references`]) — all cached ONCE, like
//! the target latents. Each step feeds the DiT the image stream the denoise loop does
//! ([`joint_images`]: references in order, then the noised target) over the same layout, so the
//! positional ids and RoPE offsets are the render path's; the DiT returns the target block's
//! velocity only, so the loss covers target tokens only. Edit targets keep their aspect ratio
//! (`edit_target_size`) because their references do. The preflight prices the reference latents,
//! the longer joint sequence and the vision tower, and refuses a reference whose fit the Qwen3-VL
//! processor would rebind ([`reference_fit`]) from its header alone, before any weight loads.
//! Saved edit adapters additionally carry [`EDIT_ADAPTER_MARKER`]; a resume refuses to continue a
//! run of the other mode; previews render edits conditioned on the first item's references.
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
    Trainer, TrainerDescriptor, TrainingConfig, TrainingItem, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use candle_gen::gen_core::{
    self, CancelFlag, Image, LoadSpec, Modality, NetworkType, Precision, Progress, RgbaImage,
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

use crate::config::{
    SchedulerConfig, TextEncoderConfig, TransformerConfig, VaeConfig, VisionConfig,
    IMAGE_TOKENS_PER_SLOT, MAX_REFERENCE_IMAGES, VAE_SCALE_FACTOR,
};
use crate::loader;
use crate::pipeline::{
    create_noise, decode_rgb, denoise, encode_references, joint_branch, joint_images,
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
use crate::transformer::{JointLayout, QwenImage21Transformer, BLOCK_ADAPTER_TARGETS};
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

/// The extra `__metadata__` entry an **edit** adapter (one trained on an edit-pair dataset)
/// carries on top of [`ADAPTER_PROVENANCE`] — the MLX twin's marker, verbatim — so the product layer
/// can tell an instruction-edit adapter from a text-to-image one without inspecting tensors.
/// Text-to-image adapters do not carry the key at all. The adapter itself is the same PEFT/LyCORIS
/// file either way and loads through the same host.
pub const EDIT_ADAPTER_MARKER: (&str, &str) = ("trainingMode", "edit");

/// Provenance + licence (+ the edit marker on an edit run) for every saved adapter.
fn provenance_meta(edit: bool) -> HashMap<String, String> {
    ADAPTER_PROVENANCE
        .iter()
        .chain(edit.then_some(&EDIT_ADAPTER_MARKER))
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
///
/// These two counts are the **forward** retention, and they hold for every SDPA call of a block at
/// once: 22 B per score element at bf16, measured at 22.1–22.5 on CUDA (sc-24163).
pub const SCORE_F32_TENSORS: u64 = 4;
/// f32 `[heads, Sq, Sk]` tensors the attention **backward** holds at its peak, on top of the forward
/// retention, for the **one** SDPA call (or chunk) being backpropagated (sc-24163).
///
/// The peak is candle's `Div` rule for `probs = exp / sum`. It holds the incoming cotangent, the
/// quotient `g / sum`, the numerator's accumulator and the `zeros_like` it was added to, the product
/// `g · exp`, the materialized `sum²` (a full-size square of the broadcast denominator), their
/// quotient, and the denominator's accumulator and its `zeros_like`. That is nine tensors. They are
/// all alive at once because a rule that multiplies the cotangent by a forward tensor carrying an op
/// returns a tensor that carries an op too, and the gradient-store entry built from it keeps that
/// chain until its node is processed. The structural count matches the measurement: 36 B per
/// element. An isolated composable attention measured 58–63 B per score element in total on CUDA
/// (forward 22 + backward 36), and the probe over the real block at the production geometry fit
/// exactly 58.0 per element of its largest call.
///
/// The old count was 2 tensors over **every** call's elements. That was too few for the call in
/// flight and too many for the others, which are backpropagated one at a time.
pub const BACKWARD_SCORE_F32_TENSORS: u64 = 9;
/// f32 `[S, inner]` gradients in flight at once during a block's backward.
const BACKWARD_HIDDEN_GRADS: u64 = 4;
/// `[S, inner]` compute-width tensors the block backward's **retained gradient chains** hold at the
/// attention peak, beyond [`BLOCK_SAVED_HIDDEN`]'s forward set (sc-24163).
///
/// candle computes a gradient for **every** input of a rule, including inputs that never reach a
/// trainable leaf: the frozen base weights, the modulation's token mask and the RoPE tables. Those
/// entries are never consumed, so they live until the segment's backward ends, and each one keeps
/// the chain it was built from. For example, the weight gradient `xᵀ·g` of a base projection holds
/// that projection's output cotangent. By the attention peak the SwiGLU, `to_out` and the gated
/// residuals have already been processed, so their cotangents and the three processed modulation
/// slots' mask chains are among what is live. This count is **measured**, not enumerated
/// tensor-by-tensor: a CUDA probe of the real checkpointed step over the production block geometry
/// (768²…1024², four sizes) fit a per-token term of 352.75 B per channel exactly. That leaves
/// 70.75 B per channel above the forward and in-flight counts, which is 35.4 bf16 tensors, rounded
/// up.
pub const BACKWARD_RETAINED_HIDDEN: u64 = 36;
/// `[S, inner]` compute-width cotangent accumulators for q, k and v. They are already live when a
/// **later** SDPA call's scores backpropagate: an edit layout processes the target call first, then
/// each condition block, and the target call has already accumulated into q, k and v (sc-24163).
pub const BACKWARD_QKV_COTANGENTS: u64 = 3;
/// Copies each frozen base projection's dead weight gradient keeps (sc-24163): the `zeros_like`
/// accumulator `GradStore::or_insert` made, the `xᵀ·g` product, and their sum. The product carries
/// an op, so the sum keeps both alive. That is three `[out, in]` copies at the compute width per
/// projection, held until the segment's backward ends. By the attention peak the SwiGLU's three
/// projections and `to_out` have been processed. For the production block that is 1.007 GB at
/// bf16; the probe's constant term measured 1.021 GB.
pub const DEAD_WEIGHT_GRAD_COPIES: u64 = 3;
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
    /// Bytes of the Qwen3-VL **vision** tower (`model.visual.*`) as the trainer materialises it —
    /// loaded only by an edit run, which encodes its references as vision context (sc-24162). `0`
    /// when the snapshot ships none.
    pub vision_tower_bytes: u64,
    /// The vision tower's hidden width (`vision_config.hidden_size`); `0` without a tower.
    pub vision_hidden: u64,
    /// The vision tower's attention heads; `0` without a tower.
    pub vision_heads: u64,
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
        let vision_prefix = format!("{}.", loader::VISION_TOWER_PREFIX);
        let (mut text_encoder_bytes, mut vision_tower_bytes) = (0u64, 0u64);
        for header in headers("text_encoder")? {
            if header.name.starts_with(&prefix) {
                text_encoder_bytes += bytes(&header)?;
            } else if header.name.starts_with(&vision_prefix) {
                vision_tower_bytes += bytes(&header)?;
            }
        }
        let (vision_hidden, vision_heads) = loader::load_vision_config(root)?.map_or((0, 0), |v| {
            (
                v.tower.hidden_size.max(0) as u64,
                v.tower.num_heads.max(0) as u64,
            )
        });
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
            vision_tower_bytes,
            vision_hidden,
            vision_heads,
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
    /// Latent tokens of the largest **target**. `0` means the square `edge × edge` target every
    /// text-to-image item trains at; an edit run keeps each target's aspect ratio
    /// (`edit_target_size`), so it prices its largest target explicitly.
    pub target_tokens: u64,
    /// The longest conditioning sequence (caption or preview prompt), in **text** tokens — the
    /// vision slots of an edit prompt are counted by [`reference_tokens`](Self::reference_tokens).
    pub caption_tokens: u64,
    /// Edit runs: the largest per-prompt sum of reference latent tokens — the condition blocks the
    /// joint sequence carries ahead of the target. `0` for text-to-image.
    pub reference_tokens: u64,
    /// Edit runs: the latent tokens (= ViT patches) of the single largest reference — its
    /// VAE-encode and vision-tower transients. `0` for text-to-image.
    pub largest_reference_tokens: u64,
    /// Edit runs: reference latent tokens cached across the whole dataset. `0` for text-to-image.
    pub reference_cache_tokens: u64,
    /// Score elements (per head) of the block-causal **prefix** attention calls for the costliest
    /// prompt — `Σ (end − start)·end` over its layout's prefix segments ([`prefix_score_elements`]):
    /// each prefix segment's rows attend to every key up to that segment's end. `0` falls back to
    /// the whole prefix squared (`(caption_tokens + reference_tokens)²`), which is exact for a
    /// text-only prefix and an upper bound otherwise.
    pub prefix_scores: u64,
    /// Score elements (per head) of the costliest **single** prefix attention call:
    /// `max (end − start)·end` over the prefix segments ([`largest_prefix_call_elements`]). The
    /// attention backward runs one call at a time, so its transient is sized by the largest call,
    /// whether that is this one or the target's. `0` falls back to the whole prefix squared.
    pub largest_prefix_call: u64,
    /// Rows the block-causal attention copies for the prefix calls: `Σ (end − start) + 3·end` over
    /// the prefix segments ([`prefix_copy_rows`]). Each call takes contiguous copies of its query rows,
    /// of the keys and values up to its end and of those keys transposed, and the graph retains those
    /// copies. `0` falls back to `4 ·` the prefix.
    pub prefix_copy_rows: u64,
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
/// scores — `heads·(T·S + Σ (end − start)·end)` elements over the prefix segments (the text plus,
/// for an edit run, every condition block: the target rows attend to every key, each block-causal
/// prefix segment's rows to the keys up to its end — `L²` for a text-only prefix), each held
/// [`SCORE_COMPUTE_TENSORS`] times at the
/// compute width and [`SCORE_F32_TENSORS`] times at f32 — plus the attention's contiguous q/k/v
/// copies and, for LoKr, the vec-trick intermediates. A **dense** step retains that for every block
/// (candle is eager: the graph holds it until the backward); a **checkpointed** step retains each
/// block's `[S, inner]` input plus ONE block's set (the recompute) and the boundary copy the
/// segmented VJP makes. Both add the retained pre-block forward and the backward's peak: the
/// largest single attention call's [`BACKWARD_SCORE_F32_TENSORS`] f32 score tensors, the in-flight
/// and retained `[S, inner]` gradient chains, and the frozen projections' dead weight gradients
/// ([`DEAD_WEIGHT_GRAD_COPIES`]). On a dense step the last two accumulate over every block.
///
/// sc-24163 measured this against the CUDA driver's live high-water. With the backward priced as two
/// f32 tensors over every call's scores and no retained chains, the step came out 26–45 % low.
pub fn training_footprint(facts: &FootprintFacts, shape: &TrainingShape) -> TrainingFootprint {
    let w = shape.compute_width;
    let side = shape.edge as u64 / facts.pixels_per_token.max(1);
    let image_tokens = if shape.target_tokens > 0 {
        shape.target_tokens
    } else {
        side * side
    };
    let text = shape.caption_tokens;
    // The block-causal prefix: the text and (edit) every condition block ahead of the target.
    let prefix = text + shape.reference_tokens;
    let seq = image_tokens + prefix;
    let token_pixels = facts.pixels_per_token * facts.pixels_per_token;
    // The largest single VAE encode: the square edge, the (aspect-preserving) target, or (edit) a
    // reference fitted to the vision resolution, which can exceed the training edge.
    let pixels = (shape.edge as u64 * shape.edge as u64)
        .max(image_tokens * token_pixels)
        .max(shape.largest_reference_tokens * token_pixels);

    // Caches (f32): caption text rows and packed latents (targets + edit references), per item.
    let caption_cache = shape.items * text * facts.text_hidden * F32_WIDTH;
    let latent_cache = (shape.items * image_tokens + shape.reference_cache_tokens)
        * facts.latent_channels
        * F32_WIDTH;

    // 1. captions: the language tower + one caption's per-layer live set. An edit caption also
    //    runs the vision tower (resident, plus one reference's patch stream and its full patch
    //    attention), and its language sequence carries one vision slot per merged 2×2 latent group.
    let vision_slots = shape.reference_tokens / IMAGE_TOKENS_PER_SLOT as u64;
    let language = text + vision_slots;
    let vision = if shape.reference_tokens > 0 {
        let patches = shape.largest_reference_tokens;
        facts.vision_tower_bytes
            + TEXT_ENCODER_LIVE_HIDDEN * patches * facts.vision_hidden * F32_WIDTH
            + TEXT_ENCODER_SCORE_MATRICES * facts.vision_heads * patches * patches * F32_WIDTH
    } else {
        0
    };
    let caption_phase = facts.text_encoder_bytes
        + vision
        + TEXT_ENCODER_LIVE_HIDDEN * language * facts.text_hidden * F32_WIDTH
        + TEXT_ENCODER_SCORE_MATRICES * facts.text_heads * language * language * F32_WIDTH;

    // 2. latents: the whole VAE + one image's full-resolution encode maps, with the caches growing.
    let latent_phase = facts.vae_encoder_bytes
        + facts.vae_decoder_bytes
        + VAE_LIVE_MAPS * facts.vae_encode_channels * pixels * F32_WIDTH
        + caption_cache
        + latent_cache;

    // 3. train: the DiT at the compute width + trainable state + caches + the step.
    let hidden = seq * facts.inner * w;
    let hidden_f32 = seq * facts.inner * F32_WIDTH;
    // The target rows attend to every key; each prefix segment's rows to the keys up to its end.
    let prefix_scores = if shape.prefix_scores > 0 {
        shape.prefix_scores
    } else {
        prefix * prefix
    };
    let score_elements = facts.heads * (image_tokens * seq + prefix_scores);
    // Forward retention: every call's scores, held by the block's graph until its backward ends.
    let scores = score_elements * (SCORE_COMPUTE_TENSORS * w + SCORE_F32_TENSORS * F32_WIDTH);
    // Backward transient: the calls backpropagate one at a time, so the largest one sizes it. A call
    // over the i32 guard's budget runs in query chunks, and the in-flight chunk holds the full set.
    // The other chunks' score cotangents (compute width) stay alive meanwhile: each chunk's `kᵀ`
    // gradient keeps its own, and every chunk shares the one `kᵀ`, which is processed last.
    let largest_prefix_call = if shape.largest_prefix_call > 0 {
        shape.largest_prefix_call
    } else {
        prefix * prefix
    };
    let largest_call = facts.heads * (image_tokens * seq).max(largest_prefix_call);
    let chunk = largest_call.min(candle_gen::ATTN_SCORES_BUDGET as u64);
    let score_backward =
        BACKWARD_SCORE_F32_TENSORS * chunk * F32_WIDTH + (largest_call - chunk) * w;
    let block_hidden = BLOCK_SAVED_HIDDEN * hidden
        + BLOCK_SAVED_HIDDEN_F32 * hidden_f32
        + BLOCK_SAVED_PER_MLP_RATIO * facts.mlp_ratio * hidden;
    // The block-causal attention's contiguous copies: each prefix call's query rows, its keys and
    // values up to its end and its transposed keys, plus the target call's query rows and its
    // transposed keys (the whole sequence). The target call's keys and values are the whole, already
    // contiguous, sequence, so they are not copied again.
    let prefix_copy_rows = if shape.prefix_copy_rows > 0 {
        shape.prefix_copy_rows
    } else {
        4 * prefix
    };
    let copies = (prefix_copy_rows + image_tokens + seq) * facts.inner * w;
    // The frozen base projections whose dead weight gradients are held at the attention peak: the
    // SwiGLU's three and `to_out`.
    let dead_weight_grads = DEAD_WEIGHT_GRAD_COPIES
        * w
        * (3 * facts.mlp_ratio * facts.inner * facts.inner + facts.inner * facts.inner);
    let block_backward = (BACKWARD_RETAINED_HIDDEN + BACKWARD_QKV_COTANGENTS) * hidden
        + BACKWARD_HIDDEN_GRADS * hidden_f32;
    let lokr = |per_token: u64| seq * per_token * w;
    let prelude = PRELUDE_SAVED_HIDDEN * hidden + lokr(shape.adapter.lokr_global_per_token);
    let (retained, backward) = if shape.checkpointed {
        // Every block's input stashed, the boundary copy the segmented VJP makes, and ONE block's
        // recomputed graph with its backward.
        (
            facts.num_layers * hidden
                + hidden
                + block_hidden
                + copies
                + scores
                + lokr(shape.adapter.lokr_block_per_token),
            score_backward + block_backward + dead_weight_grads,
        )
    } else {
        // One graph over every block, backpropagated in one `GradStore`: every block's forward set
        // is retained, and every processed block's dead weight gradients and retained chains
        // accumulate until the backward ends.
        (
            facts.num_layers * (block_hidden + copies + scores)
                + lokr(shape.adapter.lokr_blocks_per_token),
            score_backward + facts.num_layers * (block_backward + dead_weight_grads),
        )
    };
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
    if shape.reference_tokens > 0 {
        advice.push(
            "use fewer reference images per edit (each one lengthens the joint sequence)"
                .to_string(),
        );
    }
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
        // LoRA/LoKr adapters only — no control-branch training path.
        supports_control: false,
        // Adapter-only: the shared `validate_full_finetune_request` floor rejects a full tune.
        supports_full_finetune: false,
        // Instruction-edit datasets (sc-24162), capped at the render path's own reference limit —
        // the one constant `collect_references`/`validate_reference_count` enforce.
        max_reference_images: MAX_REFERENCE_IMAGES as u32,
    }
}

/// The production [`Trainer`] for `qwen_image_2_1` on candle: a frozen dense base that caches a
/// captioned (or edit-pair) dataset to Qwen3-VL conditioning + VAE latents (staged, one heavy
/// component at a time), then runs the LoRA/LoKr loop with the shared runtime glue (LR schedule,
/// gradient accumulation, checkpoints + resume, cancel, previews, progress).
pub struct QwenImage21Trainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
    tokenizer: TextTokenizer,
    /// Tokens of the system-role prefix the conditioning drops.
    drop_count: usize,
    scheduler: SchedulerConfig,
    /// The snapshot's Qwen3-VL vision geometry, when it ships a vision tower — required by an edit
    /// run (its references are vision context, exactly as on the render path).
    vision: Option<VisionConfig>,
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
        // Shared with the MLX twin (sc-24163): one refusal, one message.
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
        let tokenizer = loader::load_tokenizer(&root)?;
        let drop_count = system_prompt_drop_count(&tokenizer)?;
        let scheduler = loader::load_scheduler_config(&root)?;
        let vision = loader::load_vision_config(&root)?;
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
            vision,
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
/// timestep/loss knobs) plus this trainer's own refusals of inputs it would otherwise silently
/// ignore.
fn validate_request(req: &TrainingRequest) -> Result<()> {
    validate_flow_match_request(req, LABEL)?;
    if req
        .items
        .iter()
        .any(|item| item.control_image_path.is_some())
    {
        return Err(Error::Msg(format!(
            "{LABEL}: control images are not part of Qwen-Image 2.1 LoRA/LoKr training"
        )));
    }
    // The worker forwards its whole `advanced` map as `model_options` and native trainers parse
    // only the keys they own, so an unknown key is ignored — but a key that would switch this run
    // into a workflow it does not read from there (reference / control conditioning) is refused
    // rather than silently trained without it. A key the worker sends in its *off* state (null, an
    // empty list or string, `{}`, `false`, `"none"`) selects nothing and is ignored like any other.
    // The list, the off-value rule and the message are gen-core's, shared with the MLX twin
    // (sc-24163), so the two backends cannot drift.
    gen_core::train::refuse_reference_control_model_options(LABEL, req)?;
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

/// The `__metadata__` key an adapter's training mode is stamped under ([`EDIT_ADAPTER_MARKER`] on
/// instruction-edit adapters, here and on the MLX twin); a text-to-image adapter carries no stamp.
const TRAINING_MODE_KEY: &str = "trainingMode";

/// Refuse resuming this run from a run of the other training mode. The factors of an
/// instruction-edit run were trained on a different conditioning than a text-to-image run's, so
/// continuing one as the other — just because the factor shapes match — would silently mix two
/// objectives.
///
/// The resume bundle itself carries no mode (the shared `save_resume` writes none), so the mode is
/// read off the **adapter checkpoints** of `stem` in `dir` — exactly the
/// [`checkpoint_filename`] spelling `{stem}-step{digits}.safetensors`, never the
/// `{stem}-step{digits}.resume.safetensors` bundles that share its prefix; every bundle is written
/// only after one of them — plus the bundle's own metadata in case a future writer stamps it:
/// * a **text-to-image** run (`edit = false`) refuses any file stamped with a mode other than
///   text-to-image;
/// * an **edit** run refuses any file stamped otherwise, and any intermediate checkpoint that is
///   unstamped (a text-to-image run's).
fn check_resume_training_mode(dir: &Path, stem: &str, snapshot: &Path, edit: bool) -> Result<()> {
    let prefix = format!("{stem}-step");
    let mut checkpoints = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_checkpoint = name
                .strip_prefix(&prefix)
                .and_then(|rest| rest.strip_suffix(".safetensors"))
                .is_some_and(|step| !step.is_empty() && step.bytes().all(|b| b.is_ascii_digit()));
            if is_checkpoint {
                checkpoints.push(entry.path());
            }
        }
    }
    checkpoints.sort();
    let run_mode = |edit: bool| {
        if edit {
            "instruction-edit"
        } else {
            "text-to-image"
        }
    };
    let files = std::iter::once((snapshot.to_path_buf(), true))
        .chain(checkpoints.into_iter().map(|path| (path, false)));
    for (file, is_bundle) in files {
        let meta = gen_core::weightsmeta::safetensors_file_metadata(&file)
            .map_err(|e| Error::Msg(format!("{LABEL}: read {}: {e}", file.display())))?;
        let written_edit = match meta.get(TRAINING_MODE_KEY).map(String::as_str) {
            None if is_bundle => continue,
            None | Some("t2i" | "text_to_image") => false,
            Some(mode) if mode == EDIT_ADAPTER_MARKER.1 => true,
            Some(mode) => {
                return Err(Error::Msg(format!(
                    "{LABEL}: {} was written by a `{mode}` training run; this {} run cannot \
                     resume from it — start a fresh run, or resume it with the trainer mode that \
                     wrote it",
                    file.display(),
                    run_mode(edit)
                )));
            }
        };
        if written_edit != edit {
            return Err(Error::Msg(format!(
                "{LABEL}: {} was written by a `{}` training run; this {} run cannot resume from \
                 it — start a fresh run, or resume it with the trainer mode that wrote it",
                file.display(),
                if written_edit {
                    EDIT_ADAPTER_MARKER.1
                } else {
                    "text-to-image"
                },
                run_mode(edit)
            )));
        }
    }
    Ok(())
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
/// resize to `[1, 3, edge, edge]` in `[−1, 1]`, then [`encode_rgb_latents`] →
/// `[1, (edge/16)², z_dim]` (f32).
fn encode_latents(vae: &QwenImage21Vae, path: &Path, edge: u32, device: &Device) -> Result<Tensor> {
    encode_rgb_latents(vae, load_image_tensor(path, edge, device)?)
}

/// An RGB NCHW `[1, 3, h, w]` image in `[−1, 1]` → its packed latent: widen to opaque RGBA (a
/// constant `+1` alpha plane) when the VAE takes four channels, take the posterior **mode**,
/// normalise `(z − mean)/std`, and flatten unpatched to `[1, (h/16)·(w/16), z_dim]` (f32).
fn encode_rgb_latents(vae: &QwenImage21Vae, rgb: Tensor) -> Result<Tensor> {
    let input = if vae.config().in_channels == 4 {
        let (_, _, h, w) = rgb.dims4()?;
        let alpha = Tensor::ones((1, 1, h, w), DType::F32, rgb.device())?;
        Tensor::cat(&[&rgb, &alpha], 1)?
    } else {
        rgb
    };
    let mode = vae.encode_mode(&input)?;
    Ok(pack_latents(&vae.normalize(&mode)?)?.detach())
}

/// Decode `path` and resize the **whole** picture (no crop) to `width × height` (Lanczos, the
/// dataset loader's filter) → RGB NCHW `[1, 3, height, width]` in `[−1, 1]` (`px/127.5 − 1`, the
/// dataset loader's normalisation) on `device`. An edit target's encode input.
fn load_image_resized(path: &Path, width: u32, height: u32, device: &Device) -> Result<Tensor> {
    let img = image::open(path)
        .map_err(|e| Error::Msg(format!("{LABEL}: open image {}: {e}", path.display())))?
        .to_rgb8();
    let resized = if img.dimensions() == (width, height) {
        img
    } else {
        image::imageops::resize(&img, width, height, image::imageops::FilterType::Lanczos3)
    };
    let (w, h) = (width as usize, height as usize);
    let mut data = vec![0f32; 3 * h * w];
    for (x, y, px) in resized.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        for c in 0..3 {
            data[c * h * w + y * w + x] = px[c] as f32 / 127.5 - 1.0;
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, h, w), &Device::Cpu)?.to_device(device)?)
}

/// Decode a reference image file as RGBA8 — upstream's `img.convert("RGBA")`: an RGB file is the
/// opaque case (`A = 255`, exactly what the render path's `Conditioning::Reference` widening
/// produces), a file with alpha keeps it (the `Conditioning::ReferenceRgba` case).
fn decode_reference(path: &Path) -> Result<RgbaImage> {
    let rgba = image::open(path)
        .map_err(|e| {
            Error::Msg(format!(
                "{LABEL}: decode reference image {}: {e}",
                path.display()
            ))
        })?
        .to_rgba8();
    let (width, height) = rgba.dimensions();
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
        .map(|path| decode_reference(path))
        .collect()
}

/// The memory-relevant budget of one prompt's joint layout ([`edit_prompt_layout`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PromptBudget {
    /// Text rows the DiT sees (every text segment).
    text_tokens: u64,
    /// Σ condition-block latent tokens (every image block but the target).
    reference_tokens: u64,
    /// Latent tokens of the single largest condition block.
    largest_reference_tokens: u64,
    /// [`prefix_score_elements`] of the layout.
    prefix_scores: u64,
    /// [`largest_prefix_call_elements`] of the layout.
    largest_prefix_call: u64,
    /// [`prefix_copy_rows`] of the layout.
    prefix_copy_rows: u64,
}

impl PromptBudget {
    fn of(layout: &JointLayout) -> Self {
        use crate::transformer::Segment as S;
        let blocks = layout.segments.len().saturating_sub(1);
        let mut budget = Self {
            text_tokens: 0,
            reference_tokens: 0,
            largest_reference_tokens: 0,
            prefix_scores: prefix_score_elements(layout),
            largest_prefix_call: largest_prefix_call_elements(layout),
            prefix_copy_rows: prefix_copy_rows(layout),
        };
        for segment in &layout.segments[..blocks] {
            match *segment {
                S::Text { len } => budget.text_tokens += len as u64,
                S::Image { height, width } => {
                    let tokens = (height * width) as u64;
                    budget.reference_tokens += tokens;
                    budget.largest_reference_tokens = budget.largest_reference_tokens.max(tokens);
                }
            }
        }
        budget
    }
}

/// The exact joint layout an edit prompt assembles at a `(width, height)` target, from the
/// tokenizer and the reference **headers** alone — what [`joint_branch`] builds from the encoded
/// conditioning, without loading a weight: the image-conditioned template's ids with each
/// `<|image_pad|>` placeholder expanded to its reference's vision slots (one per merged 2×2 latent
/// group, [`reference_fit`]'s grid), the system prefix dropped, each slot run one condition block,
/// the target block last. A test pins it to the branch the real encoder assembles. Each
/// reference's fit goes through [`reference_fit`], which also **refuses** a reference whose fit the
/// Qwen3-VL processor would rebind — the exact predicate `prepare_reference` applies — so a bad
/// reference fails here, before any weight loads, rather than after the tower is resident.
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
    let mut grids = Vec::with_capacity(reference_paths.len());
    for (index, path) in reference_paths.iter().enumerate() {
        let size = image::image_dimensions(path).map_err(|e| {
            Error::Msg(format!(
                "{LABEL}: read reference image {}: {e}",
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
                    "{LABEL}: the image-conditioned template carries more placeholders than the \
                     {} reference images",
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
            "{LABEL}: the image-conditioned template placed {next} of {} reference images in a \
             {}-token prompt (system prefix {drop})",
            grids.len(),
            expanded.len()
        )));
    }
    let mut segments = Vec::with_capacity(2 * grids.len() + 2);
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

/// Per-head score elements of a layout's block-causal **prefix** attention calls:
/// `Σ (end − start)·end` over [`JointLayout::prefix_segments`] — each segment's rows attend to every
/// key before its end (the text rows are masked causally inside that, but the scores are still
/// materialised). `L²` for the text-to-image layout.
pub fn prefix_score_elements(layout: &JointLayout) -> u64 {
    layout
        .prefix_segments()
        .iter()
        .map(|&(start, end, _)| ((end - start) * end) as u64)
        .sum()
}

/// Per-head score elements of a layout's costliest **single** prefix attention call:
/// `max (end − start)·end` over [`JointLayout::prefix_segments`]. `L²` for the text-to-image
/// layout. The attention backward runs one call at a time, so the larger of this and the target
/// call sizes its transient (sc-24163).
pub fn largest_prefix_call_elements(layout: &JointLayout) -> u64 {
    layout
        .prefix_segments()
        .iter()
        .map(|&(start, end, _)| ((end - start) * end) as u64)
        .max()
        .unwrap_or(0)
}

/// Rows a layout's prefix attention calls copy: `Σ (end − start) + 3·end` over
/// [`JointLayout::prefix_segments`]. Each call takes contiguous copies of its query rows and of the
/// keys and values before its end, and the SDPA makes one more of those keys, transposed. The
/// training graph retains all of them (sc-24163). `4·L` for the text-to-image layout.
pub fn prefix_copy_rows(layout: &JointLayout) -> u64 {
    layout
        .prefix_segments()
        .iter()
        .map(|&(start, end, _)| ((end - start) + 3 * end) as u64)
        .sum()
}

/// The size an item's **target** trains at. A captioned (text-to-image) item is the centre-cropped
/// `edge × edge` square, as before. An edit pair's target keeps its **aspect ratio** — the render
/// path's own fit, [`calculate_dimensions`]`(edge², w/h)` on the 32-px grid — because its references
/// keep theirs: cropping the target square while the references stay whole would teach the adapter
/// "zoom into the middle" and break the spatial correspondence the edit is about. Reads only the
/// image header. The MLX twin's rule, verbatim.
fn edit_target_size(item: &TrainingItem, edge: u32) -> Result<(u32, u32)> {
    if !item.is_edit_pair() {
        return Ok((edge, edge));
    }
    let (w, h) = image::image_dimensions(&item.image_path).map_err(|e| {
        Error::Msg(format!(
            "{LABEL}: read target image {}: {e}",
            item.image_path.display()
        ))
    })?;
    if w == 0 || h == 0 {
        return Err(Error::Msg(format!(
            "{LABEL}: target image {} is {w}x{h}",
            item.image_path.display()
        )));
    }
    Ok(calculate_dimensions(
        f64::from(edge) * f64::from(edge),
        f64::from(w) / f64::from(h),
    ))
}

/// Latent tokens of a `(width, height)` image.
fn target_tokens((width, height): (u32, u32)) -> u64 {
    (width / VAE_SCALE_FACTOR) as u64 * (height / VAE_SCALE_FACTOR) as u64
}

/// An item's packed target latent at [`edit_target_size`]: a captioned item's centre-cropped square
/// (unchanged), an edit pair's whole picture at its aspect-preserving fit.
fn encode_item_target(
    vae: &QwenImage21Vae,
    item: &TrainingItem,
    edge: u32,
    device: &Device,
) -> Result<Tensor> {
    if item.is_edit_pair() {
        let (width, height) = edit_target_size(item, edge)?;
        encode_rgb_latents(
            vae,
            load_image_resized(&item.image_path, width, height, device)?,
        )
    } else {
        encode_latents(vae, &item.image_path, edge, device)
    }
}

/// One prompt's [`JointBranch`] through the render path's own assembly — the tower's
/// [`QwenImage21TextEncoder::encode_conditioning`] (the image-conditioned template with vision
/// tokens when `references` is non-empty, the text-to-image template otherwise) then
/// [`joint_branch`] at the target's `(width, height)` — with the text rows cached as detached f32.
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
    Ok(JointBranch {
        text: branch.text.to_dtype(DType::F32)?.detach(),
        layout: branch.layout,
    })
}

/// One dataset item's text side, exactly as phase 1 caches it: its ordered references
/// host-preprocessed by [`prepare_conditioning_references`], then [`encode_branch`] at the item's
/// target size ([`edit_target_size`]). Returns the branch and the prepared references (the first
/// item's also condition edit previews).
fn item_branch(
    encoder: &QwenImage21TextEncoder,
    tokenizer: &TextTokenizer,
    drop: usize,
    item: &TrainingItem,
    edge: u32,
) -> Result<(JointBranch, Vec<PreparedReference>)> {
    let references = prepare_conditioning_references(encoder, &decode_references(item)?)?;
    let size = edit_target_size(item, edge)?;
    let branch = encode_branch(encoder, tokenizer, drop, &item.caption, &references, size)?;
    Ok((branch, references))
}

/// An edit item's reference latents, in reference order, through the render path's own host
/// preprocessing ([`prepare_references`]) and VAE encode ([`encode_references`]). Re-prepared from
/// disk (host work only) so the caption phase keeps no pixel buffers alive. Empty for a captioned
/// item, or when `vision` is `None` (a text-to-image run).
fn encode_item_references(
    vae: &QwenImage21Vae,
    vision: Option<&VisionConfig>,
    item: &TrainingItem,
    device: &Device,
) -> Result<Vec<Tensor>> {
    match vision {
        Some(vision) if item.is_edit_pair() => {
            let prepared = prepare_references(&decode_references(item)?, vision, device)?;
            Ok(encode_references(vae, &prepared)?
                .into_iter()
                .map(|latent| latent.detach())
                .collect())
        }
        _ => Ok(Vec::new()),
    }
}

/// One cached training example: the target latent and the joint-sequence pieces the step feeds
/// the DiT — the conditioning rows and layout of [`joint_branch`] and (edit) the ordered reference
/// latents of [`encode_references`].
struct Cached {
    /// Packed target latent `[1, h·w, C]` (f32).
    x0: Tensor,
    /// The branch's text rows `[1, text_len, hidden]` (f32).
    text: Tensor,
    /// The joint layout, target block last.
    layout: JointLayout,
    /// Packed reference latents, in reference order (empty for text-to-image).
    references: Vec<Tensor>,
}

/// `candle_core::Error` from the crate error, for the checkpoint segments' closures.
fn to_core(e: Error) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// The per-step inputs of one loss evaluation.
struct StepInputs<'a> {
    /// Packed target latent `[1, target_tokens, C]` (f32).
    x0: &'a Tensor,
    /// The conditioning text rows ([`JointBranch::text`]).
    text: &'a Tensor,
    /// The joint layout ([`JointBranch::layout`]): text and condition blocks, target last.
    layout: &'a JointLayout,
    /// Packed reference latents in reference order — empty for text-to-image.
    references: &'a [Tensor],
    noise: &'a Tensor,
    t: f32,
}

/// One micro-step's forward+backward over the trainable factors: build `x_t` at flow-match `t`,
/// predict the velocity through [`QwenImage21Transformer::forward_train_joint`] over the joint
/// sequence — the denoise loop's own image stream ([`joint_images`]: references in order, then the
/// noised target) on the step's layout, raw `t`, no sign flip — regress the **target block's**
/// velocity onto `noise − x0` in f32, and return `(loss, grads)` keyed by `vars`. The DiT returns
/// the target block only, so the condition tokens never enter the loss.
///
/// `checkpoint` selects the gradient-checkpointed backward: the retained pre-block forward
/// ([`QwenImage21Transformer::train_prelude`] — the global projections, so a global target trains
/// through ordinary autograd), every block as one segment of the shared segmented VJP
/// ([`checkpointed_backward_with_input_grad`]) carrying `[x, modulation rows, norm_out rows]` across
/// the boundaries, and the head + loss as the final segment. The recovered boundary cotangent is
/// then stitched back through the retained pre-block forward. Numerically the dense grads (the
/// `checkpointed_grads_match_dense_for_lora_and_lokr` gate).
fn compute_loss_grads(
    dit: &QwenImage21Transformer,
    vars: &[Var],
    step: &StepInputs<'_>,
    mae: bool,
    checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let StepInputs {
        x0,
        text,
        layout,
        references,
        noise,
        t,
    } = *step;
    let (x_t, target) = flow_match::build_batch(x0, noise, t as f64)?;
    let x_t = x_t.to_dtype(dit.compute_dtype())?;
    let images = joint_images(references, &x_t);
    if !checkpoint {
        let v = dit.forward_train_joint(text, &images, t, layout)?;
        let loss = velocity_loss(&v, &target, mae)?;
        let value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        return Ok((value, loss.backward()?));
    }

    let prelude = dit.train_prelude_joint(text, &images, t, layout)?;
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
/// [`DECODE_TILE_EDGE`]. An edit run previews an **edit**: the prompt branches were assembled with
/// the first dataset item's references and its cached reference latents condition the denoise,
/// exactly as a reference render does. A best-effort nicety: failures are logged by the caller,
/// never fatal.
#[allow(clippy::too_many_arguments)]
fn render_sample(
    dit: &QwenImage21Transformer,
    vae: &QwenImage21Vae,
    scheduler: &SchedulerConfig,
    pos: &JointBranch,
    neg: Option<&JointBranch>,
    reference_latents: &[Tensor],
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
        // Instruction-edit datasets: the shared floor caps references at this descriptor's
        // `max_reference_images` (the render path's own cap) and refuses mixed or hybrid
        // datasets; the snapshot must also carry the vision tower the references go through.
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        if req.items.iter().any(TrainingItem::is_edit_pair) && self.vision.is_none() {
            return Err(missing_vision_tower().into());
        }
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
        // Edit mode is a different training input, so its floor is re-checked here rather than
        // trusted to a prior `validate` call.
        gen_core::train::validate_edit_request(&self.descriptor, req)?;
        validate_request(req)?;
        let edit = req.items.iter().any(TrainingItem::is_edit_pair);
        let vision = if edit {
            Some(self.vision.clone().ok_or_else(missing_vision_tower)?)
        } else {
            None
        };
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

        // --- preflight: the derived peak against this device, before any weight is read ---
        // Edit previews condition on the first item's references (see `render_sample`). A bad
        // reference (one the Qwen3-VL processor would rebind) is refused here, from its header.
        let sample_reference_paths: &[PathBuf] = if edit {
            &req.items[0].reference_image_paths
        } else {
            &[]
        };
        let mut longest = 0u64;
        let (mut reference_tokens, mut largest_reference_tokens, mut reference_cache_tokens) =
            (0u64, 0u64, 0u64);
        let mut largest_target_tokens = 0u64;
        for item in &req.items {
            largest_target_tokens =
                largest_target_tokens.max(target_tokens(edit_target_size(item, edge)?));
        }
        let (mut prefix_scores, mut largest_prefix_call, mut prefix_copy_rows) = (0u64, 0u64, 0u64);
        let mut prompts: Vec<PreflightPrompt<'_>> = Vec::new();
        for item in &req.items {
            prompts.push((
                item.caption.as_str(),
                item.reference_image_paths.as_slice(),
                edit_target_size(item, edge)?,
                true,
            ));
        }
        // Previews render square at `edge²`.
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
                    let layout = edit_prompt_layout(
                        &self.tokenizer,
                        self.drop_count,
                        vision,
                        text,
                        reference_paths,
                        size,
                    )?;
                    let budget = PromptBudget::of(&layout);
                    reference_tokens = reference_tokens.max(budget.reference_tokens);
                    largest_reference_tokens =
                        largest_reference_tokens.max(budget.largest_reference_tokens);
                    if cached {
                        reference_cache_tokens += budget.reference_tokens;
                    }
                    prefix_scores = prefix_scores.max(budget.prefix_scores);
                    largest_prefix_call = largest_prefix_call.max(budget.largest_prefix_call);
                    prefix_copy_rows = prefix_copy_rows.max(budget.prefix_copy_rows);
                    budget.text_tokens
                }
                None => {
                    let tokens = caption_tokens(&self.tokenizer, self.drop_count, text)?;
                    prefix_scores = prefix_scores.max(tokens * tokens);
                    largest_prefix_call = largest_prefix_call.max(tokens * tokens);
                    prefix_copy_rows = prefix_copy_rows.max(4 * tokens);
                    tokens
                }
            };
            longest = longest.max(tokens);
        }
        let shape = TrainingShape {
            edge,
            target_tokens: largest_target_tokens,
            caption_tokens: longest,
            reference_tokens,
            largest_reference_tokens,
            reference_cache_tokens,
            prefix_scores,
            largest_prefix_call,
            prefix_copy_rows,
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

        // --- 1. captions: the Qwen3-VL tower, encoded ONCE, then dropped ---
        // Every prompt goes through the render path's own assembly: references preprocessed by
        // `prepare_conditioning_references`, the (image-conditioned, for edit) template encoded by
        // `encode_conditioning`, then `joint_branch` → the text rows + joint layout the DiT sees.
        // A captioned item has no references, so this is the text-to-image conditioning exactly.
        // An edit run loads the vision tower too (its references are vision context).
        on_progress(TrainingProgress::LoadingModel);
        let (branches, sample_caps, sample_neg) = {
            let encoder: QwenImage21TextEncoder = loader::load_text_encoder_from(
                &self.root.join("text_encoder"),
                &device,
                vision.as_ref(),
            )?;
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
            let mut branches: Vec<JointBranch> = Vec::with_capacity(req.items.len());
            let mut sample_references = Vec::new();
            for (i, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let (branch, references) =
                    item_branch(&encoder, &self.tokenizer, self.drop_count, item, edge)?;
                branches.push(branch);
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

        // --- 2. latents: the VAE encodes each image (and edit reference) ONCE; the encoder half
        //        is then dropped ---
        let mut vae = loader::load_vae(&self.root, &device)?;
        let total = req.items.len() as u32;
        let mut cache: Vec<Cached> = Vec::with_capacity(req.items.len());
        for (i, (item, branch)) in req.items.iter().zip(branches).enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let x0 = encode_item_target(&vae, item, edge, &device)?;
            let references = encode_item_references(&vae, vision.as_ref(), item, &device)?;
            cache.push(Cached {
                x0,
                text: branch.text,
                layout: branch.layout,
                references,
            });
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
        // Provenance + licence on every saved adapter; an edit adapter is also marked as one.
        let meta = provenance_meta(edit);
        let mae = flow_match::is_mae(cfg);

        // --- resume: continue from the latest snapshot of THIS adapter in output_dir, if any ---
        // `load_resume` refuses a snapshot whose factor surface (rank, network type, targets),
        // training config or dataset fingerprint differs from this run's.
        let mut start_step = 0u32;
        let mut update_idx = 0u32;
        if cfg.resume {
            if let Some((snapshot, _)) = find_latest_resume(&req.output_dir, &stem) {
                check_resume_training_mode(&req.output_dir, &stem, &snapshot, edit)?;
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
                &StepInputs {
                    x0: &sample.x0,
                    text: &sample.text,
                    layout: &sample.layout,
                    references: &sample.references,
                    noise: &noise,
                    t,
                },
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
                // Edit previews condition on the first item's cached reference latents.
                let sample_reference_latents: &[Tensor] =
                    if edit { &cache[0].references } else { &[] };
                for (i, (prompt, pos)) in sample_caps.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    match render_sample(
                        &dit,
                        vae,
                        &self.scheduler,
                        pos,
                        sample_neg.as_ref(),
                        sample_reference_latents,
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
        assert_eq!(d.max_reference_images as usize, MAX_REFERENCE_IMAGES);
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

    fn edit_req(count: usize) -> TrainingRequest {
        let mut req = req_with(base_config());
        req.items = vec![TrainingItem::edit_pair(
            PathBuf::from("/nonexistent/target.png"),
            "make the swatch blue".into(),
            (0..count)
                .map(|i| PathBuf::from(format!("/nonexistent/ref{i}.png")))
                .collect(),
        )];
        req
    }

    /// AC (sc-24162): the reference cap is the render path's own — the constant
    /// `validate_reference_count` enforces — and the shared floor refuses one more, refuses mixed
    /// datasets, and both `validate` and `train` surface it (nothing is read for the refusal).
    ///
    /// *Mutation that reds this:* `max_reference_images` left at 0 (every edit dataset becomes a
    /// typed `Unsupported`) or hard-coded to a number other than the render path's cap.
    #[test]
    fn validate_caps_edit_references_at_the_render_paths_limit() {
        assert_eq!(
            trainer_descriptor().max_reference_images as usize,
            MAX_REFERENCE_IMAGES
        );
        assert!(crate::reference::validate_reference_count(MAX_REFERENCE_IMAGES).is_ok());
        assert!(crate::reference::validate_reference_count(MAX_REFERENCE_IMAGES + 1).is_err());
        let mut t = trainer();
        assert!(t.validate(&edit_req(1)).is_ok());
        assert!(t.validate(&edit_req(MAX_REFERENCE_IMAGES)).is_ok());
        let over = edit_req(MAX_REFERENCE_IMAGES + 1);
        for err in [
            t.validate(&over).unwrap_err(),
            t.train(&over, &mut |_| {}).unwrap_err(),
        ] {
            assert!(err.to_string().contains("at most 10"), "{err}");
        }

        let mut mixed = edit_req(1);
        mixed.items.push(TrainingItem::captioned(
            PathBuf::from("/nonexistent/x.png"),
            "a swatch".into(),
        ));
        let err = t.validate(&mixed).unwrap_err().to_string();
        assert!(err.contains("item 1 has none"), "{err}");
    }

    /// An edit dataset on a snapshot that ships no Qwen3-VL vision tower is the render path's own
    /// typed refusal, at `validate` — never a run that would condition on nothing.
    #[test]
    fn an_edit_dataset_without_a_vision_tower_is_unsupported() {
        let dir = scratch("novision");
        let root = dir.path().join("snapshot");
        copy_dir(&tiny_snapshot(), &root);
        let config_path = root.join("text_encoder/config.json");
        let mut config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        config.as_object_mut().unwrap().remove("vision_config");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
        let t = QwenImage21Trainer::load_on(&LoadSpec::new(WeightsSource::Dir(root)), Device::Cpu)
            .unwrap();
        assert!(
            t.validate(&req_with(base_config())).is_ok(),
            "T2I still fine"
        );
        let err = t.validate(&edit_req(1)).unwrap_err();
        assert!(matches!(err, gen_core::Error::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains("vision tower"), "{err}");
    }

    /// `load` refuses a control / extra-control / IP-adapter / identity overlay as a typed
    /// `Unsupported` — gen-core's one refusal and message, shared with the MLX twin (sc-24163).
    ///
    /// *Mutation that reds this:* dropping the `refuse_trainer_load_overlays` call from `load_on`.
    #[test]
    fn load_refuses_control_ip_adapter_and_identity_overlays() {
        let dense = LoadSpec::new(WeightsSource::Dir(tiny_snapshot()));
        let other = || WeightsSource::Dir(PathBuf::from("/nonexistent/overlay"));
        let mut identity = dense.clone();
        identity.identity = Some(Default::default());
        for spec in [
            dense.clone().with_control(other()),
            dense.clone().with_extra_control(other()),
            dense.clone().with_ip_adapter(other()),
            identity,
        ] {
            match QwenImage21Trainer::load_on(&spec, Device::Cpu).err() {
                Some(Error::Unsupported(message)) => assert_eq!(
                    message,
                    "qwen_image_2_1 trainer: control / IP-adapter / identity overlays are not \
                     part of text-to-image LoRA/LoKr training"
                ),
                other => panic!("an overlay must be a typed Unsupported, got {other:?}"),
            }
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
                    .insert("references".into(), serde_json::json!(["/r.png"]));
            },
            "model_options `references`",
        );
        bad(
            &|r| {
                r.items[0]
                    .model_options
                    .insert("controlType".into(), serde_json::json!("canny"));
            },
            "model_options `controlType`",
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

    /// The SceneWorks worker forwards its whole `advanced` map as `model_options` on every run;
    /// keys this trainer does not own are ignored (the "parse only your keys" worker contract), and
    /// the reference/control keys it does refuse are sent in their *off* state (null, an empty list
    /// or string, `false`, `"none"`), which selects nothing — so a defaults-shaped map, on the
    /// config and on the items, passes `validate`. The same keys turned *on* are still refused.
    ///
    /// *Mutation that reds this:* `gen_core::train::model_option_selects_something` treating
    /// any non-null value as present (the
    /// original rule — `[]`, `""`, `false` and `"none"` then refuse a defaults-shaped run).
    #[test]
    fn a_sceneworks_shaped_advanced_map_passes_validate() {
        let t = trainer();
        let advanced = serde_json::json!({
            "mixedPrecision": "bf16",
            "cacheLatents": true,
            "networkType": "lora",
            "sampleEvery": 250,
            "samplePrompts": ["a swatch"],
            "gradientCheckpointing": true,
            "optimizer": "adamw",
            "lrScheduler": "constant",
            "timestepType": "sigmoid",
            "captionDropout": 0.05,
            "controlType": null,
            "control_type": "none",
            "references": [],
            "referenceImages": "",
            "reference_images": "  ",
            "referenceImagePaths": {},
            "controlImage": false,
            "control_image": "None",
        });
        let mut req = req_with(base_config());
        req.config.model_options = advanced.as_object().unwrap().clone();
        req.items[0].model_options = advanced.as_object().unwrap().clone();
        t.validate(&req)
            .expect("unknown advanced keys and off-state refused keys are ignored, not refused");

        for (key, on) in [
            ("references", serde_json::json!(["/r.png"])),
            ("referenceImages", serde_json::json!("/r.png")),
            ("controlImage", serde_json::json!(true)),
            ("controlType", serde_json::json!("canny")),
            ("referenceImagePaths", serde_json::json!({ "0": "/r.png" })),
        ] {
            let mut on_req = req.clone();
            on_req.items[0].model_options.insert(key.into(), on);
            let err = t.validate(&on_req).unwrap_err().to_string();
            assert!(err.contains(&format!("model_options `{key}`")), "{err}");
        }
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
            vision_tower_bytes: 0,
            vision_hidden: 1152,
            vision_heads: 16,
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
            target_tokens: 0,
            caption_tokens: 64,
            reference_tokens: 0,
            largest_reference_tokens: 0,
            reference_cache_tokens: 0,
            prefix_scores: 0,
            largest_prefix_call: 0,
            prefix_copy_rows: 0,
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

    /// `Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758` as
    /// [`FootprintFacts::from_snapshot`] reads it at bf16 (component width 2), captured from the
    /// release snapshot for sc-24163.
    fn release_snapshot_facts() -> FootprintFacts {
        FootprintFacts {
            dit_elements: 7_115_124_736,
            text_encoder_bytes: 15_136_811_008,
            vae_encoder_bytes: 157_384_384,
            vae_decoder_bytes: 518_096_424,
            num_layers: 32,
            inner: 4096,
            heads: 32,
            mlp_ratio: 3,
            latent_channels: 64,
            text_hidden: 4096,
            text_heads: 32,
            vae_encode_channels: 96,
            vae_decode_channels: 144,
            pixels_per_token: 16,
            vision_tower_bytes: 1_152_776_672,
            vision_hidden: 1152,
            vision_heads: 16,
        }
    }

    /// The trainer's preflight shape for one sc-24163 measurement cell (rank 16, adamw8bit, previews
    /// on, gradient checkpointing on, bf16), as the trainer derived it from that cell's dataset.
    fn measured_cell(
        network: NetworkType,
        edge: u32,
        edit: Option<(u64, (u64, u64), u64)>,
    ) -> TrainingShape {
        let side = (edge / 16) as u64;
        let adapter = production_adapter(network);
        let base = TrainingShape {
            edge,
            target_tokens: side * side,
            caption_tokens: 25,
            reference_tokens: 0,
            largest_reference_tokens: 0,
            reference_cache_tokens: 0,
            prefix_scores: 625,
            largest_prefix_call: 625,
            prefix_copy_rows: 100,
            items: 8,
            compute_width: 2,
            adapter,
            optimizer_state_per_param: 2,
            checkpointed: true,
            sampling: true,
        };
        match edit {
            None => base,
            // 6 instruction pairs; `refs` reference latents per prompt (4096 each), its prefix calls.
            Some((refs, (prefix_scores, largest_prefix_call), prefix_copy_rows)) => TrainingShape {
                caption_tokens: if refs == 1 { 37 } else { 43 },
                reference_tokens: refs * 4096,
                largest_reference_tokens: 4096,
                reference_cache_tokens: 6 * refs * 4096,
                prefix_scores,
                largest_prefix_call,
                prefix_copy_rows,
                items: 6,
                ..base
            },
        }
    }

    /// sc-24163 (E13/E11): the checkpointed training step is priced at its **measured** peak, never
    /// below it.
    ///
    /// Each cell is the A1 request run through this trainer for 10 steps on CUDA (RTX PRO 6000,
    /// driver 596.36, one process per cell, budget disabled). The measured value is the driver's
    /// live high-water `CU_MEMPOOL_ATTR_USED_MEM_HIGH` over the whole run, in bytes. Source:
    /// SceneWorks `docs/calibration/sc-24163/cuda-measurements.json` (`a5.trainingFootprint`) on
    /// `feature/sc-24107-qwen-image-2-1-lora`. Before this fix the preflight derived 22.4 / 21.8 /
    /// 35.1 / 51.1 / 51.4 GiB, 26–45 % under, and admitted a 1024² run it could not hold.
    ///
    /// *Mutations that red this:* `BACKWARD_SCORE_F32_TENSORS` back to 2, or the score backward
    /// sized over every call; `BACKWARD_RETAINED_HIDDEN` or `DEAD_WEIGHT_GRAD_COPIES` at 0; the q/k/v
    /// copies or the pending q/k/v cotangents dropped (the edit cells go under); the backward over
    /// every call's elements (the edit cells go over 1.15).
    #[test]
    fn the_checkpointed_step_covers_the_measured_cuda_peaks() {
        let facts = release_snapshot_facts();
        // The runs' own adapter sizes (rank 16 over every block projection).
        assert_eq!(
            production_adapter(NetworkType::Lora).trainable_params,
            41_943_040
        );
        let lokr = production_adapter(NetworkType::Lokr);
        assert_eq!(
            (lokr.trainable_params, lokr.lokr_block_per_token),
            (1_671_168, 38_912)
        );
        // Text 8 / reference 64×64 / text 29 / target: Σ (end−start)·end, the reference call, copies.
        let one_ref = Some((1, (16_929_905, 16_809_984), 28_868));
        let cells = [
            (
                "t2i lora 768²",
                measured_cell(NetworkType::Lora, 768, None),
                30_361_842_468u64,
            ),
            (
                "t2i lokr 768²",
                measured_cell(NetworkType::Lokr, 768, None),
                29_850_168_644,
            ),
            (
                "t2i lora 1024²",
                measured_cell(NetworkType::Lora, 1024, None),
                54_794_203_428,
            ),
            (
                "edit lokr 768² 1 ref",
                measured_cell(NetworkType::Lokr, 768, one_ref),
                69_364_708_316,
            ),
            (
                "edit lora 768² 1 ref",
                measured_cell(NetworkType::Lora, 768, one_ref),
                69_641_733_180,
            ),
        ];
        for (label, shape, measured) in cells {
            let fp = training_footprint(&facts, &shape);
            let predicted = fp.peak();
            eprintln!(
                "[sc-24163] {label}: predicted {:.2} GiB (step stage), measured {:.2} GiB, ×{:.3}",
                gib(predicted),
                gib(measured),
                predicted as f64 / measured as f64
            );
            assert_eq!(
                predicted, fp.train_phase,
                "{label}: the step is the peak stage"
            );
            assert!(
                predicted >= measured,
                "{label}: {predicted} under the measured {measured}"
            );
            assert!(
                predicted as f64 <= measured as f64 * 1.15,
                "{label}: {predicted} more than 15 % over the measured {measured}"
            );
        }
        // Two references per edit at 768² filled the 97,295 MiB card and spilled (A1), so its need
        // is above that, and the refusal it got on a 96 GB card must stand.
        let two_refs = measured_cell(
            NetworkType::Lokr,
            768,
            Some((2, (50_685_299, 33_611_776), 82_224)),
        );
        let predicted = training_footprint(&facts, &two_refs).peak();
        assert!(predicted > 97_295 << 20, "two-reference edit: {predicted}");
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

    /// The text-to-image layout of [`fixed_batch`]: 5 text rows, a 4×4 target.
    fn t2i_layout() -> JointLayout {
        JointLayout::text_to_image(5, 4, 4)
    }

    /// A fixed synthetic **edit** batch: two references (2×2 and 2×4 latent grids) interleaved with
    /// text exactly as `joint_layout` interleaves an image-conditioned prompt, and the 4×4 target
    /// last — the MLX twin's fixture.
    struct EditBatch {
        x0: Tensor,
        text: Tensor,
        noise: Tensor,
        references: Vec<Tensor>,
        layout: JointLayout,
    }

    impl EditBatch {
        fn step(&self, t: f32) -> StepInputs<'_> {
            StepInputs {
                x0: &self.x0,
                text: &self.text,
                layout: &self.layout,
                references: &self.references,
                noise: &self.noise,
                t,
            }
        }
    }

    fn fixed_edit_batch(dit: &QwenImage21Transformer) -> EditBatch {
        use crate::transformer::Segment as S;
        let c = dit.config();
        let channels = c.in_channels;
        let layout = JointLayout {
            segments: vec![
                S::Text { len: 3 },
                S::Image {
                    height: 2,
                    width: 2,
                },
                S::Text { len: 1 },
                S::Image {
                    height: 2,
                    width: 4,
                },
                S::Text { len: 4 },
                S::Image {
                    height: 4,
                    width: 4,
                },
            ],
        };
        EditBatch {
            x0: randn(&[1, 16, channels], 1),
            text: randn(&[1, 8, c.context_in_dim], 2),
            noise: randn(&[1, 16, channels], 3),
            references: vec![randn(&[1, 4, channels], 4), randn(&[1, 8, channels], 5)],
            layout,
        }
    }

    /// [`compute_loss_grads`] on a text-to-image sample (no references, [`t2i_layout`]).
    #[allow(clippy::too_many_arguments)]
    fn t2i_loss_grads(
        dit: &QwenImage21Transformer,
        vars: &[Var],
        x0: &Tensor,
        text: &Tensor,
        noise: &Tensor,
        t: f32,
        mae: bool,
        checkpoint: bool,
    ) -> Result<(f32, GradStore)> {
        let layout = t2i_layout();
        compute_loss_grads(
            dit,
            vars,
            &StepInputs {
                x0,
                text,
                layout: &layout,
                references: &[],
                noise,
                t,
            },
            mae,
            checkpoint,
        )
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
                let (loss, mut grads) =
                    t2i_loss_grads(&dit, &set.vars, &x0, &ctx, &noise, 0.5, false, checkpoint)
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
                    t2i_loss_grads(&dit, &set.vars, &x0, &ctx, &noise, 0.4, false, checkpoint)
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
            let edit = fixed_edit_batch(&dit);
            for checkpoint in [false, true] {
                // Text-to-image, then an edit step (f32 cached reference latents into the half-
                // precision joint forward).
                let steps = [
                    t2i_loss_grads(&dit, &set.vars, &x0, &ctx, &noise, 0.3, true, checkpoint),
                    compute_loss_grads(&dit, &set.vars, &edit.step(0.3), true, checkpoint),
                ];
                for step in steps {
                    let (loss, grads) = step.unwrap();
                    assert!(loss.is_finite(), "{network:?}/{checkpoint}: {loss}");
                    for v in &set.vars {
                        let g = grads.get(v.as_tensor()).expect("gradient");
                        assert_eq!(g.dtype(), DType::F32, "master-weight grads stay f32");
                    }
                }
            }
            set_frozen(&mut dit, true).unwrap();
            let t2i = JointBranch {
                text: ctx.clone(),
                layout: t2i_layout(),
            };
            let edit_branch = JointBranch {
                text: edit.text.clone(),
                layout: edit.layout.clone(),
            };
            for (branch, references) in [(&t2i, &[][..]), (&edit_branch, &edit.references[..])] {
                let image = render_sample(
                    &dit,
                    &vae,
                    &scheduler,
                    branch,
                    Some(branch),
                    references,
                    3,
                    64,
                    2,
                    2.0,
                    &CancelFlag::new(),
                )
                .unwrap();
                assert_eq!((image.width, image.height), (64, 64));
            }
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

            // The multi-segment EDIT layout (text / condition / text / condition / text / target):
            // block-causal prefix segments and the shared-`t = 0` condition rows go through the
            // same equivalence — render `forward_joint` vs `forward_train_joint`, and the
            // `train_prelude_joint` → blocks → head pieces vs the training forward.
            let edit = fixed_edit_batch(&dit);
            let images = joint_images(&edit.references, &edit.x0);
            let render = dit
                .forward_joint(&edit.text, &images, 0.6, &edit.layout)
                .unwrap();
            let train = dit
                .forward_train_joint(&edit.text, &images, 0.6, &edit.layout)
                .unwrap();
            let prelude = dit
                .train_prelude_joint(&edit.text, &images, 0.6, &edit.layout)
                .unwrap();
            let mut x = prelude.x.clone();
            for index in 0..dit.num_blocks() {
                x = dit
                    .train_block(index, &x, &prelude.modulation, &prelude.geometry)
                    .unwrap();
            }
            let pieces = dit
                .train_head(&x, &prelude.out_rows, &prelude.geometry)
                .unwrap();
            assert_eq!(render.dims(), &[1, 16, dit.config().out_channels]);
            let (d_train, d_pieces) = (max_diff(&render, &train), max_diff(&train, &pieces));
            eprintln!(
                "[sc-24162] edit layout adapted={adapted}: render vs train {d_train:.2e}, pieces \
                 {d_pieces:.2e}"
            );
            assert!(d_train < 1e-4, "edit adapted={adapted}: {d_train}");
            assert_eq!(d_pieces, 0.0, "edit pieces are the training forward");
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

    /// The exact key → shape layout the MLX trainer writes for the DEFAULT target surface of `cfg`
    /// on the tiny DiT (bare dotted paths; LoRA `lora_A.weight [r,in]` / `lora_B.weight [out,r]` /
    /// `alpha [1]`; LoKr `lokr_w1` and `lokr_w2` or `lokr_w2_a`/`lokr_w2_b` by PEFT's `use_w2` rule)
    /// — spelled here independently of the candle installer AND of candle's target resolution: the
    /// paths are the literal per-block Linear names × `num_layers`, the shapes the block's literal
    /// `[out, in]` from the config's widths (`inner = heads·head_dim`, `hidden = inner·mlp_ratio`).
    fn mlx_layout(cfg: &TrainingConfig) -> std::collections::BTreeMap<String, Vec<usize>> {
        assert!(
            cfg.lora_target_modules.is_empty(),
            "mlx_layout spells the default (every block Linear) surface only"
        );
        let dit_cfg =
            TransformerConfig::from_json_file(&tiny_snapshot().join("transformer/config.json"))
                .unwrap();
        let inner = dit_cfg.num_attention_heads * dit_cfg.attention_head_dim;
        let hidden = inner * dit_cfg.mlp_ratio;
        let block_linears = [
            ("attn.to_q", (inner, inner)),
            ("attn.to_k", (inner, inner)),
            ("attn.to_v", (inner, inner)),
            ("attn.to_out.0", (inner, inner)),
            ("img_mlp.gate_layer", (hidden, inner)),
            ("img_mlp.proj", (hidden, inner)),
            ("img_mlp.out", (inner, hidden)),
        ];
        let targets = (0..dit_cfg.num_layers).flat_map(|layer| {
            block_linears
                .iter()
                .map(move |(name, shape)| (format!("transformer_blocks.{layer}.{name}"), *shape))
        });
        let r = cfg.rank as usize;
        let mut out = std::collections::BTreeMap::new();
        for (path, (o, i)) in targets {
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
                // alpha ≠ rank, so a reload that dropped or inverted the `alpha/rank` scale shows.
                alpha: 8.0,
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
            assert_eq!(meta["alpha"], "8");
            if network == NetworkType::Lokr {
                assert_eq!(meta["decomposeFactor"], "-1");
            }
            for (k, v) in ADAPTER_PROVENANCE {
                assert_eq!(meta[k], v, "{k}");
            }
            assert!(
                !meta.contains_key(TRAINING_MODE_KEY),
                "a text-to-image adapter carries no training-mode stamp"
            );

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

    fn max_abs(t: &Tensor) -> f32 {
        t.abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// Review (PR #1146): a saved adapter reloaded through the inference host computes **the
    /// trainer-side function** — not merely "a different velocity". The trainer's own (frozen)
    /// adapted DiT and a fresh DiT with the saved file installed by `crate::adapters::install` agree
    /// to ~1e-5 on the text-to-image AND an edit layout, at `alpha ≠ rank` (rank 4 / alpha 8, and
    /// LoKr rank 2 / alpha 8), for LoRA and LoKr with a low-rank AND a full `w2`.
    ///
    /// *Mutations that red this:* the reload dropping the `alpha/rank` scale (or using
    /// `rank/alpha`), the LoKr reload transposing `w2`, or `save_adapter` writing a stale alpha.
    #[test]
    fn a_saved_adapter_reloads_to_the_trainer_side_forward() {
        for (network, rank, kind) in [
            (NetworkType::Lora, 4, AdapterKind::Lora),
            (NetworkType::Lokr, 2, AdapterKind::Lokr),
            (NetworkType::Lokr, 4, AdapterKind::Lokr),
        ] {
            let cfg = TrainingConfig {
                rank,
                alpha: 8.0,
                network_type: network,
                ..base_config()
            };
            let mut trained = dit_at(DType::F32);
            let paths: Vec<String> = resolve_targets(trained.config(), &cfg)
                .into_iter()
                .map(|(p, _)| p)
                .collect();
            let set = install_adapters(&mut trained, &paths, &cfg, &Device::Cpu).unwrap();
            perturb(&set);
            set_frozen(&mut trained, true).unwrap();

            let dir = scratch("reload_eq");
            let path = dir.path().join("adapter.safetensors");
            save_adapter(&set, &provenance_meta(false), &path).unwrap();
            let saved = tensors(&path);
            let meta = metadata(&path);
            assert_eq!(meta["alpha"], "8");
            assert_eq!(meta["rank"], rank.to_string());
            if network == NetworkType::Lokr {
                let (full, low) = (
                    saved.keys().any(|k| k.ends_with(".lokr_w2")),
                    saved.keys().any(|k| k.ends_with(".lokr_w2_a")),
                );
                assert!(
                    if rank == 4 {
                        full && !low
                    } else {
                        low && !full
                    },
                    "rank {rank}: full w2 {full}, low-rank w2 {low}"
                );
            }

            let base = dit_at(DType::F32);
            let mut reloaded = loader::load_transformer(&tiny_snapshot(), &Device::Cpu).unwrap();
            crate::adapters::install(
                &mut reloaded,
                &[AdapterSpec::new(path.clone(), 1.0, kind)],
                Tier::Bf16,
                &Device::Cpu,
            )
            .unwrap();

            let (x0, ctx, _) = fixed_batch(&base);
            let edit = fixed_edit_batch(&base);
            let edit_images: Vec<&Tensor> = joint_images(&edit.references, &edit.x0);
            let forwards = |dit: &QwenImage21Transformer| {
                [
                    dit.forward(&x0, &ctx, 0.5, 4, 4).unwrap(),
                    dit.forward_joint(&edit.text, &edit_images, 0.5, &edit.layout)
                        .unwrap(),
                ]
            };
            let (want, got, bare) = (forwards(&trained), forwards(&reloaded), forwards(&base));
            for (i, ((want, got), bare)) in want.iter().zip(&got).zip(&bare).enumerate() {
                let moved = max_abs(&(want - bare).unwrap());
                let diff = max_abs(&(want - got).unwrap());
                let peak = max_abs(want).max(1.0);
                eprintln!(
                    "[sc-24162] {network:?} r{rank} a8 layout {i}: adapter moves {moved:.3e}, \
                     reload vs trainer {diff:.3e}"
                );
                assert!(moved > 1e-4, "{network:?} r{rank}: the adapter must matter");
                assert!(
                    diff <= 1e-5 * peak,
                    "{network:?} r{rank} layout {i}: reloaded forward differs from the trainer's \
                     by {diff:.3e}"
                );
            }
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
            req.output_dir = interrupted_dir.clone();
            assert!(trainer().train(&req, &mut |_| {}).is_err());

            // A checkpoint of the same adapter stamped by an instruction-edit run blocks a
            // text-to-image resume, even with otherwise matching settings.
            let mut dit = dit_at(DType::F32);
            let set = install(&mut dit, NetworkType::Lora, 4, vec![]);
            let meta = provenance_meta(true);
            assert_eq!(meta[TRAINING_MODE_KEY], "edit");
            save_adapter(
                &set,
                &meta,
                &interrupted_dir.join(checkpoint_filename("adapter", 1)),
            )
            .unwrap();
            let mut req = request(
                dir.path(),
                dataset(dir.path(), 2),
                TrainingConfig {
                    save_every: 2,
                    resume: true,
                    ..cfg.clone()
                },
            );
            req.output_dir = interrupted_dir;
            let err = trainer().train(&req, &mut |_| {}).unwrap_err().to_string();
            assert!(err.contains("`edit` training run"), "{err}");
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

    /// Review (PR #1146): encoding after the trainer dropped the VAE encoder is an actionable
    /// error whose text reads as one sentence (it once carried a run of embedded spaces from a
    /// missing `\` line continuation).
    #[test]
    fn a_dropped_vae_encoder_refuses_with_a_clean_message() {
        let mut vae = loader::load_vae(&tiny_snapshot(), &Device::Cpu).unwrap();
        vae.drop_encoder();
        let image = Tensor::zeros((1, 4, 32, 32), DType::F32, &Device::Cpu).unwrap();
        let err = vae.encode_mode(&image).unwrap_err().to_string();
        assert!(
            err.contains("the trainer frees it once its dataset latents are cached"),
            "{err}"
        );
        assert!(!err.contains("  "), "no embedded whitespace runs: {err:?}");
    }

    // ── instruction-edit training (sc-24162) ────────────────────────────────────────────────────

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

    /// The RGB8 [`Image`] of a PNG on disk — what a render request carries for a `Reference`.
    fn rgb_image(path: &Path) -> Image {
        let img = image::open(path).unwrap().to_rgb8();
        Image {
            width: img.width(),
            height: img.height(),
            pixels: img.into_raw(),
        }
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn assert_bit_equal(what: &str, got: &Tensor, want: &Tensor) {
        assert_eq!(got.dims(), want.dims(), "{what}: shape");
        let (g, w) = (host(got), host(want));
        assert!(
            g.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{what}: the trainer's tensor differs from the render path's"
        );
    }

    /// AC (E11): the edit trainer assembles **the same joint sequence as the render path**. One
    /// edit item (a square RGB, a 4:1 RGB and a transparent RGBA reference with a non-constant
    /// alpha, in that order) goes through the trainer's own caching functions (`item_branch`,
    /// `encode_item_references` — the exact calls phases 1/2 make), and the same pictures + prompt
    /// go through the code the render path RUNS (`crate::assemble_reference_branches`, then
    /// `encode_references`), not a re-implementation of it. Layout, positional ids (hence RoPE
    /// offsets), text rows and every reference latent must be bit-identical, the RGBA reference's
    /// alpha must survive to its latent, and the layout must carry the references as distinct blocks
    /// in dataset order.
    ///
    /// *Mutations that red this:* `decode_reference` using `to_rgb8` (drops the alpha → the RGBA
    /// reference latent differs); `decode_references` reversing/sorting the paths (layout and
    /// latents differ); `item_branch` assembling at a different size than the render request.
    #[test]
    fn edit_training_assembles_the_render_paths_joint_sequence() {
        use crate::transformer::Segment as S;
        use candle_gen::gen_core::{Conditioning, GenerationRequest};

        let root = tiny_snapshot();
        let tmp = scratch("joint");
        let square = write_png(tmp.path(), "square.png", 64, 64, 0);
        let wide = write_png(tmp.path(), "wide.png", 256, 64, 90);
        let (layer, layer_rgba) = write_rgba_png(tmp.path(), "layer.png", 64, 64);
        assert!(!layer_rgba.is_opaque(), "the fixture must carry real alpha");
        let target = write_png(tmp.path(), "target.png", 64, 64, 33);
        let prompt = "put the second image's colours onto the first, under the third";
        let edge = 64u32;
        let dev = Device::Cpu;

        let tokenizer = loader::load_tokenizer(&root).unwrap();
        let drop = system_prompt_drop_count(&tokenizer).unwrap();
        let encoder = loader::load_text_encoder(&root, &dev).unwrap();
        let vision = loader::load_vision_config(&root).unwrap().unwrap();
        let vae = loader::load_vae(&root, &dev).unwrap();

        // The trainer's path, from the dataset item.
        let item = TrainingItem::edit_pair(
            target.clone(),
            prompt.into(),
            vec![square.clone(), wide.clone(), layer.clone()],
        );
        let (trained, _) = item_branch(&encoder, &tokenizer, drop, &item, edge).unwrap();
        let trained_refs = encode_item_references(&vae, Some(&vision), &item, &dev).unwrap();

        // The render path's own assembly, from a request carrying the same pictures.
        let req = GenerationRequest {
            prompt: prompt.into(),
            width: edge,
            height: edge,
            conditioning: vec![
                Conditioning::Reference {
                    image: rgb_image(&square),
                    strength: None,
                },
                Conditioning::Reference {
                    image: rgb_image(&wide),
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
            crate::assemble_reference_branches(&encoder, &tokenizer, &req, drop, false).unwrap();
        assert!(rendered.neg.is_none());
        let rendered_refs = encode_references(&vae, &rendered.references).unwrap();

        assert_eq!(trained.layout, rendered.pos.layout, "joint layout");
        // The preflight's header-only layout is the one the encoder really assembles, so the
        // memory it prices is this sequence's.
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
        let mut flattened = decode_reference(&layer).unwrap();
        for px in flattened.pixels.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let flat_latent = encode_references(
            &vae,
            &prepare_references(&[flattened], &vision, &dev).unwrap(),
        )
        .unwrap();
        assert_ne!(
            host(&flat_latent[0]),
            host(&trained_refs[2]),
            "the transparent reference's alpha must reach the VAE encode"
        );

        // The layout: three distinct reference blocks in dataset order (4×4, the 4:1 2×8, 4×4)
        // and the 4×4 target last.
        let blocks: Vec<(usize, usize)> = trained
            .layout
            .segments
            .iter()
            .filter_map(|s| match *s {
                S::Image { height, width } => Some((height, width)),
                S::Text { .. } => None,
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

    /// An edit target keeps its aspect ratio — a 2:1 target trains as a 2:1 target block, not a
    /// centre-cropped square — so it stays spatially aligned with its (whole) references.
    /// Text-to-image items are unchanged (square).
    ///
    /// *Mutation that reds this:* `edit_target_size` returning `(edge, edge)` for edit pairs, or
    /// `encode_item_target` centre-cropping an edit target.
    #[test]
    fn a_wide_edit_target_trains_as_a_wide_target_block() {
        use crate::transformer::Segment as S;
        let root = tiny_snapshot();
        let tmp = scratch("wide");
        let target = write_png(tmp.path(), "wide_target.png", 256, 128, 5);
        let reference = write_png(tmp.path(), "ref.png", 64, 64, 9);
        let edge = 128u32;
        let dev = Device::Cpu;
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
        let encoder = loader::load_text_encoder(&root, &dev).unwrap();
        let (branch, _) = item_branch(&encoder, &tokenizer, drop, &item, edge).unwrap();
        let vision = loader::load_vision_config(&root).unwrap().unwrap();
        assert_eq!(
            edit_prompt_layout(
                &tokenizer,
                drop,
                &vision,
                "widen it",
                &item.reference_image_paths,
                size
            )
            .unwrap(),
            branch.layout,
            "the preflight prices the wide target's real layout"
        );
        let h = 96 / 16;
        assert_eq!(
            branch.layout.segments.last(),
            Some(&S::Image {
                height: h,
                width: 2 * h
            }),
            "the target block is {h}×{}",
            2 * h
        );

        let vae = loader::load_vae(&root, &dev).unwrap();
        let x0 = encode_item_target(&vae, &item, edge, &dev).unwrap();
        assert_eq!(
            x0.dim(1).unwrap(),
            h * 2 * h,
            "the target latent covers the whole 2:1 picture"
        );
        let square = encode_item_target(&vae, &captioned, edge, &dev).unwrap();
        assert_eq!(square.dim(1).unwrap(), (128 / 16) * (128 / 16));
    }

    /// A reference the Qwen3-VL processor would rebind (its fit leaves the pixel budget) is refused
    /// by the header-only preflight with the render path's own `Unsupported` — before the text
    /// encoder or vision tower load — and `prepare_reference` refuses the same picture identically.
    ///
    /// *Mutation that reds this:* `edit_prompt_tokens` using `reference_target_size` without the
    /// `smart_resize` check (the refusal then only comes from `prepare_reference`, after
    /// `LoadingModel`).
    #[test]
    fn a_rebinding_reference_is_refused_before_any_weight_loads() {
        let tmp = scratch("rebind");
        // 16:1 → the tiny 64-px fit is 256×32 = 8192 px, over the tiny processor's pixel cap.
        let thin = write_png(tmp.path(), "thin.png", 512, 32, 1);
        let target = write_png(tmp.path(), "target.png", 64, 64, 2);
        let vision = loader::load_vision_config(&tiny_snapshot())
            .unwrap()
            .unwrap();
        let tokenizer = loader::load_tokenizer(&tiny_snapshot()).unwrap();
        let drop = system_prompt_drop_count(&tokenizer).unwrap();
        let header_refusal = match edit_prompt_layout(
            &tokenizer,
            drop,
            &vision,
            "edit",
            std::slice::from_ref(&thin),
            (64, 64),
        ) {
            Err(Error::Unsupported(message)) => message,
            other => panic!("expected the render path's Unsupported, got {other:?}"),
        };
        assert!(header_refusal.contains("smart_resize"), "{header_refusal}");
        match crate::reference::prepare_reference(
            &decode_reference(&thin).unwrap(),
            0,
            &vision,
            &Device::Cpu,
        ) {
            Err(Error::Unsupported(message)) => assert_eq!(message, header_refusal),
            other => panic!("prepare_reference must refuse the same picture, got {other:?}"),
        }

        let mut t = trainer();
        let out = scratch("rebind_out");
        let mut req = req_with(TrainingConfig {
            resolution: 64,
            steps: 2,
            ..base_config()
        });
        req.items = vec![TrainingItem::edit_pair(target, "edit".into(), vec![thin])];
        req.output_dir = out.path().to_path_buf();
        let mut events = Vec::new();
        let err = t
            .train_impl(&req, &mut |p| events.push(format!("{p:?}")))
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
        assert!(
            !events.iter().any(|e| e.starts_with("LoadingModel")),
            "the refusal must precede every load: {events:?}"
        );
        assert_eq!(std::fs::read_dir(out.path()).unwrap().count(), 0);
    }

    /// A resume continues the same **training mode**: an edit run never resumes from a
    /// text-to-image run's checkpoints (or vice versa) just because the factor shapes match. The
    /// resume bundle itself is unstamped (the shared writer stamps no mode), so the mode is read off
    /// the adapter checkpoints.
    ///
    /// The bundles are the real ones (`save_resume` → `{stem}-step{N:06}.resume.safetensors`, which
    /// shares the checkpoints' `{stem}-step` prefix), found by the real `find_latest_resume`.
    ///
    /// *Mutations that red this:* `check_resume_training_mode` ignoring `edit` (an edit run then
    /// resumes a text-to-image run's factors); the checkpoint scan matching the unstamped
    /// `.resume.safetensors` bundle as a text-to-image checkpoint (every edit resume is refused).
    #[test]
    fn a_resume_across_training_modes_is_refused() {
        let mut dit = dit_at(DType::F32);
        let set = install(&mut dit, NetworkType::Lora, 4, vec![]);
        let cfg = base_config();
        let opt = TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
        let run = |dir: &Path, edit: bool| {
            save_adapter(
                &set,
                &provenance_meta(edit),
                &dir.join(checkpoint_filename("adapter", 2)),
            )
            .unwrap();
            save_resume(dir, "adapter", 2, 1, &opt, &set, &cfg, "fingerprint").unwrap();
            let (bundle, step) = find_latest_resume(dir, "adapter").expect("the real bundle");
            assert_eq!(step, 2);
            assert!(bundle.to_string_lossy().ends_with(".resume.safetensors"));
            bundle
        };
        let t2i_dir = scratch("mode_t2i");
        let t2i_bundle = run(t2i_dir.path(), false);
        let edit_dir = scratch("mode_edit");
        let edit_bundle = run(edit_dir.path(), true);

        check_resume_training_mode(t2i_dir.path(), "adapter", &t2i_bundle, false).unwrap();
        check_resume_training_mode(edit_dir.path(), "adapter", &edit_bundle, true).unwrap();
        let err = check_resume_training_mode(t2i_dir.path(), "adapter", &t2i_bundle, true)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`text-to-image` training run; this instruction-edit run"),
            "{err}"
        );
        let err = check_resume_training_mode(edit_dir.path(), "adapter", &edit_bundle, false)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`edit` training run; this text-to-image run"),
            "{err}"
        );
    }

    /// `n` edit pairs in `dir`: distinct non-square targets, each with an RGB and a transparent
    /// RGBA reference.
    fn edit_dataset(dir: &Path, n: usize) -> Vec<TrainingItem> {
        let rgb = write_png(dir, "edit_ref_rgb.png", 64, 64, 11);
        let (rgba, _) = write_rgba_png(dir, "edit_ref_rgba.png", 64, 64);
        (0..n)
            .map(|i| {
                TrainingItem::edit_pair(
                    write_png(
                        dir,
                        &format!("edit_target{i}.png"),
                        96,
                        64,
                        40 + 30 * i as u32,
                    ),
                    format!("recolour the first image like the second, variant {i}"),
                    vec![rgb.clone(), rgba.clone()],
                )
            })
            .collect()
    }

    /// AC (review): an **edit** run checkpoints and resumes like a text-to-image one — interrupted
    /// after step 2 (its `save_every = 2` bundle on disk, next to the `trainingMode = edit`
    /// checkpoint) and resumed to step 4, it lands on the same factors as the straight 4-step edit
    /// run, with and without gradient accumulation.
    ///
    /// *Mutation that reds this:* the resume-mode scan matching the unstamped
    /// `.resume.safetensors` bundle (the edit resume is then refused as a text-to-image one).
    #[test]
    fn an_edit_resume_round_trip_equals_the_straight_run() {
        for accum in [1u32, 2] {
            let dir = scratch("edit_resume");
            let items = edit_dataset(dir.path(), 2);
            let cfg = TrainingConfig {
                resolution: 64,
                gradient_accumulation: accum,
                ..base_config()
            };
            let straight = {
                let mut req = request(dir.path(), items.clone(), cfg.clone());
                req.output_dir = dir.path().join("straight");
                trainer().train(&req, &mut |_| {}).unwrap()
            };

            let interrupted_dir = dir.path().join("resumed");
            let resume_cfg = TrainingConfig {
                save_every: 2,
                ..cfg.clone()
            };
            let mut req = request(dir.path(), items.clone(), resume_cfg.clone());
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
            let (bundle, _) = find_latest_resume(&interrupted_dir, "adapter").expect("a bundle");
            assert!(bundle.to_string_lossy().ends_with(".resume.safetensors"));
            let checkpoint = interrupted_dir.join(checkpoint_filename("adapter", 2));
            assert_eq!(
                metadata(&checkpoint)[EDIT_ADAPTER_MARKER.0],
                EDIT_ADAPTER_MARKER.1
            );

            let mut req = request(
                dir.path(),
                items,
                TrainingConfig {
                    resume: true,
                    ..resume_cfg
                },
            );
            req.output_dir = interrupted_dir;
            let mut trained_steps = Vec::new();
            let resumed = trainer()
                .train(&req, &mut |p| {
                    if let TrainingProgress::Training { step, .. } = p {
                        trained_steps.push(step);
                    }
                })
                .unwrap_or_else(|e| panic!("accum {accum}: the edit resume must continue: {e}"));
            assert_eq!(trained_steps, [3, 4], "accum {accum}");

            let (a, b) = (
                tensors(&straight.adapter_path),
                tensors(&resumed.adapter_path),
            );
            assert_eq!(a.len(), b.len());
            for (key, ta) in &a {
                let diff = max_abs(&(ta - &b[key]).unwrap());
                assert!(diff <= 1e-6, "accum {accum}: {key} differs by {diff}");
            }
        }
    }

    /// AC: edit training learns the **conditional** velocity, not one memorised sample — with a
    /// fresh seeded noise draw and timestep every step (as the train loop samples them), the
    /// windowed-average loss falls; and the references genuinely condition the forward: different
    /// reference latents give a different loss at the same factors. Dense and checkpointed.
    ///
    /// *Mutations that red this:* the step ignoring `references` (the two losses coincide);
    /// regressing onto the wrong target or not stepping the factors (the window does not fall).
    #[test]
    fn edit_loss_decreases_over_steps_and_depends_on_the_references() {
        let cfg = base_config();
        for checkpoint in [false, true] {
            let mut dit = dit_at(DType::F32);
            let set = install(&mut dit, NetworkType::Lora, 4, vec![]);
            let batch = fixed_edit_batch(&dit);

            let other = EditBatch {
                references: batch
                    .references
                    .iter()
                    .enumerate()
                    .map(|(i, r)| randn(r.dims(), 50 + i as u64))
                    .collect(),
                x0: batch.x0.clone(),
                text: batch.text.clone(),
                noise: batch.noise.clone(),
                layout: batch.layout.clone(),
            };
            let (with_refs, _) =
                compute_loss_grads(&dit, &set.vars, &batch.step(0.5), false, checkpoint).unwrap();
            let (with_other, _) =
                compute_loss_grads(&dit, &set.vars, &other.step(0.5), false, checkpoint).unwrap();
            assert!(
                (with_refs - with_other).abs() > 1e-7,
                "the reference latents must enter the forward: {with_refs} vs {with_other}"
            );

            let mut opt =
                TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
            let mut losses = Vec::new();
            for step in 1..=120u32 {
                let noise = flow_match::sample_noise(
                    batch.x0.dims(),
                    flow_match::noise_seed(77, step),
                    &Device::Cpu,
                )
                .unwrap();
                let t = flow_match::sample_unit_timestep(
                    &cfg.timestep_type,
                    &cfg.timestep_bias,
                    flow_match::timestep_seed(77, step),
                );
                let inputs = StepInputs {
                    noise: &noise,
                    ..batch.step(t)
                };
                let (loss, mut grads) =
                    compute_loss_grads(&dit, &set.vars, &inputs, false, checkpoint).unwrap();
                assert!(loss.is_finite(), "non-finite loss {loss}");
                losses.push(loss);
                candle_gen::train::optim::clip_grad_norm(&mut grads, &set.vars, 1.0).unwrap();
                opt.step(&grads).unwrap();
            }
            let window = |w: &[f32]| w.iter().sum::<f32>() / w.len() as f32;
            let (first, last) = (window(&losses[..20]), window(&losses[losses.len() - 20..]));
            eprintln!(
                "[sc-24162] ckpt={checkpoint} varying-noise edit loss window {first:.5} -> {last:.5}"
            );
            assert!(
                last < 0.9 * first,
                "ckpt={checkpoint}: the windowed edit loss must fall under varying noise/t: \
                 {losses:?}"
            );
        }
    }

    /// The gradient-checkpointed joint backward is the dense one on an **edit** layout too: same
    /// loss, same grads — LoRA and LoKr, block and global targets.
    #[test]
    fn checkpointed_edit_grads_match_dense() {
        let with_globals: Vec<String> = ["attn.to_v", "img_mlp.proj", "img_in", "proj_out"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        for network in [NetworkType::Lora, NetworkType::Lokr] {
            for modules in [vec![], with_globals.clone()] {
                let mut dit = dit_at(DType::F32);
                let set = install(&mut dit, network, 4, modules);
                perturb(&set);
                let batch = fixed_edit_batch(&dit);
                let run = |checkpoint| {
                    compute_loss_grads(&dit, &set.vars, &batch.step(0.4), false, checkpoint)
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
                }
                assert!(max_rel < 1e-4, "{network:?}: {max_rel:.2e}");
            }
        }
    }

    /// AC: the preflight prices an edit run's references — their cached latents (latent and train
    /// stages), the longer joint sequence (train stage), the vision tower plus the vision slots of
    /// the image-conditioned prompt (caption stage) — and the largest (aspect-preserving) target.
    ///
    /// *Mutation that reds this:* dropping `shape.reference_tokens` from `seq`, or
    /// `reference_cache_tokens` from the latent cache, or the vision term from the caption stage,
    /// or `training_footprint` ignoring `target_tokens`.
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
        // The refusal names the reference lever.
        let err = check_training_footprint(&facts, &edit, 1 << 30)
            .unwrap_err()
            .to_string();
        assert!(err.contains("fewer reference images"), "{err}");

        // The largest (aspect-preserving) edit target is priced, not the square edge.
        let square = (1024 / 16) * (1024 / 16);
        let explicit_square = TrainingShape {
            target_tokens: square,
            ..t2i
        };
        assert_eq!(
            training_footprint(&facts, &explicit_square),
            a,
            "target_tokens = edge² is the square default"
        );
        let wide = TrainingShape {
            target_tokens: 2 * square,
            ..t2i
        };
        let c = training_footprint(&facts, &wide);
        assert!(c.latent_phase > a.latent_phase && c.train_phase > a.train_phase);
    }

    /// Review: the block-causal prefix is priced exactly, not as `prefix²`. On the fixture edit
    /// layout (text 3 / 2×2 block / text 1 / 2×4 block / text 4 / target) the prefix segments are
    /// `[0,3) [3,7) [7,8) [8,16) [16,20)`, so the prefix calls materialise
    /// `3·3 + 4·7 + 1·8 + 8·16 + 4·20 = 253` scores per head, not `20² = 400`; the text-to-image
    /// prefix is `L²`. The footprint moves by exactly those elements' retained widths, the
    /// fallback (`prefix_scores = 0`) is `prefix²`, and the preflight's layout prices its prompt.
    ///
    /// *Mutations that red this:* `prefix_score_elements` summing `(end − start)²` or `end²`;
    /// `training_footprint` ignoring `prefix_scores` (the delta collapses to 0).
    #[test]
    fn the_prefix_attention_is_priced_exactly() {
        let dit = dit_at(DType::F32);
        let edit = fixed_edit_batch(&dit);
        assert_eq!(prefix_score_elements(&edit.layout), 253);
        assert_eq!(prefix_score_elements(&t2i_layout()), 25);
        let budget = PromptBudget::of(&edit.layout);
        assert_eq!(
            budget,
            PromptBudget {
                text_tokens: 8,
                reference_tokens: 12,
                largest_reference_tokens: 8,
                prefix_scores: 253,
                // The costliest prefix call is the 2×4 block's `8·16`; the copies are
                // `(3+3·3) + (4+3·7) + (1+3·8) + (8+3·16) + (4+3·20)`.
                largest_prefix_call: 128,
                prefix_copy_rows: 182,
            }
        );
        assert_eq!(largest_prefix_call_elements(&t2i_layout()), 25);
        assert_eq!(prefix_copy_rows(&t2i_layout()), 20);

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
        // Checkpointed, no previews: the prefix scores are retained once at the compute width
        // (SCORE_COMPUTE_TENSORS) and at f32 (SCORE_F32_TENSORS). The backward transient is sized by
        // the largest single call, here the target's, so the prefix sum does not reach it.
        let per_element = facts.heads
            * (SCORE_COMPUTE_TENSORS * base.compute_width + SCORE_F32_TENSORS * F32_WIDTH);
        assert_eq!(at(400) - at(253), (400 - 253) * per_element);
        assert_eq!(at(0), at(400), "the fallback is the whole prefix squared");
    }

    /// The facts carry the snapshot's own vision tower: its `model.visual.*` bytes and its config
    /// widths, priced at the component width like the language tower.
    #[test]
    fn facts_read_the_vision_tower_off_the_snapshot() {
        let facts = FootprintFacts::from_snapshot(&tiny_snapshot(), 4).unwrap();
        let vision = loader::load_vision_config(&tiny_snapshot())
            .unwrap()
            .unwrap();
        assert!(facts.vision_tower_bytes > 0);
        assert_eq!(facts.vision_hidden, vision.tower.hidden_size as u64);
        assert_eq!(facts.vision_heads, vision.tower.num_heads as u64);
        let half = FootprintFacts::from_snapshot(&tiny_snapshot(), 2).unwrap();
        assert_eq!(half.vision_tower_bytes * 2, facts.vision_tower_bytes);
    }

    /// AC: a full `Trainer::train` over an edit dataset (an RGB and a transparent RGBA reference,
    /// non-square targets) writes an adapter with the MLX layout, the provenance and the
    /// `trainingMode = edit` marker; renders edit previews on the way; and the adapter, loaded by
    /// the GENERATOR through `LoadSpec::adapters`, changes a tiny candle edit render against the
    /// bare base, while the adapted edit render also differs from the adapted render without
    /// references (the references condition the adapted model).
    #[test]
    fn a_trained_edit_adapter_changes_the_edit_render() {
        use candle_gen::gen_core::{Conditioning, GenerationOutput, GenerationRequest};

        let dir = scratch("edit_run");
        let data = dir.path();
        let rgb_ref = write_png(data, "ref_rgb.png", 64, 64, 11);
        let (rgba_ref, rgba_image) = write_rgba_png(data, "ref_rgba.png", 64, 64);
        let items: Vec<TrainingItem> = (0..2)
            .map(|i| {
                TrainingItem::edit_pair(
                    write_png(data, &format!("target{i}.png"), 96, 64, 40 + 30 * i),
                    format!("recolour the first image like the second, variant {i}"),
                    vec![rgb_ref.clone(), rgba_ref.clone()],
                )
            })
            .collect();
        let cfg = TrainingConfig {
            resolution: 64,
            steps: 6,
            learning_rate: 5e-2,
            alpha: 8.0,
            sample_every: 3,
            sample_prompts: vec!["an edit preview".into()],
            sample_steps: 2,
            ..base_config()
        };
        let req = request(data, items, cfg.clone());
        let mut events = Vec::new();
        let out = trainer()
            .train(&req, &mut |p| events.push(p))
            .unwrap_or_else(|e| panic!("edit run: {e}"));
        assert_eq!(out.steps, 6);
        assert!(out.final_loss.is_finite());
        let samples: Vec<(u32, u32)> = events
            .iter()
            .filter_map(|e| match e {
                TrainingProgress::Sample { image, .. } => Some((image.width, image.height)),
                _ => None,
            })
            .collect();
        assert_eq!(samples, [(64, 64), (64, 64)], "an edit preview per cadence");

        let saved = tensors(&out.adapter_path);
        let layout: std::collections::BTreeMap<String, Vec<usize>> = saved
            .iter()
            .map(|(k, v)| (k.clone(), v.dims().to_vec()))
            .collect();
        assert_eq!(layout, mlx_layout(&cfg));
        let meta = metadata(&out.adapter_path);
        assert_eq!(meta[EDIT_ADAPTER_MARKER.0], EDIT_ADAPTER_MARKER.1);
        for (k, v) in ADAPTER_PROVENANCE {
            assert_eq!(meta[k], v, "{k}");
        }

        // The edit render, through the generator, bare and adapted.
        let render = |adapted: bool, with_references: bool| -> Vec<u8> {
            let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_snapshot()));
            if adapted {
                spec = spec.with_adapters(vec![AdapterSpec::new(
                    out.adapter_path.clone(),
                    1.0,
                    AdapterKind::Lora,
                )]);
            }
            let generator = crate::load(&spec).expect("the generator loads");
            let request = GenerationRequest {
                prompt: "recolour the first image like the second, variant 0".into(),
                width: 64,
                height: 64,
                steps: Some(2),
                seed: Some(5),
                conditioning: if with_references {
                    vec![
                        Conditioning::Reference {
                            image: rgb_image(&rgb_ref),
                            strength: None,
                        },
                        Conditioning::ReferenceRgba {
                            image: rgba_image.clone(),
                            strength: None,
                        },
                    ]
                } else {
                    Vec::new()
                },
                ..Default::default()
            };
            match generator.generate(&request, &mut |_| {}).unwrap() {
                GenerationOutput::Images(images) => images[0].pixels.clone(),
                other => panic!("expected RGB images, got {other:?}"),
            }
        };
        let differing = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();
        let (bare_edit, adapted_edit, bare_t2i, adapted_t2i) = (
            render(false, true),
            render(true, true),
            render(false, false),
            render(true, false),
        );
        // What the adapter does to each render: the per-byte signed delta against the bare base.
        let effect = |adapted: &[u8], bare: &[u8]| -> Vec<i16> {
            adapted
                .iter()
                .zip(bare)
                .map(|(a, b)| i16::from(*a) - i16::from(*b))
                .collect()
        };
        let (edit_effect, t2i_effect) = (
            effect(&adapted_edit, &bare_edit),
            effect(&adapted_t2i, &bare_t2i),
        );
        let vs_bare = differing(&bare_edit, &adapted_edit);
        let effect_gap = edit_effect
            .iter()
            .zip(&t2i_effect)
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "[sc-24162] adapted edit render differs from bare in {vs_bare} bytes; the adapter's \
             effect differs between the edit and the reference-free render in {effect_gap} bytes \
             (of {})",
            bare_edit.len()
        );
        assert!(
            vs_bare > 0,
            "the trained edit adapter must change the edit render"
        );
        // A zero adapter makes both effects all-zero (equal); an adapter whose effect ignored the
        // references would make them equal too.
        assert!(
            effect_gap > 0,
            "the adapter's effect on the render must depend on the references"
        );
    }
}
