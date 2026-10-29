//! LTX-2.3/2.5 LoRA **training** in pure Rust on mlx-rs. The original 2.3 route is the Rust port of
//! SceneWorks' pure-MLX `_LtxMlxLoraBackend` / `LtxMlxLoraTrainer` (`training_adapters.py:3249-3628`),
//! realizing the core [`Trainer`] contract (epic 3039). Retiring the Python version removes the last
//! Python-MLX trainer (blocks sc-3049 cutover → sc-3242 `mlx-video` drop).
//!
//! Built on the same functional-autograd mechanism the Z-Image spike proved (sc-3042) and the
//! sc-3043 runtime glue, but LTX has its **own** adapter seam: its [`crate::transformer::Linear`]
//! carries a per-pass [`LoraStack`](crate::transformer) (not the core `AdaptableLinear`), so this
//! module uses the LTX-local `Linear::set_train_lora` training seam and its own target
//! enumeration / save, while reusing the core [`LoraParams`] + grad-accumulation helpers and the
//! runtime (schedule / dataset / checkpoint).
//!
//! **What is LTX-specific:**
//!   * **2.3 video-only forward over `LtxDiT`.** The reference loads the AV model and trains with
//!     `audio=None`; [`LtxDiT`] is exactly that video-only reduction (the AV checkpoint embeds the
//!     same `transformer_blocks.{i}` video blocks), and the trained video-attention adapter reloads
//!     onto the AvDiT inference path unchanged.
//!   * **2.5 flexible AV forward over `AvDiT`.** All fifteen upstream workflows select generated,
//!     frozen or absent video/audio modalities, intrinsic and tensor-backed conditions, and separate
//!     loss masks. The dev/q4 base is required so preview validation can run the 30-step CFG/STG route.
//!   * **Rectified-flow target = `noise - clean`.** LTX denoises with `x_t - σ·v` over
//!     `x_t = (1-σ)·x0 + σ·noise` and feeds the **raw** transformer output straight to `to_denoised`
//!     (no negation, unlike Z-Image), so the velocity that recovers `x0` is `v = noise - x0`. The
//!     **timestep fed to the DiT is the raw σ** (broadcast over tokens), σ ~ U(1e-3, 1-1e-3). MSE.
//!   * **2.3 latent layout.** A still image VAE-encodes (single frame T=1) to a normalized latent
//!     `(1,128,1,h,w)`, flattened to the patchified `(1, S, 128)` the DiT consumes; the position
//!     grid is built once per resolution bucket (sc-2127; one grid when buckets are off). The 24 GB Gemma text encoder is freed
//!     after the one-time prompt-embed cache (mirroring the reference), before the train loop.
//!     2.5 instead consumes schema-checked prepared video/audio safetensors and caches Gemma-4 AV
//!     contexts before freeing the encoder.
//!   * **Adapter surface.** 2.3 retains `attn1`/`attn2` `to_q/k/v/to_out.0`; 2.5 expands those
//!     suffixes over video/audio self/text attention and both cross-modal directions, with optional
//!     video/audio FF projections and no synthetic FF bias target. Residual LoRA over the (Q4) base — the base
//!     is frozen, gradients flow only through the trainable factors (functional autograd handles the
//!     `quantized_matmul` base as a constant). Saved as `{module}.lora_A/B.weight` + `.alpha` (the
//!     `to_out.0` diffusers spelling the inference loader normalizes), so it round-trips through
//!     the generation-specific strict adapter loader.
//!   * **LoRA-only.** The reference LTX MLX trainer has no LoKr (LTX *inference* supports LoKr via
//!     sc-2393, but no LoKr trainer exists); LoKr requests are rejected with that explanation.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_edges, center_crop_square};
use mlx_gen::train::lora::{accumulate_grads, adapter_optimizer_update, average_grads, LoraParams};
use mlx_gen::train::loss::{prepared_subject_mask_weight, reduce_loss};
use mlx_gen::train::perceptual::{
    combine_step_loss, AuxDriver, AuxModelFootprint, Parameterization, PerceptualPath, StepPlan,
    X0Decoder,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::train::subject_mask::{CropBox, PreparedSubjectMask};
use mlx_gen::train::taehv::{TaehvConfig, TaehvDecoder};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{
    gen_core, LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingConfig, TrainingOutput, TrainingProgress, TrainingReport, TrainingRequest,
    WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, divide, multiply, subtract};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::{LtxConfig, LtxVaeConfig, SplitModel};
use crate::gemma::GemmaConfig;
use crate::gemma4_te::Ltx25TextEncoder;
use crate::model::{MODEL_25_ID, MODEL_ID};
use crate::pipeline::preprocess_conditioning_image;
use crate::positions::{
    create_audio_position_grid, create_audio_position_grid_with, create_position_grid,
    create_position_grid_with, SPATIAL_SCALE,
};
use crate::text_encoder::LtxTextEncoder;
use crate::tokenizer::{Ltx25Tokenizer, LtxTokenizer};
use crate::transformer::{AvDiT, AvPerturbation, BlockLoraRef, LtxAdaptable, LtxDiT, Precision};
use crate::vae::LtxVideoVae;
use mlx_gen::gen_core::ltx_checkpoint::LtxComponent;
use mlx_gen::gen_core::BucketSchedule;

// The trainer's concerns, split out of this file (sc-27342): the LTX-2.5 workflow / modality /
// token planning, the prepared-bundle IO and cached examples, the LTX-2.5 preview validation
// sampler, and the perceptual (depth-anchoring) auxiliary path. The trainer itself stays here.
mod perceptual;
mod plan;
mod prepared;
mod validation;
use perceptual::*;
pub use plan::*;
use prepared::*;
pub use validation::*;

/// Gemma prompt token budget for caption encoding (the captions are short; padding tokens are
/// attended with `mask=None`, matching the reference `Modality(context_mask=None)`).
const MAX_PROMPT_TOKENS: usize = 128;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637).
const SAMPLE_PROMPT_CAP: usize = 4;
const LTX25_VIDEO_LATENT_CHANNELS: i32 = 128;
const LTX25_AUDIO_FLAT_CHANNELS: i32 = 128;

/// The reference `inject_video_attention_lora` default targets (`DEFAULT_LORA_TARGET_MODULES`,
/// `training_adapters.py:72`), restricted to `attn1`/`attn2`. `to_out.0` is the diffusers spelling
/// the inference loader normalizes to the checkpoint's `to_out`.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// One LoRA-trained attention `Linear`: its diffusers save spelling (e.g. `…attn1.to_out.0`), the
/// resolution segments after the `to_out.0`→`to_out` normalization, and the factor-map keys.
#[derive(Clone)]
struct LtxLoraTarget {
    save_path: String,
    segs: Vec<String>,
    a_key: Rc<str>,
    b_key: Rc<str>,
}

/// LoRA trainer for LTX-2.3 and the split Gemma-4 LTX-2.5 route, implementing the core
/// [`Trainer`] surface: a frozen LtxDiT (f32 activations × Q4/Q8 weights) + VAE + text encoder +
/// tokenizer that caches a captioned image dataset to (normalized latent, prompt-embed) pairs,
/// then runs the functional-autograd rectified-flow loop with the sc-3043 runtime glue, and writes
/// a LoRA that round-trips through [`crate::apply_ltx_adapters`].
///
/// **Single-use** (F-055): `train` frees the Gemma text encoder + tokenizer (~24 GB) after the
/// embed cache, so the instance cannot run a second job — `validate` (hence `train`) rejects a reuse
/// up front. Construct a fresh trainer (via [`load_trainer`]) per job.
enum TrainingTokenizer {
    Gemma3(LtxTokenizer),
    Gemma4(Ltx25Tokenizer),
}

impl TrainingTokenizer {
    fn encode(&self, prompt: &str, max_length: usize) -> Result<(Array, Array)> {
        match self {
            Self::Gemma3(tokenizer) => tokenizer.encode(prompt, max_length),
            Self::Gemma4(tokenizer) => tokenizer.encode(prompt, max_length),
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum TrainingTextEncoder {
    Gemma3(LtxTextEncoder),
    Gemma4(Ltx25TextEncoder),
}

impl TrainingTextEncoder {
    fn encode(&self, ids: &Array, mask: &Array) -> Result<Array> {
        match self {
            Self::Gemma3(encoder) => encoder.encode(ids, mask),
            Self::Gemma4(encoder) => encoder.encode(ids, mask),
        }
    }

    fn encode_av(&self, ids: &Array, mask: &Array) -> Result<(Array, Option<Array>)> {
        match self {
            Self::Gemma3(encoder) => Ok((encoder.encode(ids, mask)?, None)),
            Self::Gemma4(encoder) => {
                let (video, audio) = encoder.encode_av(ids, mask)?;
                Ok((video, Some(audio)))
            }
        }
    }
}

pub struct LtxTrainer {
    descriptor: TrainerDescriptor,
    /// Freed after the one-time prompt-embed cache (the 24 GB Gemma backbone), before the loop.
    tokenizer: Option<TrainingTokenizer>,
    text_encoder: Option<TrainingTextEncoder>,
    vae: LtxVideoVae,
    transformer: TrainingTransformer,
    cfg: LtxConfig,
}

#[allow(clippy::large_enum_variant)]
enum TrainingTransformer {
    Ltx23(LtxDiT),
    Ltx25(AvDiT),
}

impl TrainingTransformer {
    #[cfg(test)]
    fn set_sdpa_checkpoint(&mut self, on: bool) {
        match self {
            Self::Ltx23(dit) => dit.set_sdpa_checkpoint(on),
            Self::Ltx25(dit) => dit.set_sdpa_checkpoint(on),
        }
    }

    #[cfg(test)]
    fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        match self {
            Self::Ltx23(dit) => dit.cast_weights(dtype),
            Self::Ltx25(_) => Err("ltx_2_5 test helper cannot cast the full AV base".into()),
        }
    }
}

#[cfg(test)]
impl std::ops::Deref for TrainingTransformer {
    type Target = LtxDiT;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ltx23(dit) => dit,
            Self::Ltx25(_) => panic!("2.3-only test helper used with LTX-2.5"),
        }
    }
}

#[cfg(test)]
impl std::ops::DerefMut for TrainingTransformer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Ltx23(dit) => dit,
            Self::Ltx25(_) => panic!("2.3-only test helper used with LTX-2.5"),
        }
    }
}

fn trainer_descriptor() -> TrainerDescriptor {
    trainer_descriptor_for(MODEL_ID)
}

fn trainer_descriptor_25() -> TrainerDescriptor {
    trainer_descriptor_for(MODEL_25_ID)
}

fn trainer_descriptor_for(id: &'static str) -> TrainerDescriptor {
    TrainerDescriptor {
        id,
        family: "ltx",
        backend: "mlx",
        modality: Modality::Video,
        supports_lora: true,
        // The reference LTX MLX trainer is LoRA-only; LoKr training is unsupported (see `validate`).
        supports_lokr: false,
        // No control-branch training path (F-006). The LTX trainer must reject a control request
        // rather than silently training a plain LoRA (F-055).
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
        // update.
        // sc-2127 (epic 2123): multi-resolution buckets on the 2.3 still-image path only (one cached
        // latent + RoPE grid per spatial edge). 2.5 trains on externally prepared latent packs
        // (`ltxPreparedBundlePath`) whose `videoShape`/`audioShape`, condition masks and token plan
        // are fixed by the pack — there are no pixels to re-encode at another edge and
        // `cfg.resolution` is unused — so it keeps `NONE` and the shared floor refuses buckets.
        // sc-24828 (epic 2123): LTX-2.3 trains on decoded, centre-cropped images and honours the
        // subject-masked loss (one weight per cached (item, bucket) latent). LTX-2.5 trains only on
        // preprocessed latent bundles with no image aligned to the latent, so it does not declare
        // it (refused with the reason by [`refuse_ltx25_subject_mask`]).
        // sc-24830 (epic 2123): depth anchoring through TAELTX2.3 (upstream's tiny decoder for
        // LTX-2.3 and LTX-2.5) — on the LTX-2.3 still-image path (dense and block-checkpointed)
        // and on every LTX-2.5 workflow that generates video (the prepared bundle's clean video
        // latent's selected frames decoded, each `k > 0` with its predecessor; audio ignored). The
        // six 2.5 workflows with no generated video are
        // refused with the reason by [`refuse_ltx25_depth_anchoring`].
        // sc-24833 (epic 2123): the VAE anchor (same TAELTX2.3 decode → FLUX.2 encoder taps, per
        // decoded frame) wherever depth anchoring runs — through the same builder and the same
        // video-workflow refusal. No E-LatentLPIPS: no published weights match this latent family.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: id != MODEL_25_ID,
            subject_mask_loss: id == MODEL_ID,
            depth_anchoring: true,
            // sc-24831: the face losses ride the same shared builder arms + x0 decoder.
            identity_loss: true,
            face_landmark_loss: true,
            // sc-24832: the body losses ride the same builder arms as depth anchoring.
            body_proportion_loss: true,
            body_shape_loss: true,
            normal_loss: true,
            vae_anchor_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// sc-24828 — LTX-2.5 cannot honour subject-masked loss: every one of its workflows trains on a
/// preprocessed video/audio latent bundle (`model_options`), with no decodable image whose crop is
/// aligned to the latent, so a single image mask has nothing to line up with. A typed
/// `Unsupported` naming that reason, raised by `validate`, the start of `train` (before any
/// caching) and the weights-free [`validate_ltx25_training_request`] preflight — ahead of the
/// generic technique floor, so the caller sees why.
fn refuse_ltx25_subject_mask(id: &str, req: &TrainingRequest) -> gen_core::Result<()> {
    if id == MODEL_25_ID && req.config.subject_mask_loss.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} trainer: subject-masked loss needs a training image aligned to the latent, but \
             LTX-2.5 trains on preprocessed video/audio latent bundles that carry none; turn \
             subject-masked loss off for this trainer"
        )));
    }
    // sc-24832: the subject-restricted normal loss reads the same (absent) aligned mask.
    if id == MODEL_25_ID
        && mlx_gen::train::subject_mask::PerceptualSubjectMasks::needed(&req.config)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} trainer: the normal loss restricted to the subject needs a training image aligned \
             to the latent, but LTX-2.5 trains on preprocessed latent bundles that carry none; turn \
             body_losses.normal_restrict_to_subject off for this trainer"
        )));
    }
    Ok(())
}

mod ltx25_restricted_normal_tests;

/// sc-24830 — LTX-2.5 runs depth anchoring on its generated video stream, so it refuses it for the
/// six workflows that generate no video (`v2a_lora`, whose video is frozen conditioning, and the
/// audio-only `t2a_lora`, `audio_extend_lora`, `audio_inpainting_lora`, `audio_suffix_lora`,
/// `a2a_ic_lora`): a typed `Unsupported` naming the workflow, raised by `validate`, the start of
/// `train` (before any caching) and the weights-free [`validate_ltx25_training_request`] preflight
/// — ahead of the generic technique floor, so the caller sees why. Also validates the
/// [`DEPTH_ANCHORING_FRAMES_KEY`] knob.
fn refuse_ltx25_depth_anchoring(id: &str, req: &TrainingRequest) -> gen_core::Result<()> {
    // Every decoded-x0 perceptual loss (depth anchoring, the sc-24831 identity / face-landmark
    // losses, the sc-24833 VAE anchor, the sc-24832 body losses) decodes the same generated video
    // frames, so the refusal covers whichever ones are on and names them.
    let pixel_losses = mlx_gen_perceptual::enabled_pixel_aux_losses(&req.config);
    if id != MODEL_25_ID || pixel_losses.is_empty() {
        return Ok(());
    }
    depth_anchoring_frames(&req.config)?;
    // The resolved plan (per-request modality overrides applied) when it parses; else the named
    // workflow's canonical plan — an otherwise-malformed request is reported by the plan
    // validation that follows, but never slips a video-less workflow past this refusal.
    let (workflow, video) = match Ltx25TrainingPlan::from_request(req) {
        Ok(plan) => (plan.workflow, plan.video.map(|v| v.is_generated)),
        Err(_) => {
            let Some(workflow) = req
                .config
                .model_options
                .get("ltxWorkflow")
                .and_then(serde_json::Value::as_str)
                .and_then(|id| Ltx25Workflow::parse(id).ok())
            else {
                return Ok(());
            };
            (workflow, workflow.plan().video.map(|v| v.is_generated))
        }
    };
    if video != Some(true) {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} trainer: depth anchoring and the other decoded-x0 losses ({}) decode the \
             generated video stream, but workflow `{}` generates no video (the video stream is \
             {}); turn them off for this workflow",
            pixel_losses.join(", "),
            workflow.id(),
            if video.is_some() {
                "frozen conditioning"
            } else {
                "absent"
            }
        )));
    }
    Ok(())
}

/// `model_options` key selecting how many latent frames of an LTX-2.5 video clip the depth loss
/// decodes per step (sc-24830). Each selected latent frame decodes through TAELTX2.3 to one pixel
/// frame — frame 0 as a `T = 1` clip, a frame `k > 0` together with its predecessor `k − 1` as a
/// `T = 2` clip, keeping its last output frame; the frames are evenly spaced over the clip's latent
/// frames that carry generated (loss) tokens, fixed per example. Positive integer; default
/// [`DEFAULT_DEPTH_ANCHORING_FRAMES`]. Bounds the per-step decode + Depth-Anything-V2 memory.
pub const DEPTH_ANCHORING_FRAMES_KEY: &str = "depthAnchoringFrames";

/// The default [`DEPTH_ANCHORING_FRAMES_KEY`]: two latent frames per step.
pub const DEFAULT_DEPTH_ANCHORING_FRAMES: usize = 2;

/// The validated [`DEPTH_ANCHORING_FRAMES_KEY`] of `cfg` (default when absent).
fn depth_anchoring_frames(cfg: &TrainingConfig) -> Result<usize> {
    match cfg.model_options.get(DEPTH_ANCHORING_FRAMES_KEY) {
        None => Ok(DEFAULT_DEPTH_ANCHORING_FRAMES),
        Some(value) => value
            .as_u64()
            .filter(|&n| n >= 1)
            .map(|n| n as usize)
            .ok_or_else(|| {
                mlx_gen::Error::Msg(format!(
                    "ltx_2_5 trainer: `{DEPTH_ANCHORING_FRAMES_KEY}` must be a positive integer, \
                     got {value}"
                ))
            }),
    }
}

/// The latent frames of a `(F, H, W)`-token clip the depth loss decodes (sc-24830): `k` frames
/// evenly spaced over the frames holding at least one generated (`loss_mask > 0`) target token —
/// conditioning frames would decode their clean latent and train nothing. Empty when no frame is
/// generated.
fn select_depth_frames(
    loss_mask: &[f32],
    frames: usize,
    frame_tokens: usize,
    k: usize,
) -> Vec<i32> {
    let candidates: Vec<i32> = (0..frames)
        .filter(|&f| {
            loss_mask[f * frame_tokens..(f + 1) * frame_tokens]
                .iter()
                .any(|&m| m > 0.0)
        })
        .map(|f| f as i32)
        .collect();
    let n = candidates.len();
    if k >= n {
        return candidates;
    }
    if k == 1 {
        return vec![candidates[n / 2]];
    }
    let mut picked: Vec<i32> = (0..k)
        .map(|i| candidates[(i * (n - 1) + (k - 1) / 2) / (k - 1)])
        .collect();
    picked.dedup();
    picked
}

/// Construct the trainer from an LTX-2.3 split-weight snapshot directory (transformer / VAE /
/// connector + the Gemma-3-12B text-encoder snapshot resolved like inference). The transformer loads
/// at **f32 activations × quantized weights** (`quant_f32`) for clean autograd — the base is frozen,
/// gradients flow only through the trainable LoRA factors. Registered via [`mlx_gen::TrainerRegistration`].
///
/// The weights load lazily (sc-2124): construction only checks the spec (and that the Gemma-3
/// override exists), so `validate` and `train`'s refusal floors never read weights; see
/// [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    snapshot_root_23(spec)?;
    crate::model::resolve_gemma_dir(spec.text_encoder.as_ref())?;
    Ok(Box::new(mlx_gen::train::lazy::LazyTrainer::new(
        trainer_descriptor(),
        validate_floors,
        {
            let spec = spec.clone();
            move || load_weights_23(&spec)
        },
    )))
}

/// The LTX-2.3 snapshot directory a trainer spec names — a single file is refused.
fn snapshot_root_23(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(mlx_gen::Error::Msg(
            "ltx_2_3 trainer expects a split-weight snapshot directory (transformer.safetensors \
             / vae_*.safetensors / connector.safetensors), not a single file"
                .into(),
        )),
    }
}

/// The LTX-2.3 weight load behind [`load_trainer`], run by [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer) on first need.
fn load_weights_23(spec: &LoadSpec) -> Result<LtxTrainer> {
    load_trainer_from_dir(snapshot_root_23(spec)?, spec.text_encoder.as_ref())
}

/// Construct the LTX-2.5 trainer from its split Gemma-4 bundle.  This follows the ordinary
/// provider's component resolver so training cannot quietly combine a 2.5 DiT with a Gemma-3
/// override or a 2.3-shaped VAE configuration.
///
/// The weights load lazily (sc-2124): construction only checks the spec, so `validate` and `train`'s
/// refusal floors never read weights; see [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer).
pub fn load_trainer_25(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    snapshot_root_25(spec)?;
    Ok(Box::new(mlx_gen::train::lazy::LazyTrainer::new(
        trainer_descriptor_25(),
        validate_floors,
        {
            let spec = spec.clone();
            move || load_weights_25(&spec)
        },
    )))
}

/// The LTX-2.5 split-component directory a trainer spec names — a single checkpoint is refused.
fn snapshot_root_25(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(path) => Ok(path),
        WeightsSource::File(_) => Err(mlx_gen::Error::Msg(
            "ltx_2_5 trainer expects a split-component directory, not a single checkpoint".into(),
        )),
    }
}

/// The LTX-2.5 weight load behind [`load_trainer_25`], run by [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer) on first need.
fn load_weights_25(spec: &LoadSpec) -> Result<LtxTrainer> {
    let root = snapshot_root_25(spec)?;
    let bundle = crate::bundle::resolve_split_bundle(spec)?;
    crate::bundle::assert_gemma_version(&bundle)?;
    ensure_ltx25_training_variant(crate::dev_sampler::from_bundle(&bundle)?)?;
    let split = SplitModel::from_model_dir(root)?;
    ensure_ltx25_training_tier(split)?;
    let cfg = LtxConfig::from_bundle(&bundle)?;
    let vae_cfg = LtxVaeConfig::from_bundle(&bundle, LtxComponent::ConvVideoVae)?;
    let transformer_w = Weights::from_file(bundle.require(LtxComponent::Transformer)?.path())?;
    let video_w = Weights::from_file(bundle.require(LtxComponent::ConvVideoVae)?.path())?;
    let connector_path = bundle
        .require(LtxComponent::Transformer)?
        .path()
        .with_file_name("connector.safetensors");
    if !connector_path.is_file() {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_5 trainer: missing connector.safetensors beside {}",
            bundle.require(LtxComponent::Transformer)?.path().display()
        )));
    }
    let connector_w = Weights::from_file(&connector_path)?;
    let text_encoder = Ltx25TextEncoder::from_bundle_av(
        &bundle,
        &connector_w,
        &cfg,
        Precision::quant_bf16(split.bits, split.group),
        mlx_gen::gen_core::OffloadPolicy::Resident,
    )?;
    let tokenizer =
        Ltx25Tokenizer::from_packed_te_file(bundle.require(LtxComponent::TextEncoder)?.path())?;
    let transformer = AvDiT::from_weights(
        &transformer_w,
        &cfg,
        Precision::quant_f32(split.bits, split.group),
    )?;
    // Converted 2.5 tiers split the encoder into a sibling file; raw upstream split bundles keep
    // it inside the ConvVAE component.  Match the ordinary provider's documented preference rather
    // than making training depend on one publication layout.
    let split_encoder = root.join("vae_encoder.safetensors");
    let encoder_path = if split_encoder.is_file() {
        split_encoder
    } else {
        bundle
            .require(LtxComponent::ConvVideoVae)?
            .path()
            .to_path_buf()
    };
    let vae = LtxVideoVae::from_weights_lazy_encoder(&video_w, encoder_path, &vae_cfg)?;
    // Materialize at load (sc-24245; see mlx_gen_qwen_image::loader::load_transformer_with).
    for w in [&connector_w, &transformer_w, &video_w] {
        w.materialize_accessed()?;
    }
    Ok(LtxTrainer {
        descriptor: trainer_descriptor_25(),
        tokenizer: Some(TrainingTokenizer::Gemma4(tokenizer)),
        text_encoder: Some(TrainingTextEncoder::Gemma4(text_encoder)),
        vae,
        transformer: TrainingTransformer::Ltx25(transformer),
        cfg,
    })
}

/// The concrete-typed loader behind [`load_trainer`] (sc-4942 — the first-step memory harness needs
/// the concrete [`LtxTrainer`] to reach `.transformer` / `.vae`, which a `Box<dyn Trainer>` hides).
///
/// `te_override` is `LoadSpec::text_encoder` — the bundled Gemma-3 dir the self-contained LTX install
/// ships beside the tier weights (sc-8827/sc-9989). Threaded into [`resolve_gemma_dir`] exactly as the
/// inference path does (`model.rs`), which as of sc-13664 **requires** the slot: `None` is a load-time
/// error (no env / HF-cache fallback), so the trainer must be handed an explicit TE dir.
fn load_trainer_from_dir(root: &Path, te_override: Option<&WeightsSource>) -> Result<LtxTrainer> {
    // Resolve (and validate) the Gemma-3 TE location up front — an absent/bad `LoadSpec::text_encoder`
    // override fails fast here, ahead of the split-weight load, and keeps the wiring unit-testable
    // without the ~24 GB base snapshot (sc-9989). Only the path is resolved here; the heavy
    // `Weights::from_dir` load stays below with the rest of the component loads.
    let gemma_dir = crate::model::resolve_gemma_dir(te_override)?;

    let split = SplitModel::from_model_dir(root)?;
    let cfg = LtxConfig::from_model_dir(root)?;
    let vae_config = LtxVaeConfig::from_model_dir(root)?;

    let gemma_w = Weights::from_dir(&gemma_dir)?;
    let gemma_quant = crate::model::resolve_gemma_quant(&gemma_dir)?;
    let connector_w = Weights::from_file(root.join("connector.safetensors"))?;
    let transformer_w = Weights::from_file(root.join("transformer.safetensors"))?;
    let vae_dec_w = Weights::from_file(root.join("vae_decoder.safetensors"))?;
    let vae_enc_w = Weights::from_file(root.join("vae_encoder.safetensors"))?;

    // Video-only text encoder (bf16, the reference TE dtype); we cast its embeds to f32 per-item for
    // the f32 training forward.
    let text_encoder = LtxTextEncoder::from_weights(
        &gemma_w,
        &connector_w,
        GemmaConfig::gemma_3_12b(),
        gemma_quant,
        &cfg,
        // bf16 activations at the checkpoint's own quant geometry: the connector and the
        // feature-extractor Linear take the packed arm on an LTX-2.5 tier and the dense arm on
        // LTX-2.3's dense `connector.safetensors`, decided per tensor by `.scales` presence.
        Precision::quant_bf16(split.bits, split.group),
    )?;
    let transformer = LtxDiT::from_weights(
        &transformer_w,
        &cfg,
        Precision::quant_f32(split.bits, split.group),
    )?;
    let vae = LtxVideoVae::from_weights(&vae_dec_w, Some(&vae_enc_w), &vae_config)?;
    // Materialize at load (sc-24245; see mlx_gen_qwen_image::loader::load_transformer_with).
    for w in [
        &gemma_w,
        &connector_w,
        &transformer_w,
        &vae_dec_w,
        &vae_enc_w,
    ] {
        w.materialize_accessed()?;
    }
    let tokenizer = LtxTokenizer::from_dir(&gemma_dir)?;

    Ok(LtxTrainer {
        descriptor: trainer_descriptor(),
        tokenizer: Some(TrainingTokenizer::Gemma3(tokenizer)),
        text_encoder: Some(TrainingTextEncoder::Gemma3(text_encoder)),
        vae,
        transformer: TrainingTransformer::Ltx23(transformer),
        cfg,
    })
}

// The trainer registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION_25 = trainer_descriptor_25 => load_trainer_25
}

/// Capability-free request validation, factored out of [`Trainer::validate`] so it can be
/// unit-tested without a loaded trainer. Rejects an empty dataset, zero rank, LoKr (LoRA-only
/// trainer), and unsupported optimizers. The single-use / text-encoder-present check stays in
/// [`Trainer::validate`], which has the trainer state to inspect (F-055).
fn validate_request(req: &TrainingRequest, label: &str) -> Result<()> {
    if req.items.is_empty() {
        return Err(format!("{label}: dataset is empty").into());
    }
    if req.config.rank == 0 {
        return Err(format!("{label}: rank must be > 0").into());
    }
    if req.config.network_type == NetworkType::Lokr {
        return Err(format!(
            "{label}: LoKr training is not supported — the reference LTX MLX \
                    trainer is LoRA-only. (LTX *inference* supports LoKr via sc-2393, but no \
                    LoKr trainer exists yet; that would be a separate extension.)"
        )
        .into());
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "{label}: optimizer '{}' is not available on MLX training (supported: \
             adamw, adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    Ok(())
}

/// Weights-free (with respect to the multi-gigabyte base model) LTX-2.5 request preflight. Product
/// callers can run this before constructing a trainer: it resolves the complete workflow and Dev
/// validation contract, requires every prepared-pack path, and validates each safetensors header,
/// schema, modality shape and tensor-backed condition key. The same function is the request half of
/// [`LtxTrainer::validate`], so preflight and execution cannot drift.
pub fn validate_ltx25_training_request(req: &TrainingRequest) -> Result<()> {
    let descriptor = trainer_descriptor_25();
    refuse_ltx25_subject_mask(descriptor.id, req)?;
    refuse_ltx25_depth_anchoring(descriptor.id, req)?;
    // The shared floors keep their typed variant across the seam (`?` maps
    // `gen_core::Error::Unsupported` 1:1): a capability gap must stay `Unsupported` for the worker,
    // never be flattened to a message (sc-24161).
    gen_core::train::validate_control_request(&descriptor, req)?;
    gen_core::train::validate_full_finetune_request(&descriptor, req)?;
    gen_core::train::validate_training_techniques(&descriptor, req)?;
    gen_core::train::validate_edit_request(&descriptor, req)?;
    validate_request(req, "ltx_2_5 trainer")?;
    validate_ltx25_adapter_scale(req.config.alpha)?;
    let plan = Ltx25TrainingPlan::from_request(req)?;
    for item in &req.items {
        let path = prepared_bundle_path(item)?;
        if !path.is_file() {
            return Err(format!(
                "ltx_2_5 trainer: prepared bundle does not exist: {}",
                path.display()
            )
            .into());
        }
        let weights = Weights::from_file(path)?;
        validate_prepared_bundle_contract(&weights, path, &plan)?;
    }
    Ok(())
}

/// Every weights-free [`Trainer::validate`] floor — all of it but the single-use check.
fn validate_floors(descriptor: &TrainerDescriptor, req: &TrainingRequest) -> gen_core::Result<()> {
    // Shared control-training floor (F-006 / F-055): the LTX trainer has no control branch, so a
    // request carrying `control_type` / per-item control images is rejected (typed `Unsupported`)
    // rather than silently training a plain LoRA and reporting success.
    gen_core::train::validate_control_request(descriptor, req)?;
    // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
    // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA.
    gen_core::train::validate_full_finetune_request(descriptor, req)?;
    refuse_ltx25_subject_mask(descriptor.id, req)?;
    refuse_ltx25_depth_anchoring(descriptor.id, req)?;
    // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
    // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
    gen_core::train::validate_training_techniques(descriptor, req)?;
    gen_core::train::validate_edit_request(descriptor, req)?;
    if descriptor.id == MODEL_25_ID {
        validate_ltx25_training_request(req)?;
    } else {
        validate_request(req, &format!("{} trainer", descriptor.id))?;
    }
    Ok(())
}

impl Trainer for LtxTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Single-use enforcement (F-055): `train` frees the Gemma text encoder + tokenizer (~24 GB)
        // after the embed cache, so a second `train` on the same instance can't re-encode. Fail here,
        // up front (validate runs before any progress is emitted), instead of with a late, confusing
        // "text encoder missing" mid-run. Construct a fresh trainer (via `load_trainer`) per job.
        let label = format!("{} trainer", self.descriptor.id);
        if self.text_encoder.is_none() || self.tokenizer.is_none() {
            return Err(format!(
                "{label}: single-use — the Gemma text encoder was freed after the \
                        first train() to reclaim ~24 GB; construct a fresh trainer for each job"
            )
            .into());
        }
        validate_floors(self.descriptor(), req)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        self.train_with_report(req, on_progress)
            .map(|report| report.output)
    }

    /// The primary entry point (sc-27451): a preview sample that fails to render is reported as a
    /// [`gen_core::train::PREVIEW_SAMPLE_FAILED`] warning instead of only on stderr.
    fn train_with_report(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingReport> {
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        refuse_ltx25_subject_mask(self.descriptor.id, req)?;
        refuse_ltx25_depth_anchoring(self.descriptor.id, req)?;
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        let mut warnings = Vec::new();
        let output = self.train_impl(req, on_progress, &mut warnings)?;
        Ok(TrainingReport { output, warnings })
    }
}

impl LtxTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720), keeping `?` on `mlx_rs`/family helpers transparent here.
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
        warnings: &mut Vec<gen_core::GenerationWarning>,
    ) -> Result<TrainingOutput> {
        if matches!(self.transformer, TrainingTransformer::Ltx25(_)) {
            return self.train_25_impl(req, on_progress);
        }
        self.train_23_impl(req, on_progress, warnings)
    }

    fn train_23_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
        warnings: &mut Vec<gen_core::GenerationWarning>,
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let TrainingTransformer::Ltx23(transformer) = &mut self.transformer else {
            unreachable!("dispatcher selected the 2.3 path")
        };
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);
        // sc-2127 — one pixel edge (multiple of 32) per resolution bucket (just `[resolution]` when
        // buckets are off), and its latent tokens per side. The memory guard and previews size for
        // the largest (epic 2123 E7).
        let edges = bucket_edges(cfg);
        let latent_edges: Vec<usize> = edges.iter().map(|&e| latent_edge_of(e)).collect();
        let latent_edge = guard_latent_edge(&edges);

        // sc-4942 — LTX trains in **f32 activations** (× the Q-packed base), NOT bf16, even though the
        // SceneWorks worker passes `train_dtype=bf16` (sc-4881). MEASURED on real weights (the
        // `bf16_grads_direction_and_memory_vs_f32` harness): a bf16 activation cast DECORRELATES the
        // gradient from the f32 (quality) path — global cosine 0.31–0.45, with the early/deep K
        // projections of BOTH attentions pointing ~opposite — because the 48-block distilled DiT is
        // chaos-sensitive (the same reason inference uses `quant_f32`, not `quant_bf16`, for quality;
        // see `transformer::Precision`/`Mode::QuantF32`). bf16 would save ~30% memory (1024: 28 vs 43 GB)
        // but the f32 working set already fits the video tier with the checkpointing levers below
        // (1024 ≈ 43 GB attn-ckpt / 27 GB block-ckpt, measured 128 GB box; the guard auto-scales), so
        // honoring bf16 would trade training quality for memory we do not need. (`LtxDiT::cast_weights`
        // stays available — the bf16 harness that produced this finding exercises it — but the
        // production trainer never invokes it; this trainer is f32-only.)
        let compute_dtype = Dtype::Float32;

        // sc-4942 — fail-fast pre-flight memory guard (the sc-4874 mechanism, ported to LTX). The dense
        // (non-block-checkpointed) first step materializes the whole forward graph in one MLX `eval`; at
        // high resolution that working set can exceed unified memory and the OS hard-kills the worker
        // with an UNCATCHABLE SIGKILL. We predict it and refuse up front with an actionable, catchable
        // error — BEFORE the (~minutes-long) latent caching — when gradient checkpointing is not
        // enabled. (LTX is LoRA-only, so the LoRA-path condition is always met.)
        let will_checkpoint = cfg.gradient_checkpointing;
        // Epic 2123 E7 (sc-24830): the aux-loss models (TAELTX2.3 + every enabled arm)
        // count against the budget on BOTH paths — a checkpointed depth job must not skip it.
        let aux_gb = perceptual_footprint_gb(
            cfg,
            edges.iter().copied().max().unwrap_or(0),
            req.items.len() * edges.len(),
        );
        if !will_checkpoint || aux_gb > 0.0 {
            preflight_memory_guard(cfg, latent_edge, aux_gb, will_checkpoint)?;
        }

        // Epic 2123 depth anchoring (sc-24830): load the frozen TAELTX2.3 decoder +
        // Depth-Anything-V2 before the caching pass, so a missing checkpoint fails fast.
        let mut perceptual = load_perceptual_path(cfg)?;

        // --- prepare → load → cache: normalized latents + prompt embeds (then free the TE) ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127). Per entry: patchified clean
        // latent, prompt embeds, and (subject-masked loss, sc-24828) that latent's patchified
        // loss-weight map — `None` when the technique is off.
        let mut cache: Vec<(Array, Array, Option<Array>)> =
            Vec::with_capacity(req.items.len() * edges.len());
        // sc-5637 — preview-sample prompts, pre-encoded inside the `te`/`tok` scope below (the Gemma
        // encoder is freed before the train loop). LTX is distilled (no CFG) → one ctx per prompt.
        let mut sample_ctxs: Vec<(String, Array)> = Vec::new();
        {
            let te = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg("ltx_2_3 trainer: text encoder missing".into())
            })?;
            let tok = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| mlx_gen::Error::Msg("ltx_2_3 trainer: tokenizer missing".into()))?;
            for (i, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                on_progress(TrainingProgress::Caching {
                    current: i as u32 + 1,
                    total,
                });
                let img = center_crop_square(&decode_image(&item.image_path)?);
                // sc-24828: the item's mask is read and checked once, resampled per bucket below.
                let subject_mask = PreparedSubjectMask::load_if_enabled(
                    "ltx_2_3 trainer",
                    item,
                    cfg.subject_mask_loss.as_ref(),
                )?;
                let (ids, mask) = tok.encode(&item.caption, MAX_PROMPT_TOKENS)?;
                let ctx = to_dtype(&te.encode(&ids, &mask)?, Dtype::Float32)?; // (1, L, 4096)
                eval([&ctx])?;
                for (clean, mask_weight) in encode_bucket_latents(
                    &edges,
                    |edge| {
                        let prep = preprocess_conditioning_image(&img, edge, edge)?; // (1,3,1,edge,edge)
                        self.vae.encode(&prep) // (1,128,1,le,le), normalized, f32
                    },
                    |shape| still_subject_weight("ltx_2_3 trainer", subject_mask.as_ref(), shape),
                )? {
                    cache.push((clean, ctx.clone(), mask_weight));
                }
            }
            // sc-5637 — pre-encode the preview-sample prompts while the encoder is still resident.
            if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled()
            {
                for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                    let (ids, mask) = tok.encode(prompt, MAX_PROMPT_TOKENS)?;
                    let ctx = to_dtype(&te.encode(&ids, &mask)?, Dtype::Float32)?;
                    eval([&ctx])?;
                    sample_ctxs.push((prompt.clone(), ctx));
                }
            }
        }
        if cache.is_empty() {
            // sc-4895 — a cancel tripped during caching is a genuine cancellation → typed
            // `Error::Canceled` (bridged 1:1 to `gen_core::Error::Canceled`); an empty cache with no
            // cancel is a real "no usable dataset items" error.
            if req.cancel.is_cancelled() {
                return Err(mlx_gen::Error::Canceled);
            }
            return Err("ltx_2_3 trainer: no usable dataset items".into());
        }
        // Free the Gemma text encoder + tokenizer (~24 GB) before training — they are only needed for
        // the one-time embed cache (mirrors the reference `prepare_dataset` release).
        self.text_encoder = None;
        self.tokenizer = None;

        let sampling_enabled = !sample_ctxs.is_empty();

        // Epic 2123 E8: each (item, bucket) entry's perceptual reference (TAELTX2.3 decode of its
        // cached clean latent → DA2 depth) is computed exactly once per job, by the `AuxDriver`
        // before the loop.
        if let Some(path) = perceptual.as_mut() {
            // sc-24832: the job's subject masks (restricted normal loss) reach every reference,
            // cropped like the image and resampled onto its decoded size.
            path.attach_subject_masks(mlx_gen::train::subject_mask::PerceptualSubjectMasks::load(
                "ltx trainer",
                &req.items,
                cfg,
                latent_edges.len(),
                CropBox::center_square,
            )?);
        }

        // The RoPE position grid is identical across items at a fixed latent resolution (single
        // frame) — build it once per bucket (sc-2127: each step uses its sampled bucket's grid).
        // Preview-sample rendering (sc-5637) uses the largest bucket's grid at `latent_edge`.
        let bucket_positions: Vec<Array> = latent_edges
            .iter()
            .map(|&le| create_position_grid(1, 1, le, le))
            .collect();
        let preview_positions = &bucket_positions[largest_bucket(&latent_edges)];

        // --- adapter targets + trainable factors ---
        let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
            DEFAULT_TARGET_SUFFIXES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            cfg.lora_target_modules.clone()
        };
        let (targets, mut params) = build_targets(
            transformer,
            self.cfg.num_layers,
            &suffixes,
            cfg.rank as i32,
            cfg.seed,
        )?;
        if targets.is_empty() {
            return Err(
                "ltx_2_3 trainer: no LoRA targets resolved (check lora_target_modules)".into(),
            );
        }
        let alpha = cfg.alpha;
        let rank = cfg.rank as f32;
        let mae = {
            let lt = cfg.loss_type.to_ascii_lowercase();
            lt == "mae" || lt == "l1"
        };

        // sc-4942 — gradient checkpointing. Group the resolved targets by their owning block so the
        // checkpointed forward can thread each block's LoRA factors as explicit recompute inputs. Every
        // target of this trainer lives in a `transformer_blocks.{i}.attn{1,2}` leaf, so the grouping
        // covers the whole adapter surface.
        let n_layers = self.cfg.num_layers as usize;
        let block_targets = group_block_targets(&targets, n_layers);
        // Gradient checkpointing is an OPT-IN OPTION (the SceneWorks "Gradient Checkpointing" toggle),
        // never auto-forced — a run that would OOM is caught instead by the fail-fast pre-flight guard
        // above, which recommends this flag rather than silently changing the user's training dynamics.
        let use_checkpoint = cfg.gradient_checkpointing;
        let checkpoint_block: Option<&[Vec<BlockLoraRef>]> = if use_checkpoint {
            Some(&block_targets)
        } else {
            None
        };
        // sc-4942 — attention-segment checkpointing is on for the dense (non-block-checkpointed) path:
        // it is numerically identical to the retained backward (same decomposed attention, recomputed)
        // and removes the dominant seq² per-block retention — the flash-backward surrogate every torch
        // trainer gets from its fused SDPA kernel. When whole-block checkpointing is on it goes OFF (the
        // block recompute already covers attention; nesting would recompute it twice for no win).
        transformer.set_sdpa_checkpoint(!use_checkpoint);

        // AdamW with wd=0 is identical to Adam, so the one optimizer covers both choices.
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

        // --- resume (F-125): continue from the latest snapshot of THIS adapter in output_dir, if any ---
        let mut update_idx: u32 = 0;
        let mut start_step: u32 = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
                params = loaded;
                start_step = meta.step;
                update_idx = meta.update_idx;
                eprintln!(
                    "[F-125] resuming from step {start_step} (optimizer update {update_idx})"
                );
            }
        }

        // --- train loop ---
        // sc-2127: which cached (item, bucket) latent each step trains on (round-robin over items
        // for a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
        let schedule =
            BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
        // Epic 2123 E8: the perceptual alternation interleaves optimizer windows over the
        // schedule's epochs (sc-2124) — a pure function of the step, so a resumed run needs no
        // replay.
        let mut aux_driver = match perceptual {
            Some(path) => Some(aux_driver(
                path,
                &cache,
                &latent_edges,
                &schedule,
                accum,
                &Default::default(),
            )?),
            None => None,
        };
        let mut accumulated: Option<LoraParams> = None;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (entry, bucket) = step_entry(&schedule, step);
            let (clean, ctx, mask_weight) = &cache[entry];
            let positions = &bucket_positions[bucket];
            // σ ~ U(1e-3, 1-1e-3), deterministic in seed (the reference's uniform timestep).
            let mut sigma = {
                let k = random::key(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64))?;
                random::uniform::<_, f32>(1e-3f32, 1.0 - 1e-3, &[1], Some(&k))?.item::<f32>()
            };
            // Epic 2123 E8: plan the step's loss terms; an aux-only step trains at the σ remapped
            // into the loss window.
            let planned = match aux_driver.as_mut() {
                Some(d) => d.sample(step, &schedule).plan(sigma)?,
                None => None,
            };
            if let Some(p) = &planned {
                sigma = p.plan.noise_level;
            }
            let aux = planned.as_ref().map(|p| AuxStep {
                path: p.path,
                plan: &p.plan,
                image: p.entry,
                latent_edge: latent_edges[bucket],
            });
            let noise = random::normal::<f32>(
                clean.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let (losses, grads) = compute_step_loss_grads(
                transformer,
                &params,
                &targets,
                alpha,
                rank,
                clean,
                ctx,
                positions,
                sigma,
                &noise,
                mae,
                mask_weight.as_ref(),
                checkpoint_block,
                compute_dtype,
                aux,
            )?;
            last_loss = losses.total;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                // F-017: average by the ACTUAL in-window count, not the full `accum`. The final-step
                // flush is usually a partial window (cfg.steps % accum != 0); dividing by `accum`
                // down-scaled that update (halved effective LR on the tail). Mirrors z-image/lens
                // F-069. (When step%accum==0 the window is the full `accum`.)
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

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                save_lora(&params, &targets, alpha, cfg.rank, &ckpt)?;
                checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // sc-5637 — periodic best-effort preview frames from the in-progress adapter. Install the
            // current factors concretely for the forward-only render (the next step's traced `loss_fn`
            // re-installs them); a render failure logs and is skipped, never failing the training run.
            if sampling_enabled && cfg.sample_every > 0 && step % cfg.sample_every == 0 {
                let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
                install_train_lora(transformer, &params, &targets, alpha, rank, lora_dtype)?;
                let total = sample_ctxs.len() as u32;
                for (i, (prompt, ctx)) in sample_ctxs.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match crate::pipeline::render_sample(
                        transformer,
                        &self.vae,
                        ctx,
                        preview_positions,
                        sample_seed,
                        latent_edge,
                        compute_dtype,
                    ) {
                        Ok(image) => on_progress(TrainingProgress::Sample {
                            step,
                            index: i as u32 + 1,
                            total,
                            prompt: prompt.clone(),
                            image,
                        }),
                        Err(e) => {
                            eprintln!(
                                "[sc-5637] {MODEL_ID} preview sample failed at step {step} \
                                 (prompt {}): {e} — skipping this preview, training continues",
                                i + 1
                            );
                            warnings.push(gen_core::train::preview_sample_failed(
                                step,
                                i as u32 + 1,
                                &e,
                            ));
                        }
                    }
                }
            }
        }

        // Cancelled before completing a single step (`steps == 0` is rejected upstream by
        // `validate`): the LoRA factors are still freshly initialized with `B = 0`, a no-op adapter.
        // Surface the typed `Error::Canceled` (sc-4895, bridged 1:1 to `gen_core::Error::Canceled`)
        // rather than writing a valid-looking `.safetensors` and returning `Ok` — downstream tooling
        // would otherwise ship an identity LoRA as a trained artifact (F-040).
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_lora(&params, &targets, alpha, cfg.rank, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }

    fn train_25_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let plan = Ltx25TrainingPlan::from_request(req)?;
        let cfg = &req.config;
        let compute_dtype = Dtype::Float32;
        on_progress(TrainingProgress::Preparing);
        // sc-24830 (epic 2123 E7 + E8): depth anchoring on the generated video stream — the
        // memory guard (decoded frames × the prepared video grid) and the frozen models load
        // BEFORE caching, so a missing checkpoint or an over-budget run fails fast.
        let depth_frames = if mlx_gen_perceptual::any_aux_loss(cfg) {
            let k = depth_anchoring_frames(cfg)?;
            let (f, h, w) = ltx25_video_grids(&req.items)?;
            let aux_gb = ltx25_perceptual_footprint_gb(
                cfg,
                h,
                w,
                k.min(f.max(1) as usize) as u32,
                req.items.len(),
            );
            ltx25_aux_memory_guard(cfg, aux_gb, get_memory_limit())?;
            Some(k)
        } else {
            None
        };
        let perceptual = load_perceptual_path_for(cfg, "ltx_2_5 trainer", ltx25_decoder())?;
        on_progress(TrainingProgress::LoadingModel);

        let mut cached = Vec::with_capacity(req.items.len());
        let mut sample_contexts: Vec<(String, Array, Array)> = Vec::new();
        let mut negative_contexts: Option<(Array, Array)> = None;
        {
            let encoder = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg("ltx_2_5 trainer: text encoder missing".into())
            })?;
            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| mlx_gen::Error::Msg("ltx_2_5 trainer: tokenizer missing".into()))?;
            for (index, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                on_progress(TrainingProgress::Caching {
                    current: index as u32 + 1,
                    total: req.items.len() as u32,
                });
                let (ids, mask) = tokenizer.encode(&item.caption, MAX_PROMPT_TOKENS)?;
                let (video_context, audio_context) = encoder.encode_av(&ids, &mask)?;
                let audio_context = audio_context.ok_or_else(|| {
                    mlx_gen::Error::Msg(
                        "ltx_2_5 trainer: Gemma-4 encoder has no audio connector".into(),
                    )
                })?;
                let video_context = to_dtype(&video_context, Dtype::Float32)?;
                let audio_context = to_dtype(&audio_context, Dtype::Float32)?;
                let example = load_prepared_example(
                    item,
                    &plan,
                    &video_context,
                    Some(&audio_context),
                    cfg.seed,
                    index as u64,
                    depth_frames,
                )?;
                eval([&video_context, &audio_context])?;
                cached.push(example);
            }
            if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled()
            {
                for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                    let (ids, mask) = tokenizer.encode(prompt, MAX_PROMPT_TOKENS)?;
                    let (video, audio) = encoder.encode_av(&ids, &mask)?;
                    let video = to_dtype(&video, Dtype::Float32)?;
                    let audio = to_dtype(
                        &audio.ok_or_else(|| {
                            mlx_gen::Error::Msg(
                                "ltx_2_5 trainer: validation requires audio text context".into(),
                            )
                        })?,
                        Dtype::Float32,
                    )?;
                    eval([&video, &audio])?;
                    sample_contexts.push((prompt.clone(), video, audio));
                }
                let (ids, mask) = tokenizer.encode("", MAX_PROMPT_TOKENS)?;
                let (video, audio) = encoder.encode_av(&ids, &mask)?;
                let video = to_dtype(&video, Dtype::Float32)?;
                let audio = to_dtype(
                    &audio.ok_or_else(|| {
                        mlx_gen::Error::Msg(
                            "ltx_2_5 trainer: validation requires negative audio context".into(),
                        )
                    })?,
                    Dtype::Float32,
                )?;
                eval([&video, &audio])?;
                negative_contexts = Some((video, audio));
            }
        }
        if cached.is_empty() {
            return if req.cancel.is_cancelled() {
                Err(mlx_gen::Error::Canceled)
            } else {
                Err("ltx_2_5 trainer: no usable prepared examples".into())
            };
        }
        self.text_encoder = None;
        self.tokenizer = None;

        let TrainingTransformer::Ltx25(transformer) = &mut self.transformer else {
            unreachable!("dispatcher selected the 2.5 path")
        };
        let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
            DEFAULT_TARGET_SUFFIXES
                .iter()
                .map(|value| value.to_string())
                .collect()
        } else {
            cfg.lora_target_modules.clone()
        };
        let (targets, mut params) = build_targets_25(
            transformer,
            self.cfg.num_layers,
            &suffixes,
            cfg.rank as i32,
            cfg.seed,
        )?;
        if targets.is_empty() {
            return Err(
                "ltx_2_5 trainer: no AV LoRA targets resolved (check lora_target_modules)".into(),
            );
        }
        // AV whole-block checkpointing is not yet a separate path; the attention-segment checkpoint
        // still bounds all six attention score graphs while preserving the full AV forward.
        transformer.set_sdpa_checkpoint(true);

        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut optimizer =
            TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;
        let accumulation = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accumulation, cfg.lr_warmup_steps);
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("lora")
            .to_string();
        let mut update_index = 0;
        let mut start_step = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, metadata) = checkpoint::load_resume(&snapshot, &mut optimizer)?;
                params = loaded;
                start_step = metadata.step;
                update_index = metadata.update_idx;
            }
        }

        let mae = matches!(cfg.loss_type.to_ascii_lowercase().as_str(), "mae" | "l1");
        // sc-24830: each example's depth reference (its clean video frames decoded → DA2) once,
        // and the alternation over the round-robin windows (a pure function of the step, so a
        // resumed run needs no replay).
        let mut aux_driver = match perceptual {
            Some(path) => Some(ltx25_aux_driver(
                path,
                &cached,
                accumulation,
                &Default::default(),
            )?),
            None => None,
        };
        let mut accumulated: Option<LoraParams> = None;
        let mut final_loss = 0.0;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let mut sigma = {
                let key =
                    random::key(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64))?;
                random::uniform::<_, f32>(1e-3, 1.0 - 1e-3, &[1], Some(&key))?.item::<f32>()
            };
            let item = (step as usize - 1) % cached.len();
            let example = &cached[item];
            // sc-24830: plan the step; an aux-only step trains at σ remapped into the loss window.
            let step_plan = match aux_driver.as_mut() {
                Some(driver) => {
                    let key = driver.key(step);
                    let plan = driver.path().plan(key, item, sigma)?;
                    sigma = plan.noise_level;
                    Some(plan)
                }
                _ => None,
            };
            let prepared = prepare_cached_ltx25(
                example,
                sigma,
                cfg.seed.wrapping_add(step as u64).wrapping_mul(2),
            )?;
            let aux = match (
                aux_driver.as_ref().map(AuxDriver::path),
                step_plan.as_ref(),
                example.depth.as_ref(),
            ) {
                (Some(path), Some(plan), Some(depth)) => Some(Ltx25AuxStep {
                    path,
                    plan,
                    image: item,
                    depth,
                }),
                (Some(_), Some(_), None) => {
                    return Err("ltx_2_5 trainer: depth anchoring has no video frames".into())
                }
                _ => None,
            };
            let (losses, grads) = compute_ltx25_step_loss_grads(
                transformer,
                &params,
                &targets,
                cfg.alpha,
                cfg.rank as f32,
                &prepared,
                mae,
                compute_dtype,
                aux,
            )?;
            final_loss = losses.total;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;
            if step % accumulation == 0 || step == cfg.steps {
                optimizer.set_lr_scaled(lr_multiplier(
                    cfg.lr_scheduler,
                    update_index,
                    total_updates,
                    warmup_updates,
                ));
                let window = if step % accumulation == 0 {
                    accumulation
                } else {
                    step % accumulation
                };
                let average = average_grads(
                    accumulated.take().expect("update requires gradients"),
                    window,
                )?;
                // Epic 2123 (sc-24827): clip → gradient noise → step → weight noise.
                adapter_optimizer_update(
                    &mut optimizer,
                    &mut params,
                    &average,
                    cfg,
                    update_index,
                    cfg.seed,
                )?;
                update_index += 1;
            }
            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: final_loss,
            });
            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                save_lora(
                    &params,
                    &targets,
                    cfg.alpha,
                    cfg.rank,
                    &req.output_dir.join(checkpoint_filename(&stem, step)),
                )?;
                checkpoint::save_resume(
                    &req.output_dir,
                    &stem,
                    step,
                    update_index,
                    &optimizer,
                    &params,
                )?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            if cfg.sample_every > 0 && step % cfg.sample_every == 0 && !sample_contexts.is_empty() {
                install_train_lora(
                    transformer,
                    &params,
                    &targets,
                    cfg.alpha,
                    cfg.rank as f32,
                    None,
                )?;
                let negatives = negative_contexts.as_ref().ok_or_else(|| {
                    mlx_gen::Error::Msg(
                        "ltx_2_5 trainer: negative validation context missing".into(),
                    )
                })?;
                for (index, (prompt, video_context, audio_context)) in
                    sample_contexts.iter().enumerate()
                {
                    let image = render_ltx25_validation_sample(
                        transformer,
                        &self.vae,
                        video_context,
                        audio_context,
                        &negatives.0,
                        &negatives.1,
                        &plan.validation,
                        cfg.seed
                            .wrapping_add(step as u64)
                            .wrapping_add(index as u64),
                        &req.cancel,
                    )?;
                    on_progress(TrainingProgress::Sample {
                        step,
                        index: index as u32 + 1,
                        total: sample_contexts.len() as u32,
                        prompt: prompt.clone(),
                        image,
                    });
                }
            }
        }
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_lora(&params, &targets, cfg.alpha, cfg.rank, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss,
        })
    }
}

/// Flatten a single-frame VAE latent `(1, 128, 1, le, le)` to the patchified `(1, S, 128)` the DiT
/// consumes (`S = le·le`) — the reference's `transpose(reshape(latent, (B, C, -1)), (0, 2, 1))`.
fn flatten_latent(latent: &Array) -> Result<Array> {
    let sh = latent.shape(); // [1, 128, 1, le, le]
    let (b, c) = (sh[0], sh[1]);
    let s = sh[2] * sh[3] * sh[4];
    let flat = latent.reshape(&[b, c, s])?; // (1, 128, S)
    Ok(flat.transpose_axes(&[0, 2, 1])?) // (1, S, 128)
}

/// The inverse of [`flatten_latent`] for a square single-frame latent: the patchified
/// `(1, S, 128)` with `S = latent_edge²` back to the decoder's NCHW `(1, 128, le, le)` (the
/// per-channel-normalized space `LtxVideoVae::encode` produces).
fn unflatten_latent(flat: &Array, latent_edge: usize) -> Result<Array> {
    let sh = flat.shape();
    let le = latent_edge as i32;
    if sh.len() != 3 || sh[1] != le * le {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_3 trainer: expected a patchified (B, {}, C) still latent, got {sh:?}",
            le * le
        )));
    }
    Ok(flat
        .transpose_axes(&[0, 2, 1])?
        .reshape(&[sh[0], sh[2], le, le])?)
}

/// One LTX-2.3 item's cached latents, one per bucket edge in `edges` order (the item-major layout
/// [`BucketSchedule::cache_index`] indexes, sc-2127): `encode(edge)` the unpatchified
/// `(1, 128, 1, le, le)` latent, `mask_weight(latent.shape())` — the item's subject mask resampled
/// onto **that** latent's grid (sc-24828) — then both patchified by the same [`flatten_latent`] →
/// `(1, S, 128)` so the weight lines up token-for-token with that bucket's clean latent; each pair
/// evaluated so it survives the encoder's drop.
fn encode_bucket_latents(
    edges: &[u32],
    mut encode: impl FnMut(u32) -> Result<Array>,
    mut mask_weight: impl FnMut(&[i32]) -> Result<Option<Array>>,
) -> Result<Vec<(Array, Option<Array>)>> {
    edges
        .iter()
        .map(|&edge| {
            let latent = encode(edge)?;
            let weight = mask_weight(latent.shape())?
                .map(|w| flatten_latent(&w))
                .transpose()?;
            let clean = flatten_latent(&latent)?;
            eval(std::iter::once(&clean).chain(weight.as_ref()))?;
            Ok((clean, weight))
        })
        .collect()
}

/// The subject-masked loss weight (sc-24828) of one unpatchified still latent
/// `(1, 128, 1, le, le)`: the image went through `center_crop_square`, so the mask's crop is
/// [`CropBox::center_square`], area-averaged onto the latent's `(le, le)` grid and broadcast over
/// every channel — the latent's exact shape. `None` when off.
fn still_subject_weight(
    label: &str,
    mask: Option<&PreparedSubjectMask>,
    latent_shape: &[i32],
) -> Result<Option<Array>> {
    prepared_subject_mask_weight(label, mask, CropBox::center_square, latent_shape)
}

/// `to_out.0` → `to_out`, the only diffusers→checkpoint rename in the attention LoRA surface (the
/// inference loader does the same in `adapters::normalize`); other suffixes pass through.
fn resolve_segments(save_path: &str) -> Vec<String> {
    crate::adapters::normalize_ltx_key(save_path)
        .split('.')
        .map(String::from)
        .collect()
}

const LTX25_ATTENTION_MODULES: [&str; 6] = [
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
];

/// Expand the reference's suffix matching against the complete AV model.  Short attention suffixes
/// intentionally hit all six attention modules; FF patterns remain modality-specific exactly as in
/// the upstream YAML. No bias target is manufactured: `ff_bias=false` means the 2.5 video FF base
/// legitimately has no bias while its two weight projections remain trainable.
fn ltx25_target_paths(num_layers: i32, suffixes: &[String]) -> Vec<String> {
    let mut paths = Vec::new();
    let push_unique = |paths: &mut Vec<String>, path: String| {
        if !paths.contains(&path) {
            paths.push(path);
        }
    };
    for i in 0..num_layers {
        for suffix in suffixes {
            match suffix.as_str() {
                "to_q" | "to_k" | "to_v" | "to_out.0" | "to_gate_logits" => {
                    for module in LTX25_ATTENTION_MODULES {
                        push_unique(
                            &mut paths,
                            format!("transformer_blocks.{i}.{module}.{suffix}"),
                        );
                    }
                }
                "ff.net.0.proj" | "ff.net.2" => {
                    push_unique(&mut paths, format!("transformer_blocks.{i}.{suffix}"));
                    push_unique(&mut paths, format!("transformer_blocks.{i}.audio_{suffix}"));
                }
                "ff" => {
                    for projection in ["ff.net.0.proj", "ff.net.2"] {
                        push_unique(&mut paths, format!("transformer_blocks.{i}.{projection}"));
                        push_unique(
                            &mut paths,
                            format!("transformer_blocks.{i}.audio_{projection}"),
                        );
                    }
                }
                "audio_ff.net.0.proj" | "audio_ff.net.2" => {
                    push_unique(&mut paths, format!("transformer_blocks.{i}.{suffix}"));
                }
                qualified if qualified.contains('.') => {
                    push_unique(&mut paths, format!("transformer_blocks.{i}.{qualified}"));
                }
                _ => {}
            }
        }
    }
    paths
}

fn build_targets_from_paths(
    dit: &mut impl LtxAdaptable,
    paths: impl IntoIterator<Item = String>,
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    let mut targets = Vec::new();
    let mut params = LoraParams::new();
    let small = Array::from_slice(&[0.02f32], &[1]);
    for (idx, save_path) in paths.into_iter().enumerate() {
        let segs = resolve_segments(&save_path);
        let seg_refs: Vec<&str> = segs.iter().map(String::as_str).collect();
        let Some(lin) = dit.adaptable_mut(&seg_refs) else {
            continue;
        };
        let shape = lin.base_shape();
        let (out_f, in_f) = (shape[0], shape[1]);
        let a_key: Rc<str> = Rc::from(format!("{save_path}.lora_a"));
        let b_key: Rc<str> = Rc::from(format!("{save_path}.lora_b"));
        let ka = random::key(seed.wrapping_add(2 * idx as u64 + 1))?;
        let a = multiply(
            &random::normal::<f32>(&[rank, in_f], None, None, Some(&ka))?,
            &small,
        )?;
        let b = Array::zeros::<f32>(&[out_f, rank])?;
        eval([&a, &b])?;
        params.insert(a_key.clone(), a);
        params.insert(b_key.clone(), b);
        targets.push(LtxLoraTarget {
            save_path,
            segs,
            a_key,
            b_key,
        });
    }
    Ok((targets, params))
}

fn build_targets_25(
    dit: &mut AvDiT,
    num_layers: i32,
    suffixes: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    build_targets_from_paths(dit, ltx25_target_paths(num_layers, suffixes), rank, seed)
}

/// Enumerate the `attn1`/`attn2` × `suffixes` targets across the DiT's `num_layers` blocks, resolve
/// each on the (mutable) DiT, read its `[out,in]` base shape, and initialise the trainable factors
/// the reference `_MlxLoRALinear` way — `A ~ N(0, 0.02)` `[rank,in]`, `B = 0` `[out,rank]` — keyed
/// `{save_path}.lora_a` / `.lora_b`. Targets that do not resolve (a missing gated branch, a typo'd
/// suffix) are skipped.
fn build_targets(
    dit: &mut LtxDiT,
    num_layers: i32,
    suffixes: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    let mut paths = Vec::new();
    for i in 0..num_layers {
        for attn in ["attn1", "attn2"] {
            for suf in suffixes {
                paths.push(format!("transformer_blocks.{i}.{attn}.{suf}"));
            }
        }
    }
    build_targets_from_paths(dit, paths, rank, seed)
}

/// Inject the current trainable factors as one LoRA residual per target via the LTX training seam —
/// transpose `[r,in]`→`[in,r]` and `[out,r]`→`[r,out]`, fold `alpha/rank` into `b` — so the residual
/// is `(x·Aᵀ·Bᵀ)·(alpha/rank)`, matching the reference `_MlxLoRALinear`. Differentiable.
fn install_train_lora(
    dit: &mut impl LtxAdaptable,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    lora_dtype: Option<Dtype>,
) -> MlxResult<()> {
    for t in targets {
        let a = params[&t.a_key].t(); // [r,in] -> [in,r]
        let b = params[&t.b_key]
            .t()
            .multiply(Array::from_slice(&[alpha / rank], &[1]))?; // [out,r] -> [r,out] · (α/r)

        // sc-4942 — under the bf16 training cast the f32 factors must join the bf16 stream, or every
        // adapted Linear re-promotes its block to f32 (defeating the activation saving). No-op in f32.
        let (a, b) = match lora_dtype {
            Some(dt) => (a.as_dtype(dt)?, b.as_dtype(dt)?),
            None => (a, b),
        };
        let seg_refs: Vec<&str> = t.segs.iter().map(String::as_str).collect();
        let lin = dit
            .adaptable_mut(&seg_refs)
            .ok_or_else(|| Exception::custom(format!("LoRA target not found: {}", t.save_path)))?;
        lin.set_train_lora(a, b);
    }
    Ok(())
}

/// Group resolved targets by their owning block (sc-4942) — `block_targets[i]` lists block `i`'s
/// trainable LoRA targets as the block-local path (`segs` minus the `transformer_blocks.{i}` prefix)
/// plus the factor-map keys, for the gradient-checkpoint closure. Every target lives in a
/// `transformer_blocks.{i}.attn{1,2}` leaf, so the grouping is exhaustive.
fn group_block_targets(targets: &[LtxLoraTarget], n_layers: usize) -> Vec<Vec<BlockLoraRef>> {
    let mut out: Vec<Vec<BlockLoraRef>> = (0..n_layers).map(|_| Vec::new()).collect();
    for t in targets {
        // segs = ["transformer_blocks", "{i}", attn, suffix...]; the block-local path is segs[2..].
        if t.segs.len() < 3 || t.segs[0] != "transformer_blocks" {
            continue;
        }
        let Ok(i) = t.segs[1].parse::<usize>() else {
            continue;
        };
        if i >= n_layers {
            continue;
        }
        out[i].push(BlockLoraRef {
            local: t.segs[2..].to_vec(),
            a_key: t.a_key.to_string(),
            b_key: t.b_key.to_string(),
        });
    }
    out
}

/// One forward+backward over the trainable factors: build the rectified-flow input `x_t`, inject the
/// factors, run the video DiT, regress the raw velocity toward `noise - clean`, return `(loss, grads)`.
///
/// `checkpoint_block`, when `Some`, lists each block's trainable targets and switches the forward to
/// the gradient-checkpointed path (sc-4942) — each block recomputes its activations in the backward
/// instead of retaining them. `None` runs the dense (attention-segment-checkpointed) forward.
/// `dtype` is the training compute dtype (sc-4942): for bf16 the DiT weights were cast once in
/// `train_impl` and `preprocess` casts the activation stream, so the LoRA factors are cast at install
/// (here and inside the checkpoint segment) to keep the whole graph bf16; the noising / loss / grads
/// stay f32.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &mut LtxDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    clean: &Array,
    context: &Array,
    positions: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    checkpoint_block: Option<&[Vec<BlockLoraRef>]>,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) = compute_step_loss_grads(
        dit,
        params,
        targets,
        alpha,
        rank,
        clean,
        context,
        positions,
        sigma,
        noise,
        mae,
        mask_weight,
        checkpoint_block,
        dtype,
        None,
    )?;
    Ok((losses.total, grads))
}

/// [`compute_loss_grads`] with the step's perceptual plan (epic 2123 E8): on an aux-only step the
/// diffusion term is not computed and the loss is the weighted perceptual term on the DiT's x0
/// estimate `x_t − σ·v` (the raw output regresses `noise − clean`), unpatchified to the decoder's
/// NCHW layout and decoded by TAELTX2.3; summed (`every_n == 1`) aux terms ride along on a
/// diffusion step. The dense and block-checkpointed forwards land in the same closure, so both
/// carry it. With `aux = None` (or a plan with no aux loss) the traced graph is exactly the
/// pre-epic-2123 one.
#[allow(clippy::too_many_arguments)]
fn compute_step_loss_grads(
    dit: &mut LtxDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    clean: &Array,
    context: &Array,
    positions: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    checkpoint_block: Option<&[Vec<BlockLoraRef>]>,
    dtype: Dtype,
    aux: Option<AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    // x_t = (1-σ)·clean + σ·noise; target = noise - clean (the raw-output velocity); timestep = σ.
    // x_t / context stay f32 here; `preprocess` casts the activation stream to the compute dtype.
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = add(&multiply(clean, &one_minus)?, &multiply(noise, &s)?)?;
    let x_t_f32 = x_t.clone();
    let target = subtract(noise, clean)?;
    let timestep = Array::from_slice(&[sigma], &[1, 1]); // (B, 1), broadcast over tokens
    let ctx = context.clone();
    let pos = positions.clone();
    let mask_weight = mask_weight.cloned();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        let v = match checkpoint_block {
            Some(bt) => dit
                .forward_with_main_checkpointed(&x_t, &timestep, &ctx, None, &pos, &p, bt, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => {
                install_train_lora(dit, &p, targets, alpha, rank, lora_dtype)?;
                // `None`: content-keyed RoPE memo (the per-stage epoch fast path is inference-only;
                // training positions are constant within a step, so the content compare hits — sc-7141).
                dit.forward(&x_t, &timestep, &ctx, None, &pos, None)
                    .map_err(|e| Exception::custom(e.to_string()))?
            }
        };
        let diffusion = if diffusion_on {
            let diff = subtract(&v, &target)?;
            // MSE / MAE, subject-mask weighted when on (sc-24828) — reduces to a 0-d scalar (grad
            // requires a scalar cotangent). Dense and block-checkpointed forwards both land here.
            Some(reduce_loss(&diff, mask_weight.as_ref(), mae)?)
        } else {
            None
        };
        let aux_term = match &aux {
            Some(a) if aux_on => {
                // x0 estimate in f32 from the raw velocity (`noise − clean`): x0 = x_t − σ·v.
                let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma }
                    .recover_x0(&x_t_f32, &v.as_dtype(Dtype::Float32)?)
                    .and_then(|x| unflatten_latent(&x, a.latent_edge))
                    .map_err(|e| Exception::custom(e.to_string()))?;
                a.path
                    .aux_loss(a.plan, a.image, &x0_hat)
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

fn masked_modality_loss(
    prediction: &Array,
    modality: &Ltx25PreparedModality,
    mae: bool,
) -> MlxResult<Array> {
    if modality.loss_denominator <= 0.0 {
        return Err(Exception::custom(
            "ltx_2_5 trainer: generated modality has an empty loss mask",
        ));
    }
    let diff = subtract(prediction, &modality.target)?;
    let point = if mae { diff.abs()? } else { diff.square()? };
    let masked = multiply(&point, &modality.loss_mask)?;
    divide(
        &masked.sum(None)?,
        Array::from_f32(modality.loss_denominator),
    )
}

/// Full flexible-strategy AV loss: each present modality has its own target and loss mask; frozen
/// modalities participate in cross-modal attention but contribute zero loss. Audio-only and
/// video-only workflows execute only their stream, matching upstream's optional modalities.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn compute_ltx25_loss_grads(
    dit: &mut AvDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    batch: &Ltx25PreparedBatch,
    mae: bool,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) =
        compute_ltx25_step_loss_grads(dit, params, targets, alpha, rank, batch, mae, dtype, None)?;
    Ok((losses.total, grads))
}

/// One LTX-2.5 aux-loss step's view of the perceptual path (sc-24830): the plan, the example's
/// reference key and its depth frame selection.
struct Ltx25AuxStep<'a> {
    path: &'a PerceptualPath,
    plan: &'a StepPlan,
    /// The example index (selects its cached reference).
    image: usize,
    depth: &'a Ltx25DepthFrames,
}

/// The depth x0 estimate of the generated video stream (sc-24830): over the target tokens (the
/// appended reference tokens dropped), `x0 = x_t − t·v` with the per-token timestep `t` (the
/// planned σ on generated tokens, `0` on conditioning tokens — whose x0 is then their clean
/// latent), the velocity regressing `noise − clean`; the selected frames as the decoder's NCHW.
fn ltx25_video_x0(
    prediction: &Array,
    video: &Ltx25PreparedModality,
    depth: &Ltx25DepthFrames,
) -> Result<Array> {
    use mlx_rs::ops::indexing::IndexOp;
    let tokens = video.target_tokens as i32;
    let noisy = video.noisy.index((.., ..tokens));
    let timestep = video.timestep.index((.., ..tokens)).expand_dims(2)?;
    let velocity = prediction.index((.., ..tokens)).as_dtype(Dtype::Float32)?;
    let x0 = subtract(&noisy, &multiply(&timestep, &velocity)?)?;
    depth.nchw(&x0)
}

/// [`compute_ltx25_loss_grads`] with the step's perceptual plan (sc-24830): on an aux-only step the
/// (video + audio) velocity term is not computed and the loss is the weighted depth term on the
/// generated video's x0 estimate ([`ltx25_video_x0`]); summed (`every_n == 1`) aux terms ride along
/// on a diffusion step. With `aux = None` (or a plan with no aux loss) the traced graph is exactly
/// the pre-epic-2123 one.
#[allow(clippy::too_many_arguments)]
fn compute_ltx25_step_loss_grads(
    dit: &mut AvDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    batch: &Ltx25PreparedBatch,
    mae: bool,
    dtype: Dtype,
    aux: Option<Ltx25AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    if batch.video.is_none() && batch.audio.is_none() {
        return Err("ltx_2_5 trainer: prepared batch has no modalities".into());
    }
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    if aux_on && batch.video.is_none() {
        return Err("ltx_2_5 trainer: depth anchoring needs the generated video stream".into());
    }
    let prepared = batch.clone();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        install_train_lora(dit, &p, targets, alpha, rank, lora_dtype)?;
        let mut video_prediction_out: Option<Array> = None;
        let diffusion = match (&prepared.video, &prepared.audio) {
            (Some(video), Some(audio)) => {
                let (video_prediction, audio_prediction) = dit
                    .forward(
                        &video.noisy,
                        &video.timestep,
                        &video.context,
                        None,
                        &video.positions,
                        &audio.noisy,
                        &audio.timestep,
                        &audio.context,
                        None,
                        &audio.positions,
                        None,
                        None,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let diffusion = if diffusion_on {
                    let video_loss = (video.loss_denominator > 0.0)
                        .then(|| masked_modality_loss(&video_prediction, video, mae))
                        .transpose()?;
                    let audio_loss = (audio.loss_denominator > 0.0)
                        .then(|| masked_modality_loss(&audio_prediction, audio, mae))
                        .transpose()?;
                    Some(match (video_loss, audio_loss) {
                        (Some(video), Some(audio)) => add(&video, &audio)?,
                        (Some(loss), None) | (None, Some(loss)) => loss,
                        (None, None) => {
                            return Err(Exception::custom(
                                "ltx_2_5 trainer: both modality loss masks are empty",
                            ))
                        }
                    })
                } else {
                    None
                };
                video_prediction_out = Some(video_prediction);
                diffusion
            }
            (Some(video), None) => {
                let prediction = dit
                    .forward_video_only(
                        &video.noisy,
                        &video.timestep,
                        &video.context,
                        None,
                        &video.positions,
                        None,
                        None,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let diffusion = diffusion_on
                    .then(|| masked_modality_loss(&prediction, video, mae))
                    .transpose()?;
                video_prediction_out = Some(prediction);
                diffusion
            }
            (None, Some(audio)) => {
                let prediction = dit
                    .forward_audio_only(
                        &audio.noisy,
                        &audio.timestep,
                        &audio.context,
                        None,
                        &audio.positions,
                        None,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                diffusion_on
                    .then(|| masked_modality_loss(&prediction, audio, mae))
                    .transpose()?
            }
            (None, None) => unreachable!(),
        };
        let aux_term = match (&aux, &prepared.video, &video_prediction_out) {
            (Some(a), Some(video), Some(prediction)) if aux_on => {
                let x0 = ltx25_video_x0(prediction, video, a.depth)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                a.path
                    .aux_loss(a.plan, a.image, &x0)
                    .map_err(|e| Exception::custom(e.to_string()))?
                    .map(|t| t.weighted)
            }
            _ => None,
        };
        let zero = || Array::from_f32(0.0);
        let d_out = diffusion.clone().unwrap_or_else(zero);
        let a_out = aux_term.clone().unwrap_or_else(zero);
        let total =
            combine_step_loss(diffusion, aux_term).map_err(|e| Exception::custom(e.to_string()))?;
        Ok(vec![total, d_out, a_out])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (value, grads) = vg(params.clone(), 0)?;
    let losses = StepLosses {
        total: value[0].item::<f32>(),
        diffusion: diffusion_on.then(|| value[1].item::<f32>()),
        aux: aux_on.then(|| value[2].item::<f32>()),
    };
    Ok((losses, grads))
}

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the LTX
/// latent token count `s` (the trainer trains single-frame still latents, so `s = (edge/32)²`). With
/// attention-segment checkpointing always on (sc-4942) the seq² attention term is demoted to a single
/// layer's backward transient, so the measured f32 curve is essentially **linear** in `s`: the
/// constant is the resident base (the Q-packed 22B DiT after the Gemma TE is freed) plus the
/// f32-activation working set, the slope the per-token retained hidden activations across the 48
/// blocks.
///
/// CALIBRATED from `first_step_attn_ckpt_sweep` (128 GB Mac, rank 8 / 384 targets / batch 1, f32):
/// s=256 → 23.3 GB, s=576 → 31.5 GB, s=1024 → 42.6 GB (fit error < 0.2 GB). Refit if that harness
/// prints materially different numbers. (LTX trains f32 only — see the `compute_dtype` note in
/// `train_impl`; there is no bf16 production path to size.)
fn projected_dense_peak_gb(s: f64) -> f64 {
    16.9 + 0.0251 * s
}

/// Latent tokens per side for a bucketed pixel `edge` (the VAE's ×32 spatial stride, floor 1).
fn latent_edge_of(edge: u32) -> usize {
    (edge / SPATIAL_SCALE as u32).max(1) as usize
}

/// The latent edge the pre-flight guard and previews size for: the largest bucket's (epic 2123 E7).
fn guard_latent_edge(edges: &[u32]) -> usize {
    edges.iter().map(|&e| latent_edge_of(e)).max().unwrap_or(1)
}

/// The item-major cache entry and its bucket (whose RoPE grid the step uses) for 1-based `step`
/// (sc-2127). With one bucket the entry is the pre-bucket `(step-1) % n_items` and the bucket is 0.
fn step_entry(schedule: &BucketSchedule, step: u32) -> (usize, usize) {
    let entry = schedule.cache_index((step - 1) as usize);
    (entry, entry % schedule.n_buckets())
}

/// The bucket index of the largest edge in `edges` (the first on a tie; `0` for an empty list) —
/// the bucket training previews render at (sc-2127).
fn largest_bucket(edges: &[usize]) -> usize {
    edges
        .iter()
        .enumerate()
        .fold(0, |best, (i, &e)| if e > edges[best] { i } else { best })
}

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead (sc-4942 — the sc-4874 mechanism).
/// `latent_edge` is the latent tokens per side (`edge/32`); the token count is `latent_edge²` (the
/// trainer trains single-frame still latents). The budget is MLX's reported memory limit (≈ the
/// device's recommended working set) × 0.85 for worker/host headroom. Only consulted when gradient
/// checkpointing is OFF. With resolution buckets the caller passes the LARGEST bucket's latent edge
/// (sc-2127 / epic 2123 E7): the per-step working set peaks at the biggest latent, and the cached
/// still latents (KBs–MBs per item) are not part of the projection.
///
/// Epic 2123 E7 (sc-24830): `extra_gb` is the training-time auxiliary models' footprint
/// ([`perceptual_footprint_gb`]); with it the guard also runs with gradient checkpointing on
/// (`checkpointed`), stacking on the resident base ([`checkpointed_baseline_gb`]).
fn preflight_memory_guard(
    cfg: &TrainingConfig,
    latent_edge: usize,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    preflight_memory_guard_with_budget(cfg, latent_edge, extra_gb, checkpointed, get_memory_limit())
}

/// The checkpointed baseline the auxiliary-model guard stacks on: the resident-base term of
/// [`projected_dense_peak_gb`] (block checkpointing removes most of the activation term; the base
/// stays). A lower bound.
fn checkpointed_baseline_gb() -> f64 {
    projected_dense_peak_gb(0.0)
}

/// [`preflight_memory_guard`] against an explicit budget (`budget_bytes`, the live MLX limit in
/// production) — so the guard's arithmetic is testable on any host. A checkpointed refusal goes
/// through the shared [`mlx_gen_perceptual::check_aux_memory`], naming `cfg`'s enabled aux losses.
fn preflight_memory_guard_with_budget(
    cfg: &TrainingConfig,
    latent_edge: usize,
    extra_gb: f64,
    checkpointed: bool,
    budget_bytes: usize,
) -> Result<()> {
    let s = (latent_edge * latent_edge) as f64;
    let projected = if checkpointed {
        checkpointed_baseline_gb()
    } else {
        projected_dense_peak_gb(s)
    } + extra_gb;
    let budget_gb = budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if checkpointed {
        let px = latent_edge * SPATIAL_SCALE as usize;
        return mlx_gen_perceptual::check_aux_memory(
            "ltx_2_3 trainer",
            cfg,
            &format!("a checkpointed training step at resolution {px}"),
            extra_gb,
            projected,
            safe,
            &format!("{budget_gb:.0} GB MLX limit × 0.85"),
        );
    }
    if projected > safe {
        let px = latent_edge * SPATIAL_SCALE as usize;
        return Err(mlx_gen_perceptual::name_aux_losses(
            format!(
                "ltx_2_3 trainer: a dense first training step at resolution {px} needs ~{projected:.0} GB \
                 (the forward working set materializes in one allocation), exceeding this machine's ~{safe:.0} GB \
                 safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS would hard-kill the \
                 worker (SIGKILL) at the first step with no recoverable error (sc-4874/sc-4942). Enable Gradient \
                 Checkpointing (recomputes block activations in the backward) or reduce the training resolution."
            )
            .into(),
            cfg,
            extra_gb,
        ));
    }
    Ok(())
}

/// Write the trained LoRA as safetensors keyed by the LTX module paths — `{module}.lora_A.weight`
/// `[rank,in]`, `{module}.lora_B.weight` `[out,rank]`, scalar `{module}.alpha` (= `alpha`) — the
/// reference `_save_lora` format, reloadable by [`crate::apply_ltx_adapters`] (which folds
/// `scale = alpha/rank`). `networkType`/`rank`/`alpha` metadata mirrors the other family trainers.
fn save_lora(
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: u32,
    path: &Path,
) -> Result<()> {
    let alphas: Vec<(String, Array)> = targets
        .iter()
        .map(|t| {
            (
                format!("{}.alpha", t.save_path),
                Array::from_slice(&[alpha], &[1]),
            )
        })
        .collect();
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        entries.push((format!("{}.lora_A.weight", t.save_path), &params[&t.a_key]));
        entries.push((format!("{}.lora_B.weight", t.save_path), &params[&t.b_key]));
    }
    for (k, v) in &alphas {
        entries.push((k.clone(), v));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".to_string(), "lora".to_string());
    meta.insert("rank".to_string(), rank.to_string());
    meta.insert("alpha".to_string(), alpha.to_string());
    // The split LTX-2.5 provider consumes this explicit file-wide contract and deliberately does
    // not infer a rank from a factor.  Retain the legacy keys above so LTX-2.3/third-party loaders
    // remain byte-for-byte compatible while a newly trained adapter can be selected by either route.
    meta.insert("lora_rank".to_string(), rank.to_string());
    meta.insert("lora_alpha".to_string(), alpha.to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
}

/// Decode an image file (PNG/JPEG) into the core RGB8 [`Image`].
fn decode_image(path: &Path) -> Result<Image> {
    let dynimg = mlx_gen::gen_core::__model_input_result!(
        image::open(path),
        "crates/media/mlx-gen/mlx-gen-ltx/src/training.rs::decode_image/image::open",
        Reference,
        &[path]
    )
    .map_err(|e| mlx_gen::Error::Msg(format!("decode image {}: {e}", path.display())))?;
    Ok(dynimg.to_rgb8().into())
}

// ===========================================================================================
// sc-4942 — first-step memory + grad-parity harness (weight-gated, run as its own process).
//
// Ports the z-image sc-4874/4886/4887 `first_step_repro` harness to LTX: drives the exact inner
// training step (`compute_loss_grads` + the step-1 grad `eval`) directly, sweeping resolution with
// MLX peak-memory probes, and asserts the three levers' invariants on REAL weights:
//   * attention-segment checkpointing is bit-identical to the retained backward,
//   * block (gradient) checkpointing matches the dense path within fp tolerance,
//   * bf16 grads point the same way as f32 and materially shrink the working set.
//
//   cargo test -p mlx-gen-ltx --release --lib first_step -- --ignored --nocapture
// ===========================================================================================
mod first_step_repro;

mod bucket_tests;

mod preflight_tests;

mod validate_request_tests;

mod ltx25_tiny_production_lifecycle;

mod load_trainer_tests;

/// sc-24830 (epic 2123 depth anchoring) — the LTX-2.3 step seam on a tiny synthetic video DiT
/// (`rung4_block_window_tests::tiny_cfg` / `tiny_weight_map`: 4 layers, 4 latent channels) with a
/// random-init tiny-width TAEHV carrying TAELTX2.3's hyperparameters (patch 4, 8× temporal) at 4
/// latent channels, and a random-init tiny Depth-Anything-V2. Drives the same
/// [`compute_step_loss_grads`] / [`aux_driver`]
/// `train_23_impl` runs; the LTX-2.5 refusal through its typed reason. CPU; no weights downloaded.
mod depth_anchoring_tests;

/// sc-24830 — LTX-2.5 depth anchoring on the tiny synthetic AV DiT
/// (`rung4_block_window_tests::tiny_cfg`: 4 latent channels, video 2 frames × 2 × 2 tokens + audio)
/// with a random-init tiny-width TAEHV carrying TAELTX2.3's hyperparameters at 4 latent channels and
/// a random-init tiny Depth-Anything-V2. Drives the same [`compute_ltx25_step_loss_grads`] /
/// [`Ltx25DepthFrames`] / [`refuse_ltx25_depth_anchoring`] `train_25_impl` runs. CPU.
mod ltx25_depth_anchoring_tests;
