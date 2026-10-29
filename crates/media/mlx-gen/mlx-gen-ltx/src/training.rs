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
use mlx_gen::train::dataset::{bucket_edges, center_crop_square, decode_image};
use mlx_gen::train::lora::{accumulate_grads, adapter_optimizer_update, average_grads, LoraParams};
use mlx_gen::train::loss::{prepared_subject_mask_weight, reduce_loss};
use mlx_gen::train::perceptual::{
    combine_step_loss, AuxDriver, AuxModelFootprint, Parameterization, PerceptualPath, StepPlan,
    X0Decoder,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::train::subject_mask::{CropBox, PreparedSubjectMask};
use mlx_gen::train::taehv::{TaehvConfig, TaehvDecoder};
use mlx_gen::train::trainer_loop::SAMPLE_PROMPT_CAP;
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
// sampler, the perceptual (depth-anchoring) auxiliary path, the descriptors / request floors, the
// lazy loaders, the two version-specific trainer bodies (kept apart so what they share stays
// visible), the shared adapter seam and the memory guard. The trainer type and the version
// dispatch stay here.
mod adapter;
mod descriptor;
mod load;
mod memory_guard;
mod perceptual;
mod plan;
mod prepared;
mod test_support;
mod trainer_23;
mod trainer_25;
mod validation;
use adapter::*;
pub use descriptor::*;
pub use load::*;
use memory_guard::*;
use perceptual::*;
pub use plan::*;
use prepared::*;
#[cfg(test)]
use test_support::*;
use trainer_23::*;
#[cfg(test)]
use trainer_25::*;
pub use validation::*;

/// Gemma prompt token budget for caption encoding (the captions are short; padding tokens are
/// attended with `mask=None`, matching the reference `Modality(context_mask=None)`).
const MAX_PROMPT_TOKENS: usize = 128;

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

mod ltx25_restricted_normal_tests;

// The trainer registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION_25 = trainer_descriptor_25 => load_trainer_25
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
