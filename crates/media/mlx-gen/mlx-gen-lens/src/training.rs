//! LoRA/LoKr **training** on the Lens DiT, in pure Rust on mlx-rs (sc-5148, epic 3164) — the
//! native-MLX replacement for the Python `lens_train_runner.py` (torch in `/opt/lens-venv`), the last
//! Python holdout for Lens (zero-Python north star, epic 3482).
//!
//! [`LensTrainer`] realizes the core [`Trainer`] contract on the real 48-block Lens
//! MMDiT, mirroring `ZImageTurboTrainer` / `mlx_gen_z_image` — the model crates don't use mlx-rs's
//! `Module` system (hand-rolled `&self` forwards over raw `Array`s), so training uses the **functional
//! autograd**: the trainable factors live OUTSIDE the model in a [`LoraParams`] map, re-injected each
//! step into the target [`AdaptableLinear`](mlx_gen::adapters::AdaptableLinear)s via the shared core
//! seam ([`mlx_gen::train::lora`]), stepped with `keyed_value_and_grad` + the core [`TrainOptimizer`] +
//! `clip_grad_norm`. The injection mirrors the inference reload op-for-op, so the trained adapter
//! round-trips through [`apply_lens_adapters`](crate::adapters::apply_lens_adapters) (sc-3174)
//! bit-for-bit.
//!
//! ## What is Lens-specific (everything else reuses the family-agnostic core unchanged)
//!
//! Ported from `lens_train_runner.py`:
//!   * **Flow-match velocity target = `noise − x0`** with the transformer **timestep = `t`** (the noise
//!     fraction) fed directly. The Lens DiT [`forward`](crate::dit::LensTransformer::forward) returns
//!     the **raw** patch-space velocity (no negation — the pipeline feeds it to `FlowMatchEuler::step`
//!     un-negated), so the regression target is the velocity itself. This is the **opposite sign** of
//!     the Z-Image trainer, whose Rust `forward()` negates → target `noise − x0` *with* `timestep =
//!     1 − σ`. `x_t = (1 − t)·x0 + t·noise`.
//!   * **Latents by inverting the Lens `_decode`.** The Lens latent space *is* the Flux.2 one, so a
//!     pixel → `[1, seq, 128]` training latent is exactly the Flux.2 `encode_init_latents` chain
//!     (`preprocess_ref_image → Flux2Vae::encode_mean → patchify → bn-normalize → pack`). Uses the
//!     deterministic latent **mean** (the only public encode path + the established mlx-gen img2img
//!     convention); the Python's `latent_dist.sample()` reparam-noise is a minor regularizer dropped
//!     deliberately.
//!   * **Caption features.** The pipeline's positive-only `encode_one`: tokenize → the gpt-oss
//!     [`encode`](crate::text_encoder::encoder::LensTextEncoder::encode) (4 captured layers) → slice
//!     at [`TXT_OFFSET`] → a ones mask. Single-conditional (no CFG), matching the Python.
//!   * **Targets** default to `img_qkv`/`txt_qkv`/`to_out.0`/`to_add_out` (the `AdaptableHost for
//!     LensTransformer` paths, sc-3174); LoKr reconstructs at `LOKR_DTYPE` (what the lens adapter
//!     loader uses, so the trained LoKr round-trips). The gpt-oss encoder loads **Q8** (~12 GB vs
//!     ~40 GB dense bf16) — frozen, used only to cache caption features, then dropped before the train
//!     loop (the 32 GB-Mac free pattern); Q8 also matches the Q8 inference default (sc-3172/sc-5105).
//!
//! Registered under the **`lens`** id (the base, non-distilled `microsoft/Lens` — sc-1583; arch-
//! identical to `lens_turbo`, so the adapter applies to both, sc-3174).
//!
//! ## Memory hardening (sc-5170, the z-image sc-4874/4886/4887 analog)
//! Production-resolution (1024) Lens LoRA training is memory-hardened with the z-image pattern:
//!   * **SDPA-segment checkpointing** is always on in training (LoRA and LoKr) — the joint SDPA runs
//!     inside an `mlx::checkpoint` so its backward recomputes the attention rather than retaining the
//!     `[heads, joint, joint]` probability matrix (the dominant seq² term; MLX has no fused SDPA
//!     backward). Numerically identical, and the flash-backward surrogate every torch trainer gets.
//!   * **`gradient_checkpointing`** (the SceneWorks toggle) is an opt-in OPTION (LoRA only): each of
//!     the 48 dual-stream blocks recomputes its activations in the backward via
//!     [`LensTransformer::forward_with_main_checkpointed`](crate::dit::LensTransformer::forward_with_main_checkpointed),
//!     threading the per-block LoRA factors as explicit checkpoint inputs so the adapter graph
//!     survives the recompute. LoKr keeps the dense path (caught by the guard) — mirroring z-image.
//!   * **Fail-fast OOM preflight guard** — `preflight_memory_guard` projects the dense first-step
//!     peak from resolution (a fitted curve) and, when checkpointing is off and the run would exceed
//!     this machine's memory budget, returns a catchable, actionable error BEFORE the (minutes-long)
//!     latent caching — converting the otherwise-uncatchable SIGKILL into a recommendation to enable
//!     the toggle. The default-off functional path is unaffected.

use std::path::Path;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::{self, BucketSchedule};
use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_edges, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, adapter_optimizer_update, average_grads, build_lokr_targets,
    build_lora_targets, LoraParams, TrainAdapter,
};
use mlx_gen::train::loss::{prepared_subject_mask_weight, reduce_loss};
use mlx_gen::train::perceptual::{
    combine_step_loss, AuxAlternation, Parameterization, PerceptualPath, StepPlan,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::train::subject_mask::{CropBox, PreparedSubjectMask};
use mlx_gen::train::tae::TinyDecoderSpec;
use mlx_gen::weights::Weights;
use mlx_gen::{
    Error, LoadSpec, Modality, NetworkType, Precision, Quant, Result, TrainOptimizer, Trainer,
    TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, multiply, ones, split_sections, subtract};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use mlx_gen_flux2::{load_vae, pack_latents, patchify_latents, preprocess_ref_image, Flux2Vae};

use crate::config::GptOssConfig;
use crate::dit::{LensDitConfig, LensTransformer};
use crate::pipeline::{assemble_conditioning, DEFAULT_DATE, VAE_SCALE_FACTOR};
use crate::registry::MODEL_ID_BASE;
use crate::text::{LensTokenizer, TXT_OFFSET};
use crate::text_encoder::encoder::LensTextEncoder;

/// The lens adapter loader reconstructs LoKr deltas at bf16 (`src/adapters/loader.rs`); training must
/// reconstruct at the same dtype so the trained LoKr round-trips through `apply_lens_adapters`.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637).
const SAMPLE_PROMPT_CAP: usize = 4;

/// The gpt-oss encoder is loaded Q8 for the trainer (~12 GB vs ~40 GB dense bf16): it is frozen and
/// used only to cache caption features once, then dropped. Q8 is the Lens inference default (sc-3172),
/// so the cached features match the deployed encode path.
const TRAINER_ENCODER_QUANT: Option<Quant> = Some(Quant::Q8);

/// The Lens trainer default target modules (`lens_train_runner.DEFAULT_LORA_TARGET_MODULES`): the
/// dual-stream joint-attention projections. `to_out` is an `nn.ModuleList([Linear, Identity])`, so the
/// trainable Linear is `to_out.0` (sc-2218); `img_qkv`/`txt_qkv` are the fused per-stream QKV.
const DEFAULT_TARGET_MODULES: [&str; 4] = ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"];

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

/// `(x_t, target)` for a single sample at flow-match `t`: `x_t = (1−t)·x0 + t·noise`,
/// `target = noise − x0` (the velocity the **raw** Lens DiT output is regressed onto; the transformer
/// timestep is `t` itself, fed by the caller — see the module docs for the sign vs Z-Image).
fn build_batch(x0: &Array, noise: &Array, t: f32) -> Result<(Array, Array)> {
    let one_minus = Array::from_slice(&[1.0 - t], &[1]);
    let s = Array::from_slice(&[t], &[1]);
    let x_t = add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?;
    Ok((x_t, target))
}

/// The production [`Trainer`] for the base `microsoft/Lens` DiT: a frozen base (gpt-oss encoder + Lens
/// MMDiT + Flux.2 VAE + tokenizer) that caches a captioned dataset to VAE-latents/caption-features,
/// then runs the functional-autograd LoRA/LoKr loop with the core runtime glue (LR schedule, gradient
/// accumulation, checkpoint cadence, cancel, progress bands), writing a PEFT adapter that reloads
/// through the inference path.
pub struct LensTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: LensTokenizer,
    /// The 20 B-param gpt-oss encoder, in an `Option` so it can be **dropped after caching** — it is
    /// idle during training (every caption is already cached), yet a multi-GB resident.
    encoder: Option<LensTextEncoder>,
    transformer: LensTransformer,
    vae: Flux2Vae,
    /// The compute dtype (bf16 production / f32 tight-gate), fixed at load from `spec.precision`.
    dtype: Dtype,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID_BASE,
        family: "lens",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        // LoRA/LoKr only — no control-branch training path (F-006).
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
        // update.
        // sc-2127 (epic 2123): honors `resolution_buckets` — one cached latent (+ its own latent
        // grid) per item per bucket edge, walked through a `BucketSchedule`; the pre-flight guard
        // and the preview render size for the largest edge.
        // sc-24828 (epic 2123): honors `subject_mask_loss` on its one (LoRA/LoKr, dense or
        // block-checkpointed) loss path; each (item, bucket) entry's weight is packed exactly like
        // that entry's latent.
        // sc-24830 (epic 2123): depth anchoring — the shared decoded-x0 perceptual path (TAEF2
        // decode of the unpacked flow x0 estimate → Depth-Anything-V2 → cached round-trip
        // reference) on the dense and block-checkpointed forwards.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: true,
            subject_mask_loss: true,
            depth_anchoring: true,
            // sc-24831: the face losses ride the same shared builder arms + x0 decoder.
            identity_loss: true,
            face_landmark_loss: true,
            // sc-24832: the body losses ride the same builder arms as depth anchoring
            // (decoded-x0 pixel losses through this trainer's x0 decoder).
            body_proportion_loss: true,
            body_shape_loss: true,
            normal_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// Construct the trainer from a `microsoft/Lens` snapshot directory (the diffusers multi-component
/// tree: `tokenizer/ text_encoder/ transformer/ vae/`). The DiT is loaded **dense** (the adapter host);
/// the encoder is Q8. `spec.precision` selects the compute dtype (bf16 default / f32 tight-gate).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "lens trainer expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let tokenizer = LensTokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))?;
    let enc_cfg = GptOssConfig::lens();
    let enc_w = Weights::from_dir(root.join("text_encoder"))?;
    let encoder =
        LensTextEncoder::from_weights_quant(enc_w, &enc_cfg, dtype, TRAINER_ENCODER_QUANT)?;
    let dit_cfg = LensDitConfig::lens();
    let dit_w = Weights::from_dir(root.join("transformer"))?;
    let transformer = LensTransformer::from_weights(&dit_w, &dit_cfg, dtype)?;
    // Materialize at load (sc-24245; see `mlx_gen_qwen_image::loader::load_transformer_with`).
    dit_w.materialize_accessed()?;
    let vae = load_vae(&root)?;
    Ok(Box::new(LensTrainer {
        descriptor: trainer_descriptor(),
        tokenizer,
        encoder: Some(encoder),
        transformer,
        vae,
        dtype,
    }))
}

// The trainer registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

/// Normalize a free-form config string the way the trainer's own parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation, factored out so it can be unit-tested without a loaded
/// trainer. Rejects an empty dataset, zero rank, **zero steps** (a 0-step run would write a no-op
/// `B = 0` identity adapter), an unsupported optimizer, and an unrecognized
/// `timestep_type`/`timestep_bias`/`loss_type` (rather than silently falling back to a default).
/// `gradient_checkpointing` is now a supported toggle (sc-5170) — the checkpointed DiT forward + the
/// fail-fast OOM preflight guard are wired in [`LensTrainer::train_impl`].
fn validate_request(req: &TrainingRequest) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err("lens trainer: dataset is empty".into());
    }
    if cfg.rank == 0 {
        return Err("lens trainer: rank must be > 0".into());
    }
    if cfg.steps == 0 {
        return Err("lens trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(format!(
            "lens trainer: optimizer '{}' is not available on MLX training (supported: adamw, adam, \
             rose, prodigy)",
            cfg.optimizer
        )
        .into());
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&cfg.timestep_type).as_str()) {
        return Err(format!(
            "lens trainer: timestep_type '{}' is not recognized (supported: {})",
            cfg.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
        return Err(format!(
            "lens trainer: timestep_bias '{}' is not recognized (supported: {})",
            cfg.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
        return Err(format!(
            "lens trainer: loss_type '{}' is not recognized (supported: {})",
            cfg.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

impl Trainer for LensTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared control-training floor (F-006): a LoRA-only trainer must reject a control-branch
        // request (typed `Unsupported`) rather than silently training a plain adapter.
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
        // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA.
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
        // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        validate_request(req)?;
        // Non-default `lora_target_modules` that match no adaptable module on the DiT would train zero
        // parameters yet "succeed". Catch it here, where the loaded DiT is available to match against.
        if resolve_target_paths(&self.transformer, &req.config).is_empty() {
            return Err(format!(
                "lens trainer: lora_target_modules {:?} matched no adaptable module on the Lens DiT \
                 (targets are img_qkv/txt_qkv/to_out.0/to_add_out)",
                req.config.lora_target_modules
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
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl LensTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720), keeping `?` on `mlx_rs`/family helpers transparent here.
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        validate_request(req)?;
        let cfg = &req.config;

        let target_paths = resolve_target_paths(&self.transformer, cfg);
        if target_paths.is_empty() {
            return Err(format!(
                "lens trainer: lora_target_modules {:?} matched no adaptable module on the Lens DiT",
                cfg.lora_target_modules
            )
            .into());
        }

        // The DiT compute dtype is fixed at load (`spec.precision`); the Lens DiT has no
        // cast-after-load, so `train_dtype` is *enforced* against it (never a silent no-op). The
        // common case (TrainingConfig default bf16 + LoadSpec default Bf16) matches, so this only fires
        // on an explicit f32-vs-bf16 mismatch — telling the caller to load at the matching precision.
        let want_bf16 = {
            let t = cfg.train_dtype.trim();
            t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16")
        };
        let loaded_bf16 = self.dtype == Dtype::Bfloat16;
        if want_bf16 != loaded_bf16 {
            return Err(format!(
                "lens trainer: train_dtype '{}' does not match the loaded precision ({}). Load the \
                 trainer with {} to train at that dtype.",
                cfg.train_dtype,
                if loaded_bf16 { "bf16" } else { "f32" },
                if want_bf16 {
                    "Precision::Bf16"
                } else {
                    "Precision::Fp32"
                }
            )
            .into());
        }
        let compute_dtype = self.dtype;
        // bf16 mixed precision: cast the folded LoRA residual to the activation dtype so the adapted
        // Linear stays bf16 (else it silently re-promotes the chain to f32). f32 → no cast.
        let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);

        on_progress(TrainingProgress::Preparing);
        // sc-2127 — one training edge per resolution bucket (just `[resolution]` when buckets are
        // off). The pre-flight guard and the preview render use the largest (epic 2123 E7); each
        // cached latent carries its OWN grid (see `latent_grid`) for the DiT's img ids.
        let edges = bucket_edges(cfg);
        let edge = preflight_edge(&edges);

        // sc-5170 — fail-fast pre-flight memory guard. The dense (non-block-checkpointed) first step
        // materializes the whole forward graph in one MLX `eval`; at high resolution that working set
        // can exceed unified memory and the OS hard-kills the worker with an UNCATCHABLE SIGKILL (no
        // in-process error — the run just appears to hang at the last cached latent). We cannot catch
        // that kill, so we predict it and refuse up front with a catchable, actionable error BEFORE
        // the (minutes-long) latent caching, UNLESS the run will block-checkpoint (LoRA + the toggle).
        // LoKr always takes the dense path (no clean thread-as-input form), so it is guarded
        // regardless of the toggle.
        let will_checkpoint =
            matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
        // Epic 2123 E7: the training-time aux models (TAEF2 + Depth-Anything-V2) count against the
        // budget on BOTH paths; one cached reference per (item, bucket) entry, sized at the largest
        // edge. A checkpointed run with no aux models stays unguarded (see the guard).
        let aux_gb = perceptual_footprint_gb(cfg, edge, req.items.len() * edges.len());
        preflight_memory_guard(edge, want_bf16, aux_gb, will_checkpoint)?;

        // Epic 2123 depth anchoring: load the frozen decoder + aux models before the caching pass,
        // so a missing/corrupt aux checkpoint fails fast.
        let mut perceptual = load_perceptual_path(cfg)?;

        // --- prepare → load → cache: VAE-latents + 4-layer caption features into memory ---
        on_progress(TrainingProgress::LoadingModel); // base model is already resident from load_trainer
        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127). See [`CacheEntry`].
        let mut cache: Vec<CacheEntry> = Vec::with_capacity(req.items.len() * edges.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            // sc-24828: the item's subject mask is read + checked once, resampled per bucket.
            let subject_mask = PreparedSubjectMask::load_if_enabled(
                "lens trainer",
                item,
                cfg.subject_mask_loss.as_ref(),
            )?;
            let encoder = self.encoder.as_ref().ok_or_else(|| {
                Error::Msg(
                    "lens trainer: text encoder already freed (caching after train loop)".into(),
                )
            })?;
            let (features, mask) =
                encode_caption(&self.tokenizer, encoder, &item.caption, compute_dtype)?;
            let mut to_eval: Vec<&Array> = Vec::with_capacity(features.len() + 1);
            to_eval.push(&mask);
            to_eval.extend(features.iter());
            eval(to_eval)?;
            // The caption features + mask are resolution-independent: encode them once, then one
            // latent per bucket edge, each tagged with its own grid side and (sc-24828) paired with
            // its own packed subject-mask weight.
            for (&edge, (x0, mask_weight)) in edges.iter().zip(encode_buckets(
                &edges,
                subject_mask.as_ref(),
                |edge| encode_latents(&self.vae, &img, edge), // [1, seq, 128]
            )?) {
                cache.push((
                    x0,
                    features.clone(),
                    mask.clone(),
                    latent_grid(edge),
                    mask_weight,
                ));
            }
        }
        if cache.is_empty() {
            // A cancel mid-cache is a genuine cancellation → typed `Error::Canceled`; an empty cache
            // with no cancel is a real "no usable dataset items" error.
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            return Err("lens trainer: no usable dataset items".into());
        }

        // Epic 2123 E8: each (item, bucket) entry's perceptual reference is computed exactly once
        // per job, here, before the loop.
        if let Some(path) = perceptual.as_mut() {
            // sc-24832: the job's subject masks (restricted normal loss) reach every reference,
            // cropped like the image and resampled onto its decoded size.
            path.attach_subject_masks(mlx_gen::train::subject_mask::PerceptualSubjectMasks::load(
                "lens trainer",
                &req.items,
                cfg,
                edges.len(),
                CropBox::center_square,
            )?);
            prepare_perceptual_references(path, &cache)?;
        }

        // sc-5637 — pre-encode the preview-sample prompts into the conditioning batch the preview
        // render expects, while the 20 B-param encoder is still resident (freed just below). The
        // trainer is the third producer of Lens conditioning (beside the registry and the struct API),
        // so it routes through the SAME `assemble_conditioning` gate rather than hand-rolling the
        // batching (sc-17616): above guidance 1.0 that yields the joint CFG batch (`[2, …]` = positive
        // then empty-negative — the trainer always uses the empty negative, i.e. zero features + zero
        // mask), and at guidance 1.0 the combine reduces to `cond` and `render_sample` runs a B=1
        // forward, so the uncond half is not batched at all.
        let sample_caps: Vec<(String, Vec<Array>, Array)> = if cfg.sample_every > 0
            && !cfg.sample_prompts.is_empty()
            && !req.cancel.is_cancelled()
        {
            let encoder = self.encoder.as_ref().ok_or_else(|| {
                Error::Msg("lens trainer: text encoder already freed (sample pre-encode)".into())
            })?;
            let mut caps = Vec::with_capacity(cfg.sample_prompts.len().min(SAMPLE_PROMPT_CAP));
            for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                let positive = encode_caption(&self.tokenizer, encoder, prompt, compute_dtype)?;
                let (features, mask) = assemble_conditioning(
                    positive,
                    cfg.sample_guidance_scale,
                    compute_dtype,
                    || Ok(None), // the trainer's preview negative is always empty
                )?;
                let mut to_eval: Vec<&Array> = Vec::with_capacity(features.len() + 1);
                to_eval.push(&mask);
                to_eval.extend(features.iter());
                eval(to_eval)?;
                caps.push((prompt.clone(), features, mask));
            }
            caps
        } else {
            Vec::new()
        };
        let sampling_enabled = !sample_caps.is_empty();

        // Every caption is cached now — free the 20 B-param encoder and evict its buffers before the
        // train loop, reclaiming that resident for the DiT working set.
        self.encoder = None;
        mlx_rs::memory::clear_cache();

        // --- adapter targets + params (LoRA or LoKr) + optimizer ---
        let rank = cfg.rank as f32;
        let (adapter, mut params) = match cfg.network_type {
            NetworkType::Lora => {
                let (targets, params) = build_lora_targets(
                    &mut self.transformer,
                    &target_paths,
                    cfg.rank as i32,
                    cfg.seed,
                )?;
                (TrainAdapter::Lora { targets }, params)
            }
            NetworkType::Lokr => {
                let (targets, params) = build_lokr_targets(
                    &mut self.transformer,
                    &target_paths,
                    cfg.rank as i32,
                    cfg.decompose_factor,
                    cfg.seed,
                )?;
                (TrainAdapter::Lokr { targets }, params)
            }
        };
        let alpha = cfg.alpha;
        let mae = {
            let lt = cfg.loss_type.to_ascii_lowercase();
            lt == "mae" || lt == "l1"
        };

        // sc-5170 — gradient checkpointing. Collect, per block, the adapter-routable LOCAL paths
        // trained on it (e.g. `"attn.img_qkv"`), in trained-file order — the factors a checkpoint
        // segment threads as explicit inputs. Every Lens adapter target lives in a block, so this
        // covers the whole trainable surface.
        let n_layers = self.transformer.cfg.num_layers;
        let mut block_local_targets: Vec<Vec<String>> = vec![Vec::new(); n_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    if i < n_layers {
                        block_local_targets[i].push(local.to_string());
                    }
                }
            }
        }
        // Opt-in OPTION (the SceneWorks "Gradient Checkpointing" toggle), never auto-forced — a run
        // that would OOM is caught instead by the pre-flight guard above, which recommends this flag
        // rather than silently changing the user's training dynamics. LoRA only — LoKr (a captured-
        // param Kronecker reconstruction) falls back to the dense path.
        let use_checkpoint =
            matches!(adapter, TrainAdapter::Lora { .. }) && cfg.gradient_checkpointing;
        let checkpoint_blocks: Option<&[Vec<String>]> = if use_checkpoint {
            Some(&block_local_targets)
        } else {
            None
        };
        // SDPA-segment checkpointing is ALWAYS on in training (LoRA and LoKr): numerically identical
        // to the retained backward (same decomposed attention, recomputed) and removes the dominant
        // seq² per-block retention. When whole-block checkpointing is on, the per-block SDPA flag goes
        // OFF (the block recompute already covers attention; nesting would recompute it twice).
        self.transformer.set_sdpa_checkpoint(!use_checkpoint);

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
        // sc-2127: which cached (item, bucket) latent each step trains on (round-robin over items for
        // a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
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
            let (losses, grads) = run_train_step(
                &mut self.transformer,
                &params,
                &adapter,
                cfg,
                &cache,
                &schedule,
                perceptual.as_mut().zip(alternation.as_mut()),
                step,
                mae,
                compute_dtype,
                lora_dtype,
                checkpoint_blocks,
            )?;
            last_loss = losses.total;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                // The final update can fire with fewer than `accum` grads when `steps` isn't a
                // multiple of the accumulation; dividing by `accum` would down-scale that step. Divide
                // by the actual in-window count instead (F-069).
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
                adapter.save(&params, alpha, rank, cfg.decompose_factor, "", &ckpt)?;
                checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // sc-5637 — periodic best-effort previews from the in-progress adapter (mirrors z-image).
            // Install the current factors as concrete adapters for the forward-only render; the next
            // step's traced `loss_fn` re-installs them. A render failure must NOT abort the long
            // training run — log and continue.
            if sampling_enabled && step % cfg.sample_every == 0 {
                adapter.install_as(
                    &mut self.transformer,
                    &params,
                    alpha,
                    rank,
                    lora_dtype,
                    LOKR_DTYPE,
                )?;
                let total = sample_caps.len() as u32;
                for (i, (prompt, features, mask)) in sample_caps.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match crate::pipeline::render_sample(
                        &self.transformer,
                        &self.vae,
                        features,
                        mask,
                        sample_seed,
                        edge,
                        cfg.sample_steps.max(1) as usize,
                        cfg.sample_guidance_scale,
                        compute_dtype,
                        &req.cancel,
                    ) {
                        Ok(image) => on_progress(TrainingProgress::Sample {
                            step,
                            index: i as u32 + 1,
                            total,
                            prompt: prompt.clone(),
                            image,
                        }),
                        Err(e) => eprintln!(
                            "[sc-5637] {MODEL_ID_BASE} preview sample failed at step {step} \
                             (prompt {}): {e} — skipping this preview, training continues",
                            i + 1
                        ),
                    }
                }
            }
        }

        // Cancelled before a single step completed (`steps == 0` is rejected by `validate`): the
        // factors are still the `B = 0` no-op init. Surface the cancellation rather than writing a
        // valid-looking identity adapter as a trained artifact.
        if steps_run == 0 {
            return Err(Error::Canceled);
        }

        // --- save final adapter (the diffusers/PEFT format `apply_lens_adapters` loads, sc-3174) ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        adapter.save(
            &params,
            alpha,
            rank,
            cfg.decompose_factor,
            "",
            &adapter_path,
        )?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Number of caption tokens assumed by the pre-flight projection. The unified attention sequence is
/// `img_len + txt_len`; `img_len = (edge/16)²` dominates (4096 at edge 1024), so the exact caption
/// length barely shifts the projection — this is the length the guard's fit was measured at, kept as a
/// representative constant rather than threading the (per-item, variable) real caption length.
const PREFLIGHT_TXT_TOKENS: f64 = 64.0;

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the
/// unified token count `s = img_len + txt_len` — an empirical fit to peaks measured on the 128 GB
/// target. The structure follows the sc-4874 decomposition `weights + linear·s + quad·s²`: the
/// constant is the resident DiT base (the gpt-oss encoder is freed before the train loop), the linear
/// term is the per-token hidden-state activations across the 48 dual-stream blocks, and the quadratic
/// term is the seq² attention transient — demoted from "one retained `[24-head, joint, joint]`
/// probability matrix per block" to a single block's backward transient by the always-on SDPA-segment
/// checkpointing (sc-5170). bf16 roughly halves the weights + activation terms.
///
/// MEASURED (`first_step_ckpt_sweep`, 128 GB Mac17,6, rank 16 / 192 targets / batch 1, caption
/// `txt_len = 64`) with SDPA-segment checkpointing on and only the DiT resident (the gpt-oss encoder
/// is freed before the train loop, the VAE is idle), AFTER the sc-5188 bf16 re-promotion fix:
///   f32  edge 512/768/1024 (s = 1088/2368/4160) → 22.78 / 31.83 / 45.22 GB
///   bf16 edge 512/768/1024                       → 11.52 / 16.04 / 22.79 GB
/// The exact 3-point fit reproduces all three per dtype. bf16 is now ~half f32 (the expected halving,
/// like z-image). Until sc-5188 a strong-f32 `1 + scale` AdaLN modulation constant silently re-promoted
/// the bf16 DiT stream to f32 at block 0, so the whole DiT ran f32 in "bf16" mode and peaked ~2-2.4 GB
/// ABOVE f32 (the f32 weight up-casts piled on top of f32 activations). The DiT is smaller per-layer
/// than z-image's, so this curve is measured fresh (not reused). Assumes micro-batch 1; refit (via
/// `first_step_ckpt_sweep`) if the activation shape or resident set changes.
/// `projection_matches_measured_curve` pins it to the measured points.
fn projected_dense_peak_gb(s: f64, bf16: bool) -> f64 {
    if bf16 {
        PREFLIGHT_BF16.0 + PREFLIGHT_BF16.1 * s + PREFLIGHT_BF16.2 * s * s
    } else {
        PREFLIGHT_F32.0 + PREFLIGHT_F32.1 * s + PREFLIGHT_F32.2 * s * s
    }
}

/// `(weights, linear, quad)` fit constants for [`projected_dense_peak_gb`] — the exact 3-point fits to
/// the measured `first_step_ckpt_sweep` peaks (see its docs). `projection_matches_measured_curve`
/// enforces the measured anchors; refit both tuples if the sweep prints materially different numbers.
const PREFLIGHT_F32: (f64, f64, f64) = (15.43, 6.618e-3, 1.308e-7);
const PREFLIGHT_BF16: (f64, f64, f64) = (7.875, 3.266e-3, 7.666e-8);

/// The edge the pre-flight guard (and the preview render) sizes for: the LARGEST bucket edge (epic
/// 2123 E7) — the dense first step's working set is set by the biggest latent the run will ever train
/// on, whichever bucket the schedule happens to draw first.
fn preflight_edge(edges: &[u32]) -> u32 {
    edges.iter().copied().max().unwrap_or(0)
}

/// The Lens latent grid side for a training `edge`: a cell maps to a 16×16 pixel tile (Flux.2 8× VAE ∘
/// 2× DiT patchify), so an `edge`-square image packs to `[1, grid², 128]` and the DiT's img ids span a
/// `grid × grid` lattice. The ÷32 bucket guarantees the VAE-encoded `edge/8` is even, so the 2×2
/// patchify divides cleanly. Per bucket (sc-2127): every cached latent carries the grid of the edge
/// it was encoded at.
fn latent_grid(edge: u32) -> usize {
    (edge / VAE_SCALE_FACTOR) as usize // latent_h == latent_w (square)
}

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead. The budget is MLX's own reported
/// memory limit (≈ the device's recommended working set); the rest is [`check_preflight_budget`].
/// Consulted when gradient checkpointing is OFF (LoKr, or LoRA with the toggle off), and — whenever
/// the training-time aux models add memory (`extra_gb`, epic 2123 E7) — when it is on too.
fn preflight_memory_guard(edge: u32, bf16: bool, extra_gb: f64, checkpointed: bool) -> Result<()> {
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    check_preflight_budget_with_aux(edge, bf16, budget_gb, extra_gb, checkpointed)
}

/// The pure guard logic (no MLX global state, so it is unit-testable): refuse if the projected dense
/// first-step peak exceeds `budget_gb × 0.85`. `edge` is the bucketed training edge; the unified token
/// count is `(edge/16)²` (latent /8, patch 2) plus a representative caption block. The 0.85 leaves
/// headroom for the worker/host — exceeding it is the regime where the dense run was observed to die.
#[cfg(test)]
fn check_preflight_budget(edge: u32, bf16: bool, budget_gb: f64) -> Result<()> {
    check_preflight_budget_with_aux(edge, bf16, budget_gb, 0.0, false)
}

/// [`check_preflight_budget`] plus the epic-2123 aux models (`extra_gb`, [`perceptual_footprint_gb`])
/// on top of the DiT projection. With `checkpointed`, the projection is the resident base
/// (`projected_dense_peak_gb(0)`: no fitted checkpointed curve exists, so the resident DiT is the
/// lower bound the aux models stack on), and a checkpointed run with no aux models is not guarded.
fn check_preflight_budget_with_aux(
    edge: u32,
    bf16: bool,
    budget_gb: f64,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    if checkpointed && extra_gb <= 0.0 {
        return Ok(());
    }
    let tokens_per_side = (edge as f64 / 16.0).ceil();
    let s = tokens_per_side * tokens_per_side + PREFLIGHT_TXT_TOKENS;
    let projected = if checkpointed {
        projected_dense_peak_gb(0.0, bf16)
    } else {
        projected_dense_peak_gb(s, bf16)
    } + extra_gb;
    let safe = budget_gb * 0.85;
    if projected > safe && checkpointed {
        return Err(format!(
            "lens trainer: a checkpointed training step at resolution {edge} with the \
             depth-anchoring models (~{extra_gb:.1} GB for the tiny decoder and Depth-Anything-V2) \
             needs at least ~{projected:.0} GB, exceeding this machine's ~{safe:.0} GB safe budget \
             ({budget_gb:.0} GB MLX limit × 0.85). Use a smaller depth model or reduce the training \
             resolution."
        )
        .into());
    }
    if projected > safe {
        return Err(format!(
            "lens trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             (the forward working set materializes in one allocation), exceeding this machine's \
             ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS \
             would hard-kill the worker (SIGKILL) at the first step with no recoverable error \
             (sc-4874/sc-5170). Enable Gradient Checkpointing (recomputes block activations in the \
             backward) or reduce the training resolution."
        )
        .into());
    }
    Ok(())
}

/// Resolve the config's target-module *suffixes* (default [`DEFAULT_TARGET_MODULES`]) to full dotted
/// paths by matching them against every adapter-routable module on the DiT — the same suffix-match
/// PEFT's `LoraConfig(target_modules=…)` does (`transformer_blocks.{i}.attn.{suffix}`).
fn resolve_target_paths(transformer: &LensTransformer, cfg: &TrainingConfig) -> Vec<String> {
    let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
        DEFAULT_TARGET_MODULES
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    AdaptableHost::adaptable_paths(transformer)
        .into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
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

/// Encode a center-cropped square image into a Lens training latent `[1, latent·latent, 128]` — the
/// inverse of the Lens `_decode`. The Lens latent space *is* the Flux.2 one, so this is the Flux.2
/// `encode_init_latents` chain built from public helpers: `preprocess_ref_image` (resize + `[−1,1]`
/// NHWC) → `Flux2Vae::encode_mean` (latent mean) → NCHW → 2×2 `patchify_latents` →
/// `Flux2Vae::bn_normalize_nchw` → `pack_latents`. `pack_latents`/`patchify_latents` are plain
/// row-major reshapes consistent with the lens `vae::decode` plain-reshape path, so the latent lives in
/// exactly the space the DiT predicts in. `crop_to_even`/`match_latent_spatial_size` (in the fork's
/// `encode_init_latents`) are no-ops at the ÷32-bucketed square edge, so they are elided.
fn encode_latents(vae: &Flux2Vae, image: &Image, edge: u32) -> Result<Array> {
    let pre = preprocess_ref_image(image, edge, edge)?; // NHWC [1, edge, edge, 3]
    let enc = vae.encode_mean(&pre)?; // NHWC [1, edge/8, edge/8, 32]
    let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for the packing helpers
    let patchified = patchify_latents(&enc)?; // [1, 128, edge/16, edge/16]
    let normed = vae.bn_normalize_nchw(&patchified)?; // (x − mean)/std on the packed 128-ch
    pack_latents(&normed) // [1, latent·latent, 128]
}

/// Lay a latent-grid tensor `[1, C, H, W]` out EXACTLY like [`encode_latents`] lays out the VAE
/// latent: 2×2 `patchify_latents` → `pack_latents`, so element `[0, tok, feat]` of the result sits
/// over the same latent cell as `x0[0, tok, feat]`. (The bn-normalize between the two is a
/// per-channel affine on VALUES, not a layout op, so a weight map skips it.)
fn pack_like_latent(grid: &Array) -> Result<Array> {
    pack_latents(&patchify_latents(grid)?)
}

/// The subject-mask loss weight (sc-24828) for one cached (item, bucket) latent `x0`, packed like it
/// (`[1, seq, 4·C]`, `seq = (H/2)·(W/2)` on the square bucket): the item's already-loaded mask is
/// center-cropped like the image, area-averaged onto THIS bucket's UNPACKED `[1, C, H, W]` latent
/// grid, then run through [`pack_like_latent`]. `None` when the technique is off.
fn latent_subject_mask_weight(
    mask: Option<&PreparedSubjectMask>,
    x0: &Array,
) -> Result<Option<Array>> {
    let sh = x0.shape();
    let (seq, feat) = (sh[1], sh[2]);
    let side = (seq as f64).sqrt().round() as i32;
    if sh.len() != 3 || side * side != seq || feat % 4 != 0 {
        return Err(Error::Msg(format!(
            "lens trainer: subject mask needs a square packed latent, got {sh:?}"
        )));
    }
    let unpacked = [1, feat / 4, side * 2, side * 2];
    prepared_subject_mask_weight("lens trainer", mask, CropBox::center_square, &unpacked)?
        .map(|w| pack_like_latent(&w))
        .transpose()
}

/// Encode a caption into its per-layer DiT text features (sliced at [`TXT_OFFSET`]) + the valid mask —
/// the pipeline's positive-only `encode_one` (single-conditional training; the Python keeps the
/// positives of `encode_prompt(neg="")`). Returns `(features, mask)`: `features` is 4 × `[1, S, 2880]`,
/// `mask` is `[1, S]` (all-1; a single prompt is unpadded).
fn encode_caption(
    tokenizer: &LensTokenizer,
    encoder: &LensTextEncoder,
    caption: &str,
    dtype: Dtype,
) -> Result<(Vec<Array>, Array)> {
    let out = tokenizer.encode(caption, DEFAULT_DATE)?;
    let l = out.ids.len() as i32;
    let offset = TXT_OFFSET as i32;
    if l <= offset {
        return Err(format!(
            "lens trainer: caption tokenized to {l} tokens (≤ the {offset}-token harmony preamble), \
             leaving no conditioning tokens"
        )
        .into());
    }
    let input_ids = Array::from_slice(&out.ids, &[1, l]);
    // Training-loss encode: the outer train loop checks cancel between steps, so no per-layer hook here.
    let layers = encoder.encode(&input_ids, None)?; // num_text_layers × [1, L, 2880]
                                                    // `[:, offset:, :]` — split at the offset along the sequence axis, keep the tail.
    let features = layers
        .iter()
        .map(|f| Ok(split_sections(f, &[offset], 1)?[1].as_dtype(dtype)?))
        .collect::<Result<Vec<_>>>()?;
    let mask = ones::<f32>(&[1, l - offset])?;
    Ok((features, mask))
}

/// Sample a normalized flow-match timestep (interpolation coefficient) `t ∈ [1e-3, 1−1e-3]` — a
/// faithful port of the SceneWorks `sample_training_timestep` (identical to the Z-Image trainer):
/// `sigmoid(randn)` by default, `uniform` for linear, `(uniform + sigmoid(randn))/2` for weighted;
/// bias `high` → `√t`, `low` → `t²`. Deterministic in `seed`.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = timestep_type.trim().to_ascii_lowercase().replace('-', "_");
    let t = match ttype.as_str() {
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
    let bias = timestep_bias
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_");
    let t = match bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    Ok(t.clamp(1e-3, 1.0 - 1e-3))
}

/// One forward+backward over the trainable adapter factors: inject `params` (LoRA or LoKr), run the
/// Lens DiT, regress the **raw** `forward()` velocity onto `noise − x0`, return `(loss, grads)`. The
/// transformer timestep is `t` (the noise fraction) directly. `dtype` is the training compute dtype:
/// `x_t`/features are cast at entry (the weights were loaded at this dtype), the LoRA factors are cast
/// inside the traced install (`lora_dtype`), so the DiT graph runs at `dtype`; the noising math, loss,
/// and grads stay f32.
///
/// `checkpoint_blocks`, when `Some`, lists per-block LOCAL LoRA target paths and switches the forward
/// to the gradient-checkpointed path (sc-5170) — each block recomputes its activations in the backward
/// instead of retaining them. `None` runs the dense (activation-retaining) forward. Either way the
/// per-block SDPA-segment checkpointing flag is the caller's responsibility (set on `transformer`).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    transformer: &mut LensTransformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    features: &[Array],
    mask: &Array,
    t: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    latent: usize,
    checkpoint_blocks: Option<&[Vec<String>]>,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) = compute_step_loss_grads(
        transformer,
        params,
        adapter,
        alpha,
        rank,
        x0,
        features,
        mask,
        t,
        noise,
        mae,
        mask_weight,
        dtype,
        lora_dtype,
        latent,
        checkpoint_blocks,
        None,
    )?;
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

/// A packed Lens latent `[1, grid², 4·C]` (the [`encode_latents`] layout: `patchify_latents` →
/// `pack_latents`) back to the unpatchified NCHW grid `[1, C, 2·grid, 2·grid]` — the exact inverse
/// of those two reshapes. The values stay batch-normalized: TAEF2 decodes the FLUX.2 transformer's
/// (batch-normalized) latent unpatchified (its published diffusers wrapper swaps the VAE's latent
/// batch-norm for an identity one), so no de-normalization is applied.
fn unpack_to_decoder_layout(packed: &Array, grid: usize) -> Result<Array> {
    let sh = packed.shape();
    let g = grid as i32;
    if sh.len() != 3 || sh[1] != g * g || sh[2] % 4 != 0 {
        return Err(Error::Msg(format!(
            "lens trainer: expected a packed [B, {}, 4·C] latent, got {sh:?}",
            g * g
        )));
    }
    let (b, c4) = (sh[0], sh[2]);
    let c = c4 / 4;
    // Inverse pack: [B, g·g, 4C] → [B, 4C, g, g].
    let grid_nchw = packed
        .reshape(&[b, g, g, c4])?
        .transpose_axes(&[0, 3, 1, 2])?;
    // Inverse patchify: channel `c·4 + i·2 + j` at `(h, w)` → pixel `(2h + i, 2w + j)` of `c`.
    Ok(grid_nchw
        .reshape(&[b, c, 2, 2, g, g])?
        .transpose_axes(&[0, 1, 4, 2, 5, 3])?
        .reshape(&[b, c, 2 * g, 2 * g])?)
}

/// [`compute_loss_grads`] with the step's perceptual plan (epic 2123 E8): on an aux-only step the
/// diffusion term is not computed (it contributes zero) and the loss is the weighted perceptual term
/// on the model's x0 estimate `x0 = x_t − t·v` (the raw Lens velocity regresses `noise − x0` with
/// `x_t = (1−t)·x0 + t·noise`), unpacked to TAEF2's `[1, 32, h/8, w/8]` layout
/// ([`unpack_to_decoder_layout`]). With `aux = None` (or a diffusion-only plan with no aux loss) the
/// traced graph is exactly the pre-epic-2123 one; both the dense and the block-checkpointed forwards
/// carry the aux term.
#[allow(clippy::too_many_arguments)]
fn compute_step_loss_grads(
    transformer: &mut LensTransformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    features: &[Array],
    mask: &Array,
    t: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    latent: usize,
    checkpoint_blocks: Option<&[Vec<String>]>,
    aux: Option<AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    let (x_t_f32, target) = build_batch(x0, noise, t)?;
    let mask_weight = mask_weight.cloned();
    let x_t = x_t_f32.as_dtype(dtype)?; // no-op in f32 mode
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    let timestep = Array::from_slice(&[t], &[1]);
    let feats: Vec<Array> = features
        .iter()
        .map(|f| f.as_dtype(dtype))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mask = mask.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL targets so the dense path (and any non-checkpointed targets) train through
        // ordinary autograd; on the checkpointed path the block adapters installed here are simply
        // replaced inside each checkpoint segment by the explicit-input factors, so they cost nothing.
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        let v = match checkpoint_blocks {
            Some(locals) => transformer
                .forward_with_main_checkpointed(
                    &x_t,
                    &feats,
                    Some(&mask),
                    &timestep,
                    1,
                    latent,
                    latent,
                    &p,
                    locals,
                    alpha,
                )
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => transformer
                .forward(&x_t, &feats, Some(&mask), &timestep, 1, latent, latent)
                .map_err(|e| Exception::custom(e.to_string()))?,
        };
        let diffusion = if diffusion_on {
            let diff = subtract(&v, &target)?;
            // MSE / MAE, subject-mask weighted when on (sc-24828) — reduces to a 0-d scalar (grad
            // requires a scalar cotangent).
            Some(reduce_loss(&diff, mask_weight.as_ref(), mae)?)
        } else {
            None
        };
        let aux_term = match &aux {
            Some(a) if aux_on => {
                // x0 estimate in f32 from the raw velocity (x0 = x_t − t·v), unpacked for TAEF2.
                let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma: t }
                    .recover_x0(&x_t_f32, &v.as_dtype(Dtype::Float32)?)
                    .and_then(|x| unpack_to_decoder_layout(&x, latent))
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

/// Lens's x0 decoder for the shared aux-loss builder (epic 2123 E8): TAEF2 (`madebyollin/taef2`,
/// the FLUX.2 32-channel latent API — the Lens latent space IS the FLUX.2 one).
fn taef2_decoder() -> mlx_gen_perceptual::DecoderSpec {
    mlx_gen_perceptual::DecoderSpec::Tiny {
        name: "TAEF2",
        config: TinyDecoderSpec::taef2(),
    }
}

/// Build the epic-2123 perceptual path through the shared builder: `None` when no aux loss is
/// enabled (nothing loads; every step is the plain diffusion step).
fn load_perceptual_path(cfg: &TrainingConfig) -> Result<Option<PerceptualPath>> {
    mlx_gen_perceptual::build_perceptual_path(
        cfg,
        &mlx_gen_perceptual::AuxLossContext {
            label: "lens trainer",
            decoder: taef2_decoder(),
            latent_lpips: None,
        },
    )
}

/// Extra training memory (GB) the enabled aux losses add at the bucketed `edge` with `entries`
/// cached references (epic 2123 E7). `0` when no aux loss is enabled.
fn perceptual_footprint_gb(cfg: &TrainingConfig, edge: u32, entries: usize) -> f64 {
    mlx_gen_perceptual::perceptual_footprint_gb(
        cfg,
        &taef2_decoder(),
        mlx_gen_perceptual::AuxGeometry::image(edge, entries),
    )
}

/// Compute every cache entry's perceptual reference once (its clean packed latent, unpacked to the
/// decoder layout at the entry's own grid), keyed per (item, bucket) entry.
fn prepare_perceptual_references(path: &mut PerceptualPath, cache: &[CacheEntry]) -> Result<()> {
    for (i, (x0, _, _, grid, _)) in cache.iter().enumerate() {
        path.ensure_reference(i, &unpack_to_decoder_layout(x0, *grid)?)?;
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
    transformer: &mut LensTransformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    cfg: &TrainingConfig,
    cache: &[CacheEntry],
    schedule: &BucketSchedule,
    perceptual: Option<(&mut PerceptualPath, &mut AuxAlternation)>,
    step: u32,
    mae: bool,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    checkpoint_blocks: Option<&[Vec<String>]>,
) -> Result<(StepLosses, LoraParams)> {
    let k = (step - 1) as usize;
    let (item, _bucket) = schedule.sample(k);
    let entry = schedule.cache_index(k);
    let (x0, features, mask, latent, mask_weight) = &cache[entry];
    let mut t = sample_sigma(
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
    let plan;
    let aux = match perceptual {
        Some((path, alternation)) => {
            // Normally a no-op (references were computed once, before the loop).
            path.ensure_reference(entry, &unpack_to_decoder_layout(x0, *latent)?)?;
            plan = path.plan(alternation.key(step, item), entry, t)?;
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
        adapter,
        cfg.alpha,
        cfg.rank as f32,
        x0,
        features,
        mask,
        t,
        &noise,
        mae,
        mask_weight.as_ref(),
        dtype,
        lora_dtype,
        *latent,
        checkpoint_blocks,
        aux,
    )
}

// ===========================================================================================
// sc-5170 — first-step memory/grad-parity harness (weight-gated, run as its own process).
//
// Drives the exact inner training step (`compute_loss_grads` + the step-1 grad `eval` the real loop
// forces at the optimizer step) directly on the REAL Lens DiT — synthesizing the latent + caption
// features (the encoder/VAE are irrelevant to the DiT working set, which is what is under test) —
// sweeping resolution with MLX peak-memory probes around it. The sweep produces the points the
// `projected_dense_peak_gb` guard is fit to; the parity tests prove the checkpointed forward and the
// SDPA-segment checkpointing do not change the gradients.
//
//   cargo test -p mlx-gen-lens --release --lib first_step -- --ignored --nocapture
//   cargo test -p mlx-gen-lens --release --lib grads_match -- --ignored --nocapture
// ===========================================================================================
/// One `train_impl` cache entry (item-major, `cache[item * n_buckets + bucket]`, sc-2127): packed
/// clean latent, caption features, caption mask, that latent's grid side, and — subject-masked loss
/// on (sc-24828) — that bucket's packed latent loss-weight map (`None` when off).
type CacheEntry = (Array, Vec<Array>, Array, usize, Option<Array>);

/// sc-2127 × sc-24828: one item's clean latent per bucket edge (`encode(edge)`, item-major
/// order), each paired with its subject-mask loss weight — the item's already-loaded mask cropped
/// with the center square `center_crop_square` cuts, area-averaged onto THAT bucket's latent grid
/// and laid out like that latent. Packed like the latent by
/// [`latent_subject_mask_weight`]. `None` weights when masked loss is off.
fn encode_buckets(
    edges: &[u32],
    mask: Option<&PreparedSubjectMask>,
    mut encode: impl FnMut(u32) -> Result<Array>,
) -> Result<Vec<(Array, Option<Array>)>> {
    edges
        .iter()
        .map(|&edge| {
            let x0 = encode(edge)?;
            let mask_weight = latent_subject_mask_weight(mask, &x0)?;
            eval([&x0])?;
            Ok((x0, mask_weight))
        })
        .collect()
}

#[cfg(test)]
mod first_step_repro {
    use super::*;
    use mlx_gen::train::lora::build_lora_targets;
    use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
    use std::path::PathBuf;

    /// The base Lens snapshot root from the required `LENS_SNAPSHOT` env var. sc-13668: there is no
    /// implicit default — the source snapshot path must be passed in explicitly. The base is the
    /// `SceneWorks/Lens` flat-diffusers training rehost (sc-8797); Microsoft pulled the original
    /// `microsoft/Lens` repo.
    fn snapshot() -> Option<PathBuf> {
        std::env::var("LENS_SNAPSHOT").ok().map(PathBuf::from)
    }

    #[test]
    fn source_root_requires_explicit_env_no_default() {
        let key = "LENS_SNAPSHOT";
        let saved = std::env::var(key).ok();
        std::env::remove_var(key);
        assert!(
            snapshot().is_none(),
            "the source snapshot root must come from {key}: sc-13668 removed the implicit default"
        );
        std::env::set_var(key, "/sentinel/lens");
        assert_eq!(snapshot(), Some(PathBuf::from("/sentinel/lens")));
        match saved {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    fn gb(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Load just the Lens DiT at `dtype` (the only component whose activations drive the first-step
    /// working set; the encoder is freed before the train loop and the VAE is idle).
    fn build_dit(dtype: Dtype) -> LensTransformer {
        let root = snapshot().expect("set LENS_SNAPSHOT to the SceneWorks/Lens snapshot root");
        let w = Weights::from_dir(root.join("transformer")).unwrap();
        LensTransformer::from_weights(&w, &LensDitConfig::lens(), dtype).unwrap()
    }

    /// Default-target LoRA factors (rank 16) on a freshly-loaded DiT — the production target surface
    /// (192 = 48 blocks × 4 joint-attention projections).
    fn build_targets(dit: &mut LensTransformer) -> (TrainAdapter, LoraParams) {
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(dit, &cfg);
        let (targets, params) = build_lora_targets(dit, &target_paths, 16, 7).unwrap();
        (TrainAdapter::Lora { targets }, params)
    }

    /// Per-block LOCAL LoRA target paths (mirrors `train_impl`), for driving the checkpointed path.
    fn block_local_targets(dit: &LensTransformer) -> Vec<Vec<String>> {
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(dit, &cfg);
        let n_layers = dit.cfg.num_layers;
        let mut out: Vec<Vec<String>> = vec![Vec::new(); n_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    if i < n_layers {
                        out[i].push(local.to_string());
                    }
                }
            }
        }
        out
    }

    /// Synthesize one training batch at `edge`: a clean latent `[1, (edge/16)², 128]`, matching noise,
    /// 4 caption-feature layers `[1, txt_len, 2880]`, and an all-valid mask `[1, txt_len]`. The latent
    /// magnitude is irrelevant — the graph SIZE (driven by resolution) is the variable under test.
    fn synth(edge: u32) -> (Array, Array, Vec<Array>, Array, usize) {
        let latent = (edge / VAE_SCALE_FACTOR) as usize;
        let seq = (latent * latent) as i32;
        let txt = 64i32; // PREFLIGHT_TXT_TOKENS — a representative caption length
        let x0 = random::normal::<f32>(&[1, seq, 128], None, None, Some(&random::key(1).unwrap()))
            .unwrap();
        let noise =
            random::normal::<f32>(&[1, seq, 128], None, None, Some(&random::key(2).unwrap()))
                .unwrap();
        let feats: Vec<Array> = (0..4)
            .map(|k| {
                random::normal::<f32>(
                    &[1, txt, 2880],
                    None,
                    None,
                    Some(&random::key(10 + k).unwrap()),
                )
                .unwrap()
            })
            .collect();
        let mask = ones::<f32>(&[1, txt]).unwrap();
        mlx_rs::transforms::eval(
            std::iter::once(&x0)
                .chain(std::iter::once(&noise))
                .chain(std::iter::once(&mask))
                .chain(feats.iter())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        (x0, noise, feats, mask, latent)
    }

    /// Run a single first training step at `edge` and report peak GPU memory across the
    /// forward+backward (forces the backward grad eval — the real step-1 kill point). `sdpa_ckpt` arms
    /// the always-on SDPA-segment checkpointing; `checkpoint_blocks` switches on whole-block
    /// checkpointing.
    #[allow(clippy::too_many_arguments)]
    fn one_step(
        dit: &mut LensTransformer,
        adapter: &TrainAdapter,
        params: &LoraParams,
        edge: u32,
        dtype: Dtype,
        checkpoint_blocks: Option<&[Vec<String>]>,
        sdpa_ckpt: bool,
        tag: &str,
    ) -> (f32, f64) {
        dit.set_sdpa_checkpoint(sdpa_ckpt);
        let (x0, noise, feats, mask, latent) = synth(edge);
        let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
        clear_cache();
        reset_peak_memory();
        let before = get_active_memory();
        let t0 = std::time::Instant::now();
        let (loss, grads) = compute_loss_grads(
            dit,
            params,
            adapter,
            16.0,
            16.0,
            &x0,
            &feats,
            &mask,
            0.5,
            &noise,
            false,
            None,
            dtype,
            lora_dtype,
            latent,
            checkpoint_blocks,
        )
        .unwrap();
        // `compute_loss_grads` only forces the loss (forward). The real trainer forces the backward at
        // the step-1 optimizer `eval`; do the same here so the peak reflects the true working set.
        eval(grads.values()).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let peak = get_peak_memory();
        eprintln!(
            "  [edge {edge:>4} {tag}] seq {}  loss {loss:.5}  active-before {:.2} GB  peak {:.2} GB  step {secs:.2}s",
            latent * latent,
            gb(before),
            gb(peak)
        );
        (loss, gb(peak))
    }

    /// Max relative grad diff between two param maps.
    fn max_rel_diff(ga: &LoraParams, gb_: &LoraParams) -> f32 {
        let mut max_rel = 0f32;
        for (k, a) in ga {
            let b = gb_.get(k).expect("same keys");
            let num = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
            let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        max_rel
    }

    /// Grads for one configuration at a small (resolution-agnostic for the math) edge.
    fn grads_at(
        dit: &mut LensTransformer,
        adapter: &TrainAdapter,
        params: &LoraParams,
        edge: u32,
        checkpoint_blocks: Option<&[Vec<String>]>,
        sdpa_ckpt: bool,
    ) -> LoraParams {
        dit.set_sdpa_checkpoint(sdpa_ckpt);
        let (x0, noise, feats, mask, latent) = synth(edge);
        let (_l, g) = compute_loss_grads(
            dit,
            params,
            adapter,
            16.0,
            16.0,
            &x0,
            &feats,
            &mask,
            0.5,
            &noise,
            false,
            None,
            Dtype::Float32,
            None,
            latent,
            checkpoint_blocks,
        )
        .unwrap();
        eval(g.values()).unwrap();
        g
    }

    /// Whole-block gradient checkpointing must not change the math: the checkpointed forward+grads
    /// must match the dense path within fp tolerance (it reuses the same install + block forward,
    /// recompute-only). Compares the production paths — dense (SDPA-ckpt on) vs block-checkpointed
    /// (SDPA-ckpt off, block recompute covers attention).
    #[test]
    #[ignore = "needs real SceneWorks/Lens weights; run as its own process"]
    fn checkpointed_grads_match_dense() {
        let mut dit = build_dit(Dtype::Float32);
        let (adapter, params) = build_targets(&mut dit);
        let locals = block_local_targets(&dit);
        let edge = 256u32; // small → dense is cheap; the math is resolution-agnostic
        let g_dense = grads_at(&mut dit, &adapter, &params, edge, None, true);
        let g_ckpt = grads_at(&mut dit, &adapter, &params, edge, Some(&locals), false);
        let max_rel = max_rel_diff(&g_dense, &g_ckpt);
        eprintln!("[sc-5170] checkpointed-vs-dense grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "checkpointed grads must match dense within tolerance: max rel {max_rel:.2e}"
        );
    }

    /// The always-on SDPA-segment checkpointing must not change the math: grads with the SDPA
    /// checkpoint on must match the retained backward (flag off). Same decomposed attention,
    /// recomputed instead of retained → expect (near-)bit-identical.
    #[test]
    #[ignore = "needs real SceneWorks/Lens weights; run as its own process"]
    fn attn_ckpt_grads_match_retained() {
        let mut dit = build_dit(Dtype::Float32);
        let (adapter, params) = build_targets(&mut dit);
        let edge = 256u32;
        let g_retained = grads_at(&mut dit, &adapter, &params, edge, None, false);
        let g_ckpt = grads_at(&mut dit, &adapter, &params, edge, None, true);
        let max_rel = max_rel_diff(&g_retained, &g_ckpt);
        eprintln!("[sc-5170] attn-ckpt-vs-retained grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-5,
            "SDPA-segment checkpointing must not change grads: max rel {max_rel:.2e}"
        );
    }

    /// The fit basis: first-step peak by resolution on the production DENSE path (SDPA-segment
    /// checkpointing always on), f32 then bf16. These printed points are what `projected_dense_peak_gb`
    /// is fit to — refit the `PREFLIGHT_F32`/`PREFLIGHT_BF16` constants if this prints materially
    /// different numbers. Plus a block-checkpointed 1024 point (the OOM mitigation).
    #[test]
    #[ignore = "needs real SceneWorks/Lens weights; run as its own process (may SIGKILL at 1024 dense)"]
    fn first_step_ckpt_sweep() {
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let tag = if dtype == Dtype::Float32 {
                "f32"
            } else {
                "bf16"
            };
            eprintln!("[sc-5170] dense first-step sweep ({tag}), SDPA-ckpt on:");
            let mut dit = build_dit(dtype);
            let (adapter, params) = build_targets(&mut dit);
            for edge in [512u32, 768, 1024] {
                let _ = one_step(
                    &mut dit,
                    &adapter,
                    &params,
                    edge,
                    dtype,
                    None,
                    true,
                    &format!("dense {tag}"),
                );
            }
            // The OOM mitigation: block-checkpointed 1024 (SDPA-ckpt off, block recompute covers it).
            let locals = block_local_targets(&dit);
            let _ = one_step(
                &mut dit,
                &adapter,
                &params,
                1024,
                dtype,
                Some(&locals),
                false,
                &format!("blk-ckpt {tag}"),
            );
            drop(dit);
            clear_cache();
        }
    }

    /// The fix demonstration: at production resolution 1024, gradient checkpointing must drop the
    /// first-step peak below the dense path's. Runs the dense step first (baseline), then the
    /// checkpointed step, and asserts a reduction that fits unified memory.
    #[test]
    #[ignore = "needs real SceneWorks/Lens weights; run as its own process"]
    fn first_step_1024_checkpointed_vs_dense() {
        let mut dit = build_dit(Dtype::Bfloat16);
        let (adapter, params) = build_targets(&mut dit);
        let locals = block_local_targets(&dit);
        let n: usize = locals.iter().map(|v| v.len()).sum();
        eprintln!("[sc-5170] checkpointing {n} LoRA targets across the 48-block stack");
        let (_, dense_peak) = one_step(
            &mut dit,
            &adapter,
            &params,
            1024,
            Dtype::Bfloat16,
            None,
            true,
            "dense bf16",
        );
        let (_, ckpt_peak) = one_step(
            &mut dit,
            &adapter,
            &params,
            1024,
            Dtype::Bfloat16,
            Some(&locals),
            false,
            "blk-ckpt bf16",
        );
        eprintln!(
            "[sc-5170] edge 1024 bf16  dense {dense_peak:.2} GB  ckpt {ckpt_peak:.2} GB  ({:.0}% reduction)",
            100.0 * (1.0 - ckpt_peak / dense_peak)
        );
        assert!(
            ckpt_peak < dense_peak,
            "checkpointing must reduce the first-step peak: dense {dense_peak:.2} vs ckpt {ckpt_peak:.2} GB"
        );
        assert!(
            ckpt_peak < 128.0,
            "checkpointed peak must fit unified memory: {ckpt_peak:.2} GB"
        );
    }

    /// sc-5188 — the silent-re-promotion detector. bf16 training must use materially LESS memory than
    /// f32, because bf16 activations are half-size and bf16 matmuls don't up-cast the weights. A
    /// strong-f32 constant anywhere in the DiT forward (the `1 + scale` AdaLN modulation was the
    /// culprit — block.rs `modulate` / transformer.rs `AdaLayerNormContinuous`) re-promotes the bf16
    /// stream to f32 at block 0 and cascades f32 through the whole net, erasing the bf16 win and adding
    /// the f32 weight up-casts ON TOP (so bf16 peaked ABOVE f32). This asserts the bf16 first-step peak
    /// is well below f32 at a mid bucket — a future re-promotion regression would lift the ratio back
    /// toward (or above) 1.0 and trip this. Measured ratio ≈ 0.50 (16.04 vs 31.83 GB at edge 768).
    #[test]
    #[ignore = "needs real SceneWorks/Lens weights; run as its own process"]
    fn bf16_peak_below_f32_no_silent_repromotion() {
        let edge = 768u32; // mid bucket — big enough to expose a re-promotion, cheap enough to run
        let f32_peak = {
            let mut dit = build_dit(Dtype::Float32);
            let (adapter, params) = build_targets(&mut dit);
            let (_, p) = one_step(
                &mut dit,
                &adapter,
                &params,
                edge,
                Dtype::Float32,
                None,
                true,
                "f32",
            );
            drop(dit);
            clear_cache();
            p
        };
        let bf16_peak = {
            let mut dit = build_dit(Dtype::Bfloat16);
            let (adapter, params) = build_targets(&mut dit);
            let (_, p) = one_step(
                &mut dit,
                &adapter,
                &params,
                edge,
                Dtype::Bfloat16,
                None,
                true,
                "bf16",
            );
            drop(dit);
            clear_cache();
            p
        };
        eprintln!(
            "[sc-5188] edge {edge}  f32 {f32_peak:.2} GB  bf16 {bf16_peak:.2} GB  (ratio {:.2})",
            bf16_peak / f32_peak
        );
        assert!(
            bf16_peak < f32_peak * 0.85,
            "bf16 first-step peak {bf16_peak:.2} GB must be well below f32 {f32_peak:.2} GB — a \
             strong-f32 constant is re-promoting the bf16 DiT stream to f32 (sc-5188)"
        );
    }

    /// Flattened cosine over all elements (f32).
    fn forward_cosine(a: &Array, b: &Array) -> f32 {
        let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
        let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
        let dot = a.multiply(&b).unwrap().sum(None).unwrap().item::<f32>();
        let na = a
            .multiply(&a)
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>()
            .sqrt();
        let nb = b
            .multiply(&b)
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>()
            .sqrt();
        dot / (na * nb)
    }

    /// sc-5188 — bf16 inference FIDELITY (the inference-side complement to the memory guard, and the
    /// guard the bf16 path never had — which is exactly why the re-promotion hid). With the fix the
    /// bf16 DiT forward genuinely runs bf16 (fed bf16 inputs, as the pipeline does); this asserts that
    /// true-bf16 output stays close to the f32 output on real weights. A cosine well below 1 would mean
    /// the bf16 path is lossy/broken — not merely lower precision. (Pre-fix this was a vacuous ~1.0
    /// because the bf16 stream WAS silently f32.) Inputs are fed at the run dtype, mirroring inference:
    /// the pipeline casts latents + caption features to the DiT dtype before the forward.
    #[test]
    #[ignore = "needs real SceneWorks/Lens weights; run as its own process"]
    fn bf16_forward_matches_f32() {
        let edge = 256u32; // small — the dtype fidelity is resolution-agnostic
        let (x0, _noise, feats, mask, latent) = synth(edge);
        let timestep = Array::from_slice(&[0.5f32], &[1]);

        let out_f32 = {
            let dit = build_dit(Dtype::Float32);
            dit.forward(&x0, &feats, Some(&mask), &timestep, 1, latent, latent)
                .unwrap()
        };
        // Feed bf16 inputs (the inference convention: the pipeline casts latents + features to the DiT
        // dtype). Without the sc-5188 fix the AdaLN `1 + scale` would re-promote this straight back to
        // f32 at block 0, making the test vacuous; with the fix the stream stays bf16 end to end.
        let x0_b = x0.as_dtype(Dtype::Bfloat16).unwrap();
        let feats_b: Vec<Array> = feats
            .iter()
            .map(|f| f.as_dtype(Dtype::Bfloat16).unwrap())
            .collect();
        let out_bf16 = {
            let dit = build_dit(Dtype::Bfloat16);
            dit.forward(&x0_b, &feats_b, Some(&mask), &timestep, 1, latent, latent)
                .unwrap()
        };
        let cos = forward_cosine(&out_f32, &out_bf16);
        eprintln!("[sc-5188] bf16-vs-f32 DiT forward cosine {cos:.6}");
        assert!(
            cos > 0.99,
            "bf16 DiT forward must stay close to f32 (sound bf16, not broken): cosine {cos:.6}"
        );
    }
}

#[cfg(test)]
mod preflight_tests {
    use super::{check_preflight_budget, projected_dense_peak_gb};

    /// The empirical fit must reproduce the measured first-step peaks and stay monotonic — it is the
    /// basis of the pre-flight OOM guard, so a regression here silently mis-sizes the guard. The
    /// points come from `first_step_ckpt_sweep` (the training dense path always runs SDPA-segment
    /// checkpointing since sc-5170). s = (edge/16)² + 64: edge 512/768/1024 → 1088/2368/4160.
    #[test]
    fn projection_matches_measured_curve() {
        for (s, measured) in [(1088.0, 22.78), (2368.0, 31.83), (4160.0, 45.22)] {
            let p = projected_dense_peak_gb(s, false);
            assert!(
                (p - measured).abs() < 1.0,
                "f32 projection at s={s} = {p:.2} GB, expected ≈{measured} GB"
            );
        }
        for (s, measured) in [(1088.0, 11.52), (2368.0, 16.04), (4160.0, 22.79)] {
            let p = projected_dense_peak_gb(s, true);
            assert!(
                (p - measured).abs() < 1.0,
                "bf16 projection at s={s} = {p:.2} GB, expected ≈{measured} GB"
            );
        }
        // Monotonic increasing in token count, in both dtypes.
        for bf16 in [false, true] {
            assert!(projected_dense_peak_gb(1088.0, bf16) < projected_dense_peak_gb(2368.0, bf16));
            assert!(projected_dense_peak_gb(2368.0, bf16) < projected_dense_peak_gb(4160.0, bf16));
        }
        // Post sc-5188 (bf16 re-promotion fix) the Lens bf16 dense path is ~half f32 (like z-image). A
        // regression that re-promotes the bf16 stream to f32 would lift bf16 back to/above f32, so pin
        // bf16 < f32 — the projected-curve analog of the real-weight `bf16_peak_below_f32` guard.
        assert!(projected_dense_peak_gb(4160.0, true) < projected_dense_peak_gb(4160.0, false));
    }

    /// The guard must FIRE (catchable error, not SIGKILL) when the dense first-step peak exceeds the
    /// machine's safe budget, and PASS when it fits — the sc-5170 acceptance for the dense-OOM case.
    #[test]
    fn guard_fires_over_budget_and_passes_under() {
        // A 24 GB-class budget (safe ≈ 20.4 GB): dense 1024 (~47 GB) must be refused with an
        // actionable error that recommends Gradient Checkpointing.
        let err = check_preflight_budget(1024, true, 24.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Gradient Checkpointing"), "got: {err}");
        assert!(
            err.contains("1024"),
            "error should name the resolution: {err}"
        );
        // A 128 GB-class budget (safe ≈ 108 GB) comfortably fits dense 1024 in both dtypes.
        assert!(check_preflight_budget(1024, true, 128.0).is_ok());
        assert!(check_preflight_budget(1024, false, 128.0).is_ok());
        // A 64 GB-class budget (safe ≈ 54 GB) fits 1024 dense in both dtypes (f32 ~45 GB, bf16 ~23 GB
        // post sc-5188) but a much larger resolution still overflows — the guard is machine-aware, not
        // a fixed threshold. f32 1440 (~78 GB) exceeds the 64 GB safe budget; bf16 1440 (~40 GB) fits
        // 64 GB but not a 32 GB machine.
        assert!(check_preflight_budget(1024, true, 64.0).is_ok());
        assert!(check_preflight_budget(1024, false, 64.0).is_ok());
        assert!(check_preflight_budget(1440, false, 64.0).is_err());
        assert!(check_preflight_budget(1440, true, 32.0).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::CancelFlag;
    use std::path::PathBuf;

    /// sc-24828 × sc-2127: with mask loss on and two buckets, each bucket's weight map has THAT
    /// bucket's latent shape, and the background (right half of the center-square crop, with
    /// `background_weight` 0) is zero at both grids. The 48×32 image's center square is
    /// x ∈ [8, 40); the subject is x < 24 — the crop's left half.
    /// Packed latent `[1, (g/2)², 4·C]`, C = 2: cell `(y, x)` sits at token
    /// `(y/2)·(g/2) + x/2`, feature `c·4 + (y%2)·2 + x%2`.
    #[test]
    fn subject_mask_weight_is_computed_per_bucket() {
        let tmp = tempfile::tempdir().unwrap();
        let (iw, ih) = (48u32, 32u32);
        let image_path = tmp.path().join("a.png");
        image::RgbImage::new(iw, ih).save(&image_path).unwrap();
        let mask_path = tmp.path().join("a_mask.png");
        image::GrayImage::from_fn(iw, ih, |x, _| image::Luma([if x < 24 { 255 } else { 0 }]))
            .save(&mask_path)
            .unwrap();
        let item = gen_core::TrainingItem {
            image_path,
            caption: "a".into(),
            subject_mask_path: Some(mask_path),
            ..Default::default()
        };
        let mask_cfg = gen_core::SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        };
        let mask =
            mlx_gen::train::subject_mask::PreparedSubjectMask::load("t", &item, &mask_cfg).unwrap();
        let entries = encode_buckets(&[32, 48], Some(&mask), |edge| {
            let g = (edge / 8) as i32;
            Ok(Array::zeros::<f32>(&[1, (g / 2) * (g / 2), 8])?)
        })
        .unwrap();
        assert_eq!(entries.len(), 2);
        for ((x0, w), g) in entries.iter().zip([4usize, 6]) {
            let w = w.as_ref().expect("mask loss on ⇒ a weight map");
            assert_eq!(
                w.shape(),
                x0.shape(),
                "weight must match its own bucket's latent"
            );
            let dense = mlx_rs::ops::multiply(w, Array::ones::<f32>(w.shape()).unwrap()).unwrap();
            let v = dense.as_slice::<f32>();
            for y in 0..g {
                for x in 0..g {
                    let val = v[((y / 2) * (g / 2) + x / 2) * 8 + (y % 2) * 2 + x % 2];
                    if x < g / 2 {
                        assert!(val > 0.99, "subject cell ({y},{x}) of {g}x{g} = {val}");
                    } else {
                        assert_eq!(val, 0.0, "background cell ({y},{x}) of {g}x{g}");
                    }
                }
            }
        }
    }

    fn base_config() -> TrainingConfig {
        TrainingConfig {
            rank: 8,
            steps: 10,
            ..Default::default()
        }
    }

    fn req_with(config: TrainingConfig) -> TrainingRequest {
        TrainingRequest {
            items: vec![mlx_gen::TrainingItem {
                image_path: PathBuf::from("/tmp/x.png"),
                caption: "a swatch".into(),
                control_image_path: None,
                model_options: Default::default(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            }],
            config,
            output_dir: PathBuf::from("/tmp/lens_unused"),
            file_name: "lora.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    #[test]
    fn descriptor_is_the_base_lens_id() {
        let d = trainer_descriptor();
        assert_eq!(d.id, "lens");
        assert_eq!(d.family, "lens");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.supports_lora && d.supports_lokr);
        // sc-24828: the one loss path (dense + block-checkpointed) reduces through the packed
        // subject-mask weight.
        assert!(d.techniques.subject_mask_loss);
    }

    /// sc-24828: the subject-mask weight reaches BOTH backward paths (dense + block-checkpointed) of
    /// [`compute_loss_grads`] on a tiny synthetic Lens DiT (seeded random weights for every tensor
    /// `LensTransformer::from_weights` reads — the `tests/cfg_gate_equivalence.rs` geometry). The
    /// weight is built on the unpacked `[1, C, 2·L, 2·L]` grid and packed by [`pack_like_latent`].
    /// An all-ones map equals the unweighted loss; an all-zero map gives loss exactly 0 and all-zero
    /// adapter grads on both paths; a half map lands strictly between and agrees across paths.
    #[test]
    fn subject_mask_weight_reaches_both_backward_paths() {
        use mlx_gen::train::loss::subject_mask_weight;
        use std::collections::HashMap;
        let dcfg = LensDitConfig {
            patch_size: 2,
            in_channels: 8,
            out_channels: 2,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            inner_dim: 16,
            enc_hidden_dim: 6,
            axes_dims_rope: [2, 2, 4],
            num_text_layers: 2,
        };
        let (dim, hidden) = (dcfg.inner_dim, 32);
        let txt_in = dcfg.enc_hidden_dim * dcfg.num_text_layers as i32;
        let out_w = dcfg.patch_size * dcfg.patch_size * dcfg.out_channels;
        let mut shapes: Vec<(String, Vec<i32>)> = vec![
            ("img_in.weight".into(), vec![dim, dcfg.in_channels]),
            ("img_in.bias".into(), vec![dim]),
            ("txt_in.weight".into(), vec![dim, txt_in]),
            ("txt_in.bias".into(), vec![dim]),
            (
                "time_text_embed.timestep_embedder.linear_1.weight".into(),
                vec![dim, 256],
            ),
            (
                "time_text_embed.timestep_embedder.linear_1.bias".into(),
                vec![dim],
            ),
            (
                "time_text_embed.timestep_embedder.linear_2.weight".into(),
                vec![dim, dim],
            ),
            (
                "time_text_embed.timestep_embedder.linear_2.bias".into(),
                vec![dim],
            ),
            ("norm_out.linear.weight".into(), vec![2 * dim, dim]),
            ("norm_out.linear.bias".into(), vec![2 * dim]),
            ("proj_out.weight".into(), vec![out_w, dim]),
            ("proj_out.bias".into(), vec![out_w]),
        ];
        for i in 0..dcfg.num_text_layers {
            shapes.push((format!("txt_norm.{i}.weight"), vec![dcfg.enc_hidden_dim]));
        }
        for b in 0..dcfg.num_layers {
            let p = format!("transformer_blocks.{b}");
            for m in ["img_mod.1", "txt_mod.1"] {
                shapes.push((format!("{p}.{m}.weight"), vec![6 * dim, dim]));
                shapes.push((format!("{p}.{m}.bias"), vec![6 * dim]));
            }
            for n in ["img_norm1", "img_norm2", "txt_norm1", "txt_norm2"] {
                shapes.push((format!("{p}.{n}.weight"), vec![dim]));
            }
            for n in ["img_qkv", "txt_qkv"] {
                shapes.push((format!("{p}.attn.{n}.weight"), vec![3 * dim, dim]));
                shapes.push((format!("{p}.attn.{n}.bias"), vec![3 * dim]));
            }
            for n in ["to_out.0", "to_add_out"] {
                shapes.push((format!("{p}.attn.{n}.weight"), vec![dim, dim]));
                shapes.push((format!("{p}.attn.{n}.bias"), vec![dim]));
            }
            for n in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
                shapes.push((format!("{p}.attn.{n}.weight"), vec![dcfg.head_dim]));
            }
            for m in ["img_mlp", "txt_mlp"] {
                shapes.push((format!("{p}.{m}.w1.weight"), vec![hidden, dim]));
                shapes.push((format!("{p}.{m}.w3.weight"), vec![hidden, dim]));
                shapes.push((format!("{p}.{m}.w2.weight"), vec![dim, hidden]));
            }
        }
        let draw = |shape: &[i32], k: u64| {
            random::normal::<f32>(shape, None, None, Some(&random::key(k).unwrap())).unwrap()
        };
        let map: HashMap<String, Array> = shapes
            .iter()
            .enumerate()
            .map(|(i, (n, sh))| (n.clone(), draw(sh, i as u64 + 1)))
            .collect();
        let mut dit =
            LensTransformer::from_weights(&Weights::from_map(map), &dcfg, Dtype::Float32).unwrap();
        let cfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&dit, &cfg);
        assert!(!target_paths.is_empty());
        let (targets, params) = build_lora_targets(&mut dit, &target_paths, 4, 7).unwrap();
        // Non-zero factors on both sides (lora_b inits at zero, which would zero the lora_a grads
        // trivially).
        let scale = Array::from_slice(&[0.05f32], &[1]);
        let params: LoraParams = params
            .iter()
            .enumerate()
            .map(|(i, (k, v))| {
                (
                    k.clone(),
                    multiply(draw(v.shape(), 100 + i as u64), &scale).unwrap(),
                )
            })
            .collect();
        let adapter = TrainAdapter::Lora { targets };
        let mut locals: Vec<Vec<String>> = vec![Vec::new(); dcfg.num_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    locals[i].push(local.to_string());
                }
            }
        }
        assert!(locals.iter().any(|l| !l.is_empty()));

        // Packed latent `[1, L·L, 4·C]` (L = 2 patch cells, C = 2 unpacked channels).
        let latent = 2usize;
        let unpacked = [1i32, 2, 2 * latent as i32, 2 * latent as i32];
        let x0 = pack_like_latent(&draw(&unpacked, 201)).unwrap();
        let noise = draw(x0.shape(), 202);
        let feats: Vec<Array> = (0..dcfg.num_text_layers)
            .map(|i| draw(&[1, 3, dcfg.enc_hidden_dim], 300 + i as u64))
            .collect();
        let mask = Array::ones::<f32>(&[1, 3]).unwrap();
        let side = 2 * latent;
        let wmap = |v: &[f32]| {
            pack_like_latent(&subject_mask_weight(v, side, side, &unpacked).unwrap()).unwrap()
        };
        let mut run = |weight: Option<&Array>, ckpt: bool| {
            let (l, g) = compute_loss_grads(
                &mut dit,
                &params,
                &adapter,
                4.0,
                4.0,
                &x0,
                &feats,
                &mask,
                0.5,
                &noise,
                false,
                weight,
                Dtype::Float32,
                None,
                latent,
                ckpt.then_some(locals.as_slice()),
            )
            .unwrap();
            eval(g.values()).unwrap();
            (l, g)
        };
        let n = side * side;
        let (plain, _) = run(None, false);
        let ones = wmap(&vec![1.0; n]);
        assert!((run(Some(&ones), false).0 - plain).abs() < 1e-6);
        let zeros = wmap(&vec![0.0; n]);
        for ckpt in [false, true] {
            let (loss, grads) = run(Some(&zeros), ckpt);
            assert_eq!(
                loss, 0.0,
                "ckpt={ckpt}: an all-background map must zero the loss"
            );
            assert!(!grads.is_empty());
            for (k, g) in &grads {
                let m = g.abs().unwrap().max(None).unwrap().item::<f32>();
                assert_eq!(m, 0.0, "ckpt={ckpt}: nonzero adapter grad on {k}");
            }
        }
        let half: Vec<f32> = (0..n)
            .map(|i| if i % side < side / 2 { 1.0 } else { 0.0 })
            .collect();
        let half = wmap(&half);
        let (dense, _) = run(Some(&half), false);
        let (ckpt, _) = run(Some(&half), true);
        assert!(dense > 0.0 && dense < plain, "{dense} vs {plain}");
        assert!(
            (dense - ckpt).abs() < 1e-4,
            "dense {dense} vs checkpoint {ckpt}"
        );
    }

    /// sc-24828: the subject-mask weight is packed EXACTLY like the training latent. A 16×16 mask whose
    /// 4×4 pixel blocks each carry a distinct grey level area-averages onto a 4×4 unpacked latent
    /// grid (cell `(y, x)` = that block's level); `latent_subject_mask_weight` must then put cell
    /// `(2i+dy, 2j+dx)` at packed token `i·2 + j`, feature `c·4 + dy·2 + dx` for every channel `c` —
    /// the 2×2 `patchify_latents` → `pack_latents` order `encode_latents` applies to `x0`.
    #[test]
    fn subject_mask_weight_is_packed_like_the_latent() {
        let level = |y: u32, x: u32| 10 + (y * 4 + x) * 15;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mask.png");
        image::GrayImage::from_fn(16, 16, |px, py| image::Luma([level(py / 4, px / 4) as u8]))
            .save(&path)
            .unwrap();
        let mut cfg = base_config();
        cfg.subject_mask_loss = Some(gen_core::SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        });
        let mut item = req_with(base_config()).items.remove(0);
        item.image_path = path.clone();
        item.subject_mask_path = Some(path);
        // Packed latent `[1, (4/2)·(4/2), 4·C]` with C = 2 unpacked channels.
        let x0 = Array::zeros::<f32>(&[1, 4, 8]).unwrap();
        let prepared =
            PreparedSubjectMask::load_if_enabled("t", &item, cfg.subject_mask_loss.as_ref())
                .unwrap();
        let w = latent_subject_mask_weight(prepared.as_ref(), &x0)
            .unwrap()
            .expect("technique on ⇒ a weight");
        assert_eq!(w.shape(), x0.shape());
        let w = multiply(&w, Array::ones::<f32>(w.shape()).unwrap()).unwrap();
        let v = w.as_slice::<f32>();
        for (i, j) in [(0u32, 0u32), (0, 1), (1, 0), (1, 1)] {
            for c in 0..2u32 {
                for (dy, dx) in [(0u32, 0u32), (0, 1), (1, 0), (1, 1)] {
                    let tok = (i * 2 + j) as usize;
                    let feat = (c * 4 + dy * 2 + dx) as usize;
                    let got = v[tok * 8 + feat];
                    let want = level(2 * i + dy, 2 * j + dx) as f32 / 255.0;
                    assert!(
                        (got - want).abs() < 1e-6,
                        "token {tok} feature {feat}: {got} != cell ({}, {}) = {want}",
                        2 * i + dy,
                        2 * j + dx
                    );
                }
            }
        }
        // Off ⇒ no weight (and no file read).
        let off = PreparedSubjectMask::load_if_enabled(
            "t",
            &item,
            base_config().subject_mask_loss.as_ref(),
        )
        .unwrap();
        assert!(off.is_none());
        assert!(latent_subject_mask_weight(off.as_ref(), &x0)
            .unwrap()
            .is_none());
    }

    #[test]
    fn descriptor_declares_resolution_buckets() {
        // sc-2127: the shared technique floor only lets `resolution_buckets` through when declared.
        assert!(trainer_descriptor().techniques.resolution_buckets);
    }

    #[test]
    fn preflight_sizes_for_the_largest_bucket_edge() {
        let rb = |resolution, repeats| gen_core::ResolutionBucket {
            resolution,
            repeats,
        };
        // Buckets off: the guard edge is the single legacy edge.
        let mut cfg = TrainingConfig {
            resolution: 512,
            ..base_config()
        };
        assert_eq!(preflight_edge(&bucket_edges(&cfg)), 512);
        // Buckets [512, 1024] with `resolution` 512: the guard must size for 1024 (epic 2123 E7).
        cfg.resolution_buckets = vec![rb(512, 16), rb(1024, 1)];
        let edges = bucket_edges(&cfg);
        assert_eq!(edges, vec![512, 1024]);
        assert_eq!(preflight_edge(&edges), 1024);
        // A 24 GB-class budget (safe ≈ 20.4 GB): dense bf16 fits 512 (~11.5 GB) but not 1024
        // (~22.8 GB) — so the bucketed run is refused even though `resolution` alone would pass.
        assert!(check_preflight_budget(512, true, 24.0).is_ok());
        assert!(check_preflight_budget(preflight_edge(&edges), true, 24.0).is_err());
    }

    #[test]
    fn latent_grid_is_per_bucket_edge() {
        // sc-2127: each cached latent carries the grid of ITS edge — the DiT's img ids span
        // `grid × grid`, and `grid²` must equal the packed latent's token count `(edge/16)²`.
        let rb = |resolution, repeats| gen_core::ResolutionBucket {
            resolution,
            repeats,
        };
        let cfg = TrainingConfig {
            resolution_buckets: vec![rb(512, 16), rb(770, 4), rb(1024, 1)],
            ..base_config()
        };
        let grids: Vec<usize> = bucket_edges(&cfg).into_iter().map(latent_grid).collect();
        assert_eq!(grids, vec![32, 48, 64]);
        for edge in bucket_edges(&cfg) {
            let tokens = (edge as usize / 16).pow(2);
            assert_eq!(latent_grid(edge).pow(2), tokens, "edge {edge}");
        }
    }

    #[test]
    fn validate_rejects_empty_dataset_and_zero_rank_steps() {
        let mut r = req_with(base_config());
        r.items.clear();
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("dataset is empty"));

        let r = req_with(TrainingConfig {
            rank: 0,
            ..base_config()
        });
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("rank"));

        let r = req_with(TrainingConfig {
            steps: 0,
            ..base_config()
        });
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("steps"));
    }

    #[test]
    fn validate_accepts_gradient_checkpointing() {
        // sc-5170 — gradient_checkpointing is now a supported toggle (checkpointed DiT forward + OOM
        // preflight guard), no longer a hard rejection. It must pass capability-free validation; the
        // checkpointed path is exercised in `train_impl` (and the real-weight harness).
        let r = req_with(TrainingConfig {
            gradient_checkpointing: true,
            ..base_config()
        });
        assert!(validate_request(&r).is_ok());
    }

    #[test]
    fn validate_rejects_unrecognized_optimizer_timestep_loss() {
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
        // The recognized spellings (incl. alias normalization) pass.
        assert!(validate_request(&req_with(TrainingConfig {
            timestep_type: "Weighted".into(),
            timestep_bias: "high-noise".into(),
            loss_type: "L1".into(),
            optimizer: "adamw".into(),
            ..base_config()
        }))
        .is_ok());
    }

    #[test]
    fn build_batch_is_lens_velocity_with_no_sign_flip() {
        // target = noise − x0 (the RAW Lens DiT velocity; the OPPOSITE sign of z-image's negated
        // forward), and x_t = (1−t)·x0 + t·noise. Timestep `t` is passed straight to the DiT (the
        // caller), unlike z-image's `1 − σ` — covered by checking the interpolation here.
        let x0 = Array::from_slice(&[2.0f32, 4.0, 6.0], &[1, 3, 1]);
        let noise = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3, 1]);
        let t = 0.25f32;
        let (x_t, target) = build_batch(&x0, &noise, t).unwrap();
        // target = noise − x0 = [-1, -3, -5]
        assert_eq!(target.as_slice::<f32>(), &[-1.0, -3.0, -5.0]);
        // x_t = 0.75·x0 + 0.25·noise = [1.75, 3.25, 4.75]
        let xt = x_t.as_slice::<f32>();
        for (got, want) in xt.iter().zip([1.75f32, 3.25, 4.75].iter()) {
            assert!((got - want).abs() < 1e-6, "x_t {got} != {want}");
        }
    }

    #[test]
    fn sample_sigma_is_deterministic_and_in_range() {
        for kind in ["sigmoid", "linear", "weighted"] {
            for bias in ["balanced", "high", "low"] {
                let a = sample_sigma(kind, bias, 42).unwrap();
                let b = sample_sigma(kind, bias, 42).unwrap();
                assert_eq!(a, b, "{kind}/{bias} must be deterministic in seed");
                assert!(
                    (1e-3..=1.0 - 1e-3).contains(&a),
                    "{kind}/{bias} t={a} out of range"
                );
            }
        }
        // high-noise bias (√t) lifts the value vs low-noise bias (t²) for the same draw.
        assert!(
            sample_sigma("sigmoid", "high", 7).unwrap()
                > sample_sigma("sigmoid", "low", 7).unwrap()
        );
    }
}

/// sc-24830 (epic 2123 depth anchoring) — the Lens step seam ([`run_train_step`] /
/// [`compute_step_loss_grads`]) on a tiny random-init Lens DiT (the `tests/cfg_gate_equivalence.rs`
/// geometry: 2 blocks, packed 8-channel latent = 2 unpacked channels), a random-init tiny 2-channel
/// TAESD decoder and a random-init tiny Depth-Anything-V2. Seconds; no weights downloaded.
#[cfg(test)]
mod depth_anchoring_tests {
    use super::*;
    use mlx_gen::train::perceptual::AuxLossSchedule;
    use std::collections::HashMap;

    fn tiny_dit() -> LensTransformer {
        let dcfg = LensDitConfig {
            patch_size: 2,
            in_channels: 8,
            out_channels: 2,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            inner_dim: 16,
            enc_hidden_dim: 6,
            axes_dims_rope: [2, 2, 4],
            num_text_layers: 2,
        };
        let (dim, hidden) = (dcfg.inner_dim, 32);
        let txt_in = dcfg.enc_hidden_dim * dcfg.num_text_layers as i32;
        let out_w = dcfg.patch_size * dcfg.patch_size * dcfg.out_channels;
        let mut shapes: Vec<(String, Vec<i32>)> = vec![
            ("img_in.weight".into(), vec![dim, dcfg.in_channels]),
            ("img_in.bias".into(), vec![dim]),
            ("txt_in.weight".into(), vec![dim, txt_in]),
            ("txt_in.bias".into(), vec![dim]),
            (
                "time_text_embed.timestep_embedder.linear_1.weight".into(),
                vec![dim, 256],
            ),
            (
                "time_text_embed.timestep_embedder.linear_1.bias".into(),
                vec![dim],
            ),
            (
                "time_text_embed.timestep_embedder.linear_2.weight".into(),
                vec![dim, dim],
            ),
            (
                "time_text_embed.timestep_embedder.linear_2.bias".into(),
                vec![dim],
            ),
            ("norm_out.linear.weight".into(), vec![2 * dim, dim]),
            ("norm_out.linear.bias".into(), vec![2 * dim]),
            ("proj_out.weight".into(), vec![out_w, dim]),
            ("proj_out.bias".into(), vec![out_w]),
        ];
        for i in 0..dcfg.num_text_layers {
            shapes.push((format!("txt_norm.{i}.weight"), vec![dcfg.enc_hidden_dim]));
        }
        for b in 0..dcfg.num_layers {
            let p = format!("transformer_blocks.{b}");
            for m in ["img_mod.1", "txt_mod.1"] {
                shapes.push((format!("{p}.{m}.weight"), vec![6 * dim, dim]));
                shapes.push((format!("{p}.{m}.bias"), vec![6 * dim]));
            }
            for n in ["img_norm1", "img_norm2", "txt_norm1", "txt_norm2"] {
                shapes.push((format!("{p}.{n}.weight"), vec![dim]));
            }
            for n in ["img_qkv", "txt_qkv"] {
                shapes.push((format!("{p}.attn.{n}.weight"), vec![3 * dim, dim]));
                shapes.push((format!("{p}.attn.{n}.bias"), vec![3 * dim]));
            }
            for n in ["to_out.0", "to_add_out"] {
                shapes.push((format!("{p}.attn.{n}.weight"), vec![dim, dim]));
                shapes.push((format!("{p}.attn.{n}.bias"), vec![dim]));
            }
            for n in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
                shapes.push((format!("{p}.attn.{n}.weight"), vec![dcfg.head_dim]));
            }
            for m in ["img_mlp", "txt_mlp"] {
                shapes.push((format!("{p}.{m}.w1.weight"), vec![hidden, dim]));
                shapes.push((format!("{p}.{m}.w3.weight"), vec![hidden, dim]));
                shapes.push((format!("{p}.{m}.w2.weight"), vec![dim, hidden]));
            }
        }
        let scale = Array::from_slice(&[0.3f32], &[1]);
        let map: HashMap<String, Array> = shapes
            .iter()
            .enumerate()
            .map(|(i, (n, sh))| {
                let a = random::normal::<f32>(
                    sh,
                    None,
                    None,
                    Some(&random::key(i as u64 + 1).unwrap()),
                )
                .unwrap();
                (n.clone(), multiply(&a, &scale).unwrap())
            })
            .collect();
        LensTransformer::from_weights(&Weights::from_map(map), &dcfg, Dtype::Float32).unwrap()
    }

    fn aux_sched() -> AuxLossSchedule {
        AuxLossSchedule {
            weight: 0.1,
            t_min: 0.0,
            t_max: 1.0,
            every_n: 2,
        }
    }

    fn path_with(s: AuxLossSchedule) -> PerceptualPath {
        mlx_gen_perceptual::testing::tiny_depth_path(2, s).unwrap()
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

    fn rnd(shape: &[i32], k: u64) -> Array {
        let a = random::normal::<f32>(shape, None, None, Some(&random::key(k).unwrap())).unwrap();
        eval([&a]).unwrap();
        a
    }

    /// One cache entry at latent grid `g`: a packed `[1, g², 8]` clean latent (laid out by the real
    /// `pack_like_latent` from an unpacked `[1, 2, 2g, 2g]` grid) + tiny caption features.
    fn entry(g: usize, k: u64) -> CacheEntry {
        let side = 2 * g as i32;
        let x0 = pack_like_latent(&rnd(&[1, 2, side, side], k)).unwrap();
        let feats = (0..2).map(|i| rnd(&[1, 3, 6], 300 + k + i)).collect();
        (x0, feats, Array::ones::<f32>(&[1, 3]).unwrap(), g, None)
    }

    struct Fixture {
        dit: LensTransformer,
        adapter: TrainAdapter,
        params: LoraParams,
        locals: Vec<Vec<String>>,
    }

    fn fixture(cfg: &TrainingConfig) -> Fixture {
        let mut dit = tiny_dit();
        let paths = resolve_target_paths(&dit, cfg);
        let (targets, params) =
            build_lora_targets(&mut dit, &paths, cfg.rank as i32, cfg.seed).unwrap();
        let mut locals: Vec<Vec<String>> = vec![Vec::new(); 2];
        for p in &paths {
            if let Some((idx, local)) = p
                .strip_prefix("transformer_blocks.")
                .and_then(|r| r.split_once('.'))
            {
                locals[idx.parse::<usize>().unwrap()].push(local.to_string());
            }
        }
        Fixture {
            dit,
            adapter: TrainAdapter::Lora { targets },
            params,
            locals,
        }
    }

    fn single_bucket(n: usize) -> BucketSchedule {
        BucketSchedule::new(
            n,
            &[gen_core::ResolutionBucket {
                resolution: 32,
                repeats: 1,
            }],
            7,
        )
    }

    fn step(
        f: &mut Fixture,
        cfg: &TrainingConfig,
        cache: &[CacheEntry],
        schedule: &BucketSchedule,
        path: Option<(&mut PerceptualPath, &mut AuxAlternation)>,
        n: u32,
        ckpt: bool,
    ) -> (StepLosses, LoraParams) {
        let locals = f.locals.clone();
        let (l, g) = run_train_step(
            &mut f.dit,
            &f.params,
            &f.adapter,
            cfg,
            cache,
            schedule,
            path,
            n,
            false,
            Dtype::Float32,
            None,
            ckpt.then_some(locals.as_slice()),
        )
        .unwrap();
        eval(g.values()).unwrap();
        (l, g)
    }

    /// `unpack_to_decoder_layout` exactly inverts `patchify_latents` → `pack_latents` (so TAEF2
    /// sees the 32-channel grid the VAE produced, cell for cell). Mutation: swap the `i`/`j` axes
    /// in the inverse patchify ⇒ red.
    #[test]
    fn unpack_inverts_the_lens_packing() {
        let sq = rnd(&[1, 3, 6, 6], 6);
        let packed = pack_like_latent(&sq).unwrap();
        let back = unpack_to_decoder_layout(&packed, 3).unwrap();
        assert_eq!(back.shape(), sq.shape());
        let err = back
            .subtract(&sq)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert_eq!(err, 0.0);
        assert!(unpack_to_decoder_layout(&packed, 2).is_err());
    }

    /// AC (a)/(b), dense and block-checkpointed. Mutation: force `diffusion_on = true` ⇒ red.
    #[test]
    fn depth_step_trains_the_lora_through_depth_only() {
        for ckpt in [false, true] {
            let cfg = cfg();
            let mut f = fixture(&cfg);
            let cache = vec![entry(2, 100)];
            let schedule = single_bucket(1);
            let mut p = path_with(aux_sched());
            prepare_perceptual_references(&mut p, &cache).unwrap();
            let mut alt = AuxAlternation::new(1, 1);
            let (diff, _) = step(
                &mut f,
                &cfg,
                &cache,
                &schedule,
                Some((&mut p, &mut alt)),
                1,
                ckpt,
            );
            assert_eq!(diff.aux, None, "ckpt={ckpt}");
            assert_eq!(Some(diff.total), diff.diffusion);
            let (depth, g) = step(
                &mut f,
                &cfg,
                &cache,
                &schedule,
                Some((&mut p, &mut alt)),
                2,
                ckpt,
            );
            assert_eq!(depth.diffusion, None, "ckpt={ckpt}");
            let aux = depth.aux.expect("depth term");
            assert!(aux > 0.0 && aux.is_finite(), "ckpt={ckpt}: {aux}");
            assert_eq!(depth.total, aux);
            let gb: f32 = g
                .iter()
                .filter(|(k, _)| k.ends_with(".lora_b"))
                .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
                .sum();
            assert!(gb > 0.0 && gb.is_finite(), "ckpt={ckpt}: LoRA-B |Σ| = {gb}");
        }
    }

    /// The aux step trains at `t` remapped into `[0.6, 0.8]`, and its depth term is the depth loss
    /// of the explicitly recovered and unpacked `x_t − t·v`. Mutations: keep the sampled `t` ⇒ red;
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
        let cache = vec![entry(2, 100)];
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
        let (x0, feats, mask, g, _) = &cache[0];
        let noise = random::normal::<f32>(
            x0.shape(),
            None,
            None,
            Some(&random::key(cfg.seed.wrapping_add(2).wrapping_mul(2) + 1).unwrap()),
        )
        .unwrap();
        f.adapter
            .install_as(&mut f.dit, &f.params, 4.0, 4.0, None, LOKR_DTYPE)
            .unwrap();
        let (x_t, _) = build_batch(x0, &noise, t).unwrap();
        let v = f
            .dit
            .forward(
                &x_t,
                feats,
                Some(mask),
                &Array::from_slice(&[t], &[1]),
                1,
                *g,
                *g,
            )
            .unwrap();
        let x0_hat = subtract(&x_t, multiply(&v, Array::from_f32(t)).unwrap()).unwrap();
        let want = p
            .aux_loss(&plan, 0, &unpack_to_decoder_layout(&x0_hat, *g).unwrap())
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

    /// Per-image alternation (round-robin N = 2) and per-entry references across two buckets of
    /// different grids. Mutations: key the plan on the global step ⇒ red; pass the item as the
    /// `AuxStep` entry ⇒ a bucket-1 depth step compares against bucket 0's reference ⇒ red.
    #[test]
    fn alternation_is_per_image_with_per_entry_references() {
        let cfg = cfg();
        let mut f = fixture(&cfg);
        // Round-robin, one bucket.
        let cache = vec![entry(2, 100), entry(2, 101)];
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
        // Two buckets: grid 2 and grid 3, item-major.
        let cache = vec![entry(2, 110), entry(3, 111), entry(2, 112), entry(3, 113)];
        let schedule = BucketSchedule::new(
            2,
            &[
                gen_core::ResolutionBucket {
                    resolution: 32,
                    repeats: 1,
                },
                gen_core::ResolutionBucket {
                    resolution: 48,
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
            // Recompute the step against the SCHEDULED entry's latent and reference.
            let k = (n - 1) as usize;
            let (item, entry) = (schedule.sample(k).0, schedule.cache_index(k));
            let raw = sample_sigma(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(n as u64),
            )
            .unwrap();
            let plan = p.plan(replay.key(n, item), entry, raw).unwrap();
            let (x0, feats, mask, g, _) = &cache[entry];
            let noise = random::normal::<f32>(
                x0.shape(),
                None,
                None,
                Some(&random::key(cfg.seed.wrapping_add(n as u64).wrapping_mul(2) + 1).unwrap()),
            )
            .unwrap();
            let (expected, _) = compute_step_loss_grads(
                &mut f.dit,
                &f.params,
                &f.adapter,
                cfg.alpha,
                cfg.rank as f32,
                x0,
                feats,
                mask,
                plan.noise_level,
                &noise,
                false,
                None,
                Dtype::Float32,
                None,
                *g,
                None,
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

    /// E1: depth off ⇒ nothing loaded, no footprint, and the step is bit-identical to the
    /// pre-epic-2123 closure; a diffusion-only step of an enabled path is bit-identical too.
    /// Mutation: flip MAE/MSE in the diffusion term ⇒ red.
    #[test]
    fn everything_off_is_bit_identical_to_the_legacy_step() {
        assert!(load_perceptual_path(&TrainingConfig::default())
            .unwrap()
            .is_none());
        assert_eq!(
            perceptual_footprint_gb(&TrainingConfig::default(), 1024, 10),
            0.0
        );
        let off_cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            seed: 7,
            ..Default::default()
        };
        let mut f = fixture(&off_cfg);
        let cache = vec![entry(2, 100), entry(2, 101)];
        let schedule = single_bucket(2);
        let (off, g_off) = step(&mut f, &off_cfg, &cache, &schedule, None, 1, false);
        assert_eq!(off.aux, None);
        let (x0, feats, mask, g, _) = &cache[0];
        let t = sample_sigma(
            &off_cfg.timestep_type,
            &off_cfg.timestep_bias,
            off_cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(1),
        )
        .unwrap();
        let noise = random::normal::<f32>(
            x0.shape(),
            None,
            None,
            Some(&random::key(off_cfg.seed.wrapping_add(1).wrapping_mul(2) + 1).unwrap()),
        )
        .unwrap();
        let (x_t, target) = build_batch(x0, &noise, t).unwrap();
        let timestep = Array::from_slice(&[t], &[1]);
        let (feats, mask, g) = (feats.clone(), mask.clone(), *g);
        let dit = &mut f.dit;
        let adapter = &f.adapter;
        let legacy = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
            adapter.install_as(dit, &p, 4.0, 4.0, None, LOKR_DTYPE)?;
            let v = dit
                .forward(&x_t, &feats, Some(&mask), &timestep, 1, g, g)
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

    /// E7: the estimate grows by TAEF2 + DA2 (more for Large); the guard refuses at a synthetic
    /// budget between the DiT projection and projection + aux on the dense and the checkpointed
    /// path; a checkpointed run without aux is unguarded. Mutations: drop `+ extra_gb` ⇒ red;
    /// return early for every checkpointed run ⇒ red.
    #[test]
    fn memory_estimate_includes_the_aux_models() {
        let mut on = TrainingConfig::default();
        on.depth_anchoring.schedule = aux_sched();
        let small = perceptual_footprint_gb(&on, 1024, 10);
        on.depth_anchoring.model_size = gen_core::train::DepthModelSize::Large;
        let large = perceptual_footprint_gb(&on, 1024, 10);
        assert!(small > 0.0 && large - small > 1.0, "{small} / {large}");
        // TAEF2 (pooled, 32ch) costs more than a plain TAEF1 decoder would.
        assert!(
            TinyDecoderSpec::taef2().footprint(1024, 1024).param_bytes
                > TinyDecoderSpec::taef1().footprint(1024, 1024).param_bytes
        );
        let tokens = (1024.0f64 / 16.0).powi(2) + PREFLIGHT_TXT_TOKENS;
        for (checkpointed, base) in [
            (false, projected_dense_peak_gb(tokens, true)),
            (true, projected_dense_peak_gb(0.0, true)),
        ] {
            let budget = (base + large / 2.0) / 0.85;
            assert!(check_preflight_budget_with_aux(1024, true, budget, 0.0, checkpointed).is_ok());
            assert!(
                check_preflight_budget_with_aux(1024, true, budget, large, checkpointed).is_err(),
                "checkpointed={checkpointed}"
            );
        }
        assert!(check_preflight_budget_with_aux(1024, true, 0.001, 0.0, true).is_ok());
    }

    /// E3: Lens declares depth anchoring.
    #[test]
    fn descriptor_declares_depth_anchoring() {
        assert!(trainer_descriptor().techniques.depth_anchoring);
    }

    /// A missing decoder is a named error (TAEF2).
    #[test]
    fn missing_aux_weights_are_named() {
        let tmp = tempfile::tempdir().unwrap();
        let mut c = cfg();
        c.perceptual_decoder_dir = Some(tmp.path().join("no-taef2"));
        c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
        let err = load_perceptual_path(&c)
            .err()
            .expect("must fail")
            .to_string();
        assert!(err.contains("TAEF2"), "{err}");
    }
}
