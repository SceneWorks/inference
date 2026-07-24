//! Native-MLX control-branch **TRAINER** for Krea 2 (sc-10177 / epic 10159 "B5") — the MLX twin of
//! the Candle `krea_2_control` trainer. Trains a control branch on the **frozen `krea_2_turbo` base**
//! and deploys on Turbo: there is no raw→turbo crossing, because the branch copies the base's own
//! blocks and injects back into that same frozen stream (the branch and base are one checkpoint).
//!
//! ## How it trains (the MLX-native shape)
//! The inference [`Krea2ControlBranch`](crate::control) reads its weights from `&self`, which
//! `keyed_value_and_grad` cannot differentiate. So the trainable branch lives **outside** the model in
//! an external param map ([`LoraParams`]) keyed IDENTICALLY to the overlay on-disk format
//! (`blocks.{i}.<leaf>`, RMSNorm scales pre-folded as `*.weight_p1`, `blocks.{i}.proj_out.weight`), and
//! each step **reconstructs** a real [`Krea2ControlBranch`] from that map inside the traced loss and
//! calls its own [`forward`](crate::control::Krea2ControlBranch) at `control_scale = 1.0`. Every hop —
//! `Weights::insert` → `Krea2ControlBranch::from_weights` → the block matmuls — is a traced handle on the
//! same array, so gradients reach the map; and because the *inference* forward IS the training forward,
//! train == inference is structurally true. Saving is then just dumping the map (plus `meta.inject_offset`,
//! written OUTSIDE the trace so its `.item()` read never runs mid-graph), which loads back through the
//! unmodified inference [`Krea2ControlBranch::from_weights`] — the round-trip AC.
//!
//! Reuses the [`KreaRawTrainer`](crate::training) scaffold — the functional-autograd loop
//! (`keyed_value_and_grad` over an external param map re-injected each step), the core
//! [`TrainOptimizer`], the flow-match velocity target (`noise − x0`, timestep `t`), the
//! encode-cache-then-drop-TE pattern, and SDPA-segment gradient checkpointing. Unlike LoRA (rank-r
//! factors onto `AdaptableLinear`s), the control branch's full block weights ARE the differentiation
//! leaves, reconstructed-from-map inside the traced loss.
//!
//! The magnitude-control recipe mirrors the Candle trainer's run-3 fix (control_trainer.rs): a
//! **two-group AdamW** — the branch bodies at the full lr with no decay, the zero-init injection
//! projections at [`PROJ_LR_MULT`]×lr with decoupled weight decay [`PROJ_WEIGHT_DECAY`] — because
//! AdamW's normalized steps otherwise regrow a zero-init projection to unit gain regardless of lr.
//!
//! STATUS: **dense forward + SDPA-segment checkpointing**. Whole-branch gradient checkpointing (the
//! recompute of the branch/main block activations) and the OOM pre-flight guard are the memory-hardening
//! follow-up (the Candle lane needed the segmented VJP ≥ 512² on a 96 GB card); on a 128 GB Mac the
//! first real runs train at a bounded resolution until that lands.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use mlx_gen::gen_core;
use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{accumulate_grads, average_grads, LoraParams};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::weights::Weights;
use mlx_gen::{
    Error, LoadSpec, Modality, Precision, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{add, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::Krea2Config;
use crate::control::{
    Krea2ControlBranch, DEFAULT_INJECT_OFFSET, DEFAULT_N_CONTROL_BLOCKS, META_INJECT_OFFSET,
};
use crate::loader::{load_text_encoder, load_transformer};
use crate::text_encoder::{KreaTextEncoder, KreaTokenizer};
use crate::training::{build_batch, decode_image, encode_caption, encode_latents, sample_sigma};
use crate::transformer::Krea2Transformer;
use crate::vae::{load_vae, QwenVae};

/// Registry id for the Krea control-branch trainer — the SAME id as the Candle trainer (sc-10163), so
/// a plan routes to whichever backend is present. The produced overlay records
/// `baseModel: krea_2_turbo` (trained and deployed on Turbo).
pub const KREA_2_CONTROL_ID: &str = "krea_2_control";

/// The frozen base the control overlay is trained on AND applied at (Turbo — CFG-free, distilled,
/// 8-step). Recorded in the overlay meta. There is deliberately no raw→turbo crossing (§ module doc).
const OVERLAY_BASE_MODEL: &str = "krea_2_turbo";

/// The Qwen3-VL-4B condition encoder loads Q8 for the trainer (frozen; caches caption features once,
/// then dropped before the train loop) — mirrors [`crate::training`].
const TRAINER_ENCODER_BITS: i32 = 8;

/// lr multiplier for the zero-init injection-projection optimizer group (Candle run-3 fix): the
/// projections step at `PROJ_LR_MULT × lr` so their gain grows slowly, not at AdamW's normalized rate.
const PROJ_LR_MULT: f32 = 0.1;
/// Decoupled weight decay for the injection-projection group (Candle run-3 fix) — structural magnitude
/// control on the injection gain, which a bare lr reduction cannot provide under AdamW.
const PROJ_WEIGHT_DECAY: f32 = 0.05;

/// Which optimizer group an overlay key belongs to: the zero-init injection projections
/// (`blocks.{i}.proj_out.weight`) get the reduced-lr + weight-decay group, everything else (the block
/// bodies) the full-lr group. The single predicate both the train loop and its test read.
fn is_proj_key(key: &str) -> bool {
    key.contains("proj_out")
}

/// Capability-free floor for a control-training request (rich `Result`, so it is unit-testable without
/// a loaded trainer): control_type must be set, the dataset non-empty, steps > 0, and the optimizer
/// supported. The shared per-item control-image + unsupported-capability checks live in
/// [`gen_core::train::validate_control_request`], called alongside this by [`Trainer::validate`].
fn validate_request(req: &TrainingRequest) -> Result<()> {
    let cfg = &req.config;
    if cfg.control_type.is_none() {
        return Err("krea control trainer: control_type must be set (e.g. \"depth\")".into());
    }
    if req.items.is_empty() {
        return Err("krea control trainer: dataset is empty".into());
    }
    if cfg.steps == 0 {
        return Err("krea control trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(format!(
            "krea control trainer: optimizer '{}' is not available on MLX training (supported: \
             adamw, adam, rose, prodigy)",
            cfg.optimizer
        )
        .into());
    }
    Ok(())
}

/// The MLX Krea control-branch trainer. Holds the frozen base components; the trainable branch is
/// built from the base weights in `train()` (the reconstruct-from-map init).
pub struct KreaControlTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: KreaTokenizer,
    /// Qwen3-VL-4B encoder, `Option` so it can be dropped after caption caching (idle in the loop).
    encoder: Option<KreaTextEncoder>,
    transformer: Krea2Transformer,
    vae: QwenVae,
    /// The DiT config — needed to reconstruct the [`Krea2ControlBranch`] from the trainable map each
    /// step (block dims) and to init the branch from the base's first-N blocks (`from_base`).
    config: Krea2Config,
    /// The snapshot root the base loaded from — re-opened in `train()` to copy the first-N block
    /// weights into the trainable branch (the `from_base` init), mirroring the Candle trainer which
    /// reads the base `transformer/` `Weights` directly.
    root: PathBuf,
    /// Compute dtype (bf16 production / f32 tight-gate), fixed at load from `spec.precision`.
    dtype: Dtype,
}

fn control_trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: KREA_2_CONTROL_ID,
        family: "krea_2",
        backend: "mlx",
        modality: Modality::Image,
        // A control-branch trainer, not a LoRA/LoKr one.
        supports_lora: false,
        supports_lokr: false,
        supports_control: true,
    }
}

/// Load the control trainer from a `krea/Krea-2-Turbo` snapshot dir (the frozen base the branch copies
/// its blocks from). Mirrors [`load_trainer`](crate::training::load_trainer); the trainable branch is
/// built from the base weights inside `train()`.
pub fn load_control_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(KreaControlTrainer::load(spec)?))
}

impl KreaControlTrainer {
    /// Concrete loader (the [`load_control_trainer`] registration boxes this). Returns the concrete
    /// type so weight-gated tests can exercise `from_base` / the overlay round-trip directly.
    pub(crate) fn load(spec: &LoadSpec) -> Result<Self> {
        let root =
            match &spec.weights {
                WeightsSource::Dir(p) => p.clone(),
                WeightsSource::File(_) => return Err(Error::Msg(
                    "krea control trainer expects a Krea-2-Turbo snapshot directory (tokenizer/ \
                     text_encoder/ transformer/ vae/), not a single .safetensors file"
                        .into(),
                )),
            };
        let dtype = match spec.precision {
            Precision::Bf16 => Dtype::Bfloat16,
            Precision::Fp32 => Dtype::Float32,
        };
        let config = Krea2Config::from_snapshot(&root)?;
        let tokenizer = KreaTokenizer::from_snapshot(&root)?;
        let mut encoder = load_text_encoder(&root)?;
        encoder.quantize(TRAINER_ENCODER_BITS)?;
        let mut transformer = load_transformer(&root)?;
        if transformer.compute_dtype() != dtype {
            transformer.cast_weights(dtype)?;
        }
        let vae = load_vae(&root)?;
        Ok(KreaControlTrainer {
            descriptor: control_trainer_descriptor(),
            tokenizer,
            encoder: Some(encoder),
            transformer,
            vae,
            config,
            root,
            dtype,
        })
    }
}

// Bridges the crate's rich `Result` into backend-neutral `gen_core::Result`.
mlx_gen::register_trainer! {
    pub(crate) const CONTROL_TRAINER_REGISTRATION = control_trainer_descriptor => load_control_trainer
}

impl Trainer for KreaControlTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared control floor (F-006): control_type set on a control-capable trainer ⇒ every item
        // must carry a control image (else typed reject). A control run MUST name its control type —
        // the shared floor treats an UNSET control_type as the LoRA no-op, so `validate_request`
        // requires it (plus dataset/steps/optimizer) explicitly.
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        validate_request(req).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl KreaControlTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`], keeping `?` on `mlx_rs`/family helpers transparent here.
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let cfg = &req.config;
        // Re-assert the floor in rich-`Result` form (validate() already ran in the worker path, but
        // train_impl is self-sufficient): control_type, dataset, steps, optimizer.
        validate_request(req)?;
        let control_type = cfg
            .control_type
            .clone()
            .expect("validate_request guarantees control_type is set");

        let compute_dtype = self.dtype;
        let n_blocks = DEFAULT_N_CONTROL_BLOCKS;
        let inject_offset = DEFAULT_INJECT_OFFSET;

        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);

        // --- initialise the trainable branch param map from the frozen base's first-N blocks (bodies
        // copied, RMSNorm scales folded to `weight_p1`, `proj_out` zero-init). Keys == the overlay
        // on-disk format, so the saved map loads unchanged through inference. ---
        let mut params = self.init_branch_from_base(n_blocks, compute_dtype)?;
        eval(params.values())?;

        // --- prepare → load → cache: VAE-latents (target + control) + caption features into memory ---
        on_progress(TrainingProgress::LoadingModel); // base already resident from load_control_trainer
        let total = req.items.len() as u32;
        // Per item: (target latent x0, control tokens embedded through the frozen img_in, caption ctx).
        let mut cache: Vec<(Array, Array, Array)> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let ctrl_path = item.control_image_path.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "krea control trainer: item {i} has no control image (validate should have \
                     rejected this)"
                ))
            })?;
            let target_raw = decode_image(&item.image_path)?;
            let ctrl_raw = decode_image(ctrl_path)?;
            // Fail LOUDLY on a size mismatch: the target and its control map are center-cropped
            // INDEPENDENTLY, so a shared square crop only keeps them registered when both start at the
            // same dimensions. Differing sizes would silently train on misaligned pairs (a quietly-bad
            // adapter), so refuse up front — the control map must be emitted at the target's dimensions.
            if (target_raw.width, target_raw.height) != (ctrl_raw.width, ctrl_raw.height) {
                return Err(Error::Msg(format!(
                    "krea control trainer: item {i} target image ({}×{}) and control image ({}×{}) \
                     differ in size — they must be pixel-aligned (identical dimensions) so the shared \
                     center-crop keeps the control map registered to the target",
                    target_raw.width, target_raw.height, ctrl_raw.width, ctrl_raw.height
                )));
            }
            let img = center_crop_square(&target_raw);
            let x0 = encode_latents(&self.vae, &img, edge)?; // [1, 16, edge/8, edge/8]

            // The per-item control conditioning: VAE-encode the aligned control map (depth/pose/…) into
            // the SAME normalized latent space, then patch-embed it through the FROZEN base `img_in`
            // (step-invariant) — exactly what the inference branch adds onto its image tokens.
            let ctrl_img = center_crop_square(&ctrl_raw);
            let ctrl_latent = encode_latents(&self.vae, &ctrl_img, edge)?;
            let ctrl_tokens = self.transformer.embed_latent(&ctrl_latent)?; // [1, img_len, hidden]

            let encoder = self.encoder.as_ref().ok_or_else(|| {
                Error::Msg(
                    "krea control trainer: text encoder already freed (caching after train loop)"
                        .into(),
                )
            })?;
            let context = encode_caption(&self.tokenizer, encoder, &item.caption)?;
            eval([&x0, &ctrl_tokens, &context])?;
            cache.push((x0, ctrl_tokens, context));
        }
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            return Err("krea control trainer: no usable dataset items".into());
        }

        // Every caption is cached now — free the 4 B-param encoder and evict its buffers before the
        // train loop, reclaiming that resident for the DiT + branch working set.
        self.encoder = None;
        mlx_rs::memory::clear_cache();

        // SDPA-segment gradient checkpointing on the frozen main stack (always-on in training): its
        // backward recomputes attention rather than retaining the seq² probability matrix. The
        // reconstructed branch blocks get the same flag inside the traced loss (see `control_loss_grads`).
        self.transformer.set_sdpa_checkpoint(true);

        // --- two-group AdamW (Candle run-3 fix): bodies at full lr / no decay, zero-init projections
        // at PROJ_LR_MULT×lr WITH decoupled weight decay. Each optimizer steps only its own keys. ---
        let mut opt_body = TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, 0.0)?;
        let mut opt_proj = TrainOptimizer::from_config(
            &cfg.optimizer,
            cfg.learning_rate * PROJ_LR_MULT,
            PROJ_WEIGHT_DECAY,
        )?;

        let accum = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("control")
            .to_string();

        // --- train loop ---
        let mut accumulated: Option<LoraParams> = None;
        let mut update_idx: u32 = 0;
        let mut last_loss = 0.0f32;
        let mut steps_run: u32 = 0;
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, ctrl_tokens, context) = &cache[((step - 1) as usize) % cache.len()];
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
            let (loss, grads) = self.control_loss_grads(
                &params,
                x0,
                ctrl_tokens,
                context,
                t,
                &noise,
                compute_dtype,
            )?;
            last_loss = loss;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                // The final update can fire with fewer than `accum` grads when `steps` isn't a multiple
                // of the accumulation; divide by the actual in-window count.
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
                // Split the single clipped grad map by key into the two optimizer groups: `proj_out`
                // projections vs the block bodies. Each optimizer only ever sees (and steps) its keys.
                let proj_grads: LoraParams = clipped
                    .iter()
                    .filter(|(k, _)| is_proj_key(k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                let body_grads: LoraParams = clipped
                    .iter()
                    .filter(|(k, _)| !is_proj_key(k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                opt_body.set_lr_scaled(mult);
                opt_proj.set_lr_scaled(mult);
                opt_body.step(&mut params, &body_grads)?;
                opt_proj.step(&mut params, &proj_grads)?;
                eval(params.values())?;
                update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            // Intermediate overlay checkpoints (no optimizer-state resume — the two-group optimizer has
            // no single-snapshot form, matching the Candle lane's checkpoint+final-only policy).
            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                save_overlay(&params, inject_offset, compute_dtype, &ckpt)?;
                write_meta_sidecar(&ckpt, &control_type, n_blocks, edge)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // Cancelled before a single step completed: the branch is still the zero-init identity. Surface
        // the cancellation rather than writing a valid-looking no-op overlay.
        if steps_run == 0 {
            return Err(Error::Canceled);
        }

        // --- save the final overlay (the flat `blocks.{i}.*` + `meta.inject_offset` safetensors the
        // inference `Krea2ControlBranch::from_weights` loads unchanged) + its `.json` meta sidecar. ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        eval(params.values())?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_overlay(&params, inject_offset, compute_dtype, &adapter_path)?;
        write_meta_sidecar(&adapter_path, &control_type, n_blocks, edge)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }

    /// Initialise the trainable branch param map from the frozen base's first-`n` single-stream blocks
    /// — the MLX twin of Candle `ControlBranch::from_base`. Copies each block's matmul weights (cast to
    /// the branch dtype), folds its four RMSNorm scales to the overlay `*.weight_p1` (`scale + 1`, f32),
    /// copies the `scale_shift_table`, and zero-inits the `proj_out` output projection (the ControlNet
    /// identity seam). Keys are exactly the overlay on-disk format, so the map reconstructs a
    /// [`Krea2ControlBranch`] verbatim and — once trained — saves back to a checkpoint that loads
    /// unchanged through inference. Re-opens the snapshot `transformer/` `Weights` (mmap; only the `n`
    /// copied blocks are ever materialised).
    ///
    /// The trainable branch is always DENSE (its weights are the differentiation leaves): when the base
    /// snapshot's matmuls are group-wise-quantized (the deployed Turbo turnkey packs the attn/FFN to
    /// Q8/Q4), each copied projection is DEQUANTIZED to dense `dtype` first ([`dense_leaf`]). The
    /// zero-init `proj_out` keeps the branch a step-0 identity regardless, and the trained dense bf16
    /// overlay injects onto the frozen (packed) base exactly as the shipped pose overlay does.
    fn init_branch_from_base(&self, n: usize, dtype: Dtype) -> Result<LoraParams> {
        let w = Weights::from_dir(self.root.join("transformer"))?;
        let hidden = self.config.hidden_size as i32;
        let mut params: LoraParams = HashMap::with_capacity(n * 15);
        let mut put = |key: String, a: Array| {
            params.insert(Rc::from(key.as_str()), a);
        };
        for i in 0..n {
            let base = format!("transformer_blocks.{i}");
            let blk = format!("blocks.{i}");

            // scale_shift_table — flattened to the shipped overlay's on-disk `[1, 1, 6·hidden]`. The base
            // stores `[6, hidden]`; both round-trip through the branch loader's element-count reshape, but
            // emitting the candle shape makes an MLX- vs candle-trained overlay a byte-for-byte structural
            // twin (all 99 tensors identical in key/dtype/shape). Shape-agnostic to AdamW; a no-op through
            // `from_weights`. Verified the candle `[1,1,36864]` and this both load+run by the parity test.
            put(
                format!("{blk}.scale_shift_table"),
                w.require(&format!("{base}.scale_shift_table"))?
                    .as_dtype(dtype)?
                    .reshape(&[1, 1, 6 * hidden])?,
            );

            // RMSNorm scales → pre-folded `weight_p1` (`scale + 1`), f32 — the overlay convention the
            // inference `RmsScale::from_weights` prefers verbatim.
            for (src, dst) in [
                ("norm1.weight", "norm1.weight_p1"),
                ("norm2.weight", "norm2.weight_p1"),
                ("attn.norm_q.weight", "attn.norm_q.weight_p1"),
                ("attn.norm_k.weight", "attn.norm_k.weight_p1"),
            ] {
                let scale = w
                    .require(&format!("{base}.{src}"))?
                    .as_dtype(Dtype::Float32)?;
                put(format!("{blk}.{dst}"), add(&scale, Array::from_f32(1.0))?);
            }

            // Body matmul weights — DENSE (dequantized from the base's packed Q8/Q4 codes when the
            // snapshot is a quantized turnkey, else copied straight), cast to the branch dtype.
            for stem in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_gate",
                "attn.to_out.0",
                "ff.gate",
                "ff.up",
                "ff.down",
            ] {
                put(
                    format!("{blk}.{stem}.weight"),
                    dense_leaf(&w, &format!("{base}.{stem}"), dtype)?,
                );
            }

            // Zero-init `[hidden, hidden]` output projection — the ControlNet identity seam (step 0 is
            // an exact base forward until the projection grows).
            put(
                format!("{blk}.proj_out.weight"),
                Array::zeros::<f32>(&[hidden, hidden])?.as_dtype(dtype)?,
            );
        }
        Ok(params)
    }

    /// One forward+backward over the trainable branch map: reconstruct a [`Krea2ControlBranch`] from
    /// `params` inside the traced loss, run the control-branched velocity prediction (residual scale
    /// **1.0** — `control_scale` is an inference knob), regress the raw velocity onto `noise − x0`, and
    /// return `(loss, grads)` keyed identically to `params`. The DiT timestep is `t` (the noise
    /// fraction) directly.
    ///
    /// The step-invariant prep (text fusion + RoPE) is built from the FROZEN base **outside** the
    /// traced closure (its inputs carry no trainable params), so text_fusion runs once per step rather
    /// than being retained in the branch's backward.
    #[allow(clippy::too_many_arguments)]
    fn control_loss_grads(
        &self,
        params: &LoraParams,
        x0: &Array,
        ctrl_tokens: &Array,
        context: &Array,
        t: f32,
        noise: &Array,
        dtype: Dtype,
    ) -> Result<(f32, LoraParams)> {
        let (x_t, target) = build_batch(x0, noise, t)?;
        let x_t = x_t.as_dtype(dtype)?; // no-op in f32 mode
        let timestep = Array::from_slice(&[t], &[1]);
        // Frozen step-invariant conditioning (no trainable inputs → a detached constant in the trace).
        let prep = self.transformer.prepare(context, None, &x_t)?;
        let ctrl_tokens = ctrl_tokens.clone();
        let dit = &self.transformer;
        let cfg = &self.config;

        let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
            // Reconstruct the branch from the traced map (keys == overlay format), then run its OWN
            // inference forward — grads flow back through `from_weights` → the block matmuls into `p`.
            let mut w = Weights::empty();
            for (k, v) in p.iter() {
                w.insert(k.to_string(), v.clone());
            }
            let mut branch = Krea2ControlBranch::from_weights(&w, cfg)
                .map_err(|e| Exception::custom(e.to_string()))?;
            branch.set_sdpa_checkpoint(true);
            let v = branch
                .forward(dit, &x_t, &timestep, &prep, &ctrl_tokens, 1.0)
                .map_err(|e| Exception::custom(e.to_string()))?;
            let diff = subtract(&v, &target)?;
            let loss = diff.square()?.mean(None)?; // MSE on the velocity (grad wants a 0-d scalar)
            Ok(vec![loss])
        };
        let mut vg = keyed_value_and_grad(loss_fn);
        let (val, grads) = vg(params.clone(), 0)?;
        Ok((val[0].item::<f32>(), grads))
    }
}

/// Read a base matmul projection at `key_base` (e.g. `transformer_blocks.0.attn.to_q`) as a DENSE
/// `[out, in]` weight in `dtype`: when the base snapshot packed it group-wise (a Q8/Q4 turnkey — the
/// `{key_base}.scales`/`.biases` are present), dequantize it back to dense at the Krea
/// [`crate::quant::GROUP_SIZE`]; otherwise return the dense `.weight` directly. The trainable branch
/// must be dense (its weights are the autograd leaves), and a dense bf16 overlay is exactly what the
/// inference branch loads.
fn dense_leaf(w: &Weights, key_base: &str, dtype: Dtype) -> Result<Array> {
    let wq = w.require(&format!("{key_base}.weight"))?;
    match w.get(&format!("{key_base}.scales")) {
        Some(scales) => {
            let biases = w.require(&format!("{key_base}.biases"))?;
            let group_size = crate::quant::GROUP_SIZE;
            let bits = mlx_gen::quant::packed_bits(wq, scales, group_size)?;
            let dense = mlx_rs::ops::dequantize(wq, scales, biases, group_size, bits)?;
            Ok(dense.as_dtype(dtype)?)
        }
        None => Ok(wq.as_dtype(dtype)?),
    }
}

/// Write the trained branch map to a flat `.safetensors` in the overlay on-disk format — every
/// `blocks.{i}.*` param plus the `meta.inject_offset` tensor (written HERE, outside any trace, so its
/// value is a plain host-side write and never a mid-graph `.item()`). Reloads through the unmodified
/// inference [`Krea2ControlBranch::from_weights`] — the round-trip AC.
///
/// Each param is normalized to its CANONICAL overlay dtype so a TRAINED overlay matches the shipped
/// on-disk format byte-for-byte: RMSNorm scales (`*.weight_p1`) stay **f32**; every body matmul, the
/// zero-init `proj_out`, and `scale_shift_table` are `dtype` (**bf16** in production) — the shipped
/// candle overlay's 70×BF16 / 29×F32 split. AdamW keeps **f32 master weights** during training
/// (standard mixed precision — the higher-quality path; bf16 master weights would underflow updates at
/// production lr), so the resident map is f32 after the first step; casting HERE deploys the bf16 the
/// inference branch loads (halving the file), with zero effect on the f32 training precision.
fn save_overlay(
    params: &LoraParams,
    inject_offset: usize,
    dtype: Dtype,
    path: &Path,
) -> Result<()> {
    let mut owned: Vec<(String, Array)> = params
        .iter()
        .map(|(k, v)| {
            let target = if k.ends_with("weight_p1") {
                Dtype::Float32
            } else {
                dtype
            };
            Ok((k.to_string(), v.as_dtype(target)?))
        })
        .collect::<Result<_>>()?;
    // `meta.inject_offset` is a plain host-side f32 scalar written outside any trace.
    owned.push((
        META_INJECT_OFFSET.to_string(),
        Array::from_slice(&[inject_offset as f32], &[1]),
    ));
    // Deterministic key order (not required by safetensors, but keeps the file byte-stable run-to-run).
    owned.sort_by(|a, b| a.0.cmp(&b.0));
    let refs: Vec<&Array> = owned.iter().map(|(_, v)| v).collect();
    eval(refs)?;
    let entries: Vec<(String, &Array)> = owned.iter().map(|(k, v)| (k.clone(), v)).collect();
    Array::save_safetensors(entries, None::<&HashMap<String, String>>, path)?;
    Ok(())
}

/// Write the overlay's `.json` meta sidecar — the fields the SceneWorks worker reads to register the
/// trained control model (identical shape to the Candle trainer's, so a candle- vs mlx-trained overlay
/// is indistinguishable to registration): block count, base model, family, the control-type `kind`, and
/// the encode resolution.
fn write_meta_sidecar(
    path: &Path,
    control_type: &str,
    n_blocks: usize,
    resolution: u32,
) -> Result<()> {
    let meta = serde_json::json!({
        "n_blocks": n_blocks,
        "baseModel": OVERLAY_BASE_MODEL,
        "family": "krea_2",
        "kind": format!("{control_type}_control_branch"),
        "resolution": resolution,
    });
    std::fs::write(path.with_extension("json"), meta.to_string())
        .map_err(|e| Error::Msg(format!("krea control trainer: write overlay meta: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{CancelFlag, TrainingConfig, TrainingItem, TrainingRequest};
    use std::path::PathBuf;

    fn control_config() -> TrainingConfig {
        TrainingConfig {
            steps: 10,
            control_type: Some("depth".into()),
            ..Default::default()
        }
    }

    fn req_with(config: TrainingConfig, items: Vec<TrainingItem>) -> TrainingRequest {
        TrainingRequest {
            items,
            config,
            output_dir: PathBuf::from("/tmp/krea_control_unused"),
            file_name: "control.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    /// A one-item control dataset (image + aligned control map).
    fn ctrl_items() -> Vec<TrainingItem> {
        vec![TrainingItem::with_control(
            PathBuf::from("/tmp/x.png"),
            "a street at dusk".into(),
            PathBuf::from("/tmp/x.depth.png"),
        )]
    }

    #[test]
    fn descriptor_is_the_control_id() {
        let d = control_trainer_descriptor();
        assert_eq!(d.id, "krea_2_control");
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // A control-branch trainer: control true, LoRA/LoKr false (it trains full block weights).
        assert!(d.supports_control);
        assert!(!d.supports_lora && !d.supports_lokr);
    }

    #[test]
    fn reachable_via_trainer_registry_by_id() {
        assert!(
            crate::provider_registry()
                .unwrap()
                .trainers()
                .copied()
                .any(|r| (r.descriptor)().id == KREA_2_CONTROL_ID),
            "trainer id {KREA_2_CONTROL_ID} not registered"
        );
    }

    #[test]
    fn validate_request_floor() {
        // Well-formed control request passes.
        assert!(validate_request(&req_with(control_config(), ctrl_items())).is_ok());

        // Missing control_type (the LoRA no-op the shared floor would allow) is rejected here.
        let no_type = TrainingConfig {
            control_type: None,
            ..control_config()
        };
        assert!(validate_request(&req_with(no_type, ctrl_items()))
            .unwrap_err()
            .to_string()
            .contains("control_type"));

        // Empty dataset.
        assert!(validate_request(&req_with(control_config(), vec![]))
            .unwrap_err()
            .to_string()
            .contains("dataset is empty"));

        // Zero steps.
        let zero_steps = TrainingConfig {
            steps: 0,
            ..control_config()
        };
        assert!(validate_request(&req_with(zero_steps, ctrl_items()))
            .unwrap_err()
            .to_string()
            .contains("steps"));

        // Unsupported optimizer.
        let bad_opt = TrainingConfig {
            optimizer: "nope".into(),
            ..control_config()
        };
        assert!(validate_request(&req_with(bad_opt, ctrl_items())).is_err());
    }

    /// The shared F-006 floor, exercised against THIS (control-capable) descriptor: a control_type set
    /// with an item that lacks a control image is rejected; every item carrying one passes; and an
    /// unset control_type is a no-op (LoRA path).
    #[test]
    fn shared_control_floor_requires_per_item_control_image() {
        let desc = control_trainer_descriptor();

        // Item missing its control image → typed Msg (not Unsupported — this trainer DOES support
        // control; the gap is the missing per-item image).
        let missing = vec![TrainingItem::captioned(
            PathBuf::from("/tmp/x.png"),
            "a street".into(),
        )];
        let err =
            gen_core::train::validate_control_request(&desc, &req_with(control_config(), missing))
                .unwrap_err();
        assert!(
            err.to_string().contains("control image"),
            "expected a per-item control-image rejection, got {err:?}"
        );

        // Every item carrying a control image → ok.
        assert!(gen_core::train::validate_control_request(
            &desc,
            &req_with(control_config(), ctrl_items())
        )
        .is_ok());

        // control_type unset → the shared floor is a no-op (our own validate_request rejects it, but
        // the shared floor does not — that separation is the point).
        let no_type = TrainingConfig {
            control_type: None,
            ..control_config()
        };
        assert!(
            gen_core::train::validate_control_request(&desc, &req_with(no_type, ctrl_items()))
                .is_ok()
        );
    }

    /// The two-group partition: only `blocks.{i}.proj_out.weight` is the reduced-lr + weight-decay
    /// group; every block-body leaf is the full-lr group. Over the full from_base keyset the two are
    /// disjoint and cover everything.
    #[test]
    fn is_proj_key_partitions_overlay_keys() {
        let proj = ["blocks.0.proj_out.weight", "blocks.6.proj_out.weight"];
        let body = [
            "blocks.0.attn.to_q.weight",
            "blocks.0.attn.to_out.0.weight",
            "blocks.0.attn.norm_q.weight_p1",
            "blocks.0.norm1.weight_p1",
            "blocks.0.scale_shift_table",
            "blocks.0.ff.down.weight",
        ];
        assert!(proj.iter().all(|k| is_proj_key(k)));
        assert!(body.iter().all(|k| !is_proj_key(k)));
    }

    // ===========================================================================================
    // sc-10177 — real-weight structural harness (weight-gated, run as its own process):
    //   KREA_TURBO_DIR=/path/to/Krea-2-Turbo \
    //     cargo test -p mlx-gen-krea --release --lib from_base -- --ignored --nocapture
    // ===========================================================================================
    fn snapshot() -> Option<PathBuf> {
        std::env::var("KREA_TURBO_DIR").ok().map(PathBuf::from)
    }

    /// `from_base` produces the overlay on-disk format — zero-init `proj_out`, RMSNorm scales folded to
    /// `*.weight_p1` (the raw `*.weight` absent), the eight body matmuls + `scale_shift_table` present —
    /// and the resulting map reconstructs a `Krea2ControlBranch` verbatim (the round-trip AC: a trained
    /// overlay loads unchanged through the inference loader).
    #[test]
    #[ignore = "needs real krea/Krea-2-Turbo weights; run as its own process"]
    fn from_base_produces_overlay_format_that_roundtrips() {
        use mlx_gen::WeightsSource;

        let root = snapshot().expect("set KREA_TURBO_DIR to the krea/Krea-2-Turbo snapshot root");
        let spec = LoadSpec::new(WeightsSource::Dir(root));
        let trainer = KreaControlTrainer::load(&spec).unwrap();

        let n = DEFAULT_N_CONTROL_BLOCKS;
        let params = trainer.init_branch_from_base(n, Dtype::Bfloat16).unwrap();
        eval(params.values()).unwrap();

        for i in 0..n {
            // Zero-init projection.
            let p = params
                .get(format!("blocks.{i}.proj_out.weight").as_str())
                .expect("proj_out key present");
            assert_eq!(
                p.abs().unwrap().max(None).unwrap().item::<f32>(),
                0.0,
                "blocks.{i}.proj_out.weight must be zero-init"
            );
            // Folded norm present; raw scale absent (the overlay `*.weight_p1` convention).
            assert!(params.contains_key(format!("blocks.{i}.norm1.weight_p1").as_str()));
            assert!(!params.contains_key(format!("blocks.{i}.norm1.weight").as_str()));
            // A body matmul + the modulation table. `scale_shift_table` is emitted in the shipped
            // candle overlay's on-disk shape `[1, 1, 6·hidden]` (a byte-for-byte structural twin), NOT
            // the base's `[6, hidden]` — both round-trip, but matching keeps MLX/candle overlays identical.
            assert!(params.contains_key(format!("blocks.{i}.attn.to_q.weight").as_str()));
            let sst = params
                .get(format!("blocks.{i}.scale_shift_table").as_str())
                .expect("scale_shift_table key present");
            assert_eq!(
                sst.shape(),
                &[1, 1, 6 * trainer.config.hidden_size as i32],
                "scale_shift_table must be emitted in the shipped overlay's [1,1,6·hidden] shape"
            );
        }

        // Round-trips through the UNMODIFIED inference loader.
        let mut w = Weights::empty();
        for (k, v) in &params {
            w.insert(k.to_string(), v.clone());
        }
        let branch = Krea2ControlBranch::from_weights(&w, &trainer.config).unwrap();
        assert_eq!(branch.num_blocks(), n);
        assert_eq!(branch.inject_offset(), DEFAULT_INJECT_OFFSET);
    }

    /// The keystone real-weight gate (the MLX twin of Candle `backward_reaches_branch_and_descends`):
    /// with the branch nudged off its zero-init, (a) the gradient of the reconstruct-from-map loss
    /// actually REACHES the branch — the `proj_out` projection AND a block body get finite, nonzero
    /// gradients — and (b) the two-group optimizer DESCENDS a fixed over-fit probe by a wide margin.
    /// This is the proof the weightless tests can't give: that the novel reconstruct-from-map path
    /// differentiates and learns, not just that it wires up. Run POSE first with a real corpus for
    /// end-to-end parity; this tiny synthetic-batch smoke isolates "the trainer descends" cheaply.
    #[test]
    #[ignore = "needs real krea/Krea-2-Turbo weights; run as its own process"]
    fn reconstruct_from_map_grad_flows_and_descends() {
        use mlx_gen::WeightsSource;

        let root = snapshot().expect("set KREA_TURBO_DIR to the krea/Krea-2-Turbo snapshot root");
        let spec = LoadSpec::new(WeightsSource::Dir(root));
        eprintln!("[sc-10177] loading trainer …");
        let trainer = KreaControlTrainer::load(&spec).unwrap();
        let dtype = Dtype::Bfloat16;
        eprintln!("[sc-10177] trainer loaded; init branch from base (dequant) …");

        let mut params = trainer
            .init_branch_from_base(DEFAULT_N_CONTROL_BLOCKS, dtype)
            .unwrap();
        // Nudge the zero-init `proj_out` off zero so there is a gradient signal into the bodies too
        // (at exact step 0 `d(residual)/d(body) = proj_outᵀ = 0` — correct ControlNet dynamics).
        for (k, v) in params.iter_mut() {
            if is_proj_key(k) {
                let noise =
                    random::normal::<f32>(v.shape(), None, None, Some(&random::key(7).unwrap()))
                        .unwrap()
                        .as_dtype(v.dtype())
                        .unwrap();
                let nudged = add(
                    &*v,
                    mlx_rs::ops::multiply(&noise, Array::from_f32(0.02)).unwrap(),
                )
                .unwrap();
                *v = nudged;
            }
        }
        eval(params.values()).unwrap();

        // A tiny synthetic batch: a 64² image → [1,16,8,8] latent (16 image tokens), a random control
        // latent embedded through the frozen `img_in`, and a real caption encode. A FIXED noise makes
        // the descent an over-fit of one point — the cleanest "does the optimizer lower the loss it is
        // given" signal (Candle's fixed-batch discipline).
        let x0 = random::normal::<f32>(&[1, 16, 8, 8], None, None, Some(&random::key(1).unwrap()))
            .unwrap();
        let ctrl_latent =
            random::normal::<f32>(&[1, 16, 8, 8], None, None, Some(&random::key(2).unwrap()))
                .unwrap();
        let ctrl_tokens = trainer.transformer.embed_latent(&ctrl_latent).unwrap();
        let context = encode_caption(
            &trainer.tokenizer,
            trainer.encoder.as_ref().expect("encoder resident"),
            "a quiet street at dusk",
        )
        .unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(3).unwrap())).unwrap();

        // (a) grad reaches the branch — proj_out finite+nonzero, a body finite.
        let (loss0, grads) = trainer
            .control_loss_grads(&params, &x0, &ctrl_tokens, &context, 0.5, &noise, dtype)
            .unwrap();
        assert!(loss0.is_finite(), "loss must be finite, got {loss0}");
        let proj_g = grads
            .get("blocks.0.proj_out.weight")
            .expect("proj_out gradient present");
        let proj_absmax = proj_g.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(
            proj_absmax.is_finite() && proj_absmax > 0.0,
            "proj_out gradient must be finite+nonzero — this proves the reconstruct-from-map path \
             differentiates back to the param map; got {proj_absmax}"
        );
        let body_g = grads
            .get("blocks.0.attn.to_q.weight")
            .expect("body gradient present");
        assert!(body_g
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>()
            .is_finite());
        eprintln!(
            "[sc-10177] grad-flow OK: loss0={loss0}, proj_out grad |max|={proj_absmax} (reconstruct-\
             from-map differentiates); starting descent …"
        );

        // (b) two-group AdamW descends the fixed over-fit probe. A DETERMINISTIC single-point overfit
        // (fixed x0/ctrl/context/noise/sigma) → gradient descent must LOWER the loss monotonically; we
        // assert a strict decrease and print the ratio. (A specific-percent bar like Candle's tiny-DiT
        // test isn't guaranteed here: the residual is RMS-clamped and injects through 27 frozen 12B
        // blocks toward a RANDOM target, so the achievable drop is bounded — the sound gate is "does
        // the optimizer move the loss down at all", which proves loss + grads + optimizer are wired.)
        let mut opt_body = TrainOptimizer::from_config("adamw", 3e-3, 0.0).unwrap();
        let mut opt_proj =
            TrainOptimizer::from_config("adamw", 3e-3 * PROJ_LR_MULT, PROJ_WEIGHT_DECAY).unwrap();
        let before = loss0;
        for step in 0..6 {
            let (l, grads) = trainer
                .control_loss_grads(&params, &x0, &ctrl_tokens, &context, 0.5, &noise, dtype)
                .unwrap();
            eprintln!("[sc-10177] descent step {step}: loss={l}");
            let (clipped, _norm) = clip_grad_norm(&grads, 1.0).unwrap();
            let clipped: LoraParams = clipped
                .into_iter()
                .map(|(k, v)| (k, v.into_owned()))
                .collect();
            let proj_grads: LoraParams = clipped
                .iter()
                .filter(|(k, _)| is_proj_key(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let body_grads: LoraParams = clipped
                .iter()
                .filter(|(k, _)| !is_proj_key(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            opt_body.step(&mut params, &body_grads).unwrap();
            opt_proj.step(&mut params, &proj_grads).unwrap();
            eval(params.values()).unwrap();
        }
        // The true post-training loss on the same fixed probe (after all 12 updates).
        let after = trainer
            .control_loss_grads(&params, &x0, &ctrl_tokens, &context, 0.5, &noise, dtype)
            .unwrap()
            .0;
        eprintln!(
            "[sc-10177] fixed-probe loss {before} -> {after} (ratio {})",
            after / before
        );
        assert!(
            after < before,
            "two-group AdamW must lower the fixed-probe overfit loss (deterministic descent): \
             {before} -> {after}"
        );
    }

    /// The **parity foundation** (sc-10177 AC "prove on pose"): the *shipped candle pose overlay*
    /// (`~/Models/aether/krea2-pose-control/control_step5000.safetensors`) loads through the UNMODIFIED
    /// MLX inference loader (`from_source`, the exact production path) AND runs a real branched forward
    /// to finite output — with the trained branch measurably steering the base (`control_scale = 0.6`
    /// differs from the bit-exact `control_scale = 0` base). This is the empirical proof that MLX
    /// consumes the candle-trained overlay end-to-end — the `[1,1,36864]` `scale_shift_table` reshape,
    /// the `*.weight_p1` RmsScale path, and the `meta.inject_offset` tensor all — so an MLX-trained
    /// overlay in the SAME on-disk format (post-reshape, a structural twin) is interchangeable.
    ///
    ///   KREA_TURBO_DIR=~/Models/aether/krea2-turbo \
    ///   KREA_POSE_OVERLAY=~/Models/aether/krea2-pose-control/control_step5000.safetensors \
    ///     cargo test -p mlx-gen-krea --release --lib shipped_candle_pose_overlay -- --ignored --nocapture
    #[test]
    #[ignore = "needs real krea/Krea-2-Turbo weights + the shipped candle pose overlay; own process"]
    fn shipped_candle_pose_overlay_loads_and_runs_through_mlx_inference() {
        use mlx_gen::WeightsSource;

        let root = snapshot().expect("set KREA_TURBO_DIR to the krea/Krea-2-Turbo snapshot root");
        let overlay = PathBuf::from(
            std::env::var("KREA_POSE_OVERLAY")
                .expect("set KREA_POSE_OVERLAY to the shipped candle pose overlay .safetensors"),
        );
        let spec = LoadSpec::new(WeightsSource::Dir(root));
        eprintln!("[sc-10177] loading base trainer (for DiT/tokenizer/encoder) …");
        let trainer = KreaControlTrainer::load(&spec).unwrap();

        // Load the SHIPPED candle overlay through the production inference path (File → Weights →
        // from_weights). num_blocks / inject_offset are READ FROM the overlay (the meta tensor).
        eprintln!(
            "[sc-10177] loading shipped candle pose overlay via from_source (production path) …"
        );
        let branch =
            Krea2ControlBranch::from_source(&WeightsSource::File(overlay), &trainer.config)
                .unwrap();
        assert_eq!(
            branch.num_blocks(),
            DEFAULT_N_CONTROL_BLOCKS,
            "shipped pose overlay must carry N=7 branch blocks"
        );
        assert_eq!(
            branch.inject_offset(),
            DEFAULT_INJECT_OFFSET,
            "shipped pose overlay meta.inject_offset must read back as 1"
        );

        // Synthetic-but-valid inputs, assembled exactly as inference does: a [1,16,8,8] latent (16 image
        // tokens), a control latent embedded through the frozen `img_in`, a real caption encode.
        let dtype = Dtype::Bfloat16;
        let x0 = random::normal::<f32>(&[1, 16, 8, 8], None, None, Some(&random::key(11).unwrap()))
            .unwrap()
            .as_dtype(dtype)
            .unwrap();
        let ctrl_latent =
            random::normal::<f32>(&[1, 16, 8, 8], None, None, Some(&random::key(12).unwrap()))
                .unwrap();
        let ctrl_tokens = trainer.transformer.embed_latent(&ctrl_latent).unwrap();
        let context = encode_caption(
            &trainer.tokenizer,
            trainer.encoder.as_ref().expect("encoder resident"),
            "a person standing in a doorway",
        )
        .unwrap();
        let timestep = Array::from_slice(&[0.5f32], &[1]);
        let prep = trainer.transformer.prepare(&context, None, &x0).unwrap();

        // control_scale = 0 → bit-exact base passthrough (branch skipped); 0.6 → the recommended band
        // runs the full injection. Both must be finite, and the trained branch must MOVE the output.
        eprintln!("[sc-10177] running base forward (control_scale=0) …");
        let base_v = branch
            .forward(
                &trainer.transformer,
                &x0,
                &timestep,
                &prep,
                &ctrl_tokens,
                0.0,
            )
            .unwrap();
        eprintln!("[sc-10177] running branched forward (control_scale=0.6) …");
        let ctrl_v = branch
            .forward(
                &trainer.transformer,
                &x0,
                &timestep,
                &prep,
                &ctrl_tokens,
                0.6,
            )
            .unwrap();

        let base_absmax = base_v.abs().unwrap().max(None).unwrap().item::<f32>();
        let ctrl_absmax = ctrl_v.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(
            base_absmax.is_finite() && ctrl_absmax.is_finite(),
            "both forwards must be finite (base |max|={base_absmax}, control |max|={ctrl_absmax})"
        );
        let effect = subtract(&ctrl_v, &base_v)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        eprintln!(
            "[sc-10177] shipped candle overlay runs through MLX: base |max|={base_absmax}, \
             control |max|={ctrl_absmax}, injection effect |Δ|max={effect}"
        );
        assert!(
            effect > 0.0,
            "the TRAINED candle branch must steer the base output (control_scale=0.6 ≠ base); \
             |Δ|max={effect} — a zero effect would mean MLX loaded the overlay but ignored its weights"
        );
    }

    /// The **real-data DEPTH pipeline gate** (epic 10159 depth): drives the full [`train_impl`] over a
    /// real aligned depth corpus (scenewright `scene-corpus` output — target photo + a `.depth.png`
    /// control map at the SAME dims), exercising the entire real-data path the synthetic descent gate
    /// bypasses: `decode_image` on real jpgs, the 900×600→square center-crop, `encode_latents` (VAE),
    /// `embed_latent` on the control map, `encode_caption`, the cache loop, and — critically — the
    /// **alignment guard** (which passes now that the corpus emits control maps at each target's native
    /// dims; it would have REJECTED the pre-fix 720×540 maps). Asserts the distinct, non-flaky claim the
    /// descent gate can't make: the real depth pipeline RUNS to completion, every per-step loss is
    /// finite, and the saved overlay **round-trips through the unmodified inference `from_source`** (a
    /// TRAINED depth overlay, N=7 / inject_offset=1). NOT a descent claim (train_impl cycles a different
    /// item/noise/σ per step — that's the descent gate's job) and NOT a quality claim (empty captions,
    /// 256² — a smoke, not a shippable overlay).
    ///
    ///   KREA_TURBO_DIR=~/Models/aether/krea2-turbo \
    ///   KREA_DEPTH_CORPUS=/path/to/scene-corpus/out \
    ///     cargo test -p mlx-gen-krea --release --lib depth_trainer_runs_real_corpus -- --ignored --nocapture
    #[test]
    #[ignore = "needs real krea/Krea-2-Turbo weights + a scene-corpus depth corpus; own process"]
    fn depth_trainer_runs_real_corpus_pipeline_and_saves_overlay() {
        use mlx_gen::WeightsSource;

        let root = snapshot().expect("set KREA_TURBO_DIR to the krea/Krea-2-Turbo snapshot root");
        let corpus =
            PathBuf::from(std::env::var("KREA_DEPTH_CORPUS").expect(
                "set KREA_DEPTH_CORPUS to a scene-corpus depth-corpus dir (manifest.jsonl)",
            ));

        // Parse the scene-corpus manifest.jsonl → control items. `target` is absolute; `control` is
        // relative to the corpus dir (`train/{id}.depth.png`); caption may be empty (JoyCaption TODO).
        let manifest = std::fs::read_to_string(corpus.join("manifest.jsonl"))
            .expect("read manifest.jsonl from KREA_DEPTH_CORPUS");
        let mut items = Vec::new();
        for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
            let row: serde_json::Value = serde_json::from_str(line).expect("manifest row is JSON");
            let target = PathBuf::from(row["target"].as_str().expect("row.target"));
            let control = corpus.join(row["control"].as_str().expect("row.control"));
            let caption = row["caption"].as_str().unwrap_or("").to_string();
            items.push(TrainingItem::with_control(target, caption, control));
        }
        assert!(!items.is_empty(), "corpus manifest has no items");
        eprintln!("[depth] {} corpus items; loading trainer …", items.len());

        // A short, low-res SMOKE: 2 steps (exercises cache-cycling + a save), 256² (tractable per step),
        // real AdamW so the saved overlay differs from init. Descent/quality are explicitly out of scope.
        let config = TrainingConfig {
            steps: 2,
            control_type: Some("depth".into()),
            resolution: 256,
            learning_rate: 1e-4,
            seed: 7,
            ..Default::default()
        };
        let out_dir = corpus.join("overlay_out");
        let req = TrainingRequest {
            items,
            config,
            output_dir: out_dir.clone(),
            file_name: "depth_control.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        };

        let spec = LoadSpec::new(WeightsSource::Dir(root));
        let mut trainer = KreaControlTrainer::load(&spec).unwrap();

        // Collect every per-step loss via the progress callback; assert each is finite (the real
        // forward+backward on real depth pairs produces a usable gradient signal, not a NaN).
        let mut losses: Vec<f32> = Vec::new();
        let mut on_progress = |p: TrainingProgress| {
            if let TrainingProgress::Training { step, loss, .. } = p {
                eprintln!("[depth] step {step}: loss={loss}");
                losses.push(loss);
            } else {
                eprintln!("[depth] progress: {p:?}");
            }
        };
        let out = trainer
            .train(&req, &mut on_progress)
            .expect("train_impl must complete on the real depth corpus (alignment guard passes)");

        assert_eq!(out.steps, 2, "should have run all requested steps");
        assert!(out.final_loss.is_finite(), "final loss must be finite");
        assert!(!losses.is_empty(), "progress must report per-step losses");
        assert!(
            losses.iter().all(|l| l.is_finite()),
            "every per-step loss must be finite, got {losses:?}"
        );

        // The trained depth overlay round-trips through the UNMODIFIED inference loader (from_source) —
        // the same round-trip AC as the pose overlay, now on a depth overlay from the REAL pipeline.
        let branch = Krea2ControlBranch::from_source(
            &WeightsSource::File(out.adapter_path.clone()),
            &trainer.config,
        )
        .expect("trained depth overlay must load through inference from_source");
        assert_eq!(branch.num_blocks(), DEFAULT_N_CONTROL_BLOCKS);
        assert_eq!(branch.inject_offset(), DEFAULT_INJECT_OFFSET);
        eprintln!(
            "[depth] real-corpus pipeline OK: {} steps, final_loss={}, overlay {} round-trips (N={}, \
             inject_offset={})",
            out.steps,
            out.final_loss,
            out.adapter_path.display(),
            branch.num_blocks(),
            branch.inject_offset(),
        );
    }
}
