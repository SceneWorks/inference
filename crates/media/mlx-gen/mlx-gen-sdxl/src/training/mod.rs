//! sc-3045 — LoRA/LoKr **training** on the SDXL U-Net, in pure Rust on mlx-rs. The SDXL realization
//! of the core [`Trainer`] contract (epic 3039), built on the same functional-autograd mechanism the
//! Z-Image trainer proved (sc-3042/3044) and the host-generic factor machinery hoisted to core
//! ([`mlx_gen::train::lora`], sc-3045). Parity target = the SceneWorks torch `SdxlLoraTrainer` /
//! `_SdxlLoraBackend`.
//!
//! The whole prepare→cache→train→save lifecycle is the shared SDXL-family backbone
//! ([`family::train_family`], sc-7781 — also driven by `mlx-gen-kolors`); this module supplies the
//! SDXL deltas via [`SdxlFamilyHooks`] and keeps the registration glue + the memory-fit curve local.
//!
//! **What is SDXL-specific here** (the hook bodies; everything else is the shared backbone):
//!   * **Noise / objective — discrete DDPM in the vendored sigma-space.** SDXL inference runs the
//!     vendored k-diffusion Euler-Ancestral sampler ([`EulerSampler`]): latents are stored
//!     *renormalized*, `scale_model_input` is the identity, and the per-step time `t` is the float
//!     sigma-table index in `[0, 1000]` that the U-Net's sinusoidal embedding consumes. Crucially the
//!     renormalized model input `(x0 + σ·noise)·rsqrt(σ²+1)` is **algebraically identical** to the
//!     diffusers DDPM `noisy = √(ᾱ)·x0 + √(1−ᾱ)·noise` (since `rsqrt(σ²+1) = √(ᾱ)`,
//!     `σ·rsqrt(σ²+1) = √(1−ᾱ)`), and the **epsilon** target is the unit `noise`. So training reuses
//!     the crate's own [`EulerSampler::add_noise_with`] at a sampled integer table-index `t` — making
//!     train/inference consistent **by construction** — and regresses the U-Net's `eps` toward
//!     `noise`. (SDXL-base is epsilon-prediction; the v-prediction the torch reference's
//!     `prediction_type` branch supports is never taken for SDXL-base, and the crate's eps-only
//!     sampler could not consume a v-pred adapter — so eps is the correct and only objective here.)
//!     `t` is sampled **uniform over the integer table indices `[1, 1000]`**, which maps 1:1 onto the
//!     diffusers `randint(0, 1000)` the torch trainer uses (the table is `concat([0], σ_1..σ_1000)`).
//!   * **`added_cond_kwargs`.** The U-Net forward takes the pooled `text_embeds` (CLIP-bigG pooled)
//!     and the 6-element `time_ids`. The crate's inference path hardcodes
//!     `time_ids = [512,512,0,0,512,512]` (the vendored `generate_latents` quirk — it ignores the
//!     real size); training feeds the **same** [`text_time_ids`] so the conditioning the LoRA learns
//!     under matches what inference applies it under. (This deliberately diverges from the torch
//!     trainer's real-resolution time_ids — that would mismatch this engine's inference.)
//!   * **Dual-CLIP conditioning.** `encoder_hidden_states = concat(CLIP-L.hidden[-2], bigG.hidden[-2])`
//!     and pooled `text_embeds = bigG.pooled`, via [`encode_conditioning_windows`]. Single forward
//!     per CLIP window, no CFG (the torch ref encodes with `do_classifier_free_guidance=False`).
//!   * **f32 base.** The U-Net + both text encoders + VAE load at f32 for clean autograd (the
//!     inference path runs fp16; the trained f32 factors merge into the fp16 base at load, casts
//!     handled by the loader). The VAE encodes the f32 init image to the scaled latent `x0`.
//!   * **Adapter surface, matched to inference consumption.** LoRA targets the **complete** UNet
//!     attention surface (down/mid/up `to_q/k/v/to_out.0`) — what `LoraCoverage::Complete`
//!     (`model::load`'s default) merges, and what the torch PEFT suffix-match selects. LoKr targets
//!     the **vendored** surface (down/up attention only): the SDXL LoKr loader keeps `mid_block` out
//!     (sc-2640), so training mid_block LoKr would produce factors no inference path reads. LoRA
//!     saves PEFT keys under `base_model.model.unet.` (what `_SdxlLoraBackend` emits); LoKr saves the
//!     bare `<path>.lokr_*` keys; both reconstruct at **f32** (the SDXL merge dtype).

pub mod family;

use std::path::PathBuf;

use mlx_gen::{
    gen_core, Image, LoadSpec, Modality, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::{random, Array, Dtype};

use crate::config::DiffusionConfig;
use crate::model::MODEL_ID;
use crate::pipeline::{encode_conditioning_windows, render_sample, text_time_ids};
use crate::sampler::EulerSampler;
use crate::text_encoder::ClipTextEncoder;
use crate::tokenizer::ClipBpeTokenizer;
use crate::unet::UNet2DConditionModel;
use crate::vae::Autoencoder;

use family::{train_family, SdxlFamilyHooks, TrainTimestep};

/// The SDXL family deltas behind the shared [`train_family`] backbone (sc-7781): the dual-CLIP
/// encoder pair + tokenizer, and the Euler sampler whose sigma table drives the renormalized DDPM
/// noising. Held in [`SdxlTrainer::hooks`].
struct SdxlHooks {
    tokenizer: ClipBpeTokenizer,
    /// The dual CLIP encoders, in `Option`s so they can be **dropped after the caching loop** (sc-4941,
    /// 32 GB-Mac headroom): they are idle during training (every prompt is already encoded to the
    /// cached conditioning), so freeing them (~3.3 GB at f32) before the train loop leaves more of the
    /// unified-memory budget for the U-Net working set.
    te1: Option<ClipTextEncoder>,
    te2: Option<ClipTextEncoder>,
    /// The SDXL noise schedule (the same sigma table the inference Euler-Ancestral sampler uses);
    /// training reuses its [`EulerSampler::add_noise_with`] for the renormalized DDPM noising.
    sampler: EulerSampler,
}

impl SdxlHooks {
    /// The two CLIP encoders, or a typed error if they were already freed after caching.
    fn encoders(&self) -> Result<(&ClipTextEncoder, &ClipTextEncoder)> {
        match (&self.te1, &self.te2) {
            (Some(a), Some(b)) => Ok((a, b)),
            _ => Err(mlx_gen::Error::Msg(
                "sdxl trainer: text encoders already freed (encode after caching)".into(),
            )),
        }
    }
}

impl SdxlFamilyHooks for SdxlHooks {
    fn label(&self) -> &'static str {
        "sdxl"
    }

    /// Caption → `(conditioning [1, n·77, 2048], pooled [1, 1280])`: tokenize (no negative —
    /// training is CFG-off), run both CLIP encoders, and assemble the SDXL dual-CLIP conditioning +
    /// pooled embed exactly as the inference [`encode_conditioning_windows`] path.
    ///
    /// Long captions are windowed rather than truncated (sc-20528) — the trainer must condition on
    /// the same tokens inference will, or a LoRA trained on a clipped caption is being taught the
    /// wrong association. `n == 1` for every caption inside CLIP's context, which is the
    /// pre-sc-20528 encoding unchanged.
    fn encode_prompt(&self, caption: &str) -> Result<(Array, Array)> {
        let (te1, te2) = self.encoders()?;
        let tokens = self.tokenizer.tokenize_windows(caption, None)?;
        encode_conditioning_windows(te1, te2, &tokens)
    }

    /// Preview-sample CFG batch (`[2, …]` = positive then empty-negative): the tokenizer builds the
    /// `Some("")` negative row — aligned to the positive's window count (sc-20528) — so
    /// `encode_conditioning_windows` produces the `[2, …]` conditioning SDXL's real-CFG preview
    /// denoise needs at one sequence length.
    fn encode_sample_cfg(&self, prompt: &str) -> Result<(Array, Array)> {
        let (te1, te2) = self.encoders()?;
        let tokens = self.tokenizer.tokenize_windows(prompt, Some(""))?;
        encode_conditioning_windows(te1, te2, &tokens)
    }

    fn free_text_encoders(&mut self) {
        self.te1 = None;
        self.te2 = None;
    }

    /// SDXL micro-conditioning `time_ids`, hardcoded `[512,512,0,0,512,512]` (the vendored
    /// `generate_latents` quirk) — ignores the real `edge` so the LoRA trains under the conditioning
    /// inference applies it under.
    fn time_ids(&self, batch: i32, _edge: u32) -> Array {
        text_time_ids(batch)
    }

    /// Sample a **uniform integer** DDPM timestep over the sigma-table indices `[1, max_time]` (the
    /// vendored table is `concat([0], σ_1..σ_1000)`, so index `t` maps to diffusers `ᾱ[t-1]` — a
    /// uniform draw here equals the torch trainer's `randint(0, num_train_timesteps)`). Deterministic
    /// in `seed`. At an integer `t` the sampler's sigma interpolation is exact (`σ = σ_t`).
    fn sample_timestep(&self, seed: u64) -> Result<TrainTimestep> {
        let k = random::key(seed)?;
        let max_t = self.sampler.max_time(); // 1000.0
        let u = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k))?.item::<f32>();
        // floor(1 + u·max_t) ∈ [1, max_t] (u ∈ [0,1)); clamp the u→1 edge defensively.
        let t = (1.0 + u * max_t).floor().clamp(1.0, max_t);
        Ok(TrainTimestep::Sigma(t))
    }

    /// Renormalized model input `(x0 + σ(t)·noise)·rsqrt(σ(t)²+1)` — algebraically the diffusers DDPM
    /// `noisy`. Reusing the sampler's own `add_noise_with` makes the training input bit-consistent with
    /// the inference convention.
    fn add_noise(&self, x0: &Array, noise: &Array, t: TrainTimestep) -> Result<Array> {
        match t {
            TrainTimestep::Sigma(s) => self.sampler.add_noise_with(x0, noise, s),
            TrainTimestep::Index(_) => Err(mlx_gen::Error::Msg(
                "sdxl trainer: expected a sigma-table (float) timestep".into(),
            )),
        }
    }

    fn peak_gb(&self, p: f64, bf16: bool) -> f64 {
        projected_dense_peak_gb(p, bf16)
    }

    /// `ᾱ = 1/(σ(t)²+1)` — [`EulerSampler::add_noise_with`]'s renormalized input
    /// `(x0 + σ·noise)·rsqrt(σ²+1)` is `√ᾱ·x0 + √(1−ᾱ)·noise` with `rsqrt(σ²+1) = √ᾱ`.
    fn alpha_bar(&self, t: TrainTimestep) -> Result<f32> {
        match t {
            TrainTimestep::Sigma(s) => Ok(sdxl_alpha_bar(&self.sampler, s)),
            TrainTimestep::Index(_) => Err(mlx_gen::Error::Msg(
                "sdxl trainer: expected a sigma-table (float) timestep".into(),
            )),
        }
    }

    /// The table index over `max_time` (index 1000 = σ_max, pure noise).
    fn noise_level(&self, t: TrainTimestep) -> f32 {
        t.unet_time() / self.sampler.max_time()
    }

    /// `round(level · max_time)` clamped onto the trained integer indices `[1, max_time]`.
    fn timestep_at(&self, level: f32) -> TrainTimestep {
        let max_t = self.sampler.max_time();
        TrainTimestep::Sigma((level * max_t).round().clamp(1.0, max_t))
    }

    fn render_sample(
        &self,
        unet: &UNet2DConditionModel,
        vae: &Autoencoder,
        conditioning: &Array,
        pooled: &Array,
        guidance: f32,
        seed: u64,
        edge: u32,
        steps: usize,
        _dtype: Dtype,
    ) -> Result<Image> {
        render_sample(
            unet,
            vae,
            &self.sampler,
            conditioning,
            pooled,
            guidance,
            seed,
            edge,
            steps,
        )
    }
}

/// LoRA/LoKr trainer for Stable Diffusion XL, implementing the core [`Trainer`] surface: a frozen
/// f32 base (U-Net + dual CLIP + VAE + tokenizer) that drives the shared SDXL-family backbone
/// ([`train_family`]) — caching a captioned image dataset to VAE-latents + dual-CLIP
/// conditioning/pooled embeds, then running the functional-autograd loop — and writes an adapter that
/// round-trips through the SDXL inference loader.
pub struct SdxlTrainer {
    descriptor: TrainerDescriptor,
    vae: Autoencoder,
    unet: UNet2DConditionModel,
    hooks: SdxlHooks,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "sdxl",
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
        // sc-2127 (epic 2123): honors `resolution_buckets` — the shared family backbone caches one
        // latent (+ its edge's `time_ids`) per bucket and walks them through a `BucketSchedule`.
        // sc-24828 (epic 2123): subject-masked ε loss, wired in the shared `train_family` (the
        // weight is resampled per bucket next to each latent).
        // sc-24830 (epic 2123): depth anchoring — the shared decoded-x0 perceptual path (TAESDXL
        // decode of the ε-prediction's x0 → Depth-Anything-V2 → cached round-trip reference),
        // wired in the shared `train_family` on both the dense and block-checkpointed forwards.
        // sc-24833 (epic 2123): the VAE anchor (same family decoder → FLUX.2 encoder taps) and
        // E-LatentLPIPS (this latent family's published weights), both through the shared aux-loss
        // builder this trainer already drives.
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
            vae_anchor_loss: true,
            latent_lpips_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// Construct the trainer from an SDXL snapshot directory (the diffusers multi-component tree:
/// `tokenizer/ text_encoder/ text_encoder_2/ unet/ vae/`). Loads the base at **f32** (training needs
/// the dense, high-precision base for clean autograd; inference runs fp16). Registered via
/// [`mlx_gen::TrainerRegistration`].
///
/// The weights load lazily (sc-2124): construction only checks the spec, so `validate` and `train`'s
/// refusal floors never read weights; see [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    snapshot_root(spec)?;
    Ok(Box::new(mlx_gen::train::lazy::LazyTrainer::new(
        trainer_descriptor(),
        validate_floors,
        {
            let spec = spec.clone();
            move || load_weights(&spec)
        },
    )))
}

/// The snapshot directory a trainer spec names — a single `.safetensors` file is refused.
fn snapshot_root(spec: &LoadSpec) -> Result<&PathBuf> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(mlx_gen::Error::Msg(
            "sdxl trainer expects a snapshot directory (tokenizer/ text_encoder/ text_encoder_2/ \
             unet/ vae/), not a single .safetensors file"
                .into(),
        )),
    }
}

/// The weight load behind [`load_trainer`], run by [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer) on first need.
fn load_weights(spec: &LoadSpec) -> Result<SdxlTrainer> {
    let root = snapshot_root(spec)?;
    Ok(SdxlTrainer {
        descriptor: trainer_descriptor(),
        vae: crate::loader::load_vae(root)?,
        unet: crate::loader::load_unet(root)?,
        hooks: SdxlHooks {
            tokenizer: crate::loader::load_tokenizer(root)?,
            te1: Some(crate::loader::load_text_encoder_1(root)?),
            te2: Some(crate::loader::load_text_encoder_2(root)?),
            sampler: EulerSampler::new(&DiffusionConfig::sdxl_base(), true)?,
        },
    })
}

// The trainer registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

/// Every weights-free [`Trainer::validate`] floor — the whole of it (none needs the loaded base).
fn validate_floors(descriptor: &TrainerDescriptor, req: &TrainingRequest) -> gen_core::Result<()> {
    // Shared control-training floor (F-006): a LoRA-only trainer must reject a control-branch
    // request (typed `Unsupported`) rather than silently training a plain adapter.
    gen_core::train::validate_control_request(descriptor, req)?;
    // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
    // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA.
    gen_core::train::validate_full_finetune_request(descriptor, req)?;
    // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
    // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
    gen_core::train::validate_training_techniques(descriptor, req)?;
    gen_core::train::validate_edit_request(descriptor, req)?;
    if req.items.is_empty() {
        return Err("sdxl trainer: dataset is empty".into());
    }
    if req.config.rank == 0 {
        return Err("sdxl trainer: rank must be > 0".into());
    }
    // F-023: steps == 0 makes the `1..=steps` loop empty and the run returns `Canceled` (the
    // family.rs comment claims validate rejects this — it didn't). z-image checks it; mirror.
    if req.config.steps == 0 {
        return Err("sdxl trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "sdxl trainer: optimizer '{}' is not available on MLX training (supported: adamw, \
             adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    Ok(())
}

impl Trainer for SdxlTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_floors(self.descriptor(), req)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        self.validate(req)?;
        train_family(&mut self.hooks, &mut self.unet, &self.vae, req, on_progress)
            .map_err(Into::into)
    }
}

/// The SDXL renormalized-DDPM `ᾱ` at sigma-table index `t`: `1/(σ(t)²+1)`.
fn sdxl_alpha_bar(sampler: &EulerSampler, t: f32) -> f32 {
    let s = sampler.sigma(t);
    1.0 / (s * s + 1.0)
}

/// Projected dense first-step peak memory, in GB, as a function of the latent pixel count
/// `p = (edge/8)²` (the SDXL VAE downscales /8; the U-Net working set is dominated by the conv-resnet
/// activations that scale with this spatial extent — unlike z-image, whose peak is the attention
/// seq² term). An empirical fit to peaks measured on the 128 GB target (`first_step_memory_sweep`,
/// rank 16 / 560 LoRA targets / batch 1), AFTER the CLIP encoders are freed (the train-loop working
/// set — what the guard models, since the guard projects the loop peak that follows caching).
/// The structure is `resident + linear·p + quad·p²`: the constant is the resident base (U-Net + VAE,
/// encoders freed), the linear term the per-pixel conv activations across the down/mid/up stack, the
/// small quadratic the attention seq² at the 64²/32² grids. bf16 roughly halves all three. Assumes
/// micro-batch 1 (the loop's actual shape); refit if the LoRA-target count or batch changes.
fn projected_dense_peak_gb(p: f64, bf16: bool) -> f64 {
    // Measured AFTER the CLIP encoders are freed (the train-loop working set): `first_step_memory_sweep`
    // on the 128 GB target, f32 512/768/1024/1280 → 15.0/22.3/36.4/60.4 GB; bf16 → 7.7/11.3/18.3/30.4
    // GB. The bf16 1024 peak (~18 GB) fits a 32 GB Mac; 1280 (~30 GB) is the headroom edge. p=(edge/8)².
    if bf16 {
        5.56 + 4.258e-4 * p + 2.157e-8 * p * p
    } else {
        10.78 + 8.582e-4 * p + 4.293e-8 * p * p
    }
}

#[cfg(test)]
mod preflight_tests {
    use super::projected_dense_peak_gb;

    /// The empirical fit must reproduce the measured (post-encoder-free) first-step peaks within a few
    /// GB and stay monotonic — it is the basis of the pre-flight OOM guard. p = (edge/8)²: edge
    /// 512→4096, 768→9216, 1024→16384, 1280→25600. Measured (`first_step_memory_sweep`, 128 GB):
    /// f32 15.0/22.3/36.4/60.4 GB; bf16 7.7/11.3/18.3/30.4 GB.
    #[test]
    fn projection_matches_measured_curve() {
        for (p, measured) in [
            (4096.0, 15.0),
            (9216.0, 22.3),
            (16384.0, 36.4),
            (25600.0, 60.4),
        ] {
            let proj = projected_dense_peak_gb(p, false);
            assert!(
                (proj - measured).abs() < 3.0,
                "f32 projection at p={p} = {proj:.1} GB, expected ≈{measured} GB"
            );
        }
        for (p, measured) in [
            (4096.0, 7.7),
            (9216.0, 11.3),
            (16384.0, 18.3),
            (25600.0, 30.4),
        ] {
            let proj = projected_dense_peak_gb(p, true);
            assert!(
                (proj - measured).abs() < 3.0,
                "bf16 projection at p={p} = {proj:.1} GB, expected ≈{measured} GB"
            );
        }
        // Monotonic increasing; bf16 strictly below f32.
        assert!(projected_dense_peak_gb(4096.0, false) < projected_dense_peak_gb(16384.0, false));
        assert!(projected_dense_peak_gb(16384.0, true) < projected_dense_peak_gb(16384.0, false));
        // sc-4941's 32 GB-Mac goal: bf16 1024 LoRA training fits a 32 GB box. Such a box reports an
        // MLX working-set limit of ~22 GB; the guard's budget is limit × 0.85 ≈ 18.7 GB. bf16 1024
        // (~18.3 GB) clears it; f32 1024 (~36 GB) does not — so an f32 run on a 32 GB box is correctly
        // steered to bf16 / lower resolution rather than SIGKILLed.
        assert!(projected_dense_peak_gb(16384.0, true) < 18.7); // bf16 1024 fits a 32 GB box
        assert!(projected_dense_peak_gb(16384.0, false) > 18.7); // f32 1024 does not
    }

    /// sc-24828: SDXL declares subject-masked loss (wired in the shared `train_family`).
    #[test]
    fn descriptor_declares_subject_mask_loss() {
        assert!(super::trainer_descriptor().techniques.subject_mask_loss);
    }

    /// sc-2127 / epic 2123 E7: with buckets `[512, 1024]` the pre-flight guard projects the 1024
    /// bucket's peak — equal to a 1024-only run and above a 512-only run.
    #[test]
    fn guard_projection_sizes_for_the_largest_bucket() {
        use super::family::dense_peak_for_edges;
        for bf16 in [false, true] {
            let mixed = dense_peak_for_edges(projected_dense_peak_gb, &[512, 1024], bf16);
            let at_1024 = dense_peak_for_edges(projected_dense_peak_gb, &[1024], bf16);
            let at_512 = dense_peak_for_edges(projected_dense_peak_gb, &[512], bf16);
            assert_eq!(mixed, at_1024);
            assert!(mixed.1 > at_512.1);
        }
    }

    /// sc-2127: the SDXL trainer declares multi-resolution bucket support.
    #[test]
    fn descriptor_declares_resolution_buckets() {
        assert!(super::trainer_descriptor().techniques.resolution_buckets);
    }
}

// ===========================================================================================
// sc-4941 (sibling of z-image sc-4874) — first-step peak-memory characterization for the SDXL
// U-Net LoRA trainer. The story's explicit mandate is "measure before assuming z-image magnitude":
// the SDXL U-Net's attention runs at SMALLER latent grids (64²/32², not z-image's unified 64²×30
// blocks) and its conv resnets have no seq² term, so the first-step working set must be measured,
// not extrapolated from z-image's 135 GB-at-1024 curve. This harness drives the exact inner step
// (`compute_loss_grads` + the backward grad `eval` the real loop forces) at swept resolution with
// MLX peak probes around it.
//
//   cargo test -p mlx-gen-sdxl --release --lib first_step -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod first_step_repro {
    use super::*;
    use family::{compute_loss_grads, resolve_target_paths};
    use mlx_gen::media::Image;
    use mlx_gen::train::dataset::center_crop_square;
    use mlx_gen::train::lora::{build_lora_targets, LoraParams, TrainAdapter};
    use mlx_gen::TrainingConfig;
    use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
    use mlx_rs::transforms::eval;
    use std::path::PathBuf;

    use crate::pipeline::encode_init_latents;

    /// Resolve the SDXL diffusers snapshot root from the required `SDXL_SNAPSHOT` env var. sc-13668:
    /// there is no implicit default — the source snapshot path must be passed in explicitly.
    fn snapshot() -> Option<PathBuf> {
        std::env::var("SDXL_SNAPSHOT").ok().map(PathBuf::from)
    }

    #[test]
    fn source_root_requires_explicit_env_no_default() {
        let key = "SDXL_SNAPSHOT";
        let saved = std::env::var(key).ok();
        std::env::remove_var(key);
        assert!(
            snapshot().is_none(),
            "the source snapshot root must come from {key}: sc-13668 removed the implicit default"
        );
        std::env::set_var(key, "/sentinel/sdxl");
        assert_eq!(snapshot(), Some(PathBuf::from("/sentinel/sdxl")));
        match saved {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    /// A solid-colour `edge`×`edge` RGB source image (latent magnitude is irrelevant; the graph
    /// size — driven by resolution — is the variable under test).
    fn swatch(edge: u32) -> Image {
        let mut img = image::RgbImage::new(edge, edge);
        for px in img.pixels_mut() {
            *px = image::Rgb([180u8, 60, 90]);
        }
        Image {
            width: edge,
            height: edge,
            pixels: img.into_raw(),
        }
    }

    fn gb(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Run one first training step at `edge` in `dtype` (caller casts the U-Net to match) and report
    /// the peak GPU memory across forward+backward (forces the grad eval — the real step-1 kill point).
    #[allow(clippy::too_many_arguments)]
    fn one_step(
        trainer: &mut SdxlTrainer,
        adapter: &TrainAdapter,
        params: &LoraParams,
        cond: &Array,
        pooled: &Array,
        edge: u32,
        dtype: Dtype,
        checkpoint_targets: Option<Vec<String>>,
        tag: &str,
    ) -> Result<(f32, f64)> {
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge)?;
        let noise = random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1)?))?;
        let time_ids = text_time_ids(1);
        eval([&x0, &noise]).unwrap();

        clear_cache();
        reset_peak_memory();
        let before = get_active_memory();
        let t0 = std::time::Instant::now();
        let (loss, grads) = compute_loss_grads(
            &trainer.hooks,
            &mut trainer.unet,
            params,
            adapter,
            16.0,
            16.0,
            &x0,
            cond,
            pooled,
            &time_ids,
            TrainTimestep::Sigma(500.0),
            &noise,
            false,
            None,
            dtype,
            checkpoint_targets,
        )?;
        eval(grads.values())?; // force the backward (true working set)
        let secs = t0.elapsed().as_secs_f64();
        let peak = get_peak_memory();
        eprintln!(
            "[sc-4941]   edge {edge:>4} {tag}  loss {loss:.5}  active-before {:.2} GB  peak {:.2} GB  step {secs:.2}s",
            gb(before),
            gb(peak)
        );
        Ok((loss, gb(peak)))
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

    fn build_trainer_and_adapter() -> (SdxlTrainer, TrainAdapter, LoraParams, Array, Array) {
        let root = snapshot()
            .expect("set SDXL_SNAPSHOT to the stable-diffusion-xl-base-1.0 snapshot root");
        let mut trainer = SdxlTrainer {
            descriptor: trainer_descriptor(),
            vae: crate::loader::load_vae(&root).unwrap(),
            unet: crate::loader::load_unet(&root).unwrap(),
            hooks: SdxlHooks {
                tokenizer: crate::loader::load_tokenizer(&root).unwrap(),
                te1: Some(crate::loader::load_text_encoder_1(&root).unwrap()),
                te2: Some(crate::loader::load_text_encoder_2(&root).unwrap()),
                sampler: EulerSampler::new(&DiffusionConfig::sdxl_base(), true).unwrap(),
            },
        };
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&trainer.unet, &cfg);
        let (targets, params) =
            build_lora_targets(&mut trainer.unet, &target_paths, 16, 7).unwrap();
        let (cond, pooled) = trainer
            .hooks
            .encode_prompt("a solid colour swatch")
            .unwrap();
        eval([&cond, &pooled]).unwrap();
        // Drop the CLIP encoders exactly as the backbone does after caching, so the measured peaks
        // reflect the post-free training working set.
        trainer.hooks.te1 = None;
        trainer.hooks.te2 = None;
        mlx_rs::memory::clear_cache();
        eprintln!(
            "[sc-4941] loaded SDXL trainer (encoders freed); {} LoRA targets; cond {:?} pooled {:?}",
            targets.len(),
            cond.shape(),
            pooled.shape()
        );
        (
            trainer,
            TrainAdapter::Lora { targets },
            params,
            cond,
            pooled,
        )
    }

    /// Sweep resolution tiny → production, printing the dense first-step peak curve in f32 then bf16
    /// (cond/pooled cast to match). These measured points are the basis of the `projected_dense_peak_gb`
    /// guard fit — refit the constants if this prints materially different numbers.
    #[test]
    #[ignore = "needs real SDXL weights; run as its own process"]
    fn first_step_memory_sweep() {
        let (mut trainer, adapter, params, cond, pooled) = build_trainer_and_adapter();
        eprintln!("[sc-4941] SDXL dense f32 first-step sweep:");
        for edge in [256u32, 512, 768, 1024, 1280] {
            let _ = one_step(
                &mut trainer,
                &adapter,
                &params,
                &cond,
                &pooled,
                edge,
                Dtype::Float32,
                None,
                "f32",
            )
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4941] casting U-Net to bf16…");
        trainer.unet.cast_weights(Dtype::Bfloat16).unwrap();
        let cond_b = cond.as_dtype(Dtype::Bfloat16).unwrap();
        let pooled_b = pooled.as_dtype(Dtype::Bfloat16).unwrap();
        let tp: Vec<String> = match &adapter {
            TrainAdapter::Lora { targets } => targets.iter().map(|t| t.path.clone()).collect(),
            _ => Vec::new(),
        };
        clear_cache();
        eprintln!("[sc-4941] SDXL dense bf16 first-step sweep:");
        for edge in [256u32, 512, 768, 1024, 1280] {
            let _ = one_step(
                &mut trainer,
                &adapter,
                &params,
                &cond_b,
                &pooled_b,
                edge,
                Dtype::Bfloat16,
                None,
                "bf16",
            )
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4941] SDXL bf16 BLOCK-CHECKPOINTED first-step sweep (1024/1280 — the 32 GB lever):");
        for edge in [1024u32, 1280, 1536] {
            let _ = one_step(
                &mut trainer,
                &adapter,
                &params,
                &cond_b,
                &pooled_b,
                edge,
                Dtype::Bfloat16,
                Some(tp.clone()),
                "bf16-ckpt",
            )
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4941] sweep complete");
    }

    /// sc-4941 — always-bit-identical: the SDPA-segment checkpoint (opt-in) must not change grads vs
    /// the retained backward. Same decomposed attention, recomputed instead of retained.
    #[test]
    #[ignore = "needs real SDXL weights; run as its own process"]
    fn attn_ckpt_grads_match_retained() {
        let (mut trainer, adapter, params, cond, pooled) = build_trainer_and_adapter();
        let edge = 256u32;
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        let time_ids = text_time_ids(1);
        eval([&x0, &noise]).unwrap();
        let grads_of = |t: &mut SdxlTrainer, on: bool| -> LoraParams {
            t.unet.set_sdpa_checkpoint(on);
            let (_l, g) = compute_loss_grads(
                &t.hooks,
                &mut t.unet,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cond,
                &pooled,
                &time_ids,
                TrainTimestep::Sigma(500.0),
                &noise,
                false,
                None,
                Dtype::Float32,
                None,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };
        let g_retained = grads_of(&mut trainer, false);
        let g_ckpt = grads_of(&mut trainer, true);
        let max_rel = max_rel_diff(&g_retained, &g_ckpt);
        eprintln!("[sc-4941] attn-ckpt-vs-retained grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-5,
            "attention-segment checkpointing must not change grads: max rel {max_rel:.2e}"
        );
    }

    /// sc-4941 — block (gradient) checkpointing must not change the math: the per-block checkpointed
    /// forward+grads must match the dense path within fp tolerance (it reuses the same install + block
    /// forward, recompute-only). This is the correctness gate for the `gradient_checkpointing` lever.
    #[test]
    #[ignore = "needs real SDXL weights; run as its own process"]
    fn block_ckpt_grads_match_dense() {
        let (mut trainer, adapter, params, cond, pooled) = build_trainer_and_adapter();
        let edge = 256u32; // math is resolution-agnostic; small enough that the dense path is cheap
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        let time_ids = text_time_ids(1);
        eval([&x0, &noise]).unwrap();
        let tp: Vec<String> = match &adapter {
            TrainAdapter::Lora { targets } => targets.iter().map(|t| t.path.clone()).collect(),
            _ => unreachable!("LoRA adapter in this harness"),
        };
        let grads_of = |t: &mut SdxlTrainer, ck: Option<Vec<String>>| -> LoraParams {
            let (_l, g) = compute_loss_grads(
                &t.hooks,
                &mut t.unet,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cond,
                &pooled,
                &time_ids,
                TrainTimestep::Sigma(500.0),
                &noise,
                false,
                None,
                Dtype::Float32,
                ck,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };
        let g_dense = grads_of(&mut trainer, None);
        let g_ckpt = grads_of(&mut trainer, Some(tp));
        let max_rel = max_rel_diff(&g_dense, &g_ckpt);
        eprintln!("[sc-4941] block-ckpt-vs-dense grad max relative diff: {max_rel:.2e}");
        // Recompute-vs-retained fp noise (not a structural diff — a real checkpointing bug, like the
        // duplicate-output VJP corruption this gate caught, shows up at ~1e0). The bound is a few e-3
        // because the conv-heavy recompute reorders fp accumulation; a genuine mismatch is orders of
        // magnitude larger.
        assert!(
            max_rel < 5e-3,
            "block checkpointing must match the dense grads: max rel {max_rel:.2e}"
        );
    }

    /// sc-4941 — bf16 is mixed precision, NOT bit parity: assert the bf16 grads point the same way as
    /// f32 (global cosine + large-norm cosine) and the bf16 working set is genuinely smaller (a silent
    /// f32 re-promotion in the forward would pass the cosine check while saving nothing — the memory
    /// ratio IS the dtype assertion). Runs f32 first (the cast is destructive), then casts to bf16.
    #[test]
    #[ignore = "needs real SDXL weights; run as its own process"]
    fn bf16_grads_direction_and_memory_vs_f32() {
        let (mut trainer, adapter, params, cond, pooled) = build_trainer_and_adapter();

        // Grad reference at 256 in f32.
        let edge = 256u32;
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        let time_ids = text_time_ids(1);
        eval([&x0, &noise]).unwrap();
        let grads_of =
            |t: &mut SdxlTrainer, c: &Array, p: &Array, dt: Dtype| -> (f32, LoraParams) {
                let (l, g) = compute_loss_grads(
                    &t.hooks,
                    &mut t.unet,
                    &params,
                    &adapter,
                    16.0,
                    16.0,
                    &x0,
                    c,
                    p,
                    &time_ids,
                    TrainTimestep::Sigma(500.0),
                    &noise,
                    false,
                    None,
                    dt,
                    None,
                )
                .unwrap();
                eval(g.values()).unwrap();
                (l, g)
            };
        let (f32_loss, g_f32) = grads_of(&mut trainer, &cond, &pooled, Dtype::Float32);

        // Memory A/B at 768 in f32 (big enough that activations dominate).
        let (_, f32_peak) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cond,
            &pooled,
            768,
            Dtype::Float32,
            None,
            "f32",
        )
        .unwrap();

        trainer.unet.cast_weights(Dtype::Bfloat16).unwrap();
        clear_cache();
        let cond_b = cond.as_dtype(Dtype::Bfloat16).unwrap();
        let pooled_b = pooled.as_dtype(Dtype::Bfloat16).unwrap();
        let (bf16_loss, g_bf16) = grads_of(&mut trainer, &cond_b, &pooled_b, Dtype::Bfloat16);
        assert!(
            bf16_loss.is_finite(),
            "bf16 loss must be finite: {bf16_loss}"
        );
        eprintln!("[sc-4941] loss f32 {f32_loss:.5} vs bf16 {bf16_loss:.5}");

        let mut per: Vec<(String, f32, f32, f32)> = Vec::new(); // (key, cos, na, nb)
        let (mut gdot, mut gna2, mut gnb2) = (0f64, 0f64, 0f64);
        for (k, a) in &g_f32 {
            let b = g_bf16.get(k).expect("same keys");
            let dot = a.multiply(b).unwrap().sum(None).unwrap().item::<f32>();
            let na2 = a.square().unwrap().sum(None).unwrap().item::<f32>();
            let nb2 = b.square().unwrap().sum(None).unwrap().item::<f32>();
            gdot += dot as f64;
            gna2 += na2 as f64;
            gnb2 += nb2 as f64;
            let (na, nb) = (na2.sqrt(), nb2.sqrt());
            if na > 1e-12 && nb > 1e-12 {
                per.push((k.to_string(), dot / (na * nb), na, nb));
            }
        }
        let global_cos = (gdot / (gna2.sqrt() * gnb2.sqrt())) as f32;
        per.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        let max_norm = per.iter().map(|p| p.2).fold(0f32, f32::max);
        eprintln!("[sc-4941] bf16-vs-f32 grads: global cosine {global_cos:.5}; worst per-param:");
        for (k, c, na, nb) in per.iter().take(8) {
            eprintln!(
                "    {k}: cos {c:.4}  |g| {na:.3e} vs {nb:.3e}  rel-norm {:.2e}",
                na / max_norm
            );
        }
        let min_large = per
            .iter()
            .filter(|p| p.2 >= 0.01 * max_norm)
            .map(|p| p.1)
            .fold(1f32, f32::min);
        eprintln!("[sc-4941] min cosine among params with |g| >= 1% of max: {min_large:.4}");
        assert!(
            global_cos > 0.995,
            "bf16 global grad must point the same way as f32: {global_cos:.5}"
        );
        assert!(
            min_large > 0.95,
            "a large-norm param's bf16 grad diverged from f32 (systematic bug): {min_large:.4}"
        );

        let (_, bf16_peak) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cond_b,
            &pooled_b,
            768,
            Dtype::Bfloat16,
            None,
            "bf16",
        )
        .unwrap();
        eprintln!(
            "[sc-4941] 768 peak f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB ({:.0}%)",
            100.0 * bf16_peak / f32_peak
        );
        assert!(
            bf16_peak < 0.70 * f32_peak,
            "bf16 must materially shrink the working set: f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB"
        );
    }
}

/// sc-24830 (epic 2123 depth anchoring) — the SDXL-family step seam ([`family::run_train_step`] /
/// [`family::compute_step_loss_grads`], shared with Kolors) on a tiny random-init U-Net
/// ([`family::test_support`]: 2 blocks, 4 latent channels) with the real [`SdxlHooks`] noise
/// schedule, a random-init tiny TAESDXL-layout decoder (4 latent channels) and a random-init tiny
/// Depth-Anything-V2. Seconds, a few MB; no weights downloaded.
#[cfg(test)]
mod depth_anchoring_tests {
    use super::*;
    use family::test_support::{tiny_unet, TINY_CONTEXT_DIM, TINY_POOLED_DIM};
    use family::{
        aux_driver, compute_step_loss_grads, load_perceptual_path, perceptual_footprint_gb,
        preflight_memory_guard_with_budget, resolve_target_paths, run_train_step, trained_timestep,
        AuxStep, CachedSample, StepLosses,
    };
    use mlx_gen::gen_core::BucketSchedule;
    use mlx_gen::train::lora::{build_lora_targets, LoraParams, TrainAdapter};
    use mlx_gen::train::loss::reduce_loss;
    use mlx_gen::train::perceptual::{
        AuxDriver, AuxLossSchedule, Parameterization, PerceptualPath,
    };
    use mlx_gen::TrainingConfig;
    use mlx_rs::error::{Exception, Result as MlxResult};
    use mlx_rs::transforms::{eval, keyed_value_and_grad};

    /// The production hooks with a two-token CLIP vocabulary (the tests never tokenize) and the
    /// real SDXL sigma table.
    fn hooks() -> SdxlHooks {
        let tmp = tempfile::tempdir().unwrap();
        let vocab = tmp.path().join("vocab.json");
        let merges = tmp.path().join("merges.txt");
        std::fs::write(&vocab, r#"{"<|startoftext|>": 0, "<|endoftext|>": 1}"#).unwrap();
        std::fs::write(&merges, "#version: 0.2\n").unwrap();
        SdxlHooks {
            tokenizer: ClipBpeTokenizer::from_files(&vocab, &merges).unwrap(),
            te1: None,
            te2: None,
            sampler: EulerSampler::new(&DiffusionConfig::sdxl_base(), true).unwrap(),
        }
    }

    fn schedule() -> AuxLossSchedule {
        AuxLossSchedule {
            weight: 0.1,
            t_min: 0.0,
            t_max: 1.0,
            every_n: 2,
        }
    }

    fn path() -> PerceptualPath {
        path_with(schedule())
    }

    fn path_with(schedule: AuxLossSchedule) -> PerceptualPath {
        mlx_gen_perceptual::testing::tiny_depth_path(4, schedule).unwrap()
    }

    fn cfg() -> TrainingConfig {
        let mut cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            seed: 7,
            train_dtype: "f32".into(),
            ..Default::default()
        };
        cfg.depth_anchoring.schedule = schedule();
        cfg
    }

    /// `n` cached items: clean NHWC `[1, 6, 4, 4]` latents (h 6 ≠ 4 channels, so a missed NHWC→NCHW
    /// transpose is a shape error, not a silent mis-read; 48×32 decoded) + tiny conditioning.
    fn cache_n(n: u64) -> Vec<CachedSample> {
        (0..n)
            .map(|i| {
                let r = |shape: &[i32], k: u64| {
                    random::normal::<f32>(shape, None, None, Some(&random::key(k).unwrap()))
                        .unwrap()
                };
                let x0 = r(&[1, 6, 4, 4], 100 + i);
                let cond = r(&[1, 7, TINY_CONTEXT_DIM], 200 + i);
                let pooled = r(&[1, TINY_POOLED_DIM], 300 + i);
                eval([&x0, &cond, &pooled]).unwrap();
                CachedSample {
                    x0,
                    cond,
                    pooled,
                    time_ids: crate::pipeline::text_time_ids(1),
                    mask_weight: None,
                }
            })
            .collect()
    }

    fn adapter(
        unet: &mut UNet2DConditionModel,
        cfg: &TrainingConfig,
    ) -> (TrainAdapter, LoraParams) {
        let paths = resolve_target_paths(unet, cfg);
        assert!(!paths.is_empty());
        let (targets, params) =
            build_lora_targets(unet, &paths, cfg.rank as i32, cfg.seed).unwrap();
        (TrainAdapter::Lora { targets }, params)
    }

    fn checkpoint_targets(adapter: &TrainAdapter) -> Vec<String> {
        match adapter {
            TrainAdapter::Lora { targets } => targets.iter().map(|t| t.path.clone()).collect(),
            _ => unreachable!(),
        }
    }

    fn single_bucket(n: usize) -> BucketSchedule {
        BucketSchedule::new(
            n,
            &[mlx_gen::gen_core::ResolutionBucket {
                resolution: 32,
                repeats: 1,
            }],
            7,
        )
    }

    fn prepared(cache: &[CachedSample], accum: u32) -> AuxDriver {
        aux_driver(
            path(),
            cache,
            &single_bucket(cache.len()),
            accum,
            &Default::default(),
        )
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        hooks: &SdxlHooks,
        unet: &mut UNet2DConditionModel,
        params: &LoraParams,
        adapter: &TrainAdapter,
        cfg: &TrainingConfig,
        cache: &[CachedSample],
        path: Option<&mut AuxDriver>,
        n: u32,
        ckpt: Option<Vec<String>>,
    ) -> (StepLosses, LoraParams) {
        let schedule = single_bucket(cache.len());
        let (l, g) = run_train_step(
            hooks,
            unet,
            params,
            adapter,
            cfg,
            cache,
            &schedule,
            path,
            n,
            false,
            Dtype::Float32,
            ckpt,
        )
        .unwrap();
        eval(g.values()).unwrap();
        (l, g)
    }

    fn abs_sum(g: &LoraParams, filter: &str) -> f32 {
        let picked: Vec<f32> = g
            .iter()
            .filter(|(k, _)| k.ends_with(filter))
            .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
            .collect();
        assert!(!picked.is_empty(), "no '{filter}' grads");
        picked.iter().sum()
    }

    /// The SDXL ε parameterization is exact: with the true noise as the "prediction", the x0
    /// recovery inverts [`SdxlFamilyHooks::add_noise`] at the hook's `ᾱ`. Mutations: return
    /// `σ²/(σ²+1)` (or use the `t − 1` table entry) from `alpha_bar` ⇒ the recovery misses x0 ⇒ red.
    #[test]
    fn epsilon_recovery_inverts_the_sdxl_noising() {
        let h = hooks();
        let x0 = random::normal::<f32>(&[1, 4, 4, 4], None, None, Some(&random::key(1).unwrap()))
            .unwrap();
        let noise =
            random::normal::<f32>(&[1, 4, 4, 4], None, None, Some(&random::key(2).unwrap()))
                .unwrap();
        for t in [1.0f32, 37.0, 500.0, 999.0, 1000.0] {
            let ts = TrainTimestep::Sigma(t);
            let noisy = h.add_noise(&x0, &noise, ts).unwrap();
            let alpha_bar = h.alpha_bar(ts).unwrap();
            let rec = Parameterization::Epsilon { alpha_bar }
                .recover_x0(&noisy, &noise)
                .unwrap();
            let err = rec
                .subtract(&x0)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            // √ᾱ at t = 1000 is ~0.068, so f32 rounding of x_t is amplified ~15×.
            assert!(err < 2e-3, "t={t}: |x0_hat − x0|max = {err}");
        }
    }

    /// An aux-only step's remapped noise level lands on the trained integer table (`[1, 1000]`)
    /// and the mapping round-trips. Mutation: drop the `clamp(1, max)` ⇒ level 0 maps to index 0
    /// (σ = 0, never trained) ⇒ red.
    #[test]
    fn noise_level_maps_onto_the_trained_sdxl_indices() {
        let h = hooks();
        for t in [1.0f32, 250.0, 1000.0] {
            let level = h.noise_level(TrainTimestep::Sigma(t));
            assert_eq!(h.timestep_at(level).unet_time(), t);
        }
        assert_eq!(h.timestep_at(0.0).unet_time(), 1.0);
        assert_eq!(h.timestep_at(1.0).unet_time(), 1000.0);
    }

    /// AC (a)/(b), dense and block-checkpointed: a depth step (an image's every 2nd update) trains
    /// the LoRA through the depth term alone — no diffusion term, total == aux, non-zero finite
    /// LoRA-B gradient — and a diffusion step carries no depth term. Mutation: force
    /// `diffusion_on = true` in `compute_step_loss_grads` ⇒ the depth step reports a diffusion term
    /// ⇒ red.
    #[test]
    fn depth_step_trains_the_lora_through_depth_only() {
        let h = hooks();
        for ckpt in [false, true] {
            let mut unet = tiny_unet(3).unwrap();
            let cfg = cfg();
            let (adapter, params) = adapter(&mut unet, &cfg);
            let targets = ckpt.then(|| checkpoint_targets(&adapter));
            let cache = cache_n(1);
            let mut d = prepared(&cache, 1);
            let (diff, _) = step(
                &h,
                &mut unet,
                &params,
                &adapter,
                &cfg,
                &cache,
                Some(&mut d),
                1,
                targets.clone(),
            );
            assert_eq!(
                diff.aux, None,
                "ckpt={ckpt}: diffusion step has no depth term"
            );
            assert_eq!(Some(diff.total), diff.diffusion);
            let (depth, g) = step(
                &h,
                &mut unet,
                &params,
                &adapter,
                &cfg,
                &cache,
                Some(&mut d),
                2,
                targets,
            );
            assert_eq!(
                depth.diffusion, None,
                "ckpt={ckpt}: no diffusion on a depth step"
            );
            let aux = depth.aux.expect("depth step carries the depth term");
            assert!(
                aux > 0.0 && aux.is_finite(),
                "ckpt={ckpt}: depth term {aux}"
            );
            assert_eq!(depth.total, aux);
            let gb = abs_sum(&g, ".lora_b");
            assert!(gb > 0.0 && gb.is_finite(), "ckpt={ckpt}: LoRA-B |Σ| = {gb}");
        }
    }

    /// Epic 2123 E8 (the Candle `a_reverted_aux_step_trains_at_the_remapped_level` twin): a step the
    /// alternation claims for a loss that skips the image is reverted to diffusion by
    /// `StepPlan::without_skipped` — and still trains at the plan's remapped timestep, never the raw
    /// sampled one; an unclaimed diffusion step keeps the sampled timestep exactly. Mutation: remap
    /// only `if !plan.diffusion` (the pre-fix code) ⇒ the reverted step trains at the raw `t` ⇒ red.
    #[test]
    fn a_reverted_aux_step_trains_at_the_remapped_level() {
        use mlx_gen::gen_core::train::plan_step;
        let h = hooks();
        let window = AuxLossSchedule {
            weight: 0.5,
            t_min: 0.6,
            t_max: 0.9,
            every_n: 2,
        };
        let raw = h.timestep_at(0.1);
        let reverted = plan_step(&[window], 2, h.noise_level(raw)).without_skipped(|_| true);
        assert!(
            reverted.diffusion && reverted.aux.is_empty(),
            "{reverted:?}"
        );
        let t = trained_timestep(&h, raw, &reverted);
        assert_eq!(
            t.unet_time(),
            h.timestep_at(reverted.noise_level).unet_time()
        );
        assert_ne!(
            t.unet_time(),
            raw.unet_time(),
            "the remapped level, not the sampled one"
        );
        let diffusion = plan_step(&[window], 1, h.noise_level(raw));
        assert_eq!(
            trained_timestep(&h, raw, &diffusion).unet_time(),
            raw.unet_time()
        );
    }

    /// The aux step trains at the remapped timestep (window `[0.6, 0.8]` ⇒ index ∈ [600, 800]),
    /// with that timestep's `ᾱ` — the step equals a direct `compute_step_loss_grads` at
    /// `timestep_at(plan.noise_level)`. Mutation: keep the sampled `t` on an aux step ⇒ red.
    #[test]
    fn aux_step_trains_at_the_remapped_timestep() {
        let h = hooks();
        let mut unet = tiny_unet(3).unwrap();
        let mut cfg = cfg();
        let window = AuxLossSchedule {
            t_min: 0.6,
            t_max: 0.8,
            ..schedule()
        };
        cfg.depth_anchoring.schedule = window;
        let (adapter, params) = adapter(&mut unet, &cfg);
        let cache = cache_n(1);
        let mut d = aux_driver(
            path_with(window),
            &cache,
            &single_bucket(cache.len()),
            1,
            &Default::default(),
        )
        .unwrap();
        let mut l = None;
        for n in 1..=2 {
            l = Some(step(
                &h,
                &mut unet,
                &params,
                &adapter,
                &cfg,
                &cache,
                Some(&mut d),
                n,
                None,
            ));
        }
        let (depth, _) = l.unwrap();
        assert!(depth.aux.is_some());
        // Recompute step 2 directly at the remapped timestep.
        let raw = h
            .sample_timestep(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(2))
            .unwrap();
        let plan = d.path().plan(2, 0, h.noise_level(raw)).unwrap();
        let t = h.timestep_at(plan.noise_level);
        assert!((600.0..=800.0).contains(&t.unet_time()), "{t:?}");
        assert_ne!(t.unet_time(), raw.unet_time());
        let noise = random::normal::<f32>(
            &[1, 6, 4, 4],
            None,
            None,
            Some(&random::key(cfg.seed.wrapping_add(2).wrapping_mul(2) + 1).unwrap()),
        )
        .unwrap();
        let c = &cache[0];
        let (expected, _) = compute_step_loss_grads(
            &h,
            &mut unet,
            &params,
            &adapter,
            cfg.alpha,
            cfg.rank as f32,
            &c.x0,
            &c.cond,
            &c.pooled,
            &c.time_ids,
            t,
            &noise,
            false,
            None,
            Dtype::Float32,
            None,
            Some(AuxStep {
                path: d.path(),
                plan: &plan,
                entry: 0,
            }),
        )
        .unwrap();
        assert_eq!(depth, expected);
    }

    /// Interleaved alternation through the real step seam (sc-2124): with N = 2 images and
    /// `every_n = 2` no two depth steps run back to back and every image gets a diffusion and a
    /// depth step within 3 epochs, and with gradient accumulation 2 both micro-steps of a window
    /// share one kind. Mutations: key the plan on the bare global step (no period) ⇒ image 0 never
    /// trains depth ⇒ red; build the alternation with accumulation 1 ⇒ red.
    #[test]
    fn alternation_interleaves_per_window() {
        let h = hooks();
        for (n_items, accum, steps) in [(2u64, 1u32, 6u32), (3, 2, 8)] {
            let mut unet = tiny_unet(3).unwrap();
            let cfg = cfg();
            let (adapter, params) = adapter(&mut unet, &cfg);
            let cache = cache_n(n_items);
            let mut d = prepared(&cache, accum);
            let schedule = single_bucket(cache.len());
            let kinds: Vec<(usize, bool)> = (1..=steps)
                .map(|n| {
                    let (l, _) = step(
                        &h,
                        &mut unet,
                        &params,
                        &adapter,
                        &cfg,
                        &cache,
                        Some(&mut d),
                        n,
                        None,
                    );
                    (schedule.sample((n - 1) as usize).0, l.aux.is_some())
                })
                .collect();
            if accum == 1 {
                assert!(
                    kinds.windows(2).all(|w| !(w[0].1 && w[1].1)),
                    "two depth steps in a row: {kinds:?}"
                );
                for image in 0..n_items as usize {
                    let mine: Vec<bool> = kinds
                        .iter()
                        .filter(|(i, _)| *i == image)
                        .map(|k| k.1)
                        .collect();
                    assert!(mine.contains(&true) && mine.contains(&false), "{kinds:?}");
                }
            } else {
                for w in kinds.chunks(2) {
                    assert_eq!(w[0].1, w[1].1, "window {w:?} mixes kinds ({kinds:?})");
                }
                assert!(kinds.iter().any(|k| k.1), "{kinds:?}");
            }
            assert_eq!(d.path().reference_computations(), cache.len());
        }
    }

    /// sc-2127 integration: with two resolution buckets (item-major cache, different latent sizes)
    /// the depth reference is built once per (item, bucket) entry and every depth step trains its
    /// scheduled entry against that entry's own reference (same decode size), while the alternation
    /// stays keyed on the step's window. Mutations: key the reference / plan on the item instead of
    /// the entry in `run_train_step` ⇒ a depth step on bucket 1 compares against bucket 0's
    /// reference (shape mismatch) or the counter stops at the item count ⇒ red.
    #[test]
    fn two_buckets_keep_per_entry_references() {
        let h = hooks();
        let mut unet = tiny_unet(3).unwrap();
        let cfg = cfg();
        let (adapter, params) = adapter(&mut unet, &cfg);
        let items = 2usize;
        // cache[item * 2 + bucket]: bucket 0 = 6×4 latents, bucket 1 = 4×6.
        let mut cache = Vec::new();
        for (i, base) in cache_n(items as u64).into_iter().enumerate() {
            let other = random::normal::<f32>(
                &[1, 4, 6, 4],
                None,
                None,
                Some(&random::key(500 + i as u64).unwrap()),
            )
            .unwrap();
            let (cond, pooled, time_ids) = (
                base.cond.clone(),
                base.pooled.clone(),
                base.time_ids.clone(),
            );
            cache.push(base);
            cache.push(CachedSample {
                x0: other,
                cond,
                pooled,
                time_ids,
                mask_weight: None,
            });
        }
        let schedule = BucketSchedule::new(
            items,
            &[
                mlx_gen::gen_core::ResolutionBucket {
                    resolution: 48,
                    repeats: 1,
                },
                mlx_gen::gen_core::ResolutionBucket {
                    resolution: 32,
                    repeats: 1,
                },
            ],
            7,
        );
        let mut d = aux_driver(path(), &cache, &schedule, 1, &Default::default()).unwrap();
        let steps = 2 * schedule.epoch_len() as u32;
        let mut depth_on_bucket1 = false;
        for n in 1..=steps {
            let (l, _) = run_train_step(
                &h,
                &mut unet,
                &params,
                &adapter,
                &cfg,
                &cache,
                &schedule,
                Some(&mut d),
                n,
                false,
                Dtype::Float32,
                None,
            )
            .unwrap();
            let entry = schedule.cache_index((n - 1) as usize);
            depth_on_bucket1 |= l.aux.is_some() && entry % 2 == 1;
        }
        assert!(depth_on_bucket1, "no depth step on the second bucket");
        assert_eq!(d.path().reference_computations(), cache.len());
    }

    /// E1: with depth off nothing is loaded, the estimate adds nothing, and the step is
    /// bit-identical to the pre-epic-2123 closure (reproduced here verbatim); a diffusion-only step
    /// of an enabled path is bit-identical too. Mutation: always add the aux output (e.g. a
    /// 0-weighted copy) into the total, or reorder the reduction ⇒ red.
    #[test]
    fn everything_off_is_bit_identical_to_the_legacy_step() {
        assert!(load_perceptual_path("sdxl", &TrainingConfig::default())
            .unwrap()
            .is_none());
        assert_eq!(
            perceptual_footprint_gb(&TrainingConfig::default(), 1024, 10),
            0.0
        );
        let h = hooks();
        let mut unet = tiny_unet(3).unwrap();
        let cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            seed: 7,
            ..Default::default()
        };
        let (adapter, params) = adapter(&mut unet, &cfg);
        let cache = cache_n(2);
        let (off, g_off) = step(
            &h, &mut unet, &params, &adapter, &cfg, &cache, None, 1, None,
        );
        assert_eq!(off.aux, None);

        // The pre-sc-24830 `compute_loss_grads` body for step 1 (item 0).
        let c = &cache[0];
        let t = h
            .sample_timestep(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(1))
            .unwrap();
        let noise = random::normal::<f32>(
            &[1, 6, 4, 4],
            None,
            None,
            Some(&random::key(cfg.seed.wrapping_add(1).wrapping_mul(2) + 1).unwrap()),
        )
        .unwrap();
        let noisy = h.add_noise(&c.x0, &noise, t).unwrap();
        let (cond, pooled, ids) = (c.cond.clone(), c.pooled.clone(), c.time_ids.clone());
        let unet_ref = &mut unet;
        let adapter_ref = &adapter;
        let legacy = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
            adapter_ref.install_as(unet_ref, &p, 4.0, 4.0, None, Dtype::Float32)?;
            let eps = unet_ref
                .forward(&noisy, t.unet_time(), &cond, &pooled, &ids)
                .map_err(|e| Exception::custom(e.to_string()))?;
            let diff = eps.subtract(&noise)?;
            Ok(vec![
                reduce_loss(&diff, None, false).map_err(|e| Exception::custom(e.to_string()))?
            ])
        };
        let (val, g_legacy) = keyed_value_and_grad(legacy)(params.clone(), 0).unwrap();
        eval(g_legacy.values()).unwrap();
        assert_eq!(off.total, val[0].item::<f32>());
        assert_eq!(off.diffusion, Some(off.total));
        let bits =
            |a: &Array| -> Vec<u32> { a.as_slice::<f32>().iter().map(|x| x.to_bits()).collect() };
        for (k, v) in &g_legacy {
            assert_eq!(
                bits(v),
                bits(&g_off[k]),
                "{k}: off grads differ from legacy"
            );
        }

        // A diffusion-only step of an enabled path takes exactly the same graph.
        let dcfg = self::cfg();
        let mut unet2 = tiny_unet(3).unwrap();
        let (adapter2, params2) = self::adapter(&mut unet2, &dcfg);
        let mut d = prepared(&cache, 1);
        let (on1, g_on1) = step(
            &h,
            &mut unet2,
            &params2,
            &adapter2,
            &dcfg,
            &cache,
            Some(&mut d),
            1,
            None,
        );
        let mut unet3 = tiny_unet(3).unwrap();
        let (adapter3, params3) = self::adapter(&mut unet3, &dcfg);
        let (none1, g_none1) = step(
            &h, &mut unet3, &params3, &adapter3, &dcfg, &cache, None, 1, None,
        );
        assert_eq!(on1, none1);
        for (k, v) in &g_none1 {
            assert_eq!(bits(v), bits(&g_on1[k]), "{k}");
        }
    }

    /// E7: depth anchoring grows the estimate by the TAESDXL + DA2 footprint (more for Large), and
    /// the guard refuses at a synthetic budget between the U-Net projection and projection + aux on
    /// the dense AND the checkpointed path (host-independent); a checkpointed run without aux
    /// models is not guarded (no fitted checkpointed curve). Mutations: drop `+ extra_gb` ⇒ the aux
    /// case passes ⇒ red; return early for every checkpointed run ⇒ the checkpointed aux case
    /// passes ⇒ red.
    #[test]
    fn memory_estimate_includes_the_aux_models() {
        let mut on = TrainingConfig::default();
        on.depth_anchoring.schedule = schedule();
        let small = perceptual_footprint_gb(&on, 1024, 10);
        on.depth_anchoring.model_size = mlx_gen::gen_core::train::DepthModelSize::Large;
        let large = perceptual_footprint_gb(&on, 1024, 10);
        assert!(small > 0.0 && large > small, "small {small} large {large}");
        assert!(large - small > 1.0, "DA2-Large weights alone are ~1.3 GB");

        let h = hooks();
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        let between = |base: f64| ((base + large / 2.0) / 0.85 * GIB) as usize;
        let edges = [1024u32];
        let dense = h.peak_gb(128.0 * 128.0, true);
        let ckpt = h.peak_gb(0.0, true);
        for (checkpointed, base) in [(false, dense), (true, ckpt)] {
            let budget = between(base);
            assert!(
                preflight_memory_guard_with_budget(
                    &h,
                    &on,
                    &edges,
                    true,
                    0.0,
                    checkpointed,
                    budget
                )
                .is_ok(),
                "checkpointed={checkpointed}"
            );
            assert!(
                preflight_memory_guard_with_budget(
                    &h,
                    &on,
                    &edges,
                    true,
                    large,
                    checkpointed,
                    budget
                )
                .is_err(),
                "checkpointed={checkpointed}"
            );
        }
        // The checkpointed refusal names the enabled aux losses (the shared E7 guard).
        let err =
            preflight_memory_guard_with_budget(&h, &on, &edges, true, large, true, between(ckpt))
                .unwrap_err()
                .to_string();
        assert!(err.contains("[depth]"), "{err}");
        // Checkpointed with no aux models: not guarded, even at a starvation budget.
        assert!(preflight_memory_guard_with_budget(&h, &on, &edges, true, 0.0, true, 1).is_ok());
        assert!(preflight_memory_guard_with_budget(&h, &on, &edges, true, 0.0, false, 1).is_err());
        let roomy = ((dense + large) / 0.85 * GIB) as usize * 2;
        for checkpointed in [false, true] {
            assert!(preflight_memory_guard_with_budget(
                &h,
                &on,
                &edges,
                true,
                large,
                checkpointed,
                roomy
            )
            .is_ok());
        }
    }

    /// E3: SDXL declares depth anchoring.
    #[test]
    fn descriptor_declares_depth_anchoring() {
        assert!(trainer_descriptor().techniques.depth_anchoring);
    }

    /// A missing aux checkpoint is a clear error naming the decoder, before any caching.
    #[test]
    fn missing_aux_weights_are_named() {
        let tmp = tempfile::tempdir().unwrap();
        let mut c = cfg();
        c.perceptual_decoder_dir = Some(tmp.path().join("no-taesdxl"));
        c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
        let err = load_perceptual_path("sdxl", &c)
            .err()
            .expect("must fail")
            .to_string();
        assert!(err.contains("TAESDXL"), "{err}");
    }
}
