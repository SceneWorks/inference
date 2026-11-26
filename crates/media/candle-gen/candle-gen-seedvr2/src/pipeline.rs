//! SeedVR2 image-mode pipeline — candle port of `mlx-gen-seedvr2/src/pipeline.rs` (sc-5157).
//!
//! One-step super-resolution: preprocess the LR image (bicubic upscale to target, optional `softness`
//! pre-blur, [-1,1]) → VAE encode → conditioning latent (encoded latent + ones-mask) → concat fresh
//! noise → DiT (one step) → `latents = noise − DiT_out` → VAE decode → crop → LAB+wavelet color
//! correction → RGB8.
//!
//! **Video mode (sc-5926):** the 5-D temporal pass. `generate_video` sizes a temporal chunk against
//! the device VRAM budget ([`crate::video`]), runs each chunk through the same `T`-aware
//! encode/condition/denoise/decode path, per-frame color-corrects, and cross-fades chunk overlaps to
//! close the causal-VAE seam. It falls back to a per-frame (`T=1`) path under tight memory, and to
//! per-frame **spatial tiling** when even one full-resolution frame exceeds the budget (HD).
//!
//! The negative-prompt conditioning is a precomputed embedding (bundled `data/neg_embed.safetensors`,
//! no runtime text encoder), loaded at construction.

use std::cell::RefCell;

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::driver::{
    run_candle_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use candle_gen::gen_core::{self, imageops, BoundaryState, CancelFlag, Image, Progress, Quant};
use candle_gen::{CandleError, Result as CResult};
use rand::{rngs::StdRng, SeedableRng};

use crate::config::{DitConfig, TIMESTEP};
use crate::dit::Seedvr2Transformer;
use crate::vae::Seedvr2Vae;
use crate::video::{self, Chunk, ChunkPlan, Seedvr2Route};
use crate::weights::SourceWeights;
use crate::{color, convert};

fn core_err(err: CandleError) -> gen_core::Error {
    err.into()
}

fn to_candle_err(err: CandleError) -> candle_gen::candle_core::Error {
    match err {
        CandleError::Candle(inner) => inner,
        other => candle_gen::candle_core::Error::Msg(other.to_string()),
    }
}

/// Prediction seam for [`Seedvr2Driver`], allowing production DiT forwards across all five
/// SeedVR2 routes and weights-free Tier 0 conformance tests to share one [`FamilyDriver`].
pub trait Seedvr2Forward {
    /// Prepare unit noise entering restoration unit `unit_index` (defaults to reusing `current`).
    fn prepare_unit_noise(&self, _unit_index: usize, current: &Tensor) -> CResult<Tensor> {
        Ok(current.clone())
    }

    /// Predict `dit_out` for `noise` at `unit_index` so `Seedvr2Driver` evaluates `noise − dit_out`.
    fn predict(&self, unit_index: usize, noise: &Tensor) -> CResult<Tensor>;
}

impl<F> Seedvr2Forward for F
where
    F: Fn(usize, &Tensor) -> CResult<Tensor>,
{
    fn predict(&self, unit_index: usize, noise: &Tensor) -> CResult<Tensor> {
        self(unit_index, noise)
    }
}

struct UnitForward<P, F> {
    prepare: P,
    predict: F,
}

impl<P, F> Seedvr2Forward for UnitForward<P, F>
where
    P: Fn(usize, &Tensor) -> CResult<Tensor>,
    F: Fn(usize, &Tensor) -> CResult<Tensor>,
{
    fn prepare_unit_noise(&self, unit_index: usize, current: &Tensor) -> CResult<Tensor> {
        (self.prepare)(unit_index, current)
    }

    fn predict(&self, unit_index: usize, noise: &Tensor) -> CResult<Tensor> {
        (self.predict)(unit_index, noise)
    }
}

/// Typed [`FamilyDriver`] executing SeedVR2's 1-step Euler restoration (`x_clean = noise − dit_out`).
pub struct Seedvr2Driver<F> {
    sigmas: Vec<f32>,
    forward: F,
    support: DriverSupport,
}

impl<F> Seedvr2Driver<F> {
    /// Construct a SeedVR2 driver with an explicit restoration schedule (`len == total_steps + 1`).
    pub fn new(sigmas: Vec<f32>, forward: F) -> Self {
        Self {
            sigmas,
            forward,
            support: DriverSupport {
                driver_id: "candle_seedvr2",
                arbitrary_slice: false,
                carry_history: false,
                reset_history: false,
                renoise_from_clean: false,
                geometry_change: false,
                read_only_boundary_estimate: false,
                previous_estimate_capture: false,
                uses_global_rng: false,
                refusal_reasons: DriverRefusalReasons::default(),
            },
        }
    }

    /// Construct a SeedVR2 driver over `unit_count` 1-step restoration units at `TIMESTEP -> 0.0`.
    pub fn for_units(unit_count: usize, forward: F) -> Self {
        let mut sigmas = vec![TIMESTEP as f32; unit_count];
        sigmas.push(0.0);
        Self::new(sigmas, forward)
    }
}

impl<F: Seedvr2Forward> FamilyDriver<Tensor> for Seedvr2Driver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Tensor>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Tensor>,
    ) -> gen_core::Result<BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        if input.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        if self.sigmas.len() <= input.span.end {
            return Err(gen_core::Error::Msg(format!(
                "seedvr2 driver: schedule length {} too short for span {:?}",
                self.sigmas.len(),
                input.span
            )));
        }
        state.enter_slice(&input)?;
        candle_gen::execution::schedule_f64(
            "seedvr2_one_step_euler_v1",
            &[TIMESTEP, 0.0],
            "noise_minus_dit_output",
        );
        let mut current = match input.state {
            BoundaryState::XSigma(latent)
            | BoundaryState::CleanEstimate { latent, .. }
            | BoundaryState::RouteSpecific { state: latent, .. } => latent,
        };
        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: self.sigmas[step],
                next_sigma: self.sigmas[step + 1],
            };
            hooks.observe(&info, &current)?;
            let unit_noise = self
                .forward
                .prepare_unit_noise(step, &current)
                .map_err(core_err)?;
            let dit_out = self.forward.predict(step, &unit_noise).map_err(core_err)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            let denoised = (&unit_noise - &dit_out)
                .map_err(CandleError::from)
                .map_err(core_err)?;
            current = hooks.post_step(&info, denoised)?;
            hooks.materialize(&info, &current)?;
            state.record_outer_step(step);
            hooks.emit_progress(&info);
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
        }
        state.finish_slice(input.span.end);
        Ok(BoundaryState::XSigma(current))
    }
}

/// Clamp an explicit chunk to the live memory plan and record execution policy.
fn resolve_chunk_plan(plan: ChunkPlan, chunk_override: Option<i32>) -> ChunkPlan {
    let selected = video::resolve_chunk_plan(plan, chunk_override);
    candle_gen::execution::policy(
        "seedvr2_budget_route",
        serde_json::json!(format!("{selected:?}")),
    );
    selected
}

/// Post-decode color-correction luminance weight (the reference `apply_color_correction` default).
const LUMINANCE_WEIGHT: f32 = 0.8;

pub struct Seedvr2Pipeline {
    pub vae: Seedvr2Vae,
    pub transformer: Seedvr2Transformer,
    neg_embed: Tensor,
    dtype: DType,
    device: Device,
    /// Resident weight bytes (VAE + DiT at `dtype`) — drives the video memory-budget chunk sizer.
    weights_bytes: usize,
}

/// Estimate resident weight bytes for the video memory budget: the raw `fp16` checkpoint file sizes
/// scaled by the load `dtype` (`BF16` keeps the 2 B/param footprint, `F32` doubles it). File sizes
/// (vs summing per-tensor) match the wan `dit_resident_bytes` convention; the safetensors header
/// overhead is negligible.
fn resident_weight_bytes(files: &[&std::path::Path], dt: DType) -> usize {
    let raw: u64 = files
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum();
    let ratio = match dt {
        DType::F32 => 2.0, // fp16-on-disk → f32 resident
        _ => 1.0,          // bf16/fp16 resident
    };
    (raw as f64 * ratio) as usize
}

/// The bundled precomputed negative-prompt embedding → `(1, Lt, 5120)` at `dt`.
fn load_neg_embed(dt: DType, dev: &Device) -> CResult<Tensor> {
    const BYTES: &[u8] = include_bytes!("../data/neg_embed.safetensors");
    let map = candle_gen::candle_core::safetensors::load_buffer(BYTES, dev)?;
    let emb = map
        .get("embedding")
        .ok_or_else(|| CandleError::Msg("seedvr2 neg-embed: missing `embedding`".into()))?;
    Ok(emb.to_dtype(dt)?.unsqueeze(0)?)
}

/// Deterministic N(0,1) noise of a 5-D shape (CPU `StdRng`/ChaCha, launch-portable per seed).
fn seeded_normal5(
    seed: u64,
    shape: (usize, usize, usize, usize, usize),
    dt: DType,
    dev: &Device,
) -> CResult<Tensor> {
    candle_gen::execution::stream("seedvr2_chunk_or_tile_noise", seed);
    candle_gen::execution::runtime(dev, dt);
    candle_gen::execution::policy(
        "seedvr2_noise_shape",
        serde_json::json!([shape.0, shape.1, shape.2, shape.3, shape.4]),
    );
    let (a, b, c, d, e) = shape;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, a * b * c * d * e);
    Ok(Tensor::from_vec(data, shape, dev)?.to_dtype(dt)?)
}

/// `(1,3,H,W)` in [-1,1] → RGB8 [`Image`].
fn decoded_to_image(decoded: &Tensor) -> Result<Image> {
    let (_b, _c, h, w) = decoded.dims4()?;
    let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let u8s = candle_gen::round_rgb8(&scaled)?.to_device(&Device::Cpu)?;
    let chw = u8s.squeeze(0)?; // (3,H,W)
    let pixels = chw.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Bail with the typed [`CandleError::Canceled`] when the caller's cooperative cancel flag has
/// tripped.
fn check_canceled(cancel: Option<&CancelFlag>) -> CResult<()> {
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(CandleError::Canceled);
    }
    Ok(())
}

impl Seedvr2Pipeline {
    /// Build from already-converted candle-layout VAE + DiT weights + a neg-embed (parity tests).
    pub fn from_parts(
        vae: Seedvr2Vae,
        transformer: Seedvr2Transformer,
        neg_embed: Tensor,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            vae,
            transformer,
            neg_embed,
            dtype,
            device,
            weights_bytes: 0,
        }
    }

    /// Load from a raw `numz/SeedVR2_comfyUI` checkpoint dir: convert in-memory (no Python), cast to
    /// `dt`, attach the bundled neg-embed. `dit_file` selects 3B/7B.
    pub fn load(
        raw_dir: impl AsRef<std::path::Path>,
        dit_file: &str,
        cfg: &DitConfig,
        dt: DType,
        device: &Device,
    ) -> CResult<Self> {
        let dir = raw_dir.as_ref();
        let vae_path = dir.join("ema_vae_fp16.safetensors");
        let dit_path = dir.join(dit_file);
        let weights_bytes = resident_weight_bytes(&[vae_path.as_path(), dit_path.as_path()], dt);
        // Stream each checkpoint one at a time and convert+cast tensor-by-tensor: the raw fp16 map is
        // consumed (moved into `convert_*`) and each raw tensor drops as its `dt` cast is produced, so
        // the raw set and the cast copy never both live for all weights (peak load memory ~1× rather
        // than 2×/3× — sc-9042/F-058). VAE is fully built (and its raw map freed) before the 7B DiT is
        // read, bounding peak further. Loaded values are identical to the prior two-phase convert+cast.
        let vae_w = SourceWeights::open(&vae_path, device, dt, |k| vec![k.to_owned()])?;
        let vae = Seedvr2Vae::from_weights(&vae_w)?;
        drop(vae_w);
        let dit_w = SourceWeights::open(&dit_path, device, dt, convert::dit_targets)?;
        let transformer = Seedvr2Transformer::from_weights(&dit_w, cfg)?;
        let neg_embed = load_neg_embed(dt, device)?;
        let mut p = Self::from_parts(vae, transformer, neg_embed, dt, device.clone());
        p.weights_bytes = weights_bytes;
        Ok(p)
    }

    /// Quantize the DiT Linears to `quant` (`Q4_0`/`Q8_0`) — Linear-only (sc-5927); the VAE stays
    /// dense. Call once after [`Self::load`], before the pipeline is shared. `weights_bytes` is
    /// intentionally **not** reduced — keeping the dense estimate makes the video chunk-size budget
    /// conservative (quant shrinks the weights, not the activations, so the headroom stays safe).
    pub fn quantize(&mut self, quant: Quant) -> CResult<()> {
        self.transformer.quantize(quant)
    }

    /// Resident weight bytes (VAE + DiT at `dtype`) used by the video/image budget planners.
    pub fn weights_bytes(&self) -> usize {
        self.weights_bytes
    }

    /// Build the static condition `[latent, ones-mask]` → `(B, 17, T', h, w)`.
    fn condition(latent: &Tensor) -> Result<Tensor> {
        let (b, _c, t, h, w) = latent.dims5()?;
        let mask = Tensor::ones((b, 1, t, h, w), latent.dtype(), latent.device())?;
        Tensor::cat(&[latent, &mask], 1)
    }

    fn dit_forward(&self, noise: &Tensor, condition: &Tensor) -> CResult<Tensor> {
        let model_input = Tensor::cat(&[noise, condition], 1)?; // (B,33,T',h,w)
        Ok(self
            .transformer
            .forward(&model_input, &self.neg_embed, TIMESTEP)?)
    }

    /// One denoise step driven through [`Seedvr2Driver`] (`[TIMESTEP, 0.0]`): `noise − DiT_out`.
    fn denoise(&self, noise: &Tensor, condition: &Tensor) -> CResult<Tensor> {
        let driver = Seedvr2Driver::for_units(1, |_step, current_noise: &Tensor| {
            self.dit_forward(current_noise, condition)
        });
        let cancel = CancelFlag::default();
        let mut st = DriverRunState::new();
        let mut on_progress = |_| {};
        let mut hooks = StepHooks::new(&mut on_progress);
        let out = run_candle_driver_slices(
            &driver,
            [SliceSpec::full(1)],
            BoundaryState::XSigma(noise.clone()),
            1,
            0,
            &cancel,
            None,
            &mut st,
            &mut hooks,
            |_, out, _| Ok(out),
        )?;
        let BoundaryState::XSigma(latents) = out else {
            return Err(CandleError::Msg(
                "seedvr2 driver returned non-XSigma state".into(),
            ));
        };
        Ok(latents)
    }

    /// Decode latents and crop to `(true_h, true_w)` (first frame) → `(1,3,true_h,true_w)`.
    fn decode_crop(&self, latents: &Tensor, true_h: usize, true_w: usize) -> Result<Tensor> {
        let decoded = self.vae.decode(latents)?; // (1,3,T,H,W)
        decoded
            .narrow(2, 0, 1)?
            .squeeze(2)? // (1,3,H,W)
            .narrow(2, 0, true_h)?
            .narrow(3, 0, true_w)?
            .contiguous()
    }

    /// Full model path (no color correction): preprocessed image + injected noise → decoded crop.
    /// Public for the golden parity harness (the engine-level parity check).
    pub fn run_model(
        &self,
        processed: &Tensor,
        noise: &Tensor,
        true_h: usize,
        true_w: usize,
    ) -> Result<Tensor> {
        let latent = self.vae.encode(processed)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(noise, &cond).map_err(to_candle_err)?;
        self.decode_crop(&latents, true_h, true_w)
    }

    /// End-to-end upscale: LR `image` → `(width, height)` super-resolved RGB8 image.
    ///
    /// Spatial-tiles when a single full-resolution pass would exceed the memory budget (sc-6225) — the
    /// image analog of the video path's HD-tiling fallback (sc-5926). See [`Self::generate_budgeted`].
    ///
    /// `cancel` is the caller's cooperative cancel flag, checked before each spatial tile on the
    /// tiled path (sc-12465/F-013 — a tiled 4096² still is up to 9 full encode→DiT→decode passes,
    /// so a mid-image cancel must not wait for the whole image); the one-pass path is a single unit
    /// and relies on the caller's per-image check (the gen-core per-step cancel contract).
    pub fn generate(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        cancel: Option<&CancelFlag>,
    ) -> CResult<Image> {
        self.generate_with_progress(image, width, height, seed, softness, cancel, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_progress(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        cancel: Option<&CancelFlag>,
        on_step: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Image> {
        self.generate_budgeted_with_progress(
            image,
            width,
            height,
            seed,
            softness,
            video::safe_budget_gib(),
            cancel,
            on_step,
        )
    }

    /// [`Self::generate`] with the safe peak-GB ceiling injected.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_budgeted(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        safe_gib: f64,
        cancel: Option<&CancelFlag>,
    ) -> CResult<Image> {
        self.generate_budgeted_with_progress(
            image, width, height, seed, softness, safe_gib, cancel, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_budgeted_with_progress(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        safe_gib: f64,
        cancel: Option<&CancelFlag>,
        on_step: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Image> {
        let over_budget = matches!(
            video::plan_chunk_size_with(self.weights_bytes, height as i32, width as i32, safe_gib),
            ChunkPlan::OverBudget { .. }
        );
        let over_vae_cap = width.max(height) as i32 > video::VAE_SAFE_DECODE_EDGE_PX;
        candle_gen::execution::policy(
            "seedvr2_still_route",
            serde_json::json!({
                "tiled": over_budget || over_vae_cap, "memory_fallback": over_budget,
                "decoder_cap_fallback": over_vae_cap
            }),
        );
        if over_budget || over_vae_cap {
            let tile = video::plan_spatial_tile_px(self.weights_bytes, safe_gib);
            let overlap = video::SPATIAL_OVERLAP.min(tile / 2);
            return self.generate_tiled_with_params(
                image, width, height, seed, softness, tile, overlap, cancel, on_step,
            );
        }
        self.generate_single_pass_unit(image, width, height, seed, softness, cancel, on_step)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_still_for_route(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        route: &Seedvr2Route,
        cancel: Option<&CancelFlag>,
        on_step: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Image> {
        match route {
            Seedvr2Route::StillSinglePass => self
                .generate_single_pass_unit(image, width, height, seed, softness, cancel, on_step),
            Seedvr2Route::StillSpatialTiled {
                tile_edge_px,
                overlap_px,
                ..
            } => self.generate_tiled_with_params(
                image,
                width,
                height,
                seed,
                softness,
                *tile_edge_px,
                *overlap_px,
                cancel,
                on_step,
            ),
            _ => Err(CandleError::Msg(
                "seedvr2: expected a still route for image generation".into(),
            )),
        }
    }

    fn prepare_frame_unit(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
    ) -> CResult<(Tensor, Tensor, Tensor)> {
        let processed = self.preprocess(image, width, height, softness)?; // (1,3,H,W)
        let latent = self.vae.encode(&processed)?;
        let (_b, _c, lt, lh, lw) = latent.dims5()?;
        let noise = seeded_normal5(seed, (1, 16, lt, lh, lw), self.dtype, &self.device)?;
        let cond = Self::condition(&latent)?;
        Ok((processed, noise, cond))
    }

    fn finish_frame_unit(
        &self,
        latents: &Tensor,
        processed: &Tensor,
        height: usize,
        width: usize,
    ) -> CResult<Image> {
        let decoded = self.decode_crop(latents, height, width)?; // (1,3,H,W)
        let corrected = color::apply_color_correction(
            &decoded.to_dtype(DType::F32)?,
            &processed.to_dtype(DType::F32)?,
            LUMINANCE_WEIGHT,
        )?;
        Ok(decoded_to_image(&corrected)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_driver_units(
        units: usize,
        width: usize,
        height: usize,
        seed: u64,
        device: &Device,
        cancel: Option<&CancelFlag>,
        prepare: impl Fn(usize, &Tensor) -> CResult<Tensor>,
        predict: impl Fn(usize, &Tensor) -> CResult<Tensor>,
        on_post_step: &mut dyn FnMut(&DriverStepInfo, Tensor) -> gen_core::Result<Tensor>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<()> {
        let fallback_cancel = CancelFlag::default();
        let cancel_flag = cancel.unwrap_or(&fallback_cancel);
        let driver = Seedvr2Driver::for_units(units, UnitForward { prepare, predict });
        let mut st = DriverRunState::new();
        let mut hooks = StepHooks::new(on_progress);
        hooks.on_post_step = Some(on_post_step);
        let mut spec = SliceSpec::full(units);
        spec.geometry = Some((width as u32, height as u32));
        let dummy = Tensor::zeros((1,), DType::F32, device)?;
        run_candle_driver_slices(
            &driver,
            [spec],
            BoundaryState::XSigma(dummy),
            units,
            seed,
            cancel_flag,
            None,
            &mut st,
            &mut hooks,
            |_, out, _| Ok(out),
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_single_pass_unit(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        cancel: Option<&CancelFlag>,
        mut on_step: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Image> {
        let active: RefCell<Option<(Tensor, Tensor)>> = RefCell::new(None);
        let result_img: RefCell<Option<Image>> = RefCell::new(None);
        let mut progress = |p: Progress| {
            if let (Some(cb), Progress::Step { current, total }) = (on_step.as_deref_mut(), p) {
                cb(current as usize, total as usize);
            }
        };
        let mut post_step = |_info: &DriverStepInfo, latents: Tensor| -> gen_core::Result<Tensor> {
            let (processed, _) = active
                .borrow_mut()
                .take()
                .ok_or_else(|| gen_core::Error::Msg("seedvr2: missing frame state".into()))?;
            let img = self
                .finish_frame_unit(&latents, &processed, height, width)
                .map_err(core_err)?;
            *result_img.borrow_mut() = Some(img);
            Ok(latents)
        };
        Self::run_driver_units(
            1,
            width,
            height,
            seed,
            &self.device,
            cancel,
            |_unit, _cur| {
                let (processed, noise, cond) =
                    self.prepare_frame_unit(image, width, height, seed, softness)?;
                *active.borrow_mut() = Some((processed, cond));
                Ok(noise)
            },
            |_unit, noise| {
                let (_, cond) = active
                    .borrow()
                    .as_ref()
                    .map(|(p, c)| (p.clone(), c.clone()))
                    .ok_or_else(|| CandleError::Msg("seedvr2: unit not prepared".into()))?;
                self.dit_forward(noise, &cond)
            },
            &mut post_step,
            &mut progress,
        )?;
        Ok(result_img
            .into_inner()
            .expect("one single-pass still image produced"))
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_tiled_with_params(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
        on_step: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Image> {
        let processed = self
            .preprocess(image, width, height, softness)?
            .unsqueeze(2)?; // (1,3,1,H,W)
        let decoded =
            self.run_frame_tiled_with_progress(&processed, seed, tile, overlap, cancel, on_step)?;
        Ok(self
            .frames_from_decoded(&decoded, &processed, 1)?
            .into_iter()
            .next()
            .expect("one tiled frame"))
    }

    /// LR `Image` → `(1,3,height,width)` in [-1,1] at the model dtype. Bicubic resize to target;
    /// optional `softness` pre-blur via a smaller round-trip.
    pub fn preprocess(
        &self,
        image: &Image,
        width: usize,
        height: usize,
        softness: f32,
    ) -> CResult<Tensor> {
        candle_gen::execution::policy(
            "seedvr2_preprocess",
            serde_json::json!({
                "softness_bits": softness.clamp(0.0, 1.0).to_bits(),
                "luminance_weight_bits": LUMINANCE_WEIGHT.to_bits(), "resize": "bicubic-u8-v1"
            }),
        );
        let (ih, iw) = (image.height as usize, image.width as usize);
        let resized: Vec<f32> = if softness > 0.0 {
            let factor = 1.0 + softness.clamp(0.0, 1.0) * 7.0;
            let dw = ((width as f32 / factor) as usize).max(2);
            let dh = ((height as f32 / factor) as usize).max(2);
            let down = imageops::resize_bicubic_u8(&image.pixels, ih, iw, dh, dw)?;
            let down_u8: Vec<u8> = down
                .iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect();
            imageops::resize_bicubic_u8(&down_u8, dh, dw, height, width)?
        } else {
            imageops::resize_bicubic_u8(&image.pixels, ih, iw, height, width)?
        };
        // HWC [0,255] → [-1,1] → (1,3,H,W)
        let arr = Tensor::from_vec(resized, (height, width, 3), &self.device)?;
        let arr = (arr.affine(2.0 / 255.0, -1.0))?;
        Ok(arr
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?
            .contiguous()?)
    }

    // -----------------------------------------------------------------------
    // video mode (sc-5926): multi-frame 5-D pass + temporal chunking/overlap + HD tiling
    // -----------------------------------------------------------------------

    /// Decode latents and crop spatially to `(true_h, true_w)`, **keeping all `T` frames** →
    /// `(B,3,T,true_h,true_w)`. The 5-D analog of `Self::decode_crop` (which keeps only frame 0).
    pub fn decode_crop_5d(&self, latents: &Tensor, true_h: usize, true_w: usize) -> Result<Tensor> {
        let decoded = self.vae.decode(latents)?; // (B,3,T,H,W)
        decoded
            .narrow(3, 0, true_h)?
            .narrow(4, 0, true_w)?
            .contiguous()
    }

    /// Multi-frame model path (no color correction): a preprocessed clip `(1,3,T,H,W)` + injected
    /// noise `(1,16,T',h,w)` → decoded crop `(1,3,T,true_h,true_w)`. The video analog of
    /// [`Self::run_model`]; `encode`/`condition`/`denoise` are already `T`-aware. Public for the harness.
    pub fn run_model_5d(
        &self,
        processed: &Tensor,
        noise: &Tensor,
        true_h: usize,
        true_w: usize,
    ) -> Result<Tensor> {
        let latent = self.vae.encode(processed)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(noise, &cond).map_err(to_candle_err)?;
        self.decode_crop_5d(&latents, true_h, true_w)
    }

    /// Per-frame color-correct a decoded clip `(1,3,T,Hc,Wc)` against its preprocessed `style`
    /// `(1,3,Ts,Hc,Wc)` → `count` RGB8 frames. Frame `t` matches style frame `min(t, Ts-1)`.
    fn frames_from_decoded(
        &self,
        decoded: &Tensor,
        style: &Tensor,
        count: usize,
    ) -> CResult<Vec<Image>> {
        let style_t = style.dim(2)?;
        let mut out = Vec::with_capacity(count);
        for t in 0..count {
            let d = decoded.narrow(2, t, 1)?.squeeze(2)?; // (1,3,Hc,Wc)
            let s = style.narrow(2, t.min(style_t - 1), 1)?.squeeze(2)?;
            let corrected = color::apply_color_correction(
                &d.to_dtype(DType::F32)?,
                &s.to_dtype(DType::F32)?,
                LUMINANCE_WEIGHT,
            )?;
            out.push(decoded_to_image(&corrected)?);
        }
        Ok(out)
    }

    /// Preprocess one temporal chunk: pixel-frames `[start, start+len)` of `frames`, clamping past the
    /// end to the last real frame (last-frame padding) → `(1,3,len,H,W)` in `[-1,1]`.
    fn preprocess_chunk(
        &self,
        frames: &[Image],
        start: i32,
        len: i32,
        width: usize,
        height: usize,
        softness: f32,
    ) -> CResult<Tensor> {
        let n = frames.len() as i32;
        let mut per = Vec::with_capacity(len as usize);
        for j in 0..len {
            let idx = (start + j).clamp(0, n - 1) as usize;
            per.push(
                self.preprocess(&frames[idx], width, height, softness)?
                    .unsqueeze(2)?, // (1,3,1,H,W)
            );
        }
        let refs: Vec<&Tensor> = per.iter().collect();
        Ok(Tensor::cat(&refs, 2)?)
    }

    /// End-to-end **video** upscale: a sequence of LR `frames` → upscaled `(width, height)` RGB8
    /// frames (same count). Sizes the temporal chunk against the memory budget (or `chunk_override`),
    /// processes each chunk through the 5-D model path with one-step Euler, per-frame color-corrects,
    /// and cross-fades chunk overlaps to close the causal-VAE seam ([`crate::video`]). Falls back to
    /// the per-frame (`T=1`) path under tight memory, and to per-frame **spatial tiling** when even one
    /// full-resolution frame exceeds the budget (HD), so peak stays bounded at any resolution.
    ///
    /// `cancel` is the caller's cooperative cancel flag (the gen-core video per-step cancel contract,
    /// sc-8972/sc-11227; mirrors `Sam3VideoModel::propagate`): checked before each (seconds-to-minutes)
    /// chunk/frame — and before each spatial tile within a tiled frame (sc-12465) — so a mid-clip
    /// cancel is honored promptly instead of after the whole upscale, surfacing
    /// the typed [`CandleError::Canceled`]. `progress` is invoked `(units_done, units_total)` after each
    /// completed chunk/frame (real per-unit progress, 1-based `units_done`), replacing the old
    /// fixed-placeholder single step; `units_total` is the number of chunks (chunked path) or frames
    /// (per-frame / spatial-tiling paths).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_video(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        chunk_override: Option<i32>,
        cancel: Option<&CancelFlag>,
        progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Vec<Image>> {
        candle_gen::execution::policy(
            "seedvr2_video_processing",
            serde_json::json!({
                "softness_bits": softness.to_bits(), "luminance_weight_bits": LUMINANCE_WEIGHT.to_bits(),
                "frame_count": frames.len(), "size": [width, height]
            }),
        );
        let n = frames.len() as i32;
        if n == 0 {
            return Ok(Vec::new());
        }
        if width.max(height) as i32 > video::VAE_SAFE_DECODE_EDGE_PX {
            let tile = video::plan_spatial_tile_px(self.weights_bytes, video::safe_budget_gib());
            let overlap = video::SPATIAL_OVERLAP.min(tile / 2);
            return self.generate_video_tiled_with_params(
                frames, width, height, seed, softness, tile, overlap, cancel, progress,
            );
        }
        let selected = resolve_chunk_plan(
            video::plan_chunk_size(self.weights_bytes, height as i32, width as i32),
            chunk_override,
        );
        let chunk = match selected {
            ChunkPlan::Chunked(c) => c,
            ChunkPlan::PerFrame => {
                return self.generate_video_per_frame(
                    frames, width, height, seed, softness, cancel, progress,
                )
            }
            ChunkPlan::OverBudget { .. } => {
                let tile =
                    video::plan_spatial_tile_px(self.weights_bytes, video::safe_budget_gib());
                let overlap = video::SPATIAL_OVERLAP.min(tile / 2);
                return self.generate_video_tiled_with_params(
                    frames, width, height, seed, softness, tile, overlap, cancel, progress,
                );
            }
        };

        let plan = video::plan_chunks(n, chunk, video::DEFAULT_OVERLAP);
        self.generate_video_temporal_chunked(
            frames,
            width,
            height,
            seed,
            softness,
            chunk,
            video::DEFAULT_OVERLAP,
            &plan,
            cancel,
            progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_video_for_route(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        route: &Seedvr2Route,
        cancel: Option<&CancelFlag>,
        progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Vec<Image>> {
        candle_gen::execution::policy(
            "seedvr2_video_processing",
            serde_json::json!({
                "softness_bits": softness.to_bits(), "luminance_weight_bits": LUMINANCE_WEIGHT.to_bits(),
                "frame_count": frames.len(), "size": [width, height]
            }),
        );
        match route {
            Seedvr2Route::VideoTemporalChunked {
                chunk_frames,
                overlap_frames,
                chunks,
            } => {
                resolve_chunk_plan(ChunkPlan::Chunked(*chunk_frames), None);
                self.generate_video_temporal_chunked(
                    frames,
                    width,
                    height,
                    seed,
                    softness,
                    *chunk_frames,
                    *overlap_frames,
                    chunks,
                    cancel,
                    progress,
                )
            }
            Seedvr2Route::VideoPerFrame { .. } => {
                resolve_chunk_plan(ChunkPlan::PerFrame, None);
                self.generate_video_per_frame(
                    frames, width, height, seed, softness, cancel, progress,
                )
            }
            Seedvr2Route::VideoSpatialTiled {
                tile_edge_px,
                overlap_px,
                ..
            } => self.generate_video_tiled_with_params(
                frames,
                width,
                height,
                seed,
                softness,
                *tile_edge_px,
                *overlap_px,
                cancel,
                progress,
            ),
            _ => Err(CandleError::Msg(
                "seedvr2: expected a video route for video clip generation".into(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_video_temporal_chunked(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        chunk: i32,
        overlap: i32,
        plan: &[Chunk],
        cancel: Option<&CancelFlag>,
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Vec<Image>> {
        let n = frames.len() as i32;
        if n == 0 || plan.is_empty() {
            return Ok(Vec::new());
        }
        candle_gen::execution::policy(
            "seedvr2_temporal_chunks",
            serde_json::json!({
                "chunk": chunk, "overlap": overlap,
                "units": plan.iter().map(|c| (c.start, c.len)).collect::<Vec<_>>()
            }),
        );
        let total = plan.len();
        let active: RefCell<Option<(Tensor, Tensor, Chunk)>> = RefCell::new(None);
        let out: RefCell<Vec<Image>> = RefCell::new(Vec::with_capacity(n.max(0) as usize));
        let mut on_progress = |p: Progress| {
            if let (Some(cb), Progress::Step { current, total }) = (progress.as_deref_mut(), p) {
                cb(current as usize, total as usize);
            }
        };
        let mut post_step = |_info: &DriverStepInfo, latents: Tensor| -> gen_core::Result<Tensor> {
            let (clip, _, c) = active
                .borrow_mut()
                .take()
                .ok_or_else(|| gen_core::Error::Msg("seedvr2: missing chunk state".into()))?;
            let decoded = self
                .decode_crop_5d(&latents, height, width)
                .map_err(CandleError::from)
                .map_err(core_err)?;
            let real_count = video::real_frame_count(c.start, c.len, n) as usize;
            let chunk_imgs = self
                .frames_from_decoded(&decoded, &clip, real_count)
                .map_err(core_err)?;
            video::assemble_overlap_chunk(&mut out.borrow_mut(), c.start, chunk_imgs, n, overlap);
            Ok(latents)
        };
        Self::run_driver_units(
            total,
            width,
            height,
            seed,
            &self.device,
            cancel,
            |chunk_idx, _cur| {
                let c = plan[chunk_idx];
                let clip =
                    self.preprocess_chunk(frames, c.start, c.len, width, height, softness)?;
                let latent = self.vae.encode(&clip)?;
                let (_b, _c, lt, lh, lw) = latent.dims5()?;
                let noise = seeded_normal5(seed, (1, 16, lt, lh, lw), self.dtype, &self.device)?;
                let cond = Self::condition(&latent)?;
                *active.borrow_mut() = Some((clip, cond, c));
                Ok(noise)
            },
            |_chunk_idx, noise| {
                let (_, cond, _) = active
                    .borrow()
                    .as_ref()
                    .map(|(clip, cond, c)| (clip.clone(), cond.clone(), *c))
                    .ok_or_else(|| CandleError::Msg("seedvr2: chunk unit not prepared".into()))?;
                self.dit_forward(noise, &cond)
            },
            &mut post_step,
            &mut on_progress,
        )?;
        Ok(out.into_inner())
    }

    /// Per-frame (`T=1`) video fallback driven through [`Seedvr2Driver`] + [`run_candle_driver_slices`].
    #[allow(clippy::too_many_arguments)]
    fn generate_video_per_frame(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        cancel: Option<&CancelFlag>,
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Vec<Image>> {
        candle_gen::execution::policy("seedvr2_video_route", serde_json::json!("per_frame"));
        let total = frames.len();
        if total == 0 {
            return Ok(Vec::new());
        }
        let active: RefCell<Option<(Tensor, Tensor)>> = RefCell::new(None);
        let out: RefCell<Vec<Image>> = RefCell::new(Vec::with_capacity(total));
        let mut on_progress = |p: Progress| {
            if let (Some(cb), Progress::Step { current, total }) = (progress.as_deref_mut(), p) {
                cb(current as usize, total as usize);
            }
        };
        let mut post_step = |_info: &DriverStepInfo, latents: Tensor| -> gen_core::Result<Tensor> {
            let (processed, _) = active
                .borrow_mut()
                .take()
                .ok_or_else(|| gen_core::Error::Msg("seedvr2: missing frame state".into()))?;
            let img = self
                .finish_frame_unit(&latents, &processed, height, width)
                .map_err(core_err)?;
            out.borrow_mut().push(img);
            Ok(latents)
        };
        Self::run_driver_units(
            total,
            width,
            height,
            seed,
            &self.device,
            cancel,
            |frame_idx, _cur| {
                let (processed, noise, cond) =
                    self.prepare_frame_unit(&frames[frame_idx], width, height, seed, softness)?;
                *active.borrow_mut() = Some((processed, cond));
                Ok(noise)
            },
            |_frame_idx, noise| {
                let (_, cond) = active
                    .borrow()
                    .as_ref()
                    .map(|(p, c)| (p.clone(), c.clone()))
                    .ok_or_else(|| {
                        CandleError::Msg("seedvr2: per-frame unit not prepared".into())
                    })?;
                self.dit_forward(noise, &cond)
            },
            &mut post_step,
            &mut on_progress,
        )?;
        Ok(out.into_inner())
    }

    /// HD spatial-tiling video path (sc-5201): each frame is upscaled per-frame (`T=1`) with its
    /// spatial tiles driven through [`Seedvr2Driver`].
    #[allow(clippy::too_many_arguments)]
    fn generate_video_tiled_with_params(
        &self,
        frames: &[Image],
        width: usize,
        height: usize,
        seed: u64,
        softness: f32,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Vec<Image>> {
        let total = frames.len();
        let mut out = Vec::with_capacity(total);
        for (i, f) in frames.iter().enumerate() {
            check_canceled(cancel)?;
            let processed = self.preprocess(f, width, height, softness)?.unsqueeze(2)?; // (1,3,1,H,W)
            let decoded = self.run_frame_tiled(&processed, seed, tile, overlap, cancel)?;
            let imgs = self.frames_from_decoded(&decoded, &processed, 1)?;
            out.push(imgs.into_iter().next().expect("one frame"));
            if let Some(cb) = progress.as_deref_mut() {
                cb(i + 1, total);
            }
            check_canceled(cancel)?;
        }
        Ok(out)
    }

    /// Upscale one preprocessed frame `(1,3,1,H,W)` by spatial tiling through [`Seedvr2Driver`].
    pub fn run_frame_tiled(
        &self,
        processed: &Tensor,
        seed: u64,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
    ) -> CResult<Tensor> {
        self.run_frame_tiled_with_progress(processed, seed, tile, overlap, cancel, None)
    }

    /// [`Self::run_frame_tiled`] with a per-tile progress hook driven through [`Seedvr2Driver`] +
    /// [`run_candle_driver_slices`].
    #[allow(clippy::too_many_arguments)]
    pub fn run_frame_tiled_with_progress(
        &self,
        processed: &Tensor,
        seed: u64,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
        mut on_step: Option<&mut dyn FnMut(usize, usize)>,
    ) -> CResult<Tensor> {
        let (_b, _c, _t, height, width) = processed.dims5()?; // (1,3,1,H,W)
        let (h_i, w_i) = (height as i32, width as i32);
        let plan = video::plan_spatial_tiles(h_i, w_i, tile, overlap);
        candle_gen::execution::policy(
            "seedvr2_spatial_tiles",
            serde_json::json!({
                "tile": tile, "overlap": overlap, "height": height, "width": width,
                "plan": format!("{plan:?}")
            }),
        );
        let n_tiles = plan.len();
        let active: RefCell<Option<(video::SpatialTile, usize, usize, Tensor)>> =
            RefCell::new(None);
        let acc: RefCell<Option<Tensor>> = RefCell::new(None);
        let wsum: RefCell<Option<Tensor>> = RefCell::new(None);
        let mut on_progress = |p: Progress| {
            if let (Some(cb), Progress::Step { current, total }) = (on_step.as_deref_mut(), p) {
                cb(current as usize, total as usize);
            }
        };
        let mut post_step = |_info: &DriverStepInfo, latents: Tensor| -> gen_core::Result<Tensor> {
            let (t, th, tw, _) = active
                .borrow_mut()
                .take()
                .ok_or_else(|| gen_core::Error::Msg("seedvr2: missing tile state".into()))?;
            let (y0, x0) = (t.y0 as usize, t.x0 as usize);
            let decoded = self
                .decode_crop_5d(&latents, th, tw)
                .map_err(CandleError::from)
                .map_err(core_err)?;
            let wvec = video::feather_weight(
                th as i32,
                tw as i32,
                t.y0 > 0,
                t.y1 < h_i,
                t.x0 > 0,
                t.x1 < w_i,
                overlap,
            );
            let weight = Tensor::from_vec(wvec, (1, 1, 1, th, tw), &self.device)
                .map_err(CandleError::from)
                .map_err(core_err)?
                .to_dtype(self.dtype)
                .map_err(CandleError::from)
                .map_err(core_err)?;
            let (pad_r, pad_b) = (width - x0 - tw, height - y0 - th);
            let wdec = decoded
                .broadcast_mul(&weight)
                .and_then(|t| t.pad_with_zeros(3, y0, pad_b))
                .and_then(|t| t.pad_with_zeros(4, x0, pad_r))
                .map_err(CandleError::from)
                .map_err(core_err)?;
            let wpad = weight
                .pad_with_zeros(3, y0, pad_b)
                .and_then(|t| t.pad_with_zeros(4, x0, pad_r))
                .map_err(CandleError::from)
                .map_err(core_err)?;
            let next_acc = match acc.borrow_mut().take() {
                Some(a) => a.add(&wdec).map_err(CandleError::from).map_err(core_err)?,
                None => wdec,
            };
            *acc.borrow_mut() = Some(next_acc);
            let next_wsum = match wsum.borrow_mut().take() {
                Some(a) => a.add(&wpad).map_err(CandleError::from).map_err(core_err)?,
                None => wpad,
            };
            *wsum.borrow_mut() = Some(next_wsum);
            Ok(decoded)
        };
        Self::run_driver_units(
            n_tiles,
            width,
            height,
            seed,
            &self.device,
            cancel,
            |tile_idx, _cur| {
                let t = plan[tile_idx];
                let (th, tw) = ((t.y1 - t.y0) as usize, (t.x1 - t.x0) as usize);
                let (y0, x0) = (t.y0 as usize, t.x0 as usize);
                let tile_clip = processed.narrow(3, y0, th)?.narrow(4, x0, tw)?;
                let latent = self.vae.encode(&tile_clip)?;
                let (_b, _c, lt, lh, lw) = latent.dims5()?;
                let noise = seeded_normal5(seed, (1, 16, lt, lh, lw), self.dtype, &self.device)?;
                let cond = Self::condition(&latent)?;
                *active.borrow_mut() = Some((t, th, tw, cond));
                Ok(noise)
            },
            |_tile_idx, noise| {
                let (_, _, _, cond) = active
                    .borrow()
                    .as_ref()
                    .map(|(t, th, tw, cond)| (*t, *th, *tw, cond.clone()))
                    .ok_or_else(|| CandleError::Msg("seedvr2: tile unit not prepared".into()))?;
                self.dit_forward(noise, &cond)
            },
            &mut post_step,
            &mut on_progress,
        )?;
        Ok(acc
            .into_inner()
            .expect("≥1 tile")
            .broadcast_div(&wsum.into_inner().expect("≥1 tile"))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_canceled_surfaces_typed_canceled_only_on_trip() {
        assert!(check_canceled(None).is_ok(), "no flag → no-op");
        let flag = CancelFlag::new();
        assert!(check_canceled(Some(&flag)).is_ok(), "untripped → no-op");
        flag.cancel();
        assert!(
            matches!(check_canceled(Some(&flag)), Err(CandleError::Canceled)),
            "tripped → typed Canceled"
        );
    }

    /// sc-12465 / F-013: verify the [`Seedvr2Driver`] spatial-tile unit loop checks cancellation at
    /// every tile boundary (no tiles run after cancel is tripped, pre-cancel runs 0 tiles, untripped
    /// runs all 9 tiles of a 4096² plan).
    #[test]
    fn spatial_tile_loop_cancels_between_tiles() {
        let plan = video::plan_spatial_tiles(
            4096,
            4096,
            video::VAE_SAFE_DECODE_EDGE_PX,
            video::SPATIAL_OVERLAP,
        );
        assert_eq!(plan.len(), 9, "4096² @ tile 1536 must be a 3×3 plan");
        let n_tiles = plan.len();

        let flag = CancelFlag::new();
        let tiles_done = std::cell::Cell::new(0usize);
        let driver = Seedvr2Driver::for_units(n_tiles, |_tile, x: &Tensor| {
            tiles_done.set(tiles_done.get() + 1);
            Ok((x * 0.0)?)
        });
        let mut st = DriverRunState::new();
        let mut on_p = |_| {};
        let mut hooks = StepHooks::new(&mut on_p);
        let init = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        let result = run_candle_driver_slices(
            &driver,
            [SliceSpec::full(n_tiles)],
            BoundaryState::XSigma(init),
            n_tiles,
            0,
            &flag,
            None,
            &mut st,
            &mut hooks,
            |_, out, _| Ok(out),
        );
        assert!(result.is_ok(), "untripped flag must not abort the loop");
        assert_eq!(tiles_done.get(), n_tiles);

        for trip_after in [0usize, 1, 3] {
            let flag = CancelFlag::new();
            if trip_after == 0 {
                flag.cancel();
            }
            let tiles_done = std::cell::Cell::new(0usize);
            let driver = Seedvr2Driver::for_units(n_tiles, |_tile, x: &Tensor| {
                tiles_done.set(tiles_done.get() + 1);
                if tiles_done.get() == trip_after {
                    flag.cancel();
                }
                Ok((x * 0.0)?)
            });
            let mut st = DriverRunState::new();
            let mut on_p = |_| {};
            let mut hooks = StepHooks::new(&mut on_p);
            let init = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
            let result = run_candle_driver_slices(
                &driver,
                [SliceSpec::full(n_tiles)],
                BoundaryState::XSigma(init),
                n_tiles,
                0,
                &flag,
                None,
                &mut st,
                &mut hooks,
                |_, out, _| Ok(out),
            );
            assert!(
                matches!(result, Err(CandleError::Canceled)),
                "tripped flag must surface the typed Canceled (trip_after={trip_after})"
            );
            assert_eq!(
                tiles_done.get(),
                trip_after,
                "no tile may run after the flag trips (trip_after={trip_after})"
            );
        }
    }

    /// The per-unit progress contract (sc-11227): the loops report `(done, total)` with `done`
    /// running 1..=total (monotonic, 1-based, ending exactly at `total`) — real per-chunk/per-frame
    /// progress, NOT the old fixed `Step { current: 1, total: 1 }` placeholder that jumped to "done".
    /// This asserts the callback-driving arithmetic the three loops share (`cb(i + 1, total)`).
    #[test]
    fn progress_callback_reports_monotonic_one_based_units() {
        for total in [1usize, 4, 20] {
            let mut seen: Vec<(usize, usize)> = Vec::new();
            // Mirror the loops' `if let Some(cb) = progress.as_deref_mut() { cb(i + 1, total) }`.
            let mut cb = |done: usize, tot: usize| seen.push((done, tot));
            let mut progress: Option<&mut dyn FnMut(usize, usize)> = Some(&mut cb);
            for i in 0..total {
                if let Some(cb) = progress.as_deref_mut() {
                    cb(i + 1, total);
                }
            }
            assert_eq!(seen.len(), total, "one progress event per completed unit");
            assert_eq!(seen.first().copied(), Some((1, total)), "1-based start");
            assert_eq!(seen.last().copied(), Some((total, total)), "ends at total");
            assert!(
                seen.windows(2).all(|w| w[1].0 > w[0].0),
                "strictly increasing (monotonic real progress, not a fixed placeholder): {seen:?}"
            );
        }
    }
}

#[cfg(test)]
mod execution_identity_tests {
    use super::*;

    fn report(
        plan: ChunkPlan,
        requested: Option<i32>,
    ) -> candle_gen::gen_core::generator::ExecutionIdentity {
        let attempt = candle_gen::execution::Attempt::begin();
        let _selected = resolve_chunk_plan(plan, requested);
        let _noise = seeded_normal5(5, (1, 1, 1, 1, 1), DType::F32, &Device::Cpu).unwrap();
        candle_gen::execution::schedule_f64(
            "seedvr2_one_step_euler_v1",
            &[TIMESTEP, 0.0],
            "noise_minus_dit_output",
        );
        let output = candle_gen::gen_core::GenerationOutput::Video {
            frames: vec![Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            }],
            fps: 24,
            audio: None,
        };
        attempt.finish_image(&crate::descriptor(), &output)
    }

    #[test]
    fn report_binds_effective_budget_fallback_and_clamped_chunk() {
        let safe = report(ChunkPlan::Chunked(8), None);
        assert!(safe.resolved().is_some());
        assert_eq!(safe, report(ChunkPlan::Chunked(8), Some(128)));
        assert_ne!(safe, report(ChunkPlan::Chunked(16), None));
        let per_frame = report(ChunkPlan::PerFrame, Some(128));
        assert_ne!(safe, per_frame);
        assert_eq!(per_frame, report(ChunkPlan::PerFrame, None));
        assert_ne!(
            per_frame,
            report(
                ChunkPlan::OverBudget {
                    needed_gib: 20.0,
                    safe_gib: 2.0
                },
                Some(128)
            )
        );
    }

    #[test]
    fn repeated_tile_streams_keep_the_active_still_output_index() {
        let attempt = candle_gen::execution::Attempt::begin();
        for output in 0..2 {
            candle_gen::execution::draw("image_initial_noise", 9 + output as u64, output);
            for _tile in 0..2 {
                let _noise =
                    seeded_normal5(9 + output as u64, (1, 1, 1, 1, 1), DType::F32, &Device::Cpu)
                        .unwrap();
            }
        }
        candle_gen::execution::schedule_f64(
            "seedvr2_one_step_euler_v1",
            &[TIMESTEP, 0.0],
            "noise_minus_dit_output",
        );
        let output = candle_gen::gen_core::GenerationOutput::Images(vec![
            Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            },
            Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            },
        ]);
        let identity = attempt.finish_image(&crate::descriptor(), &output);
        let resolved = identity.resolved().unwrap();
        resolved
            .validate(crate::descriptor().id, "candle", 2)
            .unwrap();
        let candle_gen::gen_core::generator::ExecutionRandomness::Draws { draws, .. } =
            &resolved.randomness
        else {
            panic!("draws")
        };
        assert_eq!(
            draws.iter().map(|d| d.output_index).collect::<Vec<_>>(),
            [0, 0, 0, 1, 1, 1]
        );
        assert_eq!(
            draws.iter().map(|d| d.seed.unwrap()).collect::<Vec<_>>(),
            [9, 9, 9, 10, 10, 10]
        );
    }
}

#[cfg(test)]
mod driver_conformance_tests {
    use super::*;
    use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

    fn make_constant(val: f32) -> Tensor {
        Tensor::from_vec(vec![val; 16 * 2 * 2], (1, 16, 1, 2, 2), &Device::Cpu).unwrap()
    }

    #[test]
    fn seedvr2_driver_passes_tier0_conformance_without_weights() {
        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 7,
            base_geometry: (512, 512),
            alt_geometry: (1024, 1024),
            make_driver: || {
                Box::new(Seedvr2Driver::for_units(4, |_step, noise: &Tensor| {
                    Ok((noise * 0.25)?)
                }))
            },
            make_initial: || make_constant(1.0),
            latents_equal: |a, b| {
                let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                va.len() == vb.len() && va.iter().zip(vb).all(|(x, y)| x.to_bits() == y.to_bits())
            },
        })
        .expect("Candle Seedvr2Driver passes Tier 0 conformance");
        assert_eq!(report.driver_id, "candle_seedvr2");
        assert_eq!(report.total_steps, 4);
        assert!(report.verified_cancellation);
    }

    #[test]
    fn seedvr2_driver_computes_one_step_euler_restoration_and_unit_cancellation() {
        let driver = Seedvr2Driver::for_units(1, |_unit, _noise: &Tensor| Ok(make_constant(0.375)));
        let cancel = CancelFlag::default();
        let mut st = DriverRunState::new();
        let mut on_progress = |_| {};
        let mut hooks = StepHooks::new(&mut on_progress);
        let out = run_candle_driver_slices(
            &driver,
            [SliceSpec::full(1)],
            BoundaryState::XSigma(make_constant(1.0)),
            1,
            5,
            &cancel,
            None,
            &mut st,
            &mut hooks,
            |_, out, _| Ok(out),
        )
        .unwrap();
        let BoundaryState::XSigma(tensor) = out else {
            panic!("expected XSigma");
        };
        let vals = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals.iter().all(|&v| (v - 0.625).abs() < 1e-6));

        let c1 = CancelFlag::default();
        let c1_hook = c1.clone();
        let mut st1 = DriverRunState::new();
        let mut ticks = 0usize;
        let mut on_p1 = |_: Progress| {
            ticks += 1;
            c1_hook.cancel();
        };
        let mut hooks1 = StepHooks::new(&mut on_p1);
        let canceled = run_candle_driver_slices(
            &driver,
            [SliceSpec::full(1)],
            BoundaryState::XSigma(make_constant(1.0)),
            1,
            5,
            &c1,
            None,
            &mut st1,
            &mut hooks1,
            |_, out, _| Ok(out),
        );
        assert!(matches!(canceled, Err(CandleError::Canceled)));
        assert_eq!(ticks, 1);
    }
}
