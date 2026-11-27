//! SeedVR2 image-mode pipeline (sc-4813).
//!
//! Ties the VAE + DiT into the one-step super-resolution path of the mflux reference
//! `SeedVR2.generate_image`: preprocess the LR image (PIL-bicubic upscale to target, optional
//! `softness` pre-blur, [-1,1]) → VAE encode → conditioning latent (encoded latent + ones-mask) →
//! concat fresh noise → DiT (one step) → 1-step Euler (`latents = noise − DiT_out`) → VAE decode →
//! crop → LAB+wavelet color correction ([`crate::color`]) → RGB8.
//!
//! The negative-prompt conditioning is a precomputed embedding (`pos_emb.safetensors`, no runtime
//! text encoder), bundled in the crate (`data/neg_embed.safetensors`) and loaded at construction.

use std::cell::RefCell;

use mlx_gen::driver::{
    run_mlx_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use mlx_gen::gen_core::{self, BoundaryState};
use mlx_gen::image::{decoded_to_image, resize_bicubic_u8};
use mlx_gen::vae_tiling::accumulate_along_axis;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Image, Progress, Result};
use mlx_rs::ops::{concatenate_axis, divide, multiply, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use crate::config::DitConfig;
use crate::dit::Seedvr2Transformer;
use crate::vae::Seedvr2Vae;
use crate::video::{self, Chunk, ChunkPlan, Seedvr2Route};
use crate::{color, convert};

/// The 1-step Euler timestep (= `num_train_steps`, which the SeedVR2 scheduler defaults to 1000).
pub const TIMESTEP: f32 = 1000.0;
/// Post-decode color-correction luminance weight (the reference `apply_color_correction` default).
const LUMINANCE_WEIGHT: f32 = 0.8;

fn core_err(err: Error) -> gen_core::Error {
    err.into()
}

/// Forward prediction seam for [`Seedvr2Driver`], allowing production DiT forwards across all
/// five SeedVR2 routes and weights-free Tier 0 conformance tests to share one [`FamilyDriver`].
pub trait Seedvr2Forward {
    /// Prepare unit noise entering restoration unit `unit_index` (defaults to reusing `current`).
    fn prepare_unit_noise(&self, _unit_index: usize, current: &Array) -> Result<Array> {
        Ok(current.clone())
    }

    /// Predict `dit_out` for `noise` at `unit_index` so `Seedvr2Driver` evaluates `noise − dit_out`.
    fn predict(&self, unit_index: usize, noise: &Array) -> Result<Array>;
}

impl<F> Seedvr2Forward for F
where
    F: Fn(usize, &Array) -> Result<Array>,
{
    fn predict(&self, unit_index: usize, noise: &Array) -> Result<Array> {
        self(unit_index, noise)
    }
}

struct UnitForward<P, F> {
    prepare: P,
    predict: F,
}

impl<P, F> Seedvr2Forward for UnitForward<P, F>
where
    P: Fn(usize, &Array) -> Result<Array>,
    F: Fn(usize, &Array) -> Result<Array>,
{
    fn prepare_unit_noise(&self, unit_index: usize, current: &Array) -> Result<Array> {
        (self.prepare)(unit_index, current)
    }

    fn predict(&self, unit_index: usize, noise: &Array) -> Result<Array> {
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
                driver_id: "mlx_seedvr2",
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
        let mut sigmas = vec![TIMESTEP; unit_count];
        sigmas.push(0.0);
        Self::new(sigmas, forward)
    }
}

impl<F: Seedvr2Forward> FamilyDriver<Array> for Seedvr2Driver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> gen_core::Result<BoundaryState<Array>> {
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
            let denoised = subtract(&unit_noise, &dit_out)
                .map_err(Error::from)
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

pub struct Seedvr2Pipeline {
    pub vae: Seedvr2Vae,
    pub transformer: Seedvr2Transformer,
    neg_embed: Option<Array>,
    dtype: Dtype,
    quant_bits: Option<i32>,
    /// Resident weight bytes (VAE + DiT at `dtype`) — drives the video memory-budget chunk sizer.
    weights_bytes: usize,
}

/// Cast the floating checkpoint tensors while preserving source-native operators and their scales.
fn cast_weights(w: &Weights, dt: Dtype) -> Result<Weights> {
    let mut out = w.clone();
    out.cast_all(dt)?;
    Ok(out)
}

/// Prepare raw SeedVR2 DiT weights exactly as the production loader does: cast each source tensor
/// once, then duplicate the converted shared-block keys as refcounted handles of that cast buffer.
fn prepare_dit(raw: &Weights, dt: Dtype) -> Result<Weights> {
    convert::convert_dit(&cast_weights(raw, dt)?)
}

/// Estimate resident weight bytes for the video memory budget: the raw `fp16` checkpoint file sizes
/// scaled by the load `dtype` (`Bfloat16` keeps the 2 B/param footprint, `Float32` doubles it). File
/// sizes (vs summing per-tensor) match the wan `dit_resident_bytes` convention; the safetensors header
/// overhead is negligible. The raw file stores each shared-block attention tensor once (under
/// `.all`), and since [`Seedvr2Pipeline::load`] casts before duplicating the `_vid`/`_txt` handles
/// (F-012) the resident model holds one buffer per source tensor too — so this file-size estimate
/// matches the actual resident bytes (it previously under-counted the duplicated ~1.15 GB on 3B).
fn resident_weight_bytes(files: &[&std::path::Path], dt: Dtype) -> usize {
    let raw: u64 = files
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum();
    let ratio = match dt {
        Dtype::Float32 => 2.0, // fp16-on-disk → f32 resident
        _ => 1.0,              // bf16/fp16 resident
    };
    (raw as f64 * ratio) as usize
}

/// The bundled precomputed negative-prompt embedding → `(1, 58, 5120)` at `dt`.
fn load_neg_embed(dt: Dtype) -> Result<Array> {
    const BYTES: &[u8] = include_bytes!("../data/neg_embed.safetensors");
    let dir = std::env::temp_dir();
    let path = dir.join("mlx_gen_seedvr2_neg_embed.safetensors");
    // Materialize atomically: write to a process-unique temp file, then `rename` into place (atomic
    // on POSIX). The previous `!exists() { write }` left a *partial* file if a write was interrupted,
    // which every later run then `exists()`-skipped and failed to deserialize, and blindly trusted a
    // pre-existing (stale or hostile) file. Rewriting+renaming each load self-heals corruption and
    // overwrites a planted file; the embed is tiny (~0.6 MB) so the cost is negligible.
    // sc-5276 — filed as F-026 of the 2026-06-13 review, whose document is not in this repo; the
    // in-tree reviews reuse F-026 for three unrelated findings, so cite the story.
    let tmp = dir.join(format!(
        "mlx_gen_seedvr2_neg_embed.{}.tmp",
        std::process::id()
    ));
    std::fs::write(&tmp, BYTES)?;
    std::fs::rename(&tmp, &path)?;
    let w = Weights::from_file(&path)?;
    // Materialize at load (sc-24245) so the first denoise never waits on this read.
    w.materialize()?;
    Ok(w.require("embedding")?.as_dtype(dt)?.expand_dims(0)?)
}

impl Seedvr2Pipeline {
    /// Build from already-converted (MLX-layout) VAE + DiT weights. Used by the parity tests with an
    /// injected neg-embed; `generate` is unavailable until [`Self::load`] sets the bundled embed.
    pub fn from_weights(vae_w: &Weights, dit_w: &Weights, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            vae: Seedvr2Vae::from_weights(vae_w)?,
            transformer: Seedvr2Transformer::from_weights(dit_w, cfg)?,
            neg_embed: None,
            dtype: Dtype::Float32,
            quant_bits: None,
            weights_bytes: 0,
        })
    }

    /// Load from a raw `numz/SeedVR2_comfyUI` checkpoint dir: convert in-memory (no Python), cast to
    /// `dt`, and attach the bundled neg-embed. `dit_file` selects 3B/7B.
    pub fn load(
        raw_dir: impl AsRef<std::path::Path>,
        dit_file: &str,
        cfg: &DitConfig,
        dt: Dtype,
    ) -> Result<Self> {
        let dir = raw_dir.as_ref();
        let vae_path = dir.join("ema_vae_fp16.safetensors");
        let dit_path = dir.join(dit_file);
        let weights_bytes = resident_weight_bytes(&[&vae_path, &dit_path], dt);
        let vae_w = cast_weights(&convert::convert_vae(&Weights::from_file(&vae_path)?)?, dt)?;
        // DiT: cast BEFORE converting (F-012). `convert_dit` duplicates each shared-block
        // `.attn.{proj_qkv,proj_out,norm_q,norm_k}.all` tensor into `_vid`/`_txt` keys as two
        // refcounted handles of ONE array; casting after the duplication gave each key its own
        // `as_dtype` node, materializing two identical buffers per shared tensor (~1.15 GB extra
        // on the 3B at bf16 — MLX does not CSE across arrays). Casting first makes both keys
        // share the single cast buffer. The reorder is exact for every key: `cast_weights` is
        // key-independent and `convert_dit` is dtype-independent (renames + handle clones only).
        // The VAE has no duplicated keys, so its order stays as-is.
        let dit_w = prepare_dit(&Weights::from_file(&dit_path)?, dt)?;
        let mut p = Self::from_weights(&vae_w, &dit_w, cfg)?;
        // Materialize at load, after the cast and before the registry's `quantize` (sc-24245; see
        // `mlx_gen_qwen_image::loader::load_transformer_with`).
        vae_w.materialize_accessed()?;
        dit_w.materialize_accessed()?;
        p.neg_embed = Some(load_neg_embed(dt)?);
        p.dtype = dt;
        p.weights_bytes = weights_bytes;
        Ok(p)
    }

    /// The bundled negative-prompt embedding `(1,58,5120)` (set by [`Self::load`]).
    pub fn neg_embed(&self) -> Option<&Array> {
        self.neg_embed.as_ref()
    }

    /// Quantize the DiT Linears to `bits` (4 or 8) — group-wise affine, Linear-only (sc-5198). The
    /// VAE stays dense (conv-dominated; its tiny attention Linears are left at `dtype`). `weights_bytes`
    /// is intentionally **not** reduced — keeping the dense estimate makes the video chunk-size budget
    /// conservative (quant shrinks weights, not activations, so the headroom is safe).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.transformer.quantize(bits)?;
        self.quant_bits = Some(bits);
        Ok(())
    }

    /// Resident weight bytes (VAE + DiT at `dtype`) used by the budget & route planners.
    pub fn weights_bytes(&self) -> usize {
        self.weights_bytes
    }

    /// Encode the preprocessed image to the conditioning latent `(B,16,T',h,w)` (scaled mean).
    pub fn encode(&self, processed: &Array) -> Result<Array> {
        self.vae.encode(processed)
    }

    /// Build the static condition `[latent, ones-mask]` → `(B, 17, T', h, w)`.
    pub fn condition(latent: &Array) -> Result<Array> {
        let sh = latent.shape();
        let mask =
            Array::ones::<f32>(&[sh[0], 1, sh[2], sh[3], sh[4]])?.as_dtype(latent.dtype())?;
        Ok(concatenate_axis(&[latent, &mask], 1)?)
    }

    fn dit_forward(
        &self,
        noise: &Array,
        condition: &Array,
        neg_embed: &Array,
        timestep: &Array,
    ) -> Result<Array> {
        let model_input = concatenate_axis(&[noise, condition], 1)?; // (B,33,T',h,w)
        self.transformer.forward(&model_input, neg_embed, timestep)
    }

    /// One denoise step driven through [`Seedvr2Driver`] (`[TIMESTEP, 0.0]`): `noise − DiT_out`.
    pub fn denoise(
        &self,
        noise: &Array,
        condition: &Array,
        neg_embed: &Array,
        timestep: &Array,
    ) -> Result<Array> {
        let driver = Seedvr2Driver::for_units(1, |_step, current_noise: &Array| {
            self.dit_forward(current_noise, condition, neg_embed, timestep)
        });
        let cancel = CancelFlag::new();
        let mut st = DriverRunState::new();
        let mut on_progress = |_| {};
        let mut hooks = StepHooks::new(&mut on_progress);
        let out = run_mlx_driver_slices(
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
            return Err(Error::Msg(
                "seedvr2 driver returned non-XSigma state".into(),
            ));
        };
        Ok(latents)
    }

    /// Decode latents and crop to `(true_h, true_w)` → `(B,3,true_h,true_w)`.
    pub fn decode_crop(&self, latents: &Array, true_h: i32, true_w: i32) -> Result<Array> {
        let decoded = self.vae.decode(latents)?; // (B,3,T,H,W)
        let t0 = decoded.take_axis(Array::from_int(0), 2)?; // first frame -> (B,3,H,W)
        let h_idx = Array::from_slice(&(0..true_h).collect::<Vec<i32>>(), &[true_h]);
        let w_idx = Array::from_slice(&(0..true_w).collect::<Vec<i32>>(), &[true_w]);
        Ok(t0.take_axis(h_idx, 2)?.take_axis(w_idx, 3)?)
    }

    /// Full model path (no color correction): preprocessed image + injected noise → decoded crop.
    pub fn run_model(
        &self,
        processed: &Array,
        noise: &Array,
        neg_embed: &Array,
        timestep: &Array,
        true_h: i32,
        true_w: i32,
    ) -> Result<Array> {
        let latent = self.encode(processed)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(noise, &cond, neg_embed, timestep)?;
        self.decode_crop(&latents, true_h, true_w)
    }

    /// End-to-end upscale: LR `image` → `(width, height)` super-resolved RGB8 image.
    ///
    /// `softness` (0..1) pre-blurs the input by round-tripping through a `1 + 7·softness`× smaller
    /// size (the reference `--softness`). Both dims must be multiples of 16 (the registry validates).
    ///
    /// Spatial-tiles when a single full-resolution pass would exceed the memory budget (sc-6067) — the
    /// image analog of the video path's HD-tiling fallback (sc-5201). See [`Self::generate_budgeted`].
    pub fn generate(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        cancel: &CancelFlag,
    ) -> Result<Image> {
        // Progress-free wrapper (unchanged public API): the registry path uses
        // `generate_with_progress` to thread the folded per-tile / per-image bar (F-162/F-164).
        self.generate_with_progress(image, width, height, seed, softness, cancel, &mut |_, _| {})
    }

    /// [`Self::generate`] with a per-tile progress hook (F-162/F-164, sc-11133): `on_step(tile_idx,
    /// n_tiles)` fires once per completed tile on the spatial-tiling path, and once as `(1, 1)` on the
    /// single-pass path, so the registry can fold a monotone, non-restarting bar across the count loop.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_progress(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Image> {
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

    /// [`Self::generate`] with the safe peak-GB ceiling injected (so the spatial-tiling path is
    /// unit-testable without a multi-GB target — mirrors [`crate::video::plan_chunk_size_with`]).
    /// When even a single full-resolution pass would exceed `safe_gib`, the image is upscaled by
    /// feather-blended spatial tiling ([`Self::run_frame_tiled`], the parity-gated sc-5201 tiler)
    /// rather than one allocation that would blow past Metal's max single-buffer size and panic the
    /// worker (sc-6067); otherwise the one-pass still path runs (numerically unchanged from before).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_budgeted(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        safe_gib: f64,
        cancel: &CancelFlag,
    ) -> Result<Image> {
        self.generate_budgeted_with_progress(
            image,
            width,
            height,
            seed,
            softness,
            safe_gib,
            cancel,
            &mut |_, _| {},
        )
    }

    /// [`Self::generate_budgeted`] with the per-tile progress hook threaded through (F-162/F-164):
    /// the tiled branch reports `(tile_idx, n_tiles)` per tile; the single-pass branch reports `(1, 1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_budgeted_with_progress(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        safe_gib: f64,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Image> {
        let resolved = video::resolve_still_route(width, height, self.weights_bytes, safe_gib);
        self.generate_still_for_route_with_policy(
            image,
            width,
            height,
            seed,
            softness,
            &resolved.route,
            resolved.over_budget,
            resolved.over_vae_cap,
            cancel,
            on_step,
        )
    }

    /// Execute one single-pass still image upscale (or one `VideoPerFrame` frame) through
    /// [`Seedvr2Driver`] and [`run_mlx_driver_slices`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_still_for_route(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        route: &Seedvr2Route,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Image> {
        let over_vae_cap = width.max(height) > video::VAE_SAFE_DECODE_EDGE_PX;
        let over_budget = matches!(route, Seedvr2Route::StillSpatialTiled { .. })
            && (!over_vae_cap
                || matches!(
                    video::plan_chunk_size(self.weights_bytes, height, width),
                    ChunkPlan::OverBudget { .. }
                ));
        self.generate_still_for_route_with_policy(
            image,
            width,
            height,
            seed,
            softness,
            route,
            over_budget,
            over_vae_cap,
            cancel,
            on_step,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_still_for_route_with_policy(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        route: &Seedvr2Route,
        over_budget: bool,
        over_vae_cap: bool,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Image> {
        mlx_gen::video_execution::policy(
            "seedvr2_image_route",
            serde_json::json!({"over_budget":over_budget,"over_vae_cap":over_vae_cap,"dtype":format!("{:?}",self.dtype),"quant_bits":self.quant_bits,"softness_bits":softness.to_bits()}),
        );
        match route {
            Seedvr2Route::StillSinglePass => self
                .generate_single_pass_unit(image, width, height, seed, softness, cancel, on_step),
            Seedvr2Route::StillSpatialTiled {
                overlap_px, tiles, ..
            } => self.generate_tiled_with_plan(
                image,
                width,
                height,
                seed,
                softness,
                *overlap_px,
                tiles,
                cancel,
                on_step,
            ),
            _ => Err(Error::Msg(
                "seedvr2: expected a still route for image generation".into(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_frame_unit(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
    ) -> Result<(Array, Array, Array)> {
        let processed = self.preprocess(image, width, height, softness)?; // (1,3,H,W)
        let latent = self.encode(&processed)?;
        let sh = latent.shape();
        let key = random::key(seed)?;
        mlx_gen::video_execution::draw_current("seedvr2_same_seed_per_chunk_or_tile", seed);
        mlx_gen::video_execution::sampling(
            "seedvr2",
            "one_step_euler",
            "flow_velocity",
            serde_json::json!({"timestep_bits":TIMESTEP.to_bits()}),
        );
        mlx_gen::video_execution::schedule("seedvr2", &[1.0, 0.0]);
        let noise = random::normal::<f32>(&[1, 16, sh[2], sh[3], sh[4]], None, None, Some(&key))?
            .as_dtype(self.dtype)?;
        let cond = Self::condition(&latent)?;
        Ok((processed, noise, cond))
    }

    fn finish_frame_unit(
        &self,
        latents: &Array,
        processed: &Array,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        let decoded = self.decode_crop(latents, height, width)?; // (1,3,H,W)
        let style = processed
            .take_axis(
                Array::from_slice(&(0..height).collect::<Vec<i32>>(), &[height]),
                2,
            )?
            .take_axis(
                Array::from_slice(&(0..width).collect::<Vec<i32>>(), &[width]),
                3,
            )?;
        let corrected = color::apply_color_correction(
            &decoded.as_dtype(Dtype::Float32)?,
            &style.as_dtype(Dtype::Float32)?,
            LUMINANCE_WEIGHT,
        )?;
        decoded_to_image(&corrected)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_driver_units(
        units: usize,
        width: i32,
        height: i32,
        seed: u64,
        cancel: &CancelFlag,
        prepare: impl Fn(usize, &Array) -> Result<Array>,
        predict: impl Fn(usize, &Array) -> Result<Array>,
        on_post_step: &mut dyn FnMut(&DriverStepInfo, Array) -> gen_core::Result<Array>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<()> {
        let driver = Seedvr2Driver::for_units(units, UnitForward { prepare, predict });
        let mut st = DriverRunState::new();
        let mut hooks = StepHooks::new(on_progress);
        hooks.on_post_step = Some(on_post_step);
        let mut spec = SliceSpec::full(units);
        spec.geometry = Some((width as u32, height as u32));
        run_mlx_driver_slices(
            &driver,
            [spec],
            BoundaryState::XSigma(Array::from_f32(0.0)),
            units,
            seed,
            cancel,
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
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Image> {
        let neg = self
            .neg_embed
            .as_ref()
            .ok_or_else(|| Error::Msg("seedvr2: negative embedding not loaded — construct via Seedvr2Pipeline::load, not from_weights".into()))?;
        let ts = Array::from_f32(TIMESTEP);
        let active: RefCell<Option<(Array, Array)>> = RefCell::new(None);
        let result_img: RefCell<Option<Image>> = RefCell::new(None);
        let mut progress = |p: Progress| {
            if let Progress::Step { current, total } = p {
                on_step(current as usize, total as usize);
            }
        };
        let mut post_step = |_info: &DriverStepInfo, latents: Array| -> gen_core::Result<Array> {
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
                    .ok_or_else(|| Error::Msg("seedvr2: unit not prepared".into()))?;
                self.dit_forward(noise, &cond, neg, &ts)
            },
            &mut post_step,
            &mut progress,
        )?;
        Ok(result_img
            .into_inner()
            .expect("one single-pass still image produced"))
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_tiled_with_plan(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        overlap: i32,
        plan: &[video::SpatialTile],
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Image> {
        let neg = self
            .neg_embed
            .as_ref()
            .ok_or_else(|| Error::Msg("seedvr2: negative embedding not loaded — construct via Seedvr2Pipeline::load, not from_weights".into()))?
            .clone();
        let processed = self.preprocess_frame(image, width, height, softness)?; // (1,3,1,H,W)
        let decoded =
            self.run_frame_tiled_with_plan(&processed, seed, overlap, plan, &neg, cancel, on_step)?;
        self.frames_from_decoded(&decoded, &processed, 1, cancel)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("seedvr2 tiled upscale: produced no frame".into()))
    }

    /// Preprocess one LR `Image` to a single-frame clip `(1,3,1,height,width)` in `[-1,1]` at the model
    /// dtype (the `preprocess` image + a temporal axis). Public for the tiling gate.
    pub fn preprocess_frame(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        softness: f32,
    ) -> Result<Array> {
        Ok(self
            .preprocess(image, width, height, softness)?
            .expand_dims(2)?)
    }

    /// LR `Image` → `(1,3,height,width)` in `[-1,1]` at the model dtype. PIL-exact bicubic resize to
    /// the target; optional `softness` pre-blur via a smaller round-trip.
    fn preprocess(&self, image: &Image, width: i32, height: i32, softness: f32) -> Result<Array> {
        mlx_gen::video_execution::policy(
            "seedvr2_preprocess",
            serde_json::json!({
                "softness_bits":softness.clamp(0.0,1.0).to_bits(),"target":[width,height],
                "luminance_weight_bits":LUMINANCE_WEIGHT.to_bits()
            }),
        );
        let (ih, iw) = (image.height as usize, image.width as usize);
        let (oh, ow) = (height as usize, width as usize);
        let resized: Vec<f32> = if softness > 0.0 {
            let factor = 1.0 + softness.clamp(0.0, 1.0) * 7.0;
            let dw = ((width as f32 / factor) as usize).max(2);
            let dh = ((height as f32 / factor) as usize).max(2);
            let down = resize_bicubic_u8(&image.pixels, ih, iw, dh, dw)?; // f32 [0,255]
            let down_u8: Vec<u8> = down
                .iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect();
            resize_bicubic_u8(&down_u8, dh, dw, oh, ow)?
        } else {
            resize_bicubic_u8(&image.pixels, ih, iw, oh, ow)?
        };
        // HWC [0,255] f32 → [-1,1] → (1,3,H,W)
        let arr = Array::from_slice(&resized, &[height, width, 3]);
        let arr = subtract(
            &multiply(&arr, Array::from_f32(2.0 / 255.0))?,
            Array::from_f32(1.0),
        )?;
        Ok(arr
            .transpose_axes(&[2, 0, 1])?
            .expand_dims(0)?
            .as_dtype(self.dtype)?)
    }

    // -----------------------------------------------------------------------
    // video mode (sc-4814): multi-frame 5-D pass + temporal chunking/overlap
    // -----------------------------------------------------------------------

    /// Decode latents and crop spatially to `(true_h, true_w)`, **keeping all `T` frames** →
    /// `(B,3,T,true_h,true_w)`. The 5-D analog of [`Self::decode_crop`] (which keeps only frame 0).
    pub fn decode_crop_5d(&self, latents: &Array, true_h: i32, true_w: i32) -> Result<Array> {
        let decoded = self.vae.decode(latents)?; // (B,3,T,H,W)
        let h_idx = Array::from_slice(&(0..true_h).collect::<Vec<i32>>(), &[true_h]);
        let w_idx = Array::from_slice(&(0..true_w).collect::<Vec<i32>>(), &[true_w]);
        Ok(decoded.take_axis(h_idx, 3)?.take_axis(w_idx, 4)?)
    }

    /// Multi-frame model path (no color correction): a preprocessed clip `(1,3,T,H,W)` + injected
    /// noise `(1,16,T',h,w)` → decoded crop `(1,3,T,true_h,true_w)`. The video analog of
    /// [`Self::run_model`]; `encode`/`condition`/`denoise` are already `T`-aware.
    pub fn run_model_5d(
        &self,
        processed: &Array,
        noise: &Array,
        neg_embed: &Array,
        timestep: &Array,
        true_h: i32,
        true_w: i32,
    ) -> Result<Array> {
        let latent = self.encode(processed)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(noise, &cond, neg_embed, timestep)?;
        self.decode_crop_5d(&latents, true_h, true_w)
    }

    /// Per-frame color-correct a decoded clip `(1,3,T,Hc,Wc)` against its preprocessed `style`
    /// `(1,3,Ts,Hc,Wc)` → `count` RGB8 frames. `count` includes only real frames needed by
    /// assembly, never trailing last-frame padding. Frame `t` matches style frame `min(t, Ts-1)`
    /// (the reference `to_uint8_frames`).
    fn frames_from_decoded(
        &self,
        decoded: &Array,
        style: &Array,
        count: i32,
        cancel: &CancelFlag,
    ) -> Result<Vec<Image>> {
        let style_t = style.shape()[2];
        let mut out = Vec::with_capacity(count as usize);
        for t in 0..count {
            // F-163: per-frame color correction is single-threaded host math (5-level wavelet + three
            // full-frame sorts, `as_slice`-materialized), so a whole chunk of it was uncancellable.
            // Each iteration already forces the prior frame's work to host, making this a genuine
            // per-frame yield point (no `eval` needed) — bounds cancel latency to one frame.
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let d = decoded.take_axis(Array::from_int(t), 2)?; // (1,3,Hc,Wc)
            let s = style.take_axis(Array::from_int(t.min(style_t - 1)), 2)?;
            let corrected = color::apply_color_correction(
                &d.as_dtype(Dtype::Float32)?,
                &s.as_dtype(Dtype::Float32)?,
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
        width: i32,
        height: i32,
        softness: f32,
    ) -> Result<Array> {
        let n = frames.len() as i32;
        let mut per = Vec::with_capacity(len as usize);
        for j in 0..len {
            let idx = (start + j).clamp(0, n - 1) as usize;
            per.push(
                self.preprocess(&frames[idx], width, height, softness)?
                    .expand_dims(2)?,
            );
        }
        let refs: Vec<&Array> = per.iter().collect();
        Ok(concatenate_axis(&refs, 2)?)
    }

    /// End-to-end **video** upscale: a sequence of LR `frames` → upscaled `(width, height)` RGB8
    /// frames (same count). Sizes the temporal chunk against the memory budget (or `chunk_override`),
    /// processes each chunk through the 5-D model path with one-step Euler via [`Seedvr2Driver`],
    /// per-frame color-corrects, and cross-fades chunk overlaps to close the causal-VAE seam
    /// ([`crate::video`]). Falls back to per-frame (`T=1`) under tight memory, and to per-frame
    /// **spatial tiling** when even one full-resolution frame exceeds the budget (HD — sc-5201).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_video(
        &self,
        frames: &[Image],
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        chunk_override: Option<i32>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let resolved = video::resolve_video_route(
            frames.len(),
            width,
            height,
            self.weights_bytes,
            video::safe_budget_gib(),
            chunk_override,
        );
        self.generate_video_for_route(
            frames,
            width,
            height,
            seed,
            softness,
            &resolved.route,
            cancel,
            on_progress,
        )
    }

    /// Execute a compiled [`Seedvr2Route`] video upscale via [`Seedvr2Driver`] and [`run_mlx_driver_slices`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_video_for_route(
        &self,
        frames: &[Image],
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        route: &Seedvr2Route,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        match route {
            Seedvr2Route::VideoTemporalChunked {
                chunk_frames,
                overlap_frames,
                chunks,
            } => self.generate_video_temporal_chunked(
                frames,
                width,
                height,
                seed,
                softness,
                *chunk_frames,
                *overlap_frames,
                chunks,
                cancel,
                on_progress,
            ),
            Seedvr2Route::VideoPerFrame { .. } => self.generate_video_per_frame(
                frames,
                width,
                height,
                seed,
                softness,
                cancel,
                on_progress,
            ),
            Seedvr2Route::VideoSpatialTiled {
                overlap_px,
                tiles_per_frame,
                ..
            } => self.generate_video_tiled_with_plan(
                frames,
                width,
                height,
                seed,
                softness,
                *overlap_px,
                tiles_per_frame,
                cancel,
                on_progress,
            ),
            _ => Err(Error::Msg(
                "seedvr2: expected a video route for video clip generation".into(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_video_temporal_chunked(
        &self,
        frames: &[Image],
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        chunk: i32,
        overlap: i32,
        plan: &[Chunk],
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let n = frames.len() as i32;
        if n == 0 || plan.is_empty() {
            return Ok(Vec::new());
        }
        mlx_gen::video_execution::policy(
            "seedvr2_video_chunks",
            serde_json::json!({"chunk":chunk,"overlap":overlap,"plan":format!("{plan:?}"),"dtype":format!("{:?}",self.dtype),"quant_bits":self.quant_bits,"softness_bits":softness.to_bits()}),
        );
        let neg = self
            .neg_embed
            .as_ref()
            .ok_or_else(|| Error::Msg("seedvr2: negative embedding not loaded — construct via Seedvr2Pipeline::load, not from_weights".into()))?;
        let ts = Array::from_f32(TIMESTEP);
        let total_chunks = plan.len();
        let active: RefCell<Option<(Array, Array, Chunk)>> = RefCell::new(None);
        let out: RefCell<Vec<Image>> = RefCell::new(Vec::with_capacity(n as usize));

        let mut post_step = |_info: &DriverStepInfo, latents: Array| -> gen_core::Result<Array> {
            let (clip, _, c) = active
                .borrow_mut()
                .take()
                .ok_or_else(|| gen_core::Error::Msg("seedvr2: missing chunk state".into()))?;
            let decoded = self
                .decode_crop_5d(&latents, height, width)
                .map_err(core_err)?;
            let real_count = video::real_frame_count(c.start, c.len, n);
            let chunk_frames = self
                .frames_from_decoded(&decoded, &clip, real_count, cancel)
                .map_err(core_err)?;
            video::assemble_overlap_chunk(&mut out.borrow_mut(), c.start, chunk_frames, n, overlap);
            Ok(latents)
        };
        Self::run_driver_units(
            total_chunks,
            width,
            height,
            seed,
            cancel,
            |chunk_idx, _cur| {
                let c = plan[chunk_idx];
                let clip =
                    self.preprocess_chunk(frames, c.start, c.len, width, height, softness)?;
                let latent = self.encode(&clip)?;
                let sh = latent.shape();
                let key = random::key(seed)?;
                mlx_gen::video_execution::draw_current("seedvr2_same_seed_per_chunk_or_tile", seed);
                mlx_gen::video_execution::sampling(
                    "seedvr2",
                    "one_step_euler",
                    "flow_velocity",
                    serde_json::json!({"timestep_bits":TIMESTEP.to_bits()}),
                );
                mlx_gen::video_execution::schedule("seedvr2", &[1.0, 0.0]);
                let noise =
                    random::normal::<f32>(&[1, 16, sh[2], sh[3], sh[4]], None, None, Some(&key))?
                        .as_dtype(self.dtype)?;
                let cond = Self::condition(&latent)?;
                *active.borrow_mut() = Some((clip, cond, c));
                Ok(noise)
            },
            |_chunk_idx, noise| {
                let (_, cond, _) = active
                    .borrow()
                    .as_ref()
                    .map(|(clip, cond, c)| (clip.clone(), cond.clone(), *c))
                    .ok_or_else(|| Error::Msg("seedvr2: chunk unit not prepared".into()))?;
                self.dit_forward(noise, &cond, neg, &ts)
            },
            &mut post_step,
            on_progress,
        )?;
        Ok(out.into_inner())
    }

    /// Per-frame (`T=1`) video fallback driven through [`Seedvr2Driver`] + [`run_mlx_driver_slices`].
    #[allow(clippy::too_many_arguments)]
    fn generate_video_per_frame(
        &self,
        frames: &[Image],
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let total = frames.len();
        if total == 0 {
            return Ok(Vec::new());
        }
        let neg = self
            .neg_embed
            .as_ref()
            .ok_or_else(|| Error::Msg("seedvr2: negative embedding not loaded — construct via Seedvr2Pipeline::load, not from_weights".into()))?;
        let ts = Array::from_f32(TIMESTEP);
        let active: RefCell<Option<(Array, Array)>> = RefCell::new(None);
        let out: RefCell<Vec<Image>> = RefCell::new(Vec::with_capacity(total));
        let mut post_step = |_info: &DriverStepInfo, latents: Array| -> gen_core::Result<Array> {
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
                    .ok_or_else(|| Error::Msg("seedvr2: per-frame unit not prepared".into()))?;
                self.dit_forward(noise, &cond, neg, &ts)
            },
            &mut post_step,
            on_progress,
        )?;
        Ok(out.into_inner())
    }

    /// HD spatial-tiling video path (sc-5201): each frame is upscaled per-frame (`T=1`) with its
    /// spatial tiles driven through [`Seedvr2Driver`].
    #[allow(clippy::too_many_arguments)]
    fn generate_video_tiled_with_plan(
        &self,
        frames: &[Image],
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        overlap: i32,
        plan: &[video::SpatialTile],
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Image>> {
        let neg = self
            .neg_embed
            .as_ref()
            .ok_or_else(|| Error::Msg("seedvr2: negative embedding not loaded — construct via Seedvr2Pipeline::load, not from_weights".into()))?
            .clone();
        let total = frames.len() as u32;
        let mut out = Vec::with_capacity(frames.len());
        for (i, f) in frames.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let processed = self
                .preprocess(f, width, height, softness)?
                .expand_dims(2)?; // (1,3,1,H,W)
            let decoded = self.run_frame_tiled_with_plan(
                &processed,
                seed,
                overlap,
                plan,
                &neg,
                cancel,
                &mut |_, _| {},
            )?;
            let imgs = self.frames_from_decoded(&decoded, &processed, 1, cancel)?;
            out.push(
                imgs.into_iter()
                    .next()
                    .ok_or_else(|| Error::Msg("seedvr2 tiled upscale: produced no frame".into()))?,
            );
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
        }
        Ok(out)
    }

    /// Upscale one preprocessed frame `(1,3,1,H,W)` by spatial tiling through [`Seedvr2Driver`].
    pub fn run_frame_tiled(
        &self,
        processed: &Array,
        seed: u64,
        tile: i32,
        overlap: i32,
        neg: &Array,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        self.run_frame_tiled_with_progress(
            processed,
            seed,
            tile,
            overlap,
            neg,
            cancel,
            &mut |_, _| {},
        )
    }

    /// [`Self::run_frame_tiled`] with a per-tile progress hook driven by [`Seedvr2Driver`] +
    /// [`run_mlx_driver_slices`].
    #[allow(clippy::too_many_arguments)]
    pub fn run_frame_tiled_with_progress(
        &self,
        processed: &Array,
        seed: u64,
        tile: i32,
        overlap: i32,
        neg: &Array,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Array> {
        let sh = processed.shape(); // (1,3,1,H,W)
        let (height, width) = (sh[3], sh[4]);
        let plan = video::plan_spatial_tiles(height, width, tile, overlap);
        self.run_frame_tiled_with_plan(processed, seed, overlap, &plan, neg, cancel, on_step)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_frame_tiled_with_plan(
        &self,
        processed: &Array,
        seed: u64,
        overlap: i32,
        plan: &[video::SpatialTile],
        neg: &Array,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Array> {
        let sh = processed.shape(); // (1,3,1,H,W)
        let (height, width) = (sh[3], sh[4]);
        let n_tiles = plan.len();
        let ts = Array::from_f32(TIMESTEP);
        let fold = RefCell::new(TileFoldAccumulator::new(plan, height, width));
        let active: RefCell<Option<(video::SpatialTile, i32, i32, Array)>> = RefCell::new(None);
        let mut progress = |p: Progress| {
            if let Progress::Step { current, total } = p {
                on_step(current as usize, total as usize);
            }
        };
        let mut post_step = |info: &DriverStepInfo, latents: Array| -> gen_core::Result<Array> {
            let (t, th, tw, _) = active
                .borrow_mut()
                .take()
                .ok_or_else(|| gen_core::Error::Msg("seedvr2: missing tile state".into()))?;
            let decoded = self.decode_crop_5d(&latents, th, tw).map_err(core_err)?;
            let wvec = video::feather_weight(
                th,
                tw,
                t.y0 > 0,
                t.y1 < height,
                t.x0 > 0,
                t.x1 < width,
                overlap,
            );
            let weight = Array::from_slice(&wvec, &[1, 1, 1, th, tw])
                .as_dtype(self.dtype)
                .map_err(Error::from)
                .map_err(core_err)?;
            let weighted = multiply(&decoded, &weight)
                .map_err(Error::from)
                .map_err(core_err)?;
            fold.borrow_mut()
                .push_tile(info.outer_step, &t, weighted, weight)
                .map_err(core_err)?;
            Ok(decoded)
        };
        Self::run_driver_units(
            n_tiles,
            width,
            height,
            seed,
            cancel,
            |tile_idx, _cur| {
                let t = plan[tile_idx];
                let (th, tw) = (t.y1 - t.y0, t.x1 - t.x0);
                mlx_gen::video_execution::policy(
                    "seedvr2_spatial_tile",
                    serde_json::json!({"rect":[t.x0,t.y0,t.x1,t.y1],"overlap":overlap,"dtype":format!("{:?}",self.dtype),"quant_bits":self.quant_bits}),
                );
                let y_idx = Array::from_slice(&(t.y0..t.y1).collect::<Vec<i32>>(), &[th]);
                let x_idx = Array::from_slice(&(t.x0..t.x1).collect::<Vec<i32>>(), &[tw]);
                let tile_clip = processed.take_axis(y_idx, 3)?.take_axis(x_idx, 4)?;
                let latent = self.encode(&tile_clip)?;
                let lsh = latent.shape();
                let key = random::key(seed)?;
                mlx_gen::video_execution::draw_current("seedvr2_same_seed_per_chunk_or_tile", seed);
                mlx_gen::video_execution::sampling(
                    "seedvr2",
                    "one_step_euler",
                    "flow_velocity",
                    serde_json::json!({"timestep_bits":TIMESTEP.to_bits()}),
                );
                mlx_gen::video_execution::schedule("seedvr2", &[1.0, 0.0]);
                let noise = random::normal::<f32>(
                    &[1, 16, lsh[2], lsh[3], lsh[4]],
                    None,
                    None,
                    Some(&key),
                )?
                .as_dtype(self.dtype)?;
                let cond = Self::condition(&latent)?;
                *active.borrow_mut() = Some((t, th, tw, cond));
                Ok(noise)
            },
            |_tile_idx, noise| {
                let (_, _, _, cond) = active
                    .borrow()
                    .as_ref()
                    .map(|(t, th, tw, cond)| (*t, *th, *tw, cond.clone()))
                    .ok_or_else(|| Error::Msg("seedvr2: tile unit not prepared".into()))?;
                self.dit_forward(noise, &cond, neg, &ts)
            },
            &mut post_step,
            &mut progress,
        )?;
        fold.into_inner().finish()
    }
}

struct TileFoldAccumulator {
    height: i32,
    width: i32,
    row_len: usize,
    acc: Option<Array>,
    wsum: Option<Array>,
    row_acc: Option<Array>,
    row_wsum: Option<Array>,
}

impl TileFoldAccumulator {
    fn new(plan: &[video::SpatialTile], height: i32, width: i32) -> Self {
        let row_len = plan
            .first()
            .map(|first| plan.iter().filter(|t| t.y0 == first.y0).count().max(1))
            .unwrap_or(1);
        Self {
            height,
            width,
            row_len,
            acc: None,
            wsum: None,
            row_acc: None,
            row_wsum: None,
        }
    }

    fn push_tile(
        &mut self,
        tile_idx: usize,
        t: &video::SpatialTile,
        weighted: Array,
        weight: Array,
    ) -> Result<()> {
        let (th, tw) = (t.y1 - t.y0, t.x1 - t.x0);
        self.row_acc = Some(accumulate_along_axis(
            self.row_acc.take(),
            &weighted,
            4,
            t.x0,
            tw,
            self.width,
        )?);
        self.row_wsum = Some(accumulate_along_axis(
            self.row_wsum.take(),
            &weight,
            4,
            t.x0,
            tw,
            self.width,
        )?);
        if (tile_idx + 1).is_multiple_of(self.row_len) {
            let strip = self.row_acc.take().expect("the row opened above");
            let strip_weight = self.row_wsum.take().expect("the row opened above");
            self.acc = Some(accumulate_along_axis(
                self.acc.take(),
                &strip,
                3,
                t.y0,
                th,
                self.height,
            )?);
            self.wsum = Some(accumulate_along_axis(
                self.wsum.take(),
                &strip_weight,
                3,
                t.y0,
                th,
                self.height,
            )?);
        }
        let live: Vec<&Array> = [
            self.row_acc.as_ref(),
            self.row_wsum.as_ref(),
            self.acc.as_ref(),
            self.wsum.as_ref(),
        ]
        .into_iter()
        .flatten()
        .collect();
        eval(live)?;
        Ok(())
    }

    fn finish(self) -> Result<Array> {
        if self.row_acc.is_some() || self.row_wsum.is_some() {
            return Err(Error::Msg(
                "seedvr2 tiled upscale: the tile plan left a row unclosed — it is not the row-major \
                 product grid `plan_spatial_tiles` emits"
                    .into(),
            ));
        }
        let acc = self
            .acc
            .ok_or_else(|| Error::Msg("seedvr2 tiled upscale: plan had no tiles".into()))?;
        let wsum = self
            .wsum
            .ok_or_else(|| Error::Msg("seedvr2 tiled upscale: plan had no tiles".into()))?;
        Ok(divide(&acc, &wsum)?)
    }
}

/// Fold each tile's weighted decode into the full frame, **x axis first** (sc-18320).
#[cfg(test)]
fn fold_spatial_tiles(
    plan: &[video::SpatialTile],
    height: i32,
    width: i32,
    mut weighted_tile: impl FnMut(&video::SpatialTile) -> Result<(Array, Array)>,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<Array> {
    let n_tiles = plan.len();
    let mut fold = TileFoldAccumulator::new(plan, height, width);
    for (tile_idx, t) in plan.iter().enumerate() {
        let (weighted, weight) = weighted_tile(t)?;
        fold.push_tile(tile_idx, t, weighted, weight)?;
        on_step(tile_idx + 1, n_tiles);
    }
    fold.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The **predecessor assembly**, verbatim: every weighted tile *and* its feather weight padded to
    /// the full frame, added into two full-frame accumulators, divided once at the end. A copy, not a
    /// refactor — grading the fold against a paraphrase of itself would prove nothing.
    fn pad_to_full_reference(
        plan: &[video::SpatialTile],
        height: i32,
        width: i32,
        mut weighted_tile: impl FnMut(&video::SpatialTile) -> Result<(Array, Array)>,
    ) -> Result<Array> {
        let mut acc: Option<Array> = None;
        let mut wsum: Option<Array> = None;
        for t in plan {
            let (weighted, weight) = weighted_tile(t)?;
            let pad_spec = [
                (0, 0),
                (0, 0),
                (0, 0),
                (t.y0, height - t.y1),
                (t.x0, width - t.x1),
            ];
            let wdec = mlx_rs::ops::pad(&weighted, &pad_spec[..], None, None)?;
            let wpad = mlx_rs::ops::pad(&weight, &pad_spec[..], None, None)?;
            acc = Some(match acc {
                Some(a) => mlx_rs::ops::add(&a, &wdec)?,
                None => wdec,
            });
            wsum = Some(match wsum {
                Some(a) => mlx_rs::ops::add(&a, &wpad)?,
                None => wpad,
            });
            eval([acc.as_ref().unwrap(), wsum.as_ref().unwrap()])?;
        }
        Ok(divide(acc.expect("≥1 tile"), wsum.expect("≥1 tile"))?)
    }

    /// sc-18320: the x-then-y fold must reproduce the pad-every-tile-to-the-full-frame assembly it
    /// replaced, on a **ragged** multi-row grid, with the shipped feather weights and a
    /// position-dependent per-tile "decode" that differs between neighbours (so the feather weights
    /// and their accumulated normalizer are both load-bearing — a mis-placed tile, a dropped weight,
    /// or a row closed at the wrong index cannot pass by reconstructing an identity).
    #[test]
    fn spatial_fold_matches_the_pad_to_full_assembly() {
        let (height, width, tile, overlap) = (200, 296, 128, 32);
        let plan = video::plan_spatial_tiles(height, width, tile, overlap);
        let rows = plan.iter().filter(|t| t.x0 == 0).count();
        let cols = plan.iter().filter(|t| t.y0 == 0).count();
        assert!(
            rows > 1 && cols > 1 && plan.len() == rows * cols,
            "the fixture must be a genuine multi-row product grid, got {rows}×{cols} for {} tiles",
            plan.len()
        );
        // `tile_axis` keeps every tile full-size and shifts the LAST one back to land on the edge, so
        // the trailing row/column overlaps its neighbour by more than `overlap`. That irregular stride
        // is the fixture's raggedness — a fold that assumed a uniform stride would place it wrong.
        let last = plan.last().expect("a non-empty plan");
        assert_eq!((last.y1, last.x1), (height, width), "the plan must cover");
        assert!(
            plan.iter()
                .any(|t| t.y0 > 0 && t.y0 % (tile - overlap) != 0)
                || plan
                    .iter()
                    .any(|t| t.x0 > 0 && t.x0 % (tile - overlap) != 0),
            "the fixture must include a shifted-back edge tile"
        );

        // Position-dependent and tile-dependent: value varies with absolute (y, x) AND carries a
        // per-tile offset, so overlapping tiles disagree exactly where the feather blends them.
        let tile_fn = |t: &video::SpatialTile| -> Result<(Array, Array)> {
            let (th, tw) = (t.y1 - t.y0, t.x1 - t.x0);
            let values: Vec<f32> = (0..3 * th * tw)
                .map(|i| {
                    let channel = i / (th * tw);
                    let y = t.y0 + (i / tw % th);
                    let x = t.x0 + (i % tw);
                    ((y as f32) * 0.013 + (x as f32) * 0.007 + (channel as f32) * 0.11).sin()
                        + (t.y0 + t.x0) as f32 * 1e-3
                })
                .collect();
            let decoded = Array::from_slice(&values, &[1, 3, 1, th, tw]);
            let wvec = video::feather_weight(
                th,
                tw,
                t.y0 > 0,
                t.y1 < height,
                t.x0 > 0,
                t.x1 < width,
                overlap,
            );
            let weight = Array::from_slice(&wvec, &[1, 1, 1, th, tw]);
            Ok((multiply(&decoded, &weight)?, weight))
        };

        let reference = pad_to_full_reference(&plan, height, width, tile_fn).unwrap();
        let mut steps = Vec::new();
        let folded = fold_spatial_tiles(&plan, height, width, tile_fn, &mut |done, total| {
            steps.push((done, total))
        })
        .unwrap();
        reference.eval().unwrap();
        folded.eval().unwrap();

        assert_eq!(folded.shape(), &[1, 3, 1, height, width]);
        assert_eq!(folded.shape(), reference.shape());
        let max = folded
            .as_slice::<f32>()
            .iter()
            .zip(reference.as_slice::<f32>())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-5,
            "the fold moved the assembled frame: max|Δ|={max:.3e}"
        );
        // F-164 liveness is unchanged: one report per tile, 1-based, against the tile count.
        assert_eq!(
            steps,
            (1..=plan.len())
                .map(|i| (i, plan.len()))
                .collect::<Vec<_>>()
        );
    }

    /// sc-18320: the fold closes a row exactly at the grid's row boundaries.
    ///
    /// `row_len` is derived from the plan (count of tiles sharing the first tile's `y0`) rather than
    /// recomputed from geometry, so pin that derivation against what `plan_spatial_tiles` actually
    /// emits: x varies fastest, and `(idx + 1) % row_len == 0` must hold exactly for the tiles that
    /// end a row. Closing a row early would fold a partial strip into the frame at the wrong offset.
    /// A **non-square** grid is load-bearing here — it is what separates a correct derivation from one
    /// that counted rows instead of columns.
    #[test]
    fn spatial_fold_row_boundaries_match_the_plan_grid() {
        let (height, width, tile, overlap) = (200, 296, 128, 32);
        let plan = video::plan_spatial_tiles(height, width, tile, overlap);
        let row_len = plan
            .first()
            .map(|first| plan.iter().filter(|t| t.y0 == first.y0).count().max(1))
            .unwrap_or(1);
        let rows = plan.iter().filter(|t| t.x0 == plan[0].x0).count();
        assert_ne!(rows, row_len, "the grid must not be square");
        assert_eq!(
            rows * row_len,
            plan.len(),
            "the plan must be a product grid"
        );
        for (idx, t) in plan.iter().enumerate() {
            assert_eq!(
                (idx + 1) % row_len == 0,
                t.x1 == width,
                "tile {idx} at x[{}, {}) disagrees with the row boundary",
                t.x0,
                t.x1
            );
        }
    }

    /// sc-18320: a plan that is not the row-major product grid leaves a row open, and the trailing
    /// band would vanish from BOTH accumulators — so the divide would still succeed and quietly return
    /// a frame missing its last strip. Refuse instead. Cannot happen with `plan_spatial_tiles` today;
    /// the guard exists because `row_len` is derived from the plan's layout rather than handed down by
    /// the loop bounds, so a future plan shape could silently violate it.
    #[test]
    fn spatial_fold_refuses_a_plan_that_leaves_a_row_open() {
        let (height, width) = (200, 296);
        let mut plan = video::plan_spatial_tiles(height, width, 128, 32);
        plan.pop().expect("a non-empty plan");
        let result = fold_spatial_tiles(
            &plan,
            height,
            width,
            |t| {
                let (th, tw) = (t.y1 - t.y0, t.x1 - t.x0);
                let ones = vec![1.0f32; (3 * th * tw) as usize];
                let weight = vec![1.0f32; (th * tw) as usize];
                Ok((
                    Array::from_slice(&ones, &[1, 3, 1, th, tw]),
                    Array::from_slice(&weight, &[1, 1, 1, th, tw]),
                ))
            },
            &mut |_, _| {},
        );
        assert!(
            matches!(result, Err(Error::Msg(ref m)) if m.contains("left a row unclosed")),
            "an incomplete grid must be refused, got {result:?}"
        );
    }

    /// sc-18320: a cancel raised by the tile closure aborts the fold rather than being swallowed by
    /// the accumulation, and no progress is reported for the tile that never completed.
    #[test]
    fn spatial_fold_propagates_a_tile_failure_without_reporting_it() {
        let (height, width) = (200, 296);
        let plan = video::plan_spatial_tiles(height, width, 128, 32);
        let mut steps = 0usize;
        let result = fold_spatial_tiles(
            &plan,
            height,
            width,
            |_| Err(Error::Canceled),
            &mut |_, _| steps += 1,
        );
        assert!(matches!(result, Err(Error::Canceled)));
        assert_eq!(steps, 0);
    }

    /// The raw checkpoint keys of one shared block's attention (stored once under `.all`), at the
    /// given shapes, in fp16 (the dtype the real file stores). Values are random so buffers are
    /// genuinely materialized (MLX `full`/broadcast outputs can be stride-0 views); the RNG is
    /// explicitly keyed — the process-global RNG state binds to the first test thread's Metal
    /// stream and panics later tests with "There is no Stream(gpu, 0) in current thread".
    fn synth_shared_attn_block(w: &mut Weights, idx: i32, dim: i32, heads: i32, head_dim: i32) {
        let qkv = 3 * heads * head_dim;
        for (i, (name, shape)) in [
            ("proj_qkv.all.weight", vec![qkv, dim]),
            ("proj_out.all.weight", vec![dim, dim]),
            ("proj_out.all.bias", vec![dim]),
            ("norm_q.all.weight", vec![head_dim]),
            ("norm_k.all.weight", vec![head_dim]),
        ]
        .into_iter()
        .enumerate()
        {
            let key = random::key((idx as u64) << 8 | i as u64).unwrap();
            let a = random::normal::<f32>(&shape, None, None, Some(&key))
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            a.eval().unwrap();
            w.insert(format!("blocks.{idx}.attn.{name}"), a);
        }
    }

    /// F-012 pin: `Seedvr2Pipeline::load`'s cast-then-convert order makes the duplicated
    /// `_vid`/`_txt` keys of a shared block share ONE materialized buffer, where the previous
    /// convert-then-cast order materialized two (each key got its own `as_dtype` node).
    #[test]
    fn cast_then_convert_shares_one_buffer_for_duplicated_keys() {
        let mut raw = Weights::empty();
        synth_shared_attn_block(&mut raw, 10, 64, 2, 32);

        // Exercise the exact preparation function used by `Seedvr2Pipeline::load`.
        let shared = prepare_dit(&raw, Dtype::Float32).unwrap();
        for sub in ["proj_qkv", "proj_out"] {
            let v = shared
                .require(&format!("blocks.10.attn.{sub}_vid.weight"))
                .unwrap();
            let t = shared
                .require(&format!("blocks.10.attn.{sub}_txt.weight"))
                .unwrap();
            assert_eq!(
                v.as_slice::<f32>().as_ptr(),
                t.as_slice::<f32>().as_ptr(),
                "{sub}: _vid/_txt must share one buffer after cast-then-convert"
            );
        }

        // the pre-F-012 order (convert, then per-key cast) demonstrably duplicates the buffer —
        // this is the waste the reorder removes; if MLX ever starts CSE-ing identical cast nodes
        // this assertion (not the fix) is what becomes stale.
        let dup = cast_weights(&convert::convert_dit(&raw).unwrap(), Dtype::Float32).unwrap();
        let v = dup.require("blocks.10.attn.proj_qkv_vid.weight").unwrap();
        let t = dup.require("blocks.10.attn.proj_qkv_txt.weight").unwrap();
        assert_ne!(
            v.as_slice::<f32>().as_ptr(),
            t.as_slice::<f32>().as_ptr(),
            "convert-then-cast materializes two buffers (the F-012 waste)"
        );
    }

    #[test]
    fn prepare_dit_preserves_file_backed_projection_across_shared_streams() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.safetensors");
        let base = "blocks.10.attn.proj_out.all";
        let header = serde_json::json!({
            (format!("{base}.weight")): {
                "dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]
            },
            (format!("{base}.weight_scale")): {
                "dtype": "F32", "shape": [], "data_offsets": [16, 20]
            }
        });
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0x38u8; 16]);
        let scale = 0.251_234_56_f32;
        bytes.extend(scale.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        let scope = mlx_gen::gen_core::CheckpointReceiptScope::default();
        let prepared = scope
            .with(|| prepare_dit(&Weights::from_file(&path).unwrap(), Dtype::Bfloat16))
            .unwrap();
        let vid_key = "blocks.10.attn.proj_out_vid.weight";
        let txt_key = "blocks.10.attn.proj_out_txt.weight";
        if mlx_gen::metal_available() {
            let vid = prepared.source_projection(vid_key).unwrap().unwrap();
            let txt = prepared.source_projection(txt_key).unwrap().unwrap();
            let vid = vid.operands();
            let txt = txt.operands();
            assert_eq!(vid[0].1.as_slice::<u8>(), &[0x38; 16]);
            assert_eq!(
                vid[0].1.as_slice::<u8>().as_ptr(),
                txt[0].1.as_slice::<u8>().as_ptr()
            );
            assert_eq!(vid[1].1.as_slice::<f32>(), &[scale]);
            assert_eq!(txt[1].1.as_slice::<f32>(), &[scale]);
        } else {
            assert_eq!(prepared.peek(vid_key).unwrap().dtype(), Dtype::Bfloat16);
            assert_eq!(prepared.peek(txt_key).unwrap().dtype(), Dtype::Bfloat16);
        }
        prepared.materialize().unwrap();
        let facts = scope.facts().expect("a measured physical projection");
        assert_eq!(facts.receipt().tensor_count, 1);
        assert_eq!(facts.receipt().source_bytes, 20);
    }

    /// F-012 measurement harness (synthetic tensors at the real 3B shapes — the real weights are
    /// multi-GB and not fetched in CI): resident bytes of the 22 shared blocks' attention tensors
    /// under the old (convert→cast) vs new (cast→convert) order, at bf16 (the default load dtype).
    /// Expected: old ≈ 2.31 GB, new ≈ 1.15 GB, delta ≈ 1.15 GB. `#[ignore]`: allocates ~3.5 GB
    /// transiently; run manually with
    /// `cargo test --locked -p mlx-gen-seedvr2 --lib f012 -- --ignored --nocapture`.
    #[test]
    #[ignore = "multi-GB measurement harness; run manually with --ignored"]
    fn f012_measure_3b_shared_attn_resident_bytes() {
        use mlx_rs::memory::{clear_cache, get_active_memory};

        // The 3B config: vid_dim 2560, 20 heads × 128, layers 10..32 shared (22 blocks).
        let mut raw = Weights::empty();
        for idx in 10..32 {
            synth_shared_attn_block(&mut raw, idx, 2560, 20, 128);
        }
        // one shared-block attn set = qkv (3·2560·2560) + out (2560²+2560) + 2 norms (128 each)
        let per_block_params: usize = 3 * 2560 * 2560 + 2560 * 2560 + 2560 + 128 + 128;
        let shared_bytes_bf16 = 22 * per_block_params * 2; // 1_153_557_504 ≈ 1.15 GB

        let eval_all = |w: &Weights| {
            let arrays: Vec<&Array> = w.keys().map(|k| w.get(k).unwrap()).collect();
            eval(arrays).unwrap();
        };

        clear_cache();
        let base = get_active_memory();
        let old = cast_weights(&convert::convert_dit(&raw).unwrap(), Dtype::Bfloat16).unwrap();
        eval_all(&old);
        clear_cache();
        let old_bytes = get_active_memory().saturating_sub(base);
        drop(old);
        clear_cache();

        let base = get_active_memory();
        let new = convert::convert_dit(&cast_weights(&raw, Dtype::Bfloat16).unwrap()).unwrap();
        eval_all(&new);
        clear_cache();
        let new_bytes = get_active_memory().saturating_sub(base);

        eprintln!(
            "[F-012] shared-attn resident: old(convert→cast)={old_bytes} B ({:.3} GB)  \
             new(cast→convert)={new_bytes} B ({:.3} GB)  delta={} B ({:.3} GB)  \
             expected shared set={shared_bytes_bf16} B",
            old_bytes as f64 / 1e9,
            new_bytes as f64 / 1e9,
            old_bytes.saturating_sub(new_bytes),
            old_bytes.saturating_sub(new_bytes) as f64 / 1e9,
        );
        let tol = shared_bytes_bf16 / 20; // 5% allocator-granularity slack
        assert!(
            new_bytes.abs_diff(shared_bytes_bf16) < tol,
            "new order must hold ~one bf16 copy: {new_bytes} vs {shared_bytes_bf16}"
        );
        assert!(
            old_bytes
                .saturating_sub(new_bytes)
                .abs_diff(shared_bytes_bf16)
                < tol,
            "reorder must save ~one bf16 copy (~1.15 GB): old {old_bytes} new {new_bytes}"
        );
    }
}

#[cfg(test)]
mod driver_conformance_tests {
    use super::*;
    use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

    fn make_constant(val: f32) -> Array {
        Array::from_slice(&[val; 16 * 2 * 2], &[1, 16, 1, 2, 2])
    }

    fn eval_and_vec(a: &Array) -> Vec<f32> {
        a.eval().unwrap();
        a.as_slice::<f32>().to_vec()
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
                Box::new(Seedvr2Driver::for_units(4, |_step, noise: &Array| {
                    Ok(multiply(noise, Array::from_f32(0.25))?)
                }))
            },
            make_initial: || make_constant(1.0),
            latents_equal: |a, b| {
                let va = eval_and_vec(a);
                let vb = eval_and_vec(b);
                va.len() == vb.len() && va.iter().zip(vb).all(|(x, y)| x.to_bits() == y.to_bits())
            },
        })
        .expect("MLX Seedvr2Driver passes Tier 0 conformance");
        assert_eq!(report.driver_id, "mlx_seedvr2");
        assert_eq!(report.total_steps, 4);
        assert!(report.verified_cancellation);
    }

    #[test]
    fn seedvr2_driver_computes_one_step_euler_restoration_and_unit_cancellation() {
        // 1-step Euler: x_clean = noise - dit_out -> 1.0 - 0.375 = 0.625.
        let driver = Seedvr2Driver::for_units(1, |_unit, _noise: &Array| Ok(make_constant(0.375)));
        let cancel = CancelFlag::new();
        let mut st = DriverRunState::new();
        let mut on_progress = |_| {};
        let mut hooks = StepHooks::new(&mut on_progress);
        let out = run_mlx_driver_slices(
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
        let BoundaryState::XSigma(arr) = out else {
            panic!("expected XSigma");
        };
        let vals = eval_and_vec(&arr);
        assert!(vals.iter().all(|&v| (v - 0.625).abs() < 1e-6));

        // Mid-unit cancel on 1-step restoration stops cleanly after step progress emits.
        let c1 = CancelFlag::new();
        let c1_hook = c1.clone();
        let mut st1 = DriverRunState::new();
        let mut ticks = 0usize;
        let mut on_p1 = |_: Progress| {
            ticks += 1;
            c1_hook.cancel();
        };
        let mut hooks1 = StepHooks::new(&mut on_p1);
        let canceled = run_mlx_driver_slices(
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
        assert!(matches!(canceled, Err(Error::Canceled)));
        assert_eq!(ticks, 1);
    }
}
