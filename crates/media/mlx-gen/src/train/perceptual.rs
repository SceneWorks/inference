//! The **shared decoded-x0 perceptual auxiliary-loss path** (epic 2123 E8, sc-2125).
//!
//! Every auxiliary perceptual loss of the perceptual-character-LoRA epic — depth anchoring
//! (sc-2125, this story), ArcFace identity + face landmarks, ViTPose/HybrIK/Sapiens body losses,
//! and the latent-space VAE-anchor / E-LatentLPIPS losses — runs through this one module instead
//! of re-implementing the plumbing per loss or per trainer:
//!
//! 1. **Recover x0** from the model prediction for the trainer's parameterisation
//!    ([`Parameterization::recover_x0`]).
//! 2. **Decode x0** with the family's small differentiable decoder ([`X0Decoder`]; TAEF1 for the
//!    Flux-VAE 16-channel families — [`super::tae::TinyDecoder`]). Losses whose
//!    [`PerceptualLoss::input`] is [`PerceptualInput::Latents`] skip the decode.
//! 3. **Run the frozen auxiliary model** ([`PerceptualLoss::features`]) — differentiable end to
//!    end in its input, frozen in its own weights (they are captured constants, never trainable
//!    params, so autograd only produces gradients for the adapter factors).
//! 4. **Compare against a per-image reference** cached once per job
//!    ([`PerceptualPath::ensure_reference`]): the reference is the same features computed from
//!    the training image's own encode→decode round trip (the trainer's cached clean latent through
//!    the same decoder), so the loss has a true zero floor.
//! 5. **Schedule**: each loss carries its own [`AuxLossSchedule`] (weight, inclusive noise-level
//!    window, alternation period); [`plan_step`] turns the schedules into a per-step
//!    [`StepPlan`] — which terms contribute this step. On an aux-only step the diffusion loss
//!    contributes **zero**.
//!
//! ## How a trainer uses it
//! ```text
//! let mut path = PerceptualPath::new(Some(Box::new(taef1)), vec![AuxLoss { schedule, loss }])?;
//! for (i, latent) in cached_latents { path.ensure_reference(i, &latent)?; }   // once per image
//! // per step:
//! let plan = path.plan(step, sigma);
//! // inside the traced loss closure:
//! let x0 = Parameterization::FlowNoiseMinusX0 { sigma }.recover_x0(&x_t, &pred)?;
//! let aux = path.aux_loss(&plan, image_idx, &x0_nchw)?;                       // weighted sum
//! let total = combine_step_loss(plan.diffusion.then_some(diffusion), aux.map(|a| a.weighted))?;
//! ```
//!
//! ## How a new aux loss plugs in
//! Implement [`PerceptualLoss`] (name, input kind, frozen-model `features`, `compare`, and the two
//! memory figures), give it an [`AuxLossSchedule`], and push an [`AuxLoss`] onto the trainer's
//! [`PerceptualPath`]. Decode, reference caching, the timestep window, alternation, and the
//! memory estimate (E7) come for free.

use std::collections::HashMap;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{abs, add, divide, maximum, multiply, subtract};
use mlx_rs::Array;

pub use gen_core::train::AuxLossSchedule;

use super::tae::TinyDecoder;
use crate::{Error, Result};

/// How a trainer's model output relates to the clean sample, with the per-step noise scalar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Parameterization {
    /// Flow matching where the model regresses `noise − x0` (Z-Image's negated forward, the
    /// `x_t = (1−σ)·x0 + σ·noise` convention): `x0 = x_t − σ·pred`.
    FlowNoiseMinusX0 { sigma: f32 },
    /// Flow matching where the model regresses `x0 − noise` (the raw diffusers flow output):
    /// `x0 = x_t + σ·pred`.
    FlowX0MinusNoise { sigma: f32 },
    /// ε-prediction: `x0 = (x_t − √(1−ᾱ)·ε) / √ᾱ`.
    Epsilon { alpha_bar: f32 },
    /// v-prediction: `x0 = √ᾱ·x_t − √(1−ᾱ)·v`.
    VPrediction { alpha_bar: f32 },
}

impl Parameterization {
    /// Recover the model's x0 estimate from the noisy input and its prediction. Pure MLX ops —
    /// differentiable in `pred`.
    pub fn recover_x0(&self, x_t: &Array, pred: &Array) -> Result<Array> {
        let s = |v: f32| Array::from_f32(v);
        Ok(match *self {
            Self::FlowNoiseMinusX0 { sigma } => subtract(x_t, &multiply(pred, s(sigma))?)?,
            Self::FlowX0MinusNoise { sigma } => add(x_t, &multiply(pred, s(sigma))?)?,
            Self::Epsilon { alpha_bar } => {
                let sa = alpha_bar.max(1e-8).sqrt();
                let s1 = (1.0 - alpha_bar).max(0.0).sqrt();
                divide(&subtract(x_t, &multiply(pred, s(s1))?)?, s(sa))?
            }
            Self::VPrediction { alpha_bar } => {
                let sa = alpha_bar.max(0.0).sqrt();
                let s1 = (1.0 - alpha_bar).max(0.0).sqrt();
                subtract(&multiply(x_t, s(sa))?, &multiply(pred, s(s1))?)?
            }
        })
    }
}

/// A frozen, differentiable latent → pixel decoder for the decoded-x0 losses.
pub trait X0Decoder {
    /// Model-space latents NCHW `[B, C, h, w]` → pixels NHWC `[B, H, W, 3]` in `[0, 1]`,
    /// differentiable in `latents`.
    fn decode(&self, latents: &Array) -> Result<Array>;
    /// Resident parameter bytes (trainer memory estimate, E7).
    fn param_bytes(&self) -> u64;
    /// Training working set of one differentiable decode to `out_h × out_w` pixels, in bytes.
    fn training_working_set_bytes(&self, out_h: u32, out_w: u32) -> u64;
}

impl X0Decoder for TinyDecoder {
    fn decode(&self, latents: &Array) -> Result<Array> {
        TinyDecoder::decode(self, latents)
    }
    fn param_bytes(&self) -> u64 {
        TinyDecoder::param_bytes(self)
    }
    fn training_working_set_bytes(&self, out_h: u32, out_w: u32) -> u64 {
        self.config().training_working_set_bytes(out_h, out_w)
    }
}

/// What a [`PerceptualLoss`] consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerceptualInput {
    /// The decoded x0 pixels (NHWC `[B, H, W, 3]`, `[0, 1]`) — depth, identity, body losses.
    DecodedPixels,
    /// The x0 latent itself (NCHW) — the VAE-anchor / E-LatentLPIPS losses, which skip the decode.
    Latents,
}

/// One frozen auxiliary perceptual model + its comparison.
pub trait PerceptualLoss {
    /// Short name for errors/diagnostics (e.g. `"depth"`).
    fn name(&self) -> &'static str;
    /// What [`features`](Self::features) consumes.
    fn input(&self) -> PerceptualInput {
        PerceptualInput::DecodedPixels
    }
    /// Frozen-model features of the input — called with the live x0 (inside the traced loss, so it
    /// must be differentiable in its input) and once per image for the reference.
    fn features(&self, input: &Array) -> Result<Array>;
    /// Scalar loss between the live features and the cached reference (differentiable in `live`).
    fn compare(&self, live: &Array, reference: &Array) -> Result<Array>;
    /// Resident parameter bytes of the frozen model (E7).
    fn param_bytes(&self) -> u64;
    /// Training working set of one differentiable forward + backward, in bytes (E7).
    fn training_working_set_bytes(&self) -> u64;
}

/// A scheduled auxiliary loss.
pub struct AuxLoss {
    pub schedule: AuxLossSchedule,
    pub loss: Box<dyn PerceptualLoss>,
}

/// Which loss terms contribute on one training step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepPlan {
    /// Whether the diffusion loss contributes. `false` ⇔ an aux-only step (the diffusion term
    /// contributes zero).
    pub diffusion: bool,
    /// Indices (into the path's losses) of the aux losses that contribute, ascending.
    pub aux: Vec<usize>,
}

impl StepPlan {
    /// The plain diffusion step (no aux term) — every step when no aux loss is enabled.
    pub fn diffusion_only() -> Self {
        Self {
            diffusion: true,
            aux: Vec::new(),
        }
    }
}

/// Turn per-loss schedules into the plan for 1-based micro-step `step` at noise level `t`:
///
/// - a loss is *live* this step when it is enabled and `t` is inside its window;
/// - if any live loss with `every_n ≥ 2` claims the step (`step % every_n == 0`), the step is
///   **aux-only**: every claiming live loss plus every live `every_n == 1` loss contributes, and the
///   diffusion loss contributes zero;
/// - otherwise the diffusion loss contributes, plus every live `every_n == 1` (summed) loss.
///
/// A claimed step whose `t` falls outside every claiming loss's window therefore trains the
/// diffusion loss (no step is wasted).
pub fn plan_step(schedules: &[AuxLossSchedule], step: u32, t: f32) -> StepPlan {
    let live = |s: &AuxLossSchedule| s.is_enabled() && s.in_window(t);
    let claiming: Vec<usize> = schedules
        .iter()
        .enumerate()
        .filter(|(_, s)| live(s) && s.every_n >= 2 && step.is_multiple_of(s.every_n))
        .map(|(i, _)| i)
        .collect();
    let summed = schedules
        .iter()
        .enumerate()
        .filter(|(_, s)| live(s) && s.every_n == 1)
        .map(|(i, _)| i);
    let mut aux: Vec<usize> = summed.chain(claiming.iter().copied()).collect();
    aux.sort_unstable();
    StepPlan {
        diffusion: claiming.is_empty(),
        aux,
    }
}

/// The weighted aux contribution of one step, plus each contributing loss's raw value.
pub struct AuxTerms {
    /// `Σ weight_i · loss_i` over the plan's aux losses — the term added to the step loss.
    pub weighted: Array,
    /// `(loss index, unweighted loss)` for diagnostics.
    pub per_loss: Vec<(usize, Array)>,
}

/// Sum the step's loss terms: the diffusion loss (when the plan has it) plus the weighted aux sum
/// (when the plan has any aux loss). Both absent is a planning bug and errors.
pub fn combine_step_loss(diffusion: Option<Array>, aux: Option<Array>) -> Result<Array> {
    match (diffusion, aux) {
        (Some(d), Some(a)) => Ok(add(&d, &a)?),
        (Some(d), None) => Ok(d),
        (None, Some(a)) => Ok(a),
        (None, None) => Err(Error::Msg(
            "perceptual path: a training step with neither a diffusion nor an aux loss term".into(),
        )),
    }
}

/// The trainer-owned perceptual path: decoder + scheduled losses + the per-image reference cache.
pub struct PerceptualPath {
    decoder: Option<Box<dyn X0Decoder>>,
    losses: Vec<AuxLoss>,
    /// image index → one reference per loss (same order as `losses`).
    references: HashMap<usize, Vec<Array>>,
    reference_computations: usize,
}

impl PerceptualPath {
    /// Build the path. A loss that consumes decoded pixels requires a decoder.
    pub fn new(decoder: Option<Box<dyn X0Decoder>>, losses: Vec<AuxLoss>) -> Result<Self> {
        if decoder.is_none() {
            if let Some(l) = losses
                .iter()
                .find(|l| l.loss.input() == PerceptualInput::DecodedPixels)
            {
                return Err(Error::Msg(format!(
                    "perceptual path: the '{}' loss decodes x0 but no x0 decoder was provided",
                    l.loss.name()
                )));
            }
        }
        Ok(Self {
            decoder,
            losses,
            references: HashMap::new(),
            reference_computations: 0,
        })
    }

    /// Whether any loss is enabled (otherwise every step is [`StepPlan::diffusion_only`]).
    pub fn is_active(&self) -> bool {
        self.losses.iter().any(|l| l.schedule.is_enabled())
    }

    /// The losses, in index order.
    pub fn losses(&self) -> &[AuxLoss] {
        &self.losses
    }

    /// The plan for 1-based micro-step `step` at noise level `t` ([`plan_step`]).
    pub fn plan(&self, step: u32, t: f32) -> StepPlan {
        let schedules: Vec<AuxLossSchedule> = self.losses.iter().map(|l| l.schedule).collect();
        plan_step(&schedules, step, t)
    }

    fn needs_pixels<'a>(&self, idxs: impl IntoIterator<Item = &'a usize>) -> bool {
        idxs.into_iter()
            .any(|&i| self.losses[i].loss.input() == PerceptualInput::DecodedPixels)
    }

    fn decode(&self, latents: &Array) -> Result<Array> {
        self.decoder
            .as_ref()
            .ok_or_else(|| Error::Msg("perceptual path: no x0 decoder".into()))?
            .decode(latents)
    }

    /// Compute and cache image `image`'s reference features from its clean latent (NCHW, model
    /// space — the trainer's cached VAE encode of the training image), once per image per job: a
    /// second call for the same image is a no-op. The reference is evaluated eagerly and carries no
    /// autograd history.
    pub fn ensure_reference(&mut self, image: usize, clean_latents: &Array) -> Result<()> {
        if self.references.contains_key(&image) {
            return Ok(());
        }
        let all: Vec<usize> = (0..self.losses.len()).collect();
        let pixels = if self.needs_pixels(&all) {
            Some(self.decode(clean_latents)?)
        } else {
            None
        };
        let mut refs = Vec::with_capacity(self.losses.len());
        for l in &self.losses {
            let input = match l.loss.input() {
                PerceptualInput::DecodedPixels => pixels.as_ref().expect("decoded above"),
                PerceptualInput::Latents => clean_latents,
            };
            let f = mlx_rs::stop_gradient(l.loss.features(input)?)?;
            f.eval()?;
            refs.push(f);
        }
        self.references.insert(image, refs);
        self.reference_computations += 1;
        Ok(())
    }

    /// How many images have had their references computed this job (each exactly once).
    pub fn reference_computations(&self) -> usize {
        self.reference_computations
    }

    /// The weighted aux term for `plan` on image `image`'s live x0 latent (NCHW, model space),
    /// differentiable in `x0`. `None` when the plan has no aux loss. Requires
    /// [`ensure_reference`](Self::ensure_reference) for `image` first.
    pub fn aux_loss(&self, plan: &StepPlan, image: usize, x0: &Array) -> Result<Option<AuxTerms>> {
        if plan.aux.is_empty() {
            return Ok(None);
        }
        let refs = self.references.get(&image).ok_or_else(|| {
            Error::Msg(format!(
                "perceptual path: no cached reference for image {image} (ensure_reference first)"
            ))
        })?;
        let pixels = if self.needs_pixels(&plan.aux) {
            Some(self.decode(x0)?)
        } else {
            None
        };
        let mut weighted: Option<Array> = None;
        let mut per_loss = Vec::with_capacity(plan.aux.len());
        for &i in &plan.aux {
            let l = &self.losses[i];
            let input = match l.loss.input() {
                PerceptualInput::DecodedPixels => pixels.as_ref().expect("decoded above"),
                PerceptualInput::Latents => x0,
            };
            let raw = l.loss.compare(&l.loss.features(input)?, &refs[i])?;
            let w = multiply(&raw, Array::from_f32(l.schedule.weight))?;
            weighted = Some(match weighted {
                Some(acc) => add(&acc, &w)?,
                None => w,
            });
            per_loss.push((i, raw));
        }
        Ok(Some(AuxTerms {
            weighted: weighted.expect("plan.aux is non-empty"),
            per_loss,
        }))
    }

    /// Resident parameter bytes of the decoder + every enabled loss's frozen model (E7).
    pub fn param_bytes(&self) -> u64 {
        if !self.is_active() {
            return 0;
        }
        let dec = self.decoder.as_ref().map_or(0, |d| d.param_bytes());
        dec + self
            .losses
            .iter()
            .filter(|l| l.schedule.is_enabled())
            .map(|l| l.loss.param_bytes())
            .sum::<u64>()
    }

    /// Peak extra training memory the path adds at an `out_h × out_w` decode, in bytes (E7):
    /// resident weights, the cached references, plus the decode working set and the largest single
    /// loss working set (the aux terms of one step run inside one traced backward, so they are
    /// summed conservatively).
    pub fn training_footprint_bytes(&self, out_h: u32, out_w: u32, images: usize) -> u64 {
        if !self.is_active() {
            return 0;
        }
        let enabled: Vec<&AuxLoss> = self
            .losses
            .iter()
            .filter(|l| l.schedule.is_enabled())
            .collect();
        let decode = if enabled
            .iter()
            .any(|l| l.loss.input() == PerceptualInput::DecodedPixels)
        {
            self.decoder
                .as_ref()
                .map_or(0, |d| d.training_working_set_bytes(out_h, out_w))
        } else {
            0
        };
        let losses: u64 = enabled
            .iter()
            .map(|l| l.loss.training_working_set_bytes())
            .sum();
        let refs: u64 = self
            .references
            .values()
            .next()
            .map(|r| r.iter().map(|a| a.nbytes() as u64).sum::<u64>())
            .unwrap_or(0)
            * images as u64;
        self.param_bytes() + decode + losses + refs
    }
}

// ----------------------------------------------------------------------------------------------
// Depth-consistency losses (MiDaS SSI-L1 + multi-scale gradient matching), the comparison half of
// depth anchoring — a port of ai-toolkit-perceptual `toolkit/depth_consistency.py`
// (`ssi_l1`, `multiscale_grad_loss`, `compute_depth_consistency_loss`, fork commit 6e01a6e).
// ----------------------------------------------------------------------------------------------

/// Upstream `ssi_weight` default.
pub const DEPTH_SSI_WEIGHT: f32 = 1.0;
/// Upstream `grad_weight` default.
pub const DEPTH_GRAD_WEIGHT: f32 = 0.5;
/// Upstream `grad_scales` default.
pub const DEPTH_GRAD_SCALES: usize = 4;

fn flat_rows(x: &Array) -> Result<Array> {
    let sh = x.shape();
    if sh.len() != 3 {
        return Err(Error::Msg(format!(
            "depth loss expects [B, H, W] maps, got shape {sh:?}"
        )));
    }
    Ok(x.reshape(&[sh[0], sh[1] * sh[2]])?)
}

/// Scale-and-shift-invariant L1 between depth maps `[B, H, W]`: per sample, solve
/// `min_{s,t} ‖s·pred + t − target‖²` in closed form (differentiable in `pred`), then the mean
/// absolute error of the aligned prediction. Returns `(loss, s, t)` with `s`/`t` (`[B, 1]`)
/// gradient-stopped.
pub fn ssi_l1(pred: &Array, target: &Array) -> Result<(Array, Array, Array)> {
    let p = flat_rows(pred)?;
    let g = flat_rows(target)?;
    let mean = |a: &Array| a.mean_axes(&[1], true);
    let mean_p = mean(&p)?;
    let mean_g = mean(&g)?;
    let var_p = subtract(&mean(&multiply(&p, &p)?)?, &multiply(&mean_p, &mean_p)?)?;
    let cov = subtract(&mean(&multiply(&p, &g)?)?, &multiply(&mean_p, &mean_g)?)?;
    let s = divide(&cov, &maximum(&var_p, Array::from_f32(1e-6))?)?;
    let t = subtract(&mean_g, &multiply(&s, &mean_p)?)?;
    let aligned = add(&multiply(&s, &p)?, &t)?;
    let loss = abs(&subtract(&aligned, &g)?)?.mean(None)?;
    Ok((loss, mlx_rs::stop_gradient(&s)?, mlx_rs::stop_gradient(&t)?))
}

/// 2×2 average pool of `[B, H, W]` (odd trailing row/col dropped, like torch `avg_pool2d`).
fn avg_pool2(x: &Array) -> Result<Option<Array>> {
    let sh = x.shape();
    let (b, h, w) = (sh[0], sh[1] / 2, sh[2] / 2);
    if h == 0 || w == 0 {
        return Ok(None);
    }
    let cropped = x.index((.., ..2 * h, ..2 * w));
    Ok(Some(
        cropped
            .reshape(&[b, h, 2, w, 2])?
            .mean_axes(&[2, 4], false)?,
    ))
}

/// Multi-scale L1 gradient-matching loss between depth maps `[B, H, W]` (MiDaS): at each of
/// `scales` 2×-pooled levels, the mean |Δx| + |Δy| of `pred − target`, averaged over levels.
pub fn multiscale_grad_loss(pred: &Array, target: &Array, scales: usize) -> Result<Array> {
    let mut p = pred.clone();
    let mut g = target.clone();
    let mut total: Option<Array> = None;
    for k in 0..scales {
        if k > 0 {
            match (avg_pool2(&p)?, avg_pool2(&g)?) {
                (Some(pp), Some(gg)) => {
                    p = pp;
                    g = gg;
                }
                _ => break,
            }
        }
        let d = subtract(&p, &g)?;
        let sh = d.shape();
        let mut level = Array::from_f32(0.0);
        if sh[2] > 1 {
            let dx = abs(&subtract(d.index((.., .., 1..)), d.index((.., .., ..-1)))?)?;
            level = add(&level, &dx.mean(None)?)?;
        }
        if sh[1] > 1 {
            let dy = abs(&subtract(d.index((.., 1.., ..)), d.index((.., ..-1, ..)))?)?;
            level = add(&level, &dy.mean(None)?)?;
        }
        total = Some(match total {
            Some(acc) => add(&acc, &level)?,
            None => level,
        });
    }
    let total = total.unwrap_or_else(|| Array::from_f32(0.0));
    Ok(divide(&total, Array::from_f32(scales as f32))?)
}

/// The full depth-consistency loss: `DEPTH_SSI_WEIGHT · ssi_l1 + DEPTH_GRAD_WEIGHT ·
/// multiscale_grad(s·pred + t, target)`, with `s`/`t` the gradient-stopped SSI alignment (so the
/// prediction's arbitrary depth scale never leaks into the gradient term). Invariant to any
/// positive or negative scale and any shift of `pred`.
pub fn depth_consistency_loss(pred: &Array, target: &Array) -> Result<Array> {
    let (ssi, s, t) = ssi_l1(pred, target)?;
    let sh = pred.shape();
    let s3 = s.reshape(&[sh[0], 1, 1])?;
    let t3 = t.reshape(&[sh[0], 1, 1])?;
    let aligned = add(&multiply(pred, &s3)?, &t3)?;
    let grad = multiscale_grad_loss(&aligned, target, DEPTH_GRAD_SCALES)?;
    Ok(add(
        &multiply(&ssi, Array::from_f32(DEPTH_SSI_WEIGHT))?,
        &multiply(&grad, Array::from_f32(DEPTH_GRAD_WEIGHT))?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::random;
    use mlx_rs::transforms::{eval, grad};

    fn sched(weight: f32, t_min: f32, t_max: f32, every_n: u32) -> AuxLossSchedule {
        AuxLossSchedule {
            weight,
            t_min,
            t_max,
            every_n,
        }
    }

    fn map(seed: u64, shape: &[i32]) -> Array {
        random::uniform::<_, f32>(0.0f32, 1.0f32, shape, Some(&random::key(seed).unwrap())).unwrap()
    }

    fn scalar(a: &Array) -> f32 {
        eval([a]).unwrap();
        a.item::<f32>()
    }

    /// AC3: the SSI loss (and the full depth loss) is invariant to scale and shift of the
    /// predicted depth. Mutation: replace `aligned = s·p + t` with `aligned = p` in `ssi_l1` ⇒ the
    /// scaled/shifted losses differ ⇒ red.
    #[test]
    fn ssi_and_depth_loss_are_scale_and_shift_invariant() {
        let pred = map(1, &[2, 12, 10]);
        let target = map(2, &[2, 12, 10]);
        let (base, _, _) = ssi_l1(&pred, &target).unwrap();
        let base = scalar(&base);
        let full = scalar(&depth_consistency_loss(&pred, &target).unwrap());
        assert!(
            base > 0.01,
            "a non-trivial pair must have a non-zero loss: {base}"
        );
        for (a, b) in [(3.0f32, 0.0f32), (0.25, 7.0), (1.0, -2.5), (-2.0, 1.0)] {
            let moved = add(
                multiply(&pred, Array::from_f32(a)).unwrap(),
                Array::from_f32(b),
            )
            .unwrap();
            let (l, _, _) = ssi_l1(&moved, &target).unwrap();
            let l = scalar(&l);
            assert!(
                (l - base).abs() <= 1e-4 * base.max(1.0),
                "ssi under ({a}, {b}): {l} vs {base}"
            );
            let f = scalar(&depth_consistency_loss(&moved, &target).unwrap());
            assert!(
                (f - full).abs() <= 1e-4 * full.max(1.0),
                "depth loss under ({a}, {b}): {f} vs {full}"
            );
        }
    }

    /// The self-loss is zero (the cached round-trip reference is a reachable target), and the loss
    /// is differentiable in the prediction.
    #[test]
    fn depth_loss_has_a_zero_floor_and_flows_gradient() {
        let target = map(3, &[1, 9, 9]);
        let self_loss = scalar(&depth_consistency_loss(&target, &target).unwrap());
        assert!(self_loss.abs() < 1e-5, "self loss {self_loss}");
        let f = |p: &Array| -> mlx_rs::error::Result<Array> {
            depth_consistency_loss(p, &target)
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
        };
        let g = grad(f)(&map(4, &[1, 9, 9])).unwrap();
        assert!(scalar(&g.abs().unwrap().sum(None).unwrap()) > 0.0);
    }

    #[test]
    fn multiscale_grad_loss_sees_structure_not_offset() {
        let t = map(5, &[1, 16, 16]);
        let shifted = add(&t, Array::from_f32(0.3)).unwrap();
        assert!(scalar(&multiscale_grad_loss(&shifted, &t, 4).unwrap()) < 1e-6);
        let other = map(6, &[1, 16, 16]);
        assert!(scalar(&multiscale_grad_loss(&other, &t, 4).unwrap()) > 0.01);
    }

    /// AC1 (alternation): with `every_n = 2` the depth steps and diffusion steps alternate; on a
    /// depth step the diffusion term is off. Mutation: `diffusion: true` in `plan_step` ⇒ red.
    #[test]
    fn strict_alternation_pattern() {
        let s = [sched(0.1, 0.0, 1.0, 2)];
        let pattern: Vec<bool> = (1..=6)
            .map(|step| plan_step(&s, step, 0.5).diffusion)
            .collect();
        assert_eq!(pattern, vec![true, false, true, false, true, false]);
        for step in 1..=6 {
            let p = plan_step(&s, step, 0.5);
            assert_eq!(p.aux.is_empty(), p.diffusion, "step {step}: {p:?}");
        }
    }

    #[test]
    fn period_three_window_and_sum_mode() {
        let s = [sched(0.1, 0.2, 0.6, 3)];
        // every 3rd step is aux-only when in window.
        assert!(!plan_step(&s, 3, 0.4).diffusion);
        assert!(plan_step(&s, 4, 0.4).diffusion && plan_step(&s, 4, 0.4).aux.is_empty());
        // a claimed step out of window trains diffusion instead (no wasted step).
        assert_eq!(plan_step(&s, 6, 0.9), StepPlan::diffusion_only());
        // window bounds are inclusive.
        assert!(!plan_step(&s, 3, 0.2).diffusion && !plan_step(&s, 3, 0.6).diffusion);
        // every_n = 1 sums with diffusion on every in-window step.
        let sum = [sched(0.1, 0.0, 1.0, 1)];
        assert_eq!(
            plan_step(&sum, 7, 0.5),
            StepPlan {
                diffusion: true,
                aux: vec![0]
            }
        );
        // off ⇒ diffusion only, always.
        let off = [AuxLossSchedule::OFF];
        for step in 1..=4 {
            assert_eq!(plan_step(&off, step, 0.5), StepPlan::diffusion_only());
        }
        // a summed loss rides along on another loss's aux-only step.
        let both = [sched(0.1, 0.0, 1.0, 2), sched(0.2, 0.0, 1.0, 1)];
        assert_eq!(
            plan_step(&both, 2, 0.5),
            StepPlan {
                diffusion: false,
                aux: vec![0, 1]
            }
        );
    }

    #[test]
    fn recover_x0_inverts_each_parameterisation() {
        let x0 = map(7, &[1, 2, 3, 3]);
        let n = map(8, &[1, 2, 3, 3]);
        let close = |a: &Array, b: &Array| {
            let d = scalar(&subtract(a, b).unwrap().abs().unwrap().max(None).unwrap());
            assert!(d < 1e-5, "max diff {d}");
        };
        let sigma = 0.3f32;
        let lerp = |a: f32, b: f32| {
            add(
                multiply(&x0, Array::from_f32(a)).unwrap(),
                multiply(&n, Array::from_f32(b)).unwrap(),
            )
            .unwrap()
        };
        let x_t = lerp(1.0 - sigma, sigma);
        let v_neg = subtract(&n, &x0).unwrap();
        close(
            &Parameterization::FlowNoiseMinusX0 { sigma }
                .recover_x0(&x_t, &v_neg)
                .unwrap(),
            &x0,
        );
        let v = subtract(&x0, &n).unwrap();
        close(
            &Parameterization::FlowX0MinusNoise { sigma }
                .recover_x0(&x_t, &v)
                .unwrap(),
            &x0,
        );
        let ab = 0.64f32;
        let x_t = lerp(ab.sqrt(), (1.0 - ab).sqrt());
        close(
            &Parameterization::Epsilon { alpha_bar: ab }
                .recover_x0(&x_t, &n)
                .unwrap(),
            &x0,
        );
        let vp = subtract(
            multiply(&n, Array::from_f32(ab.sqrt())).unwrap(),
            multiply(&x0, Array::from_f32((1.0 - ab).sqrt())).unwrap(),
        )
        .unwrap();
        close(
            &Parameterization::VPrediction { alpha_bar: ab }
                .recover_x0(&x_t, &vp)
                .unwrap(),
            &x0,
        );
    }

    /// A latent-input toy loss (mean-squared vs reference) to exercise the cache + plumbing
    /// without a decoder.
    struct LatentMse;
    impl PerceptualLoss for LatentMse {
        fn name(&self) -> &'static str {
            "latent-mse"
        }
        fn input(&self) -> PerceptualInput {
            PerceptualInput::Latents
        }
        fn features(&self, input: &Array) -> Result<Array> {
            Ok(input.clone())
        }
        fn compare(&self, live: &Array, reference: &Array) -> Result<Array> {
            Ok(subtract(live, reference)?.square()?.mean(None)?)
        }
        fn param_bytes(&self) -> u64 {
            0
        }
        fn training_working_set_bytes(&self) -> u64 {
            0
        }
    }

    struct PixelLoss;
    impl PerceptualLoss for PixelLoss {
        fn name(&self) -> &'static str {
            "pixel"
        }
        fn features(&self, input: &Array) -> Result<Array> {
            Ok(input.clone())
        }
        fn compare(&self, live: &Array, reference: &Array) -> Result<Array> {
            Ok(subtract(live, reference)?.abs()?.mean(None)?)
        }
        fn param_bytes(&self) -> u64 {
            0
        }
        fn training_working_set_bytes(&self) -> u64 {
            0
        }
    }

    #[test]
    fn a_pixel_loss_without_a_decoder_is_refused() {
        let r = PerceptualPath::new(
            None,
            vec![AuxLoss {
                schedule: sched(0.1, 0.0, 1.0, 2),
                loss: Box::new(PixelLoss),
            }],
        );
        assert!(r.is_err());
    }

    #[test]
    fn references_are_computed_once_per_image_and_weighted() {
        let mut path = PerceptualPath::new(
            None,
            vec![AuxLoss {
                schedule: sched(0.5, 0.0, 1.0, 2),
                loss: Box::new(LatentMse),
            }],
        )
        .unwrap();
        let lat = [map(10, &[1, 2, 2, 2]), map(11, &[1, 2, 2, 2])];
        for _epoch in 0..3 {
            for (i, l) in lat.iter().enumerate() {
                path.ensure_reference(i, l).unwrap();
            }
        }
        assert_eq!(path.reference_computations(), 2);
        let plan = path.plan(2, 0.5);
        let live = add(&lat[0], Array::from_f32(1.0)).unwrap();
        let terms = path.aux_loss(&plan, 0, &live).unwrap().unwrap();
        assert!((scalar(&terms.weighted) - 0.5).abs() < 1e-5);
        assert!((scalar(&terms.per_loss[0].1) - 1.0).abs() < 1e-5);
        assert!(path
            .aux_loss(&path.plan(1, 0.5), 0, &live)
            .unwrap()
            .is_none());
        // An image with no reference is an error, never a silent zero.
        assert!(path.aux_loss(&plan, 5, &live).is_err());
        assert!(combine_step_loss(None, None).is_err());
    }
}
