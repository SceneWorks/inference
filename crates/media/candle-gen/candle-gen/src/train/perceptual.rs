//! The **shared decoded-x0 perceptual auxiliary-loss path** for the Candle trainers (epic 2123 E8,
//! sc-24830) — the Candle twin of `mlx_gen::train::perceptual` (sc-2125), same shape, same policy.
//!
//! Every auxiliary perceptual loss of the perceptual-character-LoRA epic — depth anchoring, ArcFace
//! identity + face landmarks, body losses, VAE-anchor / E-LatentLPIPS — runs through this module:
//!
//! 1. **Recover x0** from the model prediction ([`Parameterization::recover_x0`]).
//! 2. **Decode x0** with the family's small differentiable decoder ([`X0Decoder`]; the TAESD family
//!    in [`super::tae::TinyDecoder`], TAEHV for the video-VAE families). Losses whose
//!    [`PerceptualLoss::input`] is [`PerceptualInput::Latents`] skip the decode, so they also run on
//!    a family with no decoder.
//! 3. **Build a per-image reference once per job** ([`PerceptualPath::ensure_reference`]) from the
//!    trainer's cached clean latent (decoded round trip for pixel losses), in the loss's own
//!    reference type, or `Ok(None)` ⇒ the image is unusable for that loss and is skipped.
//! 4. **Run the frozen model on the live x0** ([`PerceptualLoss::loss`]) — differentiable in the
//!    live input; the model's weights are plain (untracked) tensors, so `backward()` only produces
//!    adapter gradients.
//! 5. **Schedule**: the backend-neutral policy in [`gen_core::train::aux_schedule`] (re-exported):
//!    [`AuxAlternation`], [`plan_step`], [`StepPlan`]; [`PerceptualPath::plan`] applies the per-image
//!    skips.
//!
//! Candle autograd note: fused candle ops built with `apply_op*_no_bwd` (`candle_nn::ops::layer_norm`,
//! `rms_norm`, `softmax_last_dim`) and `upsample_bilinear2d` have **no backward** — a perceptual
//! model on this path must use composable ops or the gradient silently stops at that op.
//!
//! Trainers do not assemble losses themselves: the shared builder (`candle-gen-perceptual`'s
//! `build_perceptual_path`) turns a `TrainingConfig` into a ready [`PerceptualPath`].

use std::any::Any;
use std::collections::HashMap;

use candle_core::{Tensor, D};

pub use crate::gen_core::train::aux_schedule::{
    combine_step_terms, perceptual_footprint_bytes, plan_step, AuxAlternation, AuxModelFootprint,
    StepPlan,
};
pub use crate::gen_core::train::AuxLossSchedule;

use super::tae::TinyDecoder;
use crate::{CandleError, Result};

/// How a trainer's model output relates to the clean sample, with the per-step noise scalar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Parameterization {
    /// Flow matching where the model regresses `noise − x0` (`x_t = (1−σ)·x0 + σ·noise`):
    /// `x0 = x_t − σ·pred`.
    FlowNoiseMinusX0 { sigma: f32 },
    /// Flow matching where the model regresses `x0 − noise`: `x0 = x_t + σ·pred`.
    FlowX0MinusNoise { sigma: f32 },
    /// ε-prediction: `x0 = (x_t − √(1−ᾱ)·ε) / √ᾱ`.
    Epsilon { alpha_bar: f32 },
    /// v-prediction: `x0 = √ᾱ·x_t − √(1−ᾱ)·v`.
    VPrediction { alpha_bar: f32 },
}

impl Parameterization {
    /// Recover the model's x0 estimate from the noisy input and its prediction (same dtype/shape).
    /// Pure tensor ops — differentiable in `pred`.
    pub fn recover_x0(&self, x_t: &Tensor, pred: &Tensor) -> Result<Tensor> {
        Ok(match *self {
            Self::FlowNoiseMinusX0 { sigma } => (x_t - (pred * sigma as f64)?)?,
            Self::FlowX0MinusNoise { sigma } => (x_t + (pred * sigma as f64)?)?,
            Self::Epsilon { alpha_bar } => {
                let sa = (alpha_bar.max(1e-8) as f64).sqrt();
                let s1 = ((1.0 - alpha_bar).max(0.0) as f64).sqrt();
                ((x_t - (pred * s1)?)? / sa)?
            }
            Self::VPrediction { alpha_bar } => {
                let sa = (alpha_bar.max(0.0) as f64).sqrt();
                let s1 = ((1.0 - alpha_bar).max(0.0) as f64).sqrt();
                ((x_t * sa)? - (pred * s1)?)?
            }
        })
    }
}

/// A frozen, differentiable latent → pixel decoder for the decoded-x0 losses.
pub trait X0Decoder: Send + Sync {
    /// Model-space latents NCHW `[B, C, h, w]` → pixels NHWC `[B, H, W, 3]` in `[0, 1]`,
    /// differentiable in `latents`.
    fn decode(&self, latents: &Tensor) -> Result<Tensor>;
}

impl X0Decoder for TinyDecoder {
    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        TinyDecoder::decode(self, latents)
    }
}

/// What a [`PerceptualLoss`] consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerceptualInput {
    /// The decoded x0 pixels (NHWC `[B, H, W, 3]`, `[0, 1]`) — depth, identity, body losses.
    DecodedPixels,
    /// The x0 latent itself (NCHW) — the latent-space losses, which skip the decode.
    Latents,
}

/// A loss's per-image reference: whatever type the loss defines, read back in
/// [`PerceptualLoss::loss`] with [`reference_as`].
pub type LossReference = Box<dyn Any + Send + Sync>;

/// Borrow a [`LossReference`] as the loss's own type `T` (a mismatch is a loss bug ⇒ error).
pub fn reference_as<'a, T: 'static>(loss: &str, reference: &'a dyn Any) -> Result<&'a T> {
    reference.downcast_ref::<T>().ok_or_else(|| {
        CandleError::Msg(format!(
            "perceptual path: the '{loss}' loss was handed a reference of another type"
        ))
    })
}

/// One frozen auxiliary perceptual model + its comparison.
pub trait PerceptualLoss: Send + Sync {
    /// Short name for errors/diagnostics (e.g. `"depth"`).
    fn name(&self) -> &'static str;
    /// What [`reference`](Self::reference) and [`loss`](Self::loss) consume.
    fn input(&self) -> PerceptualInput {
        PerceptualInput::DecodedPixels
    }
    /// Build an image's reference from its clean input (decoded round trip, or the clean latent
    /// for [`PerceptualInput::Latents`]). Called once per image per job; the input is detached and
    /// the loss must return gradient-free data. `Ok(None)` ⇒ the image is unusable for this loss.
    fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>>;
    /// [`reference`](Self::reference) with the image's **subject mask** in hand — `[H, W]` f32 in
    /// `[0, 1]` at the decoded pixel size (1 = subject), when the trainer has one (sc-24832: the
    /// normal loss restricted to the subject reads it). The default ignores the mask.
    fn reference_with_mask(
        &self,
        clean: &Tensor,
        subject_mask: Option<&Tensor>,
    ) -> Result<Option<LossReference>> {
        let _ = subject_mask;
        self.reference(clean)
    }
    /// The unweighted scalar loss of the live input for one image, given its reference.
    /// Differentiable in `live`.
    fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor>;
}

/// A scheduled auxiliary loss.
pub struct AuxLoss {
    pub schedule: AuxLossSchedule,
    pub loss: Box<dyn PerceptualLoss>,
}

/// The weighted aux contribution of one step, plus each contributing loss's raw value.
pub struct AuxTerms {
    /// `Σ weight_i · loss_i` over the plan's aux losses — the term added to the step loss.
    pub weighted: Tensor,
    /// `(loss index, unweighted loss)` for diagnostics.
    pub per_loss: Vec<(usize, Tensor)>,
}

/// Sum the step's loss terms ([`combine_step_terms`]). Both absent errors.
pub fn combine_step_loss(diffusion: Option<Tensor>, aux: Option<Tensor>) -> Result<Tensor> {
    combine_step_terms(
        diffusion,
        aux,
        |d, a| Ok((d + a)?),
        || {
            CandleError::Msg(
                "perceptual path: a training step with neither a diffusion nor an aux loss term"
                    .into(),
            )
        },
    )
}

/// The trainer-owned perceptual path: decoder + scheduled losses + the per-image reference cache.
pub struct PerceptualPath {
    decoder: Option<Box<dyn X0Decoder>>,
    losses: Vec<AuxLoss>,
    references: HashMap<usize, Vec<Option<LossReference>>>,
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
                return Err(CandleError::Msg(format!(
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

    /// Whether any loss is enabled.
    pub fn is_active(&self) -> bool {
        self.losses.iter().any(|l| l.schedule.is_enabled())
    }

    /// The losses, in index order.
    pub fn losses(&self) -> &[AuxLoss] {
        &self.losses
    }

    fn needs_pixels<'a>(&self, idxs: impl IntoIterator<Item = &'a usize>) -> bool {
        idxs.into_iter()
            .any(|&i| self.losses[i].loss.input() == PerceptualInput::DecodedPixels)
    }

    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.decoder
            .as_ref()
            .ok_or_else(|| CandleError::Msg("perceptual path: no x0 decoder".into()))?
            .decode(latents)
    }

    fn references_of(&self, image: usize) -> Result<&[Option<LossReference>]> {
        self.references
            .get(&image)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                CandleError::Msg(format!(
                    "perceptual path: no cached reference for image {image} (ensure_reference first)"
                ))
            })
    }

    /// Build and cache image `image`'s per-loss references from its clean latent (NCHW, model
    /// space), once per image per job (a repeat call is a no-op). Trainers that cache one latent per
    /// (item, resolution bucket) key it per cache entry. The inputs are detached.
    pub fn ensure_reference(&mut self, image: usize, clean_latents: &Tensor) -> Result<()> {
        self.ensure_reference_with_mask(image, clean_latents, None)
    }

    /// [`ensure_reference`](Self::ensure_reference) with the image's subject mask (`[H, W]` f32 at
    /// the decoded pixel size), handed to every loss's
    /// [`PerceptualLoss::reference_with_mask`] (sc-24832).
    pub fn ensure_reference_with_mask(
        &mut self,
        image: usize,
        clean_latents: &Tensor,
        subject_mask: Option<&Tensor>,
    ) -> Result<()> {
        if self.references.contains_key(&image) {
            return Ok(());
        }
        let clean = clean_latents.detach();
        let all: Vec<usize> = (0..self.losses.len()).collect();
        let pixels = if self.needs_pixels(&all) {
            Some(self.decode(&clean)?.detach())
        } else {
            None
        };
        let mut refs = Vec::with_capacity(self.losses.len());
        for l in &self.losses {
            let input = match l.loss.input() {
                PerceptualInput::DecodedPixels => pixels.as_ref().expect("decoded above"),
                PerceptualInput::Latents => &clean,
            };
            refs.push(l.loss.reference_with_mask(input, subject_mask)?);
        }
        self.references.insert(image, refs);
        self.reference_computations += 1;
        Ok(())
    }

    /// How many images have had their references built this job (each exactly once).
    pub fn reference_computations(&self) -> usize {
        self.reference_computations
    }

    /// Whether loss `loss` has a usable reference for `image`.
    pub fn is_usable(&self, image: usize, loss: usize) -> Result<bool> {
        Ok(self.references_of(image)?[loss].is_some())
    }

    /// The plan for alternation `key` on `image` at sampled noise level `raw_t`: [`plan_step`],
    /// minus the losses this image is unusable for. Requires the image's reference.
    pub fn plan(&self, key: u32, image: usize, raw_t: f32) -> Result<StepPlan> {
        let refs = self.references_of(image)?;
        let schedules: Vec<AuxLossSchedule> = self.losses.iter().map(|l| l.schedule).collect();
        Ok(plan_step(&schedules, key, raw_t).without_skipped(|i| refs[i].is_none()))
    }

    /// The weighted aux term for `plan` on image `image`'s live x0 latent (NCHW, model space, f32),
    /// differentiable in `x0`. `None` when the plan has no aux loss.
    pub fn aux_loss(&self, plan: &StepPlan, image: usize, x0: &Tensor) -> Result<Option<AuxTerms>> {
        if plan.aux.is_empty() {
            return Ok(None);
        }
        let refs = self.references_of(image)?;
        let pixels = if self.needs_pixels(&plan.aux) {
            Some(self.decode(x0)?)
        } else {
            None
        };
        let mut weighted: Option<Tensor> = None;
        let mut per_loss = Vec::with_capacity(plan.aux.len());
        for &i in &plan.aux {
            let l = &self.losses[i];
            let reference = refs[i].as_deref().ok_or_else(|| {
                CandleError::Msg(format!(
                    "perceptual path: the '{}' loss is skipped for image {image} but the plan \
                     names it",
                    l.loss.name()
                ))
            })?;
            let input = match l.loss.input() {
                PerceptualInput::DecodedPixels => pixels.as_ref().expect("decoded above"),
                PerceptualInput::Latents => x0,
            };
            let raw = l.loss.loss(input, reference)?;
            let w = (&raw * l.schedule.weight as f64)?;
            weighted = Some(match weighted {
                Some(acc) => (acc + w)?,
                None => w,
            });
            per_loss.push((i, raw));
        }
        Ok(Some(AuxTerms {
            weighted: weighted.expect("plan.aux is non-empty"),
            per_loss,
        }))
    }
}

// ----------------------------------------------------------------------------------------------
// Depth-consistency losses (MiDaS SSI-L1 + multi-scale gradient matching) — the same port of
// ai-toolkit-perceptual `toolkit/depth_consistency.py` (fork 6e01a6e) as the MLX kit.
// ----------------------------------------------------------------------------------------------

/// Upstream `ssi_weight` default.
pub const DEPTH_SSI_WEIGHT: f32 = 1.0;
/// Upstream `grad_weight` default.
pub const DEPTH_GRAD_WEIGHT: f32 = 0.5;
/// Upstream `grad_scales` default.
pub const DEPTH_GRAD_SCALES: usize = 4;

fn flat_rows(x: &Tensor) -> Result<Tensor> {
    let (b, h, w) = x.dims3().map_err(|_| {
        CandleError::Msg(format!(
            "depth loss expects [B, H, W] maps, got shape {:?}",
            x.dims()
        ))
    })?;
    Ok(x.reshape((b, h * w))?)
}

/// Scale-and-shift-invariant L1 between depth maps `[B, H, W]` (closed-form per-sample `s`, `t`,
/// differentiable in `pred`). Returns `(loss, s, t)` with `s`/`t` (`[B, 1]`) detached.
pub fn ssi_l1(pred: &Tensor, target: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let p = flat_rows(pred)?;
    let g = flat_rows(target)?;
    let mean = |a: &Tensor| a.mean_keepdim(D::Minus1);
    let mean_p = mean(&p)?;
    let mean_g = mean(&g)?;
    let var_p = (mean(&p.sqr()?)? - mean_p.sqr()?)?;
    let cov = (mean(&(&p * &g)?)? - (&mean_p * &mean_g)?)?;
    let s = cov.broadcast_div(&var_p.clamp(1e-6f32, f32::MAX)?)?;
    let t = (&mean_g - (&s * &mean_p)?)?;
    let aligned = p.broadcast_mul(&s)?.broadcast_add(&t)?;
    let loss = (aligned - &g)?.abs()?.mean_all()?;
    Ok((loss, s.detach(), t.detach()))
}

/// 2×2 average pool of `[B, H, W]` (odd trailing row/col dropped, like torch `avg_pool2d`).
fn avg_pool2(x: &Tensor) -> Result<Option<Tensor>> {
    let (b, h, w) = x.dims3()?;
    let (h2, w2) = (h / 2, w / 2);
    if h2 == 0 || w2 == 0 {
        return Ok(None);
    }
    let cropped = x.narrow(1, 0, 2 * h2)?.narrow(2, 0, 2 * w2)?;
    Ok(Some(cropped.reshape((b, h2, 2, w2, 2))?.mean(4)?.mean(2)?))
}

/// Multi-scale L1 gradient-matching loss between depth maps `[B, H, W]`.
pub fn multiscale_grad_loss(pred: &Tensor, target: &Tensor, scales: usize) -> Result<Tensor> {
    let mut p = pred.clone();
    let mut g = target.clone();
    let mut total: Option<Tensor> = None;
    let device = pred.device();
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
        let d = (&p - &g)?;
        let (_, h, w) = d.dims3()?;
        let mut level = Tensor::new(0f32, device)?.to_dtype(d.dtype())?;
        if w > 1 {
            let dx = (d.narrow(2, 1, w - 1)? - d.narrow(2, 0, w - 1)?)?.abs()?;
            level = (level + dx.mean_all()?)?;
        }
        if h > 1 {
            let dy = (d.narrow(1, 1, h - 1)? - d.narrow(1, 0, h - 1)?)?.abs()?;
            level = (level + dy.mean_all()?)?;
        }
        total = Some(match total {
            Some(acc) => (acc + level)?,
            None => level,
        });
    }
    let total = match total {
        Some(t) => t,
        None => Tensor::new(0f32, device)?,
    };
    Ok((total / scales as f64)?)
}

/// The full depth-consistency loss: `DEPTH_SSI_WEIGHT · ssi_l1 + DEPTH_GRAD_WEIGHT ·
/// multiscale_grad(s·pred + t, target)` with `s`/`t` the detached SSI alignment. Invariant to any
/// scale and shift of `pred`.
pub fn depth_consistency_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (ssi, s, t) = ssi_l1(pred, target)?;
    let b = pred.dim(0)?;
    let s3 = s.reshape((b, 1, 1))?;
    let t3 = t.reshape((b, 1, 1))?;
    let aligned = pred.broadcast_mul(&s3)?.broadcast_add(&t3)?;
    let grad = multiscale_grad_loss(&aligned, target, DEPTH_GRAD_SCALES)?;
    Ok(((ssi * DEPTH_SSI_WEIGHT as f64)? + (grad * DEPTH_GRAD_WEIGHT as f64)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn sched(weight: f32, t_min: f32, t_max: f32, every_n: u32) -> AuxLossSchedule {
        AuxLossSchedule {
            weight,
            t_min,
            t_max,
            every_n,
        }
    }

    fn map(seed: u64, shape: &[usize]) -> Tensor {
        // Deterministic pseudo-random values in [0, 1).
        let n: usize = shape.iter().product();
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v: Vec<f32> = (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32) / (1u64 << 24) as f32
            })
            .collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    fn scalar(a: &Tensor) -> f32 {
        a.to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// "Decoder" for the plumbing tests: NCHW → NHWC, unchanged values.
    struct Identity;
    impl X0Decoder for Identity {
        fn decode(&self, latents: &Tensor) -> Result<Tensor> {
            Ok(latents.permute((0, 2, 3, 1))?.contiguous()?)
        }
    }

    /// The toy per-image crop loss (the face/body losses' shape): the reference stores a bounding
    /// box of the bright channel-0 region + the clean crop mean; images without one are unusable.
    struct BrightCrop;
    struct CropRef {
        bbox: (usize, usize, usize, usize),
        mean: Tensor,
    }
    fn crop(px: &Tensor, b: (usize, usize, usize, usize)) -> Tensor {
        px.narrow(1, b.0, b.1 - b.0)
            .unwrap()
            .narrow(2, b.2, b.3 - b.2)
            .unwrap()
    }
    impl PerceptualLoss for BrightCrop {
        fn name(&self) -> &'static str {
            "bright-crop"
        }
        fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
            let v = clean.i_ch0()?;
            let (h, w) = (v.len(), v[0].len());
            let (mut y0, mut y1, mut x0, mut x1) = (h, 0, w, 0);
            for (y, row) in v.iter().enumerate() {
                for (x, &p) in row.iter().enumerate() {
                    if p > 0.5 {
                        (y0, y1, x0, x1) = (y0.min(y), y1.max(y + 1), x0.min(x), x1.max(x + 1));
                    }
                }
            }
            if y1 == 0 {
                return Ok(None);
            }
            let bbox = (y0, y1, x0, x1);
            let mean = crop(clean, bbox).mean_all()?;
            Ok(Some(Box::new(CropRef { bbox, mean })))
        }
        fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor> {
            let r = reference_as::<CropRef>(self.name(), reference)?;
            let m = crop(live, r.bbox).mean_all()?;
            Ok((m - &r.mean)?.sqr()?)
        }
    }
    trait Ch0 {
        fn i_ch0(&self) -> Result<Vec<Vec<f32>>>;
    }
    impl Ch0 for Tensor {
        fn i_ch0(&self) -> Result<Vec<Vec<f32>>> {
            Ok(self.get(0)?.narrow(2, 0, 1)?.squeeze(2)?.to_vec2::<f32>()?)
        }
    }

    /// A clean NCHW image `[1, 3, 6, 6]` of 0.1 with a bright 2×3 patch (rows 1..3, cols 2..5) in
    /// channel 0 when `face`.
    fn clean_image(face: bool) -> (Tensor, Vec<f32>) {
        let mut v = vec![0.1f32; 3 * 36];
        if face {
            for y in 1..3 {
                for x in 2..5 {
                    v[y * 6 + x] = 0.9;
                }
            }
        }
        (
            Tensor::from_vec(v.clone(), (1, 3, 6, 6), &Device::Cpu).unwrap(),
            v,
        )
    }

    fn crop_path() -> PerceptualPath {
        PerceptualPath::new(
            Some(Box::new(Identity)),
            vec![AuxLoss {
                schedule: sched(0.5, 0.0, 1.0, 2),
                loss: Box::new(BrightCrop),
            }],
        )
        .unwrap()
    }

    /// A crop loss applies its reference box to the LIVE decode. Mutation: crop the live pixels
    /// with the full frame instead of `r.bbox` ⇒ the outside change moves the loss ⇒ red.
    #[test]
    fn a_crop_loss_applies_its_reference_box_to_the_live_decode() {
        let mut path = crop_path();
        let (clean, v) = clean_image(true);
        path.ensure_reference(0, &clean).unwrap();
        let plan = path.plan(2, 0, 0.5).unwrap();
        assert!(!plan.diffusion && plan.aux == vec![0], "{plan:?}");
        let at = |v: &[f32]| {
            let x0 = Tensor::from_vec(v.to_vec(), (1, 3, 6, 6), &Device::Cpu).unwrap();
            scalar(&path.aux_loss(&plan, 0, &x0).unwrap().unwrap().weighted)
        };
        assert!(at(&v).abs() < 1e-10, "self loss");
        let mut outside = v.clone();
        outside[5 * 6] = 0.0;
        assert!(
            at(&outside).abs() < 1e-10,
            "outside-box change must not count"
        );
        let mut inside = v.clone();
        inside[6 + 3] = 0.1;
        assert!(at(&inside) > 1e-4, "inside-box change must count");
    }

    /// An unusable image is skipped and its claimed step falls back to diffusion. Mutation: drop
    /// `without_skipped` in `PerceptualPath::plan` ⇒ the plan names the loss ⇒ red.
    #[test]
    fn an_unusable_image_is_skipped_and_falls_back_to_diffusion() {
        let mut path = crop_path();
        let (clean, _) = clean_image(false);
        path.ensure_reference(1, &clean).unwrap();
        assert!(!path.is_usable(1, 0).unwrap());
        let plan = path.plan(2, 1, 0.5).unwrap();
        assert!(plan.diffusion && plan.aux.is_empty(), "{plan:?}");
        assert!(path.aux_loss(&plan, 1, &clean).unwrap().is_none());
        assert!(path.plan(2, 7, 0.5).is_err());
    }

    #[test]
    fn a_pixel_loss_without_a_decoder_is_refused() {
        assert!(PerceptualPath::new(
            None,
            vec![AuxLoss {
                schedule: sched(0.1, 0.0, 1.0, 2),
                loss: Box::new(BrightCrop),
            }],
        )
        .is_err());
    }

    /// A latent-input loss runs with no decoder at all (families without a tiny decoder).
    struct LatentMean;
    impl PerceptualLoss for LatentMean {
        fn name(&self) -> &'static str {
            "latent-mean"
        }
        fn input(&self) -> PerceptualInput {
            PerceptualInput::Latents
        }
        fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
            Ok(Some(Box::new(clean.mean_all()?)))
        }
        fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor> {
            let r = reference_as::<Tensor>(self.name(), reference)?;
            Ok((live.mean_all()? - r)?.sqr()?)
        }
    }

    /// Mutation: make `ensure_reference` decode unconditionally ⇒ "no x0 decoder" error ⇒ red.
    #[test]
    fn a_latent_loss_runs_without_a_decoder() {
        let mut path = PerceptualPath::new(
            None,
            vec![AuxLoss {
                schedule: sched(1.0, 0.0, 1.0, 1),
                loss: Box::new(LatentMean),
            }],
        )
        .unwrap();
        let (clean, _) = clean_image(true);
        path.ensure_reference(0, &clean).unwrap();
        let plan = path.plan(1, 0, 0.5).unwrap();
        assert!(plan.diffusion && plan.aux == vec![0], "{plan:?}");
        let live = (&clean + 0.5).unwrap();
        let l = scalar(&path.aux_loss(&plan, 0, &live).unwrap().unwrap().weighted);
        assert!((l - 0.25).abs() < 1e-5, "{l}");
    }

    /// References are built once per image however often `ensure_reference` is called. Mutation:
    /// drop the `contains_key` early return ⇒ counter 6 ⇒ red.
    #[test]
    fn references_are_computed_once_per_image() {
        let mut path = crop_path();
        let imgs = [clean_image(true).0, clean_image(false).0];
        for _epoch in 0..3 {
            for (i, l) in imgs.iter().enumerate() {
                path.ensure_reference(i, l).unwrap();
            }
        }
        assert_eq!(path.reference_computations(), 2);
        assert!(combine_step_loss(None, None).is_err());
    }

    /// Alternation through the path: with `every_n = 2` an image alternates diffusion / aux on its
    /// own visits. Mutation: key the plan on the global step (`path.plan(step, …)`) with 2 images
    /// round-robin ⇒ image 0 is locked to one kind ⇒ red.
    #[test]
    fn every_image_alternates_on_its_own_visits() {
        let mut path = crop_path();
        for i in 0..2 {
            path.ensure_reference(i, &clean_image(true).0).unwrap();
        }
        let mut alt = AuxAlternation::new(2, 1);
        let mut kinds = vec![Vec::new(), Vec::new()];
        for step in 1..=8u32 {
            let image = ((step - 1) % 2) as usize;
            let plan = path.plan(alt.key(step, image), image, 0.5).unwrap();
            kinds[image].push(plan.diffusion);
        }
        for k in &kinds {
            assert_eq!(k, &vec![true, false, true, false], "{kinds:?}");
        }
    }

    /// The SSI loss (and the full depth loss) is invariant to scale and shift of the predicted
    /// depth. Mutation: replace `aligned = s·p + t` with `aligned = p` in `ssi_l1` ⇒ red.
    #[test]
    fn ssi_and_depth_loss_are_scale_and_shift_invariant() {
        let pred = map(1, &[2, 12, 10]);
        let target = map(2, &[2, 12, 10]);
        let base = scalar(&ssi_l1(&pred, &target).unwrap().0);
        let full = scalar(&depth_consistency_loss(&pred, &target).unwrap());
        assert!(base > 0.01, "non-trivial pair: {base}");
        for (a, b) in [(3.0f64, 0.0f64), (0.25, 7.0), (1.0, -2.5), (-2.0, 1.0)] {
            let moved = pred.affine(a, b).unwrap();
            let l = scalar(&ssi_l1(&moved, &target).unwrap().0);
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

    /// Zero self-loss, and the loss is differentiable in the prediction.
    #[test]
    fn depth_loss_has_a_zero_floor_and_flows_gradient() {
        let target = map(3, &[1, 9, 9]);
        assert!(scalar(&depth_consistency_loss(&target, &target).unwrap()).abs() < 1e-5);
        let p = Var::from_tensor(&map(4, &[1, 9, 9])).unwrap();
        let loss = depth_consistency_loss(p.as_tensor(), &target).unwrap();
        let grads = loss.backward().unwrap();
        let g = grads.get(p.as_tensor()).expect("pred has a gradient");
        assert!(scalar(&g.abs().unwrap().sum_all().unwrap()) > 0.0);
    }

    #[test]
    fn multiscale_grad_loss_sees_structure_not_offset() {
        let t = map(5, &[1, 16, 16]);
        let shifted = (&t + 0.3).unwrap();
        assert!(scalar(&multiscale_grad_loss(&shifted, &t, 4).unwrap()) < 1e-6);
        let other = map(6, &[1, 16, 16]);
        assert!(scalar(&multiscale_grad_loss(&other, &t, 4).unwrap()) > 0.01);
    }

    /// Mutation: flip the sign in any `recover_x0` arm ⇒ red.
    #[test]
    fn recover_x0_inverts_each_parameterisation() {
        let x0 = map(7, &[1, 2, 3, 3]);
        let n = map(8, &[1, 2, 3, 3]);
        let close = |a: &Tensor, b: &Tensor| {
            let d = scalar(&(a - b).unwrap().abs().unwrap().max_all().unwrap());
            assert!(d < 1e-5, "max diff {d}");
        };
        let sigma = 0.3f32;
        let lerp = |a: f64, b: f64| ((&x0 * a).unwrap() + (&n * b).unwrap()).unwrap();
        let x_t = lerp(1.0 - sigma as f64, sigma as f64);
        close(
            &Parameterization::FlowNoiseMinusX0 { sigma }
                .recover_x0(&x_t, &(&n - &x0).unwrap())
                .unwrap(),
            &x0,
        );
        close(
            &Parameterization::FlowX0MinusNoise { sigma }
                .recover_x0(&x_t, &(&x0 - &n).unwrap())
                .unwrap(),
            &x0,
        );
        let ab = 0.64f32;
        let x_t = lerp((ab as f64).sqrt(), (1.0 - ab as f64).sqrt());
        close(
            &Parameterization::Epsilon { alpha_bar: ab }
                .recover_x0(&x_t, &n)
                .unwrap(),
            &x0,
        );
        let vp = ((&n * (ab as f64).sqrt()).unwrap() - (&x0 * (1.0 - ab as f64).sqrt()).unwrap())
            .unwrap();
        close(
            &Parameterization::VPrediction { alpha_bar: ab }
                .recover_x0(&x_t, &vp)
                .unwrap(),
            &x0,
        );
    }
}
