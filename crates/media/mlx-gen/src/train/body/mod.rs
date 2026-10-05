//! **Body losses** on the shared decoded-x0 perceptual path (epic 2123, sc-24832) — MLX port.
//!
//! Three frozen, differentiable models, each behind a [`PerceptualLoss`] with its own per-image
//! [`LossReference`]:
//! - [`BodyProportionLoss`] — [`vitpose::VitPose`] keypoints → bone-length ratios vs the
//!   reference's ratios (visibility-weighted L1 + a missing-keypoint penalty);
//! - [`BodyShapeLoss`] — [`hybrik::HybrikEncoder`] betas of the person crop vs the reference's
//!   betas (L1, gated by their cosine);
//! - [`NormalLoss`] — [`sapiens::SapiensNormal`] unit normals vs the reference's (`(1 − cos) +
//!   L1`), optionally averaged over the subject mask only.
//!
//! Every loss builds its reference with the same ViTPose pass ([`VitPose::detect`]): an image with
//! **no detected person** returns `Ok(None)` and the shared path skips the loss for it (zero
//! contribution, the step falls back to diffusion). The backend-neutral half — configs, parameter
//! counts, footprints and all host-side geometry — is [`gen_core::train::body`].

pub mod hybrik;
pub mod sapiens;
pub mod vitpose;

use std::any::Any;
use std::rc::Rc;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    abs, add, divide, greater, greater_equal, less, logical_and, matmul, maximum, minimum,
    multiply, r#where, subtract,
};
use mlx_rs::Array;

pub use gen_core::train::body::{
    BodyLossesConfig, BodyModelFootprint, HybrikConfig, SapiensConfig, VitPoseConfig,
};
use gen_core::train::body::{
    keypoint_box, MIN_MEAN_RATIO_VISIBILITY, MISSING_REFERENCE_VISIBILITY, NUM_BODY_RATIOS,
    NUM_HEAD_RATIOS, RATIO_VISIBILITY_KEYPOINTS, VIS_THRESHOLD,
};

use super::perceptual::{reference_as, AuxLoss, AuxModelFootprint, LossReference, PerceptualLoss};
use crate::weights::{join, to_f32, Weights};
use crate::{Error, Result};

pub use hybrik::HybrikEncoder;
pub use sapiens::SapiensNormal;
pub use vitpose::VitPose;

// ---------------------------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------------------------

/// `prefix.key`, promoted to f32.
pub(crate) fn w32(w: &Weights, prefix: &str, key: &str) -> Result<Array> {
    to_f32(w.require(&join(prefix, key))?)
}

/// torch OIHW conv weight → MLX OHWI.
pub(crate) fn conv_ohwi(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// An eval-mode BatchNorm folded to `(scale, shift)`: `y = x · scale + shift`.
pub(crate) fn bn_fold(w: &Weights, prefix: &str, bn: &str, eps: f32) -> Result<(Array, Array)> {
    let g = |k: &str| w32(w, prefix, &format!("{bn}.{k}"));
    let scale = divide(
        &g("weight")?,
        &add(&g("running_var")?, Array::from_f32(eps))?.sqrt()?,
    )?;
    let shift = subtract(&g("bias")?, &multiply(&g("running_mean")?, &scale)?)?;
    Ok((scale, shift))
}

/// `(x − mean) / std` over the last (channel) axis of an NHWC map.
pub(crate) fn normalize(x: &Array, mean: [f32; 3], std: [f32; 3]) -> Result<Array> {
    let m = Array::from_slice(&mean, &[1, 1, 1, 3]);
    let s = Array::from_slice(&std, &[1, 1, 1, 3]);
    Ok(divide(&subtract(x, &m)?, &s)?)
}

/// One axis of a separable resample, `[out, in]` (see [`gen_core::train::body::resize_weights`]).
pub struct AxisMatrix(Array);

impl AxisMatrix {
    /// From row-major `[out, in]` weights.
    pub fn from_weights(out: usize, input: usize, w: Vec<f32>) -> Self {
        Self(Array::from_slice(&w, &[out as i32, input as i32]))
    }
    /// torch bilinear resize `input → out`.
    pub fn resize(input: usize, out: usize, align_corners: bool) -> Self {
        Self::from_weights(
            out,
            input,
            gen_core::train::body::resize_weights(input, out, align_corners),
        )
    }
    /// torch `grid_sample` (bilinear, zeros, `align_corners=True`) of `source = o·scale + offset`.
    pub fn affine(input: usize, out: usize, scale: f32, offset: f32) -> Self {
        Self::from_weights(
            out,
            input,
            gen_core::train::body::affine_sample_weights(input, out, scale, offset),
        )
    }
}

/// Separable resample of an NHWC map: `out[b, i, j, c] = Σ ay[i, h] · ax[j, w] · x[b, h, w, c]`
/// — two matmuls, differentiable in `x`.
pub fn resample_nhwc(x: &Array, ay: &AxisMatrix, ax: &AxisMatrix) -> Result<Array> {
    let t = x.transpose_axes(&[0, 3, 1, 2])?; // [B, C, H, W]
    let t = matmul(&ay.0, &t)?; // [B, C, oh, W]
    let t = matmul(&t, ax.0.t())?; // [B, C, oh, ow]
    Ok(t.transpose_axes(&[0, 2, 3, 1])?)
}

// ---------------------------------------------------------------------------------------------
// Bone-length ratios (upstream `_compute_ratios`).
// ---------------------------------------------------------------------------------------------

/// Keypoints `[B, 17, 2]` + confidences `[B, 17]` → `(ratios [B, N], visibility [B, N])`, `N = 8`
/// body ratios (+2 head ratios with `include_head`), exactly upstream's formulas. Differentiable
/// in the keypoints.
pub fn body_ratios(kp: &Array, vis: &Array, include_head: bool) -> Result<(Array, Array)> {
    let point = |i: i32| kp.index((.., i, ..));
    let len = |d: Array| -> Result<Array> {
        Ok(maximum(&d.square()?.sum_axes(&[-1], false)?, Array::from_f32(1e-6))?.sqrt()?)
    };
    let dist = |i: i32, j: i32| len(subtract(point(i), point(j))?);
    let half = Array::from_f32(0.5);
    let avg = |a: Array, b: Array| -> Result<Array> { Ok(multiply(&add(&a, &b)?, &half)?) };
    let upper_arm = avg(dist(5, 7)?, dist(6, 8)?)?;
    let forearm = avg(dist(7, 9)?, dist(8, 10)?)?;
    let thigh = avg(dist(11, 13)?, dist(12, 14)?)?;
    let shin = avg(dist(13, 15)?, dist(14, 16)?)?;
    let shoulder_mid = avg(point(5), point(6))?;
    let hip_mid = avg(point(11), point(12))?;
    let torso = len(subtract(&shoulder_mid, &hip_mid)?)?;
    let shoulder_w = dist(5, 6)?;
    let hip_w = dist(11, 12)?;
    let floor = |a: &Array, f: f32| maximum(a, Array::from_f32(f));
    let height = floor(&add(&add(&torso, &thigh)?, &shin)?, 1e-4)?;
    let mut ratios = vec![
        divide(&upper_arm, &height)?,
        divide(&forearm, &height)?,
        divide(&thigh, &height)?,
        divide(&shin, &height)?,
        divide(&torso, &height)?,
        divide(&shoulder_w, &floor(&hip_w, 1e-4)?)?,
        divide(&upper_arm, &floor(&forearm, 1e-4)?)?,
        divide(&thigh, &floor(&shin, 1e-4)?)?,
    ];
    if include_head {
        let head_h = len(subtract(point(0), &shoulder_mid)?)?;
        ratios.push(divide(&head_h, &height)?);
        ratios.push(divide(&dist(3, 4)?, &floor(&shoulder_w, 1e-4)?)?);
    }
    let n = NUM_BODY_RATIOS + if include_head { NUM_HEAD_RATIOS } else { 0 };
    let mut visibility = Vec::with_capacity(n);
    for group in &RATIO_VISIBILITY_KEYPOINTS[..n] {
        let cols: Vec<Array> = group.iter().map(|&k| vis.index((.., k as i32))).collect();
        let refs: Vec<&Array> = cols.iter().collect();
        visibility.push(mlx_rs::ops::stack_axis(&refs, -1)?.min_axes(&[-1], false)?);
    }
    let r: Vec<&Array> = ratios.iter().collect();
    let v: Vec<&Array> = visibility.iter().collect();
    Ok((
        mlx_rs::ops::stack_axis(&r, -1)?,
        mlx_rs::ops::stack_axis(&v, -1)?,
    ))
}

/// Upstream's `ref_ratios` substitution: a live ratio whose visibility is below
/// [`VIS_THRESHOLD`] is replaced by the (gradient-stopped) reference ratio and its visibility
/// zeroed.
pub fn substitute_low_confidence(
    ratios: &Array,
    vis: &Array,
    reference: &Array,
) -> Result<(Array, Array)> {
    let low = less(vis, Array::from_f32(VIS_THRESHOLD))?;
    Ok((
        r#where(&low, &mlx_rs::stop_gradient(reference)?, ratios)?,
        r#where(&low, Array::from_f32(0.0), vis)?,
    ))
}

/// Upstream's per-sample body-proportion loss (`SDTrainer` body-proportion block, without the
/// `t` weight — the shared path's schedule owns timing): visibility-weighted mean |Δratio| +
/// the fraction of confident reference ratios the live image dropped. Scalar (batch mean).
pub fn proportion_comparison(
    ref_ratios: &Array,
    ref_vis: &Array,
    live_ratios: &Array,
    live_vis: &Array,
) -> Result<Array> {
    let combined = minimum(ref_vis, live_vis)?;
    let num = multiply(&abs(&subtract(live_ratios, ref_ratios)?)?, &combined)?.sum_axes(&[-1], false)?;
    let den = maximum(&combined.sum_axes(&[-1], false)?, Array::from_f32(1e-6))?;
    let high = greater_equal(ref_vis, Array::from_f32(MISSING_REFERENCE_VISIBILITY))?;
    let dropped = logical_and(&high, &less(live_vis, Array::from_f32(VIS_THRESHOLD))?)?;
    let missing = dropped.as_dtype(mlx_rs::Dtype::Float32)?.sum_axes(&[-1], false)?;
    let high_n = maximum(
        &high.as_dtype(mlx_rs::Dtype::Float32)?.sum_axes(&[-1], false)?,
        Array::from_f32(1.0),
    )?;
    Ok(add(&divide(&num, &den)?, &divide(&missing, &high_n)?)?.mean(None)?)
}

// ---------------------------------------------------------------------------------------------
// Person detection (reference time).
// ---------------------------------------------------------------------------------------------

/// What the reference-time ViTPose pass found in an image.
pub struct PersonDetection {
    /// `[1, N]` ratios (gradient-free).
    pub ratios: Array,
    /// `[1, N]` ratio visibilities.
    pub ratio_vis: Array,
    /// The confident keypoints' box `[x1, y1, x2, y2]` in frame pixels.
    pub person_box: [f32; 4],
}

impl VitPose {
    /// Detect the person in a clean decode `[1, H, W, 3]` (outside autograd): `None` when the mean
    /// visibility of the eight body ratios is below [`MIN_MEAN_RATIO_VISIBILITY`] (upstream's
    /// no-body rule) or fewer than two keypoints are confident.
    pub fn detect(&self, clean: &Array, include_head: bool) -> Result<Option<PersonDetection>> {
        let (heatmaps, warp) = self.forward_pixels(clean)?;
        let (coords, conf) = vitpose::heatmaps_to_keypoints(&heatmaps)?;
        let (ratios, ratio_vis) = body_ratios(&coords, &conf, include_head)?;
        let (ratios, ratio_vis) = (
            mlx_rs::stop_gradient(&ratios)?,
            mlx_rs::stop_gradient(&ratio_vis)?,
        );
        mlx_rs::transforms::eval([&ratios, &ratio_vis, &coords, &conf])?;
        let body_vis = ratio_vis
            .index((.., ..NUM_BODY_RATIOS as i32))
            .mean(None)?
            .item::<f32>();
        if body_vis < MIN_MEAN_RATIO_VISIBILITY {
            return Ok(None);
        }
        let k = coords.shape()[1] as usize;
        let c: Vec<f32> = coords.reshape(&[-1])?.as_slice::<f32>().to_vec();
        let points: Vec<(f32, f32)> = (0..k)
            .map(|i| warp.keypoint_to_input(c[2 * i], c[2 * i + 1]))
            .collect();
        let confidence: Vec<f32> = conf.reshape(&[-1])?.as_slice::<f32>().to_vec();
        let sh = clean.shape();
        let Some(person_box) =
            keypoint_box(&points, &confidence, sh[1] as usize, sh[2] as usize)
        else {
            return Ok(None);
        };
        Ok(Some(PersonDetection {
            ratios,
            ratio_vis,
            person_box,
        }))
    }
}

// ---------------------------------------------------------------------------------------------
// The three losses.
// ---------------------------------------------------------------------------------------------

/// The per-image reference of [`BodyProportionLoss`].
pub struct ProportionReference {
    pub ratios: Array,
    pub ratio_vis: Array,
}

/// ViTPose bone-length-ratio loss (upstream body-proportion loss).
pub struct BodyProportionLoss {
    pose: Rc<VitPose>,
    include_head: bool,
}

impl BodyProportionLoss {
    pub fn new(pose: Rc<VitPose>, include_head: bool) -> Self {
        Self { pose, include_head }
    }
}

impl PerceptualLoss for BodyProportionLoss {
    fn name(&self) -> &'static str {
        "body-proportion"
    }

    fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
        Ok(self.pose.detect(clean, self.include_head)?.map(|d| {
            Box::new(ProportionReference {
                ratios: d.ratios,
                ratio_vis: d.ratio_vis,
            }) as LossReference
        }))
    }

    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<ProportionReference>(self.name(), reference)?;
        let (heatmaps, _) = self.pose.forward_pixels(live)?;
        let (coords, conf) = vitpose::heatmaps_to_keypoints(&heatmaps)?;
        let (ratios, vis) = body_ratios(&coords, &conf, self.include_head)?;
        let (ratios, vis) = substitute_low_confidence(&ratios, &vis, &r.ratios)?;
        proportion_comparison(&r.ratios, &r.ratio_vis, &ratios, &vis)
    }
}

/// The per-image reference of [`BodyShapeLoss`].
pub struct ShapeReference {
    /// `[1, 10]` reference betas.
    pub betas: Array,
    /// The square person crop `(y0, y1, x0, x1)` applied to the live decode too.
    pub crop: (usize, usize, usize, usize),
}

/// HybrIK SMPL-beta loss (upstream body-shape loss).
pub struct BodyShapeLoss {
    pose: Rc<VitPose>,
    hybrik: HybrikEncoder,
    min_cos: f32,
}

impl BodyShapeLoss {
    pub fn new(pose: Rc<VitPose>, hybrik: HybrikEncoder, min_cos: f32) -> Self {
        Self {
            pose,
            hybrik,
            min_cos,
        }
    }
}

/// Upstream's body-shape comparison: `mean |live − ref|`, counted only while the (gradient-free)
/// cosine of the two beta vectors exceeds `min_cos` (else zero). Scalar.
pub fn shape_comparison(reference: &Array, live: &Array, min_cos: f32) -> Result<Array> {
    let dot = multiply(reference, live)?.sum_axes(&[-1], false)?;
    let norms = multiply(
        &reference.square()?.sum_axes(&[-1], false)?.sqrt()?,
        &live.square()?.sum_axes(&[-1], false)?.sqrt()?,
    )?;
    let cos = mlx_rs::stop_gradient(divide(&dot, &maximum(&norms, Array::from_f32(1e-8))?)?)?;
    let gate = greater(&cos, Array::from_f32(min_cos))?.as_dtype(mlx_rs::Dtype::Float32)?;
    let l1 = abs(&subtract(live, reference)?)?.mean_axes(&[-1], false)?;
    Ok(multiply(&l1, &gate)?.mean(None)?)
}

impl PerceptualLoss for BodyShapeLoss {
    fn name(&self) -> &'static str {
        "body-shape"
    }

    fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
        let Some(d) = self.pose.detect(clean, false)? else {
            return Ok(None);
        };
        let sh = clean.shape();
        let crop = HybrikEncoder::crop_for(d.person_box, sh[1] as usize, sh[2] as usize);
        let betas = mlx_rs::stop_gradient(self.hybrik.forward_crop(clean, crop)?)?;
        betas.eval()?;
        Ok(Some(Box::new(ShapeReference { betas, crop })))
    }

    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<ShapeReference>(self.name(), reference)?;
        shape_comparison(&r.betas, &self.hybrik.forward_crop(live, r.crop)?, self.min_cos)
    }
}

/// The per-image reference of [`NormalLoss`].
pub struct NormalReference {
    /// `[1, S, S, 3]` unit normals of the clean decode.
    pub normals: Array,
    /// `[S, S]` subject mask on the normal grid (restricted mode only).
    pub mask: Option<Array>,
}

/// Sapiens surface-normal loss (upstream normal loss).
pub struct NormalLoss {
    pose: Rc<VitPose>,
    sapiens: SapiensNormal,
    restrict_to_subject: bool,
}

impl NormalLoss {
    pub fn new(pose: Rc<VitPose>, sapiens: SapiensNormal, restrict_to_subject: bool) -> Self {
        Self {
            pose,
            sapiens,
            restrict_to_subject,
        }
    }
}

impl PerceptualLoss for NormalLoss {
    fn name(&self) -> &'static str {
        "normal"
    }

    fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
        self.reference_with_mask(clean, None)
    }

    fn reference_with_mask(
        &self,
        clean: &Array,
        subject_mask: Option<&Array>,
    ) -> Result<Option<LossReference>> {
        if self.pose.detect(clean, false)?.is_none() {
            return Ok(None);
        }
        let mask = if self.restrict_to_subject {
            let m = subject_mask.ok_or_else(|| {
                Error::Msg(
                    "normal loss: restricted to the subject but the trainer handed no subject mask \
                     for this image"
                        .into(),
                )
            })?;
            let m = mlx_rs::stop_gradient(self.sapiens.mask_to_normal_grid(m)?)?;
            m.eval()?;
            Some(m)
        } else {
            None
        };
        let normals = mlx_rs::stop_gradient(self.sapiens.forward_pixels(clean)?)?;
        normals.eval()?;
        Ok(Some(Box::new(NormalReference { normals, mask })))
    }

    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<NormalReference>(self.name(), reference)?;
        sapiens::normal_comparison(
            &r.normals,
            &self.sapiens.forward_pixels(live)?,
            r.mask.as_ref(),
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Construction + memory.
// ---------------------------------------------------------------------------------------------

/// The enabled body losses of `cfg`, loaded from their checkpoint directories (ViTPose shared by
/// all three as the person detector), each with its own schedule — the body arm of a trainer's
/// [`super::perceptual::PerceptualPath`]. Empty when every body loss is off (nothing is loaded).
pub fn body_aux_losses(cfg: &BodyLossesConfig) -> Result<Vec<AuxLoss>> {
    if !cfg.any_enabled() {
        return Ok(Vec::new());
    }
    let dir = |d: &Option<std::path::PathBuf>, what: &str| {
        d.clone().ok_or_else(|| {
            Error::Msg(format!(
                "body losses: the {what} checkpoint directory is unset"
            ))
        })
    };
    let load_err = |what: &str, d: &std::path::Path, e: Error| {
        Error::Msg(format!(
            "body losses: could not load {what} from {}: {e}",
            d.display()
        ))
    };
    let pose_dir = dir(&cfg.pose_model_dir, "ViTPose+")?;
    let pose = Rc::new(
        VitPose::from_dir(&pose_dir, VitPoseConfig::plus_base())
            .map_err(|e| load_err("ViTPose+", &pose_dir, e))?,
    );
    let mut out = Vec::new();
    if cfg.proportion.is_enabled() {
        out.push(AuxLoss {
            schedule: cfg.proportion,
            loss: Box::new(BodyProportionLoss::new(pose.clone(), cfg.include_head)),
        });
    }
    if cfg.shape.is_enabled() {
        let d = dir(&cfg.shape_model_dir, "HybrIK")?;
        let hybrik = HybrikEncoder::from_dir(&d, HybrikConfig::resnet34())
            .map_err(|e| load_err("HybrIK", &d, e))?;
        out.push(AuxLoss {
            schedule: cfg.shape,
            loss: Box::new(BodyShapeLoss::new(pose.clone(), hybrik, cfg.shape_min_cos)),
        });
    }
    if cfg.normal.is_enabled() {
        let d = dir(&cfg.normal_model_dir, "Sapiens normal")?;
        let sapiens = SapiensNormal::from_dir(&d, SapiensConfig::normal_0_3b())
            .map_err(|e| load_err("Sapiens normal", &d, e))?;
        out.push(AuxLoss {
            schedule: cfg.normal,
            loss: Box::new(NormalLoss::new(pose, sapiens, cfg.normal_restrict_to_subject)),
        });
    }
    Ok(out)
}

/// The E7 footprints of the enabled body losses' models (see
/// [`gen_core::train::body::body_loss_footprints`]), for
/// [`super::perceptual::perceptual_footprint_bytes`].
pub fn body_loss_footprints(cfg: &BodyLossesConfig) -> Vec<AuxModelFootprint> {
    gen_core::train::body::body_loss_footprints(cfg)
        .into_iter()
        .map(|f| AuxModelFootprint {
            param_bytes: f.param_bytes,
            working_set_bytes: f.working_set_bytes,
            reference_bytes_per_image: f.reference_bytes_per_image,
        })
        .collect()
}

#[cfg(test)]
mod tests;
