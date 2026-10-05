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
//! counts, footprints and all host-side geometry — is [`mlx_gen::gen_core::train::body`].

pub mod hybrik;
pub mod sapiens;
pub mod vitpose;

use std::any::Any;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    abs, add, divide, ge, gt, logical_and, lt, maximum, minimum, multiply, r#where, subtract,
};
use mlx_rs::Array;

pub use mlx_gen::gen_core::train::body::{
    body_arm_footprint, BodyArm, BodyLossesConfig, BodyModelFootprint, HybrikConfig, SapiensConfig,
    TwoTap, VitPoseConfig,
};
use mlx_gen::gen_core::train::body::{
    keypoint_box, VitPoseWarp, MIN_MEAN_RATIO_VISIBILITY, MISSING_REFERENCE_VISIBILITY,
    NUM_BODY_RATIOS, NUM_HEAD_RATIOS, RATIO_VISIBILITY_KEYPOINTS, VIS_THRESHOLD,
};

use mlx_gen::train::perceptual::{reference_as, AuxModelFootprint, LossReference, PerceptualLoss};
use mlx_gen::weights::{join, to_f32, Weights};
use mlx_gen::{Error, Result};

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

/// One axis of a separable resample (see [`mlx_gen::gen_core::train::body::resize_weights`]), in
/// the exact two-tap gather form ([`TwoTap`]) — a dense matmul here ran on a reduced-precision
/// GEMM and missed torch by ~1e-3.
pub struct AxisMatrix {
    i0: Array,
    w0: Array,
    i1: Array,
    w1: Array,
    out: i32,
}

impl AxisMatrix {
    /// From row-major `[out, in]` weights (at most two taps per row).
    pub fn from_weights(out: usize, input: usize, w: Vec<f32>) -> Self {
        let t = TwoTap::from_matrix(&w, out, input);
        let n = out as i32;
        Self {
            i0: Array::from_slice(&t.i0, &[n]),
            w0: Array::from_slice(&t.w0, &[n]),
            i1: Array::from_slice(&t.i1, &[n]),
            w1: Array::from_slice(&t.w1, &[n]),
            out: n,
        }
    }
    /// torch bilinear resize `input → out`.
    pub fn resize(input: usize, out: usize, align_corners: bool) -> Self {
        Self::from_weights(
            out,
            input,
            mlx_gen::gen_core::train::body::resize_weights(input, out, align_corners),
        )
    }
    /// torch `grid_sample` (bilinear, zeros, `align_corners=True`) of `source = o·scale + offset`.
    pub fn affine(input: usize, out: usize, scale: f32, offset: f32) -> Self {
        Self::from_weights(
            out,
            input,
            mlx_gen::gen_core::train::body::affine_sample_weights(input, out, scale, offset),
        )
    }

    /// Resample `axis` of the rank-4 `x`.
    fn apply(&self, x: &Array, axis: i32) -> Result<Array> {
        let mut shape = vec![1i32; 4];
        shape[axis as usize] = self.out;
        let a = multiply(&x.take_axis(&self.i0, axis)?, &self.w0.reshape(&shape)?)?;
        let b = multiply(&x.take_axis(&self.i1, axis)?, &self.w1.reshape(&shape)?)?;
        Ok(add(&a, &b)?)
    }
}

/// Separable resample of an NHWC map: `out[b, i, j, c] = Σ ay[i, h] · ax[j, w] · x[b, h, w, c]`
/// — gathers and weighted adds, exact in f32 and differentiable in `x`.
pub fn resample_nhwc(x: &Array, ay: &AxisMatrix, ax: &AxisMatrix) -> Result<Array> {
    ax.apply(&ay.apply(x, 1)?, 2)
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
    let low = lt(vis, Array::from_f32(VIS_THRESHOLD))?;
    Ok((
        r#where(&low, &mlx_rs::stop_gradient(reference)?, ratios)?,
        r#where(&low, Array::from_f32(0.0), vis)?,
    ))
}

/// Upstream's per-sample body-proportion loss (`SDTrainer` body-proportion block; its `t_ratio`
/// scale is the loss's `timestep_weight`): visibility-weighted mean |Δratio| + the fraction of
/// confident reference ratios the live image dropped. Scalar (batch mean).
pub fn proportion_comparison(
    ref_ratios: &Array,
    ref_vis: &Array,
    live_ratios: &Array,
    live_vis: &Array,
) -> Result<Array> {
    let combined = minimum(ref_vis, live_vis)?;
    let num =
        multiply(&abs(&subtract(live_ratios, ref_ratios)?)?, &combined)?.sum_axes(&[-1], false)?;
    let den = maximum(&combined.sum_axes(&[-1], false)?, Array::from_f32(1e-6))?;
    let high = ge(ref_vis, Array::from_f32(MISSING_REFERENCE_VISIBILITY))?;
    let dropped = logical_and(&high, &lt(live_vis, Array::from_f32(VIS_THRESHOLD))?)?;
    let missing = dropped
        .as_dtype(mlx_rs::Dtype::Float32)?
        .sum_axes(&[-1], false)?;
    let high_n = maximum(
        &high
            .as_dtype(mlx_rs::Dtype::Float32)?
            .sum_axes(&[-1], false)?,
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
    /// Detect the person in a clean decode `[B, H, W, 3]` (outside autograd): `None` when the mean
    /// visibility of all the ratios in use is below [`MIN_MEAN_RATIO_VISIBILITY`] (upstream's
    /// no-body rule) or fewer than two keypoints are confident. The person box spans the confident
    /// keypoints of every frame.
    pub fn detect(&self, clean: &Array, include_head: bool) -> Result<Option<PersonDetection>> {
        let (heatmaps, warp) = self.forward_pixels(clean)?;
        let (coords, conf) = vitpose::heatmaps_to_keypoints(&heatmaps)?;
        let sh = clean.shape();
        detection_from_keypoints(
            &coords,
            &conf,
            &warp,
            include_head,
            sh[1] as usize,
            sh[2] as usize,
        )
    }
}

/// [`VitPose::detect`] past the heatmaps: the `[B, 17, 2]` keypoints (normalized heatmap
/// coordinates) and `[B, 17]` confidences of an `in_h × in_w` frame batch → the detection.
fn detection_from_keypoints(
    coords: &Array,
    conf: &Array,
    warp: &VitPoseWarp,
    include_head: bool,
    in_h: usize,
    in_w: usize,
) -> Result<Option<PersonDetection>> {
    let (ratios, ratio_vis) = body_ratios(coords, conf, include_head)?;
    let (ratios, ratio_vis) = (
        mlx_rs::stop_gradient(&ratios)?,
        mlx_rs::stop_gradient(&ratio_vis)?,
    );
    mlx_rs::transforms::eval([&ratios, &ratio_vis, coords, conf])?;
    // Upstream `encode`: the mean over ALL the ratios in use (10 with the head ratios).
    let mean_vis = ratio_vis.mean(None)?.item::<f32>();
    if mean_vis < MIN_MEAN_RATIO_VISIBILITY {
        return Ok(None);
    }
    // The person box spans every frame's keypoints (a video reference's person moves).
    let c: Vec<f32> = coords.reshape(&[-1])?.as_slice::<f32>().to_vec();
    let points: Vec<(f32, f32)> = (0..c.len() / 2)
        .map(|i| warp.keypoint_to_input(c[2 * i], c[2 * i + 1]))
        .collect();
    let confidence: Vec<f32> = conf.reshape(&[-1])?.as_slice::<f32>().to_vec();
    let Some(person_box) = keypoint_box(&points, &confidence, in_h, in_w) else {
        return Ok(None);
    };
    Ok(Some(PersonDetection {
        ratios,
        ratio_vis,
        person_box,
    }))
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

    /// Upstream's `t_ratio` weighting ([`mlx_gen::gen_core::train::body::body_loss_timestep_weight`]).
    fn timestep_weight(&self, noise_level: f32) -> f32 {
        mlx_gen::gen_core::train::body::body_loss_timestep_weight(noise_level)
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
    let gate = gt(&cos, Array::from_f32(min_cos))?.as_dtype(mlx_rs::Dtype::Float32)?;
    let l1 = abs(&subtract(live, reference)?)?.mean_axes(&[-1], false)?;
    Ok(multiply(&l1, &gate)?.mean(None)?)
}

impl PerceptualLoss for BodyShapeLoss {
    fn name(&self) -> &'static str {
        "body-shape"
    }

    /// Upstream's `t_ratio` weighting ([`mlx_gen::gen_core::train::body::body_loss_timestep_weight`]).
    fn timestep_weight(&self, noise_level: f32) -> f32 {
        mlx_gen::gen_core::train::body::body_loss_timestep_weight(noise_level)
    }

    fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
        let Some(d) = self.pose.detect(clean, false)? else {
            return Ok(None);
        };
        let sh = clean.shape();
        let (h, w) = (sh[1] as usize, sh[2] as usize);
        // Upstream: the reference betas come from `encode` (`int()`-truncated crop), the live ones
        // from `forward` (rounded crop) of the same person box.
        let crop = HybrikEncoder::crop_for(d.person_box, h, w);
        let betas = mlx_rs::stop_gradient(
            self.hybrik
                .forward_crop(clean, HybrikEncoder::encode_crop_for(d.person_box, h, w))?,
        )?;
        betas.eval()?;
        Ok(Some(Box::new(ShapeReference { betas, crop })))
    }

    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<ShapeReference>(self.name(), reference)?;
        shape_comparison(
            &r.betas,
            &self.hybrik.forward_crop(live, r.crop)?,
            self.min_cos,
        )
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

    /// Upstream's `t_ratio` weighting ([`mlx_gen::gen_core::train::body::body_loss_timestep_weight`]).
    fn timestep_weight(&self, noise_level: f32) -> f32 {
        mlx_gen::gen_core::train::body::body_loss_timestep_weight(noise_level)
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
        let normals = mlx_rs::stop_gradient(self.sapiens.encode_pixels(clean)?)?;
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

fn model_dir(d: &Option<PathBuf>, what: &str, knob: &str) -> Result<PathBuf> {
    d.clone().ok_or_else(|| {
        Error::Msg(format!(
            "body losses: the {what} checkpoint directory is unset (body_losses.{knob})"
        ))
    })
}

fn load_err(what: &str, dir: &Path, e: Error) -> Error {
    Error::Msg(format!(
        "body losses: could not load {what} from {}: {e}",
        dir.display()
    ))
}

thread_local! {
    /// The ViTPose every enabled body loss shares (it is the proportion encoder and all three
    /// losses' person detector): loaded once per checkpoint directory while any loss holds it.
    static SHARED_POSE: RefCell<Option<(PathBuf, Weak<VitPose>)>> = const { RefCell::new(None) };
}

/// The shared ViTPose+ base of `cfg.pose_model_dir` — loaded by the first body loss that asks and
/// reused by the others (one resident copy, as [`body_arm_footprint`] budgets).
pub fn shared_pose(cfg: &BodyLossesConfig) -> Result<Rc<VitPose>> {
    let dir = model_dir(&cfg.pose_model_dir, "ViTPose+", "pose_model_dir")?;
    if let Some(p) = SHARED_POSE.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|(d, _)| *d == dir)
            .and_then(|(_, w)| w.upgrade())
    }) {
        return Ok(p);
    }
    let pose = Rc::new(
        VitPose::from_dir(&dir, VitPoseConfig::plus_base())
            .map_err(|e| load_err("ViTPose+", &dir, e))?,
    );
    SHARED_POSE.with(|c| *c.borrow_mut() = Some((dir, Rc::downgrade(&pose))));
    Ok(pose)
}

/// The body-proportion loss of `cfg` (the builder's `body-proportion` arm).
pub fn proportion_loss(cfg: &BodyLossesConfig) -> Result<Box<dyn PerceptualLoss>> {
    Ok(Box::new(BodyProportionLoss::new(
        shared_pose(cfg)?,
        cfg.include_head,
    )))
}

/// The body-shape loss of `cfg` (the builder's `body-shape` arm).
pub fn shape_loss(cfg: &BodyLossesConfig) -> Result<Box<dyn PerceptualLoss>> {
    let d = model_dir(&cfg.shape_model_dir, "HybrIK", "shape_model_dir")?;
    let hybrik = HybrikEncoder::from_dir(&d, HybrikConfig::resnet34())
        .map_err(|e| load_err("HybrIK", &d, e))?;
    Ok(Box::new(BodyShapeLoss::new(
        shared_pose(cfg)?,
        hybrik,
        cfg.shape_min_cos,
    )))
}

/// The normal loss of `cfg` (the builder's `normal` arm).
pub fn normal_loss(cfg: &BodyLossesConfig) -> Result<Box<dyn PerceptualLoss>> {
    let d = model_dir(&cfg.normal_model_dir, "Sapiens normal", "normal_model_dir")?;
    let sapiens = SapiensNormal::from_dir(&d, SapiensConfig::normal_0_3b())
        .map_err(|e| load_err("Sapiens normal", &d, e))?;
    Ok(Box::new(NormalLoss::new(
        shared_pose(cfg)?,
        sapiens,
        cfg.normal_restrict_to_subject,
    )))
}

/// One body arm's E7 footprint ([`body_arm_footprint`]: its model + the shared ViTPose on the
/// first enabled arm).
pub fn arm_footprint(cfg: &BodyLossesConfig, arm: BodyArm) -> AuxModelFootprint {
    let f = body_arm_footprint(cfg, arm);
    AuxModelFootprint {
        param_bytes: f.param_bytes,
        working_set_bytes: f.working_set_bytes,
        reference_bytes_per_image: f.reference_bytes_per_image,
    }
}

#[cfg(test)]
mod tests;
