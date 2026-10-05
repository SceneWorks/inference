//! **Body losses** on the shared decoded-x0 perceptual path (epic 2123, sc-24832) — the candle
//! twin of `mlx-gen-body`.
//!
//! Three frozen, differentiable models behind [`PerceptualLoss`], each with its own per-image
//! [`LossReference`]: [`BodyProportionLoss`] ([`vitpose::VitPose`] bone-length ratios),
//! [`BodyShapeLoss`] ([`hybrik::HybrikEncoder`] SMPL betas of the person crop) and [`NormalLoss`]
//! ([`sapiens::SapiensNormal`] unit normals, optionally subject-masked). Every loss builds its
//! reference with the same ViTPose pass ([`VitPose::detect`]); an image with **no detected person**
//! returns `Ok(None)` and the shared path skips the loss for it. The backend-neutral half — configs,
//! parameter counts, footprints, all host geometry — is [`candle_gen::gen_core::train::body`], so
//! both backends compute the same thing.
//!
//! Candle autograd: LayerNorm is composable (the fused op has no backward), strided convs pad and
//! crop explicitly (candle's Conv2D backward breaks on unequal H/W remainders), resamples are
//! index-selects, so the gradient reaches the decoded pixels and through them the adapter.

pub mod hybrik;
pub mod sapiens;
pub mod vitpose;

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::ops::softmax;
use candle_gen::train::perceptual::{
    reference_as, AuxModelFootprint, LossReference, PerceptualLoss,
};
use candle_gen::weights::Weights;
use candle_gen::{CandleError, Result};

pub use candle_gen::gen_core::train::body::{
    body_arm_footprint, BodyArm, BodyLossesConfig, BodyModelFootprint, HybrikConfig, SapiensConfig,
    TwoTap, VitPoseConfig,
};
use candle_gen::gen_core::train::body::{
    keypoint_box, MIN_MEAN_RATIO_VISIBILITY, MISSING_REFERENCE_VISIBILITY, NUM_BODY_RATIOS,
    NUM_HEAD_RATIOS, RATIO_VISIBILITY_KEYPOINTS, VIS_THRESHOLD,
};

pub use hybrik::HybrikEncoder;
pub use sapiens::SapiensNormal;
pub use vitpose::VitPose;

// ---------------------------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------------------------

/// Every `*.safetensors` in `dir`, merged, f32 on `device`.
pub(crate) fn load_dir(dir: &Path, device: &Device) -> Result<Weights> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| CandleError::Msg(format!("cannot read {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .filter(|p| !candle_gen::gen_core::weightsmeta::is_hidden_file(p))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "no .safetensors checkpoint in {}",
            dir.display()
        )));
    }
    Weights::from_files(&files, device, DType::F32)
}

fn join(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

/// `prefix.key`, as f32.
pub(crate) fn w32(w: &Weights, prefix: &str, key: &str) -> Result<Tensor> {
    Ok(w.require(&join(prefix, key))?.to_dtype(DType::F32)?)
}

/// An eval-mode BatchNorm folded to a per-channel `(scale, shift)` over NCHW.
pub(crate) struct BatchNorm {
    scale: Tensor,
    shift: Tensor,
}

impl BatchNorm {
    pub(crate) fn apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)?)
    }
}

pub(crate) fn bn_fold(w: &Weights, prefix: &str, bn: &str, eps: f32) -> Result<BatchNorm> {
    let g = |k: &str| w32(w, prefix, &format!("{bn}.{k}"));
    let scale = g("weight")?.div(&(g("running_var")? + eps as f64)?.sqrt()?)?;
    let shift = (g("bias")? - g("running_mean")?.mul(&scale)?)?;
    let c = scale.elem_count();
    Ok(BatchNorm {
        scale: scale.reshape((1, c, 1, 1))?,
        shift: shift.reshape((1, c, 1, 1))?,
    })
}

/// NCHW conv with an OIHW kernel. A strided conv pads and crops to the extent its windows read
/// (identical forward values) so candle's Conv2D backward sees zero remainders on both axes.
pub(crate) fn conv2d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let mut xc = x.contiguous()?;
    let mut padding = padding;
    if stride > 1 {
        let (_, _, h, wd) = xc.dims4()?;
        let k = (w.dim(2)?, w.dim(3)?);
        let xp = xc
            .pad_with_zeros(2, padding, padding)?
            .pad_with_zeros(3, padding, padding)?;
        let used = |n: usize, k: usize| ((n + 2 * padding - k) / stride) * stride + k;
        xc = xp
            .narrow(2, 0, used(h, k.0))?
            .narrow(3, 0, used(wd, k.1))?
            .contiguous()?;
        padding = 0;
    }
    let mut y = xc.conv2d(w, padding, stride, 1, 1)?;
    if let Some(b) = bias {
        y = y.broadcast_add(&b.reshape((1, b.elem_count(), 1, 1))?)?;
    }
    Ok(y)
}

/// A dense linear over the last dim (`[out, in]` weight + bias).
pub(crate) fn linear(x: &Tensor, wb: &(Tensor, Tensor)) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let inp = *dims.last().expect("rank >= 1");
    let lead: usize = dims[..dims.len() - 1].iter().product();
    let y = x
        .reshape((lead, inp))?
        .matmul(&wb.0.t()?)?
        .broadcast_add(&wb.1)?;
    let mut out = dims;
    *out.last_mut().unwrap() = wb.0.dim(0)?;
    Ok(y.reshape(out)?)
}

/// Composable LayerNorm over the last dim (candle's fused op has no backward).
pub(crate) fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(xc
        .broadcast_div(&(var + eps)?.sqrt()?)?
        .broadcast_mul(w)?
        .broadcast_add(b)?)
}

/// No-mask scaled-dot-product attention over `[b, h, n, d]`.
pub(crate) fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let attn = (q.contiguous()?.matmul(&kt)? * scale)?;
    Ok(softmax(&attn, D::Minus1)?.matmul(&v.contiguous()?)?)
}

/// `(x − mean) / std` over the channel axis of an NHWC map.
pub(crate) fn normalize(x: &Tensor, mean: [f32; 3], std: [f32; 3]) -> Result<Tensor> {
    let dev = x.device();
    let m = Tensor::from_slice(&mean, (1, 1, 1, 3), dev)?;
    let s = Tensor::from_slice(&std, (1, 1, 1, 3), dev)?;
    Ok(x.broadcast_sub(&m)?.broadcast_div(&s)?)
}

/// One axis of a separable resample in the exact two-tap gather form ([`TwoTap`]).
pub struct AxisMatrix {
    i0: Tensor,
    w0: Tensor,
    i1: Tensor,
    w1: Tensor,
    out: usize,
}

impl AxisMatrix {
    /// From row-major `[out, in]` weights (at most two taps per row).
    pub fn from_weights(out: usize, input: usize, w: Vec<f32>, dev: &Device) -> Result<Self> {
        let t = TwoTap::from_matrix(&w, out, input);
        Ok(Self {
            i0: Tensor::from_vec(t.i0, out, dev)?,
            w0: Tensor::from_vec(t.w0, out, dev)?,
            i1: Tensor::from_vec(t.i1, out, dev)?,
            w1: Tensor::from_vec(t.w1, out, dev)?,
            out,
        })
    }
    /// torch bilinear resize `input → out`.
    pub fn resize(input: usize, out: usize, align_corners: bool, dev: &Device) -> Result<Self> {
        Self::from_weights(
            out,
            input,
            candle_gen::gen_core::train::body::resize_weights(input, out, align_corners),
            dev,
        )
    }
    /// torch `grid_sample` (bilinear, zeros, `align_corners=True`) of `source = o·scale + offset`.
    pub fn affine(input: usize, out: usize, scale: f32, offset: f32, dev: &Device) -> Result<Self> {
        Self::from_weights(
            out,
            input,
            candle_gen::gen_core::train::body::affine_sample_weights(input, out, scale, offset),
            dev,
        )
    }

    fn apply(&self, x: &Tensor, axis: usize) -> Result<Tensor> {
        let mut shape = vec![1usize; 4];
        shape[axis] = self.out;
        let a = x
            .index_select(&self.i0, axis)?
            .broadcast_mul(&self.w0.reshape(shape.clone())?)?;
        let b = x
            .index_select(&self.i1, axis)?
            .broadcast_mul(&self.w1.reshape(shape)?)?;
        Ok((a + b)?)
    }
}

/// Separable resample of an NHWC map (rows then columns), differentiable in `x`.
pub fn resample_nhwc(x: &Tensor, ay: &AxisMatrix, ax: &AxisMatrix) -> Result<Tensor> {
    ax.apply(&ay.apply(&x.contiguous()?, 1)?, 2)
}

// ---------------------------------------------------------------------------------------------
// Bone-length ratios (upstream `_compute_ratios`).
// ---------------------------------------------------------------------------------------------

/// Keypoints `[B, 17, 2]` + confidences `[B, 17]` → `(ratios [B, N], visibility [B, N])` — the
/// MLX twin's formulas exactly.
pub fn body_ratios(kp: &Tensor, vis: &Tensor, include_head: bool) -> Result<(Tensor, Tensor)> {
    let point = |i: usize| -> Result<Tensor> { Ok(kp.narrow(1, i, 1)?.squeeze(1)?) };
    let len = |d: Tensor| -> Result<Tensor> {
        Ok(d.sqr()?.sum(D::Minus1)?.clamp(1e-6f32, f32::MAX)?.sqrt()?)
    };
    let dist = |i: usize, j: usize| -> Result<Tensor> { len((point(i)? - point(j)?)?) };
    let avg = |a: Tensor, b: Tensor| -> Result<Tensor> { Ok(((a + b)? * 0.5)?) };
    let floor = |a: &Tensor, f: f32| -> Result<Tensor> { Ok(a.clamp(f, f32::MAX)?) };
    let upper_arm = avg(dist(5, 7)?, dist(6, 8)?)?;
    let forearm = avg(dist(7, 9)?, dist(8, 10)?)?;
    let thigh = avg(dist(11, 13)?, dist(12, 14)?)?;
    let shin = avg(dist(13, 15)?, dist(14, 16)?)?;
    let shoulder_mid = avg(point(5)?, point(6)?)?;
    let hip_mid = avg(point(11)?, point(12)?)?;
    let torso = len((&shoulder_mid - &hip_mid)?)?;
    let shoulder_w = dist(5, 6)?;
    let hip_w = dist(11, 12)?;
    let height = floor(&((&torso + &thigh)? + &shin)?, 1e-4)?;
    let mut ratios = vec![
        upper_arm.div(&height)?,
        forearm.div(&height)?,
        thigh.div(&height)?,
        shin.div(&height)?,
        torso.div(&height)?,
        shoulder_w.div(&floor(&hip_w, 1e-4)?)?,
        upper_arm.div(&floor(&forearm, 1e-4)?)?,
        thigh.div(&floor(&shin, 1e-4)?)?,
    ];
    if include_head {
        let head_h = len((point(0)? - &shoulder_mid)?)?;
        ratios.push(head_h.div(&height)?);
        ratios.push(dist(3, 4)?.div(&floor(&shoulder_w, 1e-4)?)?);
    }
    let n = NUM_BODY_RATIOS + if include_head { NUM_HEAD_RATIOS } else { 0 };
    let mut visibility = Vec::with_capacity(n);
    for group in &RATIO_VISIBILITY_KEYPOINTS[..n] {
        let cols: Vec<Tensor> = group
            .iter()
            .map(|&k| vis.narrow(1, k, 1))
            .collect::<std::result::Result<_, _>>()?;
        visibility.push(Tensor::cat(&cols, 1)?.min(1)?);
    }
    Ok((
        Tensor::stack(&ratios, D::Minus1)?,
        Tensor::stack(&visibility, D::Minus1)?,
    ))
}

/// Upstream's `ref_ratios` substitution (low-visibility live ratio ← detached reference, its
/// visibility ← 0).
pub fn substitute_low_confidence(
    ratios: &Tensor,
    vis: &Tensor,
    reference: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let low = vis.lt(VIS_THRESHOLD)?;
    Ok((
        low.where_cond(&reference.detach(), ratios)?,
        low.where_cond(&vis.zeros_like()?, vis)?,
    ))
}

/// Upstream's per-sample body-proportion loss (no `t` weight). Scalar (batch mean).
pub fn proportion_comparison(
    ref_ratios: &Tensor,
    ref_vis: &Tensor,
    live_ratios: &Tensor,
    live_vis: &Tensor,
) -> Result<Tensor> {
    let combined = ref_vis.minimum(live_vis)?;
    let num = (live_ratios - ref_ratios)?
        .abs()?
        .mul(&combined)?
        .sum(D::Minus1)?;
    let den = combined.sum(D::Minus1)?.clamp(1e-6f32, f32::MAX)?;
    let high = ref_vis
        .ge(MISSING_REFERENCE_VISIBILITY)?
        .to_dtype(DType::F32)?;
    let low_live = live_vis.lt(VIS_THRESHOLD)?.to_dtype(DType::F32)?;
    let missing = high.mul(&low_live)?.sum(D::Minus1)?;
    let high_n = high.sum(D::Minus1)?.clamp(1f32, f32::MAX)?;
    Ok((num.div(&den)? + missing.div(&high_n)?)?.mean_all()?)
}

// ---------------------------------------------------------------------------------------------
// Person detection (reference time).
// ---------------------------------------------------------------------------------------------

/// What the reference-time ViTPose pass found in an image.
pub struct PersonDetection {
    pub ratios: Tensor,
    pub ratio_vis: Tensor,
    pub person_box: [f32; 4],
}

impl VitPose {
    /// Detect the person in a clean decode `[1, H, W, 3]`: `None` below
    /// [`MIN_MEAN_RATIO_VISIBILITY`] mean body-ratio visibility or with fewer than two confident
    /// keypoints (the MLX twin's rule).
    pub fn detect(&self, clean: &Tensor, include_head: bool) -> Result<Option<PersonDetection>> {
        let clean = clean.detach();
        let (heatmaps, warp) = self.forward_pixels(&clean)?;
        let (coords, conf) = vitpose::heatmaps_to_keypoints(&heatmaps)?;
        let (ratios, ratio_vis) = body_ratios(&coords, &conf, include_head)?;
        let (ratios, ratio_vis) = (ratios.detach(), ratio_vis.detach());
        let body_vis = ratio_vis
            .narrow(1, 0, NUM_BODY_RATIOS)?
            .mean_all()?
            .to_scalar::<f32>()?;
        if body_vis < MIN_MEAN_RATIO_VISIBILITY {
            return Ok(None);
        }
        let c: Vec<f32> = coords.flatten_all()?.to_vec1()?;
        let points: Vec<(f32, f32)> = (0..c.len() / 2)
            .map(|i| warp.keypoint_to_input(c[2 * i], c[2 * i + 1]))
            .collect();
        let confidence: Vec<f32> = conf.flatten_all()?.to_vec1()?;
        let (_, h, w, _) = clean.dims4()?;
        let Some(person_box) = keypoint_box(&points, &confidence, h, w) else {
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
    pub ratios: Tensor,
    pub ratio_vis: Tensor,
}

/// ViTPose bone-length-ratio loss.
pub struct BodyProportionLoss {
    pose: Arc<VitPose>,
    include_head: bool,
}

impl BodyProportionLoss {
    pub fn new(pose: Arc<VitPose>, include_head: bool) -> Self {
        Self { pose, include_head }
    }
}

impl PerceptualLoss for BodyProportionLoss {
    fn name(&self) -> &'static str {
        "body-proportion"
    }

    fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
        Ok(self.pose.detect(clean, self.include_head)?.map(|d| {
            Box::new(ProportionReference {
                ratios: d.ratios,
                ratio_vis: d.ratio_vis,
            }) as LossReference
        }))
    }

    fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor> {
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
    pub betas: Tensor,
    pub crop: (usize, usize, usize, usize),
}

/// HybrIK SMPL-beta loss.
pub struct BodyShapeLoss {
    pose: Arc<VitPose>,
    hybrik: HybrikEncoder,
    min_cos: f32,
}

impl BodyShapeLoss {
    pub fn new(pose: Arc<VitPose>, hybrik: HybrikEncoder, min_cos: f32) -> Self {
        Self {
            pose,
            hybrik,
            min_cos,
        }
    }
}

/// Upstream's body-shape comparison: `mean |live − ref|`, zero unless the detached cosine of the
/// two beta vectors exceeds `min_cos`. Scalar.
pub fn shape_comparison(reference: &Tensor, live: &Tensor, min_cos: f32) -> Result<Tensor> {
    let dot = (reference * live)?.sum(D::Minus1)?;
    let norms = (reference.sqr()?.sum(D::Minus1)?.sqrt()? * live.sqr()?.sum(D::Minus1)?.sqrt()?)?;
    let cos = dot.div(&norms.clamp(1e-8f32, f32::MAX)?)?.detach();
    let gate = cos.gt(min_cos)?.to_dtype(DType::F32)?;
    let l1 = (live - reference)?.abs()?.mean(D::Minus1)?;
    Ok(l1.mul(&gate)?.mean_all()?)
}

impl PerceptualLoss for BodyShapeLoss {
    fn name(&self) -> &'static str {
        "body-shape"
    }

    fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
        let Some(d) = self.pose.detect(clean, false)? else {
            return Ok(None);
        };
        let (_, h, w, _) = clean.dims4()?;
        let crop = HybrikEncoder::crop_for(d.person_box, h, w);
        let betas = self.hybrik.forward_crop(&clean.detach(), crop)?.detach();
        Ok(Some(Box::new(ShapeReference { betas, crop })))
    }

    fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor> {
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
    pub normals: Tensor,
    pub mask: Option<Tensor>,
}

/// Sapiens surface-normal loss.
pub struct NormalLoss {
    pose: Arc<VitPose>,
    sapiens: SapiensNormal,
    restrict_to_subject: bool,
}

impl NormalLoss {
    pub fn new(pose: Arc<VitPose>, sapiens: SapiensNormal, restrict_to_subject: bool) -> Self {
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

    fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
        self.reference_with_mask(clean, None)
    }

    fn reference_with_mask(
        &self,
        clean: &Tensor,
        subject_mask: Option<&Tensor>,
    ) -> Result<Option<LossReference>> {
        if self.pose.detect(clean, false)?.is_none() {
            return Ok(None);
        }
        let mask = if self.restrict_to_subject {
            let m = subject_mask.ok_or_else(|| {
                CandleError::Msg(
                    "normal loss: restricted to the subject but the trainer handed no subject mask \
                     for this image"
                        .into(),
                )
            })?;
            Some(self.sapiens.mask_to_normal_grid(m)?.detach())
        } else {
            None
        };
        let normals = self.sapiens.forward_pixels(&clean.detach())?.detach();
        Ok(Some(Box::new(NormalReference { normals, mask })))
    }

    fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor> {
        let r = reference_as::<NormalReference>(self.name(), reference)?;
        sapiens::normal_comparison(
            &r.normals,
            &self.sapiens.forward_pixels(live)?,
            r.mask.as_ref(),
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Construction + memory (the builder arms).
// ---------------------------------------------------------------------------------------------

fn model_dir(d: &Option<PathBuf>, what: &str, knob: &str) -> Result<PathBuf> {
    d.clone().ok_or_else(|| {
        CandleError::Msg(format!(
            "body losses: the {what} checkpoint directory is unset (body_losses.{knob})"
        ))
    })
}

fn load_err(what: &str, dir: &Path, e: CandleError) -> CandleError {
    CandleError::Msg(format!(
        "body losses: could not load {what} from {}: {e}",
        dir.display()
    ))
}

/// The ViTPose every enabled body loss shares, loaded once per checkpoint directory while any
/// loss holds it.
static SHARED_POSE: Mutex<Option<(PathBuf, Weak<VitPose>)>> = Mutex::new(None);

/// The shared ViTPose+ base of `cfg.pose_model_dir` on `device` (one resident copy, as
/// [`body_arm_footprint`] budgets).
pub fn shared_pose(cfg: &BodyLossesConfig, device: &Device) -> Result<Arc<VitPose>> {
    let dir = model_dir(&cfg.pose_model_dir, "ViTPose+", "pose_model_dir")?;
    let mut slot = SHARED_POSE
        .lock()
        .map_err(|_| CandleError::Msg("body losses: shared ViTPose lock poisoned".into()))?;
    if let Some(p) = slot
        .as_ref()
        .filter(|(d, _)| *d == dir)
        .and_then(|(_, w)| w.upgrade())
        .filter(|p| p.device_matches(device))
    {
        return Ok(p);
    }
    let pose = Arc::new(
        VitPose::from_dir(&dir, VitPoseConfig::plus_base(), device)
            .map_err(|e| load_err("ViTPose+", &dir, e))?,
    );
    *slot = Some((dir, Arc::downgrade(&pose)));
    Ok(pose)
}

/// The body-proportion loss of `cfg` (the builder's `body-proportion` arm).
pub fn proportion_loss(cfg: &BodyLossesConfig, device: &Device) -> Result<Box<dyn PerceptualLoss>> {
    Ok(Box::new(BodyProportionLoss::new(
        shared_pose(cfg, device)?,
        cfg.include_head,
    )))
}

/// The body-shape loss of `cfg` (the builder's `body-shape` arm).
pub fn shape_loss(cfg: &BodyLossesConfig, device: &Device) -> Result<Box<dyn PerceptualLoss>> {
    let d = model_dir(&cfg.shape_model_dir, "HybrIK", "shape_model_dir")?;
    let hybrik = HybrikEncoder::from_dir(&d, HybrikConfig::resnet34(), device)
        .map_err(|e| load_err("HybrIK", &d, e))?;
    Ok(Box::new(BodyShapeLoss::new(
        shared_pose(cfg, device)?,
        hybrik,
        cfg.shape_min_cos,
    )))
}

/// The normal loss of `cfg` (the builder's `normal` arm).
pub fn normal_loss(cfg: &BodyLossesConfig, device: &Device) -> Result<Box<dyn PerceptualLoss>> {
    let d = model_dir(&cfg.normal_model_dir, "Sapiens normal", "normal_model_dir")?;
    let sapiens = SapiensNormal::from_dir(&d, SapiensConfig::normal_0_3b(), device)
        .map_err(|e| load_err("Sapiens normal", &d, e))?;
    Ok(Box::new(NormalLoss::new(
        shared_pose(cfg, device)?,
        sapiens,
        cfg.normal_restrict_to_subject,
    )))
}

/// One body arm's E7 footprint ([`body_arm_footprint`]).
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
