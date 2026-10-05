//! The two **decoded-x0 face losses** of the perceptual-character-LoRA epic (epic 2123, sc-24831) on
//! the shared perceptual path ([`mlx_gen::train::perceptual`]): the ArcFace **identity loss** and the
//! MediaPipe FaceMesh **landmark loss**, ported from ai-toolkit-perceptual (fork commit 6e01a6e,
//! `toolkit/face_id.py` `DifferentiableFaceEncoder` / `DifferentiableLandmarkEncoder` and their loss
//! blocks in `SDTrainer.py`).
//!
//! **Reference time** (once per cached image, outside autograd — [`PerceptualLoss::reference`]): the
//! frozen SCRFD detector ([`FaceBoxDetector`]) finds the largest face on the image's own
//! encode→decode round trip. No face ⇒ `Ok(None)`, which the shared path turns into a per-image skip
//! (an aux-claimed step falls back to the diffusion loss). Otherwise the face box is expanded by 15 %
//! and rounded exactly as upstream ([`face_crop_box`]) and stored in the per-image reference together
//! with that crop's ArcFace embedding (identity) or normalized FaceMesh landmarks (landmarks).
//!
//! **Loss time** (inside the traced step — [`PerceptualLoss::loss`]): the **live** decoded x0 is cut
//! with the stored box and resampled by fixed interpolation matrices ([`crop_resize`]: zero-pad to
//! square + bilinear to 112² for ArcFace, bilinear to 256² for FaceMesh — torch
//! `F.interpolate(..., align_corners=False)` exactly), so the whole chain is differentiable in the
//! decoded pixels and the frozen models contribute captured constants only.
//!
//! - identity: `loss = 1 − cos(normalize(ArcFace(x0_face)), target)` where `target` is the image's own
//!   reference embedding or the normalized dataset mean ([`IdentityReferenceMode`]); a step whose live
//!   cosine is `<= min_cos` contributes **zero** (upstream's gate against pushing on a hallucinated
//!   non-face — upstream also notes it "stands in for" a per-crop detector on the 5-D path).
//! - landmarks: FaceMesh `out[0] → [478, 3][..., :2]`, centred on the nose tip (1) and scaled by the
//!   inner-eye distance (133–362, floored at 0.01); the loss is the region-weighted mean landmark
//!   distance `(3·jaw + 2·lips + 1·(eyes+nose)) / 6` with a `1e-6` floor under each `sqrt`.
//!
//! ArcFace input stays **RGB** (insightface's canonical `swapRB=True` preprocessing, which the native
//! glintr100 port and its onnx goldens use); upstream flips to BGR on both its reference and live
//! sides, so the choice only has to be consistent, which it is.

use std::any::Any;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use mlx_gen::gen_core::train::{IdentityLossConfig, IdentityReferenceMode};
use mlx_gen::train::perceptual::{
    reference_as, AuxModelFootprint, LossReference, PerceptualLoss,
};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{add, divide, matmul, maximum, multiply, subtract};
use mlx_rs::Array;

use crate::face::detector_blob;
use crate::iresnet::ArcFace;
use crate::program::Program;
use crate::scrfd::Scrfd;

/// SCRFD detector checkpoint in the face-analysis stack dir (the `instantid_face_stack` bundle).
pub const SCRFD_FILE: &str = "scrfd_10g.safetensors";
/// ArcFace checkpoint in the face-analysis stack dir.
pub const ARCFACE_FILE: &str = "arcface_iresnet100.safetensors";
/// Converted FaceMesh-v2 program checkpoint (`tools/convert_mp_facemesh_v2.py`).
pub const FACEMESH_FILE: &str = "face_landmarks_detector.safetensors";

/// Upstream's face-box expansion on every side (`bw * 0.15`, `bh * 0.15`).
pub const FACE_CROP_PAD: f64 = 0.15;
/// ArcFace input edge.
pub const ARCFACE_INPUT: usize = 112;
/// FaceMesh input edge.
pub const FACEMESH_INPUT: usize = 256;
/// Number of FaceMesh-v2 landmarks.
pub const FACEMESH_LANDMARKS: usize = 478;

/// MediaPipe FaceMesh jaw / face-oval indices (weight 3).
pub const FACE_OVAL: [i32; 36] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
/// Lip indices (weight 2).
pub const LIPS: [i32; 20] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
];
/// Mid-face indices — left eye, right eye, nose (weight 1).
pub const MIDFACE: [i32; 45] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381,
    380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 1, 2, 98, 327, 168, 6, 197,
    195, 5, 4, 19, 94, 370,
];
const NOSE_TIP: i32 = 1;
const LEFT_INNER_EYE: i32 = 133;
const RIGHT_INNER_EYE: i32 = 362;

/// A half-open integer pixel box `[x0, x1) × [y0, y1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropBox {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl CropBox {
    fn width(&self) -> usize {
        self.x1 - self.x0
    }
    fn height(&self) -> usize {
        self.y1 - self.y0
    }
}

/// Upstream's face crop box for a detector `bbox` `[x1, y1, x2, y2]` on an `h × w` image: expand by
/// [`FACE_CROP_PAD`] per side, round half-to-even (Python `round`), clamp to the image; a box that
/// collapses falls back to the full frame (upstream's `else: crop = pixels[i:i+1]`).
pub fn face_crop_box(bbox: [f32; 4], h: usize, w: usize) -> CropBox {
    let [x1, y1, x2, y2] = bbox.map(|v| v as f64);
    let (bw, bh) = (x2 - x1, y2 - y1);
    let (pw, ph) = (bw * FACE_CROP_PAD, bh * FACE_CROP_PAD);
    let r = |v: f64| v.round_ties_even() as i64;
    let cx1 = r(x1 - pw).max(0);
    let cy1 = r(y1 - ph).max(0);
    let cx2 = r(x2 + pw).min(w as i64);
    let cy2 = r(y2 + ph).min(h as i64);
    if cx2 > cx1 && cy2 > cy1 {
        CropBox {
            x0: cx1 as usize,
            y0: cy1 as usize,
            x1: cx2 as usize,
            y1: cy2 as usize,
        }
    } else {
        CropBox {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        }
    }
}

/// The `[out, in]` matrix of torch's bilinear `F.interpolate(align_corners=False)` (no antialias)
/// along one axis: source `max((o + 0.5)·in/out − 0.5, 0)`, taps `⌊s⌋` and `min(⌊s⌋+1, in−1)`.
pub fn bilinear_matrix(out: usize, input: usize) -> Vec<f32> {
    let mut m = vec![0f32; out * input];
    let scale = input as f64 / out as f64;
    for o in 0..out {
        let src = ((o as f64 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (src.floor() as usize).min(input - 1);
        let i1 = (i0 + 1).min(input - 1);
        let l1 = (src - i0 as f64) as f32;
        m[o * input + i0] += 1.0 - l1;
        m[o * input + i1] += l1;
    }
    m
}

/// The resample matrix for `len` pixels placed at `offset` inside a zero-padded axis of `padded`
/// pixels, resized to `out`: the padded axis's [`bilinear_matrix`] restricted to the real columns
/// (the padding is zero, so its columns contribute nothing).
fn axis_matrix(out: usize, padded: usize, offset: usize, len: usize) -> Array {
    let full = bilinear_matrix(out, padded);
    let mut m = Vec::with_capacity(out * len);
    for o in 0..out {
        m.extend_from_slice(&full[o * padded + offset..o * padded + offset + len]);
    }
    Array::from_slice(&m, &[out as i32, len as i32])
}

/// Cut `b` out of NHWC `[1, H, W, C]` pixels and resample it to `out × out`, differentiably (two
/// constant matmuls). `square` zero-pads the shorter side first (centred, upstream's identity crop);
/// otherwise the crop is stretched (upstream's landmark crop).
pub fn crop_resize(px: &Array, b: CropBox, square: bool, out: usize) -> Result<Array> {
    let (h, w) = (b.height(), b.width());
    let c = px.shape()[3];
    let crop = px.index((
        0,
        b.y0 as i32..b.y1 as i32,
        b.x0 as i32..b.x1 as i32,
        ..,
    )); // [h, w, C]
    let (sy, oy, sx, ox) = if square && w != h {
        let s = w.max(h);
        let d = s - w.min(h);
        if w > h {
            (s, d / 2, w, 0)
        } else {
            (h, 0, s, d / 2)
        }
    } else {
        (h, 0, w, 0)
    };
    let ry = axis_matrix(out, sy, oy, h); // [out, h]
    let rx = axis_matrix(out, sx, ox, w); // [out, w]
    let y = matmul(&ry, &crop.reshape(&[h as i32, (w as i32) * c])?)?; // [out, w·C]
    let y = y
        .reshape(&[out as i32, w as i32, c])?
        .transpose_axes(&[0, 2, 1])?; // [out, C, w]
    let z = matmul(&y, &rx.transpose_axes(&[1, 0])?)?; // [out, C, out]
    Ok(z.transpose_axes(&[0, 2, 1])?
        .reshape(&[1, out as i32, out as i32, c])?)
}

/// L2-normalize the last axis.
fn l2_normalize(x: &Array) -> Result<Array> {
    let n = mlx_rs::ops::sqrt(&x.square()?.sum_axes(&[-1], true)?)?;
    Ok(divide(x, &maximum(&n, Array::from_f32(1e-12))?)?)
}

/// Read NHWC `[1, H, W, 3]` pixels in `[0, 1]` back as an RGB `u8` buffer (round, clamp) — the
/// detector's input. Reshape first: a decode may be a transposed (non-row-contiguous) view.
fn to_rgb_u8(px: &Array) -> Result<(Vec<u8>, usize, usize)> {
    let s = px.shape();
    if s.len() != 4 || s[0] != 1 || s[3] != 3 {
        return Err(Error::Msg(format!(
            "face loss: expected NHWC [1, H, W, 3] pixels, got {s:?}"
        )));
    }
    let (h, w) = (s[1] as usize, s[2] as usize);
    let flat = mlx_rs::ops::clip(&multiply(px, Array::from_f32(255.0))?, (0.0f32, 255.0f32))?
        .round(None)?
        .reshape(&[-1])?;
    flat.eval()?;
    let v = flat
        .try_as_slice::<f32>()
        .map_err(|e| Error::Msg(format!("face loss: pixel readback: {e}")))?;
    Ok((v.iter().map(|&p| p as u8).collect(), h, w))
}

fn hw(px: &Array) -> (usize, usize) {
    let s = px.shape();
    (s[1] as usize, s[2] as usize)
}

/// Finds the largest face on an RGB `u8` image (reference time only).
pub trait FaceBoxDetector {
    /// `[x1, y1, x2, y2]` of the largest face in original-image pixels, or `None`.
    fn largest_face(&self, rgb: &[u8], h: usize, w: usize) -> Result<Option<[f32; 4]>>;
}

/// The SCRFD-10g detector at insightface's defaults (score 0.5, NMS 0.4), largest face first.
pub struct ScrfdDetector {
    scrfd: Scrfd,
    pub det_thresh: f32,
    pub nms_thresh: f32,
}

impl ScrfdDetector {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        Ok(Self {
            scrfd: Scrfd::from_weights(w)?,
            det_thresh: 0.5,
            nms_thresh: 0.4,
        })
    }
}

impl FaceBoxDetector for ScrfdDetector {
    fn largest_face(&self, rgb: &[u8], h: usize, w: usize) -> Result<Option<[f32; 4]>> {
        let (blob, det_scale) = detector_blob(rgb, h, w)?;
        let dets = self
            .scrfd
            .detect(&blob, det_scale, self.det_thresh, self.nms_thresh)?;
        let area = |b: &[f32; 4]| (b[2] - b[0]) * (b[3] - b[1]);
        Ok(dets
            .iter()
            .map(|d| d.bbox)
            .max_by(|a, b| area(a).total_cmp(&area(b))))
    }
}

/// Detect the reference face on decoded round-trip pixels ⇒ its crop box, or `None` (skip).
fn reference_box(detector: &dyn FaceBoxDetector, clean: &Array) -> Result<Option<CropBox>> {
    let (rgb, h, w) = to_rgb_u8(clean)?;
    Ok(detector
        .largest_face(&rgb, h, w)?
        .map(|bbox| face_crop_box(bbox, h, w)))
}

fn check_live(name: &str, live: &Array, image_hw: (usize, usize)) -> Result<()> {
    if hw(live) != image_hw {
        return Err(Error::Msg(format!(
            "{name} loss: live decode is {:?} but the reference was built at {image_hw:?}",
            hw(live)
        )));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------
// Identity
// ------------------------------------------------------------------------------------------------

/// The per-image identity reference: the stored face box and that crop's unit ArcFace embedding.
pub struct IdentityReference {
    pub crop: CropBox,
    pub image_hw: (usize, usize),
    /// `[D]`, unit norm, gradient-free.
    pub embedding: Array,
}

#[derive(Default)]
struct DatasetMean {
    sum: Option<Array>,
    count: usize,
    /// Set by the first loss call; the references are complete from then on.
    frozen: Option<Array>,
}

/// The ArcFace identity loss (see the module docs).
pub struct IdentityLoss {
    arcface: ArcFace,
    detector: Rc<dyn FaceBoxDetector>,
    min_cos: f32,
    mode: IdentityReferenceMode,
    mean: RefCell<DatasetMean>,
}

impl IdentityLoss {
    pub fn new(
        arcface: ArcFace,
        detector: Rc<dyn FaceBoxDetector>,
        min_cos: f32,
        mode: IdentityReferenceMode,
    ) -> Self {
        Self {
            arcface,
            detector,
            min_cos,
            mode,
            mean: RefCell::new(DatasetMean::default()),
        }
    }

    /// Unit ArcFace embedding `[D]` of `px`'s face crop `b` (differentiable in `px`).
    pub fn embed(&self, px: &Array, b: CropBox) -> Result<Array> {
        let crop = crop_resize(px, b, true, ARCFACE_INPUT)?;
        // `(px·255 − 127.5) / 127.5` = `2·px − 1`.
        let x = subtract(&multiply(&crop, Array::from_f32(2.0))?, Array::from_f32(1.0))?;
        let emb = self.arcface.forward(&x)?; // [1, D]
        Ok(l2_normalize(&emb)?.reshape(&[-1])?)
    }

    /// The target embedding for `r` under the configured mode.
    fn target(&self, r: &IdentityReference) -> Result<Array> {
        match self.mode {
            IdentityReferenceMode::PerImage => Ok(r.embedding.clone()),
            IdentityReferenceMode::DatasetAverage => {
                let mut m = self.mean.borrow_mut();
                if m.frozen.is_none() {
                    let sum = m.sum.as_ref().ok_or_else(|| {
                        Error::Msg("identity loss: dataset average of zero references".into())
                    })?;
                    let mean = l2_normalize(&divide(sum, Array::from_f32(m.count as f32))?)?;
                    mean.eval()?;
                    m.frozen = Some(mean);
                }
                Ok(m.frozen.clone().expect("frozen above"))
            }
        }
    }

    /// `1 − cos` gated by `cos > min_cos`, with the cosine returned for diagnostics.
    pub fn loss_and_cos(&self, live: &Array, r: &IdentityReference) -> Result<(Array, Array)> {
        check_live("identity", live, r.image_hw)?;
        let emb = self.embed(live, r.crop)?;
        let cos = multiply(&emb, &self.target(r)?)?.sum(None)?;
        let gate = mlx_rs::stop_gradient(&cos)?
            .gt(Array::from_f32(self.min_cos))?
            .as_type::<f32>()?;
        let loss = multiply(&subtract(Array::from_f32(1.0), &cos)?, &gate)?;
        Ok((loss, cos))
    }
}

impl PerceptualLoss for IdentityLoss {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
        let Some(crop) = reference_box(self.detector.as_ref(), clean)? else {
            return Ok(None);
        };
        let embedding = mlx_rs::stop_gradient(&self.embed(clean, crop)?)?;
        embedding.eval()?;
        if self.mode == IdentityReferenceMode::DatasetAverage {
            let mut m = self.mean.borrow_mut();
            if m.frozen.is_some() {
                return Err(Error::Msg(
                    "identity loss: a dataset-average reference was added after training began; \
                     every image's reference must be built before the first step"
                        .into(),
                ));
            }
            m.sum = Some(match m.sum.take() {
                Some(s) => add(&s, &embedding)?,
                None => embedding.clone(),
            });
            m.count += 1;
        }
        Ok(Some(Box::new(IdentityReference {
            crop,
            image_hw: hw(clean),
            embedding,
        })))
    }

    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<IdentityReference>(self.name(), reference)?;
        Ok(self.loss_and_cos(live, r)?.0)
    }
}

// ------------------------------------------------------------------------------------------------
// Landmarks
// ------------------------------------------------------------------------------------------------

/// The per-image landmark reference: the stored face box and that crop's normalized landmarks.
pub struct LandmarkReference {
    pub crop: CropBox,
    pub image_hw: (usize, usize),
    /// `[478, 2]`, normalized, gradient-free.
    pub landmarks: Array,
}

/// Centre `[N, 478, 2]` landmarks on the nose tip and scale by the inner-eye distance (≥ 0.01).
pub fn normalize_landmarks(lm: &Array) -> Result<Array> {
    let pick = |i: i32| lm.index((.., i..i + 1, ..));
    let centered = subtract(lm, &pick(NOSE_TIP))?;
    let d = subtract(&pick(LEFT_INNER_EYE), &pick(RIGHT_INNER_EYE))?;
    let inter = mlx_rs::ops::sqrt(&d.square()?.sum_axes(&[-1], true)?)?;
    Ok(divide(&centered, &maximum(&inter, Array::from_f32(0.01))?)?)
}

/// Upstream's region-weighted landmark distance between `[478, 2]` sets.
pub fn landmark_distance(gen: &Array, reference: &Array) -> Result<Array> {
    let region = |idx: &[i32]| -> Result<Array> {
        let ix = Array::from_slice(idx, &[idx.len() as i32]);
        let d = subtract(&gen.take_axis(&ix, 0)?, &reference.take_axis(&ix, 0)?)?;
        let sq = maximum(&d.square()?.sum_axes(&[-1], false)?, Array::from_f32(1e-6))?;
        Ok(mlx_rs::ops::sqrt(&sq)?.mean(None)?)
    };
    let jaw = multiply(&region(&FACE_OVAL)?, Array::from_f32(3.0))?;
    let lips = multiply(&region(&LIPS)?, Array::from_f32(2.0))?;
    let mid = region(&MIDFACE)?;
    Ok(divide(&add(&add(&jaw, &lips)?, &mid)?, Array::from_f32(6.0))?)
}

/// The FaceMesh landmark loss (see the module docs).
pub struct FaceLandmarkLoss {
    mesh: Program,
    detector: Rc<dyn FaceBoxDetector>,
}

impl FaceLandmarkLoss {
    pub fn new(mesh: Program, detector: Rc<dyn FaceBoxDetector>) -> Self {
        Self { mesh, detector }
    }

    /// Normalized `[478, 2]` landmarks of `px`'s face crop `b` (differentiable in `px`).
    pub fn landmarks(&self, px: &Array, b: CropBox) -> Result<Array> {
        let crop = crop_resize(px, b, false, FACEMESH_INPUT)?;
        let out = self.mesh.forward(&crop)?;
        let raw = out
            .first()
            .ok_or_else(|| Error::Msg("face-landmark loss: FaceMesh has no outputs".into()))?;
        if raw.size() != FACEMESH_LANDMARKS * 3 {
            return Err(Error::Msg(format!(
                "face-landmark loss: FaceMesh output 0 has {} values, want {}",
                raw.size(),
                FACEMESH_LANDMARKS * 3
            )));
        }
        let xy = raw
            .reshape(&[1, FACEMESH_LANDMARKS as i32, 3])?
            .index((.., .., 0..2));
        Ok(normalize_landmarks(&xy)?.reshape(&[FACEMESH_LANDMARKS as i32, 2])?)
    }
}

impl PerceptualLoss for FaceLandmarkLoss {
    fn name(&self) -> &'static str {
        "face-landmark"
    }

    fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
        let Some(crop) = reference_box(self.detector.as_ref(), clean)? else {
            return Ok(None);
        };
        let landmarks = mlx_rs::stop_gradient(&self.landmarks(clean, crop)?)?;
        landmarks.eval()?;
        Ok(Some(Box::new(LandmarkReference {
            crop,
            image_hw: hw(clean),
            landmarks,
        })))
    }

    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<LandmarkReference>(self.name(), reference)?;
        check_live("face-landmark", live, r.image_hw)?;
        landmark_distance(&self.landmarks(live, r.crop)?, &r.landmarks)
    }
}

// ------------------------------------------------------------------------------------------------
// Loading + memory (E7)
// ------------------------------------------------------------------------------------------------

fn load_weights(p: &Path) -> Result<Weights> {
    Weights::from_file(p)
        .map_err(|e| Error::Msg(format!("face loss: could not load {}: {e}", p.display())))
}

fn load_detector(face_dir: &Path) -> Result<Rc<dyn FaceBoxDetector>> {
    Ok(Rc::new(ScrfdDetector::from_weights(&load_weights(
        &face_dir.join(SCRFD_FILE),
    )?)?))
}

/// Load the identity loss from the face-analysis stack dir (`face_dir/`[`SCRFD_FILE`] +
/// `face_dir/`[`ARCFACE_FILE`]) with `cfg`'s gate and reference mode.
pub fn load_identity_loss(face_dir: &Path, cfg: &IdentityLossConfig) -> Result<IdentityLoss> {
    let arcface = ArcFace::from_weights(&load_weights(&face_dir.join(ARCFACE_FILE))?)?;
    Ok(IdentityLoss::new(
        arcface,
        load_detector(face_dir)?,
        cfg.min_cos,
        cfg.reference_mode,
    ))
}

/// Load the face-landmark loss: SCRFD from `face_dir/`[`SCRFD_FILE`], FaceMesh from
/// `mesh_dir/`[`FACEMESH_FILE`].
pub fn load_face_landmark_loss(face_dir: &Path, mesh_dir: &Path) -> Result<FaceLandmarkLoss> {
    Ok(FaceLandmarkLoss::new(
        Program::from_file(mesh_dir.join(FACEMESH_FILE))?,
        load_detector(face_dir)?,
    ))
}

/// Published SCRFD-10g (bnkps) parameter count (insightface model zoo: 4.23 M).
pub const SCRFD_10G_PARAMS: u64 = 4_230_000;
/// Upper bound of the FaceMesh-v2 landmark detector's parameters: the upstream checkpoint is a
/// 5.21 MB f32 pickle (≈ 1.30 M floats; upstream's docstring says 1.2 M).
pub const FACEMESH_V2_PARAMS: u64 = 1_310_000;
/// IResNet stage widths (every insightface ArcFace checkpoint).
const IRESNET_WIDTHS: [u64; 4] = [64, 128, 256, 512];
const IRESNET_STEM: u64 = 64;
const ARCFACE_EMBEDDING: u64 = 512;

/// Parameters of an IResNet ArcFace with per-stage block counts `layers` (analytic, from the
/// architecture; glintr100 = `[3,13,30,3]` ⇒ ≈ 65.2 M).
pub fn arcface_param_count(layers: [usize; 4]) -> u64 {
    let mut n = 3 * IRESNET_STEM * 9 + 2 * IRESNET_STEM; // stem conv (+bias) + prelu
    let mut cin = IRESNET_STEM;
    for (&nb, &c) in layers.iter().zip(&IRESNET_WIDTHS) {
        for b in 0..nb as u64 {
            let bin = if b == 0 { cin } else { c };
            n += 2 * bin + 9 * bin * c + c + c + 9 * c * c + c;
            if b == 0 {
                n += bin * c + c;
            }
        }
        cin = c;
    }
    n + 2 * cin + cin * 49 * ARCFACE_EMBEDDING + ARCFACE_EMBEDDING + 2 * ARCFACE_EMBEDDING
}

/// Conservative training working set of one differentiable ArcFace forward + backward at 112²: the
/// activations every block retains (its input, bn1, conv1, PReLU and conv2/residual outputs), f32,
/// ×2 for the cotangents. An estimate, not a measurement.
pub fn arcface_working_set_bytes(layers: [usize; 4]) -> u64 {
    let mut side = ARCFACE_INPUT as u64;
    let mut floats = 2 * IRESNET_STEM * side * side; // stem conv + prelu
    let mut cin = IRESNET_STEM;
    for (&nb, &c) in layers.iter().zip(&IRESNET_WIDTHS) {
        for b in 0..nb {
            let bin = if b == 0 { cin } else { c };
            let out = if b == 0 { side / 2 } else { side };
            floats += 2 * bin * side * side + 2 * c * side * side + 2 * c * out * out;
            side = out;
        }
        cin = c;
    }
    floats * 4 * 2
}

/// The SCRFD detector each face loss loads (reference-time forward on a 640² blob, no backward):
/// its weights plus ≈ 32 input-sized f32 maps live at the widest stage.
fn detector_footprint() -> AuxModelFootprint {
    AuxModelFootprint {
        param_bytes: SCRFD_10G_PARAMS * 4,
        working_set_bytes: 32 * 640 * 640 * 3 * 4,
        reference_bytes_per_image: 0,
    }
}

fn plus(a: AuxModelFootprint, b: AuxModelFootprint) -> AuxModelFootprint {
    AuxModelFootprint {
        param_bytes: a.param_bytes + b.param_bytes,
        working_set_bytes: a.working_set_bytes + b.working_set_bytes,
        reference_bytes_per_image: a.reference_bytes_per_image + b.reference_bytes_per_image,
    }
}

/// E7 footprint of the identity loss (its SCRFD detector + an IResNet ArcFace of `layers`; the
/// shipped face stack is glintr100, [`crate::iresnet::IRESNET100_LAYERS`]). The crop is a fixed
/// 112², so the figure does not depend on the training resolution.
pub fn identity_loss_footprint(layers: [usize; 4]) -> AuxModelFootprint {
    plus(
        detector_footprint(),
        AuxModelFootprint {
            param_bytes: arcface_param_count(layers) * 4,
            working_set_bytes: arcface_working_set_bytes(layers),
            // Unit embedding + box.
            reference_bytes_per_image: ARCFACE_EMBEDDING * 4 + 64,
        },
    )
}

/// E7 footprint of the face-landmark loss (its SCRFD detector + FaceMesh-v2 on a fixed 256² crop).
pub fn face_landmark_loss_footprint() -> AuxModelFootprint {
    let input = (FACEMESH_INPUT * FACEMESH_INPUT * 3 * 4) as u64;
    plus(
        detector_footprint(),
        AuxModelFootprint {
            param_bytes: FACEMESH_V2_PARAMS * 4,
            // MobileNet-class graph at 256²: ≈ 64 input-sized f32 maps retained, ×2 cotangents.
            working_set_bytes: 64 * input * 2,
            reference_bytes_per_image: (FACEMESH_LANDMARKS * 2 * 4) as u64 + 64,
        },
    )
}

/// On-disk stand-ins for the face-loss checkpoints, for tests of the builder / trainers that must
/// never download real weights: a weightless SCRFD (scalar zeros — loads, never forwarded), a tiny
/// synthetic IResNet ArcFace, and a tiny FaceMesh-shaped fx-program.
pub mod testing {
    use std::collections::HashMap;
    use std::path::Path;

    use mlx_gen::{Error, Result};
    use mlx_rs::Array;

    use super::{ARCFACE_FILE, FACEMESH_FILE, SCRFD_FILE};
    use crate::synth;

    /// Key → shape of a tiny IResNet (stem 8, widths 8/16/32/64, blocks `[1,2,1,1]`, 32-d) — the
    /// parity fixture's architecture.
    pub fn tiny_arcface_shapes() -> Vec<(String, Vec<usize>)> {
        let mut out = Vec::new();
        let conv = |out: &mut Vec<(String, Vec<usize>)>, p: &str, cin, cout, k| {
            out.push((format!("{p}.weight"), vec![cout, k, k, cin]));
            out.push((format!("{p}.bias"), vec![cout]));
        };
        let aff = |out: &mut Vec<(String, Vec<usize>)>, p: &str, c| {
            out.push((format!("{p}.scale"), vec![c]));
            out.push((format!("{p}.shift"), vec![c]));
        };
        conv(&mut out, "stem.conv", 3, 8, 3);
        out.push(("stem.prelu.weight".into(), vec![8]));
        let mut cin = 8;
        for (li, (nb, c)) in [1usize, 2, 1, 1].into_iter().zip([8, 16, 32, 64]).enumerate() {
            for b in 0..nb {
                let p = format!("layer{}.{b}", li + 1);
                let bin = if b == 0 { cin } else { c };
                aff(&mut out, &format!("{p}.bn1"), bin);
                conv(&mut out, &format!("{p}.conv1"), bin, c, 3);
                out.push((format!("{p}.prelu.weight"), vec![c]));
                conv(&mut out, &format!("{p}.conv2"), c, c, 3);
                if b == 0 {
                    conv(&mut out, &format!("{p}.downsample"), bin, c, 1);
                }
            }
            cin = c;
        }
        aff(&mut out, "bn2", cin);
        out.push(("fc.weight".into(), vec![32, cin * 49]));
        out.push(("fc.bias".into(), vec![32]));
        aff(&mut out, "features", 32);
        out
    }

    fn save(
        pairs: Vec<(String, Array)>,
        meta: Option<&HashMap<String, String>>,
        path: &Path,
    ) -> Result<()> {
        Array::save_safetensors(pairs.iter().map(|(k, a)| (k.as_str(), a)), meta, path)
            .map_err(|e| Error::Msg(format!("write {}: {e}", path.display())))
    }

    /// Write `dir/scrfd_10g.safetensors` (weightless) + `dir/arcface_iresnet100.safetensors`
    /// (tiny synthetic IResNet).
    pub fn write_face_stack(dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| Error::Msg(e.to_string()))?;
        let scrfd = crate::face::scrfd_schema_keys()
            .into_iter()
            .map(|k| (k, Array::from_f32(0.0)))
            .collect();
        save(scrfd, None, &dir.join(SCRFD_FILE))?;
        let arc = tiny_arcface_shapes()
            .into_iter()
            .map(|(k, s)| {
                let t = synth::tensor(0x24831A, &k, &s);
                (k, t)
            })
            .collect();
        save(arc, None, &dir.join(ARCFACE_FILE))
    }

    /// Write `dir/face_landmarks_detector.safetensors`: a tiny program with FaceMesh's I/O contract
    /// (`[N,3,256,256]` → `[N,1,1,1434]`).
    pub fn write_facemesh(dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| Error::Msg(e.to_string()))?;
        let program = r#"{"inputs":["x"],"outputs":["y"],"nodes":[
            {"op":"maxpool2d","out":"p","inputs":["x"],"kernel":[32,32],"stride":[32,32],"padding":[0,0]},
            {"op":"conv2d","out":"h","inputs":["p"],"weight":"head.weight","bias":"head.bias",
             "stride":[1,1],"padding":[0,0],"dilation":[1,1],"groups":1},
            {"op":"reshape","out":"y","inputs":["h"],"shape":[-1,1,1,1434]}]}"#;
        let meta = HashMap::from([
            ("format".to_string(), crate::program::FORMAT.to_string()),
            ("program".to_string(), program.to_string()),
        ]);
        let params = vec![
            (
                "head.weight".to_string(),
                synth::tensor(0x24831B, "head.weight", &[1434, 3, 8, 8]),
            ),
            (
                "head.bias".to_string(),
                synth::tensor(0x24831B, "head.bias", &[1434]),
            ),
        ];
        save(params, Some(&meta), &dir.join(FACEMESH_FILE))
    }
}

#[cfg(test)]
mod tests;
