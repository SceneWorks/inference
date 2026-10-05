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
//! - identity ([`IdentityScorer`]): `cos = center(ArcFace(x0_face)) · center(target)` with upstream's
//!   bias centring `center(e) = normalize(normalize(e) − noise_mean)`; `target` is the image's own
//!   reference embedding or the normalized dataset mean ([`IdentityReferenceMode`], which also
//!   normalizes by the image's clean score: `max(0, 1 − cos / clean)`); a frame whose cosine is
//!   `<= min_cos` contributes **zero** (upstream's gate against pushing on a hallucinated non-face),
//!   and the loss averages over the open frames.
//! - landmarks: FaceMesh `out[0] → [478, 3][..., :2]`, centred on the nose tip (1) and scaled by the
//!   inner-eye distance (133–362, floored at 0.01); the loss is the region-weighted mean landmark
//!   distance `(3·jaw + 2·lips + 1·(eyes+nose)) / 6` with a `1e-6` floor under each `sqrt`; with the
//!   identity loss on, a frame is gated by the (shared) identity scorer's `cos > min_cos`.
//! - both losses are scaled by the step's noise level (upstream's `t_ratio`,
//!   [`PerceptualLoss::timestep_weight`]); a reference frame without a detected face is retried on a
//!   gray-padded copy (upstream's tight-close-up fallback) before it is skipped.
//!
//! ArcFace input stays **RGB** (insightface's canonical `swapRB=True` preprocessing, which the native
//! glintr100 port and its onnx goldens use); upstream flips to BGR on both its reference and live
//! sides, so the choice only has to be consistent, which it is.

use std::any::Any;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use mlx_gen::gen_core::train::{IdentityLossConfig, IdentityReferenceMode};
use mlx_gen::train::perceptual::{reference_as, LossReference, PerceptualLoss};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{add, divide, matmul, maximum, multiply, subtract};
use mlx_rs::Array;

use crate::face::detector_blob;
use crate::iresnet::ArcFace;
use crate::program::Program;
use crate::scrfd::Scrfd;

pub use mlx_gen::gen_core::train::face_loss::{
    arcface_param_count, bilinear_matrix, crop_resample_matrices, face_crop_box,
    face_landmark_loss_footprint, identity_loss_footprint, CropBox, ARCFACE_FILE, ARCFACE_INPUT,
    FACEMESH_FILE, FACEMESH_INPUT, FACEMESH_LANDMARKS, FACEMESH_V2_PARAMS, FACE_CROP_PAD,
    SCRFD_10G_PARAMS, SCRFD_FILE,
};
use mlx_gen::gen_core::train::face_loss::{
    detect_with_retry, face_loss_timestep_weight, synth, IDENTITY_CLEAN_COS_FLOOR,
    IDENTITY_NOISE_SAMPLES, IDENTITY_NOISE_SEED, INNER_EYES, INTER_EYE_FLOOR, LANDMARK_EPS,
    LANDMARK_REGIONS, NOSE_TIP,
};

/// Cut `b` out of the first frame of NHWC `[N, H, W, C]` pixels and resample it to `out × out`, differentiably (two
/// constant matmuls). `square` zero-pads the shorter side first (centred, upstream's identity crop);
/// otherwise the crop is stretched (upstream's landmark crop).
pub fn crop_resize(px: &Array, b: CropBox, square: bool, out: usize) -> Result<Array> {
    let (h, w) = (b.height(), b.width());
    let c = px.shape()[3];
    let crop = px.index((0, b.y0 as i32..b.y1 as i32, b.x0 as i32..b.x1 as i32, ..)); // [h, w, C]
    let (ry, rx) = crop_resample_matrices(b, square, out);
    let ry = Array::from_slice(&ry, &[out as i32, h as i32]);
    let rx = Array::from_slice(&rx, &[out as i32, w as i32]);
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

/// Read NHWC `[N, H, W, 3]` pixels in `[0, 1]` back as one RGB `u8` buffer per frame (round, clamp)
/// — the detector's input. Reshape first: a decode may be a transposed (non-row-contiguous) view.
fn to_rgb_u8_frames(px: &Array) -> Result<(Vec<Vec<u8>>, usize, usize)> {
    let s = px.shape();
    if s.len() != 4 || s[0] < 1 || s[3] != 3 {
        return Err(Error::Msg(format!(
            "face loss: expected NHWC [N, H, W, 3] pixels, got {s:?}"
        )));
    }
    let (n, h, w) = (s[0] as usize, s[1] as usize, s[2] as usize);
    let flat = mlx_rs::ops::clip(&multiply(px, Array::from_f32(255.0))?, (0.0f32, 255.0f32))?
        .round(None)?
        .reshape(&[-1])?;
    flat.eval()?;
    let v = flat
        .try_as_slice::<f32>()
        .map_err(|e| Error::Msg(format!("face loss: pixel readback: {e}")))?;
    let per = h * w * 3;
    Ok((
        (0..n)
            .map(|f| v[f * per..(f + 1) * per].iter().map(|&p| p as u8).collect())
            .collect(),
        h,
        w,
    ))
}

/// `(frames, height, width)` of NHWC pixels.
fn geometry(px: &Array) -> (usize, usize, usize) {
    let s = px.shape();
    (s[0] as usize, s[1] as usize, s[2] as usize)
}

/// Frame `f` of NHWC pixels as `[1, H, W, C]`.
fn frame(px: &Array, f: usize) -> Array {
    px.index(f as i32..f as i32 + 1)
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

/// Detect the reference face on every frame of the decoded round trip ⇒ each frame's crop box, or
/// `None` for a frame without a face. Upstream's retry runs on a miss: the frame is re-detected
/// centred in a gray border ([`detect_with_retry`]), and a hit is mapped back to the frame.
fn reference_boxes(detector: &dyn FaceBoxDetector, clean: &Array) -> Result<Vec<Option<CropBox>>> {
    let (frames, h, w) = to_rgb_u8_frames(clean)?;
    frames
        .iter()
        .map(|rgb| {
            Ok(
                detect_with_retry(rgb, h, w, |img, ih, iw| detector.largest_face(img, ih, iw))?
                    .map(|bbox| face_crop_box(bbox, h, w)),
            )
        })
        .collect()
}

fn check_live(name: &str, live: &Array, frames: usize, image_hw: (usize, usize)) -> Result<()> {
    let (n, h, w) = geometry(live);
    if (n, (h, w)) != (frames, image_hw) {
        return Err(Error::Msg(format!(
            "{name} loss: live decode is {n}×{h}×{w} but the reference was built at \
             {frames}×{}×{}",
            image_hw.0, image_hw.1
        )));
    }
    Ok(())
}

/// Upstream's masked mean: `Σ term·gate / max(Σ gate, 1)` over the face-bearing frames, the gates
/// gradient-free (a frame whose gate is closed contributes nothing to numerator or denominator).
fn gated_mean(terms: Vec<(Array, Array)>) -> Result<Array> {
    let mut num: Option<Array> = None;
    let mut den: Option<Array> = None;
    for (term, gate) in terms {
        let gate = mlx_rs::stop_gradient(&gate)?;
        let t = multiply(&term, &gate)?;
        num = Some(match num {
            Some(n) => add(&n, &t)?,
            None => t,
        });
        den = Some(match den {
            Some(d) => add(&d, &gate)?,
            None => gate,
        });
    }
    let (num, den) = num
        .zip(den)
        .ok_or_else(|| Error::Msg("face loss: no frame carries a reference".into()))?;
    Ok(divide(&num, &maximum(&den, Array::from_f32(1.0))?)?)
}

// ------------------------------------------------------------------------------------------------
// Identity
// ------------------------------------------------------------------------------------------------

/// One frame's identity reference: its face box and that crop's unit ArcFace embedding (before
/// bias-centring — the scorer centres at loss time).
pub struct IdentityFrame {
    pub crop: CropBox,
    /// `[D]`, unit norm, gradient-free.
    pub embedding: Array,
}

/// The per-image identity reference: per decoded frame (one for an image; the step's frames for a
/// video decoder), the frame's face box + embedding, or `None` for a frame without a face.
pub struct IdentityReference {
    pub image_hw: (usize, usize),
    pub frames: Vec<Option<IdentityFrame>>,
}

impl IdentityReference {
    /// A single-frame reference.
    pub fn single(crop: CropBox, image_hw: (usize, usize), embedding: Array) -> Self {
        Self {
            image_hw,
            frames: vec![Some(IdentityFrame { crop, embedding })],
        }
    }
}

#[derive(Default)]
struct DatasetMean {
    sum: Option<Array>,
    count: usize,
    /// Set by the first score; the references are complete from then on.
    frozen: Option<Array>,
}

/// The frozen ArcFace scorer behind the identity loss — and behind the landmark loss's identity
/// gate, which shares it (one ArcFace, one dataset mean). Ports upstream's scoring:
///
/// - **bias centring**: ArcFace maps every non-face to a tight cluster (~0.5 cosine against faces),
///   so the mean unit embedding of [`IDENTITY_NOISE_SAMPLES`] noise images is subtracted from both
///   sides and the result re-normalized — `center(e) = normalize(e − noise_mean)`; a non-face then
///   scores ~0 and the `min_cos` gate means what it says;
/// - **dataset-average** targets (`identity_loss_use_average`): every image is compared with the
///   normalized mean of all face-bearing reference embeddings, and its loss is normalized by its own
///   clean score `max(cos(center(own), center(mean)), 0.1)` — `max(0, 1 − cos / clean)` — so a
///   profile shot that only reaches 0.7 is not pushed past 0.7.
pub struct IdentityScorer {
    arcface: ArcFace,
    /// `[D]` mean unit embedding of the noise set (not normalized, upstream's `_identity_mean_embed`).
    noise_mean: Array,
    min_cos: f32,
    mode: IdentityReferenceMode,
    mean: RefCell<DatasetMean>,
}

impl IdentityScorer {
    /// Build the scorer, averaging the ArcFace embeddings of the [`IDENTITY_NOISE_SAMPLES`]
    /// counter-based noise images ([`synth::identity_noise_image`] — identical on both backends),
    /// one forward at a time (inference only; bounded by the identity loss's E7 working set).
    pub fn new(arcface: ArcFace, min_cos: f32, mode: IdentityReferenceMode) -> Result<Self> {
        let edge = ARCFACE_INPUT;
        let mut sum: Option<Array> = None;
        for i in 0..IDENTITY_NOISE_SAMPLES {
            let px = synth::identity_noise_image(IDENTITY_NOISE_SEED, i, edge);
            let x = Array::from_slice(&px, &[1, edge as i32, edge as i32, 3]);
            let x = subtract(&multiply(&x, Array::from_f32(2.0))?, Array::from_f32(1.0))?;
            let e = l2_normalize(&arcface.forward(&x)?)?.reshape(&[-1])?;
            e.eval()?;
            sum = Some(match sum {
                Some(s) => add(&s, &e)?,
                None => e,
            });
        }
        let noise_mean = divide(
            sum.expect("IDENTITY_NOISE_SAMPLES > 0"),
            Array::from_f32(IDENTITY_NOISE_SAMPLES as f32),
        )?;
        noise_mean.eval()?;
        Ok(Self {
            arcface,
            noise_mean,
            min_cos,
            mode,
            mean: RefCell::new(DatasetMean::default()),
        })
    }

    /// The bias direction (`[D]`).
    pub fn noise_mean(&self) -> &Array {
        &self.noise_mean
    }

    /// Unit ArcFace embedding `[D]` of the face crop `b` of `px`'s first frame (differentiable in
    /// `px`).
    pub fn embed(&self, px: &Array, b: CropBox) -> Result<Array> {
        let crop = crop_resize(px, b, true, ARCFACE_INPUT)?;
        // `(px·255 − 127.5) / 127.5` = `2·px − 1`.
        let x = subtract(
            &multiply(&crop, Array::from_f32(2.0))?,
            Array::from_f32(1.0),
        )?;
        let emb = self.arcface.forward(&x)?; // [1, D]
        Ok(l2_normalize(&emb)?.reshape(&[-1])?)
    }

    /// `normalize(e − noise_mean)`.
    pub fn center(&self, e: &Array) -> Result<Array> {
        l2_normalize(&subtract(e, &self.noise_mean)?)
    }

    /// One reference frame (gradient-free); `register` adds it to the dataset mean (the identity
    /// loss does; the landmark gate, sharing the scorer, does not — it would double count).
    fn reference_frame(&self, px: &Array, crop: CropBox, register: bool) -> Result<IdentityFrame> {
        let embedding = mlx_rs::stop_gradient(&self.embed(px, crop)?)?;
        embedding.eval()?;
        if register && self.mode == IdentityReferenceMode::DatasetAverage {
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
        Ok(IdentityFrame { crop, embedding })
    }

    /// The frozen dataset mean (normalized), computed on first use.
    fn dataset_mean(&self) -> Result<Array> {
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

    /// `(cos, clean, gate)` of a live frame against its reference frame: the bias-centred cosine
    /// (differentiable in `live`), the clean-cos normalizer (`1` per-image; the image's own clean
    /// score vs the dataset mean, floored at 0.1, in average mode) and the gradient-free gate
    /// `cos > min_cos`.
    pub fn frame_score(&self, live: &Array, rf: &IdentityFrame) -> Result<(Array, Array, Array)> {
        let live_c = self.center(&self.embed(live, rf.crop)?)?;
        let (target, clean) = match self.mode {
            IdentityReferenceMode::PerImage => (rf.embedding.clone(), Array::from_f32(1.0)),
            IdentityReferenceMode::DatasetAverage => {
                let avg_c = self.center(&self.dataset_mean()?)?;
                let own = multiply(&self.center(&rf.embedding)?, &avg_c)?.sum(None)?;
                let clean = maximum(&own, Array::from_f32(IDENTITY_CLEAN_COS_FLOOR))?;
                (self.dataset_mean()?, mlx_rs::stop_gradient(&clean)?)
            }
        };
        let cos = multiply(&live_c, &self.center(&target)?)?.sum(None)?;
        let gate = mlx_rs::stop_gradient(&cos)?
            .gt(Array::from_f32(self.min_cos))?
            .as_type::<f32>()?;
        Ok((cos, clean, gate))
    }
}

/// The ArcFace identity loss (see the module docs): per face-bearing frame
/// `max(0, 1 − cos / clean)`, gated by `cos > min_cos`, averaged over the frames whose gate is open
/// (`max(Σ gate, 1)`), weighted by the noise level (upstream's `t_ratio`).
pub struct IdentityLoss {
    scorer: Rc<IdentityScorer>,
    detector: Rc<dyn FaceBoxDetector>,
}

impl IdentityLoss {
    pub fn new(scorer: Rc<IdentityScorer>, detector: Rc<dyn FaceBoxDetector>) -> Self {
        Self { scorer, detector }
    }

    /// The shared scorer.
    pub fn scorer(&self) -> &Rc<IdentityScorer> {
        &self.scorer
    }

    /// The loss and the mean bias-centred cosine (diagnostics) of `live` against `r`.
    pub fn loss_and_cos(&self, live: &Array, r: &IdentityReference) -> Result<(Array, Array)> {
        check_live("identity", live, r.frames.len(), r.image_hw)?;
        let mut terms = Vec::new();
        let mut coses = Vec::new();
        for (f, rf) in r.frames.iter().enumerate() {
            let Some(rf) = rf else { continue };
            let (cos, clean, gate) = self.scorer.frame_score(&frame(live, f), rf)?;
            let term = maximum(
                &subtract(Array::from_f32(1.0), &divide(&cos, &clean)?)?,
                Array::from_f32(0.0),
            )?;
            terms.push((term, gate));
            coses.push((cos, Array::from_f32(1.0)));
        }
        Ok((gated_mean(terms)?, gated_mean(coses)?))
    }
}

impl PerceptualLoss for IdentityLoss {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
        let boxes = reference_boxes(self.detector.as_ref(), clean)?;
        if boxes.iter().all(Option::is_none) {
            return Ok(None);
        }
        let mut frames = Vec::with_capacity(boxes.len());
        for (f, crop) in boxes.into_iter().enumerate() {
            frames.push(match crop {
                Some(crop) => Some(self.scorer.reference_frame(&frame(clean, f), crop, true)?),
                None => None,
            });
        }
        let (_, h, w) = geometry(clean);
        Ok(Some(Box::new(IdentityReference {
            image_hw: (h, w),
            frames,
        })))
    }

    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<IdentityReference>(self.name(), reference)?;
        Ok(self.loss_and_cos(live, r)?.0)
    }

    fn timestep_weight(&self, noise_level: f32) -> f32 {
        face_loss_timestep_weight(noise_level)
    }
}

// ------------------------------------------------------------------------------------------------
// Landmarks
// ------------------------------------------------------------------------------------------------

/// One frame's landmark reference: its face box, that crop's normalized landmarks and — when the
/// identity loss is on — its identity reference for the landmark gate.
pub struct LandmarkFrame {
    pub crop: CropBox,
    /// `[478, 2]`, normalized, gradient-free.
    pub landmarks: Array,
    pub identity: Option<IdentityFrame>,
}

/// The per-image landmark reference: per decoded frame, the frame's face box + landmarks, or
/// `None` for a frame without a face.
pub struct LandmarkReference {
    pub image_hw: (usize, usize),
    pub frames: Vec<Option<LandmarkFrame>>,
}

/// Centre `[N, 478, 2]` landmarks on the nose tip and scale by the inner-eye distance (≥ 0.01).
pub fn normalize_landmarks(lm: &Array) -> Result<Array> {
    let pick = |i: usize| lm.index((.., i as i32..i as i32 + 1, ..));
    let centered = subtract(lm, pick(NOSE_TIP))?;
    let d = subtract(pick(INNER_EYES.0), pick(INNER_EYES.1))?;
    let inter = mlx_rs::ops::sqrt(&d.square()?.sum_axes(&[-1], true)?)?;
    Ok(divide(
        &centered,
        &maximum(&inter, Array::from_f32(INTER_EYE_FLOOR))?,
    )?)
}

/// Upstream's region-weighted landmark distance between `[478, 2]` sets:
/// `Σ_r w_r · mean_i sqrt(max(‖g_i − r_i‖², ε)) / Σ_r w_r` over [`LANDMARK_REGIONS`].
pub fn landmark_distance(gen: &Array, reference: &Array) -> Result<Array> {
    let mut total: Option<Array> = None;
    let mut weights = 0.0f32;
    for (idx, w) in LANDMARK_REGIONS {
        let idx: Vec<i32> = idx.iter().map(|&i| i as i32).collect();
        let ix = Array::from_slice(&idx, &[idx.len() as i32]);
        let d = subtract(&gen.take_axis(&ix, 0)?, &reference.take_axis(&ix, 0)?)?;
        let sq = maximum(
            &d.square()?.sum_axes(&[-1], false)?,
            Array::from_f32(LANDMARK_EPS),
        )?;
        let term = multiply(&mlx_rs::ops::sqrt(&sq)?.mean(None)?, Array::from_f32(w))?;
        total = Some(match total {
            Some(t) => add(&t, &term)?,
            None => term,
        });
        weights += w;
    }
    Ok(divide(
        total.expect("three regions"),
        Array::from_f32(weights),
    )?)
}

/// The FaceMesh landmark loss (see the module docs): per face-bearing frame the region-weighted
/// landmark distance — gated, when the identity loss is on, by the identity scorer's `cos > min_cos`
/// (upstream reuses the identity cosine: no landmark push on a hallucinated non-face) — averaged
/// over the open frames and weighted by the noise level.
pub struct FaceLandmarkLoss {
    mesh: Program,
    detector: Rc<dyn FaceBoxDetector>,
    gate: Option<Rc<IdentityScorer>>,
}

impl FaceLandmarkLoss {
    /// `gate`: the identity loss's scorer when the identity loss is enabled (upstream gates the
    /// landmark loss on the identity cosine only then).
    pub fn new(
        mesh: Program,
        detector: Rc<dyn FaceBoxDetector>,
        gate: Option<Rc<IdentityScorer>>,
    ) -> Self {
        Self {
            mesh,
            detector,
            gate,
        }
    }

    /// Normalized `[478, 2]` landmarks of the face crop `b` of `px`'s first frame (differentiable
    /// in `px`).
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
        let boxes = reference_boxes(self.detector.as_ref(), clean)?;
        if boxes.iter().all(Option::is_none) {
            return Ok(None);
        }
        let mut frames = Vec::with_capacity(boxes.len());
        for (f, crop) in boxes.into_iter().enumerate() {
            frames.push(match crop {
                Some(crop) => {
                    let px = frame(clean, f);
                    let landmarks = mlx_rs::stop_gradient(&self.landmarks(&px, crop)?)?;
                    landmarks.eval()?;
                    let identity = match &self.gate {
                        Some(s) => Some(s.reference_frame(&px, crop, false)?),
                        None => None,
                    };
                    Some(LandmarkFrame {
                        crop,
                        landmarks,
                        identity,
                    })
                }
                None => None,
            });
        }
        let (_, h, w) = geometry(clean);
        Ok(Some(Box::new(LandmarkReference {
            image_hw: (h, w),
            frames,
        })))
    }

    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<LandmarkReference>(self.name(), reference)?;
        check_live("face-landmark", live, r.frames.len(), r.image_hw)?;
        let mut terms = Vec::new();
        for (f, rf) in r.frames.iter().enumerate() {
            let Some(rf) = rf else { continue };
            let px = frame(live, f);
            let gate = match (&self.gate, &rf.identity) {
                (Some(s), Some(id)) => s.frame_score(&px, id)?.2,
                (Some(_), None) => {
                    return Err(Error::Msg(
                        "face-landmark loss: gated loss with an ungated reference".into(),
                    ))
                }
                (None, _) => Array::from_f32(1.0),
            };
            terms.push((
                landmark_distance(&self.landmarks(&px, rf.crop)?, &rf.landmarks)?,
                gate,
            ));
        }
        gated_mean(terms)
    }

    fn timestep_weight(&self, noise_level: f32) -> f32 {
        face_loss_timestep_weight(noise_level)
    }
}

// ------------------------------------------------------------------------------------------------
// Loading
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

type ScorerKey = (PathBuf, u32, IdentityReferenceMode);

thread_local! {
    /// Live identity scorers by (face dir, min_cos, mode): the identity loss and the landmark
    /// loss's gate, built by separate builder arms of one job, share ONE scorer (one resident
    /// ArcFace, one dataset mean). Weak, so a finished job's scorer is freed.
    static SCORERS: RefCell<Vec<(ScorerKey, Weak<IdentityScorer>)>> = const { RefCell::new(Vec::new()) };
}

/// The identity scorer for `face_dir/`[`ARCFACE_FILE`] and `cfg`'s gate + mode, shared with any
/// live scorer of the same key (see `SCORERS`).
pub fn shared_identity_scorer(
    face_dir: &Path,
    cfg: &IdentityLossConfig,
) -> Result<Rc<IdentityScorer>> {
    let key: ScorerKey = (
        face_dir.to_path_buf(),
        cfg.min_cos.to_bits(),
        cfg.reference_mode,
    );
    if let Some(s) = SCORERS.with(|m| {
        let mut m = m.borrow_mut();
        m.retain(|(_, w)| w.strong_count() > 0);
        m.iter()
            .find(|(k, _)| *k == key)
            .and_then(|(_, w)| w.upgrade())
    }) {
        return Ok(s);
    }
    let arcface = ArcFace::from_weights(&load_weights(&face_dir.join(ARCFACE_FILE))?)?;
    let scorer = Rc::new(IdentityScorer::new(
        arcface,
        cfg.min_cos,
        cfg.reference_mode,
    )?);
    SCORERS.with(|m| m.borrow_mut().push((key, Rc::downgrade(&scorer))));
    Ok(scorer)
}

/// Load the identity loss from the face-analysis stack dir (`face_dir/`[`SCRFD_FILE`] +
/// `face_dir/`[`ARCFACE_FILE`]) with `cfg`'s gate and reference mode.
pub fn load_identity_loss(face_dir: &Path, cfg: &IdentityLossConfig) -> Result<IdentityLoss> {
    Ok(IdentityLoss::new(
        shared_identity_scorer(face_dir, cfg)?,
        load_detector(face_dir)?,
    ))
}

/// Load the face-landmark loss: SCRFD from `face_dir/`[`SCRFD_FILE`], FaceMesh from
/// `mesh_dir/`[`FACEMESH_FILE`]; `identity` (the identity loss's config, when it is enabled) gates
/// it on the shared identity scorer.
pub fn load_face_landmark_loss(
    face_dir: &Path,
    mesh_dir: &Path,
    identity: Option<&IdentityLossConfig>,
) -> Result<FaceLandmarkLoss> {
    let gate = identity
        .map(|cfg| shared_identity_scorer(face_dir, cfg))
        .transpose()?;
    Ok(FaceLandmarkLoss::new(
        Program::from_file(mesh_dir.join(FACEMESH_FILE))?,
        load_detector(face_dir)?,
        gate,
    ))
}

/// On-disk stand-ins for the face-loss checkpoints, for tests of the builder / trainers that must
/// never download real weights: a weightless SCRFD (scalar zeros — loads, never forwarded), the
/// parity fixture's tiny synthetic IResNet ArcFace, and a tiny FaceMesh-I/O fx-program.
pub mod testing {
    use std::collections::HashMap;
    use std::path::Path;

    use mlx_gen::gen_core::fx_program::FORMAT;
    use mlx_gen::gen_core::train::face_loss::synth::{
        tiny_arcface_shapes, tiny_facemesh_shapes, ARCFACE_SEED, FACEMESH_SEED,
        TINY_FACEMESH_PROGRAM,
    };
    use mlx_gen::{Error, Result};
    use mlx_rs::Array;

    use super::{ARCFACE_FILE, FACEMESH_FILE, SCRFD_FILE};
    use crate::synth;

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
                let t = synth::tensor(ARCFACE_SEED, &k, &s);
                (k, t)
            })
            .collect();
        save(arc, None, &dir.join(ARCFACE_FILE))
    }

    /// Write `dir/face_landmarks_detector.safetensors`: a tiny program with FaceMesh's I/O contract
    /// (`[N,3,256,256]` → `[N,1,1,1434]`).
    pub fn write_facemesh(dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| Error::Msg(e.to_string()))?;
        let meta = HashMap::from([
            ("format".to_string(), FORMAT.to_string()),
            ("program".to_string(), TINY_FACEMESH_PROGRAM.to_string()),
        ]);
        let params = tiny_facemesh_shapes()
            .into_iter()
            .map(|(k, s)| {
                let t = synth::tensor(FACEMESH_SEED, &k, &s);
                (k, t)
            })
            .collect();
        save(params, Some(&meta), &dir.join(FACEMESH_FILE))
    }
}

#[cfg(test)]
mod tests;
