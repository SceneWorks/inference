//! The two **decoded-x0 face losses** of the perceptual-character-LoRA epic (epic 2123, sc-24831) on
//! the Candle shared perceptual path ([`candle_gen::train::perceptual`]) — the twin of
//! `mlx-gen-face`'s `train` (read its module docs for the full contract). The backend-neutral half —
//! crop box, resample matrices, region weights, footprints — lives in
//! [`candle_gen::gen_core::train::face_loss`], so both backends crop, resample, normalize, weight
//! and budget identically; the committed torch fixture (`crates/media/face_loss_fixtures/`) pins
//! both against upstream.
//!
//! - identity: bias-centred cosine `center(ArcFace(x0_face)) · center(target)` (`center(e) =
//!   normalize(e − noise_mean)`), loss `max(0, 1 − cos / clean)` gated by `cos > min_cos` and averaged
//!   over the open frames; `target` is the image's own reference embedding or the normalized dataset
//!   mean (with the image's clean-cos normalizer).
//! - landmarks: FaceMesh `out[0] → [478, 3][..., :2]`, nose-centred and inner-eye-scaled; loss is the
//!   region-weighted mean landmark distance, gated by the shared identity scorer when the identity
//!   loss is on.
//! - both are scaled by the step's noise level (upstream's `t_ratio`); a missed reference face is
//!   retried on a gray-padded copy.
//! - no face on the reference round trip ⇒ `reference() == None` ⇒ the shared path skips the image.

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

use candle_gen::candle_core::{DType, Device, Tensor, D};
pub use candle_gen::gen_core::train::face_loss::{
    arcface_param_count, crop_resample_matrices, face_crop_box, face_landmark_loss_footprint,
    identity_loss_footprint, CropBox, ARCFACE_FILE, ARCFACE_INPUT, FACEMESH_FILE, FACEMESH_INPUT,
    FACEMESH_LANDMARKS, SCRFD_FILE,
};
use candle_gen::gen_core::train::face_loss::{
    detect_with_retry, face_loss_timestep_weight, synth, IDENTITY_CLEAN_COS_FLOOR,
    IDENTITY_NOISE_SAMPLES, IDENTITY_NOISE_SEED, INNER_EYES, INTER_EYE_FLOOR, LANDMARK_EPS,
    LANDMARK_REGIONS, NOSE_TIP,
};
use candle_gen::gen_core::train::{IdentityLossConfig, IdentityReferenceMode};
use candle_gen::train::perceptual::{reference_as, LossReference, PerceptualLoss};
use candle_gen::{CandleError, Result};

use crate::common::Weights;
use crate::face::detector_blob;
use crate::iresnet::ArcFace;
use crate::program::Program;
use crate::scrfd::Scrfd;

fn err(m: impl Into<String>) -> CandleError {
    CandleError::Msg(m.into())
}

/// Cut `b` out of single-frame NHWC `[1, H, W, C]` pixels and resample it to NCHW `[1, C, out, out]`,
/// differentiably (two constant matmuls; see [`crop_resample_matrices`]).
pub fn crop_resize(px: &Tensor, b: CropBox, square: bool, out: usize) -> Result<Tensor> {
    let (h, w) = (b.height(), b.width());
    let crop = px
        .narrow(1, b.y0, h)?
        .narrow(2, b.x0, w)?
        .squeeze(0)?
        .permute((2, 0, 1))?
        .contiguous()?; // [C, h, w]
    let (ry, rx) = crop_resample_matrices(b, square, out);
    let ry = Tensor::from_vec(ry, (out, h), px.device())?;
    let rxt = Tensor::from_vec(rx, (out, w), px.device())?
        .t()?
        .contiguous()?;
    let y = ry.broadcast_matmul(&crop)?; // [C, out, w]
    let z = y.broadcast_matmul(&rxt)?; // [C, out, out]
    Ok(z.unsqueeze(0)?)
}

/// L2-normalize the last axis.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let n = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(1e-12)?;
    Ok(x.broadcast_div(&n)?)
}

/// NHWC `[N, H, W, 3]` pixels in `[0, 1]` → one RGB `u8` buffer per frame (round, clamp).
fn to_rgb_u8_frames(px: &Tensor) -> Result<(Vec<Vec<u8>>, usize, usize)> {
    let (n, h, w, c) = px.dims4()?;
    if n < 1 || c != 3 {
        return Err(err(format!(
            "face loss: expected NHWC [N, H, W, 3] pixels, got {:?}",
            px.dims()
        )));
    }
    let v = (px.detach() * 255.0)?
        .clamp(0f32, 255f32)?
        .round()?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
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
fn geometry(px: &Tensor) -> Result<(usize, usize, usize)> {
    let (n, h, w, _) = px.dims4()?;
    Ok((n, h, w))
}

/// Frame `f` of NHWC pixels as `[1, H, W, C]`.
fn frame(px: &Tensor, f: usize) -> Result<Tensor> {
    Ok(px.narrow(0, f, 1)?)
}

/// Finds the largest face on an RGB `u8` image (reference time only).
pub trait FaceBoxDetector: Send + Sync {
    /// `[x1, y1, x2, y2]` of the largest face in original-image pixels, or `None`.
    fn largest_face(&self, rgb: &[u8], h: usize, w: usize) -> Result<Option<[f32; 4]>>;
}

/// The SCRFD-10g detector at insightface's defaults (score 0.5, NMS 0.4), largest face first.
pub struct ScrfdDetector {
    scrfd: Scrfd,
    device: Device,
    pub det_thresh: f32,
    pub nms_thresh: f32,
}

impl FaceBoxDetector for ScrfdDetector {
    fn largest_face(&self, rgb: &[u8], h: usize, w: usize) -> Result<Option<[f32; 4]>> {
        let (blob, det_scale) = detector_blob(rgb, h, w, &self.device)?;
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

/// Detect the reference face on every frame of the decoded round trip, with upstream's gray-pad
/// retry on a miss ([`detect_with_retry`]).
fn reference_boxes(detector: &dyn FaceBoxDetector, clean: &Tensor) -> Result<Vec<Option<CropBox>>> {
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

fn check_live(name: &str, live: &Tensor, frames: usize, image_hw: (usize, usize)) -> Result<()> {
    let (n, h, w) = geometry(live)?;
    if (n, (h, w)) != (frames, image_hw) {
        return Err(err(format!(
            "{name} loss: live decode is {n}×{h}×{w} but the reference was built at \
             {frames}×{}×{}",
            image_hw.0, image_hw.1
        )));
    }
    Ok(())
}

/// Upstream's masked mean `Σ term·gate / max(Σ gate, 1)` (gates detached).
fn gated_mean(terms: Vec<(Tensor, Tensor)>) -> Result<Tensor> {
    let mut num: Option<Tensor> = None;
    let mut den: Option<Tensor> = None;
    for (term, gate) in terms {
        let gate = gate.detach();
        let t = (&term * &gate)?;
        num = Some(match num {
            Some(n) => (n + t)?,
            None => t,
        });
        den = Some(match den {
            Some(d) => (d + gate)?,
            None => gate,
        });
    }
    let (num, den) = num
        .zip(den)
        .ok_or_else(|| err("face loss: no frame carries a reference"))?;
    Ok(num.broadcast_div(&den.maximum(1.0)?)?)
}

fn scalar_tensor(v: f32, device: &Device) -> Result<Tensor> {
    Ok(Tensor::new(v, device)?)
}

// ------------------------------------------------------------------------------------------------
// Identity
// ------------------------------------------------------------------------------------------------

/// One frame's identity reference: its face box and that crop's unit ArcFace embedding.
pub struct IdentityFrame {
    pub crop: CropBox,
    /// `[D]`, unit norm, detached.
    pub embedding: Tensor,
}

/// The per-image identity reference: per decoded frame, the frame's face box + embedding, or `None`
/// for a frame without a face.
pub struct IdentityReference {
    pub image_hw: (usize, usize),
    pub frames: Vec<Option<IdentityFrame>>,
}

impl IdentityReference {
    /// A single-frame reference.
    pub fn single(crop: CropBox, image_hw: (usize, usize), embedding: Tensor) -> Self {
        Self {
            image_hw,
            frames: vec![Some(IdentityFrame { crop, embedding })],
        }
    }
}

#[derive(Default)]
struct DatasetMean {
    sum: Option<Tensor>,
    count: usize,
    frozen: Option<Tensor>,
}

/// The frozen ArcFace scorer behind the identity loss and the landmark loss's identity gate — the
/// twin of `mlx-gen-face`'s `IdentityScorer` (bias centring against the mean embedding of
/// [`IDENTITY_NOISE_SAMPLES`] counter-based noise images, dataset-average targets with the clean-cos
/// normalizer).
pub struct IdentityScorer {
    arcface: ArcFace,
    noise_mean: Tensor,
    min_cos: f32,
    mode: IdentityReferenceMode,
    mean: Mutex<DatasetMean>,
}

impl IdentityScorer {
    /// Build the scorer, averaging the unit ArcFace embeddings of the noise set on `device`, one
    /// forward at a time.
    pub fn new(
        arcface: ArcFace,
        min_cos: f32,
        mode: IdentityReferenceMode,
        device: &Device,
    ) -> Result<Self> {
        let edge = ARCFACE_INPUT;
        let mut sum: Option<Tensor> = None;
        for i in 0..IDENTITY_NOISE_SAMPLES {
            let px = synth::identity_noise_image(IDENTITY_NOISE_SEED, i, edge);
            let x = Tensor::from_vec(px, (1, edge, edge, 3), device)?
                .permute((0, 3, 1, 2))?
                .contiguous()?
                .affine(2.0, -1.0)?;
            let e = l2_normalize(&arcface.forward(&x)?)?.flatten_all()?;
            sum = Some(match sum {
                Some(s) => (s + e)?,
                None => e,
            });
        }
        let noise_mean =
            (sum.expect("IDENTITY_NOISE_SAMPLES > 0") / IDENTITY_NOISE_SAMPLES as f64)?;
        Ok(Self {
            arcface,
            noise_mean,
            min_cos,
            mode,
            mean: Mutex::new(DatasetMean::default()),
        })
    }

    /// The bias direction (`[D]`).
    pub fn noise_mean(&self) -> &Tensor {
        &self.noise_mean
    }

    /// Unit ArcFace embedding `[D]` of the face crop `b` of `px`'s first frame (differentiable in
    /// `px`).
    pub fn embed(&self, px: &Tensor, b: CropBox) -> Result<Tensor> {
        let crop = crop_resize(&frame(px, 0)?, b, true, ARCFACE_INPUT)?;
        // `(px·255 − 127.5) / 127.5` = `2·px − 1`.
        let x = crop.affine(2.0, -1.0)?;
        let emb = self.arcface.forward(&x)?; // [1, D]
        Ok(l2_normalize(&emb)?.flatten_all()?)
    }

    /// `normalize(e − noise_mean)`.
    pub fn center(&self, e: &Tensor) -> Result<Tensor> {
        l2_normalize(&(e - &self.noise_mean)?)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, DatasetMean> {
        candle_gen::lock_recover(&self.mean)
    }

    fn reference_frame(&self, px: &Tensor, crop: CropBox, register: bool) -> Result<IdentityFrame> {
        let embedding = self.embed(&px.detach(), crop)?.detach();
        if register && self.mode == IdentityReferenceMode::DatasetAverage {
            let mut m = self.lock();
            if m.frozen.is_some() {
                return Err(err(
                    "identity loss: a dataset-average reference was added after training began; \
                     every image's reference must be built before the first step",
                ));
            }
            m.sum = Some(match m.sum.take() {
                Some(s) => (s + &embedding)?,
                None => embedding.clone(),
            });
            m.count += 1;
        }
        Ok(IdentityFrame { crop, embedding })
    }

    fn dataset_mean(&self) -> Result<Tensor> {
        let mut m = self.lock();
        if m.frozen.is_none() {
            let sum = m
                .sum
                .as_ref()
                .ok_or_else(|| err("identity loss: dataset average of zero references"))?;
            let mean = l2_normalize(&(sum / m.count as f64)?)?;
            m.frozen = Some(mean);
        }
        Ok(m.frozen.clone().expect("frozen above"))
    }

    /// `(cos, clean, gate)` of a live frame against its reference frame (see the MLX twin).
    pub fn frame_score(
        &self,
        live: &Tensor,
        rf: &IdentityFrame,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dev = live.device();
        let live_c = self.center(&self.embed(live, rf.crop)?)?;
        let (target, clean) = match self.mode {
            IdentityReferenceMode::PerImage => (rf.embedding.clone(), scalar_tensor(1.0, dev)?),
            IdentityReferenceMode::DatasetAverage => {
                let avg = self.dataset_mean()?;
                let own = (self.center(&rf.embedding)? * self.center(&avg)?)?.sum_all()?;
                (avg, own.maximum(IDENTITY_CLEAN_COS_FLOOR as f64)?.detach())
            }
        };
        let cos = (&live_c * self.center(&target)?)?.sum_all()?;
        let gate = cos.detach().gt(self.min_cos as f64)?.to_dtype(DType::F32)?;
        Ok((cos, clean, gate))
    }
}

/// The ArcFace identity loss (see the MLX twin).
pub struct IdentityLoss {
    scorer: Arc<IdentityScorer>,
    detector: Arc<dyn FaceBoxDetector>,
}

impl IdentityLoss {
    pub fn new(scorer: Arc<IdentityScorer>, detector: Arc<dyn FaceBoxDetector>) -> Self {
        Self { scorer, detector }
    }

    /// The shared scorer.
    pub fn scorer(&self) -> &Arc<IdentityScorer> {
        &self.scorer
    }

    /// The loss and the mean bias-centred cosine (diagnostics) of `live` against `r`.
    pub fn loss_and_cos(&self, live: &Tensor, r: &IdentityReference) -> Result<(Tensor, Tensor)> {
        check_live("identity", live, r.frames.len(), r.image_hw)?;
        let mut terms = Vec::new();
        let mut coses = Vec::new();
        for (f, rf) in r.frames.iter().enumerate() {
            let Some(rf) = rf else { continue };
            let (cos, clean, gate) = self.scorer.frame_score(&frame(live, f)?, rf)?;
            let term = cos.broadcast_div(&clean)?.affine(-1.0, 1.0)?.maximum(0.0)?;
            terms.push((term, gate));
            coses.push((cos, scalar_tensor(1.0, live.device())?));
        }
        Ok((gated_mean(terms)?, gated_mean(coses)?))
    }
}

impl PerceptualLoss for IdentityLoss {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
        let boxes = reference_boxes(self.detector.as_ref(), clean)?;
        if boxes.iter().all(Option::is_none) {
            return Ok(None);
        }
        let mut frames = Vec::with_capacity(boxes.len());
        for (f, crop) in boxes.into_iter().enumerate() {
            frames.push(match crop {
                Some(crop) => Some(self.scorer.reference_frame(&frame(clean, f)?, crop, true)?),
                None => None,
            });
        }
        let (_, h, w) = geometry(clean)?;
        Ok(Some(Box::new(IdentityReference {
            image_hw: (h, w),
            frames,
        })))
    }

    fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor> {
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

/// One frame's landmark reference: its face box, normalized landmarks and (gated) identity
/// reference.
pub struct LandmarkFrame {
    pub crop: CropBox,
    /// `[478, 2]`, normalized, detached.
    pub landmarks: Tensor,
    pub identity: Option<IdentityFrame>,
}

/// The per-image landmark reference.
pub struct LandmarkReference {
    pub image_hw: (usize, usize),
    pub frames: Vec<Option<LandmarkFrame>>,
}

/// Centre `[478, 2]` landmarks on the nose tip and scale by the inner-eye distance (≥ 0.01).
pub fn normalize_landmarks(lm: &Tensor) -> Result<Tensor> {
    let pick = |i: usize| lm.narrow(0, i, 1);
    let centered = lm.broadcast_sub(&pick(NOSE_TIP)?)?;
    let d = (pick(INNER_EYES.0)? - pick(INNER_EYES.1)?)?;
    let inter = d
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .maximum(INTER_EYE_FLOOR as f64)?;
    Ok(centered.broadcast_div(&inter)?)
}

/// Upstream's region-weighted landmark distance between `[478, 2]` sets.
pub fn landmark_distance(gen: &Tensor, reference: &Tensor) -> Result<Tensor> {
    let mut total: Option<Tensor> = None;
    let mut weights = 0.0f64;
    for (idx, w) in LANDMARK_REGIONS {
        let ix: Vec<u32> = idx.iter().map(|&i| i as u32).collect();
        let ix = Tensor::from_vec(ix, idx.len(), gen.device())?;
        let d = (gen.index_select(&ix, 0)? - reference.index_select(&ix, 0)?)?;
        let dist = d
            .sqr()?
            .sum(D::Minus1)?
            .maximum(LANDMARK_EPS as f64)?
            .sqrt()?
            .mean_all()?;
        let term = (dist * w as f64)?;
        total = Some(match total {
            Some(t) => (t + term)?,
            None => term,
        });
        weights += w as f64;
    }
    Ok((total.expect("three regions") / weights)?)
}

/// The FaceMesh landmark loss (see the MLX twin), gated by the shared identity scorer when the
/// identity loss is on.
pub struct FaceLandmarkLoss {
    mesh: Program,
    detector: Arc<dyn FaceBoxDetector>,
    gate: Option<Arc<IdentityScorer>>,
}

impl FaceLandmarkLoss {
    pub fn new(
        mesh: Program,
        detector: Arc<dyn FaceBoxDetector>,
        gate: Option<Arc<IdentityScorer>>,
    ) -> Self {
        Self {
            mesh,
            detector,
            gate,
        }
    }

    /// Normalized `[478, 2]` landmarks of the face crop `b` of `px`'s first frame (differentiable
    /// in `px`).
    pub fn landmarks(&self, px: &Tensor, b: CropBox) -> Result<Tensor> {
        let crop = crop_resize(&frame(px, 0)?, b, false, FACEMESH_INPUT)?;
        let out = self.mesh.forward(&crop)?;
        let raw = out
            .first()
            .ok_or_else(|| err("face-landmark loss: FaceMesh has no outputs"))?;
        if raw.elem_count() != FACEMESH_LANDMARKS * 3 {
            return Err(err(format!(
                "face-landmark loss: FaceMesh output 0 has {} values, want {}",
                raw.elem_count(),
                FACEMESH_LANDMARKS * 3
            )));
        }
        let xy = raw
            .contiguous()?
            .reshape((FACEMESH_LANDMARKS, 3))?
            .narrow(1, 0, 2)?;
        normalize_landmarks(&xy)
    }
}

impl PerceptualLoss for FaceLandmarkLoss {
    fn name(&self) -> &'static str {
        "face-landmark"
    }

    fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
        let boxes = reference_boxes(self.detector.as_ref(), clean)?;
        if boxes.iter().all(Option::is_none) {
            return Ok(None);
        }
        let clean = clean.detach();
        let mut frames = Vec::with_capacity(boxes.len());
        for (f, crop) in boxes.into_iter().enumerate() {
            frames.push(match crop {
                Some(crop) => {
                    let px = frame(&clean, f)?;
                    let identity = match &self.gate {
                        Some(s) => Some(s.reference_frame(&px, crop, false)?),
                        None => None,
                    };
                    Some(LandmarkFrame {
                        crop,
                        landmarks: self.landmarks(&px, crop)?.detach(),
                        identity,
                    })
                }
                None => None,
            });
        }
        let (_, h, w) = geometry(&clean)?;
        Ok(Some(Box::new(LandmarkReference {
            image_hw: (h, w),
            frames,
        })))
    }

    fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor> {
        let r = reference_as::<LandmarkReference>(self.name(), reference)?;
        check_live("face-landmark", live, r.frames.len(), r.image_hw)?;
        let mut terms = Vec::new();
        for (f, rf) in r.frames.iter().enumerate() {
            let Some(rf) = rf else { continue };
            let px = frame(live, f)?;
            let gate = match (&self.gate, &rf.identity) {
                (Some(s), Some(id)) => s.frame_score(&px, id)?.2,
                (Some(_), None) => {
                    return Err(err(
                        "face-landmark loss: gated loss with an ungated reference",
                    ))
                }
                (None, _) => scalar_tensor(1.0, live.device())?,
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

fn load_weights(p: &Path, device: &Device) -> Result<Weights> {
    Weights::from_file(p, device)
        .map_err(|e| err(format!("face loss: could not load {}: {e}", p.display())))
}

fn load_detector(face_dir: &Path, device: &Device) -> Result<Arc<dyn FaceBoxDetector>> {
    let scrfd = Scrfd::from_weights(&load_weights(&face_dir.join(SCRFD_FILE), device)?)?;
    Ok(Arc::new(ScrfdDetector {
        scrfd,
        device: device.clone(),
        det_thresh: 0.5,
        nms_thresh: 0.4,
    }))
}

type ScorerKey = (PathBuf, u32, IdentityReferenceMode, String);

/// Live identity scorers by (face dir, min_cos, mode, device): the identity loss and the landmark
/// loss's gate, built by separate builder arms of one job, share ONE scorer. Weak, so a finished
/// job's scorer is freed.
static SCORERS: Mutex<Vec<(ScorerKey, Weak<IdentityScorer>)>> = Mutex::new(Vec::new());

/// The identity scorer for `face_dir/`[`ARCFACE_FILE`] and `cfg`'s gate + mode on `device`, shared
/// with any live scorer of the same key.
pub fn shared_identity_scorer(
    face_dir: &Path,
    cfg: &IdentityLossConfig,
    device: &Device,
) -> Result<Arc<IdentityScorer>> {
    let key: ScorerKey = (
        face_dir.to_path_buf(),
        cfg.min_cos.to_bits(),
        cfg.reference_mode,
        format!("{device:?}"),
    );
    let mut scorers = candle_gen::lock_recover(&SCORERS);
    scorers.retain(|(_, w)| w.strong_count() > 0);
    if let Some(s) = scorers
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, w)| w.upgrade())
    {
        return Ok(s);
    }
    let arcface = ArcFace::from_weights(&load_weights(&face_dir.join(ARCFACE_FILE), device)?)?;
    let scorer = Arc::new(IdentityScorer::new(
        arcface,
        cfg.min_cos,
        cfg.reference_mode,
        device,
    )?);
    scorers.push((key, Arc::downgrade(&scorer)));
    Ok(scorer)
}

/// Load the identity loss from the face-analysis stack dir (`face_dir/`[`SCRFD_FILE`] +
/// `face_dir/`[`ARCFACE_FILE`]) with `cfg`'s gate and reference mode, onto `device`.
pub fn load_identity_loss(
    face_dir: &Path,
    cfg: &IdentityLossConfig,
    device: &Device,
) -> Result<IdentityLoss> {
    Ok(IdentityLoss::new(
        shared_identity_scorer(face_dir, cfg, device)?,
        load_detector(face_dir, device)?,
    ))
}

/// Load the face-landmark loss: SCRFD from `face_dir/`[`SCRFD_FILE`], FaceMesh from
/// `mesh_dir/`[`FACEMESH_FILE`], onto `device`; `identity` (the identity loss's config, when it is
/// enabled) gates it on the shared identity scorer.
pub fn load_face_landmark_loss(
    face_dir: &Path,
    mesh_dir: &Path,
    identity: Option<&IdentityLossConfig>,
    device: &Device,
) -> Result<FaceLandmarkLoss> {
    let gate = identity
        .map(|cfg| shared_identity_scorer(face_dir, cfg, device))
        .transpose()?;
    Ok(FaceLandmarkLoss::new(
        Program::from_file(mesh_dir.join(FACEMESH_FILE), device)?,
        load_detector(face_dir, device)?,
        gate,
    ))
}

/// On-disk stand-ins for the face-loss checkpoints (the twin of `mlx-gen-face`'s `testing`): a
/// weightless SCRFD (minimal shapes — loads, never forwarded), the parity fixture's tiny synthetic
/// IResNet ArcFace, and a tiny FaceMesh-I/O fx-program.
pub mod testing {
    use std::collections::HashMap;
    use std::path::Path;

    use candle_gen::candle_core::{Device, Tensor};
    use candle_gen::gen_core::fx_program::FORMAT;
    use candle_gen::gen_core::train::face_loss::synth::{
        scrfd_standin_shapes, tiny_arcface_shapes, tiny_facemesh_shapes, ARCFACE_SEED,
        FACEMESH_SEED, TINY_FACEMESH_PROGRAM,
    };
    use candle_gen::{CandleError, Result};

    use super::{ARCFACE_FILE, FACEMESH_FILE, SCRFD_FILE};
    use crate::synth;

    fn save(
        tensors: HashMap<String, Tensor>,
        meta: Option<HashMap<String, String>>,
        path: &Path,
    ) -> Result<()> {
        // candle-core implements safetensors 0.7's `View` for `Tensor`; this writer (unlike
        // candle-core's own save) carries the metadata.
        let views: Vec<(String, Tensor)> = tensors.into_iter().collect();
        safetensors::serialize_to_file(views, meta, path)
            .map_err(|e| CandleError::Msg(format!("write {}: {e}", path.display())))
    }

    /// Write `dir/scrfd_10g.safetensors` (weightless) + `dir/arcface_iresnet100.safetensors`
    /// (tiny synthetic IResNet).
    pub fn write_face_stack(dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| CandleError::Msg(e.to_string()))?;
        let dev = Device::Cpu;
        let mut scrfd = HashMap::new();
        for (k, s) in scrfd_standin_shapes() {
            scrfd.insert(
                k,
                Tensor::zeros(s, candle_gen::candle_core::DType::F32, &dev)?,
            );
        }
        save(scrfd, None, &dir.join(SCRFD_FILE))?;
        let mut arc = HashMap::new();
        for (k, s) in tiny_arcface_shapes() {
            let t = synth::tensor(ARCFACE_SEED, &k, &s, &dev)?;
            arc.insert(k, t);
        }
        save(arc, None, &dir.join(ARCFACE_FILE))
    }

    /// Write `dir/face_landmarks_detector.safetensors`: a tiny program with FaceMesh's I/O
    /// contract (`[N,3,256,256]` → `[N,1,1,1434]`).
    pub fn write_facemesh(dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| CandleError::Msg(e.to_string()))?;
        let dev = Device::Cpu;
        let mut params = HashMap::new();
        for (k, s) in tiny_facemesh_shapes() {
            let t = synth::tensor(FACEMESH_SEED, &k, &s, &dev)?;
            params.insert(k, t);
        }
        let meta = HashMap::from([
            ("format".to_string(), FORMAT.to_string()),
            ("program".to_string(), TINY_FACEMESH_PROGRAM.to_string()),
        ]);
        save(params, Some(meta), &dir.join(FACEMESH_FILE))
    }
}

#[cfg(test)]
mod tests;
