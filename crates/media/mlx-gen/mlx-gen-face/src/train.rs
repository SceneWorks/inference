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
    FACEMESH_FILE, FACEMESH_INPUT, FACEMESH_LANDMARKS, FACEMESH_V2_PARAMS, FACE_CROP_PAD, SCRFD_10G_PARAMS,
    SCRFD_FILE,
};
use mlx_gen::gen_core::train::face_loss::{
    INNER_EYES, INTER_EYE_FLOOR, LANDMARK_EPS, LANDMARK_REGIONS, NOSE_TIP,
};

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
    let pick = |i: usize| lm.index((.., i as i32..i as i32 + 1, ..));
    let centered = subtract(lm, &pick(NOSE_TIP))?;
    let d = subtract(&pick(INNER_EYES.0), &pick(INNER_EYES.1))?;
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
        &total.expect("three regions"),
        Array::from_f32(weights),
    )?)
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
