//! DWPose YOLOX/RTMW execution, preprocessing, SimCC decoding and OpenPose mapping.
use crate::image_sampling::sample_rgb_bilinear;
use crate::session::ort_err;
use crate::{Error, Result};
use image::RgbImage;
use ort::{session::Session, value::Tensor};
use std::path::Path;
const DET: usize = 640;
const PW: usize = 288; // pose model input width
const PH: usize = 384; // pose model input height
const MEAN: [f32; 3] = [123.675, 116.28, 103.53]; // BGR-channel order (rtmlib feeds BGR)
const STD: [f32; 3] = [58.395, 57.12, 57.375];
const PAD: f32 = 1.25;
const SPLIT: f32 = 2.0;
const COCO_TO_OPENPOSE: [(usize, usize); 17] = [
    (0, 0),
    (2, 6),
    (3, 8),
    (4, 10),
    (5, 5),
    (6, 7),
    (7, 9),
    (8, 12),
    (9, 14),
    (10, 16),
    (11, 11),
    (12, 13),
    (13, 15),
    (14, 2),
    (15, 1),
    (16, 4),
    (17, 3),
];

// ---------------------------------------------------------------------------
// pure detector math (ported from the spike crate; unit-tested without weights)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Box4 {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

fn yolox_preprocess(img: &RgbImage) -> (Vec<f32>, f32) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let ratio = (DET as f32 / h).min(DET as f32 / w);
    let new_w = (w * ratio) as usize;
    let new_h = (h * ratio) as usize;
    let sx = w / new_w.max(1) as f32;
    let sy = h / new_h.max(1) as f32;
    let mut chw = vec![114.0f32; 3 * DET * DET];
    for c in 0..3 {
        let plane = c * DET * DET;
        for dy in 0..new_h {
            let src_y = (dy as f32 + 0.5) * sy - 0.5; // cv2 INTER_LINEAR half-pixel
            for dx in 0..new_w {
                let src_x = (dx as f32 + 0.5) * sx - 0.5;
                let rgb = sample_rgb_bilinear(
                    img,
                    src_x.clamp(0.0, w - 1.0),
                    src_y.clamp(0.0, h - 1.0),
                    0.0,
                );
                chw[plane + dy * DET + dx] = rgb[2 - c];
            }
        }
    }
    (chw, ratio)
}

/// This YOLOX export bakes in NMS: `dets`(1,N,5)=[xyxy,score]. rtmlib's
/// `shape[-1]==5` branch: boxes /= ratio, keep score > 0.3.
///
/// The shape/length are validated against the `(…, N, 5)` contract before indexing
/// (F-102): a mis-pinned ONNX export with a rank < 2 shape or a `dets` buffer shorter
/// than `N*5` would panic on `shape[1]` / `dets[base + 4]` and unwind the async task;
/// return an `Engine` error instead (sc-8904).
fn yolox_decode(dets: &[f32], shape: &[i64], ratio: f32) -> Result<Vec<Box4>> {
    let mut out = Vec::new();
    if shape.len() < 2 {
        return Err(Error::Engine(format!(
            "yolox output rank {} < 2 (expected (…, N, 5))",
            shape.len()
        )));
    }
    if shape.last().copied() != Some(5) {
        return Err(Error::Engine(format!(
            "yolox output last dimension {:?}, expected 5 for embedded-NMS detections",
            shape.last()
        )));
    }
    let n = shape[1].max(0) as usize;
    if dets.len() < n.saturating_mul(5) {
        return Err(Error::Engine(format!(
            "yolox output has {} values but (N={n} × 5) needs {}",
            dets.len(),
            n.saturating_mul(5)
        )));
    }
    for i in 0..n {
        let base = i * 5;
        if dets[base + 4] > 0.3 {
            out.push(Box4 {
                x1: dets[base] / ratio,
                y1: dets[base + 1] / ratio,
                x2: dets[base + 2] / ratio,
                y2: dets[base + 3] / ratio,
            });
        }
    }
    Ok(out)
}

/// Build the (1,3,PH,PW) BGR-normalized pose input for one bbox + return the
/// aspect-fixed crop geometry (center, scale) needed to rescale the output.
fn pose_preprocess(img: &RgbImage, b: &Box4) -> (Vec<f32>, f32, f32, f32, f32) {
    let cx = (b.x1 + b.x2) / 2.0;
    let cy = (b.y1 + b.y2) / 2.0;
    let sw0 = (b.x2 - b.x1) * PAD;
    let sh0 = (b.y2 - b.y1) * PAD;
    let aspect = PW as f32 / PH as f32; // 0.75
    let (sw, sh) = if sw0 > sh0 * aspect {
        (sw0, sw0 / aspect)
    } else {
        (sh0 * aspect, sh0)
    };
    let x0 = cx - sw / 2.0;
    let y0 = cy - sh / 2.0;
    let mut chw = vec![0.0f32; 3 * PH * PW];
    for c in 0..3 {
        let plane = c * PH * PW;
        let (m, sd) = (MEAN[c], STD[c]);
        for dy in 0..PH {
            let src_y = y0 + dy as f32 * sh / PH as f32; // affine inverse (no half-pixel)
            for dx in 0..PW {
                let src_x = x0 + dx as f32 * sw / PW as f32;
                let rgb = sample_rgb_bilinear(img, src_x, src_y, 0.0);
                chw[plane + dy * PW + dx] = (rgb[2 - c] - m) / sd;
            }
        }
    }
    (chw, cx, cy, sw, sh)
}

#[inline]
fn argmax(slice: &[f32]) -> (usize, f32) {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    (bi, bv)
}

/// Decode SimCC outputs → 133 `[x, y, score]` in original px.
///
/// The SimCC-x shape/length are validated against the `(1, K, Wx)` contract before
/// indexing (F-102): a mis-pinned RTMW export with a rank < 3 shape, `K == 0`, or a
/// `simcc_x`/`simcc_y` buffer shorter than `K*Wx` / `K*Wy` would panic on `sx_shape[2]`
/// or the per-keypoint slices and unwind the async task; return an `Engine` error
/// instead (sc-8904).
fn pose_decode(
    simcc_x: &[f32],
    sx_shape: &[i64],
    simcc_y: &[f32],
    sy_shape: &[i64],
    crop: (f32, f32, f32, f32),
) -> Result<Vec<[f32; 3]>> {
    let (cx, cy, sw, sh) = crop;
    if sx_shape.len() < 3 {
        return Err(Error::Engine(format!(
            "rtmw simcc_x rank {} < 3 (expected (1, K, Wx))",
            sx_shape.len()
        )));
    }
    if sy_shape.len() < 3 {
        return Err(Error::Engine(format!(
            "rtmw simcc_y rank {} < 3 (expected (1, K, Wy))",
            sy_shape.len()
        )));
    }
    let k = sx_shape[1].max(0) as usize;
    let wx = sx_shape[2].max(0) as usize;
    if k == 0 {
        return Err(Error::Engine("rtmw simcc_x has 0 keypoints".to_owned()));
    }
    let y_k = sy_shape[1].max(0) as usize;
    if y_k != k {
        return Err(Error::Engine(format!(
            "rtmw simcc keypoint dimensions disagree: x reports {k}, y reports {y_k}"
        )));
    }
    let wy = sy_shape[2].max(0) as usize;
    if wx == 0 || wy == 0 {
        return Err(Error::Engine(format!(
            "rtmw simcc bins must be non-zero (Wx={wx}, Wy={wy})"
        )));
    }
    if simcc_x.len() < k.saturating_mul(wx) || simcc_y.len() < k.saturating_mul(wy) {
        return Err(Error::Engine(format!(
            "rtmw simcc buffers too short: x has {} (needs {}), y has {} (needs {})",
            simcc_x.len(),
            k.saturating_mul(wx),
            simcc_y.len(),
            k.saturating_mul(wy)
        )));
    }
    let x0 = cx - sw / 2.0;
    let y0 = cy - sh / 2.0;
    Ok((0..k)
        .map(|j| {
            let (xloc, mvx) = argmax(&simcc_x[j * wx..(j + 1) * wx]);
            let (yloc, mvy) = argmax(&simcc_y[j * wy..(j + 1) * wy]);
            let val = 0.5 * (mvx + mvy);
            let (lx, ly) = if val <= 0.0 {
                (-1.0, -1.0)
            } else {
                (xloc as f32, yloc as f32)
            };
            let px = (lx / SPLIT) / PW as f32 * sw + x0;
            let py = (ly / SPLIT) / PH as f32 * sh + y0;
            [px, py, val]
        })
        .collect())
}

// ---------------------------------------------------------------------------
// keypoint conversion (pure; ported from pose_adapters.py)
// ---------------------------------------------------------------------------

/// One person's whole-body record: body18 + `hands[left21,right21]` + face68, each
/// `[x, y, conf]` normalized to `[0,1]`.
pub struct PoseRecord {
    pub keypoints: Vec<[f32; 3]>,
    pub hands: [Vec<[f32; 3]>; 2],
    pub face: Vec<[f32; 3]>,
}

/// Convert one person's raw 133 `[x,y,score]` (original px) into the SceneWorks
/// OpenPose record, normalized by (w,h). Mirrors `wholebody_to_openpose`.
///
/// Requires the full COCO-WholeBody-133 keypoint set (F-102): the body/hand/face slices
/// index `kps` up to 132, so a shorter buffer (a mis-pinned RTMW export with a different
/// keypoint count) would panic on `kps[i]` and unwind the async task. Return an `Engine`
/// error instead of indexing OOB (sc-8904).
pub fn wholebody_to_openpose(kps: &[[f32; 3]], w: f32, h: f32) -> Result<PoseRecord> {
    if kps.len() < 133 {
        return Err(Error::Engine(format!(
            "rtmw decode produced {} keypoints, expected 133 (COCO-WholeBody)",
            kps.len()
        )));
    }
    let pt = |i: usize| [kps[i][0] / w, kps[i][1] / h, kps[i][2]];
    let seq = |lo: usize, hi: usize| (lo..hi).map(pt).collect::<Vec<_>>();
    let mut body = vec![[0.0f32; 3]; 18];
    for (op, coco) in COCO_TO_OPENPOSE {
        body[op] = pt(coco);
    }
    let (ls, rs) = (kps[5], kps[6]); // shoulders → neck
    body[1] = [
        (ls[0] + rs[0]) / 2.0 / w,
        (ls[1] + rs[1]) / 2.0 / h,
        ls[2].min(rs[2]),
    ];
    Ok(PoseRecord {
        keypoints: body,
        hands: [seq(91, 112), seq(112, 133)],
        face: seq(23, 91),
    })
}

/// Re-normalize a source-aspect pose into a centered `max(w,h)` SQUARE (pad short
/// axis, never crop) so the stored pose is aspect-canonical. Mirrors `squareify`.
pub fn squareify(rec: &PoseRecord, w: f32, h: f32) -> PoseRecord {
    let side = w.max(h);
    let ox = (side - w) / 2.0;
    let oy = (side - h) / 2.0;
    let sq = |p: &[f32; 3]| [(p[0] * w + ox) / side, (p[1] * h + oy) / side, p[2]];
    PoseRecord {
        keypoints: rec.keypoints.iter().map(sq).collect(),
        hands: [
            rec.hands[0].iter().map(sq).collect(),
            rec.hands[1].iter().map(sq).collect(),
        ],
        face: rec.face.iter().map(sq).collect(),
    }
}

pub struct Detector {
    det: Session,
    pose: Session,
    pub device: &'static str,
}

/// Run a single CHW float input; returns each f32 output as (shape, data), skipping
/// non-f32 outputs (YOLOX's int64 `labels`).
fn run(
    session: &mut Session,
    shape: [i64; 4],
    data: Vec<f32>,
) -> Result<Vec<(Vec<i64>, Vec<f32>)>> {
    let input = Tensor::from_array((shape.to_vec(), data)).map_err(ort_err)?;
    let outputs = session.run(ort::inputs![input]).map_err(ort_err)?;
    let mut out = Vec::new();
    for i in 0..outputs.len() {
        if let Ok((shp, d)) = outputs[i].try_extract_tensor::<f32>() {
            out.push((shp.to_vec(), d.to_vec()));
        }
    }
    Ok(out)
}

impl Detector {
    /// Finalize optional ORT profiling configured by the caller's loader.
    /// No profiling is enabled by the engine or by ordinary execution.
    pub fn finish_profiling(&mut self) -> Result<Vec<std::path::PathBuf>> {
        Ok(vec![
            self.det.end_profiling().map_err(ort_err)?.into(),
            self.pose.end_profiling().map_err(ort_err)?.into(),
        ])
    }

    /// Build the cached detector, preferring the hardware EP (CoreML on Mac / CUDA
    /// off-Mac) and falling back to CPU if the provider can't initialise (mirrors
    /// Python `load_pose_detector`).
    pub fn load(
        det_path: &Path,
        pose_path: &Path,
        load: &mut dyn crate::session::SessionLoader,
    ) -> Result<Self> {
        let ([det, pose], device) =
            crate::session::load_sessions([det_path, pose_path], "DWPose", load)?;
        Ok(Self { det, pose, device })
    }

    /// Detect every person in one image → per-person raw 133 `[x,y,score]` (original
    /// px), largest-person-area first.
    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<Vec<[f32; 3]>>> {
        let (din, ratio) = yolox_preprocess(img);
        let dout = run(&mut self.det, [1, 3, DET as i64, DET as i64], din)?;
        let boxes = match dout.first() {
            Some((shape, data)) => yolox_decode(data, shape, ratio)?,
            None => Vec::new(),
        };
        let mut people = Vec::new();
        for b in &boxes {
            let (pin, cx, cy, sw, sh) = pose_preprocess(img, b);
            let pout = run(&mut self.pose, [1, 3, PH as i64, PW as i64], pin)?;
            if pout.len() < 2 {
                continue;
            }
            // disambiguate simcc_x (last dim 576) vs simcc_y (768) by shape
            let (xi, yi) = if pout[0].0.last() == Some(&((PW * 2) as i64)) {
                (0, 1)
            } else {
                (1, 0)
            };
            people.push(pose_decode(
                &pout[xi].1,
                &pout[xi].0,
                &pout[yi].1,
                &pout[yi].0,
                (cx, cy, sw, sh),
            )?);
        }
        Ok(people)
    }
}

#[cfg(test)]
#[path = "pose_tests.rs"]
mod tests;
