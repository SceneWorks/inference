//! Backend-neutral half of the epic-2123 **face losses** (sc-24831) — the ArcFace identity loss and
//! the MediaPipe FaceMesh landmark loss — shared by the MLX (`mlx-gen-face::train`) and Candle
//! (`candle-gen-face::train`) ports so both crop, resample, normalize, weight and budget identically:
//!
//! - upstream's face crop box ([`face_crop_box`]: SCRFD box + 15 % per side, Python half-to-even
//!   rounding, clamped; a collapsed box falls back to the frame) and the constant resample matrices
//!   of its crop paths ([`crop_resample_matrices`]: optional centred zero-pad to square, then torch
//!   `F.interpolate(bilinear, align_corners=False)` — applied by each backend as two matmuls, so the
//!   crop is differentiable in the decoded pixels);
//! - the FaceMesh region indices and weights of upstream's landmark loss;
//! - the E7 footprints ([`identity_loss_footprint`], [`face_landmark_loss_footprint`]);
//! - the counter-based synthetic generator ([`synth`]) the cross-backend parity fixture
//!   (`crates/media/face_loss_fixtures/`) is pinned on.
//!
//! Upstream reference: ai-toolkit-perceptual @ 6e01a6e, `toolkit/face_id.py`
//! (`DifferentiableFaceEncoder`, `DifferentiableLandmarkEncoder`) and `SDTrainer.py`'s loss blocks.

use super::aux_schedule::AuxModelFootprint;

/// SCRFD detector checkpoint in the face-analysis stack dir (the `instantid_face_stack` bundle).
pub const SCRFD_FILE: &str = "scrfd_10g.safetensors";
/// ArcFace checkpoint in the face-analysis stack dir (antelopev2 `glintr100`, iresnet100).
pub const ARCFACE_FILE: &str = "arcface_iresnet100.safetensors";
/// Converted FaceMesh-v2 program checkpoint (`tools/convert_mp_facemesh_v2.py`).
pub const FACEMESH_FILE: &str = "face_landmarks_detector.safetensors";

/// Upstream's face-box expansion on every side (`bw * 0.15`, `bh * 0.15`).
pub const FACE_CROP_PAD: f64 = 0.15;
/// ArcFace input edge.
pub const ARCFACE_INPUT: usize = 112;
/// FaceMesh input edge.
pub const FACEMESH_INPUT: usize = 256;
/// Number of FaceMesh-v2 landmarks (output 0 is `[N, 1, 1, 478·3]`).
pub const FACEMESH_LANDMARKS: usize = 478;
/// The landmark the normalization centres on (nose tip).
pub const NOSE_TIP: usize = 1;
/// The inner-eye pair whose distance scales the normalized landmarks.
pub const INNER_EYES: (usize, usize) = (133, 362);
/// Floor on the inner-eye distance.
pub const INTER_EYE_FLOOR: f32 = 0.01;
/// Floor under each landmark distance's `sqrt` (no NaN gradient at 0).
pub const LANDMARK_EPS: f32 = 1e-6;

/// MediaPipe FaceMesh jaw / face-oval indices.
pub const FACE_OVAL: [usize; 36] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
/// Lip indices.
pub const LIPS: [usize; 20] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
];
/// Mid-face indices — left eye, right eye, nose.
pub const MIDFACE: [usize; 45] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381,
    380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 1, 2, 98, 327, 168, 6, 197,
    195, 5, 4, 19, 94, 370,
];
/// `(indices, weight)` of each landmark region; the loss is `Σ w·mean_dist / Σ w` (3 + 2 + 1 = 6).
pub const LANDMARK_REGIONS: [(&[usize], f32); 3] =
    [(&FACE_OVAL, 3.0), (&LIPS, 2.0), (&MIDFACE, 1.0)];

/// A half-open integer pixel box `[x0, x1) × [y0, y1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropBox {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl CropBox {
    pub fn width(&self) -> usize {
        self.x1 - self.x0
    }
    pub fn height(&self) -> usize {
        self.y1 - self.y0
    }
}

/// Upstream's face crop box for a detector `bbox` `[x1, y1, x2, y2]` on an `h × w` image: expand by
/// [`FACE_CROP_PAD`] per side, round half-to-even (Python `round`), clamp to the image; a box that
/// collapses falls back to the full frame (upstream's `else: crop = pixels[i:i+1]`).
pub fn face_crop_box(bbox: [f32; 4], h: usize, w: usize) -> CropBox {
    let [x1, y1, x2, y2] = bbox.map(|v| v as f64);
    let (pw, ph) = ((x2 - x1) * FACE_CROP_PAD, (y2 - y1) * FACE_CROP_PAD);
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

/// The `[out, in]` matrix (row-major) of torch's bilinear `F.interpolate(align_corners=False)` (no
/// antialias) along one axis: source `max((o + 0.5)·in/out − 0.5, 0)`, taps `⌊s⌋` and
/// `min(⌊s⌋+1, in−1)`.
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

/// `len` real pixels at `offset` inside a zero-padded axis of `padded`, resized to `out`: the padded
/// axis's [`bilinear_matrix`] restricted to the real columns (padding is zero, so its columns drop).
fn axis_matrix(out: usize, padded: usize, offset: usize, len: usize) -> Vec<f32> {
    let full = bilinear_matrix(out, padded);
    let mut m = Vec::with_capacity(out * len);
    for o in 0..out {
        m.extend_from_slice(&full[o * padded + offset..o * padded + offset + len]);
    }
    m
}

/// The row matrix `ry` (`[out, h]`) and column matrix `rx` (`[out, w]`) that resample box `b`'s
/// `h × w` crop to `out × out` as `ry · crop · rxᵀ`. `square` first zero-pads the shorter side,
/// centred (`diff // 2` before) — upstream's identity crop; otherwise the crop is stretched —
/// upstream's landmark crop.
pub fn crop_resample_matrices(b: CropBox, square: bool, out: usize) -> (Vec<f32>, Vec<f32>) {
    let (h, w) = (b.height(), b.width());
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
    (axis_matrix(out, sy, oy, h), axis_matrix(out, sx, ox, w))
}

/// Upstream's reference-detection retry (`FaceIDExtractor._detect`): when no face is found, the
/// detector runs again on the image centred in a mid-gray (128) border of `max(h, w) / 4` per side
/// (tight close-ups otherwise fire no anchors). Returns `(padded RGB u8, padded h, padded w, pad)`.
pub fn pad_for_detection_retry(rgb: &[u8], h: usize, w: usize) -> (Vec<u8>, usize, usize, usize) {
    let pad = h.max(w) / 4;
    let (ph, pw) = (h + 2 * pad, w + 2 * pad);
    let mut out = vec![128u8; ph * pw * 3];
    for y in 0..h {
        let src = &rgb[y * w * 3..(y + 1) * w * 3];
        let dst = ((y + pad) * pw + pad) * 3;
        out[dst..dst + w * 3].copy_from_slice(src);
    }
    (out, ph, pw, pad)
}

/// Map a box found on the [`pad_for_detection_retry`] image back to the original `h × w` image:
/// subtract the pad and clamp to the frame.
pub fn unpad_detection_box(bbox: [f32; 4], pad: usize, h: usize, w: usize) -> [f32; 4] {
    let p = pad as f32;
    [
        (bbox[0] - p).clamp(0.0, w as f32),
        (bbox[1] - p).clamp(0.0, h as f32),
        (bbox[2] - p).clamp(0.0, w as f32),
        (bbox[3] - p).clamp(0.0, h as f32),
    ]
}

/// Detect the largest face with upstream's retry: `detect(rgb, h, w)` first, then on the
/// [`pad_for_detection_retry`] image, mapping a retry hit back with [`unpad_detection_box`].
pub fn detect_with_retry<E>(
    rgb: &[u8],
    h: usize,
    w: usize,
    mut detect: impl FnMut(&[u8], usize, usize) -> Result<Option<[f32; 4]>, E>,
) -> Result<Option<[f32; 4]>, E> {
    if let Some(b) = detect(rgb, h, w)? {
        return Ok(Some(b));
    }
    let (padded, ph, pw, pad) = pad_for_detection_retry(rgb, h, w);
    Ok(detect(&padded, ph, pw)?.map(|b| unpad_detection_box(b, pad, h, w)))
}

/// Upstream's face-loss timestep weighting (`id_weight = lm_weight = t_ratio`): the loss term of a
/// step at noise level `t ∈ [0, 1]` (flow `σ`, or `t / T`) is scaled by `t` — the face of a
/// high-noise x0 prediction is a genuine generation, a low-noise one is mostly the input.
pub fn face_loss_timestep_weight(noise_level: f32) -> f32 {
    noise_level.clamp(0.0, 1.0)
}

/// Number of noise images whose mean ArcFace embedding is the identity loss's bias direction
/// (upstream: 200).
pub const IDENTITY_NOISE_SAMPLES: usize = 200;
/// Seed of the identity loss's noise set (see [`synth::identity_noise_image`]).
pub const IDENTITY_NOISE_SEED: u64 = 0x5EED_24831;
/// Upstream's floor on a dataset-average clean-cos target.
pub const IDENTITY_CLEAN_COS_FLOOR: f32 = 0.1;

/// Published SCRFD-10g (bnkps) parameter count (insightface model zoo: 4.23 M).
pub const SCRFD_10G_PARAMS: u64 = 4_230_000;
/// Upper bound of the FaceMesh-v2 landmark detector's parameters: the upstream checkpoint is a
/// 5.21 MB f32 pickle (≈ 1.30 M floats; upstream's docstring says 1.2 M).
pub const FACEMESH_V2_PARAMS: u64 = 1_310_000;
/// iresnet100 block counts per layer — antelopev2 `glintr100`, the shipped face stack.
pub const IRESNET100_LAYERS: [usize; 4] = [3, 13, 30, 3];
/// iresnet50 block counts per layer — buffalo_l `w600k_r50`, upstream's identity checkpoint.
pub const IRESNET50_LAYERS: [usize; 4] = [3, 4, 14, 3];
const IRESNET_WIDTHS: [u64; 4] = [64, 128, 256, 512];
const IRESNET_STEM: u64 = 64;
const ARCFACE_EMBEDDING: u64 = 512;

/// Parameters of an insightface IResNet ArcFace with per-stage block counts `layers` (analytic;
/// glintr100 ⇒ ≈ 65.2 M, w600k_r50 ⇒ ≈ 43.6 M).
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
    let mut floats = 2 * IRESNET_STEM * side * side;
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

/// E7 footprint of the identity loss: its SCRFD detector + an IResNet ArcFace of `layers` (the
/// shipped face stack is glintr100, [`IRESNET100_LAYERS`]). The crop is a fixed 112², so the figure
/// does not depend on the training resolution.
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

/// E7 footprint of the face-landmark loss: its SCRFD detector + FaceMesh-v2 on a fixed 256² crop.
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

/// The counter-based synthetic generator of the face-loss parity fixture (twin of `synth`/`image`
/// in `crates/media/face_loss_fixtures/produce_face_loss_fixtures.py`). Every value is a pure
/// function of `(seed, key, index)`: `u_i = (splitmix64(k + i) >> 40) / 2²⁴` with
/// `k = splitmix64(seed ^ fnv1a64(key))`; a tensor value is `offset + (2·u_i − 1)·half` with
/// `(offset, half)` from [`synth::role`].
pub mod synth {
    use crate::train::splitmix64;

    /// Seed of the fixture's tiny ArcFace.
    pub const ARCFACE_SEED: u64 = 0x24831A;
    /// Seed of the fixture's FaceMesh-shaped program.
    pub const FACEMESH_SEED: u64 = 0x24831B;
    /// Seed of the fixture's images.
    pub const IMAGE_SEED: u64 = 0x24831C;

    /// 64-bit FNV-1a of `s`.
    pub fn fnv1a64(s: &str) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        h
    }

    /// `n` values in `[0, 1)` (f64) for `(seed, key)`.
    pub fn uniform(seed: u64, key: &str, n: usize) -> Vec<f64> {
        let k = splitmix64(seed ^ fnv1a64(key));
        (0..n as u64)
            .map(|i| (splitmix64(k.wrapping_add(i)) >> 40) as f64 / 16_777_216.0)
            .collect()
    }

    /// `(offset, half-range)` of a synthetic tensor by key/shape.
    pub fn role(key: &str, shape: &[usize]) -> (f64, f64) {
        if key.contains("prelu") || key.ends_with(".slope") {
            return (0.25, 0.05);
        }
        if key.ends_with(".scale") {
            return (1.0, 0.1);
        }
        if key.ends_with(".shift") || key.ends_with(".bias") {
            return (0.0, 0.05);
        }
        if shape.len() >= 2 {
            let n: usize = shape.iter().product();
            return (0.0, (3.0 / (n / shape[0]) as f64).sqrt());
        }
        (0.0, 0.1)
    }

    /// The synthetic f32 values of tensor `key` with `shape` (row-major).
    pub fn values(seed: u64, key: &str, shape: &[usize]) -> Vec<f32> {
        let (off, half) = role(key, shape);
        let n: usize = shape.iter().product();
        uniform(seed, key, n)
            .into_iter()
            .map(|u| (off + (2.0 * u - 1.0) * half) as f32)
            .collect()
    }

    /// A synthetic NHWC `h × w × 3` image in `[0, 1]` (row-major values).
    pub fn image(seed: u64, key: &str, h: usize, w: usize) -> Vec<f32> {
        uniform(seed, key, h * w * 3)
            .into_iter()
            .map(|u| u as f32)
            .collect()
    }

    /// `n` standard-normal values for `(seed, key)` (Box–Muller over [`uniform`] pairs:
    /// `√(−2 ln(u₀ + 2⁻²⁵)) · cos(2π u₁)`), f64.
    pub fn gaussian(seed: u64, key: &str, n: usize) -> Vec<f64> {
        let u = uniform(seed, key, 2 * n);
        (0..n)
            .map(|i| {
                let r = (-2.0 * (u[2 * i] + 2f64.powi(-25)).ln()).sqrt();
                r * (2.0 * std::f64::consts::PI * u[2 * i + 1]).cos()
            })
            .collect()
    }

    /// A `clamp(randn · 0.3 + 0.5, 0, 1)` noise image (upstream's bias-direction noise) for
    /// `(seed, key)` at `h × w`, NHWC row-major, from the counter-based [`gaussian`].
    pub fn noise_image(seed: u64, key: &str, h: usize, w: usize) -> Vec<f32> {
        gaussian(seed, key, h * w * 3)
            .into_iter()
            .map(|z| (z * 0.3 + 0.5).clamp(0.0, 1.0) as f32)
            .collect()
    }

    /// Noise image `index` of the identity loss's bias-direction set at `edge × edge` — identical on
    /// every backend.
    pub fn identity_noise_image(seed: u64, index: usize, edge: usize) -> Vec<f32> {
        noise_image(seed, &format!("identity-noise-{index}"), edge, edge)
    }

    /// Key → shape of the fixture's tiny IResNet (stem 8, widths 8/16/32/64, blocks `[1,2,1,1]`,
    /// 32-d head; conv kernels OHWI like the converted face-stack files).
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
        for (li, (nb, c)) in [1usize, 2, 1, 1]
            .into_iter()
            .zip([8, 16, 32, 64])
            .enumerate()
        {
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

    /// Every key the native SCRFD-10g loaders (`Scrfd::from_weights`, both backends) require —
    /// stem + `[3,4,2,3]` backbone (stages 2-4 block 0 carry a downsample) + PAFPN neck +
    /// per-stride heads {8,16,32} — with a minimal stand-in shape (`[1,1,1,1]` kernels, `[1]`
    /// vectors): a weightless detector that loads on either backend and is never forwarded.
    pub fn scrfd_standin_shapes() -> Vec<(String, Vec<usize>)> {
        let mut keys: Vec<String> = Vec::new();
        fn conv_into(keys: &mut Vec<String>, p: &str) {
            keys.push(format!("{p}.weight"));
            keys.push(format!("{p}.bias"));
        }
        let conv = conv_into;
        for p in [
            "stem.conv0",
            "stem.conv1",
            "stem.conv2",
            "neck.lateral0",
            "neck.lateral1",
            "neck.lateral2",
            "neck.fpn0",
            "neck.fpn1",
            "neck.fpn2",
            "neck.down0",
            "neck.down1",
            "neck.pafpn0",
            "neck.pafpn1",
        ] {
            conv(&mut keys, p);
        }
        for (l, nb) in [(1usize, 3usize), (2, 4), (3, 2), (4, 3)] {
            for b in 0..nb {
                conv(&mut keys, &format!("stage{l}.{b}.conv1"));
                conv(&mut keys, &format!("stage{l}.{b}.conv2"));
                if b == 0 && l > 1 {
                    conv(&mut keys, &format!("stage{l}.{b}.downsample"));
                }
            }
        }
        for stride in [8, 16, 32] {
            let p = format!("head{stride}");
            for c in ["stem0", "stem1", "stem2", "cls", "reg", "kps"] {
                conv(&mut keys, &format!("{p}.{c}"));
            }
            keys.push(format!("{p}.scale"));
        }
        keys.into_iter()
            .map(|k| {
                let shape = if k.ends_with(".weight") {
                    vec![1, 1, 1, 1]
                } else {
                    vec![1]
                };
                (k, shape)
            })
            .collect()
    }

    /// The tiny FaceMesh-I/O stand-in program the trainer/builder tests write (`[N,3,256,256]` →
    /// max-pool 32 → 8×8 conv → `[N,1,1,1434]`), with its parameter shapes.
    pub const TINY_FACEMESH_PROGRAM: &str = r#"{"inputs":["x"],"outputs":["y"],"nodes":[
        {"op":"maxpool2d","out":"p","inputs":["x"],"kernel":[32,32],"stride":[32,32],"padding":[0,0]},
        {"op":"conv2d","out":"h","inputs":["p"],"weight":"head.weight","bias":"head.bias",
         "stride":[1,1],"padding":[0,0],"dilation":[1,1],"groups":1},
        {"op":"reshape","out":"y","inputs":["h"],"shape":[-1,1,1,1434]}]}"#;

    /// Parameter shapes of [`TINY_FACEMESH_PROGRAM`].
    pub fn tiny_facemesh_shapes() -> Vec<(String, Vec<usize>)> {
        vec![
            ("head.weight".into(), vec![1434, 3, 8, 8]),
            ("head.bias".into(), vec![1434]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python `round` is half-to-even; a box edge on .5 must round to even like upstream.
    /// Mutation: `round_ties_even` → `round` ⇒ x0 becomes 3 ⇒ red.
    #[test]
    fn face_crop_box_rounds_half_to_even_and_falls_back_to_the_frame() {
        // bw = 20 ⇒ pad 3 ⇒ x1 − pad = 2.5 ⇒ 2 (even).
        let b = face_crop_box([5.5, 10.0, 25.5, 30.0], 64, 64);
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (2, 7, 28, 33));
        let full = face_crop_box([70.0, 70.0, 80.0, 80.0], 64, 64);
        assert_eq!((full.x0, full.y0, full.x1, full.y1), (0, 0, 64, 64));
    }

    #[test]
    fn bilinear_matrix_rows_sum_to_one_and_identity_at_equal_size() {
        let m = bilinear_matrix(7, 13);
        for r in m.chunks(13) {
            assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        let id = bilinear_matrix(5, 5);
        for (i, r) in id.chunks(5).enumerate() {
            for (j, v) in r.iter().enumerate() {
                assert_eq!(*v, if i == j { 1.0 } else { 0.0 });
            }
        }
        // Square padding drops the pad columns: a 4-wide crop padded to 6 keeps 4 columns whose
        // rows no longer sum to one at the edges.
        let b = CropBox {
            x0: 0,
            y0: 0,
            x1: 4,
            y1: 6,
        };
        let (ry, rx) = crop_resample_matrices(b, true, 3);
        assert_eq!((ry.len(), rx.len()), (3 * 6, 3 * 4));
    }

    /// The analytic glintr100 count matches the published 65.2 M (261 MB f32 onnx), w600k_r50 the
    /// published 43.6 M. Mutation: drop the downsample term ⇒ red.
    #[test]
    fn arcface_sizes_match_the_published_checkpoints() {
        let r100 = arcface_param_count(IRESNET100_LAYERS);
        assert!((65_000_000..65_400_000).contains(&r100), "{r100}");
        let r50 = arcface_param_count(IRESNET50_LAYERS);
        assert!((43_400_000..43_800_000).contains(&r50), "{r50}");
    }

    /// E7: each face loss carries its detector plus its own model and a per-image reference.
    /// Mutations: drop the detector from `identity_loss_footprint` ⇒ red; drop the ArcFace term ⇒ red.
    #[test]
    fn face_loss_footprints_count_detector_and_model() {
        let det = SCRFD_10G_PARAMS * 4;
        let id = identity_loss_footprint(IRESNET100_LAYERS);
        assert_eq!(
            id.param_bytes,
            det + arcface_param_count(IRESNET100_LAYERS) * 4
        );
        assert!(id.param_bytes > 270_000_000 && id.working_set_bytes > 0);
        assert!(id.reference_bytes_per_image >= 512 * 4);
        let lm = face_landmark_loss_footprint();
        assert_eq!(lm.param_bytes, det + FACEMESH_V2_PARAMS * 4);
        assert!(lm.reference_bytes_per_image >= (FACEMESH_LANDMARKS * 2 * 4) as u64);
    }

    /// Upstream's retry: a miss re-detects on the image centred in a 128-gray border of
    /// `max(h, w) / 4`, and the hit is mapped back (pad subtracted, clamped to the frame); a first
    /// hit is returned untouched. Mutations: skip the retry ⇒ `None` ⇒ red; forget to subtract the
    /// pad ⇒ red; drop the clamp ⇒ the negative / overflowing edges survive ⇒ red; pad with 0 ⇒ the
    /// border check reds.
    #[test]
    fn detection_retries_on_a_gray_padded_image_and_maps_the_box_back() {
        let (h, w) = (40usize, 60usize);
        let rgb: Vec<u8> = (0..h * w * 3).map(|i| (i % 7) as u8).collect();
        let mut calls = Vec::new();
        let got = detect_with_retry(&rgb, h, w, |img, ih, iw| {
            calls.push((ih, iw));
            if (ih, iw) == (h, w) {
                return Ok::<_, ()>(None);
            }
            // The padded frame: gray border, the original pixels at (pad, pad).
            let pad = 15;
            assert_eq!((ih, iw), (h + 2 * pad, w + 2 * pad));
            assert_eq!(&img[..3], &[128, 128, 128]);
            let at = ((pad + 2) * iw + pad + 3) * 3;
            assert_eq!(&img[at..at + 3], &rgb[(2 * w + 3) * 3..(2 * w + 3) * 3 + 3]);
            // A face straddling the top-left of the original frame and past its right edge.
            Ok(Some([10.0, 20.0, 90.0, 50.0]))
        })
        .unwrap();
        assert_eq!(calls, vec![(h, w), (70, 90)]);
        assert_eq!(got, Some([0.0, 5.0, 60.0, 35.0]));
        let first = detect_with_retry(&rgb, h, w, |_, _, _| {
            Ok::<_, ()>(Some([1.0, 2.0, 3.0, 4.0]))
        });
        assert_eq!(first.unwrap(), Some([1.0, 2.0, 3.0, 4.0]));
        let none = detect_with_retry(&rgb, h, w, |_, _, _| Ok::<_, ()>(None));
        assert_eq!(none.unwrap(), None);
    }

    /// Pinned values (the Python producer's generator) — a drift breaks the cross-backend fixture.
    /// Mutation: drop the `>> 40` ⇒ red.
    #[test]
    fn generator_matches_the_producer() {
        assert_eq!(synth::fnv1a64("live"), 0xbf66_95ad_6966_058f);
        let u = synth::uniform(synth::IMAGE_SEED, "live", 3);
        assert_eq!(
            u,
            [0.17715787887573242, 0.5028018951416016, 0.6082891225814819]
        );
    }
}
