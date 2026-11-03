//! YOLO11m person detection: shared letterbox/decode/NMS and native providers.
use crate::image_sampling::sample_rgb_bilinear;
use crate::session::ort_err;
use crate::{Error, Result};
use image::RgbImage;
#[cfg(all(target_os = "macos", feature = "mlx"))]
use mlx_gen::nn::{conv2d, silu, upsample_nearest};
#[cfg(all(target_os = "macos", feature = "mlx"))]
use mlx_gen::weights::Weights;
#[cfg(all(target_os = "macos", feature = "mlx"))]
use mlx_gen::Result as MlxResult;
#[cfg(all(target_os = "macos", feature = "mlx"))]
use mlx_rs::ops::indexing::TryIndexOp;
#[cfg(all(target_os = "macos", feature = "mlx"))]
use mlx_rs::ops::{
    add, concatenate_axis, maximum, multiply, pad, sigmoid, softmax_axis, split, split_sections,
    subtract, sum_axis,
};
#[cfg(all(target_os = "macos", feature = "mlx"))]
use mlx_rs::Array;
use std::path::Path;

// The off-Mac (Windows/Linux candle GPU-worker lane) backend is `ort` (onnxruntime) with the
// CUDA execution provider + a CPU fallback (sc-5498) — the same EP pattern as `pose_jobs`.
use ort::session::Session;
use ort::value::Tensor;

const DET: usize = 640;
#[cfg(all(target_os = "macos", feature = "mlx"))]
const ANCHORS: i32 = 8400;
const PERSON_CLASS: usize = 0;
const PAD_VALUE: f32 = 114.0;
const NMS_IOU: f32 = 0.7;
/// One person box in the *original frame's* pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl Detection {
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }
    fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

/// Letterbox geometry: the resize ratio and the centered left/top pad (px) in the 640
/// input. `un_x`/`un_y` invert it back to original-frame px.
#[derive(Clone, Copy, Debug)]
struct Letterbox {
    ratio: f32,
    pad_x: f32,
    pad_y: f32,
}

impl Letterbox {
    fn compute(w: u32, h: u32) -> Self {
        let (w, h) = (w as f32, h as f32);
        let ratio = (DET as f32 / w).min(DET as f32 / h);
        let new_w = (w * ratio).round();
        let new_h = (h * ratio).round();
        // Ultralytics splits the pad in half with a -0.1 bias on the lead edge.
        let pad_x = ((DET as f32 - new_w) / 2.0 - 0.1).round();
        let pad_y = ((DET as f32 - new_h) / 2.0 - 0.1).round();
        Self {
            ratio,
            pad_x,
            pad_y,
        }
    }

    /// Map a letterbox-space coordinate back into original-frame px.
    fn un_x(&self, x: f32) -> f32 {
        (x - self.pad_x) / self.ratio
    }
    fn un_y(&self, y: f32) -> f32 {
        (y - self.pad_y) / self.ratio
    }
}

/// Destination layout for the shared letterbox preprocessor: NHWC (the layout
/// `runtime_macos::media::nn::conv2d` expects, macOS) or NCHW (the `yolo11m.onnx` export expects,
/// off-Mac). The two backends' letterbox + cv2 half-pixel sampling are byte-identical;
/// only the destination index differs (sc-8906, F-104).
#[derive(Clone, Copy)]
enum InputLayout {
    /// `hwc[(y * DET + x) * 3 + c]`.
    #[cfg(any(test, all(target_os = "macos", feature = "mlx")))]
    Nhwc,
    /// `chw[c * DET * DET + y * DET + x]`.
    Nchw,
}

impl InputLayout {
    /// Flat destination index for pixel `(x, y)` channel `c` in the `3 * DET * DET` buffer.
    #[inline]
    fn index(self, x: usize, y: usize, c: usize) -> usize {
        match self {
            #[cfg(any(test, all(target_os = "macos", feature = "mlx")))]
            InputLayout::Nhwc => (y * DET + x) * 3 + c,
            InputLayout::Nchw => c * DET * DET + y * DET + x,
        }
    }
}

/// Build the (1,640,640,3) RGB `/255` letterboxed 640² input in `layout` order and return
/// the geometry. The letterbox math + cv2 INTER_LINEAR half-pixel sampling are shared by
/// both backends (macOS NHWC / off-Mac NCHW) — previously two byte-identical cfg-gated
/// copies that could silently drift (sc-8906, F-104); only [`InputLayout::index`] differs.
fn preprocess_letterbox(img: &RgbImage, layout: InputLayout) -> (Vec<f32>, Letterbox) {
    let lb = Letterbox::compute(img.width(), img.height());
    let (w, h) = (img.width() as f32, img.height() as f32);
    let new_w = (w * lb.ratio).round();
    let new_h = (h * lb.ratio).round();
    let sx = w / new_w.max(1.0);
    let sy = h / new_h.max(1.0);
    let new_w = new_w as usize;
    let new_h = new_h as usize;
    let pad_x = lb.pad_x as usize;
    let pad_y = lb.pad_y as usize;

    let mut buf = vec![PAD_VALUE / 255.0; DET * DET * 3];
    for dy in 0..new_h {
        let src_y = (dy as f32 + 0.5) * sy - 0.5; // cv2 INTER_LINEAR half-pixel
        let y = dy + pad_y;
        for dx in 0..new_w {
            let src_x = (dx as f32 + 0.5) * sx - 0.5;
            let x = dx + pad_x;
            let rgb = sample_rgb_bilinear(
                img,
                src_x.clamp(0.0, w - 1.0),
                src_y.clamp(0.0, h - 1.0),
                0.0,
            );
            for (channel, value) in rgb.into_iter().enumerate() {
                buf[layout.index(x, y, channel)] = value / 255.0;
            }
        }
    }
    (buf, lb)
}

/// Decode the (1,84,8400) channel-major output into person boxes (original px),
/// pre-NMS. `data` is laid out as `data[channel * anchors + anchor]`.
///
/// The output shape/length are validated against the `(1, channels, anchors)` contract
/// before any indexing (F-102): a user can pin an arbitrary ONNX export via
/// `SCENEWORKS_PERSON_DETECTOR_WEIGHTS` whose output rank/length differs, and reading
/// `shape[1]`/`shape[2]` or `data[channel*anchors + a]` on a shorter buffer would panic
/// (OOB) and unwind the async task. Return an `Engine` error instead (sc-8904).
fn decode(
    data: &[f32],
    shape: &[i64],
    lb: &Letterbox,
    conf: f32,
    frame_w: u32,
    frame_h: u32,
) -> Result<Vec<Detection>> {
    if shape.len() < 3 {
        return Err(Error::Engine(format!(
            "yolo11m output rank {} < 3 (expected (1, channels, anchors))",
            shape.len()
        )));
    }
    let channels = shape[1].max(0) as usize; // 84 = 4 box + 80 classes
    let anchors = shape[2].max(0) as usize; // 8400
    if channels < 5 {
        return Err(Error::Engine(format!(
            "yolo11m output has {channels} channels, expected at least 5 (box + person score)"
        )));
    }
    if data.len() < channels.saturating_mul(anchors) {
        return Err(Error::Engine(format!(
            "yolo11m output has {} values but the (channels={channels} × anchors={anchors}) \
             contract needs {}",
            data.len(),
            channels.saturating_mul(anchors)
        )));
    }
    let score_ch = 4 + PERSON_CLASS;
    let (fw, fh) = (frame_w as f32, frame_h as f32);
    let mut out = Vec::new();
    for a in 0..anchors {
        let score = data[score_ch * anchors + a];
        if score <= conf {
            continue;
        }
        let cx = data[a];
        let cy = data[anchors + a];
        let bw = data[2 * anchors + a];
        let bh = data[3 * anchors + a];
        let x1 = lb.un_x(cx - bw / 2.0).clamp(0.0, fw);
        let y1 = lb.un_y(cy - bh / 2.0).clamp(0.0, fh);
        let x2 = lb.un_x(cx + bw / 2.0).clamp(0.0, fw);
        let y2 = lb.un_y(cy + bh / 2.0).clamp(0.0, fh);
        out.push(Detection {
            x1,
            y1,
            x2,
            y2,
            score,
        });
    }
    Ok(out)
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let union = a.area() + b.area() - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Greedy non-max suppression, score-descending, single (person) class.
fn nms(mut dets: Vec<Detection>, iou_thr: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep: Vec<Detection> = Vec::new();
    for det in dets {
        if keep.iter().all(|k| iou(k, &det) <= iou_thr) {
            keep.push(det);
        }
    }
    keep
}

/// Captured forward outputs: the final `(1,84,8400)` plus the per-block reference
/// points the parity oracle (`refs.safetensors`) checks. Block tensors stay NHWC; the
/// oracle is NCHW, so the test transposes before comparing. The block fields exist only
/// for the parity test — the production `detect_people` path reads `output` alone.
#[cfg(all(target_os = "macos", feature = "mlx"))]
pub struct YoloForward {
    #[cfg_attr(not(test), allow(dead_code))]
    pub block4: Array,
    #[cfg_attr(not(test), allow(dead_code))]
    pub block10: Array,
    #[cfg_attr(not(test), allow(dead_code))]
    pub block16: Array,
    #[cfg_attr(not(test), allow(dead_code))]
    pub block19: Array,
    #[cfg_attr(not(test), allow(dead_code))]
    pub block22: Array,
    pub output: Array,
}

/// Loaded YOLO11m detector: the fused MLX weights plus the precomputed detect anchors
/// and strides (constant for the fixed 640 input).
#[cfg(all(target_os = "macos", feature = "mlx"))]
pub struct Yolo {
    weights: Weights,
    anchor_x: Array,
    anchor_y: Array,
    stride: Array,
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
impl Yolo {
    pub fn from_weights(weights: Weights) -> Self {
        let (ax, ay, st) = Self::build_anchors();
        Self {
            weights,
            anchor_x: Array::from_slice(&ax, &[1, ANCHORS, 1]),
            anchor_y: Array::from_slice(&ay, &[1, ANCHORS, 1]),
            stride: Array::from_slice(&st, &[1, ANCHORS, 1]),
        }
    }

    /// Detect-head anchor centers (cell + 0.5) and per-anchor strides, row-major over
    /// the three feature grids (80²@8, 40²@16, 20²@32) — the order the Detect head
    /// flattens and concatenates its scales.
    fn build_anchors() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut ax = Vec::with_capacity(ANCHORS as usize);
        let mut ay = Vec::with_capacity(ANCHORS as usize);
        let mut st = Vec::with_capacity(ANCHORS as usize);
        for (grid, stride) in [(80usize, 8.0f32), (40, 16.0), (20, 32.0)] {
            for gy in 0..grid {
                for gx in 0..grid {
                    ax.push(gx as f32 + 0.5);
                    ay.push(gy as f32 + 0.5);
                    st.push(stride);
                }
            }
        }
        (ax, ay, st)
    }

    fn w(&self, key: &str) -> MlxResult<&Array> {
        self.weights.require(key)
    }

    /// Ultralytics `Conv` (fused conv+BN): `conv2d` over a `<prefix>.conv.weight/bias`
    /// pair, optionally followed by SiLU.
    pub fn conv(
        &self,
        x: &Array,
        prefix: &str,
        stride: i32,
        padding: i32,
        act: bool,
    ) -> MlxResult<Array> {
        let w = self.w(&format!("{prefix}.conv.weight"))?;
        let b = self.w(&format!("{prefix}.conv.bias"))?;
        let y = conv2d(x, w, Some(b), stride, padding)?;
        if act {
            silu(&y)
        } else {
            Ok(y)
        }
    }

    /// A bare `nn.Conv2d` (the Detect head's `cv2.*.2` / `cv3.*.2` leaves): plain
    /// `<prefix>.weight/bias`, no activation.
    fn conv_raw(&self, x: &Array, prefix: &str, stride: i32, padding: i32) -> MlxResult<Array> {
        let w = self.w(&format!("{prefix}.weight"))?;
        let b = self.w(&format!("{prefix}.bias"))?;
        conv2d(x, w, Some(b), stride, padding)
    }

    /// Depthwise 3×3 conv (stride 1, pad 1) over NHWC `x`, from a `<prefix>.conv` pair
    /// with a `[C,3,3,1]` weight. `mlx_rs` conv2d only supports `groups=1`, so this is
    /// the 9-tap shift-multiply-accumulate equivalent: zero-pad by 1, then for each
    /// kernel tap broadcast the per-channel weight over a shifted slice and sum.
    pub fn depthwise3x3(&self, x: &Array, prefix: &str, act: bool) -> MlxResult<Array> {
        let w = self.w(&format!("{prefix}.conv.weight"))?; // (C,3,3,1)
        let b = self.w(&format!("{prefix}.conv.bias"))?; // (C,)
        let sh = x.shape();
        let (h, wd, c) = (sh[1], sh[2], sh[3]);
        // Per-channel taps from the host buffer: layout (C,3,3,1) → idx = cc*9 + ky*3 + kx.
        let wdat = w.as_slice::<f32>();
        let xp = pad(x, &[(0, 0), (1, 1), (1, 1), (0, 0)][..], None, None)?;
        let mut acc: Option<Array> = None;
        for ky in 0..3i32 {
            for kx in 0..3i32 {
                let tap: Vec<f32> = (0..c as usize)
                    .map(|cc| wdat[cc * 9 + (ky * 3 + kx) as usize])
                    .collect();
                let tap = Array::from_slice(&tap, &[1, 1, 1, c]);
                let sl = xp.try_index((.., ky..ky + h, kx..kx + wd, ..))?;
                let term = multiply(&sl, &tap)?;
                acc = Some(match acc {
                    Some(a) => add(&a, &term)?,
                    None => term,
                });
            }
        }
        let y = add(acc.expect("3x3 kernel has taps"), b)?;
        if act {
            silu(&y)
        } else {
            Ok(y)
        }
    }

    /// Ultralytics `Bottleneck`: Conv(k3) → Conv(k3) with an optional residual add
    /// (all callers here have `c1 == c2`, so the add is gated only by `shortcut`).
    fn bottleneck(&self, x: &Array, prefix: &str, shortcut: bool) -> MlxResult<Array> {
        let y = self.conv(x, &format!("{prefix}.cv1"), 1, 1, true)?;
        let y = self.conv(&y, &format!("{prefix}.cv2"), 1, 1, true)?;
        if shortcut {
            Ok(add(x, &y)?)
        } else {
            Ok(y)
        }
    }

    /// Ultralytics `C3k` (n=2): `cv3(cat(m(cv1(x)), cv2(x)))`.
    fn c3k(&self, x: &Array, prefix: &str, shortcut: bool) -> MlxResult<Array> {
        let p1 = self.conv(x, &format!("{prefix}.cv1"), 1, 0, true)?;
        let p2 = self.conv(x, &format!("{prefix}.cv2"), 1, 0, true)?;
        let q = self.bottleneck(&p1, &format!("{prefix}.m.0"), shortcut)?;
        let q = self.bottleneck(&q, &format!("{prefix}.m.1"), shortcut)?;
        let cat = concatenate_axis(&[&q, &p2], 3)?;
        self.conv(&cat, &format!("{prefix}.cv3"), 1, 0, true)
    }

    /// Ultralytics `C3k2` (n=1, c3k=True): split cv1 in half, run a C3k on the second
    /// half, then `cv2(cat(a, b, m))`.
    pub fn c3k2(&self, x: &Array, i: usize, shortcut: bool) -> MlxResult<Array> {
        let cv1 = self.conv(x, &format!("model.{i}.cv1"), 1, 0, true)?;
        let parts = split(&cv1, 2, 3)?;
        let (a, b) = (parts[0].clone(), parts[1].clone());
        let m = self.c3k(&b, &format!("model.{i}.m.0"), shortcut)?;
        let cat = concatenate_axis(&[&a, &b, &m], 3)?;
        self.conv(&cat, &format!("model.{i}.cv2"), 1, 0, true)
    }

    /// Ultralytics `SPPF`: cv1 → three chained 5×5 max-pools (s1,p2) → cat → cv2.
    pub fn sppf(&self, x: &Array, i: usize) -> MlxResult<Array> {
        let y = self.conv(x, &format!("model.{i}.cv1"), 1, 0, true)?;
        let p1 = Self::maxpool5(&y)?;
        let p2 = Self::maxpool5(&p1)?;
        let p3 = Self::maxpool5(&p2)?;
        let cat = concatenate_axis(&[&y, &p1, &p2, &p3], 3)?;
        self.conv(&cat, &format!("model.{i}.cv2"), 1, 0, true)
    }

    /// 5×5 max-pool, stride 1, pad 2 — `-inf` pad then elementwise max over the 25
    /// shifted windows (mlx_rs has no pooling op).
    fn maxpool5(x: &Array) -> MlxResult<Array> {
        let sh = x.shape();
        let (h, wd) = (sh[1], sh[2]);
        let ninf = Array::from_f32(f32::NEG_INFINITY);
        let xp = pad(x, &[(0, 0), (2, 2), (2, 2), (0, 0)][..], Some(ninf), None)?;
        let mut acc: Option<Array> = None;
        for ky in 0..5i32 {
            for kx in 0..5i32 {
                let sl = xp.try_index((.., ky..ky + h, kx..kx + wd, ..))?;
                acc = Some(match acc {
                    Some(a) => maximum(&a, &sl)?,
                    None => sl,
                });
            }
        }
        Ok(acc.expect("5x5 window has taps"))
    }

    /// Ultralytics `C2PSA` (block 10): cv1 → split, PSABlock on the second half, cat, cv2.
    pub fn c2psa(&self, x: &Array, i: usize) -> MlxResult<Array> {
        let cv1 = self.conv(x, &format!("model.{i}.cv1"), 1, 0, true)?;
        let parts = split(&cv1, 2, 3)?;
        let (a, b) = (parts[0].clone(), parts[1].clone());
        let b = self.psablock(&b, &format!("model.{i}.m.0"))?;
        let cat = concatenate_axis(&[&a, &b], 3)?;
        self.conv(&cat, &format!("model.{i}.cv2"), 1, 0, true)
    }

    /// `PSABlock`: `x = x + attn(x); x = x + ffn(x)` (ffn = Conv(k1,SiLU) → Conv(k1,no act)).
    fn psablock(&self, x: &Array, prefix: &str) -> MlxResult<Array> {
        let a = self.attention(x, &format!("{prefix}.attn"))?;
        let b1 = add(x, &a)?;
        let f = self.conv(&b1, &format!("{prefix}.ffn.0"), 1, 0, true)?;
        let f = self.conv(&f, &format!("{prefix}.ffn.1"), 1, 0, false)?;
        Ok(add(&b1, &f)?)
    }

    /// Ultralytics `Attention` (num_heads=4, key_dim=32, head_dim=64): 1×1 qkv conv,
    /// per-head scaled-dot-product attention, plus the depthwise positional-encoding of
    /// `v`, then a 1×1 projection. Operates with the spatial map flattened to N=H·W
    /// tokens; channels are head-major (`head*128 + d`) exactly as the torch `view` lays
    /// them out.
    fn attention(&self, x: &Array, prefix: &str) -> MlxResult<Array> {
        let sh = x.shape();
        let (h, wd, c) = (sh[1], sh[2], sh[3]);
        let n = h * wd;
        let (nh, kd, hd) = (4i32, 32i32, 64i32);

        let qkv = self.conv(x, &format!("{prefix}.qkv"), 1, 0, false)?; // (1,H,W,512)
        let qkv = qkv.reshape(&[n, nh, 2 * kd + hd])?; // (N,4,128)
        let parts = split_sections(&qkv, &[kd, 2 * kd], 2)?; // q(32), k(32), v(64)
        let (q, k, v) = (&parts[0], &parts[1], &parts[2]);

        let qh = q.transpose_axes(&[1, 0, 2])?; // (4,N,32)
        let kh = k.transpose_axes(&[1, 2, 0])?; // (4,32,N)
        let scale = Array::from_f32((kd as f32).powf(-0.5));
        // The two SDPA matmuls run on the CPU stream (sc-3734). MLX's Metal `matmul` uses a
        // reduced-precision simdgroup path (≈1e-3 relative — NOT true fp32), which is the
        // sole source of the C2PSA divergence: on the GPU this attention drifts ~7e-3 from
        // the fp32 oracle, vs ~5e-6 on the CPU stream (every conv / the depthwise PE / SPPF
        // are already exact). The map here is tiny (N=400, 4 heads, once per forward), so the
        // CPU detour is negligible. See `yolo11_mlx_per_block_isolation` + docs/sc-3734.
        let cpu = mlx_rs::StreamOrDevice::cpu();
        let attn = multiply(&qh.matmul_device(&kh, &cpu)?, &scale)?; // (4,N,N)
        let attn = softmax_axis(&attn, -1, true)?;
        let vh = v.transpose_axes(&[1, 0, 2])?; // (4,N,64)
        let out = attn.matmul_device(&vh, &cpu)?.transpose_axes(&[1, 0, 2])?; // (N,4,64)
        let x_attn = out.reshape(&[1, h, wd, c])?; // channel = head*64 + d

        let v_nhwc = v.reshape(&[1, h, wd, c])?; // v as a feature map for the PE conv
        let pe = self.depthwise3x3(&v_nhwc, &format!("{prefix}.pe"), false)?;
        let x = add(&x_attn, &pe)?;
        self.conv(&x, &format!("{prefix}.proj"), 1, 0, false)
    }

    /// One Detect scale: box branch (cv2 → 64 = 4·reg_max) and class branch (cv3 → 80),
    /// concatenated and flattened to `(1, Hi·Wi, 144)` (row-major, matching the anchors).
    fn detect_scale(&self, x: &Array, i: usize) -> MlxResult<Array> {
        let bx = self.conv(x, &format!("model.23.cv2.{i}.0"), 1, 1, true)?;
        let bx = self.conv(&bx, &format!("model.23.cv2.{i}.1"), 1, 1, true)?;
        let bx = self.conv_raw(&bx, &format!("model.23.cv2.{i}.2"), 1, 0)?; // 64

        let cx = self.depthwise3x3(x, &format!("model.23.cv3.{i}.0.0"), true)?;
        let cx = self.conv(&cx, &format!("model.23.cv3.{i}.0.1"), 1, 0, true)?;
        let cx = self.depthwise3x3(&cx, &format!("model.23.cv3.{i}.1.0"), true)?;
        let cx = self.conv(&cx, &format!("model.23.cv3.{i}.1.1"), 1, 0, true)?;
        let cx = self.conv_raw(&cx, &format!("model.23.cv3.{i}.2"), 1, 0)?; // 80

        let cat = concatenate_axis(&[&bx, &cx], 3)?; // (1,Hi,Wi,144)
        let sh = cat.shape();
        Ok(cat.reshape(&[1, sh[1] * sh[2], 144])?)
    }

    /// Detect head: assemble the three scales, DFL-decode the box distances, project to
    /// xywh via the precomputed anchors/strides, sigmoid the classes, and emit the
    /// `(1,84,8400)` channel-major tensor the `decode()` consumer expects.
    fn detect(&self, p3: &Array, p4: &Array, p5: &Array) -> MlxResult<Array> {
        let s0 = self.detect_scale(p3, 0)?;
        let s1 = self.detect_scale(p4, 1)?;
        let s2 = self.detect_scale(p5, 2)?;
        let cat = concatenate_axis(&[&s0, &s1, &s2], 1)?; // (1,8400,144)
        let parts = split_sections(&cat, &[64], 2)?; // box(64), cls(80)
        let (box_, cls) = (&parts[0], &parts[1]);

        // DFL: per side, softmax over 16 bins then expected value (the bin indices live
        // in the conv weight, [0..15]).
        let bins = self
            .w("model.23.dfl.conv.weight")?
            .reshape(&[1, 1, 1, 16])?;
        let probs = softmax_axis(&box_.reshape(&[1, ANCHORS, 4, 16])?, 3, true)?;
        let dist = sum_axis(&multiply(&probs, &bins)?, 3, false)?; // (1,8400,4) = l,t,r,b
        let d = split_sections(&dist, &[1, 2, 3], 2)?;
        let (l, t, r, b) = (&d[0], &d[1], &d[2], &d[3]);

        // dist2bbox(xywh) * stride, with anchor centers (cell + 0.5).
        let x1 = subtract(&self.anchor_x, l)?;
        let y1 = subtract(&self.anchor_y, t)?;
        let x2 = add(&self.anchor_x, r)?;
        let y2 = add(&self.anchor_y, b)?;
        let half = Array::from_f32(0.5);
        let cx = multiply(&multiply(&add(&x1, &x2)?, &half)?, &self.stride)?;
        let cy = multiply(&multiply(&add(&y1, &y2)?, &half)?, &self.stride)?;
        let bw = multiply(&subtract(&x2, &x1)?, &self.stride)?;
        let bh = multiply(&subtract(&y2, &y1)?, &self.stride)?;
        let dbox = concatenate_axis(&[&cx, &cy, &bw, &bh], 2)?; // (1,8400,4)

        let cls = sigmoid(cls)?;
        let hwc = concatenate_axis(&[&dbox, &cls], 2)?; // (1,8400,84)
        Ok(hwc.transpose_axes(&[0, 2, 1])?) // (1,84,8400) channel-major
    }

    /// Full YOLO11m forward, capturing the oracle block points. Shortcut residuals are
    /// on for every C3k2 (the C3k2 default; the yaml never overrides it).
    pub fn run(&self, x: &Array) -> MlxResult<YoloForward> {
        // backbone
        let x = self.conv(x, "model.0", 2, 1, true)?; // 320,64
        let x = self.conv(&x, "model.1", 2, 1, true)?; // 160,128
        let x = self.c3k2(&x, 2, true)?; // 160,256
        let x = self.conv(&x, "model.3", 2, 1, true)?; // 80,256
        let x = self.c3k2(&x, 4, true)?; // 80,512
        let block4 = x.clone();
        let x = self.conv(&x, "model.5", 2, 1, true)?; // 40,512
        let x = self.c3k2(&x, 6, true)?; // 40,512
        let b6 = x.clone();
        let x = self.conv(&x, "model.7", 2, 1, true)?; // 20,512
        let x = self.c3k2(&x, 8, true)?; // 20,512
        let x = self.sppf(&x, 9)?; // 20,512
        let x = self.c2psa(&x, 10)?; // 20,512

        // One saved copy of the C2PSA output, reused by the P5 neck concat (below) and returned in
        // `YoloForward.block10` (sc-8952).
        let block10 = x.clone();
        // neck (PANet)
        let x = upsample_nearest(&x, 2)?; // 40,512
        let x = concatenate_axis(&[&x, &b6], 3)?; // 40,1024
        let x = self.c3k2(&x, 13, true)?; // 40,512
        let b13 = x.clone();
        let x = upsample_nearest(&x, 2)?; // 80,512
        let x = concatenate_axis(&[&x, &block4], 3)?; // 80,1024
        let p3 = self.c3k2(&x, 16, true)?; // 80,256  → P3
        let x = self.conv(&p3, "model.17", 2, 1, true)?; // 40,256
        let x = concatenate_axis(&[&x, &b13], 3)?; // 40,768
        let p4 = self.c3k2(&x, 19, true)?; // 40,512  → P4
        let x = self.conv(&p4, "model.20", 2, 1, true)?; // 20,512
        let x = concatenate_axis(&[&x, &block10], 3)?; // 20,1024
        let p5 = self.c3k2(&x, 22, true)?; // 20,512  → P5

        let output = self.detect(&p3, &p4, &p5)?;
        Ok(YoloForward {
            block4,
            block10,
            block16: p3,
            block19: p4,
            block22: p5,
            output,
        })
    }

    pub fn detect_people(&self, img: &RgbImage, conf: f32) -> Result<Vec<Detection>> {
        let (input, lb) = preprocess_letterbox(img, InputLayout::Nhwc);
        let x = Array::from_slice(&input, &[1, DET as i32, DET as i32, 3]);
        // A forward/reshape failure is an engine-execution fault, not a bad user payload —
        // classify it as `Engine` like the `ort` sibling + the SAM2/SAM3 propagate paths
        // (sc-8906, F-104); the old `InvalidPayload` mislabeled an MLX crash as user error.
        let out = self
            .run(&x)
            .map_err(|e| Error::Engine(format!("yolo11m forward: {e}")))?
            .output;
        // The head ends in a transpose, so `out` is a non-contiguous view; `as_slice`
        // would hand back the *physical* (pre-transpose) buffer. Flatten first to force a
        // logical-order copy → the `(1,84,8400)` channel-major layout `decode` indexes.
        let out = out
            .reshape(&[-1])
            .map_err(|e| Error::Engine(format!("yolo11m output reshape: {e}")))?;
        let data = out.as_slice::<f32>();
        let shape = [1_i64, 84, ANCHORS as i64];
        let raw = decode(data, &shape, &lb, conf, img.width(), img.height())?;
        Ok(nms(raw, NMS_IOU))
    }
}

/// Loaded ONNX YOLO11m detector: the `ort` session plus the device it runs on.
pub struct OrtYolo {
    session: Session,
    pub device: &'static str,
}

impl OrtYolo {
    /// Finalize optional ORT profiling configured by the caller's loader.
    /// No profiling is enabled by the engine or by ordinary execution.
    pub fn finish_profiling(&mut self) -> Result<Vec<std::path::PathBuf>> {
        Ok(vec![self.session.end_profiling().map_err(ort_err)?.into()])
    }

    /// Load the detector, preferring the CUDA EP and falling back to a CPU session if the
    /// provider can't initialise (mirrors `pose_jobs::Detector::load`).
    pub fn load(path: &Path, load: &mut dyn crate::session::SessionLoader) -> Result<Self> {
        let ([session], device) = crate::session::load_sessions([path], "yolo11m", load)?;
        Ok(Self { session, device })
    }

    /// Detect every person in one frame → NMS'd boxes in original px.
    pub fn detect_people(&mut self, img: &RgbImage, conf: f32) -> Result<Vec<Detection>> {
        let (input, lb) = preprocess_letterbox(img, InputLayout::Nchw);
        let tensor = Tensor::from_array(([1_i64, 3, DET as i64, DET as i64].to_vec(), input))
            .map_err(ort_err)?;
        let outputs = self.session.run(ort::inputs![tensor]).map_err(ort_err)?;
        // The Ultralytics export has a single output `(1,84,8400)` channel-major — the SAME
        // tensor the MLX forward was reverse-engineered to reproduce. Read its reported shape
        // so `decode` indexes it correctly regardless of the exact class count.
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let shape: Vec<i64> = shape.to_vec();
        let raw = decode(data, &shape, &lb, conf, img.width(), img.height())?;
        Ok(nms(raw, NMS_IOU))
    }
}

#[cfg(test)]
#[path = "person_tests.rs"]
mod tests;
