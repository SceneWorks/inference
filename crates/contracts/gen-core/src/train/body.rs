//! **Body losses** (epic 2123, sc-24832) — the backend-neutral half of the three decoded-x0 body
//! perceptual losses ported from ai-toolkit-perceptual (fork commit 6e01a6e):
//!
//! - **body proportion** (`toolkit/body_id.py` `DifferentiableBodyProportionEncoder`): ViTPose+
//!   base COCO keypoints of the decoded x0 → eight pose-invariant bone-length ratios (ten with the
//!   head ratios), compared visibility-weighted against the reference image's ratios;
//! - **body shape** (`toolkit/body_shape.py` `DifferentiableBodyShapeEncoder`): HybrIK's ResNet-34
//!   backbone + beta head → 10 SMPL shape coefficients, L1 against the reference's betas, gated by
//!   their cosine (upstream `body_shape_loss_min_cos`). Only the beta head is used, so **no SMPL
//!   model file is needed**;
//! - **surface normals** (`toolkit/normal_id.py` `DifferentiableNormalEncoder`): Sapiens-0.3B
//!   normal map of the letterboxed x0, per-pixel `(1 − cos) + L1` against the reference map,
//!   optionally restricted to the item's subject mask (upstream `perceptual_restrict_to_body`).
//!
//! What lives here, shared by the MLX and Candle ports so both backends compute the same thing:
//! the typed config ([`BodyLossesConfig`]), the model hyper-parameters with exact parameter
//! counts ([`VitPoseConfig`], [`HybrikConfig`], [`SapiensConfig`]) and the E7 footprint figures,
//! and the host-side geometry: the separable resampling matrices every differentiable crop/resize
//! is built from ([`resize_weights`], [`affine_sample_weights`], [`nearest_weights`]), ViTPose's
//! full-frame affine warp ([`VitPoseWarp`]), HybrIK's square person crop ([`hybrik_square_crop`]),
//! Sapiens' letterbox ([`Letterbox`]), and the reference-time person box ([`keypoint_box`]).
//!
//! **Person detection.** All three losses need "is there a person in this reference image".
//! Upstream gates body proportion on ViTPose's own keypoint confidence (its video path needs no
//! other detector) and never really gates the other two. Here every body loss gates on the same
//! reference-time ViTPose pass: a reference whose mean body-ratio visibility is below
//! [`MIN_MEAN_RATIO_VISIBILITY`] has **no detected person** and the loss is skipped for that image.
//! The confident keypoints' bounding box is the person box HybrIK crops to.

use std::path::PathBuf;

use super::AuxLossSchedule;

/// Minimum keypoint confidence for a ratio to be trusted (upstream `VIS_THRESHOLD`).
pub const VIS_THRESHOLD: f32 = 0.2;
/// A reference whose mean ratio visibility is below this has no detected person (upstream
/// `encode`: `ratio_vis.mean() < 0.1` ⇒ zero embedding ⇒ masked out of the loss).
pub const MIN_MEAN_RATIO_VISIBILITY: f32 = 0.1;
/// A reference ratio at least this confident whose live counterpart dropped below
/// [`VIS_THRESHOLD`] counts as a missing keypoint in the visibility penalty (upstream `0.5`).
pub const MISSING_REFERENCE_VISIBILITY: f32 = 0.5;
/// Body ratios (upstream `NUM_BODY_RATIOS`).
pub const NUM_BODY_RATIOS: usize = 8;
/// Extra head ratios with [`BodyLossesConfig::include_head`] (upstream `NUM_HEAD_RATIOS`).
pub const NUM_HEAD_RATIOS: usize = 2;
/// COCO keypoints ViTPose predicts.
pub const NUM_KEYPOINTS: usize = 17;
/// SMPL shape coefficients HybrIK predicts.
pub const BETA_DIM: usize = 10;
/// Side of the square normal map both the reference and the live map are resampled to (upstream
/// `NORMAL_SIZE`).
pub const NORMAL_SIZE: usize = 256;
/// ImageNet normalization (ViTPose, Sapiens).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet normalization (ViTPose, Sapiens).
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// HybrIK's RGB normalization (upstream `img_mean`).
pub const HYBRIK_MEAN: [f32; 3] = [0.406, 0.457, 0.480];
/// HybrIK's RGB normalization (upstream `img_std`).
pub const HYBRIK_STD: [f32; 3] = [0.225, 0.224, 0.229];

/// For each ratio (body ratios first, then the two head ratios) the COCO keypoints whose minimum
/// confidence is that ratio's visibility (upstream `vis_list`).
pub const RATIO_VISIBILITY_KEYPOINTS: [&[usize]; NUM_BODY_RATIOS + NUM_HEAD_RATIOS] = [
    &[5, 6, 7, 8],
    &[7, 8, 9, 10],
    &[11, 12, 13, 14],
    &[13, 14, 15, 16],
    &[5, 6, 11, 12],
    &[5, 6, 11, 12],
    &[5, 6, 7, 8, 9, 10],
    &[11, 12, 13, 14, 15, 16],
    &[0, 5, 6],
    &[3, 4, 5, 6],
];

/// [`TrainingConfig::body_losses`](super::TrainingConfig::body_losses) — the three body losses'
/// schedules, their knobs and the frozen checkpoints they run. Every loss is off by default.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyLossesConfig {
    /// ViTPose bone-length-ratio loss. Off by default.
    pub proportion: AuxLossSchedule,
    /// Add the two head ratios (nose→shoulders height, ear-to-ear width) to the eight body ratios
    /// (upstream `body_proportion_include_head`, default `false`).
    pub include_head: bool,
    /// HybrIK SMPL-beta loss. Off by default.
    pub shape: AuxLossSchedule,
    /// The shape loss only counts while the live betas' cosine to the reference exceeds this
    /// (upstream `body_shape_loss_min_cos`, default `0.2`), in [`Self::SHAPE_MIN_COS_RANGE`].
    pub shape_min_cos: f32,
    /// Sapiens surface-normal loss. Off by default.
    pub normal: AuxLossSchedule,
    /// Average the normal loss over the item's subject mask only (upstream
    /// `perceptual_restrict_to_body`). Needs every item's
    /// [`subject_mask_path`](super::TrainingItem::subject_mask_path).
    pub normal_restrict_to_subject: bool,
    /// The ViTPose+ base checkpoint (`usyd-community/vitpose-plus-base`, `model.safetensors`).
    /// Required whenever **any** body loss is on: it is the proportion encoder and the
    /// reference-time person detector of all three.
    pub pose_model_dir: Option<PathBuf>,
    /// The HybrIK ResNet-34 checkpoint (`model.safetensors`, HybrIK key layout). Required with the
    /// shape loss on.
    pub shape_model_dir: Option<PathBuf>,
    /// The Sapiens-0.3B normal checkpoint (`model.safetensors`, Sapiens key layout). Required with
    /// the normal loss on.
    pub normal_model_dir: Option<PathBuf>,
}

impl BodyLossesConfig {
    /// Upstream `body_shape_loss_min_cos` default.
    pub const DEFAULT_SHAPE_MIN_COS: f32 = 0.2;
    /// Inclusive bounds of [`shape_min_cos`](Self::shape_min_cos) (a cosine).
    pub const SHAPE_MIN_COS_RANGE: (f32, f32) = (-1.0, 1.0);

    /// Whether any of the three losses is on.
    pub fn any_enabled(&self) -> bool {
        self.proportion.is_enabled() || self.shape.is_enabled() || self.normal.is_enabled()
    }

    /// Ratios the proportion loss compares (8, or 10 with the head ratios).
    pub fn ratio_count(&self) -> usize {
        NUM_BODY_RATIOS + if self.include_head { NUM_HEAD_RATIOS } else { 0 }
    }

    /// Reject malformed schedules or knobs (`label` prefixes the message).
    pub fn validate(&self, label: &str) -> Result<(), String> {
        for (name, s) in [
            ("body proportion loss", &self.proportion),
            ("body shape loss", &self.shape),
            ("normal loss", &self.normal),
        ] {
            s.validate(name).map_err(|m| format!("{label}: {m}"))?;
        }
        let (lo, hi) = Self::SHAPE_MIN_COS_RANGE;
        let c = self.shape_min_cos;
        if !c.is_finite() || c < lo || c > hi {
            return Err(format!(
                "{label}: body_losses.shape_min_cos must be in [{lo}, {hi}], got {c}"
            ));
        }
        Ok(())
    }
}

impl Default for BodyLossesConfig {
    fn default() -> Self {
        Self {
            proportion: AuxLossSchedule::OFF,
            include_head: false,
            shape: AuxLossSchedule::OFF,
            shape_min_cos: Self::DEFAULT_SHAPE_MIN_COS,
            normal: AuxLossSchedule::OFF,
            normal_restrict_to_subject: false,
            pose_model_dir: None,
            shape_model_dir: None,
            normal_model_dir: None,
        }
    }
}

/// Pre-load memory figures of one body model (epic 2123 E7), backend-neutral; each backend wraps
/// them into its perceptual path's footprint type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BodyModelFootprint {
    /// Resident f32 weights.
    pub param_bytes: u64,
    /// One differentiable forward + backward.
    pub working_set_bytes: u64,
    /// One cached per-image reference.
    pub reference_bytes_per_image: u64,
}

const F32: u64 = 4;

// ---------------------------------------------------------------------------------------------
// ViTPose+ (HF transformers `VitPoseForPoseEstimation`, `usyd-community/vitpose-plus-base`).
// ---------------------------------------------------------------------------------------------

/// ViTPose(+) hyper-parameters (HF `VitPoseConfig` + its `VitPoseBackboneConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct VitPoseConfig {
    /// Embedding width (768 for base).
    pub hidden_size: usize,
    /// Transformer layers (12).
    pub num_layers: usize,
    /// Attention heads (12).
    pub num_heads: usize,
    /// FFN expansion (4).
    pub mlp_ratio: usize,
    /// ViTPose+ mixture-of-experts MLP: experts (6; `1` = plain MLP).
    pub num_experts: usize,
    /// Width of the expert slice of the MLP output (192).
    pub part_features: usize,
    /// Which expert (dataset head) runs — `0` = COCO, what upstream passes (`dataset_index=0`).
    pub expert_index: usize,
    /// Input `(height, width)` (256, 192).
    pub image_size: (usize, usize),
    /// Patch size (16); the patch conv also pads by 2 (HF `VitPoseBackbonePatchEmbeddings`).
    pub patch_size: usize,
    /// LayerNorm epsilon (HF default 1e-12).
    pub layer_norm_eps: f32,
    /// Keypoints (17 COCO).
    pub num_keypoints: usize,
    /// `true` = the "simple" decoder (ReLU → ×4 bilinear → 3×3 conv); `false` = the classic
    /// decoder (2× deconv+BN+ReLU → 1×1 conv), which `vitpose-plus-base` ships.
    pub use_simple_decoder: bool,
    /// Classic-decoder deconv width (256).
    pub deconv_channels: usize,
}

impl VitPoseConfig {
    /// `usyd-community/vitpose-plus-base`.
    pub fn plus_base() -> Self {
        Self {
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_ratio: 4,
            num_experts: 6,
            part_features: 192,
            expert_index: 0,
            image_size: (256, 192),
            patch_size: 16,
            layer_norm_eps: 1e-12,
            num_keypoints: NUM_KEYPOINTS,
            use_simple_decoder: false,
            deconv_channels: 256,
        }
    }

    /// A tiny ViTPose-shaped config (same graph, toy widths) for synthetic tests.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 16,
            num_layers: 2,
            num_heads: 2,
            mlp_ratio: 2,
            num_experts: 2,
            part_features: 4,
            expert_index: 0,
            image_size: (64, 48),
            patch_size: 8,
            layer_norm_eps: 1e-6,
            num_keypoints: NUM_KEYPOINTS,
            use_simple_decoder: false,
            // HF's classic decoder hard-codes 256 deconv channels; the tiny reference fixture is
            // an HF model, so the tiny config keeps it.
            deconv_channels: 256,
        }
    }

    /// Patch grid `(h, w)` of the padded patch conv: `(size + 4 − patch) / patch + 1`.
    pub fn grid(&self) -> (usize, usize) {
        let g = |s: usize| (s + 4 - self.patch_size) / self.patch_size + 1;
        (g(self.image_size.0), g(self.image_size.1))
    }

    /// Heatmap `(h, w)`: the patch grid ×4 (two stride-2 deconvs, or the ×4 upsample).
    pub fn heatmap_size(&self) -> (usize, usize) {
        let (h, w) = self.grid();
        (h * 4, w * 4)
    }

    /// Exact parameter count (all experts and BatchNorm running stats included, as stored).
    pub fn param_count(&self) -> u64 {
        let c = self.hidden_size as u64;
        let inter = c * self.mlp_ratio as u64;
        let (gh, gw) = self.grid();
        let n = (gh * gw) as u64;
        let p = self.patch_size as u64;
        let embed = c * 3 * p * p + c + (n + 1) * c;
        let mlp = if self.num_experts > 1 {
            let part = self.part_features as u64;
            (c * inter + inter)
                + (inter * (c - part) + (c - part))
                + self.num_experts as u64 * (inter * part + part)
        } else {
            (c * inter + inter) + (inter * c + c)
        };
        let layer = 4 * c + 4 * (c * c + c) + mlp;
        let k = self.num_keypoints as u64;
        let head = if self.use_simple_decoder {
            c * k * 9 + k
        } else {
            let d = self.deconv_channels as u64;
            c * d * 16 + 4 * d + d * d * 16 + 4 * d + d * k + k
        };
        embed + self.num_layers as u64 * layer + 2 * c + head
    }

    /// Conservative training working set of one differentiable forward + backward: per layer the
    /// retained token activations (LN in/out, Q/K/V, attention out, residual, MLP hidden pre/post
    /// activation) and the attention matrix, plus the decoder maps, plus the resampled input —
    /// doubled for the backward's transient gradients.
    pub fn training_working_set_bytes(&self) -> u64 {
        let c = self.hidden_size as u64;
        let (gh, gw) = self.grid();
        let n = (gh * gw) as u64;
        let inter = c * self.mlp_ratio as u64;
        let per_layer = n * (8 * c + 3 * inter) + self.num_heads as u64 * n * n;
        let (hh, hw) = self.heatmap_size();
        let d = self.deconv_channels.max(self.hidden_size) as u64;
        let decoder = 6 * d * (hh * hw) as u64;
        let input = 3 * (self.image_size.0 * self.image_size.1) as u64 * 2;
        2 * F32 * (self.num_layers as u64 * per_layer + decoder + input)
    }

    /// E7 figures; the reference holds `ratio_count` ratios + visibilities and a person box.
    pub fn footprint(&self, ratio_count: usize) -> BodyModelFootprint {
        BodyModelFootprint {
            param_bytes: self.param_count() * F32,
            working_set_bytes: self.training_working_set_bytes(),
            reference_bytes_per_image: (2 * ratio_count as u64 + 4) * F32,
        }
    }
}

/// ViTPose's full-frame affine input warp (HF `VitPoseImageProcessor` with the whole image as the
/// box: `box_to_center_and_scale(.., padding_factor=1.25)` + `get_warp_matrix(rotation 0)`), as the
/// per-axis map `source = output_index · scale + offset` in input pixel-index coordinates — exactly
/// what upstream's `affine_grid` + `grid_sample(align_corners=True)` sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VitPoseWarp {
    pub scale_x: f32,
    pub offset_x: f32,
    pub scale_y: f32,
    pub offset_y: f32,
    /// The model input `(height, width)`.
    pub out: (usize, usize),
}

impl VitPoseWarp {
    /// The warp of an `in_h × in_w` frame onto a model input of `out = (h, w)`.
    pub fn full_frame(in_h: usize, in_w: usize, out: (usize, usize)) -> Self {
        let (out_h, out_w) = out;
        let (w, h) = (in_w as f64, in_h as f64);
        let aspect = out_w as f64 / out_h as f64;
        let (mut tw, mut th) = (w, h);
        if tw > aspect * th {
            th = tw / aspect;
        } else if tw < aspect * th {
            tw = th * aspect;
        }
        // `scale * 200` with `scale = size / 200 * 1.25` — the padded target box in pixels.
        let (tw, th) = (tw * 1.25, th * 1.25);
        let (cx, cy) = (w * 0.5, h * 0.5);
        Self {
            scale_x: (tw / (out_w as f64 - 1.0)) as f32,
            offset_x: (cx - 0.5 * tw) as f32,
            scale_y: (th / (out_h as f64 - 1.0)) as f32,
            offset_y: (cy - 0.5 * th) as f32,
            out,
        }
    }

    /// A keypoint in DSNT-normalized heatmap coordinates (`[-1, 1]` over pixel centres) → input
    /// pixel coordinates `(x, y)`.
    pub fn keypoint_to_input(&self, x_norm: f32, y_norm: f32) -> (f32, f32) {
        let (out_h, out_w) = self.out;
        let ox = (x_norm + 1.0) * 0.5 * out_w as f32 - 0.5;
        let oy = (y_norm + 1.0) * 0.5 * out_h as f32 - 0.5;
        (
            ox * self.scale_x + self.offset_x,
            oy * self.scale_y + self.offset_y,
        )
    }
}

/// The person box `[x1, y1, x2, y2]` (input pixels, clamped to the frame) spanned by the
/// keypoints at least [`VIS_THRESHOLD`] confident; `None` when fewer than two are.
pub fn keypoint_box(
    keypoints: &[(f32, f32)],
    confidence: &[f32],
    in_h: usize,
    in_w: usize,
) -> Option<[f32; 4]> {
    let pts: Vec<(f32, f32)> = keypoints
        .iter()
        .zip(confidence)
        .filter(|(_, &c)| c >= VIS_THRESHOLD)
        .map(|(&p, _)| p)
        .collect();
    if pts.len() < 2 {
        return None;
    }
    let clamp = |v: f32, hi: usize| v.clamp(0.0, hi as f32);
    let x1 = clamp(pts.iter().map(|p| p.0).fold(f32::INFINITY, f32::min), in_w);
    let x2 = clamp(pts.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max), in_w);
    let y1 = clamp(pts.iter().map(|p| p.1).fold(f32::INFINITY, f32::min), in_h);
    let y2 = clamp(pts.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max), in_h);
    (x2 > x1 && y2 > y1).then_some([x1, y1, x2, y2])
}

// ---------------------------------------------------------------------------------------------
// Separable resampling matrices (row-major `[out, in]`): every differentiable crop/resize of the
// body losses is `A_y · X · A_xᵀ`, a plain matmul pair on either backend.
// ---------------------------------------------------------------------------------------------

/// torch `F.interpolate(mode="bilinear", antialias=False)` along one axis of length `in_len`
/// resized to `out_len`.
pub fn resize_weights(in_len: usize, out_len: usize, align_corners: bool) -> Vec<f32> {
    let mut m = vec![0.0f32; out_len * in_len];
    let last = in_len - 1;
    for o in 0..out_len {
        let src = if align_corners {
            if out_len == 1 {
                0.0
            } else {
                o as f64 * last as f64 / (out_len - 1) as f64
            }
        } else {
            ((o as f64 + 0.5) * in_len as f64 / out_len as f64 - 0.5).max(0.0)
        };
        let lo = (src.floor() as usize).min(last);
        let hi = (lo + 1).min(last);
        let frac = (src - lo as f64).clamp(0.0, 1.0) as f32;
        m[o * in_len + lo] += 1.0 - frac;
        m[o * in_len + hi] += frac;
    }
    m
}

/// torch `F.grid_sample(mode="bilinear", padding_mode="zeros", align_corners=True)` along one axis
/// for the affine map `source = o · scale + offset` (input pixel-index coordinates): each
/// bracketing tap outside `[0, in_len − 1]` contributes zero.
pub fn affine_sample_weights(in_len: usize, out_len: usize, scale: f32, offset: f32) -> Vec<f32> {
    let mut m = vec![0.0f32; out_len * in_len];
    for o in 0..out_len {
        let src = o as f64 * scale as f64 + offset as f64;
        let x0 = src.floor();
        let w1 = (src - x0) as f32;
        for (tap, w) in [(x0, 1.0 - w1), (x0 + 1.0, w1)] {
            if tap >= 0.0 && tap <= (in_len - 1) as f64 {
                m[o * in_len + tap as usize] += w;
            }
        }
    }
    m
}

/// torch `F.interpolate(mode="nearest")` along one axis (`src = floor(o · in / out)`).
pub fn nearest_weights(in_len: usize, out_len: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; out_len * in_len];
    for o in 0..out_len {
        let src = ((o as f64 * in_len as f64 / out_len as f64).floor() as usize).min(in_len - 1);
        m[o * in_len + src] = 1.0;
    }
    m
}

// ---------------------------------------------------------------------------------------------
// HybrIK (ResNet-34 backbone + linear beta head).
// ---------------------------------------------------------------------------------------------

/// HybrIK shape-encoder hyper-parameters: a torchvision-layout ResNet (BasicBlocks) + the
/// `fc1 → fc2 → decshape` beta head (no nonlinearity, dropout is identity at eval).
#[derive(Clone, Debug, PartialEq)]
pub struct HybrikConfig {
    /// BasicBlocks per stage (`[3, 4, 6, 3]` = ResNet-34).
    pub blocks: [usize; 4],
    /// Stage widths (`[64, 128, 256, 512]`); the stem conv outputs `widths[0]`.
    pub widths: [usize; 4],
    /// Beta-head hidden width (1024).
    pub fc_hidden: usize,
    /// Square input side (256).
    pub input_size: usize,
    /// BatchNorm epsilon (1e-5).
    pub bn_eps: f32,
}

impl HybrikConfig {
    /// HybrIK's ResNet-34 (`hybrik_resnet34`).
    pub fn resnet34() -> Self {
        Self {
            blocks: [3, 4, 6, 3],
            widths: [64, 128, 256, 512],
            fc_hidden: 1024,
            input_size: 256,
            bn_eps: 1e-5,
        }
    }

    /// A tiny ResNet-shaped config (same graph) for synthetic tests.
    pub fn tiny() -> Self {
        Self {
            blocks: [1, 1, 1, 1],
            widths: [4, 4, 8, 8],
            fc_hidden: 8,
            input_size: 32,
            bn_eps: 1e-5,
        }
    }

    /// Exact parameter count (BatchNorm affine + running stats; `init_shape` included).
    pub fn param_count(&self) -> u64 {
        let bn = |c: u64| 4 * c;
        let conv = |i: u64, o: u64, k: u64| i * o * k * k;
        let w0 = self.widths[0] as u64;
        let mut n = conv(3, w0, 7) + bn(w0);
        let mut cin = w0;
        for (s, (&blocks, &w)) in self.blocks.iter().zip(&self.widths).enumerate() {
            let w = w as u64;
            for b in 0..blocks {
                let i = if b == 0 { cin } else { w };
                n += conv(i, w, 3) + bn(w) + conv(w, w, 3) + bn(w);
                if b == 0 && (s > 0 || cin != w) {
                    n += conv(cin, w, 1) + bn(w);
                }
            }
            cin = w;
        }
        let h = self.fc_hidden as u64;
        n + (cin * h + h) + (h * h + h) + (h * BETA_DIM as u64 + BETA_DIM as u64) + BETA_DIM as u64
    }

    /// Conservative training working set of one differentiable forward + backward at the square
    /// input: every BasicBlock retains ~5 maps of its output size (conv/BN/ReLU in, residual), the
    /// stem 4 at half and quarter resolution — doubled for the backward.
    pub fn training_working_set_bytes(&self) -> u64 {
        let s = self.input_size as u64;
        let mut px = (s / 2) * (s / 2);
        let mut elems = 4 * self.widths[0] as u64 * px + 3 * s * s;
        px /= 4; // maxpool
        for (stage, (&blocks, &w)) in self.blocks.iter().zip(&self.widths).enumerate() {
            if stage > 0 {
                px /= 4;
            }
            elems += 5 * blocks as u64 * w as u64 * px.max(1);
        }
        2 * F32 * elems
    }

    /// E7 figures; the reference holds 10 betas and a crop box.
    pub fn footprint(&self) -> BodyModelFootprint {
        BodyModelFootprint {
            param_bytes: self.param_count() * F32,
            working_set_bytes: self.training_working_set_bytes(),
            reference_bytes_per_image: (BETA_DIM as u64 + 4) * F32,
        }
    }
}

/// HybrIK's square person crop of an `in_h × in_w` frame (upstream `forward` with a person box):
/// side `max(w, h) · 1.25` around the box centre, rounded and clamped to the frame. Returns
/// half-open `(y0, y1, x0, x1)`; a degenerate crop falls back to the whole frame (upstream's
/// `else: crop = pixels[i:i+1]`).
pub fn hybrik_square_crop(bbox: [f32; 4], in_h: usize, in_w: usize) -> (usize, usize, usize, usize) {
    let [x1, y1, x2, y2] = bbox.map(|v| v as f64);
    let (cx, cy) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0);
    let half = (x2 - x1).max(y2 - y1) * 1.25 / 2.0;
    let lo = |v: f64| (v.round().max(0.0)) as usize;
    let cx1 = lo(cx - half);
    let cy1 = lo(cy - half);
    let cx2 = ((cx + half).round().max(0.0) as usize).min(in_w);
    let cy2 = ((cy + half).round().max(0.0) as usize).min(in_h);
    if cx2 > cx1 && cy2 > cy1 {
        (cy1, cy2, cx1, cx2)
    } else {
        (0, in_h, 0, in_w)
    }
}

// ---------------------------------------------------------------------------------------------
// Sapiens (ViT + deconv normal head).
// ---------------------------------------------------------------------------------------------

/// Sapiens normal-estimator hyper-parameters (`sapiens_0.3b_normal_render_people`).
#[derive(Clone, Debug, PartialEq)]
pub struct SapiensConfig {
    /// Embedding width (1024).
    pub embed_dim: usize,
    /// Transformer layers (24).
    pub num_layers: usize,
    /// Attention heads (16).
    pub num_heads: usize,
    /// FFN width (4096).
    pub ffn_dim: usize,
    /// Patch size (16); the patch conv pads by 2.
    pub patch_size: usize,
    /// The stored position-embedding grid `(h, w)` (64 × 48 — the native 1024 × 768 input).
    pub pos_grid: (usize, usize),
    /// Decoder width (768).
    pub decoder_channels: usize,
    /// Deconv stages (3, each ×2).
    pub decoder_stages: usize,
    /// The training-time letterbox target `(h, w)` for a portrait frame (512 × 384, half the
    /// native size — upstream `_best_orientation_train`); a landscape frame uses `(w, h)`.
    pub train_size: (usize, usize),
    /// LayerNorm epsilon (1e-6, the mmpretrain ViT setting).
    pub layer_norm_eps: f32,
    /// InstanceNorm epsilon (1e-5, torch default).
    pub instance_norm_eps: f32,
    /// Side of the square map the normals are resampled to before comparing (upstream
    /// `NORMAL_SIZE` = [`NORMAL_SIZE`]).
    pub normal_size: usize,
}

impl SapiensConfig {
    /// `facebook/sapiens-normal-0.3b`.
    pub fn normal_0_3b() -> Self {
        Self {
            embed_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            ffn_dim: 4096,
            patch_size: 16,
            pos_grid: (64, 48),
            decoder_channels: 768,
            decoder_stages: 3,
            train_size: (512, 384),
            layer_norm_eps: 1e-6,
            instance_norm_eps: 1e-5,
            normal_size: NORMAL_SIZE,
        }
    }

    /// A tiny Sapiens-shaped config (same graph) for synthetic tests.
    pub fn tiny() -> Self {
        Self {
            embed_dim: 16,
            num_layers: 2,
            num_heads: 2,
            ffn_dim: 32,
            patch_size: 8,
            pos_grid: (8, 6),
            decoder_channels: 8,
            decoder_stages: 3,
            train_size: (32, 24),
            layer_norm_eps: 1e-6,
            instance_norm_eps: 1e-5,
            normal_size: 16,
        }
    }

    /// Patch grid of a `h × w` input (padded patch conv).
    pub fn grid(&self, h: usize, w: usize) -> (usize, usize) {
        let g = |s: usize| (s + 4 - self.patch_size) / self.patch_size + 1;
        (g(h), g(w))
    }

    /// Exact parameter count.
    pub fn param_count(&self) -> u64 {
        let c = self.embed_dim as u64;
        let f = self.ffn_dim as u64;
        let p = self.patch_size as u64;
        let (ph, pw) = self.pos_grid;
        let embed = 3 * c * p * p + c + (ph * pw) as u64 * c;
        let layer = 4 * c + (c * 3 * c + 3 * c) + (c * c + c) + (c * f + f) + (f * c + c);
        let d = self.decoder_channels as u64;
        let mut dec = 0;
        for i in 0..self.decoder_stages {
            let cin = if i == 0 { c } else { d };
            dec += cin * d * 16 + (d * d + d);
        }
        dec += d * 3 + 3;
        embed + self.num_layers as u64 * layer + 2 * c + dec
    }

    /// Conservative training working set at the training letterbox: per layer the retained
    /// token activations and the attention matrix, the decoder's retained maps at each ×2 stage
    /// (deconv out, norm, SiLU, conv, norm, SiLU), the output and the 256² resample — doubled for
    /// the backward.
    pub fn training_working_set_bytes(&self) -> u64 {
        let (gh, gw) = self.grid(self.train_size.0, self.train_size.1);
        let n = (gh * gw) as u64;
        let c = self.embed_dim as u64;
        let per_layer = n * (8 * c + 2 * self.ffn_dim as u64) + self.num_heads as u64 * n * n;
        let d = self.decoder_channels as u64;
        let mut dec = 0;
        let mut px = n;
        for _ in 0..self.decoder_stages {
            px *= 4;
            dec += 6 * d * px;
        }
        let out = 3 * px + 2 * 3 * (self.normal_size * self.normal_size) as u64;
        let input = 3 * (self.train_size.0 * self.train_size.1) as u64 * 2;
        2 * F32 * (self.num_layers as u64 * per_layer + dec + out + input)
    }

    /// E7 figures; the reference holds a 3 × 256² normal map and (restricted) a 256² mask.
    pub fn footprint(&self, restrict_to_subject: bool) -> BodyModelFootprint {
        let px = (self.normal_size * self.normal_size) as u64;
        BodyModelFootprint {
            param_bytes: self.param_count() * F32,
            working_set_bytes: self.training_working_set_bytes(),
            reference_bytes_per_image: (3 + u64::from(restrict_to_subject)) * px * F32,
        }
    }

    /// The training letterbox of an `in_h × in_w` frame (upstream `_best_orientation_train` +
    /// `_letterbox_tensor`).
    pub fn letterbox(&self, in_h: usize, in_w: usize) -> Letterbox {
        let (ph, pw) = self.train_size;
        let (target_h, target_w) = if in_h >= in_w { (ph, pw) } else { (pw, ph) };
        let scale = (target_w as f64 / in_w as f64).min(target_h as f64 / in_h as f64);
        let new_w = ((in_w as f64 * scale) as usize).clamp(1, target_w);
        let new_h = ((in_h as f64 * scale) as usize).clamp(1, target_h);
        Letterbox {
            target_h,
            target_w,
            new_h,
            new_w,
            pad_top: (target_h - new_h) / 2,
            pad_left: (target_w - new_w) / 2,
        }
    }
}

/// A letterbox: resize to `new_h × new_w` (bilinear, `align_corners=False`), then zero-pad to
/// `target_h × target_w` with the image at (`pad_top`, `pad_left`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Letterbox {
    pub target_h: usize,
    pub target_w: usize,
    pub new_h: usize,
    pub new_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
}

impl Letterbox {
    /// The letterbox along one axis as a single `[target, in]` matrix: the bilinear resize rows,
    /// shifted by the pad (pad rows are zero).
    pub fn axis_weights(target: usize, new: usize, pad: usize, in_len: usize) -> Vec<f32> {
        let resize = resize_weights(in_len, new, false);
        let mut m = vec![0.0f32; target * in_len];
        m[pad * in_len..(pad + new) * in_len].copy_from_slice(&resize);
        m
    }
}

/// The total E7 footprint of the enabled body losses' models on `images` cached references:
/// ViTPose whenever any loss is on (it is every loss's person detector), HybrIK with the shape
/// loss, Sapiens with the normal loss. Empty when all are off.
pub fn body_loss_footprints(cfg: &BodyLossesConfig) -> Vec<BodyModelFootprint> {
    if !cfg.any_enabled() {
        return Vec::new();
    }
    let mut out = vec![VitPoseConfig::plus_base().footprint(cfg.ratio_count())];
    if cfg.shape.is_enabled() {
        out.push(HybrikConfig::resnet34().footprint());
    }
    if cfg.normal.is_enabled() {
        out.push(SapiensConfig::normal_0_3b().footprint(cfg.normal_restrict_to_subject));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The analytic counts match the published checkpoints. ViTPose+ base: `model.safetensors`
    /// is 501 627 628 bytes (f32 + header + the two int64 `num_batches_tracked`). Sapiens-0.3B:
    /// the `.pth` is 1 358 042 350 bytes (f32 state dict + pickle overhead). Mutation: drop the
    /// experts term from `VitPoseConfig::param_count` ⇒ red.
    #[test]
    fn param_counts_match_the_published_checkpoints() {
        let vit = VitPoseConfig::plus_base().param_count() * 4;
        assert!(
            vit <= 501_627_628 && 501_627_628 - vit < 100_000,
            "vitpose {vit}"
        );
        let sap = SapiensConfig::normal_0_3b().param_count() * 4;
        assert!(
            sap <= 1_358_042_350 && 1_358_042_350 - sap < 2_000_000,
            "sapiens {sap}"
        );
        // torchvision resnet34 has 21 797 672 params of which 513 000 are its 1000-way fc; the
        // BN running stats add 2 × 8 512 BN channels (= 17 024); HybrIK's head adds
        // fc1/fc2/decshape + init_shape.
        let resnet_body = 21_797_672 - 513_000 + 17_024;
        let head = (512 * 1024 + 1024) + (1024 * 1024 + 1024) + (1024 * 10 + 10) + 10;
        assert_eq!(HybrikConfig::resnet34().param_count(), resnet_body + head);
    }

    #[test]
    fn vitpose_grid_and_heatmap_match_the_hf_shapes() {
        let c = VitPoseConfig::plus_base();
        assert_eq!(c.grid(), (16, 12));
        assert_eq!(c.heatmap_size(), (64, 48));
        let s = SapiensConfig::normal_0_3b();
        assert_eq!(s.grid(1024, 768), s.pos_grid);
        assert_eq!(s.grid(512, 384), (32, 24));
    }

    /// The full-frame warp maps the model input onto the padded, aspect-fixed box centred on the
    /// frame: the input centre samples the frame centre, and the box is 1.25 × the frame along the
    /// limiting axis. Mutation: drop the `* 1.25` ⇒ red.
    #[test]
    fn vitpose_warp_centres_and_pads_the_frame() {
        let w = VitPoseWarp::full_frame(512, 384, (256, 192)); // same 4:3 aspect as the input
        let centre_x = 95.5 * w.scale_x + w.offset_x;
        let centre_y = 127.5 * w.scale_y + w.offset_y;
        assert!((centre_x - 192.0).abs() < 1e-3 && (centre_y - 256.0).abs() < 1e-3);
        assert!((w.scale_x * 191.0 - 384.0 * 1.25).abs() < 1e-2);
        assert!((w.scale_y * 255.0 - 512.0 * 1.25).abs() < 1e-2);
        // A wide frame is limited by its width.
        let wide = VitPoseWarp::full_frame(100, 400, (256, 192));
        assert!((wide.scale_x * 191.0 - 500.0).abs() < 1e-2);
        assert!((wide.scale_y * 255.0 - 400.0 / 0.75 * 1.25).abs() < 1e-1);
        // The keypoint inverse lands the heatmap centre on the frame centre.
        let (x, y) = w.keypoint_to_input(0.0, 0.0);
        assert!((x - 192.0).abs() < 1.0 && (y - 256.0).abs() < 1.0, "{x} {y}");
    }

    #[test]
    fn resampling_matrices_follow_torch() {
        // Downsample 4 → 2 (half-pixel centres): rows average pairs.
        assert_eq!(resize_weights(4, 2, false), vec![0.5, 0.5, 0., 0., 0., 0., 0.5, 0.5]);
        // Identity.
        assert_eq!(resize_weights(3, 3, false), vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]);
        // grid_sample zeros padding: a tap left of 0 contributes nothing.
        let m = affine_sample_weights(3, 3, 1.0, -0.5);
        assert_eq!(&m[0..3], &[0.5, 0.0, 0.0]);
        assert_eq!(&m[3..6], &[0.5, 0.5, 0.0]);
        assert_eq!(nearest_weights(4, 2), vec![1., 0., 0., 0., 0., 0., 1., 0.]);
    }

    #[test]
    fn hybrik_crop_and_letterbox_geometry() {
        // A 20×40 person box centred at (50, 60) → side 50, clamped inside a 100×100 frame.
        assert_eq!(
            hybrik_square_crop([40.0, 40.0, 60.0, 80.0], 100, 100),
            (35, 85, 25, 75)
        );
        assert_eq!(hybrik_square_crop([5.0, 5.0, 5.0, 5.0], 10, 10), (0, 10, 0, 10));
        let s = SapiensConfig::normal_0_3b();
        let lb = s.letterbox(1024, 1024);
        assert_eq!((lb.target_h, lb.target_w, lb.new_h, lb.new_w), (512, 384, 384, 384));
        assert_eq!((lb.pad_top, lb.pad_left), (64, 0));
        let land = s.letterbox(768, 1024);
        assert_eq!((land.target_h, land.target_w, land.new_h, land.new_w), (384, 512, 384, 512));
        let m = Letterbox::axis_weights(4, 2, 1, 2);
        assert_eq!(m, vec![0., 0., 1., 0., 0., 1., 0., 0.]);
    }

    #[test]
    fn keypoint_box_needs_two_confident_points() {
        let kps = [(10.0, 20.0), (30.0, 5.0), (200.0, 200.0)];
        assert_eq!(
            keypoint_box(&kps, &[0.9, 0.5, 0.1], 100, 100),
            Some([10.0, 5.0, 30.0, 20.0])
        );
        assert_eq!(keypoint_box(&kps, &[0.9, 0.1, 0.1], 100, 100), None);
    }

    /// E7: ViTPose is counted whenever any body loss is on; the shape and normal models only with
    /// their loss. Mutation: count ViTPose only for the proportion loss ⇒ red.
    #[test]
    fn footprints_follow_the_enabled_losses() {
        let mut cfg = BodyLossesConfig::default();
        assert!(body_loss_footprints(&cfg).is_empty());
        cfg.normal.weight = 0.1;
        let f = body_loss_footprints(&cfg);
        assert_eq!(f.len(), 2);
        assert_eq!(f[0].param_bytes, VitPoseConfig::plus_base().param_count() * 4);
        assert_eq!(f[1].param_bytes, SapiensConfig::normal_0_3b().param_count() * 4);
        assert!(f[1].working_set_bytes > 1 << 30, "Sapiens-0.3B backward is GB-scale");
        cfg.shape.weight = 0.1;
        assert_eq!(body_loss_footprints(&cfg).len(), 3);
    }

    #[test]
    fn config_validation_names_the_bad_knob() {
        let mut cfg = BodyLossesConfig::default();
        assert!(cfg.validate("t").is_ok());
        cfg.shape_min_cos = 1.5;
        assert!(cfg.validate("t").unwrap_err().contains("shape_min_cos"));
        cfg.shape_min_cos = 0.2;
        cfg.normal.t_min = 0.9;
        cfg.normal.t_max = 0.1;
        assert!(cfg.validate("t").unwrap_err().contains("normal loss"));
    }
}
