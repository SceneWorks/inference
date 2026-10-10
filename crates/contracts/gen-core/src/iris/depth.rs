//! Iris-3B **monocular depth** (`iris3b/downstream/depth.py` + `scripts/depth.py` @
//! [`super::UPSTREAM_CODE_REVISION`]) — the backend-neutral half: request/result types, the source
//! preprocessing (max-side control, patch-multiple Lanczos resize, `[-1, 1]` mapping), the bilinear
//! resize of the prediction back to the source size, the metadata a consumer records, and the pure
//! presentation adapters. The tensor forward lives in `mlx-gen-iris` / `candle-gen-iris`.
//!
//! ## Value convention (E5)
//!
//! [`DepthMap`] is the model's raw output at the source resolution: **affine-invariant relative log
//! depth**, float32, row-major `H × W`. Training regressed `log(depth)` with its 2nd/98th
//! percentiles mapped to −1 (near) and +1 (far), so values sit around `[-1, 1]` but are **not
//! clamped, not normalized and not metric** — compare to ground truth by fitting a scale and shift in
//! log space. [`near_bright_unit`], [`near_bright_control_image`] and [`colorize_inferno`] derive
//! presentation forms from a borrowed map and never mutate it.
//!
//! ## Preprocessing (caller vs provider)
//!
//! Upstream's `DepthPredictor.__call__` first applies `ImageOps.exif_transpose(image).convert("RGB")`
//! — that orientation step is the **caller's** job (an [`Image`] carries no EXIF). Everything after
//! it is here: the long side is capped at `max_side` ([`DepthResolution::Capped`], default 1024) or
//! kept ([`DepthResolution::Native`], upstream's `max_side=0`), each side is rounded to the patch
//! grid with Python's round-half-even, the image is Lanczos-resized when that changes its size, the
//! prediction is resized back bilinearly (`align_corners=False`).

use super::{
    IrisTask, DEPTH_MODEL_ID, UPSTREAM_CODE_REVISION, UPSTREAM_WEIGHTS_REPO,
    UPSTREAM_WEIGHTS_REVISION,
};
use crate::imageops::resize_lanczos_u8;
use crate::media::Image;
use crate::runtime::{CancelFlag, Progress};
use crate::{Error, Result};

/// Upstream's default `max_side` (`DepthPredictor.__call__`, `scripts/depth.py --max-side`).
pub const DEFAULT_DEPTH_MAX_SIDE: u32 = 1024;
/// Input channels the depth model concatenates after RGB (a zero channel, `IrisDepth.forward`).
pub const DEPTH_EXTRA_INPUT_CHANNELS: usize = 1;
/// [`DepthMetadata::value_convention`] of every Iris depth map.
pub const VALUE_CONVENTION: &str = "relative_log_depth";
/// Human-readable statement of [`VALUE_CONVENTION`].
pub const VALUE_CONVENTION_DESCRIPTION: &str = "affine-invariant relative log depth: log(depth) \
    with training-time 2nd/98th percentiles mapped to -1 (near) and +1 (far); float32 HxW at the \
    source resolution, not clamped, not normalized, not metric";

/// How large the model's input is (upstream's `max_side`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthResolution {
    /// Downscale so the long side is at most this many pixels (never upscale). Must be `> 0`.
    Capped(u32),
    /// Run at the source resolution (rounded to the patch grid) — upstream's `max_side=0`.
    Native,
}

impl Default for DepthResolution {
    fn default() -> Self {
        DepthResolution::Capped(DEFAULT_DEPTH_MAX_SIDE)
    }
}

impl DepthResolution {
    /// Upstream's `max_side` integer (`0` = native).
    pub fn max_side(self) -> u32 {
        match self {
            DepthResolution::Capped(side) => side,
            DepthResolution::Native => 0,
        }
    }
}

/// One depth request. `image` is the (already EXIF-oriented) RGB8 source.
#[derive(Clone, Debug, Default)]
pub struct DepthRequest {
    pub image: Image,
    pub resolution: DepthResolution,
    pub cancel: CancelFlag,
}

/// Check a request before any tensor work.
pub fn validate_depth_request(req: &DepthRequest) -> Result<()> {
    let img = &req.image;
    if img.width == 0 || img.height == 0 {
        return Err(Error::Msg(format!(
            "{DEPTH_MODEL_ID}: the source image is {}x{} — both sides must be > 0",
            img.width, img.height
        )));
    }
    let need = img.width as usize * img.height as usize * Image::CHANNELS;
    if img.pixels.len() != need {
        return Err(Error::Msg(format!(
            "{DEPTH_MODEL_ID}: the source pixel buffer holds {} bytes; a {}x{} RGB8 image needs \
             {need}",
            img.pixels.len(),
            img.width,
            img.height
        )));
    }
    if req.resolution == DepthResolution::Capped(0) {
        return Err(Error::Msg(format!(
            "{DEPTH_MODEL_ID}: a max-side cap of 0 is not a size — use DepthResolution::Native \
             for the source resolution"
        )));
    }
    Ok(())
}

/// The geometry of one prediction: source size and the patch-grid size the model sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthInputPlan {
    pub source_width: u32,
    pub source_height: u32,
    pub model_width: u32,
    pub model_height: u32,
}

impl DepthInputPlan {
    /// Whether the input is resized for the model (and the prediction resized back).
    pub fn resized(&self) -> bool {
        (self.model_width, self.model_height) != (self.source_width, self.source_height)
    }
}

/// `DepthPredictor.__call__`'s size law: `scale = min(1, max_side / max(w, h))` (1 when native),
/// each side `max(patch, round(side · scale / patch) · patch)` with Python's round-half-even.
pub fn plan_depth_input(
    width: u32,
    height: u32,
    resolution: DepthResolution,
    patch: usize,
) -> Result<DepthInputPlan> {
    if width == 0 || height == 0 || patch == 0 {
        return Err(Error::Msg(format!(
            "{DEPTH_MODEL_ID}: cannot plan a {width}x{height} input on a {patch}-pixel patch grid"
        )));
    }
    let scale = match resolution {
        DepthResolution::Native => 1.0f64,
        DepthResolution::Capped(0) => {
            return Err(Error::Msg(format!(
                "{DEPTH_MODEL_ID}: a max-side cap of 0 is not a size"
            )))
        }
        DepthResolution::Capped(side) => (side as f64 / width.max(height) as f64).min(1.0),
    };
    let p = patch as f64;
    let side = |s: u32| -> u32 { ((s as f64 * scale / p).round_ties_even() * p).max(p) as u32 };
    Ok(DepthInputPlan {
        source_width: width,
        source_height: height,
        model_width: side(width),
        model_height: side(height),
    })
}

/// The model input `[3, H', W']` (CHW, row-major) in `[-1, 1]`: the source Lanczos-resized to the
/// plan's model size when it differs (PIL's 8-bit `Image.resize(LANCZOS)`, bit-exact), then
/// `x / 255 · 2 − 1` in f32 like upstream's `rgb / 255 * 2 - 1`.
pub fn prepare_depth_input(image: &Image, plan: &DepthInputPlan) -> Result<Vec<f32>> {
    let (w, h) = (plan.model_width as usize, plan.model_height as usize);
    if (image.width, image.height) != (plan.source_width, plan.source_height) {
        return Err(Error::Msg(format!(
            "{DEPTH_MODEL_ID}: the image is {}x{}, the plan was made for {}x{}",
            image.width, image.height, plan.source_width, plan.source_height
        )));
    }
    let hwc: Vec<f32> = if plan.resized() {
        resize_lanczos_u8(
            &image.pixels,
            plan.source_height as usize,
            plan.source_width as usize,
            h,
            w,
        )?
    } else {
        image.pixels.iter().map(|&v| v as f32).collect()
    };
    let mut chw = vec![0f32; 3 * h * w];
    for (i, px) in hwc.chunks_exact(3).enumerate() {
        for c in 0..3 {
            chw[c * h * w + i] = px[c] / 255.0 * 2.0 - 1.0;
        }
    }
    Ok(chw)
}

/// torch `F.interpolate(mode="bilinear", align_corners=False)` (no antialias) of one `in_h × in_w`
/// plane to `out_h × out_w`, f32 — upstream's resize of the prediction back to the source size.
pub fn interpolate_bilinear(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<f32>> {
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 || src.len() != in_h * in_w {
        return Err(Error::Msg(format!(
            "interpolate_bilinear: {} samples for {in_h}x{in_w} -> {out_h}x{out_w}",
            src.len()
        )));
    }
    // `area_pixel_compute_source_index`: src = scale·(dst + 0.5) − 0.5, clamped at 0; the upper
    // neighbour is the clamped `+1`.
    let axis = |in_len: usize, out_len: usize| -> Vec<(usize, usize, f32, f32)> {
        let scale = in_len as f32 / out_len as f32;
        (0..out_len)
            .map(|d| {
                let r = (scale * (d as f32 + 0.5) - 0.5).max(0.0);
                let i0 = r as usize;
                let step = usize::from(i0 < in_len - 1);
                let l1 = r - i0 as f32;
                (i0, i0 + step, 1.0 - l1, l1)
            })
            .collect()
    };
    let rows = axis(in_h, out_h);
    let cols = axis(in_w, out_w);
    let mut out = Vec::with_capacity(out_h * out_w);
    for &(h0, h1, wh0, wh1) in &rows {
        let (r0, r1) = (&src[h0 * in_w..][..in_w], &src[h1 * in_w..][..in_w]);
        for &(w0, w1, ww0, ww1) in &cols {
            out.push(wh0 * (ww0 * r0[w0] + ww1 * r0[w1]) + wh1 * (ww0 * r1[w0] + ww1 * r1[w1]));
        }
    }
    Ok(out)
}

/// A float32 `H × W` depth map, row-major. Iris maps follow [`VALUE_CONVENTION`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}

/// Resize the model-resolution prediction (`plan.model_height × plan.model_width`) back to the
/// source size (identity when the input was not resized).
pub fn depth_to_source(prediction: Vec<f32>, plan: &DepthInputPlan) -> Result<DepthMap> {
    let values = if plan.resized() {
        interpolate_bilinear(
            &prediction,
            plan.model_height as usize,
            plan.model_width as usize,
            plan.source_height as usize,
            plan.source_width as usize,
        )?
    } else if prediction.len() == (plan.source_width * plan.source_height) as usize {
        prediction
    } else {
        return Err(Error::Msg(format!(
            "{DEPTH_MODEL_ID}: the prediction has {} samples for a {}x{} plan",
            prediction.len(),
            plan.model_width,
            plan.model_height
        )));
    };
    Ok(DepthMap {
        width: plan.source_width,
        height: plan.source_height,
        values,
    })
}

/// What produced a [`DepthMap`] — everything a consumer records next to it (E5).
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMetadata {
    /// [`DEPTH_MODEL_ID`].
    pub model_id: &'static str,
    /// `"mlx"` | `"candle"`.
    pub backend: &'static str,
    /// Always [`IrisTask::Depth`].
    pub task: IrisTask,
    /// [`UPSTREAM_CODE_REVISION`] — the source the port mirrors.
    pub code_revision: &'static str,
    /// [`UPSTREAM_WEIGHTS_REPO`] at [`UPSTREAM_WEIGHTS_REVISION`] (`depth/`) — the export the port
    /// is verified against. The provisioning consumer owns the guarantee that the staged directory
    /// is that revision; [`Self::config_sha256`] fingerprints what was actually loaded.
    pub weights_repo: &'static str,
    pub weights_revision: &'static str,
    /// sha256 of the loaded export's `config.yaml`.
    pub config_sha256: String,
    /// `"bfloat16"` (upstream's CUDA autocast; the default) or `"float32"` (upstream's CPU path).
    pub compute_dtype: &'static str,
    pub resolution: DepthResolution,
    pub source_width: u32,
    pub source_height: u32,
    pub model_width: u32,
    pub model_height: u32,
    /// `"lanczos"` when the source was resized for the model, else `"none"`.
    pub input_resample: &'static str,
    /// `"bilinear"` (`align_corners=False`) when the prediction was resized back, else `"none"`.
    pub output_resample: &'static str,
    /// [`VALUE_CONVENTION`].
    pub value_convention: &'static str,
}

impl DepthMetadata {
    /// The per-prediction fields from a plan; `backend`, `config_sha256` and `compute_dtype` come
    /// from the loaded provider.
    pub fn new(
        backend: &'static str,
        config_sha256: String,
        compute_dtype: &'static str,
        resolution: DepthResolution,
        plan: &DepthInputPlan,
    ) -> Self {
        let resample = |name: &'static str| if plan.resized() { name } else { "none" };
        Self {
            model_id: DEPTH_MODEL_ID,
            backend,
            task: IrisTask::Depth,
            code_revision: UPSTREAM_CODE_REVISION,
            weights_repo: UPSTREAM_WEIGHTS_REPO,
            weights_revision: UPSTREAM_WEIGHTS_REVISION,
            config_sha256,
            compute_dtype,
            resolution,
            source_width: plan.source_width,
            source_height: plan.source_height,
            model_width: plan.model_width,
            model_height: plan.model_height,
            input_resample: resample("lanczos"),
            output_resample: resample("bilinear"),
            value_convention: VALUE_CONVENTION,
        }
    }
}

/// A depth prediction and its provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthOutput {
    pub map: DepthMap,
    pub metadata: DepthMetadata,
}

/// The Iris depth task, implemented by `mlx_gen_iris::depth` and `candle_gen_iris::depth`.
pub trait IrisDepthEstimator {
    /// `"mlx"` | `"candle"`.
    fn backend(&self) -> &'static str;
    /// Predict the raw relative-log-depth map at the source resolution. Emits one
    /// `Progress::Step { current: 1, total: 1 }` after the forward; a cancelled request returns
    /// [`Error::Canceled`] and no map.
    fn estimate(
        &self,
        req: &DepthRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<DepthOutput>;
}

/// numpy `percentile(values, q)` (method `linear`).
fn percentile(sorted: &[f32], q: f64) -> f32 {
    let n = sorted.len();
    let index = q / 100.0 * (n - 1) as f64;
    let lo = index.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let t = index - lo as f64;
    let (a, b) = (sorted[lo] as f64, sorted[hi] as f64);
    // numpy `_lerp`: a + (b − a)·t, evaluated from the far end when t ≥ 0.5
    let v = if t >= 0.5 {
        b - (b - a) * (1.0 - t)
    } else {
        a + (b - a) * t
    };
    v as f32
}

/// The explicit **near-bright** adapter for depth-control consumers, upstream's `colorize` scaling
/// without the colormap: `clip((p98 − d) / max(p98 − p2, 1e-6), 0, 1)` over the map's own 2nd–98th
/// percentile range — 1 = near, 0 = far. Returns a new `H × W` buffer; the map is not touched.
pub fn near_bright_unit(map: &DepthMap) -> Vec<f32> {
    if map.values.is_empty() {
        return Vec::new();
    }
    let mut sorted = map.values.clone();
    sorted.sort_by(f32::total_cmp);
    let (low, high) = (percentile(&sorted, 2.0), percentile(&sorted, 98.0));
    let range = (high - low).max(1e-6);
    map.values
        .iter()
        .map(|&d| ((high - d) / range).clamp(0.0, 1.0))
        .collect()
}

/// [`near_bright_unit`] as an RGB8 control image (gray broadcast, `round(255 · near)`) at the map's
/// size — the near-bright depth convention control adapters consume.
pub fn near_bright_control_image(map: &DepthMap) -> Image {
    let pixels = near_bright_unit(map)
        .into_iter()
        .flat_map(|v| {
            let g = (v * 255.0).round() as u8;
            [g, g, g]
        })
        .collect();
    Image {
        width: map.width,
        height: map.height,
        pixels,
    }
}

/// Upstream's preview, `colorize(depth, "inferno")`: [`near_bright_unit`] through matplotlib's
/// 256-entry `inferno` colormap (`index = min(⌊256 · near⌋, 255)`), `round(255 · rgb)`.
pub fn colorize_inferno(map: &DepthMap) -> Image {
    let pixels = near_bright_unit(map)
        .into_iter()
        .flat_map(|v| INFERNO[((v * 256.0) as usize).min(255)])
        .collect();
    Image {
        width: map.width,
        height: map.height,
        pixels,
    }
}

/// matplotlib `colormaps["inferno"]` (N = 256), each entry `round(255 · rgb)`.
const INFERNO: [[u8; 3]; 256] = [
    [0, 0, 4],
    [1, 0, 5],
    [1, 1, 6],
    [1, 1, 8],
    [2, 1, 10],
    [2, 2, 12],
    [2, 2, 14],
    [3, 2, 16],
    [4, 3, 18],
    [4, 3, 20],
    [5, 4, 23],
    [6, 4, 25],
    [7, 5, 27],
    [8, 5, 29],
    [9, 6, 31],
    [10, 7, 34],
    [11, 7, 36],
    [12, 8, 38],
    [13, 8, 41],
    [14, 9, 43],
    [16, 9, 45],
    [17, 10, 48],
    [18, 10, 50],
    [20, 11, 52],
    [21, 11, 55],
    [22, 11, 57],
    [24, 12, 60],
    [25, 12, 62],
    [27, 12, 65],
    [28, 12, 67],
    [30, 12, 69],
    [31, 12, 72],
    [33, 12, 74],
    [35, 12, 76],
    [36, 12, 79],
    [38, 12, 81],
    [40, 11, 83],
    [41, 11, 85],
    [43, 11, 87],
    [45, 11, 89],
    [47, 10, 91],
    [49, 10, 92],
    [50, 10, 94],
    [52, 10, 95],
    [54, 9, 97],
    [56, 9, 98],
    [57, 9, 99],
    [59, 9, 100],
    [61, 9, 101],
    [62, 9, 102],
    [64, 10, 103],
    [66, 10, 104],
    [68, 10, 104],
    [69, 10, 105],
    [71, 11, 106],
    [73, 11, 106],
    [74, 12, 107],
    [76, 12, 107],
    [77, 13, 108],
    [79, 13, 108],
    [81, 14, 108],
    [82, 14, 109],
    [84, 15, 109],
    [85, 15, 109],
    [87, 16, 110],
    [89, 16, 110],
    [90, 17, 110],
    [92, 18, 110],
    [93, 18, 110],
    [95, 19, 110],
    [97, 19, 110],
    [98, 20, 110],
    [100, 21, 110],
    [101, 21, 110],
    [103, 22, 110],
    [105, 22, 110],
    [106, 23, 110],
    [108, 24, 110],
    [109, 24, 110],
    [111, 25, 110],
    [113, 25, 110],
    [114, 26, 110],
    [116, 26, 110],
    [117, 27, 110],
    [119, 28, 109],
    [120, 28, 109],
    [122, 29, 109],
    [124, 29, 109],
    [125, 30, 109],
    [127, 30, 108],
    [128, 31, 108],
    [130, 32, 108],
    [132, 32, 107],
    [133, 33, 107],
    [135, 33, 107],
    [136, 34, 106],
    [138, 34, 106],
    [140, 35, 105],
    [141, 35, 105],
    [143, 36, 105],
    [144, 37, 104],
    [146, 37, 104],
    [147, 38, 103],
    [149, 38, 103],
    [151, 39, 102],
    [152, 39, 102],
    [154, 40, 101],
    [155, 41, 100],
    [157, 41, 100],
    [159, 42, 99],
    [160, 42, 99],
    [162, 43, 98],
    [163, 44, 97],
    [165, 44, 96],
    [166, 45, 96],
    [168, 46, 95],
    [169, 46, 94],
    [171, 47, 94],
    [173, 48, 93],
    [174, 48, 92],
    [176, 49, 91],
    [177, 50, 90],
    [179, 50, 90],
    [180, 51, 89],
    [182, 52, 88],
    [183, 53, 87],
    [185, 53, 86],
    [186, 54, 85],
    [188, 55, 84],
    [189, 56, 83],
    [191, 57, 82],
    [192, 58, 81],
    [193, 58, 80],
    [195, 59, 79],
    [196, 60, 78],
    [198, 61, 77],
    [199, 62, 76],
    [200, 63, 75],
    [202, 64, 74],
    [203, 65, 73],
    [204, 66, 72],
    [206, 67, 71],
    [207, 68, 70],
    [208, 69, 69],
    [210, 70, 68],
    [211, 71, 67],
    [212, 72, 66],
    [213, 74, 65],
    [215, 75, 63],
    [216, 76, 62],
    [217, 77, 61],
    [218, 78, 60],
    [219, 80, 59],
    [221, 81, 58],
    [222, 82, 56],
    [223, 83, 55],
    [224, 85, 54],
    [225, 86, 53],
    [226, 87, 52],
    [227, 89, 51],
    [228, 90, 49],
    [229, 92, 48],
    [230, 93, 47],
    [231, 94, 46],
    [232, 96, 45],
    [233, 97, 43],
    [234, 99, 42],
    [235, 100, 41],
    [235, 102, 40],
    [236, 103, 38],
    [237, 105, 37],
    [238, 106, 36],
    [239, 108, 35],
    [239, 110, 33],
    [240, 111, 32],
    [241, 113, 31],
    [241, 115, 29],
    [242, 116, 28],
    [243, 118, 27],
    [243, 120, 25],
    [244, 121, 24],
    [245, 123, 23],
    [245, 125, 21],
    [246, 126, 20],
    [246, 128, 19],
    [247, 130, 18],
    [247, 132, 16],
    [248, 133, 15],
    [248, 135, 14],
    [248, 137, 12],
    [249, 139, 11],
    [249, 140, 10],
    [249, 142, 9],
    [250, 144, 8],
    [250, 146, 7],
    [250, 148, 7],
    [251, 150, 6],
    [251, 151, 6],
    [251, 153, 6],
    [251, 155, 6],
    [251, 157, 7],
    [252, 159, 7],
    [252, 161, 8],
    [252, 163, 9],
    [252, 165, 10],
    [252, 166, 12],
    [252, 168, 13],
    [252, 170, 15],
    [252, 172, 17],
    [252, 174, 18],
    [252, 176, 20],
    [252, 178, 22],
    [252, 180, 24],
    [251, 182, 26],
    [251, 184, 29],
    [251, 186, 31],
    [251, 188, 33],
    [251, 190, 35],
    [250, 192, 38],
    [250, 194, 40],
    [250, 196, 42],
    [250, 198, 45],
    [249, 199, 47],
    [249, 201, 50],
    [249, 203, 53],
    [248, 205, 55],
    [248, 207, 58],
    [247, 209, 61],
    [247, 211, 64],
    [246, 213, 67],
    [246, 215, 70],
    [245, 217, 73],
    [245, 219, 76],
    [244, 221, 79],
    [244, 223, 83],
    [244, 225, 86],
    [243, 227, 90],
    [243, 229, 93],
    [242, 230, 97],
    [242, 232, 101],
    [242, 234, 105],
    [241, 236, 109],
    [241, 237, 113],
    [241, 239, 117],
    [241, 241, 121],
    [242, 242, 125],
    [242, 244, 130],
    [243, 245, 134],
    [243, 246, 138],
    [244, 248, 142],
    [245, 249, 146],
    [246, 250, 150],
    [248, 251, 154],
    [249, 252, 157],
    [250, 253, 161],
    [252, 255, 164],
];

/// Run the shared depth pipeline around a backend forward: validate, plan, prepare, `forward`
/// (`[3, H', W']` input → `H' × W'` prediction), resize back, describe. Backends call this so both
/// apply the identical host-side source preprocessing.
pub fn estimate_with(
    req: &DepthRequest,
    patch: usize,
    describe: impl FnOnce(&DepthInputPlan) -> DepthMetadata,
    on_progress: &mut dyn FnMut(Progress),
    forward: impl FnOnce(&[f32], &DepthInputPlan) -> Result<Vec<f32>>,
) -> Result<DepthOutput> {
    validate_depth_request(req)?;
    let plan = plan_depth_input(req.image.width, req.image.height, req.resolution, patch)?;
    if req.cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let input = prepare_depth_input(&req.image, &plan)?;
    let prediction = forward(&input, &plan)?;
    if req.cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    on_progress(Progress::Step {
        current: 1,
        total: 1,
    });
    let map = depth_to_source(prediction, &plan)?;
    Ok(DepthOutput {
        map,
        metadata: describe(&plan),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32) -> Image {
        Image {
            width,
            height,
            pixels: (0..width * height * 3)
                .map(|i| (i * 37 % 256) as u8)
                .collect(),
        }
    }

    #[test]
    fn the_size_law_matches_upstream() {
        let plan = |w, h, r| {
            let p = plan_depth_input(w, h, r, 16).unwrap();
            (p.model_width, p.model_height)
        };
        // capped: 4000x3000 -> scale 0.256 -> (1024, 768)
        assert_eq!(plan(4000, 3000, DepthResolution::default()), (1024, 768));
        // never upscaled; rounded to the patch grid
        assert_eq!(plan(100, 50, DepthResolution::default()), (96, 48));
        // native keeps the size, rounded half-to-even: 1000/16 = 62.5 -> 62 patches
        assert_eq!(plan(1000, 750, DepthResolution::Native), (992, 752));
        // a tiny side is at least one patch
        assert_eq!(plan(3, 5, DepthResolution::Native), (16, 16));
        // Python round-half-even: 24/16 = 1.5 -> 2 patches, 40/16 = 2.5 -> 2 patches
        assert_eq!(plan(24, 40, DepthResolution::Native), (32, 32));
        let p = plan_depth_input(64, 32, DepthResolution::Native, 16).unwrap();
        assert!(!p.resized());
        assert!(plan_depth_input(64, 32, DepthResolution::Capped(0), 16).is_err());
    }

    #[test]
    fn requests_are_validated() {
        let mut req = DepthRequest {
            image: image(4, 3),
            ..Default::default()
        };
        validate_depth_request(&req).unwrap();
        req.image.pixels.pop();
        assert!(validate_depth_request(&req).is_err());
        req.image = image(4, 3);
        req.resolution = DepthResolution::Capped(0);
        assert!(validate_depth_request(&req).is_err());
        req.resolution = DepthResolution::Native;
        req.image = image(0, 3);
        assert!(validate_depth_request(&req).is_err());
    }

    #[test]
    fn input_maps_to_minus_one_one_channel_major() {
        let img = Image {
            width: 2,
            height: 1,
            pixels: vec![0, 255, 51, 255, 0, 102],
        };
        let plan = plan_depth_input(2, 1, DepthResolution::Native, 1).unwrap();
        let x = prepare_depth_input(&img, &plan).unwrap();
        let v = |b: u8| b as f32 / 255.0 * 2.0 - 1.0;
        assert_eq!(x, [v(0), v(255), v(255), v(0), v(51), v(102)]);
    }

    #[test]
    fn bilinear_matches_torch_reference_values() {
        // torch.nn.functional.interpolate(torch.tensor([[0., 1.], [2., 3.]])[None, None],
        //     size=(3, 4), mode="bilinear", align_corners=False)
        let out = interpolate_bilinear(&[0.0, 1.0, 2.0, 3.0], 2, 2, 3, 4).unwrap();
        let want = [
            0.0, 0.25, 0.75, 1.0, //
            1.0, 1.25, 1.75, 2.0, //
            2.0, 2.25, 2.75, 3.0,
        ];
        for (a, b) in out.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{out:?}");
        }
        // downsampling picks the two-tap blend, no antialias: 4 -> 2 samples (0.5, 2.5)
        let out = interpolate_bilinear(&[0.0, 1.0, 2.0, 3.0], 1, 4, 1, 2).unwrap();
        assert_eq!(out, [0.5, 2.5]);
        assert!(interpolate_bilinear(&[0.0], 1, 2, 1, 1).is_err());
    }

    #[test]
    fn presentation_adapters_do_not_touch_the_raw_map() {
        let values: Vec<f32> = (0..100).map(|i| i as f32 / 50.0 - 1.0).collect();
        let map = DepthMap {
            width: 10,
            height: 10,
            values: values.clone(),
        };
        let near = near_bright_unit(&map);
        assert_eq!(map.values, values, "the raw map is untouched");
        // numpy: percentile(linspace, [2, 98]) on 100 points -> index 1.98 / 97.02
        let (low, high) = (-1.0 + 1.98 / 50.0, -1.0 + 97.02 / 50.0);
        let want = |d: f32| ((high - d) / (high - low)).clamp(0.0, 1.0);
        for (i, &v) in near.iter().enumerate() {
            assert!((v - want(values[i])).abs() < 1e-5, "{i}: {v}");
        }
        assert_eq!(near[0], 1.0, "nearest is brightest");
        assert_eq!(near[99], 0.0, "farthest is darkest");
        let control = near_bright_control_image(&map);
        assert_eq!((control.width, control.height), (10, 10));
        assert_eq!(&control.pixels[..3], &[255, 255, 255]);
        assert_eq!(&control.pixels[297..], &[0, 0, 0]);
        let preview = colorize_inferno(&map);
        assert_eq!(&preview.pixels[..3], &INFERNO[255]);
        assert_eq!(&preview.pixels[297..], &INFERNO[0]);
        // a constant map does not divide by zero
        let flat = DepthMap {
            width: 2,
            height: 1,
            values: vec![0.3, 0.3],
        };
        assert_eq!(near_bright_unit(&flat), [0.0, 0.0]);
    }

    #[test]
    fn the_pipeline_resizes_back_and_describes_itself() {
        let req = DepthRequest {
            image: image(30, 20),
            resolution: DepthResolution::Capped(16),
            ..Default::default()
        };
        let mut steps = Vec::new();
        let out = estimate_with(
            &req,
            4,
            |plan| DepthMetadata::new("test", "c".repeat(64), "float32", req.resolution, plan),
            &mut |p| steps.push(p),
            |input, plan| {
                assert_eq!((plan.model_width, plan.model_height), (16, 12));
                assert_eq!(input.len(), 3 * 16 * 12);
                Ok(vec![0.5; 16 * 12])
            },
        )
        .unwrap();
        assert_eq!((out.map.width, out.map.height), (30, 20));
        assert!(out.map.values.iter().all(|&v| (v - 0.5).abs() < 1e-6));
        assert_eq!(
            steps,
            [Progress::Step {
                current: 1,
                total: 1
            }]
        );
        let m = &out.metadata;
        assert_eq!(m.model_id, "iris_3b_depth");
        assert_eq!(
            (m.input_resample, m.output_resample),
            ("lanczos", "bilinear")
        );
        assert_eq!(m.value_convention, VALUE_CONVENTION);
        assert_eq!((m.model_width, m.model_height), (16, 12));

        let cancel = CancelFlag::new();
        cancel.cancel();
        let req = DepthRequest { cancel, ..req };
        let err = estimate_with(
            &req,
            4,
            |plan| DepthMetadata::new("test", String::new(), "float32", req.resolution, plan),
            &mut |_| {},
            |_, _| panic!("a cancelled request never runs the forward"),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Canceled));
    }
}
