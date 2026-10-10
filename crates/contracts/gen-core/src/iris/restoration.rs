//! Iris-3B **restoration / upscaling** (`upscaler/`, story sc-25683) — the backend-neutral half of
//! upstream `iris3b/downstream/restoration.py` + `scripts/upscale.py`.
//!
//! The restorer is a one-step model: the low-quality image is upsampled bicubically to the output
//! size, mapped to `[-1, 1]`, and run through the Iris backbone **once** at model time
//! `sigma · num_train_timesteps` with the shipped empty-prompt states (no text encoder, E4); the
//! restored pixels are `x − sigma · v`. Everything around that forward is host-side and lives here,
//! so the MLX and Candle providers share one definition of it:
//!
//! * the **planner** ([`plan`]) — budget, output size, small-image enlargement, padding and the tile
//!   grid, returned *before* any execution so a consumer can show them;
//! * the **pixel path** ([`restore`]) — the torch-exact bicubic upsample (`scale_factor`, `a = −0.75`),
//!   the antialiased bicubic enlarge/shrink (`a = −0.5`), 50 %-overlap tiling with upstream's
//!   Gaussian fusion window, the wavelet colour fix and the 8-bit quantization — calling back into
//!   the backend only for each tile's velocity.
//!
//! The backend owns only `velocity(tile) -> v`; tile iteration, cancellation between tiles and
//! per-tile progress are here.

pub use super::downstream::{EmptyPrompt, EMPTY_PROMPT_FILE};
use super::downstream::{TaskExport, TaskSettings};
use super::{IrisConfig, IrisTask};
use crate::media::Image;
use crate::runtime::{CancelFlag, LoadSpec, Progress};
use crate::transform::{InputBudget, InputSizing, TargetSize, TransformRequest};
use crate::{Error, Result};

/// Registry id of the Iris-3B restoration transform.
pub const MODEL_ID: &str = "iris_3b_restore";
/// `Restorer.__call__(scale=4.0)` — the release default.
pub const DEFAULT_SCALE: f64 = 4.0;
/// `fit_budget(short_side=512, long_side=1024)` — `scripts/upscale.py`'s default input budget.
pub const INPUT_BUDGET: InputBudget = InputBudget {
    short_side: 512,
    long_side: 1024,
};
/// `color_fix=True` — the release default.
pub const DEFAULT_COLOR_FIX: bool = true;
/// The release export's `task.tile` (the 1024² fine-tuning crop) — read from `config.yaml` at load.
pub const RELEASE_TILE: u32 = 1024;
/// The release export's `task.sigma` — read from `config.yaml` at load.
pub const RELEASE_SIGMA: f64 = 0.5;
/// Colour fix: à-trous binomial pyramid dilations (effective radius 16 px).
pub const COLOR_FIX_DILATIONS: [usize; 5] = [1, 2, 4, 8, 16];

// ---------------------------------------------------------------------------------------------
// Export identity + resources
// ---------------------------------------------------------------------------------------------

/// The `task` section of a restoration export's `config.yaml`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestorationSettings {
    /// One-step noise level (`model_t / 1000`); the forward runs at `sigma · num_train_timesteps`.
    pub sigma: f64,
    /// Tile side (the fine-tuning crop); tiles overlap by 50 % (stride `tile / 2`).
    pub tile: u32,
}

impl RestorationSettings {
    /// Read `sigma` / `tile` from the export's `task` section (task identity is already checked by
    /// [`TaskExport`]: a generation checkpoint or a depth export never reaches this).
    pub fn from_task(task: &TaskSettings, model_id: &str) -> Result<Self> {
        let get = |key: &str| {
            task.get(key)
                .ok_or_else(|| Error::Msg(format!("{model_id}: config.yaml task.{key} is missing")))
        };
        let sigma: f64 = get("sigma")?
            .parse()
            .map_err(|_| Error::Msg(format!("{model_id}: task.sigma is not a number")))?;
        let tile: u32 = get("tile")?
            .parse()
            .map_err(|_| Error::Msg(format!("{model_id}: task.tile is not a positive integer")))?;
        Ok(Self { sigma, tile })
    }

    /// Check the settings against the backbone: a positive finite sigma, a tile that is a positive
    /// multiple of the patch (so every tile patchifies), and upstream's v-prediction requirement.
    pub fn validate(&self, config: &IrisConfig, model_id: &str) -> Result<()> {
        if !(self.sigma.is_finite() && self.sigma > 0.0 && self.sigma <= 1.0) {
            return Err(Error::Unsupported(format!(
                "{model_id}: task.sigma = {} is outside (0, 1]",
                self.sigma
            )));
        }
        let patch = config.model.patch_size as u32;
        if self.tile < 2 || patch == 0 || !self.tile.is_multiple_of(patch) {
            return Err(Error::Unsupported(format!(
                "{model_id}: task.tile = {} is not a positive multiple of model.patch_size = {patch}",
                self.tile
            )));
        }
        if config.flow.prediction != "v" {
            return Err(Error::Unsupported(format!(
                "{model_id}: the restorer expects a v-prediction parent (flow.prediction = {})",
                config.flow.prediction
            )));
        }
        Ok(())
    }

    /// Model time of the one-step forward (`sigma · num_train_timesteps`, as upstream's
    /// `self.time`, an f64 product handed to torch as f32).
    pub fn model_time(&self, config: &IrisConfig) -> f32 {
        (self.sigma * config.flow.num_train_timesteps as f64) as f32
    }

    /// The fixed tile geometry of this export.
    pub fn geometry(&self, config: &IrisConfig) -> TileGeometry {
        TileGeometry {
            tile: self.tile,
            patch: config.model.patch_size as u32,
        }
    }
}

/// The resolved, identity-checked resources of the restoration task: the shared downstream export
/// closure ([`TaskExport`], the `upscaler/` folder) plus its validated `task` settings.
#[derive(Clone, Debug, PartialEq)]
pub struct RestorationResources {
    pub export: TaskExport,
    pub settings: RestorationSettings,
}

impl RestorationResources {
    /// Resolve the restoration task from a load spec: `spec.weights` is the `upscaler/` export
    /// directory. The task runs **without** the text encoder (E4), so any component — or a
    /// generation / depth export staged in its place — is a typed refusal ([`TaskExport`]).
    pub fn from_spec(spec: &LoadSpec, model_id: &str) -> Result<Self> {
        Self::checked(
            TaskExport::from_spec(spec, IrisTask::Restoration, model_id)?,
            model_id,
        )
    }

    /// [`Self::from_spec`] on a directory.
    pub fn from_dir(dir: &std::path::Path, model_id: &str) -> Result<Self> {
        Self::checked(
            TaskExport::from_dir(dir, IrisTask::Restoration, model_id)?,
            model_id,
        )
    }

    fn checked(export: TaskExport, model_id: &str) -> Result<Self> {
        let settings = RestorationSettings::from_task(&export.settings, model_id)?;
        settings.validate(&export.config, model_id)?;
        Ok(Self { export, settings })
    }

    pub fn config(&self) -> &IrisConfig {
        &self.export.config
    }

    /// Read the shipped empty-prompt states.
    pub fn empty_prompt(&self) -> Result<EmptyPrompt> {
        self.export.empty_prompt()
    }
}

// ---------------------------------------------------------------------------------------------
// Request → options
// ---------------------------------------------------------------------------------------------

/// The resolved controls of one restoration request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestorationOptions {
    /// Output = `floor(input · scale)` per side; any positive finite scale (1× = restore only).
    pub scale: f64,
    pub input_sizing: InputSizing,
    pub color_fix: bool,
}

impl Default for RestorationOptions {
    fn default() -> Self {
        Self {
            scale: DEFAULT_SCALE,
            input_sizing: InputSizing::Budgeted,
            color_fix: DEFAULT_COLOR_FIX,
        }
    }
}

/// The f64 scale a [`TargetSize::Scale`] means: the shortest decimal that round-trips the f32
/// (`1.3f32` → `1.3`), i.e. the Python float a caller of upstream `Restorer(scale=…)` writes.
/// Widening the f32 directly (`1.2999999523…`) would floor a 1000-px side at ×1.3 to 1299 instead
/// of torch's `floor(1000 · 1.3) = 1300`, and skew the bicubic coordinate scale `1 / scale`.
fn request_scale(s: f32) -> f64 {
    // `f32`'s `Display` is the shortest round-trip decimal; every such string parses as an f64
    // (incl. `NaN` / `inf`, which `check_scale` then refuses).
    s.to_string().parse().unwrap_or(f64::NAN)
}

impl RestorationOptions {
    /// Resolve a [`TransformRequest`], refusing every field the restorer does not honour by name
    /// (it is deterministic and one-step: no seed, strength or step count; its output size is a
    /// scale factor, not a min-edge or explicit resolution).
    pub fn from_request(req: &TransformRequest, model_id: &str) -> Result<Self> {
        let scale = match req.target {
            TargetSize::ModelDefault => DEFAULT_SCALE,
            TargetSize::Scale(s) => request_scale(s),
            TargetSize::MinEdge(_) | TargetSize::Resolution { .. } => {
                return Err(Error::Unsupported(format!(
                    "{model_id}: the Iris-3B restorer sizes its output by a scale factor only \
                     (upstream `Restorer(scale=…)`); request TargetSize::Scale"
                )))
            }
        };
        let refusals = [
            (
                "seed",
                req.seed.is_some(),
                "the restorer is deterministic (one forward, no noise)",
            ),
            (
                "strength",
                req.strength.is_some(),
                "the restorer has no strength control",
            ),
            (
                "steps",
                req.steps.is_some_and(|s| s != 1),
                "the restorer is a single forward pass",
            ),
        ];
        for (field, set, why) in refusals {
            if set {
                return Err(Error::Unsupported(format!(
                    "{model_id}: `{field}` is not a control of the Iris-3B restorer — {why}"
                )));
            }
        }
        let options = Self {
            scale,
            input_sizing: req.input_sizing,
            color_fix: req.color_fix.unwrap_or(DEFAULT_COLOR_FIX),
        };
        options.check_scale(model_id)?;
        Ok(options)
    }

    fn check_scale(&self, model_id: &str) -> Result<()> {
        if !(self.scale.is_finite() && self.scale > 0.0) {
            return Err(Error::Unsupported(format!(
                "{model_id}: scale {} is not a positive finite factor",
                self.scale
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------------------------

/// Width × height of one stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dims {
    pub width: u32,
    pub height: u32,
}

impl Dims {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// The export's fixed tile geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileGeometry {
    pub tile: u32,
    pub patch: u32,
}

impl TileGeometry {
    /// The released `upscaler/` export (1024-px tiles, 16-px patches).
    pub const RELEASE: TileGeometry = TileGeometry {
        tile: RELEASE_TILE,
        patch: 16,
    };

    /// `tile // 2` — 50 % overlap.
    pub fn stride(&self) -> u32 {
        self.tile / 2
    }
}

/// Everything a restoration will do, computed before it runs.
#[derive(Clone, Debug, PartialEq)]
pub struct RestorationPlan {
    /// The image as handed in.
    pub source: Dims,
    pub input_sizing: InputSizing,
    /// The image after the input budget (`== source` when no downscale was needed or under
    /// [`InputSizing::Original`]).
    pub input: Dims,
    /// Whether the budget actually downscaled the input (Lanczos).
    pub budget_applied: bool,
    pub scale: f64,
    /// The exact output geometry: `floor(input · scale)` per side.
    pub output: Dims,
    /// The size the model runs at: the output, or — when the output's short side is at most one
    /// tile — the output enlarged so its short side is exactly one tile (resized back afterwards).
    pub processing: Dims,
    pub enlarged: bool,
    /// `processing` zero-padded up to a multiple of the patch (bottom/right).
    pub padded: Dims,
    pub tile: u32,
    pub stride: u32,
    /// Tile top offsets (empty when the padded image is a single tile).
    pub tile_rows: Vec<u32>,
    /// Tile left offsets (empty when the padded image is a single tile).
    pub tile_cols: Vec<u32>,
    pub color_fix: bool,
}

impl RestorationPlan {
    /// One forward over the whole padded image (upstream `tiled`'s `height <= tile and width <=
    /// tile` branch, no fusion window).
    pub fn single_pass(&self) -> bool {
        self.tile_rows.is_empty()
    }

    /// Model forwards this plan runs.
    pub fn forward_count(&self) -> usize {
        if self.single_pass() {
            1
        } else {
            self.tile_rows.len() * self.tile_cols.len()
        }
    }

    /// `(top, left)` of every tile, in upstream's row-major order.
    pub fn tiles(&self) -> Vec<(u32, u32)> {
        if self.single_pass() {
            return vec![(0, 0)];
        }
        self.tile_rows
            .iter()
            .flat_map(|&top| self.tile_cols.iter().map(move |&left| (top, left)))
            .collect()
    }
}

/// `fit_budget`: the size an input is downscaled to (never upscaled) so its short side is at most
/// `budget.short_side` and its long side at most `budget.long_side`. Python's `round` (half to
/// even) on the f64 products, like upstream.
pub fn fit_budget_dims(source: Dims, budget: InputBudget) -> Dims {
    let (w, h) = (source.width as f64, source.height as f64);
    let scale = 1.0f64
        .min(budget.short_side as f64 / w.min(h))
        .min(budget.long_side as f64 / w.max(h));
    if scale < 1.0 {
        Dims::new(
            ((w * scale).round_ties_even() as u32).max(1),
            ((h * scale).round_ties_even() as u32).max(1),
        )
    } else {
        source
    }
}

/// `tile_positions`: start offsets along one axis; the last tile is flush with the far edge.
pub fn tile_positions(size: u32, tile: u32, stride: u32) -> Vec<u32> {
    if size <= tile {
        return vec![0];
    }
    let mut out: Vec<u32> = (0..size - tile).step_by(stride.max(1) as usize).collect();
    out.push(size - tile);
    out
}

/// `floor(side · scale)` — torch's `F.interpolate(scale_factor=…)` output size (f64 product).
pub fn scaled_side(side: u32, scale: f64) -> u64 {
    (side as f64 * scale).floor() as u64
}

/// Plan a restoration of a `source`-sized image. Pure and backend-neutral: the consumer calls it
/// with the release geometry ([`TileGeometry::RELEASE`]) to show the processing size, tile grid and
/// output size before running; the provider calls it with the loaded export's geometry. A scale
/// whose output rounds to zero pixels (or overflows) is a typed refusal — never a different scale.
pub fn plan(
    source: Dims,
    options: &RestorationOptions,
    geometry: TileGeometry,
    model_id: &str,
) -> Result<RestorationPlan> {
    options.check_scale(model_id)?;
    if source.width == 0 || source.height == 0 {
        return Err(Error::Msg(format!(
            "{model_id}: the input image is {}x{} — both sides must be positive",
            source.width, source.height
        )));
    }
    if geometry.patch == 0 || geometry.tile < 2 || !geometry.tile.is_multiple_of(geometry.patch) {
        return Err(Error::Unsupported(format!(
            "{model_id}: tile {} is not a positive multiple of patch {}",
            geometry.tile, geometry.patch
        )));
    }
    let input = match options.input_sizing {
        InputSizing::Budgeted => fit_budget_dims(source, INPUT_BUDGET),
        InputSizing::Original => source,
    };
    let out_w = scaled_side(input.width, options.scale);
    let out_h = scaled_side(input.height, options.scale);
    if out_w == 0 || out_h == 0 {
        return Err(Error::Unsupported(format!(
            "{model_id}: scale {} of a {}x{} input is a {out_w}x{out_h} output — pick a larger \
             scale",
            options.scale, input.width, input.height
        )));
    }
    // Every stage size is computed in u64 and must fit u32: the output, the enlarged processing
    // size (a 1×N output enlarges its long side by `tile`) and its patch padding.
    let fits = |stage: &str, w: u64, h: u64| -> Result<Dims> {
        match (u32::try_from(w), u32::try_from(h)) {
            (Ok(w), Ok(h)) => Ok(Dims::new(w, h)),
            _ => Err(Error::Unsupported(format!(
                "{model_id}: scale {} of a {}x{} input needs a {w}x{h} {stage} — too large",
                options.scale, input.width, input.height
            ))),
        }
    };
    let output = fits("output", out_w, out_h)?;
    let short = output.width.min(output.height);
    let tile = u64::from(geometry.tile);
    let (processing, enlarged) = if short <= geometry.tile {
        // `ratio = tile / min(out_size)`; `max(tile, round(side · ratio))` per side.
        let ratio = geometry.tile as f64 / short as f64;
        // f64 → u64 saturates; anything near u64::MAX is refused by `fits` anyway.
        let side = |s: u32| ((s as f64 * ratio).round_ties_even() as u64).max(tile);
        (
            fits(
                "enlarged processing size",
                side(output.width),
                side(output.height),
            )?,
            true,
        )
    } else {
        (output, false)
    };
    let patch = u64::from(geometry.patch);
    let pad = |s: u32| u64::from(s).div_ceil(patch) * patch;
    let padded = fits(
        "patch-padded processing size",
        pad(processing.width),
        pad(processing.height),
    )?;
    let stride = geometry.stride();
    let (tile_rows, tile_cols) = if padded.height <= geometry.tile && padded.width <= geometry.tile
    {
        (Vec::new(), Vec::new())
    } else {
        (
            tile_positions(padded.height, geometry.tile, stride),
            tile_positions(padded.width, geometry.tile, stride),
        )
    };
    Ok(RestorationPlan {
        source,
        input_sizing: options.input_sizing,
        input,
        budget_applied: input != source,
        scale: options.scale,
        output,
        processing,
        enlarged,
        padded,
        tile: geometry.tile,
        stride,
        tile_rows,
        tile_cols,
        color_fix: options.color_fix,
    })
}

/// [`RestorationOptions::from_request`] + [`plan`] for a whole request (the image's own size) — what
/// both providers' `validate`/`apply` run, and what a consumer calls with
/// [`TileGeometry::RELEASE`] to preview a request.
pub fn plan_request(
    req: &TransformRequest,
    geometry: TileGeometry,
    model_id: &str,
) -> Result<RestorationPlan> {
    let options = RestorationOptions::from_request(req, model_id)?;
    let (w, h) = (req.image.width as usize, req.image.height as usize);
    if req.image.pixels.len() != w * h * 3 {
        return Err(Error::Msg(format!(
            "{model_id}: a {w}x{h} RGB image needs {} bytes, got {}",
            w * h * 3,
            req.image.pixels.len()
        )));
    }
    plan(
        Dims::new(req.image.width, req.image.height),
        &options,
        geometry,
        model_id,
    )
}

// ---------------------------------------------------------------------------------------------
// Pixel ops (planar CHW f32, the torch layout) — torch-exact f32 arithmetic where it matters
// ---------------------------------------------------------------------------------------------

/// A planar `[C, H, W]` f32 image.
#[derive(Clone, Debug, PartialEq)]
pub struct Planes {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Planes {
    pub fn zeros(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
            data: vec![0.0; channels * height * width],
        }
    }

    /// `torch.from_numpy(np.array(rgb8)).permute(2, 0, 1) / 255`.
    pub fn from_rgb8(image: &Image) -> Result<Self> {
        let (w, h) = (image.width as usize, image.height as usize);
        if image.pixels.len() != w * h * 3 {
            return Err(Error::Msg(format!(
                "iris: a {w}x{h} RGB image needs {} bytes, got {}",
                w * h * 3,
                image.pixels.len()
            )));
        }
        let mut data = vec![0f32; 3 * h * w];
        for (i, px) in image.pixels.chunks_exact(3).enumerate() {
            for c in 0..3 {
                data[c * h * w + i] = px[c] as f32 / 255.0;
            }
        }
        Ok(Self {
            channels: 3,
            height: h,
            width: w,
            data,
        })
    }

    /// `(y.clamp(0, 1)[0].permute(1, 2, 0) * 255).round().byte()` (round half to even).
    pub fn to_rgb8(&self) -> Image {
        let (h, w) = (self.height, self.width);
        let mut pixels = vec![0u8; h * w * 3];
        for c in 0..3 {
            for i in 0..h * w {
                let v = self.data[c * h * w + i].clamp(0.0, 1.0) * 255.0;
                pixels[i * 3 + c] = v.round_ties_even() as u8;
            }
        }
        Image {
            width: w as u32,
            height: h as u32,
            pixels,
        }
    }

    fn plane(&self, c: usize) -> &[f32] {
        let n = self.height * self.width;
        &self.data[c * n..(c + 1) * n]
    }

    /// `[C, top:top+h, left:left+w]` (in bounds).
    pub fn crop(&self, top: usize, left: usize, height: usize, width: usize) -> Planes {
        let mut out = Planes::zeros(self.channels, height, width);
        for c in 0..self.channels {
            for y in 0..height {
                let src = c * self.height * self.width + (top + y) * self.width + left;
                let dst = c * height * width + y * width;
                out.data[dst..dst + width].copy_from_slice(&self.data[src..src + width]);
            }
        }
        out
    }
}

/// `cubic_convolution1` (`|x| ≤ 1`).
fn cubic1(x: f32, a: f32) -> f32 {
    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
}

/// `cubic_convolution2` (`1 < |x| < 2`).
fn cubic2(x: f32, a: f32) -> f32 {
    ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
}

/// The non-antialiased bicubic taps of one axis (`HelperInterpCubic::compute_indices_weights`):
/// source index `scale·(i + 0.5) − 0.5` in f32 with `scale = f32(1 / scale_factor)`,
/// `guard_index_and_lambda`, Keys `a = −0.75`, replicate-clamped indices.
fn bicubic_taps(in_size: usize, out_size: usize, coord_scale: f32) -> Vec<([usize; 4], [f32; 4])> {
    const A: f32 = -0.75;
    (0..out_size)
        .map(|i| {
            let real = coord_scale * (i as f32 + 0.5) - 0.5;
            let index = (real.floor() as i64).min(in_size as i64 - 1);
            let t = (real - index as f32).clamp(0.0, 1.0);
            let w = [
                cubic2(t + 1.0, A),
                cubic1(t, A),
                cubic1(1.0 - t, A),
                cubic2((1.0 - t) + 1.0, A),
            ];
            let idx = std::array::from_fn(|j| {
                (index - 1 + j as i64).clamp(0, in_size as i64 - 1) as usize
            });
            (idx, w)
        })
        .collect()
}

/// `F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)` — output
/// `floor(side · scale)`, coordinate scale `1 / scale` (not `in / out`), f32 accumulation over the
/// 4×4 window in torch's order (rows outer, columns inner).
pub fn bicubic_scale_factor(src: &Planes, scale: f64) -> Planes {
    let out_h = scaled_side(src.height as u32, scale) as usize;
    let out_w = scaled_side(src.width as u32, scale) as usize;
    let coord = (1.0 / scale) as f32;
    let rows = bicubic_taps(src.height, out_h, coord);
    let cols = bicubic_taps(src.width, out_w, coord);
    let mut out = Planes::zeros(src.channels, out_h, out_w);
    for c in 0..src.channels {
        let plane = src.plane(c);
        let dst = &mut out.data[c * out_h * out_w..(c + 1) * out_h * out_w];
        for (y, (ri, rw)) in rows.iter().enumerate() {
            for (x, (ci, cw)) in cols.iter().enumerate() {
                let row = |r: usize| {
                    let base = &plane[r * src.width..];
                    let mut acc = base[ci[0]] * cw[0];
                    for (&i, &w) in ci.iter().zip(cw).skip(1) {
                        acc += base[i] * w;
                    }
                    acc
                };
                let mut acc = row(ri[0]) * rw[0];
                for (&r, &w) in ri.iter().zip(rw).skip(1) {
                    acc += row(r) * w;
                }
                dst[y * out_w + x] = acc;
            }
        }
    }
    out
}

/// The antialiased bicubic taps of one axis (`_compute_indices_min_size_weights_aa`, Keys
/// `a = −0.5`, support scaled by the downscale factor, window clipped to the image and
/// renormalized) — all in f32 like torch's float kernel.
fn bicubic_aa_taps(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f32>)> {
    const A: f32 = -0.5;
    let filter = |x: f32| {
        let x = x.abs();
        if x < 1.0 {
            cubic1(x, A)
        } else if x < 2.0 {
            cubic2(x, A)
        } else {
            0.0
        }
    };
    let scale = in_size as f32 / out_size as f32;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let max_interp = (support.ceil() as usize) * 2 + 1;
    (0..out_size)
        .map(|i| {
            let center = scale * (i as f32 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xsize = (((center + support + 0.5) as i64).min(in_size as i64) - xmin)
                .clamp(0, max_interp as i64) as usize;
            let mut w: Vec<f32> = (0..xsize)
                .map(|j| filter((j as f32 + xmin as f32 - center + 0.5) * invscale))
                .collect();
            let total: f32 = w.iter().fold(0.0, |a, &b| a + b);
            if total != 0.0 {
                for v in &mut w {
                    *v /= total;
                }
            }
            (xmin as usize, w)
        })
        .collect()
}

/// `F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)` — separable,
/// horizontal pass first, an axis whose size is unchanged skipped (torch's
/// `_separable_upsample_generic_Nd_kernel_impl`).
pub fn bicubic_aa_resize(src: &Planes, out_h: usize, out_w: usize) -> Planes {
    let mut cur = src.clone();
    if out_w != cur.width {
        let taps = bicubic_aa_taps(cur.width, out_w);
        let mut next = Planes::zeros(cur.channels, cur.height, out_w);
        for c in 0..cur.channels {
            for y in 0..cur.height {
                let row = &cur.data[(c * cur.height + y) * cur.width..][..cur.width];
                let dst = &mut next.data[(c * cur.height + y) * out_w..][..out_w];
                for (x, (xmin, w)) in taps.iter().enumerate() {
                    let mut acc = row[*xmin] * w[0];
                    for (j, wj) in w.iter().enumerate().skip(1) {
                        acc += row[xmin + j] * wj;
                    }
                    dst[x] = acc;
                }
            }
        }
        cur = next;
    }
    if out_h != cur.height {
        let taps = bicubic_aa_taps(cur.height, out_h);
        let mut next = Planes::zeros(cur.channels, out_h, cur.width);
        let w_ = cur.width;
        for c in 0..cur.channels {
            let plane = &cur.data[c * cur.height * w_..(c + 1) * cur.height * w_];
            for (y, (ymin, w)) in taps.iter().enumerate() {
                let dst = &mut next.data[(c * out_h + y) * w_..][..w_];
                for x in 0..w_ {
                    let mut acc = plane[ymin * w_ + x] * w[0];
                    for (j, wj) in w.iter().enumerate().skip(1) {
                        acc += plane[(ymin + j) * w_ + x] * wj;
                    }
                    dst[x] = acc;
                }
            }
        }
        cur = next;
    }
    cur
}

/// `gaussian_window(tile)`: `[tile, tile]` fusion weights, variance 0.01 in tile-normalized
/// coordinates, vertical centre `tile / 2`, horizontal centre `(tile − 1) / 2` (upstream's exact,
/// deliberately asymmetric centres), computed in f32 like torch.
pub fn gaussian_window(tile: u32) -> Vec<f32> {
    let n = tile as usize;
    let area = (tile as f64 * tile as f64) as f32;
    let axis = |centre: f64| -> Vec<f32> {
        let centre = centre as f32;
        (0..n)
            .map(|i| {
                let d = i as f32 - centre;
                (-(d * d) / area / 0.02f32).exp()
            })
            .collect()
    };
    let cols = axis((tile as f64 - 1.0) / 2.0);
    let rows = axis(tile as f64 / 2.0);
    let mut out = Vec::with_capacity(n * n);
    for r in &rows {
        for c in &cols {
            out.push(r * c);
        }
    }
    out
}

/// `_blur`: depthwise 3×3 binomial `[1 2 1]ᵀ[1 2 1] / 16` at `dilation`, replicate padding.
fn blur(src: &[f32], h: usize, w: usize, dilation: usize) -> Vec<f32> {
    const K: [[f32; 3]; 3] = [
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    ];
    let d = dilation as i64;
    let clamp_y = |y: i64| y.clamp(0, h as i64 - 1) as usize;
    let clamp_x = |x: i64| x.clamp(0, w as i64 - 1) as usize;
    let mut out = vec![0f32; h * w];
    for y in 0..h {
        let ys = [clamp_y(y as i64 - d), y, clamp_y(y as i64 + d)];
        for x in 0..w {
            let xs = [clamp_x(x as i64 - d), x, clamp_x(x as i64 + d)];
            let mut acc = 0f32;
            for (ky, &yy) in ys.iter().enumerate() {
                for (kx, &xx) in xs.iter().enumerate() {
                    acc += src[yy * w + xx] * K[ky][kx];
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// `_split` of one plane: `(high, low)` frequency parts that sum back to the input.
fn split_plane(plane: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut high = vec![0f32; plane.len()];
    let mut low = plane.to_vec();
    for &d in &COLOR_FIX_DILATIONS {
        let blurred = blur(&low, h, w, d);
        for i in 0..high.len() {
            high[i] += low[i] - blurred[i];
        }
        low = blurred;
    }
    (high, low)
}

/// `wavelet_color_fix(content, reference)`: `content`'s high frequencies on `reference`'s low
/// frequencies (both the same size). Channels run on scoped threads.
pub fn wavelet_color_fix(content: &Planes, reference: &Planes) -> Result<Planes> {
    if (content.channels, content.height, content.width)
        != (reference.channels, reference.height, reference.width)
    {
        return Err(Error::Msg(format!(
            "iris: colour fix needs equal sizes, got {}x{} and {}x{}",
            content.width, content.height, reference.width, reference.height
        )));
    }
    let (h, w) = (content.height, content.width);
    let planes: Vec<Vec<f32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..content.channels)
            .map(|c| {
                let (cp, rp) = (content.plane(c), reference.plane(c));
                scope.spawn(move || {
                    let (high, _) = split_plane(cp, h, w);
                    let (_, low) = split_plane(rp, h, w);
                    high.iter().zip(&low).map(|(a, b)| a + b).collect()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("colour-fix worker panicked"))
            .collect()
    });
    Ok(Planes {
        channels: content.channels,
        height: h,
        width: w,
        data: planes.concat(),
    })
}

/// `fit_budget`'s resize: PIL `Image.LANCZOS` on the 8-bit RGB image (bit-exact fixed point).
pub fn fit_budget_image(image: &Image, to: Dims) -> Result<Image> {
    if (image.width, image.height) == (to.width, to.height) {
        return Ok(image.clone());
    }
    let resized = crate::imageops::resize_lanczos_u8(
        &image.pixels,
        image.height as usize,
        image.width as usize,
        to.height as usize,
        to.width as usize,
    )?;
    Ok(Image {
        width: to.width,
        height: to.height,
        pixels: resized.iter().map(|&v| v as u8).collect(),
    })
}

// ---------------------------------------------------------------------------------------------
// The restoration driver
// ---------------------------------------------------------------------------------------------

/// Run a planned restoration. `velocity(tile, height, width)` is the backend's one-step forward:
/// it receives one `[3, height, width]` CHW f32 crop in `[-1, 1]` and returns the model's velocity
/// `v` (same layout); the restored crop is `x − sigma · v` (f32). The cancel flag is checked before
/// every tile and before post-processing — a cancel is [`Error::Canceled`] and no image.
/// [`Progress::Step`] reports each finished tile (`current` of the plan's forward count);
/// [`Progress::Decoding`] marks the post-processing (crop, resize back, colour fix, quantize).
pub fn restore(
    image: &Image,
    plan: &RestorationPlan,
    sigma: f64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    velocity: &mut TileVelocity,
) -> Result<Image> {
    Ok(
        restore_detailed(image, plan, sigma, cancel, on_progress, velocity)?
            .restored
            .to_rgb8(),
    )
}

/// A backend's one-step forward over one tile: `(tile CHW f32 in [-1, 1], height, width)` → the
/// velocity `v` in the same layout.
pub type TileVelocity<'a> = dyn FnMut(&[f32], usize, usize) -> Result<Vec<f32>> + 'a;

/// The float stages of one restoration (what [`restore`] quantizes).
#[derive(Clone, Debug, PartialEq)]
pub struct RestorationOutput {
    /// The fused model output over the padded image, in `[-1, 1]` model space (upstream `tiled()`).
    pub fused: Planes,
    /// The restored image in `[0, 1]` (pre-clamp): cropped, resized back when enlarged, and
    /// colour-fixed when requested.
    pub restored: Planes,
}

/// [`restore`] without the final 8-bit quantization (parity tests and float consumers).
pub fn restore_detailed(
    image: &Image,
    plan: &RestorationPlan,
    sigma: f64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    velocity: &mut TileVelocity,
) -> Result<RestorationOutput> {
    if Dims::new(image.width, image.height) != plan.source {
        return Err(Error::Msg(format!(
            "iris: the plan is for a {}x{} image, got {}x{}",
            plan.source.width, plan.source.height, image.width, image.height
        )));
    }
    let image = if plan.budget_applied {
        fit_budget_image(image, plan.input)?
    } else {
        image.clone()
    };
    let lq = Planes::from_rgb8(&image)?;
    let reference = bicubic_scale_factor(&lq, plan.scale);
    debug_assert_eq!(
        (reference.width as u32, reference.height as u32),
        (plan.output.width, plan.output.height)
    );
    let x = if plan.enlarged {
        bicubic_aa_resize(
            &reference,
            plan.processing.height as usize,
            plan.processing.width as usize,
        )
    } else {
        reference.clone()
    };
    // `F.pad(x * 2 − 1, …)`: zero padding bottom/right to the patch grid.
    let (ph, pw) = (plan.padded.height as usize, plan.padded.width as usize);
    let mut padded = Planes::zeros(3, ph, pw);
    for c in 0..3 {
        for y in 0..x.height {
            for xx in 0..x.width {
                padded.data[(c * ph + y) * pw + xx] =
                    x.data[(c * x.height + y) * x.width + xx] * 2.0 - 1.0;
            }
        }
    }
    drop(x);
    let sigma = sigma as f32;
    let mut restore_tile = |crop: &Planes| -> Result<Vec<f32>> {
        let v = velocity(&crop.data, crop.height, crop.width)?;
        if v.len() != crop.data.len() {
            return Err(Error::Msg(format!(
                "iris: the backend returned {} values for a {} element tile",
                v.len(),
                crop.data.len()
            )));
        }
        Ok(crop
            .data
            .iter()
            .zip(&v)
            .map(|(x, v)| x - sigma * v)
            .collect())
    };
    let total = plan.forward_count() as u32;
    let fused = if plan.single_pass() {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let data = restore_tile(&padded)?;
        on_progress(Progress::Step { current: 1, total });
        Planes { data, ..padded }
    } else {
        let tile = plan.tile as usize;
        let window = gaussian_window(plan.tile);
        let mut out = Planes::zeros(3, ph, pw);
        let mut weight = vec![0f32; ph * pw];
        for (k, (top, left)) in plan.tiles().into_iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let (top, left) = (top as usize, left as usize);
            let restored = restore_tile(&padded.crop(top, left, tile, tile))?;
            for c in 0..3 {
                for y in 0..tile {
                    let row = (c * ph + top + y) * pw + left;
                    let src = c * tile * tile + y * tile;
                    for xx in 0..tile {
                        out.data[row + xx] += restored[src + xx] * window[y * tile + xx];
                    }
                }
            }
            for y in 0..tile {
                for xx in 0..tile {
                    weight[(top + y) * pw + left + xx] += window[y * tile + xx];
                }
            }
            on_progress(Progress::Step {
                current: k as u32 + 1,
                total,
            });
        }
        for plane in out.data.chunks_exact_mut(ph * pw) {
            for (v, w) in plane.iter_mut().zip(&weight) {
                *v /= w;
            }
        }
        out
    };
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    on_progress(Progress::Decoding);
    // `(y[..., :height, :width] + 1) / 2`
    let (h, w) = (
        plan.processing.height as usize,
        plan.processing.width as usize,
    );
    let mut y = fused.crop(0, 0, h, w);
    for v in &mut y.data {
        *v = (*v + 1.0) / 2.0;
    }
    if plan.enlarged {
        y = bicubic_aa_resize(&y, plan.output.height as usize, plan.output.width as usize);
    }
    if plan.color_fix {
        y = wavelet_color_fix(&y, &reference)?;
    }
    Ok(RestorationOutput { fused, restored: y })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::WeightsSource;

    fn opts(scale: f64, input_sizing: InputSizing) -> RestorationOptions {
        RestorationOptions {
            scale,
            input_sizing,
            color_fix: true,
        }
    }

    #[test]
    fn release_plan_matches_the_upstream_docstring() {
        // "4x of a 512x512 input is a 2048x2048 output, 9 tiles; 512x1024 gives 21 tiles."
        let p = plan(
            Dims::new(512, 512),
            &RestorationOptions::default(),
            TileGeometry::RELEASE,
            MODEL_ID,
        )
        .unwrap();
        assert_eq!(p.output, Dims::new(2048, 2048));
        assert_eq!(p.forward_count(), 9);
        assert!(!p.enlarged);
        let p = plan(
            Dims::new(512, 1024),
            &RestorationOptions::default(),
            TileGeometry::RELEASE,
            MODEL_ID,
        )
        .unwrap();
        assert_eq!(p.output, Dims::new(2048, 4096));
        assert_eq!(p.forward_count(), 21);
        // "a 256x256 input is exactly one tile"
        let p = plan(
            Dims::new(256, 256),
            &RestorationOptions::default(),
            TileGeometry::RELEASE,
            MODEL_ID,
        )
        .unwrap();
        assert!(p.single_pass());
        assert!(p.enlarged);
        assert_eq!(p.processing, Dims::new(1024, 1024));
    }

    #[test]
    fn budget_downscales_and_never_upscales() {
        let b = INPUT_BUDGET;
        assert_eq!(fit_budget_dims(Dims::new(300, 200), b), Dims::new(300, 200));
        assert_eq!(
            fit_budget_dims(Dims::new(1100, 700), b),
            Dims::new(805, 512)
        );
        assert_eq!(
            fit_budget_dims(Dims::new(4000, 500), b),
            Dims::new(1024, 128)
        );
        let budgeted = plan(
            Dims::new(1100, 700),
            &opts(4.0, InputSizing::Budgeted),
            TileGeometry::RELEASE,
            MODEL_ID,
        )
        .unwrap();
        assert!(budgeted.budget_applied);
        assert_eq!(budgeted.output, Dims::new(3220, 2048));
        let original = plan(
            Dims::new(1100, 700),
            &opts(4.0, InputSizing::Original),
            TileGeometry::RELEASE,
            MODEL_ID,
        )
        .unwrap();
        assert!(!original.budget_applied);
        assert_eq!(original.output, Dims::new(4400, 2800));
    }

    #[test]
    fn every_positive_scale_has_exact_floor_geometry() {
        for (src, scale, out) in [
            ((640, 480), 1.0, (640, 480)),
            ((13, 9), 2.5, (32, 22)),
            ((333, 101), 3.0, (999, 303)),
            ((1000, 800), 0.5, (500, 400)),
        ] {
            let p = plan(
                Dims::new(src.0, src.1),
                &opts(scale, InputSizing::Original),
                TileGeometry::RELEASE,
                MODEL_ID,
            )
            .unwrap();
            assert_eq!(p.output, Dims::new(out.0, out.1), "{src:?} x{scale}");
        }
    }

    #[test]
    fn request_scales_are_the_decimal_the_caller_wrote() {
        // torch's `floor(side · scale_factor)` with the Python float the caller wrote: a widened
        // `1.3f32` (1.2999999523…) would give 1299×1039 and 6 here.
        for (src, scale, want_scale, out) in [
            ((1000, 800), 1.3f32, 1.3f64, (1300, 1040)),
            ((10, 10), 0.7, 0.7, (7, 7)),
            ((20, 10), 2.3, 2.3, (46, 23)),
        ] {
            let req = TransformRequest {
                target: TargetSize::Scale(scale),
                input_sizing: InputSizing::Original,
                ..Default::default()
            };
            let o = RestorationOptions::from_request(&req, MODEL_ID).unwrap();
            assert_eq!(o.scale, want_scale, "{scale}");
            let p = plan(Dims::new(src.0, src.1), &o, TileGeometry::RELEASE, MODEL_ID).unwrap();
            assert_eq!(p.output, Dims::new(out.0, out.1), "{src:?} x{scale}");
        }
    }

    #[test]
    fn oversized_processing_and_padding_are_refused() {
        // A 1×5,000,000 output enlarges its long side by tile/1 = 1024 → 5.12e9 px (> u32).
        let err = plan(
            Dims::new(1, 5_000_000),
            &opts(1.0, InputSizing::Original),
            TileGeometry::RELEASE,
            MODEL_ID,
        )
        .unwrap_err();
        assert!(
            matches!(&err, Error::Unsupported(m) if m.contains("enlarged processing size")),
            "{err:?}"
        );
        // An output that fits u32 but whose patch padding does not.
        let err = plan(
            Dims::new(u32::MAX - 3, 2000),
            &opts(1.0, InputSizing::Original),
            TileGeometry::RELEASE,
            MODEL_ID,
        )
        .unwrap_err();
        assert!(
            matches!(&err, Error::Unsupported(m) if m.contains("patch-padded processing size")),
            "{err:?}"
        );
    }

    #[test]
    fn non_positive_and_vanishing_scales_are_refused() {
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY, 0.001] {
            let err = plan(
                Dims::new(100, 100),
                &opts(scale, InputSizing::Original),
                TileGeometry::RELEASE,
                MODEL_ID,
            )
            .unwrap_err();
            assert!(matches!(err, Error::Unsupported(_)), "{scale}: {err:?}");
        }
    }

    #[test]
    fn tile_positions_end_flush() {
        assert_eq!(tile_positions(1024, 1024, 512), [0]);
        assert_eq!(tile_positions(1536, 1024, 512), [0, 512]);
        assert_eq!(tile_positions(2048, 1024, 512), [0, 512, 1024]);
        assert_eq!(tile_positions(1040, 1024, 512), [0, 16]);
    }

    #[test]
    fn request_fields_are_resolved_or_refused_by_name() {
        let o = RestorationOptions::from_request(&TransformRequest::default(), MODEL_ID).unwrap();
        assert_eq!(o, RestorationOptions::default());
        for (req, field) in [
            (
                TransformRequest {
                    seed: Some(1),
                    ..Default::default()
                },
                "seed",
            ),
            (
                TransformRequest {
                    strength: Some(0.5),
                    ..Default::default()
                },
                "strength",
            ),
            (
                TransformRequest {
                    steps: Some(4),
                    ..Default::default()
                },
                "steps",
            ),
            (
                TransformRequest {
                    target: TargetSize::MinEdge(1024),
                    ..Default::default()
                },
                "scale factor",
            ),
        ] {
            let err = RestorationOptions::from_request(&req, MODEL_ID).unwrap_err();
            assert!(
                matches!(&err, Error::Unsupported(m) if m.contains(field)),
                "{field}: {err:?}"
            );
        }
        let one = TransformRequest {
            target: TargetSize::Scale(1.0),
            input_sizing: InputSizing::Original,
            color_fix: Some(false),
            steps: Some(1),
            ..Default::default()
        };
        let o = RestorationOptions::from_request(&one, MODEL_ID).unwrap();
        assert_eq!(o.scale, 1.0);
        assert_eq!(o.input_sizing, InputSizing::Original);
        assert!(!o.color_fix);
    }

    #[test]
    fn wrong_task_exports_are_refused() {
        let export = |config: &str| {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("config.yaml"), config).unwrap();
            std::fs::write(dir.path().join("model.safetensors"), b"").unwrap();
            std::fs::write(dir.path().join(EMPTY_PROMPT_FILE), b"").unwrap();
            RestorationResources::from_dir(dir.path(), MODEL_ID)
        };
        let err = export("model:\n  block: single_stream\n").unwrap_err();
        assert!(
            matches!(&err, Error::Unsupported(m) if m.contains("generation")),
            "{err}"
        );
        let err = export("model:\n  block: single_stream\ntask:\n  name: depth\n").unwrap_err();
        assert!(
            matches!(&err, Error::Unsupported(m) if m.contains("'depth'")),
            "{err}"
        );
        let ok = "task:\n  name: restoration\n  sigma: 0.5\n  tile: 1024\n";
        let task = super::super::downstream::parse_task_settings(ok)
            .unwrap()
            .unwrap();
        let s = RestorationSettings::from_task(&task, MODEL_ID).unwrap();
        assert_eq!(export(ok).unwrap().settings, s);
        assert_eq!(
            s,
            RestorationSettings {
                sigma: 0.5,
                tile: 1024
            }
        );
        let cfg = IrisConfig::parse(ok).unwrap();
        s.validate(&cfg, MODEL_ID).unwrap();
        assert_eq!(s.model_time(&cfg), 500.0);
        let bad = RestorationSettings { tile: 1000, ..s };
        assert!(bad.validate(&cfg, MODEL_ID).is_err());
    }

    #[test]
    fn a_text_encoder_component_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        spec.components.insert(
            super::super::TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(dir.path().to_path_buf()),
        );
        let err = RestorationResources::from_spec(&spec, MODEL_ID).unwrap_err();
        assert!(
            matches!(&err, Error::Unsupported(m) if m.contains("text_encoder")),
            "{err}"
        );
    }

    #[test]
    fn gaussian_window_is_peaked_at_upstreams_centres() {
        let w = gaussian_window(8);
        // rows centre 4.0 (exact peak row 4), cols centre 3.5 (cols 3 and 4 tie)
        let at = |r: usize, c: usize| w[r * 8 + c];
        assert_eq!(at(4, 3), at(4, 4));
        assert!(at(4, 3) > at(3, 3) && at(4, 3) > at(5, 3));
        assert!((at(4, 3) - (-(0.25f32) / 64.0 / 0.02).exp()).abs() < 1e-7);
    }

    #[test]
    fn colour_fix_of_an_image_onto_itself_is_identity() {
        let mut p = Planes::zeros(3, 9, 11);
        for (i, v) in p.data.iter_mut().enumerate() {
            *v = ((i * 37) % 101) as f32 / 100.0;
        }
        let fixed = wavelet_color_fix(&p, &p).unwrap();
        for (a, b) in fixed.data.iter().zip(&p.data) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn cancellation_between_tiles_yields_no_image() {
        let p = plan(
            Dims::new(40, 24),
            &opts(4.0, InputSizing::Original),
            TileGeometry {
                tile: 64,
                patch: 16,
            },
            MODEL_ID,
        )
        .unwrap();
        assert_eq!(p.forward_count(), 8);
        let image = Image {
            width: 40,
            height: 24,
            pixels: vec![128; 40 * 24 * 3],
        };
        let cancel = CancelFlag::new();
        let mut steps = Vec::new();
        let mut calls = 0;
        let err = restore(
            &image,
            &p,
            0.5,
            &cancel.clone(),
            &mut |e| steps.push(e),
            &mut |tile, _, _| {
                calls += 1;
                if calls == 3 {
                    cancel.cancel();
                }
                Ok(vec![0.0; tile.len()])
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::Canceled), "{err:?}");
        assert_eq!(calls, 3);
        assert_eq!(steps.len(), 3);
    }

    #[test]
    fn zero_velocity_reproduces_the_bicubic_reference() {
        // v = 0 ⇒ every tile is its input; fusion is a normalized average of identical values, so
        // the output is the bicubic upsample (colour fix of an image onto its own source).
        let p = plan(
            Dims::new(40, 24),
            &opts(4.0, InputSizing::Original),
            TileGeometry {
                tile: 64,
                patch: 16,
            },
            MODEL_ID,
        )
        .unwrap();
        let mut pixels = Vec::new();
        for i in 0..40 * 24 * 3 {
            pixels.push(((i * 53) % 256) as u8);
        }
        let image = Image {
            width: 40,
            height: 24,
            pixels,
        };
        let mut progress = Vec::new();
        let out = restore(
            &image,
            &p,
            0.5,
            &CancelFlag::new(),
            &mut |e| progress.push(e),
            &mut |tile, _, _| Ok(vec![0.0; tile.len()]),
        )
        .unwrap();
        assert_eq!((out.width, out.height), (160, 96));
        let reference = bicubic_scale_factor(&Planes::from_rgb8(&image).unwrap(), 4.0).to_rgb8();
        let max = out
            .pixels
            .iter()
            .zip(&reference.pixels)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(max <= 1, "max diff {max}");
        assert_eq!(
            progress.last(),
            Some(&Progress::Decoding),
            "post-processing is reported after the tiles"
        );
        assert_eq!(
            progress
                .iter()
                .filter(|e| matches!(e, Progress::Step { .. }))
                .count(),
            8
        );
    }
}
