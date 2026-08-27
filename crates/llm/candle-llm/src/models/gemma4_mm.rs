//! Gemma 4 unified's multimodal front-ends: the **encoder-free** vision embedder and the audio
//! projector (sc-18772) — the Candle port of mlx-llm's module of the same name.
//!
//! Unlike every other vision path in this crate there is no ViT here. Gemma 4 ships five vision
//! tensors and one audio tensor, and the decoder does the rest:
//!
//! ```text
//! image  -> 48x48x3 patches -> patch_ln1 -> patch_dense -> patch_ln2
//!                                        -> + pos_embedding[row] + pos_embedding[col]
//!                                        -> pos_norm -> embed_vision   -> [N, hidden]
//! audio  -> 640-sample frames                          -> embed_audio  -> [M, hidden]
//! ```
//!
//! Both results are spliced into the token embeddings at `image_token_id` / `audio_token_id` rows,
//! each span framed by the checkpoint's begin/end-of-image (`boi`/`eoi`) and begin/end-of-audio
//! (`boa`/`eoa`) tokens.
//!
//! **Two shipped weight layouts.** `google/gemma-4-12B-it` nests everything under `model.`
//! (`model.vision_embedder.*`, `model.embed_vision.*`, `model.embed_audio.*`), while the LTX-2.5
//! packed text encoder repacks the same tensors flat (`vision_model.*`, `multi_modal_projector.*`,
//! `audio_projector.*`) with an identical `gemma_config`. The config cannot tell them apart — both
//! declare `Gemma4UnifiedForConditionalGeneration` — so [`Gemma4Layout`] discriminates on a
//! signature *tensor key* and fails closed when neither is present.

use candle_core::{DType, Device, Tensor};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::primitives::nn::{layer_norm, linear};
use crate::primitives::Weights;

/// Which of the two shipped Gemma 4 tensor layouts a checkpoint uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gemma4Layout {
    /// `google/gemma-4-12B-it`: `model.vision_embedder.*`, `model.embed_vision.*`,
    /// `model.embed_audio.*`, decoder under `model.language_model.*`.
    HfUnified,
    /// The LTX-2.5 packed text encoder: `vision_model.*`, `multi_modal_projector.*`,
    /// `audio_projector.*`, decoder flat under `model.*`.
    LtxPacked,
}

impl Gemma4Layout {
    /// Discriminate on a signature tensor key rather than the config: both layouts declare the same
    /// `architectures` / `model_type`, so only the checkpoint itself distinguishes them.
    ///
    /// The probe is the **decoder's** embedding table, not a vision tensor: a Gemma 4 checkpoint may
    /// legitimately ship without the vision/audio front-ends (a text-only re-export), but it can
    /// never ship without its token embeddings, so this stays decidable for every Gemma 4 snapshot.
    pub fn detect(w: &Weights) -> Result<Self> {
        if w.contains("model.language_model.embed_tokens.weight") {
            Ok(Self::HfUnified)
        } else if w.contains("model.embed_tokens.weight") {
            Ok(Self::LtxPacked)
        } else {
            Err(Error::MissingTensor(
                "Gemma 4: neither `model.language_model.embed_tokens.weight` (the HF unified \
                 layout) nor `model.embed_tokens.weight` (the LTX-2.5 packed layout) is present; \
                 cannot determine the checkpoint's tensor layout"
                    .to_string(),
            ))
        }
    }

    /// The decoder's weight-key root for this layout.
    pub fn decoder_root(self) -> &'static str {
        match self {
            Self::HfUnified => "model.language_model",
            Self::LtxPacked => "model",
        }
    }

    fn vision_prefix(self) -> &'static str {
        match self {
            Self::HfUnified => "model.vision_embedder",
            Self::LtxPacked => "vision_model",
        }
    }

    fn vision_proj_prefix(self) -> &'static str {
        match self {
            Self::HfUnified => "model.embed_vision",
            Self::LtxPacked => "multi_modal_projector",
        }
    }

    fn audio_proj_prefix(self) -> &'static str {
        match self {
            Self::HfUnified => "model.embed_audio",
            Self::LtxPacked => "audio_projector",
        }
    }
}

/// The multimodal token ids and front-end geometry read from a Gemma 4 unified `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4MmConfig {
    /// `image_token_id` — the soft-token id each image patch row replaces (258880 upstream).
    pub image_token_id: i32,
    /// `boi_token_id` — begin-of-image (255999 upstream).
    pub boi_token_id: i32,
    /// `eoi_token_id` — end-of-image (258882 upstream).
    pub eoi_token_id: i32,
    /// `audio_token_id` — the soft-token id each audio frame row replaces (258881 upstream).
    pub audio_token_id: i32,
    /// `boa_token_id` — begin-of-audio (256000 upstream).
    pub boa_token_id: i32,
    /// `eoa_token_index` — end-of-audio (258883 upstream). Upstream spells this one `_index`.
    pub eoa_token_id: i32,
    /// Vision geometry, when the config carries a `vision_config`.
    pub vision: Option<Gemma4VisionConfig>,
    /// Audio geometry, when the config carries an `audio_config`.
    pub audio: Option<Gemma4AudioConfig>,
}

/// The `vision_config` block's geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4VisionConfig {
    /// `model_patch_size` — the pixel side of one soft token's square patch (48 upstream, i.e.
    /// `patch_size * pooling_kernel_size` = 16 * 3).
    pub patch_pixels: usize,
    /// `mm_posemb_size` — entries per positional-embedding axis (1120 upstream).
    pub posemb_size: usize,
    /// `num_soft_tokens` — the per-image soft-token budget (280 upstream).
    pub max_soft_tokens: usize,
    /// `rms_norm_eps` — reused as the LayerNorm epsilon for the three vision norms.
    pub norm_eps: f64,
}

impl Gemma4VisionConfig {
    /// The flattened length of one patch: `patch_pixels² · 3` (RGB). 6912 upstream.
    pub fn patch_elems(&self) -> usize {
        self.patch_pixels * self.patch_pixels * 3
    }
}

/// The `audio_config` block's geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4AudioConfig {
    /// `audio_samples_per_token` — mono samples per audio soft token (640 upstream = 40 ms at
    /// 16 kHz), which is also the projector's input width.
    pub samples_per_token: usize,
    /// The sample rate the framing assumes (16 000 upstream). A clip at any other rate is rejected
    /// rather than silently resampled.
    pub sample_rate: u32,
    /// `audio_seq_length` — the per-clip audio soft-token budget (750 upstream).
    pub max_soft_tokens: usize,
}

/// Upstream's audio sample rate, used when the snapshot ships no `processor_config.json`.
const DEFAULT_AUDIO_SAMPLE_RATE: u32 = 16_000;
/// Upstream's `audio_seq_length` when the snapshot ships no `processor_config.json`.
const DEFAULT_AUDIO_MAX_SOFT_TOKENS: usize = 750;

impl Gemma4MmConfig {
    /// Parse the multimodal block out of a Gemma 4 unified `config.json` value (the **top level**,
    /// not the `text_config` descent — every field here lives beside `text_config`, not inside it).
    ///
    /// `processor` is the snapshot's `processor_config.json` when present; it carries the audio
    /// sample rate and sequence cap, which `config.json` does not.
    pub fn from_json(v: &Value, processor: Option<&Value>) -> Result<Self> {
        let req_id = |key: &str| -> Result<i32> {
            v.get(key)
                .and_then(|x| x.as_i64())
                .map(|x| x as i32)
                .ok_or_else(|| {
                    Error::Config(format!(
                        "Gemma 4 multimodal config: missing `{key}`; a checkpoint that declares a \
                         vision or audio front-end must name the token id it splices at"
                    ))
                })
        };

        let vision = match v.get("vision_config") {
            None => None,
            Some(vc) => {
                let usize_at = |key: &str| -> Option<usize> {
                    vc.get(key)
                        .and_then(|x| x.as_u64())
                        .and_then(|x| usize::try_from(x).ok())
                        .filter(|&x| x > 0)
                };
                let patch_pixels = match usize_at("model_patch_size") {
                    Some(p) => p,
                    None => match (usize_at("patch_size"), usize_at("pooling_kernel_size")) {
                        (Some(p), Some(k)) => p * k,
                        _ => {
                            return Err(Error::Config(
                                "Gemma 4 vision_config: needs `model_patch_size`, or both \
                                 `patch_size` and `pooling_kernel_size` to derive it"
                                    .to_string(),
                            ))
                        }
                    },
                };
                Some(Gemma4VisionConfig {
                    patch_pixels,
                    posemb_size: usize_at("mm_posemb_size").ok_or_else(|| {
                        Error::Config(
                            "Gemma 4 vision_config: missing `mm_posemb_size` (the positional \
                             embedding table's per-axis length)"
                                .to_string(),
                        )
                    })?,
                    max_soft_tokens: usize_at("num_soft_tokens").ok_or_else(|| {
                        Error::Config(
                            "Gemma 4 vision_config: missing `num_soft_tokens` (the per-image soft \
                             token budget)"
                                .to_string(),
                        )
                    })?,
                    norm_eps: vc
                        .get("rms_norm_eps")
                        .and_then(|x| x.as_f64())
                        .unwrap_or(1e-6),
                })
            }
        };

        let audio = match v.get("audio_config") {
            None => None,
            Some(ac) => {
                let fe = processor.and_then(|p| p.get("feature_extractor"));
                let samples_per_token = ac
                    .get("audio_samples_per_token")
                    .or_else(|| fe.and_then(|f| f.get("audio_samples_per_token")))
                    .or_else(|| ac.get("audio_embed_dim"))
                    .and_then(|x| x.as_u64())
                    .and_then(|x| usize::try_from(x).ok())
                    .filter(|&x| x > 0)
                    .ok_or_else(|| {
                        Error::Config(
                            "Gemma 4 audio_config: missing `audio_samples_per_token` (and no \
                             `audio_embed_dim` to fall back on); cannot frame a waveform"
                                .to_string(),
                        )
                    })?;
                Some(Gemma4AudioConfig {
                    samples_per_token,
                    sample_rate: fe
                        .and_then(|f| f.get("sampling_rate"))
                        .and_then(|x| x.as_u64())
                        .and_then(|x| u32::try_from(x).ok())
                        .filter(|&x| x > 0)
                        .unwrap_or(DEFAULT_AUDIO_SAMPLE_RATE),
                    max_soft_tokens: processor
                        .and_then(|p| p.get("audio_seq_length"))
                        .and_then(|x| x.as_u64())
                        .and_then(|x| usize::try_from(x).ok())
                        .filter(|&x| x > 0)
                        .unwrap_or(DEFAULT_AUDIO_MAX_SOFT_TOKENS),
                })
            }
        };

        Ok(Self {
            image_token_id: req_id("image_token_id")?,
            boi_token_id: req_id("boi_token_id")?,
            eoi_token_id: req_id("eoi_token_id")?,
            audio_token_id: req_id("audio_token_id")?,
            boa_token_id: req_id("boa_token_id")?,
            eoa_token_id: v
                .get("eoa_token_index")
                .or_else(|| v.get("eoa_token_id"))
                .and_then(|x| x.as_i64())
                .map(|x| x as i32)
                .ok_or_else(|| {
                    Error::Config(
                        "Gemma 4 multimodal config: missing `eoa_token_index` (end-of-audio)"
                            .to_string(),
                    )
                })?,
            vision,
            audio,
        })
    }
}

/// The encoder-free vision embedder: three LayerNorms, one dense projection, and a factorized 2-D
/// positional-embedding table.
pub struct Gemma4VisionEmbedder {
    patch_ln1_w: Tensor,
    patch_ln1_b: Tensor,
    patch_dense_w: Tensor,
    patch_dense_b: Tensor,
    patch_ln2_w: Tensor,
    patch_ln2_b: Tensor,
    pos_row: Tensor,
    pos_col: Tensor,
    pos_norm_w: Tensor,
    pos_norm_b: Tensor,
    proj_w: Tensor,
    cfg: Gemma4VisionConfig,
    dtype: DType,
    device: Device,
}

impl Gemma4VisionEmbedder {
    /// Whether `w` carries this layout's vision tensors at all.
    pub fn present(w: &Weights, layout: Gemma4Layout) -> bool {
        w.contains(&format!("{}.patch_dense.weight", layout.vision_prefix()))
            && w.contains(&format!(
                "{}.embedding_projection.weight",
                layout.vision_proj_prefix()
            ))
    }

    /// Load the tower in `dtype`. Every tensor is required: a partially-present vision embedder is a
    /// broken checkpoint, not a text-only one ([`present`](Self::present) answered that already).
    pub fn from_weights(
        w: &Weights,
        layout: Gemma4Layout,
        cfg: Gemma4VisionConfig,
        dtype: DType,
    ) -> Result<Self> {
        let vp = layout.vision_prefix();
        let get = |key: String| -> Result<Tensor> { Ok(w.require(&key)?.to_dtype(dtype)?) };

        let pos = get(format!("{vp}.pos_embedding"))?;
        let shape = pos.dims().to_vec();
        if shape.len() != 3 || shape[1] != 2 {
            return Err(Error::Config(format!(
                "Gemma 4 vision: `{vp}.pos_embedding` must be [posemb_size, 2, hidden] (a row axis \
                 and a column axis); got {shape:?}"
            )));
        }
        if shape[0] != cfg.posemb_size {
            return Err(Error::Config(format!(
                "Gemma 4 vision: `{vp}.pos_embedding` has {} positions but vision_config declares \
                 mm_posemb_size = {}",
                shape[0], cfg.posemb_size
            )));
        }
        let hidden = shape[2];
        // Split the two axes once, at load, so the per-image forward is two plain gathers.
        let pos_row = pos.i((.., 0, ..))?.contiguous()?;
        let pos_col = pos.i((.., 1, ..))?.contiguous()?;

        let patch_dense_w = get(format!("{vp}.patch_dense.weight"))?;
        let dense_shape = patch_dense_w.dims().to_vec();
        let expected_in = cfg.patch_elems();
        if dense_shape != [hidden, expected_in] {
            return Err(Error::Config(format!(
                "Gemma 4 vision: `{vp}.patch_dense.weight` is {dense_shape:?} but the config's \
                 {}x{} RGB patch flattens to {expected_in} and pos_embedding's hidden is {hidden}",
                cfg.patch_pixels, cfg.patch_pixels
            )));
        }

        Ok(Self {
            patch_ln1_w: get(format!("{vp}.patch_ln1.weight"))?,
            patch_ln1_b: get(format!("{vp}.patch_ln1.bias"))?,
            patch_dense_w,
            patch_dense_b: get(format!("{vp}.patch_dense.bias"))?,
            patch_ln2_w: get(format!("{vp}.patch_ln2.weight"))?,
            patch_ln2_b: get(format!("{vp}.patch_ln2.bias"))?,
            pos_row,
            pos_col,
            pos_norm_w: get(format!("{vp}.pos_norm.weight"))?,
            pos_norm_b: get(format!("{vp}.pos_norm.bias"))?,
            proj_w: get(format!(
                "{}.embedding_projection.weight",
                layout.vision_proj_prefix()
            ))?,
            cfg,
            dtype,
            device: w.device().clone(),
        })
    }

    /// The vision geometry this tower was built for.
    pub fn config(&self) -> &Gemma4VisionConfig {
        &self.cfg
    }

    /// Embed one image's patches. `patches` is `[N, patch_elems]` in row-major grid order (see
    /// [`patchify`]); `grid` is that image's `(rows, cols)`, whose product must be `N`.
    pub fn forward(&self, patches: &Tensor, grid: (usize, usize)) -> Result<Tensor> {
        let (gh, gw) = grid;
        let n = patches.dims()[0];
        if gh * gw != n {
            return Err(Error::Config(format!(
                "Gemma 4 vision: {n} patches but the grid is {gh}x{gw} = {}",
                gh * gw
            )));
        }
        let eps = self.cfg.norm_eps;
        let x = layer_norm(patches, &self.patch_ln1_w, &self.patch_ln1_b, eps)?;
        let x = linear(&x, &self.patch_dense_w, Some(&self.patch_dense_b))?;
        let x = layer_norm(&x, &self.patch_ln2_w, &self.patch_ln2_b, eps)?;

        let (row_idx, col_idx) = posemb_indices(gh, gw, self.cfg.posemb_size);
        let rows = Tensor::from_vec(row_idx, n, &self.device)?;
        let cols = Tensor::from_vec(col_idx, n, &self.device)?;
        let pos = self
            .pos_row
            .index_select(&rows, 0)?
            .add(&self.pos_col.index_select(&cols, 0)?)?;
        let x = x.add(&pos.to_dtype(self.dtype)?)?;
        let x = layer_norm(&x, &self.pos_norm_w, &self.pos_norm_b, eps)?;
        linear(&x, &self.proj_w, None)
    }
}

/// The audio projector: one dense map from a raw 640-sample frame to the decoder's hidden width.
pub struct Gemma4AudioEmbedder {
    proj_w: Tensor,
    cfg: Gemma4AudioConfig,
}

impl Gemma4AudioEmbedder {
    /// Whether `w` carries this layout's audio projection.
    pub fn present(w: &Weights, layout: Gemma4Layout) -> bool {
        w.contains(&format!(
            "{}.embedding_projection.weight",
            layout.audio_proj_prefix()
        ))
    }

    /// Load the projector, checking its input width against the config's frame size.
    pub fn from_weights(
        w: &Weights,
        layout: Gemma4Layout,
        cfg: Gemma4AudioConfig,
        dtype: DType,
    ) -> Result<Self> {
        let key = format!(
            "{}.embedding_projection.weight",
            layout.audio_proj_prefix()
        );
        let proj_w = w.require(&key)?.to_dtype(dtype)?;
        let shape = proj_w.dims().to_vec();
        if shape.len() != 2 || shape[1] != cfg.samples_per_token {
            return Err(Error::Config(format!(
                "Gemma 4 audio: `{key}` is {shape:?} but the config frames audio at {} samples per \
                 token, so the projection must take that many inputs",
                cfg.samples_per_token
            )));
        }
        Ok(Self { proj_w, cfg })
    }

    /// The audio geometry this projector was built for.
    pub fn config(&self) -> &Gemma4AudioConfig {
        &self.cfg
    }

    /// Embed framed audio. `frames` is `[M, samples_per_token]`; returns `[M, hidden]`.
    pub fn forward(&self, frames: &Tensor) -> Result<Tensor> {
        let width = frames.dims()[1];
        if width != self.cfg.samples_per_token {
            return Err(Error::Config(format!(
                "Gemma 4 audio: frames are {width} samples wide, expected {}",
                self.cfg.samples_per_token
            )));
        }
        linear(frames, &self.proj_w, None)
    }
}

/// Both front-ends plus the token ids that place their output, as loaded from one checkpoint.
pub struct Gemma4Mm {
    /// The multimodal token ids and geometry.
    pub cfg: Gemma4MmConfig,
    /// The vision embedder, present iff the checkpoint ships one.
    pub vision: Option<Gemma4VisionEmbedder>,
    /// The audio projector, present iff the checkpoint ships one.
    pub audio: Option<Gemma4AudioEmbedder>,
}

impl Gemma4Mm {
    /// Load whichever front-ends the checkpoint carries. A `vision_config` without vision tensors
    /// (or the reverse) yields `None` for that modality — the provider then declares it unsupported
    /// rather than advertising a capability the weights cannot serve.
    pub fn from_weights(
        w: &Weights,
        layout: Gemma4Layout,
        cfg: Gemma4MmConfig,
        dtype: DType,
    ) -> Result<Self> {
        let vision = match (&cfg.vision, Gemma4VisionEmbedder::present(w, layout)) {
            (Some(vc), true) => Some(Gemma4VisionEmbedder::from_weights(
                w,
                layout,
                vc.clone(),
                dtype,
            )?),
            _ => None,
        };
        let audio = match (&cfg.audio, Gemma4AudioEmbedder::present(w, layout)) {
            (Some(ac), true) => Some(Gemma4AudioEmbedder::from_weights(
                w,
                layout,
                ac.clone(),
                dtype,
            )?),
            _ => None,
        };
        Ok(Self { cfg, vision, audio })
    }
}

// --- pure host geometry (no tensors; unit-testable on any device) -------------------------------
//
// These mirror `mlx_llm::models::gemma4_mm`'s functions of the same names exactly — the accepted
// leaf duplication that keeps the two engine crates independent (see `primitives/mod.rs`). Each is
// covered by the same assertions on both sides, so a divergence shows up as a test failure rather
// than as a silent backend disagreement.

use candle_core::IndexOp;

/// Resample the `posemb_size`-entry positional table onto a `gh x gw` patch grid, returning the row
/// and column table indices of each patch in row-major order.
///
/// The table is far longer than any real grid side (1120 vs at most 280), so a grid samples it
/// rather than indexing it directly: patch row `r` of `gh` reads table entry
/// `floor(r * posemb_size / gh)`, which spreads the grid evenly across the table and is exact when
/// `gh` divides `posemb_size`.
pub fn posemb_indices(gh: usize, gw: usize, posemb_size: usize) -> (Vec<u32>, Vec<u32>) {
    let last = posemb_size.saturating_sub(1);
    let mut rows = Vec::with_capacity(gh * gw);
    let mut cols = Vec::with_capacity(gh * gw);
    for r in 0..gh {
        let ri = ((r * posemb_size) / gh.max(1)).min(last) as u32;
        for c in 0..gw {
            let ci = ((c * posemb_size) / gw.max(1)).min(last) as u32;
            rows.push(ri);
            cols.push(ci);
        }
    }
    (rows, cols)
}

/// Choose the `(rows, cols)` patch grid for a `width x height` image: the aspect-preserving grid
/// with the most patches that still fits `max_soft_tokens`. Both dimensions are at least 1.
pub fn soft_token_grid(width: usize, height: usize, max_soft_tokens: usize) -> (usize, usize) {
    let budget = max_soft_tokens.max(1);
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let aspect = w / h;
    // Aspect fidelity first, token count second — NOT the other way round. Maximizing tokens alone
    // picks a grid one row taller than the true ratio whenever that squeezes more patches under the
    // budget (a 1:2 image would take 23x12 = 276 over the exact 22x11 = 242), and since the image is
    // then resized onto that grid, the stretch is baked into what the model sees and into where the
    // resampled positional table places every patch. A systematic distortion on every portrait image
    // is a worse trade than 34 fewer soft tokens.
    const EXACT: f64 = 1e-9;
    let mut best = (1usize, 1usize);
    let mut best_score = (f64::INFINITY, 0usize);
    for gh in 1..=budget {
        let gw = ((gh as f64) * aspect).round().max(1.0) as usize;
        let count = gh * gw;
        if count > budget {
            continue;
        }
        let distortion = ((gw as f64 / gh as f64) / aspect).ln().abs();
        let better = distortion < best_score.0 - EXACT
            || ((distortion - best_score.0).abs() <= EXACT && count > best_score.1);
        if better {
            best = (gh, gw);
            best_score = (distortion, count);
        }
    }
    best
}

/// Split an RGB image of exactly `gh*patch x gw*patch` pixels into row-major patches, each flattened
/// in `(row, col, channel)` order — the layout `patch_dense`'s `[hidden, patch²·3]` weight expects.
///
/// `pixels` is `width * height * 3` interleaved RGB samples **on the 0..255 scale** — exactly what
/// [`resize_bicubic_u8`](crate::image::resize_bicubic_u8) returns. The rescale happens here, once,
/// as `sample / 255`: the shipped processor sets `do_rescale: true` with `rescale_factor: 1/255`
/// and `do_normalize: false` (mean 0, std 1), so there is no mean/std step to apply afterwards.
pub fn patchify(pixels: &[f32], width: usize, height: usize, patch: usize) -> Result<Vec<f32>> {
    if patch == 0 {
        return Err(Error::Config("Gemma 4 vision: patch size 0".to_string()));
    }
    if width % patch != 0 || height % patch != 0 {
        return Err(Error::Config(format!(
            "Gemma 4 vision: {width}x{height} is not a whole number of {patch}x{patch} patches; \
             the caller must resize to a patch multiple first"
        )));
    }
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(Error::Config(format!(
            "Gemma 4 vision: {width}x{height} RGB needs {expected} bytes, got {}",
            pixels.len()
        )));
    }
    let (gh, gw) = (height / patch, width / patch);
    let mut out = vec![0f32; gh * gw * patch * patch * 3];
    let patch_elems = patch * patch * 3;
    for pr in 0..gh {
        for pc in 0..gw {
            let base = (pr * gw + pc) * patch_elems;
            for y in 0..patch {
                let src_row = (pr * patch + y) * width * 3 + pc * patch * 3;
                let dst_row = base + y * patch * 3;
                for i in 0..patch * 3 {
                    out[dst_row + i] = pixels[src_row + i] / 255.0;
                }
            }
        }
    }
    Ok(out)
}

/// Frame a mono waveform into `[M, samples_per_token]` rows, zero-padding the final partial frame
/// and truncating at `max_soft_tokens`.
pub fn audio_frames(
    samples: &[f32],
    samples_per_token: usize,
    max_soft_tokens: usize,
) -> Result<Vec<f32>> {
    if samples_per_token == 0 {
        return Err(Error::Config(
            "Gemma 4 audio: samples_per_token 0".to_string(),
        ));
    }
    if samples.is_empty() {
        return Err(Error::Config(
            "Gemma 4 audio: an empty clip carries no conditioning".to_string(),
        ));
    }
    let frames = samples
        .len()
        .div_ceil(samples_per_token)
        .min(max_soft_tokens.max(1));
    let mut out = vec![0f32; frames * samples_per_token];
    let copy = (frames * samples_per_token).min(samples.len());
    out[..copy].copy_from_slice(&samples[..copy]);
    Ok(out)
}

/// The number of frames [`audio_frames`] will emit for a clip.
pub fn audio_frame_count(
    sample_count: usize,
    samples_per_token: usize,
    max_soft_tokens: usize,
) -> usize {
    if samples_per_token == 0 || sample_count == 0 {
        return 0;
    }
    sample_count
        .div_ceil(samples_per_token)
        .min(max_soft_tokens.max(1))
}

/// Replace each occurrence of `marker` in `ids` with `begin`, `counts[i]` copies of `marker`, then
/// `end` — Gemma 4's framed soft-token span.
pub fn expand_framed_placeholders(
    ids: &[i32],
    marker: i32,
    begin: i32,
    end: i32,
    counts: &[usize],
) -> Result<Vec<i32>> {
    let seen = ids.iter().filter(|&&id| id == marker).count();
    if seen != counts.len() {
        return Err(Error::Config(format!(
            "Gemma 4: {seen} `{marker}` placeholder(s) in the rendered prompt but {} count(s) \
             supplied",
            counts.len()
        )));
    }
    let extra: usize = counts.iter().map(|c| c + 2).sum();
    let mut out = Vec::with_capacity(ids.len() - seen + extra);
    let mut next = 0usize;
    for &id in ids {
        if id == marker {
            out.push(begin);
            out.extend(std::iter::repeat_n(marker, counts[next]));
            out.push(end);
            next += 1;
        } else {
            out.push(id);
        }
    }
    Ok(out)
}

/// Lift host patch rows into an `[N, patch_elems]` tensor in `dtype`.
pub fn patch_tensor(
    flat: &[f32],
    patch_elems: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let n = flat.len() / patch_elems.max(1);
    Ok(Tensor::from_slice(flat, (n, patch_elems), device)?.to_dtype(dtype)?)
}

/// Lift host audio frames into an `[M, samples_per_token]` tensor in `dtype`.
pub fn frame_tensor(
    flat: &[f32],
    samples_per_token: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let m = flat.len() / samples_per_token.max(1);
    Ok(Tensor::from_slice(flat, (m, samples_per_token), device)?.to_dtype(dtype)?)
}

/// Concatenate per-visual feature blocks into the single `[total, hidden]` buffer the splice
/// consumes, preserving document order.
pub fn concat_features(blocks: &[Tensor]) -> Result<Tensor> {
    match blocks {
        [] => Err(Error::Config(
            "Gemma 4: no multimodal features to splice".to_string(),
        )),
        [one] => Ok(one.clone()),
        many => Ok(Tensor::cat(many, 0)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn unified_config() -> Value {
        json!({
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
            "boi_token_id": 255999,
            "eoi_token_id": 258882,
            "image_token_id": 258880,
            "boa_token_id": 256000,
            "eoa_token_index": 258883,
            "audio_token_id": 258881,
            "text_config": { "model_type": "gemma4_unified_text" },
            "vision_config": {
                "model_type": "gemma4_unified_vision",
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "model_patch_size": 48,
                "mm_posemb_size": 1120,
                "num_soft_tokens": 280,
                "mm_embed_dim": 3840,
                "output_proj_dims": 3840,
                "rms_norm_eps": 1e-6
            },
            "audio_config": {
                "model_type": "gemma4_unified_audio",
                "audio_embed_dim": 640,
                "audio_samples_per_token": 640,
                "rms_norm_eps": 1e-6
            }
        })
    }

    fn processor_config() -> Value {
        json!({
            "audio_seq_length": 750,
            "image_seq_length": 280,
            "feature_extractor": {
                "audio_samples_per_token": 640,
                "sampling_rate": 16000
            }
        })
    }

    #[test]
    fn parses_the_shipped_token_ids_and_geometry() {
        let cfg = Gemma4MmConfig::from_json(&unified_config(), Some(&processor_config())).unwrap();
        assert_eq!(cfg.image_token_id, 258880);
        assert_eq!(cfg.boi_token_id, 255999);
        assert_eq!(cfg.eoi_token_id, 258882);
        assert_eq!(cfg.audio_token_id, 258881);
        assert_eq!(cfg.boa_token_id, 256000);
        assert_eq!(cfg.eoa_token_id, 258883);

        let v = cfg.vision.expect("a vision_config");
        assert_eq!(v.patch_pixels, 48);
        assert_eq!(v.patch_elems(), 6912, "48x48 RGB flattens to 6912");
        assert_eq!(v.posemb_size, 1120);
        assert_eq!(v.max_soft_tokens, 280);

        let a = cfg.audio.expect("an audio_config");
        assert_eq!(a.samples_per_token, 640);
        assert_eq!(a.sample_rate, 16_000);
        assert_eq!(a.max_soft_tokens, 750);
    }

    #[test]
    fn derives_the_patch_side_from_its_factors_when_unstated() {
        let mut v = unified_config();
        v["vision_config"]
            .as_object_mut()
            .unwrap()
            .remove("model_patch_size");
        let cfg = Gemma4MmConfig::from_json(&v, None).unwrap();
        assert_eq!(cfg.vision.unwrap().patch_pixels, 48);
    }

    #[test]
    fn a_config_missing_a_token_id_is_refused_not_defaulted() {
        let mut v = unified_config();
        v.as_object_mut().unwrap().remove("image_token_id");
        let err = Gemma4MmConfig::from_json(&v, None).expect_err("must refuse");
        assert!(
            format!("{err}").contains("image_token_id"),
            "the error must name the missing key: {err}"
        );
    }

    #[test]
    fn text_only_config_parses_with_no_front_ends() {
        let mut v = unified_config();
        v.as_object_mut().unwrap().remove("vision_config");
        v.as_object_mut().unwrap().remove("audio_config");
        let cfg = Gemma4MmConfig::from_json(&v, None).unwrap();
        assert!(cfg.vision.is_none());
        assert!(cfg.audio.is_none());
    }

    #[test]
    fn audio_sample_rate_falls_back_to_upstreams_when_no_processor_config() {
        let cfg = Gemma4MmConfig::from_json(&unified_config(), None).unwrap();
        let a = cfg.audio.unwrap();
        assert_eq!(a.sample_rate, 16_000);
        assert_eq!(a.max_soft_tokens, 750);
    }

    #[test]
    fn soft_token_grid_fits_the_budget_and_tracks_aspect() {
        // A square image gets a square grid within budget: 16x16 = 256 <= 280, and 17x17 = 289 > 280.
        assert_eq!(soft_token_grid(480, 480, 280), (16, 16));
        // A 2:1 landscape image gets an EXACTLY 2:1 grid, the largest that fits: 11x22 = 242
        // (12x24 = 288 overruns). The nearby 12x23 = 276 packs in more patches but is not 2:1 —
        // preferring it would stretch the image, which is the trade this rule refuses.
        assert_eq!(soft_token_grid(960, 480, 280), (11, 22));
        // Portrait mirrors it exactly.
        assert_eq!(soft_token_grid(480, 960, 280), (22, 11));
        // A 5:3 image likewise lands on an exact ratio: 12x20 = 240.
        assert_eq!(soft_token_grid(500, 300, 280), (12, 20));
        // Every grid stays within the budget and is never empty.
        for (w, h) in [(960, 480), (480, 960), (500, 300), (1, 999), (999, 1), (37, 53)] {
            let (gh, gw) = soft_token_grid(w, h, 280);
            assert!(gh * gw <= 280, "{w}x{h} -> grid {gh}x{gw} exceeds the budget");
            assert!(gh >= 1 && gw >= 1, "{w}x{h} -> empty grid {gh}x{gw}");
        }
        // Smaller than one patch still yields a real span.
        assert_eq!(soft_token_grid(10, 10, 280), (16, 16));
        // A tiny budget cannot produce an empty grid.
        assert_eq!(soft_token_grid(100, 100, 1), (1, 1));
    }

    #[test]
    fn posemb_indices_spread_the_table_across_the_grid() {
        let (rows, cols) = posemb_indices(2, 2, 1120);
        assert_eq!(rows, vec![0, 0, 560, 560]);
        assert_eq!(cols, vec![0, 560, 0, 560]);
        let (rows, cols) = posemb_indices(2, 3, 1120);
        assert_eq!(rows, vec![0, 0, 0, 560, 560, 560]);
        assert_eq!(cols, vec![0, 373, 746, 0, 373, 746]);
        let (rows, cols) = posemb_indices(1, 1, 1120);
        assert_eq!((rows, cols), (vec![0], vec![0]));
        let (rows, _) = posemb_indices(1120, 1, 1120);
        assert_eq!(rows.len(), 1120);
        assert!(rows.iter().all(|&i| i < 1120));
        assert_eq!(rows[1119], 1119);
    }

    #[test]
    fn patchify_reads_each_patch_in_row_col_channel_order() {
        let (w, h, p) = (4usize, 2usize, 2usize);
        let mut pixels = vec![0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    pixels[(y * w + x) * 3 + c] = (y * 40 + x * 10 + c) as f32;
                }
            }
        }
        let out = patchify(&pixels, w, h, p).unwrap();
        assert_eq!(out.len(), 2 * p * p * 3);
        // Patch 0 covers x in 0..2; its first row is pixels (0,0) and (0,1) => 0,1,2, 10,11,12.
        let expect0: Vec<f32> = [0, 1, 2, 10, 11, 12]
            .iter()
            .map(|v| *v as f32 / 255.0)
            .collect();
        assert_eq!(&out[..6], &expect0[..]);
        // Patch 1 covers x in 2..4 — so the SECOND patch must read the right-hand columns, which is
        // what fails if the patch walk is column-major or the row stride is wrong.
        let expect1: Vec<f32> = [20, 21, 22, 30, 31, 32]
            .iter()
            .map(|v| *v as f32 / 255.0)
            .collect();
        assert_eq!(&out[p * p * 3..p * p * 3 + 6], &expect1[..]);
        // Rescale is 1/255 with no mean/std step, so a full-scale sample maps to exactly 1.0.
        assert!(patchify(&[255f32; 12], 2, 2, 2)
            .unwrap()
            .iter()
            .all(|v| *v == 1.0));
    }

    #[test]
    fn patchify_refuses_a_non_patch_multiple() {
        let err = patchify(&vec![0f32; 5 * 2 * 3], 5, 2, 2).expect_err("must refuse");
        assert!(format!("{err}").contains("whole number"), "{err}");
        assert!(patchify(&vec![0f32; 10], 2, 2, 2).is_err());
    }

    #[test]
    fn audio_frames_pad_the_tail_and_cap_the_span() {
        let samples: Vec<f32> = (0..700).map(|i| i as f32).collect();
        let out = audio_frames(&samples, 640, 750).unwrap();
        assert_eq!(out.len(), 2 * 640);
        assert_eq!(out[699], 699.0);
        assert!(out[700..].iter().all(|v| *v == 0.0), "tail must be zeros");
        assert_eq!(audio_frame_count(700, 640, 750), 2);
        assert_eq!(audio_frames(&[0.5], 640, 750).unwrap().len(), 640);
        assert_eq!(audio_frame_count(1, 640, 750), 1);
        let long: Vec<f32> = vec![1.0; 640 * 10];
        assert_eq!(audio_frames(&long, 640, 3).unwrap().len(), 3 * 640);
        assert_eq!(audio_frame_count(640 * 10, 640, 3), 3);
        assert!(audio_frames(&[], 640, 750).is_err());
    }

    #[test]
    fn expand_framed_placeholders_wraps_each_span() {
        let ids = vec![7, 258880, 9];
        let out = expand_framed_placeholders(&ids, 258880, 255999, 258882, &[3]).unwrap();
        assert_eq!(out, vec![7, 255999, 258880, 258880, 258880, 258882, 9]);
        let ids = vec![258880, 5, 258880];
        let out = expand_framed_placeholders(&ids, 258880, 255999, 258882, &[1, 2]).unwrap();
        assert_eq!(
            out,
            vec![255999, 258880, 258882, 5, 255999, 258880, 258880, 258882]
        );
        assert_eq!(
            expand_framed_placeholders(&[1, 2, 3], 258880, 255999, 258882, &[]).unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn expand_framed_placeholders_refuses_a_count_mismatch() {
        let err = expand_framed_placeholders(&[258880, 258880], 258880, 255999, 258882, &[3])
            .expect_err("must refuse");
        assert!(
            format!("{err}").contains("2 `258880` placeholder(s)"),
            "{err}"
        );
        assert!(expand_framed_placeholders(&[258880], 258880, 255999, 258882, &[1, 1]).is_err());
    }

    /// The encoder-free tower on synthetic weights: proves the wiring end-to-end on CPU (LayerNorms,
    /// the dense projection, both positional axes, and the output projection) without a real
    /// checkpoint, and proves the *spatial* claim the positional table exists for.
    #[test]
    fn vision_embedder_forward_is_position_sensitive() {
        use std::collections::HashMap;
        let dev = Device::Cpu;
        let (patch, hidden, posemb) = (2usize, 4usize, 8usize);
        let elems = patch * patch * 3; // 12
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let ones = |n: usize| Tensor::ones(n, DType::F32, &dev).unwrap();
        let zeros = |n: usize| Tensor::zeros(n, DType::F32, &dev).unwrap();
        t.insert("model.embed_tokens.weight".into(), zeros(1));
        t.insert("vision_model.patch_ln1.weight".into(), ones(elems));
        t.insert("vision_model.patch_ln1.bias".into(), zeros(elems));
        // A deterministic, non-degenerate dense map.
        let dense: Vec<f32> = (0..hidden * elems).map(|i| (i % 7) as f32 * 0.1).collect();
        t.insert(
            "vision_model.patch_dense.weight".into(),
            Tensor::from_vec(dense, (hidden, elems), &dev).unwrap(),
        );
        t.insert("vision_model.patch_dense.bias".into(), zeros(hidden));
        t.insert("vision_model.patch_ln2.weight".into(), ones(hidden));
        t.insert("vision_model.patch_ln2.bias".into(), zeros(hidden));
        // Distinct row/col position vectors so a swapped or dropped axis changes the output.
        //
        // The values must vary WITHIN each position vector, not just between positions: `pos_norm`
        // is a LayerNorm, which subtracts the per-row mean, so two positions differing by a constant
        // offset (what a plain `i * 0.01` ramp produces) are indistinguishable downstream and this
        // test would pass on a broken gather. A non-monotone pattern gives each position its own
        // shape, not just its own level.
        let pos: Vec<f32> = (0..posemb * 2 * hidden)
            .map(|i| (((i * 37) % 23) as f32) * 0.05)
            .collect();
        t.insert(
            "vision_model.pos_embedding".into(),
            Tensor::from_vec(pos, (posemb, 2, hidden), &dev).unwrap(),
        );
        t.insert("vision_model.pos_norm.weight".into(), ones(hidden));
        t.insert("vision_model.pos_norm.bias".into(), zeros(hidden));
        t.insert(
            "multi_modal_projector.embedding_projection.weight".into(),
            Tensor::from_vec(
                (0..hidden * hidden).map(|i| (i % 5) as f32 * 0.2).collect::<Vec<f32>>(),
                (hidden, hidden),
                &dev,
            )
            .unwrap(),
        );
        let w = Weights::from_map(t, dev.clone());
        assert_eq!(Gemma4Layout::detect(&w).unwrap(), Gemma4Layout::LtxPacked);

        let cfg = Gemma4VisionConfig {
            patch_pixels: patch,
            posemb_size: posemb,
            max_soft_tokens: 280,
            norm_eps: 1e-6,
        };
        let tower =
            Gemma4VisionEmbedder::from_weights(&w, Gemma4Layout::LtxPacked, cfg, DType::F32)
                .unwrap();

        // Two identical patches in a 1x2 grid: the *content* is the same, so any difference in the
        // output rows can only have come from the positional embedding.
        let one = vec![0.5f32; elems];
        let mut both = one.clone();
        both.extend_from_slice(&one);
        let patches = patch_tensor(&both, elems, &dev, DType::F32).unwrap();
        let out = tower.forward(&patches, (1, 2)).unwrap();
        assert_eq!(out.dims(), &[2, hidden]);
        let rows: Vec<Vec<f32>> = out.to_vec2().unwrap();
        assert!(
            rows[0].iter().zip(&rows[1]).any(|(a, b)| (a - b).abs() > 1e-5),
            "identical patches at different columns must not embed identically — the positional \
             table is what distinguishes them"
        );
        assert!(
            rows.iter().flatten().all(|v| v.is_finite()),
            "vision embedding must be finite"
        );

        // The same two patches as a 2x1 grid walk the ROW axis instead of the column axis, so the
        // output must differ from the 1x2 case. This is the assertion that fails if the two
        // positional axes are swapped or if one is dropped.
        let transposed = tower.forward(&patches, (2, 1)).unwrap();
        let trows: Vec<Vec<f32>> = transposed.to_vec2().unwrap();
        assert!(
            trows[1]
                .iter()
                .zip(&rows[1])
                .any(|(a, b)| (a - b).abs() > 1e-5),
            "a 2x1 grid must not embed like a 1x2 grid — row and column axes are distinct"
        );

        // A grid whose product disagrees with the patch count is refused, not silently reshaped.
        assert!(tower.forward(&patches, (1, 3)).is_err());
    }

    /// The audio projector on synthetic weights: shape, finiteness, and that different frames map to
    /// different embeddings (a projector wired to a constant would pass a shape-only check).
    #[test]
    fn audio_embedder_projects_each_frame() {
        use std::collections::HashMap;
        let dev = Device::Cpu;
        let (spt, hidden) = (4usize, 3usize);
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "model.embed_tokens.weight".into(),
            Tensor::zeros(1, DType::F32, &dev).unwrap(),
        );
        t.insert(
            "audio_projector.embedding_projection.weight".into(),
            Tensor::from_vec(
                (0..hidden * spt).map(|i| (i + 1) as f32 * 0.25).collect::<Vec<f32>>(),
                (hidden, spt),
                &dev,
            )
            .unwrap(),
        );
        let w = Weights::from_map(t, dev.clone());
        let cfg = Gemma4AudioConfig {
            samples_per_token: spt,
            sample_rate: 16_000,
            max_soft_tokens: 750,
        };
        let proj =
            Gemma4AudioEmbedder::from_weights(&w, Gemma4Layout::LtxPacked, cfg, DType::F32).unwrap();

        let framed = audio_frames(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], spt, 750).unwrap();
        assert_eq!(framed.len(), 2 * spt, "6 samples at 4/frame => 2 frames");
        let frames = frame_tensor(&framed, spt, &dev, DType::F32).unwrap();
        let out = proj.forward(&frames).unwrap();
        assert_eq!(out.dims(), &[2, hidden]);
        let rows: Vec<Vec<f32>> = out.to_vec2().unwrap();
        assert!(rows.iter().flatten().all(|v| v.is_finite()));
        assert!(
            rows[0].iter().zip(&rows[1]).any(|(a, b)| (a - b).abs() > 1e-6),
            "distinct waveforms must produce distinct audio embeddings"
        );

        // A frame width the projector was not built for is refused rather than broadcast.
        let wrong = Tensor::zeros((1, spt + 1), DType::F32, &dev).unwrap();
        assert!(proj.forward(&wrong).is_err());
    }

    /// A checkpoint whose `audio_config` frames at a width the projection does not accept must fail
    /// the load, not produce silently mis-shaped conditioning.
    #[test]
    fn audio_projector_width_must_match_the_config() {
        use std::collections::HashMap;
        let dev = Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "model.embed_tokens.weight".into(),
            Tensor::zeros(1, DType::F32, &dev).unwrap(),
        );
        t.insert(
            "audio_projector.embedding_projection.weight".into(),
            Tensor::zeros((3, 4), DType::F32, &dev).unwrap(),
        );
        let w = Weights::from_map(t, dev.clone());
        let cfg = Gemma4AudioConfig {
            samples_per_token: 640, // the projection takes 4
            sample_rate: 16_000,
            max_soft_tokens: 750,
        };
        let err = Gemma4AudioEmbedder::from_weights(&w, Gemma4Layout::LtxPacked, cfg, DType::F32)
            .map(|_| ())
            .expect_err("a width mismatch must fail the load");
        assert!(format!("{err}").contains("640"), "{err}");
    }
}
