//! LTX-2.5 **DiffVAE** video decoder — `NADiffusionDecoder` on MLX (sc-18766).
//!
//! `vae/ltx-2.5-video-vae-bf16.safetensors` declares `_class_name: CausalDiffusionVAE`: the same
//! conv `Encoder` the conv VAE uses (see [`crate::vae::LtxVideoVae::encoder_only`]) paired with a
//! decoder that is not a conv stack at all. This module is that decoder.
//!
//! ## Shape of the thing
//!
//! Five stages, all **channels-last** `(B, T, H, W, C)` — the reference's own layout, and the one
//! MLX wants:
//!
//! | | stage 1 | stage 2 | stage 3 | stage 4 | stage 5 |
//! | --- | --- | --- | --- | --- | --- |
//! | channels | 2048 | 1024 | 512 | 512 | 256 |
//! | blocks | 4 | 6 | 4 | 2 | 8 |
//! | 3-D window | 3x7x7 | 3x7x7 | 3x5x5 | 3x5x5 | 11x11x11 |
//!
//! Stages 1-4 are **deterministic**: pre-norm blocks of 3-D neighborhood attention + SwiGLU, each
//! followed by a `Linear` + channels-last pixel-shuffle upsample (strides `1x2x2`, `2x1x1`,
//! `2x2x2`, `2x2x2`). Their output is the *context* volume, at pixel resolution divided by the
//! patch size. Stage 5 is the **diffusion** stage: eight blocks that denoise patchified noisy
//! pixels `x_t`, injecting the context through a per-block `context_proj` and modulating on a
//! shared AdaLN-Zero built from the timestep. `model_output_type: x0` with
//! `default_num_inference_steps: 1` means the shipped checkpoint runs exactly one step and returns
//! its prediction — but the Euler loop is implemented, so a `v`-parameterised or multi-step
//! checkpoint runs correctly rather than silently wrongly.
//!
//! ## Neighborhood attention without NATTEN
//!
//! Upstream backs the 3-D windows with NATTEN/CUTLASS-FNA, which is CUDA-only. [`na3d`] is this
//! port's own implementation of the same operator: for each query the attended window is
//! `[clamp(i - k/2, 0, L - k), + k)` per axis — NATTEN's rule of **shifting the window inward** at
//! the border rather than clamping and masking, which is also what upstream's vendored
//! `fallback_na.eager` backend implements and therefore what the committed goldens encode. Queries
//! are tiled; every tile of a tile-row shares one additive mask assembled from three tiny per-axis
//! masks, and each tile is one `scaled_dot_product_attention` call, so the `[Nq, Nk]` score matrix
//! is never materialized.
//!
//! ## Memory
//!
//! Everything runs **f32** — the crate's VAE convention, and the pmetal bf16 SDPA/GEMM hazards
//! (`tests/bf16_sdpa_bug.rs`) are exactly the kind of thing a 22-block attention stack amplifies.
//! The lever for large geometries is therefore [`NaDiffusionDecoder::decode_tiled`], not precision:
//! stages 1-3 run once over the whole volume, and stages 4-5 run per tile with a separable
//! trapezoidal pixel blend. The temporal axis is tiled with the **same halo discipline** as the
//! spatial ones — a tiler that starves time produces a plausible-looking but temporally smeared
//! clip instead of an error, which is the failure mode `tests/vae_decode_tiling_parity.rs` was
//! written for on the conv decoder.
//!
//! Measured on this Mac (M-series, f32, real weights, 2026-08-19 —
//! `tests/ltx_2_5_diffvae_parity.rs::peak_memory_and_quality_at_production_geometries`):
//!
//! | geometry | untiled | tiled (stage-4 tile / overlap) |
//! | --- | --- | --- |
//! | 768x512x25 | 6.5 s, **12.29 GiB** | 16.3 s, **5.43 GiB** — `[13, 32, 48]` / `[20, 20, 20]` |
//! | 1280x704x25 | 14.6 s, **26.15 GiB** | 24.3 s, **9.77 GiB** — `[13, 44, 80]` / `[20, 20, 20]` |
//!
//! Peak tracks pixel area almost exactly (2.13x the area, 2.13x the peak), so the untiled path is
//! the one that runs out first; tiling trades ~1.6x the time for ~2.5x less memory and lands within
//! 65-69 dB of the untiled decode. The temporal axis usually cannot be split at these lengths — the
//! stage-5 halo is 20 stage-4 cells, which is more frames than a 25-frame clip has — so the tile
//! above is spatial, and an unsplit axis' overlap is not held to the `overlap < tile` rule.
//!
//! Reference: `Lightricks/LTX-2` v1.2.0, commit `d151147788a9284cca791edc6ce898007e727fe6`,
//! `packages/ltx-core/src/ltx_core/model/video_vae/{diffusion_video_decoder,diffusion_tiling}.py`
//! and `.../video_vae/transformer/*`. Goldens: `tools/dump_ltx25_diffvae_golden.py`.

use std::path::Path;

use serde_json::Value;

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, divide, matmul, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{linear, silu, timestep_sincos};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{Error, Result};

use crate::contiguous;
use crate::vae::{patchify, unpatchify};

/// `nn.RMSNorm(..., eps=1e-6)` — every norm in the decoder, deterministic and diffusion alike.
const NORM_EPS: f32 = 1e-6;

/// RoPE base for the neighborhood attention's absolute positional rotation
/// (`NeighborhoodAttention3D.rope_base`).
const ROPE_BASE: f64 = 10_000.0;

/// `SpatioTemporalScaleFactors.default()` — pinned by `diffusion_video_decoder.py`, unchanged from
/// LTX-2.3. `(time, height, width)` pixels per latent cell. The released ladder's own upsample
/// strides multiply out to exactly this (asserted in the unit tests); the decoder derives its scale
/// from the ladder rather than reading the constant, so a differently-shaped checkpoint is decoded
/// at its own geometry instead of at 2.3's.
pub const VIDEO_SCALE_FACTORS: [i32; 3] = [8, 32, 32];

/// SwiGLU `mlp_ratio` (`NABlock` / `DiffusionNABlock`): `hidden = round_up(4 * dim, 16)`.
const MLP_RATIO: i32 = 4;

/// The SwiGLU hidden width a block of `dim` channels declares.
fn mlp_hidden(dim: i32) -> i32 {
    (MLP_RATIO * dim + 15) / 16 * 16
}

/// Token-tile for the SwiGLU hidden buffer (upstream `DEFAULT_SWIGLU_TILE_SIZE`). Bounds the
/// `[tokens, 4*dim]` intermediate, which at stage-5 production geometry would otherwise be the
/// single largest allocation in the decoder.
const SWIGLU_TILE_TOKENS: i32 = 16_384;

/// Per-tile `Nq * Nk` budget for [`na3d`]. The score matrix itself never lands (SDPA is fused), but
/// the additive mask does, so this caps one mask at 64 MiB of f32 while keeping tiles large enough
/// that the attention is still a real GEMM. Upstream's eager backend uses `2**25` for the same
/// knob; the tighter value here trades a little recompute for a smaller transient.
const NA_TILE_BUDGET: i64 = 1 << 24;

/// One masked-out score, per axis. Three of these sum to about -3e30 — still finite in f32, so the
/// softmax sees a hard zero rather than a NaN.
const MASK_NEG: f32 = -1e30;

// ---------------------------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------------------------

/// What the stage-5 blocks predict (`vae.model_output_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelOutputType {
    /// The final step's prediction **is** the clean sample; no trailing Euler update.
    X0,
    /// The blocks predict velocity; every step (including the last) is an Euler update.
    Velocity,
}

impl ModelOutputType {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "x0" => Ok(Self::X0),
            "v" => Ok(Self::Velocity),
            other => Err(Error::Msg(format!(
                "ltx diffvae: unknown vae.model_output_type {other:?} (expected x0 | v)"
            ))),
        }
    }
}

/// The `NADiffusionDecoder` structure, read from `vae.decoder` (plus `vae.model_output_type`, which
/// upstream keeps a sibling of `decoder` rather than inside it).
///
/// Channel widths ride on the weights; this drives the *structure* — how many stages, how deep,
/// how wide a window, and where the upsample hops sit.
#[derive(Clone, Debug, PartialEq)]
pub struct NaDiffusionDecoderConfig {
    /// Latent channels entering `conv_in` (128).
    pub in_channels: i32,
    /// RGB channels leaving `conv_out` after unpatchify (3).
    pub out_channels: i32,
    /// Spatial patch size of the stage-5 pixel grid (4).
    pub patch_size: i32,
    /// Attention head width; every stage channel count must be a multiple of it (64).
    pub head_dim: i32,
    /// Per-stage channels, stages 1..=5.
    pub stage_channels: Vec<i32>,
    /// Per-stage block depths, stages 1..=5.
    pub stage_depths: Vec<i32>,
    /// Per-stage 3-D window `(K_t, K_h, K_w)` for the deterministic stages.
    pub stage_kernels: Vec<[i32; 3]>,
    /// `(stride, out-channel reduction)` per upsample hop — one fewer than the stage count.
    pub upsamples: Vec<([i32; 3], i32)>,
    /// The diffusion stage's own window, which is wider than any deterministic stage's.
    pub stage5_kernel: [i32; 3],
    /// Explicit stage-5 width; `None` means "the last stage channel count".
    pub stage5_channels: Option<i32>,
    /// Timestep-embedding width feeding the shared AdaLN-Zero (384).
    pub t_emb_dim: i32,
    /// Steps the shipped schedule runs (`linspace(1, 1/N, N)`); 1 for the released checkpoint.
    pub default_num_inference_steps: i32,
    /// Timesteps are multiplied by this before the sinusoidal embedding (1000.0).
    pub timestep_scale_multiplier: f32,
    /// What the stage-5 blocks predict.
    pub model_output_type: ModelOutputType,
}

fn get_i32(v: &Value, key: &str, default: i32) -> i32 {
    v.get(key)
        .and_then(Value::as_i64)
        .map_or(default, |x| x as i32)
}

fn triple(v: &Value, what: &str) -> Result<[i32; 3]> {
    let arr = v.as_array().filter(|a| a.len() == 3).ok_or_else(|| {
        Error::Msg(format!(
            "ltx diffvae: {what} must be a 3-element array, got {v}"
        ))
    })?;
    let mut out = [0i32; 3];
    for (slot, item) in out.iter_mut().zip(arr) {
        *slot = item.as_i64().ok_or_else(|| {
            Error::Msg(format!(
                "ltx diffvae: {what} entries must be integers, got {v}"
            ))
        })? as i32;
    }
    Ok(out)
}

impl NaDiffusionDecoderConfig {
    /// Parse the `vae` block of an `embedded_config.json` (or a component checkpoint's
    /// `__metadata__.config.vae`, which is the same shape).
    ///
    /// Every architecture field is **required**: unlike the conv VAE — where an absent block list
    /// means "a 2.3 tree that predates the embedded config", so the 2.3 defaults are the right
    /// answer — there has never been a default `NADiffusionDecoder`, and inventing one would build
    /// a differently-shaped decoder against real weights.
    pub fn from_embedded_vae(v: &Value) -> Result<Self> {
        let dec = v.get("decoder").filter(|d| d.is_object()).ok_or_else(|| {
            Error::Msg("ltx diffvae: config.vae has no `decoder` block".to_string())
        })?;
        let class = dec.get("_class_name").and_then(Value::as_str);
        if class != Some("NADiffusionDecoder") {
            return Err(Error::Msg(format!(
                "ltx diffvae: expected vae.decoder._class_name = NADiffusionDecoder, got {class:?}"
            )));
        }
        let require = |key: &str| -> Result<&Value> {
            dec.get(key)
                .ok_or_else(|| Error::Msg(format!("ltx diffvae: vae.decoder has no `{key}`")))
        };
        let list_of = |key: &str| -> Result<Vec<i64>> {
            require(key)?
                .as_array()
                .ok_or_else(|| {
                    Error::Msg(format!("ltx diffvae: vae.decoder.{key} must be an array"))
                })?
                .iter()
                .map(|x| {
                    x.as_i64().ok_or_else(|| {
                        Error::Msg(format!("ltx diffvae: vae.decoder.{key} must hold integers"))
                    })
                })
                .collect()
        };

        let stage_channels: Vec<i32> = list_of("stage_channels")?
            .iter()
            .map(|&x| x as i32)
            .collect();
        let stage_depths: Vec<i32> = list_of("stage_depths")?.iter().map(|&x| x as i32).collect();
        let stage_kernels = require("stage_kernels")?
            .as_array()
            .ok_or_else(|| {
                Error::Msg("ltx diffvae: vae.decoder.stage_kernels must be an array".into())
            })?
            .iter()
            .map(|k| triple(k, "vae.decoder.stage_kernels[]"))
            .collect::<Result<Vec<_>>>()?;
        let upsamples = require("upsamples")?
            .as_array()
            .ok_or_else(|| Error::Msg("ltx diffvae: vae.decoder.upsamples must be an array".into()))?
            .iter()
            .map(|entry| {
                let pair = entry.as_array().filter(|a| a.len() == 2).ok_or_else(|| {
                    Error::Msg(format!(
                        "ltx diffvae: each vae.decoder.upsamples entry must be [stride, reduction], got {entry}"
                    ))
                })?;
                let stride = triple(&pair[0], "vae.decoder.upsamples[][0]")?;
                let reduction = pair[1].as_i64().ok_or_else(|| {
                    Error::Msg("ltx diffvae: upsample reduction must be an integer".to_string())
                })? as i32;
                Ok((stride, reduction))
            })
            .collect::<Result<Vec<_>>>()?;

        let cfg = Self {
            in_channels: get_i32(dec, "in_channels", 128),
            out_channels: get_i32(dec, "out_channels", 3),
            patch_size: get_i32(dec, "patch_size", 4),
            head_dim: get_i32(dec, "head_dim", get_i32(dec, "na_head_dim", 64)),
            stage_channels,
            stage_depths,
            stage_kernels,
            upsamples,
            stage5_kernel: triple(require("stage5_kernel")?, "vae.decoder.stage5_kernel")?,
            stage5_channels: dec
                .get("stage5_channels")
                .and_then(Value::as_i64)
                .map(|x| x as i32),
            t_emb_dim: get_i32(dec, "t_emb_dim", 384),
            default_num_inference_steps: get_i32(dec, "default_num_inference_steps", 2),
            timestep_scale_multiplier: dec
                .get("timestep_scale_multiplier")
                .and_then(Value::as_f64)
                .unwrap_or(1.0) as f32,
            model_output_type: ModelOutputType::parse(
                v.get("model_output_type")
                    .and_then(Value::as_str)
                    .unwrap_or("v"),
            )?,
        };
        if let Some(mode) = dec.get("spatial_padding_mode").and_then(Value::as_str) {
            if mode != "zeros" {
                return Err(Error::Msg(format!(
                    "ltx diffvae: vae.decoder.spatial_padding_mode {mode:?} is not implemented \
                     (the decoder has no spatial convolutions; only `zeros` is meaningful)"
                )));
            }
        }
        cfg.validated()
    }

    /// Load from a converted model directory's `embedded_config.json` (`vae` block).
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let path = root.join("embedded_config.json");
        let text = std::fs::read_to_string(&path)?;
        let root_cfg: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("ltx diffvae: parse {}: {e}", path.display())))?;
        let vae = root_cfg.get("vae").ok_or_else(|| {
            Error::Msg(format!(
                "ltx diffvae: {} has no `vae` block",
                path.display()
            ))
        })?;
        Self::from_embedded_vae(vae)
    }

    /// Reject shapes the decoder cannot be built from, at load rather than at the first forward.
    fn validated(self) -> Result<Self> {
        let n = self.stage_channels.len();
        if n < 2 || self.stage_depths.len() != n || self.stage_kernels.len() != n {
            return Err(Error::Msg(format!(
                "ltx diffvae: stage_channels/stage_depths/stage_kernels must agree and hold >= 2 \
                 stages, got {}/{}/{}",
                n,
                self.stage_depths.len(),
                self.stage_kernels.len()
            )));
        }
        if self.upsamples.len() != n - 1 {
            return Err(Error::Msg(format!(
                "ltx diffvae: expected {} upsample hops for {n} stages, got {}",
                n - 1,
                self.upsamples.len()
            )));
        }
        if self.head_dim < 2 || self.head_dim % 2 != 0 {
            return Err(Error::Msg(format!(
                "ltx diffvae: head_dim must be a positive even number, got {}",
                self.head_dim
            )));
        }
        for &c in self
            .stage_channels
            .iter()
            .chain(self.stage5_channels.iter())
        {
            if c <= 0 || c % self.head_dim != 0 {
                return Err(Error::Msg(format!(
                    "ltx diffvae: stage width {c} is not a positive multiple of head_dim {}",
                    self.head_dim
                )));
            }
        }
        for kernel in self
            .stage_kernels
            .iter()
            .chain(std::iter::once(&self.stage5_kernel))
        {
            if kernel.iter().any(|&k| k < 1) {
                return Err(Error::Msg(format!(
                    "ltx diffvae: window sizes must be >= 1, got {kernel:?}"
                )));
            }
        }
        for (stride, reduction) in &self.upsamples {
            if stride.iter().any(|&s| s < 1) || *reduction < 1 {
                return Err(Error::Msg(format!(
                    "ltx diffvae: upsample ({stride:?}, {reduction}) must hold positive factors"
                )));
            }
        }
        if self.default_num_inference_steps < 1 {
            return Err(Error::Msg(format!(
                "ltx diffvae: default_num_inference_steps must be >= 1, got {}",
                self.default_num_inference_steps
            )));
        }
        if self.patch_size < 1 || self.out_channels < 1 || self.in_channels < 1 {
            return Err(Error::Msg(
                "ltx diffvae: patch_size / in_channels / out_channels must be >= 1".into(),
            ));
        }
        Ok(self)
    }

    /// Stage-5 feature width: explicit if declared, else the last stage's channels.
    pub fn stage5_width(&self) -> i32 {
        self.stage5_channels
            .unwrap_or_else(|| *self.stage_channels.last().expect("validated: >= 2 stages"))
    }

    /// Latent frames replicated at the tail before stages 1-4, to keep the last real frame off the
    /// neighborhood window's shifted border. `(stage_kernels[0].t / 2) * 2`, per upstream.
    pub fn ghost_latent_frames(&self) -> i32 {
        (self.stage_kernels[0][0] / 2) * 2
    }

    /// Per-axis latent floor so every stage's window fits its grid
    /// (`diffusion_tiling.all_stages_min_tile_size`).
    pub fn min_latent_shape(&self) -> [i32; 3] {
        let mut cumulative = [1i32; 3];
        let mut mins = [1i32; 3];
        for (stage, kernel) in self
            .stage_kernels
            .iter()
            .enumerate()
            .take(self.upsamples.len())
        {
            for axis in 0..3 {
                mins[axis] = mins[axis].max(ceil_div(kernel[axis], cumulative[axis]));
            }
            let stride = self.upsamples[stage].0;
            for (axis, slot) in cumulative.iter_mut().enumerate() {
                *slot *= stride[axis];
            }
        }
        for axis in 0..3 {
            mins[axis] = mins[axis].max(ceil_div(self.stage5_kernel[axis], cumulative[axis]));
        }
        mins
    }

    /// Minimum stage-4-input extent so stages 4 and 5 each see at least their window
    /// (`diffusion_tiling.compute_tile_min_size`).
    pub fn min_tile_shape(&self) -> [i32; 3] {
        let last = self.upsamples.len() - 1;
        let stride = self.upsamples[last].0;
        let kernel4 = self.stage_kernels[last];
        let mut out = [0i32; 3];
        for axis in 0..3 {
            out[axis] = kernel4[axis].max(ceil_div(self.stage5_kernel[axis], stride[axis]));
        }
        out
    }

    /// One-sided stage-4/5 halos in stage-4-input units (`diffusion_tiling.compute_tile_halos`),
    /// reduced to the dominant per-axis halo — the minimum overlap a tiling may use.
    pub fn tile_halo(&self) -> [i32; 3] {
        let last = self.upsamples.len() - 1;
        let stride = self.upsamples[last].0;
        let kernel4 = self.stage_kernels[last];
        let depth4 = self.stage_depths[last];
        let depth5 = *self.stage_depths.last().expect("validated");
        let mut out = [0i32; 3];
        for axis in 0..3 {
            let halo4 = depth4 * (kernel4[axis] / 2);
            let halo5 = ceil_div(depth5 * (self.stage5_kernel[axis] / 2), stride[axis]);
            out[axis] = halo4.max(halo5);
        }
        out
    }

    /// Pixels (and frames) per latent cell, from the ladder's own hops: the product of the upsample
    /// strides, spatially times the patch size. Equals [`VIDEO_SCALE_FACTORS`] for the released
    /// checkpoint.
    pub fn pixel_scale(&self) -> [i32; 3] {
        let mut scale = [1i32; 3];
        for (stride, _) in &self.upsamples {
            for axis in 0..3 {
                scale[axis] *= stride[axis];
            }
        }
        [
            scale[0],
            scale[1] * self.patch_size,
            scale[2] * self.patch_size,
        ]
    }

    /// Stage-4-input `(T, H, W)` for a latent extent, after the first `n-1` upsample hops.
    pub fn stage4_shape(&self, latent_t: i32, latent_h: i32, latent_w: i32) -> [i32; 3] {
        let mut out = [latent_t, latent_h, latent_w];
        for (stride, _) in self.upsamples.iter().take(self.upsamples.len() - 1) {
            for axis in 0..3 {
                out[axis] *= stride[axis];
            }
            if stride[0] == 2 {
                out[0] -= 1;
            }
        }
        out
    }

    /// Stage-5 pixel `(F, H, W)` a stage-4 extent expands to — the shape the noise must have.
    pub fn stage5_pixel_shape(
        &self,
        stage4: [i32; 3],
        drop_leading_frame: bool,
        pad_trailing: bool,
    ) -> [i32; 3] {
        let last = self.upsamples.len() - 1;
        let stride = self.upsamples[last].0;
        let mut frames = stage4[0] * stride[0];
        if drop_leading_frame && stride[0] == 2 {
            frames -= 1;
        }
        if pad_trailing {
            frames = frames.max(self.stage5_kernel[0]);
        }
        [
            frames,
            stage4[1] * stride[1] * self.patch_size,
            stage4[2] * stride[2] * self.patch_size,
        ]
    }

    /// Stage-5 noise geometry for an untiled decode of a `(T, H, W)` latent — the shape
    /// [`NaDiffusionDecoder::decode`] requires. Applies the latent size floor first, exactly as
    /// decode does, so a below-floor latent reports the geometry it will actually be run at.
    pub fn noise_shape(&self, latent_t: i32, latent_h: i32, latent_w: i32) -> [i32; 3] {
        let min = self.min_latent_shape();
        let stage4 = self.stage4_shape(
            latent_t.max(min[0]),
            latent_h.max(min[1]),
            latent_w.max(min[2]),
        );
        self.stage5_pixel_shape(stage4, true, true)
    }
}

fn ceil_div(a: i32, b: i32) -> i32 {
    debug_assert!(b > 0);
    (a + b - 1) / b
}

// ---------------------------------------------------------------------------------------------
// Small layers
// ---------------------------------------------------------------------------------------------

/// `y = x Wᵀ` — the SwiGLU projections carry no bias.
fn linear_nobias(x: &Array, w: &Array) -> Result<Array> {
    Ok(matmul(x, w.t())?)
}

fn weight(w: &Weights, key: &str) -> Result<Array> {
    to_dtype(w.require(key)?, Dtype::Float32)
}

/// `[out, in]` weight + `[out]` bias.
struct Linear {
    w: Array,
    b: Array,
}

impl Linear {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: weight(w, &format!("{prefix}.weight"))?,
            b: weight(w, &format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        linear(x, &self.w, &self.b)
    }

    fn out_features(&self) -> i32 {
        self.w.shape()[0]
    }
}

/// `w_down(silu(x W_gateᵀ) * (x W_upᵀ))`, tiled over tokens so the `[tokens, hidden]` intermediate
/// stays bounded (upstream `swiglu_tiled`). Tiling changes nothing numerically — each token's
/// result depends only on itself.
struct SwiGlu {
    w_gate: Array,
    w_up: Array,
    w_down: Array,
}

impl SwiGlu {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w_gate: weight(w, &format!("{prefix}.w_gate.weight"))?,
            w_up: weight(w, &format!("{prefix}.w_up.weight"))?,
            w_down: weight(w, &format!("{prefix}.w_down.weight"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape().to_vec();
        let dim = *shape.last().expect("rank >= 1");
        let tokens: i32 = shape[..shape.len() - 1].iter().product();
        let flat = contiguous(x)?.reshape(&[tokens, dim])?;
        if tokens <= SWIGLU_TILE_TOKENS {
            return Ok(self.tile(&flat)?.reshape(&shape)?);
        }
        let mut parts: Vec<Array> = Vec::new();
        let mut start = 0;
        while start < tokens {
            let end = (start + SWIGLU_TILE_TOKENS).min(tokens);
            let part = self.tile(&slice_axis(&flat, 0, start, end)?)?;
            part.eval()?;
            parts.push(part);
            start = end;
        }
        let refs: Vec<&Array> = parts.iter().collect();
        Ok(concatenate_axis(&refs, 0)?.reshape(&shape)?)
    }

    fn tile(&self, flat: &Array) -> Result<Array> {
        let gate = silu(&linear_nobias(flat, &self.w_gate)?)?;
        let up = linear_nobias(flat, &self.w_up)?;
        linear_nobias(&multiply(&gate, &up)?, &self.w_down)
    }
}

/// Slice `x` along `axis` to `[start, end)`.
fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let rank = x.ndim() as i32;
    let axis = if axis < 0 { axis + rank } else { axis };
    let len = x.shape()[axis as usize];
    if start < 0 || end > len || start >= end {
        return Err(Error::Msg(format!(
            "ltx diffvae: slice [{start}, {end}) is out of range for axis {axis} of length {len}"
        )));
    }
    if start == 0 && end == len {
        return Ok(x.clone());
    }
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), axis)?)
}

/// Pad or crop `axis` to `size`. `symmetric` splits the difference across both edges
/// (`before = need / 2`), replicating the edge slice; otherwise the tail is repeated or dropped.
/// Returns the resized array and the `(before, after)` element counts, in that axis's units.
fn resize_axis(x: &Array, axis: i32, size: i32, symmetric: bool) -> Result<(Array, (i32, i32))> {
    let rank = x.ndim() as i32;
    let axis = if axis < 0 { axis + rank } else { axis };
    let len = x.shape()[axis as usize];
    if size < 1 {
        return Err(Error::Msg(format!(
            "ltx diffvae: resize target must be >= 1, got {size}"
        )));
    }
    if len == size {
        return Ok((x.clone(), (0, 0)));
    }
    if len < size {
        let need = size - len;
        let (before, after) = if symmetric {
            (need / 2, need - need / 2)
        } else {
            (0, need)
        };
        let mut parts: Vec<Array> = Vec::new();
        if before > 0 {
            parts.push(Array::repeat_axis::<f32>(
                slice_axis(x, axis, 0, 1)?,
                before,
                axis,
            )?);
        }
        parts.push(x.clone());
        if after > 0 {
            parts.push(Array::repeat_axis::<f32>(
                slice_axis(x, axis, len - 1, len)?,
                after,
                axis,
            )?);
        }
        let refs: Vec<&Array> = parts.iter().collect();
        return Ok((concatenate_axis(&refs, axis)?, (before, after)));
    }
    let need = len - size;
    let (before, after) = if symmetric {
        (need / 2, need - need / 2)
    } else {
        (0, need)
    };
    Ok((slice_axis(x, axis, before, before + size)?, (before, after)))
}

// ---------------------------------------------------------------------------------------------
// 3-D neighborhood attention
// ---------------------------------------------------------------------------------------------

/// Per-index window start along one axis: `clamp(i - k/2, 0, L - k)` with `k = min(kernel, L)`.
/// This is NATTEN's shifted-window rule — near the border the window slides inward and keeps its
/// full size, instead of being clipped and renormalised.
fn window_starts(len: i32, kernel: i32) -> Vec<i32> {
    let k = kernel.min(len);
    let lo = len - k;
    let half = k / 2;
    (0..len).map(|i| (i - half).clamp(0, lo)).collect()
}

/// Query-tile extents keeping one tile's `Nq * Nk` under [`NA_TILE_BUDGET`]. Halves the axis with
/// the largest tile-to-window ratio, which is the one paying the most for its halo.
fn pick_tiles(dims: [i32; 3], kernels: [i32; 3]) -> [i32; 3] {
    let mut tiles = dims;
    let cost = |t: [i32; 3]| -> i64 {
        let nq: i64 = t.iter().map(|&x| x as i64).product();
        let nk: i64 = (0..3)
            .map(|a| dims[a].min(t[a] + kernels[a] - 1) as i64)
            .product();
        nq.saturating_mul(nk)
    };
    while cost(tiles) > NA_TILE_BUDGET && tiles.iter().any(|&t| t > 1) {
        let mut best = 0usize;
        let mut best_ratio = f64::NEG_INFINITY;
        for axis in 0..3 {
            let ratio = tiles[axis] as f64 / kernels[axis] as f64;
            if ratio > best_ratio {
                best_ratio = ratio;
                best = axis;
            }
        }
        if tiles[best] <= 1 {
            break;
        }
        tiles[best] = ((tiles[best] + 1) / 2).max(1);
    }
    tiles
}

/// Additive `[n, nk]` visibility mask for one axis of one tile: `0` inside the window, [`MASK_NEG`]
/// outside.
fn axis_mask(
    starts: &[i32],
    q0: i32,
    q1: i32,
    kernel_eff: i32,
    key_start: i32,
    key_len: i32,
) -> Array {
    let n = (q1 - q0) as usize;
    let mut data = vec![MASK_NEG; n * key_len as usize];
    for (j, row) in data.chunks_mut(key_len as usize).enumerate() {
        let lo = starts[q0 as usize + j] - key_start;
        for slot in row.iter_mut().skip(lo as usize).take(kernel_eff as usize) {
            *slot = 0.0;
        }
    }
    Array::from_slice(&data, &[n as i32, key_len])
}

/// 3-D neighborhood attention over `(B, T, H, W, NH, HD)` tensors, returning
/// `(B, T, H, W, NH * HD)`.
///
/// `q` must already carry the `head_dim^-0.5` scale (the SDPA call runs at `scale = 1.0`), and both
/// `q` and `k` their rotary positions. Semantics are NATTEN `na3d`'s: each query attends
/// `[clamp(i - k/2, 0, L - k), + k)` on every axis, so at the border the window slides inward and
/// keeps its full size instead of being clipped and renormalised.
pub fn na3d(q: &Array, k: &Array, v: &Array, kernel: [i32; 3]) -> Result<Array> {
    let sh = q.shape().to_vec();
    if sh.len() != 6 {
        return Err(Error::Msg(format!(
            "ltx diffvae: na3d expects (B, T, H, W, NH, HD), got {sh:?}"
        )));
    }
    let (b, t, h, w, nh, hd) = (sh[0], sh[1], sh[2], sh[3], sh[4], sh[5]);
    let dims = [t, h, w];
    let mut kernels = [0i32; 3];
    for axis in 0..3 {
        if dims[axis] < kernel[axis] {
            return Err(Error::Msg(format!(
                "ltx diffvae: 3-D neighborhood attention needs each dim >= its window; got \
                 (T,H,W)=({t},{h},{w}) vs kernel {kernel:?}"
            )));
        }
        kernels[axis] = kernel[axis].min(dims[axis]);
    }
    let starts: Vec<Vec<i32>> = (0..3).map(|a| window_starts(dims[a], kernels[a])).collect();
    let tiles = pick_tiles(dims, kernels);

    // Key extent covering a query tile: from the first query's window start to the last query's
    // window end.
    let key_range = |axis: usize, q0: i32, q1: i32| -> (i32, i32) {
        let lo = starts[axis][q0 as usize];
        let hi = starts[axis][(q1 - 1) as usize] + kernels[axis];
        (lo, hi - lo)
    };

    let mut t_parts: Vec<Array> = Vec::new();
    let mut t0 = 0;
    while t0 < t {
        let t1 = (t0 + tiles[0]).min(t);
        let (kt0, ktn) = key_range(0, t0, t1);
        let qt = slice_axis(q, 1, t0, t1)?;
        let kt = slice_axis(k, 1, kt0, kt0 + ktn)?;
        let vt = slice_axis(v, 1, kt0, kt0 + ktn)?;
        let mt = axis_mask(&starts[0], t0, t1, kernels[0], kt0, ktn);

        let mut h_parts: Vec<Array> = Vec::new();
        let mut h0 = 0;
        while h0 < h {
            let h1 = (h0 + tiles[1]).min(h);
            let (kh0, khn) = key_range(1, h0, h1);
            let qth = slice_axis(&qt, 2, h0, h1)?;
            let kth = slice_axis(&kt, 2, kh0, kh0 + khn)?;
            let vth = slice_axis(&vt, 2, kh0, kh0 + khn)?;
            let mh = axis_mask(&starts[1], h0, h1, kernels[1], kh0, khn);

            let mut w_parts: Vec<Array> = Vec::new();
            let mut w0 = 0;
            while w0 < w {
                let w1 = (w0 + tiles[2]).min(w);
                let (kw0, kwn) = key_range(2, w0, w1);
                let mw = axis_mask(&starts[2], w0, w1, kernels[2], kw0, kwn);

                let (nt, nhq, nw) = (t1 - t0, h1 - h0, w1 - w0);
                let nq = nt * nhq * nw;
                let nk = ktn * khn * kwn;
                // Separable additive mask: `[nq, nk]` assembled by broadcasting three small
                // per-axis masks, so the only 2-D allocation is the result.
                let mask = add(
                    &add(
                        &mt.reshape(&[nt, 1, 1, ktn, 1, 1])?,
                        &mh.reshape(&[1, nhq, 1, 1, khn, 1])?,
                    )?,
                    &mw.reshape(&[1, 1, nw, 1, 1, kwn])?,
                )?
                .reshape(&[1, 1, nq, nk])?;

                let qs = contiguous(&slice_axis(&qth, 3, w0, w1)?)?
                    .reshape(&[b, nq, nh, hd])?
                    .transpose_axes(&[0, 2, 1, 3])?;
                let ks = contiguous(&slice_axis(&kth, 3, kw0, kw0 + kwn)?)?
                    .reshape(&[b, nk, nh, hd])?
                    .transpose_axes(&[0, 2, 1, 3])?;
                let vs = contiguous(&slice_axis(&vth, 3, kw0, kw0 + kwn)?)?
                    .reshape(&[b, nk, nh, hd])?
                    .transpose_axes(&[0, 2, 1, 3])?;

                let out = scaled_dot_product_attention(
                    &contiguous(&qs)?,
                    &contiguous(&ks)?,
                    &contiguous(&vs)?,
                    1.0,
                    &mask,
                    None,
                )?;
                let out = out
                    .transpose_axes(&[0, 2, 1, 3])?
                    .reshape(&[b, nt, nhq, nw, nh * hd])?;
                let out = contiguous(&out)?;
                // Evaluate per tile: forcing one lazy graph over every tile of a production-sized
                // stage-5 volume is the shape that fails as
                // `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored` (sc-18760).
                out.eval()?;
                w_parts.push(out);
                w0 = w1;
            }
            let refs: Vec<&Array> = w_parts.iter().collect();
            h_parts.push(concatenate_axis(&refs, 3)?);
            h0 = h1;
        }
        let refs: Vec<&Array> = h_parts.iter().collect();
        t_parts.push(concatenate_axis(&refs, 2)?);
        t0 = t1;
    }
    let refs: Vec<&Array> = t_parts.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}

// ---------------------------------------------------------------------------------------------
// Rotary positions
// ---------------------------------------------------------------------------------------------

/// Split `head_dim` across the `(T, H, W)` rotary chunks — upstream `default_rope_dim_split`.
fn rope_dim_split(head_dim: i32) -> Result<[i32; 3]> {
    if head_dim % 8 != 0 {
        return Err(Error::Msg(format!(
            "ltx diffvae: head_dim must be a multiple of 8 for the default rope split, got {head_dim}"
        )));
    }
    let mut d_t = (head_dim / 4) / 2 * 2;
    let mut d_hw = (head_dim - d_t) / 2;
    if d_hw % 2 != 0 {
        d_t -= 2;
        d_hw = (head_dim - d_t) / 2;
    }
    Ok([d_t, d_hw, d_hw])
}

/// `1 / base^(i / dim)` for even `i` — the rotary inverse frequencies, built in f64 as upstream
/// does before the single cast to f32.
fn rope_inv_freqs(dim: i32, base: f64) -> Array {
    let values: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| (1.0 / base.powf(i as f64 / dim as f64)) as f32)
        .collect();
    Array::from_slice(&values, &[dim / 2])
}

/// Rotate one axis chunk of `(B, T, H, W, NH, D)` by absolute positions along `axis`
/// (1 = T, 2 = H, 3 = W). Pairs are adjacent (`[x0, x1], [x2, x3], ...`).
fn rotate_axis(x: &Array, inv: &Array, axis: usize) -> Result<Array> {
    let sh = x.shape().to_vec();
    let d = *sh.last().expect("rank >= 1");
    let half = d / 2;
    let pairs = contiguous(x)?.reshape(&[sh[0], sh[1], sh[2], sh[3], sh[4], half, 2])?;
    let xe = slice_axis(&pairs, 6, 0, 1)?.reshape(&[sh[0], sh[1], sh[2], sh[3], sh[4], half])?;
    let xo = slice_axis(&pairs, 6, 1, 2)?.reshape(&[sh[0], sh[1], sh[2], sh[3], sh[4], half])?;

    let len = sh[axis];
    let positions: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let pos = Array::from_slice(&positions, &[len, 1]);
    let mut ang_shape = [1i32, 1, 1, 1, 1, half];
    ang_shape[axis] = len;
    let ang = multiply(&pos, &inv.reshape(&[1, half])?)?.reshape(&ang_shape)?;
    let (cos, sin) = (ang.cos()?, ang.sin()?);

    let re = subtract(&multiply(&xe, &cos)?, &multiply(&xo, &sin)?)?;
    let ro = add(&multiply(&xe, &sin)?, &multiply(&xo, &cos)?)?;
    let stacked = concatenate_axis(
        &[
            &re.reshape(&[sh[0], sh[1], sh[2], sh[3], sh[4], half, 1])?,
            &ro.reshape(&[sh[0], sh[1], sh[2], sh[3], sh[4], half, 1])?,
        ],
        6,
    )?;
    Ok(stacked.reshape(&sh)?)
}

// ---------------------------------------------------------------------------------------------
// Attention / blocks
// ---------------------------------------------------------------------------------------------

struct NaAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    proj: Linear,
    q_norm: Array,
    k_norm: Array,
    inv: [Array; 3],
    split: [i32; 3],
    kernel: [i32; 3],
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl NaAttention {
    /// The checkpoint ships one fused `qkv` Linear; upstream splits it into three at load, and so
    /// does this — three narrow GEMMs never materialise a `3*dim`-wide intermediate.
    fn load(w: &Weights, prefix: &str, kernel: [i32; 3], head_dim: i32) -> Result<Self> {
        let fused_w = weight(w, &format!("{prefix}.qkv.weight"))?;
        let fused_b = weight(w, &format!("{prefix}.qkv.bias"))?;
        let rows = fused_w.shape()[0];
        if rows % 3 != 0 {
            return Err(Error::Msg(format!(
                "ltx diffvae: {prefix}.qkv.weight has {rows} rows, not divisible by 3"
            )));
        }
        let dim = rows / 3;
        if dim % head_dim != 0 {
            return Err(Error::Msg(format!(
                "ltx diffvae: {prefix} width {dim} is not a multiple of head_dim {head_dim}"
            )));
        }
        let part = |i: i32| -> Result<Linear> {
            Ok(Linear {
                w: contiguous(&slice_axis(&fused_w, 0, i * dim, (i + 1) * dim)?)?,
                b: contiguous(&slice_axis(&fused_b, 0, i * dim, (i + 1) * dim)?)?,
            })
        };
        let split = rope_dim_split(head_dim)?;
        Ok(Self {
            to_q: part(0)?,
            to_k: part(1)?,
            to_v: part(2)?,
            proj: Linear::load(w, &format!("{prefix}.proj"))?,
            q_norm: weight(w, &format!("{prefix}.q_norm.weight"))?,
            k_norm: weight(w, &format!("{prefix}.k_norm.weight"))?,
            inv: [
                rope_inv_freqs(split[0], ROPE_BASE),
                rope_inv_freqs(split[1], ROPE_BASE),
                rope_inv_freqs(split[2], ROPE_BASE),
            ],
            split,
            kernel,
            num_heads: dim / head_dim,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn width(&self) -> i32 {
        self.num_heads * self.head_dim
    }

    /// Q/K/V as `(B, T, H, W, NH, HD)`.
    fn project(&self, x: &Array) -> Result<(Array, Array, Array)> {
        let sh = x.shape().to_vec();
        let heads = [sh[0], sh[1], sh[2], sh[3], self.num_heads, self.head_dim];
        Ok((
            self.to_q.forward(x)?.reshape(&heads)?,
            self.to_k.forward(x)?.reshape(&heads)?,
            self.to_v.forward(x)?.reshape(&heads)?,
        ))
    }

    /// Absolute rotary positions over the three axes, applied to the head-split tensor.
    ///
    /// Positions are the tile's own `0..len`, not a global origin. Upstream does the same and
    /// documents why it is exact under tiling: every window is local, and a shared phase offset
    /// cancels inside that window's softmax.
    fn rope(&self, x: &Array) -> Result<Array> {
        let [d_t, d_h, _] = self.split;
        let d = self.head_dim;
        let xt = rotate_axis(&slice_axis(x, 5, 0, d_t)?, &self.inv[0], 1)?;
        let xh = rotate_axis(&slice_axis(x, 5, d_t, d_t + d_h)?, &self.inv[1], 2)?;
        let xw = rotate_axis(&slice_axis(x, 5, d_t + d_h, d)?, &self.inv[2], 3)?;
        Ok(concatenate_axis(&[&xt, &xh, &xw], 5)?)
    }

    /// `proj(na3d(rope(norm(qkv(x)))))` for a channels-last `(B, T, H, W, C)` input.
    fn forward(&self, x: &Array) -> Result<Array> {
        let (q, k, v) = self.project(x)?;
        let q = multiply(
            &rms_norm(&q, &self.q_norm, NORM_EPS)?,
            Array::from_f32(self.scale),
        )?;
        let k = rms_norm(&k, &self.k_norm, NORM_EPS)?;
        let q = contiguous(&self.rope(&q)?)?;
        let k = contiguous(&self.rope(&k)?)?;
        let v = contiguous(&v)?;
        let out = na3d(&q, &k, &v, self.kernel)?;
        self.proj.forward(&out)
    }
}

/// Deterministic (stages 1-4) block: `x + attn(norm1(x))`, then `x + swiglu(norm2(x))`.
struct NaBlock {
    norm1: Array,
    attn: NaAttention,
    norm2: Array,
    mlp: SwiGlu,
}

impl NaBlock {
    fn load(w: &Weights, prefix: &str, kernel: [i32; 3], head_dim: i32) -> Result<Self> {
        Ok(Self {
            norm1: weight(w, &format!("{prefix}.norm1.weight"))?,
            attn: NaAttention::load(w, &format!("{prefix}.attn"), kernel, head_dim)?,
            norm2: weight(w, &format!("{prefix}.norm2.weight"))?,
            mlp: SwiGlu::load(w, &format!("{prefix}.mlp"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = rms_norm(x, &self.norm1, NORM_EPS)?;
        let x = add(x, &self.attn.forward(&y)?)?;
        let y = rms_norm(&x, &self.norm2, NORM_EPS)?;
        Ok(add(&x, &self.mlp.forward(&y)?)?)
    }
}

/// Stage-5 block: inject context, then AdaLN-modulated attention and MLP residuals.
struct DiffusionBlock {
    context_proj: Linear,
    /// `[7, dim]`, added to the shared AdaLN chunks before use.
    scale_shift_table: Array,
    norm1: Array,
    attn: NaAttention,
    norm2: Array,
    mlp: SwiGlu,
}

/// The seven AdaLN-Zero chunks (`AdaLNZero.NUM_CHUNKS`). The three gate slots exist for checkpoint
/// shape compatibility and are unused by the block — the released checkpoint carries no `gate_*`
/// parameters at all, so there is nothing folded into the projections either.
const ADALN_CHUNKS: i32 = 7;

impl DiffusionBlock {
    fn load(w: &Weights, prefix: &str, kernel: [i32; 3], head_dim: i32) -> Result<Self> {
        let table = weight(w, &format!("{prefix}.scale_shift_table"))?;
        if table.ndim() != 2 || table.shape()[0] != ADALN_CHUNKS {
            return Err(Error::Msg(format!(
                "ltx diffvae: {prefix}.scale_shift_table must be [{ADALN_CHUNKS}, dim], got {:?}",
                table.shape()
            )));
        }
        Ok(Self {
            context_proj: Linear::load(w, &format!("{prefix}.context_proj"))?,
            scale_shift_table: table,
            norm1: weight(w, &format!("{prefix}.norm1.weight"))?,
            attn: NaAttention::load(w, &format!("{prefix}.attn"), kernel, head_dim)?,
            norm2: weight(w, &format!("{prefix}.norm2.weight"))?,
            mlp: SwiGlu::load(w, &format!("{prefix}.mlp"))?,
        })
    }

    /// `modulation` holds the seven shared AdaLN chunks, each `(1, 1, 1, 1, dim)`.
    fn forward(&self, context: &Array, x: &Array, modulation: &[Array]) -> Result<Array> {
        let chunk = |i: i32| -> Result<Array> {
            let row =
                slice_axis(&self.scale_shift_table, 0, i, i + 1)?.reshape(&[1, 1, 1, 1, -1])?;
            Ok(add(&modulation[i as usize], &row)?)
        };
        let (scale_msa, shift_msa) = (chunk(0)?, chunk(1)?);
        let (scale_mlp, shift_mlp) = (chunk(3)?, chunk(4)?);

        let x = add(x, &self.context_proj.forward(context)?)?;
        let y = modulate(
            &rms_norm(&x, &self.norm1, NORM_EPS)?,
            &scale_msa,
            &shift_msa,
        )?;
        let x = add(&x, &self.attn.forward(&y)?)?;
        let y = modulate(
            &rms_norm(&x, &self.norm2, NORM_EPS)?,
            &scale_mlp,
            &shift_mlp,
        )?;
        Ok(add(&x, &self.mlp.forward(&y)?)?)
    }
}

/// `x * (1 + scale) + shift` (upstream `layers.modulate`).
fn modulate(x: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    Ok(add(
        &multiply(x, &add(scale, Array::from_f32(1.0))?)?,
        shift,
    )?)
}

/// `Linear` + channels-last pixel shuffle (upstream `LinearPixelShuffleUpsample`).
struct PixelShuffleUpsample {
    proj: Linear,
    stride: [i32; 3],
}

impl PixelShuffleUpsample {
    fn load(w: &Weights, prefix: &str, stride: [i32; 3]) -> Result<Self> {
        Ok(Self {
            proj: Linear::load(w, &format!("{prefix}.proj"))?,
            stride,
        })
    }

    fn out_channels(&self) -> i32 {
        self.proj.out_features() / (self.stride[0] * self.stride[1] * self.stride[2])
    }

    /// `(B, T, H, W, C) -> (B, T*p1 [-1], H*p2, W*p3, C_out)`.
    ///
    /// A temporal stride of 2 duplicates the leading frame; dropping it preserves the causal 1:2
    /// (composed 1:8) frame mapping. `drop_leading_frame` must be true **only** for the chunk
    /// holding the tensor's true `t = 0` — a later tile has no duplicate of its own to drop.
    fn forward(&self, x: &Array, drop_leading_frame: bool) -> Result<Array> {
        let sh = x.shape().to_vec();
        let (b, t, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        let [p1, p2, p3] = self.stride;
        let c = self.out_channels();
        let y = self
            .proj
            .forward(x)?
            .reshape(&[b, t, h, w, c, p1, p2, p3])?;
        // (b, t, p1, h, p2, w, p3, c)
        let y = y.transpose_axes(&[0, 1, 5, 2, 6, 3, 7, 4])?;
        let y = contiguous(&y)?.reshape(&[b, t * p1, h * p2, w * p3, c])?;
        if p1 == 2 && drop_leading_frame {
            return slice_axis(&y, 1, 1, y.shape()[1]);
        }
        Ok(y)
    }
}

// ---------------------------------------------------------------------------------------------
// The decoder
// ---------------------------------------------------------------------------------------------

/// Tiling for [`NaDiffusionDecoder::decode_tiled`], in **stage-4-input** units.
///
/// Stages 1-3 always run once over the whole volume (they are cheap relative to stage 5 and their
/// windows are small); the split applies to stages 4-5, whose halos this describes. Sizes and
/// overlaps below [`NaDiffusionDecoderConfig::min_tile_shape`] /
/// [`NaDiffusionDecoderConfig::tile_halo`] are rejected rather than silently clamped: an
/// under-haloed tiling does not error at decode time, it just returns a smeared clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiffVaeTiling {
    /// Tile extent per axis, in stage-4-input cells.
    pub tile: [i32; 3],
    /// Overlap between neighbouring tiles per axis, in stage-4-input cells.
    pub overlap: [i32; 3],
}

impl DiffVaeTiling {
    /// A tiling with the minimum legal overlap for `cfg` and the given per-axis tile extents.
    pub fn with_min_overlap(cfg: &NaDiffusionDecoderConfig, tile: [i32; 3]) -> Self {
        Self {
            tile,
            overlap: cfg.tile_halo(),
        }
    }

    fn validated(self, cfg: &NaDiffusionDecoderConfig) -> Result<Self> {
        let min_tile = cfg.min_tile_shape();
        let halo = cfg.tile_halo();
        for axis in 0..3 {
            if self.tile[axis] < min_tile[axis] {
                return Err(Error::Msg(format!(
                    "ltx diffvae: tile {} on axis {axis} is below the stage-4/5 window floor {}",
                    self.tile[axis], min_tile[axis]
                )));
            }
            if self.overlap[axis] < halo[axis] {
                return Err(Error::Msg(format!(
                    "ltx diffvae: overlap {} on axis {axis} is below the stage-4/5 halo {} — a \
                     starved axis corrupts the blend without erroring",
                    self.overlap[axis], halo[axis]
                )));
            }
            // `overlap < tile` is checked in `decode_tiled`, where the axis' real extent is known:
            // an axis whose tile already covers the volume is never split, so its overlap is
            // irrelevant. Enforcing it here would forbid spatially tiling a clip too short to tile
            // in time — which, with a 20-cell stage-5 halo, is most of them.
        }
        Ok(self)
    }
}

/// One tile's interval along one axis, with the linear ramps that blend it into its neighbours.
#[derive(Clone, Copy, Debug)]
struct Interval {
    start: i32,
    end: i32,
    left_ramp: i32,
    right_ramp: i32,
}

/// Split `[0, len)` into `size`-long intervals overlapping by `overlap` (upstream `split_by_size`).
///
/// A short trailing tile is **grown leftward** to `min_tile`, widening its neighbour's right ramp
/// to match — an axis that divides unevenly would otherwise end in a sliver too narrow for stage 4
/// or 5 to attend over.
fn split_by_size(len: i32, size: i32, overlap: i32, min_tile: i32) -> Vec<Interval> {
    if len <= size {
        return vec![Interval {
            start: 0,
            end: len,
            left_ramp: 0,
            right_ramp: 0,
        }];
    }
    let step = size - overlap;
    let count = (len + size - 2 * overlap - 1) / step;
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count {
        let start = i * step;
        let end = if i == count - 1 { len } else { start + size };
        out.push(Interval {
            start,
            end,
            left_ramp: if i == 0 { 0 } else { overlap },
            right_ramp: if i == count - 1 { 0 } else { overlap },
        });
    }
    if out.len() >= 2 {
        let last = *out.last().expect("len >= 2");
        if last.end - last.start < min_tile {
            let new_start = (last.end - min_tile).max(0);
            let prev = out[out.len() - 2];
            let new_overlap = prev.end - new_start;
            let n = out.len();
            out[n - 2].right_ramp = new_overlap;
            out[n - 1].start = new_start;
            out[n - 1].left_ramp = new_overlap;
        }
    }
    out
}

/// Coverage / ramp consistency for a split, so a bad layout is an error rather than a dim seam.
fn validate_split(intervals: &[Interval], len: i32, min_tile: i32, axis: usize) -> Result<()> {
    let bad = |why: String| {
        Err(Error::Msg(format!(
            "ltx diffvae: axis {axis} tiling — {why}"
        )))
    };
    if intervals.is_empty() || intervals[0].start != 0 || intervals[intervals.len() - 1].end != len
    {
        return bad(format!("tiles must cover [0, {len})"));
    }
    for (i, iv) in intervals.iter().enumerate() {
        let length = iv.end - iv.start;
        if length < min_tile.min(len) {
            return bad(format!(
                "tile {i} is {length} long, below the stage-4/5 window floor {min_tile}"
            ));
        }
        // Ramps may meet (a tile narrower than twice its overlap), in which case they multiply and
        // the blend is no longer a partition of unity — which is why the driver always normalises
        // by the accumulated weight profile instead of assuming one. A ramp longer than the tile
        // itself is still nonsense.
        if iv.left_ramp < 0 || iv.right_ramp < 0 || iv.left_ramp > length || iv.right_ramp > length
        {
            return bad(format!(
                "tile {i} has ramps that do not fit its length {length}"
            ));
        }
        if i > 0 {
            let overlap = intervals[i - 1].end - iv.start;
            if overlap < 0 || intervals[i - 1].right_ramp != overlap || iv.left_ramp != overlap {
                return bad(format!("tiles {}/{i} disagree about their overlap", i - 1));
            }
        }
    }
    Ok(())
}

/// Propagate an interval through one upsample hop. `causal` applies the pixel-shuffle
/// duplicate-frame drop that `PixelShuffleUpsample` performs on the temporal axis.
fn propagate(interval: Interval, stride: i32, causal: bool) -> Interval {
    let mut out = Interval {
        start: interval.start * stride,
        end: interval.end * stride,
        left_ramp: interval.left_ramp * stride,
        right_ramp: interval.right_ramp * stride,
    };
    if causal && stride == 2 {
        out.end -= 1;
        if interval.start != 0 {
            out.start -= 1;
        }
    }
    out
}

/// Linear-ramp trapezoid of `length`, matching upstream `compute_trapezoidal_mask_1d` with
/// `left_starts_from_0=False`. Two neighbouring tiles' ramps sum to exactly 1 over their overlap.
fn trapezoid(length: i32, left_ramp: i32, right_ramp: i32) -> Vec<f32> {
    let left = left_ramp.clamp(0, length);
    let right = right_ramp.clamp(0, length);
    let mut mask = vec![1.0f32; length as usize];
    for (i, slot) in mask.iter_mut().take(left as usize).enumerate() {
        *slot *= (i + 1) as f32 / (left + 1) as f32;
    }
    for (i, slot) in mask.iter_mut().rev().take(right as usize).enumerate() {
        *slot *= (i + 1) as f32 / (right + 1) as f32;
    }
    mask
}

/// A latent grown to the stage floors and carrying its trailing ghost frames, plus the
/// `(before, after)` spatial pads — in latent cells — that decode must crop back off.
type PreparedLatent = (Array, (i32, i32), (i32, i32));

/// The LTX-2.5 `NADiffusionDecoder`.
pub struct NaDiffusionDecoder {
    cfg: NaDiffusionDecoderConfig,
    /// `per_channel_statistics`, `(1, C, 1, 1, 1)` — the encoder normalises its latent, so the
    /// decoder un-normalises before `conv_in`.
    stat_mean: Array,
    stat_std: Array,
    conv_in: Linear,
    det_stages: Vec<Vec<NaBlock>>,
    upsamples: Vec<PixelShuffleUpsample>,
    t_linear1: Linear,
    t_linear2: Linear,
    shared_adaln: Linear,
    conv_in_x_t: Linear,
    diff_blocks: Vec<DiffusionBlock>,
    norm_out: Array,
    conv_out: Linear,
}

impl NaDiffusionDecoder {
    /// Sinusoidal timestep-projection width (`Timesteps(num_channels=256, flip_sin_to_cos=True,
    /// downscale_freq_shift=0)`).
    const TIME_PROJ_DIM: usize = 256;

    /// Build from a converted `vae_diffusion_decoder.safetensors` weight map.
    pub fn from_weights(w: &Weights, cfg: &NaDiffusionDecoderConfig) -> Result<Self> {
        let stages = cfg.stage_channels.len();
        let det = stages - 1;
        let mut det_stages = Vec::with_capacity(det);
        let mut upsamples = Vec::with_capacity(det);
        for stage in 0..det {
            let kernel = cfg.stage_kernels[stage];
            let mut blocks = Vec::with_capacity(cfg.stage_depths[stage] as usize);
            for i in 0..cfg.stage_depths[stage] {
                blocks.push(NaBlock::load(
                    w,
                    &format!("det_stages.{stage}.{i}"),
                    kernel,
                    cfg.head_dim,
                )?);
            }
            det_stages.push(blocks);
            upsamples.push(PixelShuffleUpsample::load(
                w,
                &format!("upsamples.{stage}"),
                cfg.upsamples[stage].0,
            )?);
        }
        let mut diff_blocks =
            Vec::with_capacity(*cfg.stage_depths.last().expect("validated") as usize);
        for i in 0..*cfg.stage_depths.last().expect("validated") {
            diff_blocks.push(DiffusionBlock::load(
                w,
                &format!("diff_blocks.{i}"),
                cfg.stage5_kernel,
                cfg.head_dim,
            )?);
        }

        let latent_c = cfg.in_channels;
        let stat = |key: &str| -> Result<Array> {
            let a = weight(w, key)?;
            if a.size() as i32 != latent_c {
                return Err(Error::Msg(format!(
                    "ltx diffvae: {key} has {} entries, expected {latent_c}",
                    a.size()
                )));
            }
            Ok(a.reshape(&[1, latent_c, 1, 1, 1])?)
        };

        let decoder = Self {
            cfg: cfg.clone(),
            stat_mean: stat("per_channel_statistics.mean")?,
            stat_std: stat("per_channel_statistics.std")?,
            conv_in: Linear::load(w, "conv_in")?,
            det_stages,
            upsamples,
            t_linear1: Linear::load(w, "t_embedder.mlp.0")?,
            t_linear2: Linear::load(w, "t_embedder.mlp.2")?,
            shared_adaln: Linear::load(w, "shared_adaln.proj")?,
            conv_in_x_t: Linear::load(w, "conv_in_x_t")?,
            diff_blocks,
            norm_out: weight(w, "norm_out.weight")?,
            conv_out: Linear::load(w, "conv_out")?,
        };
        decoder.check_widths()?;
        Ok(decoder)
    }

    /// Cross-check the declared structure against the loaded widths. A config/weight disagreement
    /// here would otherwise surface as a reshape failure several stages deep, or — where the
    /// numbers happen to line up — as a plausible but wrong picture.
    fn check_widths(&self) -> Result<()> {
        let cfg = &self.cfg;
        let expect = |what: &str, got: i32, want: i32| -> Result<()> {
            if got != want {
                return Err(Error::Msg(format!(
                    "ltx diffvae: {what} is {got} in the weights but {want} in the config"
                )));
            }
            Ok(())
        };
        expect(
            "conv_in output width",
            self.conv_in.out_features(),
            cfg.stage_channels[0],
        )?;
        for (stage, blocks) in self.det_stages.iter().enumerate() {
            expect(
                &format!("det stage {stage} width"),
                blocks[0].attn.width(),
                cfg.stage_channels[stage],
            )?;
            expect(
                &format!("upsample {stage} output width"),
                self.upsamples[stage].out_channels(),
                cfg.stage_channels[stage + 1],
            )?;
        }
        let c5 = cfg.stage5_width();
        expect(
            "conv_in_x_t output width",
            self.conv_in_x_t.out_features(),
            c5,
        )?;
        expect(
            "conv_in_x_t input width",
            self.conv_in_x_t.w.shape()[1],
            cfg.out_channels * cfg.patch_size * cfg.patch_size,
        )?;
        expect(
            "shared_adaln output width",
            self.shared_adaln.out_features(),
            ADALN_CHUNKS * c5,
        )?;
        expect(
            "t_embedder output width",
            self.t_linear2.out_features(),
            cfg.t_emb_dim,
        )?;
        for (i, block) in self.diff_blocks.iter().enumerate() {
            expect(&format!("diff block {i} width"), block.attn.width(), c5)?;
            expect(
                &format!("diff block {i} context width"),
                block.context_proj.w.shape()[1],
                *cfg.stage_channels.last().expect("validated"),
            )?;
        }
        expect(
            "conv_out output width",
            self.conv_out.out_features(),
            cfg.out_channels * cfg.patch_size * cfg.patch_size,
        )?;
        // The SwiGLU hidden width is `round_up(mlp_ratio * dim, 16)`; a checkpoint that used a
        // different ratio would load silently and run at the wrong capacity.
        for (stage, blocks) in self.det_stages.iter().enumerate() {
            let dim = cfg.stage_channels[stage];
            expect(
                &format!("det stage {stage} mlp hidden"),
                blocks[0].mlp.w_gate.shape()[0],
                mlp_hidden(dim),
            )?;
        }
        Ok(())
    }

    /// The structure this decoder was built for.
    pub fn config(&self) -> &NaDiffusionDecoderConfig {
        &self.cfg
    }

    /// `(B, C, T, H, W) -> (B, T, H, W, C)` with the per-channel statistics undone.
    fn un_normalize(&self, latent: &Array) -> Result<Array> {
        let x = add(&multiply(latent, &self.stat_std)?, &self.stat_mean)?;
        contiguous(&x.transpose_axes(&[0, 2, 3, 4, 1])?)
    }

    /// Stages 1..=3 over the whole (already ghost-padded) latent → the stage-4 input feature.
    fn stages_1_to_3(&self, latent: &Array) -> Result<Array> {
        let mut x = self.conv_in.forward(&self.un_normalize(latent)?)?;
        for stage in 0..self.upsamples.len() - 1 {
            x = self.run_det_stage(&x, stage, true)?;
        }
        Ok(x)
    }

    fn run_det_stage(&self, x: &Array, stage: usize, drop_leading_frame: bool) -> Result<Array> {
        let mut x = x.clone();
        for block in &self.det_stages[stage] {
            x = block.forward(&x)?;
            // Stage by stage rather than one lazy graph over the whole decoder (sc-18760).
            x.eval()?;
        }
        self.upsamples[stage].forward(&x, drop_leading_frame)
    }

    /// Stage 4 → the stage-5 context, with the trailing ghost frames cropped back off.
    fn stage_4(&self, x: &Array, drop_leading_frame: bool, pad_trailing: bool) -> Result<Array> {
        let last = self.upsamples.len() - 1;
        let x = self.run_det_stage(x, last, drop_leading_frame)?;
        if !pad_trailing {
            return Ok(x);
        }
        let ghost = self.cfg.ghost_latent_frames() * self.cfg.pixel_scale()[0];
        if ghost <= 0 {
            return Ok(x);
        }
        let frames = x.shape()[1];
        let content = (frames - ghost).max(1);
        let keep = frames.min(content.max(self.cfg.stage5_kernel[0]));
        Ok(resize_axis(&x, 1, keep, false)?.0)
    }

    /// The shared AdaLN-Zero chunks for one timestep.
    fn modulation(&self, t: f32) -> Result<Vec<Array>> {
        let scaled = Array::from_slice(&[t * self.cfg.timestep_scale_multiplier], &[1]);
        let proj = timestep_sincos(&scaled, Self::TIME_PROJ_DIM, 10_000.0, 0.0)?;
        let emb = self
            .t_linear2
            .forward(&silu(&self.t_linear1.forward(&proj)?)?)?;
        let h = self.shared_adaln.forward(&silu(&emb)?)?;
        let c5 = self.cfg.stage5_width();
        (0..ADALN_CHUNKS)
            .map(|i| Ok(slice_axis(&h, 1, i * c5, (i + 1) * c5)?.reshape(&[1, 1, 1, 1, c5])?))
            .collect()
    }

    /// One deterministic block, in isolation. Exposed for parity work: a whole-decode mismatch says
    /// nothing about *where*, and the alternative — inferring it from the picture — is guesswork.
    /// `x` is channels-last `(B, T, H, W, C)` at that stage's width.
    pub fn det_block(&self, stage: usize, index: usize, x: &Array) -> Result<Array> {
        let blocks = self.det_stages.get(stage).ok_or_else(|| {
            Error::Msg(format!(
                "ltx diffvae: no deterministic stage {stage} (have {})",
                self.det_stages.len()
            ))
        })?;
        let block = blocks.get(index).ok_or_else(|| {
            Error::Msg(format!(
                "ltx diffvae: stage {stage} has {} blocks, not {index}",
                blocks.len()
            ))
        })?;
        block.forward(x)
    }

    /// One stage-5 block, in isolation, with the modulation the shared AdaLN produces at `t`.
    pub fn diffusion_block(
        &self,
        index: usize,
        context: &Array,
        x: &Array,
        t: f32,
    ) -> Result<Array> {
        let block = self.diff_blocks.get(index).ok_or_else(|| {
            Error::Msg(format!(
                "ltx diffvae: there are {} diffusion blocks, not {index}",
                self.diff_blocks.len()
            ))
        })?;
        block.forward(context, x, &self.modulation(t)?)
    }

    /// The two big intermediates of an untiled decode: the stage-1-3 feature and the stage-5
    /// context. Exposed for the same reason as [`Self::det_block`] — so a parity failure names a
    /// stage instead of a picture.
    pub fn stage_features(&self, latent: &Array) -> Result<(Array, Array)> {
        self.check_latent(latent)?;
        let (padded, _, _) = self.prepare_latent(&to_dtype(latent, Dtype::Float32)?)?;
        let feature = self.stages_1_to_3(&padded)?;
        feature.eval()?;
        let context = self.stage_4(&feature, true, true)?;
        context.eval()?;
        Ok((feature, context))
    }

    /// The seven shared AdaLN-Zero chunks at `t`, concatenated into `(1, 7 * stage5_width)` in
    /// chunk order. The block-level modulation is this plus each block's `scale_shift_table`.
    pub fn adaln_chunks(&self, t: f32) -> Result<Array> {
        let chunks = self.modulation(t)?;
        let flat: Vec<Array> = chunks
            .iter()
            .map(|c| c.reshape(&[1, -1]).map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Array> = flat.iter().collect();
        Ok(concatenate_axis(&refs, 1)?)
    }

    /// The timestep embedding itself — exposed so a port failure localises to the embedder rather
    /// than to whatever the eight stage-5 blocks did with it.
    pub fn timestep_embedding(&self, t: f32) -> Result<Array> {
        let scaled = Array::from_slice(&[t * self.cfg.timestep_scale_multiplier], &[1]);
        let proj = timestep_sincos(&scaled, Self::TIME_PROJ_DIM, 10_000.0, 0.0)?;
        self.t_linear2
            .forward(&silu(&self.t_linear1.forward(&proj)?)?)
    }

    /// One stage-5 pass: patchified `x_t` + context → the model's pixel-space prediction.
    fn diff_step(&self, context: &Array, x_t: &Array, t: f32) -> Result<Array> {
        let modulation = self.modulation(t)?;
        let patched = patchify(x_t, self.cfg.patch_size)?;
        let patched = contiguous(&patched.transpose_axes(&[0, 2, 3, 4, 1])?)?;
        let mut x = self.conv_in_x_t.forward(&patched)?;
        for block in &self.diff_blocks {
            x = block.forward(context, &x, &modulation)?;
            x.eval()?;
        }
        let x = self
            .conv_out
            .forward(&rms_norm(&x, &self.norm_out, NORM_EPS)?)?;
        let x = contiguous(&x.transpose_axes(&[0, 4, 1, 2, 3])?)?;
        unpatchify(&x, self.cfg.patch_size)
    }

    /// The reverse-diffusion schedule: `linspace(1, 1/N, N)`, computed the way `torch.linspace`
    /// does (`start + i * step`) so a multi-step schedule lands on the same floats the reference
    /// samples at.
    fn timesteps(&self) -> Vec<f32> {
        let n = self.cfg.default_num_inference_steps;
        if n == 1 {
            return vec![1.0];
        }
        let step = (1.0 / n as f32 - 1.0) / (n - 1) as f32;
        (0..n).map(|i| 1.0 + i as f32 * step).collect()
    }

    /// One Euler update: advance `x_t` from `t_now` to `t_next` given the model's prediction.
    fn euler_step(&self, x_t: &Array, model_out: &Array, t_now: f32, t_next: f32) -> Result<Array> {
        let velocity = match self.cfg.model_output_type {
            ModelOutputType::Velocity => model_out.clone(),
            ModelOutputType::X0 => divide(&subtract(x_t, model_out)?, Array::from_f32(t_now))?,
        };
        Ok(subtract(
            x_t,
            &multiply(&velocity, Array::from_f32(t_now - t_next))?,
        )?)
    }

    /// Stages 4-5 for one stage-4 feature extent.
    fn decode_one_tile(
        &self,
        feature: &Array,
        x_t_init: &Array,
        is_origin: bool,
        pad_trailing: bool,
    ) -> Result<Array> {
        let context = self.stage_4(feature, is_origin, pad_trailing)?;
        context.eval()?;
        let schedule = self.timesteps();
        let mut x_t = x_t_init.clone();
        for i in 0..schedule.len() - 1 {
            let out = self.diff_step(&context, &x_t, schedule[i])?;
            x_t = self.euler_step(&x_t, &out, schedule[i], schedule[i + 1])?;
            x_t.eval()?;
        }
        let last = *schedule.last().expect("schedule holds >= 1 step");
        let out = self.diff_step(&context, &x_t, last)?;
        match self.cfg.model_output_type {
            ModelOutputType::X0 => Ok(out),
            ModelOutputType::Velocity => self.euler_step(&x_t, &out, last, 0.0),
        }
    }

    /// Apply the latent size floor and the trailing ghost frames, returning the padded latent plus
    /// the `(before, after)` spatial pads that decode must crop back off.
    fn prepare_latent(&self, latent: &Array) -> Result<PreparedLatent> {
        let min = self.cfg.min_latent_shape();
        let (x, _) = resize_axis(latent, 2, latent.shape()[2].max(min[0]), false)?;
        let (x, h_pad) = resize_axis(&x, 3, x.shape()[3].max(min[1]), true)?;
        let (x, w_pad) = resize_axis(&x, 4, x.shape()[4].max(min[2]), true)?;
        let ghost = self.cfg.ghost_latent_frames();
        let padded = if ghost > 0 {
            resize_axis(&x, 2, x.shape()[2] + ghost, false)?.0
        } else {
            x.clone()
        };
        Ok((padded, h_pad, w_pad))
    }

    /// Crop a decoded pixel volume back to the content geometry the caller asked for.
    fn crop_to_content(
        &self,
        pixels: &Array,
        frames: i32,
        height: i32,
        width: i32,
        h_pad: (i32, i32),
        w_pad: (i32, i32),
    ) -> Result<Array> {
        let (x, _) = resize_axis(pixels, 2, frames, false)?;
        let h_before = h_pad.0 * self.cfg.pixel_scale()[1];
        let x = if h_pad.0 + h_pad.1 > 0 {
            slice_axis(&x, 3, h_before, h_before + height)?
        } else {
            resize_axis(&x, 3, height, true)?.0
        };
        let w_before = w_pad.0 * self.cfg.pixel_scale()[2];
        let x = if w_pad.0 + w_pad.1 > 0 {
            slice_axis(&x, 4, w_before, w_before + width)?
        } else {
            resize_axis(&x, 4, width, true)?.0
        };
        Ok(x)
    }

    fn check_latent(&self, latent: &Array) -> Result<()> {
        if latent.ndim() != 5 || latent.shape()[1] != self.cfg.in_channels {
            return Err(Error::Msg(format!(
                "ltx diffvae: expected a (B, {}, T, H, W) latent, got {:?}",
                self.cfg.in_channels,
                latent.shape()
            )));
        }
        Ok(())
    }

    fn check_noise(&self, noise: &Array, want: [i32; 3], latent_batch: i32) -> Result<()> {
        let want_shape = [
            latent_batch,
            self.cfg.out_channels,
            want[0],
            want[1],
            want[2],
        ];
        if noise.shape() != want_shape {
            return Err(Error::Msg(format!(
                "ltx diffvae: stage-5 noise must be {want_shape:?}, got {:?} (see \
                 NaDiffusionDecoderConfig::noise_shape)",
                noise.shape()
            )));
        }
        Ok(())
    }

    /// Untiled decode. `latent` is `(B, 128, T, H, W)`; `noise` is the stage-5 `x_t` at
    /// [`NaDiffusionDecoderConfig::noise_shape`]. Returns `(B, 3, F, H*32, W*32)` in `[-1, 1]`.
    ///
    /// The noise is an explicit argument rather than drawn internally so a decode is reproducible
    /// and comparable across backends; [`Self::decode_seeded`] is the convenience wrapper that
    /// draws it.
    pub fn decode(&self, latent: &Array, noise: &Array) -> Result<Array> {
        self.check_latent(latent)?;
        let sh = latent.shape().to_vec();
        let (b, lt, lh, lw) = (sh[0], sh[2], sh[3], sh[4]);
        self.check_noise(noise, self.cfg.noise_shape(lt, lh, lw), b)?;

        let latent_f32 = to_dtype(latent, Dtype::Float32)?;
        let (padded, h_pad, w_pad) = self.prepare_latent(&latent_f32)?;
        let feature = self.stages_1_to_3(&padded)?;
        feature.eval()?;
        let pixels =
            self.decode_one_tile(&feature, &to_dtype(noise, Dtype::Float32)?, true, true)?;
        let scale = self.cfg.pixel_scale();
        let out = self.crop_to_content(
            &pixels,
            (lt - 1) * scale[0] + 1,
            lh * scale[1],
            lw * scale[2],
            h_pad,
            w_pad,
        )?;
        contiguous(&out)
    }

    /// Tiled decode: stages 1-3 once over the whole volume, stages 4-5 per tile, blended in pixel
    /// space with separable trapezoids.
    ///
    /// Takes the same full-canvas `noise` as [`Self::decode`] and slices each tile's `x_t` out of
    /// it, so a tiled and an untiled decode of the same inputs are directly comparable — which is
    /// what makes the seam tests able to say anything at all.
    pub fn decode_tiled(
        &self,
        latent: &Array,
        noise: &Array,
        tiling: &DiffVaeTiling,
    ) -> Result<Array> {
        self.check_latent(latent)?;
        let tiling = tiling.validated(&self.cfg)?;
        let sh = latent.shape().to_vec();
        let (b, lt, lh, lw) = (sh[0], sh[2], sh[3], sh[4]);
        self.check_noise(noise, self.cfg.noise_shape(lt, lh, lw), b)?;

        let latent_f32 = to_dtype(latent, Dtype::Float32)?;
        let (padded, h_pad, w_pad) = self.prepare_latent(&latent_f32)?;
        let min = self.cfg.min_latent_shape();
        let work = [lt.max(min[0]), lh.max(min[1]), lw.max(min[2])];
        let stage4 = self.cfg.stage4_shape(work[0], work[1], work[2]);
        let canvas = self.cfg.stage5_pixel_shape(stage4, true, true);

        let feature = self.stages_1_to_3(&padded)?;
        feature.eval()?;
        let noise_f32 = to_dtype(noise, Dtype::Float32)?;

        let last = self.upsamples.len() - 1;
        let stride = self.cfg.upsamples[last].0;
        // Pixel intervals per axis: propagate the stage-4 split through the final upsample hop
        // (temporal: with the duplicate-frame drop) and, spatially, through unpatchify.
        let min_tile = self.cfg.min_tile_shape();
        let axes: Vec<Vec<(Interval, Interval)>> = (0..3)
            .map(|axis| {
                if tiling.tile[axis] < stage4[axis] && tiling.overlap[axis] >= tiling.tile[axis] {
                    return Err(Error::Msg(format!(
                        "ltx diffvae: axis {axis} splits {} cells into {}-cell tiles, so its \
                         overlap {} must be smaller than the tile",
                        stage4[axis], tiling.tile[axis], tiling.overlap[axis]
                    )));
                }
                let split = split_by_size(
                    stage4[axis],
                    tiling.tile[axis],
                    tiling.overlap[axis],
                    min_tile[axis],
                );
                validate_split(&split, stage4[axis], min_tile[axis], axis)?;
                Ok(split
                    .into_iter()
                    .map(|iv| {
                        let up = propagate(iv, stride[axis], axis == 0);
                        let px = if axis == 0 {
                            up
                        } else {
                            propagate(up, self.cfg.patch_size, false)
                        };
                        (iv, px)
                    })
                    .collect())
            })
            .collect::<Result<Vec<_>>>()?;

        let mut accumulator: Option<Array> = None;
        let mut weights: [Vec<f32>; 3] = [
            vec![0.0; canvas[0] as usize],
            vec![0.0; canvas[1] as usize],
            vec![0.0; canvas[2] as usize],
        ];
        let mut weight_seen = [false; 3];

        for (t_s4, t_px) in &axes[0] {
            let is_origin = t_s4.start == 0;
            let pad_trailing = t_s4.end == stage4[0];
            // The trailing tile keeps the ghost frames stages 1-3 produced; stage 4 crops them.
            let t_end = if pad_trailing {
                feature.shape()[1]
            } else {
                t_s4.end
            };
            let feature_t = slice_axis(&feature, 1, t_s4.start, t_end)?;
            for (h_s4, h_px) in &axes[1] {
                let feature_th = slice_axis(&feature_t, 2, h_s4.start, h_s4.end)?;
                for (w_s4, w_px) in &axes[2] {
                    let tile_feature =
                        contiguous(&slice_axis(&feature_th, 3, w_s4.start, w_s4.end)?)?;
                    let content = [
                        t_s4.end - t_s4.start,
                        h_s4.end - h_s4.start,
                        w_s4.end - w_s4.start,
                    ];
                    let stage5 = self
                        .cfg
                        .stage5_pixel_shape(content, is_origin, pad_trailing);

                    // The tile's slice of the shared canvas, edge-extended to the stage-5 extent
                    // with the same policy the size floor uses — fresh noise would make the halo
                    // inconsistent with its neighbours, which is what the blend relies on.
                    let mut x_t = slice_axis(&noise_f32, 2, t_px.start, t_px.end)?;
                    x_t = slice_axis(&x_t, 3, h_px.start, h_px.end)?;
                    x_t = slice_axis(&x_t, 4, w_px.start, w_px.end)?;
                    x_t = resize_axis(&x_t, 2, stage5[0], false)?.0;
                    x_t = resize_axis(&x_t, 3, stage5[1], true)?.0;
                    x_t = resize_axis(&x_t, 4, stage5[2], true)?.0;

                    let tile = self.decode_one_tile(
                        &tile_feature,
                        &contiguous(&x_t)?,
                        is_origin,
                        pad_trailing,
                    )?;
                    let (tile, _) = resize_axis(&tile, 2, t_px.end - t_px.start, false)?;
                    let (tile, _) = resize_axis(&tile, 3, h_px.end - h_px.start, true)?;
                    let (tile, _) = resize_axis(&tile, 4, w_px.end - w_px.start, true)?;

                    let mt = trapezoid(t_px.end - t_px.start, t_px.left_ramp, t_px.right_ramp);
                    let mh = trapezoid(h_px.end - h_px.start, h_px.left_ramp, h_px.right_ramp);
                    let mw = trapezoid(w_px.end - w_px.start, w_px.left_ramp, w_px.right_ramp);
                    let mask = multiply(
                        &multiply(
                            Array::from_slice(&mt, &[1, 1, mt.len() as i32, 1, 1]),
                            Array::from_slice(&mh, &[1, 1, 1, mh.len() as i32, 1]),
                        )?,
                        Array::from_slice(&mw, &[1, 1, 1, 1, mw.len() as i32]),
                    )?;
                    let weighted = multiply(&tile, &mask)?;

                    let placed = mlx_rs::ops::pad(
                        &weighted,
                        &[
                            (0, 0),
                            (0, 0),
                            (t_px.start, canvas[0] - t_px.end),
                            (h_px.start, canvas[1] - h_px.end),
                            (w_px.start, canvas[2] - w_px.end),
                        ][..],
                        Array::from_f32(0.0),
                        None,
                    )?;
                    accumulator = Some(match accumulator {
                        None => placed,
                        Some(acc) => add(&acc, &placed)?,
                    });
                    if let Some(acc) = accumulator.as_ref() {
                        acc.eval()?;
                    }

                    for (axis, (interval, mask_1d)) in [(*t_px, &mt), (*h_px, &mh), (*w_px, &mw)]
                        .iter()
                        .enumerate()
                    {
                        // Each axis' weight profile is accumulated once — the tile grid is a full
                        // product, so the 3-D weight is the outer product of the three profiles.
                        let counted = match axis {
                            0 => h_px.start == 0 && w_px.start == 0,
                            1 => t_px.start == 0 && w_px.start == 0,
                            _ => t_px.start == 0 && h_px.start == 0,
                        };
                        if !counted {
                            continue;
                        }
                        weight_seen[axis] = true;
                        for (i, value) in mask_1d.iter().enumerate() {
                            weights[axis][interval.start as usize + i] += value;
                        }
                    }
                }
            }
        }

        let accumulator = accumulator
            .ok_or_else(|| Error::Msg("ltx diffvae: tiled decode produced no tiles".to_string()))?;
        debug_assert!(weight_seen.iter().all(|&s| s));
        let weight_3d = multiply(
            &multiply(
                Array::from_slice(&weights[0], &[1, 1, canvas[0], 1, 1]),
                Array::from_slice(&weights[1], &[1, 1, 1, canvas[1], 1]),
            )?,
            Array::from_slice(&weights[2], &[1, 1, 1, 1, canvas[2]]),
        )?;
        let blended = divide(&accumulator, &weight_3d)?;

        let scale = self.cfg.pixel_scale();
        let out = self.crop_to_content(
            &blended,
            (lt - 1) * scale[0] + 1,
            lh * scale[1],
            lw * scale[2],
            h_pad,
            w_pad,
        )?;
        contiguous(&out)
    }

    /// [`Self::decode`] / [`Self::decode_tiled`] with the stage-5 noise drawn from `seed`.
    pub fn decode_seeded(
        &self,
        latent: &Array,
        seed: u64,
        tiling: Option<&DiffVaeTiling>,
    ) -> Result<Array> {
        self.check_latent(latent)?;
        let sh = latent.shape().to_vec();
        let shape5 = self.cfg.noise_shape(sh[2], sh[3], sh[4]);
        let key = mlx_rs::random::key(seed)?;
        let noise = mlx_rs::random::normal::<f32>(
            &[
                sh[0],
                self.cfg.out_channels,
                shape5[0],
                shape5[1],
                shape5[2],
            ],
            None,
            None,
            Some(&key),
        )?;
        match tiling {
            Some(t) => self.decode_tiled(latent, &noise, t),
            None => self.decode(latent, &noise),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Loader plumbing
// ---------------------------------------------------------------------------------------------

/// The component name [`crate::convert::convert_vae_components`] writes an `NADiffusionDecoder`
/// under. Deliberately **not** `vae_decoder`: that name is the conv decoder's, and a directory
/// where the two were interchangeable is one where a mis-selected file renders garbage.
pub const DIFFUSION_DECODER_COMPONENT: &str = "vae_diffusion_decoder";

/// Keys the converter carries through but the decoder never reads.
///
/// `type_emb` is a 128-wide vector the checkpoint ships and no reference module consumes — the
/// upstream loader drops it on a non-strict `load_state_dict`. It is listed here so the unused-key
/// audit in `tests/ltx_2_5_vae_conformance.rs` distinguishes "known dead weight" from "the port
/// forgot to load something".
pub const UNUSED_DECODER_KEYS: &[&str] = &["type_emb"];

fn push_linear_keys(keys: &mut Vec<String>, prefix: &str) {
    keys.push(format!("{prefix}.weight"));
    keys.push(format!("{prefix}.bias"));
}

fn push_block_keys(keys: &mut Vec<String>, prefix: &str, diffusion: bool) {
    keys.push(format!("{prefix}.norm1.weight"));
    keys.push(format!("{prefix}.norm2.weight"));
    push_linear_keys(keys, &format!("{prefix}.attn.qkv"));
    push_linear_keys(keys, &format!("{prefix}.attn.proj"));
    keys.push(format!("{prefix}.attn.q_norm.weight"));
    keys.push(format!("{prefix}.attn.k_norm.weight"));
    keys.push(format!("{prefix}.mlp.w_gate.weight"));
    keys.push(format!("{prefix}.mlp.w_up.weight"));
    keys.push(format!("{prefix}.mlp.w_down.weight"));
    if diffusion {
        push_linear_keys(keys, &format!("{prefix}.context_proj"));
        keys.push(format!("{prefix}.scale_shift_table"));
    }
}

/// Every weight key [`NaDiffusionDecoder::from_weights`] reads for `cfg`, for loader audits.
pub fn expected_weight_keys(cfg: &NaDiffusionDecoderConfig) -> Vec<String> {
    let mut keys = vec![
        "per_channel_statistics.mean".to_string(),
        "per_channel_statistics.std".to_string(),
        "norm_out.weight".to_string(),
    ];
    for prefix in [
        "conv_in",
        "conv_in_x_t",
        "conv_out",
        "shared_adaln.proj",
        "t_embedder.mlp.0",
        "t_embedder.mlp.2",
    ] {
        push_linear_keys(&mut keys, prefix);
    }
    for stage in 0..cfg.upsamples.len() {
        for i in 0..cfg.stage_depths[stage] {
            push_block_keys(&mut keys, &format!("det_stages.{stage}.{i}"), false);
        }
        push_linear_keys(&mut keys, &format!("upsamples.{stage}.proj"));
    }
    for i in 0..*cfg.stage_depths.last().expect("validated") {
        push_block_keys(&mut keys, &format!("diff_blocks.{i}"), true);
    }
    keys.sort();
    keys
}

/// Classify a `Weights` map as an `NADiffusionDecoder` — used by the converter and by loaders that
/// must tell the two 2.5 video decoders apart from the tensors alone.
pub fn looks_like_diffusion_decoder(w: &Weights) -> bool {
    let mut det = false;
    let mut diff = false;
    for key in w.keys() {
        det |= key.starts_with("det_stages.");
        diff |= key.starts_with("diff_blocks.");
    }
    det && diff
}

#[cfg(test)]
mod tests;
