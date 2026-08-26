//! LTX-2.5 **DiffVAE** video decoder — `NADiffusionDecoder` on candle/CUDA (sc-18767).
//!
//! The candle twin of `mlx_gen_ltx::diff_vae` (sc-18766, inference PR #696). Same operator, same
//! window rule, same tiling arithmetic, and the **same committed absolute-error goldens** — this
//! crate's parity test loads `mlx-gen-ltx/tests/fixtures/ltx25_diffvae_golden.safetensors` by path
//! rather than re-recording it, which is what makes a cross-backend divergence a red test rather
//! than two independently plausible pictures.
//!
//! ## Shape of the thing
//!
//! Five stages, all **channels-last** `(B, T, H, W, C)` — the reference's own layout, and the one
//! every projection here is a plain `[out, in]` GEMM against:
//!
//! | | stage 1 | stage 2 | stage 3 | stage 4 | stage 5 |
//! | --- | --- | --- | --- | --- | --- |
//! | channels | 2048 | 1024 | 512 | 512 | 256 |
//! | blocks | 4 | 6 | 4 | 2 | 8 |
//! | 3-D window | 3x7x7 | 3x7x7 | 3x5x5 | 3x5x5 | 11x11x11 |
//!
//! Stages 1-4 are **deterministic**: pre-norm blocks of 3-D neighborhood attention + SwiGLU, each
//! followed by a `Linear` + channels-last pixel-shuffle upsample (strides `1x2x2`, `2x1x1`,
//! `2x2x2`, `2x2x2`). Their output is the *context* volume. Stage 5 is the **diffusion** stage:
//! eight blocks that denoise patchified noisy pixels `x_t`, injecting the context through a
//! per-block `context_proj` and modulating on a shared AdaLN-Zero built from the timestep.
//! `model_output_type: x0` with `default_num_inference_steps: 1` means the shipped checkpoint runs
//! exactly one step and returns its prediction — the Euler loop is implemented anyway, so a
//! `v`-parameterised or multi-step checkpoint runs correctly rather than silently wrongly.
//!
//! ## Loading: no conversion step
//!
//! Unlike the MLX port — whose weights are channels-last and whose loader therefore needs
//! `convert_vae_components` — every tensor this decoder reads is a PyTorch-layout `[out, in]`
//! matrix or a 1-D norm/bias, so the released `vae/ltx-2.5-video-vae-bf16.safetensors` is consumed
//! **verbatim**: a [`VarBuilder`] rooted at `decoder.` plus the file-root
//! `per_channel_statistics.{mean,std}-of-means`. That is the same "no remap on candle" property
//! sc-18765 established for the conv VAE, and [`NaDiffusionDecoder::load`] is where it is asserted
//! (a checkpoint missing any key is a load error, never a default).
//!
//! ## Neighborhood attention: implemented here, no kernel library
//!
//! Upstream backs the 3-D windows with NATTEN/CUTLASS-FNA. That is a PyTorch CUDA extension with
//! no Rust binding, and vendoring its kernels would put a second CUDA build system and a second
//! licence inside a crate whose only other GPU code is candle's own. [`na3d`] is therefore this
//! port's own implementation of the same operator, mirroring the MLX one so the two backends agree
//! **by construction** rather than by two independent readings of NATTEN's border rule: for each
//! query the attended window is `[clamp(i - k/2, 0, L - k), + k)` per axis — the window **slides
//! inward** at the border and keeps its full size, instead of being clipped and renormalised.
//! That is NATTEN's rule, it is what upstream's own vendored `fallback_na.eager` backend
//! implements, and it is therefore what the committed goldens encode.
//!
//! Queries are tiled; every tile shares one additive mask assembled from three tiny per-axis masks.
//! candle has no fused SDPA on the paths this crate builds for, so unlike MLX the `[Nq, Nk]` score
//! matrix **does** land — [`NA_SCORE_BUDGET`] bounds it by additionally chunking the head axis,
//! which is the one place this port's memory schedule legitimately differs from the MLX one.
//!
//! ## Memory
//!
//! Everything runs **f32**, the crate's VAE convention. The lever for large geometries is
//! [`NaDiffusionDecoder::decode_tiled`], not precision: stages 1-3 run once over the whole volume,
//! and stages 4-5 run per tile with a separable trapezoidal pixel blend normalised by the
//! accumulated weight profile. The temporal axis is tiled with the **same halo discipline** as the
//! spatial ones — a tiler that starves time produces a plausible-looking but temporally smeared
//! clip instead of an error.
//!
//! Reference: `Lightricks/LTX-2` v1.2.0, commit `d151147788a9284cca791edc6ce898007e727fe6`,
//! `packages/ltx-core/src/ltx_core/model/video_vae/{diffusion_video_decoder,diffusion_tiling}.py`
//! and `.../video_vae/transformer/*`. Goldens: `mlx-gen/tools/dump_ltx25_diffvae_golden.py`.

use std::path::Path;

use serde_json::Value;

use candle_gen::candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen::candle_nn::ops::{rms_norm, silu, softmax_last_dim};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::ltx_checkpoint::LtxCheckpointMetadata;

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
pub const VIDEO_SCALE_FACTORS: [usize; 3] = [8, 32, 32];

/// SwiGLU `mlp_ratio` (`NABlock` / `DiffusionNABlock`): `hidden = round_up(4 * dim, 16)`.
const MLP_RATIO: usize = 4;

/// The SwiGLU hidden width a block of `dim` channels declares.
fn mlp_hidden(dim: usize) -> usize {
    (MLP_RATIO * dim).div_ceil(16) * 16
}

/// Token-tile for the SwiGLU hidden buffer (upstream `DEFAULT_SWIGLU_TILE_SIZE`). Bounds the
/// `[tokens, 4*dim]` intermediate, which at stage-5 production geometry would otherwise be the
/// single largest allocation in the decoder.
const SWIGLU_TILE_TOKENS: usize = 16_384;

/// Per-tile `Nq * Nk` budget for [`na3d`]'s **mask**, mirroring the MLX port: one additive mask is
/// capped at 64 MiB of f32 while tiles stay large enough that the attention is still a real GEMM.
const NA_TILE_BUDGET: usize = 1 << 24;

/// Per-chunk `heads * Nq * Nk` budget for [`na3d`]'s **scores**.
///
/// This has no MLX counterpart on purpose. MLX's `scaled_dot_product_attention` is fused, so there
/// the `[Nq, Nk]` scores never land and only the mask needs bounding. candle materialises
/// `q @ kᵀ` for every head at once, so without a second bound a stage-1 tile (32 heads) would
/// allocate 32x the mask. Chunking the head axis keeps that buffer at ~256 MiB of f32 and changes
/// nothing numerically — each head's softmax is independent.
pub const NA_SCORE_BUDGET: usize = 1 << 26;

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
    pub in_channels: usize,
    /// RGB channels leaving `conv_out` after unpatchify (3).
    pub out_channels: usize,
    /// Spatial patch size of the stage-5 pixel grid (4).
    pub patch_size: usize,
    /// Attention head width; every stage channel count must be a multiple of it (64).
    pub head_dim: usize,
    /// Per-stage channels, stages 1..=5.
    pub stage_channels: Vec<usize>,
    /// Per-stage block depths, stages 1..=5.
    pub stage_depths: Vec<usize>,
    /// Per-stage 3-D window `(K_t, K_h, K_w)` for the deterministic stages.
    pub stage_kernels: Vec<[usize; 3]>,
    /// `(stride, out-channel reduction)` per upsample hop — one fewer than the stage count.
    pub upsamples: Vec<([usize; 3], usize)>,
    /// The diffusion stage's own window, which is wider than any deterministic stage's.
    pub stage5_kernel: [usize; 3],
    /// Explicit stage-5 width; `None` means "the last stage channel count".
    pub stage5_channels: Option<usize>,
    /// Timestep-embedding width feeding the shared AdaLN-Zero (384).
    pub t_emb_dim: usize,
    /// Steps the shipped schedule runs (`linspace(1, 1/N, N)`); 1 for the released checkpoint.
    pub default_num_inference_steps: usize,
    /// Timesteps are multiplied by this before the sinusoidal embedding (1000.0).
    pub timestep_scale_multiplier: f64,
    /// What the stage-5 blocks predict.
    pub model_output_type: ModelOutputType,
}

fn get_usize(v: &Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(Value::as_u64)
        .map_or(default, |x| x as usize)
}

fn triple(v: &Value, what: &str) -> Result<[usize; 3]> {
    let arr = v.as_array().filter(|a| a.len() == 3).ok_or_else(|| {
        Error::Msg(format!(
            "ltx diffvae: {what} must be a 3-element array, got {v}"
        ))
    })?;
    let mut out = [0usize; 3];
    for (slot, item) in out.iter_mut().zip(arr) {
        *slot = item.as_u64().ok_or_else(|| {
            Error::Msg(format!(
                "ltx diffvae: {what} entries must be non-negative integers, got {v}"
            ))
        })? as usize;
    }
    Ok(out)
}

impl NaDiffusionDecoderConfig {
    /// Parse the `vae` block of a checkpoint's `__metadata__["config"]`.
    ///
    /// Every architecture field is **required** — the stage ladder, and also the two fields that
    /// parameterise the sampler rather than the shape (`vae.model_output_type` and
    /// `vae.decoder.timestep_scale_multiplier`). Unlike the conv VAE — where an absent block list
    /// means "a 2.3 tree that predates the embedded config", so the 2.3 defaults are the right
    /// answer — there has never been a default `NADiffusionDecoder`, and inventing one would build
    /// a differently-shaped or differently-sampled decoder against real weights: defaulting
    /// `model_output_type` to `v` against the released `x0` checkpoint, or the scale multiplier to
    /// `1.0` against the released `1000.0`, decodes silently wrongly rather than failing.
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
        let list_of = |key: &str| -> Result<Vec<usize>> {
            require(key)?
                .as_array()
                .ok_or_else(|| {
                    Error::Msg(format!("ltx diffvae: vae.decoder.{key} must be an array"))
                })?
                .iter()
                .map(|x| {
                    x.as_u64().map(|v| v as usize).ok_or_else(|| {
                        Error::Msg(format!(
                            "ltx diffvae: vae.decoder.{key} must hold non-negative integers"
                        ))
                    })
                })
                .collect()
        };

        let stage_channels = list_of("stage_channels")?;
        let stage_depths = list_of("stage_depths")?;
        let stage_kernels = require("stage_kernels")?
            .as_array()
            .ok_or_else(|| {
                Error::Msg("ltx diffvae: vae.decoder.stage_kernels must be an array".to_string())
            })?
            .iter()
            .map(|k| triple(k, "vae.decoder.stage_kernels[]"))
            .collect::<Result<Vec<_>>>()?;
        let upsamples = require("upsamples")?
            .as_array()
            .ok_or_else(|| {
                Error::Msg("ltx diffvae: vae.decoder.upsamples must be an array".to_string())
            })?
            .iter()
            .map(|entry| {
                let pair = entry.as_array().filter(|a| a.len() == 2).ok_or_else(|| {
                    Error::Msg(format!(
                        "ltx diffvae: each vae.decoder.upsamples entry must be \
                         [stride, reduction], got {entry}"
                    ))
                })?;
                let stride = triple(&pair[0], "vae.decoder.upsamples[][0]")?;
                let reduction = pair[1].as_u64().ok_or_else(|| {
                    Error::Msg(
                        "ltx diffvae: upsample reduction must be a non-negative integer"
                            .to_string(),
                    )
                })? as usize;
                Ok((stride, reduction))
            })
            .collect::<Result<Vec<_>>>()?;

        let cfg = Self {
            in_channels: get_usize(dec, "in_channels", 128),
            out_channels: get_usize(dec, "out_channels", 3),
            patch_size: get_usize(dec, "patch_size", 4),
            head_dim: get_usize(dec, "head_dim", get_usize(dec, "na_head_dim", 64)),
            stage_channels,
            stage_depths,
            stage_kernels,
            upsamples,
            stage5_kernel: triple(require("stage5_kernel")?, "vae.decoder.stage5_kernel")?,
            stage5_channels: dec
                .get("stage5_channels")
                .and_then(Value::as_u64)
                .map(|x| x as usize),
            t_emb_dim: get_usize(dec, "t_emb_dim", 384),
            default_num_inference_steps: get_usize(dec, "default_num_inference_steps", 2),
            timestep_scale_multiplier: require("timestep_scale_multiplier")?.as_f64().ok_or_else(
                || {
                    Error::Msg(
                        "ltx diffvae: vae.decoder.timestep_scale_multiplier must be a number"
                            .to_string(),
                    )
                },
            )?,
            model_output_type: ModelOutputType::parse(
                v.get("model_output_type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        Error::Msg(
                            "ltx diffvae: config.vae has no `model_output_type` (x0 | v); it \
                             parameterises the sampler and must never be guessed"
                                .to_string(),
                        )
                    })?,
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

    /// Read the structure straight out of a released component checkpoint's
    /// `__metadata__["config"]["vae"]` — the candle path, with no conversion step in between.
    pub fn from_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let meta =
            LtxCheckpointMetadata::from_file(path).map_err(|e| Error::Msg(format!("{e}")))?;
        let vae = meta.section("vae").ok_or_else(|| {
            Error::Msg(format!(
                "ltx diffvae: {} carries no __metadata__[\"config\"][\"vae\"] block",
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
        if self.head_dim < 2 || !self.head_dim.is_multiple_of(2) {
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
            if c == 0 || c % self.head_dim != 0 {
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
            return Err(Error::Msg(
                "ltx diffvae: default_num_inference_steps must be >= 1, got 0".to_string(),
            ));
        }
        if self.patch_size < 1 || self.out_channels < 1 || self.in_channels < 1 {
            return Err(Error::Msg(
                "ltx diffvae: patch_size / in_channels / out_channels must be >= 1".to_string(),
            ));
        }
        Ok(self)
    }

    /// Stage-5 feature width: explicit if declared, else the last stage's channels.
    pub fn stage5_width(&self) -> usize {
        self.stage5_channels
            .unwrap_or_else(|| *self.stage_channels.last().expect("validated: >= 2 stages"))
    }

    /// Latent frames replicated at the tail before stages 1-4, to keep the last real frame off the
    /// neighborhood window's shifted border. `(stage_kernels[0].t / 2) * 2`, per upstream.
    pub fn ghost_latent_frames(&self) -> usize {
        (self.stage_kernels[0][0] / 2) * 2
    }

    /// Per-axis latent floor so every stage's window fits its grid
    /// (`diffusion_tiling.all_stages_min_tile_size`).
    pub fn min_latent_shape(&self) -> [usize; 3] {
        let mut cumulative = [1usize; 3];
        let mut mins = [1usize; 3];
        for (stage, kernel) in self
            .stage_kernels
            .iter()
            .enumerate()
            .take(self.upsamples.len())
        {
            for axis in 0..3 {
                mins[axis] = mins[axis].max(kernel[axis].div_ceil(cumulative[axis]));
            }
            let stride = self.upsamples[stage].0;
            for (axis, slot) in cumulative.iter_mut().enumerate() {
                *slot *= stride[axis];
            }
        }
        for axis in 0..3 {
            mins[axis] = mins[axis].max(self.stage5_kernel[axis].div_ceil(cumulative[axis]));
        }
        mins
    }

    /// Minimum stage-4-input extent so stages 4 and 5 each see at least their window
    /// (`diffusion_tiling.compute_tile_min_size`).
    pub fn min_tile_shape(&self) -> [usize; 3] {
        let last = self.upsamples.len() - 1;
        let stride = self.upsamples[last].0;
        let kernel4 = self.stage_kernels[last];
        let mut out = [0usize; 3];
        for axis in 0..3 {
            out[axis] = kernel4[axis].max(self.stage5_kernel[axis].div_ceil(stride[axis]));
        }
        out
    }

    /// One-sided stage-4/5 halos in stage-4-input units (`diffusion_tiling.compute_tile_halos`),
    /// reduced to the dominant per-axis halo — the minimum overlap a tiling may use.
    pub fn tile_halo(&self) -> [usize; 3] {
        let last = self.upsamples.len() - 1;
        let stride = self.upsamples[last].0;
        let kernel4 = self.stage_kernels[last];
        let depth4 = self.stage_depths[last];
        let depth5 = *self.stage_depths.last().expect("validated");
        let mut out = [0usize; 3];
        for axis in 0..3 {
            let halo4 = depth4 * (kernel4[axis] / 2);
            let halo5 = (depth5 * (self.stage5_kernel[axis] / 2)).div_ceil(stride[axis]);
            out[axis] = halo4.max(halo5);
        }
        out
    }

    /// Pixels (and frames) per latent cell, from the ladder's own hops: the product of the upsample
    /// strides, spatially times the patch size. Equals [`VIDEO_SCALE_FACTORS`] for the released
    /// checkpoint.
    pub fn pixel_scale(&self) -> [usize; 3] {
        let mut scale = [1usize; 3];
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
    pub fn stage4_shape(&self, latent_t: usize, latent_h: usize, latent_w: usize) -> [usize; 3] {
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
        stage4: [usize; 3],
        drop_leading_frame: bool,
        pad_trailing: bool,
    ) -> [usize; 3] {
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
    pub fn noise_shape(&self, latent_t: usize, latent_h: usize, latent_w: usize) -> [usize; 3] {
        let min = self.min_latent_shape();
        let stage4 = self.stage4_shape(
            latent_t.max(min[0]),
            latent_h.max(min[1]),
            latent_w.max(min[2]),
        );
        self.stage5_pixel_shape(stage4, true, true)
    }
}

// ---------------------------------------------------------------------------------------------
// Small layers
// ---------------------------------------------------------------------------------------------

/// `y = x Wᵀ (+ b)` over a channels-last tensor of any rank: the trailing axis is the feature axis
/// and every leading axis is a token. Flattened to one `[tokens, in]` GEMM rather than a broadcast
/// matmul, which candle would otherwise expand into `tokens / T` separate calls.
fn affine(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let in_features = *dims.last().expect("rank >= 1");
    let tokens: usize = dims[..dims.len() - 1].iter().product();
    let flat = x.contiguous()?.reshape((tokens, in_features))?;
    let y = flat.matmul(&w.t()?.contiguous()?)?;
    let y = match b {
        Some(bias) => y.broadcast_add(&bias.reshape((1, bias.elem_count()))?)?,
        None => y,
    };
    let mut out_dims = dims;
    *out_dims.last_mut().expect("rank >= 1") = w.dim(0)?;
    y.reshape(out_dims)
}

/// `[out, in]` weight + `[out]` bias.
#[derive(Debug)]
struct Linear {
    w: Tensor,
    b: Tensor,
}

impl Linear {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let w = vb.get_unchecked(&format!("{prefix}.weight"))?;
        let (out, _) = w.dims2()?;
        Ok(Self {
            b: vb.get(out, &format!("{prefix}.bias"))?,
            w,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        affine(x, &self.w, Some(&self.b))
    }

    fn out_features(&self) -> usize {
        self.w.dims()[0]
    }

    fn in_features(&self) -> usize {
        self.w.dims()[1]
    }
}

/// `w_down(silu(x W_gateᵀ) * (x W_upᵀ))`, tiled over tokens so the `[tokens, hidden]` intermediate
/// stays bounded (upstream `swiglu_tiled`). Tiling changes nothing numerically — each token's
/// result depends only on itself.
#[derive(Debug)]
struct SwiGlu {
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
}

impl SwiGlu {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            w_gate: vb.get_unchecked(&format!("{prefix}.w_gate.weight"))?,
            w_up: vb.get_unchecked(&format!("{prefix}.w_up.weight"))?,
            w_down: vb.get_unchecked(&format!("{prefix}.w_down.weight"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let dim = *dims.last().expect("rank >= 1");
        let tokens: usize = dims[..dims.len() - 1].iter().product();
        let flat = x.contiguous()?.reshape((tokens, dim))?;
        if tokens <= SWIGLU_TILE_TOKENS {
            return self.tile(&flat)?.reshape(dims);
        }
        let mut parts: Vec<Tensor> = Vec::new();
        let mut start = 0;
        while start < tokens {
            let len = SWIGLU_TILE_TOKENS.min(tokens - start);
            parts.push(self.tile(&flat.narrow(0, start, len)?)?);
            start += len;
        }
        Tensor::cat(&parts, 0)?.reshape(dims)
    }

    fn tile(&self, flat: &Tensor) -> Result<Tensor> {
        let gate = silu(&affine(flat, &self.w_gate, None)?)?;
        let up = affine(flat, &self.w_up, None)?;
        affine(&(gate * up)?, &self.w_down, None)
    }
}

/// Slice `x` along `axis` to `[start, end)`.
fn slice_axis(x: &Tensor, axis: usize, start: usize, end: usize) -> Result<Tensor> {
    let len = x.dims()[axis];
    if end > len || start >= end {
        return Err(Error::Msg(format!(
            "ltx diffvae: slice [{start}, {end}) is out of range for axis {axis} of length {len}"
        )));
    }
    if start == 0 && end == len {
        return Ok(x.clone());
    }
    x.narrow(axis, start, end - start)
}

/// Pad or crop `axis` to `size`. `symmetric` splits the difference across both edges
/// (`before = need / 2`), replicating the edge slice; otherwise the tail is repeated or dropped.
/// Returns the resized tensor and the `(before, after)` element counts, in that axis's units.
fn resize_axis(
    x: &Tensor,
    axis: usize,
    size: usize,
    symmetric: bool,
) -> Result<(Tensor, (usize, usize))> {
    let len = x.dims()[axis];
    if size < 1 {
        return Err(Error::Msg(
            "ltx diffvae: resize target must be >= 1, got 0".to_string(),
        ));
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
        let mut parts: Vec<Tensor> = Vec::new();
        if before > 0 {
            let edge = slice_axis(x, axis, 0, 1)?;
            parts.extend(std::iter::repeat_n(edge, before));
        }
        parts.push(x.clone());
        if after > 0 {
            let edge = slice_axis(x, axis, len - 1, len)?;
            parts.extend(std::iter::repeat_n(edge, after));
        }
        return Ok((Tensor::cat(&parts, axis)?, (before, after)));
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
fn window_starts(len: usize, kernel: usize) -> Vec<usize> {
    let k = kernel.min(len);
    let lo = len - k;
    let half = k / 2;
    (0..len).map(|i| i.saturating_sub(half).min(lo)).collect()
}

/// Query-tile extents keeping one tile's `Nq * Nk` under `tile_budget` (production:
/// [`NA_TILE_BUDGET`]). Halves the axis with the largest tile-to-window ratio, which is the one
/// paying the most for its halo.
fn pick_tiles(dims: [usize; 3], kernels: [usize; 3], tile_budget: usize) -> [usize; 3] {
    let mut tiles = dims;
    let cost = |t: [usize; 3]| -> u128 {
        let nq: u128 = t.iter().map(|&x| x as u128).product();
        let nk: u128 = (0..3)
            .map(|a| dims[a].min(t[a] + kernels[a] - 1) as u128)
            .product();
        nq.saturating_mul(nk)
    };
    while cost(tiles) > tile_budget as u128 && tiles.iter().any(|&t| t > 1) {
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
        tiles[best] = tiles[best].div_ceil(2).max(1);
    }
    tiles
}

/// Additive `[n, key_len]` visibility mask for one axis of one tile: `0` inside the window,
/// [`MASK_NEG`] outside.
fn axis_mask(
    starts: &[usize],
    q0: usize,
    q1: usize,
    kernel_eff: usize,
    key_start: usize,
    key_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let n = q1 - q0;
    let mut data = vec![MASK_NEG; n * key_len];
    for (j, row) in data.chunks_mut(key_len).enumerate() {
        let lo = starts[q0 + j] - key_start;
        for slot in row.iter_mut().skip(lo).take(kernel_eff) {
            *slot = 0.0;
        }
    }
    Tensor::from_vec(data, (n, key_len), device)
}

/// 3-D neighborhood attention over `(B, T, H, W, NH, HD)` tensors, returning
/// `(B, T, H, W, NH * HD)`.
///
/// `q` must already carry the `head_dim^-0.5` scale, and both `q` and `k` their rotary positions.
/// Semantics are NATTEN `na3d`'s: each query attends `[clamp(i - k/2, 0, L - k), + k)` on every
/// axis, so at the border the window slides inward and keeps its full size instead of being clipped
/// and renormalised — the same rule `mlx_gen_ltx::diff_vae::na3d` implements, which is what makes
/// the two backends comparable against one set of goldens.
pub fn na3d(q: &Tensor, k: &Tensor, v: &Tensor, kernel: [usize; 3]) -> Result<Tensor> {
    na3d_with_budgets(q, k, v, kernel, NA_TILE_BUDGET, NA_SCORE_BUDGET)
}

/// How many of the `heads` flattened `(batch, head)` rows one score matmul may carry, given that
/// each row materialises `per_head` scores and the whole chunk must stay under `score_budget`.
fn head_chunk(per_head: usize, score_budget: usize, heads: usize) -> usize {
    (score_budget / per_head.max(1)).clamp(1, heads)
}

/// [`na3d`] with its two schedule budgets supplied rather than read from the consts.
///
/// Production always takes [`na3d`]. The budgets are a parameter so a test can force the
/// query-tiled / head-chunked schedule — the one real geometry takes, and the only place the
/// candle and MLX ports differ in *how* they compute the same operator — at a size a CPU unit test
/// can also brute-force.
fn na3d_with_budgets(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    kernel: [usize; 3],
    tile_budget: usize,
    score_budget: usize,
) -> Result<Tensor> {
    let sh = q.dims().to_vec();
    if sh.len() != 6 {
        return Err(Error::Msg(format!(
            "ltx diffvae: na3d expects (B, T, H, W, NH, HD), got {sh:?}"
        )));
    }
    let (b, t, h, w, nh, hd) = (sh[0], sh[1], sh[2], sh[3], sh[4], sh[5]);
    let device = q.device().clone();
    let dims = [t, h, w];
    let mut kernels = [0usize; 3];
    for axis in 0..3 {
        if dims[axis] < kernel[axis] {
            return Err(Error::Msg(format!(
                "ltx diffvae: 3-D neighborhood attention needs each dim >= its window; got \
                 (T,H,W)=({t},{h},{w}) vs kernel {kernel:?}"
            )));
        }
        kernels[axis] = kernel[axis].min(dims[axis]);
    }
    let starts: Vec<Vec<usize>> = (0..3).map(|a| window_starts(dims[a], kernels[a])).collect();
    let tiles = pick_tiles(dims, kernels, tile_budget);

    // Key extent covering a query tile: from the first query's window start to the last query's
    // window end.
    let key_range = |axis: usize, q0: usize, q1: usize| -> (usize, usize) {
        let lo = starts[axis][q0];
        let hi = starts[axis][q1 - 1] + kernels[axis];
        (lo, hi - lo)
    };

    let mut t_parts: Vec<Tensor> = Vec::new();
    let mut t0 = 0;
    while t0 < t {
        let t1 = (t0 + tiles[0]).min(t);
        let (kt0, ktn) = key_range(0, t0, t1);
        let qt = slice_axis(q, 1, t0, t1)?;
        let kt = slice_axis(k, 1, kt0, kt0 + ktn)?;
        let vt = slice_axis(v, 1, kt0, kt0 + ktn)?;
        let mt = axis_mask(&starts[0], t0, t1, kernels[0], kt0, ktn, &device)?;

        let mut h_parts: Vec<Tensor> = Vec::new();
        let mut h0 = 0;
        while h0 < h {
            let h1 = (h0 + tiles[1]).min(h);
            let (kh0, khn) = key_range(1, h0, h1);
            let qth = slice_axis(&qt, 2, h0, h1)?;
            let kth = slice_axis(&kt, 2, kh0, kh0 + khn)?;
            let vth = slice_axis(&vt, 2, kh0, kh0 + khn)?;
            let mh = axis_mask(&starts[1], h0, h1, kernels[1], kh0, khn, &device)?;

            let mut w_parts: Vec<Tensor> = Vec::new();
            let mut w0 = 0;
            while w0 < w {
                let w1 = (w0 + tiles[2]).min(w);
                let (kw0, kwn) = key_range(2, w0, w1);
                let mw = axis_mask(&starts[2], w0, w1, kernels[2], kw0, kwn, &device)?;

                let (nt, nhq, nw) = (t1 - t0, h1 - h0, w1 - w0);
                let nq = nt * nhq * nw;
                let nk = ktn * khn * kwn;
                // Separable additive mask: `[nq, nk]` assembled by broadcasting three small
                // per-axis masks, so the only 2-D allocation is the result.
                let mask = mt
                    .reshape((nt, 1, 1, ktn, 1, 1))?
                    .broadcast_add(&mh.reshape((1, nhq, 1, 1, khn, 1))?)?
                    .broadcast_add(&mw.reshape((1, 1, nw, 1, 1, kwn))?)?
                    .reshape((1, nq, nk))?;

                // `(B, nq, NH, HD)` -> `(B * NH, nq, HD)`: one flat batch of per-head attentions.
                let flatten = |x: &Tensor, n: usize| -> Result<Tensor> {
                    x.contiguous()?
                        .reshape((b, n, nh, hd))?
                        .permute((0usize, 2, 1, 3))?
                        .contiguous()?
                        .reshape((b * nh, n, hd))
                };
                let qs = flatten(&slice_axis(&qth, 3, w0, w1)?, nq)?;
                let ks = flatten(&slice_axis(&kth, 3, kw0, kw0 + kwn)?, nk)?;
                let vs = flatten(&slice_axis(&vth, 3, kw0, kw0 + kwn)?, nk)?;
                let kt_s = ks.transpose(1, 2)?.contiguous()?;

                // candle materialises the scores, so the head axis is chunked as well as the query
                // axis — see `NA_SCORE_BUDGET`.
                let per_head = nq * nk;
                let chunk = head_chunk(per_head, score_budget, b * nh);
                let mut head_parts: Vec<Tensor> = Vec::new();
                let mut head0 = 0;
                while head0 < b * nh {
                    let n_heads = chunk.min(b * nh - head0);
                    let scores = qs
                        .narrow(0, head0, n_heads)?
                        .matmul(&kt_s.narrow(0, head0, n_heads)?)?
                        .broadcast_add(&mask)?;
                    let probs = softmax_last_dim(&scores)?;
                    head_parts.push(probs.matmul(&vs.narrow(0, head0, n_heads)?)?);
                    head0 += n_heads;
                }
                let out = if head_parts.len() == 1 {
                    head_parts.remove(0)
                } else {
                    Tensor::cat(&head_parts, 0)?
                };
                let out = out
                    .reshape((b, nh, nq, hd))?
                    .permute((0usize, 2, 1, 3))?
                    .contiguous()?
                    .reshape((b, nt, nhq, nw, nh * hd))?;
                w_parts.push(out);
                w0 = w1;
            }
            h_parts.push(Tensor::cat(&w_parts, 3)?);
            h0 = h1;
        }
        t_parts.push(Tensor::cat(&h_parts, 2)?);
        t0 = t1;
    }
    Tensor::cat(&t_parts, 1)
}

// ---------------------------------------------------------------------------------------------
// Rotary positions
// ---------------------------------------------------------------------------------------------

/// Split `head_dim` across the `(T, H, W)` rotary chunks — upstream `default_rope_dim_split`.
fn rope_dim_split(head_dim: usize) -> Result<[usize; 3]> {
    if !head_dim.is_multiple_of(8) {
        return Err(Error::Msg(format!(
            "ltx diffvae: head_dim must be a multiple of 8 for the default rope split, got \
             {head_dim}"
        )));
    }
    let mut d_t = (head_dim / 4) / 2 * 2;
    let mut d_hw = (head_dim - d_t) / 2;
    if !d_hw.is_multiple_of(2) {
        d_t -= 2;
        d_hw = (head_dim - d_t) / 2;
    }
    Ok([d_t, d_hw, d_hw])
}

/// `1 / base^(i / dim)` for even `i` — the rotary inverse frequencies, built in f64 as upstream
/// does before the single cast to f32.
fn rope_inv_freqs(dim: usize, base: f64, device: &Device) -> Result<Tensor> {
    let values: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| (1.0 / base.powf(i as f64 / dim as f64)) as f32)
        .collect();
    Tensor::from_vec(values, dim / 2, device)
}

/// Rotate one axis chunk of `(B, T, H, W, NH, D)` by absolute positions along `axis`
/// (1 = T, 2 = H, 3 = W). Pairs are adjacent (`[x0, x1], [x2, x3], ...`).
fn rotate_axis(x: &Tensor, inv: &Tensor, axis: usize) -> Result<Tensor> {
    let sh = x.dims().to_vec();
    let d = *sh.last().expect("rank >= 1");
    let half = d / 2;
    let split = [sh[0], sh[1], sh[2], sh[3], sh[4], half];
    let paired = [sh[0], sh[1], sh[2], sh[3], sh[4], half, 2];
    let stack = [sh[0], sh[1], sh[2], sh[3], sh[4], half, 1];
    let pairs = x.contiguous()?.reshape(&paired[..])?;
    let xe = pairs.narrow(6, 0, 1)?.reshape(&split[..])?;
    let xo = pairs.narrow(6, 1, 1)?.reshape(&split[..])?;

    let len = sh[axis];
    let positions: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let pos = Tensor::from_vec(positions, (len, 1), x.device())?;
    let mut ang_shape = [1usize, 1, 1, 1, 1, half];
    ang_shape[axis] = len;
    let ang = pos
        .broadcast_mul(&inv.reshape((1, half))?)?
        .reshape(&ang_shape[..])?;
    let (cos, sin) = (ang.cos()?, ang.sin()?);

    let re = (xe.broadcast_mul(&cos)? - xo.broadcast_mul(&sin)?)?;
    let ro = (xe.broadcast_mul(&sin)? + xo.broadcast_mul(&cos)?)?;
    let stacked = Tensor::cat(&[re.reshape(&stack[..])?, ro.reshape(&stack[..])?], 6)?;
    stacked.contiguous()?.reshape(sh)
}

// ---------------------------------------------------------------------------------------------
// Attention / blocks
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
struct NaAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    proj: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    inv: [Tensor; 3],
    split: [usize; 3],
    kernel: [usize; 3],
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl NaAttention {
    /// The checkpoint ships one fused `qkv` Linear; upstream splits it into three at load, and so
    /// does this — three narrow GEMMs never materialise a `3*dim`-wide intermediate.
    fn load(vb: &VarBuilder, prefix: &str, kernel: [usize; 3], head_dim: usize) -> Result<Self> {
        let fused_w = vb.get_unchecked(&format!("{prefix}.qkv.weight"))?;
        let (rows, _) = fused_w.dims2()?;
        let fused_b = vb.get(rows, &format!("{prefix}.qkv.bias"))?;
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
        let part = |i: usize| -> Result<Linear> {
            Ok(Linear {
                w: fused_w.narrow(0, i * dim, dim)?.contiguous()?,
                b: fused_b.narrow(0, i * dim, dim)?.contiguous()?,
            })
        };
        let split = rope_dim_split(head_dim)?;
        let device = fused_w.device();
        Ok(Self {
            to_q: part(0)?,
            to_k: part(1)?,
            to_v: part(2)?,
            proj: Linear::load(vb, &format!("{prefix}.proj"))?,
            q_norm: vb.get(head_dim, &format!("{prefix}.q_norm.weight"))?,
            k_norm: vb.get(head_dim, &format!("{prefix}.k_norm.weight"))?,
            inv: [
                rope_inv_freqs(split[0], ROPE_BASE, device)?,
                rope_inv_freqs(split[1], ROPE_BASE, device)?,
                rope_inv_freqs(split[2], ROPE_BASE, device)?,
            ],
            split,
            kernel,
            num_heads: dim / head_dim,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn width(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Q/K/V as `(B, T, H, W, NH, HD)`.
    fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let sh = x.dims().to_vec();
        let heads = [sh[0], sh[1], sh[2], sh[3], self.num_heads, self.head_dim];
        Ok((
            self.to_q.forward(x)?.reshape(&heads[..])?,
            self.to_k.forward(x)?.reshape(&heads[..])?,
            self.to_v.forward(x)?.reshape(&heads[..])?,
        ))
    }

    /// Absolute rotary positions over the three axes, applied to the head-split tensor.
    ///
    /// Positions are the tile's own `0..len`, not a global origin. Upstream does the same and
    /// documents why it is exact under tiling: every window is local, and a shared phase offset
    /// cancels inside that window's softmax.
    fn rope(&self, x: &Tensor) -> Result<Tensor> {
        let [d_t, d_h, _] = self.split;
        let d = self.head_dim;
        let xt = rotate_axis(&slice_axis(x, 5, 0, d_t)?, &self.inv[0], 1)?;
        let xh = rotate_axis(&slice_axis(x, 5, d_t, d_t + d_h)?, &self.inv[1], 2)?;
        let xw = rotate_axis(&slice_axis(x, 5, d_t + d_h, d)?, &self.inv[2], 3)?;
        Tensor::cat(&[xt, xh, xw], 5)
    }

    /// `proj(na3d(rope(norm(qkv(x)))))` for a channels-last `(B, T, H, W, C)` input.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (q, k, v) = self.project(x)?;
        let q = (rms_norm(&q.contiguous()?, &self.q_norm, NORM_EPS)? * self.scale)?;
        let k = rms_norm(&k.contiguous()?, &self.k_norm, NORM_EPS)?;
        let q = self.rope(&q)?.contiguous()?;
        let k = self.rope(&k)?.contiguous()?;
        let v = v.contiguous()?;
        let out = na3d(&q, &k, &v, self.kernel)?;
        self.proj.forward(&out)
    }
}

/// Deterministic (stages 1-4) block: `x + attn(norm1(x))`, then `x + swiglu(norm2(x))`.
#[derive(Debug)]
struct NaBlock {
    norm1: Tensor,
    attn: NaAttention,
    norm2: Tensor,
    mlp: SwiGlu,
}

impl NaBlock {
    fn load(vb: &VarBuilder, prefix: &str, kernel: [usize; 3], head_dim: usize) -> Result<Self> {
        Ok(Self {
            norm1: vb.get_unchecked(&format!("{prefix}.norm1.weight"))?,
            attn: NaAttention::load(vb, &format!("{prefix}.attn"), kernel, head_dim)?,
            norm2: vb.get_unchecked(&format!("{prefix}.norm2.weight"))?,
            mlp: SwiGlu::load(vb, &format!("{prefix}.mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = rms_norm(&x.contiguous()?, &self.norm1, NORM_EPS)?;
        let x = (x + self.attn.forward(&y)?)?;
        let y = rms_norm(&x.contiguous()?, &self.norm2, NORM_EPS)?;
        &x + self.mlp.forward(&y)?
    }
}

/// The seven AdaLN-Zero chunks (`AdaLNZero.NUM_CHUNKS`). The three gate slots exist for checkpoint
/// shape compatibility and are unused by the block — the released checkpoint carries no `gate_*`
/// parameters at all, so there is nothing folded into the projections either.
const ADALN_CHUNKS: usize = 7;

/// Stage-5 block: inject context, then AdaLN-modulated attention and MLP residuals.
#[derive(Debug)]
struct DiffusionBlock {
    context_proj: Linear,
    /// `[7, dim]`, added to the shared AdaLN chunks before use.
    scale_shift_table: Tensor,
    norm1: Tensor,
    attn: NaAttention,
    norm2: Tensor,
    mlp: SwiGlu,
}

impl DiffusionBlock {
    fn load(vb: &VarBuilder, prefix: &str, kernel: [usize; 3], head_dim: usize) -> Result<Self> {
        let table = vb.get_unchecked(&format!("{prefix}.scale_shift_table"))?;
        if table.rank() != 2 || table.dims()[0] != ADALN_CHUNKS {
            return Err(Error::Msg(format!(
                "ltx diffvae: {prefix}.scale_shift_table must be [{ADALN_CHUNKS}, dim], got {:?}",
                table.dims()
            )));
        }
        Ok(Self {
            context_proj: Linear::load(vb, &format!("{prefix}.context_proj"))?,
            scale_shift_table: table,
            norm1: vb.get_unchecked(&format!("{prefix}.norm1.weight"))?,
            attn: NaAttention::load(vb, &format!("{prefix}.attn"), kernel, head_dim)?,
            norm2: vb.get_unchecked(&format!("{prefix}.norm2.weight"))?,
            mlp: SwiGlu::load(vb, &format!("{prefix}.mlp"))?,
        })
    }

    /// `modulation` holds the seven shared AdaLN chunks, each `(1, 1, 1, 1, dim)`.
    fn forward(&self, context: &Tensor, x: &Tensor, modulation: &[Tensor]) -> Result<Tensor> {
        let width = self.scale_shift_table.dims()[1];
        let chunk = |i: usize| -> Result<Tensor> {
            let row = self
                .scale_shift_table
                .narrow(0, i, 1)?
                .reshape((1, 1, 1, 1, width))?;
            &modulation[i] + row
        };
        let (scale_msa, shift_msa) = (chunk(0)?, chunk(1)?);
        let (scale_mlp, shift_mlp) = (chunk(3)?, chunk(4)?);

        let x = (x + self.context_proj.forward(context)?)?;
        let y = modulate(
            &rms_norm(&x.contiguous()?, &self.norm1, NORM_EPS)?,
            &scale_msa,
            &shift_msa,
        )?;
        let x = (&x + self.attn.forward(&y)?)?;
        let y = modulate(
            &rms_norm(&x.contiguous()?, &self.norm2, NORM_EPS)?,
            &scale_mlp,
            &shift_mlp,
        )?;
        &x + self.mlp.forward(&y)?
    }
}

/// `x * (1 + scale) + shift` (upstream `layers.modulate`).
fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// `Linear` + channels-last pixel shuffle (upstream `LinearPixelShuffleUpsample`).
#[derive(Debug)]
struct PixelShuffleUpsample {
    proj: Linear,
    stride: [usize; 3],
}

impl PixelShuffleUpsample {
    fn load(vb: &VarBuilder, prefix: &str, stride: [usize; 3]) -> Result<Self> {
        Ok(Self {
            proj: Linear::load(vb, &format!("{prefix}.proj"))?,
            stride,
        })
    }

    fn out_channels(&self) -> usize {
        self.proj.out_features() / (self.stride[0] * self.stride[1] * self.stride[2])
    }

    /// `(B, T, H, W, C) -> (B, T*p1 [-1], H*p2, W*p3, C_out)`.
    ///
    /// A temporal stride of 2 duplicates the leading frame; dropping it preserves the causal 1:2
    /// (composed 1:8) frame mapping. `drop_leading_frame` must be true **only** for the chunk
    /// holding the tensor's true `t = 0` — a later tile has no duplicate of its own to drop.
    fn forward(&self, x: &Tensor, drop_leading_frame: bool) -> Result<Tensor> {
        let sh = x.dims().to_vec();
        let (b, t, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        let [p1, p2, p3] = self.stride;
        let c = self.out_channels();
        let y = self
            .proj
            .forward(x)?
            .reshape([b, t, h, w, c, p1, p2, p3].as_slice())?;
        // (b, t, p1, h, p2, w, p3, c)
        let y = y.permute([0usize, 1, 5, 2, 6, 3, 7, 4].as_slice())?;
        let y = y.contiguous()?.reshape((b, t * p1, h * p2, w * p3, c))?;
        if p1 == 2 && drop_leading_frame {
            return slice_axis(&y, 1, 1, y.dims()[1]);
        }
        Ok(y)
    }
}

// ---------------------------------------------------------------------------------------------
// Tiling
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
    pub tile: [usize; 3],
    /// Overlap between neighbouring tiles per axis, in stage-4-input cells.
    pub overlap: [usize; 3],
}

impl DiffVaeTiling {
    /// A tiling with the minimum legal overlap for `cfg` and the given per-axis tile extents.
    pub fn with_min_overlap(cfg: &NaDiffusionDecoderConfig, tile: [usize; 3]) -> Self {
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
    start: usize,
    end: usize,
    left_ramp: usize,
    right_ramp: usize,
}

/// Split `[0, len)` into `size`-long intervals overlapping by `overlap` (upstream `split_by_size`).
///
/// A short trailing tile is **grown leftward** to `min_tile`, widening its neighbour's right ramp
/// to match — an axis that divides unevenly would otherwise end in a sliver too narrow for stage 4
/// or 5 to attend over.
fn split_by_size(len: usize, size: usize, overlap: usize, min_tile: usize) -> Vec<Interval> {
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
    let mut out = Vec::with_capacity(count);
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
            let new_start = last.end.saturating_sub(min_tile);
            let prev = out[out.len() - 2];
            let new_overlap = prev.end.saturating_sub(new_start);
            let n = out.len();
            out[n - 2].right_ramp = new_overlap;
            out[n - 1].start = new_start;
            out[n - 1].left_ramp = new_overlap;
        }
    }
    out
}

/// Coverage / ramp consistency for a split, so a bad layout is an error rather than a dim seam.
fn validate_split(intervals: &[Interval], len: usize, min_tile: usize, axis: usize) -> Result<()> {
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
        if iv.end < iv.start {
            return bad(format!("tile {i} runs backwards"));
        }
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
        if iv.left_ramp > length || iv.right_ramp > length {
            return bad(format!(
                "tile {i} has ramps that do not fit its length {length}"
            ));
        }
        if i > 0 {
            let previous = intervals[i - 1];
            if previous.end < iv.start {
                return bad(format!("tiles {}/{i} leave a gap", i - 1));
            }
            let overlap = previous.end - iv.start;
            if previous.right_ramp != overlap || iv.left_ramp != overlap {
                return bad(format!("tiles {}/{i} disagree about their overlap", i - 1));
            }
        }
    }
    Ok(())
}

/// Propagate an interval through one upsample hop. `causal` applies the pixel-shuffle
/// duplicate-frame drop that `PixelShuffleUpsample` performs on the temporal axis.
fn propagate(interval: Interval, stride: usize, causal: bool) -> Interval {
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
fn trapezoid(length: usize, left_ramp: usize, right_ramp: usize) -> Vec<f32> {
    let left = left_ramp.min(length);
    let right = right_ramp.min(length);
    let mut mask = vec![1.0f32; length];
    for (i, slot) in mask.iter_mut().take(left).enumerate() {
        *slot *= (i + 1) as f32 / (left + 1) as f32;
    }
    for (i, slot) in mask.iter_mut().rev().take(right).enumerate() {
        *slot *= (i + 1) as f32 / (right + 1) as f32;
    }
    mask
}

// ---------------------------------------------------------------------------------------------
// The decoder
// ---------------------------------------------------------------------------------------------

/// A latent grown to the stage floors and carrying its trailing ghost frames, plus the
/// `(before, after)` spatial pads — in latent cells — that decode must crop back off.
type PreparedLatent = (Tensor, (usize, usize), (usize, usize));

/// The LTX-2.5 `NADiffusionDecoder`, on candle.
#[derive(Debug)]
pub struct NaDiffusionDecoder {
    cfg: NaDiffusionDecoderConfig,
    /// `per_channel_statistics`, `(1, C, 1, 1, 1)` — the encoder normalises its latent, so the
    /// decoder un-normalises before `conv_in`.
    stat_mean: Tensor,
    stat_std: Tensor,
    conv_in: Linear,
    det_stages: Vec<Vec<NaBlock>>,
    upsamples: Vec<PixelShuffleUpsample>,
    t_linear1: Linear,
    t_linear2: Linear,
    shared_adaln: Linear,
    conv_in_x_t: Linear,
    diff_blocks: Vec<DiffusionBlock>,
    norm_out: Tensor,
    conv_out: Linear,
    device: Device,
}

impl NaDiffusionDecoder {
    /// Sinusoidal timestep-projection width (`Timesteps(num_channels=256, flip_sin_to_cos=True,
    /// downscale_freq_shift=0)`).
    const TIME_PROJ_DIM: usize = 256;

    /// Build from a released `CausalDiffusionVAE` checkpoint, **verbatim**.
    ///
    /// `vb` is rooted at the decoder (`decoder.` on the released file); `stats` is rooted where
    /// `per_channel_statistics.{mean,std}-of-means` live (the file root). Both are separate
    /// arguments for the same reason [`crate::vae::LtxVideoVae`] takes them separately: on LTX-2.3
    /// the statistics sit beside the decoder under `vae.`, on LTX-2.5 they sit above it.
    pub fn load(vb: VarBuilder, stats: VarBuilder, cfg: &NaDiffusionDecoderConfig) -> Result<Self> {
        let stages = cfg.stage_channels.len();
        let det = stages - 1;
        let mut det_stages = Vec::with_capacity(det);
        let mut upsamples = Vec::with_capacity(det);
        for stage in 0..det {
            let kernel = cfg.stage_kernels[stage];
            let mut blocks = Vec::with_capacity(cfg.stage_depths[stage]);
            for i in 0..cfg.stage_depths[stage] {
                blocks.push(NaBlock::load(
                    &vb,
                    &format!("det_stages.{stage}.{i}"),
                    kernel,
                    cfg.head_dim,
                )?);
            }
            det_stages.push(blocks);
            upsamples.push(PixelShuffleUpsample::load(
                &vb,
                &format!("upsamples.{stage}"),
                cfg.upsamples[stage].0,
            )?);
        }
        let depth5 = *cfg.stage_depths.last().expect("validated");
        let mut diff_blocks = Vec::with_capacity(depth5);
        for i in 0..depth5 {
            diff_blocks.push(DiffusionBlock::load(
                &vb,
                &format!("diff_blocks.{i}"),
                cfg.stage5_kernel,
                cfg.head_dim,
            )?);
        }

        let latent_c = cfg.in_channels;
        let stat = |key: &str| -> Result<Tensor> {
            stats
                .get(latent_c, key)?
                .reshape((1, latent_c, 1, 1, 1))?
                .to_dtype(DType::F32)
        };

        let decoder = Self {
            device: vb.device().clone(),
            cfg: cfg.clone(),
            stat_mean: stat(STAT_MEAN_KEY)?,
            stat_std: stat(STAT_STD_KEY)?,
            conv_in: Linear::load(&vb, "conv_in")?,
            det_stages,
            upsamples,
            t_linear1: Linear::load(&vb, "t_embedder.mlp.0")?,
            t_linear2: Linear::load(&vb, "t_embedder.mlp.2")?,
            shared_adaln: Linear::load(&vb, "shared_adaln.proj")?,
            conv_in_x_t: Linear::load(&vb, "conv_in_x_t")?,
            diff_blocks,
            norm_out: vb.get_unchecked("norm_out.weight")?,
            conv_out: Linear::load(&vb, "conv_out")?,
        };
        decoder.check_widths()?;
        Ok(decoder)
    }

    /// Cross-check the declared structure against the loaded widths. A config/weight disagreement
    /// here would otherwise surface as a reshape failure several stages deep, or — where the
    /// numbers happen to line up — as a plausible but wrong picture.
    fn check_widths(&self) -> Result<()> {
        let cfg = &self.cfg;
        let expect = |what: &str, got: usize, want: usize| -> Result<()> {
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
            self.conv_in_x_t.in_features(),
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
                block.context_proj.in_features(),
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
                blocks[0].mlp.w_gate.dims()[0],
                mlp_hidden(dim),
            )?;
        }
        Ok(())
    }

    /// The structure this decoder was built for.
    pub fn config(&self) -> &NaDiffusionDecoderConfig {
        &self.cfg
    }

    /// The device its weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `(B, C, T, H, W) -> (B, T, H, W, C)` with the per-channel statistics undone.
    fn un_normalize(&self, latent: &Tensor) -> Result<Tensor> {
        let x = latent
            .broadcast_mul(&self.stat_std)?
            .broadcast_add(&self.stat_mean)?;
        x.permute((0usize, 2, 3, 4, 1))?.contiguous()
    }

    /// Stages 1..=3 over the whole (already ghost-padded) latent → the stage-4 input feature.
    fn stages_1_to_3(&self, latent: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(&self.un_normalize(latent)?)?;
        for stage in 0..self.upsamples.len() - 1 {
            x = self.run_det_stage(&x, stage, true)?;
        }
        Ok(x)
    }

    fn run_det_stage(&self, x: &Tensor, stage: usize, drop_leading_frame: bool) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.det_stages[stage] {
            x = block.forward(&x)?;
        }
        self.upsamples[stage].forward(&x, drop_leading_frame)
    }

    /// Stage 4 → the stage-5 context, with the trailing ghost frames cropped back off.
    fn stage_4(&self, x: &Tensor, drop_leading_frame: bool, pad_trailing: bool) -> Result<Tensor> {
        let last = self.upsamples.len() - 1;
        let x = self.run_det_stage(x, last, drop_leading_frame)?;
        if !pad_trailing {
            return Ok(x);
        }
        let ghost = self.cfg.ghost_latent_frames() * self.cfg.pixel_scale()[0];
        if ghost == 0 {
            return Ok(x);
        }
        let frames = x.dims()[1];
        let content = frames.saturating_sub(ghost).max(1);
        let keep = frames.min(content.max(self.cfg.stage5_kernel[0]));
        Ok(resize_axis(&x, 1, keep, false)?.0)
    }

    /// The shared AdaLN-Zero chunks for one timestep.
    fn modulation(&self, t: f64) -> Result<Vec<Tensor>> {
        let emb = self.timestep_embedding(t)?;
        let h = self.shared_adaln.forward(&silu(&emb)?)?;
        let c5 = self.cfg.stage5_width();
        (0..ADALN_CHUNKS)
            .map(|i| h.narrow(1, i * c5, c5)?.reshape((1, 1, 1, 1, c5)))
            .collect()
    }

    /// One deterministic block, in isolation. Exposed for parity work: a whole-decode mismatch says
    /// nothing about *where*, and the alternative — inferring it from the picture — is guesswork.
    /// `x` is channels-last `(B, T, H, W, C)` at that stage's width.
    pub fn det_block(&self, stage: usize, index: usize, x: &Tensor) -> Result<Tensor> {
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
        context: &Tensor,
        x: &Tensor,
        t: f64,
    ) -> Result<Tensor> {
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
    pub fn stage_features(&self, latent: &Tensor) -> Result<(Tensor, Tensor)> {
        self.check_latent(latent)?;
        let (padded, _, _) = self.prepare_latent(&latent.to_dtype(DType::F32)?)?;
        let feature = self.stages_1_to_3(&padded)?;
        let context = self.stage_4(&feature, true, true)?;
        Ok((feature, context))
    }

    /// The seven shared AdaLN-Zero chunks at `t`, concatenated into `(1, 7 * stage5_width)` in
    /// chunk order. The block-level modulation is this plus each block's `scale_shift_table`.
    pub fn adaln_chunks(&self, t: f64) -> Result<Tensor> {
        let chunks = self.modulation(t)?;
        let flat: Vec<Tensor> = chunks
            .iter()
            .map(|c| c.reshape((1, c.elem_count())))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&flat, 1)
    }

    /// The timestep embedding itself — exposed so a port failure localises to the embedder rather
    /// than to whatever the eight stage-5 blocks did with it.
    ///
    /// diffusers `get_timestep_embedding(..., flip_sin_to_cos=True, downscale_freq_shift=0)`,
    /// i.e. `[cos, sin]`, then the two-layer SiLU MLP.
    pub fn timestep_embedding(&self, t: f64) -> Result<Tensor> {
        let half = Self::TIME_PROJ_DIM / 2;
        let neg_log = -ROPE_BASE.ln();
        let scaled = t * self.cfg.timestep_scale_multiplier;
        let angles: Vec<f32> = (0..half)
            .map(|i| (scaled * (neg_log * i as f64 / half as f64).exp()) as f32)
            .collect();
        let ang = Tensor::from_vec(angles, (1, half), &self.device)?;
        let proj = Tensor::cat(&[ang.cos()?, ang.sin()?], 1)?;
        self.t_linear2
            .forward(&silu(&self.t_linear1.forward(&proj)?)?)
    }

    /// One stage-5 pass: patchified `x_t` + context → the model's pixel-space prediction.
    fn diff_step(&self, context: &Tensor, x_t: &Tensor, t: f64) -> Result<Tensor> {
        let modulation = self.modulation(t)?;
        let patched = patchify(x_t, self.cfg.patch_size)?;
        let patched = patched.permute((0usize, 2, 3, 4, 1))?.contiguous()?;
        let mut x = self.conv_in_x_t.forward(&patched)?;
        for block in &self.diff_blocks {
            x = block.forward(context, &x, &modulation)?;
        }
        let x = self
            .conv_out
            .forward(&rms_norm(&x.contiguous()?, &self.norm_out, NORM_EPS)?)?;
        let x = x.permute((0usize, 4, 1, 2, 3))?.contiguous()?;
        unpatchify(&x, self.cfg.patch_size)
    }

    /// The reverse-diffusion schedule: `linspace(1, 1/N, N)`, computed the way `torch.linspace`
    /// does (`start + i * step`) so a multi-step schedule lands on the same floats the reference
    /// samples at.
    fn timesteps(&self) -> Vec<f64> {
        let n = self.cfg.default_num_inference_steps;
        if n == 1 {
            return vec![1.0];
        }
        let step = (1.0 / n as f64 - 1.0) / (n - 1) as f64;
        (0..n).map(|i| 1.0 + i as f64 * step).collect()
    }

    /// One Euler update: advance `x_t` from `t_now` to `t_next` given the model's prediction.
    fn euler_step(
        &self,
        x_t: &Tensor,
        model_out: &Tensor,
        t_now: f64,
        t_next: f64,
    ) -> Result<Tensor> {
        let velocity = match self.cfg.model_output_type {
            ModelOutputType::Velocity => model_out.clone(),
            ModelOutputType::X0 => ((x_t - model_out)? / t_now)?,
        };
        x_t - (velocity * (t_now - t_next))?
    }

    /// Stages 4-5 for one stage-4 feature extent.
    fn decode_one_tile(
        &self,
        feature: &Tensor,
        x_t_init: &Tensor,
        is_origin: bool,
        pad_trailing: bool,
    ) -> Result<Tensor> {
        let context = self.stage_4(feature, is_origin, pad_trailing)?;
        let schedule = self.timesteps();
        let mut x_t = x_t_init.clone();
        for i in 0..schedule.len() - 1 {
            let out = self.diff_step(&context, &x_t, schedule[i])?;
            x_t = self.euler_step(&x_t, &out, schedule[i], schedule[i + 1])?;
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
    fn prepare_latent(&self, latent: &Tensor) -> Result<PreparedLatent> {
        let min = self.cfg.min_latent_shape();
        let (x, _) = resize_axis(latent, 2, latent.dims()[2].max(min[0]), false)?;
        let (x, h_pad) = resize_axis(&x, 3, x.dims()[3].max(min[1]), true)?;
        let (x, w_pad) = resize_axis(&x, 4, x.dims()[4].max(min[2]), true)?;
        let ghost = self.cfg.ghost_latent_frames();
        let padded = if ghost > 0 {
            resize_axis(&x, 2, x.dims()[2] + ghost, false)?.0
        } else {
            x.clone()
        };
        Ok((padded, h_pad, w_pad))
    }

    /// Crop a decoded pixel volume back to the content geometry the caller asked for.
    fn crop_to_content(
        &self,
        pixels: &Tensor,
        frames: usize,
        height: usize,
        width: usize,
        h_pad: (usize, usize),
        w_pad: (usize, usize),
    ) -> Result<Tensor> {
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

    fn check_latent(&self, latent: &Tensor) -> Result<()> {
        if latent.rank() != 5 || latent.dims()[1] != self.cfg.in_channels {
            return Err(Error::Msg(format!(
                "ltx diffvae: expected a (B, {}, T, H, W) latent, got {:?}",
                self.cfg.in_channels,
                latent.dims()
            )));
        }
        Ok(())
    }

    fn check_noise(&self, noise: &Tensor, want: [usize; 3], latent_batch: usize) -> Result<()> {
        let want_shape = [
            latent_batch,
            self.cfg.out_channels,
            want[0],
            want[1],
            want[2],
        ];
        if noise.dims() != want_shape {
            return Err(Error::Msg(format!(
                "ltx diffvae: stage-5 noise must be {want_shape:?}, got {:?} (see \
                 NaDiffusionDecoderConfig::noise_shape)",
                noise.dims()
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
    pub fn decode(&self, latent: &Tensor, noise: &Tensor) -> Result<Tensor> {
        self.check_latent(latent)?;
        let sh = latent.dims().to_vec();
        let (b, lt, lh, lw) = (sh[0], sh[2], sh[3], sh[4]);
        self.check_noise(noise, self.cfg.noise_shape(lt, lh, lw), b)?;

        let (padded, h_pad, w_pad) = self.prepare_latent(&latent.to_dtype(DType::F32)?)?;
        let feature = self.stages_1_to_3(&padded)?;
        let pixels = self.decode_one_tile(&feature, &noise.to_dtype(DType::F32)?, true, true)?;
        let scale = self.cfg.pixel_scale();
        let out = self.crop_to_content(
            &pixels,
            (lt - 1) * scale[0] + 1,
            lh * scale[1],
            lw * scale[2],
            h_pad,
            w_pad,
        )?;
        out.contiguous()
    }

    /// Tiled decode: stages 1-3 once over the whole volume, stages 4-5 per tile, blended in pixel
    /// space with separable trapezoids.
    ///
    /// Takes the same full-canvas `noise` as [`Self::decode`] and slices each tile's `x_t` out of
    /// it, so a tiled and an untiled decode of the same inputs are directly comparable — which is
    /// what makes the seam tests able to say anything at all.
    pub fn decode_tiled(
        &self,
        latent: &Tensor,
        noise: &Tensor,
        tiling: &DiffVaeTiling,
    ) -> Result<Tensor> {
        self.check_latent(latent)?;
        let tiling = tiling.validated(&self.cfg)?;
        let sh = latent.dims().to_vec();
        let (b, lt, lh, lw) = (sh[0], sh[2], sh[3], sh[4]);
        self.check_noise(noise, self.cfg.noise_shape(lt, lh, lw), b)?;

        let (padded, h_pad, w_pad) = self.prepare_latent(&latent.to_dtype(DType::F32)?)?;
        let min = self.cfg.min_latent_shape();
        let work = [lt.max(min[0]), lh.max(min[1]), lw.max(min[2])];
        let stage4 = self.cfg.stage4_shape(work[0], work[1], work[2]);
        let canvas = self.cfg.stage5_pixel_shape(stage4, true, true);

        let feature = self.stages_1_to_3(&padded)?;
        let noise_f32 = noise.to_dtype(DType::F32)?;

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

        let mut accumulator: Option<Tensor> = None;
        let mut weights: [Vec<f32>; 3] = [
            vec![0.0; canvas[0]],
            vec![0.0; canvas[1]],
            vec![0.0; canvas[2]],
        ];
        let mut weight_seen = [false; 3];

        for (t_s4, t_px) in &axes[0] {
            let is_origin = t_s4.start == 0;
            let pad_trailing = t_s4.end == stage4[0];
            // The trailing tile keeps the ghost frames stages 1-3 produced; stage 4 crops them.
            let t_end = if pad_trailing {
                feature.dims()[1]
            } else {
                t_s4.end
            };
            let feature_t = slice_axis(&feature, 1, t_s4.start, t_end)?;
            for (h_s4, h_px) in &axes[1] {
                let feature_th = slice_axis(&feature_t, 2, h_s4.start, h_s4.end)?;
                for (w_s4, w_px) in &axes[2] {
                    let tile_feature =
                        slice_axis(&feature_th, 3, w_s4.start, w_s4.end)?.contiguous()?;
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
                        &x_t.contiguous()?,
                        is_origin,
                        pad_trailing,
                    )?;
                    let (tile, _) = resize_axis(&tile, 2, t_px.end - t_px.start, false)?;
                    let (tile, _) = resize_axis(&tile, 3, h_px.end - h_px.start, true)?;
                    let (tile, _) = resize_axis(&tile, 4, w_px.end - w_px.start, true)?;

                    let mt = trapezoid(t_px.end - t_px.start, t_px.left_ramp, t_px.right_ramp);
                    let mh = trapezoid(h_px.end - h_px.start, h_px.left_ramp, h_px.right_ramp);
                    let mw = trapezoid(w_px.end - w_px.start, w_px.left_ramp, w_px.right_ramp);
                    let mask = profile(&mt, 2, &self.device)?
                        .broadcast_mul(&profile(&mh, 3, &self.device)?)?
                        .broadcast_mul(&profile(&mw, 4, &self.device)?)?;
                    let weighted = tile.broadcast_mul(&mask)?;

                    let placed = weighted
                        .pad_with_zeros(2, t_px.start, canvas[0] - t_px.end)?
                        .pad_with_zeros(3, h_px.start, canvas[1] - h_px.end)?
                        .pad_with_zeros(4, w_px.start, canvas[2] - w_px.end)?;
                    accumulator = Some(match accumulator {
                        None => placed,
                        Some(acc) => (acc + placed)?,
                    });

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
                            weights[axis][interval.start + i] += value;
                        }
                    }
                }
            }
        }

        let accumulator = accumulator
            .ok_or_else(|| Error::Msg("ltx diffvae: tiled decode produced no tiles".to_string()))?;
        debug_assert!(weight_seen.iter().all(|&s| s));
        let weight_3d = profile(&weights[0], 2, &self.device)?
            .broadcast_mul(&profile(&weights[1], 3, &self.device)?)?
            .broadcast_mul(&profile(&weights[2], 4, &self.device)?)?;
        let blended = accumulator.broadcast_div(&weight_3d)?;

        let scale = self.cfg.pixel_scale();
        let out = self.crop_to_content(
            &blended,
            (lt - 1) * scale[0] + 1,
            lh * scale[1],
            lw * scale[2],
            h_pad,
            w_pad,
        )?;
        out.contiguous()
    }

    /// [`Self::decode`] / [`Self::decode_tiled`] with the stage-5 noise drawn from `seed` on CPU
    /// (`candle_gen::seed`'s `StdRng` + `StandardNormal`, the crate's own reproducible draw) and
    /// moved to the weights' device.
    pub fn decode_seeded(
        &self,
        latent: &Tensor,
        seed: u64,
        tiling: Option<&DiffVaeTiling>,
    ) -> Result<Tensor> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        self.check_latent(latent)?;
        let sh = latent.dims().to_vec();
        let shape5 = self.cfg.noise_shape(sh[2], sh[3], sh[4]);
        let shape = (
            sh[0],
            self.cfg.out_channels,
            shape5[0],
            shape5[1],
            shape5[2],
        );
        let n = shape.0 * shape.1 * shape.2 * shape.3 * shape.4;
        let mut rng = StdRng::seed_from_u64(seed);
        let values = candle_gen::seed::seeded_normal_vec(&mut rng, n);
        let noise = Tensor::from_vec(values, shape, &self.device)?;
        match tiling {
            Some(t) => self.decode_tiled(latent, &noise, t),
            None => self.decode(latent, &noise),
        }
    }
}

/// A 1-D blend profile broadcast along `axis` of a `(B, C, F, H, W)` volume.
fn profile(values: &[f32], axis: usize, device: &Device) -> Result<Tensor> {
    let mut shape = [1usize; 5];
    shape[axis] = values.len();
    Tensor::from_vec(values.to_vec(), &shape[..], device)
}

// ---------------------------------------------------------------------------------------------
// Loader plumbing
// ---------------------------------------------------------------------------------------------

/// The released file's per-channel latent mean. LTX-2.5 keeps the LTX-2.3 `-of-means` spelling that
/// [`crate::vae::LtxVideoVae`] already reads; the MLX port renames it during conversion, which is
/// exactly the conversion this port does not need.
pub const STAT_MEAN_KEY: &str = "per_channel_statistics.mean-of-means";

/// The released file's per-channel latent standard deviation. See [`STAT_MEAN_KEY`].
pub const STAT_STD_KEY: &str = "per_channel_statistics.std-of-means";

/// The `decoder.` sub-tree of a `CausalDiffusionVAE` checkpoint — where [`NaDiffusionDecoder::load`]
/// must be rooted.
pub const DECODER_PREFIX: &str = "decoder";

/// Keys the checkpoint carries under `decoder.` that the decoder never reads.
///
/// `type_emb` is a 128-wide vector the checkpoint ships and no reference module consumes — the
/// upstream loader drops it on a non-strict `load_state_dict`. It is listed here so a key audit
/// distinguishes "known dead weight" from "the port forgot to load something".
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

/// Every weight key [`NaDiffusionDecoder::load`] reads for `cfg`, relative to the `decoder.` root
/// (the two `per_channel_statistics` tensors sit above it and are not included).
pub fn expected_weight_keys(cfg: &NaDiffusionDecoderConfig) -> Vec<String> {
    let mut keys = vec!["norm_out.weight".to_string()];
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

/// Classify a key set as an `NADiffusionDecoder` — used by loaders that must tell the two LTX-2.5
/// video decoders apart from the tensors alone. Keys are as they appear in the file, so the
/// `decoder.` prefix is optional.
pub fn looks_like_diffusion_decoder<'a>(keys: impl IntoIterator<Item = &'a str>) -> bool {
    let mut det = false;
    let mut diff = false;
    for key in keys {
        let key = key.strip_prefix("decoder.").unwrap_or(key);
        det |= key.starts_with("det_stages.");
        diff |= key.starts_with("diff_blocks.");
    }
    det && diff
}

#[cfg(test)]
mod tests;
