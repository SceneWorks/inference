//! SDXL VAE (autoencoder) — port of the vendored `_vendor/mlx_sd/vae.py`. The decoder powers T2I
//! (latents → image); the encoder powers img2img (sc-2638). Runs entirely in NHWC, reusing the
//! UNet [`ResnetBlock2D`] (temb-free here) and the core conv/group-norm primitives. SDXL's
//! `scaling_factor` is **0.13025** (not SD-2.1's 0.18215). Latents are scaled by it on encode and
//! divided by it on decode.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, multiply, pad};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::{conv2d, group_norm, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LatentDecoder, Result};

use crate::config::VaeConfig;
use crate::unet::ResnetBlock2D;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;

thread_local! {
    static TRACE_DECODE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Opt-in diagnostics for one tiled decode, restored even on error. An unset variable does not
/// evaluate or modify the graph. `SDXL_VAE_TRACE_TILE=256` traces precisely that tile edge,
/// including the global head; it can be combined with the parity test's single-cell selectors.
struct DecodeTrace(bool);

impl DecodeTrace {
    fn for_tile(tile: i32) -> Self {
        let enabled = std::env::var("SDXL_VAE_TRACE_TILE")
            .ok()
            .is_some_and(|v| v.parse::<i32>() == Ok(tile));
        Self(TRACE_DECODE.with(|active| active.replace(enabled)))
    }
}

impl Drop for DecodeTrace {
    fn drop(&mut self) {
        TRACE_DECODE.with(|active| active.set(self.0));
    }
}

#[derive(Debug)]
struct DecodeStageStats {
    min: f32,
    max: f32,
    nan: u64,
    inf: u64,
}

/// Bounded diagnostic reductions, never a full activation copied/upcast to the host. Reducing
/// each row chunk before casting its scalar min/max avoids f16 sum overflow and bounds scratch.
fn decode_stage_stats(x: &Array) -> Result<DecodeStageStats> {
    let sh = x.shape();
    let row_elems = i64::from(sh[0]) * i64::from(sh[2]) * i64::from(sh[3]);
    let rows = ((1 << 20) / row_elems.max(1)).clamp(1, i64::from(sh[1])) as i32;
    let mut stats = DecodeStageStats {
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
        nan: 0,
        inf: 0,
    };
    for y in (0..sh[1]).step_by(rows as usize) {
        let end = (y + rows).min(sh[1]);
        // A contiguous row range, not a gather: the diagnostic does not allocate a full copy.
        use mlx_rs::ops::indexing::IndexOp;
        let part = x.index((.., y..end, .., ..));
        let count = |a: Array| -> Result<u64> {
            Ok(u64::from(
                a.as_dtype(Dtype::Uint32)?.sum(None)?.item::<u32>(),
            ))
        };
        stats.nan += count(part.is_nan()?)?;
        stats.inf += count(part.is_inf()?)?;
        let finite = part.is_finite()?;
        let minimum = mlx_rs::ops::r#where(&finite, &part, scalar(f32::INFINITY))?;
        let maximum = mlx_rs::ops::r#where(&finite, &part, scalar(f32::NEG_INFINITY))?;
        stats.min = stats
            .min
            .min(minimum.min(None)?.as_dtype(Dtype::Float32)?.item::<f32>());
        stats.max = stats
            .max
            .max(maximum.max(None)?.as_dtype(Dtype::Float32)?.item::<f32>());
    }
    Ok(stats)
}

pub(crate) fn trace_decode_stage(label: &str, x: &Array) -> Result<()> {
    if TRACE_DECODE.with(|active| active.get()) {
        let stats = decode_stage_stats(x)?;
        eprintln!(
            "[sdxl VAE trace] {label} {:?} {:?}: min={} max={} nan={} inf={} active={} cache={} peak={}",
            x.shape(), x.dtype(), stats.min, stats.max, stats.nan, stats.inf,
            mlx_rs::memory::get_active_memory(), mlx_rs::memory::get_cache_memory(),
            mlx_rs::memory::get_peak_memory()
        );
    }
    Ok(())
}

/// The SDXL VAE's tiling geometry for ladder rung 2 (SC-15525).
///
/// A **still-image** VAE: spatial ×8 (three `upsample_nearest(2)`+conv stages in `up_blocks.0..2`;
/// `up_blocks.3` does not upsample), and the temporal axis is a singleton, so `temporal_scale: 1`
/// non-causal makes it a no-op and only H/W tile.
///
/// `full_res_channels: 128` — `VaeConfig::sdxl_base().block_out_channels[0]`, the width of the last
/// `up_blocks` stage and of the `conv_norm_out` input, which is the widest tensor written at full
/// output resolution. That is the *correctness* input to [`VaeTiling::writable_frame_cap`], not a
/// memory estimate; it is inert at SDXL's 2048² ceiling (128 × 2048² = 5.4e8, 0.25× the bound).
///
/// Declared here rather than in `gen_core::tiling` deliberately: the presets there are the ones
/// shared by more than one crate, and this geometry has exactly one consumer.
const VAE_TILING: mlx_gen::tiling::VaeTiling = mlx_gen::tiling::VaeTiling {
    spatial_scale: 8,
    temporal_scale: 1,
    causal_temporal: false,
    full_res_channels: 128,
};

/// Single-head spatial self-attention used in the VAE mid block (the vendored `vae.Attention`).
struct VaeAttention {
    gn_w: Array,
    gn_b: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
}

impl VaeAttention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let lin = |n: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&format!("{prefix}.{n}.weight"))?.clone(),
                Some(w.require(&format!("{prefix}.{n}.bias"))?.clone()),
            ))
        };
        Ok(Self {
            gn_w: w.require(&format!("{prefix}.group_norm.weight"))?.clone(),
            gn_b: w.require(&format!("{prefix}.group_norm.bias"))?.clone(),
            q: lin("to_q")?,
            k: lin("to_k")?,
            v: lin("to_v")?,
            out: lin("to_out.0")?,
        })
    }

    /// `x`: NHWC `[B, H, W, C]`. Single-head attention over the H·W positions, residual.
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
        let y = group_norm(x, &self.gn_w, &self.gn_b, GN_GROUPS, GN_EPS)?;
        let to_seq = |a: Array| -> Result<Array> { Ok(a.reshape(&[b, 1, h * w_, c])?) };
        let q = to_seq(self.q.forward(&y)?)?;
        let k = to_seq(self.k.forward(&y)?)?;
        let v = to_seq(self.v.forward(&y)?)?;
        let scale = (c as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = self.out.forward(&o.reshape(&[b, h, w_, c])?)?;
        Ok(add(x, &o)?)
    }
}

/// Encoder/decoder macro-block: a run of (temb-free) resnets, then an optional downsample
/// (asymmetric-pad + stride-2 conv) or upsample (nearest-2× + conv). Port of the vendored
/// `vae.EncoderDecoderBlock2D`.
struct EncoderDecoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsample: Option<(Array, Array)>,
    upsample: Option<(Array, Array)>,
}

impl EncoderDecoderBlock2D {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_resnets: i32,
        add_downsample: bool,
        add_upsample: bool,
    ) -> Result<Self> {
        let resnets = (0..num_resnets)
            .map(|j| ResnetBlock2D::from_weights(w, &format!("{prefix}.resnets.{j}")))
            .collect::<Result<Vec<_>>>()?;
        let conv = |which: &str| -> Result<(Array, Array)> {
            Ok((
                crate::unet::nchw_to_nhwc(w.require(&format!("{prefix}.{which}.0.conv.weight"))?)?,
                w.require(&format!("{prefix}.{which}.0.conv.bias"))?.clone(),
            ))
        };
        Ok(Self {
            resnets,
            downsample: if add_downsample {
                Some(conv("downsamplers")?)
            } else {
                None
            },
            upsample: if add_upsample {
                Some(conv("upsamplers")?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x, None)?;
        }
        if let Some((cw, cb)) = &self.downsample {
            // Asymmetric (right/bottom) pad then stride-2, pad-0 conv (the SD downsample).
            x = pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            x = conv2d(&x, cw, Some(cb), 2, 0)?;
        }
        if let Some((cw, cb)) = &self.upsample {
            x = conv2d(&upsample_nearest(&x, 2)?, cw, Some(cb), 1, 1)?;
        }
        Ok(x)
    }

    fn forward_tiled_decode(
        &self,
        x: &Array,
        tile_edge: i32,
        upsampled_tile_edge: i32,
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        if self.downsample.is_some() {
            return Err(mlx_gen::Error::Msg(
                "SDXL tiled decode cannot run an encoder/downsample block".into(),
            ));
        }
        let mut x = x.clone();
        for (index, resnet) in self.resnets.iter().enumerate() {
            x = resnet.forward_tiled_vae(&x, tile_edge, cancel)?;
            trace_decode_stage(&format!("resnet.{index}.out"), &x)?;
        }
        if let Some((cw, cb)) = &self.upsample {
            x = upsample_tiled(&x, cw, cb, upsampled_tile_edge, cancel)?;
        }
        Ok(x)
    }

    fn upsamples(&self) -> bool {
        self.upsample.is_some()
    }
}

/// The VAE upsample convolution has no normalization between nearest-neighbor expansion and
/// convolution: unlike a resnet's normalized inputs its activation can be ~2000 (sc-26268).
fn upsample_tiled(
    x: &Array,
    weight: &Array,
    bias: &Array,
    tile: i32,
    cancel: Option<&CancelFlag>,
) -> Result<Array> {
    let up = upsample_nearest(x, 2)?;
    trace_decode_stage("upsample.nearest", &up)?;
    if x.dtype() == Dtype::Float16 {
        // Half storage, wide accumulation: only this unnormalized convolution is sensitive to
        // partial-sum overflow. The helper promotes one crop and casts each core back before
        // assembling the image, so no full-resolution f32 activation is introduced.
        mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc_f32_accum(
            &up,
            weight,
            Some(bias),
            tile,
            cancel,
            |tile| Ok(tile.clone()),
        )
    } else {
        mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc(&up, weight, Some(bias), tile, cancel, |tile| {
            Ok(tile.clone())
        })
    }
}

/// The decoder (latent → image).
struct Decoder {
    conv_in_w: Array,
    conv_in_b: Array,
    mid_resnet0: ResnetBlock2D,
    mid_attn: VaeAttention,
    mid_resnet1: ResnetBlock2D,
    up_blocks: Vec<EncoderDecoderBlock2D>,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out_w: Array,
    conv_out_b: Array,
}

impl Decoder {
    fn from_weights(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let n = cfg.block_out_channels.len();
        // decoder layers_per_block = config.layers_per_block + 1.
        let num_resnets = cfg.layers_per_block + 1;
        let up_blocks = (0..n)
            .map(|i| {
                EncoderDecoderBlock2D::from_weights(
                    w,
                    &format!("decoder.up_blocks.{i}"),
                    num_resnets,
                    false,
                    i < n - 1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in_w: crate::unet::nchw_to_nhwc(w.require("decoder.conv_in.weight")?)?,
            conv_in_b: w.require("decoder.conv_in.bias")?.clone(),
            mid_resnet0: ResnetBlock2D::from_weights(w, "decoder.mid_block.resnets.0")?,
            mid_attn: VaeAttention::from_weights(w, "decoder.mid_block.attentions.0")?,
            mid_resnet1: ResnetBlock2D::from_weights(w, "decoder.mid_block.resnets.1")?,
            up_blocks,
            norm_out_w: w.require("decoder.conv_norm_out.weight")?.clone(),
            norm_out_b: w.require("decoder.conv_norm_out.bias")?.clone(),
            conv_out_w: crate::unet::nchw_to_nhwc(w.require("decoder.conv_out.weight")?)?,
            conv_out_b: w.require("decoder.conv_out.bias")?.clone(),
        })
    }

    fn forward(&self, z: &Array) -> Result<Array> {
        self.tail(&self.head(z)?)
    }

    /// The **globally-scoped** half of the decode: `conv_in` → mid resnet → mid **attention** → mid
    /// resnet, all at latent resolution.
    ///
    /// Split out for ladder rung 2 (SC-15525). `mid_attn` is a single-head self-attention over every
    /// `H·W` latent token ([`VaeAttention::forward`]), so its result depends on the whole grid. Tiling
    /// across it would silently change the arithmetic — each tile would attend only to its own tokens
    /// — which is a *wrong decode*, not a blend artifact. The head is also cheap to run whole: at
    /// 1024² it is a `[1, 128, 128, 512]` f32 activation (32 MiB), three orders of magnitude under the
    /// full-resolution tail it protects. So the head runs once on the full latent and only the tail
    /// tiles, which is the same split `mlx_gen_qwen_image`'s `decode_tiled` uses.
    fn head(&self, z: &Array) -> Result<Array> {
        let mut x = conv2d(z, &self.conv_in_w, Some(&self.conv_in_b), 1, 1)?;
        x = self.mid_resnet0.forward(&x, None)?;
        x = self.mid_attn.forward(&x)?;
        self.mid_resnet1.forward(&x, None)
    }

    /// The dense upsample half of the decode: the four `up_blocks` (×8 nearest+conv upsample) and
    /// `conv_norm_out` → `silu` → `conv_out`. This is where full-resolution activations live.
    /// GroupNorms span the full image, so the bounded path uses [`Self::tail_tiled`] to retain those
    /// global statistics and tile each 3×3 convolution layer instead of tiling this whole tail.
    fn tail(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for ub in &self.up_blocks {
            x = ub.forward(&x)?;
        }
        let x = group_norm(&x, &self.norm_out_w, &self.norm_out_b, GN_GROUPS, GN_EPS)?;
        conv2d(&silu(&x)?, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)
    }

    /// Untiled inference tail with explicit lifetime boundaries. Every convolution and GroupNorm
    /// still sees the ENTIRE spatial tensor, with the exact dense kernel geometry. Evaluating each
    /// resnet convolution and upsample before proceeding prevents MLX's lazy graph from keeping
    /// multiple full-resolution normalization/conv intermediates simultaneously live (sc-26268:
    /// 1024² f32 reference peaked at 17.55 GiB despite a cleared allocator cache).
    fn tail_evaluated(&self, x: &Array) -> Result<Array> {
        x.eval()?;
        let mut x = x.clone();
        for up in &self.up_blocks {
            for resnet in &up.resnets {
                x = resnet.forward_evaluated_vae(&x)?;
            }
            if let Some((cw, cb)) = &up.upsample {
                let enlarged = upsample_nearest(&x, 2)?;
                enlarged.eval()?;
                // Drop the previous-resolution activation before allocating convolution scratch.
                x = enlarged;
                x = conv2d(&x, cw, Some(cb), 1, 1)?;
                x.eval()?;
            }
        }
        let norm = group_norm(&x, &self.norm_out_w, &self.norm_out_b, GN_GROUPS, GN_EPS)?;
        norm.eval()?;
        drop(x);
        let activated = silu(&norm)?;
        activated.eval()?;
        drop(norm);
        let out = conv2d(&activated, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)?;
        out.eval()?;
        Ok(out)
    }

    /// Layer-wise bounded tail with dense-image GroupNorm semantics (sc-19753).
    fn tail_tiled(
        &self,
        x: &Array,
        output_tile_edge: i32,
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(mlx_gen::CancelFlag::is_cancelled) {
            return Err(mlx_gen::Error::Canceled);
        }
        let tile_edge_at = |remaining_scale: i32| {
            ((output_tile_edge + remaining_scale - 1) / remaining_scale).max(3)
        };
        let mut remaining_scale = self
            .up_blocks
            .iter()
            .filter(|block| block.upsamples())
            .fold(1_i32, |scale, _| scale.saturating_mul(2));
        let mut x = x.clone();
        for (index, up) in self.up_blocks.iter().enumerate() {
            trace_decode_stage(&format!("up.{index}.input"), &x)?;
            let current_edge = tile_edge_at(remaining_scale);
            let next_scale = if up.upsamples() {
                (remaining_scale / 2).max(1)
            } else {
                remaining_scale
            };
            x = up.forward_tiled_decode(&x, current_edge, tile_edge_at(next_scale), cancel)?;
            trace_decode_stage(&format!("up.{index}.output"), &x)?;
            remaining_scale = next_scale;
        }
        if remaining_scale != 1 {
            return Err(mlx_gen::Error::Msg(format!(
                "SDXL tiled VAE tail ended at remaining spatial scale {remaining_scale}, expected 1"
            )));
        }
        if cancel.is_some_and(mlx_gen::CancelFlag::is_cancelled) {
            return Err(mlx_gen::Error::Canceled);
        }
        let norm = mlx_gen::vae_tiling::GlobalGroupNorm::new(
            &x,
            &self.norm_out_w,
            &self.norm_out_b,
            GN_GROUPS,
            GN_EPS,
        )?;
        mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc(
            &x,
            &self.conv_out_w,
            Some(&self.conv_out_b),
            output_tile_edge.max(3),
            cancel,
            |tile| silu(&norm.apply(tile)?),
        )
    }
}

/// The encoder (image → latent moments).
struct Encoder {
    conv_in_w: Array,
    conv_in_b: Array,
    down_blocks: Vec<EncoderDecoderBlock2D>,
    mid_resnet0: ResnetBlock2D,
    mid_attn: VaeAttention,
    mid_resnet1: ResnetBlock2D,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out_w: Array,
    conv_out_b: Array,
}

impl Encoder {
    fn from_weights(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let n = cfg.block_out_channels.len();
        let num_resnets = cfg.layers_per_block;
        let down_blocks = (0..n)
            .map(|i| {
                EncoderDecoderBlock2D::from_weights(
                    w,
                    &format!("encoder.down_blocks.{i}"),
                    num_resnets,
                    i < n - 1,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in_w: crate::unet::nchw_to_nhwc(w.require("encoder.conv_in.weight")?)?,
            conv_in_b: w.require("encoder.conv_in.bias")?.clone(),
            down_blocks,
            mid_resnet0: ResnetBlock2D::from_weights(w, "encoder.mid_block.resnets.0")?,
            mid_attn: VaeAttention::from_weights(w, "encoder.mid_block.attentions.0")?,
            mid_resnet1: ResnetBlock2D::from_weights(w, "encoder.mid_block.resnets.1")?,
            norm_out_w: w.require("encoder.conv_norm_out.weight")?.clone(),
            norm_out_b: w.require("encoder.conv_norm_out.bias")?.clone(),
            conv_out_w: crate::unet::nchw_to_nhwc(w.require("encoder.conv_out.weight")?)?,
            conv_out_b: w.require("encoder.conv_out.bias")?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = conv2d(x, &self.conv_in_w, Some(&self.conv_in_b), 1, 1)?;
        for db in &self.down_blocks {
            x = db.forward(&x)?;
        }
        x = self.mid_resnet0.forward(&x, None)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_resnet1.forward(&x, None)?;
        let x = group_norm(&x, &self.norm_out_w, &self.norm_out_b, GN_GROUPS, GN_EPS)?;
        conv2d(&silu(&x)?, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)
    }
}

/// The SDXL autoencoder. `decode` reconstructs an image from latents; `encode` produces the latent
/// mean (used to seed img2img).
pub struct Autoencoder {
    encoder: Encoder,
    decoder: Decoder,
    /// `quant_conv` [8,8,1,1] → [8,8] Linear over channels.
    quant_proj: AdaptableLinear,
    /// `post_quant_conv` [4,4,1,1] → [4,4] Linear over channels.
    post_quant_proj: AdaptableLinear,
    scaling_factor: f32,
    /// The compute dtype this VAE was loaded at — the dtype of its weights. `Float32` is the
    /// historical in-snapshot VAE (the stock SDXL VAE overflows at f16); `Float16` is the staged
    /// fp16-fix VAE (sc-26268, [`crate::loader::VAE_FP16_FIX_COMPONENT`]).
    dtype: Dtype,
}

impl Autoencoder {
    pub fn from_weights(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let squeeze_lin = |name: &str| -> Result<AdaptableLinear> {
            let cw = w.require(&format!("{name}.weight"))?;
            let sh = cw.shape();
            Ok(AdaptableLinear::dense(
                cw.reshape(&[sh[0], sh[1]])?,
                Some(w.require(&format!("{name}.bias"))?.clone()),
            ))
        };
        Ok(Self {
            encoder: Encoder::from_weights(w, cfg)?,
            decoder: Decoder::from_weights(w, cfg)?,
            quant_proj: squeeze_lin("quant_conv")?,
            post_quant_proj: squeeze_lin("post_quant_conv")?,
            scaling_factor: cfg.scaling_factor,
            dtype: w.require("decoder.conv_in.weight")?.dtype(),
        })
    }

    /// The compute dtype this VAE runs at (the dtype of its weights).
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Bring a caller tensor into this VAE's compute dtype.
    ///
    /// **An f32 VAE leaves its input untouched**, so the historical path keeps its exact op graph
    /// (an f16 latent promotes inside the first f32 op, as it always has) and stays byte-identical.
    /// A half-precision VAE casts explicitly: left to promotion, the first f32 scalar would lift the
    /// whole decode back to f32 and silently forfeit the memory the fp16 VAE exists to save.
    fn to_compute(&self, x: &Array) -> Result<Array> {
        if self.dtype == Dtype::Float32 || x.dtype() == self.dtype {
            Ok(x.clone())
        } else {
            Ok(x.as_dtype(self.dtype)?)
        }
    }

    /// Hand a half-precision VAE's output back at f32 — every consumer (`decoded_to_image`, the
    /// img2img init latents, the sampler's f32 step math) was written against the f32 VAE's
    /// output dtype. An f32 VAE's output is returned as is.
    fn to_output(&self, x: Array) -> Result<Array> {
        if self.dtype == Dtype::Float32 {
            Ok(x)
        } else {
            Ok(x.as_dtype(Dtype::Float32)?)
        }
    }

    /// `1 / scaling_factor` (or `scaling_factor`) as a scalar in this VAE's compute dtype, so the
    /// (de)normalization does not promote a half-precision decode to f32.
    fn scale(&self, value: f32) -> Result<Array> {
        let s = scalar(value);
        if self.dtype == Dtype::Float32 {
            Ok(s)
        } else {
            Ok(s.as_dtype(self.dtype)?)
        }
    }

    /// Decode latents `[B, H/8, W/8, 4]` (NHWC) → image tensor `[B, H, W, 3]` in roughly `[-1, 1]`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let latents = self.to_compute(latents)?;
        let z = multiply(&latents, self.scale(1.0 / self.scaling_factor)?)?;
        let z = self.post_quant_proj.forward(&z)?;
        self.to_output(self.decoder.forward(&z)?)
    }

    /// Untiled, evaluated inference decode. Uses exactly [`decode`](Self::decode)'s full-image
    /// convolutions, global normalization and attention, but materializes the head and every tail
    /// convolution before constructing the next. Intended for memory-bounded dense reference
    /// generation; this is NOT a tiled reference and does not change convolution geometry.
    /// Do not use under autograd (evaluation intentionally releases the lazy graph).
    pub fn decode_evaluated(&self, latents: &Array) -> Result<Array> {
        let latents = self.to_compute(latents)?;
        let z = multiply(&latents, self.scale(1.0 / self.scaling_factor)?)?;
        let z = self.post_quant_proj.forward(&z)?;
        let head = self.decoder.head(&z)?;
        head.eval()?;
        self.to_output(self.decoder.tail_evaluated(&head)?)
    }

    /// Ladder rung 2 — **bounded decode** (SC-15525): the same decode with each tail convolution
    /// evaluated on halo-expanded tiles and assembled from non-overlapping output cores, bounding
    /// convolution work without changing full-image GroupNorm statistics.
    ///
    /// `cfg` is the requested geometry; when it does not actually tile this latent
    /// ([`TilingConfig::needs_tiling`](mlx_gen::tiling::TilingConfig::needs_tiling)) the call falls through to the exact single-pass
    /// [`decode`](Self::decode) rather than assembling a one-tile plan, so a small render pays
    /// nothing. `cfg.spatial.tile_px` bounds each convolution crop. The configured overlap remains
    /// part of the public tiling contract and policy identity, but layer-wise halo/core arithmetic
    /// does not blend overlapping whole-tail outputs.
    ///
    /// **Normalization semantics.** Denormalize, `post_quant_conv`, `conv_in`, the mid resnets and
    /// mid **attention** run once on the full latent (`Decoder::head`). In the tail, every GroupNorm
    /// reduces the full layer activation; only halo-expanded 3×3 convolution work tiles. This fixes
    /// the per-crop normalization drift measured by sc-19753 without widening the quality bar.
    ///
    /// `cancel` is checked between tiles by the shared loop; decode is a dominant fraction of an SDXL
    /// render's wall clock and previously had no cancellation point at all.
    pub fn decode_tiled(
        &self,
        latents: &Array,
        cfg: &mlx_gen::tiling::TilingConfig,
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(mlx_gen::CancelFlag::is_cancelled) {
            return Err(mlx_gen::Error::Canceled);
        }
        let _trace = DecodeTrace::for_tile(cfg.spatial.as_ref().map_or(0, |s| s.tile_px));
        trace_decode_stage("latent", latents)?;
        let sh = latents.shape();
        if sh.len() != 4 {
            return Err(mlx_gen::Error::Msg(format!(
                "sdxl tiled decode expects NHWC latents [B, H, W, C], got {sh:?}"
            )));
        }
        let (hl, wl) = (sh[1], sh[2]);
        if !cfg.needs_tiling(VAE_TILING, 1, hl, wl) {
            return self.decode(latents);
        }
        let latents = self.to_compute(latents)?;
        let z = multiply(&latents, self.scale(1.0 / self.scaling_factor)?)?;
        let z = self.post_quant_proj.forward(&z)?;
        // Head once on the whole latent. The tail tiles layer-by-layer so every GroupNorm keeps the
        // dense image's statistics while each 3×3 convolution stays bounded by the selected edge.
        let head = self.decoder.head(&z)?;
        trace_decode_stage("head", &head)?;
        let tile_edge = cfg
            .spatial
            .as_ref()
            .ok_or_else(|| mlx_gen::Error::Msg("SDXL tiled decode requires spatial tiling".into()))?
            .tile_px;
        let out = self.decoder.tail_tiled(&head, tile_edge, cancel)?;
        trace_decode_stage("output_compute", &out)?;
        let out = self.to_output(out)?;
        trace_decode_stage("output_f32", &out)?;
        Ok(out)
    }

    /// Encode an image `[B, 3, ...]`-normalized NHWC `[B, H, W, 3]` → latent **mean** `[B, H/8, W/8,
    /// 4]` (scaled by `scaling_factor`). Mirrors the vendored `Autoencoder.encode` (returns the mean;
    /// img2img seeds from the mean, not a sample).
    pub fn encode_mean(&self, x: &Array) -> Result<Array> {
        let x = self.to_compute(x)?;
        let moments = self.quant_proj.forward(&self.encoder.forward(&x)?)?;
        // split into (mean, logvar) along the channel axis; keep the mean · scaling_factor.
        let c = moments.shape()[3];
        let half = c / 2;
        let idx = Array::from_slice(&(0..half).collect::<Vec<i32>>(), &[half]);
        let mean = moments.take_axis(&idx, 3)?;
        self.to_output(multiply(&mean, self.scale(self.scaling_factor)?)?)
    }
}

/// NCHW adapter for the backend-generic decode seam. The native SDXL VAE is internally NHWC, so the
/// wrapper performs only the two layout transposes around its established decode methods; the VAE
/// still owns SDXL's `1 / scaling_factor` de-normalization and all native tile geometry.
pub struct SdxlLatentDecoder<'a> {
    vae: &'a Autoencoder,
}

impl<'a> SdxlLatentDecoder<'a> {
    pub fn new(vae: &'a Autoencoder) -> Self {
        Self { vae }
    }

    fn to_nhwc(latents: &Array) -> Result<Array> {
        let shape = latents.shape();
        if shape.len() != 4 || shape[1] != mlx_gen::gen_core::SDXL_LATENT_SPACE.channels as i32 {
            return Err(Error::Msg(format!(
                "SDXL decoder expects NCHW [B,4,H,W], got {shape:?}"
            )));
        }
        Ok(latents.transpose_axes(&[0, 2, 3, 1])?)
    }

    fn to_nchw(decoded: &Array) -> Result<Array> {
        let shape = decoded.shape();
        if shape.len() != 4 || shape[3] != 3 {
            return Err(Error::Msg(format!(
                "SDXL decoder produced invalid NHWC [B,H,W,3] output {shape:?}"
            )));
        }
        Ok(decoded.transpose_axes(&[0, 3, 1, 2])?)
    }
}

impl LatentDecoder for SdxlLatentDecoder<'_> {
    fn input_latent_space(&self) -> Option<&mlx_gen::gen_core::LatentSpace> {
        Some(&mlx_gen::gen_core::SDXL_LATENT_SPACE)
    }

    fn decode(&self, latents: &Array) -> Result<Array> {
        Self::to_nchw(&self.vae.decode(&Self::to_nhwc(latents)?)?)
    }

    fn decode_tiled(
        &self,
        latents: &Array,
        tiling: &mlx_gen::tiling::TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        Self::to_nchw(
            &self
                .vae
                .decode_tiled(&Self::to_nhwc(latents)?, tiling, cancel)?,
        )
    }
}

#[cfg(test)]
mod decoder_seam_tests {
    use super::*;

    /// SC-18309 N1's weight-free layout gate. The engine historically passed NHWC latents directly
    /// to the VAE while the generic seam accepts NCHW; decoded output travels back as RGB NHWC and
    /// the seam returns RGB NCHW. Distinct values in every axis prove both production transposes
    /// preserve bytes without weakening the decoded-output RGB shape guard.
    #[test]
    fn sdxl_seam_layout_transposes_are_byte_exact() {
        let nchw_values = (0..(2 * 4 * 3 * 5))
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let nchw = Array::from_slice(&nchw_values, &[2, 4, 3, 5]);
        let nhwc = SdxlLatentDecoder::to_nhwc(&nchw).unwrap();
        assert_eq!(nhwc.shape(), &[2, 3, 5, 4]);
        let mut expected_nhwc = Vec::with_capacity(nchw_values.len());
        for batch in 0..2 {
            for height in 0..3 {
                for width in 0..5 {
                    for channel in 0..4 {
                        expected_nhwc
                            .push(nchw_values[((batch * 4 + channel) * 3 + height) * 5 + width]);
                    }
                }
            }
        }
        assert_eq!(
            nhwc.reshape(&[-1]).unwrap().as_slice::<f32>(),
            expected_nhwc
        );

        let rgb_nhwc_values = (0..(2 * 3 * 5 * 3))
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let rgb_nhwc = Array::from_slice(&rgb_nhwc_values, &[2, 3, 5, 3]);
        let rgb_nchw = SdxlLatentDecoder::to_nchw(&rgb_nhwc).unwrap();
        assert_eq!(rgb_nchw.shape(), &[2, 3, 3, 5]);
        let mut expected_nchw = Vec::with_capacity(rgb_nhwc_values.len());
        for batch in 0..2 {
            for channel in 0..3 {
                for height in 0..3 {
                    for width in 0..5 {
                        expected_nchw.push(
                            rgb_nhwc_values[((batch * 3 + height) * 5 + width) * 3 + channel],
                        );
                    }
                }
            }
        }
        assert_eq!(
            rgb_nchw.reshape(&[-1]).unwrap().as_slice::<f32>(),
            expected_nchw
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use super::*;
    use mlx_gen::tiling::TilingConfig;
    use mlx_gen::{CancelFlag, Error};

    const CHANNELS: i32 = 32;

    /// A synthetic tensor generator: `(shape, phase, scale) -> Array`.
    type Gen = fn(&[i32], f32, f32) -> Array;

    /// Smooth, position-correlated values (period ~88 elements). Neighbouring taps are nearly
    /// equal, so this fixture's convolutions act as smoothing filters — exactly what makes per-crop
    /// GroupNorm statistics drift, which is the geometry test's subject.
    fn values(shape: &[i32], phase: f32, scale: f32) -> Array {
        let count = shape.iter().product::<i32>();
        let values = (0..count)
            .map(|i| ((i as f32 + phase) * 0.071).sin() * scale)
            .collect::<Vec<_>>();
        Array::from_slice(&values, shape)
    }

    /// Decorrelated values (a fixed hash per element). A trained decoder's filters are not
    /// smoothing kernels, and the smooth generator above leaves every GroupNorm input nearly
    /// constant across space — a catastrophic-cancellation case no trained VAE presents, which
    /// would make a precision tolerance measure the fixture rather than the arithmetic.
    fn decorrelated(shape: &[i32], phase: f32, scale: f32) -> Array {
        let count = shape.iter().product::<i32>();
        let values = (0..count)
            .map(|i| {
                let h = ((i as f32) * 12.9898 + phase * 78.233).sin() * 43_758.547;
                (h - h.floor()) * 2.0 - 1.0
            })
            .map(|v| v * scale)
            .collect::<Vec<_>>();
        Array::from_slice(&values, shape)
    }

    fn insert_conv(
        gen: Gen,
        tensors: &mut HashMap<String, Array>,
        prefix: &str,
        input: i32,
        output: i32,
        kernel: i32,
        phase: f32,
    ) {
        tensors.insert(
            format!("{prefix}.weight"),
            gen(&[output, input, kernel, kernel], phase, 0.025),
        );
        tensors.insert(format!("{prefix}.bias"), gen(&[output], phase + 3.0, 0.01));
    }

    fn insert_resnet(gen: Gen, tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
        for (name, offset) in [("norm1", 0.0), ("norm2", 1.0)] {
            tensors.insert(
                format!("{prefix}.{name}.weight"),
                add(gen(&[CHANNELS], phase + offset, 0.15), Array::from_f32(1.0)).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.{name}.bias"),
                gen(&[CHANNELS], phase + offset + 2.0, 0.05),
            );
        }
        insert_conv(
            gen,
            tensors,
            &format!("{prefix}.conv1"),
            CHANNELS,
            CHANNELS,
            3,
            phase + 4.0,
        );
        insert_conv(
            gen,
            tensors,
            &format!("{prefix}.conv2"),
            CHANNELS,
            CHANNELS,
            3,
            phase + 5.0,
        );
    }

    fn insert_attention(gen: Gen, tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
        tensors.insert(
            format!("{prefix}.group_norm.weight"),
            Array::ones::<f32>(&[CHANNELS]).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.group_norm.bias"),
            gen(&[CHANNELS], phase, 0.03),
        );
        for (index, name) in ["to_q", "to_k", "to_v", "to_out.0"].iter().enumerate() {
            tensors.insert(
                format!("{prefix}.{name}.weight"),
                gen(&[CHANNELS, CHANNELS], phase + index as f32 + 1.0, 0.02),
            );
            tensors.insert(
                format!("{prefix}.{name}.bias"),
                gen(&[CHANNELS], phase + index as f32 + 5.0, 0.005),
            );
        }
    }

    fn insert_mid(gen: Gen, tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
        insert_resnet(gen, tensors, &format!("{prefix}.resnets.0"), phase);
        insert_attention(
            gen,
            tensors,
            &format!("{prefix}.attentions.0"),
            phase + 10.0,
        );
        insert_resnet(gen, tensors, &format!("{prefix}.resnets.1"), phase + 20.0);
    }

    /// A tiny synthetic SDXL-layout VAE (32 channels, one upsample) and its config.
    pub(crate) fn fixture() -> (Weights, VaeConfig) {
        fixture_with(values)
    }

    /// [`fixture`] over the [`decorrelated`] generator — the precision tests' fixture.
    fn decorrelated_fixture() -> (Weights, VaeConfig) {
        fixture_with(decorrelated)
    }

    fn fixture_with(gen: Gen) -> (Weights, VaeConfig) {
        let cfg = VaeConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels_out: 8,
            latent_channels_in: 4,
            block_out_channels: vec![CHANNELS, CHANNELS],
            layers_per_block: 1,
            norm_num_groups: 32,
            scaling_factor: 0.13025,
        };
        let mut tensors = HashMap::new();

        insert_conv(gen, &mut tensors, "decoder.conv_in", 4, CHANNELS, 3, 1.0);
        insert_mid(gen, &mut tensors, "decoder.mid_block", 10.0);
        for block in 0..2 {
            for resnet in 0..2 {
                insert_resnet(
                    gen,
                    &mut tensors,
                    &format!("decoder.up_blocks.{block}.resnets.{resnet}"),
                    40.0 + (block * 10 + resnet) as f32,
                );
            }
        }
        insert_conv(
            gen,
            &mut tensors,
            "decoder.up_blocks.0.upsamplers.0.conv",
            CHANNELS,
            CHANNELS,
            3,
            70.0,
        );
        tensors.insert(
            "decoder.conv_norm_out.weight".into(),
            Array::ones::<f32>(&[CHANNELS]).unwrap(),
        );
        tensors.insert(
            "decoder.conv_norm_out.bias".into(),
            gen(&[CHANNELS], 72.0, 0.04),
        );
        insert_conv(gen, &mut tensors, "decoder.conv_out", CHANNELS, 3, 3, 74.0);

        insert_conv(gen, &mut tensors, "encoder.conv_in", 3, CHANNELS, 3, 80.0);
        for block in 0..2 {
            insert_resnet(
                gen,
                &mut tensors,
                &format!("encoder.down_blocks.{block}.resnets.0"),
                90.0 + block as f32 * 10.0,
            );
        }
        insert_conv(
            gen,
            &mut tensors,
            "encoder.down_blocks.0.downsamplers.0.conv",
            CHANNELS,
            CHANNELS,
            3,
            112.0,
        );
        insert_mid(gen, &mut tensors, "encoder.mid_block", 120.0);
        tensors.insert(
            "encoder.conv_norm_out.weight".into(),
            Array::ones::<f32>(&[CHANNELS]).unwrap(),
        );
        tensors.insert(
            "encoder.conv_norm_out.bias".into(),
            gen(&[CHANNELS], 145.0, 0.04),
        );
        insert_conv(gen, &mut tensors, "encoder.conv_out", CHANNELS, 8, 3, 150.0);
        insert_conv(gen, &mut tensors, "quant_conv", 8, 8, 1, 160.0);
        insert_conv(gen, &mut tensors, "post_quant_conv", 4, 4, 1, 170.0);

        (Weights::from_map(tensors), cfg)
    }

    fn latent() -> Array {
        let shape = [1, 8, 9, 4];
        let count = shape.iter().product::<i32>();
        let values = (0..count)
            .map(|i| {
                let y = (i / shape[3] / shape[2]) as f32;
                let x = (i / shape[3] % shape[2]) as f32;
                (i as f32 * 0.037).sin() + y * 0.11 - x * 0.07
            })
            .collect::<Vec<_>>();
        Array::from_slice(&values, &shape)
    }

    /// This test crosses the public autoencoder seam. The synthetic input is deliberately
    /// position-dependent so replacing global GroupNorm statistics with per-crop statistics makes
    /// the assertion fail even though both paths retain the same shape.
    #[test]
    fn tiled_autoencoder_decode_tracks_dense_with_global_group_norm() {
        let (weights, cfg) = fixture();
        let vae = Autoencoder::from_weights(&weights, &cfg).unwrap();
        let latent = latent();
        let dense = vae.decode(&latent).unwrap();
        let tiled = vae
            .decode_tiled(&latent, &TilingConfig::spatial_only(8, 0), None)
            .unwrap();
        dense.eval().unwrap();
        tiled.eval().unwrap();
        assert_eq!(tiled.shape(), dense.shape());

        let max_delta = dense
            .as_slice::<f32>()
            .iter()
            .zip(tiled.as_slice::<f32>())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_delta < 3e-3,
            "layer-wise tiled SDXL VAE diverged from dense decode: max|delta|={max_delta:.3e}"
        );
    }

    /// Evaluated reference keeps the untiled arithmetic, byte for byte, while cutting the lazy
    /// graph's lifetime. This compares both output and allocator high-water on a decorrelated
    /// tiny VAE: replacing `decode_evaluated` with the lazy decode must fail the peak comparison.
    #[test]
    fn evaluated_untiled_decode_preserves_pixels_and_reduces_peak() {
        let _cpu = crate::test_device::cpu_only();
        let (weights, cfg) = decorrelated_fixture();
        let vae = f32_reference(&weights, &cfg);
        let z = latent();
        z.eval().unwrap();
        let run = |evaluated| {
            mlx_rs::memory::clear_cache();
            mlx_rs::memory::reset_peak_memory();
            let out = if evaluated {
                vae.decode_evaluated(&z).unwrap()
            } else {
                vae.decode(&z).unwrap()
            };
            out.eval().unwrap();
            let peak = mlx_rs::memory::get_peak_memory();
            (f32_values(&out), peak)
        };
        // Warm the weight casts before measuring activation peaks.
        drop(run(false));
        let (dense, lazy_peak) = run(false);
        let (evaluated, evaluated_peak) = run(true);
        eprintln!("dense reference peak bytes: lazy={lazy_peak} evaluated={evaluated_peak}");
        assert_eq!(
            dense, evaluated,
            "evaluating must not alter dense arithmetic"
        );
        assert!(
            evaluated_peak < lazy_peak,
            "evaluation must release transients: evaluated {evaluated_peak}, lazy {lazy_peak}"
        );
    }

    /// The declared tolerance for a **half-precision** decode against the f32 untiled decode of
    /// the same weights (sc-26268), as a fraction of the reference output's peak magnitude: 1%.
    /// f16 carries an 11-bit significand (relative rounding 4.9e-4 per op); a decoder chains ~20
    /// normalized conv/attention stages, so a few ulps of compounding is the expected class, and a
    /// promotion or overflow defect (the failure this guards) lands orders of magnitude above it.
    /// On SDXL's real `[-1, 1]` output 1% is 1.3/255 — below one 8-bit level in the worst pixel.
    const FP16_RELATIVE_TOLERANCE: f32 = 1e-2;

    /// The declared tolerance for an **f32** tiled decode against the f32 untiled decode: the same
    /// arithmetic up to float reassociation of the halo-cropped convolutions, so 1e-4 of peak.
    const F32_TILED_RELATIVE_TOLERANCE: f32 = 1e-4;

    /// Output tile edges for the synthetic fixture (its output is 16×18): one that does not tile,
    /// even and ragged splits, and the 3-pixel floor `tiled_conv2d_3x3_nhwc` accepts.
    const SYNTHETIC_EDGES: [i32; 5] = [32, 9, 8, 5, 3];

    fn f32_values(a: &Array) -> Vec<f32> {
        let a = a.as_dtype(Dtype::Float32).unwrap();
        a.eval().unwrap();
        a.as_slice::<f32>().to_vec()
    }

    /// `max|a - b| / max|reference|`.
    fn relative_max_delta(reference: &Array, candidate: &Array) -> f32 {
        assert_eq!(reference.shape(), candidate.shape());
        let reference = f32_values(reference);
        let candidate = f32_values(candidate);
        let peak = reference.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        assert!(
            peak > 0.0,
            "a zero reference cannot ground a relative tolerance"
        );
        let delta = reference
            .iter()
            .zip(&candidate)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        delta / peak
    }

    fn fp16_vae(weights: &Weights, cfg: &VaeConfig) -> Autoencoder {
        let mut half = weights.clone();
        half.cast_all(Dtype::Float16).unwrap();
        Autoencoder::from_weights(&half, cfg).unwrap()
    }

    /// The f32 reference for a half-precision VAE: the **same** f16-stored weights, computed at
    /// f32 — exactly the comparison the real fp16-fix checkpoint (stored f16) is held to, so the
    /// tolerance measures compute precision rather than the weights' storage rounding.
    fn f32_reference(weights: &Weights, cfg: &VaeConfig) -> Autoencoder {
        let mut rounded = weights.clone();
        rounded.cast_all(Dtype::Float16).unwrap();
        rounded.cast_all(Dtype::Float32).unwrap();
        Autoencoder::from_weights(&rounded, cfg).unwrap()
    }

    /// sc-26268 AC: the **fp16 VAE decode** matches the f32 untiled decode of the same weights
    /// within [`FP16_RELATIVE_TOLERANCE`], runs at f16 (no silent promotion back to f32), and hands
    /// its output back at f32.
    #[test]
    fn fp16_decode_matches_the_f32_untiled_decode_within_the_declared_tolerance() {
        let _cpu = crate::test_device::cpu_only();
        let (weights, cfg) = decorrelated_fixture();
        let reference = f32_reference(&weights, &cfg);
        assert_eq!(reference.dtype(), Dtype::Float32);
        let half = fp16_vae(&weights, &cfg);
        assert_eq!(half.dtype(), Dtype::Float16);

        // The production latent arrives at the U-Net's f16.
        let latent = latent().as_dtype(Dtype::Float16).unwrap();
        let dense = reference.decode(&latent).unwrap();
        assert_eq!(dense.dtype(), Dtype::Float32);
        let decoded = half.decode(&latent).unwrap();
        assert_eq!(decoded.dtype(), Dtype::Float32, "callers receive f32");
        let delta = relative_max_delta(&dense, &decoded);
        assert!(
            delta <= FP16_RELATIVE_TOLERANCE,
            "fp16 decode drifted {delta:.3e} of peak from the f32 decode (tolerance \
             {FP16_RELATIVE_TOLERANCE:.0e})"
        );
        // Non-vacuity: f16 rounding is real, so an identical output would mean the "f16" VAE never
        // ran at f16 (a promotion back to f32 that forfeits the memory saving).
        assert!(delta > 0.0, "the f16 decode must actually compute at f16");

        // The img2img encode follows the same contract.
        let image = values(&[1, 16, 18, 3], 3.0, 0.8);
        let encoded = half.encode_mean(&image).unwrap();
        assert_eq!(encoded.dtype(), Dtype::Float32);
        let delta = relative_max_delta(&reference.encode_mean(&image).unwrap(), &encoded);
        assert!(
            delta <= FP16_RELATIVE_TOLERANCE,
            "fp16 encode drifted {delta:.3e}"
        );
    }

    /// sc-26268 AC: the **fixed tiled decode** (layer-wise halo tiling + full-image GroupNorm)
    /// matches the f32 untiled decode — within [`F32_TILED_RELATIVE_TOLERANCE`] at f32 and within
    /// [`FP16_RELATIVE_TOLERANCE`] on the fp16 VAE — at every edge, including ragged splits. The
    /// pre-SC-19753 per-crop normalization moved this position-dependent fixture by orders of
    /// magnitude more (see `tiled_autoencoder_decode_tracks_dense_with_global_group_norm`).
    #[test]
    fn tiled_decode_matches_the_f32_untiled_decode_across_edges() {
        let _cpu = crate::test_device::cpu_only();
        let (weights, cfg) = decorrelated_fixture();
        let reference = f32_reference(&weights, &cfg);
        let half = fp16_vae(&weights, &cfg);
        let latent = latent().as_dtype(Dtype::Float16).unwrap();
        let dense = reference.decode(&latent).unwrap();
        for edge in SYNTHETIC_EDGES {
            let tiling = TilingConfig::spatial_only(edge, 0);
            let tiled = reference.decode_tiled(&latent, &tiling, None).unwrap();
            let delta = relative_max_delta(&dense, &tiled);
            assert!(
                delta <= F32_TILED_RELATIVE_TOLERANCE,
                "f32 tiled decode at edge {edge} drifted {delta:.3e} of peak"
            );
            let tiled_half = half.decode_tiled(&latent, &tiling, None).unwrap();
            assert_eq!(tiled_half.dtype(), Dtype::Float32);
            let delta = relative_max_delta(&dense, &tiled_half);
            assert!(
                delta <= FP16_RELATIVE_TOLERANCE,
                "fp16 tiled decode at edge {edge} drifted {delta:.3e} of peak"
            );
        }
    }

    /// The tracer must distinguish finite corruption, NaN and infinity before RGB conversion
    /// clips or casts them away. Explicit trace state avoids process-environment mutation in tests.
    #[test]
    fn decode_trace_counts_nonfinite_values_and_restores_scope() {
        let _cpu = crate::test_device::cpu_only();
        let x = Array::from_slice(
            &[-2.0f32, 3.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            &[1, 1, 5, 1],
        );
        let stats = decode_stage_stats(&x).unwrap();
        assert_eq!(
            (stats.min, stats.max, stats.nan, stats.inf),
            (-2.0, 3.0, 1, 2)
        );
        TRACE_DECODE.with(|active| active.set(false));
        {
            let _guard = DecodeTrace(TRACE_DECODE.with(|active| active.replace(true)));
            assert!(TRACE_DECODE.with(|active| active.get()));
            trace_decode_stage("test.nonfinite", &x).unwrap();
        }
        assert!(!TRACE_DECODE.with(|active| active.get()));
        // A traced finite synthetic decode has exactly the same output as the untraced call.
        let (weights, cfg) = decorrelated_fixture();
        let half = fp16_vae(&weights, &cfg);
        let z = latent().as_dtype(Dtype::Float16).unwrap();
        let decode = || {
            half.decoder
                .tail_tiled(&half.decoder.head(&z).unwrap(), 8, None)
                .unwrap()
        };
        let plain = f32_values(&decode());
        let _guard = DecodeTrace(TRACE_DECODE.with(|active| active.replace(true)));
        let traced = f32_values(&decode());
        assert_eq!(
            plain, traced,
            "stage observation must not change the output"
        );
    }

    /// Real failing kernel geometry with synthetic weights: the upsample feeds a 256²×256 half
    /// activation to a 256-channel 3×3 convolution. Positive partial sums exceed f16's range
    /// before negative terms cancel them; the mathematical result (including bias) is one.
    #[test]
    fn half_upsample_conv_keeps_cancelling_high_range_dot_products_finite() {
        for cpu in [true, false] {
            let _guard = cpu.then(crate::test_device::cpu_only);
            let x = Array::full::<f32>(&[1, 128, 128, 256], scalar(1954.0))
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let v = (0..256 * 3 * 3 * 256)
                .map(|i| if i % 256 < 128 { 0.5f32 } else { -0.5 })
                .collect::<Vec<_>>();
            let weight = Array::from_slice(&v, &[256, 3, 3, 256])
                .as_dtype(Dtype::Float16)
                .unwrap();
            let bias = Array::ones::<f32>(&[256])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let up = upsample_nearest(&x, 2).unwrap();
            let up_stats = decode_stage_stats(&up).unwrap();
            assert_eq!(
                (up_stats.nan, up_stats.inf),
                (0, 0),
                "nearest-neighbor must stay finite"
            );
            // Same kernel geometry and weights at low amplitude: no nonfinite values. This
            // distinguishes high-range accumulation from an unconditional shape/indexing defect.
            let low = conv2d(
                &multiply(&up, scalar(1.0 / 1954.0).as_dtype(Dtype::Float16).unwrap()).unwrap(),
                &weight,
                Some(&bias),
                1,
                1,
            )
            .unwrap();
            let low_stats = decode_stage_stats(&low).unwrap();
            assert_eq!((low_stats.nan, low_stats.inf), (0, 0));
            let expected = conv2d(
                &up.as_dtype(Dtype::Float32).unwrap(),
                &weight.as_dtype(Dtype::Float32).unwrap(),
                Some(&bias.as_dtype(Dtype::Float32).unwrap()),
                1,
                1,
            )
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
            let got = upsample_tiled(&x, &weight, &bias, 256, None).unwrap();
            assert_eq!(got.dtype(), Dtype::Float16, "storage must remain half");
            let stats = decode_stage_stats(&got).unwrap();
            eprintln!("cpu={cpu}: high-range upsample conv {stats:?}");
            assert_eq!(
                (stats.nan, stats.inf),
                (0, 0),
                "finite cancellation cannot overflow a partial sum"
            );
            assert!(
                relative_max_delta(&expected, &got) <= 1e-3,
                "wide accumulation must match the f32 convolution"
            );
        }
    }

    /// Scaled analogue of the failing real-weight cell: 4×4+ragged core grids at half precision,
    /// after other decode sizes have populated/released allocator buffers. Tiny synthetic weights,
    /// CPU only; retains a tight decoded-float comparison, not a clipped RGB-only comparison.
    #[test]
    fn fp16_tiled_ragged_grids_remain_finite_after_cache_reuse() {
        let _cpu = crate::test_device::cpu_only();
        let (weights, cfg) = decorrelated_fixture();
        let half = fp16_vae(&weights, &cfg);
        for edge in [16, 32, 64, 32] {
            let z = values(&[1, edge, edge, 4], 1.7, 0.2);
            let reference = f32_values(&half.decode_evaluated(&z).unwrap());
            for tile in [56, 48, 40, 32, 24, 16] {
                let decoded = half
                    .decode_tiled(&z, &TilingConfig::spatial_only(tile, 0), None)
                    .unwrap();
                let got = f32_values(&decoded);
                assert!(
                    got.iter().all(|v| v.is_finite()),
                    "edge={edge} tile={tile}: non-finite"
                );
                let peak = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
                let max = reference
                    .iter()
                    .zip(&got)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(
                    max <= 1e-2 * peak,
                    "edge={edge} tile={tile}: drift {max} peak {peak}"
                );
                drop(decoded);
                mlx_rs::memory::clear_cache();
            }
        }
    }

    #[test]
    fn tiled_autoencoder_decode_honors_pretripped_cancel() {
        let (weights, cfg) = fixture();
        let vae = Autoencoder::from_weights(&weights, &cfg).unwrap();
        let cancel = CancelFlag::new();
        cancel.cancel();
        let malformed = Array::from_slice(&[1.0_f32], &[1]);
        for tiling in [
            TilingConfig::spatial_only(8, 0),
            TilingConfig::spatial_only(4096, 64),
        ] {
            let result = vae.decode_tiled(&malformed, &tiling, Some(&cancel));
            assert!(matches!(result, Err(Error::Canceled)));
        }
    }
}
