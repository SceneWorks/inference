//! SDXL VAE (autoencoder) — port of the vendored `_vendor/mlx_sd/vae.py`. The decoder powers T2I
//! (latents → image); the encoder powers img2img (sc-2638). Runs entirely in NHWC, reusing the
//! UNet [`ResnetBlock2D`] (temb-free here) and the core conv/group-norm primitives. SDXL's
//! `scaling_factor` is **0.13025** (not SD-2.1's 0.18215). Latents are scaled by it on encode and
//! divided by it on decode.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, multiply, pad};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::{conv2d, group_norm, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::VaeConfig;
use crate::unet::ResnetBlock2D;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;

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
        for resnet in &self.resnets {
            x = resnet.forward_tiled_vae(&x, tile_edge, cancel)?;
        }
        if let Some((cw, cb)) = &self.upsample {
            let up = upsample_nearest(&x, 2)?;
            x = mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc(
                &up,
                cw,
                Some(cb),
                upsampled_tile_edge,
                cancel,
                |tile| Ok(tile.clone()),
            )?;
        }
        Ok(x)
    }

    fn upsamples(&self) -> bool {
        self.upsample.is_some()
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
        for up in &self.up_blocks {
            let current_edge = tile_edge_at(remaining_scale);
            let next_scale = if up.upsamples() {
                (remaining_scale / 2).max(1)
            } else {
                remaining_scale
            };
            x = up.forward_tiled_decode(&x, current_edge, tile_edge_at(next_scale), cancel)?;
            remaining_scale = next_scale;
        }
        if remaining_scale != 1 {
            return Err(mlx_gen::Error::Msg(format!(
                "SDXL tiled VAE tail ended at remaining spatial scale {remaining_scale}, expected 1"
            )));
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
        })
    }

    /// Decode latents `[B, H/8, W/8, 4]` (NHWC) → image tensor `[B, H, W, 3]` in roughly `[-1, 1]`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let z = multiply(latents, scalar(1.0 / self.scaling_factor))?;
        let z = self.post_quant_proj.forward(&z)?;
        self.decoder.forward(&z)
    }

    /// Ladder rung 2 — **bounded decode** (SC-15525): the same decode with the full-resolution
    /// upsample tail run tile-by-tile and trapezoidally blended, bounding the decode's peak by one
    /// tile's activations instead of the whole image's.
    ///
    /// `cfg` is the requested geometry; when it does not actually tile this latent
    /// ([`TilingConfig::needs_tiling`](mlx_gen::tiling::TilingConfig::needs_tiling)) the call falls through to the exact single-pass
    /// [`decode`](Self::decode) rather than assembling a one-tile plan, so a small render pays
    /// nothing. The blend geometry is [`mlx_gen::tiling`] and the array loop is
    /// [`mlx_gen::vae_tiling::tiled_decode`] — this crate contributes only the head/tail split and
    /// the NHWC axis mapping.
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
        if cancel.is_some_and(mlx_gen::CancelFlag::is_cancelled) {
            return Err(mlx_gen::Error::Canceled);
        }
        let z = multiply(latents, scalar(1.0 / self.scaling_factor))?;
        let z = self.post_quant_proj.forward(&z)?;
        // Head once on the whole latent. The tail tiles layer-by-layer so every GroupNorm keeps the
        // dense image's statistics while each 3×3 convolution stays bounded by the selected edge.
        let head = self.decoder.head(&z)?;
        let tile_edge = cfg
            .spatial
            .as_ref()
            .ok_or_else(|| mlx_gen::Error::Msg("SDXL tiled decode requires spatial tiling".into()))?
            .tile_px;
        self.decoder.tail_tiled(&head, tile_edge, cancel)
    }

    /// Encode an image `[B, 3, ...]`-normalized NHWC `[B, H, W, 3]` → latent **mean** `[B, H/8, W/8,
    /// 4]` (scaled by `scaling_factor`). Mirrors the vendored `Autoencoder.encode` (returns the mean;
    /// img2img seeds from the mean, not a sample).
    pub fn encode_mean(&self, x: &Array) -> Result<Array> {
        let moments = self.quant_proj.forward(&self.encoder.forward(x)?)?;
        // split into (mean, logvar) along the channel axis; keep the mean · scaling_factor.
        let c = moments.shape()[3];
        let half = c / 2;
        let idx = Array::from_slice(&(0..half).collect::<Vec<i32>>(), &[half]);
        let mean = moments.take_axis(&idx, 3)?;
        Ok(multiply(&mean, scalar(self.scaling_factor))?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use mlx_gen::tiling::TilingConfig;
    use mlx_gen::{CancelFlag, Error};

    const CHANNELS: i32 = 32;

    fn values(shape: &[i32], phase: f32, scale: f32) -> Array {
        let count = shape.iter().product::<i32>();
        let values = (0..count)
            .map(|i| ((i as f32 + phase) * 0.071).sin() * scale)
            .collect::<Vec<_>>();
        Array::from_slice(&values, shape)
    }

    fn insert_conv(
        tensors: &mut HashMap<String, Array>,
        prefix: &str,
        input: i32,
        output: i32,
        kernel: i32,
        phase: f32,
    ) {
        tensors.insert(
            format!("{prefix}.weight"),
            values(&[output, input, kernel, kernel], phase, 0.025),
        );
        tensors.insert(
            format!("{prefix}.bias"),
            values(&[output], phase + 3.0, 0.01),
        );
    }

    fn insert_resnet(tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
        for (name, offset) in [("norm1", 0.0), ("norm2", 1.0)] {
            tensors.insert(
                format!("{prefix}.{name}.weight"),
                add(
                    &values(&[CHANNELS], phase + offset, 0.15),
                    Array::from_f32(1.0),
                )
                .unwrap(),
            );
            tensors.insert(
                format!("{prefix}.{name}.bias"),
                values(&[CHANNELS], phase + offset + 2.0, 0.05),
            );
        }
        insert_conv(
            tensors,
            &format!("{prefix}.conv1"),
            CHANNELS,
            CHANNELS,
            3,
            phase + 4.0,
        );
        insert_conv(
            tensors,
            &format!("{prefix}.conv2"),
            CHANNELS,
            CHANNELS,
            3,
            phase + 5.0,
        );
    }

    fn insert_attention(tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
        tensors.insert(
            format!("{prefix}.group_norm.weight"),
            Array::ones::<f32>(&[CHANNELS]).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.group_norm.bias"),
            values(&[CHANNELS], phase, 0.03),
        );
        for (index, name) in ["to_q", "to_k", "to_v", "to_out.0"].iter().enumerate() {
            tensors.insert(
                format!("{prefix}.{name}.weight"),
                values(&[CHANNELS, CHANNELS], phase + index as f32 + 1.0, 0.02),
            );
            tensors.insert(
                format!("{prefix}.{name}.bias"),
                values(&[CHANNELS], phase + index as f32 + 5.0, 0.005),
            );
        }
    }

    fn insert_mid(tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
        insert_resnet(tensors, &format!("{prefix}.resnets.0"), phase);
        insert_attention(tensors, &format!("{prefix}.attentions.0"), phase + 10.0);
        insert_resnet(tensors, &format!("{prefix}.resnets.1"), phase + 20.0);
    }

    fn fixture() -> (Weights, VaeConfig) {
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

        insert_conv(&mut tensors, "decoder.conv_in", 4, CHANNELS, 3, 1.0);
        insert_mid(&mut tensors, "decoder.mid_block", 10.0);
        for block in 0..2 {
            for resnet in 0..2 {
                insert_resnet(
                    &mut tensors,
                    &format!("decoder.up_blocks.{block}.resnets.{resnet}"),
                    40.0 + (block * 10 + resnet) as f32,
                );
            }
        }
        insert_conv(
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
            values(&[CHANNELS], 72.0, 0.04),
        );
        insert_conv(&mut tensors, "decoder.conv_out", CHANNELS, 3, 3, 74.0);

        insert_conv(&mut tensors, "encoder.conv_in", 3, CHANNELS, 3, 80.0);
        for block in 0..2 {
            insert_resnet(
                &mut tensors,
                &format!("encoder.down_blocks.{block}.resnets.0"),
                90.0 + block as f32 * 10.0,
            );
        }
        insert_conv(
            &mut tensors,
            "encoder.down_blocks.0.downsamplers.0.conv",
            CHANNELS,
            CHANNELS,
            3,
            112.0,
        );
        insert_mid(&mut tensors, "encoder.mid_block", 120.0);
        tensors.insert(
            "encoder.conv_norm_out.weight".into(),
            Array::ones::<f32>(&[CHANNELS]).unwrap(),
        );
        tensors.insert(
            "encoder.conv_norm_out.bias".into(),
            values(&[CHANNELS], 145.0, 0.04),
        );
        insert_conv(&mut tensors, "encoder.conv_out", CHANNELS, 8, 3, 150.0);
        insert_conv(&mut tensors, "quant_conv", 8, 8, 1, 160.0);
        insert_conv(&mut tensors, "post_quant_conv", 4, 4, 1, 170.0);

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

    #[test]
    fn tiled_autoencoder_decode_honors_pretripped_cancel() {
        let (weights, cfg) = fixture();
        let vae = Autoencoder::from_weights(&weights, &cfg).unwrap();
        let cancel = CancelFlag::new();
        cancel.cancel();
        let result = vae.decode_tiled(&latent(), &TilingConfig::spatial_only(8, 0), Some(&cancel));
        assert!(matches!(result, Err(Error::Canceled)));
    }
}
