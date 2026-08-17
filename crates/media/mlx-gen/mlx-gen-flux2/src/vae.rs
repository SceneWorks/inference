//! FLUX.2 VAE (`AutoencoderKLFlux2`) — a 32-channel diffusers AutoencoderKL with two FLUX.2
//! additions: a **2×2 patchify** that folds the 32-ch latent into the 128-ch transformer space,
//! and a **BatchNorm-stats** normalization of that packed space (`bn.running_mean/var`). Port of
//! the fork's `models/flux2/model/flux2_vae/`.
//!
//! Structurally identical to the SDXL VAE (encoder/decoder, resnets, single-head mid attention,
//! GroupNorm) but with `block_out_channels = (128, 256, 512, 512)`, `latent_channels = 32`,
//! GroupNorm eps **1e-6** (SDXL uses 1e-5), `scaling_factor = 1.0`, `shift_factor = 0.0`. Runs
//! entirely in NHWC, f32 (the VAE is small; f32 dodges the bf16-GEMM bug in the mid attention and
//! is the quality target).

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, multiply, pad, sqrt};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::{conv2d, group_norm, linear, silu, upsample_nearest};
use mlx_gen::vae_tiling::{tiled_conv2d_3x3_nhwc, GlobalGroupNorm};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;
const BN_EPS: f32 = 1e-4;
const LATENT_CHANNELS: i32 = 32;
const BLOCK_OUT: [i32; 4] = [128, 256, 512, 512];
const LAYERS_PER_BLOCK: i32 = 2;
const FLUX2_TILING: mlx_gen::tiling::VaeTiling = mlx_gen::tiling::VaeTiling {
    spatial_scale: 8,
    temporal_scale: 1,
    causal_temporal: false,
    full_res_channels: 128,
};

#[cfg(test)]
mod tiling_tests {
    use super::*;

    #[test]
    fn lens_decode_candidate_physically_splits_the_1024_tail_and_edge_is_load_bearing() {
        // Lens packs 2x2, so a 1024 image enters the Flux2 decoder as a 128x128 raw latent.
        let bounded =
            mlx_gen::tiling::TilingConfig::spatial_only(512, 128).plan(FLUX2_TILING, 1, 128, 128);
        assert!(bounded.h.len() > 1 && bounded.w.len() > 1);
        assert_eq!((bounded.out_h, bounded.out_w), (1024, 1024));

        // Mutation control: doubling the edge collapses the same geometry to one tile. This proves
        // the production parameter reaches the physical plan instead of being metadata-only.
        let mutated =
            mlx_gen::tiling::TilingConfig::spatial_only(1024, 128).plan(FLUX2_TILING, 1, 128, 128);
        assert_eq!((mutated.h.len(), mutated.w.len()), (1, 1));
    }
}

/// `[O, I, H, W]` (PyTorch) → `[O, H, W, I]` (mlx conv2d), cast to f32.
fn conv_w(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?
        .transpose_axes(&[0, 2, 3, 1])?
        .as_dtype(Dtype::Float32)?)
}

fn f32w(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.as_dtype(Dtype::Float32)?)
}

/// A 1×1 conv expressed as a channel-wise Linear `[O, I]` (+ bias), f32.
fn squeeze_linear(w: &Weights, name: &str) -> Result<(Array, Array)> {
    let cw = w.require(&format!("{name}.weight"))?;
    let sh = cw.shape();
    Ok((
        cw.reshape(&[sh[0], sh[1]])?.as_dtype(Dtype::Float32)?,
        f32w(w, &format!("{name}.bias"))?,
    ))
}

/// VAE resnet block (temb-free): `silu(gn1(x)) → conv1 → silu(gn2) → conv2 + shortcut`.
struct ResnetBlock2D {
    norm1_w: Array,
    norm1_b: Array,
    conv1_w: Array,
    conv1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    conv2_w: Array,
    conv2_b: Array,
    shortcut: Option<(Array, Array)>,
}

impl ResnetBlock2D {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let shortcut = if w.get(&format!("{prefix}.conv_shortcut.weight")).is_some() {
            Some((
                conv_w(w, &format!("{prefix}.conv_shortcut.weight"))?,
                f32w(w, &format!("{prefix}.conv_shortcut.bias"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            norm1_w: f32w(w, &format!("{prefix}.norm1.weight"))?,
            norm1_b: f32w(w, &format!("{prefix}.norm1.bias"))?,
            conv1_w: conv_w(w, &format!("{prefix}.conv1.weight"))?,
            conv1_b: f32w(w, &format!("{prefix}.conv1.bias"))?,
            norm2_w: f32w(w, &format!("{prefix}.norm2.weight"))?,
            norm2_b: f32w(w, &format!("{prefix}.norm2.bias"))?,
            conv2_w: conv_w(w, &format!("{prefix}.conv2.weight"))?,
            conv2_b: f32w(w, &format!("{prefix}.conv2.bias"))?,
            shortcut,
        })
    }

    /// `x`: NHWC.
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = group_norm(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let h = conv2d(&silu(&h)?, &self.conv1_w, Some(&self.conv1_b), 1, 1)?;
        let h = group_norm(&h, &self.norm2_w, &self.norm2_b, GN_GROUPS, GN_EPS)?;
        let h = conv2d(&silu(&h)?, &self.conv2_w, Some(&self.conv2_b), 1, 1)?;
        let res = match &self.shortcut {
            Some((cw, cb)) => conv2d(x, cw, Some(cb), 1, 0)?,
            None => x.clone(),
        };
        Ok(add(&h, &res)?)
    }

    /// Normalization-correct bounded forward (sc-19753).
    ///
    /// Both GroupNorms reduce the **whole** layer activation; only the two 3×3 convolutions are
    /// evaluated on halo-expanded crops. The 1×1 shortcut has no spatial extent, so it runs whole.
    fn forward_tiled_vae(
        &self,
        x: &Array,
        tile_edge: i32,
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(mlx_gen::CancelFlag::is_cancelled) {
            return Err(mlx_gen::Error::Canceled);
        }
        let norm1 = GlobalGroupNorm::new(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let h = tiled_conv2d_3x3_nhwc(
            x,
            &self.conv1_w,
            Some(&self.conv1_b),
            tile_edge,
            cancel,
            |tile| silu(&norm1.apply(tile)?),
        )?;
        if cancel.is_some_and(mlx_gen::CancelFlag::is_cancelled) {
            return Err(mlx_gen::Error::Canceled);
        }
        let norm2 = GlobalGroupNorm::new(&h, &self.norm2_w, &self.norm2_b, GN_GROUPS, GN_EPS)?;
        let h = tiled_conv2d_3x3_nhwc(
            &h,
            &self.conv2_w,
            Some(&self.conv2_b),
            tile_edge,
            cancel,
            |tile| silu(&norm2.apply(tile)?),
        )?;
        let res = match &self.shortcut {
            Some((cw, cb)) => conv2d(x, cw, Some(cb), 1, 0)?,
            None => x.clone(),
        };
        let out = add(&h, &res)?;
        out.eval()?;
        Ok(out)
    }
}

/// Single-head spatial self-attention used in the mid block (the fork's `Flux2AttentionBlock`).
/// q/k/v/out are `nn.Linear` (with bias) — the only VAE modules the fork's `nn.quantize` hits, so
/// they are core [`AdaptableLinear`]s; the GroupNorm stays full precision (as do all the convs).
struct VaeAttention {
    gn_w: Array,
    gn_b: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
}

impl VaeAttention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        // q/k/v/out carry bias; weights are loaded f32 (the VAE runs f32). `quantize` casts to bf16
        // before packing so the scales byte-match the fork's bf16 `nn.quantize` (sc-2604 chokepoint).
        let lin = |n: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                f32w(w, &format!("{prefix}.{n}.weight"))?,
                Some(f32w(w, &format!("{prefix}.{n}.bias"))?),
            ))
        };
        Ok(Self {
            gn_w: f32w(w, &format!("{prefix}.group_norm.weight"))?,
            gn_b: f32w(w, &format!("{prefix}.group_norm.bias"))?,
            q: lin("to_q")?,
            k: lin("to_k")?,
            v: lin("to_v")?,
            o: lin("to_out.0")?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q.quantize(bits, None)?;
        self.k.quantize(bits, None)?;
        self.v.quantize(bits, None)?;
        self.o.quantize(bits, None)?;
        Ok(())
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
        let o = self.o.forward(&o.reshape(&[b, h, w_, c])?)?;
        Ok(add(x, &o)?)
    }
}

/// A run of resnets, then an optional downsample (asymmetric-pad + stride-2 conv) or upsample
/// (nearest-2× + conv). Port of `Flux2{Down,Up}EncoderBlock2D`.
struct SampleBlock {
    resnets: Vec<ResnetBlock2D>,
    downsample: Option<(Array, Array)>,
    upsample: Option<(Array, Array)>,
}

impl SampleBlock {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_resnets: i32,
        down: bool,
        up: bool,
    ) -> Result<Self> {
        let resnets = (0..num_resnets)
            .map(|j| ResnetBlock2D::from_weights(w, &format!("{prefix}.resnets.{j}")))
            .collect::<Result<Vec<_>>>()?;
        let conv = |which: &str| -> Result<(Array, Array)> {
            Ok((
                conv_w(w, &format!("{prefix}.{which}.0.conv.weight"))?,
                f32w(w, &format!("{prefix}.{which}.0.conv.bias"))?,
            ))
        };
        Ok(Self {
            resnets,
            downsample: if down {
                Some(conv("downsamplers")?)
            } else {
                None
            },
            upsample: if up { Some(conv("upsamplers")?) } else { None },
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        if let Some((cw, cb)) = &self.downsample {
            // Fork pads (right, bottom) then stride-2, pad-0 conv.
            x = pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            x = conv2d(&x, cw, Some(cb), 2, 0)?;
        }
        if let Some((cw, cb)) = &self.upsample {
            x = conv2d(&upsample_nearest(&x, 2)?, cw, Some(cb), 1, 1)?;
        }
        Ok(x)
    }

    /// Decode-side bounded forward (sc-19753): every resnet keeps whole-activation GroupNorm
    /// statistics and only the 3×3 convolutions tile. `tile_edge` bounds the resnet convolutions at
    /// this block's input resolution; `upsampled_tile_edge` bounds the post-upsample convolution,
    /// which runs at twice that resolution.
    fn forward_tiled_decode(
        &self,
        x: &Array,
        tile_edge: i32,
        upsampled_tile_edge: i32,
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        if self.downsample.is_some() {
            return Err(mlx_gen::Error::Msg(
                "FLUX.2 tiled decode cannot run an encoder/downsample block".into(),
            ));
        }
        let mut x = x.clone();
        for resnet in &self.resnets {
            x = resnet.forward_tiled_vae(&x, tile_edge, cancel)?;
        }
        if let Some((cw, cb)) = &self.upsample {
            let up = upsample_nearest(&x, 2)?;
            x = tiled_conv2d_3x3_nhwc(&up, cw, Some(cb), upsampled_tile_edge, cancel, |tile| {
                Ok(tile.clone())
            })?;
        }
        Ok(x)
    }

    fn upsamples(&self) -> bool {
        self.upsample.is_some()
    }
}

struct Encoder {
    conv_in_w: Array,
    conv_in_b: Array,
    down_blocks: Vec<SampleBlock>,
    mid_resnet0: ResnetBlock2D,
    mid_attn: VaeAttention,
    mid_resnet1: ResnetBlock2D,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out_w: Array,
    conv_out_b: Array,
}

impl Encoder {
    fn from_weights(w: &Weights) -> Result<Self> {
        let n = BLOCK_OUT.len();
        let down_blocks = (0..n)
            .map(|i| {
                SampleBlock::from_weights(
                    w,
                    &format!("encoder.down_blocks.{i}"),
                    LAYERS_PER_BLOCK,
                    i < n - 1,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in_w: conv_w(w, "encoder.conv_in.weight")?,
            conv_in_b: f32w(w, "encoder.conv_in.bias")?,
            down_blocks,
            mid_resnet0: ResnetBlock2D::from_weights(w, "encoder.mid_block.resnets.0")?,
            mid_attn: VaeAttention::from_weights(w, "encoder.mid_block.attentions.0")?,
            mid_resnet1: ResnetBlock2D::from_weights(w, "encoder.mid_block.resnets.1")?,
            norm_out_w: f32w(w, "encoder.conv_norm_out.weight")?,
            norm_out_b: f32w(w, "encoder.conv_norm_out.bias")?,
            conv_out_w: conv_w(w, "encoder.conv_out.weight")?,
            conv_out_b: f32w(w, "encoder.conv_out.bias")?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = conv2d(x, &self.conv_in_w, Some(&self.conv_in_b), 1, 1)?;
        for db in &self.down_blocks {
            x = db.forward(&x)?;
        }
        x = self.mid_resnet0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_resnet1.forward(&x)?;
        let x = group_norm(&x, &self.norm_out_w, &self.norm_out_b, GN_GROUPS, GN_EPS)?;
        conv2d(&silu(&x)?, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)
    }
}

struct Decoder {
    conv_in_w: Array,
    conv_in_b: Array,
    mid_resnet0: ResnetBlock2D,
    mid_attn: VaeAttention,
    mid_resnet1: ResnetBlock2D,
    up_blocks: Vec<SampleBlock>,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out_w: Array,
    conv_out_b: Array,
}

impl Decoder {
    fn from_weights(w: &Weights) -> Result<Self> {
        let n = BLOCK_OUT.len();
        // decoder resnets = layers_per_block + 1.
        let up_blocks = (0..n)
            .map(|i| {
                SampleBlock::from_weights(
                    w,
                    &format!("decoder.up_blocks.{i}"),
                    LAYERS_PER_BLOCK + 1,
                    false,
                    i < n - 1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in_w: conv_w(w, "decoder.conv_in.weight")?,
            conv_in_b: f32w(w, "decoder.conv_in.bias")?,
            mid_resnet0: ResnetBlock2D::from_weights(w, "decoder.mid_block.resnets.0")?,
            mid_attn: VaeAttention::from_weights(w, "decoder.mid_block.attentions.0")?,
            mid_resnet1: ResnetBlock2D::from_weights(w, "decoder.mid_block.resnets.1")?,
            up_blocks,
            norm_out_w: f32w(w, "decoder.conv_norm_out.weight")?,
            norm_out_b: f32w(w, "decoder.conv_norm_out.bias")?,
            conv_out_w: conv_w(w, "decoder.conv_out.weight")?,
            conv_out_b: f32w(w, "decoder.conv_out.bias")?,
        })
    }

    fn forward(&self, z: &Array) -> Result<Array> {
        self.forward_upsample_tail(&self.forward_pre_upsample(z)?)
    }

    /// Global-attention head, run once before a bounded decode splits the local upsample tail.
    fn forward_pre_upsample(&self, z: &Array) -> Result<Array> {
        let mut x = conv2d(z, &self.conv_in_w, Some(&self.conv_in_b), 1, 1)?;
        x = self.mid_resnet0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        self.mid_resnet1.forward(&x)
    }

    /// Spatially local, memory-spiking upsample tail.
    ///
    /// Every GroupNorm here — the two per `up_blocks` resnet and the final `conv_norm_out` — reduces
    /// the whole image. Tiling this tail as a unit would give each crop its own statistics, so the
    /// bounded route is [`Self::forward_upsample_tail_tiled`], which tiles one convolution at a time.
    fn forward_upsample_tail(&self, head: &Array) -> Result<Array> {
        let mut x = head.clone();
        for ub in &self.up_blocks {
            x = ub.forward(&x)?;
        }
        let x = group_norm(&x, &self.norm_out_w, &self.norm_out_b, GN_GROUPS, GN_EPS)?;
        conv2d(&silu(&x)?, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)
    }

    /// Layer-wise bounded upsample tail with dense-image GroupNorm semantics (sc-19753).
    ///
    /// `output_tile_edge` is expressed in **output** pixels; each stage's convolution edge is that
    /// bound divided by the upsampling still ahead of it, so the physical crop stays comparable at
    /// every resolution instead of collapsing to a single tile at latent scale.
    fn forward_upsample_tail_tiled(
        &self,
        head: &Array,
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
        let mut x = head.clone();
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
                "FLUX.2 tiled VAE tail ended at remaining spatial scale {remaining_scale}, expected 1"
            )));
        }
        if cancel.is_some_and(mlx_gen::CancelFlag::is_cancelled) {
            return Err(mlx_gen::Error::Canceled);
        }
        let norm = GlobalGroupNorm::new(&x, &self.norm_out_w, &self.norm_out_b, GN_GROUPS, GN_EPS)?;
        tiled_conv2d_3x3_nhwc(
            &x,
            &self.conv_out_w,
            Some(&self.conv_out_b),
            output_tile_edge.max(3),
            cancel,
            |tile| silu(&norm.apply(tile)?),
        )
    }
}

/// The FLUX.2 autoencoder. All tensors NHWC, f32.
pub struct Flux2Vae {
    encoder: Encoder,
    decoder: Decoder,
    quant: (Array, Array),
    post_quant: (Array, Array),
    bn_mean: Array,
    bn_std: Array,
}

impl Flux2Vae {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let bn_mean = f32w(w, "bn.running_mean")?;
        let bn_var = f32w(w, "bn.running_var")?;
        let bn_std = sqrt(&add(&bn_var, scalar(BN_EPS))?)?;
        Ok(Self {
            encoder: Encoder::from_weights(w)?,
            decoder: Decoder::from_weights(w)?,
            quant: squeeze_linear(w, "quant_conv")?,
            post_quant: squeeze_linear(w, "post_quant_conv")?,
            bn_mean,
            bn_std,
        })
    }

    /// Quantize the VAE to Q4/Q8 (group_size 64). The fork's `nn.quantize` predicate only hits
    /// `nn.Linear`, which in this VAE is exactly the encoder + decoder mid-block attention
    /// (q/k/v/out). Every Conv2d (incl. `quant_conv`/`post_quant_conv`), GroupNorm, and the
    /// BatchNorm stats are not Linears, so they stay full precision — matching the fork.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.encoder.mid_attn.quantize(bits)?;
        self.decoder.mid_attn.quantize(bits)?;
        Ok(())
    }

    /// Decode latents `[B, h, w, 32]` (NHWC) → image `[B, H, W, 3]` in ~`[-1, 1]`.
    /// `scaling_factor=1.0, shift_factor=0.0`, so the latent passes straight to `post_quant_conv`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let latents = latents.as_dtype(Dtype::Float32)?;
        let z = linear(&latents, &self.post_quant.0, &self.post_quant.1)?;
        self.decoder.forward(&z)
    }

    /// Test-only (sc-2643 byte-parity gate): the quantized `(wq, scales, biases, group_size, bits)`
    /// of the encoder mid-block attention `to_q` — the unique f32-loaded Linear-with-bias case
    /// (the rest of the VAE is Conv/GroupNorm, never quantized). `None` if the VAE is still dense.
    #[doc(hidden)]
    pub fn probe_quant_enc_q(&self) -> Option<(&Array, &Array, &Array, i32, i32)> {
        let (wq, sc, bi, _bias, gs, b) = self.encoder.mid_attn.q.quantized_params()?;
        Some((wq, sc, bi, gs, b))
    }

    /// Decode the transformer's packed output `[B, lat_h, lat_w, 128]` (NHWC): de-normalize with
    /// the BatchNorm stats, 2×2-unpatchify into `[B, lat_h·2, lat_w·2, 32]`, then `decode`.
    pub fn decode_packed_latents(&self, packed: &Array) -> Result<Array> {
        let latents = self.unpack_flux_packed_latents(packed)?;
        self.decode(&latents)
    }

    /// Bounded packed-latent decode. BatchNorm de-normalization, unpatchify, post-quant projection,
    /// and the decoder's global-attention head run once on the full latent; only the spatially-local
    /// ×8 upsample tail is bounded.
    ///
    /// **Normalization semantics (sc-19753).** The tail is bounded *layer-wise*: each 3×3
    /// convolution is evaluated on halo-expanded crops assembled from non-overlapping output cores,
    /// while every GroupNorm still reduces the whole layer activation. The previous whole-tail
    /// [`tiled_decode`](mlx_gen::vae_tiling::tiled_decode) route gave each crop its own GroupNorm
    /// statistics — a different decode, not a blend artifact. `cfg.spatial.tile_px` bounds each
    /// convolution crop in output pixels; the configured overlap remains part of the public tiling
    /// contract and policy identity, but halo/core arithmetic needs no blend of whole-tail outputs.
    pub fn decode_packed_latents_tiled(
        &self,
        packed: &Array,
        cfg: &mlx_gen::tiling::TilingConfig,
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(mlx_gen::CancelFlag::is_cancelled) {
            return Err(mlx_gen::Error::Canceled);
        }
        let latents = self
            .unpack_flux_packed_latents(packed)?
            .as_dtype(Dtype::Float32)?;
        let shape = latents.shape();
        let (h, w) = (shape[1], shape[2]);
        if !cfg.needs_tiling(FLUX2_TILING, 1, h, w) {
            return self.decode(&latents);
        }
        let tile_edge = cfg
            .spatial
            .as_ref()
            .ok_or_else(|| {
                mlx_gen::Error::Msg("FLUX.2 tiled decode requires spatial tiling".into())
            })?
            .tile_px;
        let z = linear(&latents, &self.post_quant.0, &self.post_quant.1)?;
        let head = self.decoder.forward_pre_upsample(&z)?;
        self.decoder
            .forward_upsample_tail_tiled(&head, tile_edge, cancel)
    }

    /// De-normalize and unpatchify FLUX/Lens packed-grid latents into the VAE's true raw 32-channel
    /// latent space. Input is NHWC `[B, h, w, 128]`, with channel order `c·4 + ph·2 + pw`; output
    /// is NHWC `[B, h·2, w·2, 32]`. This is the narrow shared seam for decorative latent previews
    /// and native VAE decode; callers must not project the intermediate packed 128-channel tensor.
    pub fn unpack_flux_packed_latents(&self, packed: &Array) -> Result<Array> {
        let shape = packed.shape();
        if shape.len() != 4 || shape[3] != 128 {
            return Err(mlx_gen::Error::Msg(format!(
                "FLUX packed latent must have shape [B, h, w, 128], got {shape:?}"
            )));
        }
        let packed = packed.as_dtype(Dtype::Float32)?;
        // De-normalize: x·std + mean (bn channel order = the packed 128-ch order).
        let denorm = add(&multiply(&packed, &self.bn_std)?, &self.bn_mean)?;
        unpatchify(&denorm)
    }

    /// De-normalize and unpatchify Ideogram 4 packed-token latents into the same VAE's true raw
    /// 32-channel latent space. Input is `[B, grid_h·grid_w, 128]`, with Ideogram's distinct packed
    /// channel order `(ph, pw, c)`; output is NHWC `[B, grid_h·2, grid_w·2, 32]`.
    pub fn unpack_ideogram_packed_latents(
        &self,
        packed: &Array,
        grid_h: i32,
        grid_w: i32,
    ) -> Result<Array> {
        let shape = packed.shape();
        if shape.len() != 3 || shape[1] != grid_h.saturating_mul(grid_w) || shape[2] != 128 {
            return Err(mlx_gen::Error::Msg(format!(
                "Ideogram packed latent must have shape [B, {}, 128], got {shape:?}",
                grid_h.saturating_mul(grid_w)
            )));
        }
        let packed = packed.as_dtype(Dtype::Float32)?;
        let std = self.bn_std.reshape(&[1, 1, 128])?;
        let mean = self.bn_mean.reshape(&[1, 1, 128])?;
        let denorm = add(&multiply(&packed, &std)?, &mean)?;
        unpatchify_ideogram(&denorm, grid_h, grid_w)
    }

    /// The packed-space BatchNorm de-normalization stats `(bn_std, bn_mean)` (each `[128]`, in the
    /// packed-channel order). For engines that reuse this VAE but unpatchify the packed latent
    /// themselves (e.g. Ideogram 4, whose reference does `z * bn_std + bn_mean` before its own
    /// unpatchify) rather than going through [`Self::decode_packed_latents`].
    pub fn bn_stats(&self) -> (&Array, &Array) {
        (&self.bn_std, &self.bn_mean)
    }

    /// Encode an image `[B, H, W, 3]` (NHWC, ~`[-1, 1]`) → latent **mean** `[B, H/8, W/8, 32]`.
    /// Mirrors the fork's `encode` (returns the mean; `scaling_factor=1.0, shift_factor=0.0`).
    pub fn encode_mean(&self, x: &Array) -> Result<Array> {
        let x = x.as_dtype(Dtype::Float32)?;
        let moments = linear(&self.encoder.forward(&x)?, &self.quant.0, &self.quant.1)?;
        // split (mean, logvar) along channels; keep the mean.
        let c = moments.shape()[3];
        let half = c / 2;
        let idx = Array::from_slice(&(0..half).collect::<Vec<i32>>(), &[half]);
        Ok(moments.take_axis(&idx, 3)?)
    }

    /// Forward BatchNorm-stats normalization of a **NCHW** patchified `[B, 128, h, w]` latent (the
    /// inverse of `decode_packed_latents`' de-normalize): `(x - mean) / std`, the fork's
    /// `bn_normalize_vae_encoded_latents`. Used by edit / img2img to normalize the reference VAE
    /// latent into the transformer's packed space.
    pub fn bn_normalize_nchw(&self, patchified: &Array) -> Result<Array> {
        let c = self.bn_mean.shape()[0];
        let mean = self.bn_mean.reshape(&[1, c, 1, 1])?;
        let std = self.bn_std.reshape(&[1, c, 1, 1])?;
        let x = patchified.as_dtype(Dtype::Float32)?;
        Ok(mlx_rs::ops::divide(
            &mlx_rs::ops::subtract(&x, &mean)?,
            &std,
        )?)
    }
}

/// 2×2 unpatchify (NHWC): `[B, h, w, 128]` → `[B, h·2, w·2, 32]`. Channel order `c·4 + ph·2 + pw`
/// matches the fork's NCHW `reshape(B, C/4, 2, 2, H, W) → transpose → reshape`.
fn unpatchify(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
    let c4 = c / 4;
    Ok(x.reshape(&[b, h, w_, c4, 2, 2])?
        .transpose_axes(&[0, 1, 4, 2, 5, 3])?
        .reshape(&[b, h * 2, w_ * 2, c4])?)
}

/// Ideogram's patch-major 2x2 unpatch order: `(ph, pw, c)` rather than FLUX's `(c, ph, pw)`.
fn unpatchify_ideogram(x: &Array, grid_h: i32, grid_w: i32) -> Result<Array> {
    let shape = x.shape();
    let channels = shape[2] / 4;
    Ok(x.reshape(&[shape[0], grid_h, grid_w, 2, 2, channels])?
        .transpose_axes(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[shape[0], grid_h * 2, grid_w * 2, channels])?)
}

#[allow(dead_code)]
const _: i32 = LATENT_CHANNELS; // documented; channel counts come from the checkpoint shapes.

/// Weights-free synthetic FLUX.2 VAE, for tiled-decode regressions (sc-19753).
///
/// `#[doc(hidden)]` test instrumentation rather than a `#[cfg(test)]` module because **Lens shares
/// this exact decode path** ([`mlx_gen_lens::vae::decode_with_tiling`] hands its packed grid to
/// [`Flux2Vae::decode_packed_latents_tiled`]), and a downstream crate cannot reach another crate's
/// `cfg(test)` items. Sharing the builder is what lets the Lens-side proof drive the real seam
/// instead of restating a copy of it that could drift.
#[doc(hidden)]
pub mod tiling_fixture {
    use super::*;
    use std::collections::HashMap;

    /// One channel width everywhere, so a single fixture builder covers every block. It must be a
    /// multiple of [`GN_GROUPS`] and — because the 2×2 unpatchify fixes the raw latent width at
    /// `128 / 4` — equal to the VAE's latent channel count.
    const C: i32 = 32;

    fn values(shape: &[i32], phase: f32, scale: f32) -> Array {
        let count = shape.iter().product::<i32>();
        let data = (0..count)
            .map(|i| ((i as f32 + phase) * 0.071).sin() * scale)
            .collect::<Vec<_>>();
        Array::from_slice(&data, shape)
    }

    /// Insert a `[out, in, k, k]` PyTorch-layout convolution — the layout [`conv_w`] transposes.
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

    fn insert_norm(tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
        tensors.insert(
            format!("{prefix}.weight"),
            add(values(&[C], phase, 0.15), Array::from_f32(1.0)).unwrap(),
        );
        tensors.insert(format!("{prefix}.bias"), values(&[C], phase + 2.0, 0.05));
    }

    fn insert_resnet(tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
        insert_norm(tensors, &format!("{prefix}.norm1"), phase);
        insert_norm(tensors, &format!("{prefix}.norm2"), phase + 1.0);
        insert_conv(tensors, &format!("{prefix}.conv1"), C, C, 3, phase + 4.0);
        insert_conv(tensors, &format!("{prefix}.conv2"), C, C, 3, phase + 5.0);
    }

    fn insert_attention(tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
        insert_norm(tensors, &format!("{prefix}.group_norm"), phase);
        for (index, name) in ["to_q", "to_k", "to_v", "to_out.0"].iter().enumerate() {
            tensors.insert(
                format!("{prefix}.{name}.weight"),
                values(&[C, C], phase + index as f32 + 1.0, 0.02),
            );
            tensors.insert(
                format!("{prefix}.{name}.bias"),
                values(&[C], phase + index as f32 + 5.0, 0.005),
            );
        }
    }

    fn insert_mid(tensors: &mut HashMap<String, Array>, prefix: &str, phase: f32) {
        insert_resnet(tensors, &format!("{prefix}.resnets.0"), phase);
        insert_attention(tensors, &format!("{prefix}.attentions.0"), phase + 10.0);
        insert_resnet(tensors, &format!("{prefix}.resnets.1"), phase + 20.0);
    }

    /// A structurally faithful synthetic FLUX.2 VAE: the real block/resnet counts
    /// ([`BLOCK_OUT`], [`LAYERS_PER_BLOCK`]) at one narrow channel width.
    pub fn weights() -> Weights {
        let blocks = BLOCK_OUT.len();
        let mut tensors = HashMap::new();

        insert_conv(&mut tensors, "decoder.conv_in", C, C, 3, 1.0);
        insert_mid(&mut tensors, "decoder.mid_block", 10.0);
        for block in 0..blocks {
            for resnet in 0..(LAYERS_PER_BLOCK + 1) {
                insert_resnet(
                    &mut tensors,
                    &format!("decoder.up_blocks.{block}.resnets.{resnet}"),
                    40.0 + (block as i32 * 10 + resnet) as f32,
                );
            }
            if block < blocks - 1 {
                insert_conv(
                    &mut tensors,
                    &format!("decoder.up_blocks.{block}.upsamplers.0.conv"),
                    C,
                    C,
                    3,
                    70.0 + block as f32,
                );
            }
        }
        insert_norm(&mut tensors, "decoder.conv_norm_out", 72.0);
        insert_conv(&mut tensors, "decoder.conv_out", C, 3, 3, 74.0);

        insert_conv(&mut tensors, "encoder.conv_in", 3, C, 3, 80.0);
        for block in 0..blocks {
            for resnet in 0..LAYERS_PER_BLOCK {
                insert_resnet(
                    &mut tensors,
                    &format!("encoder.down_blocks.{block}.resnets.{resnet}"),
                    90.0 + (block as i32 * 10 + resnet) as f32,
                );
            }
            if block < blocks - 1 {
                insert_conv(
                    &mut tensors,
                    &format!("encoder.down_blocks.{block}.downsamplers.0.conv"),
                    C,
                    C,
                    3,
                    112.0 + block as f32,
                );
            }
        }
        insert_mid(&mut tensors, "encoder.mid_block", 120.0);
        insert_norm(&mut tensors, "encoder.conv_norm_out", 145.0);
        insert_conv(&mut tensors, "encoder.conv_out", C, 2 * C, 3, 150.0);
        insert_conv(&mut tensors, "quant_conv", 2 * C, 2 * C, 1, 160.0);
        insert_conv(&mut tensors, "post_quant_conv", C, C, 1, 170.0);

        tensors.insert("bn.running_mean".into(), values(&[128], 180.0, 0.05));
        tensors.insert(
            "bn.running_var".into(),
            add(values(&[128], 190.0, 0.05), Array::from_f32(1.0)).unwrap(),
        );
        Weights::from_map(tensors)
    }

    /// Position-dependent packed latents `[1, grid_h, grid_w, 128]`: a per-crop GroupNorm sees
    /// visibly different statistics from the dense activation, so a tiled/dense comparison over
    /// these discriminates the defect rather than merely re-checking shapes.
    pub fn packed_latents(grid_h: i32, grid_w: i32) -> Array {
        let shape = [1, grid_h, grid_w, 128];
        let count = shape.iter().product::<i32>();
        let data = (0..count)
            .map(|i| {
                let y = (i / 128 / grid_w) as f32;
                let x = (i / 128 % grid_w) as f32;
                (i as f32 * 0.037).sin() + y * 0.11 - x * 0.07
            })
            .collect::<Vec<_>>();
        Array::from_slice(&data, &shape)
    }

    /// `max |left - right|` over two evaluated f32 arrays of equal length.
    pub fn max_abs_delta(left: &Array, right: &Array) -> f32 {
        left.as_slice::<f32>()
            .iter()
            .zip(right.as_slice::<f32>())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    }
}

/// sc-19753 — the layer-wise bounded decode, driven end-to-end across the public
/// [`Flux2Vae::decode_packed_latents_tiled`] seam on a synthetic checkpoint.
#[cfg(test)]
mod tiled_decode_tests {
    use super::tiling_fixture::{max_abs_delta, packed_latents, weights as fixture};
    use super::*;

    /// The **executed** defect control for the bound below (sc-19753).
    ///
    /// Runs the upsample tail on two halves of the same head activation and stitches them, which is
    /// exactly what the retired whole-tail `tiled_decode` route did: each crop's `GroupNorm`s reduce
    /// only their own crop. Asserting that this is observably different from the dense tail is what
    /// makes the 3e-3 tolerance below a real guard rather than one any implementation satisfies.
    ///
    /// Deliberately isolates the *tail*: the head runs once and is shared, so the divergence
    /// measured here is normalization alone, not the mid block's global attention.
    #[test]
    fn per_crop_tail_normalization_is_observably_wrong() {
        let vae = Flux2Vae::from_weights(&fixture()).unwrap();
        let packed = packed_latents(4, 5);
        let latents = vae
            .unpack_flux_packed_latents(&packed)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let z = linear(&latents, &vae.post_quant.0, &vae.post_quant.1).unwrap();
        let head = vae.decoder.forward_pre_upsample(&z).unwrap();

        let dense = vae.decoder.forward_upsample_tail(&head).unwrap();
        let width = head.shape()[2];
        let split = width / 2;
        let take = |start: i32, len: i32| {
            let idx = (start..start + len).collect::<Vec<i32>>();
            head.take_axis(Array::from_slice(&idx, &[len]), 2).unwrap()
        };
        let left = vae.decoder.forward_upsample_tail(&take(0, split)).unwrap();
        let right = vae
            .decoder
            .forward_upsample_tail(&take(split, width - split))
            .unwrap();
        let per_crop = mlx_rs::ops::concatenate_axis(&[&left, &right], 2).unwrap();

        dense.eval().unwrap();
        per_crop.eval().unwrap();
        assert_eq!(per_crop.shape(), dense.shape());
        let delta = max_abs_delta(&dense, &per_crop);
        assert!(
            delta > 1e-2,
            "per-crop tail normalization must be observably different from the dense tail, else the \
             bounded-decode tolerance proves nothing: max|delta|={delta:.3e}"
        );
    }

    /// The layer-wise bounded tail must track the dense decode. The tolerance is meaningful because
    /// `per_crop_tail_normalization_is_observably_wrong` executes the defect on this same fixture
    /// and lands orders of magnitude outside this bound.
    #[test]
    fn tiled_packed_decode_tracks_dense_with_global_group_norm() {
        let vae = Flux2Vae::from_weights(&fixture()).unwrap();
        let packed = packed_latents(4, 5);
        let dense = vae.decode_packed_latents(&packed).unwrap();
        let tiled = vae
            .decode_packed_latents_tiled(
                &packed,
                &mlx_gen::tiling::TilingConfig::spatial_only(16, 0),
                None,
            )
            .unwrap();
        dense.eval().unwrap();
        tiled.eval().unwrap();
        assert_eq!(tiled.shape(), dense.shape());
        let delta = max_abs_delta(&dense, &tiled);
        assert!(
            delta < 3e-3,
            "layer-wise tiled FLUX.2 VAE diverged from dense decode: max|delta|={delta:.3e}"
        );
    }

    /// A tiling request that does not actually split this latent must fall through to the exact
    /// single-pass decode rather than assembling a one-tile plan.
    #[test]
    fn untiled_request_falls_through_to_the_dense_decode() {
        let vae = Flux2Vae::from_weights(&fixture()).unwrap();
        let packed = packed_latents(4, 5);
        let dense = vae.decode_packed_latents(&packed).unwrap();
        let passthrough = vae
            .decode_packed_latents_tiled(
                &packed,
                &mlx_gen::tiling::TilingConfig::spatial_only(4096, 128),
                None,
            )
            .unwrap();
        dense.eval().unwrap();
        passthrough.eval().unwrap();
        assert_eq!(max_abs_delta(&dense, &passthrough), 0.0);
    }

    #[test]
    fn tiled_decode_honors_a_pretripped_cancel() {
        let vae = Flux2Vae::from_weights(&fixture()).unwrap();
        let cancel = mlx_gen::CancelFlag::new();
        cancel.cancel();
        let packed = packed_latents(4, 5);
        for tiling in [
            mlx_gen::tiling::TilingConfig::spatial_only(16, 0),
            mlx_gen::tiling::TilingConfig::spatial_only(4096, 128),
        ] {
            let result = vae.decode_packed_latents_tiled(&packed, &tiling, Some(&cancel));
            assert!(matches!(result, Err(mlx_gen::Error::Canceled)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpatchify_round_trips_patchify_ordering() {
        // Build [1,2,2,8] where channel = c*4 + ph*2 + pw, c in 0..2.
        let mut data = vec![0f32; 2 * 2 * 8];
        for hi in 0..2 {
            for wi in 0..2 {
                for ch in 0..8 {
                    data[((hi * 2 + wi) * 8) + ch] = (hi * 1000 + wi * 100 + ch) as f32;
                }
            }
        }
        let x = Array::from_slice(&data, &[1, 2, 2, 8]);
        let out = unpatchify(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 4, 2]);
        // out[b, 2*hi+ph, 2*wi+pw, c] == x[b, hi, wi, c*4+ph*2+pw]
        let o = out.as_slice::<f32>();
        let at = |hh: usize, ww: usize, cc: usize| o[((hh * 4 + ww) * 2) + cc];
        for hi in 0..2 {
            for wi in 0..2 {
                for ph in 0..2 {
                    for pw in 0..2 {
                        for c in 0..2 {
                            let got = at(2 * hi + ph, 2 * wi + pw, c);
                            let want = (hi * 1000 + wi * 100 + (c * 4 + ph * 2 + pw)) as f32;
                            assert_eq!(got, want, "mismatch at hi{hi} wi{wi} ph{ph} pw{pw} c{c}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ideogram_unpatchify_uses_patch_major_ordering() {
        // One 2x1 token grid, two raw channels: packed index = (ph*2 + pw)*2 + c.
        let data = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let packed = Array::from_slice(&data, &[1, 2, 8]);
        let out = unpatchify_ideogram(&packed, 2, 1).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2, 2]);
        let values = out.as_slice::<f32>();
        let at = |h: usize, w: usize, c: usize| values[((h * 2 + w) * 2) + c];
        for token_h in 0..2 {
            for ph in 0..2 {
                for pw in 0..2 {
                    for channel in 0..2 {
                        let got = at(token_h * 2 + ph, pw, channel);
                        let want = (token_h * 8 + (ph * 2 + pw) * 2 + channel) as f32;
                        assert_eq!(got, want);
                    }
                }
            }
        }
    }
}
