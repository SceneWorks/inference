//! The Mochi 1 AsymmVAE **decoder** (`AutoencoderKLMochi` decode path) ported to MLX.
//!
//! Decode-only and attention-free — both decoder mid-blocks are built with `add_attention=False`, so
//! the whole path is 3-D causal convs + per-frame chunked GroupNorm + a `Linear` depth-to-space
//! unpatchify. Tensors stay **NCTHW** (channels-first) throughout, mirroring the reference, and
//! transpose to channels-last only inside the conv / linear / group-norm ops (mlx convs are NDHWC).
//!
//! Structure (`MochiDecoder3D`):
//!  - `conv_in`: non-causal `1×1×1` Conv3d (`latent_channels` → `block_out_channels[-1]`);
//!  - `block_in`: `MochiMidBlock3D` (`layers_per_block[-1]` resnets, no attention);
//!  - `up_blocks[i]`: `MochiUpBlock3D` = resnets + `proj: Linear(C → C_out·t·s²)` + a reshape/permute
//!    depth-to-space unpatchify that expands `T·t, H·s, W·s`;
//!  - `block_out`: `MochiMidBlock3D` (`layers_per_block[0]` resnets, no attention);
//!  - `silu` → `proj_out: Linear(block_out_channels[0] → out_channels)`;
//!  - `drop_last_temporal_frames`: drop the leading `temporal_ratio − 1` (= 5) frames.
//!
//! `MochiResnetBlock3D` = `GroupNorm → silu → CausalConv3d(replicate) → GroupNorm → silu →
//! CausalConv3d → + residual`. `MochiChunkedGroupNorm3D` is a **per-frame** `GroupNorm(32)` (the 8-frame
//! chunking in the reference is a memory optimization — GroupNorm is per-sample independent, so the
//! result is identical without it). `CogVideoXCausalConv3d(pad_mode="replicate")` edge-pads the input:
//! spatial `(k−1)/2` symmetric, temporal `(k−1)` on the **front only** (causal).

use std::path::Path;

use mlx_rs::ops::{add, pad, PadMode};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{group_norm, linear, silu};
use mlx_gen::weights::{join, Weights};
use mlx_gen::{Error, Result};

use crate::config::MochiVaeConfig;

/// `nn.GroupNorm` default epsilon (`torch.nn.GroupNorm(eps=1e-5)`).
const GROUP_NORM_EPS: f32 = 1e-5;
/// `MochiChunkedGroupNorm3D` group count.
const NUM_GROUPS: i32 = 32;

/// A 3-D causal conv (`CogVideoXCausalConv3d`, `pad_mode="replicate"`). NCTHW I/O; the stored weight is
/// already transposed to the mlx `[out, kt, kh, kw, in]` layout at load. Spatial padding is symmetric
/// `(k−1)/2`; temporal padding is `(kt−1)` on the front only (causal). Both use **edge/replicate**
/// padding. A `1×1×1` kernel (the decoder `conv_in`) degenerates to zero padding — a plain conv.
struct CausalConv3d {
    w: Array,
    b: Array,
    kt: i32,
    kh: i32,
    kw: i32,
}

impl CausalConv3d {
    /// Load `{prefix}.conv.weight` (torch `[O, I, kt, kh, kw]`) + `{prefix}.conv.bias`, transposing the
    /// weight to the mlx `[O, kt, kh, kw, I]` layout and casting to `dtype`.
    fn from_weights(w: &Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        let weight = w.require(&join(prefix, "conv.weight"))?;
        let sh = weight.shape(); // [O, I, kt, kh, kw]
        let (kt, kh, kw) = (sh[2], sh[3], sh[4]);
        // torch [O, I, kt, kh, kw] -> mlx [O, kt, kh, kw, I].
        let w_mlx = weight.transpose_axes(&[0, 2, 3, 4, 1])?.as_dtype(dtype)?;
        Ok(Self {
            w: w_mlx,
            b: w.require(&join(prefix, "conv.bias"))?.as_dtype(dtype)?,
            kt,
            kh,
            kw,
        })
    }

    fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let time_pad = self.kt - 1;
        let h_pad = (self.kh - 1) / 2;
        let w_pad = (self.kw - 1) / 2;
        let x = if time_pad > 0 || h_pad > 0 || w_pad > 0 {
            // Replicate (edge) pad: T front-only (causal), H/W symmetric. NCTHW axes: 2=T, 3=H, 4=W.
            pad(
                x_ncthw,
                &[
                    (0, 0),
                    (0, 0),
                    (time_pad, 0),
                    (h_pad, h_pad),
                    (w_pad, w_pad),
                ][..],
                None,
                Some(PadMode::Edge),
            )?
        } else {
            x_ncthw.clone()
        };
        // NCTHW -> NDHWC, conv (valid), back to NCTHW.
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?;
        let y = mlx_gen::nn::conv3d(&x, &self.w, Some(&self.b), (1, 1, 1), (0, 0, 0))?;
        Ok(y.transpose_axes(&[0, 4, 1, 2, 3])?)
    }
}

/// `MochiChunkedGroupNorm3D` — per-frame `GroupNorm(32)` over a NCTHW tensor. Reshapes `[B,C,T,H,W]` to
/// per-frame `[B·T, H, W, C]`, applies the shared NHWC [`group_norm`], and restores NCTHW.
struct GroupNorm32 {
    weight: Array,
    bias: Array,
}

impl GroupNorm32 {
    fn from_weights(w: &Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            weight: w
                .require(&join(prefix, "norm_layer.weight"))?
                .as_dtype(dtype)?,
            bias: w
                .require(&join(prefix, "norm_layer.bias"))?
                .as_dtype(dtype)?,
        })
    }

    fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c, t, h, wd) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        // NCTHW -> [B,T,C,H,W] -> [B*T, C, H, W] -> NHWC [B*T, H, W, C].
        let x = x_ncthw
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[b * t, c, h, wd])?
            .transpose_axes(&[0, 2, 3, 1])?;
        let y = group_norm(&x, &self.weight, &self.bias, NUM_GROUPS, GROUP_NORM_EPS)?;
        // NHWC [B*T, H, W, C] -> [B*T, C, H, W] -> [B,T,C,H,W] -> NCTHW.
        Ok(y.transpose_axes(&[0, 3, 1, 2])?
            .reshape(&[b, t, c, h, wd])?
            .transpose_axes(&[0, 2, 1, 3, 4])?)
    }
}

/// `MochiResnetBlock3D`: `norm1 → silu → conv1 → norm2 → silu → conv2 → + input`. In the decoder every
/// resnet has `in_channels == out_channels`, so the residual add is direct (no shortcut conv).
struct ResnetBlock {
    norm1: GroupNorm32,
    conv1: CausalConv3d,
    norm2: GroupNorm32,
    conv2: CausalConv3d,
}

impl ResnetBlock {
    fn from_weights(w: &Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            norm1: GroupNorm32::from_weights(w, &join(prefix, "norm1"), dtype)?,
            conv1: CausalConv3d::from_weights(w, &join(prefix, "conv1"), dtype)?,
            norm2: GroupNorm32::from_weights(w, &join(prefix, "norm2"), dtype)?,
            conv2: CausalConv3d::from_weights(w, &join(prefix, "conv2"), dtype)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv1.forward(&silu(&self.norm1.forward(x)?)?)?;
        let h = self.conv2.forward(&silu(&self.norm2.forward(&h)?)?)?;
        Ok(add(&h, x)?)
    }
}

/// `MochiMidBlock3D` with `add_attention=False` — a run of resnets at a fixed channel width.
struct MidBlock {
    resnets: Vec<ResnetBlock>,
}

impl MidBlock {
    fn from_weights(w: &Weights, prefix: &str, num_layers: usize, dtype: Dtype) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            resnets.push(ResnetBlock::from_weights(
                w,
                &join(prefix, &format!("resnets.{i}")),
                dtype,
            )?);
        }
        Ok(Self { resnets })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        Ok(x)
    }
}

/// `MochiUpBlock3D`: resnets (at the input channel width) → channel-last `proj: Linear(C →
/// C_out·t·s²)` → depth-to-space unpatchify expanding `(T·t, H·s, W·s)`.
struct UpBlock {
    resnets: Vec<ResnetBlock>,
    proj_w: Array,
    proj_b: Array,
    out_ch: i32,
    t_exp: i32,
    s_exp: i32,
}

impl UpBlock {
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_layers: usize,
        out_ch: i32,
        t_exp: i32,
        s_exp: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            resnets.push(ResnetBlock::from_weights(
                w,
                &join(prefix, &format!("resnets.{i}")),
                dtype,
            )?);
        }
        Ok(Self {
            resnets,
            proj_w: w.require(&join(prefix, "proj.weight"))?.as_dtype(dtype)?,
            proj_b: w.require(&join(prefix, "proj.bias"))?.as_dtype(dtype)?,
            out_ch,
            t_exp,
            s_exp,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        // Channel-last proj: NCTHW -> [B,T,H,W,C] -> Linear -> [B,T,H,W,C_out·t·s²] -> NCTHW.
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?;
        let x = linear(&x, &self.proj_w, &self.proj_b)?;
        let x = x.transpose_axes(&[0, 4, 1, 2, 3])?;

        // Depth-to-space unpatchify (reference `MochiUpBlock3D.forward` tail):
        //   view [B, C_out, st, sh, sw, T, H, W]
        //   permute (0,1,5,2,6,3,7,4) -> [B, C_out, T, st, H, sh, W, sw]
        //   view [B, C_out, T·st, H·sh, W·sw]
        let sh = x.shape();
        let (b, t, h, wd) = (sh[0], sh[2], sh[3], sh[4]);
        let (st, ss) = (self.t_exp, self.s_exp);
        let x = x.reshape(&[b, self.out_ch, st, ss, ss, t, h, wd])?;
        let x = x.transpose_axes(&[0, 1, 5, 2, 6, 3, 7, 4])?;
        Ok(x.reshape(&[b, self.out_ch, t * st, h * ss, wd * ss])?)
    }
}

/// The Mochi 1 AsymmVAE decoder (decode-only). Holds the per-channel latent statistics for
/// de-normalization plus the ported `MochiDecoder3D`.
pub struct MochiVaeDecoder {
    conv_in: CausalConv3d,
    block_in: MidBlock,
    up_blocks: Vec<UpBlock>,
    block_out: MidBlock,
    proj_out_w: Array,
    proj_out_b: Array,
    /// `[1, C, 1, 1, 1]` per-channel latent mean (de-normalization).
    latents_mean: Array,
    /// `[1, C, 1, 1, 1]` per-channel latent std (de-normalization).
    latents_std: Array,
    scaling_factor: f32,
    temporal_ratio: i32,
    dtype: Dtype,
}

impl MochiVaeDecoder {
    /// Build the decoder from the diffusers `vae/` weight map at f32 compute precision (numerically
    /// clean; the reference decode ran bf16 — the parity residual reflects that bf16 rounding).
    pub fn from_weights(w: &Weights, cfg: &MochiVaeConfig) -> Result<Self> {
        Self::from_weights_dtype(w, cfg, Dtype::Float32)
    }

    /// As [`from_weights`](Self::from_weights) but choosing the compute dtype (bf16 to mirror the
    /// reference's shipped precision, f32 for a numerically clean decode).
    pub fn from_weights_dtype(w: &Weights, cfg: &MochiVaeConfig, dtype: Dtype) -> Result<Self> {
        let n_blocks = cfg.decoder_block_out_channels.len();
        let n_layers = cfg.layers_per_block.len();
        if n_blocks < 2 || n_layers < 2 || cfg.temporal_expansions.len() != n_blocks - 1 {
            return Err(Error::Msg(format!(
                "mochi vae: inconsistent config (blocks={n_blocks}, layers={n_layers}, \
                 temporal_expansions={})",
                cfg.temporal_expansions.len()
            )));
        }

        // conv_in is a plain nn.Conv3d (`decoder.conv_in.weight`, not the CogVideoX `.conv.weight`).
        let conv_in = load_plain_conv(w, "decoder.conv_in", dtype)?;

        // block_in: MochiMidBlock3D(block_out_channels[-1], layers_per_block[-1]).
        let block_in = MidBlock::from_weights(
            w,
            "decoder.block_in",
            cfg.layers_per_block[n_layers - 1],
            dtype,
        )?;

        // up_blocks[i]: in=block[-1-i], out=block[-2-i], layers=layers_per_block[-2-i],
        // t=temporal_expansions[-1-i], s=spatial_expansions[-1-i]  (reference decoder loop).
        let mut up_blocks = Vec::with_capacity(n_blocks - 1);
        let k = cfg.temporal_expansions.len();
        for i in 0..(n_blocks - 1) {
            let out_ch = cfg.decoder_block_out_channels[n_blocks - 2 - i] as i32;
            let num_layers = cfg.layers_per_block[n_layers - 2 - i];
            let t_exp = cfg.temporal_expansions[k - 1 - i] as i32;
            let s_exp = cfg.spatial_expansions[k - 1 - i] as i32;
            up_blocks.push(UpBlock::from_weights(
                w,
                &format!("decoder.up_blocks.{i}"),
                num_layers,
                out_ch,
                t_exp,
                s_exp,
                dtype,
            )?);
        }

        // block_out: MochiMidBlock3D(block_out_channels[0], layers_per_block[0]).
        let block_out =
            MidBlock::from_weights(w, "decoder.block_out", cfg.layers_per_block[0], dtype)?;

        let c = cfg.latent_channels as i32;
        let latents_mean =
            Array::from_slice(&cfg.latents_mean, &[1, c, 1, 1, 1]).as_dtype(dtype)?;
        let latents_std = Array::from_slice(&cfg.latents_std, &[1, c, 1, 1, 1]).as_dtype(dtype)?;

        Ok(Self {
            conv_in,
            block_in,
            up_blocks,
            block_out,
            proj_out_w: w.require("decoder.proj_out.weight")?.as_dtype(dtype)?,
            proj_out_b: w.require("decoder.proj_out.bias")?.as_dtype(dtype)?,
            latents_mean,
            latents_std,
            scaling_factor: cfg.scaling_factor,
            temporal_ratio: cfg.temporal_compression_ratio() as i32,
            dtype,
        })
    }

    /// De-normalize a raw latent (diffusers `MochiPipeline`): `z · std / scaling + mean`, per channel.
    pub fn denormalize(&self, latents: &Array) -> Result<Array> {
        let z = latents.as_dtype(self.dtype)?;
        let scaled = mlx_rs::ops::multiply(&z, &self.latents_std)?;
        let scaled = mlx_rs::ops::divide(&scaled, &scalar(self.scaling_factor, self.dtype)?)?;
        Ok(add(&scaled, &self.latents_mean)?)
    }

    /// Decode an **already-de-normalized** latent `[B, C, T_lat, H_lat, W_lat]` → video
    /// `[B, out_channels, F, H, W]` (`F = (T_lat − 1)·temporal_ratio + 1`, spatial ×8). Teacher-forced
    /// entry point (the vae_parity gate feeds the golden's `denormalized_latents`).
    pub fn decode_denormalized(&self, latents: &Array) -> Result<Array> {
        let x = latents.as_dtype(self.dtype)?;
        let mut x = self.conv_in.forward(&x)?;
        x = self.block_in.forward(&x)?;
        for up in &self.up_blocks {
            x = up.forward(&x)?;
        }
        x = self.block_out.forward(&x)?;
        x = silu(&x)?;
        // Channel-last proj_out: NCTHW -> [B,T,H,W,C] -> Linear(C->out) -> NCTHW.
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?;
        let x = linear(&x, &self.proj_out_w, &self.proj_out_b)?;
        let x = x.transpose_axes(&[0, 4, 1, 2, 3])?;
        self.drop_last_temporal_frames(&x)
    }

    /// De-normalize then decode a raw latent → video (the production entry point).
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let denorm = self.denormalize(latents)?;
        self.decode_denormalized(&denorm)
    }

    /// Drop the leading `temporal_ratio − 1` decoded frames (`drop_last_temporal_frames=True`), which
    /// realigns the causal decode to `(T_lat − 1)·ratio + 1` output frames. NCTHW temporal axis = 2.
    fn drop_last_temporal_frames(&self, x: &Array) -> Result<Array> {
        let f = x.shape()[2];
        if f >= self.temporal_ratio {
            let start = self.temporal_ratio - 1;
            let idx: Vec<i32> = (start..f).collect();
            let idx = Array::from_slice(&idx, &[idx.len() as i32]);
            Ok(x.take_axis(&idx, 2)?)
        } else {
            Ok(x.clone())
        }
    }
}

/// Load the Mochi AsymmVAE decoder from a snapshot root: reads `vae/config.json` for the decoder
/// structure + latent statistics and the `vae/` safetensors for the weights, at f32 compute precision.
pub fn load_vae_decoder(root: &Path) -> Result<MochiVaeDecoder> {
    let cfg = MochiVaeConfig::from_model_dir(root)?;
    let w = Weights::from_dir(root.join("vae"))?;
    MochiVaeDecoder::from_weights(&w, &cfg)
}

/// A scalar `Array` at `dtype` (small helper for the de-normalization divide).
fn scalar(v: f32, dtype: Dtype) -> Result<Array> {
    Ok(Array::from_f32(v).as_dtype(dtype)?)
}

/// Load a **plain** `nn.Conv3d` (`{prefix}.weight` torch `[O, I, kt, kh, kw]` + `{prefix}.bias`) as a
/// [`CausalConv3d`] — the decoder `conv_in`, whose `1×1×1` kernel makes the causal padding a no-op.
fn load_plain_conv(w: &Weights, prefix: &str, dtype: Dtype) -> Result<CausalConv3d> {
    let weight = w.require(&join(prefix, "weight"))?;
    let sh = weight.shape();
    let (kt, kh, kw) = (sh[2], sh[3], sh[4]);
    let w_mlx = weight.transpose_axes(&[0, 2, 3, 4, 1])?.as_dtype(dtype)?;
    Ok(CausalConv3d {
        w: w_mlx,
        b: w.require(&join(prefix, "bias"))?.as_dtype(dtype)?,
        kt,
        kh,
        kw,
    })
}
