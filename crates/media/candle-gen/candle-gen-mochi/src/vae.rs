//! The Mochi 1 AsymmVAE **decoder** (`AutoencoderKLMochi` decode path) ported to candle.
//!
//! Decode-only and attention-free — both decoder mid-blocks are built with `add_attention=False`, so
//! the whole path is 3-D causal convs ([`crate::conv3d`]) + per-frame chunked `GroupNorm(32)` + a
//! `Linear` depth-to-space unpatchify. Tensors stay **NCTHW** (channels-first) throughout; unlike the
//! MLX port there is no NHWC transpose dance — candle's `GroupNorm` and `conv2d` are channels-first.
//!
//! Structure (`MochiDecoder3D`):
//!  - `conv_in`: non-causal `1×1×1` conv (`latent_channels` → `block_out_channels[-1]`);
//!  - `block_in`: `MochiMidBlock3D` (`layers_per_block[-1]` resnets, no attention);
//!  - `up_blocks[i]`: `MochiUpBlock3D` = resnets + `proj: Linear(C → C_out·t·s²)` + a reshape/permute
//!    depth-to-space unpatchify that expands `T·t, H·s, W·s`;
//!  - `block_out`: `MochiMidBlock3D` (`layers_per_block[0]` resnets, no attention);
//!  - `silu` → `proj_out: Linear(block_out_channels[0] → out_channels)`;
//!  - `drop_last_temporal_frames`: drop the leading `temporal_ratio − 1` (= 5) frames.
//!
//! `MochiResnetBlock3D` = `GroupNorm → silu → CausalConv3d(replicate) → GroupNorm → silu →
//! CausalConv3d → + residual`. `MochiChunkedGroupNorm3D` is a **per-frame** `GroupNorm(32)` (the
//! reference's 8-frame chunking is a memory optimization — GroupNorm is per-sample independent, so the
//! result is identical). The decoder runs **f32** (bf16 intermediates reach O(100), outside bf16's range).

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{GroupNorm, Module, VarBuilder};
use candle_gen::{CandleError, Result};

use crate::config::MochiVaeConfig;
use crate::conv3d::{CausalConv3d, FrameCache};
use crate::nn::{linear_b, silu};

/// `torch.nn.GroupNorm` default epsilon.
const GROUP_NORM_EPS: f64 = 1e-5;
/// `MochiChunkedGroupNorm3D` group count.
const NUM_GROUPS: usize = 32;

/// Latent frames per chunk for [`MochiVaeDecoder::decode_chunked`].
///
/// The decode peak is dominated by `block_out`, which runs 128 channels at the **full** output
/// resolution, so the working set scales with the chunk while the peak stays ~flat in clip length.
/// 1 is the floor; raise it to trade memory back for fewer per-chunk boundaries.
///
/// Matches the MLX port's default, where the tradeoff is measured (848×480/151 frames: chunk=1 →
/// 24.70 GiB, chunk=2 → 37.69, chunk=4 → 65.62). Those figures are MLX's; candle's allocator and eager
/// execution differ, so treat them as the shape of the curve rather than as candle's own numbers.
pub const DEFAULT_DECODE_CHUNK_FRAMES: usize = 1;

/// `MochiChunkedGroupNorm3D` — per-frame `GroupNorm(32)` over a NCTHW tensor. Reshapes `[B,C,T,H,W]` to
/// per-frame `[B·T, C, H, W]`, applies the channels-first [`GroupNorm`], and restores NCTHW.
struct GroupNorm32 {
    gn: GroupNorm,
}

impl GroupNorm32 {
    /// Load the `norm_layer.weight`/`.bias` under `vb` (the channel count rides on the weight).
    fn load(vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get_unchecked("norm_layer.weight")?;
        let bias = vb.get_unchecked("norm_layer.bias")?;
        let c = weight.dim(0)?;
        let gn = GroupNorm::new(weight, bias, c, NUM_GROUPS, GROUP_NORM_EPS)?;
        Ok(Self { gn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        // NCTHW -> [B,T,C,H,W] -> [B·T, C, H, W] (channels-first for candle GroupNorm).
        let x = x
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * t, c, h, w))?
            .contiguous()?;
        let y = self.gn.forward(&x)?;
        // back to NCTHW.
        Ok(y.reshape((b, t, c, h, w))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?)
    }
}

/// `MochiResnetBlock3D`: `norm1 → silu → conv1 → norm2 → silu → conv2 → + input`. Every decoder resnet
/// has `in_channels == out_channels`, so the residual add is direct (no shortcut conv).
struct ResnetBlock {
    norm1: GroupNorm32,
    conv1: CausalConv3d,
    norm2: GroupNorm32,
    conv2: CausalConv3d,
}

impl ResnetBlock {
    fn load(vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: GroupNorm32::load(&vb.pp("norm1"))?,
            conv1: CausalConv3d::load(&vb.pp("conv1"), "conv")?,
            norm2: GroupNorm32::load(&vb.pp("norm2"))?,
            conv2: CausalConv3d::load(&vb.pp("conv2"), "conv")?,
        })
    }

    fn forward(&self, x: &Tensor, mut cache: Option<&mut FrameCache>) -> Result<Tensor> {
        let h = self
            .conv1
            .forward(&silu(&self.norm1.forward(x)?)?, cache.as_deref_mut())?;
        let h = self
            .conv2
            .forward(&silu(&self.norm2.forward(&h)?)?, cache)?;
        Ok((h + x)?)
    }
}

/// `MochiMidBlock3D` with `add_attention=False` — a run of resnets at a fixed channel width.
struct MidBlock {
    resnets: Vec<ResnetBlock>,
}

impl MidBlock {
    fn load(vb: &VarBuilder, num_layers: usize) -> Result<Self> {
        let rvb = vb.pp("resnets");
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            resnets.push(ResnetBlock::load(&rvb.pp(i))?);
        }
        Ok(Self { resnets })
    }

    fn forward(&self, x: &Tensor, mut cache: Option<&mut FrameCache>) -> Result<Tensor> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x, cache.as_deref_mut())?;
        }
        Ok(x)
    }
}

/// `MochiUpBlock3D`: resnets (at the input channel width) → channel-last `proj: Linear(C →
/// C_out·t·s²)` → depth-to-space unpatchify expanding `(T·t, H·s, W·s)`.
struct UpBlock {
    resnets: Vec<ResnetBlock>,
    proj_w: Tensor,
    proj_b: Tensor,
    out_ch: usize,
    t_exp: usize,
    s_exp: usize,
}

impl UpBlock {
    fn load(
        vb: &VarBuilder,
        num_layers: usize,
        out_ch: usize,
        t_exp: usize,
        s_exp: usize,
    ) -> Result<Self> {
        let rvb = vb.pp("resnets");
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            resnets.push(ResnetBlock::load(&rvb.pp(i))?);
        }
        Ok(Self {
            resnets,
            proj_w: vb.get_unchecked("proj.weight")?,
            proj_b: vb.get_unchecked("proj.bias")?,
            out_ch,
            t_exp,
            s_exp,
        })
    }

    fn forward(&self, x: &Tensor, mut cache: Option<&mut FrameCache>) -> Result<Tensor> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x, cache.as_deref_mut())?;
        }
        // Channel-last proj: NCTHW -> [B,T,H,W,C] -> Linear -> [B,T,H,W,C_out·t·s²] -> NCTHW.
        let x = x.permute((0, 2, 3, 4, 1))?.contiguous()?;
        let x = linear_b(&x, &self.proj_w, &self.proj_b)?;
        let x = x.permute((0, 4, 1, 2, 3))?.contiguous()?;

        // Depth-to-space unpatchify (reference `MochiUpBlock3D.forward` tail):
        //   view [B, C_out, st, ss, ss, T, H, W]
        //   permute (0,1,5,2,6,3,7,4) -> [B, C_out, T, st, H, ss, W, ss]
        //   view [B, C_out, T·st, H·ss, W·ss]
        let (b, _, t, h, w) = x.dims5()?;
        let (st, ss) = (self.t_exp, self.s_exp);
        // Rank-8 reshape/permute — candle's tuple `Dims`/`Shape` impls stop at arity 6, so use a Vec /
        // array here (`[usize; N]: Dims`, `Vec<usize>: Into<Shape>`).
        let x = x
            .reshape(vec![b, self.out_ch, st, ss, ss, t, h, w])?
            .permute([0usize, 1, 5, 2, 6, 3, 7, 4])?
            .contiguous()?
            .reshape((b, self.out_ch, t * st, h * ss, w * ss))?;
        Ok(x)
    }
}

/// The Mochi 1 AsymmVAE decoder (decode-only). Holds the per-channel latent statistics for
/// de-normalization plus the ported `MochiDecoder3D`.
pub struct MochiVaeDecoder {
    conv_in: CausalConv3d,
    block_in: MidBlock,
    up_blocks: Vec<UpBlock>,
    block_out: MidBlock,
    proj_out_w: Tensor,
    proj_out_b: Tensor,
    /// `[1, C, 1, 1, 1]` per-channel latent mean (de-normalization).
    latents_mean: Tensor,
    /// `[1, C, 1, 1, 1]` per-channel latent std (de-normalization).
    latents_std: Tensor,
    scaling_factor: f64,
    temporal_ratio: usize,
    dtype: DType,
}

impl MochiVaeDecoder {
    /// Build the decoder from `dec_vb` (a VarBuilder rooted at `decoder`, i.e. `conv_in.weight`,
    /// `block_in.resnets.0…`) + the config. `dtype` is the compute dtype (f32 for a clean decode).
    pub fn new(
        dec_vb: VarBuilder,
        cfg: &MochiVaeConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let n_blocks = cfg.decoder_block_out_channels.len();
        let n_layers = cfg.layers_per_block.len();
        if n_blocks < 2 || n_layers < 2 || cfg.temporal_expansions.len() != n_blocks - 1 {
            return Err(CandleError::Msg(format!(
                "mochi vae: inconsistent config (blocks={n_blocks}, layers={n_layers}, \
                 temporal_expansions={})",
                cfg.temporal_expansions.len()
            )));
        }

        // conv_in is a plain 1×1×1 conv (`decoder.conv_in.weight`, not a CogVideoX `.conv.weight`).
        let conv_in = CausalConv3d::load(&dec_vb, "conv_in")?;
        // block_in: MochiMidBlock3D(block_out_channels[-1], layers_per_block[-1]).
        let block_in = MidBlock::load(&dec_vb.pp("block_in"), cfg.layers_per_block[n_layers - 1])?;

        // up_blocks[i]: in=block[-1-i], out=block[-2-i], layers=layers_per_block[-2-i],
        // t=temporal_expansions[-1-i], s=spatial_expansions[-1-i] (reference decoder loop).
        let ub = dec_vb.pp("up_blocks");
        let k = cfg.temporal_expansions.len();
        let mut up_blocks = Vec::with_capacity(n_blocks - 1);
        for i in 0..(n_blocks - 1) {
            let out_ch = cfg.decoder_block_out_channels[n_blocks - 2 - i];
            let num_layers = cfg.layers_per_block[n_layers - 2 - i];
            let t_exp = cfg.temporal_expansions[k - 1 - i];
            let s_exp = cfg.spatial_expansions[k - 1 - i];
            up_blocks.push(UpBlock::load(&ub.pp(i), num_layers, out_ch, t_exp, s_exp)?);
        }

        // block_out: MochiMidBlock3D(block_out_channels[0], layers_per_block[0]).
        let block_out = MidBlock::load(&dec_vb.pp("block_out"), cfg.layers_per_block[0])?;

        let c = cfg.latent_channels;
        let latents_mean =
            Tensor::from_vec(cfg.latents_mean.clone(), (1, c, 1, 1, 1), device)?.to_dtype(dtype)?;
        let latents_std =
            Tensor::from_vec(cfg.latents_std.clone(), (1, c, 1, 1, 1), device)?.to_dtype(dtype)?;

        Ok(Self {
            conv_in,
            block_in,
            up_blocks,
            block_out,
            proj_out_w: dec_vb.get_unchecked("proj_out.weight")?,
            proj_out_b: dec_vb.get_unchecked("proj_out.bias")?,
            latents_mean,
            latents_std,
            scaling_factor: cfg.scaling_factor as f64,
            temporal_ratio: cfg.temporal_compression_ratio(),
            dtype,
        })
    }

    /// Load the AsymmVAE decoder from a snapshot root: `vae/config.json` + the `vae/` safetensors at
    /// f32 compute precision.
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        let cfg = MochiVaeConfig::from_model_dir(root)?;
        let vb = candle_gen::component_vb(root, "vae", DType::F32, device, "mochi vae")?;
        Self::new(vb.pp("decoder"), &cfg, DType::F32, device)
    }

    /// De-normalize a raw latent (diffusers `MochiPipeline`): `z · std / scaling + mean`, per channel.
    pub fn denormalize(&self, latents: &Tensor) -> Result<Tensor> {
        let z = latents.to_dtype(self.dtype)?;
        let scaled = z.broadcast_mul(&self.latents_std)?;
        let scaled = (scaled / self.scaling_factor)?;
        Ok(scaled.broadcast_add(&self.latents_mean)?)
    }

    /// The decode body, **without** the leading-frame drop: de-normalized latent → raw `6·T_lat`-frame
    /// video. `cache` is `None` for a single-shot decode and `Some` for one chunk of a chunked decode.
    fn decode_body(&self, latents: &Tensor, mut cache: Option<&mut FrameCache>) -> Result<Tensor> {
        let x = latents.to_dtype(self.dtype)?;
        let mut x = self.conv_in.forward(&x, cache.as_deref_mut())?;
        x = self.block_in.forward(&x, cache.as_deref_mut())?;
        for up in &self.up_blocks {
            x = up.forward(&x, cache.as_deref_mut())?;
        }
        x = self.block_out.forward(&x, cache)?;
        x = silu(&x)?;
        // Channel-last proj_out: NCTHW -> [B,T,H,W,C] -> Linear(C->out) -> NCTHW.
        let x = x.permute((0, 2, 3, 4, 1))?.contiguous()?;
        let x = linear_b(&x, &self.proj_out_w, &self.proj_out_b)?;
        Ok(x.permute((0, 4, 1, 2, 3))?.contiguous()?)
    }

    /// One [`FrameCache`] slot per causal conv, in decode-body traversal order.
    fn conv_count(&self) -> usize {
        1 + 2
            * (self.block_in.resnets.len()
                + self.block_out.resnets.len()
                + self
                    .up_blocks
                    .iter()
                    .map(|u| u.resnets.len())
                    .sum::<usize>())
    }

    /// Decode an **already-de-normalized** latent `[B, C, T_lat, H_lat, W_lat]` → video
    /// `[B, out_channels, F, H, W]` (`F = (T_lat − 1)·temporal_ratio + 1`, spatial ×8) in a single
    /// pass. Teacher-forced entry point (the `vae_parity` gate feeds the golden's
    /// `denormalized_latents`), and the reference [`decode_denormalized_chunked`] is gated against.
    ///
    /// Peak scales linearly in `T_lat`; prefer [`decode_chunked`](Self::decode_chunked) in production.
    pub fn decode_denormalized(&self, latents: &Tensor) -> Result<Tensor> {
        let x = self.decode_body(latents, None)?;
        self.drop_last_temporal_frames(&x)
    }

    /// As [`decode_denormalized`](Self::decode_denormalized), but decoding `chunk_frames` latent frames
    /// at a time and threading a [`FrameCache`] so each chunk sees the previous one's real trailing
    /// frames instead of a replicate pad. **Numerically identical** to the single-shot decode (every op
    /// is per-frame or causal), so there are no tile seams to blend — see [`FrameCache`].
    ///
    /// Peak memory becomes ~independent of clip length: a fixed conv cache plus a working set sized by
    /// `chunk_frames`, in place of a `block_out` tensor that grows with the whole clip.
    pub fn decode_denormalized_chunked(
        &self,
        latents: &Tensor,
        chunk_frames: usize,
    ) -> Result<Tensor> {
        let t_lat = latents.dim(2)?;
        let chunk = chunk_frames.max(1);
        if chunk >= t_lat {
            return self.decode_denormalized(latents);
        }

        let mut cache = FrameCache::new(self.conv_count());
        let mut chunks: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        while start < t_lat {
            let len = chunk.min(t_lat - start);
            cache.rewind();
            let v = self.decode_body(&latents.narrow(2, start, len)?, Some(&mut cache))?;
            // The full decode drops the leading `ratio − 1` frames of the whole video; chunk 0 owns
            // them (it always yields `ratio` ≥ 5 frames), and later chunks concatenate untouched.
            let v = if start == 0 {
                self.drop_last_temporal_frames(&v)?
            } else {
                v
            };
            chunks.push(v);
            start += len;
        }
        Ok(Tensor::cat(&chunks, 2)?)
    }

    /// De-normalize then decode a raw latent → video, in a single pass.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let denorm = self.denormalize(latents)?;
        self.decode_denormalized(&denorm)
    }

    /// De-normalize then chunk-decode a raw latent → video (the production entry point). See
    /// [`decode_denormalized_chunked`](Self::decode_denormalized_chunked); `chunk_frames` of
    /// [`DEFAULT_DECODE_CHUNK_FRAMES`] is the shipped default.
    pub fn decode_chunked(&self, latents: &Tensor, chunk_frames: usize) -> Result<Tensor> {
        // De-normalization is per-element on the *latent*, so doing it once up front costs a latent-
        // sized tensor (12 ch), not a decode-sized one.
        let denorm = self.denormalize(latents)?;
        self.decode_denormalized_chunked(&denorm, chunk_frames)
    }

    /// Drop the leading `temporal_ratio − 1` decoded frames (`drop_last_temporal_frames=True`), which
    /// realigns the causal decode to `(T_lat − 1)·ratio + 1` output frames. NCTHW temporal axis = 2.
    fn drop_last_temporal_frames(&self, x: &Tensor) -> Result<Tensor> {
        let f = x.dim(2)?;
        if f >= self.temporal_ratio {
            let start = self.temporal_ratio - 1;
            Ok(x.narrow(2, start, f - start)?)
        } else {
            Ok(x.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic small "random" fill (bounded so 5 GroupNorm+conv stages stay well-conditioned).
    fn rnd(shape: &[usize], seed: u64, device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                (((i as u64).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32 * 1e-6).sin()
                    * 0.05
            })
            .collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    /// A tiny, GroupNorm-valid decoder config: 32-wide stages, 1 resnet each, real expansions.
    fn tiny_cfg() -> MochiVaeConfig {
        MochiVaeConfig {
            latent_channels: 12,
            out_channels: 3,
            decoder_block_out_channels: vec![32, 32, 32, 32],
            layers_per_block: vec![1, 1, 1, 1, 1],
            temporal_expansions: vec![1, 2, 3],
            spatial_expansions: vec![2, 2, 2],
            latents_mean: vec![0.0; 12],
            latents_std: vec![1.0; 12],
            scaling_factor: 1.0,
        }
    }

    /// Insert a `MochiResnetBlock3D`'s weights (identity-ish norms, small random convs) at `pfx`.
    fn insert_resnet(
        w: &mut HashMap<String, Tensor>,
        pfx: &str,
        ch: usize,
        seed: u64,
        dev: &Device,
    ) {
        for norm in ["norm1", "norm2"] {
            w.insert(
                format!("{pfx}.{norm}.norm_layer.weight"),
                Tensor::ones(ch, DType::F32, dev).unwrap(),
            );
            w.insert(
                format!("{pfx}.{norm}.norm_layer.bias"),
                Tensor::zeros(ch, DType::F32, dev).unwrap(),
            );
        }
        for (j, conv) in ["conv1", "conv2"].iter().enumerate() {
            w.insert(
                format!("{pfx}.{conv}.conv.weight"),
                rnd(&[ch, ch, 3, 3, 3], seed + j as u64 * 7 + 1, dev),
            );
            w.insert(
                format!("{pfx}.{conv}.conv.bias"),
                Tensor::zeros(ch, DType::F32, dev).unwrap(),
            );
        }
    }

    /// The real AsymmVAE temporal geometry (`layers_per_block` `[3,3,4,6,3]` → 19 resnets → 38 `kt=3`
    /// causal convs) at 32 channels — what the chunked-decode equivalence gate needs, since the
    /// `FrameCache` has one slot per conv and the temporal coupling is what is under test.
    fn real_geometry_cfg() -> MochiVaeConfig {
        MochiVaeConfig {
            layers_per_block: vec![3, 3, 4, 6, 3],
            ..tiny_cfg()
        }
    }

    /// Build the synthetic decoder weight map for `cfg`, honoring its `layers_per_block`.
    fn synthetic_weights(cfg: &MochiVaeConfig, dev: &Device) -> HashMap<String, Tensor> {
        let mut w = HashMap::new();
        let c_last = *cfg.decoder_block_out_channels.last().unwrap();
        let c_first = cfg.decoder_block_out_channels[0];
        let lat = cfg.latent_channels;
        let n = cfg.decoder_block_out_channels.len();
        let nl = cfg.layers_per_block.len();
        let k = cfg.temporal_expansions.len();

        w.insert(
            "conv_in.weight".into(),
            rnd(&[c_last, lat, 1, 1, 1], 10, dev),
        );
        w.insert(
            "conv_in.bias".into(),
            Tensor::zeros(c_last, DType::F32, dev).unwrap(),
        );
        for r in 0..cfg.layers_per_block[nl - 1] {
            insert_resnet(
                &mut w,
                &format!("block_in.resnets.{r}"),
                c_last,
                100 + r as u64,
                dev,
            );
        }

        for i in 0..(n - 1) {
            let in_ch = cfg.decoder_block_out_channels[n - 1 - i];
            let out_ch = cfg.decoder_block_out_channels[n - 2 - i];
            let t = cfg.temporal_expansions[k - 1 - i];
            let s = cfg.spatial_expansions[k - 1 - i];
            let pfx = format!("up_blocks.{i}");
            for r in 0..cfg.layers_per_block[nl - 2 - i] {
                insert_resnet(
                    &mut w,
                    &format!("{pfx}.resnets.{r}"),
                    in_ch,
                    200 + i as u64 * 13 + r as u64,
                    dev,
                );
            }
            let proj_out = out_ch * t * s * s;
            w.insert(
                format!("{pfx}.proj.weight"),
                rnd(&[proj_out, in_ch], 300 + i as u64 * 13, dev),
            );
            w.insert(
                format!("{pfx}.proj.bias"),
                Tensor::zeros(proj_out, DType::F32, dev).unwrap(),
            );
        }

        for r in 0..cfg.layers_per_block[0] {
            insert_resnet(
                &mut w,
                &format!("block_out.resnets.{r}"),
                c_first,
                400 + r as u64,
                dev,
            );
        }
        w.insert(
            "proj_out.weight".into(),
            rnd(&[cfg.out_channels, c_first], 500, dev),
        );
        w.insert(
            "proj_out.bias".into(),
            Tensor::zeros(cfg.out_channels, DType::F32, dev).unwrap(),
        );
        w
    }

    fn max_abs(t: &Tensor) -> f32 {
        t.abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// The synthetic decoder reproduces the AsymmVAE geometry (temporal `(T−1)·6+1`, spatial ×8,
    /// out_channels 3) and is deterministic across two runs — the CPU CI-green VAE gate (no weights).
    #[test]
    fn synthetic_decode_shape_and_determinism() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vb = VarBuilder::from_tensors(synthetic_weights(&cfg, &dev), DType::F32, &dev);
        let dec =
            MochiVaeDecoder::new(vb, &cfg, DType::F32, &dev).expect("build synthetic decoder");

        // Teacher-forced latent [B=1, C=12, T_lat=2, H_lat=4, W_lat=4].
        let latent = rnd(&[1, 12, 2, 4, 4], 42, &dev);
        let v1 = dec.decode_denormalized(&latent).expect("decode 1");
        let v2 = dec.decode_denormalized(&latent).expect("decode 2");

        // Output: temporal (2-1)*6+1 = 7 frames, spatial 4*8 = 32, out_channels = 3.
        assert_eq!(v1.dims(), &[1, 3, 7, 32, 32], "decode output shape");
        let d = (&v1 - &v2)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(d, 0.0, "decode must be deterministic");
        assert!(
            v1.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|x| x.is_finite()),
            "decode produced non-finite values"
        );
    }

    /// **Chunked == single-shot, exactly** (sc-12291) — the candle mirror of the MLX
    /// `chunked_decode.rs` gate. Every op in the decoder is per-frame (GroupNorm is per-frame;
    /// silu/residual/proj are elementwise or per-position) or a causal conv fed real history by the
    /// `FrameCache`, so chunking is an exact refactor, not an approximation to be blended. Runs on the
    /// real resnet geometry, since the temporal coupling is the thing under test.
    #[test]
    fn chunked_decode_is_identical_to_single_shot() {
        let dev = Device::Cpu;
        let cfg = real_geometry_cfg();
        let vb = VarBuilder::from_tensors(synthetic_weights(&cfg, &dev), DType::F32, &dev);
        let dec = MochiVaeDecoder::new(vb, &cfg, DType::F32, &dev).expect("build decoder");

        let latent = rnd(&[1, 12, 13, 2, 2], 42, &dev);
        let single = dec.decode_denormalized(&latent).expect("single-shot");
        // Sanity: a non-constant decode, so "identical" is a real claim, not two flat tensors.
        assert!(
            max_abs(&single) > 1e-4,
            "synthetic decode is ~constant — the equivalence assertion would be vacuous"
        );

        // 1 and 4 do not divide 13 (ragged final chunk); 13 hits the `chunk >= t_lat` single-shot path.
        for chunk in [1usize, 2, 3, 4, 5, 13] {
            let chunked = dec
                .decode_denormalized_chunked(&latent, chunk)
                .unwrap_or_else(|e| panic!("chunked decode (chunk={chunk}): {e}"));
            assert_eq!(
                chunked.dims(),
                single.dims(),
                "chunk={chunk}: shape must match the single-shot decode"
            );
            let d = max_abs(&(&chunked - &single).unwrap());
            assert_eq!(
                d, 0.0,
                "chunk={chunk}: chunked decode must be identical to single-shot (max abs diff {d:.3e})"
            );
        }
    }
}
