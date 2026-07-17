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

use mlx_rs::ops::{add, concatenate_axis, pad, PadMode};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{group_norm, linear, silu};
use mlx_gen::tiling::MAX_WRITABLE_ELEMS;
use mlx_gen::weights::{join, Weights};
use mlx_gen::{CancelFlag, Error, Result};

use crate::config::MochiVaeConfig;

/// `nn.GroupNorm` default epsilon (`torch.nn.GroupNorm(eps=1e-5)`).
const GROUP_NORM_EPS: f32 = 1e-5;
/// `MochiChunkedGroupNorm3D` group count.
const NUM_GROUPS: i32 = 32;

/// Latent frames per chunk for [`MochiVaeDecoder::decode_chunked`].
///
/// The decode peak is dominated by `block_out`, which runs 128 channels at the **full** output
/// resolution, so the working set scales with the chunk. **Measured** (848×480/151 frames, real
/// weights, `get_peak_memory`, VAE weights included, one warm process):
///
/// | chunk | decode peak |
/// |---|---|
/// | **1** | **24.70 GiB** |
/// | 2 | 37.69 GiB |
/// | 4 | 65.62 GiB |
///
/// 1 is the floor, and this knob exists to lower the floor: ~13 GiB below chunk=2 is the difference
/// between Mochi fitting a 64 GB Mac with room and fitting it barely. A larger chunk trades that back
/// for fewer per-chunk syncs; raise it if decode wall-clock ever matters more than reach.
///
/// The point is the **flatness in clip length**, not the absolute number. Measured at chunk=1 in a cold
/// process (`decode_peak_is_flat_in_clip_length`):
///
/// | frames | 19 | 61 | 151 | 163 |
/// |---|---|---|---|---|
/// | peak GiB | 23.09 | 23.28 | 23.70 | 23.75 |
/// | secs | 16.8 | 45.2 | 127.4 | 427.2 |
///
/// 8.6× the frames for 1.03× the memory. (23.70 vs the table's 24.70 at 151 frames is process warmth —
/// the sweep above shares one process; treat ~24 GiB as the figure and the 1.03× as the claim.)
///
/// `decode_denormalized_chunked` clamps this to [`MochiVaeDecoder::max_safe_chunk_frames`] (6 at
/// 848×480), above which this decoder is measured to return wrong pixels — see [`MAX_WRITABLE_ELEMS`].
pub const DEFAULT_DECODE_CHUNK_FRAMES: usize = 1;

/// Per-conv temporal cache threaded through a chunked decode: each slot holds the last `kt−1` frames
/// of that conv's input from the previous chunk. `idx` resets to 0 each chunk and advances once per
/// conv in the fixed traversal order, so slots stay aligned (the [`crate::vae`] decode body is a
/// straight-line sequence, so the order is fixed). Mirrors the z48 Wan VAE's `FeatCache` idiom, and
/// the `conv_cache` diffusers threads through `AutoencoderKLMochi`'s own framewise decode.
///
/// **Why a cache and not overlap+blend.** Every op in this decoder is either per-frame (the
/// `GroupNorm(32)` is per-frame; silu/residual/proj are elementwise or per-position) or a causal conv,
/// so feeding a chunk the previous chunk's real trailing frames reproduces the single-shot decode
/// *exactly* — no seams to blend away. That matters here because the decoder's temporal receptive
/// field is **~45 latent frames** (38 stacked `kt=3` causal convs: 12 + 24 latent-rate frames through
/// `block_in`/`up_blocks[0]`, +5.3 at 3× and +4 at 6×) — *wider than a whole 5 s clip* (26 latent
/// frames). The repo's shared `mlx_gen::tiling` gives causal tiles a **1**-frame left context, which
/// would leave every tile after the first missing ~45 frames of history: not a boundary seam that a
/// trapezoidal blend can hide, but a wrong tile. See `tests/chunked_decode.rs`.
pub struct FrameCache {
    slots: Vec<Option<Array>>,
    idx: usize,
}

impl FrameCache {
    /// A cache with one slot per causal conv, all empty (chunk 0 falls back to the replicate pad).
    fn new(n: usize) -> Self {
        Self {
            slots: vec![None; n],
            idx: 0,
        }
    }
}

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

    /// `x_ncthw` → conv output (same T). `cache` threads the chunked decode: when its slot for this
    /// conv is populated (every chunk after the first) the previous chunk's real trailing frames stand
    /// in for the causal front-pad, which is what makes chunked == single-shot. `None` (or an empty
    /// slot) falls back to the reference replicate pad.
    fn forward(&self, x_ncthw: &Array, cache: Option<&mut FrameCache>) -> Result<Array> {
        let time_pad = self.kt - 1;

        // Temporal context: real history when chunking, else the causal replicate pad of frame 0.
        let xt = if time_pad == 0 {
            x_ncthw.clone()
        } else {
            match cache.as_ref().and_then(|c| c.slots[c.idx].clone()) {
                Some(prev) => concatenate_axis(&[&prev, x_ncthw], 2)?,
                None => pad(
                    x_ncthw,
                    &[(0, 0), (0, 0), (time_pad, 0), (0, 0), (0, 0)][..],
                    None,
                    Some(PadMode::Edge),
                )?,
            }
        };

        // Hand the next chunk this conv's trailing frames (taken post-concat, as diffusers does, so a
        // chunk shorter than `time_pad` still carries forward the right history), then advance the slot.
        if let Some(c) = cache {
            if time_pad > 0 {
                let tail = last_frames(&xt, time_pad)?;
                // Materialize the slice NOW. Left lazy it is a view onto `xt`, so the slot would pin
                // this conv's whole padded input — and its producing subgraph — alive until the
                // end-of-chunk eval, forcing all 38 convs' inputs to be live at once. That pinning,
                // not the working set, dominates the peak. Measured at 848×480/151 frames: it is worth
                // 61.00 → 37.69 GiB at chunk=2 and 35.01 → 24.70 GiB at chunk=1, for ~1.5× decode
                // wall-clock (the eval serializes on a GPU sync per conv). The decode runs once per
                // generation, after minutes of denoise, so the memory is the better side of that trade.
                mlx_rs::transforms::eval([&tail])?;
                c.slots[c.idx] = Some(tail);
            }
            c.idx += 1;
        }

        // Spatial replicate pad is symmetric and chunk-independent. NCTHW axes: 2=T, 3=H, 4=W.
        let h_pad = (self.kh - 1) / 2;
        let w_pad = (self.kw - 1) / 2;
        let xp = if h_pad > 0 || w_pad > 0 {
            pad(
                &xt,
                &[(0, 0), (0, 0), (0, 0), (h_pad, h_pad), (w_pad, w_pad)][..],
                None,
                Some(PadMode::Edge),
            )?
        } else {
            xt
        };

        // NCTHW -> NDHWC, conv (valid), back to NCTHW.
        let xp = xp.transpose_axes(&[0, 2, 3, 4, 1])?;
        let y = mlx_gen::nn::conv3d(&xp, &self.w, Some(&self.b), (1, 1, 1), (0, 0, 0))?;
        Ok(y.transpose_axes(&[0, 4, 1, 2, 3])?)
    }
}

/// The last `n` frames along the NCTHW temporal axis.
fn last_frames(x: &Array, n: i32) -> Result<Array> {
    let t = x.shape()[2];
    let idx: Vec<i32> = (t - n..t).collect();
    let idx = Array::from_slice(&idx, &[idx.len() as i32]);
    Ok(x.take_axis(&idx, 2)?)
}

/// Frames `[start, end)` along the NCTHW temporal axis.
fn slice_frames(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    let idx = Array::from_slice(&idx, &[idx.len() as i32]);
    Ok(x.take_axis(&idx, 2)?)
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

    fn forward(&self, x: &Array, mut cache: Option<&mut FrameCache>) -> Result<Array> {
        let h = self
            .conv1
            .forward(&silu(&self.norm1.forward(x)?)?, cache.as_deref_mut())?;
        let h = self
            .conv2
            .forward(&silu(&self.norm2.forward(&h)?)?, cache)?;
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

    fn forward(&self, x: &Array, mut cache: Option<&mut FrameCache>) -> Result<Array> {
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

    fn forward(&self, x: &Array, mut cache: Option<&mut FrameCache>) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x, cache.as_deref_mut())?;
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
    /// `∏ spatial_expansions` (8) — with `temporal_ratio`, sizes the decode's largest intermediate.
    spatial_ratio: i32,
    /// `decoder_block_out_channels[0]` (128) — the width `block_out` runs at full output resolution.
    first_block_channels: i32,
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
            spatial_ratio: cfg.spatial_compression_ratio() as i32,
            first_block_channels: cfg.decoder_block_out_channels[0] as i32,
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

    /// Elements in the decode's largest intermediate — `block_out`'s spatially-padded conv input, which
    /// runs `first_block_channels` at the full output resolution — for `t_lat` latent frames.
    fn peak_tensor_elems(&self, t_lat: i32, h_lat: i32, w_lat: i32) -> i64 {
        let t = (self.temporal_ratio as i64) * (t_lat as i64) + 2; // +2: causal front-pad (kt−1)
        let h = (h_lat as i64) * (self.spatial_ratio as i64) + 2; // +2: symmetric spatial pad
        let w = (w_lat as i64) * (self.spatial_ratio as i64) + 2;
        (self.first_block_channels as i64) * t * h * w
    }

    /// The largest chunk, in latent frames, whose decode keeps every intermediate under
    /// [`MAX_WRITABLE_ELEMS`] at this latent geometry — **6** at 848×480. 0 when even a single frame
    /// would exceed it (only reachable at resolutions far above Mochi's 848×480 design point).
    ///
    /// This decoder is where that bound was found (sc-12291). Measured on real AsymmVAE weights at
    /// 848×480, varying only clip length: an untiled decode is exact through `T_lat = 6` (`block_out` =
    /// 1.88e9 elements) and silently returns wrong values — no error, deterministic, two runs
    /// byte-identical — from `T_lat = 7` (2.19e9) on: **±2.67 where a valid video is ~[-1, 1]**, against
    /// ±0.50 from the chunked path on the same latents. `block_out` writes 128 channels at the full
    /// output resolution, which is what puts it over the line.
    ///
    /// The failure is invisible to the `vae_parity` golden, dumped at 64×64/7 frames (6.3e6 elements,
    /// ~340× under); the shipped 848×480/151-frame default is 8.13e9, ~3.8× **over**. Chunking keeps
    /// every intermediate far below it ([`DEFAULT_DECODE_CHUNK_FRAMES`] → 4.2e8, ~5× under), which is
    /// why the chunked decode is not merely cheaper but *correct where the untiled one is not*.
    pub fn max_safe_chunk_frames(&self, h_lat: i32, w_lat: i32) -> usize {
        let per_frame_row =
            self.peak_tensor_elems(1, h_lat, w_lat) - self.peak_tensor_elems(0, h_lat, w_lat);
        if per_frame_row <= 0 {
            return 1;
        }
        let headroom = MAX_WRITABLE_ELEMS - self.peak_tensor_elems(0, h_lat, w_lat);
        (headroom / per_frame_row).max(0) as usize
    }

    /// The decode body, **without** the leading-frame drop: de-normalized latent → raw `6·T_lat`-frame
    /// video. `cache` is `None` for a single-shot decode and `Some` for one chunk of a chunked decode.
    fn decode_body(&self, latents: &Array, mut cache: Option<&mut FrameCache>) -> Result<Array> {
        // Refuse a decode whose intermediates would exceed the element ceiling. MLX does not error on
        // this — it returns plausible-looking wrong pixels — so an explicit error is the only way the
        // caller ever learns. See [`MAX_WRITABLE_ELEMS`].
        let sh = latents.shape();
        let elems = self.peak_tensor_elems(sh[2], sh[3], sh[4]);
        if elems > MAX_WRITABLE_ELEMS {
            return Err(Error::Msg(format!(
                "mochi vae: a {}-latent-frame decode at {}×{} needs a {elems}-element intermediate, \
                 over the {MAX_WRITABLE_ELEMS} ceiling above which MLX silently returns wrong pixels. \
                 Decode in chunks of ≤{} latent frames (decode_chunked / decode_denormalized_chunked).",
                sh[2],
                sh[3] * self.spatial_ratio,
                sh[4] * self.spatial_ratio,
                self.max_safe_chunk_frames(sh[3], sh[4]),
            )));
        }

        let x = latents.as_dtype(self.dtype)?;
        let mut x = self.conv_in.forward(&x, cache.as_deref_mut())?;
        x = self.block_in.forward(&x, cache.as_deref_mut())?;
        for up in &self.up_blocks {
            x = up.forward(&x, cache.as_deref_mut())?;
        }
        x = self.block_out.forward(&x, cache)?;
        x = silu(&x)?;
        // Channel-last proj_out: NCTHW -> [B,T,H,W,C] -> Linear(C->out) -> NCTHW.
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?;
        let x = linear(&x, &self.proj_out_w, &self.proj_out_b)?;
        Ok(x.transpose_axes(&[0, 4, 1, 2, 3])?)
    }

    /// One `FrameCache` slot per causal conv, in decode-body traversal order.
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
    /// pass. Teacher-forced entry point (the vae_parity gate feeds the golden's
    /// `denormalized_latents`), and the reference [`decode_denormalized_chunked`](Self::decode_denormalized_chunked) is gated against.
    ///
    /// Peak scales linearly in `T_lat`; prefer [`decode_chunked`](Self::decode_chunked) in production.
    pub fn decode_denormalized(&self, latents: &Array) -> Result<Array> {
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
        latents: &Array,
        chunk_frames: usize,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let sh = latents.shape();
        let (t_lat, h_lat, w_lat) = (sh[2], sh[3], sh[4]);
        // Clamp to what the element ceiling allows: a too-large chunk would not merely use more memory,
        // it would silently corrupt (see [`MAX_WRITABLE_ELEMS`]). `decode_body` still errors if even one
        // frame is too big, so the clamp never masks an impossible geometry.
        let safe = self.max_safe_chunk_frames(h_lat, w_lat).max(1);
        let chunk = chunk_frames.clamp(1, safe) as i32;
        if chunk >= t_lat {
            return self.decode_denormalized(latents);
        }

        let mut cache = FrameCache::new(self.conv_count());
        let mut chunks: Vec<Array> = Vec::new();
        let mut start = 0i32;
        while start < t_lat {
            if cancel.is_some_and(|c| c.is_cancelled()) {
                return Err(Error::Canceled);
            }
            let end = (start + chunk).min(t_lat);
            cache.idx = 0;
            let v = self.decode_body(&slice_frames(latents, start, end)?, Some(&mut cache))?;
            // The full decode drops the leading `ratio − 1` frames of the whole video; chunk 0 owns
            // them (it always yields `ratio` ≥ 5 frames), and later chunks concatenate untouched.
            let v = if start == 0 {
                self.drop_last_temporal_frames(&v)?
            } else {
                v
            };
            chunks.push(v);
            // Materialize the chunk + the carried cache: without this the lazy graph would hold every
            // chunk's full-resolution intermediates at once and defeat the chunking entirely.
            let mut ev: Vec<&Array> = cache.slots.iter().flatten().collect();
            ev.push(chunks.last().expect("just pushed"));
            mlx_rs::transforms::eval(ev)?;
            start = end;
        }

        let refs: Vec<&Array> = chunks.iter().collect();
        Ok(concatenate_axis(&refs, 2)?)
    }

    /// De-normalize then decode a raw latent → video, in a single pass.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let denorm = self.denormalize(latents)?;
        self.decode_denormalized(&denorm)
    }

    /// De-normalize then chunk-decode a raw latent → video (the production entry point). See
    /// [`decode_denormalized_chunked`](Self::decode_denormalized_chunked)(Self::decode_denormalized_chunked); `chunk_frames` of
    /// [`DEFAULT_DECODE_CHUNK_FRAMES`] is the shipped default.
    pub fn decode_chunked(
        &self,
        latents: &Array,
        chunk_frames: usize,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        // De-normalization is per-element on the *latent*, so doing it once up front costs a latent-
        // sized tensor (12 ch), not a decode-sized one.
        let denorm = self.denormalize(latents)?;
        self.decode_denormalized_chunked(&denorm, chunk_frames, cancel)
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
