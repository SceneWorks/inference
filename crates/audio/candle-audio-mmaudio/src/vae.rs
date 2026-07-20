//! MMAudio's **latent mel-VAE decoder** (sc-13440) — the Make-An-Audio-2 mel-VAE with EDM2
//! magnitude-preserving (MP) primitives — ported natively onto the pinned candle revision.
//!
//! This is the first stage of MMAudio's 16 kHz **output path**: a DiT latent `z` (embed_dim 20)
//! decodes to an 80-band log-mel spectrogram, which [`crate::bigvgan`] then vocodes to a 16 kHz
//! waveform. Only the **decode** path is ported — the video→audio pipeline never runs the VAE
//! encoder (MMAudio itself `del`s `vae.encoder` when `need_vae_encoder=False`), so the encoder is
//! deliberately out of scope here.
//!
//! ## Faithful to `mmaudio/ext/autoencoder/{vae,vae_modules,edm2_utils}.py`
//!
//! The decoder is `Decoder1D` configured by `VAE_16k` (`data_dim=80`, `embed_dim=20`,
//! `hidden_dim=384`, `ch_mult=(1,2,4)`, `num_res_blocks=2`, `attn_layers=[3]`, `down_layers=[0]`).
//! Every layer is an [`MpConv1d`] (EDM2 `MPConv1D`): a bias-free 1-D conv whose weights are
//! **force-weight-normalized** at load (`remove_weight_norm`, replicated exactly here — see
//! [`MpConv1d::from_raw`]). The building blocks:
//!
//! - **`nonlinearity` = magnitude-preserving SiLU** `silu(x)/0.596` ([`mp_silu`]).
//! - **`mp_sum(a,b,t=0.3)`** `lerp(a,b,t)/sqrt((1-t)^2+t^2)` ([`mp_sum`]) — the residual combiner.
//! - **pixel-norm** `normalize(x, dim=1)` (`channel_normalize`) — EDM2 unit-magnitude normalize
//!   over the channel axis with `eps=1e-4` and the `1/sqrt(dim)` scaling.
//! - **`ResnetBlock1D`**: pixel-norm → `mp_silu` → conv1 → `mp_silu` → conv2, `nin_shortcut` when
//!   the channel count changes, combined by `mp_sum(·,·,0.3)`.
//! - **`AttnBlock1D`**: 1-head MP self-attention over the time axis (`qkv`/`proj_out` are 1×1
//!   MP convs, the qkv split is channel-**interleaved** `c·3+{q,k,v}`, and q/k/v are pixel-normed
//!   before scaled-dot-product attention), combined by `mp_sum(·,·,0.3)`.
//! - **`Upsample1D`**: nearest-exact ×2 then a 3-tap MP conv.
//!
//! The decode forward is `conv_in → mid(block,attn,block) → clamp(±256) → for each level: 3
//! ResnetBlocks (clamp ±256 after each) then ×2 upsample at level 1 → mp_silu → conv_out`, then
//! **unnormalize** `mel = dec·data_std + data_mean` with the fixed 80-band statistics
//! ([`DATA_MEAN_80D`]/[`DATA_STD_80D`], transcribed verbatim from the reference). One upsample
//! stage means `mel_len = 2·latent_len`.

use candle_audio::candle_core::{DType, Result as CResult, Tensor, D};
use candle_nn::VarBuilder;

/// Latent channels the 16k VAE decodes from (`VAE_16k.embed_dim`).
pub const EMBED_DIM: usize = 20;
/// Mel bands the decoder emits (`VAE_16k.data_dim`).
pub const DATA_DIM: usize = 80;
/// Backbone width (`VAE_16k.hidden_dim`).
pub const HIDDEN_DIM: usize = 384;

/// Latent channels the 44k VAE decodes from (`VAE_44k.embed_dim`). The **confirmed** 44k latent
/// channel count — the reference `vae.py::VAE_44k` sets `embed_dim=40` (the 16k VAE is 20-d), and
/// the large_44k_v2 DiT's `latent_dim=40` matches. Recorded here since it was an open unknown at
/// scoping (sc-13441).
pub const EMBED_DIM_44K: usize = 40;
/// Mel bands the 44k decoder emits (`VAE_44k.data_dim` = 128 — the 44k mel is 128-band, vs 80 @ 16k).
pub const DATA_DIM_44K: usize = 128;
/// Backbone width (`VAE_44k.hidden_dim`).
pub const HIDDEN_DIM_44K: usize = 512;

/// The parameterized latent mel-VAE decoder configuration (`vae.py::get_my_vae`). Both the 16k and
/// 44k VAEs share the same `Decoder1D` topology (`ch_mult=(1,2,4)`, `num_res_blocks=2`,
/// `attn_layers=[3]`, `down_layers=[0]` → one ×2 upsample) and differ only in these three dims.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Latent channels (`embed_dim`): 20 (16k) / 40 (44k).
    pub embed_dim: usize,
    /// Mel bands the decoder emits (`data_dim`): 80 (16k) / 128 (44k).
    pub data_dim: usize,
    /// Backbone width (`hidden_dim`): 384 (16k) / 512 (44k).
    pub hidden_dim: usize,
}

impl Config {
    /// The 16 kHz mel-VAE (`VAE_16k`): 20-d latent → 80-band log-mel, hidden 384.
    pub fn vae_16k() -> Self {
        Self {
            embed_dim: EMBED_DIM,
            data_dim: DATA_DIM,
            hidden_dim: HIDDEN_DIM,
        }
    }

    /// The 44.1 kHz mel-VAE (`VAE_44k`): 40-d latent → 128-band log-mel, hidden 512 (sc-13441).
    pub fn vae_44k() -> Self {
        Self {
            embed_dim: EMBED_DIM_44K,
            data_dim: DATA_DIM_44K,
            hidden_dim: HIDDEN_DIM_44K,
        }
    }
}

/// Channel multipliers per resolution level (`ch_mult`).
pub const CH_MULT: [usize; 3] = [1, 2, 4];
/// Residual blocks per level in the encoder; the decoder uses `+1`.
pub const NUM_RES_BLOCKS: usize = 2;
/// Activation clamp bound (`clip_act`).
pub const CLIP_ACT: f64 = 256.0;
/// EDM2 normalize epsilon (`normalize(..., eps=1e-4)`).
const EPS: f64 = 1e-4;
/// `mp_sum` interpolation used by every residual combine (`t=0.3`).
const MP_SUM_T: f64 = 0.3;

/// Per-band mel mean the 16k VAE unnormalizes with (`DATA_MEAN_80D`, verbatim from
/// `mmaudio/ext/autoencoder/vae.py`).
pub const DATA_MEAN_80D: [f32; DATA_DIM] = [
    -1.6058, -1.3676, -1.2520, -1.2453, -1.2078, -1.2224, -1.2419, -1.2439, -1.2922, -1.2927,
    -1.3170, -1.3543, -1.3401, -1.3836, -1.3907, -1.3912, -1.4313, -1.4152, -1.4527, -1.4728,
    -1.4568, -1.5101, -1.5051, -1.5172, -1.5623, -1.5373, -1.5746, -1.5687, -1.6032, -1.6131,
    -1.6081, -1.6331, -1.6489, -1.6489, -1.6700, -1.6738, -1.6953, -1.6969, -1.7048, -1.7280,
    -1.7361, -1.7495, -1.7658, -1.7814, -1.7889, -1.8064, -1.8221, -1.8377, -1.8417, -1.8643,
    -1.8857, -1.8929, -1.9173, -1.9379, -1.9531, -1.9673, -1.9824, -2.0042, -2.0215, -2.0436,
    -2.0766, -2.1064, -2.1418, -2.1855, -2.2319, -2.2767, -2.3161, -2.3572, -2.3954, -2.4282,
    -2.4659, -2.5072, -2.5552, -2.6074, -2.6584, -2.7107, -2.7634, -2.8266, -2.8981, -2.9673,
];

/// Per-band mel std the 16k VAE unnormalizes with (`DATA_STD_80D`, verbatim from the reference).
pub const DATA_STD_80D: [f32; DATA_DIM] = [
    1.0291, 1.0411, 1.0043, 0.9820, 0.9677, 0.9543, 0.9450, 0.9392, 0.9343, 0.9297, 0.9276, 0.9263,
    0.9242, 0.9254, 0.9232, 0.9281, 0.9263, 0.9315, 0.9274, 0.9247, 0.9277, 0.9199, 0.9188, 0.9194,
    0.9160, 0.9161, 0.9146, 0.9161, 0.9100, 0.9095, 0.9145, 0.9076, 0.9066, 0.9095, 0.9032, 0.9043,
    0.9038, 0.9011, 0.9019, 0.9010, 0.8984, 0.8983, 0.8986, 0.8961, 0.8962, 0.8978, 0.8962, 0.8973,
    0.8993, 0.8976, 0.8995, 0.9016, 0.8982, 0.8972, 0.8974, 0.8949, 0.8940, 0.8947, 0.8936, 0.8939,
    0.8951, 0.8956, 0.9017, 0.9167, 0.9436, 0.9690, 1.0003, 1.0225, 1.0381, 1.0491, 1.0545, 1.0604,
    1.0761, 1.0929, 1.1089, 1.1196, 1.1176, 1.1156, 1.1117, 1.1070,
];

/// Magnitude-preserving SiLU: `silu(x) / 0.596` (`mp_silu`, EDM2 Eq. 81).
pub fn mp_silu(x: &Tensor) -> CResult<Tensor> {
    candle_nn::ops::silu(x)? / 0.596
}

/// Magnitude-preserving sum: `lerp(a, b, t) / sqrt((1-t)^2 + t^2)` (`mp_sum`, EDM2 Eq. 88).
/// `lerp(a, b, t) = a + t·(b - a) = (1-t)·a + t·b`.
pub fn mp_sum(a: &Tensor, b: &Tensor, t: f64) -> CResult<Tensor> {
    let denom = ((1.0 - t).powi(2) + t.powi(2)).sqrt();
    let mixed = ((a * (1.0 - t))? + (b * t)?)?;
    mixed / denom
}

/// EDM2 unit-magnitude normalize over `dim` (`normalize(x, dim)`):
/// `x / (eps + ||x||_dim / sqrt(D))` where `D` is the size along `dim`. Used as the pixel-norm.
fn channel_normalize(x: &Tensor, dim: usize) -> CResult<Tensor> {
    let d = x.dim(dim)? as f64;
    let norm = x.sqr()?.sum_keepdim(dim)?.sqrt()?; // ||x||_2 over `dim`, keepdim
                                                   // torch: norm = eps + norm * (1/sqrt(D))  →  x / norm
    let denom = (norm * (1.0 / d.sqrt()))?.affine(1.0, EPS)?;
    x.broadcast_div(&denom)
}

/// EDM2 magnitude-preserving 1-D conv (`MPConv1D`) with weight-norm already removed.
///
/// Bias-free; padding is `kernel_size / 2` (matching the reference `conv1d(..., padding=k//2)`).
/// A per-conv scalar `gain` (`(learnable_gain + 1)` on the output convs) is applied to the output,
/// which is exact because convolution is linear in the kernel.
pub struct MpConv1d {
    weight: Tensor, // (out, in, k) — force-weight-normalized at load
    padding: usize,
    gain: f64,
}

impl MpConv1d {
    /// Build from the **raw** checkpoint weight `(out, in, k)`, replicating EDM2's
    /// `remove_weight_norm`: `w = normalize(w) / sqrt(in·k)`, which reduces to
    /// `w[o] = w_raw[o] / (||w_raw[o]||_2 + eps·sqrt(in·k))` per output channel.
    pub fn from_raw(raw: &Tensor, gain: f64) -> CResult<Self> {
        let (_out, inc, k) = raw.dims3()?;
        let fan = (inc * k) as f64;
        let norm = raw.sqr()?.sum_keepdim((1, 2))?.sqrt()?; // (out,1,1)
        let denom = norm.affine(1.0, EPS * fan.sqrt())?; // ||·|| + eps·sqrt(in·k)
        let weight = raw.broadcast_div(&denom)?;
        Ok(Self {
            weight,
            padding: k / 2,
            gain,
        })
    }

    /// Load `proj`-style: read `{prefix}.weight` at the expected shape and build.
    pub fn load(vb: VarBuilder, out: usize, inc: usize, k: usize, gain: f64) -> CResult<Self> {
        let raw = vb.get((out, inc, k), "weight")?;
        Self::from_raw(&raw, gain)
    }

    /// `(B, in, L) → (B, out, L)` (kernel-size-preserving; stride 1, dilation 1).
    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let y = x.conv1d(&self.weight, self.padding, 1, 1, 1)?;
        if (self.gain - 1.0).abs() < f64::EPSILON {
            Ok(y)
        } else {
            y * self.gain
        }
    }
}

/// `ResnetBlock1D`: pixel-norm → mp_silu → conv1 → mp_silu → conv2, `nin_shortcut` on channel
/// change, combined by `mp_sum(shortcut, h, 0.3)`.
struct ResnetBlock1d {
    conv1: MpConv1d,
    conv2: MpConv1d,
    nin_shortcut: Option<MpConv1d>,
    use_norm: bool,
}

impl ResnetBlock1d {
    fn load(vb: VarBuilder, in_dim: usize, out_dim: usize, use_norm: bool) -> CResult<Self> {
        let conv1 = MpConv1d::load(vb.pp("conv1"), out_dim, in_dim, 3, 1.0)?;
        let conv2 = MpConv1d::load(vb.pp("conv2"), out_dim, out_dim, 3, 1.0)?;
        let nin_shortcut = if in_dim != out_dim {
            Some(MpConv1d::load(
                vb.pp("nin_shortcut"),
                out_dim,
                in_dim,
                1,
                1.0,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1,
            conv2,
            nin_shortcut,
            use_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let x = if self.use_norm {
            channel_normalize(x, 1)?
        } else {
            x.clone()
        };
        let h = mp_silu(&x)?;
        let h = self.conv1.forward(&h)?;
        let h = mp_silu(&h)?;
        let h = self.conv2.forward(&h)?;
        let shortcut = match &self.nin_shortcut {
            Some(c) => c.forward(&x)?,
            None => x,
        };
        mp_sum(&shortcut, &h, MP_SUM_T)
    }
}

/// `AttnBlock1D`: single-head magnitude-preserving self-attention over the time axis.
struct AttnBlock1d {
    qkv: MpConv1d,
    proj_out: MpConv1d,
    channels: usize,
}

impl AttnBlock1d {
    fn load(vb: VarBuilder, channels: usize) -> CResult<Self> {
        let qkv = MpConv1d::load(vb.pp("qkv"), channels * 3, channels, 1, 1.0)?;
        let proj_out = MpConv1d::load(vb.pp("proj_out"), channels, channels, 1, 1.0)?;
        Ok(Self {
            qkv,
            proj_out,
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let (b, _c, l) = x.dims3()?;
        let c = self.channels;
        let y = self.qkv.forward(x)?; // (B, 3C, L)
                                      // torch reshape (B, heads=1, C, 3, L) from (B, 3C, L): the 3C axis
                                      // splits channel-interleaved as ch = c·3 + {q,k,v}.
        let y = y.reshape((b, c, 3, l))?;
        // normalize over the C axis (reference `normalize(y, dim=2)` on the heads=1 layout).
        let y = channel_normalize(&y, 1)?;
        let q = y.narrow(2, 0, 1)?.squeeze(2)?; // (B, C, L)
        let k = y.narrow(2, 1, 1)?.squeeze(2)?;
        let v = y.narrow(2, 2, 1)?.squeeze(2)?;
        // rearrange 'b c l -> b l c', SDPA over c with scale 1/sqrt(C), back to 'b c l'.
        let q = q.transpose(1, 2)?.contiguous()?; // (B, L, C)
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (c as f64).sqrt();
        let sim = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?; // (B, L, L)
        let attn = candle_nn::ops::softmax_last_dim(&sim)?;
        let out = attn.matmul(&v)?; // (B, L, C)
        let out = out.transpose(1, 2)?.contiguous()?; // (B, C, L)
        let out = self.proj_out.forward(&out)?;
        mp_sum(x, &out, MP_SUM_T)
    }
}

/// `Upsample1D`: nearest-exact ×2 followed by a 3-tap MP conv.
struct Upsample1d {
    conv: MpConv1d,
}

impl Upsample1d {
    fn load(vb: VarBuilder, channels: usize) -> CResult<Self> {
        Ok(Self {
            conv: MpConv1d::load(vb.pp("conv"), channels, channels, 3, 1.0)?,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        // nearest-exact ×2 == repeat each timestep twice (exact for an integer factor).
        let (b, c, l) = x.dims3()?;
        let up = x
            .unsqueeze(D::Minus1)? // (B, C, L, 1)
            .broadcast_as((b, c, l, 2))?
            .contiguous()?
            .reshape((b, c, l * 2))?;
        self.conv.forward(&up)
    }
}

/// One decoder resolution level: `num_res_blocks + 1` residual blocks, optional attention, and an
/// optional ×2 upsample.
struct UpLevel {
    blocks: Vec<ResnetBlock1d>,
    attn: Vec<AttnBlock1d>,
    upsample: Option<Upsample1d>,
}

/// `Decoder1D` for the 16k VAE — latent `(B, 20, L)` → mel `(B, 80, 2L)` (pre-unnormalize).
struct Decoder1d {
    conv_in: MpConv1d,
    mid_block_1: ResnetBlock1d,
    mid_attn_1: AttnBlock1d,
    mid_block_2: ResnetBlock1d,
    levels: Vec<UpLevel>, // indexed by i_level (0..num_layers)
    conv_out: MpConv1d,
}

impl Decoder1d {
    fn load(vb: VarBuilder, cfg: Config, learnable_gain: f32) -> CResult<Self> {
        let num_layers = CH_MULT.len();
        let block_in0 = cfg.hidden_dim * CH_MULT[num_layers - 1];
        let conv_in = MpConv1d::load(vb.pp("conv_in"), block_in0, cfg.embed_dim, 3, 1.0)?;
        let mid = vb.pp("mid");
        let mid_block_1 = ResnetBlock1d::load(mid.pp("block_1"), block_in0, block_in0, true)?;
        let mid_attn_1 = AttnBlock1d::load(mid.pp("attn_1"), block_in0)?;
        let mid_block_2 = ResnetBlock1d::load(mid.pp("block_2"), block_in0, block_in0, true)?;

        // Decoder up levels: down_layers shifts by +1 (each down adds one) → upsample at level 1.
        let dec_down_layers = [1usize];
        let attn_layers = [3usize]; // never matched at the 3 decoder levels
        let up = vb.pp("up");
        let mut levels: Vec<UpLevel> = Vec::with_capacity(num_layers);
        // Build levels indexed 0..num_layers to mirror `self.up[i_level]`.
        let mut block_in = block_in0;
        // reversed order builds channel sizes; store per i_level.
        let mut per_level: Vec<UpLevel> = Vec::with_capacity(num_layers);
        for i_level in (0..num_layers).rev() {
            let level_vb = up.pp(i_level);
            let block_out = cfg.hidden_dim * CH_MULT[i_level];
            let mut blocks = Vec::with_capacity(NUM_RES_BLOCKS + 1);
            let mut attn = Vec::new();
            for i_block in 0..(NUM_RES_BLOCKS + 1) {
                blocks.push(ResnetBlock1d::load(
                    level_vb.pp("block").pp(i_block),
                    block_in,
                    block_out,
                    true,
                )?);
                block_in = block_out;
                if attn_layers.contains(&i_level) {
                    attn.push(AttnBlock1d::load(
                        level_vb.pp("attn").pp(attn.len()),
                        block_in,
                    )?);
                }
            }
            let upsample = if dec_down_layers.contains(&i_level) {
                Some(Upsample1d::load(level_vb.pp("upsample"), block_in)?)
            } else {
                None
            };
            per_level.push(UpLevel {
                blocks,
                attn,
                upsample,
            });
        }
        // per_level is in reversed (high→low i_level) order; index by i_level.
        per_level.reverse();
        levels.extend(per_level);

        let conv_out = MpConv1d::load(
            vb.pp("conv_out"),
            cfg.data_dim,
            block_in,
            3,
            (learnable_gain + 1.0) as f64,
        )?;
        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            levels,
            conv_out,
        })
    }

    fn forward(&self, z: &Tensor) -> CResult<Tensor> {
        let mut h = self.conv_in.forward(z)?;
        h = self.mid_block_1.forward(&h)?;
        h = self.mid_attn_1.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;
        h = h.clamp(-CLIP_ACT, CLIP_ACT)?;
        for i_level in (0..self.levels.len()).rev() {
            let level = &self.levels[i_level];
            for (bi, block) in level.blocks.iter().enumerate() {
                h = block.forward(&h)?;
                if let Some(a) = level.attn.get(bi) {
                    h = a.forward(&h)?;
                }
                h = h.clamp(-CLIP_ACT, CLIP_ACT)?;
            }
            if let Some(up) = &level.upsample {
                h = up.forward(&h)?;
            }
        }
        h = mp_silu(&h)?;
        self.conv_out.forward(&h)
    }
}

/// The assembled 16k mel-VAE **decoder** with its unnormalize statistics.
pub struct MelVaeDecoder {
    decoder: Decoder1d,
    data_mean: Tensor, // (1, 80, 1)
    data_std: Tensor,  // (1, 80, 1)
}

impl MelVaeDecoder {
    /// Load from a `VarBuilder` rooted at the VAE checkpoint (keys `decoder.*`, `data_mean`,
    /// `data_std`). The scalar `decoder.learnable_gain` is read as f32 (it scales `conv_out`
    /// linearly).
    ///
    /// **`data_mean` / `data_std` are read from the checkpoint, not from [`DATA_MEAN_80D`] /
    /// [`DATA_STD_80D`].** Those source constants only *initialize* the registered buffers; the
    /// released 16k checkpoint carries slightly different (training-updated) values, so unnormalize
    /// must use the checkpoint's `(1, 80, 1)` buffers to match the reference bit-for-bit.
    pub fn load(vb: VarBuilder) -> CResult<Self> {
        Self::load_with_config(vb, Config::vae_16k())
    }

    /// Load a mel-VAE decoder with an explicit [`Config`] (16k or 44k, sc-13441). Identical topology;
    /// only `embed_dim`/`data_dim`/`hidden_dim` differ. `data_mean`/`data_std` are read from the
    /// checkpoint at the config's `data_dim`.
    pub fn load_with_config(vb: VarBuilder, cfg: Config) -> CResult<Self> {
        // learnable_gain is a scalar Parameter under `decoder.learnable_gain`.
        let learnable_gain = vb
            .get((), "decoder.learnable_gain")?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        let decoder = Decoder1d::load(vb.pp("decoder"), cfg, learnable_gain)?;
        let data_mean = vb
            .get((1, cfg.data_dim, 1), "data_mean")?
            .to_dtype(DType::F32)?;
        let data_std = vb
            .get((1, cfg.data_dim, 1), "data_std")?
            .to_dtype(DType::F32)?;
        Ok(Self {
            decoder,
            data_mean,
            data_std,
        })
    }

    /// Decode a latent `(B, embed_dim=20, L)` to an unnormalized 80-band log-mel `(B, 80, 2L)`.
    ///
    /// Mirrors `VAE.decode(z, unnormalize=True)` — the caller supplies the latent already in
    /// channel-first `(B, C, L)` layout (MMAudio's `features_utils.decode` transposes the DiT's
    /// `(B, L, C)` latent before this; that transpose is a caller/DiT concern, not the VAE's).
    pub fn decode(&self, z: &Tensor) -> CResult<Tensor> {
        let dec = self.decoder.forward(z)?;
        let out = dec
            .broadcast_mul(&self.data_std)?
            .broadcast_add(&self.data_mean)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::Device;

    #[test]
    fn mp_sum_matches_reference_formula() {
        let dev = Device::Cpu;
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 3, 1), &dev).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], (1, 3, 1), &dev).unwrap();
        let out = mp_sum(&a, &b, 0.3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let denom = (0.7f32.powi(2) + 0.3f32.powi(2)).sqrt();
        for (i, (av, bv)) in [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)].iter().enumerate() {
            let expect = (0.7 * av + 0.3 * bv) / denom;
            assert!((out[i] - expect).abs() < 1e-6, "mp_sum[{i}]={} ", out[i]);
        }
    }

    #[test]
    fn mp_silu_is_silu_over_0596() {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[0.0f32, 1.0, -1.0, 2.5], (1, 4, 1), &dev).unwrap();
        let out = mp_silu(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, xv) in [0.0f32, 1.0, -1.0, 2.5].iter().enumerate() {
            let silu = xv / (1.0 + (-xv).exp());
            let expect = silu / 0.596;
            assert!((out[i] - expect).abs() < 1e-5, "mp_silu[{i}]={}", out[i]);
        }
    }

    #[test]
    fn weight_norm_removal_normalizes_per_output_channel() {
        let dev = Device::Cpu;
        // A raw weight (out=2, in=3, k=3); check final norm == raw_norm / (raw_norm + eps·sqrt(9)).
        let raw = Tensor::randn(0f32, 1.0, (2, 3, 3), &dev).unwrap();
        let conv = MpConv1d::from_raw(&raw, 1.0).unwrap();
        let fan = 9f64;
        let raw_v = raw.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let w_v = conv.weight.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for o in 0..2 {
            let mut nrm = 0f64;
            for j in 0..9 {
                nrm += (raw_v[o * 9 + j] as f64).powi(2);
            }
            let nrm = nrm.sqrt();
            let denom = nrm + 1e-4 * fan.sqrt();
            for j in 0..9 {
                let expect = raw_v[o * 9 + j] as f64 / denom;
                assert!(
                    (w_v[o * 9 + j] as f64 - expect).abs() < 1e-6,
                    "weight[{o},{j}] mismatch"
                );
            }
        }
    }

    #[test]
    fn vae_configs_match_reference() {
        let k16 = Config::vae_16k();
        assert_eq!((k16.embed_dim, k16.data_dim, k16.hidden_dim), (20, 80, 384));
        let k44 = Config::vae_44k();
        // The confirmed 44k latent channel count (open unknown at scoping): embed_dim=40, 128-band mel.
        assert_eq!(
            (k44.embed_dim, k44.data_dim, k44.hidden_dim),
            (40, 128, 512)
        );
    }

    #[test]
    fn channel_normalize_unit_scale() {
        let dev = Device::Cpu;
        // (B=1, C=4, L=1): norm over channels.
        let x = Tensor::from_slice(&[3.0f32, 4.0, 0.0, 0.0], (1, 4, 1), &dev).unwrap();
        let out = channel_normalize(&x, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // ||x|| = 5, denom = 1e-4 + 5/sqrt(4) = 1e-4 + 2.5.
        let denom = 1e-4 + 5.0 / 2.0;
        assert!((out[0] - 3.0 / denom).abs() < 1e-5);
        assert!((out[1] - 4.0 / denom).abs() < 1e-5);
    }
}
