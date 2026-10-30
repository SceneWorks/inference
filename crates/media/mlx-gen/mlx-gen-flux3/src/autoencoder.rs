//! The FLUX 3 image autoencoder — the port of upstream `src/flux3/autoencoder.py` @
//! [`gen_core_flux3::UPSTREAM_CODE_REVISION`]: an 8× KL VAE (GroupNorm(32, eps 1e-6) resnets,
//! single-head mid attention, asymmetric-pad stride-2 downsample, nearest-2× upsample) followed by
//! a **2×2 patchify** into `4·EMBED_DIM` packed token channels and a **BatchNorm-stats**
//! normalization of that packed space (`bn.running_mean/var`, eps 1e-4). 16× pixels per token.
//!
//! Runs entirely in **NHWC, f32** (upstream loads it `float32` and decodes outside autocast).
//! Every width is read from the tensors (`encoder.conv_in.weight` → CH, the `down.N.block.0.conv1`
//! out channels → CH_MULT, the `down.0.block.*` count → NUM_RES_BLOCKS, `quant_conv.weight` →
//! 2·EMBED_DIM), so a miniature fixture loads through the same code as the release.
//!
//! Weight keys are upstream's own (`encoder.down.0.block.1.nin_shortcut.weight`,
//! `decoder.mid.attn_1.proj_out.bias`, `bn.running_var`, …) — the released `ae.safetensors`
//! loads unchanged.
//!
//! Delegated leaf of sc-26234: this file, `../tools/dump_flux3_ae_golden.py` and
//! `tests/ae_parity.rs` are implemented together against the frozen source; the interface below
//! is the contract the pipeline and the memory contract build on.
//!
//! ## Upstream structure (what is ported, verbatim)
//!
//! * `ResnetBlock`: `conv2(swish(norm2(conv1(swish(norm1(x)))))) + (nin_shortcut(x) | x)` — the
//!   1×1 shortcut only where `in != out` (its key presence is the switch).
//! * `AttnBlock`: `x + proj_out(sdpa(q(norm(x)), k(..), v(..)))` with q/k/v/proj_out 1×1 convs,
//!   one head over the H·W positions, default scale `1/√C`.
//! * `Downsample`: `F.pad(x, (0, 1, 0, 1))` (right, bottom) then a 3×3 stride-2 pad-0 conv.
//!   `Upsample`: nearest ×2 then a 3×3 pad-1 conv.
//! * `Encoder`: `conv_in → down[l].block[*] (+ downsample for l < last) → mid.block_1 →
//!   mid.attn_1 → mid.block_2 → conv_out(swish(norm_out)) → quant_conv` (the 1×1 `quant_conv` is
//!   part of the encoder forward). `Decoder`: `conv_in(post_quant_conv(z)) → mid → up[l] for l in
//!   reversed(levels) (NUM_RES_BLOCKS + 1 blocks each, + upsample for l > 0) →
//!   conv_out(swish(norm_out))`.
//! * `AutoEncoder.encode`: the mean half of the encoder output, 2×2 patchified in NCHW as
//!   `reshape(1, E, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4)` (packed channel `c·4 + ph·2 + pw`),
//!   BatchNorm-normalized with the running statistics (`affine=False`), tokens in row-major
//!   `latent_ids` order. `decode`: the inverse — `z·√(var + eps) + mean`, then
//!   `reshape(1, E, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)`, then the decoder.
//!
//! [`gen_core_flux3::UPSTREAM_CODE_REVISION`]: gen_core_flux3::UPSTREAM_CODE_REVISION

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, divide, multiply, pad, sqrt, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::array::scalar;
use mlx_gen::nn::{conv2d, group_norm, linear, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Pixels per packed token on each side (`DOWNSAMPLING_RATIO`): 8× VAE × 2×2 patchify.
pub const DOWNSAMPLING_RATIO: i32 = gen_core_flux3::DOWNSAMPLING_RATIO as i32;
/// GroupNorm groups of every norm in the autoencoder.
pub const GN_GROUPS: i32 = 32;
/// GroupNorm eps.
pub const GN_EPS: f32 = 1e-6;
/// BatchNorm eps of the packed-latent normalization (`AutoEncoder.bn_eps`).
pub const BN_EPS: f32 = 1e-4;

/// Side of the patchify square (`reshape(.., h, 2, w, 2)`); `DOWNSAMPLING_RATIO / 8`.
const PATCH: i32 = 2;
/// Spatial downsampling of the VAE alone (three stride-2 `Downsample`s in the release geometry;
/// derived from the level count at load, this is only the documented release value).
const RELEASE_VAE_RATIO: i32 = 8;

fn layout_error(msg: impl Into<String>) -> Error {
    Error::Msg(format!("flux3 autoencoder: {}", msg.into()))
}

/// `[O, I, kH, kW]` (PyTorch) → `[O, kH, kW, I]` (mlx conv2d), cast to f32.
fn conv_w(w: &Weights, key: &str) -> Result<Array> {
    let t = w.require(key)?;
    if t.ndim() != 4 {
        return Err(layout_error(format!(
            "{key}: expected a rank-4 conv weight, got shape {:?}",
            t.shape()
        )));
    }
    Ok(t.transpose_axes(&[0, 2, 3, 1])?.as_dtype(Dtype::Float32)?)
}

fn f32w(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.as_dtype(Dtype::Float32)?)
}

/// A 3×3 (or stride-2) conv's `(weight [O, 3, 3, I], bias [O])`, f32.
fn conv(w: &Weights, name: &str) -> Result<(Array, Array)> {
    Ok((
        conv_w(w, &format!("{name}.weight"))?,
        f32w(w, &format!("{name}.bias"))?,
    ))
}

/// A 1×1 conv `[O, I, 1, 1]` expressed as a channel-wise Linear `[O, I]` (+ bias `[O]`), f32.
/// In NHWC a 1×1 conv is exactly a matmul over the channel axis.
fn squeeze_linear(w: &Weights, name: &str) -> Result<(Array, Array)> {
    let key = format!("{name}.weight");
    let cw = w.require(&key)?;
    let sh = cw.shape();
    if sh.len() != 4 || sh[2] != 1 || sh[3] != 1 {
        return Err(layout_error(format!(
            "{key}: expected a 1x1 conv weight [O, I, 1, 1], got shape {sh:?}"
        )));
    }
    Ok((
        cw.reshape(&[sh[0], sh[1]])?.as_dtype(Dtype::Float32)?,
        f32w(w, &format!("{name}.bias"))?,
    ))
}

/// Out channels of a stored `[O, I, kH, kW]` conv weight.
fn conv_out_channels(w: &Weights, key: &str) -> Result<i32> {
    let t = w.require(key)?;
    let sh = t.shape();
    if sh.len() != 4 {
        return Err(layout_error(format!(
            "{key}: expected a rank-4 conv weight, got shape {sh:?}"
        )));
    }
    Ok(sh[0])
}

/// Upstream `normalize(channels)`: GroupNorm(32, eps 1e-6, affine).
struct Norm {
    weight: Array,
    bias: Array,
}

impl Norm {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: f32w(w, &format!("{prefix}.weight"))?,
            bias: f32w(w, &format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        group_norm(x, &self.weight, &self.bias, GN_GROUPS, GN_EPS)
    }

    fn arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        out.push(&self.weight);
        out.push(&self.bias);
    }
}

/// Upstream `ResnetBlock`: `h = conv1(swish(norm1(x))); h = conv2(swish(norm2(h)));
/// x (= nin_shortcut(x) if in != out) + h`. `dropout` is `Dropout(0.0)` — identity.
struct ResnetBlock {
    norm1: Norm,
    conv1: (Array, Array),
    norm2: Norm,
    conv2: (Array, Array),
    /// The 1×1 `nin_shortcut`, present exactly when `in_channels != out_channels`.
    nin_shortcut: Option<(Array, Array)>,
}

impl ResnetBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let conv1 = conv(w, &format!("{prefix}.conv1"))?;
        let (out_ch, in_ch) = {
            let sh = conv1.0.shape();
            (sh[0], sh[3])
        };
        let shortcut_key = format!("{prefix}.nin_shortcut");
        let nin_shortcut = if in_ch != out_ch {
            Some(squeeze_linear(w, &shortcut_key)?)
        } else {
            if w.get(&format!("{shortcut_key}.weight")).is_some() {
                return Err(layout_error(format!(
                    "{shortcut_key}: present although in == out == {in_ch} channels"
                )));
            }
            None
        };
        Ok(Self {
            norm1: Norm::from_weights(w, &format!("{prefix}.norm1"))?,
            conv1,
            norm2: Norm::from_weights(w, &format!("{prefix}.norm2"))?,
            conv2: conv(w, &format!("{prefix}.conv2"))?,
            nin_shortcut,
        })
    }

    /// `x`: NHWC.
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.norm1.forward(x)?;
        let h = conv2d(&silu(&h)?, &self.conv1.0, Some(&self.conv1.1), 1, 1)?;
        let h = self.norm2.forward(&h)?;
        let h = conv2d(&silu(&h)?, &self.conv2.0, Some(&self.conv2.1), 1, 1)?;
        let res = match &self.nin_shortcut {
            Some((lw, lb)) => linear(x, lw, lb)?,
            None => x.clone(),
        };
        Ok(add(&res, &h)?)
    }

    fn arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        self.norm1.arrays(out);
        out.push(&self.conv1.0);
        out.push(&self.conv1.1);
        self.norm2.arrays(out);
        out.push(&self.conv2.0);
        out.push(&self.conv2.1);
        if let Some((lw, lb)) = &self.nin_shortcut {
            out.push(lw);
            out.push(lb);
        }
    }
}

/// Upstream `AttnBlock`: single-head self-attention over the H·W positions of `norm(x)`, with
/// q/k/v/proj_out as 1×1 convs (channel linears here), residual.
struct AttnBlock {
    norm: Norm,
    q: (Array, Array),
    k: (Array, Array),
    v: (Array, Array),
    proj_out: (Array, Array),
}

impl AttnBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: Norm::from_weights(w, &format!("{prefix}.norm"))?,
            q: squeeze_linear(w, &format!("{prefix}.q"))?,
            k: squeeze_linear(w, &format!("{prefix}.k"))?,
            v: squeeze_linear(w, &format!("{prefix}.v"))?,
            proj_out: squeeze_linear(w, &format!("{prefix}.proj_out"))?,
        })
    }

    /// `x`: NHWC `[B, H, W, C]`. Upstream flattens each of q/k/v to `[B, 1, H·W, C]`
    /// (`reshape(b, c, hw).permute(0, 2, 1).unsqueeze(1)`) and calls
    /// `F.scaled_dot_product_attention` with its default scale `1/√C`; the result goes back to
    /// `[B, C, H, W]` and through `proj_out`. In NHWC the flatten is a plain reshape.
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
        let y = self.norm.forward(x)?;
        let seq = |p: &(Array, Array)| -> Result<Array> {
            Ok(linear(&y, &p.0, &p.1)?.reshape(&[b, 1, h * w, c])?)
        };
        let q = seq(&self.q)?;
        let k = seq(&self.k)?;
        let v = seq(&self.v)?;
        let scale = 1.0 / (c as f32).sqrt();
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = linear(
            &o.reshape(&[b, h, w, c])?,
            &self.proj_out.0,
            &self.proj_out.1,
        )?;
        Ok(add(x, &o)?)
    }

    fn arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        self.norm.arrays(out);
        for (lw, lb) in [&self.q, &self.k, &self.v, &self.proj_out] {
            out.push(lw);
            out.push(lb);
        }
    }
}

/// Upstream `mid`: `block_2(attn_1(block_1(h)))`.
struct Mid {
    block_1: ResnetBlock,
    attn_1: AttnBlock,
    block_2: ResnetBlock,
}

impl Mid {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            block_1: ResnetBlock::from_weights(w, &format!("{prefix}.block_1"))?,
            attn_1: AttnBlock::from_weights(w, &format!("{prefix}.attn_1"))?,
            block_2: ResnetBlock::from_weights(w, &format!("{prefix}.block_2"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.block_1.forward(x)?;
        let x = self.attn_1.forward(&x)?;
        self.block_2.forward(&x)
    }

    fn arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        self.block_1.arrays(out);
        self.attn_1.arrays(out);
        self.block_2.arrays(out);
    }
}

/// Number of consecutive `{prefix}.{i}.conv1.weight` keys starting at 0 — the resnet count of
/// one level, read from the checkpoint. Zero is a layout error for every caller.
fn count_blocks(w: &Weights, prefix: &str) -> Result<i32> {
    let mut n = 0;
    while w.get(&format!("{prefix}.{n}.conv1.weight")).is_some() {
        n += 1;
    }
    if n == 0 {
        return Err(Error::MissingTensor(format!("{prefix}.0.conv1.weight")));
    }
    Ok(n)
}

/// Number of consecutive `{prefix}.{l}.block.0.conv1.weight` keys starting at 0 — the level count.
fn count_levels(w: &Weights, prefix: &str) -> Result<i32> {
    let mut n = 0;
    while w
        .get(&format!("{prefix}.{n}.block.0.conv1.weight"))
        .is_some()
    {
        n += 1;
    }
    if n == 0 {
        return Err(Error::MissingTensor(format!(
            "{prefix}.0.block.0.conv1.weight"
        )));
    }
    Ok(n)
}

/// One encoder level: `NUM_RES_BLOCKS` resnets, then (for every level but the last) the
/// asymmetric-pad stride-2 `Downsample`.
struct DownLevel {
    blocks: Vec<ResnetBlock>,
    downsample: Option<(Array, Array)>,
}

impl DownLevel {
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        if let Some((cw, cb)) = &self.downsample {
            // `F.pad(x, (0, 1, 0, 1))`: one zero column on the right, one zero row at the bottom.
            x = pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            x = conv2d(&x, cw, Some(cb), 2, 0)?;
        }
        Ok(x)
    }

    fn arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        for block in &self.blocks {
            block.arrays(out);
        }
        if let Some((cw, cb)) = &self.downsample {
            out.push(cw);
            out.push(cb);
        }
    }
}

/// One decoder level: `NUM_RES_BLOCKS + 1` resnets, then (for every level but 0) the nearest-2×
/// `Upsample`.
struct UpLevel {
    blocks: Vec<ResnetBlock>,
    upsample: Option<(Array, Array)>,
}

impl UpLevel {
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        if let Some((cw, cb)) = &self.upsample {
            x = conv2d(&upsample_nearest(&x, 2)?, cw, Some(cb), 1, 1)?;
        }
        Ok(x)
    }

    fn arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        for block in &self.blocks {
            block.arrays(out);
        }
        if let Some((cw, cb)) = &self.upsample {
            out.push(cw);
            out.push(cb);
        }
    }
}

/// Upstream `Encoder` (including its trailing 1×1 `quant_conv`).
struct Encoder {
    conv_in: (Array, Array),
    down: Vec<DownLevel>,
    mid: Mid,
    norm_out: Norm,
    conv_out: (Array, Array),
    quant_conv: (Array, Array),
}

impl Encoder {
    /// Returns the encoder and the geometry it read: `(num_levels, num_res_blocks)`.
    fn from_weights(w: &Weights) -> Result<(Self, i32, i32)> {
        let levels = count_levels(w, "encoder.down")?;
        let num_res_blocks = count_blocks(w, "encoder.down.0.block")?;
        let mut down = Vec::with_capacity(levels as usize);
        for l in 0..levels {
            let prefix = format!("encoder.down.{l}");
            let n = count_blocks(w, &format!("{prefix}.block"))?;
            if n != num_res_blocks {
                return Err(layout_error(format!(
                    "{prefix}.block has {n} resnets, level 0 has {num_res_blocks}"
                )));
            }
            let blocks = (0..n)
                .map(|j| ResnetBlock::from_weights(w, &format!("{prefix}.block.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let downsample = if l != levels - 1 {
                Some(conv(w, &format!("{prefix}.downsample.conv"))?)
            } else {
                if w.get(&format!("{prefix}.downsample.conv.weight")).is_some() {
                    return Err(layout_error(format!(
                        "{prefix}.downsample present on the last level"
                    )));
                }
                None
            };
            down.push(DownLevel { blocks, downsample });
        }
        let encoder = Self {
            conv_in: conv(w, "encoder.conv_in")?,
            down,
            mid: Mid::from_weights(w, "encoder.mid")?,
            norm_out: Norm::from_weights(w, "encoder.norm_out")?,
            conv_out: conv(w, "encoder.conv_out")?,
            quant_conv: squeeze_linear(w, "encoder.quant_conv")?,
        };
        Ok((encoder, levels, num_res_blocks))
    }

    /// `x`: `[1, H, W, 3]` → moments `[1, H/8, W/8, 2·E]` (mean half first, as `chunk(2, dim=1)`).
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = conv2d(x, &self.conv_in.0, Some(&self.conv_in.1), 1, 1)?;
        for level in &self.down {
            h = level.forward(&h)?;
        }
        let h = self.mid.forward(&h)?;
        let h = self.norm_out.forward(&h)?;
        let h = conv2d(&silu(&h)?, &self.conv_out.0, Some(&self.conv_out.1), 1, 1)?;
        linear(&h, &self.quant_conv.0, &self.quant_conv.1)
    }

    fn arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        out.push(&self.conv_in.0);
        out.push(&self.conv_in.1);
        for level in &self.down {
            level.arrays(out);
        }
        self.mid.arrays(out);
        self.norm_out.arrays(out);
        out.push(&self.conv_out.0);
        out.push(&self.conv_out.1);
        out.push(&self.quant_conv.0);
        out.push(&self.quant_conv.1);
    }
}

/// Upstream `Decoder` (including its leading 1×1 `post_quant_conv`).
struct Decoder {
    post_quant_conv: (Array, Array),
    conv_in: (Array, Array),
    mid: Mid,
    /// Indexed by level, as `decoder.up.{level}`; run in reverse level order.
    up: Vec<UpLevel>,
    norm_out: Norm,
    conv_out: (Array, Array),
}

impl Decoder {
    fn from_weights(w: &Weights, levels: i32, num_res_blocks: i32) -> Result<Self> {
        let mut up = Vec::with_capacity(levels as usize);
        for l in 0..levels {
            let prefix = format!("decoder.up.{l}");
            let n = count_blocks(w, &format!("{prefix}.block"))?;
            if n != num_res_blocks + 1 {
                return Err(layout_error(format!(
                    "{prefix}.block has {n} resnets, expected NUM_RES_BLOCKS + 1 = {}",
                    num_res_blocks + 1
                )));
            }
            let blocks = (0..n)
                .map(|j| ResnetBlock::from_weights(w, &format!("{prefix}.block.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let upsample = if l != 0 {
                Some(conv(w, &format!("{prefix}.upsample.conv"))?)
            } else {
                if w.get(&format!("{prefix}.upsample.conv.weight")).is_some() {
                    return Err(layout_error(format!(
                        "{prefix}.upsample present on level 0"
                    )));
                }
                None
            };
            up.push(UpLevel { blocks, upsample });
        }
        if w.get(&format!("decoder.up.{levels}.block.0.conv1.weight"))
            .is_some()
        {
            return Err(layout_error(format!(
                "decoder has more than the encoder's {levels} levels"
            )));
        }
        Ok(Self {
            post_quant_conv: squeeze_linear(w, "decoder.post_quant_conv")?,
            conv_in: conv(w, "decoder.conv_in")?,
            mid: Mid::from_weights(w, "decoder.mid")?,
            up,
            norm_out: Norm::from_weights(w, "decoder.norm_out")?,
            conv_out: conv(w, "decoder.conv_out")?,
        })
    }

    /// `z`: `[1, H/8, W/8, E]` (already de-normalized and unpatchified) → `[1, H, W, 3]`.
    fn forward(&self, z: &Array) -> Result<Array> {
        let z = linear(z, &self.post_quant_conv.0, &self.post_quant_conv.1)?;
        let mut h = conv2d(&z, &self.conv_in.0, Some(&self.conv_in.1), 1, 1)?;
        h = self.mid.forward(&h)?;
        for level in self.up.iter().rev() {
            h = level.forward(&h)?;
        }
        let h = self.norm_out.forward(&h)?;
        conv2d(&silu(&h)?, &self.conv_out.0, Some(&self.conv_out.1), 1, 1)
    }

    fn arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        out.push(&self.post_quant_conv.0);
        out.push(&self.post_quant_conv.1);
        out.push(&self.conv_in.0);
        out.push(&self.conv_in.1);
        self.mid.arrays(out);
        for level in &self.up {
            level.arrays(out);
        }
        self.norm_out.arrays(out);
        out.push(&self.conv_out.0);
        out.push(&self.conv_out.1);
    }
}

/// The loaded autoencoder (encoder + decoder + the packed-space BatchNorm statistics).
pub struct Flux3AutoEncoder {
    encoder: Encoder,
    decoder: Decoder,
    /// `bn.running_mean` `[4·E]`, in the packed-channel order.
    bn_mean: Array,
    /// `√(bn.running_var + BN_EPS)` `[4·E]`.
    bn_std: Array,
    /// `EMBED_DIM` (the latent channels of the VAE proper; a quarter of a packed token).
    embed_dim: i32,
    /// Total spatial downsampling of the VAE proper: `2^(levels − 1)` (8 in the release).
    vae_ratio: i32,
}

impl Flux3AutoEncoder {
    /// Build from the `ae.safetensors` tensors (upstream key names, any float dtype; every
    /// parameter is cast to f32). Every width is read from the tensors; a missing key is a
    /// [`Error::MissingTensor`], a shape the layout cannot explain a named error.
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let (encoder, levels, num_res_blocks) = Encoder::from_weights(w)?;
        let decoder = Decoder::from_weights(w, levels, num_res_blocks)?;

        // Geometry: CH from conv_in, 2·E from quant_conv / conv_out, 4·E from the bn stats.
        let ch = conv_out_channels(w, "encoder.conv_in.weight")?;
        if ch % GN_GROUPS != 0 {
            return Err(layout_error(format!(
                "encoder.conv_in has {ch} out channels, not a multiple of GroupNorm({GN_GROUPS})"
            )));
        }
        let two_e = encoder.quant_conv.0.shape()[0];
        if two_e % 2 != 0 || encoder.quant_conv.0.shape()[1] != two_e {
            return Err(layout_error(format!(
                "encoder.quant_conv.weight must be [2E, 2E, 1, 1], got [{two_e}, {}, 1, 1]",
                encoder.quant_conv.0.shape()[1]
            )));
        }
        if encoder.conv_out.0.shape()[0] != two_e {
            return Err(layout_error(format!(
                "encoder.conv_out has {} out channels, quant_conv expects {two_e}",
                encoder.conv_out.0.shape()[0]
            )));
        }
        let embed_dim = two_e / 2;
        let dec_in = decoder.post_quant_conv.0.shape();
        if dec_in[0] != embed_dim || dec_in[1] != embed_dim {
            return Err(layout_error(format!(
                "decoder.post_quant_conv.weight must be [E, E, 1, 1] with E = {embed_dim}, got \
                 [{}, {}, 1, 1]",
                dec_in[0], dec_in[1]
            )));
        }
        let latent_channels = embed_dim * PATCH * PATCH;

        let bn_mean = f32w(w, "bn.running_mean")?;
        let bn_var = f32w(w, "bn.running_var")?;
        if bn_mean.shape() != [latent_channels] || bn_var.shape() != [latent_channels] {
            return Err(layout_error(format!(
                "bn.running_mean/var must be [{latent_channels}] (4·EMBED_DIM), got {:?} / {:?}",
                bn_mean.shape(),
                bn_var.shape()
            )));
        }
        let bn_std = sqrt(&add(&bn_var, scalar(BN_EPS))?)?;

        // Every level but the last downsamples by 2; the decoder mirrors it.
        let vae_ratio = 1i32 << (levels - 1);
        if vae_ratio * PATCH != DOWNSAMPLING_RATIO {
            return Err(layout_error(format!(
                "{levels} levels give a {vae_ratio}x VAE ({}x with the {PATCH}x{PATCH} patchify), \
                 the contract is {DOWNSAMPLING_RATIO}x ({RELEASE_VAE_RATIO}x VAE)",
                vae_ratio * PATCH
            )));
        }

        Ok(Self {
            encoder,
            decoder,
            bn_mean,
            bn_std,
            embed_dim,
            vae_ratio,
        })
    }

    /// Channels of one packed token: `4 · EMBED_DIM` (128 in the release).
    pub fn latent_channels(&self) -> i32 {
        self.embed_dim * PATCH * PATCH
    }

    /// `AutoEncoder.encode`: `image` `[1, H, W, 3]` f32 in `[−1, 1]` (H, W multiples of
    /// [`DOWNSAMPLING_RATIO`]) → `(tokens [h·w, 4·EMBED_DIM] f32, (h, w))`, tokens in the
    /// row-major order of [`gen_core_flux3::latent_ids`]: the encoder's mean half, 2×2
    /// patchified (`reshape(1, E, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4)` — packed channel
    /// `c·4 + ph·2 + pw`), BatchNorm-normalized with the running statistics.
    ///
    /// [`gen_core_flux3::latent_ids`]: gen_core_flux3::latent_ids
    pub fn encode(&self, image: &Array) -> Result<(Array, (i32, i32))> {
        let sh = image.shape();
        if sh.len() != 4 || sh[0] != 1 || sh[3] != 3 {
            return Err(layout_error(format!(
                "encode expects an image [1, H, W, 3], got {sh:?}"
            )));
        }
        let (height, width) = (sh[1], sh[2]);
        if height <= 0
            || width <= 0
            || height % DOWNSAMPLING_RATIO != 0
            || width % DOWNSAMPLING_RATIO != 0
        {
            return Err(layout_error(format!(
                "width and height must be multiples of {DOWNSAMPLING_RATIO}, got {width}x{height}"
            )));
        }
        let (h, w) = (height / DOWNSAMPLING_RATIO, width / DOWNSAMPLING_RATIO);
        let e = self.embed_dim;

        let image = image.as_dtype(Dtype::Float32)?;
        let moments = self.encoder.forward(&image)?;
        let msh = moments.shape();
        if msh != [1, h * PATCH, w * PATCH, 2 * e] {
            return Err(layout_error(format!(
                "encoder produced {msh:?}, expected [1, {}, {}, {}]",
                h * PATCH,
                w * PATCH,
                2 * e
            )));
        }
        // `mean, _ = moments.chunk(2, dim=1)`: the first E channels.
        let idx = Array::from_slice(&(0..e).collect::<Vec<i32>>(), &[e]);
        let mean = moments.take_axis(&idx, 3)?;
        // Patchify. NCHW `[1, E, 2h, 2w] → reshape(1, E, h, 2, w, 2) → permute(0, 1, 3, 5, 2, 4)
        // → [1, E, 2, 2, h, w] → reshape(1, 4E, h, w)`; packed channel = c·4 + ph·2 + pw. In
        // NHWC: `[1, 2h, 2w, E] → [1, h, 2, w, 2, E]` (axes b, h, ph, w, pw, c) → (b, h, w, c, ph,
        // pw) → `[1, h, w, 4E]`.
        let packed = mean
            .reshape(&[1, h, PATCH, w, PATCH, e])?
            .transpose_axes(&[0, 1, 3, 5, 2, 4])?
            .reshape(&[1, h, w, self.latent_channels()])?;
        // BatchNorm2d (eval, affine=False): (z − running_mean) / √(running_var + eps).
        let normalized = divide(&subtract(&packed, &self.bn_mean)?, &self.bn_std)?;
        // `z[0].permute(1, 2, 0).reshape(h·w, C)`: row-major over (h, w) — the NHWC flatten.
        let tokens = normalized.reshape(&[h * w, self.latent_channels()])?;
        Ok((tokens, (h, w)))
    }

    /// `AutoEncoder.decode`: `tokens` `[h·w, 4·EMBED_DIM]` (any float dtype; cast to f32 as
    /// upstream's `tokens.float()`) in row-major id order → the decoder output `[1, H, W, 3]` f32,
    /// **unclamped** (the pipeline clamps to `[−1, 1]` when converting to RGB8). Denormalizes with
    /// `z · sqrt(running_var + eps) + running_mean`, then unpatchifies
    /// (`reshape(1, E, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)`) before `post_quant_conv`.
    pub fn decode(&self, tokens: &Array, h: i32, w: i32) -> Result<Array> {
        let c = self.latent_channels();
        let sh = tokens.shape();
        if h <= 0 || w <= 0 || sh != [h * w, c] {
            return Err(layout_error(format!(
                "decode expects tokens [{}, {c}] for a {h}x{w} grid, got {sh:?}",
                h.max(0) * w.max(0)
            )));
        }
        let e = self.embed_dim;
        // `tokens.float().reshape(h, w, C).permute(2, 0, 1).unsqueeze(0)` → NHWC `[1, h, w, C]`.
        let z = tokens.as_dtype(Dtype::Float32)?.reshape(&[1, h, w, c])?;
        // `z · √(running_var + eps) + running_mean` (per packed channel).
        let z = add(&multiply(&z, &self.bn_std)?, &self.bn_mean)?;
        // Unpatchify. NCHW `[1, 4E, h, w] → reshape(1, E, 2, 2, h, w) → permute(0, 1, 4, 2, 5, 3)
        // → [1, E, h, 2, w, 2] → reshape(1, E, 2h, 2w)`. In NHWC: `[1, h, w, 4E] → [1, h, w, E,
        // 2, 2]` (axes b, h, w, c, ph, pw) → (b, h, ph, w, pw, c) → `[1, 2h, 2w, E]`.
        let z = z
            .reshape(&[1, h, w, e, PATCH, PATCH])?
            .transpose_axes(&[0, 1, 4, 2, 5, 3])?
            .reshape(&[1, h * PATCH, w * PATCH, e])?;
        let image = self.decoder.forward(&z)?;
        let ish = image.shape();
        let (height, width) = (h * PATCH * self.vae_ratio, w * PATCH * self.vae_ratio);
        if ish != [1, height, width, 3] {
            return Err(layout_error(format!(
                "decoder produced {ish:?}, expected [1, {height}, {width}, 3]"
            )));
        }
        Ok(image)
    }

    /// Every parameter array, for the load-time evaluation sweep.
    pub fn arrays(&self) -> Vec<&Array> {
        let mut out = Vec::new();
        self.encoder.arrays(&mut out);
        self.decoder.arrays(&mut out);
        out.push(&self.bn_mean);
        out.push(&self.bn_std);
        out
    }
}

// Additive training opmath. Selected-dtype parameters/activations are real:
// only one operation's rounded operands are promoted, with one low output write.
fn training_pair_cast(p: &mut (Array, Array), dtype: Dtype) -> Result<()> {
    p.0 = p.0.as_dtype(dtype)?;
    p.1 = p.1.as_dtype(dtype)?;
    Ok(())
}

fn training_conv(x: &Array, p: &(Array, Array), stride: i32, padding: i32) -> Result<Array> {
    Ok(conv2d(
        &x.as_dtype(Dtype::Float32)?,
        &p.0.as_dtype(Dtype::Float32)?,
        Some(&p.1.as_dtype(Dtype::Float32)?),
        stride,
        padding,
    )?
    .as_dtype(x.dtype())?)
}

fn training_linear(x: &Array, p: &(Array, Array)) -> Result<Array> {
    Ok(linear(
        &x.as_dtype(Dtype::Float32)?,
        &p.0.as_dtype(Dtype::Float32)?,
        &p.1.as_dtype(Dtype::Float32)?,
    )?
    .as_dtype(x.dtype())?)
}

fn training_silu(x: &Array) -> Result<Array> {
    Ok(silu(&x.as_dtype(Dtype::Float32)?)?.as_dtype(x.dtype())?)
}

impl Norm {
    fn training_cast(&mut self, dtype: Dtype) -> Result<()> {
        self.weight = self.weight.as_dtype(dtype)?;
        self.bias = self.bias.as_dtype(dtype)?;
        Ok(())
    }
    fn training_forward(&self, x: &Array) -> Result<Array> {
        Ok(group_norm(
            &x.as_dtype(Dtype::Float32)?,
            &self.weight.as_dtype(Dtype::Float32)?,
            &self.bias.as_dtype(Dtype::Float32)?,
            GN_GROUPS,
            GN_EPS,
        )?
        .as_dtype(x.dtype())?)
    }
}

impl ResnetBlock {
    fn training_cast(&mut self, dtype: Dtype) -> Result<()> {
        self.norm1.training_cast(dtype)?;
        self.norm2.training_cast(dtype)?;
        training_pair_cast(&mut self.conv1, dtype)?;
        training_pair_cast(&mut self.conv2, dtype)?;
        if let Some(p) = &mut self.nin_shortcut {
            training_pair_cast(p, dtype)?;
        }
        Ok(())
    }
    fn training_forward(&self, x: &Array) -> Result<Array> {
        let h = training_conv(
            &training_silu(&self.norm1.training_forward(x)?)?,
            &self.conv1,
            1,
            1,
        )?;
        let h = training_conv(
            &training_silu(&self.norm2.training_forward(&h)?)?,
            &self.conv2,
            1,
            1,
        )?;
        let residual = match &self.nin_shortcut {
            Some(p) => training_linear(x, p)?,
            None => x.clone(),
        };
        Ok(add(&residual, &h)?)
    }
}

impl AttnBlock {
    fn training_cast(&mut self, dtype: Dtype) -> Result<()> {
        self.norm.training_cast(dtype)?;
        for p in [&mut self.q, &mut self.k, &mut self.v, &mut self.proj_out] {
            training_pair_cast(p, dtype)?;
        }
        Ok(())
    }
    fn training_forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
        let y = self.norm.training_forward(x)?;
        let seq = |p: &(Array, Array)| -> Result<Array> {
            Ok(training_linear(&y, p)?
                .as_dtype(Dtype::Float32)?
                .reshape(&[b, 1, h * w, c])?)
        };
        let (q, k, v) = (seq(&self.q)?, seq(&self.k)?, seq(&self.v)?);
        let o = scaled_dot_product_attention(&q, &k, &v, 1.0 / (c as f32).sqrt(), None, None)?
            .reshape(&[b, h, w, c])?
            .as_dtype(x.dtype())?;
        Ok(add(x, &training_linear(&o, &self.proj_out)?)?)
    }
}

impl Encoder {
    fn training_cast(&mut self, dtype: Dtype) -> Result<()> {
        for p in [&mut self.conv_in, &mut self.conv_out, &mut self.quant_conv] {
            training_pair_cast(p, dtype)?;
        }
        for level in &mut self.down {
            for block in &mut level.blocks {
                block.training_cast(dtype)?;
            }
            if let Some(p) = &mut level.downsample {
                training_pair_cast(p, dtype)?;
            }
        }
        self.mid.block_1.training_cast(dtype)?;
        self.mid.attn_1.training_cast(dtype)?;
        self.mid.block_2.training_cast(dtype)?;
        self.norm_out.training_cast(dtype)
    }
    fn training_forward(&self, x: &Array) -> Result<Array> {
        let mut h = training_conv(x, &self.conv_in, 1, 1)?;
        for level in &self.down {
            for block in &level.blocks {
                h = block.training_forward(&h)?;
            }
            if let Some(p) = &level.downsample {
                h = pad(&h, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
                h = training_conv(&h, p, 2, 0)?;
            }
        }
        h = self.mid.block_1.training_forward(&h)?;
        h = self.mid.attn_1.training_forward(&h)?;
        h = self.mid.block_2.training_forward(&h)?;
        h = training_conv(
            &training_silu(&self.norm_out.training_forward(&h)?)?,
            &self.conv_out,
            1,
            1,
        )?;
        training_linear(&h, &self.quant_conv)
    }
}

/// Frozen encoder-only training view. CPU-stream load/cast/evaluation ensures
/// selected precision exists before subsequent GPU evaluation (MLX unified memory;
/// no physical-upload claim). Retains selected weights, never a decoder or F32
/// encoder shadow. Every operation uses F32 opmath and a selected-dtype write.
pub struct FrozenTrainingEncoder {
    encoder: Encoder,
    bn_mean: Array,
    bn_std: Array,
    embed_dim: i32,
    compute: Dtype,
}

impl FrozenTrainingEncoder {
    pub fn from_weights(w: &Weights, compute: Dtype) -> Result<Self> {
        if !matches!(compute, Dtype::Float32 | Dtype::Bfloat16 | Dtype::Float16) {
            return Err(layout_error("training compute must be F32, BF16 or F16"));
        }
        mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
            // Original statistics captured and evaluated BEFORE half parameter casts.
            let bn_mean = f32w(w, "bn.running_mean")?;
            let bn_std = sqrt(&add(&f32w(w, "bn.running_var")?, scalar(BN_EPS))?)?;
            bn_mean.eval()?;
            bn_std.eval()?;
            let (mut encoder, levels, _) = Encoder::from_weights(w)?;
            let two_e = encoder.quant_conv.0.shape()[0];
            if levels != 4
                || two_e <= 0
                || two_e % 2 != 0
                || encoder.quant_conv.0.shape()[1] != two_e
                || encoder.conv_out.0.shape()[0] != two_e
            {
                return Err(layout_error(
                    "training encoder requires source 16x geometry and matching 2E moments",
                ));
            }
            let embed_dim = two_e / 2;
            let c = embed_dim * PATCH * PATCH;
            if bn_mean.shape() != [c] || bn_std.shape() != [c] {
                return Err(layout_error(
                    "training original BN statistics must have 4E channels",
                ));
            }
            encoder.training_cast(compute)?;
            let mut arrays = Vec::new();
            encoder.arrays(&mut arrays);
            for array in arrays {
                array.eval()?;
            }
            Ok(Self {
                encoder,
                bn_mean,
                bn_std,
                embed_dim,
                compute,
            })
        })
    }

    pub fn compute_dtype(&self) -> Dtype {
        self.compute
    }
    pub fn latent_channels(&self) -> i32 {
        self.embed_dim * PATCH * PATCH
    }

    /// Selected encoder parameters followed by the two ORIGINAL F32 BN arrays.
    /// Never includes decoder arrays or a retained F32 encoder shadow.
    pub fn arrays(&self) -> Vec<&Array> {
        let mut out = Vec::new();
        self.encoder.arrays(&mut out);
        out.push(&self.bn_mean);
        out.push(&self.bn_std);
        out
    }

    /// F32 NHWC input -> deterministic mean -> patchify -> original-F32 BN,
    /// already-normalized F32 tokens in row-major grid order. No second packing.
    pub fn encode(&self, image: &Array) -> Result<(Array, (i32, i32))> {
        let sh = image.shape();
        if image.dtype() != Dtype::Float32
            || sh.len() != 4
            || sh[0] != 1
            || sh[3] != 3
            || sh[1] <= 0
            || sh[2] <= 0
            || sh[1] % DOWNSAMPLING_RATIO != 0
            || sh[2] % DOWNSAMPLING_RATIO != 0
        {
            return Err(layout_error(
                "training input must be F32 [1,H,W,3], positive sides multiple16",
            ));
        }
        let (h, w) = (sh[1] / DOWNSAMPLING_RATIO, sh[2] / DOWNSAMPLING_RATIO);
        let image = mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || -> Result<Array> {
            let selected = image.as_dtype(self.compute)?;
            selected.eval()?;
            Ok(selected)
        })?;
        let moments = self.encoder.training_forward(&image)?;
        let e = self.embed_dim;
        if moments.shape() != [1, h * PATCH, w * PATCH, 2 * e] {
            return Err(layout_error(
                "training encoder produced unexpected moments geometry",
            ));
        }
        let idx = Array::from_slice(&(0..e).collect::<Vec<i32>>(), &[e]);
        let mean = moments.take_axis(&idx, 3)?.as_dtype(Dtype::Float32)?;
        let packed = mean
            .reshape(&[1, h, PATCH, w, PATCH, e])?
            .transpose_axes(&[0, 1, 3, 5, 2, 4])?
            .reshape(&[1, h, w, self.latent_channels()])?;
        let normalized = divide(&subtract(&packed, &self.bn_mean)?, &self.bn_std)?;
        Ok((
            normalized.reshape(&[h * w, self.latent_channels()])?,
            (h, w),
        ))
    }
}
