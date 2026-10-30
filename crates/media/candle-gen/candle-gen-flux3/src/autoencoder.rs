//! The FLUX 3 image autoencoder — the port of upstream `src/flux3/autoencoder.py` @
//! [`gen_core_flux3::UPSTREAM_CODE_REVISION`], the Candle twin of
//! `mlx_gen_flux3::autoencoder`: an 8× KL VAE (GroupNorm(32, eps 1e-6) resnets, single-head mid
//! attention, asymmetric-pad stride-2 downsample, nearest-2× upsample) followed by a **2×2
//! patchify** into `4·EMBED_DIM` packed token channels and a **BatchNorm-stats** normalization of
//! that packed space (`bn.running_mean/var`, eps 1e-4). 16× pixels per token.
//!
//! Runs entirely in **NCHW, f32** (candle's native conv layout; upstream loads it `float32` and
//! decodes outside autocast). Every width is read from the tensors (`encoder.conv_in.weight` → CH,
//! the `down.N.block.0.conv1` out channels → CH_MULT, the `down.0.block.*` count → NUM_RES_BLOCKS,
//! `quant_conv.weight` → 2·EMBED_DIM), so a miniature fixture loads through the same code as the
//! release. Weight keys are upstream's own — the released `ae.safetensors` loads unchanged.
//!
//! Convolutions (including 1×1 shortcuts), upsampling, normalization, pointwise operations and
//! output assembly use bounded operation-level launches, including the 2^32-element 4096² stage.
//! GroupNorm retains full-image group statistics; this is not approximate spatial VAE tiling.
//! Mid attention uses the shared i32-overflow-safe SDPA.
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

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Conv2dConfig, VarBuilder};

mod bounded;
use bounded::{BoundedConv2d as Conv2d, BoundedGroupNorm as GroupNorm};
use candle_gen::{CandleError as Error, Result};

/// Pixels per packed token on each side (`DOWNSAMPLING_RATIO`): 8× VAE × 2×2 patchify.
pub const DOWNSAMPLING_RATIO: usize = gen_core_flux3::DOWNSAMPLING_RATIO as usize;
/// GroupNorm groups of every norm in the autoencoder.
pub const GN_GROUPS: usize = 32;
/// GroupNorm eps.
pub const GN_EPS: f64 = 1e-6;
/// BatchNorm eps of the packed-latent normalization (`AutoEncoder.bn_eps`).
pub const BN_EPS: f64 = 1e-4;

/// Side of the patchify square (`reshape(.., h, 2, w, 2)`); `DOWNSAMPLING_RATIO / 8`.
const PATCH: usize = 2;
/// Spatial downsampling of the VAE alone (three stride-2 `Downsample`s in the release geometry;
/// derived from the level count at load, this is only the documented release value).
const RELEASE_VAE_RATIO: usize = 8;

fn layout_error(msg: impl Into<String>) -> Error {
    Error::Msg(format!("flux3 autoencoder: {}", msg.into()))
}

// Keep the upload boundary injectable for CPU assertion tests: the callback sees
// rounded CPU operands, never a full-F32 parameter/input awaiting a device cast.
fn training_cast_before_upload(
    host: &Tensor,
    compute: DType,
    upload: impl FnOnce(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    if !host.device().is_cpu() {
        return Err(layout_error("training cast requires host operands"));
    }
    let rounded = host.to_dtype(compute)?;
    upload(&rounded)
}

/// The `ae.safetensors` mmap plus the key presence probe the optional modules switch on.
struct Loader<'a> {
    vb: &'a VarBuilder<'static>,
    /// Additive training-only load policy. Inference always leaves this None.
    training: Option<(DType, Device)>,
}

impl Loader<'_> {
    fn has(&self, key: &str) -> bool {
        self.vb.contains_tensor(key)
    }

    fn take(&self, key: &str) -> Result<Tensor> {
        if !self.vb.contains_tensor(key) {
            return Err(Error::Msg(format!("missing tensor: {key}")));
        }
        if let Some((compute, device)) = &self.training {
            // The training builder is CPU-backed: round BEFORE uploading a parameter,
            // never retain an F32 device map beside the selected-dtype representation.
            return training_cast_before_upload(
                &self.vb.get_unchecked(key)?,
                *compute,
                |rounded| Ok(rounded.to_device(device)?),
            );
        }
        Ok(self.vb.get_unchecked(key)?.to_dtype(DType::F32)?)
    }

    /// A 3×3 conv (`padding`, `stride`), f32 — `[O, I, 3, 3]` + `[O]`.
    fn conv(&self, name: &str, padding: usize, stride: usize) -> Result<Conv2d> {
        let weight = self.take(&format!("{name}.weight"))?;
        let dims = weight.dims().to_vec();
        if dims.len() != 4 || dims[2] != 3 || dims[3] != 3 {
            return Err(layout_error(format!(
                "{name}.weight: expected a 3x3 conv weight [O, I, 3, 3], got shape {dims:?}"
            )));
        }
        let bias = self.take(&format!("{name}.bias"))?;
        Ok(Conv2d::new(
            weight,
            Some(bias),
            Conv2dConfig {
                padding,
                stride,
                ..Default::default()
            },
        ))
    }

    /// A 1×1 conv `[O, I, 1, 1]` + `[O]`, f32.
    fn conv1x1(&self, name: &str) -> Result<Conv2d> {
        let weight = self.take(&format!("{name}.weight"))?;
        let dims = weight.dims().to_vec();
        if dims.len() != 4 || dims[2] != 1 || dims[3] != 1 {
            return Err(layout_error(format!(
                "{name}.weight: expected a 1x1 conv weight [O, I, 1, 1], got shape {dims:?}"
            )));
        }
        let bias = self.take(&format!("{name}.bias"))?;
        Ok(Conv2d::new(weight, Some(bias), Conv2dConfig::default()))
    }

    /// Upstream `normalize(channels)`: GroupNorm(32, eps 1e-6, affine).
    fn norm(&self, prefix: &str) -> Result<GroupNorm> {
        let weight = self.take(&format!("{prefix}.weight"))?;
        let bias = self.take(&format!("{prefix}.bias"))?;
        let channels = weight.dim(0)?;
        if channels % GN_GROUPS != 0 {
            return Err(layout_error(format!(
                "{prefix} normalizes {channels} channels, not a multiple of GroupNorm({GN_GROUPS})"
            )));
        }
        GroupNorm::new(weight, bias, channels, GN_GROUPS, GN_EPS)
    }
}

/// A 3×3 conv behind bounded local-padding, im2col, bias and output-copy launches.
fn conv3(x: &Tensor, conv: &Conv2d) -> Result<Tensor> {
    bounded::conv2d(x, conv)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    bounded::silu(x)
}

/// Out channels of a stored conv weight (`[O, I, kH, kW]`), by header.
fn conv_out_channels(l: &Loader<'_>, key: &str) -> Result<usize> {
    let t = l.take(key)?;
    if t.rank() != 4 {
        return Err(layout_error(format!(
            "{key}: expected a rank-4 conv weight, got shape {:?}",
            t.dims()
        )));
    }
    Ok(t.dim(0)?)
}

/// Upstream `ResnetBlock`: `h = conv1(swish(norm1(x))); h = conv2(swish(norm2(h)));
/// x (= nin_shortcut(x) if in != out) + h`. `dropout` is `Dropout(0.0)` — identity.
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    /// The 1×1 `nin_shortcut`, present exactly when `in_channels != out_channels`.
    nin_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn load(l: &Loader<'_>, prefix: &str) -> Result<Self> {
        let conv1 = l.conv(&format!("{prefix}.conv1"), 1, 1)?;
        let (out_ch, in_ch, _, _) = conv1.weight().dims4()?;
        let shortcut_key = format!("{prefix}.nin_shortcut");
        let nin_shortcut = if in_ch != out_ch {
            Some(l.conv1x1(&shortcut_key)?)
        } else {
            if l.has(&format!("{shortcut_key}.weight")) {
                return Err(layout_error(format!(
                    "{shortcut_key}: present although in == out == {in_ch} channels"
                )));
            }
            None
        };
        Ok(Self {
            norm1: l.norm(&format!("{prefix}.norm1"))?,
            conv1,
            norm2: l.norm(&format!("{prefix}.norm2"))?,
            conv2: l.conv(&format!("{prefix}.conv2"), 1, 1)?,
            nin_shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = conv3(&swish(&self.norm1.forward(x)?)?, &self.conv1)?;
        let h = conv3(&swish(&self.norm2.forward(&h)?)?, &self.conv2)?;
        let res = match &self.nin_shortcut {
            Some(sc) => bounded::conv2d(x, sc)?,
            None => x.clone(),
        };
        bounded::add(&res, &h)
    }
}

/// Upstream `AttnBlock`: single-head self-attention over the H·W positions of `norm(x)`, with
/// q/k/v/proj_out as 1×1 convs, residual.
struct AttnBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
}

impl AttnBlock {
    fn load(l: &Loader<'_>, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: l.norm(&format!("{prefix}.norm"))?,
            q: l.conv1x1(&format!("{prefix}.q"))?,
            k: l.conv1x1(&format!("{prefix}.k"))?,
            v: l.conv1x1(&format!("{prefix}.v"))?,
            proj_out: l.conv1x1(&format!("{prefix}.proj_out"))?,
        })
    }

    /// `x`: NCHW `[B, C, H, W]`. Upstream flattens each of q/k/v to `[B, 1, H·W, C]`
    /// (`reshape(b, c, hw).permute(0, 2, 1).unsqueeze(1)`) and calls
    /// `F.scaled_dot_product_attention` with its default scale `1/√C`; the result goes back to
    /// `[B, C, H, W]` and through `proj_out`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let y = self.norm.forward(x)?;
        let seq = |conv: &Conv2d| -> Result<Tensor> {
            Ok(bounded::conv2d(&y, conv)?
                .reshape((b, c, h * w))?
                .transpose(1, 2)?
                .unsqueeze(1)?
                .contiguous()?)
        };
        let q = seq(&self.q)?;
        let k = seq(&self.k)?;
        let v = seq(&self.v)?;
        let scale = 1.0 / (c as f64).sqrt();
        let o = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            None,
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?;
        let o = o
            .squeeze(1)?
            .transpose(1, 2)?
            .reshape((b, c, h, w))?
            .contiguous()?;
        bounded::add(x, &bounded::conv2d(&o, &self.proj_out)?)
    }
}

/// Upstream `mid`: `block_2(attn_1(block_1(h)))`.
struct Mid {
    block_1: ResnetBlock,
    attn_1: AttnBlock,
    block_2: ResnetBlock,
}

impl Mid {
    fn load(l: &Loader<'_>, prefix: &str) -> Result<Self> {
        Ok(Self {
            block_1: ResnetBlock::load(l, &format!("{prefix}.block_1"))?,
            attn_1: AttnBlock::load(l, &format!("{prefix}.attn_1"))?,
            block_2: ResnetBlock::load(l, &format!("{prefix}.block_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.block_1.forward(x)?;
        let x = self.attn_1.forward(&x)?;
        self.block_2.forward(&x)
    }
}

/// Number of consecutive `{prefix}.{i}.conv1.weight` keys starting at 0 — the resnet count of
/// one level, read from the checkpoint. Zero is a layout error for every caller.
fn count_blocks(l: &Loader<'_>, prefix: &str) -> Result<usize> {
    let mut n = 0;
    while l.has(&format!("{prefix}.{n}.conv1.weight")) {
        n += 1;
    }
    if n == 0 {
        return Err(Error::Msg(format!(
            "missing tensor: {prefix}.0.conv1.weight"
        )));
    }
    Ok(n)
}

/// Number of consecutive `{prefix}.{l}.block.0.conv1.weight` keys starting at 0 — the level count.
fn count_levels(l: &Loader<'_>, prefix: &str) -> Result<usize> {
    let mut n = 0;
    while l.has(&format!("{prefix}.{n}.block.0.conv1.weight")) {
        n += 1;
    }
    if n == 0 {
        return Err(Error::Msg(format!(
            "missing tensor: {prefix}.0.block.0.conv1.weight"
        )));
    }
    Ok(n)
}

/// One encoder level: `NUM_RES_BLOCKS` resnets, then (for every level but the last) the
/// asymmetric-pad stride-2 `Downsample`.
struct DownLevel {
    blocks: Vec<ResnetBlock>,
    downsample: Option<Conv2d>,
}

impl DownLevel {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        if let Some(conv) = &self.downsample {
            // `F.pad(x, (0, 1, 0, 1))`: one zero column on the right, one zero row at the bottom.
            x = bounded::downsample(&x, conv)?;
        }
        Ok(x)
    }
}

/// One decoder level: `NUM_RES_BLOCKS + 1` resnets, then (for every level but 0) the nearest-2×
/// `Upsample`.
struct UpLevel {
    blocks: Vec<ResnetBlock>,
    upsample: Option<Conv2d>,
}

impl UpLevel {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        if let Some(conv) = &self.upsample {
            let (_, _, h, w) = x.dims4()?;
            x = conv3(&bounded::nearest2d(&x, 2 * h, 2 * w)?, conv)?;
        }
        Ok(x)
    }
}

/// Upstream `Encoder` (including its trailing 1×1 `quant_conv`).
struct Encoder {
    conv_in: Conv2d,
    down: Vec<DownLevel>,
    mid: Mid,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    quant_conv: Conv2d,
}

impl Encoder {
    /// Returns the encoder and the geometry it read: `(num_levels, num_res_blocks)`.
    fn load(l: &Loader<'_>) -> Result<(Self, usize, usize)> {
        let levels = count_levels(l, "encoder.down")?;
        let num_res_blocks = count_blocks(l, "encoder.down.0.block")?;
        let mut down = Vec::with_capacity(levels);
        for level in 0..levels {
            let prefix = format!("encoder.down.{level}");
            let n = count_blocks(l, &format!("{prefix}.block"))?;
            if n != num_res_blocks {
                return Err(layout_error(format!(
                    "{prefix}.block has {n} resnets, level 0 has {num_res_blocks}"
                )));
            }
            let blocks = (0..n)
                .map(|j| ResnetBlock::load(l, &format!("{prefix}.block.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let downsample = if level != levels - 1 {
                Some(l.conv(&format!("{prefix}.downsample.conv"), 0, 2)?)
            } else {
                if l.has(&format!("{prefix}.downsample.conv.weight")) {
                    return Err(layout_error(format!(
                        "{prefix}.downsample present on the last level"
                    )));
                }
                None
            };
            down.push(DownLevel { blocks, downsample });
        }
        let encoder = Self {
            conv_in: l.conv("encoder.conv_in", 1, 1)?,
            down,
            mid: Mid::load(l, "encoder.mid")?,
            norm_out: l.norm("encoder.norm_out")?,
            conv_out: l.conv("encoder.conv_out", 1, 1)?,
            quant_conv: l.conv1x1("encoder.quant_conv")?,
        };
        Ok((encoder, levels, num_res_blocks))
    }

    /// `x`: `[1, 3, H, W]` → moments `[1, 2·E, H/8, W/8]` (mean half first, as `chunk(2, dim=1)`).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = conv3(x, &self.conv_in)?;
        for level in &self.down {
            h = level.forward(&h)?;
        }
        let h = self.mid.forward(&h)?;
        let h = conv3(&swish(&self.norm_out.forward(&h)?)?, &self.conv_out)?;
        bounded::conv2d(&h, &self.quant_conv)
    }
}

/// Upstream `Decoder` (including its leading 1×1 `post_quant_conv`).
struct Decoder {
    post_quant_conv: Conv2d,
    conv_in: Conv2d,
    mid: Mid,
    /// Indexed by level, as `decoder.up.{level}`; run in reverse level order.
    up: Vec<UpLevel>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    fn load(l: &Loader<'_>, levels: usize, num_res_blocks: usize) -> Result<Self> {
        let mut up = Vec::with_capacity(levels);
        for level in 0..levels {
            let prefix = format!("decoder.up.{level}");
            let n = count_blocks(l, &format!("{prefix}.block"))?;
            if n != num_res_blocks + 1 {
                return Err(layout_error(format!(
                    "{prefix}.block has {n} resnets, expected NUM_RES_BLOCKS + 1 = {}",
                    num_res_blocks + 1
                )));
            }
            let blocks = (0..n)
                .map(|j| ResnetBlock::load(l, &format!("{prefix}.block.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let upsample = if level != 0 {
                Some(l.conv(&format!("{prefix}.upsample.conv"), 1, 1)?)
            } else {
                if l.has(&format!("{prefix}.upsample.conv.weight")) {
                    return Err(layout_error(format!(
                        "{prefix}.upsample present on level 0"
                    )));
                }
                None
            };
            up.push(UpLevel { blocks, upsample });
        }
        if l.has(&format!("decoder.up.{levels}.block.0.conv1.weight")) {
            return Err(layout_error(format!(
                "decoder has more than the encoder's {levels} levels"
            )));
        }
        Ok(Self {
            post_quant_conv: l.conv1x1("decoder.post_quant_conv")?,
            conv_in: l.conv("decoder.conv_in", 1, 1)?,
            mid: Mid::load(l, "decoder.mid")?,
            up,
            norm_out: l.norm("decoder.norm_out")?,
            conv_out: l.conv("decoder.conv_out", 1, 1)?,
        })
    }

    /// `z`: `[1, E, H/8, W/8]` (already de-normalized and unpatchified) → `[1, 3, H, W]`.
    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let z = bounded::conv2d(z, &self.post_quant_conv)?;
        let mut h = conv3(&z, &self.conv_in)?;
        h = self.mid.forward(&h)?;
        for level in self.up.iter().rev() {
            h = level.forward(&h)?;
        }
        conv3(&swish(&self.norm_out.forward(&h)?)?, &self.conv_out)
    }
}

/// The loaded autoencoder (encoder + decoder + the packed-space BatchNorm statistics).
pub struct Flux3AutoEncoder {
    encoder: Encoder,
    decoder: Decoder,
    /// `bn.running_mean` `[1, 4·E, 1, 1]`, in the packed-channel order.
    bn_mean: Tensor,
    /// `√(bn.running_var + BN_EPS)` `[1, 4·E, 1, 1]`.
    bn_std: Tensor,
    /// `EMBED_DIM` (the latent channels of the VAE proper; a quarter of a packed token).
    embed_dim: usize,
    /// Total spatial downsampling of the VAE proper: `2^(levels − 1)` (8 in the release).
    vae_ratio: usize,
    device: Device,
}

impl Flux3AutoEncoder {
    /// Build from the `ae.safetensors` at `path` (upstream key names, any float dtype; every
    /// parameter is cast to f32) onto `device`. Every width is read from the tensors; a missing
    /// key is an error naming it (`missing tensor: <key>`), a shape the layout cannot explain a
    /// named error.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        if !path.is_file() {
            return Err(Error::Msg(format!(
                "flux3: the autoencoder file {} does not exist",
                path.display()
            )));
        }
        let vb = candle_gen::mmap_var_builder(&[path.to_path_buf()], DType::F32, device)
            .map_err(|e| Error::Msg(format!("flux3: loading {}: {e}", path.display())))?;
        Self::from_var_builder(&vb, device)
    }

    /// [`Self::load`] over an already-mapped checkpoint.
    pub fn from_var_builder(vb: &VarBuilder<'static>, device: &Device) -> Result<Self> {
        let l = Loader { vb, training: None };
        let (encoder, levels, num_res_blocks) = Encoder::load(&l)?;
        let decoder = Decoder::load(&l, levels, num_res_blocks)?;

        // Geometry: CH from conv_in, 2·E from quant_conv / conv_out, 4·E from the bn stats.
        let ch = conv_out_channels(&l, "encoder.conv_in.weight")?;
        if ch % GN_GROUPS != 0 {
            return Err(layout_error(format!(
                "encoder.conv_in has {ch} out channels, not a multiple of GroupNorm({GN_GROUPS})"
            )));
        }
        let (two_e, qin, _, _) = encoder.quant_conv.weight().dims4()?;
        if two_e % 2 != 0 || qin != two_e {
            return Err(layout_error(format!(
                "encoder.quant_conv.weight must be [2E, 2E, 1, 1], got [{two_e}, {qin}, 1, 1]"
            )));
        }
        let conv_out_ch = encoder.conv_out.weight().dim(0)?;
        if conv_out_ch != two_e {
            return Err(layout_error(format!(
                "encoder.conv_out has {conv_out_ch} out channels, quant_conv expects {two_e}"
            )));
        }
        let embed_dim = two_e / 2;
        let (dec_out, dec_in, _, _) = decoder.post_quant_conv.weight().dims4()?;
        if dec_out != embed_dim || dec_in != embed_dim {
            return Err(layout_error(format!(
                "decoder.post_quant_conv.weight must be [E, E, 1, 1] with E = {embed_dim}, got \
                 [{dec_out}, {dec_in}, 1, 1]"
            )));
        }
        let latent_channels = embed_dim * PATCH * PATCH;

        let bn_mean = l.take("bn.running_mean")?;
        let bn_var = l.take("bn.running_var")?;
        if bn_mean.dims() != [latent_channels] || bn_var.dims() != [latent_channels] {
            return Err(layout_error(format!(
                "bn.running_mean/var must be [{latent_channels}] (4·EMBED_DIM), got {:?} / {:?}",
                bn_mean.dims(),
                bn_var.dims()
            )));
        }
        let bn_std = (bn_var + BN_EPS)?.sqrt()?;

        // Every level but the last downsamples by 2; the decoder mirrors it.
        let vae_ratio = 1usize << (levels - 1);
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
            bn_mean: bn_mean.reshape((1, latent_channels, 1, 1))?,
            bn_std: bn_std.reshape((1, latent_channels, 1, 1))?,
            embed_dim,
            vae_ratio,
            device: device.clone(),
        })
    }

    /// Channels of one packed token: `4 · EMBED_DIM` (128 in the release).
    pub fn latent_channels(&self) -> usize {
        self.embed_dim * PATCH * PATCH
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `AutoEncoder.encode`: `image` `[1, 3, H, W]` f32 in `[−1, 1]` (H, W multiples of
    /// [`DOWNSAMPLING_RATIO`]) → `(tokens [h·w, 4·EMBED_DIM] f32, (h, w))`, tokens in the
    /// row-major order of [`gen_core_flux3::latent_ids`]: the encoder's mean half, 2×2
    /// patchified (`reshape(1, E, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4)` — packed channel
    /// `c·4 + ph·2 + pw`), BatchNorm-normalized with the running statistics.
    ///
    /// [`gen_core_flux3::latent_ids`]: gen_core_flux3::latent_ids
    pub fn encode(&self, image: &Tensor) -> Result<(Tensor, (usize, usize))> {
        let dims = image.dims().to_vec();
        if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
            return Err(layout_error(format!(
                "encode expects an image [1, 3, H, W], got {dims:?}"
            )));
        }
        let (height, width) = (dims[2], dims[3]);
        if height == 0
            || width == 0
            || height % DOWNSAMPLING_RATIO != 0
            || width % DOWNSAMPLING_RATIO != 0
        {
            return Err(layout_error(format!(
                "width and height must be multiples of {DOWNSAMPLING_RATIO}, got {width}x{height}"
            )));
        }
        let (h, w) = (height / DOWNSAMPLING_RATIO, width / DOWNSAMPLING_RATIO);
        let e = self.embed_dim;
        let c = self.latent_channels();

        let image = image.to_dtype(DType::F32)?.to_device(&self.device)?;
        let moments = __model_proof_output!(self.encoder.forward(&image))?;
        let mdims = moments.dims().to_vec();
        if mdims != [1, 2 * e, h * PATCH, w * PATCH] {
            return Err(layout_error(format!(
                "encoder produced {mdims:?}, expected [1, {}, {}, {}]",
                2 * e,
                h * PATCH,
                w * PATCH
            )));
        }
        // `mean, _ = moments.chunk(2, dim=1)`: the first E channels.
        let mean = moments.narrow(1, 0, e)?;
        // Patchify: `[1, E, 2h, 2w] → reshape(1, E, h, 2, w, 2) → permute(0, 1, 3, 5, 2, 4) →
        // [1, E, 2, 2, h, w] → reshape(1, 4E, h, w)`; packed channel = c·4 + ph·2 + pw.
        let packed = mean
            .reshape((1, e, h, PATCH, w, PATCH))?
            .permute((0, 1, 3, 5, 2, 4))?
            .contiguous()?
            .reshape((1, c, h, w))?;
        // BatchNorm2d (eval, affine=False): (z − running_mean) / √(running_var + eps).
        let normalized = packed
            .broadcast_sub(&self.bn_mean)?
            .broadcast_div(&self.bn_std)?;
        // `z[0].permute(1, 2, 0).reshape(h·w, C)`: row-major over (h, w).
        let tokens = normalized
            .squeeze(0)?
            .permute((1, 2, 0))?
            .contiguous()?
            .reshape((h * w, c))?;
        Ok((tokens, (h, w)))
    }

    /// `AutoEncoder.decode`: `tokens` `[h·w, 4·EMBED_DIM]` (any float dtype; cast to f32 as
    /// upstream's `tokens.float()`) in row-major id order → the decoder output `[1, 3, H, W]` f32,
    /// **unclamped** (the pipeline clamps to `[−1, 1]` when converting to RGB8). Denormalizes with
    /// `z · sqrt(running_var + eps) + running_mean`, then unpatchifies
    /// (`reshape(1, E, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)`) before `post_quant_conv`.
    pub fn decode(&self, tokens: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let c = self.latent_channels();
        let dims = tokens.dims().to_vec();
        if h == 0 || w == 0 || dims != [h * w, c] {
            return Err(layout_error(format!(
                "decode expects tokens [{}, {c}] for a {h}x{w} grid, got {dims:?}",
                h * w
            )));
        }
        let e = self.embed_dim;
        // `tokens.float().reshape(h, w, C).permute(2, 0, 1).unsqueeze(0)` → `[1, C, h, w]`.
        let z = tokens
            .to_dtype(DType::F32)?
            .to_device(&self.device)?
            .reshape((h, w, c))?
            .permute((2, 0, 1))?
            .contiguous()?
            .unsqueeze(0)?;
        // `z · √(running_var + eps) + running_mean` (per packed channel).
        let z = z
            .broadcast_mul(&self.bn_std)?
            .broadcast_add(&self.bn_mean)?;
        // Unpatchify: `[1, 4E, h, w] → reshape(1, E, 2, 2, h, w) → permute(0, 1, 4, 2, 5, 3) →
        // [1, E, h, 2, w, 2] → reshape(1, E, 2h, 2w)`.
        let z = z
            .reshape((1, e, PATCH, PATCH, h, w))?
            .permute((0, 1, 4, 2, 5, 3))?
            .contiguous()?
            .reshape((1, e, h * PATCH, w * PATCH))?;
        let image = __model_proof_output!(self.decoder.forward(&z))?;
        let idims = image.dims().to_vec();
        let (height, width) = (h * PATCH * self.vae_ratio, w * PATCH * self.vae_ratio);
        if idims != [1, 3, height, width] {
            return Err(layout_error(format!(
                "decoder produced {idims:?}, expected [1, 3, {height}, {width}]"
            )));
        }
        Ok(image)
    }
}

// Additive training forwards: frozen HF training uses fused SiLU and selected
// precision; ordinary BFL inference forwards above remain unchanged.
fn training_swish(x: &Tensor) -> Result<Tensor> {
    let out = bounded::silu(&bounded::training_cast(x, DType::F32)?)?;
    bounded::training_cast(&out, x.dtype())
}

impl ResnetBlock {
    fn training_forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = training_swish(&self.norm1.training_forward_f32_opmath(x)?)?;
        let h = bounded::training_conv_f32_opmath(&h, &self.conv1, false)?;
        let h = training_swish(&self.norm2.training_forward_f32_opmath(&h)?)?;
        let h = bounded::training_conv_f32_opmath(&h, &self.conv2, false)?;
        let res = match &self.nin_shortcut {
            Some(conv) => bounded::training_conv_f32_opmath(x, conv, false)?,
            None => x.clone(),
        };
        bounded::add(&res, &h)
    }
}

impl AttnBlock {
    fn training_forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let y = self.norm.training_forward_f32_opmath(x)?;
        let seq = |conv: &Conv2d| -> Result<Tensor> {
            // SDPA accumulates scores/softmax/value products in F32 and writes
            // selected dtype once; no rounded intermediate score cache.
            Ok(bounded::training_conv_f32_opmath(&y, conv, false)?
                .to_dtype(DType::F32)?
                .reshape((b, c, h * w))?
                .transpose(1, 2)?
                .unsqueeze(1)?
                .contiguous()?)
        };
        let (q, k, v) = (seq(&self.q)?, seq(&self.k)?, seq(&self.v)?);
        let out = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            1.0 / (c as f64).sqrt(),
            None,
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?;
        let out = out
            .squeeze(1)?
            .transpose(1, 2)?
            .reshape((b, c, h, w))?
            .contiguous()?;
        let out = bounded::training_cast(&out, x.dtype())?;
        bounded::add(
            x,
            &bounded::training_conv_f32_opmath(&out, &self.proj_out, false)?,
        )
    }
}

impl Encoder {
    fn training_forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = bounded::training_conv_f32_opmath(x, &self.conv_in, false)?;
        for level in &self.down {
            for block in &level.blocks {
                h = block.training_forward(&h)?;
            }
            if let Some(conv) = &level.downsample {
                h = bounded::training_conv_f32_opmath(&h, conv, true)?;
            }
        }
        h = self.mid.block_1.training_forward(&h)?;
        h = self.mid.attn_1.training_forward(&h)?;
        h = self.mid.block_2.training_forward(&h)?;
        let h = training_swish(&self.norm_out.training_forward_f32_opmath(&h)?)?;
        let h = bounded::training_conv_f32_opmath(&h, &self.conv_out, false)?;
        bounded::training_conv_f32_opmath(&h, &self.quant_conv, false)
    }
}

/// Encoder-only frozen training AE; selected-dtype retained parameters and
/// operation writes, with original F32 BN statistics. Convolution, GroupNorm
/// and SDPA use explicit F32 opmath. CPU half convolution is EMULATION, not a
/// native BF16/CUDA-kernel claim. No decoder is loaded or uploaded.
pub struct FrozenTrainingEncoder {
    encoder: Encoder,
    bn_mean: Tensor,
    bn_std: Tensor,
    embed_dim: usize,
    compute: DType,
    device: Device,
}

impl FrozenTrainingEncoder {
    pub fn load(path: &Path, compute: DType, device: &Device) -> Result<Self> {
        let vb = candle_gen::mmap_var_builder(&[path.to_path_buf()], DType::F32, &Device::Cpu)?;
        Self::from_var_builder(&vb, compute, device)
    }

    /// A CPU builder is mandatory: selected parameters round BEFORE upload,
    /// without a retained full-F32 device map beside the selected representation.
    pub fn from_var_builder(
        vb: &VarBuilder<'static>,
        compute: DType,
        device: &Device,
    ) -> Result<Self> {
        if !matches!(compute, DType::F32 | DType::BF16 | DType::F16) {
            return Err(layout_error("training compute must be F32, BF16 or F16"));
        }
        if !vb.device().is_cpu() {
            return Err(layout_error(
                "training weights must be CPU-backed before upload",
            ));
        }
        let original = Loader { vb, training: None };
        // Capture ORIGINAL F32 stats before constructing the half encoder.
        let mean = original.take("bn.running_mean")?;
        let var = original.take("bn.running_var")?;
        let std = (var + BN_EPS)?.sqrt()?;
        let l = Loader {
            vb,
            training: Some((compute, device.clone())),
        };
        let (encoder, levels, _) = Encoder::load(&l)?;
        let (two_e, qin, _, _) = encoder.quant_conv.weight().dims4()?;
        if two_e == 0
            || two_e % 2 != 0
            || qin != two_e
            || encoder.conv_out.weight().dim(0)? != two_e
        {
            return Err(layout_error(
                "training encoder requires matching [2E,2E,1,1] quant_conv",
            ));
        }
        if levels != 4 {
            return Err(layout_error(
                "training encoder requires source four-level packed geometry",
            ));
        }
        let embed_dim = two_e / 2;
        let channels = embed_dim * PATCH * PATCH;
        if mean.dims() != [channels] || std.dims() != [channels] {
            return Err(layout_error(
                "training original BN statistics must have 4E channels",
            ));
        }
        Ok(Self {
            encoder,
            bn_mean: mean.reshape((1, channels, 1, 1))?.to_device(device)?,
            bn_std: std.reshape((1, channels, 1, 1))?.to_device(device)?,
            embed_dim,
            compute,
            device: device.clone(),
        })
    }

    pub fn compute_dtype(&self) -> DType {
        self.compute
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn latent_channels(&self) -> usize {
        self.embed_dim * PATCH * PATCH
    }

    /// HOST F32 NCHW input -> deterministic mean -> packed -> original-F32-BN
    /// normalized F32 row-major tokens. Never patchify or normalize again.
    pub fn encode(&self, image: &Tensor) -> Result<(Tensor, (usize, usize))> {
        let dims = image.dims();
        if !image.device().is_cpu() || image.dtype() != DType::F32 {
            return Err(layout_error(
                "training input must be host CPU F32 before upload",
            ));
        }
        if dims.len() != 4
            || dims[0] != 1
            || dims[1] != 3
            || dims[2] == 0
            || dims[3] == 0
            || !dims[2].is_multiple_of(DOWNSAMPLING_RATIO)
            || !dims[3].is_multiple_of(DOWNSAMPLING_RATIO)
        {
            return Err(layout_error(
                "training encode requires [1,3,H,W], positive sides multiple16",
            ));
        }
        let (h, w) = (dims[2] / DOWNSAMPLING_RATIO, dims[3] / DOWNSAMPLING_RATIO);
        let e = self.embed_dim;
        let c = self.latent_channels();
        let image = training_cast_before_upload(image, self.compute, |rounded| {
            Ok(rounded.to_device(&self.device)?)
        })?;
        let moments = __model_proof_output!(self.encoder.training_forward(&image))?;
        if moments.dims() != [1, 2 * e, h * PATCH, w * PATCH] {
            return Err(layout_error(
                "training encoder moments differ from expected geometry",
            ));
        }
        let mean = moments.narrow(1, 0, e)?.to_dtype(DType::F32)?;
        let packed = mean
            .reshape((1, e, h, PATCH, w, PATCH))?
            .permute((0, 1, 3, 5, 2, 4))?
            .contiguous()?
            .reshape((1, c, h, w))?;
        let normalized = packed
            .broadcast_sub(&self.bn_mean)?
            .broadcast_div(&self.bn_std)?;
        let tokens = normalized
            .squeeze(0)?
            .permute((1, 2, 0))?
            .contiguous()?
            .reshape((h * w, c))?;
        Ok((tokens, (h, w)))
    }
}

#[cfg(test)]
mod training_precision_tests {
    use super::*;

    #[test]
    fn training_silu_has_one_half_write_not_bfl_split_rounding() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../mlx-gen/mlx-gen-flux3/tests/fixtures/flux3_ae_training_precision.safetensors",
        );
        let g = candle_gen::candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
        let values = |x: &Tensor| {
            x.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let got = training_swish(&g["silu_input"]).unwrap();
        assert_eq!(got.dtype(), DType::BF16);
        assert_eq!(values(&got), values(&g["silu_fused"]));
        assert_ne!(values(&got), values(&g["silu_wrong_split"]));
    }

    #[test]
    fn selected_cast_is_observed_before_upload_callback() {
        let host = Tensor::from_vec(vec![1.001f32, -0.3333], 2, &Device::Cpu).unwrap();
        for compute in [DType::F32, DType::BF16, DType::F16] {
            let expected = host
                .to_dtype(compute)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut called = false;
            let out = training_cast_before_upload(&host, compute, |at_upload| {
                called = true;
                assert!(at_upload.device().is_cpu());
                assert_eq!(at_upload.dtype(), compute);
                assert_eq!(at_upload.to_dtype(DType::F32)?.to_vec1::<f32>()?, expected);
                Ok(at_upload.clone())
            })
            .unwrap();
            assert!(called);
            assert_eq!(out.dtype(), compute);
        }
    }

    #[test]
    fn retained_encoder_weights_and_original_bn_are_distinct_precisions() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-flux3/tests/fixtures/tiny-snapshot/ae.safetensors");
        let mut tensors = candle_gen::candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
        // Construct from an inventory that has NO decoder at all.
        tensors.retain(|name, _| name.starts_with("encoder.") || name.starts_with("bn."));
        let mean = tensors["bn.running_mean"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let ae = FrozenTrainingEncoder::from_var_builder(&vb, DType::BF16, &Device::Cpu).unwrap();
        assert_eq!(ae.encoder.conv_in.weight().dtype(), DType::BF16);
        assert_eq!(ae.encoder.quant_conv.weight().dtype(), DType::BF16);
        assert_eq!(ae.encoder.mid.attn_1.q.weight().dtype(), DType::BF16);
        for level in &ae.encoder.down {
            for block in &level.blocks {
                assert_eq!(block.conv1.weight().dtype(), DType::BF16);
                assert_eq!(block.conv2.weight().dtype(), DType::BF16);
                if let Some(sc) = &block.nin_shortcut {
                    assert_eq!(sc.weight().dtype(), DType::BF16);
                }
            }
            if let Some(conv) = &level.downsample {
                assert_eq!(conv.weight().dtype(), DType::BF16);
            }
        }
        assert_eq!(ae.bn_mean.dtype(), DType::F32);
        assert_eq!(ae.bn_std.dtype(), DType::F32);
        assert_eq!(
            ae.bn_mean.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            mean
        );
        let rounded = ae
            .bn_mean
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_ne!(
            rounded.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            mean
        );
    }
}
