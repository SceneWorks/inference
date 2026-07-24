//! The CoD decoder — `_Decoder` plus its `ResnetBlock`/`AttnBlock` primitives
//! (`_vendor/mage_flow/models/modules/mage_vae.py:265-396`).
//!
//! It maps a `[B, 128, h, w]` latent to the `[B, 384, h, w]` conditioning the one-step denoiser
//! consumes. Despite the `up2x=True` in the reference docstring (`:377`) there is **no** upsampler
//! in the module: the output stays at latent resolution and the 16× expansion happens in the
//! denoiser's per-patch head.
//!
//! ```text
//! conv_in(128→384, 3×3) → [Resnet, Attn, Resnet, Attn, Resnet] → GroupNorm → SiLU → conv_out(3×3)
//! ```
//!
//! `self.ada` is `nn.Identity` (`:391`) and carries no weights, so it is not modelled.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, pad, PadMode};
use mlx_rs::Array;

use mlx_gen::nn::{silu, window_unpartition};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use super::layers::{Conv2d, LayerNormAffine};

/// `AttnBlock.patch_size` — attention is computed independently inside each 32×32 latent tile
/// (`mage_vae.py:293,384,386`).
pub const ATTN_PATCH: i32 = 32;

/// Number of `ResnetBlock`s in `_Decoder.block` (`mage_vae.py:382-388`).
pub const NUM_RESNETS: usize = 3;
/// Number of `AttnBlock`s in `_Decoder.block`.
pub const NUM_ATTNS: usize = 2;

/// `GroupNorm(32) → SiLU → Conv3×3` twice, plus a residual (`mage_vae.py:265-287`).
///
/// The decoder's three blocks are all `in_channels == out_channels == 384`, so `nin_shortcut`
/// never exists in the checkpoint and the residual is the identity. It is still modelled as an
/// `Option` because sc-14046's encoder path does hit the projecting case.
pub struct ResnetBlock {
    norm1: LayerNormAffine,
    conv1: Conv2d,
    norm2: LayerNormAffine,
    conv2: Conv2d,
    nin_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    /// Load one block under `{prefix}`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let nin_shortcut = if w.get(&format!("{prefix}.nin_shortcut.weight")).is_some() {
            Some(Conv2d::pointwise(
                w,
                &format!("{prefix}.nin_shortcut"),
                true,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: LayerNormAffine::from_weights(w, &format!("{prefix}.norm1"))?,
            conv1: Conv2d::conv3x3(w, &format!("{prefix}.conv1"))?,
            norm2: LayerNormAffine::from_weights(w, &format!("{prefix}.norm2"))?,
            conv2: Conv2d::conv3x3(w, &format!("{prefix}.conv2"))?,
            nin_shortcut,
        })
    }

    /// NHWC in, NHWC out.
    pub fn forward(&self, x_nhwc: &Array) -> Result<Array> {
        let h = self
            .conv1
            .forward(&silu(&self.norm1.group_norm(x_nhwc)?)?)?;
        let h = self.conv2.forward(&silu(&self.norm2.group_norm(&h)?)?)?;
        let residual = match &self.nin_shortcut {
            Some(c) => c.forward(x_nhwc)?,
            None => x_nhwc.clone(),
        };
        Ok(add(&residual, &h)?)
    }
}

/// Local self-attention over non-overlapping 32×32 latent tiles (`mage_vae.py:290-335`).
///
/// Two details a generic VAE-attention port would get wrong:
///
/// 1. **The softmax scale is `C^-0.5` over the full 384 channels**, not a per-head dimension
///    (`:330`) — this is single-head attention in the LDM `AttnBlock` idiom.
/// 2. **Ragged edges are `replicate`-padded, not zero-padded** (`:314-316`), and the padding is
///    then cropped back off (`:334`). This fires for any latent side that is not a multiple of 32
///    — including 256²-native images, whose 16×16 latent pads to a single 32×32 tile.
pub struct AttnBlock {
    norm: LayerNormAffine,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
    tile: i32,
}

impl AttnBlock {
    /// Load one block under `{prefix}`, attending within `tile`×`tile` latent tiles.
    pub fn from_weights(w: &Weights, prefix: &str, tile: i32) -> Result<Self> {
        if tile <= 0 {
            return Err(Error::Msg(format!(
                "mage-vae AttnBlock: tile must be > 0, got {tile}"
            )));
        }
        Ok(Self {
            norm: LayerNormAffine::from_weights(w, &format!("{prefix}.norm"))?,
            q: Conv2d::pointwise(w, &format!("{prefix}.q"), true)?,
            k: Conv2d::pointwise(w, &format!("{prefix}.k"), true)?,
            v: Conv2d::pointwise(w, &format!("{prefix}.v"), true)?,
            proj_out: Conv2d::pointwise(w, &format!("{prefix}.proj_out"), true)?,
            tile,
        })
    }

    /// NHWC in, NHWC out.
    pub fn forward(&self, x_nhwc: &Array) -> Result<Array> {
        let sh = x_nhwc.shape();
        let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);

        let normed = self.norm.group_norm(x_nhwc)?;
        let d = self.tile;
        let pad_h = (d - h % d) % d;
        let pad_w = (d - w % d) % d;

        let (hp, wp) = (h + pad_h, w + pad_w);

        // `window_partition` in core `nn` is the same reshape, but zero-pads; the reference
        // replicate-pads (`mage_vae.py:314-316`), so the tiling is spelled out here and only the
        // inverse (`window_unpartition`) is shared.
        let to_tiles = |conv: &Conv2d| -> Result<Array> {
            let t = conv.forward(&normed)?;
            let t = if pad_h != 0 || pad_w != 0 {
                pad(
                    &t,
                    &[(0, 0), (0, pad_h), (0, pad_w), (0, 0)][..],
                    None,
                    PadMode::Edge,
                )?
            } else {
                t
            };
            // [B,Hp,Wp,C] -> [B,nph,d,npw,d,C] -> [B,nph,npw,d,d,C] -> [B*np, 1, d*d, C]
            Ok(t.reshape(&[b, hp / d, d, wp / d, d, c])?
                .transpose_axes(&[0, 1, 3, 2, 4, 5])?
                .reshape(&[-1, 1, d * d, c])?)
        };

        let q = to_tiles(&self.q)?;
        let k = to_tiles(&self.k)?;
        let v = to_tiles(&self.v)?;

        // `c ** -0.5` over the CHANNEL count (`mage_vae.py:330`), single head.
        let scale = (c as f32).powf(-0.5);
        // Trailing `None` is the MLX >= 0.30 `sinks` argument (no attention sinks).
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;

        let o = window_unpartition(&o.reshape(&[-1, d, d, c])?, d, (hp, wp), (h, w))?;
        Ok(add(x_nhwc, &self.proj_out.forward(&o)?)?)
    }
}

/// `_Decoder` — latent → the denoiser's conditioning features (`mage_vae.py:376-396`).
pub struct CodDecoder {
    conv_in: Conv2d,
    resnets: [ResnetBlock; NUM_RESNETS],
    attns: [AttnBlock; NUM_ATTNS],
    norm_out: LayerNormAffine,
    conv_out: Conv2d,
}

impl CodDecoder {
    /// Load under `{prefix}` (`pipeline.y_embedder.decoder` in the published checkpoint).
    ///
    /// `block` is a `nn.Sequential` of `[Resnet, Attn, Resnet, Attn, Resnet]`, so the indices
    /// interleave: resnets at 0/2/4, attention at 1/3 (`mage_vae.py:382-388`).
    pub fn from_weights(w: &Weights, prefix: &str, tile: i32) -> Result<Self> {
        Ok(Self {
            conv_in: Conv2d::conv3x3(w, &format!("{prefix}.conv_in"))?,
            resnets: [
                ResnetBlock::from_weights(w, &format!("{prefix}.block.0"))?,
                ResnetBlock::from_weights(w, &format!("{prefix}.block.2"))?,
                ResnetBlock::from_weights(w, &format!("{prefix}.block.4"))?,
            ],
            attns: [
                AttnBlock::from_weights(w, &format!("{prefix}.block.1"), tile)?,
                AttnBlock::from_weights(w, &format!("{prefix}.block.3"), tile)?,
            ],
            norm_out: LayerNormAffine::from_weights(w, &format!("{prefix}.norm_out"))?,
            conv_out: Conv2d::conv3x3(w, &format!("{prefix}.conv_out"))?,
        })
    }

    /// `[B, h, w, 128]` latent → `[B, h, w, 384]` conditioning. NHWC in, NHWC out.
    pub fn forward(&self, z_nhwc: &Array) -> Result<Array> {
        let h = self.conv_in.forward(z_nhwc)?;
        let h = self.resnets[0].forward(&h)?;
        let h = self.attns[0].forward(&h)?;
        let h = self.resnets[1].forward(&h)?;
        let h = self.attns[1].forward(&h)?;
        let h = self.resnets[2].forward(&h)?;
        self.conv_out
            .forward(&silu(&self.norm_out.group_norm(&h)?)?)
    }
}
