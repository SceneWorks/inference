//! `_DConvDenoiser` — the one-step pixel head (`_vendor/mage_flow/models/modules/mage_vae.py:100-263,455-513`).
//!
//! Given the CoD conditioning `cond` at latent resolution and a `[B, 3, H, W]` noise tensor, it
//! produces the decoded image in a single forward. `decode` always passes **zero** noise
//! (`mage_vae.py:631`), which is preserved explicitly here rather than specialised away — the
//! zeros are constructed and concatenated exactly where the reference's `proj1(x)` and
//! `unfold(x)` would land, so the data flow stays comparable line-for-line with the reference.
//!
//! ```text
//! s = s_embedder(noise, cond)                 → [B, h, w, 384]
//! s = DiCoBlock ×21 (s, c)                    → [B, h, w, 384]      (adaLN folded at t = 0)
//! x = unfold(noise, 16) ‖ y_embedder_x(cond)  → [B·L, 256, 35]
//! x = x_embedder(x)                           → [B·L, 256, 32]      (per-patch DCT, max_freqs 8)
//! x = dec_net(x, s)                           → [B·L, 256, 32]      (SimpleMLPAdaLN ×3)
//! x = final_layer(x)                          → [B·L, 256, 3]
//! fold → [B, 3, H, W]
//! ```
//!
//! ## The two channel orderings are different, and swapping them is silent
//!
//! `y_embedder_x` is a `Conv2d(384 → 32·16²)` whose 8192 output channels are read back as
//! `(feature, position)` — **feature-major** (`mage_vae.py:503-504`: `flatten(2)` then
//! `reshape(b, -1, 256, L)`). `dec_net.cond_embed` is a `Linear(384 → 16²·32)` whose 8192 outputs
//! are read back as `(position, feature)` — **position-major** (`:239`). Swapping either is caught
//! by `tests/vae_decode_fixture.rs::tiny_decode_matches_the_torch_reference` — verified by
//! mutation, not assumed: reading the `y_embedder_x` output position-major instead moves the
//! tiny decode by max_abs **0.176** against a 2e-3 bound.

use mlx_rs::ops::{add, concatenate_axis, multiply, split, zeros_dtype};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{modulate, silu};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use super::dico::DiCoBlock;
use super::layers::{layer_norm_2d, Conv2d, LayerNormAffine, Linear, RmsNorm};
use super::timestep::TimestepEmbedder;

/// The codec's architectural dimensions.
///
/// Threaded rather than hard-coded so the same code runs the published checkpoint and the tiny
/// `tests/fixtures/mage_vae_tiny.safetensors` model, per the workspace's dimension-parametric
/// module convention. [`MageVaeShape::PUBLISHED`] is the only shape any production path uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MageVaeShape {
    /// `patch_size` (`mage_vae.py:457`) — also the codec's spatial downsample factor.
    pub patch: i32,
    /// `hidden_size` of the conditioning stream (`mage_vae.py:460`).
    pub hidden: i32,
    /// `hidden_size_x` of the per-pixel stream (`mage_vae.py:461`).
    pub hidden_x: i32,
    /// `in_channels` — RGB (`mage_vae.py:459`).
    pub in_channels: i32,
    /// `bottleneck_dim` — the latent channel count (`mage_vae.py:465`).
    pub bottleneck: i32,
    /// `num_cond_blocks` — the DiCo stack over the conditioning stream (`mage_vae.py:464`).
    pub num_cond_blocks: usize,
    /// `num_blocks - num_cond_blocks` — the `SimpleMLPAdaLN` residual blocks
    /// (`mage_vae.py:463,485`).
    pub num_mlp_blocks: usize,
    /// `NerfEmbedder.max_freqs` (`mage_vae.py:475`).
    pub max_freqs: usize,
    /// `AttnBlock.patch_size` — the CoD decoder's attention tile (`mage_vae.py:384,386`).
    pub attn_tile: i32,
}

impl MageVaeShape {
    /// The shape of the published `vae/diffusion_pytorch_model.safetensors`.
    pub const PUBLISHED: Self = Self {
        patch: 16,
        hidden: 384,
        hidden_x: 32,
        in_channels: 3,
        bottleneck: 128,
        num_cond_blocks: 21,
        num_mlp_blocks: 3,
        max_freqs: 8,
        attn_tile: super::cod_decoder::ATTN_PATCH,
    };
}

/// `BottleneckPatchEmbed` (`mage_vae.py:100-109`): patchify the image and fuse it with `cond`.
///
/// `proj1` is **unbiased** (`:105`), so on the zero-noise decode path its output is exactly zero.
struct BottleneckPatchEmbed {
    proj1: Conv2d,
    proj2: Conv2d,
}

impl BottleneckPatchEmbed {
    fn from_weights(w: &Weights, prefix: &str, patch: i32) -> Result<Self> {
        Ok(Self {
            proj1: Conv2d::from_weights(w, &format!("{prefix}.proj1"), patch, 0, 1, false)?,
            proj2: Conv2d::pointwise(w, &format!("{prefix}.proj2"), true)?,
        })
    }

    /// `x` NHWC `[B, H, W, 3]`, `cond` NHWC `[B, h, w, 384]` → `[B, h, w, 384]`.
    fn forward(&self, x_nhwc: &Array, cond_nhwc: &Array) -> Result<Array> {
        let patched = self.proj1.forward(x_nhwc)?; // [B, h, w, 128]
        self.proj2
            .forward(&concatenate_axis(&[patched, cond_nhwc.clone()], -1)?)
    }
}

/// `NerfEmbedder` (`mage_vae.py:180-208`): concatenate a fixed per-patch DCT position code onto the
/// per-pixel features and project.
struct NerfEmbedder {
    embedder: Linear,
}

impl NerfEmbedder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            embedder: Linear::from_weights(w, &format!("{prefix}.embedder.0"))?,
        })
    }

    /// `[.., P², in]` → `[.., P², 32]`, with the DCT table appended along the last axis.
    fn forward(&self, x: &Array, dct: &Array) -> Result<Array> {
        let lead = x.shape()[0];
        let dct = mlx_rs::ops::broadcast_to(dct, &[lead, dct.shape()[1], dct.shape()[2]])?;
        self.embedder
            .forward(&concatenate_axis(&[x.clone(), dct], -1)?)
    }
}

/// The fixed `fetch_pos` DCT table (`mage_vae.py:190-202`), `[1, P², max_freqs²]`.
///
/// Two easy-to-miss details, both pinned against the reference by
/// `tests/vae_decode_fixture.rs::dct_position_table_matches_the_torch_reference` (at the tiny
/// *and* the published `patch = 16` geometry), with
/// `an_integer_frequency_ramp_would_not_match` proving the check discriminates:
///
/// 1. `freqs = linspace(0, max_freqs, max_freqs)` is **8 points spanning 0…8 inclusive**, i.e. a
///    step of `8/7` — *not* `arange(8)`. A `0..7` integer ramp is the natural misreading and
///    changes every entry but the first row/column.
/// 2. `pos_x` (the within-patch **column**) pairs with the **first** frequency axis and `pos_y`
///    (the row) with the second, so the flattened index is `fx·max_freqs + fy`.
///
/// The reference builds this in the model dtype — bf16 for the published checkpoint
/// (`mage_vae.py:191,207`). It is built in f32 here: the table is a fixed positional code, not a
/// trained tensor, so there is no rounding to reproduce, and the extra precision only moves the
/// result toward the infinite-precision answer. (Contrast the DiT's timestep frequency table,
/// which *is* deliberately bf16-rounded — [`crate::config::TIMESTEP_FREQS_BF16`].)
pub fn dct_position_table(patch: i32, max_freqs: usize) -> Result<Array> {
    let p = patch as usize;
    let denom = (p - 1) as f32;
    let pos: Vec<f32> = (0..p).map(|i| i as f32 / denom).collect();
    let fstep = max_freqs as f32 / (max_freqs - 1) as f32;
    let freqs: Vec<f32> = (0..max_freqs).map(|i| i as f32 * fstep).collect();

    let mut data = Vec::with_capacity(p * p * max_freqs * max_freqs);
    for i in 0..p {
        for j in 0..p {
            for &fx in freqs.iter() {
                for &fy in freqs.iter() {
                    let coeff = 1.0 / (1.0 + fx * fy);
                    let dx = (pos[j] * fx * std::f32::consts::PI).cos();
                    let dy = (pos[i] * fy * std::f32::consts::PI).cos();
                    data.push(dx * dy * coeff);
                }
            }
        }
    }
    Ok(Array::from_slice(
        &data,
        &[1, (p * p) as i32, (max_freqs * max_freqs) as i32],
    ))
}

/// `_MLPResBlock` (`mage_vae.py:245-262`).
///
/// Its adaLN is conditioned on a **per-position** latent, not on `t`, so it is deliberately *not*
/// constant-folded (`mage_vae.py:356-358`).
struct MlpResBlock {
    in_ln: LayerNormAffine,
    fc1: Linear,
    fc2: Linear,
    adaln: Linear,
}

impl MlpResBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            in_ln: LayerNormAffine::from_weights(w, &format!("{prefix}.in_ln"))?,
            fc1: Linear::from_weights(w, &format!("{prefix}.mlp.0"))?,
            fc2: Linear::from_weights(w, &format!("{prefix}.mlp.2"))?,
            adaln: Linear::from_weights(w, &format!("{prefix}.adaLN_modulation.1"))?,
        })
    }

    fn forward(&self, x: &Array, y: &Array) -> Result<Array> {
        let parts = split(&self.adaln.forward(&silu(y)?)?, 3, -1)?;
        if parts.len() != 3 {
            return Err(Error::Msg(format!(
                "mage-vae dec_net adaLN: expected 3 chunks, got {}",
                parts.len()
            )));
        }
        let (shift, scale, gate) = (&parts[0], &parts[1], &parts[2]);
        // `in_ln` normalises over the last axis, so `layer_norm_2d` applies unchanged here.
        let h = modulate(&layer_norm_2d(x, Some(&self.in_ln))?, scale, shift, true)?;
        let h = self.fc2.forward(&silu(&self.fc1.forward(&h)?)?)?;
        Ok(add(x, &multiply(gate, &h)?)?)
    }
}

/// `SimpleMLPAdaLN` (`mage_vae.py:221-242`).
struct SimpleMlpAdaLn {
    cond_embed: Linear,
    input_proj: Linear,
    res_blocks: Vec<MlpResBlock>,
    patch: i32,
}

impl SimpleMlpAdaLn {
    fn from_weights(w: &Weights, prefix: &str, shape: MageVaeShape) -> Result<Self> {
        let res_blocks = (0..shape.num_mlp_blocks)
            .map(|i| MlpResBlock::from_weights(w, &format!("{prefix}.res_blocks.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cond_embed: Linear::from_weights(w, &format!("{prefix}.cond_embed"))?,
            input_proj: Linear::from_weights(w, &format!("{prefix}.input_proj"))?,
            res_blocks,
            patch: shape.patch,
        })
    }

    /// `x` `[B·L, P², 32]`, `c` `[B·L, 384]` → `[B·L, P², 32]`.
    fn forward(&self, x: &Array, c: &Array) -> Result<Array> {
        let mut h = self.input_proj.forward(x)?;
        // POSITION-major: `reshape(rows, P², -1)` (`mage_vae.py:239`).
        let c =
            self.cond_embed
                .forward(c)?
                .reshape(&[c.shape()[0], self.patch * self.patch, -1])?;
        for block in &self.res_blocks {
            h = block.forward(&h, &c)?;
        }
        Ok(h)
    }
}

/// `NerfFinalLayer` (`mage_vae.py:211-218`): `RMSNorm → Linear(32 → 3)`.
struct NerfFinalLayer {
    norm: RmsNorm,
    linear: Linear,
}

impl NerfFinalLayer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::from_weights(w, &format!("{prefix}.norm"))?,
            linear: Linear::from_weights(w, &format!("{prefix}.linear"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.linear.forward(&self.norm.forward(x)?)
    }
}

/// `_DConvDenoiser` (`mage_vae.py:455-513`) — everything below `y_embedder.decoder`.
pub struct DConvDenoiser {
    t_embedder: TimestepEmbedder,
    s_embedder: BottleneckPatchEmbed,
    blocks: Vec<DiCoBlock>,
    y_embedder_x: Conv2d,
    x_embedder: NerfEmbedder,
    dec_net: SimpleMlpAdaLn,
    final_layer: NerfFinalLayer,
    dct: Array,
    shape: MageVaeShape,
}

impl DConvDenoiser {
    /// Load under `{prefix}` (`pipeline` in the published checkpoint).
    pub fn from_weights(w: &Weights, prefix: &str, shape: MageVaeShape) -> Result<Self> {
        let blocks = (0..shape.num_cond_blocks)
            .map(|i| DiCoBlock::from_weights(w, &format!("{prefix}.blocks.{i}"), shape.hidden))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            t_embedder: TimestepEmbedder::from_weights(w, &format!("{prefix}.t_embedder"))?,
            s_embedder: BottleneckPatchEmbed::from_weights(
                w,
                &format!("{prefix}.s_embedder"),
                shape.patch,
            )?,
            blocks,
            y_embedder_x: Conv2d::pointwise(w, &format!("{prefix}.y_embedder_x"), true)?,
            x_embedder: NerfEmbedder::from_weights(w, &format!("{prefix}.x_embedder"))?,
            dec_net: SimpleMlpAdaLn::from_weights(w, &format!("{prefix}.dec_net"), shape)?,
            final_layer: NerfFinalLayer::from_weights(w, &format!("{prefix}.final_layer"))?,
            dct: dct_position_table(shape.patch, shape.max_freqs)?,
            shape,
        })
    }

    /// The conditioning vector this denoiser's DiCo stack is modulated by at timestep `t`.
    pub fn conditioning(&self, t: &Array, dtype: Dtype) -> Result<Array> {
        self.t_embedder.forward(t, dtype)
    }

    /// Constant-fold every DiCo block's adaLN at `t = 0` and free the MLPs
    /// (`mage_vae.py:643-651`).
    pub fn fold_adaln_at_zero(&mut self, dtype: Dtype) -> Result<()> {
        let t = zeros_dtype(&[1], dtype)?;
        let c = self.conditioning(&t, dtype)?;
        for block in &mut self.blocks {
            block.fold_adaln(&c)?;
        }
        Ok(())
    }

    /// Whether every DiCo block has had its adaLN folded.
    pub fn is_folded(&self) -> bool {
        self.blocks.iter().all(DiCoBlock::is_folded)
    }

    /// Each DiCo block's packed `[B, 6·hidden]` adaLN modulation at conditioning `c`, in block
    /// order — see [`DiCoBlock::adaln_packed`].
    pub fn adaln_packed(&self, c: &Array) -> Result<Vec<Array>> {
        self.blocks.iter().map(|b| b.adaln_packed(c)).collect()
    }

    /// `noise` NHWC `[B, H, W, 3]`, `cond` NHWC `[B, h, w, 384]`, `t` `[B]` → NHWC `[B, H, W, 3]`.
    pub fn forward(&self, noise_nhwc: &Array, t: &Array, cond_nhwc: &Array) -> Result<Array> {
        let sh = noise_nhwc.shape();
        let (b, height, width) = (sh[0], sh[1], sh[2]);
        let dtype = noise_nhwc.dtype();
        let (patch, hidden, hidden_x, in_ch) = (
            self.shape.patch,
            self.shape.hidden,
            self.shape.hidden_x,
            self.shape.in_channels,
        );
        let (hl, wl) = (height / patch, width / patch);
        let length = hl * wl;
        let p2 = patch * patch;

        let c = self.conditioning(t, dtype)?;

        // --- conditioning stream ------------------------------------------------------------
        let mut s = self.s_embedder.forward(noise_nhwc, cond_nhwc)?;
        for block in &self.blocks {
            s = block.forward(&s, &c)?;
        }
        // [B, h, w, 384] -> [B·L, 384]; NHWC is already the reference's `permute(0,2,3,1)`
        // (`mage_vae.py:500`), and the row-major (h, w) flatten matches `unfold`'s column order.
        let s = s.reshape(&[b * length, hidden])?;

        // --- per-pixel stream ---------------------------------------------------------------
        // `unfold(noise, patch, stride=patch)` (`mage_vae.py:502`) — the exact inverse of the fold
        // at the end of this function, since kernel == stride means no overlap accumulation.
        //
        // `decode` always passes zero noise, so this block is identically zero in production. It
        // is computed for real rather than substituted with `zeros` anyway: this is a `pub fn`
        // taking `noise_nhwc`, and silently ignoring the argument would give any future caller
        // (an encode round-trip, or a multi-step use of the denoiser) a half-wrong answer with
        // nothing to notice.
        let unfolded = noise_nhwc
            .reshape(&[b, hl, patch, wl, patch, in_ch])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[b * length, p2, in_ch])?;

        // FEATURE-major: `[B, h, w, 32·P²]` -> `[B, h, w, 32, P²]` -> `[B·L, P², 32]`
        // (`mage_vae.py:503-504`).
        let y = self
            .y_embedder_x
            .forward(cond_nhwc)?
            .reshape(&[b, hl, wl, hidden_x, p2])?
            .transpose_axes(&[0, 1, 2, 4, 3])?
            .reshape(&[b * length, p2, hidden_x])?;

        let x = concatenate_axis(&[unfolded, y], -1)?; // [B·L, P², 35]
        let x = self.x_embedder.forward(&x, &self.dct)?; // [B·L, P², 32]
        let x = self.dec_net.forward(&x, &s)?;
        let x = self.final_layer.forward(&x)?; // [B·L, P², 3]

        // `fold` with kernel == stride is the exact inverse of `unfold` (no overlap accumulation):
        // [B·L, P², 3] -> [B, h, w, P, P, 3] -> [B, h, P, w, P, 3] -> [B, H, W, 3].
        Ok(x.reshape(&[b, hl, wl, patch, patch, in_ch])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[b, height, width, in_ch])?)
    }
}
