//! The FLUX 3 image autoencoder — the port of upstream `src/flux3/autoencoder.py` @
//! [`gen_core::flux3::UPSTREAM_CODE_REVISION`]: an 8× KL VAE (GroupNorm(32, eps 1e-6) resnets,
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

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::Array;

/// Pixels per packed token on each side (`DOWNSAMPLING_RATIO`): 8× VAE × 2×2 patchify.
pub const DOWNSAMPLING_RATIO: i32 = mlx_gen::gen_core::flux3::DOWNSAMPLING_RATIO as i32;
/// GroupNorm groups of every norm in the autoencoder.
pub const GN_GROUPS: i32 = 32;
/// GroupNorm eps.
pub const GN_EPS: f32 = 1e-6;
/// BatchNorm eps of the packed-latent normalization (`AutoEncoder.bn_eps`).
pub const BN_EPS: f32 = 1e-4;

/// The loaded autoencoder (encoder + decoder + the packed-space BatchNorm statistics).
pub struct Flux3AutoEncoder {
    _private: (),
}

impl Flux3AutoEncoder {
    /// Build from the `ae.safetensors` tensors (upstream key names, any float dtype; every
    /// parameter is cast to f32). Every width is read from the tensors; a missing key is a
    /// [`Error::MissingTensor`], a shape the layout cannot explain a named error.
    pub fn from_weights(_w: &Weights) -> Result<Self> {
        Err(Error::Unsupported(
            "flux3: the autoencoder port is the delegated sc-26234 leaf and has not landed yet"
                .into(),
        ))
    }

    /// Channels of one packed token: `4 · EMBED_DIM` (128 in the release).
    pub fn latent_channels(&self) -> i32 {
        0
    }

    /// `AutoEncoder.encode`: `image` `[1, H, W, 3]` f32 in `[−1, 1]` (H, W multiples of
    /// [`DOWNSAMPLING_RATIO`]) → `(tokens [h·w, 4·EMBED_DIM] f32, (h, w))`, tokens in the
    /// row-major order of [`gen_core::flux3::latent_ids`]: the encoder's mean half, 2×2
    /// patchified (`reshape(1, E, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4)` — packed channel
    /// `c·4 + ph·2 + pw`), BatchNorm-normalized with the running statistics.
    pub fn encode(&self, _image: &Array) -> Result<(Array, (i32, i32))> {
        Err(Error::Unsupported(
            "flux3: the autoencoder port is the delegated sc-26234 leaf and has not landed yet"
                .into(),
        ))
    }

    /// `AutoEncoder.decode`: `tokens` `[h·w, 4·EMBED_DIM]` (any float dtype; cast to f32 as
    /// upstream's `tokens.float()`) in row-major id order → the decoder output `[1, H, W, 3]` f32,
    /// **unclamped** (the pipeline clamps to `[−1, 1]` when converting to RGB8). Denormalizes with
    /// `z · sqrt(running_var + eps) + running_mean`, then unpatchifies
    /// (`reshape(1, E, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)`) before `post_quant_conv`.
    pub fn decode(&self, _tokens: &Array, _h: i32, _w: i32) -> Result<Array> {
        Err(Error::Unsupported(
            "flux3: the autoencoder port is the delegated sc-26234 leaf and has not landed yet"
                .into(),
        ))
    }

    /// Every parameter array, for the load-time evaluation sweep.
    pub fn arrays(&self) -> Vec<&Array> {
        Vec::new()
    }
}
