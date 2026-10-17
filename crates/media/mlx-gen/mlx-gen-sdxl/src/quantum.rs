//! SDXL iOS execution quanta (sc-26270): the labels and the scratch geometry every guarded SDXL
//! quantum is priced with.
//!
//! Every helper reads its geometry off the LOADED tensors (conv weight shapes, projection base
//! shapes) and the request's actual activation shapes, never off production constants. The guards
//! themselves are `mlx_gen::ios_quantum::{run, before, finish}`, which are pass-throughs whenever
//! [`mlx_gen::ios_quantum::enabled`] is false (macOS, Linux, the simulator), so nothing here can
//! change host numerics or ordering.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::ios_quantum::{
    attention_block_bytes, conv_stage_bytes, transformer_layer_bytes, AttentionBlockGeometry,
};
use mlx_rs::Array;

/// U-Net stem: Kolors context projection, timestep/`text_time` embedding and `conv_in`.
pub(crate) const UNET_STEM: &str = "SDXL UNet stem";
/// One U-Net / ControlNet `ResnetBlock2D` (including its up-path skip concatenation).
pub(crate) const UNET_RESNET: &str = "SDXL UNet resnet";
/// A U-Net / ControlNet stride-2 downsample convolution.
pub(crate) const UNET_DOWNSAMPLE: &str = "SDXL UNet downsample";
/// A U-Net nearest-2x upsample plus its convolution.
pub(crate) const UNET_UPSAMPLE: &str = "SDXL UNet upsample";
/// A `Transformer2D`'s GroupNorm + `proj_in`.
pub(crate) const TRANSFORMER_PROJ_IN: &str = "SDXL transformer proj_in";
/// One spatial-transformer block (self-attn, cross-attn [+ IP branch], GEGLU FFN).
pub(crate) const TRANSFORMER_BLOCK: &str = "SDXL transformer block";
/// A `Transformer2D`'s `proj_out` + residual.
pub(crate) const TRANSFORMER_PROJ_OUT: &str = "SDXL transformer proj_out";
/// The ControlNet down residuals added into the U-Net skip stack.
pub(crate) const CONTROL_DOWN_INJECTION: &str = "SDXL ControlNet down injection";
/// The ControlNet mid residual added into the U-Net mid output.
pub(crate) const CONTROL_MID_INJECTION: &str = "SDXL ControlNet mid injection";
/// U-Net output GroupNorm + SiLU + `conv_out`.
pub(crate) const UNET_CONV_OUT: &str = "SDXL UNet conv_out";
/// ControlNet stem: context projection, embeddings, `conv_in` + conditioning embedding.
pub(crate) const CONTROL_STEM: &str = "SDXL ControlNet stem";
/// One ControlNet 1x1 zero-convolution head (down or mid), scaled.
pub(crate) const CONTROL_ZERO_CONV: &str = "SDXL ControlNet zero-conv";
/// One convolution of the ControlNet conditioning embedding.
pub(crate) const CONTROL_COND_CONV: &str = "SDXL ControlNet cond embedding";
/// CLIP text token + position embedding.
pub(crate) const CLIP_EMBEDDINGS: &str = "SDXL CLIP embeddings";
/// One CLIP text encoder layer.
pub(crate) const CLIP_LAYER: &str = "SDXL CLIP layer";
/// CLIP text final LayerNorm + pooled (projected) EOS token.
pub(crate) const CLIP_POOLED: &str = "SDXL CLIP final norm";
/// CLIP vision patch/class/position embedding + `pre_layrnorm`.
pub(crate) const VISION_EMBEDDINGS: &str = "SDXL CLIP vision embeddings";
/// One CLIP vision encoder layer.
pub(crate) const VISION_LAYER: &str = "SDXL CLIP vision layer";
/// IP-Adapter Resampler `proj_in`.
pub(crate) const RESAMPLER_PROJ_IN: &str = "SDXL IP resampler proj_in";
/// One IP-Adapter Resampler (perceiver attention + FFN) layer.
pub(crate) const RESAMPLER_LAYER: &str = "SDXL IP resampler layer";
/// IP-Adapter Resampler `proj_out` + `norm_out`.
pub(crate) const RESAMPLER_PROJ_OUT: &str = "SDXL IP resampler proj_out";
/// VAE decode de-normalization, `post_quant_conv` and decoder `conv_in`.
pub(crate) const VAE_DECODE_IN: &str = "SDXL VAE decode conv_in";
/// VAE encoder `conv_in`.
pub(crate) const VAE_ENCODE_IN: &str = "SDXL VAE encode conv_in";
/// One VAE (temb-free) resnet.
pub(crate) const VAE_RESNET: &str = "SDXL VAE resnet";
/// The VAE mid-block single-head attention.
pub(crate) const VAE_ATTENTION: &str = "SDXL VAE attention";
/// A VAE nearest-2x upsample plus its convolution.
pub(crate) const VAE_UPSAMPLE: &str = "SDXL VAE upsample";
/// A VAE asymmetric-pad stride-2 downsample convolution.
pub(crate) const VAE_DOWNSAMPLE: &str = "SDXL VAE downsample";
/// VAE decoder output GroupNorm + SiLU + `conv_out` (+ output cast).
pub(crate) const VAE_DECODE_OUT: &str = "SDXL VAE decode conv_out";
/// VAE encoder output GroupNorm + SiLU + `conv_out` + `quant_conv` + mean.
pub(crate) const VAE_ENCODE_OUT: &str = "SDXL VAE encode conv_out";
/// The decoded image's clamp / scale / round to 8-bit pixel values.
pub(crate) const DECODED_TO_IMAGE: &str = "SDXL decoded to image";
/// One halo-expanded tile of a layer-wise tiled VAE convolution, checked before its graph exists.
pub(crate) const VAE_TILE: &str = "SDXL VAE tile";
/// The assembled output of one tiled VAE layer (resnet / upsample / output convolution).
pub(crate) const VAE_TILED_STAGE: &str = "SDXL VAE tiled stage";

/// Elements of `a`, as u64.
pub(crate) fn elements(a: &Array) -> u64 {
    a.shape()
        .iter()
        .fold(1u64, |n, d| n.saturating_mul(*d as u64))
}

/// Logical (dequantized) weight elements of one projection, plus its bias. A packed projection is
/// priced at its full `[out, in]` extent: a dequantize/promotion copy is what the scratch floor
/// must cover.
pub(crate) fn linear_elements(linear: &AdaptableLinear) -> u64 {
    let weight = linear
        .base_shape()
        .iter()
        .fold(1u64, |n, d| n.saturating_mul(*d as u64));
    weight.saturating_add(linear.bias().map_or(0, elements))
}

/// Output-row count of one projection (the `out` of its logical `[out, in]` weight).
pub(crate) fn linear_out(linear: &AdaptableLinear) -> u64 {
    linear.base_shape().first().map_or(0, |d| *d as u64)
}

/// A conv stage at its OUTPUT extent `[b, h, w]` and the widest of `channels` — every channel count
/// the stage reads or writes, from its loaded weights.
pub(crate) fn conv_stage(b: i32, h: i32, w: i32, channels: &[i32]) -> u64 {
    let c = channels.iter().copied().max().unwrap_or(0);
    conv_stage_bytes(b as u64, h as u64, w as u64, c as u64)
}

/// Channel counts `(in, out)` of an NHWC conv weight `[out, kH, kW, in]`.
pub(crate) fn conv_channels(weight: &Array) -> (i32, i32) {
    let s = weight.shape();
    (s[s.len() - 1], s[0])
}

/// One text/vision transformer layer: `transformer_layer_bytes` over the loaded widths.
pub(crate) fn clip_layer(
    b: i32,
    tokens: i32,
    hidden: u64,
    intermediate: u64,
    heads: i32,
    head_dim: i32,
) -> u64 {
    transformer_layer_bytes(
        b as u64,
        tokens as u64,
        hidden,
        intermediate,
        heads as u64,
        head_dim as u64,
    )
}

/// The VAE mid-block single-head attention over every `h*w` token at width `c`.
pub(crate) fn vae_attention(b: i32, h: i32, w: i32, c: i32, weight_elements: u64) -> u64 {
    let tokens = (h as u64).saturating_mul(w as u64);
    attention_block_bytes(AttentionBlockGeometry {
        batch: b as u64,
        tokens,
        dim: c as u64,
        heads: 1,
        context_tokens: 0,
        ff_hidden: 0,
        query_rows: tokens,
        weight_elements,
    })
}

/// Total f32 bytes of `arrays` times `copies` — an elementwise pass (residual add, cast) over
/// already-materialized tensors.
pub(crate) fn elementwise_bytes(arrays: &[&Array], copies: u64) -> u64 {
    arrays
        .iter()
        .fold(0u64, |n, a| n.saturating_add(elements(a)))
        .saturating_mul(4)
        .saturating_mul(copies)
}

/// Check one halo-expanded tile of a layer-wise tiled VAE convolution BEFORE its norm/activation/
/// conv graph is constructed, priced at the tile's extent and `weight`'s channel counts. Called from
/// the per-tile preprocess hook of `mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc*`, which evaluates
/// each output core before the next tile is built.
pub(crate) fn before_tile(tile: &Array, weight: &Array) -> mlx_gen::Result<()> {
    let s = tile.shape();
    let (cin, cout) = conv_channels(weight);
    mlx_gen::ios_quantum::before(VAE_TILE, conv_stage(s[0], s[1], s[2], &[cin, cout]))
}

/// One layer-wise tiled VAE stage at OUTPUT extent `[b, h, w]`: four full-resolution f32 activations
/// at the widest channel count (input, assembled rows, assembled output, residual/contiguous copy).
/// The per-tile convolution scratch is checked separately by [`before_tile`].
pub(crate) fn tiled_stage(b: i32, h: i32, w: i32, channels: &[i32]) -> u64 {
    let c = channels.iter().copied().max().unwrap_or(0);
    [4u64, b as u64, h as u64, w as u64, c as u64, 4]
        .iter()
        .fold(1u64, |n, d| n.saturating_mul(*d))
}
