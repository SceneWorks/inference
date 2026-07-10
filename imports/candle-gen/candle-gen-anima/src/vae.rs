//! Anima's VAE is the Qwen-Image `AutoencoderKLQwenImage` (`qwen_image_vae.safetensors`, z_dim 16,
//! 8× spatial, per-channel `latents_mean`/`latents_std` de-norm). We reuse
//! [`candle_gen_qwen_image::vae::QwenVae`] wholesale.
//!
//! The on-disk file uses the **original** Qwen VAE naming (`conv1`, `encoder.downsamples.N`,
//! `decoder.upsamples.N`, `head.0`, `middle.N`, …). The candle `QwenVae` reads the **diffusers**
//! `AutoencoderKLQwenImage` naming (`post_quant_conv`, `decoder.conv_in`, `decoder.mid_block.*`,
//! `decoder.up_blocks.*`, `decoder.norm_out`, …) directly. So we port the Anima convert script's
//! original→diffusers **rename** ([`convert_vae_key`], rename-only — no tensor transpose; the candle
//! `QwenVae` reduces the native 5-D Conv3d weights + flattens `gamma` itself) and build a VarBuilder
//! from the renamed f32 tensors. (This is the candle counterpart of the MLX `vae.rs`, which chains
//! `convert_vae_key` → `remap_vae_keys`; candle needs no `remap_vae_keys` because its `QwenVae` speaks
//! the diffusers naming + native conv layout directly.)

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{DType, Device};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::Result;

pub use candle_gen_qwen_image::vae::QwenVae;

/// Port of the Anima convert script's `rename_residual_key` (diffusers resnet leaf names).
fn rename_residual_key(key: &str) -> String {
    key.replace(".residual.0.", ".norm1.")
        .replace(".residual.2.", ".conv1.")
        .replace(".residual.3.", ".norm2.")
        .replace(".residual.6.", ".conv2.")
        .replace(".shortcut.", ".conv_shortcut.")
}

/// Port of `rename_mid_key`.
fn rename_mid_key(key: &str) -> String {
    let k = key
        .replace(".middle.0.", ".mid_block.resnets.0.")
        .replace(".middle.1.", ".mid_block.attentions.0.")
        .replace(".middle.2.", ".mid_block.resnets.1.");
    rename_residual_key(&k)
}

/// Port of `rename_decoder_upsample_key` (flat `decoder.upsamples.N` → grouped up_blocks).
fn rename_decoder_upsample_key(key: &str) -> String {
    let suffix = key.strip_prefix("decoder.upsamples.").unwrap_or(key);
    let (index_str, rest) = suffix.split_once('.').unwrap_or((suffix, ""));
    let index: i64 = index_str.parse().unwrap_or(-1);
    let new_key = if index == 3 || index == 7 || index == 11 {
        let block_index = (index - 3) / 4;
        format!("decoder.up_blocks.{block_index}.upsamplers.0.{rest}")
    } else {
        let block_index = index / 4;
        let resnet_index = index % 4;
        format!("decoder.up_blocks.{block_index}.resnets.{resnet_index}.{rest}")
    };
    rename_residual_key(&new_key)
}

/// Map one original-naming Qwen VAE key to its diffusers `AutoencoderKLQwenImage` name (port of
/// `convert_qwen_image_vae_state_dict`, rename-only). The result is what candle `QwenVae` reads.
pub fn convert_vae_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("conv1.") {
        format!("quant_conv.{rest}")
    } else if let Some(rest) = key.strip_prefix("conv2.") {
        format!("post_quant_conv.{rest}")
    } else if let Some(rest) = key.strip_prefix("encoder.conv1.") {
        format!("encoder.conv_in.{rest}")
    } else if let Some(rest) = key.strip_prefix("decoder.conv1.") {
        format!("decoder.conv_in.{rest}")
    } else if let Some(rest) = key.strip_prefix("encoder.downsamples.") {
        rename_residual_key(&format!("encoder.down_blocks.{rest}"))
    } else if key.starts_with("decoder.upsamples.") {
        rename_decoder_upsample_key(key)
    } else if key.starts_with("encoder.middle.") || key.starts_with("decoder.middle.") {
        rename_mid_key(key)
    } else if let Some(rest) = key.strip_prefix("encoder.head.0.") {
        format!("encoder.norm_out.{rest}")
    } else if let Some(rest) = key.strip_prefix("encoder.head.2.") {
        format!("encoder.conv_out.{rest}")
    } else if let Some(rest) = key.strip_prefix("decoder.head.0.") {
        format!("decoder.norm_out.{rest}")
    } else if let Some(rest) = key.strip_prefix("decoder.head.2.") {
        format!("decoder.conv_out.{rest}")
    } else {
        rename_residual_key(key)
    }
}

/// Load the Qwen-Image VAE from Anima's single-file `qwen_image_vae.safetensors` on `device`. Run in
/// f32 (the qwen-image golden convention): the denoised latents are f32 and this avoids a mixed-dtype
/// conv, at a negligible memory cost for the small VAE.
pub fn load_vae(path: impl AsRef<Path>, device: &Device) -> Result<QwenVae> {
    // Load + cast every tensor to f32 (candle_gen::Weights coerces floats on load), then rename keys.
    let src = candle_gen::Weights::from_file(path.as_ref(), device, DType::F32)?;
    let keys: Vec<String> = src.keys().cloned().collect();
    let mut map: HashMap<String, _> = HashMap::with_capacity(keys.len());
    for k in &keys {
        map.insert(convert_vae_key(k), src.require(k)?);
    }
    let vb = VarBuilder::from_tensors(map, DType::F32, device);
    QwenVae::new(vb).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_vae_key_examples() {
        assert_eq!(convert_vae_key("conv1.weight"), "quant_conv.weight");
        assert_eq!(convert_vae_key("conv2.bias"), "post_quant_conv.bias");
        assert_eq!(
            convert_vae_key("encoder.conv1.weight"),
            "encoder.conv_in.weight"
        );
        assert_eq!(
            convert_vae_key("decoder.conv1.bias"),
            "decoder.conv_in.bias"
        );
        assert_eq!(
            convert_vae_key("encoder.head.0.gamma"),
            "encoder.norm_out.gamma"
        );
        assert_eq!(
            convert_vae_key("encoder.head.2.weight"),
            "encoder.conv_out.weight"
        );
        assert_eq!(
            convert_vae_key("decoder.head.0.gamma"),
            "decoder.norm_out.gamma"
        );
        // encoder downsample resnet: flat index preserved, residual leaf renamed.
        assert_eq!(
            convert_vae_key("encoder.downsamples.0.residual.2.weight"),
            "encoder.down_blocks.0.conv1.weight"
        );
        // decoder upsample: index 3 is the upsampler resample conv.
        assert_eq!(
            convert_vae_key("decoder.upsamples.3.resample.1.weight"),
            "decoder.up_blocks.0.upsamplers.0.resample.1.weight"
        );
        // decoder upsample resnet: index 0 → block 0, resnet 0.
        assert_eq!(
            convert_vae_key("decoder.upsamples.0.residual.0.gamma"),
            "decoder.up_blocks.0.resnets.0.norm1.gamma"
        );
        // mid-block attention (fused to_qkv is preserved — candle QwenVae reads to_qkv/proj).
        assert_eq!(
            convert_vae_key("decoder.middle.1.to_qkv.weight"),
            "decoder.mid_block.attentions.0.to_qkv.weight"
        );
        assert_eq!(
            convert_vae_key("decoder.middle.0.residual.0.gamma"),
            "decoder.mid_block.resnets.0.norm1.gamma"
        );
    }
}
