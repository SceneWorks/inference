//! ComfyUI single-file Qwen-Image DiT → candle in-memory remap seam (epic 10451 Phase 2b, sc-10670).
//!
//! SceneWorks lets a user point at an existing ComfyUI `models/` tree and generate from the weights in
//! place — no copy, no re-download (Phase 1 did this for LoRAs; sc-10668 did it for the bf16 Z-Image
//! base). This module is the Qwen-Image slice: a ComfyUI Qwen-Image install ships the DiT as one
//! single file with **BFL-native** tensor names and a **plain fp8_e4m3fn** cast (no scale companions):
//!
//! * `diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors` — the 60-layer dual-stream MMDiT, 1933
//!   tensors, **all `F8_E4M3`, zero scale tensors**, keyed `model.diffusion_model.<diffusers-name>`.
//!
//! Two observations make this the *smallest* Phase 2 quant slice:
//!
//! 1. **Keys already match.** After the `model.diffusion_model.` prefix, the inner names are the exact
//!    diffusers spelling [`crate::transformer::QwenTransformer`] reads (`img_in`, `txt_in`, `txt_norm`,
//!    `proj_out`, `norm_out.linear`, `time_text_embed.timestep_embedder.linear_{1,2}`, and every
//!    `transformer_blocks.N.{img_mod,txt_mod,attn.*,img_mlp,txt_mlp}` leaf). So the DiT remap is a pure
//!    **prefix strip** — no per-leaf aliases, unlike the Z-Image DiT's fused-qkv split.
//! 2. **No scale companions.** Plain fp8 is a straight upcast: `F8_E4M3 → bf16` per tensor. (The
//!    *scaled*-fp8 conventions — a `.scale_weight`/`.scale_input` companion, or a `.weight_scale`
//!    triplet — are the later slices sc-10671/sc-10680, which multiply by the scale *before* the same
//!    prefix strip.)
//!
//! The Qwen-Image **VAE** is also read in place when the caller passes the tree's
//! `vae/qwen_image_vae.safetensors` (epic 10451 Phase 2b, sc-10830): that file carries **native
//! WAN-VAE key names** (`conv1`, `{enc,dec}oder.{middle,downsamples,upsamples,head}.*.residual.*`),
//! remapped to the diffusers schema [`crate::vae::QwenVae`]/[`crate::vae::QwenVaeEncoder`] read by
//! [`candle_gen::comfyui_vae::wan_vae_var_builder`] — a pure key rename (values byte-identical bf16, upcast to f32 at
//! `VarBuilder` build like the snapshot VAE). When the caller does **not** pass it, the VAE falls back
//! to the resident snapshot's `vae/` (same weights, diffusers keys). The tree's Qwen2.5-VL **text
//! encoder** is still snapshot-sourced (it is *scaled*-fp8, sc-10671 territory), as is the tiny
//! tokenizer. This mirrors how the Z-Image lane remapped its BFL/ldm VAE (the analogous seam in
//! `candle-gen-z-image`'s `comfyui` module) while sourcing its tokenizer from our snapshot.
//!
//! Header-only classification (which file is a Qwen-Image DiT, and that it is plain fp8) is done
//! upstream by SceneWorks (`sceneworks-core::base_weights`, sc-10662); this module is handed a file
//! already identified as a Qwen-Image DiT.

use candle_gen::candle_core::{DType, Device};
use candle_gen::{CandleError, Result};

/// Physical namespace admission runs before binding the shared checkpoint-aware builder.
pub(crate) fn transformer_builder(
    path: &std::path::Path,
    dtype: DType,
    device: &Device,
) -> Result<candle_gen::candle_nn::VarBuilder<'static>> {
    let mut count = 0;
    let vb = candle_gen::loader::remap_file_var_builder(path, dtype, device, |key, _| {
        if key == "scaled_fp8" {
            return Ok(Vec::new());
        }
        let name=key.strip_prefix("model.diffusion_model.").filter(|n|!n.is_empty()).ok_or_else(||CandleError::Msg(format!("qwen-image ComfyUI DiT: {key} outside the required model.diffusion_model namespace")))?;
        count += 1;
        Ok(vec![candle_gen::loader::SourceView::Whole(name.to_owned())])
    })?;
    if count == 0 {
        return Err(CandleError::Msg(
            "qwen-image ComfyUI DiT has no prefixed tensors".into(),
        ));
    }
    Ok(vb)
}
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    fn remap_and_cast_comfyui_dit(
        src: HashMap<String, Tensor>,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        let dir = tempfile::tempdir().map_err(|e| CandleError::Msg(e.to_string()))?;
        let path = dir.path().join("model.safetensors");
        candle_gen::candle_core::safetensors::save(&src, &path)?;
        let vb = transformer_builder(&path, dtype, &Device::Cpu)?;
        src.keys()
            .filter_map(|k| k.strip_prefix("model.diffusion_model."))
            .map(|k| Ok((k.to_owned(), vb.get_unchecked(k)?)))
            .collect()
    }

    use super::*;
    use candle_gen::candle_core::{Device, Tensor};

    fn t(dtype: DType) -> Tensor {
        Tensor::zeros(&[4, 4], DType::F32, &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    #[test]
    fn strips_diffusion_model_prefix() {
        let mut src = HashMap::new();
        src.insert(
            "model.diffusion_model.transformer_blocks.0.attn.to_q.weight".to_string(),
            t(DType::F8E4M3),
        );
        src.insert(
            "model.diffusion_model.img_in.weight".to_string(),
            t(DType::F8E4M3),
        );
        let out = remap_and_cast_comfyui_dit(src, DType::BF16).unwrap();
        assert!(out.contains_key("transformer_blocks.0.attn.to_q.weight"));
        assert!(out.contains_key("img_in.weight"));
        // Prefixed forms are gone.
        assert!(!out.contains_key("model.diffusion_model.img_in.weight"));
    }

    #[test]
    fn upcasts_fp8_to_bf16() {
        let mut src = HashMap::new();
        src.insert(
            "model.diffusion_model.img_in.weight".to_string(),
            t(DType::F8E4M3),
        );
        let out = remap_and_cast_comfyui_dit(src, DType::BF16).unwrap();
        assert_eq!(
            out.get("img_in.weight").unwrap().dtype(),
            DType::BF16,
            "plain fp8 weight upcast to the compute dtype"
        );
    }

    #[test]
    fn rejects_bare_diffusers_keys_even_when_some_are_prefixed() {
        let mut src = HashMap::new();
        src.insert(
            "model.diffusion_model.img_in.weight".to_string(),
            t(DType::F8E4M3),
        );
        src.insert("txt_norm.weight".to_string(), t(DType::BF16));
        let error = remap_and_cast_comfyui_dit(src, DType::BF16)
            .expect_err("mixed namespaces must fail closed")
            .to_string();
        assert!(error.contains("outside the required"), "got: {error}");
    }

    #[test]
    fn rejects_a_map_with_no_prefixed_keys() {
        // No `model.diffusion_model.` anywhere → wrong file, surfaced not silently loaded.
        let mut src = HashMap::new();
        src.insert("some.other.tensor".to_string(), t(DType::BF16));
        assert!(remap_and_cast_comfyui_dit(src, DType::BF16).is_err());
    }

    // The native WAN-VAE → diffusers remap (`remap_vae_wan_to_diffusers`) is shared with the Wan2.2
    // lane and unit-tested in `candle_gen::comfyui_vae` (sc-10909); the sc-10830 in-place VAE build is
    // exercised end-to-end by `comfyui_vae_validate`.
}
