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
//! [`remap_vae_wan_to_diffusers`] — a pure key rename (values byte-identical bf16, upcast to f32 at
//! `VarBuilder` build like the snapshot VAE). When the caller does **not** pass it, the VAE falls back
//! to the resident snapshot's `vae/` (same weights, diffusers keys). The tree's Qwen2.5-VL **text
//! encoder** is still snapshot-sourced (it is *scaled*-fp8, sc-10671 territory), as is the tiny
//! tokenizer. This mirrors how the Z-Image lane remapped its BFL/ldm VAE (the analogous seam in
//! `candle-gen-z-image`'s `comfyui` module) while sourcing its tokenizer from our snapshot.
//!
//! Header-only classification (which file is a Qwen-Image DiT, and that it is plain fp8) is done
//! upstream by SceneWorks (`sceneworks-core::base_weights`, sc-10662); this module is handed a file
//! already identified as a Qwen-Image DiT.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::{CandleError, Result};

/// The ComfyUI/BFL prefix every Qwen-Image DiT tensor carries; stripped to reach the diffusers keys
/// [`crate::transformer::QwenTransformer`] reads.
const COMFY_DIT_PREFIX: &str = "model.diffusion_model.";

/// Remap + upcast a ComfyUI Qwen-Image DiT tensor map to the schema
/// [`crate::transformer::QwenTransformer`] reads, ready for `VarBuilder::from_tensors`.
///
/// Two transforms per tensor, nothing synthesized or dropped:
///
/// 1. **Strip** the `model.diffusion_model.` prefix — the inner diffusers names already match candle,
///    so this is the entire key transform (no per-leaf aliases).
/// 2. **Upcast** each weight to `dtype` (the compute dtype, bf16). The plain-fp8 checkpoint is all
///    `F8_E4M3`; the cast is a straight `to_dtype` with **no** scale companion (candle's CPU backend
///    upcasts `F8_E4M3 → bf16` directly). Integer buffers (if any) pass through unchanged.
///
/// Errors if **no** tensor carries the `model.diffusion_model.` prefix — a file whose keys we cannot
/// place is a wrong-file / wrong-family signal, surfaced rather than loaded as an empty transformer
/// (no silent fallback). A checkpoint that mixes prefixed and bare keys keeps the bare ones as-is (the
/// transformer's `VarBuilder` reads only the keys it needs); the guard only trips when *nothing*
/// matched.
pub fn remap_and_cast_comfyui_dit(
    src: HashMap<String, Tensor>,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(src.len());
    let mut stripped = 0usize;
    for (key, tensor) in src {
        let new_key = match key.strip_prefix(COMFY_DIT_PREFIX) {
            Some(rest) => {
                stripped += 1;
                rest.to_string()
            }
            None => key,
        };
        let tensor = cast_weight(&new_key, tensor, dtype)?;
        out.insert(new_key, tensor);
    }
    if stripped == 0 {
        return Err(CandleError::Msg(format!(
            "qwen-image ComfyUI DiT remap: no {COMFY_DIT_PREFIX:?}-prefixed tensors found — not a \
             ComfyUI Qwen-Image DiT (wrong file/family?)"
        )));
    }
    Ok(out)
}

/// Upcast one DiT weight to the compute `dtype`. Floating tensors (fp8 / f32 / bf16) cast; integer
/// buffers pass through unchanged (they are indices/caches, not weights, and must keep their dtype).
fn cast_weight(key: &str, tensor: Tensor, dtype: DType) -> Result<Tensor> {
    if tensor.dtype().is_int() {
        return Ok(tensor);
    }
    tensor.to_dtype(dtype).map_err(|e| {
        CandleError::Msg(format!(
            "qwen-image ComfyUI DiT remap: upcast {key:?} ({:?} → {dtype:?}) failed: {e}",
            tensor.dtype()
        ))
    })
}

// VAE: native WAN-VAE (`Wan2.1` z16 3D-causal-conv autoencoder) → diffusers `AutoencoderKLQwenImage`
// keys read by `crate::vae::{QwenVae, QwenVaeEncoder}`. Qwen-Image ships the *same physical* Wan2.1
// 16-channel VAE the Wan2.2 lane reads, so the native→diffusers key rename is shared from the
// `candle-gen` core crate (sc-10909) rather than kept per-crate — re-exported here so this module's
// `comfyui` seam and the sc-10830 call sites keep referring to `comfyui::remap_vae_wan_to_diffusers`.
pub use candle_gen::comfyui_vae::remap_vae_wan_to_diffusers;

#[cfg(test)]
mod tests {
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
    fn passes_bare_diffusers_keys_when_some_prefixed() {
        // A checkpoint that carries a stray already-bare key alongside prefixed ones keeps the bare one.
        let mut src = HashMap::new();
        src.insert(
            "model.diffusion_model.img_in.weight".to_string(),
            t(DType::F8E4M3),
        );
        src.insert("txt_norm.weight".to_string(), t(DType::BF16));
        let out = remap_and_cast_comfyui_dit(src, DType::BF16).unwrap();
        assert!(out.contains_key("img_in.weight"));
        assert!(out.contains_key("txt_norm.weight"));
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
