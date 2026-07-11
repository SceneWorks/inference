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

// ---------------------------------------------------------------------------
// VAE: native WAN-VAE (`Wan2.1` z16 3D-causal-conv autoencoder) → diffusers
//      `AutoencoderKLQwenImage` keys read by `crate::vae::{QwenVae, QwenVaeEncoder}`.
// ---------------------------------------------------------------------------

/// Remap a ComfyUI Qwen-Image VAE tensor map (`vae/qwen_image_vae.safetensors`, **native WAN-VAE
/// keys**, 194 tensors, bf16) to the diffusers `AutoencoderKLQwenImage` schema
/// [`crate::vae::QwenVae`] (decoder) / [`crate::vae::QwenVaeEncoder`] read. Pure key rename — the
/// values are byte-identical (the on-disk bf16 is upcast to the VAE compute dtype at `VarBuilder`
/// build, exactly like the snapshot VAE mmap), no per-tensor transform (the crate does the causal-3d
/// depth-tap reduction and any conv reshape at load).
///
/// The native layout stores each sub-module as a PyTorch `nn.Sequential` (integer child indices) and
/// the up/down stacks as **flat** lists; the diffusers layout uses named children and (for the
/// decoder) **nested** `up_blocks.{i}.{resnets.{j}|upsamplers.0}`. The renames:
///
/// * top-level `conv1`→`quant_conv`, `conv2`→`post_quant_conv`.
/// * `{enc,dec}.conv1`→`conv_in`; `{enc,dec}.head.0`→`norm_out`, `head.2`→`conv_out`.
/// * `{enc,dec}.middle.{0,1,2}` → `mid_block.{resnets.0, attentions.0, resnets.1}`.
/// * `encoder.downsamples.{i}` → `encoder.down_blocks.{i}` (flat, 1:1 — the crate's encoder
///   `down_blocks` is itself a flat mixed resnet/resample list).
/// * `decoder.upsamples.{i}` → `decoder.up_blocks.{i/4}.{resnets.{i%4} | upsamplers.0 when i%4==3}`
///   (each decoder up_block = 3 resnets + 1 upsampler; the last has no upsampler).
/// * resnet `residual.{0,2,3,6}`→`{norm1,conv1,norm2,conv2}`; `shortcut`→`conv_shortcut`; the
///   resample/`time_conv`/attention (`norm`/`to_qkv`/`proj`) leaves pass through. (`time_conv` is
///   present in the diffusers snapshot too — [`crate::vae`] skips it on a single image — so it is
///   renamed, not dropped.)
///
/// Errors if **no** native WAN-VAE key matched (a diffusers file, or the wrong family — surfaced, not
/// loaded as an empty VAE) or if a key sits under `{encoder,decoder}.` with an unrecognized sub-shape
/// (structural drift — surfaced, not silently mis-placed).
pub fn remap_vae_wan_to_diffusers(src: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(src.len());
    let mut remapped = 0usize;
    for (key, tensor) in src {
        match remap_vae_key(&key)? {
            Some(new_key) => {
                remapped += 1;
                out.insert(new_key, tensor);
            }
            // A stray already-diffusers / foreign key: keep it (the VAE `VarBuilder` reads only the
            // keys it needs); the guard below trips only when *nothing* matched.
            None => {
                out.insert(key, tensor);
            }
        }
    }
    if remapped == 0 {
        return Err(CandleError::Msg(
            "qwen-image ComfyUI VAE remap: no native WAN-VAE keys found (conv1/conv2, \
             {encoder,decoder}.{conv1,middle,downsamples,upsamples,head}) — not a ComfyUI \
             Qwen-Image VAE (wrong file/family?)"
                .to_string(),
        ));
    }
    Ok(out)
}

/// Map one native WAN-VAE key to its diffusers spelling, or `None` when the key is not a recognized
/// native top-level (`conv1`/`conv2`/`encoder.`/`decoder.`). A key *under* `{encoder,decoder}.` whose
/// sub-shape is unrecognized is an `Err` (structural drift), not a silent pass-through.
fn remap_vae_key(key: &str) -> Result<Option<String>> {
    // Top-level quant / post-quant convs (`conv1` = mu/logvar quant `[32,32,1,1,1]`, `conv2` =
    // post-quant `[16,16,1,1,1]`). Checked before the stem loop; `encoder.conv1`/`decoder.conv1` are
    // the per-branch input convs and do NOT start with `conv1.`.
    if let Some(tail) = key.strip_prefix("conv1.") {
        return Ok(Some(format!("quant_conv.{tail}")));
    }
    if let Some(tail) = key.strip_prefix("conv2.") {
        return Ok(Some(format!("post_quant_conv.{tail}")));
    }
    for stem in ["encoder", "decoder"] {
        let Some(rest) = key.strip_prefix(&format!("{stem}.")) else {
            continue;
        };
        return remap_vae_stem(stem, rest).map(Some);
    }
    Ok(None)
}

/// Remap the `rest` after an `{encoder|decoder}.` prefix. Splits on the fixed WAN-VAE sub-module names
/// so the flat→nested decoder arithmetic (and the encoder's 1:1 flat map) is explicit.
fn remap_vae_stem(stem: &str, rest: &str) -> Result<String> {
    if let Some(tail) = rest.strip_prefix("conv1.") {
        return Ok(format!("{stem}.conv_in.{tail}"));
    }
    if let Some(tail) = rest.strip_prefix("head.0.") {
        return Ok(format!("{stem}.norm_out.{tail}"));
    }
    if let Some(tail) = rest.strip_prefix("head.2.") {
        return Ok(format!("{stem}.conv_out.{tail}"));
    }
    if let Some(after) = rest.strip_prefix("middle.") {
        let (idx, sub) = split_idx(after, stem, "middle")?;
        // middle.0 → resnets.0, middle.1 → attentions.0, middle.2 → resnets.1.
        let named = match idx {
            0 => "resnets.0",
            1 => "attentions.0",
            2 => "resnets.1",
            _ => {
                return Err(CandleError::Msg(format!(
                    "qwen-image ComfyUI VAE remap: unexpected {stem}.middle.{idx} (expected 0..=2)"
                )))
            }
        };
        return Ok(format!(
            "{stem}.mid_block.{named}.{}",
            remap_vae_module_leaf(sub, stem)?
        ));
    }
    if let Some(after) = rest.strip_prefix("downsamples.") {
        // Encoder: the crate's `down_blocks` is itself a flat mixed list → 1:1 index.
        let (idx, sub) = split_idx(after, stem, "downsamples")?;
        return Ok(format!(
            "{stem}.down_blocks.{idx}.{}",
            remap_vae_module_leaf(sub, stem)?
        ));
    }
    if let Some(after) = rest.strip_prefix("upsamples.") {
        // Decoder: flat `upsamples` → nested `up_blocks`. Each up_block is 3 resnets + 1 upsampler
        // (4 flat slots); slot 3 is the upsampler (the last up_block has none, so its slot 3 never
        // exists on disk).
        let (idx, sub) = split_idx(after, stem, "upsamples")?;
        let (block, slot) = (idx / 4, idx % 4);
        let named = if slot == 3 {
            "upsamplers.0".to_string()
        } else {
            format!("resnets.{slot}")
        };
        return Ok(format!(
            "{stem}.up_blocks.{block}.{named}.{}",
            remap_vae_module_leaf(sub, stem)?
        ));
    }
    Err(CandleError::Msg(format!(
        "qwen-image ComfyUI VAE remap: unrecognized {stem} sub-key {rest:?}"
    )))
}

/// Split `"{idx}.{sub}"` (a flat-list child) into its numeric index and the remaining leaf.
fn split_idx<'a>(after: &'a str, stem: &str, list: &str) -> Result<(usize, &'a str)> {
    let (idx_str, sub) = after.split_once('.').ok_or_else(|| {
        CandleError::Msg(format!(
            "qwen-image ComfyUI VAE remap: malformed {stem}.{list} key {after:?}"
        ))
    })?;
    let idx: usize = idx_str.parse().map_err(|_| {
        CandleError::Msg(format!(
            "qwen-image ComfyUI VAE remap: bad index in {stem}.{list}.{after:?}"
        ))
    })?;
    Ok((idx, sub))
}

/// Remap the leaf inside a WAN-VAE sub-module (resnet / resample / attention) to its diffusers leaf.
/// Resnet `nn.Sequential` children: `residual.0`=norm1, `.2`=conv1, `.3`=norm2, `.6`=conv2 (the odd
/// indices are the parameter-free SiLU/Dropout, never on disk). `shortcut`→`conv_shortcut`. The
/// resample (`resample.1`, `time_conv`) and attention (`norm`, `to_qkv`, `proj`) leaves already match
/// and pass through. Any other leaf under a native module is `Err` (structural drift).
fn remap_vae_module_leaf(sub: &str, stem: &str) -> Result<String> {
    if let Some(tail) = sub.strip_prefix("residual.") {
        let (idx, leaf) = tail.split_once('.').ok_or_else(|| {
            CandleError::Msg(format!(
                "qwen-image ComfyUI VAE remap: malformed {stem} residual leaf {sub:?}"
            ))
        })?;
        let named = match idx {
            "0" => "norm1",
            "2" => "conv1",
            "3" => "norm2",
            "6" => "conv2",
            _ => {
                return Err(CandleError::Msg(format!(
                    "qwen-image ComfyUI VAE remap: unexpected {stem} residual index in {sub:?} \
                     (expected 0/2/3/6)"
                )))
            }
        };
        return Ok(format!("{named}.{leaf}"));
    }
    if let Some(tail) = sub.strip_prefix("shortcut.") {
        return Ok(format!("conv_shortcut.{tail}"));
    }
    // resample.1.*, time_conv.*, norm.*, to_qkv.*, proj.* already match the diffusers leaf.
    if sub.starts_with("resample.")
        || sub.starts_with("time_conv.")
        || sub.starts_with("norm.")
        || sub.starts_with("to_qkv.")
        || sub.starts_with("proj.")
    {
        return Ok(sub.to_string());
    }
    Err(CandleError::Msg(format!(
        "qwen-image ComfyUI VAE remap: unrecognized {stem} module leaf {sub:?}"
    )))
}

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

    // --- VAE (native WAN-VAE → diffusers) -------------------------------------

    fn rk(key: &str) -> String {
        remap_vae_key(key).unwrap().expect("native key should map")
    }

    #[test]
    fn vae_top_level_quant_convs() {
        assert_eq!(rk("conv1.weight"), "quant_conv.weight");
        assert_eq!(rk("conv1.bias"), "quant_conv.bias");
        assert_eq!(rk("conv2.weight"), "post_quant_conv.weight");
    }

    #[test]
    fn vae_conv_in_and_head() {
        assert_eq!(rk("decoder.conv1.weight"), "decoder.conv_in.weight");
        assert_eq!(rk("encoder.conv1.bias"), "encoder.conv_in.bias");
        assert_eq!(rk("decoder.head.0.gamma"), "decoder.norm_out.gamma");
        assert_eq!(rk("decoder.head.2.weight"), "decoder.conv_out.weight");
        assert_eq!(rk("encoder.head.2.bias"), "encoder.conv_out.bias");
    }

    #[test]
    fn vae_middle_block_resnets_and_attention() {
        assert_eq!(
            rk("decoder.middle.0.residual.0.gamma"),
            "decoder.mid_block.resnets.0.norm1.gamma"
        );
        assert_eq!(
            rk("decoder.middle.0.residual.6.weight"),
            "decoder.mid_block.resnets.0.conv2.weight"
        );
        assert_eq!(
            rk("decoder.middle.2.residual.2.bias"),
            "decoder.mid_block.resnets.1.conv1.bias"
        );
        assert_eq!(
            rk("decoder.middle.1.norm.gamma"),
            "decoder.mid_block.attentions.0.norm.gamma"
        );
        assert_eq!(
            rk("encoder.middle.1.to_qkv.weight"),
            "encoder.mid_block.attentions.0.to_qkv.weight"
        );
        assert_eq!(
            rk("encoder.middle.1.proj.bias"),
            "encoder.mid_block.attentions.0.proj.bias"
        );
    }

    #[test]
    fn vae_encoder_downsamples_are_flat_one_to_one() {
        // Resnet leaf remap under a flat 1:1 index.
        assert_eq!(
            rk("encoder.downsamples.0.residual.3.gamma"),
            "encoder.down_blocks.0.norm2.gamma"
        );
        // A resample (downsample) module: the `resample.1` leaf passes through.
        assert_eq!(
            rk("encoder.downsamples.2.resample.1.weight"),
            "encoder.down_blocks.2.resample.1.weight"
        );
        // A shortcut → conv_shortcut.
        assert_eq!(
            rk("encoder.downsamples.3.shortcut.weight"),
            "encoder.down_blocks.3.conv_shortcut.weight"
        );
        // A `time_conv` (temporal) leaf passes through (present in the diffusers snapshot, unused).
        assert_eq!(
            rk("encoder.downsamples.5.time_conv.bias"),
            "encoder.down_blocks.5.time_conv.bias"
        );
    }

    #[test]
    fn vae_decoder_upsamples_flat_to_nested() {
        // Flat slot < 3 → resnets.slot within up_block = idx/4.
        assert_eq!(
            rk("decoder.upsamples.0.residual.2.weight"),
            "decoder.up_blocks.0.resnets.0.conv1.weight"
        );
        assert_eq!(
            rk("decoder.upsamples.4.shortcut.bias"),
            "decoder.up_blocks.1.resnets.0.conv_shortcut.bias"
        );
        assert_eq!(
            rk("decoder.upsamples.10.residual.6.weight"),
            "decoder.up_blocks.2.resnets.2.conv2.weight"
        );
        // Flat slot 3 → the up_block's upsampler.
        assert_eq!(
            rk("decoder.upsamples.3.resample.1.weight"),
            "decoder.up_blocks.0.upsamplers.0.resample.1.weight"
        );
        assert_eq!(
            rk("decoder.upsamples.11.resample.1.bias"),
            "decoder.up_blocks.2.upsamplers.0.resample.1.bias"
        );
        // The last up_block (12,13,14) has only resnets (no slot 3 on disk).
        assert_eq!(
            rk("decoder.upsamples.14.residual.0.gamma"),
            "decoder.up_blocks.3.resnets.2.norm1.gamma"
        );
    }

    #[test]
    fn vae_remap_wrapper_counts_and_passes_values() {
        let mut src = HashMap::new();
        src.insert("conv1.weight".to_string(), t(DType::BF16));
        src.insert(
            "decoder.upsamples.3.resample.1.weight".to_string(),
            t(DType::BF16),
        );
        let out = remap_vae_wan_to_diffusers(src).unwrap();
        assert!(out.contains_key("quant_conv.weight"));
        assert!(out.contains_key("decoder.up_blocks.0.upsamplers.0.resample.1.weight"));
        // Values pass through unchanged (bf16 — the upcast happens at VarBuilder build).
        assert_eq!(out.get("quant_conv.weight").unwrap().dtype(), DType::BF16);
    }

    #[test]
    fn vae_rejects_non_native_map() {
        // An already-diffusers file (no native WAN keys) → surfaced, not loaded as an empty VAE.
        let mut src = HashMap::new();
        src.insert(
            "decoder.mid_block.resnets.0.conv1.weight".to_string(),
            t(DType::BF16),
        );
        assert!(remap_vae_wan_to_diffusers(src).is_err());
    }

    #[test]
    fn vae_rejects_structural_drift_under_a_stem() {
        // A key under `decoder.` with an unrecognized sub-shape is a hard error (not mis-placed).
        assert!(remap_vae_key("decoder.middle.0.residual.9.weight").is_err());
        assert!(remap_vae_key("decoder.upsamples.0.mystery.weight").is_err());
        assert!(remap_vae_key("encoder.bogus.0.weight").is_err());
    }
}
