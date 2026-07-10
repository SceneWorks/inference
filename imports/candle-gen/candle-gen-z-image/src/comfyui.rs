//! ComfyUI single-file → candle in-memory remap seam (epic 10451 Phase 2, sc-10668).
//!
//! SceneWorks lets a user point at an existing ComfyUI `models/` tree and generate
//! from the weights in place — no copy, no re-download (Phase 1 did this for LoRAs).
//! Phase 2 does it for **base** models. A ComfyUI Z-Image install ships the three
//! components as *separate* single files with ComfyUI-native tensor names:
//!
//! * `unet/z_image_turbo_bf16.safetensors` — the DiT, 453 tensors, ComfyUI-native
//!   keys (fused `attention.qkv`, `q_norm`/`k_norm`, `attention.out`).
//! * `text_encoders/qwen_3_4b.safetensors` — the Qwen3 text encoder in **standard
//!   HF** layout, which [`crate::pipeline`]'s `ZImageTextEncoder` already loads
//!   verbatim (no remap).
//! * `vae/ae.safetensors` — the VAE in **BFL/ldm** ("flux autoencoder") layout,
//!   which the diffusers-format `AutoEncoderKL` the pipeline uses does **not**
//!   accept.
//!
//! This module is the two key-schema remaps that make the DiT and VAE loadable via
//! `VarBuilder::from_tensors` (the same in-memory tensor path
//! [`crate::pipeline::Pipeline::transformer_vb_with_adapters`] already uses for
//! LoRA merge). It is the shared seam the later per-quant slices (fp8 / scaled-fp8 /
//! GGUF) extend — they add a dequant step *before* these key transforms; the key
//! transforms themselves are quant-agnostic.
//!
//! Header-only classification (which file is which family/component) is done
//! upstream by SceneWorks (`sceneworks-core::base_weights`, sc-10662); this module
//! is handed the files already identified as a Z-Image DiT and VAE.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_gen::candle_core::Tensor;
use candle_gen::{CandleError, Result};

/// The four external inputs for an in-place ComfyUI Z-Image load (sc-10668): the
/// three separate ComfyUI component files (read in place, never copied) plus the
/// directory holding our shipped `tokenizer/tokenizer.json` — the one tiny file a
/// ComfyUI tree does not ship (ComfyUI stores weights only; the tokenizer,
/// scheduler, and geometry are otherwise compiled into the loader).
///
/// The DiT and VAE are key-remapped in memory ([`remap_dit_comfyui_to_diffusers`],
/// [`remap_vae_ldm_to_diffusers`]) at component-load; the Qwen3 text encoder loads
/// verbatim.
#[derive(Clone, Debug)]
pub(crate) struct ComfyuiSources {
    /// ComfyUI DiT (`unet/z_image_turbo_bf16.safetensors`), ComfyUI-native keys.
    pub transformer_file: PathBuf,
    /// ComfyUI Qwen3 text encoder (`text_encoders/qwen_3_4b.safetensors`), HF keys.
    pub text_encoder_file: PathBuf,
    /// ComfyUI VAE (`vae/ae.safetensors`), BFL/ldm keys.
    pub vae_file: PathBuf,
    /// Directory containing `tokenizer/tokenizer.json` (our shipped Z-Image snapshot).
    pub tokenizer_dir: PathBuf,
}

/// Z-Image DiT hidden size (`Config::z_image_turbo().dim`). The fused attention
/// projection is `[3 * DIM, DIM]`; each split slice is `[DIM, DIM]`.
const DIT_DIM: usize = 3840;

/// The VAE has four resolution levels (`block_out_channels = [128,256,512,512]`).
/// The diffusers decoder numbers its `up_blocks` in the reverse order of the ldm
/// `decoder.up.*` (candle `z_image::vae` Decoder: "up_blocks order is reversed from
/// encoder down_blocks"), so `up.i` → `up_blocks.(N-1-i)`.
const VAE_LEVELS: usize = 4;

// ---------------------------------------------------------------------------
// DiT: ComfyUI-native → diffusers/candle `ZImageTransformer2DModel` keys
// ---------------------------------------------------------------------------

/// Remap a ComfyUI Z-Image DiT tensor map to the key schema
/// `candle_transformers::models::z_image::transformer::ZImageTransformer2DModel`
/// reads. Two transforms, everything else a pass-through:
///
/// 1. **Split** the fused `{blk}.attention.qkv.weight` `[3·DIM, DIM]` into
///    `{blk}.attention.{to_q,to_k,to_v}.weight` `[DIM, DIM]` each (row-split, q/k/v
///    order). This is the entire 453→521 tensor-count difference (+2 per block ×
///    34 blocks); nothing is synthesized or dropped.
/// 2. **Rename** the leaf names the two schemas spell differently:
///    `attention.q_norm`→`attention.norm_q`, `attention.k_norm`→`attention.norm_k`,
///    `attention.out`→`attention.to_out.0`; and the single-aspect head wrappers
///    `x_embedder`→`all_x_embedder.2-1`, `final_layer`→`all_final_layer.2-1`.
///
/// The block prefixes (`noise_refiner.N`, `context_refiner.N`, `layers.N`), the
/// SwiGLU `feed_forward.w{1,2,3}`, the pre/post norms, `adaLN_modulation`,
/// `cap_embedder`, `t_embedder`, and the pad tokens already match and pass through.
///
/// Errors if a fused `qkv` weight is not `[3·DIM, DIM]` — a shape we do not
/// recognize is a wrong-file/wrong-family signal, surfaced rather than mis-split
/// (no silent fallback).
pub fn remap_dit_comfyui_to_diffusers(
    src: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(src.len() + 128);
    for (key, tensor) in src {
        if let Some(prefix) = key.strip_suffix(".attention.qkv.weight") {
            let (q, k, v) = split_fused_qkv(&key, &tensor)?;
            out.insert(format!("{prefix}.attention.to_q.weight"), q);
            out.insert(format!("{prefix}.attention.to_k.weight"), k);
            out.insert(format!("{prefix}.attention.to_v.weight"), v);
            continue;
        }
        if key.ends_with(".attention.qkv.bias") {
            // Z-Image-Turbo attention is bias-free; a qkv bias would need the same
            // 3-way split. Reject rather than silently mishandle it.
            return Err(CandleError::Msg(format!(
                "z-image ComfyUI remap: unexpected fused attention bias {key:?} \
                 (Z-Image-Turbo attention is bias-free)"
            )));
        }
        out.insert(remap_dit_key(&key), tensor);
    }
    Ok(out)
}

/// Row-split a fused `[3·DIM, DIM]` projection into its q, k, v thirds.
fn split_fused_qkv(key: &str, qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = qkv.dims();
    if dims != [3 * DIT_DIM, DIT_DIM] {
        return Err(CandleError::Msg(format!(
            "z-image ComfyUI remap: fused {key:?} has shape {dims:?}, expected [{}, {DIT_DIM}]",
            3 * DIT_DIM
        )));
    }
    let q = qkv.narrow(0, 0, DIT_DIM)?.contiguous()?;
    let k = qkv.narrow(0, DIT_DIM, DIT_DIM)?.contiguous()?;
    let v = qkv.narrow(0, 2 * DIT_DIM, DIT_DIM)?.contiguous()?;
    Ok((q, k, v))
}

/// The non-qkv leaf-name renames. Applied to a single key; returns the diffusers
/// spelling (or the key unchanged when no rule matches).
fn remap_dit_key(key: &str) -> String {
    // Per-block attention leaf renames (substring — the block prefix is arbitrary).
    if let Some(rest) = key.strip_suffix(".attention.q_norm.weight") {
        return format!("{rest}.attention.norm_q.weight");
    }
    if let Some(rest) = key.strip_suffix(".attention.k_norm.weight") {
        return format!("{rest}.attention.norm_k.weight");
    }
    if let Some((head, tail)) = split_once_infix(key, ".attention.out.") {
        return format!("{head}.attention.to_out.0.{tail}");
    }
    // Single-aspect head wrappers: ComfyUI collapses the multi-aspect
    // `all_*.<patch>-<f_patch>` path to the bare name; candle keeps `2-1`.
    if let Some(tail) = key.strip_prefix("x_embedder.") {
        return format!("all_x_embedder.2-1.{tail}");
    }
    if let Some(tail) = key.strip_prefix("final_layer.") {
        return format!("all_final_layer.2-1.{tail}");
    }
    key.to_string()
}

// ---------------------------------------------------------------------------
// VAE: BFL/ldm ("flux autoencoder") → diffusers `AutoEncoderKL` keys
// ---------------------------------------------------------------------------

/// Remap a ComfyUI Z-Image VAE tensor map (BFL/ldm "flux autoencoder" layout) to
/// the diffusers `AutoencoderKL` schema `candle_transformers::models::z_image::vae`
/// reads. Z-Image aliases the FLUX.1 latent space, so `vae/ae.safetensors` is the
/// FLUX ldm autoencoder; the diffusers loader wants a different naming.
///
/// Structural renames (values are byte-identical — this is a pure key/shape map):
/// * `{enc,dec}.conv_in`/`conv_out` pass through; `norm_out`→`conv_norm_out`.
/// * `encoder.down.{i}.block.{j}` → `encoder.down_blocks.{i}.resnets.{j}`;
///   `.downsample.conv` → `.downsamplers.0.conv`.
/// * `decoder.up.{i}.block.{j}` → `decoder.up_blocks.{N-1-i}.resnets.{j}` (the
///   **reversed** up-block index); `.upsample.conv` → `.upsamplers.0.conv`.
/// * `{enc,dec}.mid.block_1/block_2` → `mid_block.resnets.0/1`.
/// * `nin_shortcut` → `conv_shortcut`.
/// * mid attention `mid.attn_1`: `norm`→`group_norm`; `q/k/v`→`to_q/to_k/to_v`;
///   `proj_out`→`to_out.0`. The ldm attention projections are Conv2d 1×1
///   (`[C, C, 1, 1]`); diffusers attention is `Linear` (`[C, C]`), so their
///   **weights are squeezed** `[C,C,1,1]→[C,C]` (biases pass through).
///
/// Errors on an attention weight whose trailing dims are not `1×1` (an unexpected
/// VAE shape — surfaced, not reshaped blindly).
pub fn remap_vae_ldm_to_diffusers(src: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(src.len());
    for (key, tensor) in src {
        let new_key = remap_vae_key(&key)?;
        // Squeeze the ldm 1×1-conv attention projection weights to diffusers Linear.
        let tensor = if is_vae_attn_proj_weight(&new_key) {
            squeeze_conv1x1_to_linear(&key, &tensor)?
        } else {
            tensor
        };
        out.insert(new_key, tensor);
    }
    Ok(out)
}

/// Map one ldm VAE key to its diffusers spelling. Splits on the fixed prefixes so
/// the per-level index arithmetic (and the decoder reversal) is explicit.
fn remap_vae_key(key: &str) -> Result<String> {
    for stem in ["encoder", "decoder"] {
        let Some(rest) = key.strip_prefix(&format!("{stem}.")) else {
            continue;
        };
        // Level blocks: `down.{i}.…` (encoder) / `up.{i}.…` (decoder).
        let level_word = if stem == "encoder" { "down" } else { "up" };
        if let Some(after) = rest.strip_prefix(&format!("{level_word}.")) {
            return remap_vae_level(stem, level_word, after);
        }
        // Mid block.
        if let Some(after) = rest.strip_prefix("mid.") {
            return Ok(format!("{stem}.mid_block.{}", remap_vae_mid(after)));
        }
        // Top-level conv_in / conv_out pass through; norm_out → conv_norm_out.
        if let Some(tail) = rest.strip_prefix("norm_out.") {
            return Ok(format!("{stem}.conv_norm_out.{tail}"));
        }
        return Ok(key.to_string());
    }
    Ok(key.to_string())
}

/// `down.{i}.…` / `up.{i}.…` → `{down_blocks|up_blocks}.{idx}.…`, applying the
/// decoder up-block index reversal and `block.j`→`resnets.j`,
/// `downsample/upsample.conv`→`downsamplers/upsamplers.0.conv`,
/// `nin_shortcut`→`conv_shortcut`.
fn remap_vae_level(stem: &str, level_word: &str, after: &str) -> Result<String> {
    let (idx_str, tail) = after.split_once('.').ok_or_else(|| {
        CandleError::Msg(format!(
            "z-image ComfyUI VAE remap: malformed level key {after:?}"
        ))
    })?;
    let idx: usize = idx_str.parse().map_err(|_| {
        CandleError::Msg(format!(
            "z-image ComfyUI VAE remap: bad level index in {after:?}"
        ))
    })?;
    let (blocks_word, sampler_word, block_idx) = if level_word == "down" {
        ("down_blocks", "downsamplers", idx)
    } else {
        // Decoder up-blocks are numbered in reverse of the ldm `up.*`.
        ("up_blocks", "upsamplers", VAE_LEVELS - 1 - idx)
    };
    if let Some(rest) = tail.strip_prefix("block.") {
        let (j, leaf) = rest.split_once('.').ok_or_else(|| {
            CandleError::Msg(format!(
                "z-image ComfyUI VAE remap: malformed block key {tail:?}"
            ))
        })?;
        let leaf = leaf.replacen("nin_shortcut.", "conv_shortcut.", 1);
        return Ok(format!(
            "{stem}.{blocks_word}.{block_idx}.resnets.{j}.{leaf}"
        ));
    }
    if let Some(rest) = tail
        .strip_prefix("downsample.")
        .or_else(|| tail.strip_prefix("upsample."))
    {
        // ldm `.downsample.conv.*` / `.upsample.conv.*` → `.{sampler}.0.conv.*`.
        let rest = rest.strip_prefix("conv.").unwrap_or(rest);
        return Ok(format!(
            "{stem}.{blocks_word}.{block_idx}.{sampler_word}.0.conv.{rest}"
        ));
    }
    Err(CandleError::Msg(format!(
        "z-image ComfyUI VAE remap: unrecognized level sub-key {tail:?}"
    )))
}

/// `mid.…` sub-key → diffusers `mid_block.…`.
fn remap_vae_mid(after: &str) -> String {
    if let Some(tail) = after.strip_prefix("block_1.") {
        return format!("resnets.0.{tail}");
    }
    if let Some(tail) = after.strip_prefix("block_2.") {
        return format!("resnets.1.{tail}");
    }
    if let Some(tail) = after.strip_prefix("attn_1.") {
        let mapped = if let Some(t) = tail.strip_prefix("norm.") {
            format!("group_norm.{t}")
        } else if let Some(t) = tail.strip_prefix("q.") {
            format!("to_q.{t}")
        } else if let Some(t) = tail.strip_prefix("k.") {
            format!("to_k.{t}")
        } else if let Some(t) = tail.strip_prefix("v.") {
            format!("to_v.{t}")
        } else if let Some(t) = tail.strip_prefix("proj_out.") {
            format!("to_out.0.{t}")
        } else {
            tail.to_string()
        };
        return format!("attentions.0.{mapped}");
    }
    after.to_string()
}

/// True for the four diffusers mid-attention projection **weights** whose ldm
/// source is a 1×1 conv that must be squeezed to a Linear.
fn is_vae_attn_proj_weight(diffusers_key: &str) -> bool {
    diffusers_key.contains(".attentions.0.")
        && (diffusers_key.ends_with(".to_q.weight")
            || diffusers_key.ends_with(".to_k.weight")
            || diffusers_key.ends_with(".to_v.weight")
            || diffusers_key.ends_with(".to_out.0.weight"))
}

/// `[C, C, 1, 1]` conv weight → `[C, C]` Linear weight. Errors if the trailing
/// spatial dims are not 1×1.
fn squeeze_conv1x1_to_linear(key: &str, w: &Tensor) -> Result<Tensor> {
    let dims = w.dims();
    match dims {
        [o, i, 1, 1] => Ok(w.reshape((*o, *i))?),
        // Already a Linear (a diffusers-format VAE slipped through) — leave it.
        [_, _] => Ok(w.clone()),
        _ => Err(CandleError::Msg(format!(
            "z-image ComfyUI VAE remap: attention weight {key:?} has shape {dims:?}, \
             expected [C, C, 1, 1] (ldm 1×1 conv) or [C, C] (diffusers Linear)"
        ))),
    }
}

/// `split_once` on a substring: returns the text before and after the first
/// occurrence of `infix`, or `None` when absent.
fn split_once_infix<'a>(s: &'a str, infix: &str) -> Option<(&'a str, &'a str)> {
    let idx = s.find(infix)?;
    Some((&s[..idx], &s[idx + infix.len()..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device, Tensor};

    fn w(dims: &[usize]) -> Tensor {
        Tensor::zeros(dims, DType::F32, &Device::Cpu).unwrap()
    }

    // --- DiT ------------------------------------------------------------------

    #[test]
    fn dit_splits_fused_qkv_into_thirds() {
        let mut src = HashMap::new();
        // A recognizable fused qkv: rows 0..DIM = 1s (q), DIM..2DIM = 2s (k), rest = 3s (v).
        let q = Tensor::full(1f32, (DIT_DIM, DIT_DIM), &Device::Cpu).unwrap();
        let k = Tensor::full(2f32, (DIT_DIM, DIT_DIM), &Device::Cpu).unwrap();
        let v = Tensor::full(3f32, (DIT_DIM, DIT_DIM), &Device::Cpu).unwrap();
        let qkv = Tensor::cat(&[&q, &k, &v], 0).unwrap();
        src.insert("layers.0.attention.qkv.weight".to_string(), qkv);

        let out = remap_dit_comfyui_to_diffusers(src).unwrap();
        assert!(!out.contains_key("layers.0.attention.qkv.weight"));
        for (name, expect) in [("to_q", 1f32), ("to_k", 2f32), ("to_v", 3f32)] {
            let t = out
                .get(&format!("layers.0.attention.{name}.weight"))
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(t.dims(), &[DIT_DIM, DIT_DIM]);
            assert_eq!(
                t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0],
                expect
            );
        }
    }

    #[test]
    fn dit_renames_norms_out_and_head_wrappers() {
        let mut src = HashMap::new();
        src.insert("layers.3.attention.q_norm.weight".into(), w(&[128]));
        src.insert("layers.3.attention.k_norm.weight".into(), w(&[128]));
        src.insert(
            "noise_refiner.1.attention.out.weight".into(),
            w(&[DIT_DIM, DIT_DIM]),
        );
        src.insert("x_embedder.weight".into(), w(&[DIT_DIM, 64]));
        src.insert("x_embedder.bias".into(), w(&[DIT_DIM]));
        src.insert("final_layer.linear.weight".into(), w(&[64, DIT_DIM]));
        src.insert(
            "final_layer.adaLN_modulation.1.weight".into(),
            w(&[DIT_DIM, 256]),
        );
        // Pass-throughs.
        src.insert(
            "layers.3.feed_forward.w1.weight".into(),
            w(&[10240, DIT_DIM]),
        );
        src.insert("cap_embedder.0.weight".into(), w(&[2560]));

        let out = remap_dit_comfyui_to_diffusers(src).unwrap();
        assert!(out.contains_key("layers.3.attention.norm_q.weight"));
        assert!(out.contains_key("layers.3.attention.norm_k.weight"));
        assert!(out.contains_key("noise_refiner.1.attention.to_out.0.weight"));
        assert!(out.contains_key("all_x_embedder.2-1.weight"));
        assert!(out.contains_key("all_x_embedder.2-1.bias"));
        assert!(out.contains_key("all_final_layer.2-1.linear.weight"));
        assert!(out.contains_key("all_final_layer.2-1.adaLN_modulation.1.weight"));
        // Unchanged keys stay put.
        assert!(out.contains_key("layers.3.feed_forward.w1.weight"));
        assert!(out.contains_key("cap_embedder.0.weight"));
    }

    #[test]
    fn dit_rejects_wrongly_shaped_qkv() {
        let mut src = HashMap::new();
        src.insert("layers.0.attention.qkv.weight".into(), w(&[100, 100]));
        assert!(remap_dit_comfyui_to_diffusers(src).is_err());
    }

    // --- VAE ------------------------------------------------------------------

    #[test]
    fn vae_encoder_down_blocks_map_without_reversal() {
        assert_eq!(
            remap_vae_key("encoder.down.0.block.0.conv1.weight").unwrap(),
            "encoder.down_blocks.0.resnets.0.conv1.weight"
        );
        assert_eq!(
            remap_vae_key("encoder.down.2.block.1.nin_shortcut.weight").unwrap(),
            "encoder.down_blocks.2.resnets.1.conv_shortcut.weight"
        );
        assert_eq!(
            remap_vae_key("encoder.down.1.downsample.conv.weight").unwrap(),
            "encoder.down_blocks.1.downsamplers.0.conv.weight"
        );
    }

    #[test]
    fn vae_decoder_up_blocks_are_reversed() {
        // ldm up.0 (highest res, last decode) → diffusers up_blocks.3.
        assert_eq!(
            remap_vae_key("decoder.up.0.block.2.conv2.weight").unwrap(),
            "decoder.up_blocks.3.resnets.2.conv2.weight"
        );
        // ldm up.3 (lowest res, first decode) → diffusers up_blocks.0.
        assert_eq!(
            remap_vae_key("decoder.up.3.upsample.conv.weight").unwrap(),
            "decoder.up_blocks.0.upsamplers.0.conv.weight"
        );
    }

    #[test]
    fn vae_mid_block_and_attention_names() {
        assert_eq!(
            remap_vae_key("encoder.mid.block_1.conv1.weight").unwrap(),
            "encoder.mid_block.resnets.0.conv1.weight"
        );
        assert_eq!(
            remap_vae_key("decoder.mid.block_2.norm2.bias").unwrap(),
            "decoder.mid_block.resnets.1.norm2.bias"
        );
        assert_eq!(
            remap_vae_key("decoder.mid.attn_1.norm.weight").unwrap(),
            "decoder.mid_block.attentions.0.group_norm.weight"
        );
        assert_eq!(
            remap_vae_key("decoder.mid.attn_1.q.weight").unwrap(),
            "decoder.mid_block.attentions.0.to_q.weight"
        );
        assert_eq!(
            remap_vae_key("decoder.mid.attn_1.proj_out.bias").unwrap(),
            "decoder.mid_block.attentions.0.to_out.0.bias"
        );
    }

    #[test]
    fn vae_top_level_conv_and_norm() {
        assert_eq!(
            remap_vae_key("encoder.conv_in.weight").unwrap(),
            "encoder.conv_in.weight"
        );
        assert_eq!(
            remap_vae_key("decoder.conv_out.bias").unwrap(),
            "decoder.conv_out.bias"
        );
        assert_eq!(
            remap_vae_key("encoder.norm_out.weight").unwrap(),
            "encoder.conv_norm_out.weight"
        );
    }

    #[test]
    fn vae_squeezes_conv1x1_attention_weight_to_linear() {
        let mut src = HashMap::new();
        src.insert("decoder.mid.attn_1.q.weight".into(), w(&[512, 512, 1, 1]));
        src.insert("decoder.mid.attn_1.q.bias".into(), w(&[512]));
        src.insert("decoder.mid.attn_1.norm.weight".into(), w(&[512]));

        let out = remap_vae_ldm_to_diffusers(src).unwrap();
        let q = out
            .get("decoder.mid_block.attentions.0.to_q.weight")
            .unwrap();
        assert_eq!(q.dims(), &[512, 512], "1×1 conv weight squeezed to Linear");
        // Bias and group_norm are not reshaped.
        assert_eq!(
            out.get("decoder.mid_block.attentions.0.to_q.bias")
                .unwrap()
                .dims(),
            &[512]
        );
        assert_eq!(
            out.get("decoder.mid_block.attentions.0.group_norm.weight")
                .unwrap()
                .dims(),
            &[512]
        );
    }
}
