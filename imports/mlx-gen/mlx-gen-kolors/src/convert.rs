//! Offline pre-quantization for Kolors (sc-9946, epic 8506): read a dense `Kwai-Kolors/Kolors-diffusers`
//! snapshot and write a packed Q4/Q8 turnkey that [`crate::model::Kolors::load`] loads with no dense
//! fp16 transient and no in-app `.quantize` peak. The bespoke tail of the Group-B converter template
//! ([`mlx_gen_sdxl::convert`] / the shared [`mlx_gen::quant`] primitives) — Kolors' text encoder is
//! **ChatGLM3-6B**, not a CLIP, so it needs its own pack predicate.
//!
//! Kolors quantizes **two** components (matching the load-time seams wired in
//! [`crate::model::Kolors::quantize`]):
//!
//! * **U-Net** — the SDXL [`mlx_gen_sdxl::UNet2DConditionModel`] reused as-is (`load_unet_kolors_dtype`),
//!   already shipped in the SDXL quant matrix (sc-8513). Every 2-D Linear packs; the shared
//!   [`quantize_map`] shape guard (2-D, `in % gs == 0`, `in >= gs`) skips the 1-D GroupNorms and the
//!   4-D convs (incl. the 1×1 `conv_shortcut`). The Kolors `encoder_hid_proj` (4096→2048) packs like
//!   any other Linear. The SDXL loader already **packed-detects** (`is_packed` guard → `quant::lin`),
//!   so no engine change — this converter just has to pack the raw diffusers `unet/`.
//! * **ChatGLM3-6B text encoder** — pack the four GLM-block projections
//!   ([`is_chatglm_target`]); the token **embedding** and every `*_layernorm` stay dense (matching
//!   [`crate::chatglm3::ChatGlmModel::quantize`], which leaves the embedding dense and never touches
//!   the norms). [`crate::chatglm3::ChatGlmLinear::load`] packed-detects the result.
//!
//! The **VAE is never quantized** (the SDXL VAE runs f32 — fp16/int8-unstable), so every tier ships a
//! dense VAE (mirror the source `vae/`). The **tokenizer** is baked: ChatGLM3 ships only a *slow*
//! tokenizer upstream, so the derived fast `tokenizer.json` (`SceneWorks/kolors-chatglm3-tokenizer`,
//! sc-4764) is written into each tier's `tokenizer/` — the per-tier turnkey layout does not hit the
//! worker's install-time overlay, so it must travel with the bytes.
//!
//! The packed triple is byte-identical to the load-time `.quantize` (shared [`quantize_map`], bf16
//! cast, group [`DEFAULT_GROUP_SIZE`]); the completeness gate is the real-weight render in
//! `tests/prequantize_real_weights.rs` (a missed site loads u32 codes as dense floats → a flat render).

use std::path::Path;

use mlx_gen::quant::{
    copy_asset, copy_dir, copy_turnkey_assets, load_dir_map, quantize_map, save_map,
    write_quantized_config, DEFAULT_GROUP_SIZE,
};
use mlx_gen::{Error, Result};

/// Kolors packs at the codebase-default group size (64) — the same
/// [`crate::chatglm3::ChatGlmModel::quantize`] and the SDXL U-Net use, so the offline pack is
/// byte-identical to the load-time seam.
const GROUP_SIZE: i32 = DEFAULT_GROUP_SIZE;

// ============================================================================================
// Pack predicates (operate on the **base** = the on-disk key minus its `.weight`).
// ============================================================================================

/// ChatGLM3 text encoder: pack the four GLM-block projections — the fused `query_key_value`, the
/// attention output `dense`, and the two MLP linears (`dense_h_to_4h` fused gate+up, `dense_4h_to_h`).
/// The token embedding (`embedding.word_embeddings`, a 2-D gather that stays dense) and every
/// `input_layernorm`/`post_attention_layernorm`/`final_layernorm` (1-D norms) are left dense — exactly
/// [`crate::chatglm3::ChatGlmModel::quantize`]'s scope. Operates on the raw diffusers key minus
/// `.weight`.
fn is_chatglm_target(base: &str) -> bool {
    base.ends_with(".self_attention.query_key_value")
        || base.ends_with(".self_attention.dense")
        || base.ends_with(".mlp.dense_h_to_4h")
        || base.ends_with(".mlp.dense_4h_to_h")
}

/// U-Net: pack every quantizable leaf, leaning on the shared [`quantize_map`] shape guard to skip the
/// 1-D norms and the 4-D convs (incl. the 1×1 `conv_shortcut`). Every 2-D `.weight` in the SDXL U-Net
/// is a quantized Linear (time/add-embedding MLPs, attention, GEGLU FFN, `proj_in`/`proj_out`, each
/// ResNet `time_emb_proj`, the Kolors `encoder_hid_proj`), so "pack all shape-eligible" is exactly the
/// load-time `.quantize` scope — identical to [`mlx_gen_sdxl::convert`]'s U-Net predicate.
fn is_unet_target(_base: &str) -> bool {
    true
}

/// Pre-quantize one component dir → a packed single `{out_stem}.safetensors` + annotated `config.json`
/// in `dst`. The whole dir is read (all `*.safetensors` shards merged — the Kolors `text_encoder/` is a
/// 3-shard fp16 ChatGLM3, the `unet/` a single fp16 file) and packed into ONE output file named so the
/// loader's `resolve_weight_file` / `Weights::from_dir` finds it exactly as the dense master
/// (`unet/diffusion_pytorch_model.safetensors`, `text_encoder/model.safetensors`). `is_target` is the
/// component's pack predicate; `bits` = 4 (Q4) or 8 (Q8) at group size [`GROUP_SIZE`].
fn quantize_component(
    src: &Path,
    dst: &Path,
    out_stem: &str,
    bits: i32,
    is_target: fn(&str) -> bool,
) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, is_target)?;
    save_map(&dst.join(format!("{out_stem}.safetensors")), &map)?;
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

/// Assemble a full pre-quantized turnkey Kolors snapshot in `dst_root`: pack the SDXL U-Net + the
/// ChatGLM3 text encoder, mirror the dense VAE, copy the scheduler, and bake the tokenizer (the derived
/// fast `tokenizer.json` from `tokenizer_json`, plus the source `tokenizer/` assets). The result loads
/// via [`crate::model::Kolors::load`] (packed weights auto-detect) with no dense transient. `bits` = 4
/// (Q4 tier) or 8 (Q8 tier). The bf16 tier is the dense source itself (no conversion — mirror it).
///
/// `tokenizer_json` is the derived fast tokenizer.json (`SceneWorks/kolors-chatglm3-tokenizer`); pass
/// `None` only when the source `tokenizer/` already contains a `tokenizer.json` (an app-installed
/// snapshot that ran the worker overlay). A tier without `tokenizer/tokenizer.json` is unloadable
/// ([`crate::tokenizer::KolorsTokenizer::from_dir`] requires it), so this is an error.
pub fn prequantize_turnkey(
    src_root: &Path,
    dst_root: &Path,
    bits: i32,
    tokenizer_json: Option<&Path>,
) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;

    // U-Net (pack-all, shape-guarded) — the SDXL loader packed-detects it.
    quantize_component(
        &src_root.join("unet"),
        &dst_root.join("unet"),
        "diffusion_pytorch_model",
        bits,
        is_unet_target,
    )?;
    // ChatGLM3-6B text encoder (the four GLM projections; embedding + norms stay dense).
    quantize_component(
        &src_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        "model",
        bits,
        is_chatglm_target,
    )?;

    // VAE stays dense (never quantized) — mirror the source `vae/` verbatim (deref symlinks).
    let vae_src = src_root.join("vae");
    if vae_src.exists() {
        copy_dir(&vae_src, &dst_root.join("vae"))?;
    }
    // Scheduler is a non-weight asset — copy verbatim.
    let sched_src = src_root.join("scheduler");
    if sched_src.exists() {
        copy_dir(&sched_src, &dst_root.join("scheduler"))?;
    }

    // Tokenizer: copy the source assets (slow tokenizer.model / vocab / config), then bake the derived
    // fast tokenizer.json (ChatGLM3 ships slow-only; the fast one is required to load).
    let tok_src = src_root.join("tokenizer");
    let tok_dst = dst_root.join("tokenizer");
    if tok_src.exists() {
        copy_dir(&tok_src, &tok_dst)?;
    }
    if let Some(tj) = tokenizer_json {
        std::fs::create_dir_all(&tok_dst)?;
        let real = std::fs::canonicalize(tj)?;
        std::fs::copy(&real, tok_dst.join("tokenizer.json"))?;
    }
    if !tok_dst.join("tokenizer.json").exists() {
        return Err(Error::Msg(
            "kolors convert: tier tokenizer/ lacks tokenizer.json — pass the derived \
             SceneWorks/kolors-chatglm3-tokenizer tokenizer.json (ChatGLM3 ships slow-only upstream)"
                .into(),
        ));
    }

    // Kolors' license file is `MODEL_LICENSE` (not the LICENSE* names the shared helper covers), so
    // carry it explicitly; `copy_turnkey_assets` handles `model_index.json` + `README.md`.
    copy_asset(src_root, dst_root, "MODEL_LICENSE")?;
    copy_turnkey_assets(src_root, dst_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::{Array, Dtype};
    use std::collections::HashMap;

    #[test]
    fn chatglm_predicate_packs_projections_skips_embedding_and_norms() {
        for base in [
            "encoder.layers.0.self_attention.query_key_value",
            "encoder.layers.0.self_attention.dense",
            "encoder.layers.13.mlp.dense_h_to_4h",
            "encoder.layers.27.mlp.dense_4h_to_h",
        ] {
            assert!(is_chatglm_target(base), "{base} should be packed");
        }
        for base in [
            "embedding.word_embeddings",
            "encoder.layers.0.input_layernorm",
            "encoder.layers.0.post_attention_layernorm",
            "encoder.final_layernorm",
        ] {
            assert!(!is_chatglm_target(base), "{base} should stay dense");
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The packed triple a ChatGLM projection becomes is byte-identical to the op the load-time
    /// `.quantize` runs (bf16 cast, group 64) — the pre-quantize-on-disk == quantize-at-load
    /// guarantee. The embedding (2-D, shape-eligible) stays dense via the predicate; a 1-D norm stays
    /// dense via the shape guard.
    #[test]
    fn quantize_map_packs_chatglm_projections_byte_identical() {
        // A fused query_key_value Linear ([out=4608-ish, in=256]) that packs.
        let w = Array::from_slice(
            &(0..192 * 256).map(|i| (i as f32).cos()).collect::<Vec<_>>(),
            &[192, 256],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert(
            "encoder.layers.0.self_attention.query_key_value.weight".into(),
            w.clone(),
        );
        // The fused qkv also carries a Linear bias (1-D) — must pass through untouched (not packed).
        map.insert(
            "encoder.layers.0.self_attention.query_key_value.bias".into(),
            Array::zeros::<f32>(&[192]).unwrap(),
        );
        // The token embedding is 2-D + shape-eligible but the predicate keeps it dense.
        map.insert(
            "embedding.word_embeddings.weight".into(),
            Array::zeros::<f32>(&[512, 256]).unwrap(),
        );
        // A 1-D RMSNorm weight stays dense (shape-guarded out).
        map.insert(
            "encoder.layers.0.input_layernorm.weight".into(),
            Array::ones::<f32>(&[256]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_chatglm_target).unwrap();

        let base = "encoder.layers.0.self_attention.query_key_value";
        let wq = out.get(&format!("{base}.weight")).expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "codes != load-time quantize");
        assert!(
            byte_equal(out.get(&format!("{base}.scales")).unwrap(), &esc),
            "scales != load-time quantize"
        );
        assert!(
            byte_equal(out.get(&format!("{base}.biases")).unwrap(), &ebi),
            "affine biases != load-time quantize"
        );
        // The Linear bias passes through unchanged (dense f32, no packed triple).
        let bias = out
            .get(&format!("{base}.bias"))
            .expect("bias passes through");
        assert_eq!(bias.dtype(), Dtype::Float32);

        // The embedding stays dense (predicate) — no packed sidecars.
        let emb = out.get("embedding.word_embeddings.weight").unwrap();
        assert_eq!(emb.dtype(), Dtype::Float32, "embedding stays dense");
        assert!(!out.contains_key("embedding.word_embeddings.scales"));
        // The 1-D norm stays dense.
        let n = out.get("encoder.layers.0.input_layernorm.weight").unwrap();
        assert_eq!(n.dtype(), Dtype::Float32, "norm stays dense");
    }
}
