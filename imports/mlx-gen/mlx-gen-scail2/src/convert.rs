//! SCAIL-2 DiT **pre-quantization** (sc-5445): pack the dense bf16 `dit.safetensors` into a
//! group-wise-affine Q4/Q8 `dit.safetensors` *on disk*, plus a `config.json` `quantization` manifest,
//! so the published snapshot loads straight into packed quantized Linears.
//!
//! This is the low-memory-floor path. Quantizing **at load** (`Scail2Dit::quantize`) still has to
//! materialize the full ~33 GB bf16 DiT before packing, so the load-time peak (and therefore
//! `minMemoryGb`) is the *bf16* footprint. Quantizing **on disk** here moves that bf16 transient to a
//! one-off offline convert; the shipped snapshot is already packed, so the consume side
//! ([`crate::model::load_lin_q`]) never builds a dense weight — the resident set is the Q4 packs.
//!
//! Mirrors `mlx_gen_wan::convert::quantize_wan_transformer` (same `mlx_rs::ops::quantize`, byte-equal
//! to `nn.quantize`), differing only in the SCAIL-2 key layout (the FFN is `ffn.0` / `ffn.2`, and the
//! I2V cross-attention adds `k_img` / `v_img`).
//!
//! Run the offline convert on macOS against the assembled bf16 snapshot — see
//! `tests/quantize_snapshot.rs` (`#[ignore]`, needs the real ~33 GB DiT).

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::quant::{quantize_map, save_map};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::Array;

/// The SCAIL-2 DiT `_quantize_predicate`: a Linear is quantized iff its weight key (minus `.weight`)
/// ends with one of these — every block's self/cross-attention `q/k/v/o`, the I2V cross-attention
/// `k_img/v_img`, and the FFN `ffn.0` / `ffn.2`. The patch / text / time / image embeddings,
/// `time_projection`, modulation tables, qk-/`norm3`-/LayerNorm norms, and the output head stay dense
/// (small + precision-sensitive — the reference skips them). Must stay in lockstep with the
/// quant-aware call sites in [`crate::model`].
pub const SCAIL2_QUANT_SUFFIXES: &[&str] = &[
    ".self_attn.q",
    ".self_attn.k",
    ".self_attn.v",
    ".self_attn.o",
    ".cross_attn.q",
    ".cross_attn.k",
    ".cross_attn.v",
    ".cross_attn.o",
    ".cross_attn.k_img",
    ".cross_attn.v_img",
    ".ffn.0",
    ".ffn.2",
];

/// `true` iff `weight_key` (an `…​.weight` tensor name) names a [`SCAIL2_QUANT_SUFFIXES`] Linear.
fn is_quant_target(weight_key: &str) -> bool {
    weight_key
        .strip_suffix(".weight")
        .is_some_and(|base| SCAIL2_QUANT_SUFFIXES.iter().any(|s| base.ends_with(s)))
}

/// Selectively Q4/Q8-quantize a SCAIL-2 DiT weight map in place: each [`SCAIL2_QUANT_SUFFIXES`]-matched
/// `{base}.weight` (cast to bf16 for fork parity, matching [`AdaptableLinear::quantize`]) becomes the
/// packed triple `{base}.weight` (u32 codes), `{base}.scales`, `{base}.biases` via MLX `quantize`; the
/// Linear's dense `{base}.bias` and every other tensor pass through unchanged. The result is the exact
/// key layout [`crate::model::load_lin_q`] reads back.
///
/// [`AdaptableLinear::quantize`]: mlx_gen::adapters::AdaptableLinear::quantize
pub fn quantize_scail2_transformer(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
) -> Result<HashMap<String, Array>> {
    // The shared packer (bf16-cast → `quantize`, byte-identical to the load-time
    // `AdaptableLinear::quantize`); `is_quant_target` receives the full `.weight` key.
    quantize_map(map, bits, group_size, |base| {
        is_quant_target(&format!("{base}.weight"))
    })
}

/// Read every tensor of `path` into an owned key→`Array` map (MLX arrays are ref-counted, so the
/// clone is a handle copy, not a buffer copy).
fn load_map(path: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_file(path)?;
    Ok(w.keys()
        .map(|k| (k.to_string(), w.get(k).expect("listed key").clone()))
        .collect())
}

/// Copy `src/config.json` to `dst/config.json` with a `"quantization": {"bits", "group_size"}` block
/// added (the manifest [`crate::config::Scail2Config::from_model_dir`] reads to enable the packed
/// loader). A missing source config starts from an empty object.
fn write_quantized_config(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    let src_cfg = src.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&std::fs::read_to_string(&src_cfg)?)
            .map_err(|e| Error::Msg(format!("scail2: parse {}: {e}", src_cfg.display())))?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Msg(format!("scail2: serialize config.json: {e}")))?;
    std::fs::write(dst.join("config.json"), text)?;
    Ok(())
}

/// Offline one-shot: read the dense bf16 `src/dit.safetensors` (+ `src/config.json`) and write a
/// pre-quantized `dst/dit.safetensors` (packed Q4/Q8) + `dst/config.json` (with the `quantization`
/// manifest). The VAE / UMT5 / CLIP / tokenizer are unchanged — the caller copies or symlinks them
/// alongside to complete the turnkey snapshot. `group_size` is the mflux/reference default of 64.
pub fn quantize_scail2_dit(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = load_map(&src.join("dit.safetensors"))?;
    let quantized = quantize_scail2_transformer(map, bits, group_size)?;
    save_map(&dst.join("dit.safetensors"), &quantized)?;
    write_quantized_config(src, dst, bits, group_size)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    #[test]
    fn predicate_matches_attn_ffn_only() {
        assert!(is_quant_target("blocks.0.self_attn.q.weight"));
        assert!(is_quant_target("blocks.39.cross_attn.k_img.weight"));
        assert!(is_quant_target("blocks.7.ffn.0.weight"));
        assert!(is_quant_target("blocks.7.ffn.2.weight"));
        // Dense surface: embeddings / projection / norms / head / biases.
        assert!(!is_quant_target("patch_embedding.weight"));
        assert!(!is_quant_target("text_embedding.0.weight"));
        assert!(!is_quant_target("time_projection.1.weight"));
        assert!(!is_quant_target("head.head.weight"));
        assert!(!is_quant_target("blocks.0.self_attn.norm_q.weight"));
        assert!(!is_quant_target("blocks.0.self_attn.q.bias"));
    }

    #[test]
    fn quantizes_predicate_keys_and_passes_through_rest() {
        // A predicate Linear (`in` a multiple of the group size) + its bias + a non-predicate weight.
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert(
            "blocks.0.self_attn.q.weight".into(),
            Array::ones::<f32>(&[64, 128]).unwrap(),
        );
        map.insert(
            "blocks.0.self_attn.q.bias".into(),
            Array::zeros::<f32>(&[64]).unwrap(),
        );
        map.insert(
            "head.head.weight".into(),
            Array::ones::<f32>(&[16, 64]).unwrap(),
        );

        let out = quantize_scail2_transformer(map, 4, 64).unwrap();

        // The predicate weight became the packed triple (u32 codes + scales + biases)…
        let wq = out
            .get("blocks.0.self_attn.q.weight")
            .expect("packed weight");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        assert!(out.contains_key("blocks.0.self_attn.q.scales"));
        assert!(out.contains_key("blocks.0.self_attn.q.biases"));
        // …its dense bias and the non-predicate weight passed through untouched.
        assert!(out.contains_key("blocks.0.self_attn.q.bias"));
        let head = out.get("head.head.weight").expect("dense head weight");
        assert_eq!(
            head.dtype(),
            Dtype::Float32,
            "non-predicate weight unchanged"
        );
        assert!(!out.contains_key("head.head.scales"));
    }
}
