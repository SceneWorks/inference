//! Offline pre-quantization: read a dense Chroma diffusers snapshot and write a packed Q4/Q8 turnkey
//! that [`crate::quant`] (via [`crate::model::load_chroma`]) loads with no dense bf16/f32 transient.
//! Mirrors `mlx_gen_sdxl::convert` / `mlx_gen_sensenova::convert` (same `mlx_gen::quant::quantize_map`,
//! byte-equal to the load-time `.quantize` seam), differing in the Chroma key layout and quant scope.
//!
//! Chroma's shipping Q4/Q8 tiers pack all three resident model components: the DiT `transformer/`,
//! the T5-XXL `text_encoder/`, and the FLUX.1 `vae/`.
//!
//! * The transformer's own `x_embedder` / `context_embedder` / top-level `proj_out` and the entire
//!   distilled-guidance **Approximator** (`distilled_guidance_layer.*`, which drives all per-block
//!   modulation) — small / precision-sensitive, kept dense to match `is_transformer_target`.
//! * T5 progressively packs every group-quantizable 2-D weight as a primary Q8 term plus a packed
//!   reconstruction residual. Most attention/FFN projections use Q4 residuals; the shared token
//!   embedding, relative-position bias, calibrated block-4 attention, and block-1 feed-forward
//!   projections use small Q8 residuals. RMSNorm scales remain dense because affine quantization
//!   does not target vectors.
//! * The otherwise-convolutional VAE packs its encoder/decoder mid-block attention projections.
//!
//! The per-component pack predicate matches the loader's `.quantize` scope exactly — a missed site (or
//! a wrongly-packed dense tensor) loads u32 codes as dense floats → a garbage render. The completeness
//! gate is the real-weight render in `tests/prequantize_real_weights.rs`.
//!
//! Group-B per-crate converter template (sc-8669 / sc-8777).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use mlx_gen::quant::{
    copy_dir, copy_turnkey_assets, quantize_map, quantize_map_with_residual_policy, save_map,
    write_quantized_config,
};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::quant::GROUP_SIZE;

/// The single packed weight file the turnkey ships for the transformer (replaces the source's sharded
/// `diffusion_pytorch_model-0000N-of-0000M.safetensors`). The loader globs `*.safetensors` under
/// `transformer/`, so one flat file suffices; its stem matches the dense master so nothing downstream
/// changes.
const TRANSFORMER_FILE: &str = "diffusion_pytorch_model.safetensors";
const AUXILIARY_FILE: &str = "model.safetensors";
/// Chroma's immutable packed auxiliary width. Q4/Q8 is a transformer tier choice; T5 and VAE use
/// Q8 in both tiers to preserve the measured image-quality envelope.
pub const AUXILIARY_BITS: i32 = 8;
/// Chroma's immutable T5 affine group size, selected by the hosted quality calibration.
pub const T5_GROUP_SIZE: i32 = 32;
/// The second-stage packed correction applied to Chroma's Q8 T5 surface.
pub const T5_RESIDUAL_BITS: i32 = 4;
/// The higher-fidelity correction used for the calibrated sensitive T5 surface.
pub const T5_SENSITIVE_RESIDUAL_BITS: i32 = 8;
/// The attention block selected by the hosted all-block sensitivity sweep.
pub const T5_SENSITIVE_RESIDUAL_BLOCK: usize = 4;
/// The feed-forward block selected as the next smallest packed correction by the same sweep.
pub const T5_SENSITIVE_RESIDUAL_FFN_BLOCK: usize = 1;
/// Exact packed bases whose small Q8 residuals protect the T5 boundaries and calibrated sublayers
/// while keeping every source weight packed.
pub const T5_SENSITIVE_RESIDUAL_BASES: &[&str] = &[
    "shared",
    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias",
    "encoder.block.4.layer.0.SelfAttention.q",
    "encoder.block.4.layer.0.SelfAttention.k",
    "encoder.block.4.layer.0.SelfAttention.v",
    "encoder.block.4.layer.0.SelfAttention.o",
    "encoder.block.1.layer.1.DenseReluDense.wi_0",
    "encoder.block.1.layer.1.DenseReluDense.wi_1",
    "encoder.block.1.layer.1.DenseReluDense.wo",
];

pub fn t5_residual_bits_for(base: &str) -> i32 {
    if T5_SENSITIVE_RESIDUAL_BASES.contains(&base) {
        T5_SENSITIVE_RESIDUAL_BITS
    } else {
        T5_RESIDUAL_BITS
    }
}

// ============================================================================================
// Pack predicate (operates on the **base** = the on-disk key minus its `.weight`).
// ============================================================================================

/// Whether a `transformer/` key's `base` is a **block Linear** the DiT quantizes — matching
/// [`crate::transformer::ChromaTransformer::quantize`] exactly:
///
/// * a **double** block (`transformer_blocks.{i}`): attention `to_q`/`to_k`/`to_v`/`to_out.0`,
///   `add_q_proj`/`add_k_proj`/`add_v_proj`/`to_add_out`, and the FFN `ff.net.0.proj`/`ff.net.2` +
///   `ff_context.net.0.proj`/`ff_context.net.2`;
/// * a **single** block (`single_transformer_blocks.{i}`): attention `to_q`/`to_k`/`to_v`, plus
///   `proj_mlp` and `proj_out`.
///
/// Everything else stays dense: the per-block QK-norms / added-norms (1-D `norm_*.weight`, also
/// shape-guarded out by [`quantize_map`]), and every top-level module (`x_embedder`,
/// `context_embedder`, the top-level `proj_out`, and the whole `distilled_guidance_layer.*`
/// Approximator). The `single_*` prefix is tested before the `transformer_blocks.` prefix so a single
/// block (which also starts with `…transformer_blocks.` textually) is classified by its own rule.
fn is_transformer_target(base: &str) -> bool {
    if let Some(rest) = base.strip_prefix("single_transformer_blocks.") {
        // rest = `{i}.<tail>`
        let Some((_i, tail)) = rest.split_once('.') else {
            return false;
        };
        return matches!(
            tail,
            "attn.to_q" | "attn.to_k" | "attn.to_v" | "proj_mlp" | "proj_out"
        );
    }
    if let Some(rest) = base.strip_prefix("transformer_blocks.") {
        let Some((_i, tail)) = rest.split_once('.') else {
            return false;
        };
        return matches!(
            tail,
            "attn.to_q"
                | "attn.to_k"
                | "attn.to_v"
                | "attn.to_out.0"
                | "attn.add_q_proj"
                | "attn.add_k_proj"
                | "attn.add_v_proj"
                | "attn.to_add_out"
                | "ff.net.0.proj"
                | "ff.net.2"
                | "ff_context.net.0.proj"
                | "ff_context.net.2"
        );
    }
    false
}

/// T5-XXL's Chroma policy packs every group-quantizable 2-D weight. The shared [`quantize_map`]
/// shape guard keeps 1-D LayerNorm/RMSNorm vectors dense.
fn is_t5_target(_base: &str) -> bool {
    true
}

/// FLUX.1 VAE packed surface: encoder/decoder mid-block attention QKV/out projections. Convolutions
/// and GroupNorms remain dense; the shared Z-Image VAE loader packed-detects these keys.
pub(crate) fn is_vae_target(base: &str) -> bool {
    base.ends_with(".to_q")
        || base.ends_with(".to_k")
        || base.ends_with(".to_v")
        || base.ends_with(".to_out.0")
}

/// Load a component dir's safetensors (single or sharded) into one key→`Array` map. Chroma ships the
/// transformer as sharded `diffusion_pytorch_model-0000N-of-0000M.safetensors`; the shard keys are
/// disjoint, so we merge them (a duplicate key across shards is a corrupt snapshot → error).
fn load_component_map(dir: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_dir(dir)?;
    let mut map: HashMap<String, Array> = HashMap::new();
    for k in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let v = w.get(&k).expect("listed key").clone();
        if map.insert(k.clone(), v).is_some() {
            return Err(Error::Msg(format!(
                "chroma convert: duplicate key `{k}` across shards in {}",
                dir.display()
            )));
        }
    }
    Ok(map)
}

fn quantize_component(
    src: &Path,
    dst: &Path,
    file: &str,
    bits: i32,
    group_size: i32,
    is_target: fn(&str) -> bool,
) -> Result<()> {
    if !src.is_dir() {
        return Err(Error::Msg(format!(
            "chroma convert: source snapshot has no {} component",
            src.display()
        )));
    }
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_component_map(src)?, bits, group_size, is_target)?;
    save_map(&dst.join(file), &map)?;
    write_quantized_config(src, dst, bits, group_size)
}

fn quantize_t5_component(src: &Path, dst: &Path, bits: i32, group_size: i32) -> Result<()> {
    if !src.is_dir() {
        return Err(Error::Msg(format!(
            "chroma convert: source snapshot has no {} component",
            src.display()
        )));
    }
    std::fs::create_dir_all(dst)?;
    let map = quantize_map_with_residual_policy(
        load_component_map(src)?,
        bits,
        group_size,
        is_t5_target,
        t5_residual_bits_for,
    )?;
    save_map(&dst.join(AUXILIARY_FILE), &map)?;
    write_quantized_config(src, dst, bits, group_size)?;
    let config_path = dst.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?).map_err(|error| {
            Error::Msg(format!(
                "chroma convert: parse {} after quantization: {error}",
                config_path.display()
            ))
        })?;
    config["quantization"]["residual_bits"] = serde_json::json!(T5_RESIDUAL_BITS);
    config["quantization"]["sensitive_residual_bits"] =
        serde_json::json!(T5_SENSITIVE_RESIDUAL_BITS);
    config["quantization"]["sensitive_residual_bases"] =
        serde_json::json!(T5_SENSITIVE_RESIDUAL_BASES);
    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&config).map_err(|error| {
            Error::Msg(format!(
                "chroma convert: serialize {}: {error}",
                config_path.display()
            ))
        })?,
    )?;
    Ok(())
}

/// Assemble a full pre-quantized turnkey Chroma snapshot in `dst_root`: pack the DiT `transformer/`
/// block Linears into one `transformer/diffusion_pytorch_model.safetensors` (+ annotated
/// `config.json`), progressively pack T5 into `text_encoder/model.safetensors`, pack the VAE
/// attention into `vae/model.safetensors`, and copy the tokenizer / scheduler / `model_index.json` /
/// license. The
/// result loads via
/// [`crate::model::load_chroma`] (packed weights auto-detect) with no dense transient. `bits` = 4 (Q4
/// tier) or 8 (Q8 tier). The **bf16 tier** is the dense source itself (no conversion — mirror it; see
/// the tier builder in `tests/prequantize_real_weights.rs`).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    prequantize_turnkey_with_t5_group_size(src_root, dst_root, bits, T5_GROUP_SIZE)
}

/// Compatibility seam for callers that used the calibration-era API. Shipping policy is immutable,
/// so any group size other than [`T5_GROUP_SIZE`] is rejected rather than producing an artifact the
/// Chroma loader should not accept.
pub fn prequantize_turnkey_with_t5_group_size(
    src_root: &Path,
    dst_root: &Path,
    bits: i32,
    t5_group_size: i32,
) -> Result<()> {
    if !matches!(bits, 4 | 8) {
        return Err(Error::Msg(format!(
            "chroma convert: bits must be 4 or 8, got {bits}"
        )));
    }
    if t5_group_size != T5_GROUP_SIZE {
        return Err(Error::Msg(format!(
            "chroma convert: T5 group size must be {T5_GROUP_SIZE}, got {t5_group_size}"
        )));
    }
    if dst_root.exists() {
        return Err(Error::Msg(format!(
            "chroma convert: destination already exists; refusing to mix packed output with stale files: {}",
            dst_root.display()
        )));
    }
    let parent = dst_root.parent().ok_or_else(|| {
        Error::Msg(format!(
            "chroma convert: destination has no parent: {}",
            dst_root.display()
        ))
    })?;
    std::fs::create_dir_all(parent)?;
    let name = dst_root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::Msg("chroma convert: destination has no UTF-8 file name".into()))?;
    let staging = parent.join(format!(".{name}.sceneworks-pack-{}", std::process::id()));
    if staging.exists() {
        return Err(Error::Msg(format!(
            "chroma convert: staging destination already exists: {}",
            staging.display()
        )));
    }
    let result = prequantize_turnkey_into(src_root, &staging, bits, t5_group_size);
    if let Err(error) = result {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    if let Err(error) = std::fs::rename(&staging, dst_root) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error.into());
    }
    Ok(())
}

fn prequantize_turnkey_into(
    src_root: &Path,
    dst_root: &Path,
    bits: i32,
    t5_group_size: i32,
) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;

    // Transformer: pack the block Linears into one flat file + annotate config.
    let tr_src = src_root.join("transformer");
    if !tr_src.is_dir() {
        return Err(Error::Msg(format!(
            "chroma convert: source snapshot {} has no transformer/ dir",
            src_root.display()
        )));
    }
    let tr_dst = dst_root.join("transformer");
    std::fs::create_dir_all(&tr_dst)?;
    let map = quantize_map(
        load_component_map(&tr_src)?,
        bits,
        GROUP_SIZE,
        is_transformer_target,
    )?;
    save_map(&tr_dst.join(TRANSFORMER_FILE), &map)?;
    write_quantized_config(&tr_src, &tr_dst, bits, GROUP_SIZE)?;

    quantize_t5_component(
        &src_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        AUXILIARY_BITS,
        t5_group_size,
    )?;
    quantize_component(
        &src_root.join("vae"),
        &dst_root.join("vae"),
        AUXILIARY_FILE,
        AUXILIARY_BITS,
        GROUP_SIZE,
        is_vae_target,
    )?;

    // Tokenizer and scheduler carry no quantizable weights.
    for rel in ["tokenizer", "scheduler"] {
        let s = src_root.join(rel);
        if s.exists() {
            copy_dir(&s, &dst_root.join(rel))?;
        }
    }
    copy_turnkey_assets(src_root, dst_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{dequantize, eq, quantize, subtract};
    use mlx_rs::{Array, Dtype};

    #[test]
    fn turnkey_refuses_invalid_bits_and_existing_destinations() {
        let missing = Path::new("definitely-missing-chroma-source");
        let invalid = prequantize_turnkey(missing, Path::new("unused-chroma-output"), 3)
            .unwrap_err()
            .to_string();
        assert!(invalid.contains("bits must be 4 or 8"));

        let invalid_group = prequantize_turnkey_with_t5_group_size(
            missing,
            Path::new("unused-chroma-output"),
            8,
            64,
        )
        .unwrap_err()
        .to_string();
        assert!(invalid_group.contains("T5 group size must be 32"));

        let destination = std::env::temp_dir().join(format!(
            "mlx-gen-chroma-existing-destination-{}",
            std::process::id()
        ));
        if destination.exists() {
            std::fs::remove_dir_all(&destination).unwrap();
        }
        std::fs::create_dir(&destination).unwrap();
        let existing = prequantize_turnkey(missing, &destination, 4)
            .unwrap_err()
            .to_string();
        assert!(existing.contains("destination already exists"));
        std::fs::remove_dir(&destination).unwrap();
    }

    #[test]
    fn predicate_matches_block_linears_only() {
        // Double-block Linears (attention + FFN) → packed.
        for base in [
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_v",
            "transformer_blocks.0.attn.to_out.0",
            "transformer_blocks.0.attn.add_q_proj",
            "transformer_blocks.0.attn.add_k_proj",
            "transformer_blocks.0.attn.add_v_proj",
            "transformer_blocks.0.attn.to_add_out",
            "transformer_blocks.18.ff.net.0.proj",
            "transformer_blocks.18.ff.net.2",
            "transformer_blocks.18.ff_context.net.0.proj",
            "transformer_blocks.18.ff_context.net.2",
            // Single-block Linears (attention + proj_mlp/proj_out) → packed.
            "single_transformer_blocks.0.attn.to_q",
            "single_transformer_blocks.0.attn.to_k",
            "single_transformer_blocks.0.attn.to_v",
            "single_transformer_blocks.37.proj_mlp",
            "single_transformer_blocks.37.proj_out",
        ] {
            assert!(is_transformer_target(base), "{base} should pack");
        }
        // Everything else stays dense: per-block norms, top-level embedders/proj_out, Approximator.
        for base in [
            "transformer_blocks.0.attn.norm_q",
            "transformer_blocks.0.attn.norm_k",
            "transformer_blocks.0.attn.norm_added_q",
            "transformer_blocks.0.attn.norm_added_k",
            "single_transformer_blocks.0.attn.norm_q",
            "x_embedder",
            "context_embedder",
            "proj_out",
            "distilled_guidance_layer.in_proj",
            "distilled_guidance_layer.out_proj",
            "distilled_guidance_layer.layers.0.linear_1",
            "distilled_guidance_layer.layers.4.linear_2",
        ] {
            assert!(!is_transformer_target(base), "{base} should stay dense");
        }
    }

    #[test]
    fn auxiliary_component_predicates_cover_the_complete_packed_surface() {
        for base in [
            "encoder.block.0.layer.0.SelfAttention.q",
            "encoder.block.4.layer.0.SelfAttention.q",
            "encoder.block.23.layer.1.DenseReluDense.wi_0",
            "encoder.block.23.layer.1.DenseReluDense.wi_1",
            "encoder.block.23.layer.1.DenseReluDense.wo",
            "shared",
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias",
        ] {
            assert!(
                is_t5_target(base),
                "{base} should be considered for packing"
            );
        }

        for base in [
            "decoder.mid_block.attentions.0.to_q",
            "decoder.mid_block.attentions.0.to_k",
            "decoder.mid_block.attentions.0.to_v",
            "decoder.mid_block.attentions.0.to_out.0",
            "encoder.mid_block.attentions.0.to_q",
        ] {
            assert!(is_vae_target(base), "{base} should be packed");
        }
        for base in [
            "decoder.mid_block.attentions.0.group_norm",
            "decoder.mid_block.resnets.0.conv1",
            "decoder.conv_in",
        ] {
            assert!(!is_vae_target(base), "{base} should stay dense");
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    #[test]
    fn auxiliary_components_pack_byte_identical_to_the_shared_load_time_seams() {
        let weight = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).cos()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let (expected_weight, expected_scales, expected_biases) =
            quantize(weight.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();

        for (base, predicate) in [
            (
                "encoder.block.0.layer.1.DenseReluDense.wi_0",
                is_t5_target as fn(&str) -> bool,
            ),
            (
                "decoder.mid_block.attentions.0.to_q",
                is_vae_target as fn(&str) -> bool,
            ),
        ] {
            let mut map = HashMap::new();
            map.insert(format!("{base}.weight"), weight.clone());
            let out = quantize_map(map, 4, GROUP_SIZE, predicate).unwrap();
            assert!(byte_equal(
                out.get(&format!("{base}.weight")).unwrap(),
                &expected_weight
            ));
            assert!(byte_equal(
                out.get(&format!("{base}.scales")).unwrap(),
                &expected_scales
            ));
            assert!(byte_equal(
                out.get(&format!("{base}.biases")).unwrap(),
                &expected_biases
            ));
        }
    }

    #[test]
    fn t5_progressive_pack_matches_selective_two_stage_affine_quantization() {
        let weight = Array::from_slice(
            &(0..64 * 128)
                .map(|i| ((i as f32) * 0.017).cos())
                .collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map = HashMap::new();
        for base in [
            "encoder.block.0.layer.1.DenseReluDense.wi_0",
            "shared",
            "encoder.block.4.layer.0.SelfAttention.q",
            "encoder.block.1.layer.1.DenseReluDense.wi_0",
        ] {
            map.insert(format!("{base}.weight"), weight.clone());
        }
        let out = quantize_map_with_residual_policy(
            map,
            8,
            GROUP_SIZE,
            is_t5_target,
            t5_residual_bits_for,
        )
        .unwrap();
        for base in [
            "encoder.block.0.layer.1.DenseReluDense.wi_0",
            "shared",
            "encoder.block.4.layer.0.SelfAttention.q",
            "encoder.block.1.layer.1.DenseReluDense.wi_0",
        ] {
            let wbf16 = weight.as_dtype(Dtype::Bfloat16).unwrap();
            let (primary_weight, primary_scales, primary_biases) =
                quantize(&wbf16, GROUP_SIZE, 8).unwrap();
            let restored = dequantize(
                &primary_weight,
                &primary_scales,
                &primary_biases,
                GROUP_SIZE,
                8,
            )
            .unwrap();
            let residual = subtract(&wbf16, &restored).unwrap();
            let residual_bits = t5_residual_bits_for(base);
            let (residual_weight, residual_scales, residual_biases) =
                quantize(&residual, GROUP_SIZE, residual_bits).unwrap();
            for (suffix, expected) in [
                ("weight", &primary_weight),
                ("scales", &primary_scales),
                ("biases", &primary_biases),
                ("residual.weight", &residual_weight),
                ("residual.scales", &residual_scales),
                ("residual.biases", &residual_biases),
            ] {
                assert!(
                    byte_equal(out.get(&format!("{base}.{suffix}")).unwrap(), expected),
                    "progressive tensor mismatch for {base}: {suffix}"
                );
            }
        }
        assert_eq!(t5_residual_bits_for("shared"), 8);
        for projection in ["q", "k", "v", "o"] {
            assert_eq!(
                t5_residual_bits_for(&format!(
                    "encoder.block.{}.layer.0.SelfAttention.{projection}",
                    T5_SENSITIVE_RESIDUAL_BLOCK
                )),
                T5_SENSITIVE_RESIDUAL_BITS
            );
        }
        for projection in ["wi_0", "wi_1", "wo"] {
            assert_eq!(
                t5_residual_bits_for(&format!(
                    "encoder.block.{}.layer.1.DenseReluDense.{projection}",
                    T5_SENSITIVE_RESIDUAL_FFN_BLOCK
                )),
                T5_SENSITIVE_RESIDUAL_BITS
            );
        }
        assert_eq!(
            t5_residual_bits_for("encoder.block.0.layer.1.DenseReluDense.wi_0"),
            4
        );
    }

    /// The packed triple a block Linear becomes is byte-identical to the op the load-time `.quantize`
    /// runs (bf16 cast, group 64) — the sc-8669 round-trip guarantee: pre-quantize-on-disk ==
    /// quantize-at-load. A top-level embedder stays dense (predicate); a 1-D norm stays dense (shape
    /// guard).
    #[test]
    fn quantize_map_packs_block_linear_byte_identical_to_load_time_quantize() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        // A double-block attention proj (packs) + a top-level embedder (dense, predicate) + a 1-D
        // QK-norm (shape-guarded dense).
        map.insert("transformer_blocks.0.attn.to_q.weight".into(), w.clone());
        map.insert(
            "x_embedder.weight".into(),
            Array::from_slice(
                &(0..64 * 128).map(|i| (i as f32).cos()).collect::<Vec<_>>(),
                &[64, 128],
            ),
        );
        map.insert(
            "transformer_blocks.0.attn.norm_q.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_transformer_target).unwrap();

        let base = "transformer_blocks.0.attn.to_q";
        let wq = out.get(&format!("{base}.weight")).expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let scales = out.get(&format!("{base}.scales")).unwrap();
        let biases = out.get(&format!("{base}.biases")).unwrap();
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "packed weight != load-time quantize");
        assert!(byte_equal(scales, &esc), "scales != load-time quantize");
        assert!(byte_equal(biases, &ebi), "biases != load-time quantize");

        // The top-level embedder stays dense (predicate) — no packed triple.
        let xe = out.get("x_embedder.weight").unwrap();
        assert_eq!(xe.dtype(), Dtype::Float32, "x_embedder unchanged (dense)");
        assert!(!out.contains_key("x_embedder.scales"));
        // The 1-D norm stays dense (shape guard).
        let n = out.get("transformer_blocks.0.attn.norm_q.weight").unwrap();
        assert_eq!(n.dtype(), Dtype::Float32, "norm unchanged");
    }
}
