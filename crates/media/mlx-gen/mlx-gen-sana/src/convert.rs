//! Offline pre-quantization: read a dense or same-tier mixed SANA MLX snapshot (`transformer/ text_encoder/ vae/`) and
//! write a packed Q4/Q8 turnkey that `crate::quant` (via [`crate::model::load`] /
//! [`crate::model::load_sprint`]) loads with no dense bf16/f32 transient. Mirrors the other Group-B
//! converters (`mlx_gen_chroma::convert` / `mlx_gen_sdxl::convert`) — the same
//! `mlx_gen::quant::quantize_map`, byte-equal to the load-time `.quantize` seam — differing in the
//! SANA key layout and the **two** quantized components.
//!
//! SANA quantizes:
//!  * the Linear-DiT `transformer/` — every trunk Linear (`is_transformer_target`): the self /
//!    cross attention `to_q/k/v` + `to_out.0`, the timestep / (Sprint) guidance / modulation MLPs, the
//!    caption-projection MLP, and `proj_out`. The `patch_embed` conv and the GLUMBConv depth/point/
//!    inverted convs stay dense (they are conv weights, not a quant target).
//!  * the Gemma-2 CHI `text_encoder/` — every decoder projection (`is_te_target`): `self_attn`
//!    `q/k/v/o_proj`, the `mlp` `gate/up/down_proj`, and `model.embed_tokens`. RMSNorms stay dense.
//!    (This is the biggest component, so packing it is where the low-RAM win is.)
//!
//! The **DC-AE `vae/`** decoder stays dense in every tier (all-conv; its Linear-attention is a
//! measurably-0% memory win and is not routed through the quant seam), so it is mirrored verbatim.
//!
//! Each per-component predicate matches the loader's `crate::quant::lin` scope exactly — a missed site
//! (or a wrongly-packed dense tensor) loads u32 codes as dense floats → a garbage render. The
//! completeness gate is the real-weight render in `tests/prequantize_real_weights.rs`. There is no
//! `config.json` in `transformer/` (the config is the hard-coded `SanaTransformerConfig`), and the
//! packed bit-width self-describes via the `.scales` shapes, so no quant-config sidecar is written.
//!
//! Group-B per-crate converter template (sc-8669).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use mlx_gen::quant::{copy_dir, copy_turnkey_assets, save_map};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::quant::GROUP_SIZE;

/// The single packed weight file each component ships (the source is already one flat file per
/// component: `transformer/diffusion_pytorch_model.safetensors`, `text_encoder/gemma-2-2b-it.safetensors`).
const TRANSFORMER_FILE: &str = "diffusion_pytorch_model.safetensors";
const TE_FILE: &str = "gemma-2-2b-it.safetensors";

// ============================================================================================
// Pack predicates (operate on the **base** = the on-disk key minus its `.weight`).
// ============================================================================================

/// Whether a `transformer/` key's `base` is a trunk **Linear** the DiT quantizes — matching the
/// `crate::quant::lin` sites in [`crate::transformer`] exactly. Everything else (the `patch_embed` +
/// GLUMBConv convs, `scale_shift_table`, `caption_norm`, the per-block affine-free norms) stays dense.
fn is_transformer_target(base: &str) -> bool {
    if let Some(rest) = base.strip_prefix("transformer_blocks.") {
        // rest = `{i}.<tail>`
        let Some((_i, tail)) = rest.split_once('.') else {
            return false;
        };
        return matches!(
            tail,
            "attn1.to_q"
                | "attn1.to_k"
                | "attn1.to_v"
                | "attn1.to_out.0"
                | "attn2.to_q"
                | "attn2.to_k"
                | "attn2.to_v"
                | "attn2.to_out.0"
        );
    }
    // Trunk-level MLPs: timestep + (Sprint) guidance embedders (both the base `time_embed.emb.…` and
    // the Sprint `time_embed.…` key layouts), the modulation Linear, the caption-projection MLP, and
    // the final `proj_out`.
    matches!(
        base,
        "time_embed.emb.timestep_embedder.linear_1"
            | "time_embed.emb.timestep_embedder.linear_2"
            | "time_embed.timestep_embedder.linear_1"
            | "time_embed.timestep_embedder.linear_2"
            | "time_embed.guidance_embedder.linear_1"
            | "time_embed.guidance_embedder.linear_2"
            | "time_embed.linear"
            | "caption_projection.linear_1"
            | "caption_projection.linear_2"
            | "proj_out"
    )
}

/// Whether a `text_encoder/` key's `base` is a Gemma-2 decoder **projection** the TE quantizes —
/// matching `mlx_gen_pid::gemma2`'s `lin` sites exactly (`self_attn` q/k/v/o + `mlp` gate/up/down).
/// `model.embed_tokens` uses the shared packed embedding loader; norms and `lm_head` stay dense.
fn is_te_target(base: &str) -> bool {
    if base == "model.embed_tokens" {
        return true;
    }
    let Some(rest) = base.strip_prefix("model.layers.") else {
        return false;
    };
    let Some((_i, tail)) = rest.split_once('.') else {
        return false;
    };
    matches!(
        tail,
        "self_attn.q_proj"
            | "self_attn.k_proj"
            | "self_attn.v_proj"
            | "self_attn.o_proj"
            | "mlp.gate_proj"
            | "mlp.up_proj"
            | "mlp.down_proj"
    )
}

/// Preserve validated same-tier affine triples in a mixed snapshot. In particular a cached q4
/// tier with a dense embedding can acquire a packed embedding WITHOUT repacking its U32 decoder
/// projections. A tier mismatch or orphan companion fails closed instead of quantizing codes.
fn quantize_map(
    mut map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
    is_target: impl Fn(&str) -> bool,
) -> Result<HashMap<String, Array>> {
    if !matches!(bits, 4 | 8) {
        return Err(Error::Msg(format!(
            "sana convert: unsupported Q{bits}; expected Q4 or Q8"
        )));
    }
    let bases: std::collections::BTreeSet<String> = map
        .keys()
        .filter_map(|key| {
            [".weight", ".scales", ".biases"]
                .iter()
                .find_map(|suffix| key.strip_suffix(suffix))
                .filter(|base| is_target(base))
                .map(str::to_owned)
        })
        .collect();
    let mut retained = HashMap::new();
    for base in bases {
        let weight_key = format!("{base}.weight");
        let scale_key = format!("{base}.scales");
        let bias_key = format!("{base}.biases");
        let weight = map
            .get(&weight_key)
            .ok_or_else(|| Error::Msg(format!("sana convert: missing {weight_key}")))?;
        if map.contains_key(&scale_key)
            || map.contains_key(&bias_key)
            || weight.dtype() == mlx_rs::Dtype::Uint32
        {
            let scales = map
                .get(&scale_key)
                .ok_or_else(|| Error::Msg(format!("sana convert: missing {scale_key}")))?;
            let biases = map
                .get(&bias_key)
                .ok_or_else(|| Error::Msg(format!("sana convert: missing {bias_key}")))?;
            let packed_bits = mlx_gen::quant::packed_bits(weight, scales, group_size)?;
            if packed_bits != bits
                || biases.shape() != scales.shape()
                || biases.dtype() != scales.dtype()
            {
                return Err(Error::Msg(format!("sana convert: incompatible affine Q{packed_bits} layout at {base} for requested Q{bits}")));
            }
            for key in [weight_key, scale_key, bias_key] {
                retained.insert(key.clone(), map.remove(&key).expect("validated packed key"));
            }
        } else if !matches!(
            weight.dtype(),
            mlx_rs::Dtype::Float32 | mlx_rs::Dtype::Float16 | mlx_rs::Dtype::Bfloat16
        ) {
            return Err(Error::Msg(format!(
                "sana convert: unsupported dense storage at {base}"
            )));
        }
    }
    let mut out = mlx_gen::quant::quantize_map(map, bits, group_size, is_target)?;
    out.extend(retained);
    Ok(out)
}

/// Load a component dir's safetensors into one key→`Array` map (a duplicate key is a corrupt
/// snapshot → error). SANA ships one flat file per component, but `from_dir` handles a sharded set too.
fn load_map(dir: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_dir(dir)?;
    let mut map: HashMap<String, Array> = HashMap::new();
    for k in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let v = w.get(&k).expect("listed key").clone();
        if map.insert(k.clone(), v).is_some() {
            return Err(Error::Msg(format!(
                "sana convert: duplicate key `{k}` in {}",
                dir.display()
            )));
        }
    }
    Ok(map)
}

/// Copy every non-`.safetensors` sidecar (e.g. `text_encoder/tokenizer.json`) from `src` → `dst`,
/// dereferencing HF-cache symlinks (`std::fs::copy` follows the link to the blob).
fn copy_sidecars(src: &Path, dst: &Path) -> Result<()> {
    for entry in std::fs::read_dir(src)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors")
            || mlx_gen::gen_core::weightsmeta::is_hidden_file(&path)
        {
            continue;
        }
        if let Some(name) = path.file_name() {
            std::fs::copy(&path, dst.join(name))?;
        }
    }
    Ok(())
}

/// Assemble a full pre-quantized turnkey SANA snapshot in `dst_root`: pack the Linear-DiT
/// `transformer/` and the Gemma-2 `text_encoder/`, mirror the dense DC-AE `vae/`, and copy the
/// tokenizer + any root license/notice verbatim (deref symlinks). Already-packed same-tier
/// projections are validated and preserved, so an older q4/q8 tier with a dense embedding can be
/// converted without requantizing its packed codes. Mismatched tiers and orphan triples refuse.
/// The result loads via
/// [`crate::model::load`] / [`crate::model::load_sprint`] (packed weights auto-detect) with no dense transient.
/// `bits` = 4 (Q4 tier) or 8 (Q8 tier). The **bf16 tier** is the dense source itself (no conversion —
/// mirror it; see the tier builder in `tests/prequantize_real_weights.rs`).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;

    // Transformer: pack the trunk Linears into one flat file.
    let tr_src = src_root.join("transformer");
    if !tr_src.is_dir() {
        return Err(Error::Msg(format!(
            "sana convert: source snapshot {} has no transformer/ dir",
            src_root.display()
        )));
    }
    let tr_dst = dst_root.join("transformer");
    std::fs::create_dir_all(&tr_dst)?;
    let tr = quantize_map(load_map(&tr_src)?, bits, GROUP_SIZE, is_transformer_target)?;
    save_map(&tr_dst.join(TRANSFORMER_FILE), &tr)?;

    // Text encoder: pack the Gemma-2 projections and token embedding, keep tokenizer.json.
    let te_src = src_root.join("text_encoder");
    if !te_src.is_dir() {
        return Err(Error::Msg(format!(
            "sana convert: source snapshot {} has no text_encoder/ dir",
            src_root.display()
        )));
    }
    let te_dst = dst_root.join("text_encoder");
    std::fs::create_dir_all(&te_dst)?;
    let te = quantize_map(load_map(&te_src)?, bits, GROUP_SIZE, is_te_target)?;
    save_map(&te_dst.join(TE_FILE), &te)?;
    copy_sidecars(&te_src, &te_dst)?;

    // DC-AE decoder stays dense (all-conv) — mirror verbatim.
    copy_dir(&src_root.join("vae"), &dst_root.join("vae"))?;

    // Root license / notice / etc.
    copy_turnkey_assets(src_root, dst_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::{Array, Dtype};

    #[test]
    fn transformer_predicate_matches_trunk_linears_only() {
        for base in [
            "transformer_blocks.0.attn1.to_q",
            "transformer_blocks.0.attn1.to_k",
            "transformer_blocks.0.attn1.to_v",
            "transformer_blocks.0.attn1.to_out.0",
            "transformer_blocks.19.attn2.to_q",
            "transformer_blocks.19.attn2.to_k",
            "transformer_blocks.19.attn2.to_v",
            "transformer_blocks.19.attn2.to_out.0",
            "time_embed.emb.timestep_embedder.linear_1",
            "time_embed.emb.timestep_embedder.linear_2",
            "time_embed.timestep_embedder.linear_1",
            "time_embed.guidance_embedder.linear_2",
            "time_embed.linear",
            "caption_projection.linear_1",
            "caption_projection.linear_2",
            "proj_out",
        ] {
            assert!(is_transformer_target(base), "{base} should pack");
        }
        // Convs + norms + tables stay dense.
        for base in [
            "patch_embed.proj",
            "transformer_blocks.0.ff.conv_inverted",
            "transformer_blocks.0.ff.conv_depth",
            "transformer_blocks.0.ff.conv_point",
            "transformer_blocks.0.attn1.norm_q",
            "transformer_blocks.0.attn2.norm_k",
            "transformer_blocks.0.scale_shift_table",
            "caption_norm",
            "scale_shift_table",
        ] {
            assert!(!is_transformer_target(base), "{base} should stay dense");
        }
    }

    #[test]
    fn te_predicate_matches_gemma_projections_and_embedding() {
        for base in [
            "model.embed_tokens",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.25.mlp.gate_proj",
            "model.layers.25.mlp.up_proj",
            "model.layers.25.mlp.down_proj",
        ] {
            assert!(is_te_target(base), "{base} should pack");
        }
        for base in [
            "lm_head",
            "model.norm",
            "model.layers.0.input_layernorm",
            "model.layers.0.post_attention_layernorm",
            "model.layers.0.pre_feedforward_layernorm",
        ] {
            assert!(!is_te_target(base), "{base} should stay dense");
        }
    }

    #[test]
    fn converter_embedding_is_consumed_by_gemma_at_both_packed_tiers() {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        for bits in [4, 8] {
            let source = Array::from_slice(
                &(0..5 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
                &[5, 128],
            );
            let norm = Array::from_slice(&[0.0f32; 128], &[128]);
            let map = HashMap::from([
                ("model.embed_tokens.weight".into(), source),
                ("model.norm.weight".into(), norm),
            ]);
            let packed = quantize_map(map, bits, GROUP_SIZE, is_te_target).unwrap();
            assert_eq!(packed["model.embed_tokens.weight"].dtype(), Dtype::Uint32);
            assert!(packed.contains_key("model.embed_tokens.scales"));
            assert_eq!(packed["model.norm.weight"].dtype(), Dtype::Float32);
            let dense = mlx_rs::ops::dequantize(
                &packed["model.embed_tokens.weight"],
                &packed["model.embed_tokens.scales"],
                &packed["model.embed_tokens.biases"],
                GROUP_SIZE,
                bits,
            )
            .unwrap();
            let cfg = mlx_gen_pid::Gemma2Config {
                hidden_size: 128,
                num_layers: 0,
                ..mlx_gen_pid::Gemma2Config::gemma_2_2b()
            };
            let dir = tempfile::tempdir().unwrap();
            let file = dir.path().join("gemma-2-2b-it.safetensors");
            save_map(&file, &packed).unwrap();
            let model = mlx_gen_pid::Gemma2::from_weights(
                &Weights::from_file(&file).unwrap(),
                "model.",
                &cfg,
            )
            .unwrap();
            let mut reference = packed;
            reference.remove("model.embed_tokens.scales");
            reference.remove("model.embed_tokens.biases");
            reference.insert("model.embed_tokens.weight".into(), dense);
            let reference =
                mlx_gen_pid::Gemma2::from_weights(&Weights::from_map(reference), "model.", &cfg)
                    .unwrap();
            let ids = Array::from_slice(&[4i32, 0, 4, 2], &[2, 2]);
            let got = model.forward(&ids, None).unwrap();
            let expected = reference.forward(&ids, None).unwrap();
            assert!(
                byte_equal(&got, &expected),
                "converter/loader embedding q{bits} mismatch"
            );
        }
    }

    #[test]
    fn mixed_tier_preserves_packed_projections_and_packs_only_dense_embedding() {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        for bits in [4, 8] {
            let dense = Array::from_slice(
                &(0..4 * 512).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
                &[4, 512],
            );
            let (codes, scales, biases) =
                quantize(dense.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, bits).unwrap();
            let base = "model.layers.0.self_attn.q_proj";
            let map = HashMap::from([
                (format!("{base}.weight"), codes.clone()),
                (format!("{base}.scales"), scales.clone()),
                (format!("{base}.biases"), biases.clone()),
                ("model.embed_tokens.weight".into(), dense),
            ]);
            let out = quantize_map(map.clone(), bits, GROUP_SIZE, is_te_target).unwrap();
            assert!(
                byte_equal(&out[&format!("{base}.weight")], &codes),
                "existing q{bits} codes were repacked"
            );
            assert!(byte_equal(&out[&format!("{base}.scales")], &scales));
            assert!(byte_equal(&out[&format!("{base}.biases")], &biases));
            assert_eq!(out["model.embed_tokens.weight"].dtype(), Dtype::Uint32);
            assert!(out.contains_key("model.embed_tokens.scales"));
            assert!(
                quantize_map(map.clone(), 12 - bits, GROUP_SIZE, is_te_target).is_err(),
                "tier mismatch must not silently reuse or repack codes"
            );
            let mut bad_shape = map.clone();
            bad_shape.insert(
                format!("{base}.biases"),
                Array::from_slice(&[0.0f32; 4], &[4, 1])
                    .as_dtype(scales.dtype())
                    .unwrap(),
            );
            assert!(quantize_map(bad_shape, bits, GROUP_SIZE, is_te_target).is_err());
            let mut bad_dtype = map.clone();
            bad_dtype.insert(
                format!("{base}.biases"),
                scales.as_dtype(Dtype::Int32).unwrap(),
            );
            assert!(quantize_map(bad_dtype, bits, GROUP_SIZE, is_te_target).is_err());
            let mut missing = map.clone();
            missing.remove(&format!("{base}.biases"));
            assert!(quantize_map(missing, bits, GROUP_SIZE, is_te_target).is_err());
            let mut orphan = map;
            orphan.remove(&format!("{base}.scales"));
            assert!(quantize_map(orphan, bits, GROUP_SIZE, is_te_target).is_err());
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// A packed trunk Linear is byte-identical to the op the load-time `.quantize` runs (bf16 cast,
    /// group 64) — the sc-8669 round-trip guarantee. A dense embedder-style key (predicate miss) and a
    /// 1-D norm (shape guard) stay dense.
    #[test]
    fn quantize_map_packs_trunk_linear_byte_identical() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert("transformer_blocks.0.attn1.to_q.weight".into(), w.clone());
        map.insert("model.layers.0.self_attn.q_proj.weight".into(), w.clone());
        map.insert(
            "patch_embed.proj.weight".into(),
            Array::from_slice(
                &(0..64 * 128).map(|i| (i as f32).cos()).collect::<Vec<_>>(),
                &[64, 128],
            ),
        );
        map.insert(
            "transformer_blocks.0.attn1.norm_q.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_transformer_target).unwrap();

        let base = "transformer_blocks.0.attn1.to_q";
        let wq = out.get(&format!("{base}.weight")).expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "packed weight != load-time quantize");
        assert!(byte_equal(
            out.get(&format!("{base}.scales")).unwrap(),
            &esc
        ));
        assert!(byte_equal(
            out.get(&format!("{base}.biases")).unwrap(),
            &ebi
        ));

        // The is_transformer_target predicate does NOT pack a TE key or the conv → both stay dense.
        assert_eq!(
            out.get("patch_embed.proj.weight").unwrap().dtype(),
            Dtype::Float32
        );
        assert!(!out.contains_key("model.layers.0.self_attn.q_proj.scales"));
        assert_eq!(
            out.get("transformer_blocks.0.attn1.norm_q.weight")
                .unwrap()
                .dtype(),
            Dtype::Float32
        );
    }
}
