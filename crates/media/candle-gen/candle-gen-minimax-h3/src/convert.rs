//! Offline pre-quantization of the MiniMax-H3-**Image** DiT on the candle lane (SW-7093): the twin
//! of `mlx_gen_minimax_h3::convert::quantize_minimax_h3_image_dit`, for hosts that cannot run MLX
//! (the Windows/CUDA runner). It writes the **same artifact**: MLX-affine packed triples
//! (`{base}.weight` u32 codes, `{base}.scales`, `{base}.biases`, group size 64) under the layout
//! [`gen_core_h3::h3_image_tier`] publishes, so either backend loads either converter's output.
//!
//! The policy is H3's, taken from [`gen_core_h3::h3_image_tier::dit_pack_bits`] (the MLX crate has a
//! test that the two agree on every published tensor name): block and token-refiner attention and
//! feed-forward and the per-block AdaLN projection are packed; the float32 I/O heads,
//! `context_embedder`, `norm_out.linear`, **the six `pdd_proj_out.{i}` per-step heads** and every norm
//! stay dense. Like the MLX converter it processes **one source shard at a time** and casts to bf16
//! before packing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{safetensors, DType, Tensor};
use candle_gen::quant::pack_mlx_affine;
use candle_gen::{CandleError, Result};
use gen_core_h3::h3_image_tier::{dit_pack_bits, DIT_DIR, GROUP_SIZE, PDD_FUSED_HEADS_FILE};

fn io<T>(r: std::io::Result<T>, what: &str) -> Result<T> {
    r.map_err(|e| CandleError::Msg(format!("h3-image convert: {what}: {e}")))
}

/// `true` iff a `[out, in]` weight can be packed at [`GROUP_SIZE`] (the shape guard the MLX
/// converter's `quantize_map` applies).
fn packable(t: &Tensor) -> bool {
    matches!(t.dims(), [_, n] if n % GROUP_SIZE == 0 && *n >= GROUP_SIZE)
}

fn source_shards(src: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(src)
        .map_err(|e| CandleError::Msg(format!("h3-image convert: read {}: {e}", src.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .filter(|p| {
            !p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.starts_with("._"))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "h3-image convert: no .safetensors in {}",
            src.display()
        )));
    }
    Ok(files)
}

/// Pack one shard's tensors: the keys the policy packs become the triple, the rest pass through.
pub fn pack_shard(
    tensors: HashMap<String, Tensor>,
    tier_bits: i32,
    adaln_bits: i32,
) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(tensors.len());
    for (name, t) in tensors {
        let packed_bits = name
            .strip_suffix(".weight")
            .and_then(|base| dit_pack_bits(base, tier_bits, adaln_bits))
            .filter(|_| packable(&t));
        let (Some(bits), Some(base)) = (packed_bits, name.strip_suffix(".weight")) else {
            out.insert(name, t);
            continue;
        };
        let weight = t.to_dtype(DType::BF16)?;
        let (wq, scales, biases) = pack_mlx_affine(&weight, bits as usize, GROUP_SIZE)?;
        out.insert(format!("{base}.weight"), wq);
        out.insert(format!("{base}.scales"), scales.to_dtype(DType::BF16)?);
        out.insert(format!("{base}.biases"), biases.to_dtype(DType::BF16)?);
    }
    Ok(out)
}

/// Pre-quantize a MiniMax-H3-Image snapshot's `transformer_diffusers/` into the tier directory
/// `dst_tier` (`transformer_diffusers/` plus `pdd_fused_heads.safetensors` copied byte for byte),
/// one source shard at a time. `bits` is the tier width (4 or 8); `adaln_bits` the AdaLN width
/// (H3's default is the tier's own). Returns the emitted shards.
pub fn quantize_image_dit(
    src_root: &Path,
    dst_tier: &Path,
    bits: i32,
    adaln_bits: i32,
    on_shard: &mut dyn FnMut(&Path) -> Result<()>,
) -> Result<Vec<PathBuf>> {
    if !matches!(bits, 4 | 8) || !matches!(adaln_bits, 4 | 8) {
        return Err(CandleError::Msg(format!(
            "h3-image convert: widths must be 4 or 8, got tier {bits} / adaln {adaln_bits}"
        )));
    }
    let src = src_root.join(DIT_DIR);
    let dst = dst_tier.join(DIT_DIR);
    let shards = source_shards(&src)?;
    io(std::fs::create_dir_all(&dst), "create the tier directory")?;

    // The marker first, as on MLX: a consumer that sees shards appearing already knows the tier.
    let mut config: serde_json::Value = serde_json::from_slice(&io(
        std::fs::read(src.join("config.json")),
        "read config.json",
    )?)
    .map_err(|e| CandleError::Msg(format!("h3-image convert: parse config.json: {e}")))?;
    config["quantization"] = serde_json::json!({ "bits": bits, "group_size": GROUP_SIZE });
    io(
        std::fs::write(
            dst.join("config.json"),
            serde_json::to_string_pretty(&config).map_err(|e| {
                CandleError::Msg(format!("h3-image convert: serialize config: {e}"))
            })?,
        ),
        "write config.json",
    )?;

    let n = shards.len();
    let mut written = Vec::with_capacity(n);
    let mut weight_map = serde_json::Map::new();
    let mut total_size: usize = 0;
    for (i, shard) in shards.iter().enumerate() {
        let tensors = safetensors::load(shard, &candle_gen::candle_core::Device::Cpu)?;
        let packed = pack_shard(tensors, bits, adaln_bits)?;
        let name = format!("diffusion_pytorch_model-{:05}-of-{n:05}.safetensors", i + 1);
        let path = dst.join(&name);
        for (key, t) in &packed {
            total_size += t.elem_count() * t.dtype().size_in_bytes();
            weight_map.insert(key.clone(), serde_json::Value::String(name.clone()));
        }
        safetensors::save(&packed, &path)?;
        drop(packed);
        on_shard(&path)?;
        written.push(path);
    }
    let mut sorted: Vec<_> = weight_map.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let index = serde_json::json!({
        "metadata": { "total_size": total_size },
        "weight_map": serde_json::Value::Object(sorted.into_iter().collect()),
    });
    io(
        std::fs::write(
            dst.join("diffusion_pytorch_model.safetensors.index.json"),
            serde_json::to_string_pretty(&index)
                .map_err(|e| CandleError::Msg(format!("h3-image convert: serialize index: {e}")))?,
        ),
        "write the index",
    )?;

    let heads = src_root.join(PDD_FUSED_HEADS_FILE);
    if heads.is_file() {
        io(
            std::fs::copy(&heads, dst_tier.join(PDD_FUSED_HEADS_FILE)),
            "copy pdd_fused_heads",
        )?;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn t(rows: usize, cols: usize, dtype: DType) -> Tensor {
        let v: Vec<f32> = (0..rows * cols)
            .map(|i| (i % 17) as f32 * 0.1 - 0.8)
            .collect();
        Tensor::from_vec(v, (rows, cols), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn synthetic() -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        for k in [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.ff.net.2.weight",
            "transformer_blocks.0.adaln_proj.linear.weight",
            "token_refiner.refiner_blocks.0.attn.to_k.weight",
        ] {
            m.insert(k.into(), t(8, 128, DType::BF16));
        }
        // Dense by policy, though 2-D and group-aligned: the heads must not be packed.
        for k in [
            "pdd_proj_out.0.weight",
            "pdd_proj_out.5.weight",
            "proj_in.weight",
            "context_embedder.weight",
            "norm_out.linear.weight",
        ] {
            m.insert(k.into(), t(8, 128, DType::F32));
        }
        m.insert(
            "transformer_blocks.0.norm1.weight".into(),
            t(1, 128, DType::BF16).flatten_all().unwrap(),
        );
        m.insert(
            "transformer_blocks.0.attn.to_q.bias".into(),
            t(1, 8, DType::BF16).flatten_all().unwrap(),
        );
        m
    }

    /// The packed set gains a `.scales`/`.biases` pair and u32 codes at the tier width; the dense
    /// set (heads, I/O, norms, biases) passes through untouched at its own dtype.
    #[test]
    fn heads_and_io_stay_dense_and_the_blocks_pack_at_each_tier() {
        for (bits, cols) in [(8, 128 / 4), (4, 128 / 8)] {
            let out = pack_shard(synthetic(), bits, bits).unwrap();
            for base in [
                "transformer_blocks.0.attn.to_q",
                "transformer_blocks.0.ff.net.2",
                "transformer_blocks.0.adaln_proj.linear",
                "token_refiner.refiner_blocks.0.attn.to_k",
            ] {
                let w = &out[&format!("{base}.weight")];
                assert_eq!(w.dtype(), DType::U32, "{base}");
                assert_eq!(w.dims(), &[8, cols], "{base} at {bits} bits");
                assert_eq!(out[&format!("{base}.scales")].dims(), &[8, 2]);
                assert_eq!(out[&format!("{base}.biases")].dtype(), DType::BF16);
            }
            for dense in [
                "pdd_proj_out.0",
                "pdd_proj_out.5",
                "proj_in",
                "context_embedder",
                "norm_out.linear",
            ] {
                let w = &out[&format!("{dense}.weight")];
                assert_eq!(
                    (w.dtype(), w.dims()),
                    (DType::F32, &[8usize, 128][..]),
                    "{dense}"
                );
                assert!(!out.contains_key(&format!("{dense}.scales")), "{dense}");
            }
            assert_eq!(
                out["transformer_blocks.0.norm1.weight"].dtype(),
                DType::BF16
            );
            assert_eq!(out["transformer_blocks.0.attn.to_q.bias"].dims(), &[8]);
        }
    }

    /// End to end over a synthetic snapshot: the layout, the marker, the index and the byte-exact
    /// heads file.
    #[test]
    fn a_synthetic_snapshot_converts_to_the_published_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("snap");
        let dit = src.join(DIT_DIR);
        std::fs::create_dir_all(&dit).unwrap();
        std::fs::write(dit.join("config.json"), r#"{"num_layers": 50}"#).unwrap();
        safetensors::save(
            &synthetic(),
            dit.join("diffusion_pytorch_model-00001-of-00001.safetensors"),
        )
        .unwrap();
        std::fs::write(src.join(PDD_FUSED_HEADS_FILE), b"heads").unwrap();
        let dst = tmp.path().join("q4");
        let mut seen = 0;
        quantize_image_dit(&src, &dst, 4, 4, &mut |_| {
            seen += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, 1);
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dst.join(DIT_DIR).join("config.json")).unwrap())
                .unwrap();
        assert_eq!(cfg["quantization"]["bits"], 4);
        assert_eq!(cfg["quantization"]["group_size"], 64);
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                dst.join(DIT_DIR)
                    .join("diffusion_pytorch_model.safetensors.index.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(index["weight_map"]["transformer_blocks.0.attn.to_q.scales"].is_string());
        assert!(index["weight_map"]["pdd_proj_out.0.weight"].is_string());
        assert!(index["weight_map"].get("pdd_proj_out.0.scales").is_none());
        assert_eq!(
            std::fs::read(dst.join(PDD_FUSED_HEADS_FILE)).unwrap(),
            b"heads"
        );
        // The marker is what the load-time tier check reads.
        assert_eq!(
            crate::tier::MiniMaxH3TierPaths::staged_bits(&dst.join(DIT_DIR)).unwrap(),
            Some(4)
        );
    }
}
