//! Offline pre-quantization: read the dense bf16 converted snapshot and write a packed Q4/Q8
//! snapshot that [`crate::quant`] loads with no dense transient. Mirrors `mlx_gen_flux2::convert` /
//! `mlx_gen_scail2::convert` (same `mlx_rs::ops::quantize`, byte-equal to the load-time
//! [`Ideogram4Pipeline::quantize`](crate::Ideogram4Pipeline::quantize) / `nn.quantize(bf16)`).
//!
//! Used at publish time (sc-5990) to ship a lean ~14 GB Q4 turnkey instead of the ~53 GB bf16
//! source. The two DiTs + the Qwen3-VL text encoder are packed; the small VAE / tokenizer /
//! scheduler pass through dense.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::ops::quantize;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::quant::GROUP_SIZE;

/// Pack every `is_target` `{base}.weight` (group-quantizable 2-D, `in % gs == 0`) into the triple
/// `{base}.weight` (u32 codes) + `.scales` + `.biases` — the weight cast to bf16 first so the pack
/// is byte-identical to the load-time quantize (and the fork's `nn.quantize(bf16)`). Every other
/// tensor (norms, 1-D, non-divisible, non-target) passes through unchanged.
fn quantize_map(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
    is_target: impl Fn(&str) -> bool,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map {
        let base = k.strip_suffix(".weight").filter(|b| is_target(b));
        let packable = base.is_some()
            && v.shape().len() == 2
            && v.shape()[1] % group_size == 0
            && v.shape()[1] >= group_size;
        if let (Some(base), true) = (base, packable) {
            let wbf16 = v.as_dtype(Dtype::Bfloat16)?;
            let (wq, scales, biases) = quantize(&wbf16, group_size, bits)?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
        } else {
            out.insert(k, v);
        }
    }
    Ok(out)
}

/// DiT pack target: every Linear. The tiny `embed_image_indicator` table (2-D) is name-excluded so
/// it stays a dense [`TokenEmbedding`](mlx_gen::nn::TokenEmbedding); the norms are 1-D and auto-skip.
fn is_dit_target(base: &str) -> bool {
    !base.contains("embed_image_indicator")
}

/// TE pack target: the layer q/k/v/o + gate/up/down Linears and the token embedding. The per-head
/// q/k RMSNorms + the input/post layernorms (all 1-D) auto-skip; `norm`-named bases are excluded for
/// clarity.
fn is_te_target(base: &str) -> bool {
    !base.contains("norm")
}

/// Read every tensor of `dir` (possibly sharded safetensors) into an owned key→`Array` map (MLX
/// arrays are ref-counted, so the clone is a handle copy, not a buffer copy).
fn load_dir_map(dir: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_dir(dir)?;
    Ok(w.keys()
        .map(|k| (k.to_string(), w.get(k).expect("listed key").clone()))
        .collect())
}

/// Materialize + write a key→`Array` map to a single `path` safetensors (one file — a packed Q4
/// component is small enough not to need sharding).
fn save_map(path: &Path, map: &HashMap<String, Array>) -> Result<()> {
    eval(map.values().collect::<Vec<_>>())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Array::save_safetensors(
        map.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        path,
    )?;
    Ok(())
}

/// Pre-quantize one DiT component dir (`transformer` or `unconditional_transformer`) → a packed
/// `model.safetensors` in `dst`. `bits` = 4 (Q4) or 8 (Q8); group size is the codebase default 64.
pub fn quantize_ideogram_dit(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, is_dit_target)?;
    save_map(&dst.join("model.safetensors"), &map)
}

/// Pre-quantize the `text_encoder` component dir → a packed `model.safetensors` in `dst`.
pub fn quantize_ideogram_text_encoder(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, is_te_target)?;
    save_map(&dst.join("model.safetensors"), &map)
}

/// Recursively copy a directory's files (one level of nesting is enough for the VAE/tokenizer dirs).
fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &target)?;
        } else {
            std::fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

/// Assemble a full pre-quantized turnkey snapshot in `dst_root`: pack the two DiTs + the TE, and
/// copy the dense VAE / tokenizer / scheduler / `config.json`s / top-level files verbatim. The
/// result loads via [`Ideogram4Pipeline::load`](crate::Ideogram4Pipeline::load) (the packed weights
/// auto-detect) with no dense transient. `bits` = 4 for the lean Q4 publish (sc-5990).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;
    quantize_ideogram_dit(
        &src_root.join("transformer"),
        &dst_root.join("transformer"),
        bits,
    )?;
    quantize_ideogram_dit(
        &src_root.join("unconditional_transformer"),
        &dst_root.join("unconditional_transformer"),
        bits,
    )?;
    quantize_ideogram_text_encoder(
        &src_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        bits,
    )?;
    // Dense passthrough — small relative to the DiTs, and the loaders read them as-is.
    for rel in ["vae", "tokenizer", "scheduler"] {
        let s = src_root.join(rel);
        if s.exists() {
            copy_dir(&s, &dst_root.join(rel))?;
        }
    }
    // Per-component config.json (HF-compat; the Rust loaders use hardcoded configs) + top-level files.
    for comp in ["transformer", "unconditional_transformer", "text_encoder"] {
        let s = src_root.join(comp).join("config.json");
        if s.exists() {
            std::fs::copy(&s, dst_root.join(comp).join("config.json"))?;
        }
    }
    for f in ["model_index.json", "LICENSE.md", "README.md"] {
        let s = src_root.join(f);
        if s.exists() {
            std::fs::copy(&s, dst_root.join(f))?;
        }
    }
    Ok(())
}
