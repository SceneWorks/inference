//! Native (Rust/MLX) Mochi 1 **snapshot → per-tier MLX** converter (story A6, sc-11990).
//!
//! The upstream `genmo/mochi-1-preview` snapshot ships the AsymmDiT transformer as **bf16** shards
//! (`diffusion_pytorch_model.safetensors.index.bf16.json` + parts), the T5-XXL text encoder as fp32
//! shards, and the AsymmVAE + tokenizer alongside. The distribution model (epic 1788 / the tier
//! strategy decision, `docs/reference/mochi-1-tier-strategy.md`) is **pre-quantized, self-contained
//! per-tier artifacts** — `q4/`, `q8/`, `bf16/` — each carrying its own `split_model.json`, mirroring
//! the LTX tier layout and the shipped Wan quant matrices. A client only ever downloads the one tier
//! it runs.
//!
//! [`convert_and_assemble`] produces **one tier** from the snapshot:
//!
//!   * load the transformer's bf16 shards (via [`crate::transformer::load_transformer_weights`], which
//!     merges the `.bf16` index) — the crate's native layout is the raw HF diffusers naming, so there
//!     are **no key renames** and **no conv transpose** here (the loader transposes `patch_embed` at
//!     load — see [`crate::transformer::MochiTransformer3DModel::from_weights`] — so the on-disk layout
//!     stays byte-identical to upstream and both the dense and quantized loaders round-trip it);
//!   * cast every float tensor to bf16 (a no-op for the already-bf16 shards; the parity-safe base for
//!     `quantize`);
//!   * for `q4`/`q8`, **selectively quantize** exactly the [`MOCHI_QUANT_SUFFIXES`] Linears (the same
//!     ones the packed loader consumes via `.scales`) with MLX `quantize` (group 64), emitting
//!     `.weight`(u32)/`.scales`/`.biases`; every other tensor (norms, adaLN modulation, patchify,
//!     pooler, caption/time embed, `proj_out`) stays dense bf16;
//!   * write `<out>/transformer/model.safetensors` + `<out>/split_model.json` (+ a sibling
//!     `quantize_config.json` for the quantized tiers, HF convention).
//!
//! The **T5-XXL text encoder, AsymmVAE, and tokenizer are shared across tiers** — [`stage_shared_components`]
//! materializes them **once** as sibling dirs (not duplicated per tier), and
//! [`crate::model::load`] resolves them from the tier dir's parent. This is the dominant download cost
//! for a very-large video model, so sharing it is the point of the layout.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::{Error, Result};
use mlx_rs::ops::quantize;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::transformer::load_transformer_weights;

/// The Mochi AsymmDiT quantization predicate: a transformer Linear is quantized iff its weight key
/// (minus the `.weight` suffix) ends with one of these — the visual attention `to_q/k/v` + `to_out.0`,
/// the joint-attention added projections `add_{q,k,v}_proj` + `to_add_out`, and the SwiGLU FFN
/// `net.0.proj`/`net.2` of both the visual `ff` and the text `ff_context`. This is **exactly** the set
/// the packed loader ([`crate::transformer::MochiLinear`]) treats as quantizable, so the on-disk
/// `.scales` the two agree on. The per-head `qk_norm` weights, adaLN modulation (`norm1*.linear`),
/// `patch_embed`, `pos_frequencies`, the time/caption embed, `norm_out`, and `proj_out` stay dense
/// (small + precision-sensitive) — the Wan/LTX stance.
pub const MOCHI_QUANT_SUFFIXES: &[&str] = &[
    ".attn1.to_q",
    ".attn1.to_k",
    ".attn1.to_v",
    ".attn1.to_out.0",
    ".attn1.add_q_proj",
    ".attn1.add_k_proj",
    ".attn1.add_v_proj",
    ".attn1.to_add_out",
    ".ff.net.0.proj",
    ".ff.net.2",
    ".ff_context.net.0.proj",
    ".ff_context.net.2",
];

/// The shared, tier-independent component subdirs copied/linked once (not per tier): the T5-XXL text
/// encoder, the AsymmVAE, and the tokenizer. Consumed straight from the diffusers snapshot layout (the
/// MLX loaders read the fp32 T5 shards + `vae/config.json` directly), so a raw copy/link is faithful.
pub const SHARED_COMPONENTS: &[&str] = &["text_encoder", "vae", "tokenizer"];

/// `true` iff `weight_key` (an entire key ending in `.weight`) is a [`MOCHI_QUANT_SUFFIXES`] Linear.
fn is_quant_target(weight_key: &str) -> bool {
    weight_key
        .strip_suffix(".weight")
        .is_some_and(|base| MOCHI_QUANT_SUFFIXES.iter().any(|s| base.ends_with(s)))
}

/// Conversion knobs. Default = the dense **bf16** tier (`quantize:false`); `bits`/`group_size` hold the
/// Q4/64 geometry, applied only when `quantize` is set. `bits` is `4` (Q4) or `8` (Q8).
#[derive(Clone, Copy, Debug)]
pub struct MochiConvertOpts {
    /// Selectively quantize the transformer's [`MOCHI_QUANT_SUFFIXES`] Linears.
    pub quantize: bool,
    /// Quantization bits (4 → Q4, 8 → Q8). Ignored when `quantize` is false.
    pub bits: i32,
    /// Affine-quant group size (the reference/mflux default 64).
    pub group_size: i32,
}

impl Default for MochiConvertOpts {
    fn default() -> Self {
        Self {
            quantize: false,
            bits: 4,
            group_size: 64,
        }
    }
}

impl MochiConvertOpts {
    /// Selective transformer quantization at `bits` (group 64) — the `q4`/`q8` tier recipe.
    pub fn quant(bits: i32) -> Self {
        Self {
            quantize: true,
            bits,
            group_size: 64,
        }
    }

    /// The canonical tier directory name for these opts (`q4` / `q8` / `bf16`).
    pub fn tier_name(&self) -> String {
        if self.quantize {
            format!("q{}", self.bits)
        } else {
            "bf16".to_string()
        }
    }
}

/// Cast every float tensor (f32/f16/bf16) to bf16 in place, preserving non-float tensors (e.g. already
/// quantized u32 packs). The parity-safe base for `quantize` — a no-op for the already-bf16 Mochi DiT.
fn cast_floats_bf16(map: &mut HashMap<String, Array>) -> Result<()> {
    for v in map.values_mut() {
        if matches!(v.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
            && v.dtype() != Dtype::Bfloat16
        {
            *v = v.as_dtype(Dtype::Bfloat16)?;
        }
    }
    Ok(())
}

/// Selectively quantize the [`MOCHI_QUANT_SUFFIXES`] Linears in place: each matched `{base}.weight`
/// (bf16) becomes `{base}.weight`(u32 packed) + `{base}.scales` + `{base}.biases` via MLX `quantize`.
/// Non-matching tensors (norms, adaLN, biases, embeds, `proj_out`) pass through untouched.
fn quantize_transformer(
    m: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(m.len());
    for (k, v) in m {
        if is_quant_target(&k) {
            let base = k
                .strip_suffix(".weight")
                .expect("is_quant_target ⇒ .weight");
            let (wq, scales, biases) = quantize(&v, group_size, bits)?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
        } else {
            out.insert(k, v);
        }
    }
    Ok(out)
}

/// Materialize + write a weight map to `dir/<name>.safetensors`.
fn save_component(dir: &Path, name: &str, weights: &HashMap<String, Array>) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let arrays: Vec<&Array> = weights.values().collect();
    eval(arrays)?;
    Array::save_safetensors(
        weights.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        dir.join(format!("{name}.safetensors")),
    )?;
    Ok(())
}

/// Pretty-print a JSON value to `path` (`json.dump(..., indent=2)`).
fn write_json(path: PathBuf, value: &serde_json::Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| Error::Msg(format!("serialize {}: {e}", path.display())))?;
    std::fs::write(&path, text)?;
    Ok(())
}

/// Build **one** Mochi tier from the snapshot at `src_snapshot` into `out_dir`. Writes
/// `<out_dir>/transformer/model.safetensors` + `<out_dir>/split_model.json` (+ `quantize_config.json`
/// for the quantized tiers), and stages the SHARED T5-XXL/VAE/tokenizer **once** as siblings of
/// `out_dir` (idempotent — see [`stage_shared_components`]). Returns `out_dir`.
///
/// The dense `bf16` tier ([`MochiConvertOpts::default`]) repacks the transformer bf16 with
/// `quantized:false`; `q4`/`q8` ([`MochiConvertOpts::quant`]) selectively quantize the
/// [`MOCHI_QUANT_SUFFIXES`] Linears. The result loads directly through [`crate::model::load`] (engine
/// id `mochi_1`) by pointing `WeightsSource` at the tier dir.
pub fn convert_and_assemble(
    src_snapshot: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    opts: &MochiConvertOpts,
) -> Result<PathBuf> {
    let src = src_snapshot.as_ref();
    let out = out_dir.as_ref();
    if !src.is_dir() {
        return Err(Error::Msg(format!(
            "mochi convert: source snapshot dir not found: {}",
            src.display()
        )));
    }
    std::fs::create_dir_all(out)?;

    // Load the bf16 DiT shards in the crate's native (raw HF diffusers) naming — no rename/transpose.
    let raw = load_transformer_weights(src)?;
    let mut transformer: HashMap<String, Array> = raw
        .keys()
        .map(|k| {
            (
                k.to_string(),
                raw.require(k).expect("key from keys()").clone(),
            )
        })
        .collect();
    if transformer.is_empty() {
        return Err(Error::Msg(format!(
            "mochi convert: no transformer weights under {}/transformer",
            src.display()
        )));
    }
    cast_floats_bf16(&mut transformer)?;
    let n_quant = if opts.quantize {
        let before = transformer.keys().filter(|k| is_quant_target(k)).count();
        transformer = quantize_transformer(transformer, opts.bits, opts.group_size)?;
        before
    } else {
        0
    };

    save_component(&out.join("transformer"), "model", &transformer)?;
    drop(transformer);

    // Shared components (staged once as siblings of the tier dir).
    let shared_root = out.parent().unwrap_or(out);
    stage_shared_components(src, shared_root)?;

    // Manifest sidecars.
    let mut manifest = serde_json::json!({
        "format": "split",
        "model_id": "mochi_1",
        "tier": opts.tier_name(),
        "components": ["transformer"],
        "source": src.display().to_string(),
        "quantized": opts.quantize,
    });
    if opts.quantize {
        manifest["quantization_bits"] = serde_json::Value::from(opts.bits);
        manifest["quantization_group_size"] = serde_json::Value::from(opts.group_size);
        manifest["quantized_linears"] = serde_json::Value::from(n_quant);
        // HF-convention sidecar (mirrors LTX). The loader reads the geometry from `split_model.json`;
        // this is emitted for downstream-tooling compatibility, not read here.
        write_json(
            out.join("quantize_config.json"),
            &serde_json::json!({"quantization": {"bits": opts.bits, "group_size": opts.group_size}}),
        )?;
    }
    write_json(out.join("split_model.json"), &manifest)?;

    Ok(out.to_path_buf())
}

/// Materialize the SHARED, tier-independent components ([`SHARED_COMPONENTS`]: the T5-XXL text encoder,
/// the AsymmVAE, and the tokenizer) into `shared_root` **once** — a symlink to each snapshot subdir
/// (cheap; the MLX loaders read the fp32 T5 shards + `vae/config.json` directly through it). Skips any
/// component that already exists in `shared_root`, so building the three tiers stages the (large) T5
/// only once. Returns the resolved shared-component paths.
pub fn stage_shared_components(
    src_snapshot: impl AsRef<Path>,
    shared_root: impl AsRef<Path>,
) -> Result<Vec<PathBuf>> {
    let src = src_snapshot.as_ref();
    let root = shared_root.as_ref();
    std::fs::create_dir_all(root)?;
    let mut staged = Vec::new();
    for component in SHARED_COMPONENTS {
        let dst = root.join(component);
        let source = src.join(component);
        if !source.exists() {
            return Err(Error::Msg(format!(
                "mochi convert: shared component `{component}` missing from snapshot {}",
                src.display()
            )));
        }
        if !dst.exists() {
            link_or_copy_dir(&source, &dst)?;
        }
        staged.push(dst);
    }
    Ok(staged)
}

/// Link `source` → `dst` (a directory symlink; the loaders read straight through it). Falls back to a
/// recursive copy when a symlink can't be created (e.g. a filesystem without symlink support).
fn link_or_copy_dir(source: &Path, dst: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        if std::os::unix::fs::symlink(source, dst).is_ok() {
            return Ok(());
        }
    }
    copy_dir_recursive(source, dst)
}

/// Recursively copy `source` into `dst`, resolving symlinks (so a copied snapshot subdir materializes
/// the blob-store targets). Only used on the symlink-unsupported fallback path.
fn copy_dir_recursive(source: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        // `metadata` follows symlinks, so a snapshot's symlink-to-blob copies the real file.
        let meta = std::fs::metadata(&from)?;
        if meta.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quant predicate matches exactly the Linears the packed loader consumes: visual `to_q/k/v` +
    /// `to_out.0`, the added `add_{q,k,v}_proj` + `to_add_out`, and both SwiGLU FFNs' `net.0.proj`/`net.2`.
    #[test]
    fn quant_predicate_selects_dit_linears() {
        for k in [
            "transformer_blocks.0.attn1.to_q.weight",
            "transformer_blocks.5.attn1.to_k.weight",
            "transformer_blocks.9.attn1.to_v.weight",
            "transformer_blocks.9.attn1.to_out.0.weight",
            "transformer_blocks.3.attn1.add_q_proj.weight",
            "transformer_blocks.3.attn1.add_k_proj.weight",
            "transformer_blocks.3.attn1.add_v_proj.weight",
            "transformer_blocks.3.attn1.to_add_out.weight",
            "transformer_blocks.7.ff.net.0.proj.weight",
            "transformer_blocks.7.ff.net.2.weight",
            "transformer_blocks.7.ff_context.net.0.proj.weight",
            "transformer_blocks.7.ff_context.net.2.weight",
        ] {
            assert!(is_quant_target(k), "should quantize: {k}");
        }
        // Dense: qk-norms, adaLN modulation, biases, patchify, pos-freq, embeds, proj_out.
        for k in [
            "transformer_blocks.0.attn1.norm_q.weight",
            "transformer_blocks.0.attn1.norm_added_k.weight",
            "transformer_blocks.0.attn1.to_out.0.bias",
            "transformer_blocks.0.norm1.linear.weight",
            "transformer_blocks.0.norm1_context.linear.weight",
            "patch_embed.proj.weight",
            "pos_frequencies",
            "time_embed.caption_proj.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
        ] {
            assert!(!is_quant_target(k), "should stay dense: {k}");
        }
        // `.ff.net.2` must not spuriously match `.ff_context.net.2` (distinct suffixes).
        assert!(is_quant_target("blocks.0.ff.net.2.weight"));
        assert!(is_quant_target("blocks.0.ff_context.net.2.weight"));
    }
}
