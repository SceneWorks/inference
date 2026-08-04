//! Offline pre-quantization: read a dense Chroma diffusers snapshot and write a packed Q4/Q8 turnkey
//! that [`crate::quant`] (via [`crate::model::load_chroma`]) loads with no dense bf16/f32 transient.
//! Mirrors `mlx_gen_sdxl::convert` / `mlx_gen_sensenova::convert` (same `mlx_gen::quant::quantize_map`,
//! byte-equal to the load-time `.quantize` seam), differing in the Chroma key layout and quant scope.
//!
//! Chroma quantizes **one** component — the DiT `transformer/` (the fork's `nn.quantize`, wired in
//! [`crate::transformer::ChromaTransformer::quantize`]): the double blocks' attention + FFN Linears
//! and the single blocks' attention + `proj_mlp`/`proj_out`. Everything else stays **dense in every
//! tier**:
//!
//! * The transformer's own `x_embedder` / `context_embedder` / top-level `proj_out` and the entire
//!   distilled-guidance **Approximator** (`distilled_guidance_layer.*`, which drives all per-block
//!   modulation) — small / precision-sensitive, kept dense to match `is_transformer_target`.
//! * The T5-XXL **text encoder** (`text_encoder/`) and the FLUX.1 16-ch **VAE** (`vae/`) are packed
//!   AT THE SELECTED TIER (sc-16462), exactly as `mlx_gen_flux::convert::prequantize_turnkey` already
//!   packs the same T5-XXL (`text_encoder_2/`) and AutoencoderKL. Carrying them at bf16 on a q4/q8
//!   tier is above-tier residency — the thing `config/tier-integrity.jsonc` exists to eliminate. T5
//!   packs at [`T5_GROUP_SIZE`] (32), which measurably halves packed-T5 render error versus the
//!   codebase-default 64 at the same width; the VAE packs its attention projections at [`GROUP_SIZE`].
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
    copy_dir, copy_turnkey_assets, quantize_map, save_map, write_quantized_config,
};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::quant::GROUP_SIZE;

/// The single packed weight file each auxiliary component ships.
const AUXILIARY_FILE: &str = "model.safetensors";

/// T5's affine group size. The relative-attention-bias table is 64 wide, so 128 cannot pack the full
/// surface; 32 is chosen over the codebase default 64 on measured render error at equal width.
pub const T5_GROUP_SIZE: i32 = 32;

/// The single packed weight file the turnkey ships for the transformer (replaces the source's sharded
/// `diffusion_pytorch_model-0000N-of-0000M.safetensors`). The loader globs `*.safetensors` under
/// `transformer/`, so one flat file suffices; its stem matches the dense master so nothing downstream
/// changes.
const TRANSFORMER_FILE: &str = "diffusion_pytorch_model.safetensors";

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

/// T5-XXL packs its complete quantizable surface — every 2-D Linear plus the token-embedding and
/// relative-position-bias tables. The shared [`quantize_map`] shape guard keeps 1-D RMSNorm vectors
/// dense (affine quantization does not target vectors). This matches `mlx_gen_flux`'s `pack_all`.
fn is_t5_target(_base: &str) -> bool {
    true
}

/// FLUX.1 VAE packed surface: encoder/decoder mid-block attention QKV/out projections. Convolutions
/// and GroupNorms stay dense; the shared Z-Image VAE loader packed-detects exactly these keys.
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
    save_map(&dst.join(AUXILIARY_FILE), &map)?;
    write_quantized_config(src, dst, bits, group_size)
}

/// Assemble a full pre-quantized turnkey Chroma snapshot in `dst_root`: pack the DiT `transformer/`
/// block Linears into one `transformer/diffusion_pytorch_model.safetensors` (+ annotated
/// `config.json`), pack the T5-XXL `text_encoder/` and FLUX.1 `vae/` **at the same width**, and copy
/// the tokenizer / scheduler / `model_index.json` / license verbatim (deref symlinks). The result
/// loads via [`crate::model::load_chroma`] (packed weights auto-detect) with no dense transient.
/// `bits` = 4 (Q4 tier) or 8 (Q8 tier). The **bf16 tier** is the dense source itself (no conversion —
/// mirror it; see the tier builder in `tests/prequantize_real_weights.rs`).
///
/// EVERY component is packed at `bits`. There is deliberately no per-component width override: the
/// tier a user selects is a statement about the whole render, not a memory budget to be spent
/// wherever it is cheapest. Running any segment above the selected tier sidesteps that choice — a
/// user who wants a better text encoder chooses q8, they do not get one silently smuggled into q4.
/// This is why `chroma1_*` accumulated six `config/tier-integrity.jsonc` exception rows while
/// `flux1_*`, which packs the same T5-XXL at its route width, has none.
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;
    let auxiliary_bits = bits;

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

    // T5-XXL + FLUX.1 VAE: pack at the auxiliary width (sc-16462) rather than mirroring bf16.
    quantize_component(
        &src_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        auxiliary_bits,
        T5_GROUP_SIZE,
        is_t5_target,
    )?;
    quantize_component(
        &src_root.join("vae"),
        &dst_root.join("vae"),
        auxiliary_bits,
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

/// Repack ONLY `text_encoder/` and `vae/` onto an existing shipped tier, copying every other path
/// byte-for-byte (`transformer/`, tokenizer, scheduler, `model_index.json`, license).
///
/// This is the sc-16462 publish operation. The shipped q4/q8 tiers already exist at pinned
/// revisions, and their transformer bytes must not move — a new revision has to change the
/// auxiliaries and nothing else, or the change stops being an isolated tier-integrity fix and
/// becomes an unreviewed transformer swap. Rebuilding the whole turnkey from the bf16 source does
/// NOT reproduce the shipped transformer byte stream (shard merge order differs even though the
/// quantized VALUES are deterministic), so copying is the only sound way to hold it fixed.
///
/// The width is DERIVED from the baseline's own packed transformer, never passed in, so this cannot
/// mint a tier whose text encoder sits above the tier the user selected.
pub fn repack_auxiliaries(baseline_root: &Path, dst_root: &Path) -> Result<()> {
    let auxiliary_bits = mlx_gen::quant::packed_quant_bits_at(&baseline_root.join("transformer"))?
        .ok_or_else(|| {
            Error::Msg(format!(
                "chroma repack: baseline tier {} has no packed transformer, so there is no selected \
                 tier to match the auxiliaries to",
                baseline_root.display()
            ))
        })?;
    if !matches!(auxiliary_bits, 4 | 8) {
        return Err(Error::Msg(format!(
            "chroma repack: baseline transformer declares Q{auxiliary_bits}; expected Q4 or Q8"
        )));
    }
    for required in ["transformer", "text_encoder", "vae"] {
        if !baseline_root.join(required).is_dir() {
            return Err(Error::Msg(format!(
                "chroma repack: baseline tier {} has no {required}/",
                baseline_root.display()
            )));
        }
    }
    if mlx_gen::quant::packed_quant_bits_at(&baseline_root.join("text_encoder"))?.is_some() {
        return Err(Error::Msg(format!(
            "chroma repack: {}/text_encoder is already packed; repacking a packed source would \
             quantize codes as if they were weights",
            baseline_root.display()
        )));
    }
    if dst_root.exists() {
        return Err(Error::Msg(format!(
            "chroma repack: destination already exists: {}",
            dst_root.display()
        )));
    }
    std::fs::create_dir_all(dst_root)?;

    // Everything that is not an auxiliary is copied verbatim.
    for entry in std::fs::read_dir(baseline_root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_str().ok_or_else(|| {
            Error::Msg("chroma repack: non-UTF-8 entry in baseline tier".to_string())
        })?;
        if matches!(name, "text_encoder" | "vae") {
            continue;
        }
        let src = entry.path();
        let dst = dst_root.join(name);
        if src.is_dir() {
            copy_dir(&src, &dst)?;
        } else {
            std::fs::copy(&src, &dst)?;
        }
    }

    quantize_component(
        &baseline_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        auxiliary_bits,
        T5_GROUP_SIZE,
        is_t5_target,
    )?;
    quantize_component(
        &baseline_root.join("vae"),
        &dst_root.join("vae"),
        auxiliary_bits,
        GROUP_SIZE,
        is_vae_target,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::{Array, Dtype};

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

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
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
