//! Offline pre-quantization: read a dense SDXL snapshot and write a packed Q4/Q8 snapshot that
//! [`crate::quant`] (via [`crate::model::load`]) loads with no dense fp16/f32 transient. Mirrors
//! `mlx_gen_z_image::convert` / `mlx_gen_flux::convert` (same `mlx_rs::ops::quantize`, byte-equal to
//! the load-time `.quantize` seams), differing in the SDXL key layout and quant scope.
//!
//! SDXL quantizes **three** components (the fork's `nn.quantize`, wired in [`crate::model::load`]):
//! the U-Net's true Linears and BOTH CLIP text encoders. The **VAE is never quantized** (it runs
//! f32 — fp16/int8-unstable), so every tier ships a dense VAE (mirror the source `vae/`). The
//! per-component pack predicates below match the loader's `.quantize` scope exactly (a missed site =
//! that layer loads u32 codes as dense floats = a garbage render — the completeness gate is the
//! real-weight render in `tests/prequantize_real_weights.rs`):
//!
//! * **U-Net** — every 2-D Linear packs; the shared [`quantize_map`] shape guard (2-D, `in % gs ==
//!   0`, `in >= gs`) skips the 1-D GroupNorms and the 4-D convs *including* the 1×1
//!   `conv_shortcut.weight` `[out, in, 1, 1]` (a Linear at load, but 4-D on disk → dense, matching
//!   `ResnetBlock2D::quantize` keeping it dense, sc-3329). The GEGLU `ff.net.0.proj` `[2·hidden, D]`
//!   packs whole — the loader row-slices the packed triple into value/gate halves, byte-identical to
//!   the dense split-then-quantize (quantization is per-row).
//! * **CLIP text encoders** — every 2-D Linear packs EXCEPT the token/position **embeddings**
//!   (2-D `.weight`, but gather lookups that stay dense — matching `ClipTextEncoder::quantize`). The
//!   TE2 `text_projection` (a bare top-level 2-D Linear) packs.
//!
//! Group-B per-crate converter template (sc-8669).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use mlx_gen::quant::{
    copy_dir, copy_turnkey_assets, quantize_map, save_map, write_quantized_config,
};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::quant::GROUP_SIZE;

// ============================================================================================
// Pack predicates (operate on the **base** = the on-disk key minus its `.weight`).
// ============================================================================================

/// U-Net: pack every quantizable leaf, leaning on the shared [`quantize_map`] shape guard to skip the
/// 1-D norms and the 4-D convs (incl. the 1×1 `conv_shortcut`). Every 2-D `.weight` in the SDXL U-Net
/// is a quantized Linear (time/add-embedding MLPs, attention, GEGLU FFN, `proj_in`/`proj_out`, each
/// ResNet `time_emb_proj`, the Kolors `encoder_hid_proj`), so "pack all shape-eligible" is exactly the
/// load-time `.quantize` scope.
fn is_unet_target(_base: &str) -> bool {
    true
}

/// CLIP text encoder: pack every 2-D Linear EXCEPT the token/position embeddings, which stay dense
/// (gather lookups, not matmuls — matching `ClipTextEncoder::quantize`). The shape guard additionally
/// skips the 1-D LayerNorm scales. Operates on the raw diffusers key minus `.weight`.
fn is_clip_target(base: &str) -> bool {
    !base.ends_with(".token_embedding") && !base.ends_with(".position_embedding")
}

/// Load exactly ONE weight variant of a component dir into a key→`Array` map. A diffusers SDXL
/// snapshot often ships several variants side-by-side (`model.safetensors` + `model.fp16.safetensors`,
/// or a sharded f32 master `*-00001-of-000NN.safetensors` alongside a single-file fp16), so a
/// `Weights::from_dir` glob would collide on the shared keys. We pick the **f32 master** (the exact
/// dense source; `quantize_map` casts to bf16 anyway), preferring: a single `{stem}.safetensors`, else
/// its `{stem}-NNNNN-of-NNNNN.safetensors` shards, else the `{stem}.fp16.safetensors` fallback. The
/// chosen files are merged (shard keys are disjoint).
fn load_component_map(dir: &Path, stem: &str) -> Result<HashMap<String, Array>> {
    let single = dir.join(format!("{stem}.safetensors"));
    let fp16 = dir.join(format!("{stem}.fp16.safetensors"));
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    if single.exists() {
        files.push(single);
    } else {
        // Collect sharded f32 files `{stem}-NNNNN-of-NNNNN.safetensors`.
        let shard_prefix = format!("{stem}-");
        for entry in std::fs::read_dir(dir)? {
            let p = entry?.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.starts_with(&shard_prefix)
                && name.ends_with(".safetensors")
                && name.contains("-of-")
            {
                files.push(p);
            }
        }
        if files.is_empty() && fp16.exists() {
            files.push(fp16);
        }
    }
    if files.is_empty() {
        return Err(Error::Msg(format!(
            "sdxl convert: no `{stem}.safetensors` (single, sharded, or .fp16) in {}",
            dir.display()
        )));
    }
    files.sort();
    let mut map: HashMap<String, Array> = HashMap::new();
    for f in files {
        let w = Weights::from_file(&f)?;
        for k in w.keys().map(str::to_string).collect::<Vec<_>>() {
            let v = w.get(&k).expect("listed key").clone();
            if map.insert(k.clone(), v).is_some() {
                return Err(Error::Msg(format!(
                    "sdxl convert: duplicate key `{k}` across `{stem}` shards in {}",
                    dir.display()
                )));
            }
        }
    }
    Ok(map)
}

/// Pre-quantize one component dir → a packed `{stem}.safetensors` + annotated `config.json` in `dst`.
/// `stem` selects the source weight file(s) ([`load_component_map`]) AND names the single packed
/// output file, so the loader's `resolve_weight_file` finds it exactly as it would the dense master
/// (`unet/diffusion_pytorch_model.safetensors`, `text_encoder/model.safetensors`). `is_target` is the
/// component's pack predicate; `bits` = 4 (Q4) or 8 (Q8) at group size [`GROUP_SIZE`].
fn quantize_component(
    src: &Path,
    dst: &Path,
    stem: &str,
    bits: i32,
    is_target: fn(&str) -> bool,
) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_component_map(src, stem)?, bits, GROUP_SIZE, is_target)?;
    let map = dense_leaves_at_compute_dtype(map)?;
    save_map(&dst.join(format!("{stem}.safetensors")), &map)?;
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

/// The compute dtype every dense leaf of a packed tier runs at — the loader casts a dense snapshot's
/// U-Net and CLIP towers to it, and (since sc-26268) a packed snapshot's dense leaves too.
const DENSE_LEAF_DTYPE: mlx_rs::Dtype = mlx_rs::Dtype::Float16;

/// Write a packed tier's **dense** leaves (convs, norms, embeddings, the packed linears' own biases)
/// at [`DENSE_LEAF_DTYPE`], leaving each affine pack (u32 codes + `.scales`/`.biases`) as
/// `quantize_map` wrote it (sc-26268).
///
/// [`load_component_map`] reads the f32 master when a source ships one, and `quantize_map` keeps
/// every non-target leaf at its source dtype — so a source with an f32 master (RealVisXL,
/// Illustrious) produced a packed U-Net carrying 1.33 GB of f32 dense leaves where `sdxl` (fp16
/// source) carries 0.67 GB. The loader casts them to f16 anyway, so the f32 bytes bought nothing
/// but download size (the bulk of `illustrious-xl-v2` q4's 3.91 GB against `sdxl` q4's 2.70 GB).
fn dense_leaves_at_compute_dtype(map: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
    let mut weights = Weights::from_map(map);
    weights.cast_unpacked_floats(DENSE_LEAF_DTYPE)?;
    Ok(weights.into_tensors())
}

/// Mirror the dense VAE into a tier: the `.fp16` variant alone when the source ships one (every
/// published SDXL-family tier's loader reads it — `loader::resolve_vae_weight_file` falls back to it
/// and upcasts), else the directory verbatim. Before sc-26268 a source shipping both variants put
/// both in every tier: 335 MB of f32 VAE no load needed beside the 167 MB fp16 one.
fn mirror_vae(src: &Path, dst: &Path) -> Result<()> {
    let fp16 = src.join("diffusion_pytorch_model.fp16.safetensors");
    if !fp16.is_file() {
        return copy_dir(src, dst);
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let path = entry?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('.') || !path.is_file() {
            continue;
        }
        if name.ends_with(".safetensors") && path != fp16 {
            continue;
        }
        std::fs::copy(&path, dst.join(name))?;
    }
    Ok(())
}

/// Assemble a full pre-quantized turnkey SDXL snapshot in `dst_root`: pack the U-Net + both CLIP text
/// encoders, mirror the dense VAE, and copy the tokenizers / scheduler / `model_index.json` verbatim.
/// The result loads via [`crate::model::load`] (packed weights auto-detect) with no dense transient.
/// `bits` = 4 (Q4 tier) or 8 (Q8 tier). The bf16 tier is the dense source itself (no conversion —
/// mirror it; see the tier builder in `tests/prequantize_real_weights.rs`).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;
    quantize_component(
        &src_root.join("unet"),
        &dst_root.join("unet"),
        "diffusion_pytorch_model",
        bits,
        is_unet_target,
    )?;
    quantize_component(
        &src_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        "model",
        bits,
        is_clip_target,
    )?;
    quantize_component(
        &src_root.join("text_encoder_2"),
        &dst_root.join("text_encoder_2"),
        "model",
        bits,
        is_clip_target,
    )?;
    // VAE stays dense (never quantized) — mirror the source `vae/` verbatim (deref symlinks).
    let vae_src = src_root.join("vae");
    if vae_src.exists() {
        mirror_vae(&vae_src, &dst_root.join("vae"))?;
    }
    // Tokenizers + scheduler are non-weight assets — copy verbatim.
    for rel in ["tokenizer", "tokenizer_2", "scheduler"] {
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
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::{Array, Dtype};

    /// sc-26268: a tier's dense leaves land at f16 whatever the source master's dtype, the pack is
    /// untouched, and a VAE shipped in both variants is mirrored as its `.fp16` file alone.
    #[test]
    fn packed_tiers_store_dense_leaves_at_f16_and_one_vae_variant() {
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert(
            "conv_in.weight".into(),
            Array::from_slice(&[0.5f32; 4], &[1, 1, 2, 2]),
        );
        map.insert("proj.weight".into(), Array::from_slice(&[1u32], &[1]));
        map.insert(
            "proj.scales".into(),
            Array::from_slice(&[1.0f32], &[1])
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
        );
        let out = dense_leaves_at_compute_dtype(map).unwrap();
        assert_eq!(out["conv_in.weight"].dtype(), Dtype::Float16);
        assert_eq!(out["proj.weight"].dtype(), Dtype::Uint32);
        assert_eq!(out["proj.scales"].dtype(), Dtype::Bfloat16);

        let tmp = tempfile::tempdir().unwrap();
        let (src, dst) = (tmp.path().join("src"), tmp.path().join("dst"));
        std::fs::create_dir_all(&src).unwrap();
        for name in [
            "config.json",
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.fp16.safetensors",
        ] {
            std::fs::write(src.join(name), name).unwrap();
        }
        mirror_vae(&src, &dst).unwrap();
        let mut names: Vec<_> = std::fs::read_dir(&dst)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(
            names,
            ["config.json", "diffusion_pytorch_model.fp16.safetensors"]
        );
    }

    #[test]
    fn clip_predicate_skips_embeddings_packs_projections() {
        for base in [
            "text_model.encoder.layers.0.self_attn.q_proj",
            "text_model.encoder.layers.0.self_attn.out_proj",
            "text_model.encoder.layers.0.mlp.fc1",
            "text_model.encoder.layers.0.mlp.fc2",
            "text_projection",
        ] {
            assert!(is_clip_target(base), "{base} should be packed");
        }
        for base in [
            "text_model.embeddings.token_embedding",
            "text_model.embeddings.position_embedding",
        ] {
            assert!(!is_clip_target(base), "{base} should stay dense");
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The packed triple a Linear becomes is byte-identical to the op the load-time `.quantize` runs
    /// (bf16 cast, group 64) — the sc-8669 round-trip guarantee: pre-quantize-on-disk ==
    /// quantize-at-load. A 4-D conv and a 1-D norm stay dense (shape-guarded out); a CLIP token
    /// embedding stays dense (predicate).
    #[test]
    fn quantize_map_packs_targets_byte_identical_to_load_time_quantize() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        // A U-Net attention Linear (packs) + a GEGLU proj (packs whole) + a 4-D conv_shortcut (dense)
        // + a 1-D GroupNorm (dense).
        map.insert(
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight".into(),
            w.clone(),
        );
        map.insert(
            "down_blocks.0.resnets.0.conv_shortcut.weight".into(),
            Array::zeros::<f32>(&[64, 32, 1, 1]).unwrap(),
        );
        map.insert(
            "down_blocks.0.resnets.0.norm1.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_unet_target).unwrap();

        let wq = out
            .get("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight")
            .expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let scales = out
            .get("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.scales")
            .unwrap();
        let biases = out
            .get("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.biases")
            .unwrap();
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "packed weight != load-time quantize");
        assert!(byte_equal(scales, &esc), "scales != load-time quantize");
        assert!(byte_equal(biases, &ebi), "biases != load-time quantize");

        // The 4-D conv_shortcut is shape-guarded out (stays dense f32, no packed triple).
        let cs = out
            .get("down_blocks.0.resnets.0.conv_shortcut.weight")
            .unwrap();
        assert_eq!(
            cs.dtype(),
            Dtype::Float32,
            "conv_shortcut unchanged (dense)"
        );
        assert!(!out.contains_key("down_blocks.0.resnets.0.conv_shortcut.scales"));
        // The 1-D norm stays dense.
        let n = out.get("down_blocks.0.resnets.0.norm1.weight").unwrap();
        assert_eq!(n.dtype(), Dtype::Float32, "norm unchanged");
    }
}
