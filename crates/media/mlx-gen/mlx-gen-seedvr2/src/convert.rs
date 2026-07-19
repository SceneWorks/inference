//! Native weight converter (sc-4813): the `numz/SeedVR2_comfyUI` checkpoints
//! (`seedvr2_ema_{3b,7b}_fp16.safetensors`, `ema_vae_fp16.safetensors`) → the MLX-native key/layout
//! the Rust modules load. Port of the mflux `SeedVR2WeightMapping`; no Python.
//!
//! - **VAE:** keys are unchanged; conv weights are torch `[out,in,kT,kH,kW]` → MLX `[out,kT,kH,kW,in]`
//!   (any 5-D weight). Everything else passes through.
//! - **DiT:** rename the dotted attention/ada submodules (`attn.proj_qkv.vid` → `attn.proj_qkv_vid`,
//!   `ada.vid` → `ada.params_vid`, `attn.rope.rope.freqs` → `attn.rope.freqs`, …). Shared layers store
//!   the attention projections once under `.all`; they are duplicated into both `_vid` and `_txt`
//!   (the attention always uses separate projections). MLP keys pass through.

use std::collections::HashMap;

use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

/// Convert the raw VAE checkpoint: transpose every 5-D conv weight to channels-last.
pub fn convert_vae(raw: &Weights) -> Result<Weights> {
    let mut out = Weights::empty();
    let keys: Vec<String> = raw.keys().map(String::from).collect();
    for k in keys {
        let v = raw.require(&k)?;
        let nv = if v.ndim() == 5 {
            v.transpose_axes(&[0, 2, 3, 4, 1])?
        } else {
            v.clone()
        };
        out.insert(k, nv);
    }
    Ok(out)
}

const ATTN_SUBS: [&str; 4] = ["proj_qkv", "proj_out", "norm_q", "norm_k"];

/// Map a raw DiT key to its converted target name(s) (two for a shared-layer attention `.all`).
fn dit_targets(k: &str) -> Vec<String> {
    // output AdaLN scale/shift live under `vid_out_ada.` in the checkpoint, flat in the model.
    if let Some(rest) = k.strip_prefix("vid_out_ada.") {
        return vec![rest.to_string()];
    }
    if k.contains(".attn.rope.rope.freqs") {
        return vec![k.replace(".attn.rope.rope.", ".attn.rope.")];
    }
    for sub in ATTN_SUBS {
        let all = format!(".attn.{sub}.all.");
        if k.contains(&all) {
            return vec![
                k.replace(&all, &format!(".attn.{sub}_vid.")),
                k.replace(&all, &format!(".attn.{sub}_txt.")),
            ];
        }
        for stream in ["vid", "txt"] {
            let from = format!(".attn.{sub}.{stream}.");
            if k.contains(&from) {
                return vec![k.replace(&from, &format!(".attn.{sub}_{stream}."))];
            }
        }
    }
    for stream in ["vid", "txt", "all"] {
        let from = format!(".ada.{stream}.");
        if k.contains(&from) {
            return vec![k.replace(&from, &format!(".ada.params_{stream}."))];
        }
    }
    vec![k.to_string()] // mlp.* and top-level pass through
}

/// Map a raw DiT key to the canonical on-disk form written by the offline converter.
///
/// Unlike [`dit_targets`], this deliberately keeps shared attention tensors under their single
/// checkpoint `.all` key. The production loader already expands that key into the model's `_vid`
/// and `_txt` names via [`convert_dit`], so serializing those two aliases would only store the same
/// tensor payload twice. Non-shared 7B attention keys and every other rename remain unchanged.
fn dit_storage_target(k: &str) -> String {
    for sub in ATTN_SUBS {
        if k.contains(&format!(".attn.{sub}.all.")) {
            return k.to_string();
        }
    }
    // Every non-shared source key has exactly one target.
    dit_targets(k)
        .into_iter()
        .next()
        .expect("dit_targets always returns at least one target")
}

/// Convert raw DiT weights to the canonical, deduplicated turnkey representation.
fn convert_dit_for_storage(raw: &Weights) -> Result<Weights> {
    let mut out = Weights::empty();
    for k in raw.keys().map(String::from).collect::<Vec<_>>() {
        out.insert(dit_storage_target(&k), raw.require(&k)?.clone());
    }
    Ok(out)
}

/// Convert the raw DiT checkpoint (key renames; no transposes — all weights are 2-D).
///
/// Shared-layer `.attn.*.all` tensors are duplicated into `_vid`/`_txt` keys as two refcounted
/// **handles of one array** — no extra buffer. Keep it that way: any caller-side per-key transform
/// applied *after* this (e.g. a dtype cast) turns each handle into its own lazy node and
/// materializes two buffers (MLX does not CSE across arrays), which is why
/// `Seedvr2Pipeline::load` casts **before** converting (F-012).
pub fn convert_dit(raw: &Weights) -> Result<Weights> {
    let mut out = Weights::empty();
    let keys: Vec<String> = raw.keys().map(String::from).collect();
    for k in keys {
        let v = raw.require(&k)?;
        for target in dit_targets(&k) {
            out.insert(target, v.clone());
        }
    }
    Ok(out)
}

/// Convert both raw files in `src_dir` and write `vae.safetensors` + `transformer.safetensors`
/// into `out_dir`. `dit_file` selects 3B/7B.
///
/// Shared 3B attention tensors remain singular under `.all` in `transformer.safetensors`; the
/// ordinary load-time [`convert_dit`] seam expands them to the model's `_vid`/`_txt` aliases.
pub fn convert_to_dir(
    src_dir: impl AsRef<std::path::Path>,
    dit_file: &str,
    out_dir: impl AsRef<std::path::Path>,
) -> Result<()> {
    let src = src_dir.as_ref();
    let out = out_dir.as_ref();
    std::fs::create_dir_all(out)?;

    let vae = convert_vae(&Weights::from_file(src.join("ema_vae_fp16.safetensors"))?)?;
    save_weights(&vae, &out.join("vae.safetensors"))?;
    let dit = convert_dit_for_storage(&Weights::from_file(src.join(dit_file))?)?;
    save_weights(&dit, &out.join("transformer.safetensors"))?;
    Ok(())
}

fn save_weights(w: &Weights, path: &std::path::Path) -> Result<()> {
    let map: HashMap<String, Array> = w
        .keys()
        .map(|k| (k.to_string(), w.get(k).unwrap().clone()))
        .collect();
    let arrays: Vec<&Array> = map.values().collect();
    eval(arrays)?;
    Array::save_safetensors(
        map.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        path,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_dit() -> Weights {
        let mut w = Weights::empty();
        w.insert(
            "blocks.10.attn.proj_qkv.all.weight",
            Array::from_slice(&vec![0.25f32; 4096], &[64, 64]),
        );
        w.insert(
            "blocks.2.attn.proj_qkv.vid.weight",
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        w.insert(
            "blocks.2.attn.proj_qkv.txt.weight",
            Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]),
        );
        w.insert("blocks.10.ada.all.weight", Array::from_f32(9.0));
        w
    }

    fn temp_file(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mlx-gen-seedvr2-{label}-{}-{}.safetensors",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ))
    }

    #[test]
    fn offline_snapshot_stores_shared_attention_once_and_loader_expands_it() {
        let raw = synthetic_dit();
        let stored = convert_dit_for_storage(&raw).unwrap();

        assert!(stored.get("blocks.10.attn.proj_qkv.all.weight").is_some());
        assert!(stored.get("blocks.10.attn.proj_qkv_vid.weight").is_none());
        assert!(stored.get("blocks.10.attn.proj_qkv_txt.weight").is_none());
        // The non-shared (7B-style) stream keys and all ordinary renames retain their old format.
        assert!(stored.get("blocks.2.attn.proj_qkv_vid.weight").is_some());
        assert!(stored.get("blocks.2.attn.proj_qkv_txt.weight").is_some());
        assert!(stored.get("blocks.10.ada.params_all.weight").is_some());

        let compact_path = temp_file("compact");
        let legacy_path = temp_file("legacy");
        save_weights(&stored, &compact_path).unwrap();
        save_weights(&convert_dit(&raw).unwrap(), &legacy_path).unwrap();

        let compact_bytes = std::fs::metadata(&compact_path).unwrap().len();
        let legacy_bytes = std::fs::metadata(&legacy_path).unwrap().len();
        assert!(
            compact_bytes + 4096 * size_of::<f32>() as u64 <= legacy_bytes,
            "one 4096-element shared tensor payload must disappear: compact={compact_bytes}, legacy={legacy_bytes}"
        );

        // Exercise the exact conversion seam used by Seedvr2Pipeline::load on the emitted file.
        let reloaded = Weights::from_file(&compact_path).unwrap();
        let loaded = convert_dit(&reloaded).unwrap();
        let vid = loaded
            .require("blocks.10.attn.proj_qkv_vid.weight")
            .unwrap();
        let txt = loaded
            .require("blocks.10.attn.proj_qkv_txt.weight")
            .unwrap();
        assert_eq!(vid.as_slice::<f32>(), txt.as_slice::<f32>());
        assert_eq!(vid.as_slice::<f32>(), &[0.25; 4096]);

        std::fs::remove_file(compact_path).unwrap();
        std::fs::remove_file(legacy_path).unwrap();
    }
}
