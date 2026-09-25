//! Resolving a staged YuE LM snapshot (stage 1 or stage 2) to the tier directory and the
//! `LoadSpec` a stage loads.
//!
//! The SceneWorks YuE LM rehosts are *tiered* (`scripts/audio/prepare_yue_assets.py`): the repo
//! root holds `sceneworks-tiers.json` plus one self-contained directory per tier — `bf16/` (the
//! upstream dense checkpoint), `q8/` and `q4/` (candle-llm prepared, projections stored as GGML
//! blocks, the tier recorded in `config.json`'s `quantization` block). A caller may provision the
//! root, one tier directory, or only `bf16/`; an asserted tier that is not staged pre-quantized is
//! quantized from the dense checkpoint on load.

use std::path::{Path, PathBuf};

use candle_audio::gen_core;
use candle_llm::core_llm::{LoadSpec, Quantize};

use crate::config::Tier;

/// The directory each tier lives in under a tiered snapshot root.
fn tier_dir_name(tier: Tier) -> &'static str {
    match tier {
        Tier::Bf16 => "bf16",
        Tier::Q8 => "q8",
        Tier::Q4 => "q4",
    }
}

/// The tier a single tier directory holds, read from its `config.json` `quantization` block
/// (none ⇒ the dense bf16 checkpoint).
fn detect_tier(dir: &Path, what: &str) -> gen_core::Result<Tier> {
    let cfg = candle_llm::ModelConfig::from_dir(dir).map_err(|e| {
        gen_core::Error::Msg(format!(
            "candle-audio-yue: {what} snapshot {}: {e}",
            dir.display()
        ))
    })?;
    Ok(match cfg.quantization.map(|q| q.bits()) {
        None => Tier::Bf16,
        Some(8) => Tier::Q8,
        Some(_) => Tier::Q4,
    })
}

/// Resolve `root` (a tiered snapshot root, or one snapshot directory) to the directory to load,
/// and name the tier that directory **stores** (a dense `Bf16` directory loaded for an asserted
/// Q8/Q4 is quantized on load — see [`lm_load_spec`]).
///
/// - A directory with its own `config.json` is one snapshot: its tier is detected. An asserted
///   Q8/Q4 over a dense snapshot quantizes on load; over a snapshot stored at a *different*
///   quantized tier it is refused (a quantized weight is never re-quantized).
/// - Otherwise `root` holds tier subdirectories: the asserted tier's directory when staged, else
///   `bf16/` (quantized on load). With nothing asserted, a lone staged tier, else `bf16/`.
/// - A tier directory whose `config.json` declares a different tier than its name is refused.
pub fn resolve_tier_dir(
    root: &Path,
    requested: Option<Tier>,
    what: &str,
) -> gen_core::Result<(PathBuf, Tier)> {
    let err = |m: String| gen_core::Error::Msg(format!("candle-audio-yue: {what} snapshot {m}"));
    if root.join("config.json").is_file() {
        let staged = detect_tier(root, what)?;
        return match requested {
            Some(want) if staged != Tier::Bf16 && want != staged => Err(err(format!(
                "{} is stored pre-quantized at {staged:?}; it cannot be loaded as {want:?}",
                root.display()
            ))),
            _ => Ok((root.to_path_buf(), staged)),
        };
    }
    let staged: Vec<Tier> = [Tier::Bf16, Tier::Q8, Tier::Q4]
        .into_iter()
        .filter(|&t| root.join(tier_dir_name(t)).join("config.json").is_file())
        .collect();
    let tier = match (requested, staged.as_slice()) {
        (Some(want), _) if staged.contains(&want) => want,
        (_, [only]) if requested.is_none() => *only,
        (_, _) if staged.contains(&Tier::Bf16) => Tier::Bf16,
        (_, _) => {
            return Err(err(format!(
                "{} holds no `config.json`, no `bf16/` to quantize from{} (staged tiers: \
                 {staged:?})",
                root.display(),
                requested.map_or(String::new(), |t| format!(
                    " and no `{}/`",
                    tier_dir_name(t)
                ))
            )))
        }
    };
    let dir = root.join(tier_dir_name(tier));
    let found = detect_tier(&dir, what)?;
    if found != tier {
        return Err(err(format!(
            "{} is labelled {tier:?} but its config.json declares {found:?}",
            dir.display()
        )));
    }
    Ok((dir, tier))
}

/// The `LoadSpec` for a resolved snapshot directory: an asserted Q8/Q4 is passed through, so a
/// dense directory is quantized on load and a prepared one is checked against its stored tier;
/// `None` / `Bf16` loads the directory as stored (its persisted `quantization` block, if any).
pub fn lm_load_spec(dir: &Path, requested: Option<Tier>) -> LoadSpec {
    let mut spec = LoadSpec::dense(dir.to_string_lossy().into_owned());
    spec.quantize = match requested {
        Some(Tier::Q8) => Some(Quantize::Q8),
        Some(Tier::Q4) => Some(Quantize::Q4),
        Some(Tier::Bf16) | None => None,
    };
    spec
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(dir: &Path, quant_bits: Option<u32>) {
        std::fs::create_dir_all(dir).unwrap();
        let quant = quant_bits.map_or(String::new(), |b| {
            format!(r#", "quantization": {{"bits": {b}, "storage": "ggml"}}"#)
        });
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                    "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
                    "num_hidden_layers": 1, "num_key_value_heads": 2, "vocab_size": 64,
                    "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
                    "max_position_embeddings": 128{quant}}}"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn a_snapshot_directory_resolves_to_itself() {
        let root = tempfile::tempdir().unwrap();
        let q8 = root.path().join("only-q8");
        stage(&q8, Some(8));
        assert_eq!(
            resolve_tier_dir(&q8, None, "s2").unwrap(),
            (q8.clone(), Tier::Q8)
        );
        assert_eq!(
            resolve_tier_dir(&q8, Some(Tier::Q8), "s2").unwrap().1,
            Tier::Q8
        );
        // A pre-quantized snapshot is never re-quantized to another tier.
        let err = resolve_tier_dir(&q8, Some(Tier::Q4), "s2").unwrap_err();
        assert!(
            err.to_string().contains("stored pre-quantized at Q8"),
            "{err}"
        );
        // A dense snapshot serves any asserted tier (quantized on load).
        let dense = root.path().join("dense");
        stage(&dense, None);
        assert_eq!(
            resolve_tier_dir(&dense, Some(Tier::Q4), "s2").unwrap(),
            (dense.clone(), Tier::Bf16)
        );
    }

    #[test]
    fn a_tiered_root_picks_the_asserted_lone_or_dense_tier() {
        let root = tempfile::tempdir().unwrap();
        stage(&root.path().join("bf16"), None);
        stage(&root.path().join("q8"), Some(8));
        let r = |t| resolve_tier_dir(root.path(), t, "s2");
        assert_eq!(r(None).unwrap(), (root.path().join("bf16"), Tier::Bf16));
        assert_eq!(
            r(Some(Tier::Q8)).unwrap(),
            (root.path().join("q8"), Tier::Q8)
        );
        // An unstaged tier falls back to bf16, quantized on load.
        assert_eq!(
            r(Some(Tier::Q4)).unwrap(),
            (root.path().join("bf16"), Tier::Bf16)
        );

        let lone = tempfile::tempdir().unwrap();
        stage(&lone.path().join("q8"), Some(8));
        assert_eq!(
            resolve_tier_dir(lone.path(), None, "s2").unwrap().1,
            Tier::Q8
        );
        // No bf16 to quantize from and the asserted tier is not staged.
        assert!(resolve_tier_dir(lone.path(), Some(Tier::Q4), "s2")
            .unwrap_err()
            .to_string()
            .contains("no `bf16/` to quantize from"));

        let mislabelled = tempfile::tempdir().unwrap();
        stage(&mislabelled.path().join("q8"), Some(4));
        assert!(resolve_tier_dir(mislabelled.path(), Some(Tier::Q8), "s2")
            .unwrap_err()
            .to_string()
            .contains("declares Q4"));
        let empty = tempfile::tempdir().unwrap();
        assert!(resolve_tier_dir(empty.path(), None, "s2").is_err());
    }

    #[test]
    fn the_load_spec_carries_the_asserted_tier() {
        let dir = Path::new("/staged/s");
        assert_eq!(lm_load_spec(dir, None).quantize, None);
        assert_eq!(lm_load_spec(dir, Some(Tier::Bf16)).quantize, None);
        assert_eq!(
            lm_load_spec(dir, Some(Tier::Q8)).quantize,
            Some(Quantize::Q8)
        );
        assert_eq!(
            lm_load_spec(dir, Some(Tier::Q4)).quantize,
            Some(Quantize::Q4)
        );
        assert_eq!(lm_load_spec(dir, None).source, "/staged/s");
    }
}
