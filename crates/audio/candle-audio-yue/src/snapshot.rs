//! Resolving a staged YuE LM snapshot to the tier directory a stage loads.
//!
//! The SceneWorks YuE LM rehosts are *tiered* (`scripts/audio/prepare_yue_assets.py`): the repo
//! root holds `sceneworks-tiers.json` plus one self-contained directory per tier — `bf16/` (the
//! upstream dense checkpoint), `q8/` and `q4/` (candle-llm prepared, projections stored as GGML
//! blocks, the tier recorded in `config.json`'s `quantization` block). A caller may provision
//! either the root or one tier directory, so both resolve here.

use std::path::{Path, PathBuf};

use candle_audio::gen_core;

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

/// Resolve `root` (a tiered snapshot root, or one tier directory) to the directory holding the
/// requested tier, and name the tier it holds.
///
/// - A directory with its own `config.json` is one tier: its tier is detected, and a `requested`
///   tier that differs is refused (the caller asserted a tier that was not staged).
/// - Otherwise `root` must hold tier subdirectories: `requested` picks one; with nothing
///   requested, a lone staged tier is used, else the dense `bf16/` (the unquantized load).
pub fn resolve_tier_dir(
    root: &Path,
    requested: Option<Tier>,
    what: &str,
) -> gen_core::Result<(PathBuf, Tier)> {
    let err = |m: String| gen_core::Error::Msg(format!("candle-audio-yue: {what} snapshot {m}"));
    if root.join("config.json").is_file() {
        let staged = detect_tier(root, what)?;
        return match requested {
            Some(want) if want != staged => Err(err(format!(
                "{} holds the {staged:?} tier, but {want:?} was requested",
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
        (Some(want), _) => {
            return Err(err(format!(
                "{} has no `{}/` tier directory (staged: {staged:?})",
                root.display(),
                tier_dir_name(want)
            )))
        }
        (None, [only]) => *only,
        (None, _) if staged.contains(&Tier::Bf16) => Tier::Bf16,
        (None, _) => {
            return Err(err(format!(
                "{} holds neither a `config.json` nor a single tier directory (staged: \
                 {staged:?}); request a tier",
                root.display()
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
    fn a_tier_directory_resolves_to_itself_and_refuses_a_different_assertion() {
        let root = tempfile::tempdir().unwrap();
        let q8 = root.path().join("only");
        stage(&q8, Some(8));
        assert_eq!(
            resolve_tier_dir(&q8, None, "s2").unwrap(),
            (q8.clone(), Tier::Q8)
        );
        assert_eq!(
            resolve_tier_dir(&q8, Some(Tier::Q8), "s2").unwrap().1,
            Tier::Q8
        );
        let err = resolve_tier_dir(&q8, Some(Tier::Q4), "s2").unwrap_err();
        assert!(err.to_string().contains("holds the Q8 tier"), "{err}");
    }

    #[test]
    fn a_tiered_root_picks_the_requested_lone_or_dense_tier() {
        let root = tempfile::tempdir().unwrap();
        stage(&root.path().join("bf16"), None);
        stage(&root.path().join("q4"), Some(4));
        let r = |t| resolve_tier_dir(root.path(), t, "s2");
        assert_eq!(r(None).unwrap(), (root.path().join("bf16"), Tier::Bf16));
        assert_eq!(
            r(Some(Tier::Q4)).unwrap(),
            (root.path().join("q4"), Tier::Q4)
        );
        assert!(r(Some(Tier::Q8))
            .unwrap_err()
            .to_string()
            .contains("no `q8/`"));

        let lone = tempfile::tempdir().unwrap();
        stage(&lone.path().join("q8"), Some(8));
        assert_eq!(
            resolve_tier_dir(lone.path(), None, "s2").unwrap().1,
            Tier::Q8
        );

        let mislabelled = tempfile::tempdir().unwrap();
        stage(&mislabelled.path().join("q8"), Some(4));
        assert!(resolve_tier_dir(mislabelled.path(), Some(Tier::Q8), "s2")
            .unwrap_err()
            .to_string()
            .contains("declares Q4"));
        let empty = tempfile::tempdir().unwrap();
        assert!(resolve_tier_dir(empty.path(), None, "s2").is_err());
    }
}
