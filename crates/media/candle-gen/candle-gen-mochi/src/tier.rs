//! Mochi 1 **MLX pre-quantized tier** resolver (sc-11990, A6 candle-ingest) — the split-file / shared-
//! component layer that lets the packed-detect seam ([`crate::transformer`]'s [`QLinear::linear_detect`]
//! (candle_gen::quant::QLinear::linear_detect) calls) fire on the REAL `SceneWorks/mochi-1-mlx` q4/q8/bf16
//! tier weights, with **no dense staging** and **no key remap**.
//!
//! ## Why a resolver is needed
//!
//! The `mlx-gen-mochi` converter (`convert.rs`, story A6) writes each tier as a **self-contained tier
//! dir** — `<snapshot>/{q4,q8,bf16}/` — carrying `transformer/model.safetensors` + `split_model.json`
//! (+ a `quantize_config.json` sidecar for the quantized tiers), while the **shared** T5-XXL text
//! encoder, AsymmVAE, and tokenizer are staged **once** as siblings of the tier dir (in its parent),
//! not duplicated per tier. So a q4 tier dir holds only the (packed) transformer; the T5/VAE it needs
//! live one level up. The raw diffusers snapshot, by contrast, holds `transformer/`, `text_encoder/`,
//! and `vae/` all under one root. [`MochiTierPaths::detect`] recognizes the tier layout (by its
//! `split_model.json` marker) and resolves the shared root to the tier dir's parent so
//! [`crate::pipeline::Pipeline::load_components`] loads the DiT from the tier dir and the T5/VAE from
//! the parent — the dense snapshot path stays unchanged (no `split_model.json` ⇒ `detect` returns `None`).
//!
//! ## Key spelling is 1:1 — no remap (unlike the LTX tier)
//!
//! The converter keeps the transformer in the **raw HF-diffusers naming** (no renames, no conv
//! transpose on disk), which is **exactly** the spelling [`crate::transformer`] already reads
//! (`transformer_blocks.N.attn1.to_q.weight`, `…ff.net.0.proj.weight`, …). So the `.scales`/`.biases`
//! siblings land at the keys the crate asks for and the packed-detect fires directly — no [`Rename`]
//! backend is needed (the LTX tier remapped `to_out.0`→`to_out`, `net.0.proj`→`proj_in`, etc.; the
//! Mochi tier does not).
//!
//! ## group_size validation (LTX `validate_group_size` precedent)
//!
//! The MLX-packed → GGML repack ([`candle_gen::quant`]) the seam runs at group **64**; a tier that ever
//! shipped a different group would mis-align. [`MochiTierPaths::validate_group_size`] reads the packed
//! tier's `quantize_config.json` `quantization.group_size` (via the shared
//! [`candle_gen::quant::PackedConfig`]) and asserts it equals [`GROUP_SIZE`], failing loudly rather than
//! rendering garbage. The dense `bf16` tier carries no `quantize_config.json` (nothing to validate).

use std::path::{Path, PathBuf};

use candle_gen::quant::PackedConfig;
use candle_gen::{CandleError, Result};

/// The Mochi MLX tiers' quant group size (the hosted q4/q8 tiers pack at 64, MLX's default and the group
/// `mlx-gen-mochi`'s `convert.rs` writes). The packed-detect seam repacks at this default; a tier at a
/// different group is rejected by [`MochiTierPaths::validate_group_size`] rather than silently mis-read.
pub const GROUP_SIZE: usize = candle_gen::quant::MLX_GROUP_SIZE; // 64

/// A resolved pre-quantized Mochi tier: the tier dir (`.../q4`, `.../q8`, or `.../bf16`) holding the
/// packed/dense `transformer/model.safetensors` + `split_model.json`, and the **shared root** where the
/// tier-independent T5-XXL / AsymmVAE / tokenizer live (the tier dir's parent, per the `convert.rs`
/// layout).
#[derive(Debug, Clone)]
pub struct MochiTierPaths {
    /// The `q4` / `q8` / `bf16` tier dir (holds `transformer/` + `split_model.json`).
    pub tier_dir: PathBuf,
    /// Where the shared `text_encoder/` + `vae/` components live — the tier dir's parent (the
    /// `convert.rs` `stage_shared_components` target), falling back to the tier dir itself for a future
    /// self-contained tier.
    pub shared_root: PathBuf,
}

impl MochiTierPaths {
    /// Detect a Mochi MLX tier at `dir`: a directory that holds **both** `split_model.json` (the tier
    /// marker `convert.rs` writes) **and** a `transformer/` subdir. Returns `None` for the raw diffusers
    /// snapshot (no `split_model.json`), so [`crate::pipeline`] keeps the legacy single-root path.
    ///
    /// The shared T5/VAE root is resolved to the tier dir's **parent** (where `stage_shared_components`
    /// links them) when that parent actually holds a `text_encoder/`; otherwise it falls back to the
    /// tier dir (a self-contained tier). The dense `bf16` tier is detected the same way as q4/q8.
    pub fn detect(dir: &Path) -> Option<Self> {
        let manifest = dir.join("split_model.json");
        let transformer = dir.join("transformer");
        if !(manifest.is_file() && transformer.is_dir()) {
            return None;
        }
        let shared_root = dir
            .parent()
            .filter(|p| p.join("text_encoder").is_dir())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| dir.to_path_buf());
        Some(Self {
            tier_dir: dir.to_path_buf(),
            shared_root,
        })
    }

    /// Parse the tier's `split_model.json` manifest.
    fn split_model(&self) -> Result<serde_json::Value> {
        let p = self.tier_dir.join("split_model.json");
        let text = std::fs::read_to_string(&p)
            .map_err(|e| CandleError::Msg(format!("mochi tier: read {}: {e}", p.display())))?;
        serde_json::from_str(&text)
            .map_err(|e| CandleError::Msg(format!("mochi tier: parse {}: {e}", p.display())))
    }

    /// Whether this tier is quantized (`split_model.json` `quantized: true` — the q4/q8 tiers). The
    /// `bf16` tier is dense (`quantized: false`).
    pub fn is_quantized(&self) -> Result<bool> {
        Ok(self
            .split_model()?
            .get("quantized")
            .and_then(|v| v.as_bool())
            .unwrap_or(false))
    }

    /// The packed [`PackedConfig`] (`quantization.{bits, group_size}`) read from the tier's
    /// `quantize_config.json` sidecar — `None` for the dense `bf16` tier (which has no such sidecar).
    /// The nested-block schema matches the shared [`PackedConfig::from_config`] (mirrors the LTX tier).
    pub fn packed_config(&self) -> Result<Option<PackedConfig>> {
        let p = self.tier_dir.join("quantize_config.json");
        if !p.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&p)
            .map_err(|e| CandleError::Msg(format!("mochi tier: read {}: {e}", p.display())))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| CandleError::Msg(format!("mochi tier: parse {}: {e}", p.display())))?;
        Ok(Some(PackedConfig::from_config(&json).ok_or_else(|| {
            CandleError::Msg(format!(
                "mochi tier: {} has no `quantization.bits` — not a packed tier config",
                p.display()
            ))
        })?))
    }

    /// The tier's declared quantization bit-width (`Some(4)` q4 / `Some(8)` q8; `None` for the dense
    /// `bf16` tier) — used by [`crate::load`] to assert a caller-supplied `spec.quantize` matches the
    /// tier the dir already is.
    pub fn manifest_bits(&self) -> Result<Option<i32>> {
        Ok(self.packed_config()?.map(|c| c.bits))
    }

    /// Read + **validate** the tier's `group_size` against the [`GROUP_SIZE`] (64) the packed loaders
    /// repack at (the LTX `validate_group_size` precedent). Fails loudly if a quantized tier ever ships
    /// a different group (the MLX→GGML repack would mis-align). A no-op for the dense `bf16` tier.
    pub fn validate_group_size(&self) -> Result<()> {
        if let Some(cfg) = self.packed_config()? {
            let g = cfg.group_size as usize;
            if g != GROUP_SIZE {
                return Err(CandleError::Msg(format!(
                    "mochi tier: quantize_config.json group_size {g} != the loader's repack group {GROUP_SIZE} \
                     — the MLX→GGML repack would mis-align. A tier at a new group needs the group threaded \
                     into the packed loaders (candle_gen::quant::*_gs already accepts it)."
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A raw diffusers snapshot (no `split_model.json`) is NOT a tier — `detect` returns `None`, so the
    /// dense single-root load path is left unchanged.
    #[test]
    fn detect_returns_none_for_non_tier_dir() {
        let tmp = std::env::temp_dir().join(format!("sc11990_nontier_{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("transformer")).unwrap();
        // No split_model.json ⇒ not a tier.
        assert!(MochiTierPaths::detect(&tmp).is_none());
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// A tier dir (`split_model.json` + `transformer/`) is detected, and the shared root resolves to the
    /// parent when the parent holds a `text_encoder/` (the `convert.rs` staged layout).
    #[test]
    fn detect_resolves_shared_root_to_parent() {
        let root = std::env::temp_dir().join(format!("sc11990_tier_{}", std::process::id()));
        let tier = root.join("q4");
        std::fs::create_dir_all(tier.join("transformer")).unwrap();
        std::fs::create_dir_all(root.join("text_encoder")).unwrap();
        std::fs::write(
            tier.join("split_model.json"),
            r#"{"quantized": true, "quantization_bits": 4, "quantization_group_size": 64}"#,
        )
        .unwrap();
        std::fs::write(
            tier.join("quantize_config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();

        let paths = MochiTierPaths::detect(&tier).expect("tier detected");
        assert_eq!(paths.tier_dir, tier);
        assert_eq!(
            paths.shared_root, root,
            "shared root is the tier dir's parent"
        );
        assert!(paths.is_quantized().unwrap());
        assert_eq!(paths.manifest_bits().unwrap(), Some(4));
        paths.validate_group_size().unwrap();

        std::fs::remove_dir_all(&root).ok();
    }

    /// A dense `bf16` tier (`quantized: false`, no `quantize_config.json`) validates trivially and has no
    /// manifest bits.
    #[test]
    fn dense_bf16_tier_has_no_packed_config() {
        let root = std::env::temp_dir().join(format!("sc11990_bf16_{}", std::process::id()));
        let tier = root.join("bf16");
        std::fs::create_dir_all(tier.join("transformer")).unwrap();
        std::fs::create_dir_all(root.join("text_encoder")).unwrap();
        std::fs::write(tier.join("split_model.json"), r#"{"quantized": false}"#).unwrap();

        let paths = MochiTierPaths::detect(&tier).expect("tier detected");
        assert!(!paths.is_quantized().unwrap());
        assert_eq!(paths.manifest_bits().unwrap(), None);
        paths.validate_group_size().unwrap(); // no-op for dense
        std::fs::remove_dir_all(&root).ok();
    }

    /// A quantized tier whose `quantize_config.json` declares a non-64 group is rejected loudly.
    #[test]
    fn validate_group_size_rejects_non_64_group() {
        let root = std::env::temp_dir().join(format!("sc11990_g32_{}", std::process::id()));
        let tier = root.join("q4");
        std::fs::create_dir_all(tier.join("transformer")).unwrap();
        std::fs::create_dir_all(root.join("text_encoder")).unwrap();
        std::fs::write(tier.join("split_model.json"), r#"{"quantized": true}"#).unwrap();
        std::fs::write(
            tier.join("quantize_config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 32}}"#,
        )
        .unwrap();

        let paths = MochiTierPaths::detect(&tier).expect("tier detected");
        assert!(
            paths.validate_group_size().is_err(),
            "a non-64 group must be rejected"
        );
        std::fs::remove_dir_all(&root).ok();
    }
}
