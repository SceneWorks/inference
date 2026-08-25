//! The 2-layer text **token refiner** (`MiniMaxH3TokenRefiner`).
//!
//! `context_embedder` maps the text encoder's 5120-wide context to `hidden_size`, and this stack
//! then refines it before its rows are scattered into the packed sequence. It is small — 21 of the
//! 638 published tensors — and easy to mistake for an optional pre-pass, so:
//!
//! * it is **not** stubbed, and `tests/dit_parity.rs` covers both blocks and the final norm
//!   against the reference;
//! * the lightx2v turbo step-distill LoRAs (sc-18724) target
//!   `token_refiner.refiner_blocks.{0,1}` on the same six leaves as `transformer_blocks`, so 24 of
//!   their 624 tensors land here. A refiner with different naming or dims puts those in
//!   `unmatched_paths` and the distill silently applies at partial coverage.
//!
//! # How it differs from a `transformer_blocks` entry
//!
//! Same leaf names, same widths (`attn` 5376 → 7168, `ff` 5376 → 2·14336 → 5376), and **two
//! removals**:
//!
//! | | `transformer_blocks.N` | `token_refiner.refiner_blocks.N` |
//! |---|---|---|
//! | AdaLN | `adaln_proj.linear` modulates both residuals | **none** — plain pre-norm residuals |
//! | rotary | MM-RoPE on q and k | **none** — `self.attn(self.norm1(x))` passes no `rotary_emb` |
//!
//! Applying the rotary here anyway is the plausible error, and it is invisible structurally: the
//! shapes are unchanged and the refiner would still produce finite, well-scaled output.

use mlx_rs::ops::add;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::dit::config::MiniMaxH3DitConfig;
use crate::dit::layers::{DitAttention, DitFeedForward, RmsNorm};

/// One refiner block: `x + attn(norm1(x))`, then `x + ff(norm2(x))`.
#[derive(Debug, Clone)]
pub struct TokenRefinerBlock {
    norm1: RmsNorm,
    attn: DitAttention,
    norm2: RmsNorm,
    ff: DitFeedForward,
}

impl TokenRefinerBlock {
    /// Load block `prefix` (e.g. `token_refiner.refiner_blocks.0`).
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            norm1: RmsNorm::from_weights(w, &format!("{prefix}.norm1"), cfg.norm_eps, dtype)?,
            attn: DitAttention::from_weights(w, &format!("{prefix}.attn"), cfg, dtype)?,
            norm2: RmsNorm::from_weights(w, &format!("{prefix}.norm2"), cfg.norm_eps, dtype)?,
            ff: DitFeedForward::from_weights(w, &format!("{prefix}.ff"), dtype)?,
        })
    }

    /// The 10 tensors a refiner block consumes — a `transformer_blocks` entry's 12 minus
    /// `adaln_proj`'s weight/bias pair.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v = RmsNorm::names(&format!("{prefix}.norm1")).to_vec();
        v.extend(RmsNorm::names(&format!("{prefix}.norm2")));
        v.extend(DitAttention::names(&format!("{prefix}.attn")));
        v.extend(DitFeedForward::names(&format!("{prefix}.ff")));
        v
    }

    /// Unmodulated, unrotated pre-norm residuals.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let attn = self.attn.forward(&self.norm1.forward(x)?, None)?;
        let x = add(x, &attn)?;
        let ff = self.ff.forward(&self.norm2.forward(&x)?)?;
        Ok(add(&x, &ff)?)
    }
}

/// The refiner stack plus its `final_norm`.
#[derive(Debug, Clone)]
pub struct TokenRefiner {
    blocks: Vec<TokenRefinerBlock>,
    final_norm: RmsNorm,
}

impl TokenRefiner {
    /// Load `token_refiner`. `cfg.num_refiner_layers` blocks — 2 in the published checkpoint.
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        cfg.validate()?;
        if cfg.num_refiner_layers <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit refiner: num_refiner_layers must be positive, got {}",
                cfg.num_refiner_layers
            )));
        }
        let blocks = (0..cfg.num_refiner_layers)
            .map(|i| {
                TokenRefinerBlock::from_weights(
                    w,
                    &format!("{prefix}.refiner_blocks.{i}"),
                    cfg,
                    dtype,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            blocks,
            // `final_norm` uses `final_norm_eps`, not `norm_eps` — the same value in the shipped
            // config, but they are separate knobs.
            final_norm: RmsNorm::from_weights(
                w,
                &format!("{prefix}.final_norm"),
                cfg.final_norm_eps,
                dtype,
            )?,
        })
    }

    /// Every tensor the refiner consumes — 21 in the published checkpoint.
    pub fn names(prefix: &str, cfg: &MiniMaxH3DitConfig) -> Vec<String> {
        let mut v: Vec<String> = (0..cfg.num_refiner_layers)
            .flat_map(|i| TokenRefinerBlock::names(&format!("{prefix}.refiner_blocks.{i}")))
            .collect();
        v.extend(RmsNorm::names(&format!("{prefix}.final_norm")));
        v
    }

    /// Blocks in order, then `final_norm`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        self.final_norm.forward(&h)
    }

    /// Blocks loaded — `num_refiner_layers`.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }
}

/// `attn.*` / `ff.*` — the same six adaptable leaves a [`crate::dit::block::DitBlock`] exposes.
/// `norm1` / `norm2` are RMSNorm gains and are deliberately unreachable (sc-18724).
impl AdaptableHost for TokenRefinerBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            ["ff", rest @ ..] => self.ff.adaptable_mut(rest),
            _ => None,
        }
    }
}

/// `refiner_blocks.{i}.…`.
///
/// **The refiner is load-bearing for the turbo LoRA and must never be stubbed** (sc-18724): 24 of the
/// 624 lightx2v turbo tensors target `token_refiner.refiner_blocks.{0,1}`, so a stubbed refiner puts
/// them in `unmatched_paths` and fails the strict install. sc-17144 already required a real refiner
/// for parity; this is a second, independent reason.
impl AdaptableHost for TokenRefiner {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["refiner_blocks", i, rest @ ..] => {
                let idx: usize = i.parse().ok()?;
                self.blocks.get_mut(idx)?.adaptable_mut(rest)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2 blocks × 10 tensors + `final_norm` = the published 21, and no `adaln_proj` among them.
    #[test]
    fn refiner_names_cover_the_published_twenty_one_tensors() {
        let cfg = MiniMaxH3DitConfig::default();
        let names = TokenRefiner::names("token_refiner", &cfg);
        assert_eq!(cfg.num_refiner_layers, 2);
        assert_eq!(TokenRefinerBlock::names("x").len(), 10);
        assert_eq!(
            super::super::block::DitBlock::names("y").len(),
            TokenRefinerBlock::names("x").len() + 2,
            "a refiner block is a transformer block minus adaln_proj's weight/bias pair"
        );
        assert_eq!(names.len(), 21, "got {names:?}");
        assert!(names.contains(&"token_refiner.final_norm.weight".to_string()));
        assert!(names.contains(&"token_refiner.refiner_blocks.1.attn.norm_k.weight".to_string()));
        assert!(
            !names.iter().any(|n| n.contains("adaln")),
            "the refiner has no AdaLN"
        );
    }
}
