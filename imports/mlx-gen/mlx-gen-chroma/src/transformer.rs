//! The Chroma DiT (`ChromaTransformer2DModel`).
//!
//! **Skeleton (sc-3835):** loads + holds the diffusers-named weight map and validates the expected
//! key surface. The distilled-guidance Approximator (sc-3836), the double/single blocks + pruned
//! adaLN modulation + MMDiT masking + RoPE (sc-3837), and the forward pass land in their own slices.

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::ChromaTransformerConfig;

/// The Chroma transformer weights + static config. Forward pass arrives in sc-3837.
pub struct ChromaTransformer {
    pub cfg: ChromaTransformerConfig,
    /// Diffusers-named weights (`x_embedder.*`, `distilled_guidance_layer.*`, `transformer_blocks.N.*`,
    /// `single_transformer_blocks.N.*`, `norm_out`/`proj_out`). Materialized into typed modules in
    /// sc-3836/sc-3837.
    pub weights: Weights,
}

impl ChromaTransformer {
    /// Load from a diffusers `transformer/` weight map. Validates that the Chroma-specific key surface
    /// is present (a structural sanity check before the typed modules are wired) and that the blocks
    /// carry **no** per-block modulation linears (the pruned-adaLN invariant).
    pub fn from_weights(w: Weights, cfg: ChromaTransformerConfig) -> Result<Self> {
        // Representative keys that must exist across the input projections, the Approximator, the two
        // block stacks, and the pruned output norm.
        let required = [
            "x_embedder.weight",
            "context_embedder.weight",
            "distilled_guidance_layer.in_proj.weight",
            "distilled_guidance_layer.out_proj.weight",
            "distilled_guidance_layer.layers.0.linear_1.weight",
            "distilled_guidance_layer.norms.0.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.add_q_proj.weight",
            "single_transformer_blocks.0.proj_mlp.weight",
            "single_transformer_blocks.0.proj_out.weight",
            "proj_out.weight",
        ];
        for k in required {
            if w.get(k).is_none() {
                return Err(Error::Msg(format!(
                    "chroma transformer: missing expected key {k:?} (not a ChromaTransformer2DModel \
                     diffusers layout?)"
                )));
            }
        }

        // Pruned-adaLN invariant: Chroma blocks have NO `.norm*.linear` weights — modulation is sliced
        // out of the Approximator output, not projected per block. A present linear would mean we
        // loaded a plain-FLUX transformer by mistake.
        if let Some(k) = w
            .keys()
            .find(|k| k.contains(".norm1.linear") || k.contains(".norm.linear"))
        {
            return Err(Error::Msg(format!(
                "chroma transformer: unexpected per-block modulation linear {k:?} — Chroma uses \
                 pruned adaLN (modulation comes from distilled_guidance_layer)"
            )));
        }

        // Block-count sanity against the config.
        let n_double = (0..)
            .take_while(|i| {
                w.get(&format!("transformer_blocks.{i}.attn.to_q.weight"))
                    .is_some()
            })
            .count();
        let n_single = (0..)
            .take_while(|i| {
                w.get(&format!("single_transformer_blocks.{i}.proj_out.weight"))
                    .is_some()
            })
            .count();
        if n_double != cfg.num_layers || n_single != cfg.num_single_layers {
            return Err(Error::Msg(format!(
                "chroma transformer: block counts {n_double} double / {n_single} single != config \
                 {} / {}",
                cfg.num_layers, cfg.num_single_layers
            )));
        }

        Ok(Self { cfg, weights: w })
    }
}
