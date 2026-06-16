//! Full Qwen3-VL text encoder: token embedding → up to 36 pre-norm decoder layers, collecting the
//! intermediate hidden states. Ideogram concatenates the hidden states at
//! [`crate::config::EXTRACTED_LAYERS`] (`0,3,…,33,35`, where index 0 = the token embedding and
//! index `k` = the output of the `k`-th layer, the HF `output_hidden_states` convention) into the
//! `13·4096 = 53248`-wide features the DiT's `llm_cond_proj` consumes.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::array::host_i32;
use mlx_gen::nn::{TextRope, TokenEmbedding};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use super::{join, Qwen3DecoderLayer};
use crate::config::{Ideogram4TextEncoderConfig, EXTRACTED_LAYERS};

pub struct Ideogram4TextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    rope: TextRope,
    /// Hidden-state indices concatenated into the output (index 0 = embedding, k = layer-k output).
    out_layers: Vec<usize>,
}

impl Ideogram4TextEncoder {
    /// Load from the converted `text_encoder` weights under `prefix` (`"language_model"`):
    /// `{prefix}.embed_tokens.weight`, `{prefix}.layers.{i}.…`. The final `{prefix}.norm.weight`
    /// is intentionally not loaded — Ideogram uses the raw (pre-final-norm) intermediate states.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &Ideogram4TextEncoderConfig,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers as usize);
        for i in 0..cfg.num_layers {
            layers.push(Qwen3DecoderLayer::from_weights(
                w,
                &join(prefix, &format!("layers.{i}")),
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        Ok(Self {
            embed_tokens: TokenEmbedding::Dense(
                w.require(&join(prefix, "embed_tokens.weight"))?.clone(),
            ),
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            out_layers: EXTRACTED_LAYERS.to_vec(),
        })
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns the concatenated hidden states
    /// `[b, s, 13·hidden]` (f32) — Ideogram's `llm` features. The final norm is never applied; only
    /// layers up to `max(out_layers)` are run (later layers cannot influence the result).
    pub fn prompt_embeds(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        let max_idx = *self.out_layers.iter().max().unwrap_or(&0);

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let mut saved: Vec<(usize, Array)> = Vec::with_capacity(self.out_layers.len());
        if self.out_layers.contains(&0) {
            saved.push((0, hidden.clone()));
        }
        for (i, layer) in self.layers.iter().take(max_idx).enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            let idx = i + 1;
            if self.out_layers.contains(&idx) {
                saved.push((idx, hidden.clone()));
            }
        }

        let pick = |idx: usize| -> Result<&Array> {
            saved
                .iter()
                .find(|(k, _)| *k == idx)
                .map(|(_, v)| v)
                .ok_or_else(|| Error::Msg(format!("ideogram te: hidden state {idx} not captured")))
        };
        // Concatenate in the order the model expects (= `out_layers` order).
        let picked: Vec<&Array> = self
            .out_layers
            .iter()
            .map(|&idx| pick(idx))
            .collect::<Result<_>>()?;
        Ok(concatenate_axis(&picked, 2)?)
    }
}

/// Additive attention mask `[b, 1, s, s]`: `0` where a query may attend (key is causal **and**
/// not padding), `-inf` otherwise. Built host-side (one-time `O(b·s²)` fill per encode).
fn build_mask(attention_mask: &Array, b: i32, s: i32) -> Result<Array> {
    let am = host_i32(attention_mask)?;
    let (b, s) = (b as usize, s as usize);
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in 0..s {
                let allowed = j <= i && am[bi * s + j] == 1;
                if !allowed {
                    data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Array::from_slice(&data, &[b as i32, 1, s as i32, s as i32]))
}
