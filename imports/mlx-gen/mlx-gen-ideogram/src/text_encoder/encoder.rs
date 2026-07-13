//! Full Qwen3-VL text encoder: token embedding → up to 36 pre-norm decoder layers, collecting the
//! OUTPUTS of the layers at [`crate::config::EXTRACTED_LAYERS`] (`0,3,…,33,35` — layer indices,
//! the upstream `_get_qwen3_vl_embeddings` `captured[layer_idx]`), interleaved into the
//! `13·4096 = 53248`-wide features the DiT's `llm_cond_proj` consumes.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::nn::{build_mask, TextRope, TokenEmbedding};
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
            embed_tokens: crate::quant::embedding(w, &join(prefix, "embed_tokens"))?,
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            out_layers: EXTRACTED_LAYERS.to_vec(),
        })
    }

    /// Quantize the token-embedding table + every decoder-layer projection in place (group-wise
    /// affine Q4/Q8). `cast_to_bf16=true` for the embedding matches the FLUX.2 Qwen3 TE path; the
    /// per-layer norms stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.embed_tokens.quantize(bits, true)?;
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns the concatenated hidden states
    /// `[b, s, 13·hidden]` (f32) — Ideogram's `llm` features. The final norm is never applied; only
    /// layers up to `max(out_layers)` are run (later layers cannot influence the result).
    pub fn prompt_embeds(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        // `out_layers` are the indices of the LAYERS whose OUTPUTS Ideogram concatenates (upstream
        // `_get_qwen3_vl_embeddings`: `captured[layer_idx] = decoder_layer(hidden)`), i.e. the
        // hidden state right *after* running layer `i` — NOT HF `output_hidden_states` indexing
        // (which would offset by one and put raw embeddings at index 0). Run up to the last needed
        // layer (`max + 1` layers) since later layers can't influence the captured set.
        let max_layer = *self.out_layers.iter().max().unwrap_or(&0);

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let mut saved: Vec<(usize, Array)> = Vec::with_capacity(self.out_layers.len());
        for (i, layer) in self.layers.iter().take(max_layer + 1).enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            if self.out_layers.contains(&i) {
                saved.push((i, hidden.clone()));
            }
        }

        let pick = |idx: usize| -> Result<&Array> {
            saved
                .iter()
                .find(|(k, _)| *k == idx)
                .map(|(_, v)| v)
                .ok_or_else(|| Error::Msg(format!("ideogram te: hidden state {idx} not captured")))
        };
        // INTERLEAVE the layers into the feature axis: stack to `[B, L, H, n]` then reshape to
        // `[B, L, H·n]`, so feature `f = h·n + layer` — the pipeline's
        // `stack(dim=0).permute(1,2,3,0).reshape`, NOT a plain per-layer concat (which would be
        // block order `layer·H + h`). The DiT's `llm_cond_proj` was trained on the interleaved
        // layout; getting it wrong yields a coherent but prompt-agnostic image.
        let expanded: Vec<Array> = self
            .out_layers
            .iter()
            .map(|&idx| Ok(pick(idx)?.expand_dims(3)?))
            .collect::<Result<_>>()?;
        let refs: Vec<&Array> = expanded.iter().collect();
        let stacked = concatenate_axis(&refs, 3)?; // [B, L, H, n]
        let sh = stacked.shape();
        Ok(stacked.reshape(&[sh[0], sh[1], sh[2] * sh[3]])?)
    }
}
