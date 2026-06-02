//! Full Qwen3 text encoder: token embedding → 36 pre-norm decoder layers, collecting the
//! intermediate hidden states. `prompt_embeds` concatenates the outputs of layers 9/18/27 into
//! the transformer's conditioning. Port of the fork's `Qwen3TextEncoder.get_prompt_embeds`.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use mlx_gen::array::host_i32;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, Qwen3DecoderLayer, TextRope};

/// Qwen3 LM dimensions (FLUX.2-klein `text_encoder`).
pub struct Qwen3TextEncoderConfig {
    pub hidden_size: i32,
    pub n_layers: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Hidden-state indices (into a list whose entry 0 is the token embedding) concatenated into
    /// `prompt_embeds`. klein: (9, 18, 27) → 3·hidden = 12288.
    pub out_layers: [usize; 3],
}

impl Qwen3TextEncoderConfig {
    pub fn klein_9b() -> Self {
        Self {
            hidden_size: 4096,
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            out_layers: [9, 18, 27],
        }
    }
}

pub struct Qwen3TextEncoder {
    embed_tokens: Array,
    layers: Vec<Qwen3DecoderLayer>,
    rope: TextRope,
    out_layers: [usize; 3],
}

impl Qwen3TextEncoder {
    /// Loads from the on-disk `model.*` tree under `prefix` (`"model"`):
    /// `{prefix}.embed_tokens.weight`, `{prefix}.layers.{i}.…`. The final `{prefix}.norm.weight`
    /// is intentionally **not** loaded — the fork computes the final norm but `get_prompt_embeds`
    /// discards it, using only the raw (pre-final-norm) intermediate layer outputs.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Qwen3TextEncoderConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(Qwen3DecoderLayer::from_weights(
                w,
                &join(prefix, &format!("layers.{i}")),
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        Ok(Self {
            embed_tokens: w.require(&join(prefix, "embed_tokens.weight"))?.clone(),
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            out_layers: cfg.out_layers,
        })
    }

    /// Token embedding (f32): `input_ids` `[b, s]` int32 → `[b, s, hidden]`.
    fn embed(&self, input_ids: &Array) -> Result<Array> {
        Ok(self
            .embed_tokens
            .take_axis(input_ids, 0)?
            .as_dtype(Dtype::Float32)?)
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns `prompt_embeds`
    /// `[b, s, 3·hidden]` (f32): the layer-9/18/27 hidden states concatenated along the feature
    /// axis. Equivalent to the fork's `stack(axis=1) → transpose(0,2,1,3) → reshape` (which, per
    /// position, lays the three layers out contiguously = a feature-axis concat).
    pub fn prompt_embeds(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        // hidden_states[0] = embeddings; hidden_states[i+1] = output of layer i.
        let mut hidden = self.embed(input_ids)?;
        let mut hidden_states: Vec<Array> = Vec::with_capacity(self.layers.len() + 1);
        hidden_states.push(hidden.clone());
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            hidden_states.push(hidden.clone());
        }

        let [a, b_, c] = self.out_layers;
        Ok(concatenate_axis(
            &[&hidden_states[a], &hidden_states[b_], &hidden_states[c]],
            2,
        )?)
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
