//! Anima's source text encoder — **Qwen3-0.6B base** (`Qwen3Model`), reusing z-image's Qwen3 decoder
//! block (`EncoderLayer`) + the shared HF RoPE. Unlike z-image (which returns the second-to-last
//! layer un-normed for its DiT `cap_feats`), Anima consumes the model's **`last_hidden_state`** — the
//! last layer output AFTER the final `norm` — matching `Qwen3Model(...).last_hidden_state`. The hidden
//! states are then mask-multiplied and fed to the `AnimaTextConditioner` as `source_hidden_states`.

use mlx_rs::fast::rms_norm;
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{build_mask, TextRope, TokenEmbedding};
use mlx_gen::weights::{join, Weights};
use mlx_gen::Result;
use mlx_gen_z_image::text_encoder::EncoderLayer;

use crate::config::Qwen3Config;

/// The Qwen3-0.6B text tower (token embed → 28 pre-norm decoder layers → final RMSNorm).
pub struct AnimaQwen3 {
    embed: TokenEmbedding,
    layers: Vec<EncoderLayer>,
    norm: Array,
    rope: TextRope,
    eps: f32,
}

impl AnimaQwen3 {
    /// `prefix` is the checkpoint root of the Qwen3 model (`"model"` for `model.embed_tokens.*`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Qwen3Config) -> Result<Self> {
        let embed = mlx_gen::quant::embedding(
            w,
            &join(prefix, "embed_tokens"),
            mlx_gen::quant::DEFAULT_GROUP_SIZE,
        )?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(EncoderLayer::from_weights(
                w,
                &join(prefix, &format!("layers.{i}")),
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        Ok(Self {
            embed,
            layers,
            norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            eps: cfg.rms_norm_eps,
        })
    }

    /// `input_ids` / `attention_mask`: `[B, S]` int32. Returns the **last_hidden_state** `[B, S, hidden]`
    /// (bf16) — the last decoder layer AFTER the final norm (a causal LM with padding masking).
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        // bf16 tower (matches the reference `text_encoder.dtype`): embed gathers f32 rows, cast to bf16.
        let mut h = self.embed.forward(input_ids)?.as_dtype(Dtype::Bfloat16)?;
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?.as_dtype(Dtype::Bfloat16)?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, &mask)?;
        }
        // last_hidden_state = final RMSNorm applied to the last layer (unlike z-image's [-2] un-normed).
        Ok(rms_norm(&h, &self.norm, self.eps)?)
    }
}
