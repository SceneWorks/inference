//! The MOSS-TTS-Realtime **local/depth transformer** (sc-13334) — `config.json.local_config`.
//!
//! This is the CSM-style depth decoder: run **once per audio frame**, it autoregressively emits
//! the frame's `rvq` (16) RVQ codebook tokens along a depth axis of length `rvq`. Depth position 0
//! is seeded by the backbone's last hidden state; each subsequent depth position embeds the token
//! sampled at the previous position (through its own `embed_tokens.N`), and every depth position
//! projects through its own per-codebook LM head (`local_lm_heads.N`). Weight-for-weight this
//! mirrors the reference `MossTTSRealtimeLocalTransformerForCausalLM.generate_local_transformer`
//! loop; sampling here is deterministic **greedy** (argmax) so a given backbone state maps to one
//! reproducible RVQ frame (the gen-core determinism law).

use candle_audio::candle_core::{Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::blocks::{causal_mask, rope_tables, BlockConfig, Layer};
use crate::config::LocalConfig;

/// The local/depth transformer: the `rvq - 1` depth embeddings, the decoder stack, the final norm,
/// and the `rvq` per-codebook LM heads.
pub struct LocalTransformer {
    embed_tokens: Vec<Embedding>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    heads: Vec<Linear>,
    head_dim: usize,
    rope_theta: f64,
    rvq: usize,
    device: Device,
}

impl LocalTransformer {
    /// Build from the checkpoint (`local_transformer.model.*` + `local_transformer.local_lm_heads.*`).
    pub fn new(cfg: &LocalConfig, vb: VarBuilder) -> CandleResult<Self> {
        let vb_m = vb.pp("local_transformer.model");
        // rvq - 1 depth embeddings (embed_tokens.0 ..= embed_tokens.rvq-2).
        let mut embed_tokens = Vec::with_capacity(cfg.rvq - 1);
        let vb_e = vb_m.pp("embed_tokens");
        for i in 0..cfg.rvq - 1 {
            embed_tokens.push(candle_nn::embedding(
                cfg.audio_vocab_size,
                cfg.hidden_size,
                vb_e.pp(i),
            )?);
        }
        let block = BlockConfig {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
            attention_bias: cfg.attention_bias,
        };
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::new(&block, vb_l.pp(i))?);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let vb_h = vb.pp("local_transformer.local_lm_heads");
        let mut heads = Vec::with_capacity(cfg.rvq);
        for i in 0..cfg.rvq {
            heads.push(candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.audio_vocab_size,
                vb_h.pp(i),
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            rvq: cfg.rvq,
            device: vb.device().clone(),
        })
    }

    /// Run the depth decoder over the current `[1, depth, H]` sequence and return the last
    /// position's post-norm hidden state `[1, H]`.
    fn hidden_last(&self, embeds: &Tensor) -> CandleResult<Tensor> {
        let depth = embeds.dim(1)?;
        let (cos, sin) = rope_tables(&self.device, depth, self.head_dim, self.rope_theta)?;
        let mask = if depth > 1 {
            Some(causal_mask(&self.device, depth, embeds.dtype())?)
        } else {
            None
        };
        let mut h = embeds.clone();
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, mask.as_ref())?;
        }
        let h = self.norm.forward(&h)?;
        h.i((.., depth - 1..depth, ..))?.reshape((1, h.dim(2)?))
    }

    /// Decode one frame's `rvq` RVQ codebook tokens (greedy) from the backbone's last hidden state
    /// `[1, 1, H]`. Deterministic: same seed hidden ⇒ same frame.
    pub fn decode_frame(&self, backbone_last_hidden: &Tensor) -> CandleResult<Vec<u32>> {
        // Depth position 0 embedding = the backbone hidden state.
        let mut depth_embeds = backbone_last_hidden.clone(); // [1, 1, H]
        let mut tokens = Vec::with_capacity(self.rvq);
        for i in 0..self.rvq {
            let hidden = self.hidden_last(&depth_embeds)?; // [1, H]
            let logits = self.heads[i].forward(&hidden)?; // [1, audio_vocab]
            let token = argmax_last(&logits)?;
            tokens.push(token);
            if i + 1 < self.rvq {
                // Next depth position embedding = embed_tokens[i](token).
                let tok_t = Tensor::from_vec(vec![token], (1, 1), &self.device)?;
                let emb = self.embed_tokens[i].forward(&tok_t)?; // [1, 1, H]
                depth_embeds = Tensor::cat(&[&depth_embeds, &emb], 1)?;
            }
        }
        Ok(tokens)
    }
}

/// Argmax over the last dim of a `[1, V]` logits tensor → the winning token id.
fn argmax_last(logits: &Tensor) -> CandleResult<u32> {
    let row: Vec<f32> = logits.reshape((logits.elem_count(),))?.to_vec1::<f32>()?;
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    Ok(best as u32)
}
