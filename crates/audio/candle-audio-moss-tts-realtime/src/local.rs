//! The MOSS-TTS-Realtime **local/depth transformer** (sc-13334) — `config.json.local_config`.
//!
//! This is the CSM-style depth decoder: run **once per audio frame**, it autoregressively emits
//! the frame's `rvq` (16) RVQ codebook tokens along a depth axis of length `rvq`. Depth position 0
//! is seeded by the backbone's last hidden state; each subsequent depth position embeds the token
//! sampled at the previous position (through its own `embed_tokens.N`), and every depth position
//! projects through its own per-codebook LM head (`local_lm_heads.N`). Weight-for-weight this
//! mirrors the reference `MossTTSRealtimeLocalTransformerForCausalLM.generate_local_transformer`
//! loop, **including its sampling** ([`crate::sampling`]: temperature / top-k / top-p + a
//! per-codebook cross-frame repetition penalty). The reference is `do_sample=True`; greedy (argmax)
//! decoding collapses this model into a repeating loop whose codec decode is silent. A **seeded**
//! PRNG keeps the sampled decode reproducible (same seed ⇒ same frame — the gen-core determinism law).

use candle_audio::candle_core::{Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::blocks::{causal_mask, rope_tables, BlockConfig, Layer};
use crate::config::LocalConfig;
use crate::sampling::{sample, Rng, SamplingParams};

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

    /// Decode one frame's `rvq` RVQ codebook tokens from the backbone's last hidden state `[1, 1, H]`,
    /// sampling each codebook (temperature / top-k / top-p + per-codebook cross-frame repetition
    /// penalty — the reference `generate_local_transformer`) with the seeded `rng`. `history` is the
    /// frames emitted so far (previous frames); codebook `i`'s repetition penalty uses `history`'s
    /// codebook-`i` column. Deterministic per seed: same backbone state + same `rng` sequence ⇒ same
    /// frame, so `generate` and `generate_streaming` agree.
    pub fn decode_frame(
        &self,
        backbone_last_hidden: &Tensor,
        history: &[Vec<u32>],
        params: &SamplingParams,
        rng: &mut Rng,
    ) -> CandleResult<Vec<u32>> {
        // Depth position 0 embedding = the backbone hidden state.
        let mut depth_embeds = backbone_last_hidden.clone(); // [1, 1, H]
        let mut tokens = Vec::with_capacity(self.rvq);
        for i in 0..self.rvq {
            let hidden = self.hidden_last(&depth_embeds)?; // [1, H]
            let logits = self.heads[i].forward(&hidden)?; // [1, audio_vocab]
            let mut row: Vec<f32> = logits.reshape((logits.elem_count(),))?.to_vec1::<f32>()?;
            // Codebook i's tokens from previous frames (the repetition-penalty history).
            let hist_i: Vec<u32> = history.iter().filter_map(|f| f.get(i).copied()).collect();
            let token = sample(&mut row, &hist_i, params, rng);
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
