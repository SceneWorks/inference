//! The MOSS-TTS-Realtime Qwen3-1.7B **backbone** (sc-13334) — `config.json.language_config`.
//!
//! At every sequence position the input is a **multi-channel** token: one text-channel id plus
//! `rvq` (16) audio-codebook ids. The backbone embeds each channel through its own embedding
//! (`embed_tokens.0` for text; `embed_tokens.1..=rvq` for the codebooks), **sums** them, and runs
//! the standard Qwen3 decoder stack, returning the post-final-norm hidden states. Those hidden
//! states seed the local/depth transformer that decodes the next frame's RVQ tokens (see
//! [`crate::decode`]). This mirrors the reference `MossTTSRealtime.get_input_embeddings` +
//! `Qwen3Model` forward exactly, weight-for-weight (`embed_tokens.N` + `language_model.*`).

use candle_audio::candle_core::{Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module, RmsNorm, VarBuilder};

use crate::blocks::{causal_mask, rope_tables, BlockConfig, Layer};
use crate::config::LanguageConfig;

/// One backbone input position: the text-channel id and the `rvq` audio-codebook ids.
#[derive(Debug, Clone)]
pub struct Frame {
    pub text: u32,
    /// The `rvq` audio-codebook ids for this position (length must equal `rvq`).
    pub audio: Vec<u32>,
}

/// The Qwen3 backbone: the multi-channel input embeddings + the decoder stack + final norm.
pub struct Backbone {
    embed_text: Embedding,
    embed_audio: Vec<Embedding>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    head_dim: usize,
    rope_theta: f64,
    hidden_size: usize,
    device: Device,
}

impl Backbone {
    /// Build the backbone from the checkpoint (`embed_tokens.*` + `language_model.*`). `vb` is the
    /// snapshot root VarBuilder (the full-model tensor namespace).
    pub fn new(
        cfg: &LanguageConfig,
        rvq: usize,
        audio_vocab_size: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let embed_text =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens.0"))?;
        // audio embeddings are embed_tokens.1 ..= embed_tokens.rvq, each over the audio-codebook
        // vocabulary (`config.json.audio_vocab_size`), sharing the backbone `hidden_size`.
        let mut embed_audio = Vec::with_capacity(rvq);
        for c in 0..rvq {
            embed_audio.push(candle_nn::embedding(
                audio_vocab_size,
                cfg.hidden_size,
                vb.pp(format!("embed_tokens.{}", c + 1)),
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
        let vb_l = vb.pp("language_model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::new(&block, vb_l.pp(i))?);
        }
        let norm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("language_model.norm"),
        )?;
        Ok(Self {
            embed_text,
            embed_audio,
            layers,
            norm,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            hidden_size: cfg.hidden_size,
            device: vb.device().clone(),
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// The summed multi-channel input embeddings for a frame sequence → `[1, T, H]`.
    fn embed(&self, frames: &[Frame]) -> CandleResult<Tensor> {
        let t = frames.len();
        let text_ids: Vec<u32> = frames.iter().map(|f| f.text).collect();
        let text_t = Tensor::from_vec(text_ids, (1, t), &self.device)?;
        let mut sum = self.embed_text.forward(&text_t)?;
        for (c, embed) in self.embed_audio.iter().enumerate() {
            let ids: Vec<u32> = frames.iter().map(|f| f.audio[c]).collect();
            let ids_t = Tensor::from_vec(ids, (1, t), &self.device)?;
            sum = (sum + embed.forward(&ids_t)?)?;
        }
        Ok(sum)
    }

    /// Full-sequence forward over `frames` → the post-norm hidden states `[1, T, H]`.
    pub fn forward(&self, frames: &[Frame]) -> CandleResult<Tensor> {
        let t = frames.len();
        let mut h = self.embed(frames)?;
        let (cos, sin) = rope_tables(&self.device, t, self.head_dim, self.rope_theta)?;
        let mask = if t > 1 {
            Some(causal_mask(&self.device, t, h.dtype())?)
        } else {
            None
        };
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, mask.as_ref())?;
        }
        self.norm.forward(&h)
    }

    /// Full-sequence forward, returning only the **last** position's hidden state `[1, 1, H]` —
    /// the seed the local/depth transformer decodes the next frame from.
    pub fn forward_last(&self, frames: &[Frame]) -> CandleResult<Tensor> {
        let h = self.forward(frames)?;
        let t = h.dim(1)?;
        h.i((.., t - 1..t, ..))?.contiguous()
    }
}
