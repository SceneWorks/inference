//! CLAP text tower (sc-12851): a faithful RoBERTa encoder + pooler port on `candle-nn`, matching
//! `transformers` `ClapTextModel` (which is `RobertaModel` with a `[CLS]`-token tanh pooler).
//!
//! Batch-of-one, unpadded: the provider embeds one query string at a time, so every token is
//! attended (no padding mask needed) and the RoBERTa position offset is computed for a full
//! sequence. `get_text_features` = pooler_output → `text_projection` → L2-normalize (the projection
//! + norm live in [`crate::model`], shared with the audio tower).

use crate::config;
use candle_audio::candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};

struct Embeddings {
    word: Embedding,
    position: Embedding,
    token_type: Embedding,
    norm: LayerNorm,
}

impl Embeddings {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            word: embedding(
                config::TEXT_VOCAB,
                config::TEXT_HIDDEN,
                vb.pp("word_embeddings"),
            )?,
            position: embedding(
                config::TEXT_MAX_POS,
                config::TEXT_HIDDEN,
                vb.pp("position_embeddings"),
            )?,
            token_type: embedding(
                config::TEXT_TYPE_VOCAB,
                config::TEXT_HIDDEN,
                vb.pp("token_type_embeddings"),
            )?,
            norm: layer_norm(config::TEXT_HIDDEN, config::TEXT_LN_EPS, vb.pp("LayerNorm"))?,
        })
    }

    fn forward(&self, input_ids: &Tensor, device: &Device) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        // RoBERTa position ids: no-padding sequence ⇒ positions [pad+1 .. pad+seq_len].
        let positions: Vec<u32> = (0..seq_len as u32)
            .map(|i| i + config::TEXT_PAD_TOKEN_ID + 1)
            .collect();
        let position_ids = Tensor::from_vec(positions, (1, seq_len), device)?;
        let token_type_ids = Tensor::zeros((1, seq_len), DType::U32, device)?;

        let words = self.word.forward(input_ids)?;
        let pos = self.position.forward(&position_ids)?;
        let types = self.token_type.forward(&token_type_ids)?;
        let sum = (words.broadcast_add(&pos)?).broadcast_add(&types)?;
        self.norm.forward(&sum)
    }
}

struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn load(vb: VarBuilder) -> Result<Self> {
        let h = config::TEXT_HIDDEN;
        Ok(Self {
            query: linear(h, h, vb.pp("query"))?,
            key: linear(h, h, vb.pp("key"))?,
            value: linear(h, h, vb.pp("value"))?,
            num_heads: config::TEXT_HEADS,
            head_dim: h / config::TEXT_HEADS,
        })
    }

    fn shape_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        x.reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (b, seq, h) = hidden.dims3()?;
        let q = self.shape_heads(&self.query.forward(hidden)?)?;
        let k = self.shape_heads(&self.key.forward(hidden)?)?;
        let v = self.shape_heads(&self.value.forward(hidden)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (b, heads, seq, head_dim)
        ctx.transpose(1, 2)?.contiguous()?.reshape((b, seq, h))
    }
}

struct Layer {
    attention: SelfAttention,
    attn_out: Linear,
    attn_norm: LayerNorm,
    intermediate: Linear,
    output: Linear,
    output_norm: LayerNorm,
}

impl Layer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let h = config::TEXT_HIDDEN;
        let i = config::TEXT_INTERMEDIATE;
        let eps = config::TEXT_LN_EPS;
        Ok(Self {
            attention: SelfAttention::load(vb.pp("attention").pp("self"))?,
            attn_out: linear(h, h, vb.pp("attention").pp("output").pp("dense"))?,
            attn_norm: layer_norm(h, eps, vb.pp("attention").pp("output").pp("LayerNorm"))?,
            intermediate: linear(h, i, vb.pp("intermediate").pp("dense"))?,
            output: linear(i, h, vb.pp("output").pp("dense"))?,
            output_norm: layer_norm(h, eps, vb.pp("output").pp("LayerNorm"))?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let attn = self.attention.forward(hidden)?;
        let attn = self.attn_out.forward(&attn)?;
        let hidden = self.attn_norm.forward(&(attn + hidden)?)?;
        let inter = self.intermediate.forward(&hidden)?.gelu_erf()?;
        let out = self.output.forward(&inter)?;
        self.output_norm.forward(&(out + &hidden)?)
    }
}

/// The RoBERTa encoder + tanh `[CLS]` pooler. `forward` returns the pooled `(1, hidden)` output that
/// `text_projection` consumes.
pub struct TextTower {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    pooler: Linear,
}

impl TextTower {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let embeddings = Embeddings::load(vb.pp("embeddings"))?;
        let mut layers = Vec::with_capacity(config::TEXT_LAYERS);
        let encoder = vb.pp("encoder");
        for i in 0..config::TEXT_LAYERS {
            layers.push(Layer::load(encoder.pp("layer").pp(i))?);
        }
        let pooler = linear(
            config::TEXT_HIDDEN,
            config::TEXT_HIDDEN,
            vb.pp("pooler").pp("dense"),
        )?;
        Ok(Self {
            embeddings,
            layers,
            pooler,
        })
    }

    /// Encode token ids → the `(1, hidden)` pooled output (`tanh(dense(hidden[:, 0]))`).
    pub fn forward(&self, input_ids: &[u32], device: &Device) -> Result<Tensor> {
        let ids = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), device)?;
        let mut hidden = self.embeddings.forward(&ids, device)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        // Pool the first ([CLS]) token: tanh(dense(hidden[:, 0])).
        let first = hidden.narrow(1, 0, 1)?.squeeze(1)?; // (1, hidden)
        self.pooler.forward(&first)?.tanh()
    }
}
