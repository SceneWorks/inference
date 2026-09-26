//! The SheetSage2 BART decoder (transformers 4.45.2 `BartDecoder` as configured by
//! `modeling_sheetsage2.py`): token embeddings (scale 1.0) tied to the output projection, learned
//! positions with BART's offset of 2, `layernorm_embedding`, post-LN layers with biased
//! self-attention (causal), cross-attention over the encoder memory, and a GELU FFN.
//!
//! Decoding is incremental: the self-attention K/V grow by one position per step and the
//! cross-attention K/V are projected from the memory **once** per window.

use candle_audio::candle_core::{Device, Tensor};

use super::config::SheetSage2Config;
use super::mert::{LayerNorm, Linear};
use super::weights::Weights;
use crate::Error;

const OFFSET: usize = 2;

#[derive(Clone, Debug)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

#[derive(Clone, Debug)]
struct Layer {
    self_attn: Attention,
    self_norm: LayerNorm,
    cross_attn: Attention,
    cross_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
}

/// The decoder weights.
#[derive(Clone, Debug)]
pub struct Decoder {
    embedding: Tensor,
    embedding_t: Tensor,
    positions: Tensor,
    norm_embedding: LayerNorm,
    layers: Vec<Layer>,
    heads: usize,
    width: usize,
    bytes: usize,
}

/// Per-window decoding state: cached cross-attention K/V and the growing self-attention K/V.
pub struct DecoderCache {
    cross: Vec<(Tensor, Tensor)>,
    selfs: Vec<Option<(Tensor, Tensor)>>,
    length: usize,
}

impl DecoderCache {
    /// Tokens decoded so far.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether nothing was decoded yet.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

impl Decoder {
    /// Load from SheetSage2's `token_embedding.weight` and `decoder.*` for `config`'s shape.
    pub(crate) fn load(
        w: &mut Weights,
        config: &SheetSage2Config,
        device: &Device,
    ) -> Result<Self, Error> {
        let (vocab, width, layers, heads, ffn, max_positions) = (
            config.vocab_size,
            config.hidden_size,
            config.decoder_layers,
            config.num_attention_heads,
            config.intermediate_size,
            config.max_output_seq_len,
        );
        let embedding = w.take("token_embedding.weight", &[vocab, width])?;
        let positions = w.take(
            "decoder.embed_positions.weight",
            &[max_positions + OFFSET, width],
        )?;
        let norm_embedding =
            LayerNorm::take(w, "decoder.layernorm_embedding", width, 1e-5, device)?;
        let mut bytes = (embedding.elem_count() + positions.elem_count()) * 4 + 2 * width * 4;
        let mut out = Vec::new();
        for i in 0..layers {
            let p = format!("decoder.layers.{i}");
            let mut lin = |name: &str, o: usize, inp: usize| -> Result<Linear, Error> {
                let l = Linear::new(
                    &w.take(&format!("{p}.{name}.weight"), &[o, inp])?,
                    Some(w.take(&format!("{p}.{name}.bias"), &[o])?),
                    device,
                )?;
                bytes += l.bytes();
                Ok(l)
            };
            let self_attn = Attention {
                q: lin("self_attn.q_proj", width, width)?,
                k: lin("self_attn.k_proj", width, width)?,
                v: lin("self_attn.v_proj", width, width)?,
                out: lin("self_attn.out_proj", width, width)?,
            };
            let cross_attn = Attention {
                q: lin("encoder_attn.q_proj", width, width)?,
                k: lin("encoder_attn.k_proj", width, width)?,
                v: lin("encoder_attn.v_proj", width, width)?,
                out: lin("encoder_attn.out_proj", width, width)?,
            };
            let fc1 = lin("fc1", ffn, width)?;
            let fc2 = lin("fc2", width, ffn)?;
            out.push(Layer {
                self_attn,
                self_norm: LayerNorm::take(
                    w,
                    &format!("{p}.self_attn_layer_norm"),
                    width,
                    1e-5,
                    device,
                )?,
                cross_attn,
                cross_norm: LayerNorm::take(
                    w,
                    &format!("{p}.encoder_attn_layer_norm"),
                    width,
                    1e-5,
                    device,
                )?,
                fc1,
                fc2,
                final_norm: LayerNorm::take(
                    w,
                    &format!("{p}.final_layer_norm"),
                    width,
                    1e-5,
                    device,
                )?,
            });
            bytes += 6 * width * 4;
        }
        let embedding = embedding.to_device(device)?;
        Ok(Self {
            embedding_t: embedding.t()?.contiguous()?,
            embedding,
            positions: positions.to_device(device)?,
            norm_embedding,
            layers: out,
            heads,
            width,
            bytes,
        })
    }

    /// Parameter bytes on the device.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Positional capacity.
    pub fn max_positions(&self) -> usize {
        self.positions.dim(0).unwrap_or(OFFSET) - OFFSET
    }

    fn split_heads(&self, x: &Tensor) -> Result<Tensor, Error> {
        let (b, t, _) = x.dims3()?;
        Ok(x.reshape((b, t, self.heads, self.width / self.heads))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    /// Project `memory` (`[1, frames, width]`) into every layer's cross-attention K/V.
    pub fn start(&self, memory: &Tensor) -> Result<DecoderCache, Error> {
        let cross = self
            .layers
            .iter()
            .map(|l| {
                Ok((
                    self.split_heads(&l.cross_attn.k.forward(memory)?)?,
                    self.split_heads(&l.cross_attn.v.forward(memory)?)?,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(DecoderCache {
            cross,
            selfs: vec![None; self.layers.len()],
            length: 0,
        })
    }

    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal_offset: Option<usize>,
    ) -> Result<Tensor, Error> {
        let head_dim = self.width / self.heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        if let Some(offset) = causal_offset {
            let (tq, tk) = (scores.dim(2)?, scores.dim(3)?);
            if tq > 1 {
                let mask: Vec<f32> = (0..tq)
                    .flat_map(|i| {
                        (0..tk).map(move |j| {
                            if j > offset + i {
                                f32::NEG_INFINITY
                            } else {
                                0.0
                            }
                        })
                    })
                    .collect();
                let mask = Tensor::from_vec(mask, (1, 1, tq, tk), scores.device())?;
                scores = scores.broadcast_add(&mask)?;
            }
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(v)?;
        let (b, h, t, d) = out.dims4()?;
        Ok(out.transpose(1, 2)?.contiguous()?.reshape((b, t, h * d))?)
    }

    /// Feed `tokens` (the whole prefix on the first call, then one token) and return the logits of
    /// the last position, `[vocab]`.
    pub fn step(&self, cache: &mut DecoderCache, tokens: &[u32]) -> Result<Tensor, Error> {
        let device = self.embedding.device();
        let n = tokens.len();
        let past = cache.length;
        if past + n > self.max_positions() {
            return Err(Error::Request(format!(
                "decoder context of {} positions exceeded",
                self.max_positions()
            )));
        }
        let ids = Tensor::from_vec(tokens.to_vec(), n, device)?;
        let embeds = self.embedding.index_select(&ids, 0)?;
        let positions = self.positions.narrow(0, past + OFFSET, n)?;
        let mut h = self
            .norm_embedding
            .forward(&(embeds + positions)?.unsqueeze(0)?)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let residual = h.clone();
            let q = self.split_heads(&layer.self_attn.q.forward(&h)?)?;
            let k_new = self.split_heads(&layer.self_attn.k.forward(&h)?)?;
            let v_new = self.split_heads(&layer.self_attn.v.forward(&h)?)?;
            let (k, v) = match cache.selfs[index].take() {
                Some((k, v)) => (
                    Tensor::cat(&[&k, &k_new], 2)?,
                    Tensor::cat(&[&v, &v_new], 2)?,
                ),
                None => (k_new, v_new),
            };
            let attended = self.attend(&q, &k, &v, Some(past))?;
            cache.selfs[index] = Some((k, v));
            h = layer
                .self_norm
                .forward(&(residual + layer.self_attn.out.forward(&attended)?)?)?;
            let residual = h.clone();
            let q = self.split_heads(&layer.cross_attn.q.forward(&h)?)?;
            let (ck, cv) = &cache.cross[index];
            let attended = self.attend(&q, ck, cv, None)?;
            h = layer
                .cross_norm
                .forward(&(residual + layer.cross_attn.out.forward(&attended)?)?)?;
            let residual = h.clone();
            let ffn = layer.fc2.forward(&layer.fc1.forward(&h)?.gelu_erf()?)?;
            h = layer.final_norm.forward(&(residual + ffn)?)?;
        }
        cache.length += n;
        let last = h.narrow(1, n - 1, 1)?.reshape((1, self.width))?;
        Ok(last.matmul(&self.embedding_t)?.flatten_all()?)
    }
}
