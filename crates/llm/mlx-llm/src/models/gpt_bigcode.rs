//! GPTBigCode / StarCoderBase decoder used by StarVector-1B.
//!
//! This is deliberately separate from the RoPE-based Llama decoder: StarCoderBase uses learned
//! absolute positions and multi-query attention, and treating it as a Llama alias would load the
//! checkpoint while producing numerically unrelated SVG source.

use mlx_rs::ops::{add, split_sections};
use mlx_rs::Array;

use crate::decode::Decode;
use crate::error::{Error, Result};
use crate::primitives::attention::sdpa_causal;
use crate::primitives::kv_cache::{ContiguousKvCache, KvCache};
use crate::primitives::nn::{embed, gelu_tanh, layer_norm, linear};
use crate::primitives::Weights;

/// Fixed decoder geometry of `starvector/starvector-1b-im2svg`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GptBigCodeConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub layers: usize,
    pub heads: i32,
    pub positions: i32,
}

impl GptBigCodeConfig {
    /// The StarCoderBase-1B decoder embedded in the StarVector-1B snapshot.
    pub const STARVECTOR_1B: Self = Self {
        vocab_size: 49_156,
        hidden_size: 2_048,
        layers: 24,
        heads: 16,
        positions: 8_192,
    };

    fn head_dim(self) -> i32 {
        self.hidden_size / self.heads
    }
}

/// Native GPTBigCode decoder with learned positions and a single MQA KV head.
pub struct GptBigCode {
    token_embedding: Array,
    position_embedding: Array,
    layers: Vec<GptBigCodeLayer>,
    final_norm_weight: Array,
    final_norm_bias: Array,
    lm_head: Array,
    cfg: GptBigCodeConfig,
}

impl GptBigCode {
    /// Load the exact StarVector decoder from its namespaced safetensors map.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: GptBigCodeConfig) -> Result<Self> {
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        let token_embedding = w.require(&key("transformer.wte.weight"))?.clone();
        // GPTBigCode ties the output projection to the token embedding. The published
        // StarVector-1B snapshot therefore intentionally has no separate `lm_head.weight`.
        let lm_head = token_embedding.clone();
        let position_embedding = w.require(&key("transformer.wpe.weight"))?.clone();
        let mut layers = Vec::with_capacity(cfg.layers);
        for index in 0..cfg.layers {
            layers.push(GptBigCodeLayer::from_weights(
                w,
                &key(&format!("transformer.h.{index}")),
                cfg,
            )?);
        }
        let model = Self {
            token_embedding,
            position_embedding,
            layers,
            final_norm_weight: w.require(&key("transformer.ln_f.weight"))?.clone(),
            final_norm_bias: w.require(&key("transformer.ln_f.bias"))?.clone(),
            lm_head,
            cfg,
        };
        w.verify_accessed_gpu_view()?;
        Ok(model)
    }

    /// Token embeddings with learned absolute position rows for a cache offset.
    pub fn embed_at(&self, ids: &Array, offset: i32) -> Result<Array> {
        let shape = ids.shape();
        if shape.len() != 2 || offset < 0 || offset + shape[1] > self.cfg.positions {
            return Err(Error::Msg(
                "gpt_bigcode: invalid absolute position range".into(),
            ));
        }
        let positions: Vec<i32> = (offset..offset + shape[1]).collect();
        let position_ids = Array::from_slice(&positions, &[1, shape[1]]);
        Ok(add(
            &embed(&self.token_embedding, ids)?,
            &embed(&self.position_embedding, &position_ids)?,
        )?)
    }

    /// Token embeddings without position rows, used to concatenate the image prefix before one
    /// shared learned-position lookup over the complete multimodal prompt.
    pub fn raw_embed(&self, ids: &Array) -> Result<Array> {
        embed(&self.token_embedding, ids)
    }

    /// Prefill multimodal embeddings. Vision rows already occupy positions `0..offset`; `embeds`
    /// must therefore include their own positional rows before this call.
    pub fn logits_from_embeds(&self, embeds: &Array, cache: &mut dyn KvCache) -> Result<Array> {
        let mut hidden = embeds.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, cache, index)?;
        }
        let hidden = layer_norm(
            &hidden,
            Some(&self.final_norm_weight),
            Some(&self.final_norm_bias),
            1e-5,
        )?;
        let sequence = hidden.shape()[1];
        let last = hidden.take_axis(Array::from_slice(&[sequence - 1], &[1]), 1)?;
        let logits = linear(&last, &self.lm_head, None)?;
        Ok(logits.reshape(&[hidden.shape()[0], self.cfg.vocab_size])?)
    }

    /// Add the learned position rows to precomputed vision/text embeddings.
    pub fn position_embeds(&self, embeds: &Array, offset: i32) -> Result<Array> {
        let sequence = embeds.shape()[1];
        if offset < 0 || offset + sequence > self.cfg.positions {
            return Err(Error::Msg(
                "gpt_bigcode: multimodal prompt exceeds 8192 positions".into(),
            ));
        }
        let positions: Vec<i32> = (offset..offset + sequence).collect();
        let ids = Array::from_slice(&positions, &[1, sequence]);
        Ok(add(embeds, &embed(&self.position_embedding, &ids)?)?)
    }

    /// A fresh MQA cache for this decoder.
    pub fn cache(&self) -> ContiguousKvCache {
        ContiguousKvCache::new(self.layers.len())
    }
}

impl Decode for GptBigCode {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.cache())
    }

    fn step(&self, ids: &Array, cache: &mut dyn KvCache, offset: i32) -> Result<Array> {
        let embeds = self.embed_at(ids, offset)?;
        self.logits_from_embeds(&embeds, cache)
    }
}

struct GptBigCodeLayer {
    ln_1_weight: Array,
    ln_1_bias: Array,
    attn: GptBigCodeAttention,
    ln_2_weight: Array,
    ln_2_bias: Array,
    fc_weight: Array,
    fc_bias: Array,
    proj_weight: Array,
    proj_bias: Array,
}

impl GptBigCodeLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: GptBigCodeConfig) -> Result<Self> {
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        Ok(Self {
            ln_1_weight: w.require(&key("ln_1.weight"))?.clone(),
            ln_1_bias: w.require(&key("ln_1.bias"))?.clone(),
            attn: GptBigCodeAttention::from_weights(w, &key("attn"), cfg)?,
            ln_2_weight: w.require(&key("ln_2.weight"))?.clone(),
            ln_2_bias: w.require(&key("ln_2.bias"))?.clone(),
            fc_weight: w.require(&key("mlp.c_fc.weight"))?.clone(),
            fc_bias: w.require(&key("mlp.c_fc.bias"))?.clone(),
            proj_weight: w.require(&key("mlp.c_proj.weight"))?.clone(),
            proj_bias: w.require(&key("mlp.c_proj.bias"))?.clone(),
        })
    }

    fn forward(&self, hidden: &Array, cache: &mut dyn KvCache, index: usize) -> Result<Array> {
        let normed = layer_norm(hidden, Some(&self.ln_1_weight), Some(&self.ln_1_bias), 1e-5)?;
        let attended = self.attn.forward(&normed, cache, index)?;
        let hidden = add(hidden, &attended)?;
        let normed = layer_norm(
            &hidden,
            Some(&self.ln_2_weight),
            Some(&self.ln_2_bias),
            1e-5,
        )?;
        // The published StarVector safetensors store every GPTBigCode projection in ordinary
        // `[out, in]` layout, matching the shared linear helper.
        let mlp = linear(&normed, &self.fc_weight, Some(&self.fc_bias))?;
        let mlp = gelu_tanh(&mlp)?;
        let mlp = linear(&mlp, &self.proj_weight, Some(&self.proj_bias))?;
        Ok(add(&hidden, &mlp)?)
    }
}

struct GptBigCodeAttention {
    qkv_weight: Array,
    qkv_bias: Array,
    out_weight: Array,
    out_bias: Array,
    cfg: GptBigCodeConfig,
}

impl GptBigCodeAttention {
    fn from_weights(w: &Weights, prefix: &str, cfg: GptBigCodeConfig) -> Result<Self> {
        Ok(Self {
            qkv_weight: w.require(&format!("{prefix}.c_attn.weight"))?.clone(),
            qkv_bias: w.require(&format!("{prefix}.c_attn.bias"))?.clone(),
            out_weight: w.require(&format!("{prefix}.c_proj.weight"))?.clone(),
            out_bias: w.require(&format!("{prefix}.c_proj.bias"))?.clone(),
            cfg,
        })
    }

    fn forward(&self, hidden: &Array, cache: &mut dyn KvCache, index: usize) -> Result<Array> {
        let shape = hidden.shape();
        let (batch, sequence) = (shape[0], shape[1]);
        let qkv = linear(hidden, &self.qkv_weight, Some(&self.qkv_bias))?;
        let head_dim = self.cfg.head_dim();
        // StarCoderBase-1B is `multi_query=true`: one K and one V head.
        let parts = split_sections(
            &qkv,
            &[self.cfg.hidden_size, self.cfg.hidden_size + head_dim],
            2,
        )?;
        let query = parts[0]
            .reshape(&[batch, sequence, self.cfg.heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let key = parts[1]
            .reshape(&[batch, sequence, 1, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let value = parts[2]
            .reshape(&[batch, sequence, 1, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let (keys, values) = cache.update(index, &key, &value)?;
        let attended = sdpa_causal(&query, &keys, &values, 1.0 / (head_dim as f32).sqrt())?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, sequence, self.cfg.hidden_size])?;
        linear(&attended, &self.out_weight, Some(&self.out_bias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::decode::{generate, GenerationConfig};
    use crate::primitives::input_ids;
    use crate::primitives::sampler::SamplingParams;
    use crate::CancelFlag;

    fn put(map: &mut HashMap<String, Array>, key: &str, values: &[f32], shape: &[i32]) {
        map.insert(key.into(), Array::from_slice(values, shape));
    }

    /// A tiny shape-valid GPTBigCode checkpoint. It is intentionally not StarVector-sized: this
    /// fixture proves the native learned-position + MQA decode path without allocating real weights.
    fn tiny_decoder() -> GptBigCode {
        let cfg = GptBigCodeConfig {
            vocab_size: 3,
            hidden_size: 4,
            layers: 1,
            heads: 2,
            positions: 8,
        };
        let mut map = HashMap::new();
        let prefix = "fixture";
        put(
            &mut map,
            "fixture.transformer.wte.weight",
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[3, 4],
        );
        put(
            &mut map,
            "fixture.transformer.wpe.weight",
            &[0.0; 32],
            &[8, 4],
        );
        for key in [
            "fixture.transformer.h.0.ln_1.weight",
            "fixture.transformer.h.0.ln_2.weight",
            "fixture.transformer.ln_f.weight",
        ] {
            put(&mut map, key, &[1.0; 4], &[4]);
        }
        for key in [
            "fixture.transformer.h.0.ln_1.bias",
            "fixture.transformer.h.0.ln_2.bias",
            "fixture.transformer.ln_f.bias",
        ] {
            put(&mut map, key, &[0.0; 4], &[4]);
        }
        // Q is 4 wide and K/V are one 2-wide MQA head each: 8 total output rows in the
        // checkpoint's published `[out, in]` projection layout.
        put(
            &mut map,
            "fixture.transformer.h.0.attn.c_attn.weight",
            &[0.0; 32],
            &[8, 4],
        );
        put(
            &mut map,
            "fixture.transformer.h.0.attn.c_attn.bias",
            &[0.0; 8],
            &[8],
        );
        put(
            &mut map,
            "fixture.transformer.h.0.attn.c_proj.weight",
            &[0.0; 16],
            &[4, 4],
        );
        put(
            &mut map,
            "fixture.transformer.h.0.attn.c_proj.bias",
            &[0.0; 4],
            &[4],
        );
        put(
            &mut map,
            "fixture.transformer.h.0.mlp.c_fc.weight",
            &[0.0; 32],
            &[8, 4],
        );
        put(
            &mut map,
            "fixture.transformer.h.0.mlp.c_fc.bias",
            &[0.0; 8],
            &[8],
        );
        put(
            &mut map,
            "fixture.transformer.h.0.mlp.c_proj.weight",
            &[0.0; 32],
            &[4, 8],
        );
        put(
            &mut map,
            "fixture.transformer.h.0.mlp.c_proj.bias",
            &[0.0; 4],
            &[4],
        );
        GptBigCode::from_weights(&Weights::from_map(map), prefix, cfg).unwrap()
    }

    #[test]
    fn tiny_decoder_loads_and_projects_with_tied_token_embeddings() {
        let model = tiny_decoder();
        assert_eq!(
            model.lm_head.as_slice::<f32>(),
            model.token_embedding.as_slice::<f32>()
        );

        let mut cache = model.cache();
        let logits = model
            .logits_from_embeds(&model.embed_at(&input_ids(&[1]), 0).unwrap(), &mut cache)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 3]);
        let logits = logits.as_slice::<f32>();
        assert!(logits[1] > logits[0]);
        assert!(logits[1] > logits[2]);
    }

    #[test]
    fn tiny_mqa_greedy_decode_is_deterministic_and_bounded() {
        let model = tiny_decoder();
        let config = GenerationConfig {
            max_new_tokens: 3,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            seed: Some(7),
            stop_tokens: Vec::new(),
        };
        let cancel = CancelFlag::new();
        let run = || {
            let mut events = Vec::new();
            let output = generate(&model, &[1], &config, &cancel, &mut |event| {
                events.push(event)
            })
            .unwrap();
            (output, events)
        };
        let (first, first_events) = run();
        let (second, second_events) = run();
        assert_eq!(first.tokens, second.tokens);
        assert_eq!(first.tokens.len(), 3);
        assert_eq!(first_events, second_events);

        // A prefill exposes only the last position logits and uses the same cache geometry as the
        // one-token continuation loop above.
        let mut cache = model.cache();
        assert_eq!(
            model
                .logits_from_embeds(&model.embed_at(&input_ids(&[1]), 0).unwrap(), &mut cache)
                .unwrap()
                .shape(),
            &[1, 3]
        );
    }
}
