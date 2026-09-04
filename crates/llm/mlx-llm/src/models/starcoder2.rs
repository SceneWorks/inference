//! Native StarCoder2 decoder used by StarVector-8B.
//!
//! StarCoder2 is not GPTBigCode: it has separate Q/K/V GQA projections, RoPE, biasful LayerNorm,
//! and a biasful GELU MLP. Keeping that shape separate prevents a compatible-looking checkpoint
//! from being decoded with StarVector-1B's learned-position MQA math.

use mlx_rs::ops::add;
use mlx_rs::Array;

use crate::decode::Decode;
use crate::error::Result;
use crate::primitives::attention::sdpa_causal;
use crate::primitives::kv_cache::{ContiguousKvCache, KvCache};
use crate::primitives::nn::{embed, gelu_tanh, layer_norm, linear};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::Weights;

/// Fixed decoder geometry of `starvector/starvector-8b-im2svg`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarCoder2Config {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub layers: usize,
    pub heads: i32,
    pub kv_heads: i32,
    pub rope_theta: f32,
    pub layer_norm_eps: f32,
}

impl StarCoder2Config {
    /// The StarCoder2-7B decoder embedded in the published StarVector-8B snapshot.
    ///
    /// The snapshot's `config.json` records the 49,152-token base vocabulary, while its tokenizer
    /// adds five StarVector tokens and the tied embedding/head is correspondingly resized to
    /// 49,157 rows. Runtime logits must follow the checkpoint tensor, not the base-vocabulary
    /// provenance field.
    pub const STARVECTOR_8B: Self = Self {
        vocab_size: 49_157,
        hidden_size: 4_608,
        intermediate_size: 18_432,
        layers: 32,
        heads: 36,
        kv_heads: 4,
        rope_theta: 1_000_000.0,
        layer_norm_eps: 1e-5,
    };

    fn head_dim(self) -> i32 {
        self.hidden_size / self.heads
    }
}

/// Native StarCoder2 decoder with tied input/output embeddings.
pub struct StarCoder2 {
    embed_tokens: Array,
    layers: Vec<StarCoder2Layer>,
    final_norm_weight: Array,
    final_norm_bias: Array,
    cfg: StarCoder2Config,
}

impl StarCoder2 {
    /// Load the exact StarVector-8B decoder from its namespaced safetensors map.
    ///
    /// The published snapshot stores no separate `lm_head.weight`: its language-model head is tied
    /// to `model.embed_tokens.weight`, so this loader deliberately uses that same array rather than
    /// accepting an unrelated optional head.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: StarCoder2Config) -> Result<Self> {
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        let embed_tokens = w.require(&key("model.embed_tokens.weight"))?.clone();
        let expected_embedding_shape = [cfg.vocab_size, cfg.hidden_size];
        if embed_tokens.shape() != expected_embedding_shape {
            return Err(crate::error::Error::Config(format!(
                "StarCoder2 tied embedding/head must be {expected_embedding_shape:?}, got {:?}",
                embed_tokens.shape()
            )));
        }
        let mut layers = Vec::with_capacity(cfg.layers);
        for index in 0..cfg.layers {
            layers.push(StarCoder2Layer::from_weights(
                w,
                &key(&format!("model.layers.{index}")),
                cfg,
            )?);
        }
        let model = Self {
            embed_tokens,
            layers,
            final_norm_weight: w.require(&key("model.norm.weight"))?.clone(),
            final_norm_bias: w.require(&key("model.norm.bias"))?.clone(),
            cfg,
        };
        w.verify_accessed_gpu_view()?;
        Ok(model)
    }

    /// Embed token ids without any positional addition; StarCoder2 positions are RoPE-only.
    pub fn embed(&self, ids: &Array) -> Result<Array> {
        embed(&self.embed_tokens, ids)
    }

    /// Prefill or decode precomputed embeddings, returning last-position logits.
    pub fn logits_from_embeds(
        &self,
        embeds: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let sequence = embeds.shape()[1];
        let rope = Rope::standard(self.cfg.head_dim(), self.cfg.rope_theta);
        let (cos, sin) = rope.cos_sin(sequence, offset, embeds.dtype())?;
        let mut hidden = embeds.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, cache, index)?;
        }
        let hidden = layer_norm(
            &hidden,
            Some(&self.final_norm_weight),
            Some(&self.final_norm_bias),
            self.cfg.layer_norm_eps,
        )?;
        let last = hidden.take_axis(Array::from_slice(&[hidden.shape()[1] - 1], &[1]), 1)?;
        let logits = linear(&last, &self.embed_tokens, None)?;
        Ok(logits.reshape(&[hidden.shape()[0], self.cfg.vocab_size])?)
    }

    /// A fresh GQA cache for this decoder.
    pub fn cache(&self) -> ContiguousKvCache {
        ContiguousKvCache::new(self.layers.len())
    }
}

impl Decode for StarCoder2 {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.cache())
    }

    fn step(&self, ids: &Array, cache: &mut dyn KvCache, offset: i32) -> Result<Array> {
        self.logits_from_embeds(&self.embed(ids)?, cache, offset)
    }
}

struct StarCoder2Layer {
    input_norm_weight: Array,
    input_norm_bias: Array,
    attn: StarCoder2Attention,
    post_attn_norm_weight: Array,
    post_attn_norm_bias: Array,
    mlp_fc_weight: Array,
    mlp_fc_bias: Array,
    mlp_proj_weight: Array,
    mlp_proj_bias: Array,
    eps: f32,
}

impl StarCoder2Layer {
    fn from_weights(w: &Weights, prefix: &str, cfg: StarCoder2Config) -> Result<Self> {
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        Ok(Self {
            input_norm_weight: w.require(&key("input_layernorm.weight"))?.clone(),
            input_norm_bias: w.require(&key("input_layernorm.bias"))?.clone(),
            attn: StarCoder2Attention::from_weights(w, &key("self_attn"), cfg)?,
            post_attn_norm_weight: w.require(&key("post_attention_layernorm.weight"))?.clone(),
            post_attn_norm_bias: w.require(&key("post_attention_layernorm.bias"))?.clone(),
            mlp_fc_weight: w.require(&key("mlp.c_fc.weight"))?.clone(),
            mlp_fc_bias: w.require(&key("mlp.c_fc.bias"))?.clone(),
            mlp_proj_weight: w.require(&key("mlp.c_proj.weight"))?.clone(),
            mlp_proj_bias: w.require(&key("mlp.c_proj.bias"))?.clone(),
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(
        &self,
        hidden: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut dyn KvCache,
        index: usize,
    ) -> Result<Array> {
        let normed = layer_norm(
            hidden,
            Some(&self.input_norm_weight),
            Some(&self.input_norm_bias),
            self.eps,
        )?;
        let attended = self.attn.forward(&normed, cos, sin, cache, index)?;
        let hidden = add(hidden, &attended)?;
        let normed = layer_norm(
            &hidden,
            Some(&self.post_attn_norm_weight),
            Some(&self.post_attn_norm_bias),
            self.eps,
        )?;
        let mlp = gelu_tanh(&linear(
            &normed,
            &self.mlp_fc_weight,
            Some(&self.mlp_fc_bias),
        )?)?;
        let mlp = linear(&mlp, &self.mlp_proj_weight, Some(&self.mlp_proj_bias))?;
        Ok(add(&hidden, &mlp)?)
    }
}

struct StarCoder2Attention {
    q_weight: Array,
    q_bias: Array,
    k_weight: Array,
    k_bias: Array,
    v_weight: Array,
    v_bias: Array,
    o_weight: Array,
    o_bias: Array,
    cfg: StarCoder2Config,
}

impl StarCoder2Attention {
    fn from_weights(w: &Weights, prefix: &str, cfg: StarCoder2Config) -> Result<Self> {
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        Ok(Self {
            q_weight: w.require(&key("q_proj.weight"))?.clone(),
            q_bias: w.require(&key("q_proj.bias"))?.clone(),
            k_weight: w.require(&key("k_proj.weight"))?.clone(),
            k_bias: w.require(&key("k_proj.bias"))?.clone(),
            v_weight: w.require(&key("v_proj.weight"))?.clone(),
            v_bias: w.require(&key("v_proj.bias"))?.clone(),
            o_weight: w.require(&key("o_proj.weight"))?.clone(),
            o_bias: w.require(&key("o_proj.bias"))?.clone(),
            cfg,
        })
    }

    fn forward(
        &self,
        hidden: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut dyn KvCache,
        index: usize,
    ) -> Result<Array> {
        let (batch, sequence) = (hidden.shape()[0], hidden.shape()[1]);
        let head_dim = self.cfg.head_dim();
        let query = linear(hidden, &self.q_weight, Some(&self.q_bias))?.reshape(&[
            batch,
            sequence,
            self.cfg.heads,
            head_dim,
        ])?;
        let key = linear(hidden, &self.k_weight, Some(&self.k_bias))?.reshape(&[
            batch,
            sequence,
            self.cfg.kv_heads,
            head_dim,
        ])?;
        let value = linear(hidden, &self.v_weight, Some(&self.v_bias))?.reshape(&[
            batch,
            sequence,
            self.cfg.kv_heads,
            head_dim,
        ])?;
        let query = apply_rope(&query, cos, sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
        let key = apply_rope(&key, cos, sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
        let value = value.transpose_axes(&[0, 2, 1, 3])?;
        let (keys, values) = cache.update(index, &key, &value)?;
        let attended = sdpa_causal(&query, &keys, &values, 1.0 / (head_dim as f32).sqrt())?;
        let merged = attended.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            sequence,
            self.cfg.hidden_size,
        ])?;
        linear(&merged, &self.o_weight, Some(&self.o_bias))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::decode::{generate, GenerationConfig};
    use crate::primitives::sampler::SamplingParams;
    use crate::CancelFlag;

    fn put(map: &mut HashMap<String, Array>, key: &str, values: &[f32], shape: &[i32]) {
        map.insert(key.into(), Array::from_slice(values, shape));
    }

    fn tiny_decoder() -> StarCoder2 {
        let cfg = StarCoder2Config {
            vocab_size: 3,
            hidden_size: 4,
            intermediate_size: 8,
            layers: 1,
            heads: 2,
            kv_heads: 1,
            rope_theta: 10_000.0,
            layer_norm_eps: 1e-5,
        };
        let mut map = HashMap::new();
        let prefix = "fixture";
        put(
            &mut map,
            "fixture.model.embed_tokens.weight",
            &[0.0; 12],
            &[3, 4],
        );
        for key in [
            "fixture.model.layers.0.input_layernorm.weight",
            "fixture.model.layers.0.post_attention_layernorm.weight",
            "fixture.model.norm.weight",
        ] {
            put(&mut map, key, &[1.0; 4], &[4]);
        }
        for key in [
            "fixture.model.layers.0.input_layernorm.bias",
            "fixture.model.layers.0.post_attention_layernorm.bias",
            "fixture.model.norm.bias",
        ] {
            put(&mut map, key, &[0.0; 4], &[4]);
        }
        for key in ["q", "o"] {
            put(
                &mut map,
                &format!("fixture.model.layers.0.self_attn.{key}_proj.weight"),
                &[0.0; 16],
                &[4, 4],
            );
            put(
                &mut map,
                &format!("fixture.model.layers.0.self_attn.{key}_proj.bias"),
                &[0.0; 4],
                &[4],
            );
        }
        for key in ["k", "v"] {
            put(
                &mut map,
                &format!("fixture.model.layers.0.self_attn.{key}_proj.weight"),
                &[0.0; 8],
                &[2, 4],
            );
            put(
                &mut map,
                &format!("fixture.model.layers.0.self_attn.{key}_proj.bias"),
                &[0.0; 2],
                &[2],
            );
        }
        put(
            &mut map,
            "fixture.model.layers.0.mlp.c_fc.weight",
            &[0.0; 32],
            &[8, 4],
        );
        put(
            &mut map,
            "fixture.model.layers.0.mlp.c_fc.bias",
            &[0.0; 8],
            &[8],
        );
        put(
            &mut map,
            "fixture.model.layers.0.mlp.c_proj.weight",
            &[0.0; 32],
            &[4, 8],
        );
        put(
            &mut map,
            "fixture.model.layers.0.mlp.c_proj.bias",
            &[0.0; 4],
            &[4],
        );
        StarCoder2::from_weights(&Weights::from_map(map), prefix, cfg).unwrap()
    }

    #[test]
    fn tiny_gqa_rope_decoder_is_deterministic_and_ties_the_lm_head() {
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
        assert_eq!(first_events, second_events);
        assert_eq!(first.tokens.len(), 3);
    }

    #[test]
    fn starvector_8b_runtime_vocab_includes_checkpoint_added_tokens() {
        assert_eq!(StarCoder2Config::STARVECTOR_8B.vocab_size, 49_157);
    }

    #[test]
    fn rejects_tied_embedding_shape_that_disagrees_with_runtime_vocab() {
        let cfg = StarCoder2Config {
            vocab_size: 3,
            hidden_size: 4,
            intermediate_size: 8,
            layers: 1,
            heads: 2,
            kv_heads: 1,
            rope_theta: 10_000.0,
            layer_norm_eps: 1e-5,
        };
        let mut map = HashMap::new();
        put(
            &mut map,
            "fixture.model.embed_tokens.weight",
            &[0.0; 16],
            &[4, 4],
        );
        let error = match StarCoder2::from_weights(&Weights::from_map(map), "fixture", cfg) {
            Ok(_) => panic!("mismatched tied embedding/head shape must fail closed"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            crate::error::Error::Config(message)
                if message.contains("must be [3, 4]") && message.contains("got [4, 4]")
        ));
    }
}
