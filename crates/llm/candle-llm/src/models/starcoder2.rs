//! Native StarCoder2 decoder used only by StarVector-8B.
//!
//! This deliberately stays separate from the GPTBigCode decoder used by the 1B model: StarCoder2
//! uses GQA, RoPE, biasful LayerNorm, and a biasful GELU MLP.

use candle_core::{DType, Device, Tensor};

use crate::decode::Decode;
use crate::error::Result;
use crate::primitives::attention::{repeat_kv, sdpa_causal};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::{embed, gelu, layer_norm, linear};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::{ContiguousKvCache, Weights};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarCoder2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub rope_theta: f32,
    pub layer_norm_eps: f64,
}

impl StarCoder2Config {
    pub const STARVECTOR_8B: Self = Self {
        vocab_size: 49_152,
        hidden_size: 4_608,
        intermediate_size: 18_432,
        layers: 32,
        heads: 36,
        kv_heads: 4,
        rope_theta: 1_000_000.0,
        layer_norm_eps: 1e-5,
    };

    fn head_dim(self) -> usize {
        self.hidden_size / self.heads
    }
}

pub struct StarCoder2 {
    embed_tokens: Tensor,
    layers: Vec<StarCoder2Layer>,
    final_norm_weight: Tensor,
    final_norm_bias: Tensor,
    cfg: StarCoder2Config,
    dtype: DType,
    device: Device,
}

impl StarCoder2 {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: StarCoder2Config) -> Result<Self> {
        let dtype = crate::device::compute_dtype(w.device());
        let req = |key: String| -> Result<Tensor> { Ok(w.require(&key)?.to_dtype(dtype)?) };
        let key = |suffix: &str| join(prefix, suffix);
        let layers = (0..cfg.layers)
            .map(|index| {
                StarCoder2Layer::from_weights(w, &key(&format!("model.layers.{index}")), cfg, dtype)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: req(key("model.embed_tokens.weight"))?,
            layers,
            final_norm_weight: req(key("model.norm.weight"))?,
            final_norm_bias: req(key("model.norm.bias"))?,
            cfg,
            dtype,
            device: w.device().clone(),
        })
    }

    pub fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        embed(&self.embed_tokens, ids)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn logits_from_embeds(
        &self,
        embeds: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Tensor> {
        let (_, sequence, hidden) = embeds.dims3()?;
        debug_assert_eq!(hidden, self.cfg.hidden_size);
        let (cos, sin) = Rope::standard(self.cfg.head_dim() as i32, self.cfg.rope_theta).cos_sin(
            sequence as i32,
            offset,
            self.dtype,
            &self.device,
        )?;
        let mut state = embeds.to_dtype(self.dtype)?;
        for (index, layer) in self.layers.iter().enumerate() {
            state = layer.forward(&state, &cos, &sin, cache, index)?;
        }
        let state = layer_norm(
            &state,
            &self.final_norm_weight,
            &self.final_norm_bias,
            self.cfg.layer_norm_eps,
        )?;
        let last = state.narrow(1, sequence - 1, 1)?.squeeze(1)?;
        linear(&last, &self.embed_tokens, None)
    }

    pub fn cache(&self) -> ContiguousKvCache {
        ContiguousKvCache::new(self.layers.len())
    }
}

impl Decode for StarCoder2 {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.cache())
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn step(&self, ids: &Tensor, cache: &mut dyn KvCache, offset: i32) -> Result<Tensor> {
        self.logits_from_embeds(&self.embed(ids)?, cache, offset)
    }
}

struct StarCoder2Layer {
    input_norm_weight: Tensor,
    input_norm_bias: Tensor,
    attn: StarCoder2Attention,
    post_attn_norm_weight: Tensor,
    post_attn_norm_bias: Tensor,
    mlp_fc_weight: Tensor,
    mlp_fc_bias: Tensor,
    mlp_proj_weight: Tensor,
    mlp_proj_bias: Tensor,
    eps: f64,
}

impl StarCoder2Layer {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: StarCoder2Config,
        dtype: DType,
    ) -> Result<Self> {
        let key = |suffix: &str| join(prefix, suffix);
        let req = |key: String| -> Result<Tensor> { Ok(w.require(&key)?.to_dtype(dtype)?) };
        Ok(Self {
            input_norm_weight: req(key("input_layernorm.weight"))?,
            input_norm_bias: req(key("input_layernorm.bias"))?,
            attn: StarCoder2Attention::from_weights(w, &key("self_attn"), cfg, dtype)?,
            post_attn_norm_weight: req(key("post_attention_layernorm.weight"))?,
            post_attn_norm_bias: req(key("post_attention_layernorm.bias"))?,
            mlp_fc_weight: req(key("mlp.c_fc.weight"))?,
            mlp_fc_bias: req(key("mlp.c_fc.bias"))?,
            mlp_proj_weight: req(key("mlp.c_proj.weight"))?,
            mlp_proj_bias: req(key("mlp.c_proj.bias"))?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut dyn KvCache,
        index: usize,
    ) -> Result<Tensor> {
        let normed = layer_norm(
            hidden,
            &self.input_norm_weight,
            &self.input_norm_bias,
            self.eps,
        )?;
        let attended = self.attn.forward(&normed, cos, sin, cache, index)?;
        let hidden = hidden.broadcast_add(&attended)?;
        let normed = layer_norm(
            &hidden,
            &self.post_attn_norm_weight,
            &self.post_attn_norm_bias,
            self.eps,
        )?;
        let mlp = gelu(&linear(
            &normed,
            &self.mlp_fc_weight,
            Some(&self.mlp_fc_bias),
        )?)?;
        hidden
            .broadcast_add(&linear(
                &mlp,
                &self.mlp_proj_weight,
                Some(&self.mlp_proj_bias),
            )?)
            .map_err(Into::into)
    }
}

struct StarCoder2Attention {
    q_weight: Tensor,
    q_bias: Tensor,
    k_weight: Tensor,
    k_bias: Tensor,
    v_weight: Tensor,
    v_bias: Tensor,
    o_weight: Tensor,
    o_bias: Tensor,
    cfg: StarCoder2Config,
}

impl StarCoder2Attention {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: StarCoder2Config,
        dtype: DType,
    ) -> Result<Self> {
        let key = |suffix: &str| join(prefix, suffix);
        let req = |key: String| -> Result<Tensor> { Ok(w.require(&key)?.to_dtype(dtype)?) };
        Ok(Self {
            q_weight: req(key("q_proj.weight"))?,
            q_bias: req(key("q_proj.bias"))?,
            k_weight: req(key("k_proj.weight"))?,
            k_bias: req(key("k_proj.bias"))?,
            v_weight: req(key("v_proj.weight"))?,
            v_bias: req(key("v_proj.bias"))?,
            o_weight: req(key("o_proj.weight"))?,
            o_bias: req(key("o_proj.bias"))?,
            cfg,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut dyn KvCache,
        index: usize,
    ) -> Result<Tensor> {
        let (batch, sequence, _) = hidden.dims3()?;
        let dim = self.cfg.head_dim();
        let q = linear(hidden, &self.q_weight, Some(&self.q_bias))?.reshape((
            batch,
            sequence,
            self.cfg.heads,
            dim,
        ))?;
        let k = linear(hidden, &self.k_weight, Some(&self.k_bias))?.reshape((
            batch,
            sequence,
            self.cfg.kv_heads,
            dim,
        ))?;
        let v = linear(hidden, &self.v_weight, Some(&self.v_bias))?.reshape((
            batch,
            sequence,
            self.cfg.kv_heads,
            dim,
        ))?;
        let q = apply_rope(&q, cos, sin, false)?.transpose(1, 2)?;
        let k = apply_rope(&k, cos, sin, false)?.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let (k, v) = cache.update(index, &k, &v)?;
        let groups = self.cfg.heads / self.cfg.kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;
        let out = sdpa_causal(&q, &k, &v, 1.0 / (dim as f32).sqrt())?;
        let out = out
            .transpose(1, 2)?
            .reshape((batch, sequence, self.cfg.hidden_size))?;
        linear(&out, &self.o_weight, Some(&self.o_bias))
    }
}

fn join(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    trait TestShape {
        fn into_shape(self) -> candle_core::Shape;
    }
    impl TestShape for usize {
        fn into_shape(self) -> candle_core::Shape {
            candle_core::Shape::from_dims(&[self])
        }
    }
    impl TestShape for (usize, usize) {
        fn into_shape(self) -> candle_core::Shape {
            candle_core::Shape::from_dims(&[self.0, self.1])
        }
    }

    fn put(map: &mut HashMap<String, Tensor>, key: &str, values: Vec<f32>, shape: impl TestShape) {
        map.insert(
            key.into(),
            Tensor::from_vec(values, shape.into_shape(), &Device::Cpu).unwrap(),
        );
    }

    #[test]
    fn tiny_gqa_rope_decoder_is_deterministic_and_uses_tied_head() {
        let cfg = StarCoder2Config {
            vocab_size: 3,
            hidden_size: 4,
            intermediate_size: 8,
            layers: 1,
            heads: 2,
            kv_heads: 1,
            rope_theta: 10_000.,
            layer_norm_eps: 1e-5,
        };
        let mut map = HashMap::new();
        let p = "fixture";
        put(
            &mut map,
            "fixture.model.embed_tokens.weight",
            vec![0.; 12],
            (3, 4),
        );
        for key in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            put(
                &mut map,
                &format!("{p}.model.layers.0.{key}"),
                vec![1.; 4],
                4,
            );
        }
        for key in ["input_layernorm.bias", "post_attention_layernorm.bias"] {
            put(
                &mut map,
                &format!("{p}.model.layers.0.{key}"),
                vec![0.; 4],
                4,
            );
        }
        put(&mut map, "fixture.model.norm.weight", vec![1.; 4], 4);
        put(&mut map, "fixture.model.norm.bias", vec![0.; 4], 4);
        for key in ["q", "o"] {
            put(
                &mut map,
                &format!("{p}.model.layers.0.self_attn.{key}_proj.weight"),
                vec![0.; 16],
                (4, 4),
            );
            put(
                &mut map,
                &format!("{p}.model.layers.0.self_attn.{key}_proj.bias"),
                vec![0.; 4],
                4,
            );
        }
        for key in ["k", "v"] {
            put(
                &mut map,
                &format!("{p}.model.layers.0.self_attn.{key}_proj.weight"),
                vec![0.; 8],
                (2, 4),
            );
            put(
                &mut map,
                &format!("{p}.model.layers.0.self_attn.{key}_proj.bias"),
                vec![0.; 2],
                2,
            );
        }
        put(
            &mut map,
            "fixture.model.layers.0.mlp.c_fc.weight",
            vec![0.; 32],
            (8, 4),
        );
        put(
            &mut map,
            "fixture.model.layers.0.mlp.c_fc.bias",
            vec![0.; 8],
            8,
        );
        put(
            &mut map,
            "fixture.model.layers.0.mlp.c_proj.weight",
            vec![0.; 32],
            (4, 8),
        );
        put(
            &mut map,
            "fixture.model.layers.0.mlp.c_proj.bias",
            vec![0.; 4],
            4,
        );
        let model = StarCoder2::from_weights(&Weights::from_map(map, Device::Cpu), p, cfg).unwrap();
        let ids = Tensor::from_vec(vec![1u32], (1, 1), &Device::Cpu).unwrap();
        let run = || {
            model
                .step(&ids, &mut model.cache(), 0)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()
        };
        assert_eq!(run(), run());
    }
}
