//! Native StarCoder2 decoder used only by StarVector-8B.
//!
//! This deliberately stays separate from the GPTBigCode decoder used by the 1B model: StarCoder2
//! uses GQA, RoPE, biasful LayerNorm, and a biasful GELU MLP.
//!
//! It implements both decode seams (story sc-24138): the reference [`Decode`] over a
//! [`ContiguousKvCache`] (kept as the parity oracle) and [`StepModel`] over the shared
//! [`StepKvCache`] — preallocated by default. Every path attends the un-expanded K/V through
//! [`sdpa_gqa_causal`] by default, so the reference loop, the growing backing and the static cache
//! are the same arithmetic; [`StarCoder2::set_attn_formulation`] selects the pre-migration
//! `repeat_kv`-expanded arithmetic ([`AttnFormulation::Expanded`]) on the reference paths and the
//! growing backing, as a labelled comparison.

use candle_core::{DType, Device, Tensor};

use crate::decode::step::{LogitsScope, StepModel, StepOutput, StepRequest};
use crate::decode::Decode;
use crate::error::{Error, Result};
use crate::primitives::attention::{repeat_kv, sdpa_causal, sdpa_gqa_causal, AttnFormulation};
use crate::primitives::decode_cache::DecodeCache;
use crate::primitives::kv_cache::{KvCache, KvCacheKind};
use crate::primitives::nn::{embed, gelu, layer_norm, linear};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::step_kv_cache::{KvLayout, LayerKvShape, StepKvCache};
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
    /// The published config records the 49,152-token base vocabulary, but the checkpoint adds five
    /// StarVector tokens and resizes its tied embedding/head to 49,157 rows.
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
    /// Which KV cache [`StepModel::new_cache_for`] builds (static by default).
    step_kv_cache: KvCacheKind,
    /// How the reference paths and a growing step cache attend: [`AttnFormulation::Gqa`] (the
    /// default — the static cache's arithmetic) or [`AttnFormulation::Expanded`] (the
    /// pre-migration `repeat_kv` + `sdpa`, selected for a comparison). The static cache attends
    /// un-expanded regardless.
    attn_formulation: AttnFormulation,
}

impl StarCoder2 {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: StarCoder2Config) -> Result<Self> {
        let dtype = crate::device::compute_dtype(w.device());
        let req = |key: String| -> Result<Tensor> { Ok(w.require(&key)?.to_dtype(dtype)?) };
        let key = |suffix: &str| join(prefix, suffix);
        let embed_tokens = req(key("model.embed_tokens.weight"))?;
        let expected_embedding_shape = [cfg.vocab_size, cfg.hidden_size];
        if embed_tokens.dims() != expected_embedding_shape {
            return Err(crate::error::Error::Config(format!(
                "StarCoder2 tied embedding/head must be {expected_embedding_shape:?}, got {:?}",
                embed_tokens.dims()
            )));
        }
        let layers = (0..cfg.layers)
            .map(|index| {
                StarCoder2Layer::from_weights(w, &key(&format!("model.layers.{index}")), cfg, dtype)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens,
            layers,
            final_norm_weight: req(key("model.norm.weight"))?,
            final_norm_bias: req(key("model.norm.bias"))?,
            cfg,
            dtype,
            device: w.device().clone(),
            step_kv_cache: KvCacheKind::Static,
            attn_formulation: AttnFormulation::Gqa,
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

    /// Last-position logits `[batch, vocab]` over `embeds` at `offset` — the reference forward,
    /// in the selected formulation ([`AttnFormulation::Gqa`] by default; see
    /// [`set_attn_formulation`](Self::set_attn_formulation)).
    pub fn logits_from_embeds(
        &self,
        embeds: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Tensor> {
        let state = self.normed_states(embeds, cache, offset, self.attn_formulation)?;
        let sequence = state.dim(1)?;
        let last = state.narrow(1, sequence - 1, 1)?.squeeze(1)?;
        linear(&last, &self.embed_tokens, None)
    }

    /// The decoder stack over `embeds` at `offset`, final-LayerNormed: `[batch, seq, hidden]`.
    fn normed_states(
        &self,
        embeds: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
        formulation: AttnFormulation,
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
            state = layer.forward(&state, &cos, &sin, cache, index, formulation)?;
        }
        layer_norm(
            &state,
            &self.final_norm_weight,
            &self.final_norm_bias,
            self.cfg.layer_norm_eps,
        )
    }

    /// The reference cache (the growing concat the [`Decode`] loop runs on).
    pub fn cache(&self) -> ContiguousKvCache {
        ContiguousKvCache::new(self.layers.len())
    }

    /// The per-layer KV geometry (uniform GQA): what the step seam preallocates and prices.
    pub fn kv_layout(&self) -> KvLayout {
        let shape = LayerKvShape {
            kv_heads: self.cfg.kv_heads,
            key_dim: self.cfg.head_dim(),
            value_dim: self.cfg.head_dim(),
            device: self.device.clone(),
        };
        KvLayout {
            layers: vec![Some(shape); self.layers.len()],
            dtype: self.dtype,
        }
    }

    /// Bytes [`new_static_cache`](Self::new_static_cache) preallocates for `capacity` positions.
    pub fn static_kv_bytes(&self, capacity: usize) -> usize {
        self.kv_layout().static_bytes(capacity)
    }

    /// A step-seam cache preallocated for `capacity` positions (`0` is [`Error::Msg`]).
    pub fn new_static_cache(&self, capacity: usize) -> Result<StepKvCache> {
        StepKvCache::preallocated(&self.kv_layout(), capacity)
    }

    /// A step-seam cache on the growing backing.
    pub fn new_step_cache(&self) -> StepKvCache {
        StepKvCache::growing(&self.kv_layout())
    }

    /// Select which KV cache [`StepModel::new_cache_for`] builds.
    pub fn set_step_kv_cache(&mut self, kind: KvCacheKind) {
        self.step_kv_cache = kind;
    }

    /// Select how the reference paths and a growing step cache attend:
    /// [`AttnFormulation::Gqa`] (the default) or [`AttnFormulation::Expanded`] (the pre-migration
    /// arithmetic, for a labelled comparison). The static cache attends un-expanded regardless.
    pub fn set_attn_formulation(&mut self, formulation: AttnFormulation) {
        self.attn_formulation = formulation;
    }

    /// The selected formulation (what the reference paths and a growing cache run).
    pub fn attn_formulation(&self) -> AttnFormulation {
        self.attn_formulation
    }

    /// Prefill `embeds` (the StarVector-8B image rows + `<svg` prompt) into a step-seam cache at
    /// its current length, in the cache's formulation; last-position logits `[batch, vocab]`.
    pub fn step_prefill_from_embeds(
        &self,
        embeds: &Tensor,
        cache: &mut StepKvCache,
    ) -> Result<Tensor> {
        let offset = DecodeCache::len(cache) + cache.rope_delta();
        let formulation = self.cache_formulation(cache);
        let state = self.normed_states(embeds, cache, offset, formulation)?;
        let sequence = state.dim(1)?;
        let last = state.narrow(1, sequence - 1, 1)?.squeeze(1)?;
        linear(&last, &self.embed_tokens, None)
    }

    /// Un-expanded attention on a static cache; the selected formulation on a growing one.
    fn cache_formulation(&self, cache: &StepKvCache) -> AttnFormulation {
        match cache.kv_kind() {
            KvCacheKind::Static => AttnFormulation::Gqa,
            KvCacheKind::Growing => self.attn_formulation,
        }
    }
}

impl StepModel for StarCoder2 {
    type Cache = StepKvCache;

    fn new_cache(&self) -> StepKvCache {
        self.new_step_cache()
    }

    /// The static cache for `capacity + overshoot` positions (or the growing backing when selected).
    fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<StepKvCache> {
        match self.step_kv_cache {
            KvCacheKind::Static => self.new_static_cache(capacity.saturating_add(overshoot)),
            KvCacheKind::Growing => Ok(self.new_step_cache()),
        }
    }

    fn attn_formulation(&self, cache: &StepKvCache) -> AttnFormulation {
        self.cache_formulation(cache)
    }

    /// Not replayable as a CUDA graph (story sc-24134): the step's RoPE offset is the cache's
    /// Rust-side length (plus its delta) and the KV lands at that host offset, so a graph would
    /// replay at the captured position (`positions_host_scalar`).
    fn graph_support(&self) -> std::result::Result<(), &'static str> {
        Err("positions_host_scalar")
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    fn forward_step(
        &self,
        cache: &mut StepKvCache,
        request: StepRequest<'_>,
    ) -> Result<StepOutput> {
        if request.is_empty()? {
            return Err(Error::Msg(
                "StarCoder2::forward_step: empty token slice".into(),
            ));
        }
        let embeds = self.embed(&request.tokens.ids(&self.device)?)?;
        let offset = DecodeCache::len(cache) + cache.rope_delta();
        let formulation = self.cache_formulation(cache);
        let state = self.normed_states(&embeds, cache, offset, formulation)?;
        let logits = match request.scope {
            LogitsScope::Last => {
                let sequence = state.dim(1)?;
                linear(
                    &state.narrow(1, sequence - 1, 1)?.squeeze(1)?,
                    &self.embed_tokens,
                    None,
                )?
            }
            // Row-wise through the same 2-D product the last-position path runs, so a one-token
            // verify step is bit-identical to a plain decode step.
            LogitsScope::All => {
                let (b, s, h) = state.dims3()?;
                linear(&state.reshape((b * s, h))?, &self.embed_tokens, None)?.reshape((
                    b,
                    s,
                    self.cfg.vocab_size,
                ))?
            }
        };
        Ok(StepOutput {
            logits,
            hidden: request.want_hidden.then_some(state),
        })
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
        formulation: AttnFormulation,
    ) -> Result<Tensor> {
        let normed = layer_norm(
            hidden,
            &self.input_norm_weight,
            &self.input_norm_bias,
            self.eps,
        )?;
        let attended = self
            .attn
            .forward(&normed, cos, sin, cache, index, formulation)?;
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
        formulation: AttnFormulation,
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
        // Head-major and contiguous, as llama's `project` hands them over: the growing cache returns
        // the first append's K/V unchanged, and `sdpa_gqa_causal` attends them directly — a bare
        // transpose of a multi-token GQA projection is a layout the CUDA matmul rejects (sc-24164).
        let q = apply_rope(&q, cos, sin, false)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = apply_rope(&k, cos, sin, false)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let (k, v) = cache.update(index, &k, &v)?;
        let scale = 1.0 / (dim as f32).sqrt();
        let out = match formulation {
            // The static cache's views, un-expanded (sc-24138).
            AttnFormulation::Gqa => sdpa_gqa_causal(&q, &k, &v, scale)?,
            // The reference arithmetic.
            AttnFormulation::Expanded => {
                let groups = self.cfg.heads / self.cfg.kv_heads;
                let k = repeat_kv(&k, groups)?;
                let v = repeat_kv(&v, groups)?;
                sdpa_causal(&q, &k, &v, scale)?
            }
        };
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
            vec![0.; 16],
            (4, 4),
        );
        let error =
            match StarCoder2::from_weights(&Weights::from_map(map, Device::Cpu), "fixture", cfg) {
                Ok(_) => panic!("mismatched tied embedding/head shape must fail closed"),
                Err(error) => error,
            };
        assert!(matches!(
            error,
            crate::error::Error::Config(message)
                if message.contains("must be [3, 4]") && message.contains("got [4, 4]")
        ));
    }

    /// A weight-file-free StarCoder2 on `device`: bounded, deterministic cos-pattern weights (a
    /// distinct phase per tensor) with unit LayerNorm scales.
    fn synthetic_decoder(cfg: StarCoder2Config, device: &Device) -> StarCoder2 {
        let mut phase = 0.0f64;
        let mut det = |dims: &[usize], scale: f64| -> Tensor {
            phase += 0.61;
            let n: usize = dims.iter().product();
            Tensor::arange(0f32, n as f32, device)
                .unwrap()
                .affine(0.0137, phase)
                .unwrap()
                .cos()
                .unwrap()
                .affine(scale, 0.0)
                .unwrap()
                .reshape(dims)
                .unwrap()
        };
        let ones = |n: usize| Tensor::ones(n, DType::F32, device).unwrap();
        let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let kv = cfg.kv_heads * cfg.head_dim();
        let p = "fixture";
        let mut map = HashMap::new();
        map.insert(
            format!("{p}.model.embed_tokens.weight"),
            det(&[cfg.vocab_size, hidden], 0.5),
        );
        map.insert(format!("{p}.model.norm.weight"), ones(hidden));
        map.insert(format!("{p}.model.norm.bias"), det(&[hidden], 0.02));
        for layer in 0..cfg.layers {
            let key = |suffix: &str| format!("{p}.model.layers.{layer}.{suffix}");
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                map.insert(key(&format!("{norm}.weight")), ones(hidden));
                map.insert(key(&format!("{norm}.bias")), det(&[hidden], 0.02));
            }
            for (proj, rows) in [
                ("q_proj", hidden),
                ("k_proj", kv),
                ("v_proj", kv),
                ("o_proj", hidden),
            ] {
                map.insert(
                    key(&format!("self_attn.{proj}.weight")),
                    det(&[rows, hidden], 0.2),
                );
                map.insert(key(&format!("self_attn.{proj}.bias")), det(&[rows], 0.02));
            }
            map.insert(key("mlp.c_fc.weight"), det(&[inter, hidden], 0.2));
            map.insert(key("mlp.c_fc.bias"), det(&[inter], 0.02));
            map.insert(key("mlp.c_proj.weight"), det(&[hidden, inter], 0.2));
            map.insert(key("mlp.c_proj.bias"), det(&[hidden], 0.02));
        }
        StarCoder2::from_weights(&Weights::from_map(map, device.clone()), p, cfg).unwrap()
    }

    /// sc-24164 regression (release gate 2, `runtime-2026.09.1-rc.1`): StarVector-8B's CUDA prefill
    /// failed with "matmul is only supported for contiguous tensors". StarCoder2 handed the growing
    /// cache bare head transposes of its GQA projections, the growing cache returns the first
    /// append unchanged, and `sdpa_gqa_causal`'s `Kᵀ` matmul rejected that layout. It takes more
    /// than one KV head and a multi-token prefill to produce it (a single KV head or a single token
    /// leaves the transpose dense), so a small synthetic decoder with 2 KV heads is prefilled with
    /// 11 positions through the growing step cache the 8B provider uses, and through the reference
    /// `Decode` path's growing cache. Both must run, copy no K/V (StarCoder2 hands the cache
    /// contiguous head-major tensors, so attention never needs its fallback copy), agree bit for
    /// bit, decode on from the prefill, and agree with the static cache's prefill.
    fn assert_multi_token_gqa_prefill_through_the_growing_cache(device: &Device) {
        use crate::primitives::kv_cache::kv_materialize_count;
        let cfg = StarCoder2Config {
            vocab_size: 48,
            hidden_size: 64,
            intermediate_size: 128,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            rope_theta: 1_000_000.0,
            layer_norm_eps: 1e-5,
        };
        let model = synthetic_decoder(cfg, device);
        let prompt = 11usize;
        // Stand-ins for the projected image rows + `<svg` prompt embeddings the provider joins.
        let embeds = Tensor::arange(0f32, (prompt * cfg.hidden_size) as f32, device)
            .unwrap()
            .affine(0.021, 0.3)
            .unwrap()
            .sin()
            .unwrap()
            .reshape((1, prompt, cfg.hidden_size))
            .unwrap();
        let host = |t: &Tensor| -> Vec<f32> {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };

        // The StarVector-8B provider's prefill: a fresh growing step cache.
        let mut growing = model.new_step_cache();
        let before = kv_materialize_count();
        let provider = model
            .step_prefill_from_embeds(&embeds, &mut growing)
            .expect("a multi-token GQA prefill through the growing cache must run");
        assert_eq!(
            kv_materialize_count(),
            before,
            "the first append attends StarCoder2's K/V as handed over: no copy"
        );
        assert_eq!(provider.dims(), &[1, cfg.vocab_size]);
        assert!(host(&provider).iter().all(|x| x.is_finite()));

        // The reference `Decode` path over the same growing concat: identical arithmetic.
        let mut reference_cache = model.cache();
        let reference = model
            .logits_from_embeds(&embeds, &mut reference_cache, 0)
            .expect("the reference prefill must run");
        assert_eq!(host(&provider), host(&reference));
        // ... and it decodes on from the prefilled cache (the `cat` path).
        let next = model
            .embed(&Tensor::new(&[[3u32]], device).unwrap())
            .unwrap();
        let step = model
            .logits_from_embeds(&next, &mut reference_cache, prompt as i32)
            .expect("a decode step after the prefill must run");
        assert!(host(&step).iter().all(|x| x.is_finite()));

        // The static cache (in-place contiguous writes, narrowed views) attends the same values.
        let mut fixed = model.new_static_cache(prompt + 4).unwrap();
        let fixed = model.step_prefill_from_embeds(&embeds, &mut fixed).unwrap();
        let (a, b) = (host(&provider), host(&fixed));
        let peak = a.iter().fold(0f32, |m, x| m.max(x.abs()));
        let diff = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        eprintln!(
            "[starcoder2 {:?}] growing vs static prefill: max|delta| = {diff} (max|logit| = \
             {peak}), bit-identical = {}",
            device,
            a == b
        );
        assert!(
            diff <= 2e-2 * peak.max(1.0),
            "growing vs static max|delta| = {diff}"
        );
    }

    /// The sc-24164 regression on the CPU lane: the CPU `gemm` reads any strides, so this pins the
    /// no-copy contract (fix (a)) rather than the matmul failure.
    #[test]
    fn multi_token_gqa_prefill_through_the_growing_cache_copies_no_kv_on_cpu() {
        assert_multi_token_gqa_prefill_through_the_growing_cache(&Device::Cpu);
    }

    /// The sc-24164 regression where it failed: on CUDA, where the matmul rejects the layout.
    #[cfg(feature = "cuda")]
    #[test]
    fn multi_token_gqa_prefill_through_the_growing_cache_runs_on_cuda() {
        let device = crate::device::new_cuda_for_test().expect("cuda device");
        assert_multi_token_gqa_prefill_through_the_growing_cache(&device);
    }
}
