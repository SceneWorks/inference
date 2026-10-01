//! The optional **PromptReasoner** (sc-3176) — the local gpt-oss `generate` path of
//! `_vendor/lens/reasoner.py::PromptReasoner`. It rewrites the user prompt into a richer
//! text-to-image prompt *before* encoding. **Off by default** (the pipeline's `enable_reasoner`); the
//! OpenAI-compatible-API path is host-agnostic and needs no MLX, so this story is only the local path.
//!
//! Turning the encoder-only gpt-oss into a **generating** model adds: the full 24-layer stack + final
//! `norm` + `lm_head`, an incremental KV-cache decode ([`GptOssDecoderLayer::forward_cached`] over a
//! per-layer [`KvCache`]), both **greedy** ([`LensReasonerModel::generate_greedy`]) and
//! **temperature-sampled** ([`LensReasonerModel::generate_sampled`], the vendor `PromptReasoner`'s
//! `temperature=0.7` decode — sc-9561 / F-105) decoding, the harmony `reasoning_effort="low"`
//! template ([`crate::text::LensTokenizer::encode_reasoner`]), and the harmony-channel output parse
//! ([`crate::text::clean_reasoner_output`]). The greedy path is retained as the KV-cache parity oracle.
//!
//! Both decodes run on the shared MLX engine's token-at-a-time loop
//! ([`generate_speculative`] with [`NoProposer`], epic sc-24432 E8) over this model
//! (`ReasonerTarget`), drawing every token through `ReasonerSampler` — the on-device argmax the
//! greedy path always took, or mlx-gen's seeded host [`sample_token`] — so a rewrite is
//! token-identical to the pre-engine loops.
//!
//! The MoE experts can be quantized (Q4/Q8, sc-3172) so the reasoner loads at the same `~12 GB` as the
//! encoder.

use std::cell::RefCell;

use mlx_rs::ops::indexing::{argmax, argmax_axis};
use mlx_rs::ops::{matmul, split_sections};
use mlx_rs::{Array, Dtype};

use mlx_gen::text_sample::{sample_token, SampleParams, SplitMix64};
use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen::{Error, Quant, Result};
use mlx_llm::core_llm::{HostSampleReason, SamplerPath};
use mlx_llm::decode::{
    generate_speculative, EngineOptions, FinishReason, GenerationConfig, LogitsScope,
    NoDraftRollback, NoProposer, Pipelining, SampledToken, SpeculativePrompt, SpeculativeTarget,
    TargetOutput, TokenSampler,
};
use mlx_llm::primitives::SamplingParams;

use crate::config::GptOssConfig;
use crate::text::{LensTokenizer, HARMONY_RETURN};
use crate::text_encoder::gpt_oss::{attention_mask, GptOssDecoderLayer, KvCache};

/// Generation default from the vendor `PromptReasoner.__init__` (`max_new_tokens`).
pub const DEFAULT_MAX_NEW_TOKENS: usize = 4096;

/// The generating gpt-oss-20b model: the full decoder stack + final norm + LM head (the
/// encoder-only [`crate::text_encoder::encoder::LensTextEncoder`] truncates the stack and drops these).
pub struct LensReasonerModel {
    embed_tokens: Array, // [vocab, hidden]
    layers: Vec<GptOssDecoderLayer>,
    final_norm: Array, // [hidden]
    lm_head: Array,    // [vocab, hidden]
    inv_freq: Array,
    attn_scaling: f32,
    sliding_window: i32,
    cfg: GptOssConfig,
}

impl LensReasonerModel {
    /// Load the full generating model from the `text_encoder` weights at `dtype`. `quant` (Q4/Q8)
    /// quantizes the MoE experts per-layer (sc-3172) so the reasoner stays `~12 GB`.
    pub fn from_weights(
        w: &Weights,
        cfg: &GptOssConfig,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let embed_tokens = w.require("model.embed_tokens.weight")?.as_dtype(dtype)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(GptOssDecoderLayer::from_weights(
                w,
                &format!("model.layers.{i}"),
                cfg,
                dtype,
                quant,
            )?);
        }
        let (inv_freq, attn_scaling) = cfg.yarn_rope();
        Ok(Self {
            embed_tokens,
            layers,
            final_norm: w.require("model.norm.weight")?.as_dtype(dtype)?,
            lm_head: w.require("lm_head.weight")?.as_dtype(dtype)?,
            inv_freq: Array::from_slice(&inv_freq, &[inv_freq.len() as i32]),
            attn_scaling,
            sliding_window: cfg.sliding_window,
            cfg: *cfg,
        })
    }

    /// Run all layers over `hidden` `[1, T, hidden]` with the per-layer caches. `prefill` ⇒ build the
    /// per-layer causal(+sliding) mask for the `T` prompt tokens; otherwise (`T == 1` decode) every
    /// cached key is valid (`mask = None`).
    fn run_layers(
        &self,
        mut hidden: Array,
        caches: &mut [KvCache],
        position: i32,
        prefill: bool,
    ) -> Result<Array> {
        let l = hidden.shape()[1];
        for (i, layer) in self.layers.iter().enumerate() {
            let sliding = if self.cfg.is_sliding(i) {
                Some(self.sliding_window)
            } else {
                None
            };
            let mask = if prefill {
                Some(attention_mask(l, sliding, hidden.dtype())?)
            } else {
                None
            };
            hidden = layer.forward_cached(
                &hidden,
                &self.inv_freq,
                self.attn_scaling,
                position,
                &mut caches[i],
                sliding,
                mask.as_ref(),
            )?;
        }
        Ok(hidden)
    }

    /// Teacher-forced per-position next-token argmax over the whole `input_ids` in **one** prefill
    /// forward (no decode loop): returns `pred[i] = argmax(logits at position i)`, the model's greedy
    /// next-token prediction given the true prefix `input_ids[..=i]`. Used to (a) compare to torch
    /// greedy at every position and (b) prove the incremental KV-cache decode is bit-equivalent to a
    /// full recompute.
    pub fn next_token_argmax(&self, input_ids: &[i32]) -> Result<Vec<i32>> {
        let mut caches: Vec<KvCache> = (0..self.cfg.num_layers).map(|_| KvCache::new()).collect();
        let prompt = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
        let hidden = self.embed_tokens.take_axis(&prompt, 0)?;
        let hidden = self.run_layers(hidden, &mut caches, 0, true)?;
        let normed = mlx_rs::fast::rms_norm(&hidden, &self.final_norm, self.cfg.rms_eps)?;
        let logits = matmul(&normed, self.lm_head.t())?; // [1, L, vocab]
        let pred = argmax_axis(&logits, 2, None)?.reshape(&[input_ids.len() as i32])?; // [L]
        Ok(pred.as_slice::<u32>().iter().map(|&i| i as i32).collect())
    }

    /// **Greedy** autoregressive generation (the parity path): prefill `input_ids`, then decode until
    /// the harmony `<|return|>` stop or `max_new_tokens`. Returns the **new** tokens (including the
    /// trailing stop, which [`clean_reasoner_output`](crate::text::clean_reasoner_output) strips) —
    /// mirroring the vendor `out_ids[0, input_len:]`. At least one token is always drawn (the
    /// prefill's), whatever the budget; `cancel` is checked before every later draw.
    pub fn generate_greedy(
        &self,
        input_ids: &[i32],
        max_new_tokens: usize,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        self.decode(
            input_ids,
            max_new_tokens,
            HARMONY_RETURN,
            ReasonerDraw::Greedy,
            cancel,
        )
    }

    /// Host `[vocab]` logits of a `[1, 1, vocab]` (or `[vocab]`) logits row, pulled to `Vec<f32>`
    /// for the *sampled* draw. The greedy draw keeps its on-device `argmax` (the KV-cache parity
    /// oracle) and is deliberately left untouched.
    fn host_logits(logits: &Array) -> Result<Vec<f32>> {
        let vocab = *logits.shape().last().expect("logits rank >= 1");
        Ok(logits
            .reshape(&[vocab])?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()
            .to_vec())
    }

    /// **Temperature-sampled** autoregressive generation (F-105): the same prefill + KV-cache decode as
    /// [`generate_greedy`](Self::generate_greedy), but each token is drawn from the shared seeded
    /// host-side sampler ([`mlx_gen::text_sample`]) rather than argmax — matching the vendor
    /// `PromptReasoner`'s `temperature`-sampled decode. Deterministic given `seed`. Returns the **new**
    /// tokens (including the trailing `<|return|>` stop). The greedy path is retained: it is the
    /// KV-cache parity oracle and the `temperature <= 0` route.
    pub fn generate_sampled(
        &self,
        input_ids: &[i32],
        max_new_tokens: usize,
        params: &SampleParams,
        seed: u64,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        self.decode(
            input_ids,
            max_new_tokens,
            HARMONY_RETURN,
            ReasonerDraw::Sampled {
                params,
                rng: SplitMix64::new(seed),
            },
            cancel,
        )
    }

    /// The reasoner decode on the shared engine: the engine prefills `input_ids` through
    /// `ReasonerTarget` and runs its token-at-a-time loop to `stop` (the harmony `<|return|>`) or the
    /// budget, every draw `draw`'s. The pre-engine loop always drew the prefill's token, so the budget
    /// is at least one; it kept the final stop in its list, which the engine does not emit, so a
    /// stop-token end appends it.
    fn decode(
        &self,
        input_ids: &[i32],
        max_new_tokens: usize,
        stop: i32,
        draw: ReasonerDraw<'_>,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        let target = ReasonerTarget {
            model: self,
            failure: RefCell::new(None),
        };
        let mut sampler = ReasonerSampler::new(draw, cancel);
        let generation = GenerationConfig {
            max_new_tokens: max_new_tokens.max(1),
            sampling: sampler.params,
            seed: None,
            stop_tokens: vec![stop],
        };
        let run = generate_speculative(
            &target,
            &mut NoProposer,
            SpeculativePrompt::Tokens(input_ids),
            &generation,
            0,
            &mlx_llm::CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                sampler: Some(&mut sampler),
                // Every draw is read back on the host before the next step; nothing pipelines.
                pipelining: Pipelining::Off,
                ..EngineOptions::default()
            },
        );
        // A model or draw failure keeps its own typed error (a cancel stays `Canceled`).
        if let Some(error) = target.failure.take().or_else(|| sampler.failure.take()) {
            return Err(error);
        }
        let run = run.map_err(|e| Error::Msg(format!("lens reasoner decode: {e}")))?;
        let mut out = run.output.tokens;
        if run.output.finish_reason == FinishReason::StopToken {
            out.extend(sampler.last);
        }
        Ok(out)
    }
}

/// The reasoner's KV cache for the engine: the per-layer caches plus the positions fed so far (a
/// sliding layer evicts, so no single layer's length is the position).
struct ReasonerCache {
    layers: Vec<KvCache>,
    len: i32,
}

/// [`LensReasonerModel`] as a target of the shared MLX engine. A forward from an empty cache is the
/// prompt prefill (the per-layer causal + sliding masks); every later forward is one decode token.
/// It serves the token-at-a-time loop only: it projects the last position, and a sliding layer's
/// eviction cannot be rolled back ([`NoDraftRollback`]).
struct ReasonerTarget<'a> {
    model: &'a LensReasonerModel,
    /// The model's own error from a failed forward, returned as-is by the decode.
    failure: RefCell<Option<Error>>,
}

impl ReasonerTarget<'_> {
    fn step(&self, cache: &mut ReasonerCache, ids: &Array, position: i32) -> Result<Array> {
        let m = self.model;
        let hidden = m.embed_tokens.take_axis(ids, 0)?; // [1, T, hidden]
        let hidden = m.run_layers(hidden, &mut cache.layers, position, cache.len == 0)?;
        cache.len += ids.shape()[1];
        let t = hidden.shape()[1];
        let last = if t > 1 {
            split_sections(&hidden, &[t - 1], 1)?[1].clone() // [1, 1, hidden]
        } else {
            hidden
        };
        let normed = mlx_rs::fast::rms_norm(&last, &m.final_norm, m.cfg.rms_eps)?;
        Ok(matmul(&normed, m.lm_head.t())?) // [1, 1, vocab]
    }
}

impl SpeculativeTarget for ReasonerTarget<'_> {
    type Cache = ReasonerCache;
    type Rollback = NoDraftRollback;

    fn new_cache(&self) -> ReasonerCache {
        ReasonerCache {
            layers: (0..self.model.cfg.num_layers)
                .map(|_| KvCache::new())
                .collect(),
            len: 0,
        }
    }

    fn cache_len(&self, cache: &ReasonerCache) -> i32 {
        cache.len
    }

    fn rollback(&self, _: usize) -> NoDraftRollback {
        NoDraftRollback
    }

    fn forward(
        &self,
        cache: &mut ReasonerCache,
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> mlx_llm::Result<TargetOutput> {
        if want_hidden || scope == LogitsScope::All {
            return Err(mlx_llm::Error::Unsupported(
                "the Lens reasoner target returns last-position logits only".into(),
            ));
        }
        match self.step(cache, ids, rope_offset) {
            Ok(logits) => Ok(TargetOutput {
                logits,
                hidden: None,
            }),
            Err(error) => {
                let message = error.to_string();
                *self.failure.borrow_mut() = Some(error);
                Err(mlx_llm::Error::Msg(message))
            }
        }
    }

    fn attention_label(&self) -> &'static str {
        // K/V heads repeated to the query heads before attention (`GptOssAttention`).
        "expanded"
    }
}

/// How the reasoner draws: the greedy on-device argmax, or mlx-gen's seeded host sampler.
enum ReasonerDraw<'a> {
    Greedy,
    Sampled {
        params: &'a SampleParams,
        rng: SplitMix64,
    },
}

/// The reasoner's draw on the engine's sampler seam — exactly the pre-engine loops' draws. Greedy is
/// the on-device `argmax` of the logits row with one index read back (ties to the lowest index, the
/// MLX `argmax` ≡ `torch.argmax` rule the parity oracle pins); sampled is mlx-gen's
/// [`sample_token`] over the f32 host row and the running prompt + generated history, from the
/// pipeline's own seeded [`SplitMix64`]. The pre-engine loops checked `cancel` before every token
/// after the first, which is where this draw checks it.
struct ReasonerSampler<'a> {
    draw: ReasonerDraw<'a>,
    cancel: Option<&'a CancelFlag>,
    /// The same knobs in the engine's vocabulary (reported, never drawn from).
    params: SamplingParams,
    draws: u64,
    /// The last token drawn — on a stop-token end, the stop the engine does not emit.
    last: Option<i32>,
    /// A draw's own error (a cancel), returned as-is by the decode.
    failure: Option<Error>,
}

impl<'a> ReasonerSampler<'a> {
    fn new(draw: ReasonerDraw<'a>, cancel: Option<&'a CancelFlag>) -> Self {
        let params = match &draw {
            ReasonerDraw::Greedy => SamplingParams::default(),
            ReasonerDraw::Sampled { params, .. } => SamplingParams {
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: usize::try_from(params.top_k).unwrap_or(0),
                presence_penalty: 0.0,
                repetition_penalty: params.repetition_penalty.unwrap_or(1.0),
                repetition_context: params.repetition_context,
            },
        };
        Self {
            draw,
            cancel,
            params,
            draws: 0,
            last: None,
            failure: None,
        }
    }

    fn draw_token(&mut self, logits: &Array, history: &[i32]) -> Result<i32> {
        if self.draws > 0 && self.cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let token = match &mut self.draw {
            ReasonerDraw::Greedy => {
                let vocab = *logits.shape().last().expect("logits rank >= 1");
                argmax(&logits.reshape(&[vocab])?, None)?.item::<u32>() as i32
            }
            ReasonerDraw::Sampled { params, rng } => {
                let host = LensReasonerModel::host_logits(logits)?;
                sample_token(&host, history, params, rng)
            }
        };
        self.draws += 1;
        self.last = Some(token);
        Ok(token)
    }
}

impl TokenSampler for ReasonerSampler<'_> {
    fn params(&self) -> &SamplingParams {
        &self.params
    }

    fn sample(
        &mut self,
        logits: &Array,
        history: &[i32],
        allowed: Option<&[bool]>,
    ) -> mlx_llm::Result<SampledToken> {
        if allowed.is_some() {
            return Err(mlx_llm::Error::Unsupported(
                "the Lens reasoner sampler takes no constraint mask".into(),
            ));
        }
        match self.draw_token(logits, history) {
            Ok(token) => Ok(SampledToken::Host(token)),
            Err(error) => {
                let message = error.to_string();
                self.failure = Some(error);
                Err(mlx_llm::Error::Msg(message))
            }
        }
    }

    fn argmax_rows(&mut self, _: &Array) -> mlx_llm::Result<Vec<i32>> {
        Err(mlx_llm::Error::Unsupported(
            "the Lens reasoner decodes token-at-a-time; it never verifies drafts".into(),
        ))
    }

    fn distribution(
        &mut self,
        _: &Array,
        _: &[i32],
        _: Option<&[bool]>,
    ) -> mlx_llm::Result<Vec<(i32, f32)>> {
        Err(mlx_llm::Error::Unsupported(
            "the Lens reasoner decodes token-at-a-time; it never verifies drafts".into(),
        ))
    }

    fn uniform(&mut self) -> f32 {
        match &mut self.draw {
            ReasonerDraw::Sampled { rng, .. } => rng.next_f32(),
            ReasonerDraw::Greedy => 0.0,
        }
    }

    fn path(&self) -> Option<SamplerPath> {
        match (&self.draw, self.draws) {
            (_, 0) => None,
            (ReasonerDraw::Greedy, _) => Some(SamplerPath::Device),
            (ReasonerDraw::Sampled { .. }, _) => {
                Some(SamplerPath::Host(HostSampleReason::Reference))
            }
        }
    }

    fn reads_history(&self) -> bool {
        matches!(self.draw, ReasonerDraw::Sampled { .. })
    }
}

/// The local PromptReasoner: the generating model + the tokenizer (harmony reasoner template +
/// output parse). The vendor default `enable=False` is the caller's concern; this is the `enable=True`
/// local-`generate` path.
pub struct LensReasoner {
    model: LensReasonerModel,
    tokenizer: LensTokenizer,
}

impl LensReasoner {
    /// Load from a Lens snapshot dir (`text_encoder/` + `tokenizer/tokenizer.json`). `quant` keeps the
    /// reasoner at `~12 GB`.
    pub fn load(
        snapshot_dir: impl AsRef<std::path::Path>,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let root = snapshot_dir.as_ref();
        let tokenizer = LensTokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))?;
        let w = Weights::from_dir(root.join("text_encoder"))?;
        let model = LensReasonerModel::from_weights(&w, &GptOssConfig::lens(), dtype, quant)?;
        // Materialize at load (sc-24245; see `mlx_gen_qwen_image::loader::load_transformer_with`).
        w.materialize_accessed()?;
        Ok(Self { model, tokenizer })
    }

    /// Refine one prompt via the local gpt-oss. `date` fills the harmony preamble (`Current date:`).
    /// `temperature` selects the decode: `<= 0` ⇒ deterministic greedy (also the KV-cache parity
    /// oracle); `> 0` ⇒ temperature sampling seeded by `seed`, matching the vendor `PromptReasoner`'s
    /// stochastic decode (F-105). Returns the cleaned final-channel rewrite, or the original `prompt`
    /// when the reasoner produced no usable final text (the vendor `clean_text_out or prompt`).
    pub fn refine(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        date: &str,
        temperature: f32,
        seed: u64,
        cancel: Option<&CancelFlag>,
    ) -> Result<String> {
        let input_ids = self.tokenizer.encode_reasoner(prompt, date)?;
        if input_ids.is_empty() {
            return Err(Error::Msg("lens reasoner: empty tokenization".into()));
        }
        let new_tokens = if temperature <= 0.0 {
            self.model
                .generate_greedy(&input_ids, max_new_tokens, cancel)?
        } else {
            self.model.generate_sampled(
                &input_ids,
                max_new_tokens,
                &SampleParams::temperature(temperature),
                seed,
                cancel,
            )?
        };
        let raw = self.tokenizer.decode(&new_tokens)?;
        let cleaned = crate::text::clean_reasoner_output(&raw);
        Ok(if cleaned.is_empty() {
            prompt.to_string()
        } else {
            cleaned
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fill in `[-scale, scale]`.
    fn fill(n: usize, seed: f32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.731).sin() * scale)
            .collect()
    }

    /// A tiny two-layer gpt-oss (layer 0 sliding with a window the decode crosses, layer 1 full),
    /// MXFP4 experts dequantized dense, vocabulary 48.
    fn tiny_reasoner() -> LensReasonerModel {
        let cfg = GptOssConfig {
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate: 32,
            num_experts: 2,
            experts_per_tok: 1,
            rms_eps: 1e-5,
            sliding_window: 4,
            ..GptOssConfig::lens()
        };
        let (h, vocab, e, i) = (32usize, 48usize, 2usize, 32usize);
        let (q, kv) = (4 * 8, 2 * 8);
        let mut w = Weights::empty();
        let mut seed = 0.0f32;
        let mut put = |w: &mut Weights, key: String, shape: &[i32], scale: f32| {
            seed += 1.0;
            let n = shape.iter().product::<i32>() as usize;
            w.insert(key, Array::from_slice(&fill(n, seed, scale), shape));
        };
        put(
            &mut w,
            "model.embed_tokens.weight".into(),
            &[vocab as i32, h as i32],
            1.0,
        );
        put(&mut w, "model.norm.weight".into(), &[h as i32], 1.0);
        put(
            &mut w,
            "lm_head.weight".into(),
            &[vocab as i32, h as i32],
            1.0,
        );
        for layer in 0..2 {
            let p = format!("model.layers.{layer}");
            for (name, out, inp) in [("q", q, h), ("k", kv, h), ("v", kv, h), ("o", h, q)] {
                put(
                    &mut w,
                    format!("{p}.self_attn.{name}_proj.weight"),
                    &[out as i32, inp as i32],
                    0.3,
                );
                put(
                    &mut w,
                    format!("{p}.self_attn.{name}_proj.bias"),
                    &[out as i32],
                    0.1,
                );
            }
            put(&mut w, format!("{p}.self_attn.sinks"), &[4], 0.5);
            put(
                &mut w,
                format!("{p}.input_layernorm.weight"),
                &[h as i32],
                1.0,
            );
            put(
                &mut w,
                format!("{p}.post_attention_layernorm.weight"),
                &[h as i32],
                1.0,
            );
            put(
                &mut w,
                format!("{p}.mlp.router.weight"),
                &[e as i32, h as i32],
                1.0,
            );
            put(&mut w, format!("{p}.mlp.router.bias"), &[e as i32], 0.1);
            for (name, rows, contract) in [("gate_up", 2 * i, h), ("down", h, i)] {
                let groups = contract / 32;
                let bytes: Vec<u8> = (0..e * rows * groups * 16)
                    .map(|k| ((k * 37 + layer * 11) % 256) as u8)
                    .collect();
                w.insert(
                    format!("{p}.mlp.experts.{name}_proj_blocks"),
                    Array::from_slice(&bytes, &[e as i32, rows as i32, groups as i32, 16]),
                );
                // e8m0 124 = 2^-3 keeps the FP4 grid (|x| <= 6) small.
                w.insert(
                    format!("{p}.mlp.experts.{name}_proj_scales"),
                    Array::from_slice(
                        &vec![124u8; e * rows * groups],
                        &[e as i32, rows as i32, groups as i32],
                    ),
                );
                put(
                    &mut w,
                    format!("{p}.mlp.experts.{name}_proj_bias"),
                    &[e as i32, rows as i32],
                    0.1,
                );
            }
        }
        LensReasonerModel::from_weights(&w, &cfg, Dtype::Float32, None).unwrap()
    }

    /// The pre-engine greedy argmax helper, verbatim.
    fn argmax_token(m: &LensReasonerModel, hidden: &Array) -> Result<i32> {
        let t = hidden.shape()[1];
        let last = if t > 1 {
            split_sections(hidden, &[t - 1], 1)?[1].clone() // [1, 1, hidden]
        } else {
            hidden.clone()
        };
        let normed = mlx_rs::fast::rms_norm(&last, &m.final_norm, m.cfg.rms_eps)?;
        let logits = matmul(&normed, m.lm_head.t())?; // [1, 1, vocab]
        let vocab = logits.shape()[2];
        let idx = argmax(&logits.reshape(&[vocab])?, None)?;
        Ok(idx.item::<u32>() as i32)
    }

    /// The pre-engine host-logits helper, verbatim.
    fn last_logits_host(m: &LensReasonerModel, hidden: &Array) -> Result<Vec<f32>> {
        let t = hidden.shape()[1];
        let last = if t > 1 {
            split_sections(hidden, &[t - 1], 1)?[1].clone() // [1, 1, hidden]
        } else {
            hidden.clone()
        };
        let normed = mlx_rs::fast::rms_norm(&last, &m.final_norm, m.cfg.rms_eps)?;
        let logits = matmul(&normed, m.lm_head.t())?; // [1, 1, vocab]
        let vocab = logits.shape()[2];
        Ok(logits
            .reshape(&[vocab])?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()
            .to_vec())
    }

    /// The pre-engine greedy loop, verbatim but for its stop (the `HARMONY_RETURN` constant, a
    /// parameter here so a tiny vocabulary can end on a stop token): the E1/E8 oracle.
    fn reference_greedy(
        m: &LensReasonerModel,
        input_ids: &[i32],
        max_new_tokens: usize,
        stop: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        let mut caches: Vec<KvCache> = (0..m.cfg.num_layers).map(|_| KvCache::new()).collect();
        let prompt = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
        let hidden = m.embed_tokens.take_axis(&prompt, 0)?;
        let hidden = m.run_layers(hidden, &mut caches, 0, true)?;
        mlx_rs::transforms::eval([&hidden])?;
        let mut next = argmax_token(m, &hidden)?;
        let mut position = input_ids.len() as i32;
        let mut out = vec![next];
        while out.len() < max_new_tokens && next != stop {
            if let Some(c) = cancel {
                if c.is_cancelled() {
                    return Err(Error::Canceled);
                }
            }
            let tok = Array::from_slice(&[next], &[1, 1]);
            let h = m.embed_tokens.take_axis(&tok, 0)?;
            let h = m.run_layers(h, &mut caches, position, false)?;
            position += 1;
            next = argmax_token(m, &h)?;
            out.push(next);
        }
        Ok(out)
    }

    /// The pre-engine sampled loop, verbatim but for its stop (as [`reference_greedy`]).
    fn reference_sampled(
        m: &LensReasonerModel,
        input_ids: &[i32],
        max_new_tokens: usize,
        stop: i32,
        params: &SampleParams,
        seed: u64,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        let mut caches: Vec<KvCache> = (0..m.cfg.num_layers).map(|_| KvCache::new()).collect();
        let prompt = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
        let hidden = m.embed_tokens.take_axis(&prompt, 0)?;
        let hidden = m.run_layers(hidden, &mut caches, 0, true)?;
        let mut history: Vec<i32> = input_ids.to_vec();
        let mut rng = SplitMix64::new(seed);
        let logits = last_logits_host(m, &hidden)?;
        let mut next = sample_token(&logits, &history, params, &mut rng);
        history.push(next);
        let mut position = input_ids.len() as i32;
        let mut out = vec![next];
        while out.len() < max_new_tokens && next != stop {
            if let Some(c) = cancel {
                if c.is_cancelled() {
                    return Err(Error::Canceled);
                }
            }
            let tok = Array::from_slice(&[next], &[1, 1]);
            let h = m.embed_tokens.take_axis(&tok, 0)?;
            let h = m.run_layers(h, &mut caches, position, false)?;
            position += 1;
            let logits = last_logits_host(m, &h)?;
            next = sample_token(&logits, &history, params, &mut rng);
            history.push(next);
            out.push(next);
        }
        Ok(out)
    }

    fn engine(
        m: &LensReasonerModel,
        prompt: &[i32],
        max: usize,
        stop: i32,
        sampling: Option<(&SampleParams, u64)>,
    ) -> Vec<i32> {
        let draw = match sampling {
            None => ReasonerDraw::Greedy,
            Some((params, seed)) => ReasonerDraw::Sampled {
                params,
                rng: SplitMix64::new(seed),
            },
        };
        m.decode(prompt, max, stop, draw, None).unwrap()
    }

    /// E8 (sc-24446): both reasoner decodes run on the shared engine and are token-identical to the
    /// pre-engine loops — greedy, and seeded temperature / penalized / top-k + nucleus draws over
    /// several seeds — across a stop-token end (a stop the run itself emits), a budget end, the
    /// zero budget (the prefill's token is still drawn) and the public `HARMONY_RETURN` entry points.
    #[test]
    fn reasoner_engine_decode_matches_the_pre_engine_loops() {
        let m = tiny_reasoner();
        let prompt = [3, 17, 42, 8, 30, 11, 5];
        let (mut stop_ends, mut budget_ends) = (0, 0);
        let mut check = |want: Vec<i32>, got: Vec<i32>, max: usize, stop: i32, what: &str| {
            assert_eq!(got, want, "{what} max {max} stop {stop}");
            if want.last() == Some(&stop) {
                stop_ends += 1;
            } else {
                budget_ends += 1;
            }
        };
        let free = reference_greedy(&m, &prompt, 24, -1, None).unwrap();
        for (max, stop) in [(24, -1), (24, free[5]), (0, -1), (1, -1)] {
            let want = reference_greedy(&m, &prompt, max, stop, None).unwrap();
            check(
                want,
                engine(&m, &prompt, max, stop, None),
                max,
                stop,
                "greedy",
            );
        }
        let samplers = [
            SampleParams::temperature(0.7),
            SampleParams::censored(1.0),
            SampleParams {
                temperature: 0.9,
                top_k: 6,
                top_p: 0.8,
                repetition_penalty: Some(1.1),
                repetition_context: 8,
            },
        ];
        for params in &samplers {
            for seed in 0..4u64 {
                let free = reference_sampled(&m, &prompt, 24, -1, params, seed, None).unwrap();
                for (max, stop) in [(24, -1), (24, free[4])] {
                    let want =
                        reference_sampled(&m, &prompt, max, stop, params, seed, None).unwrap();
                    let got = engine(&m, &prompt, max, stop, Some((params, seed)));
                    check(want, got, max, stop, &format!("{params:?} seed {seed}"));
                }
            }
        }
        assert!(
            stop_ends > 0 && budget_ends > 0,
            "stop {stop_ends}, budget {budget_ends}"
        );
        // The public entry points stop on `HARMONY_RETURN` (out of this vocabulary: a budget end).
        assert_eq!(
            m.generate_greedy(&prompt, 12, None).unwrap(),
            reference_greedy(&m, &prompt, 12, HARMONY_RETURN, None).unwrap()
        );
        let params = SampleParams::temperature(0.7);
        assert_eq!(
            m.generate_sampled(&prompt, 12, &params, 9, None).unwrap(),
            reference_sampled(&m, &prompt, 12, HARMONY_RETURN, &params, 9, None).unwrap()
        );
    }

    /// A cancel stays typed and lands where the pre-engine loop checked it: never before the
    /// prefill's token, before every later one.
    #[test]
    fn reasoner_cancel_is_checked_before_every_token_after_the_first() {
        let m = tiny_reasoner();
        let prompt = [3, 17, 42, 8];
        let cancel = CancelFlag::new();
        cancel.cancel();
        for max in [0, 1] {
            let want = reference_greedy(&m, &prompt, max, -1, Some(&cancel)).unwrap();
            assert_eq!(want.len(), 1);
            assert_eq!(
                m.generate_greedy(&prompt, max, Some(&cancel)).unwrap(),
                want
            );
        }
        assert!(matches!(
            reference_greedy(&m, &prompt, 4, -1, Some(&cancel)),
            Err(Error::Canceled)
        ));
        assert!(matches!(
            m.generate_greedy(&prompt, 4, Some(&cancel)),
            Err(Error::Canceled)
        ));
        let params = SampleParams::temperature(0.7);
        assert!(matches!(
            reference_sampled(&m, &prompt, 4, -1, &params, 1, Some(&cancel)),
            Err(Error::Canceled)
        ));
        assert!(matches!(
            m.generate_sampled(&prompt, 4, &params, 1, Some(&cancel)),
            Err(Error::Canceled)
        ));
        assert_eq!(
            m.generate_sampled(&prompt, 1, &params, 1, Some(&cancel))
                .unwrap(),
            reference_sampled(&m, &prompt, 1, -1, &params, 1, Some(&cancel)).unwrap()
        );
    }
}
