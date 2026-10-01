//! The autoregressive text-generation runtime — the candle port of `mlx-gen-sensenova`'s
//! `runtime.rs` (the slice needed by the understanding surface: VQA + interleave).
//!
//! SenseNova-U1 is a *generating* LLM, so beyond a forward pass the understanding path needs an AR
//! runtime: a prefix is prefilled into a [`KvCache`](crate::qwen3::KvCache), then tokens are decoded
//! one at a time — each new token forwarded through the cached backbone at the next temporal
//! position to produce the logits for the token after it. This module ports the reference's pieces
//! (`modeling_neo_chat.py`):
//!
//! * budgeted single-token cached forwards that produce next-token logits; and
//! * greedy/sampled rollout to an EOS or token budget (the runtime under `chat` /
//!   `answer_question`).
//!
//! Positions: text tokens advance the temporal axis by one per token (`h = w = 0`), matching the
//! reference. The understanding path ([`Path::Und`]) drives text decode.
//!
//! Every text rollout — `generate_planned` and the interleave rollout's text segments (`t2i.rs`,
//! which alternates them with gen-path image generation) — is the shared Candle engine's
//! token-at-a-time loop ([`generate_with_sampler`], epic sc-24432 E8) over the understanding path
//! (`UndStep`), drawing every token through `UndPick` — the shared candle-llm sampler
//! ([`sample_host`]: the device argmax for greedy, the shared temperature / top-k / nucleus
//! reference for a stochastic draw) from the rollout's seeded [`SplitMix64`]. Greedy streams are
//! token-identical to the pre-engine loops (ties break to the lowest index on the CPU, as they
//! did). **Behaviour change (sc-24446):** a seeded *stochastic* stream is a valid draw from the
//! same shaped distribution but no longer the pre-engine crate-local sort-based sampler's exact
//! stream — the MLX twin (`mlx-gen-sensenova`) made the same move onto its shared sampler. The
//! pipeline's cancel is bridged onto the engine's flag, so a cancel ends the rollout as the
//! engine's typed `Cancelled` finish.

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::attention_budget::AttentionPlan;
use candle_gen::gen_core::CancelFlag;
use candle_gen::{CandleError, Result};
use candle_llm::decode::StreamEvent;
use candle_llm::decode::{
    generate_with_sampler, FinishReason, GenerationConfig, LogitsScope, SpeculativePrompt,
    StepModel, StepOutput, StepRequest, TokenSampler,
};
use candle_llm::primitives::sampler::sample_host;
use candle_llm::primitives::{
    CacheMemory, DecodeCache, HostSampleReason, SamplerPath, SamplingParams, SplitMix64,
};

use crate::qwen3::{KvCache, Path, Qwen3Backbone};

/// How the next token is chosen from a logits row.
#[derive(Clone, Copy, Debug)]
pub enum Sampler {
    /// Argmax — the reference deterministic chat path.
    Greedy,
    /// Temperature + nucleus (top-p) + top-k sampling. `top_p`/`top_k` of `1.0`/`0` disable that
    /// stage; `temperature` must be `> 0`.
    Sample {
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: u64,
    },
}

impl Sampler {
    /// The knobs in the shared sampler's vocabulary: greedy, or temperature + top-k + nucleus with
    /// no penalty.
    fn params(&self) -> SamplingParams {
        match *self {
            Sampler::Greedy => SamplingParams::default(),
            Sampler::Sample {
                temperature,
                top_p,
                top_k,
                ..
            } => SamplingParams {
                temperature,
                top_p,
                top_k,
                ..SamplingParams::default()
            },
        }
    }

    /// Draw a token id from a logits row through the shared candle-llm sampler ([`sample_host`]),
    /// advancing `rng` for the stochastic variants.
    fn pick(&self, logits: &Tensor, rng: &mut SplitMix64) -> candle_llm::error::Result<i32> {
        sample_host(logits, &[], &self.params(), rng, None)
    }

    fn seed(&self) -> u64 {
        match *self {
            Sampler::Greedy => 0,
            Sampler::Sample { seed, .. } => seed,
        }
    }
}

impl Qwen3Backbone {
    pub(crate) fn decode_logits_planned(
        &self,
        token: i32,
        pos_t: i32,
        cache: &mut KvCache,
        attention: AttentionPlan<'_>,
    ) -> Result<Vec<f32>> {
        let embeds = self.embed(&[token])?;
        let logits = self.und_logits(&embeds, pos_t, cache, attention)?;
        let vocab = logits.dim(2)?;
        Ok(logits.reshape((vocab,))?.to_vec1::<f32>()?)
    }

    /// One cached understanding-path forward of `ids` (one token, already on the device) at
    /// temporal `pos_t` → its `[1, 1, vocab]` logits: the engine's decode step.
    fn step_logits(
        &self,
        ids: &Tensor,
        pos_t: i32,
        cache: &mut KvCache,
        attention: AttentionPlan<'_>,
    ) -> Result<Tensor> {
        let embeds = self.embed_ids(ids)?;
        self.und_logits(&embeds, pos_t, cache, attention)
    }

    fn und_logits(
        &self,
        embeds: &Tensor,
        pos_t: i32,
        cache: &mut KvCache,
        attention: AttentionPlan<'_>,
    ) -> Result<Tensor> {
        let hidden = self.forward_cached_planned(
            embeds,
            &[pos_t],
            &[0],
            &[0],
            Path::Und,
            cache,
            true,
            attention,
            None,
        )?;
        Ok(self.lm_head(&hidden)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_planned(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
        attention: AttentionPlan<'_>,
    ) -> Result<Vec<i32>> {
        let rollout = self.rollout_planned(
            first_logits,
            cache,
            t_idx,
            eos,
            max_new_tokens,
            sampler,
            cancel,
            attention,
        )?;
        // The pre-engine loop forwarded every emitted token, the last one included, so a budget end
        // leaves the caller's cache holding the whole stream (the engine stops before that forward).
        if rollout.stop.is_none() {
            if let Some(&last) = rollout.tokens.last() {
                let t = t_idx + rollout.tokens.len() as i32;
                self.decode_logits_planned(last, t, cache, attention)?;
            }
        }
        Ok(rollout.tokens)
    }

    /// Roll out up to `max_new_tokens` understanding-path tokens on the shared engine from
    /// `first_logits` (the prefix's last-position row), the first fed at temporal `t_idx + 1`, until
    /// any of `stops` is drawn. `cancel` is bridged onto the engine's flag — set before the run, the
    /// engine refuses it before any draw; set mid-run, it is observed after the token being emitted
    /// and the engine finishes `Cancelled` — and either way returns [`CandleError::Canceled`]. The
    /// caller's cache is lent to the engine for the run and handed back whatever the outcome.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rollout_planned(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        stops: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
        attention: AttentionPlan<'_>,
    ) -> Result<UndRollout> {
        self.rollout_observed(
            first_logits,
            cache,
            t_idx,
            stops,
            max_new_tokens,
            sampler,
            cancel,
            attention,
            &mut |_| {},
        )
    }

    /// [`rollout_planned`](Self::rollout_planned), handing every engine event to `on_event` before
    /// the cancel bridge reads the pipeline's flag.
    #[allow(clippy::too_many_arguments)]
    fn rollout_observed(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        stops: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
        attention: AttentionPlan<'_>,
        on_event: &mut dyn FnMut(&StreamEvent),
    ) -> Result<UndRollout> {
        let logits = Tensor::from_vec(
            first_logits.to_vec(),
            (1, first_logits.len()),
            self.device(),
        )?;
        let start = cache.len();
        let step = UndStep {
            backbone: self,
            attention,
            // The first decode step runs at temporal `t_idx + 1`, wherever the cache ends.
            delta: t_idx + 1 - start as i32,
            vocab: first_logits.len(),
        };
        let mut pick = UndPick::new(sampler);
        let generation = GenerationConfig {
            max_new_tokens,
            sampling: pick.params,
            seed: Some(sampler.seed()),
            stop_tokens: stops.to_vec(),
        };
        // The engine's history is the penalty window and the draw reads none; the prefix (spliced
        // image rows included) is its length in placeholder ids.
        let history = vec![0; start.max(1)];
        // The pipeline's cancel, bridged onto the engine's flag: before a run that would draw (the
        // engine's typed pre-inference refusal; a zero budget draws nothing and ends `Ok`, as the
        // pre-engine loops did) and after every emitted token (its `Cancelled` finish).
        let engine_cancel = candle_llm::decode::CancelFlag::new();
        if max_new_tokens > 0 && cancel.is_some_and(CancelFlag::is_cancelled) {
            engine_cancel.cancel();
        }
        let bridged = engine_cancel.clone();
        let mut bridge = |event: StreamEvent| {
            on_event(&event);
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                bridged.cancel();
            }
        };
        let mut lent = UndCache(std::mem::replace(cache, self.new_cache()));
        let run = generate_with_sampler(
            &step,
            SpeculativePrompt::Prefilled {
                cache: &mut lent,
                logits,
                hidden: None,
                history: &history,
                position_delta: step.delta,
                warm_proposer: false,
            },
            &generation,
            &engine_cancel,
            &mut bridge,
            None,
            None,
            &mut pick,
        );
        *cache = lent.0;
        let run = run.map_err(|e| match e {
            candle_llm::error::Error::Canceled => CandleError::Canceled,
            candle_llm::error::Error::Candle(e) => CandleError::Candle(e),
            other => CandleError::Msg(format!("sensenova text decode: {other}")),
        })?;
        if run.output.finish_reason == FinishReason::Cancelled {
            return Err(CandleError::Canceled);
        }
        let stop = (run.output.finish_reason == FinishReason::StopToken)
            .then_some(pick.last)
            .flatten();
        Ok(UndRollout {
            tokens: run.output.tokens,
            stop,
        })
    }
}

/// A text rollout on the shared engine: the tokens emitted and, on a stop-token end, the stop drawn
/// (never emitted, never fed). Every emitted token but the last was fed through the cache; on a
/// stop-token end the last was too.
pub(crate) struct UndRollout {
    pub(crate) tokens: Vec<i32>,
    pub(crate) stop: Option<i32>,
}

/// The caller's understanding cache, lent to the engine for one rollout. The token-at-a-time loop
/// never rolls it back; any other rollback is refused rather than approximated.
struct UndCache(KvCache);

impl DecodeCache for UndCache {
    fn len(&self) -> i32 {
        self.0.len() as i32
    }

    fn rollback_to(&mut self, n: i32) -> candle_llm::error::Result<()> {
        if n == self.len() {
            return Ok(());
        }
        Err(candle_llm::error::Error::Msg(format!(
            "sensenova understanding cache: no rollback to {n} from {}",
            self.len()
        )))
    }

    fn reset(&mut self) {
        self.0.clear();
    }

    fn memory(&self) -> CacheMemory {
        CacheMemory {
            live_bytes: self.0.live_bytes(),
            checkpoint_bytes: 0,
        }
    }
}

/// The understanding path as the engine's [`StepModel`]: one cached single-token forward at temporal
/// `cache.len() + delta` (`h = w = 0`), projected to its f32 logits. Token-at-a-time only — a
/// multi-token verify or a hidden-state request is refused.
struct UndStep<'a> {
    backbone: &'a Qwen3Backbone,
    attention: AttentionPlan<'a>,
    delta: i32,
    vocab: usize,
}

impl StepModel for UndStep<'_> {
    type Cache = UndCache;

    fn new_cache(&self) -> UndCache {
        UndCache(self.backbone.new_cache())
    }

    fn device(&self) -> &Device {
        self.backbone.device()
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn forward_step(
        &self,
        cache: &mut UndCache,
        request: StepRequest<'_>,
    ) -> candle_llm::error::Result<StepOutput> {
        if request.want_hidden || request.tokens.len()? != 1 {
            return Err(candle_llm::error::Error::Msg(
                "the SenseNova understanding step feeds one token, last-position logits only"
                    .into(),
            ));
        }
        let ids = request.tokens.ids(self.backbone.device())?;
        let pos_t = cache.len() + self.delta;
        let logits = self
            .backbone
            .step_logits(&ids, pos_t, &mut cache.0, self.attention)
            .map_err(|e| match e {
                CandleError::Candle(e) => candle_llm::error::Error::Candle(e),
                CandleError::Canceled => candle_llm::error::Error::Canceled,
                other => candle_llm::error::Error::Msg(other.to_string()),
            })?; // [1, 1, vocab]
        let logits = match request.scope {
            LogitsScope::Last => logits.squeeze(1)?,
            LogitsScope::All => logits,
        };
        Ok(StepOutput {
            logits,
            hidden: None,
        })
    }
}

/// A rollout's draw on the engine's sampler seam: the shared candle-llm sampler ([`Sampler::pick`],
/// [`sample_host`]) from the rollout's seeded stream. It exists only to remember the last token
/// drawn, which on a stop-token end is the stop the engine does not emit (an interleave segment
/// ends differently on each of its two stops). Greedy is the device argmax; a stochastic draw is
/// the shared host reference.
struct UndPick {
    sampler: Sampler,
    rng: SplitMix64,
    /// The same knobs in the engine's vocabulary.
    params: SamplingParams,
    /// The last token drawn — on a stop-token end, the stop the engine does not emit.
    last: Option<i32>,
}

impl UndPick {
    fn new(sampler: Sampler) -> Self {
        Self {
            sampler,
            rng: SplitMix64::new(sampler.seed()),
            params: sampler.params(),
            last: None,
        }
    }
}

impl TokenSampler for UndPick {
    fn sample(
        &mut self,
        logits: &Tensor,
        _: &[i32],
        allowed: Option<&[bool]>,
    ) -> candle_llm::error::Result<(i32, SamplerPath)> {
        if allowed.is_some() {
            return Err(candle_llm::error::Error::Msg(
                "the SenseNova rollout draw takes no constraint mask".into(),
            ));
        }
        let token = self.sampler.pick(logits, &mut self.rng)?;
        self.last = Some(token);
        let path = if self.params.temperature <= 0.0 {
            SamplerPath::Device
        } else {
            SamplerPath::Host(HostSampleReason::Reference)
        };
        Ok((token, path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_llm::primitives::sampler::argmax_host;

    fn row(logits: &[f32]) -> Tensor {
        Tensor::from_slice(logits, logits.len(), &Device::Cpu).unwrap()
    }

    #[test]
    fn greedy_breaks_ties_to_lowest_index() {
        let mut rng = SplitMix64::new(0);
        for (logits, want) in [(&[0.1, 0.5, 0.5, 0.2][..], 1), (&[3.0, 1.0, 2.0][..], 0)] {
            assert_eq!(Sampler::Greedy.pick(&row(logits), &mut rng).unwrap(), want);
        }
    }

    #[test]
    fn top_k_one_is_argmax() {
        let logits = row(&[0.1, 2.0, 0.5, 1.0]);
        let s = Sampler::Sample {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            seed: 123,
        };
        let mut rng = SplitMix64::new(s.seed());
        for _ in 0..16 {
            assert_eq!(s.pick(&logits, &mut rng).unwrap(), 1);
        }
    }

    #[test]
    fn sampling_is_seed_deterministic() {
        let logits = row(&[0.2, 1.5, 0.3, 0.9, 0.1]);
        let s = Sampler::Sample {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 42,
        };
        let run = || {
            let mut rng = SplitMix64::new(s.seed());
            (0..8)
                .map(|_| s.pick(&logits, &mut rng).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "same seed → identical token sequence");
    }

    // ---- E8 (sc-24446): the rollouts on the shared engine are the pre-engine loops ----

    use candle_gen::candle_core::DType;
    use candle_gen::candle_nn::VarBuilder;

    /// The synthetic SenseNova runtime fixture `mlx-gen-sensenova` commits (a tiny two-layer
    /// backbone), loaded through this crate's own loader, with its prefix ids and tri-axis indexes.
    fn fixture() -> (Qwen3Backbone, Vec<i32>, [Vec<i32>; 3]) {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../mlx-gen/mlx-gen-sensenova/tests/fixtures/runtime_golden.safetensors"
        );
        let tensors = candle_gen::candle_core::safetensors::load(path, &Device::Cpu).unwrap();
        let meta = std::fs::read(path).unwrap();
        let (_, header) = safetensors_candle::SafeTensors::read_metadata(&meta).unwrap();
        let meta = header.metadata().clone().unwrap();
        let m = |k: &str| meta[k].clone();
        let llm = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": m("hidden_size").parse::<u64>().unwrap(),
            "intermediate_size": m("intermediate_size").parse::<u64>().unwrap(),
            "num_hidden_layers": m("num_hidden_layers").parse::<u64>().unwrap(),
            "num_attention_heads": m("num_attention_heads").parse::<u64>().unwrap(),
            "num_key_value_heads": m("num_key_value_heads").parse::<u64>().unwrap(),
            "head_dim": m("head_dim").parse::<u64>().unwrap(),
            "rms_norm_eps": m("rms_norm_eps").parse::<f64>().unwrap(),
            "rope_theta": m("rope_theta").parse::<f64>().unwrap(),
            "rope_theta_hw": m("rope_theta_hw").parse::<f64>().unwrap(),
            "vocab_size": m("vocab_size").parse::<u64>().unwrap(),
            "attention_bias": false,
        });
        let v = serde_json::json!({ "model_type": "neo_chat", "tie_word_embeddings": false, "llm_config": llm, "vision_config": {} });
        let cfg = crate::NeoChatConfig::from_config_json(&v).unwrap();
        let ids = tensors["prefix.input_ids"]
            .flatten_all()
            .unwrap()
            .to_vec1::<i32>()
            .unwrap();
        let idx = tensors["prefix.indexes"].to_vec2::<i32>().unwrap();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let model = Qwen3Backbone::from_weights(&vb, &cfg, "language_model").unwrap();
        (model, ids, [idx[0].clone(), idx[1].clone(), idx[2].clone()])
    }

    /// Prefill the prefix: the cache, the last-position host row and the prefix's max temporal index.
    fn prefill(
        model: &Qwen3Backbone,
        ids: &[i32],
        idx: &[Vec<i32>; 3],
    ) -> (KvCache, Vec<f32>, i32) {
        let embeds = model.embed(ids).unwrap();
        let mut cache = model.new_cache();
        let hidden = model
            .forward_cached_planned(
                &embeds,
                &idx[0],
                &idx[1],
                &idx[2],
                Path::Und,
                &mut cache,
                true,
                AttentionPlan::UNBOUNDED,
                None,
            )
            .unwrap();
        let n = hidden.dim(1).unwrap();
        let logits = model.lm_head(&hidden.narrow(1, n - 1, 1).unwrap()).unwrap();
        let t_idx = *idx[0].iter().max().unwrap();
        (
            cache,
            logits.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            t_idx,
        )
    }

    /// The next decode row after a rollout — what the cache it left holds, read through one probe.
    fn probe(model: &Qwen3Backbone, cache: &mut KvCache, t: i32) -> Vec<f32> {
        model
            .decode_logits_planned(7, t + 1, cache, AttentionPlan::UNBOUNDED)
            .unwrap()
    }

    /// The pre-engine `generate_planned` loop. Its greedy draw is frozen verbatim (the host scan,
    /// ties to the lowest index — the pre-engine crate-local `argmax` was a byte copy of
    /// [`argmax_host`]); its stochastic draw is the shared sampler the engine now draws through
    /// (sc-24446: the pre-engine crate-local sort-based sampler's stream is not preserved).
    #[allow(clippy::too_many_arguments)]
    fn reference_generate(
        model: &Qwen3Backbone,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        let attention = AttentionPlan::UNBOUNDED;
        let mut rng = SplitMix64::new(sampler.seed());
        let mut logits = first_logits.to_vec();
        let mut out = Vec::new();
        let mut t = t_idx;
        for _ in 0..max_new_tokens {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(CandleError::Canceled);
            }
            let next = match sampler {
                Sampler::Greedy => argmax_host(&logits),
                Sampler::Sample { .. } => sampler.pick(&row(&logits), &mut rng).unwrap(),
            };
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            t += 1;
            logits = model.decode_logits_planned(next, t, cache, attention)?;
        }
        Ok(out)
    }

    /// The pre-engine interleave text segment (`t2i.rs`), verbatim: `(tokens, next, hit_max)`.
    fn reference_segment(
        model: &Qwen3Backbone,
        mut next: i32,
        cache: &mut KvCache,
        t_cond: &mut usize,
        stops: [i32; 2],
        total_tokens: &mut usize,
        max_new_tokens: usize,
    ) -> (Vec<i32>, i32, bool) {
        let mut gen_tokens = Vec::new();
        let mut hit_max = false;
        loop {
            if next == stops[0] || next == stops[1] {
                break;
            }
            gen_tokens.push(next);
            *total_tokens += 1;
            let logits = model
                .decode_logits_planned(next, (*t_cond + 1) as i32, cache, AttentionPlan::UNBOUNDED)
                .unwrap();
            *t_cond += 1;
            next = argmax_host(&logits);
            if *total_tokens >= max_new_tokens {
                hit_max = true;
                break;
            }
        }
        (gen_tokens, next, hit_max)
    }

    fn samplers() -> Vec<Sampler> {
        let mut all = vec![Sampler::Greedy];
        for (temperature, top_p, top_k) in [(0.8, 1.0, 0), (1.0, 0.9, 0), (1.3, 0.8, 5)] {
            for seed in [0, 7, 42] {
                all.push(Sampler::Sample {
                    temperature,
                    top_p,
                    top_k,
                    seed,
                });
            }
        }
        all
    }

    /// `generate_planned` on the engine emits the pre-engine stream — greedy token-identical to the
    /// frozen pre-engine loop, seeded stochastic identical to that loop drawing through the shared
    /// sampler — across eos ends (an eos the run itself emits), budget ends and the zero budget,
    /// and leaves the caller's cache exactly as the pre-engine loop did (same length, same
    /// next-step logits). A cancel set before the run stays typed.
    #[test]
    fn generate_on_the_engine_is_the_pre_engine_loop() {
        let (model, ids, idx) = fixture();
        let plan = AttentionPlan::UNBOUNDED;
        let (mut eos_ends, mut budget_ends) = (0, 0);
        for sampler in samplers() {
            let (mut c, first, t_idx) = prefill(&model, &ids, &idx);
            let free =
                reference_generate(&model, &first, &mut c, t_idx, &[], 12, sampler, None).unwrap();
            for (eos, max) in [
                (vec![], 12),
                (vec![free[3]], 12),
                (vec![free[0]], 12),
                (vec![], 0),
            ] {
                let (mut want_cache, _, _) = prefill(&model, &ids, &idx);
                let want = reference_generate(
                    &model,
                    &first,
                    &mut want_cache,
                    t_idx,
                    &eos,
                    max,
                    sampler,
                    None,
                )
                .unwrap();
                let (mut got_cache, _, _) = prefill(&model, &ids, &idx);
                let got = model
                    .generate_planned(
                        &first,
                        &mut got_cache,
                        t_idx,
                        &eos,
                        max,
                        sampler,
                        None,
                        plan,
                    )
                    .unwrap();
                assert_eq!(got, want, "{sampler:?} eos {eos:?} max {max}");
                assert_eq!(got_cache.len(), want_cache.len());
                let t = t_idx + want.len() as i32;
                assert_eq!(
                    probe(&model, &mut got_cache, t),
                    probe(&model, &mut want_cache, t)
                );
                if want.len() < max {
                    eos_ends += 1;
                } else if max > 0 {
                    budget_ends += 1;
                }
            }
        }
        assert!(
            eos_ends > 0 && budget_ends > 0,
            "eos {eos_ends}, budget {budget_ends}"
        );
        let cancel = CancelFlag::new();
        cancel.cancel();
        for sampler in [Sampler::Greedy, samplers()[1]] {
            let (mut c, first, t_idx) = prefill(&model, &ids, &idx);
            assert!(matches!(
                reference_generate(
                    &model,
                    &first,
                    &mut c,
                    t_idx,
                    &[],
                    4,
                    sampler,
                    Some(&cancel)
                ),
                Err(CandleError::Canceled)
            ));
            let (mut c, _, _) = prefill(&model, &ids, &idx);
            assert!(matches!(
                model.generate_planned(&first, &mut c, t_idx, &[], 4, sampler, Some(&cancel), plan),
                Err(CandleError::Canceled)
            ));
            // The cache lent to the engine comes back even from a cancelled rollout.
            assert_eq!(c.len(), ids.len());
            assert!(model
                .generate_planned(&first, &mut c, t_idx, &[], 0, sampler, Some(&cancel), plan)
                .unwrap()
                .is_empty());
        }
    }

    /// An interleave text segment on the engine (`rollout_planned` with the segment's stops and
    /// remaining budget, as `t2i.rs` calls it) emits the pre-engine segment's tokens and ends the
    /// same way: on the same stop, or at the budget (`hit_max`).
    #[test]
    fn interleave_segments_on_the_engine_are_the_pre_engine_loop() {
        let (model, ids, idx) = fixture();
        let (mut c, first, t_idx) = prefill(&model, &ids, &idx);
        let free = reference_generate(
            &model,
            &first,
            &mut c,
            t_idx,
            &[],
            12,
            Sampler::Greedy,
            None,
        )
        .unwrap();
        let (mut stop_ends, mut budget_ends) = (0, 0);
        for (stops, max_new_tokens, total) in [
            ([free[3], -1], 12, 0),
            ([-1, free[2]], 12, 0),
            ([-1, -1], 6, 0),
            ([-1, -1], 9, 4),
            ([free[0], -1], 12, 0),
            ([-1, -1], 0, 0),
        ] {
            let (mut want_cache, _, _) = prefill(&model, &ids, &idx);
            let (mut t_cond, mut want_total) = (t_idx as usize, total);
            let (want, next, hit_max) = reference_segment(
                &model,
                argmax_host(&first),
                &mut want_cache,
                &mut t_cond,
                stops,
                &mut want_total,
                max_new_tokens,
            );
            let (mut got_cache, _, _) = prefill(&model, &ids, &idx);
            let got = model
                .rollout_planned(
                    &first,
                    &mut got_cache,
                    t_idx,
                    &stops,
                    max_new_tokens.saturating_sub(total).max(1),
                    Sampler::Greedy,
                    None,
                    AttentionPlan::UNBOUNDED,
                )
                .unwrap();
            let what = format!("stops {stops:?} max {max_new_tokens} total {total}");
            assert_eq!(got.tokens, want, "{what}");
            assert_eq!(got.stop.is_none(), hit_max, "{what}");
            if !hit_max {
                assert_eq!(got.stop, Some(next), "{what}");
                assert_eq!(got_cache.len(), want_cache.len(), "{what}");
                stop_ends += 1;
            } else {
                budget_ends += 1;
            }
        }
        assert!(stop_ends > 0 && budget_ends > 0);
    }

    /// sc-24446 behaviour change: the seeded stochastic rollout now draws through the shared
    /// sampler, so the pre-engine stream is not pinned. What it must keep: a seed reproduces its
    /// stream, different seeds explore different streams, `top_k = 1` collapses to the greedy
    /// stream, and the first draw follows the shaped (temperature-softmax) distribution of its row.
    #[test]
    fn stochastic_rollouts_reproduce_per_seed_and_follow_the_shaped_distribution() {
        let (model, ids, idx) = fixture();
        let plan = AttentionPlan::UNBOUNDED;
        let (_, first, t_idx) = prefill(&model, &ids, &idx);
        let roll = |sampler: Sampler, max: usize| {
            let (mut c, _, _) = prefill(&model, &ids, &idx);
            model
                .rollout_planned(&first, &mut c, t_idx, &[], max, sampler, None, plan)
                .unwrap()
                .tokens
        };
        let sample = |seed, top_k, temperature| Sampler::Sample {
            temperature,
            top_p: 1.0,
            top_k,
            seed,
        };
        let streams: Vec<_> = (0..6).map(|seed| roll(sample(seed, 0, 1.0), 10)).collect();
        for (seed, stream) in streams.iter().enumerate() {
            assert_eq!(
                &roll(sample(seed as u64, 0, 1.0), 10),
                stream,
                "seed {seed}"
            );
        }
        assert!(
            streams.iter().any(|s| s != &streams[0]),
            "six seeds drew one stream: {streams:?}"
        );
        assert_eq!(roll(sample(9, 1, 1.0), 10), roll(Sampler::Greedy, 10));

        // The first draw (no forward) over many seeds against softmax(first / T). The fixture's
        // first row is nearly flat (a spread under one logit), so `T = 0.1` shapes it into a
        // distribution a wrong temperature or a wrong draw cannot hide in.
        let temperature = 0.1f32;
        let n = 2000;
        let mut counts = vec![0usize; first.len()];
        let (mut c, _, _) = prefill(&model, &ids, &idx);
        for seed in 0..n {
            let got = model
                .rollout_planned(
                    &first,
                    &mut c,
                    t_idx,
                    &[],
                    1,
                    sample(seed, 0, temperature),
                    None,
                    plan,
                )
                .unwrap()
                .tokens;
            counts[got[0] as usize] += 1;
        }
        let max = first.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = first
            .iter()
            .map(|&l| (f64::from(l - max) / f64::from(temperature)).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let n = n as f64;
        let mut seen = 0;
        for (token, (w, &k)) in weights.iter().zip(&counts).enumerate() {
            let p = w / total;
            // Five binomial standard deviations (the seeds are fixed, so this never flakes).
            let bound = 5.0 * (p * (1.0 - p) / n).sqrt() + 1.0 / n;
            assert!(
                (k as f64 / n - p).abs() <= bound,
                "token {token}: drawn {k} of {n}, softmax {p}"
            );
            seen += usize::from(k > 0);
        }
        assert!(seen > 1, "the first row's draw is not degenerate");
    }

    /// The pipeline's cancel, set mid-rollout, is bridged onto the engine's flag: the engine
    /// finishes `Cancelled` right after the token it was emitting, the rollout returns the typed
    /// [`CandleError::Canceled`] (greedy and stochastic), and the lent cache comes back.
    #[test]
    fn a_mid_run_cancel_ends_the_rollout_typed() {
        let (model, ids, idx) = fixture();
        let (_, first, t_idx) = prefill(&model, &ids, &idx);
        for sampler in [Sampler::Greedy, samplers()[1]] {
            let cancel = CancelFlag::new();
            let mut emitted = 0;
            let (mut c, _, _) = prefill(&model, &ids, &idx);
            let run = model.rollout_observed(
                &first,
                &mut c,
                t_idx,
                &[],
                12,
                sampler,
                Some(&cancel),
                AttentionPlan::UNBOUNDED,
                &mut |event| {
                    if let StreamEvent::Token { step, .. } = event {
                        emitted = step + 1;
                        if *step == 2 {
                            cancel.cancel();
                        }
                    }
                },
            );
            assert!(matches!(run, Err(CandleError::Canceled)), "{sampler:?}");
            assert_eq!(
                emitted, 3,
                "{sampler:?}: cancelled right after the third token"
            );
            assert_eq!(
                c.len(),
                ids.len() + 2,
                "{sampler:?}: the lent cache comes back"
            );
        }
    }
}
