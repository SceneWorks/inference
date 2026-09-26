//! Autoregressive score (ABC) and semantic-token generation over the YuE2 MoT backbone
//! (sc-22991).
//!
//! Ported from the pinned upstream `src/yue2/sampling.py::generate_tokens` and the `plan` /
//! `generate_semantic` stages of `src/yue2/pipeline.py`. The model and sampler consume **token
//! ids**; the request, its prefixes and the exact symbolic plan come from [`crate::protocol`] and
//! [`crate::plan`].
//!
//! # One decode ([`decode_tokens`])
//!
//! * The prefix plus the output budget must fit the released
//!   [`CONTEXT`](crate::protocol::CONTEXT)
//!   ([`check_generation_budget`]); a longer request is an error, never an implicit truncation,
//!   window or re-prefill. The KV cache is sized to exactly `prefix + max_tokens` positions and
//!   positions are the cache slots `0, 1, 2, …`.
//! * With guidance (`scale != 1`) the negative prefix gets its own cache; every step runs both
//!   branches and samples from `uncond + scale · (cond − uncond)` in the model dtype.
//! * A step shapes the row with [`distribution`], draws (or takes the argmax), reports
//!   the token (end id included) to the observer, and stops on the phase's end id. Otherwise the
//!   token is fed back — except after the last budgeted step, which runs no forward.
//! * `truncated` is true exactly when the budget ran out before the end id.
//! * Cancellation is checked before the prefill, between prefill chunks
//!   ([`PREFILL_CHUNK`](crate::model::PREFILL_CHUNK) positions), between the two branches'
//!   prefills and before every decode step, so at most one chunk or one step (both branches) of
//!   work runs after the flag trips. A cancelled decode returns [`gen_core::Error::Canceled`] and
//!   no partial output.
//!
//! # The stages
//!
//! The request is planned with [`SymbolicPlan::prepare`]. A [`PlanStep::GenerateAbc`] is sampled
//! by [`plan_score`] — the planner prefix with the ABC phase's controls
//! ([`GenerationConfig::abc`]) — into a [`SymbolicPlan`] carrying the exact sampled ids; a
//! [`PlanStep::Ready`] plan (`cot = off`, or an external score) needs no sampling.
//! [`generate_semantic`] then takes the plan and its [`SymbolicPlan::semantic_conditioning`] and
//! samples the codec tokens with the semantic controls ([`GenerationConfig::semantic`]). It
//! refuses conditioning that is not the plan's: the positive prefix must be the plan's prefix and
//! frame **exactly** the plan's ABC ids (`ABC_START abc… ABC_END MUSIC_START`); under guidance
//! the negative must frame the very same ids (upstream's `same_instruction_and_exact_abc`) or —
//! for `cot = off`, which has no score — only `MUSIC_START` (`instruction_only`); and between the
//! leading `EOD` and that framing both prefixes hold ordinary text ids only (`< EOD`), as
//! upstream's `[EOD] + tokenizer.encode(text)` does. Both stages restart their random stream from
//! the request seed ([`stage_rng`]), as upstream does.
//!
//! [`PlanStep::GenerateAbc`]: crate::plan::PlanStep::GenerateAbc
//! [`PlanStep::Ready`]: crate::plan::PlanStep::Ready

use std::time::Instant;

use candle_audio::candle_core::DType;
use candle_audio::gen_core;
use candle_llm::primitives::StaticKvCache;
use serde_json::{Map, Value};

use crate::model::{backend, Yue2Lm};
use crate::plan::{AbcPlanning, SemanticConditioning, SymbolicPlan};
use crate::protocol::{
    check_generation_budget, CotMode, GenerationConfig, ProtocolError, Sampling, ABC_END,
    ABC_START, CODEC_OFFSET, EOD, MUSIC_START,
};
use crate::sampling::{cfg_mix, distribution, next_token, Arith, Phase, SplitMix64, TokenRng};
use crate::tokenizer::Yue2TextTokenizer;

fn protocol_error(e: ProtocolError) -> gen_core::Error {
    gen_core::Error::Msg(format!("YuE2: {e}"))
}

/// Classifier-free guidance for one decode.
#[derive(Clone, Copy, Debug)]
pub struct Guidance<'a> {
    /// The negative (unconditional) branch's prefix.
    pub negative: &'a [u32],
    /// The guidance scale; exactly `1` is no guidance (the negative branch is not run).
    pub scale: f64,
}

/// One autoregressive decode — upstream `generate_tokens`' arguments.
#[derive(Clone, Copy, Debug)]
pub struct DecodeRequest<'a> {
    /// Which phase (mask, stop id).
    pub phase: Phase,
    /// The positive prefix.
    pub prefix: &'a [u32],
    /// This phase's sampling controls.
    pub sampling: &'a Sampling,
    /// Optional classifier-free guidance.
    pub guidance: Option<Guidance<'a>>,
    /// The historical `cot = off` sampler arithmetic (model-dtype scores, top-p keeps three).
    pub legacy_off: bool,
}

/// Observes a decode as it runs. Every method has a no-op default.
pub trait DecodeObserver {
    /// The logits row step `step` samples from — the CFG-mixed row under guidance, the
    /// conditional row otherwise — before any mask or penalty.
    fn on_logits(&mut self, step: usize, logits: &[f32]) {
        let _ = (step, logits);
    }

    /// An output token, reported once each, the end id included.
    fn on_token(&mut self, phase: Phase, token: u32) {
        let _ = (phase, token);
    }
}

/// The no-op observer.
impl DecodeObserver for () {}

/// Cancellation and observation for a decode.
pub struct Hooks<'a> {
    /// Polled at every bounded boundary (see the module docs); `true` cancels.
    pub cancelled: &'a dyn Fn() -> bool,
    /// Receives each step's logits and tokens.
    pub observer: &'a mut dyn DecodeObserver,
}

/// What a decode produced — upstream's `(history, timing, truncated)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    /// The generated tokens, the end id excluded.
    pub tokens: Vec<u32>,
    /// The phase's end id was sampled.
    pub ended: bool,
    /// The budget ran out before the end id (`!ended`).
    pub truncated: bool,
    /// Outputs including the end id.
    pub output_tokens: usize,
    /// Positive prefix length.
    pub prefix_tokens: usize,
    /// `2` under guidance, else `1`.
    pub cfg_branches: usize,
    /// Wall time of the prefill(s), seconds.
    pub prefill_seconds: f64,
    /// Wall time of the whole decode, seconds.
    pub seconds: f64,
}

/// The random stream a stage draws from: SplitMix64 seeded with the request seed, restarted for
/// each stage (upstream reseeds its generator per stage). Not PyTorch's stream (epic E9).
pub fn stage_rng(seed: u64) -> SplitMix64 {
    SplitMix64::new(seed)
}

fn arith_of(dtype: DType) -> gen_core::Result<Arith> {
    match dtype {
        DType::F32 => Ok(Arith::F32),
        DType::BF16 => Ok(Arith::Bf16),
        other => Err(gen_core::Error::Unsupported(format!(
            "YuE2 runs in F32 or BF16, not {other:?}"
        ))),
    }
}

fn check_cancel(cancelled: &dyn Fn() -> bool) -> gen_core::Result<()> {
    if cancelled() {
        Err(gen_core::Error::Canceled)
    } else {
        Ok(())
    }
}

fn host_row(logits: &candle_audio::candle_core::Tensor) -> gen_core::Result<Vec<f32>> {
    let row = logits
        .to_dtype(DType::F32)
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(backend("logits"))?;
    if let Some(i) = row.iter().position(|x| !x.is_finite()) {
        return Err(gen_core::Error::Msg(format!(
            "YuE2 produced a non-finite logit ({}) at id {i}",
            row[i]
        )));
    }
    Ok(row)
}

/// Run one autoregressive decode (see the module docs). `rng` supplies one uniform per stochastic
/// step; a greedy (`temperature == 0`) decode never draws.
pub fn decode_tokens(
    lm: &Yue2Lm,
    request: &DecodeRequest<'_>,
    rng: &mut dyn TokenRng,
    hooks: Hooks<'_>,
) -> gen_core::Result<Decoded> {
    let Hooks {
        cancelled,
        observer,
    } = hooks;
    let sampling = request.sampling;
    let model_arith = arith_of(lm.dtype())?;
    let score_arith = if request.legacy_off {
        model_arith
    } else {
        Arith::F32
    };
    let budget = sampling.max_tokens() as usize;
    let scale = request.guidance.map_or(1.0, |g| g.scale);
    if !scale.is_finite() {
        return Err(gen_core::Error::Msg(format!(
            "YuE2: guidance scale {scale} is not finite"
        )));
    }
    // Upstream's refusals before prefill: each supplied branch plus the budget must fit the
    // context — nothing is truncated to make a request fit.
    check_generation_budget(
        request.prefix.len(),
        request.guidance.map(|g| g.negative.len()),
        sampling,
        scale,
    )
    .map_err(protocol_error)?;
    let guidance = request.guidance.filter(|g| g.scale != 1.0);
    check_cancel(cancelled)?;
    let start = Instant::now();
    let prefill = |ids: &[u32]| -> gen_core::Result<(Vec<f32>, StaticKvCache)> {
        let mut cache = lm.new_cache(ids.len() + budget)?;
        let logits = lm.prefill(ids, &mut cache, || check_cancel(cancelled))?;
        Ok((host_row(&logits)?, cache))
    };
    let (mut conditional, mut positive) = prefill(request.prefix)?;
    let mut negative = match guidance {
        Some(g) => {
            check_cancel(cancelled)?;
            Some(prefill(g.negative)?)
        }
        None => None,
    };
    let prefill_seconds = start.elapsed().as_secs_f64();
    let end = request.phase.end_token();
    let mut history = Vec::new();
    let mut ended = false;
    for step in 0..budget {
        check_cancel(cancelled)?;
        let logits = match (&negative, guidance) {
            (Some((unconditional, _)), Some(g)) => {
                cfg_mix(&conditional, unconditional, g.scale, model_arith)
            }
            _ => std::mem::take(&mut conditional),
        };
        observer.on_logits(step, &logits);
        let scores = distribution(
            &logits,
            sampling,
            &history,
            step,
            request.phase,
            score_arith,
            request.legacy_off,
        );
        let token = next_token(&scores, sampling, score_arith, rng)?;
        observer.on_token(request.phase, token);
        if token == end {
            ended = true;
            break;
        }
        history.push(token);
        if step + 1 < budget {
            conditional = host_row(&lm.decode(token, &mut positive)?)?;
            if let Some((unconditional, cache)) = negative.as_mut() {
                *unconditional = host_row(&lm.decode(token, cache)?)?;
            }
        }
    }
    let output_tokens = history.len() + usize::from(ended);
    Ok(Decoded {
        tokens: history,
        ended,
        truncated: !ended,
        output_tokens,
        prefix_tokens: request.prefix.len(),
        cfg_branches: if guidance.is_some() { 2 } else { 1 },
        prefill_seconds,
        seconds: start.elapsed().as_secs_f64(),
    })
}

/// Upstream's timing record for a decode (`generate_tokens`' `timing`, minus the CUDA-graph
/// fields this runtime has no counterpart for).
fn timing_of(decoded: &Decoded) -> Map<String, Value> {
    let mut t = Map::new();
    t.insert("seconds".into(), Value::from(decoded.seconds));
    t.insert(
        "prefill_seconds".into(),
        Value::from(decoded.prefill_seconds),
    );
    t.insert("output_tokens".into(), Value::from(decoded.output_tokens));
    t.insert("content_tokens".into(), Value::from(decoded.tokens.len()));
    t.insert("prefix_tokens".into(), Value::from(decoded.prefix_tokens));
    t.insert("cfg_branches".into(), Value::from(decoded.cfg_branches));
    t.insert("execution".into(), Value::from("candle"));
    t
}

/// Sample the score for a [`PlanStep::GenerateAbc`](crate::plan::PlanStep::GenerateAbc): the
/// planner's own prefix (`EOD … ABC_START`) with the ABC phase's controls, then
/// [`AbcPlanning::finish`] with the exact sampled ids (`ABC_END` excluded) and the truncation flag.
pub fn plan_score(
    lm: &Yue2Lm,
    planning: AbcPlanning,
    tokenizer: &Yue2TextTokenizer,
    config: &GenerationConfig,
    rng: &mut dyn TokenRng,
    hooks: Hooks<'_>,
) -> gen_core::Result<(SymbolicPlan, Decoded)> {
    let decoded = decode_tokens(
        lm,
        &DecodeRequest {
            phase: Phase::Abc,
            prefix: planning.prefix(),
            sampling: config.abc(),
            guidance: None,
            legacy_off: false,
        },
        rng,
        hooks,
    )?;
    let plan = planning
        .finish(
            tokenizer,
            decoded.tokens.clone(),
            timing_of(&decoded),
            decoded.truncated,
        )
        .map_err(|e| gen_core::Error::Msg(format!("YuE2 planner: {e}")))?;
    Ok((plan, decoded))
}

/// The semantic stage's codec tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticTokens {
    /// Codec indices `0 .. CODEC_SIZE` (vocabulary id − `CODEC_OFFSET`), the end id excluded.
    pub codes: Vec<u32>,
    /// The raw decode (vocabulary ids, stop/truncation, timing).
    pub decoded: Decoded,
}

/// `ABC_START abc… ABC_END MUSIC_START` — the tail every positive prefix of `plan` ends with (for
/// `cot = off`, the empty score).
fn semantic_tail(plan: &SymbolicPlan) -> Vec<u32> {
    let mut tail = Vec::with_capacity(plan.abc_ids().len() + 3);
    tail.push(ABC_START);
    tail.extend_from_slice(plan.abc_ids());
    tail.extend_from_slice(&[ABC_END, MUSIC_START]);
    tail
}

/// The tail a guidance negative of `plan` ends with: the same exact score for `full | melody`,
/// only `MUSIC_START` for `off`.
fn negative_tail(plan: &SymbolicPlan) -> Vec<u32> {
    match plan.request().cot() {
        CotMode::Off => vec![MUSIC_START],
        CotMode::Full | CotMode::Melody => semantic_tail(plan),
    }
}

/// `EOD`, then a non-empty body of ordinary text ids (`< EOD`), then exactly `tail`.
fn check_framing(branch: &str, prefix: &[u32], tail: &[u32]) -> gen_core::Result<()> {
    let refuse = |what: &str| {
        Err(gen_core::Error::Msg(format!(
            "YuE2 semantic stage: the {branch} prefix {what}"
        )))
    };
    if prefix.first() != Some(&EOD) {
        return refuse("must start with EOD");
    }
    if prefix.len() < 1 + tail.len() || !prefix.ends_with(tail) {
        return refuse("must end with the plan's exact score framing");
    }
    let body = &prefix[1..prefix.len() - tail.len()];
    if body.is_empty() {
        return refuse("has no instruction text between EOD and the score framing");
    }
    if let Some(bad) = body.iter().find(|&&t| t >= EOD) {
        return refuse(&format!(
            "holds id {bad} before the score framing; only ordinary text ids (< EOD) belong there"
        ));
    }
    Ok(())
}

/// Check that `conditioning` is `plan`'s ([`SymbolicPlan::semantic_conditioning`]'s output).
fn check_conditioning<'a>(
    plan: &SymbolicPlan,
    conditioning: &'a SemanticConditioning,
) -> gen_core::Result<Option<Guidance<'a>>> {
    let refuse = |what: String| Err(gen_core::Error::Msg(format!("YuE2 semantic stage: {what}")));
    let cot = plan.request().cot();
    let scale = conditioning.cfg_scale;
    if !(scale.is_finite() && (0.0..=20.0).contains(&scale)) {
        return refuse(format!("cfg_scale {scale} must be finite and in [0, 20]"));
    }
    if scale != plan.request().guidance() {
        return refuse(format!(
            "cfg_scale {scale} is not the plan's guidance {}",
            plan.request().guidance()
        ));
    }
    if conditioning.legacy_off != (cot == CotMode::Off) {
        return refuse("legacy_off must be set exactly for cot = off".into());
    }
    check_framing("positive", &conditioning.positive, &semantic_tail(plan))?;
    if conditioning.positive != plan.prefix() {
        return refuse("the positive prefix is not the plan's exact prefix".into());
    }
    match (&conditioning.negative, scale != 1.0) {
        (Some(negative), true) => {
            check_framing("negative", negative, &negative_tail(plan))?;
            Ok(Some(Guidance { negative, scale }))
        }
        (None, false) => Ok(None),
        (None, true) => refuse("guidance (cfg_scale != 1) requires a negative prefix".into()),
        (Some(_), false) => refuse("a negative prefix without guidance (cfg_scale = 1)".into()),
    }
}

/// Generate the semantic codec tokens for `plan` from its `conditioning` (see the module docs),
/// with the semantic phase's controls. The `cot = off` stage samples with the historical
/// arithmetic (`legacy_off`).
pub fn generate_semantic(
    lm: &Yue2Lm,
    plan: &SymbolicPlan,
    conditioning: &SemanticConditioning,
    config: &GenerationConfig,
    rng: &mut dyn TokenRng,
    hooks: Hooks<'_>,
) -> gen_core::Result<SemanticTokens> {
    let guidance = check_conditioning(plan, conditioning)?;
    let decoded = decode_tokens(
        lm,
        &DecodeRequest {
            phase: Phase::Semantic,
            prefix: &conditioning.positive,
            sampling: config.semantic(),
            guidance,
            legacy_off: conditioning.legacy_off,
        },
        rng,
        hooks,
    )?;
    let codes = decoded.tokens.iter().map(|&t| t - CODEC_OFFSET).collect();
    Ok(SemanticTokens { codes, decoded })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::model::{synthetic, MotPaths};
    use crate::plan::PlanStep;
    use crate::protocol::{
        SongRequest, SongRequestSpec, CODEC_SIZE, CONTEXT, MUSIC_END, VOCAB_SIZE,
    };
    use crate::sampling::{self, test_sampling};
    use crate::test_fixtures;

    /// Draws from a fixed list (parity tests inject the reference's uniforms).
    pub(crate) struct Uniforms(pub std::collections::VecDeque<f32>);

    impl TokenRng for Uniforms {
        fn next_f32(&mut self) -> f32 {
            self.0.pop_front().expect("injected uniforms exhausted")
        }
    }

    #[derive(Default)]
    struct Record {
        logits: Vec<Vec<f32>>,
        tokens: Vec<u32>,
    }

    impl DecodeObserver for Record {
        fn on_logits(&mut self, _step: usize, logits: &[f32]) {
            self.logits.push(logits.to_vec());
        }
        fn on_token(&mut self, _phase: Phase, token: u32) {
            self.tokens.push(token);
        }
    }

    fn never() -> bool {
        false
    }

    fn greedy(max_tokens: i64) -> Sampling {
        test_sampling(0.0, 0.95, 100, 1.2, 50, 0, max_tokens)
    }

    const PREFIX: [u32; 7] = [EOD, 40, 1234, 99, ABC_START, ABC_END, MUSIC_START];

    fn run(
        lm: &Yue2Lm,
        req: &DecodeRequest<'_>,
        rng: &mut dyn TokenRng,
        cancelled: &dyn Fn() -> bool,
    ) -> (gen_core::Result<Decoded>, Record) {
        let mut rec = Record::default();
        let out = decode_tokens(
            lm,
            req,
            rng,
            Hooks {
                cancelled,
                observer: &mut rec,
            },
        );
        (out, rec)
    }

    #[test]
    fn budget_exhaustion_is_reported_as_truncation() {
        let lm = synthetic::model(MotPaths::Ar);
        let s = greedy(5);
        let req = DecodeRequest {
            phase: Phase::Semantic,
            prefix: &PREFIX,
            sampling: &s,
            guidance: None,
            legacy_off: false,
        };
        let (out, rec) = run(&lm, &req, &mut stage_rng(1), &never);
        let out = out.unwrap();
        assert_eq!(out.tokens.len(), 5);
        assert!(out.truncated && !out.ended);
        assert_eq!(out.output_tokens, 5);
        assert_eq!(rec.tokens, out.tokens);
        assert!(out
            .tokens
            .iter()
            .all(|&t| Phase::Semantic.allows(t) && t != MUSIC_END));
    }

    /// The end id stops the decode: it is reported to the observer, excluded from the tokens, and
    /// the decode is not truncated. Driven through the real loop with an injected draw that lands
    /// on MUSIC_END's slot of the shaped distribution.
    #[test]
    fn end_id_stops_and_is_reported_but_not_returned() {
        let lm = synthetic::model(MotPaths::Ar);
        let s = greedy(1);
        let req = DecodeRequest {
            phase: Phase::Semantic,
            prefix: &PREFIX,
            sampling: &s,
            guidance: None,
            legacy_off: false,
        };
        let (_, rec) = run(&lm, &req, &mut stage_rng(1), &never);
        let row = &rec.logits[0];
        // Drive the real loop to the end id through an injected draw: a stochastic sampler with
        // top_k large enough to include MUSIC_END, and a uniform that lands on it.
        let stochastic = test_sampling(1.0, 1.0, VOCAB_SIZE.into(), 1.0, 50, 0, 4);
        let probs = sampling::probabilities(
            &distribution(row, &stochastic, &[], 0, Phase::Semantic, Arith::F32, false),
            Arith::F32,
        );
        let total: f64 = probs.iter().map(|&p| p as f64).sum();
        let below: f64 = probs[..MUSIC_END as usize].iter().map(|&p| p as f64).sum();
        let u = ((below + probs[MUSIC_END as usize] as f64 / 2.0) / total) as f32;
        let req = DecodeRequest {
            sampling: &stochastic,
            ..req
        };
        let (out, rec) = run(&lm, &req, &mut Uniforms([u].into()), &never);
        let out = out.unwrap();
        assert_eq!(
            rec.tokens,
            vec![MUSIC_END],
            "end id reported to the observer"
        );
        assert!(out.tokens.is_empty() && out.ended && !out.truncated);
        assert_eq!(out.output_tokens, 1);
    }

    #[test]
    fn min_tokens_bars_the_end_id_in_the_real_loop() {
        let lm = synthetic::model(MotPaths::Ar);
        let s = test_sampling(1.0, 1.0, VOCAB_SIZE.into(), 1.0, 50, 3, 3);
        // Uniform 0.999999 lands past the codec range (on MUSIC_END's slot if it had mass).
        let req = DecodeRequest {
            phase: Phase::Semantic,
            prefix: &PREFIX,
            sampling: &s,
            guidance: None,
            legacy_off: false,
        };
        let (out, _) = run(
            &lm,
            &req,
            &mut Uniforms([0.0, 0.5, 0.999_999].into()),
            &never,
        );
        let out = out.unwrap();
        assert_eq!(out.tokens.len(), 3);
        assert!(out.tokens.iter().all(|&t| t != MUSIC_END));
    }

    #[test]
    fn context_budget_is_refused_not_truncated() {
        let lm = synthetic::model(MotPaths::Ar);
        let long = vec![5u32; CONTEXT - 3];
        let s = greedy(4);
        let req = DecodeRequest {
            phase: Phase::Abc,
            prefix: &long,
            sampling: &s,
            guidance: None,
            legacy_off: false,
        };
        let (out, rec) = run(&lm, &req, &mut stage_rng(1), &never);
        let err = out.unwrap_err().to_string();
        assert!(err.contains("nothing is truncated implicitly"), "{err}");
        assert!(rec.tokens.is_empty(), "nothing ran");
        let neg = vec![5u32; CONTEXT];
        let req = DecodeRequest {
            prefix: &PREFIX,
            guidance: Some(Guidance {
                negative: &neg,
                scale: 1.5,
            }),
            ..req
        };
        let err = run(&lm, &req, &mut stage_rng(1), &never)
            .0
            .unwrap_err()
            .to_string();
        assert!(err.contains("negative prefix"), "{err}");
    }

    /// Cancellation is observed before the prefill and at every step boundary: tripping it after
    /// the second token ends the decode before a third token, with the typed error.
    #[test]
    fn cancellation_is_observed_at_bounded_boundaries() {
        let lm = synthetic::model(MotPaths::Ar);
        let s = greedy(50);
        let req = DecodeRequest {
            phase: Phase::Semantic,
            prefix: &PREFIX,
            sampling: &s,
            guidance: Some(Guidance {
                negative: &[EOD, 77, MUSIC_START],
                scale: 1.5,
            }),
            legacy_off: false,
        };
        let (out, rec) = run(&lm, &req, &mut stage_rng(1), &|| true);
        assert!(matches!(out, Err(gen_core::Error::Canceled)));
        assert!(rec.logits.is_empty(), "cancelled before the prefill");

        let polls = Cell::new(0usize);
        let tokens = Cell::new(0usize);
        struct Count<'a>(&'a Cell<usize>);
        impl DecodeObserver for Count<'_> {
            fn on_token(&mut self, _: Phase, _: u32) {
                self.0.set(self.0.get() + 1);
            }
        }
        let cancel_after_two = || {
            polls.set(polls.get() + 1);
            tokens.get() >= 2
        };
        let out = decode_tokens(
            &lm,
            &req,
            &mut stage_rng(1),
            Hooks {
                cancelled: &cancel_after_two,
                observer: &mut Count(&tokens),
            },
        );
        assert!(matches!(out, Err(gen_core::Error::Canceled)));
        assert_eq!(tokens.get(), 2, "no token after the flag tripped");
        // before-prefill + positive chunk + before-negative + negative chunk + 3 step checks.
        assert_eq!(polls.get(), 7);
    }

    /// Guidance runs the negative branch and mixes it in; scale 1 runs one branch and samples the
    /// conditional row unchanged.
    #[test]
    fn guidance_mixes_the_negative_branch() {
        let lm = synthetic::model(MotPaths::Ar);
        let s = greedy(3);
        let negative = [EOD, 77, 78, ABC_START, ABC_END, MUSIC_START];
        let plain = DecodeRequest {
            phase: Phase::Semantic,
            prefix: &PREFIX,
            sampling: &s,
            guidance: None,
            legacy_off: false,
        };
        let (a, ra) = run(&lm, &plain, &mut stage_rng(1), &never);
        let one = DecodeRequest {
            guidance: Some(Guidance {
                negative: &negative,
                scale: 1.0,
            }),
            ..plain
        };
        let (b, rb) = run(&lm, &one, &mut stage_rng(1), &never);
        assert_eq!(a.as_ref().unwrap().cfg_branches, 1);
        assert_eq!(b.as_ref().unwrap().cfg_branches, 1);
        assert_eq!(ra.logits, rb.logits);
        let cfg = DecodeRequest {
            guidance: Some(Guidance {
                negative: &negative,
                scale: 3.0,
            }),
            ..plain
        };
        let (c, rc) = run(&lm, &cfg, &mut stage_rng(1), &never);
        assert_eq!(c.unwrap().cfg_branches, 2);
        // Step 0: mixed = uncond + 3 (cond − uncond) with uncond from the negative prefill.
        let mut neg_cache = lm.new_cache(negative.len() + 3).unwrap();
        let uncond = host_row(&lm.prefill(&negative, &mut neg_cache, || Ok(())).unwrap()).unwrap();
        let want = cfg_mix(&ra.logits[0], &uncond, 3.0, Arith::F32);
        assert_eq!(rc.logits[0], want);
    }

    fn tok() -> &'static Yue2TextTokenizer {
        test_fixtures::synthetic()
    }

    fn request(cot: CotMode, abc: Option<&str>, cfg_scale: Option<f64>) -> SongRequest {
        let mut spec = SongRequestSpec::new("warm piano pop, 88 BPM", "[Verse]\nNeon fades\n");
        spec.cot = cot;
        spec.abc = abc.map(str::to_string);
        spec.cfg_scale = cfg_scale;
        SongRequest::new(spec).unwrap()
    }

    fn ready(request: SongRequest) -> SymbolicPlan {
        match SymbolicPlan::prepare(request, tok()).unwrap() {
            PlanStep::Ready(plan) => plan,
            PlanStep::GenerateAbc(_) => panic!("expected a ready plan"),
        }
    }

    const SCORE: &str = "X:1\nL:1/16\nK:C\nV: Vocal\nE2G2A2G2|\n";

    fn config(semantic: Sampling) -> GenerationConfig {
        GenerationConfig::new(test_sampling(0.0, 0.9, 30, 1.005, 100, 0, 4), semantic, 32).unwrap()
    }

    fn semantic(
        lm: &Yue2Lm,
        plan: &SymbolicPlan,
        conditioning: &SemanticConditioning,
        config: &GenerationConfig,
    ) -> gen_core::Result<SemanticTokens> {
        generate_semantic(
            lm,
            plan,
            conditioning,
            config,
            &mut stage_rng(1),
            Hooks {
                cancelled: &never,
                observer: &mut (),
            },
        )
    }

    /// `ids` with `id` inserted at `at`.
    fn with(ids: &[u32], at: usize, id: u32) -> Vec<u32> {
        let mut v = ids.to_vec();
        v.insert(at, id);
        v
    }

    fn refused(result: gen_core::Result<SemanticTokens>, needle: &str, what: &str) {
        match result {
            Ok(_) => panic!("accepted {what}"),
            Err(e) => assert!(e.to_string().contains(needle), "{what}: {e}"),
        }
    }

    /// The semantic stage takes exactly the plan's conditioning: under guidance the negative must
    /// carry the plan's exact score; between `EOD` and the framing both prefixes hold ordinary text
    /// ids only; and scale, legacy flag and positive prefix must be the plan's.
    #[test]
    fn semantic_guidance_requires_the_exact_planned_score() {
        let lm = synthetic::model(MotPaths::Ar);
        let cfg = config(greedy(2));
        let plan = ready(request(CotMode::Full, Some(SCORE), Some(1.5)));
        let good = plan.semantic_conditioning(tok(), cfg.semantic()).unwrap();
        let ok = semantic(&lm, &plan, &good, &cfg).unwrap();
        assert_eq!(ok.decoded.cfg_branches, 2);
        assert!(ok.codes.iter().all(|&c| c < CODEC_SIZE));

        let neg = good.negative.clone().unwrap();
        let tail = semantic_tail(&plan);
        let body = &neg[1..neg.len() - tail.len()];
        let n = plan.abc_ids().len();
        let negative = |ids: Vec<u32>| SemanticConditioning {
            negative: Some(ids),
            ..good.clone()
        };
        let mut other_score = neg.clone();
        other_score[neg.len() - 3] += 1;
        let mut short_score = neg.clone();
        short_score.remove(neg.len() - 3);
        let mut no_score = vec![EOD];
        no_score.extend_from_slice(body);
        no_score.push(MUSIC_START);
        for (ids, needle, what) in [
            (no_score, "framing", "a negative without the score"),
            (other_score, "framing", "a negative with a different score"),
            (short_score, "framing", "a negative with a truncated score"),
            (neg[1..].to_vec(), "EOD", "a negative without EOD"),
            (
                with(&neg, 1, ABC_START),
                "ordinary",
                "score framing inside the negative body",
            ),
            (
                with(&neg, 2, MUSIC_START),
                "ordinary",
                "a stray MUSIC_START in the negative body",
            ),
        ] {
            refused(semantic(&lm, &plan, &negative(ids), &cfg), needle, what);
        }
        assert!(n > 0);

        let positive = |ids: Vec<u32>| SemanticConditioning {
            positive: ids,
            ..good.clone()
        };
        let mut other_text = good.positive.clone();
        other_text[1] = (other_text[1] + 1) % EOD;
        for (cond, needle, what) in [
            (
                positive(with(&good.positive, 1, ABC_END)),
                "ordinary",
                "a special id in the positive body",
            ),
            (
                positive(other_text),
                "exact prefix",
                "another request's positive prefix",
            ),
            (
                SemanticConditioning {
                    negative: None,
                    ..good.clone()
                },
                "requires a negative",
                "guidance without a negative",
            ),
            (
                SemanticConditioning {
                    cfg_scale: 2.0,
                    ..good.clone()
                },
                "guidance",
                "a scale that is not the plan's",
            ),
            (
                SemanticConditioning {
                    cfg_scale: f64::NAN,
                    ..good.clone()
                },
                "finite",
                "a NaN scale",
            ),
            (
                SemanticConditioning {
                    legacy_off: true,
                    ..good.clone()
                },
                "legacy_off",
                "legacy arithmetic outside cot = off",
            ),
        ] {
            refused(semantic(&lm, &plan, &cond, &cfg), needle, what);
        }

        // Without guidance: one branch, and a stray negative is refused.
        let plain = ready(request(CotMode::Full, Some(SCORE), None));
        let cond = plain.semantic_conditioning(tok(), cfg.semantic()).unwrap();
        assert!(cond.negative.is_none());
        assert_eq!(
            semantic(&lm, &plain, &cond, &cfg)
                .unwrap()
                .decoded
                .cfg_branches,
            1
        );
        let stray = SemanticConditioning {
            negative: Some(neg),
            ..cond
        };
        refused(
            semantic(&lm, &plain, &stray, &cfg),
            "without guidance",
            "a negative at scale 1",
        );
    }

    /// The `cot = off` semantic stage samples with the historical arithmetic (top-p keeps three),
    /// the planned modes with the standard one (keeps one): with a vanishing `top_p` and a draw
    /// near 1, only the legacy rule can return anything but the top-1 id.
    #[test]
    fn off_semantic_stage_uses_the_legacy_sampler() {
        let lm = synthetic::model(MotPaths::Ar);
        let cfg = config(test_sampling(1.0, 1e-6, 100, 1.0, 1, 0, 1));
        let stage = |plan: &SymbolicPlan| {
            let cond = plan.semantic_conditioning(tok(), cfg.semantic()).unwrap();
            generate_semantic(
                &lm,
                plan,
                &cond,
                &cfg,
                &mut Uniforms([0.999].into()),
                Hooks {
                    cancelled: &never,
                    observer: &mut (),
                },
            )
            .unwrap()
            .decoded
            .tokens
        };
        let direct = |prefix: &[u32], legacy_off: bool| {
            let req = DecodeRequest {
                phase: Phase::Semantic,
                prefix,
                sampling: cfg.semantic(),
                guidance: None,
                legacy_off,
            };
            run(&lm, &req, &mut Uniforms([0.999].into()), &never)
                .0
                .unwrap()
                .tokens
        };
        let off = ready(request(CotMode::Off, None, Some(1.0)));
        assert_ne!(
            direct(off.prefix(), true),
            direct(off.prefix(), false),
            "the two rules must differ here"
        );
        assert_eq!(stage(&off), direct(off.prefix(), true));
        let full = ready(request(CotMode::Full, Some(SCORE), None));
        assert_eq!(stage(&full), direct(full.prefix(), false));
    }

    /// `cot = off` guidance is instruction-only: its negative is `EOD`, ordinary instruction ids and
    /// `MUSIC_START` — no score framing and no stray specials anywhere.
    #[test]
    fn off_mode_negative_is_instruction_only() {
        let lm = synthetic::model(MotPaths::Ar);
        let cfg = config(greedy(2));
        let plan = ready(request(CotMode::Off, None, None));
        let good = plan.semantic_conditioning(tok(), cfg.semantic()).unwrap();
        assert_eq!(good.cfg_scale, CotMode::Off.default_guidance());
        let ok = semantic(&lm, &plan, &good, &cfg).unwrap();
        assert_eq!(ok.decoded.cfg_branches, 2, "off defaults to guidance 1.01");
        let neg = good.negative.clone().unwrap();
        let last = neg.len() - 1;
        let negative = |ids: Vec<u32>| SemanticConditioning {
            negative: Some(ids),
            ..good.clone()
        };
        let mut framed = neg[..last].to_vec();
        framed.extend_from_slice(&[ABC_START, ABC_END, MUSIC_START]);
        for (ids, needle, what) in [
            (framed, "ordinary", "score framing in an off negative"),
            (
                with(&neg, last, ABC_END),
                "ordinary",
                "a stray ABC_END in an off negative",
            ),
            (
                with(&neg, 1, MUSIC_START),
                "ordinary",
                "a stray MUSIC_START in an off negative",
            ),
            (neg[1..].to_vec(), "EOD", "an off negative without EOD"),
            (
                vec![EOD, MUSIC_START],
                "instruction",
                "an off negative without instruction",
            ),
        ] {
            refused(semantic(&lm, &plan, &negative(ids), &cfg), needle, what);
        }
        let special = SemanticConditioning {
            positive: with(&good.positive, 1, MUSIC_END),
            ..good.clone()
        };
        refused(
            semantic(&lm, &plan, &special, &cfg),
            "ordinary",
            "a special id in an off positive body",
        );
    }

    /// The whole symbolic path on the synthetic model, for every `cot`:
    /// [`SymbolicPlan::prepare`] → [`plan_score`] (sampling the planner's own prefix) →
    /// [`AbcPlanning::finish`] → [`SymbolicPlan::semantic_conditioning`] → [`generate_semantic`].
    ///
    /// The ABC stage draws with `u = 0` over the whole allowed range (the lowest id with mass,
    /// id 0), so the planned ids stay inside the committed test tokenizer's partial rank table,
    /// which `finish` decodes; that the planner sampled `planning.prefix()` is shown by its logits
    /// rows equalling a direct decode of that prefix.
    #[test]
    fn symbolic_plan_drives_both_stages_end_to_end() {
        let lm = synthetic::model(MotPaths::Ar);
        let abc_steps = 4;
        let cfg = GenerationConfig::new(
            test_sampling(1.0, 1.0, VOCAB_SIZE.into(), 1.0, 1, 0, abc_steps),
            test_sampling(0.0, 0.95, 100, 1.2, 50, 3, 3),
            32,
        )
        .unwrap();
        let zeros = || Uniforms(vec![0.0; abc_steps as usize].into());
        for (cot, cfg_scale, branches) in [
            (CotMode::Full, None, 1),
            (CotMode::Melody, Some(1.5), 2),
            (CotMode::Off, None, 2),
        ] {
            let plan = match SymbolicPlan::prepare(request(cot, None, cfg_scale), tok()).unwrap() {
                PlanStep::GenerateAbc(planning) => {
                    let prefix = planning.prefix().to_vec();
                    assert_eq!(prefix.last(), Some(&ABC_START));
                    let direct_req = DecodeRequest {
                        phase: Phase::Abc,
                        prefix: &prefix,
                        sampling: cfg.abc(),
                        guidance: None,
                        legacy_off: false,
                    };
                    let (direct, direct_rows) = run(&lm, &direct_req, &mut zeros(), &never);
                    let mut rows = Record::default();
                    let (plan, decoded) = plan_score(
                        &lm,
                        planning,
                        tok(),
                        &cfg,
                        &mut zeros(),
                        Hooks {
                            cancelled: &never,
                            observer: &mut rows,
                        },
                    )
                    .unwrap();
                    assert_eq!(decoded.prefix_tokens, prefix.len());
                    assert_eq!(rows.logits, direct_rows.logits, "{cot:?}: planner prefix");
                    assert_eq!(decoded.tokens, direct.unwrap().tokens);
                    assert_eq!(decoded.tokens.len(), abc_steps as usize);
                    assert_eq!(plan.abc_ids(), &decoded.tokens[..]);
                    assert_eq!(plan.truncated(), decoded.truncated);
                    assert!(plan.truncated());
                    assert_eq!(
                        plan.timing()["output_tokens"],
                        Value::from(decoded.output_tokens)
                    );
                    plan
                }
                PlanStep::Ready(plan) => {
                    assert_eq!(cot, CotMode::Off);
                    plan
                }
            };
            assert!(plan.prefix().ends_with(&semantic_tail(&plan)));
            let cond = plan.semantic_conditioning(tok(), cfg.semantic()).unwrap();
            let out = semantic(&lm, &plan, &cond, &cfg).unwrap();
            assert_eq!(out.decoded.cfg_branches, branches, "{cot:?}");
            assert_eq!(out.decoded.prefix_tokens, plan.prefix().len());
            assert_eq!(out.codes.len(), 3);
            assert!(out.decoded.truncated);
            assert!(out.codes.iter().all(|&c| c < CODEC_SIZE));
        }
    }
}
