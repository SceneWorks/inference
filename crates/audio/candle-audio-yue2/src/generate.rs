//! Autoregressive score (ABC) and semantic-token generation over the YuE2 MoT backbone
//! (sc-22991).
//!
//! Ported from the pinned upstream `src/yue2/sampling.py::generate_tokens` and the `plan` /
//! `generate_semantic` stages of `src/yue2/pipeline.py`. The model and sampler consume **token
//! ids**: the tokenizer and the request/prompt assembly (`token_prefixes`, `negative_prefix`)
//! produce the prefixes these functions take.
//!
//! # One decode ([`decode_tokens`])
//!
//! * The prefix plus the output budget must fit the released [`CONTEXT`]; a longer request is an
//!   error, never an implicit truncation, window or re-prefill. The KV cache is sized to exactly
//!   `prefix + max_tokens` positions and positions are the cache slots `0, 1, 2, …`.
//! * With guidance (`scale != 1`) the negative prefix gets its own cache; every step runs both
//!   branches and samples from `uncond + scale · (cond − uncond)` in the model dtype.
//! * A step shapes the row with [`sampling::distribution`], draws (or takes the argmax), reports
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
//! [`plan_score`] generates the score for `cot = full | melody` from the planner prefix
//! (`… ABC_START`); [`ScorePlan::supplied`] takes an external score's exact ids instead, and
//! [`ScorePlan::off`] has none. [`generate_semantic`] generates the codec tokens from the semantic
//! prefix, which must frame **exactly** the plan's ABC ids (`ABC_START abc… ABC_END MUSIC_START`);
//! under guidance the negative prefix must frame the very same ids (upstream's
//! `same_instruction_and_exact_abc`), or — for `cot = off`, which has no score — none at all
//! (`instruction_only`). Both stages restart their random stream from the request seed
//! ([`stage_rng`]), as upstream does.

use std::time::Instant;

use candle_audio::candle_core::DType;
use candle_audio::gen_core;
use candle_llm::primitives::StaticKvCache;

use crate::model::{backend, Yue2Lm};
use crate::sampling::{
    cfg_mix, distribution, next_token, Arith, Phase, Sampling, SplitMix64, TokenRng, ABC_END,
    ABC_START, CODEC_OFFSET, CONTEXT, EOD, MUSIC_START,
};

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
    sampling.validate()?;
    let model_arith = arith_of(lm.dtype())?;
    let score_arith = if request.legacy_off {
        model_arith
    } else {
        Arith::F32
    };
    let budget = sampling.max_tokens;
    let fits = |len: usize| len.checked_add(budget).is_some_and(|n| n <= CONTEXT);
    if !fits(request.prefix.len()) {
        return Err(gen_core::Error::Msg(format!(
            "YuE2: prefix ({}) + requested generation budget ({budget}) exceeds the {CONTEXT}-token \
             context; no implicit truncation",
            request.prefix.len()
        )));
    }
    let guidance = match request.guidance {
        Some(g) if g.scale != 1.0 => {
            if !g.scale.is_finite() {
                return Err(gen_core::Error::Msg(format!(
                    "YuE2: guidance scale {} is not finite",
                    g.scale
                )));
            }
            if !fits(g.negative.len()) {
                return Err(gen_core::Error::Msg(format!(
                    "YuE2: negative prefix ({}) + generation budget ({budget}) exceeds the \
                     {CONTEXT}-token context",
                    g.negative.len()
                )));
            }
            Some(g)
        }
        _ => None,
    };
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

/// The symbolic-planning mode (upstream `SongRequest.cot`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cot {
    /// Plan melody and chords.
    Full,
    /// Plan the melody only.
    Melody,
    /// No symbolic plan.
    Off,
}

impl Cot {
    /// Upstream's default guidance (`SongRequest.guidance` with no `cfg_scale`): `1.01` for `off`
    /// (instruction-only negative), `1.0` — no guidance — otherwise.
    pub fn default_cfg_scale(self) -> f64 {
        match self {
            Cot::Off => 1.01,
            Cot::Full | Cot::Melody => 1.0,
        }
    }
}

/// Where a plan's score came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanSource {
    /// `cot = off`: no score.
    None,
    /// An external score's exact ids.
    Supplied,
    /// Generated by [`plan_score`]; `truncated` when the budget ran out before `ABC_END`.
    Planned {
        /// The planner hit its token budget.
        truncated: bool,
    },
}

/// The symbolic plan the semantic stage conditions on: the exact ABC ids (no decode/re-encode).
#[derive(Clone, Debug, PartialEq)]
pub struct ScorePlan {
    cot: Cot,
    abc_ids: Vec<u32>,
    source: PlanSource,
}

fn check_abc_ids(ids: &[u32]) -> gen_core::Result<()> {
    if let Some(bad) = ids.iter().find(|&&t| t >= EOD) {
        return Err(gen_core::Error::Msg(format!(
            "YuE2: ABC id {bad} is outside the ordinary text vocabulary [0, {EOD})"
        )));
    }
    Ok(())
}

impl ScorePlan {
    /// `cot = off`: no score.
    pub fn off() -> Self {
        Self {
            cot: Cot::Off,
            abc_ids: Vec::new(),
            source: PlanSource::None,
        }
    }

    /// An external score (`cot = full | melody`), as its exact tokenizer ids.
    pub fn supplied(cot: Cot, abc_ids: Vec<u32>) -> gen_core::Result<Self> {
        if cot == Cot::Off || abc_ids.is_empty() {
            return Err(gen_core::Error::Msg(
                "YuE2: an external score requires cot = full | melody and a non-empty score".into(),
            ));
        }
        check_abc_ids(&abc_ids)?;
        Ok(Self {
            cot,
            abc_ids,
            source: PlanSource::Supplied,
        })
    }

    /// The planning mode.
    pub fn cot(&self) -> Cot {
        self.cot
    }

    /// The exact ABC ids.
    pub fn abc_ids(&self) -> &[u32] {
        &self.abc_ids
    }

    /// Where the score came from.
    pub fn source(&self) -> PlanSource {
        self.source
    }

    /// Whether the planner hit its budget.
    pub fn truncated(&self) -> bool {
        matches!(self.source, PlanSource::Planned { truncated: true })
    }

    /// The tail every semantic prefix of this plan ends with: `ABC_START abc… ABC_END
    /// MUSIC_START` (for `off`, the empty score `ABC_START ABC_END MUSIC_START`).
    pub fn semantic_tail(&self) -> Vec<u32> {
        let mut tail = Vec::with_capacity(self.abc_ids.len() + 3);
        tail.push(ABC_START);
        tail.extend_from_slice(&self.abc_ids);
        tail.extend_from_slice(&[ABC_END, MUSIC_START]);
        tail
    }

    /// The tail a guidance negative prefix of this plan ends with: the same exact score for
    /// `full | melody`, only `MUSIC_START` for `off`.
    pub fn negative_tail(&self) -> Vec<u32> {
        match self.cot {
            Cot::Off => vec![MUSIC_START],
            Cot::Full | Cot::Melody => self.semantic_tail(),
        }
    }
}

/// Generate the score for `cot = full | melody` from the planner prefix (`EOD … ABC_START`), with
/// the ABC phase's own sampling controls.
pub fn plan_score(
    lm: &Yue2Lm,
    cot: Cot,
    planner_prefix: &[u32],
    sampling: &Sampling,
    rng: &mut dyn TokenRng,
    hooks: Hooks<'_>,
) -> gen_core::Result<(ScorePlan, Decoded)> {
    if cot == Cot::Off {
        return Err(gen_core::Error::Msg(
            "YuE2: cot = off has no score to plan (use ScorePlan::off)".into(),
        ));
    }
    if planner_prefix.first() != Some(&EOD) || planner_prefix.last() != Some(&ABC_START) {
        return Err(gen_core::Error::Msg(
            "YuE2: the planner prefix must be `EOD … ABC_START`".into(),
        ));
    }
    let decoded = decode_tokens(
        lm,
        &DecodeRequest {
            phase: Phase::Abc,
            prefix: planner_prefix,
            sampling,
            guidance: None,
            legacy_off: false,
        },
        rng,
        hooks,
    )?;
    check_abc_ids(&decoded.tokens)?;
    let plan = ScorePlan {
        cot,
        abc_ids: decoded.tokens.clone(),
        source: PlanSource::Planned {
            truncated: decoded.truncated,
        },
    };
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

/// The semantic stage's conditioning: the plan, the positive prefix, and the guidance.
#[derive(Clone, Copy, Debug)]
pub struct SemanticInput<'a> {
    /// The plan the prefix frames.
    pub plan: &'a ScorePlan,
    /// The positive prefix: `EOD …` ending in [`ScorePlan::semantic_tail`].
    pub prefix: &'a [u32],
    /// The guidance negative prefix (`EOD …` ending in [`ScorePlan::negative_tail`]); required
    /// when `cfg_scale != 1`, ignored when it is exactly `1`.
    pub negative: Option<&'a [u32]>,
    /// The guidance scale, in `[0, 20]` ([`Cot::default_cfg_scale`] is the released default).
    pub cfg_scale: f64,
}

/// Generate the semantic codec tokens (see the module docs for the prefix contract).
pub fn generate_semantic(
    lm: &Yue2Lm,
    input: &SemanticInput<'_>,
    sampling: &Sampling,
    rng: &mut dyn TokenRng,
    hooks: Hooks<'_>,
) -> gen_core::Result<SemanticTokens> {
    let plan = input.plan;
    let refuse = |what: String| Err(gen_core::Error::Msg(format!("YuE2 semantic stage: {what}")));
    if !(input.cfg_scale.is_finite() && (0.0..=20.0).contains(&input.cfg_scale)) {
        return refuse(format!(
            "cfg_scale {} must be finite and in [0, 20]",
            input.cfg_scale
        ));
    }
    let tail = plan.semantic_tail();
    if input.prefix.first() != Some(&EOD) || !input.prefix.ends_with(&tail) {
        return refuse(
            "the prefix must start with EOD and end with the plan's exact score framing \
             (ABC_START abc… ABC_END MUSIC_START)"
                .into(),
        );
    }
    let guidance = if input.cfg_scale == 1.0 {
        None
    } else {
        let Some(negative) = input.negative else {
            return refuse("guidance (cfg_scale != 1) requires a negative prefix".into());
        };
        let neg_tail = plan.negative_tail();
        let body = &negative[..negative.len().saturating_sub(neg_tail.len())];
        let framed = negative.first() == Some(&EOD) && negative.ends_with(&neg_tail);
        // `off` negatives are instruction-only: no score framing anywhere before MUSIC_START.
        let stray_score = plan.cot == Cot::Off && body.contains(&ABC_START);
        if !framed || stray_score || negative.len() <= neg_tail.len() {
            return refuse(match plan.cot {
                Cot::Off => "a cot = off negative must be `EOD instruction… MUSIC_START` with no \
                             score"
                    .into(),
                _ => "the negative prefix must retain the plan's exact score \
                      (EOD instruction… ABC_START abc… ABC_END MUSIC_START)"
                    .into(),
            });
        }
        Some(Guidance {
            negative,
            scale: input.cfg_scale,
        })
    };
    let decoded = decode_tokens(
        lm,
        &DecodeRequest {
            phase: Phase::Semantic,
            prefix: input.prefix,
            sampling,
            guidance,
            legacy_off: plan.cot == Cot::Off,
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
    use crate::sampling::{self, CODEC_SIZE, MUSIC_END};

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

    fn greedy(max_tokens: usize) -> Sampling {
        Sampling {
            temperature: 0.0,
            min_tokens: 0,
            max_tokens,
            ..Sampling::SEMANTIC_DEFAULT
        }
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
        let stochastic = Sampling {
            temperature: 1.0,
            top_k: sampling::VOCAB_SIZE,
            top_p: 1.0,
            repetition_penalty: 1.0,
            min_tokens: 0,
            max_tokens: 4,
            ..Sampling::SEMANTIC_DEFAULT
        };
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
        let s = Sampling {
            temperature: 1.0,
            top_k: sampling::VOCAB_SIZE,
            top_p: 1.0,
            repetition_penalty: 1.0,
            min_tokens: 3,
            max_tokens: 3,
            ..Sampling::SEMANTIC_DEFAULT
        };
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
        assert!(err.contains("no implicit truncation"), "{err}");
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
        assert!(run(&lm, &req, &mut stage_rng(1), &never).0.is_err());
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

    fn plan_of(ids: &[u32]) -> ScorePlan {
        ScorePlan::supplied(Cot::Full, ids.to_vec()).unwrap()
    }

    /// The semantic stage refuses a negative branch that does not carry the plan's exact score,
    /// and a positive prefix that does not frame it.
    #[test]
    fn semantic_guidance_requires_the_exact_planned_score() {
        let lm = synthetic::model(MotPaths::Ar);
        let abc = [11u32, 12, 13];
        let plan = plan_of(&abc);
        let prefix = [EOD, 5, 6, ABC_START, 11, 12, 13, ABC_END, MUSIC_START];
        let good_negative = [EOD, 5, ABC_START, 11, 12, 13, ABC_END, MUSIC_START];
        let s = greedy(2);
        let go = |prefix: &[u32], negative: Option<&[u32]>, scale: f64| {
            generate_semantic(
                &lm,
                &SemanticInput {
                    plan: &plan,
                    prefix,
                    negative,
                    cfg_scale: scale,
                },
                &s,
                &mut stage_rng(1),
                Hooks {
                    cancelled: &never,
                    observer: &mut (),
                },
            )
        };
        let ok = go(&prefix, Some(&good_negative), 1.5).unwrap();
        assert_eq!(ok.decoded.cfg_branches, 2);
        assert!(ok.codes.iter().all(|&c| c < CODEC_SIZE));
        // Negative missing the score, carrying a different score, or a truncated score.
        for bad in [
            &[EOD, 5, MUSIC_START][..],
            &[EOD, 5, ABC_START, 11, 12, 14, ABC_END, MUSIC_START],
            &[EOD, 5, ABC_START, 11, 12, ABC_END, MUSIC_START],
            &[ABC_START, 11, 12, 13, ABC_END, MUSIC_START],
        ] {
            assert!(
                go(&prefix, Some(bad), 1.5).is_err(),
                "accepted negative {bad:?}"
            );
        }
        assert!(
            go(&prefix, None, 1.5).is_err(),
            "guidance without a negative"
        );
        // Scale 1: no negative needed.
        assert_eq!(go(&prefix, None, 1.0).unwrap().decoded.cfg_branches, 1);
        // The positive prefix must frame the exact plan too.
        let wrong = [EOD, 5, 6, ABC_START, 11, 12, ABC_END, MUSIC_START];
        assert!(go(&wrong, None, 1.0).is_err());
        assert!(go(&prefix, None, f64::NAN).is_err());
        assert!(go(&prefix, None, 21.0).is_err());
    }

    #[test]
    fn off_mode_negative_is_instruction_only() {
        let lm = synthetic::model(MotPaths::Ar);
        let plan = ScorePlan::off();
        let s = greedy(2);
        let go = |negative: &[u32]| {
            generate_semantic(
                &lm,
                &SemanticInput {
                    plan: &plan,
                    prefix: &PREFIX,
                    negative: Some(negative),
                    cfg_scale: Cot::Off.default_cfg_scale(),
                },
                &s,
                &mut stage_rng(1),
                Hooks {
                    cancelled: &never,
                    observer: &mut (),
                },
            )
        };
        let ok = go(&[EOD, 9, 10, MUSIC_START]).unwrap();
        assert_eq!(ok.decoded.cfg_branches, 2, "off defaults to guidance 1.01");
        assert!(go(&[EOD, 9, ABC_START, ABC_END, MUSIC_START]).is_err());
        assert!(go(&[EOD, MUSIC_START]).is_ok());
        assert!(go(&[MUSIC_START]).is_err());
    }

    #[test]
    fn planner_contract() {
        let lm = synthetic::model(MotPaths::Ar);
        let s = Sampling {
            temperature: 0.0,
            min_tokens: 0,
            max_tokens: 3,
            ..Sampling::ABC_DEFAULT
        };
        let prefix = [EOD, 40, 41, ABC_START];
        let mut unit = ();
        let (plan, dec) = plan_score(
            &lm,
            Cot::Melody,
            &prefix,
            &s,
            &mut stage_rng(1),
            Hooks {
                cancelled: &never,
                observer: &mut unit,
            },
        )
        .unwrap();
        assert_eq!(plan.abc_ids(), &dec.tokens[..]);
        assert_eq!(plan.truncated(), dec.truncated);
        assert!(plan.abc_ids().iter().all(|&t| t < EOD));
        for bad in [&[EOD, 40][..], &[40, ABC_START]] {
            assert!(plan_score(
                &lm,
                Cot::Full,
                bad,
                &s,
                &mut stage_rng(1),
                Hooks {
                    cancelled: &never,
                    observer: &mut ()
                }
            )
            .is_err());
        }
        assert!(plan_score(
            &lm,
            Cot::Off,
            &prefix,
            &s,
            &mut stage_rng(1),
            Hooks {
                cancelled: &never,
                observer: &mut ()
            }
        )
        .is_err());
        assert!(ScorePlan::supplied(Cot::Off, vec![1]).is_err());
        assert!(ScorePlan::supplied(Cot::Full, vec![]).is_err());
        assert!(ScorePlan::supplied(Cot::Full, vec![EOD]).is_err());
    }
}
