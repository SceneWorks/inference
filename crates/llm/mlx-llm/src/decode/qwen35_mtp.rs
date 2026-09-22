//! Native Qwen3.8 in-checkpoint multi-token prediction (MTP).
//!
//! The predictor proposes tokens, but the target Qwen35 model verifies every proposal. Greedy
//! decoding accepts the longest equal prefix; stochastic decoding uses exact rejection sampling.
//! Qwen35's DeltaNet cache cannot be truncated, so partial rejection restores a cloned pre-verify
//! cache and replays only the committed prefix.

use mlx_rs::transforms::eval;
use mlx_rs::Array;
use std::time::Instant;

use core_llm::speculative::{accept_token, sample_weighted, Acceptance};

use crate::decode::cancel::CancelFlag;
use crate::decode::speculative::{decide_greedy, decide_stochastic, logits_row, SpeculativeStats};
use crate::decode::stream::{
    default_seed, ConstraintMask, FinishReason, GenerationConfig, GenerationOutput,
    GenerationTimer, StreamEvent, TimedGenerationOutput,
};
use crate::decode::BufferRelease;
use crate::error::{Error, Result};
use crate::models::qwen35::Qwen35Model;
use crate::primitives::input_ids;
use crate::primitives::sampler::{sample, shaped_candidates, SplitMix64, TokenRng};

/// A Qwen3.8 multimodal prompt whose visual rows have already been encoded and fused into the
/// decoder input embeddings. MTP must seed from these embeddings and the explicit three-axis
/// M-RoPE positions; re-embedding the placeholder ids would change the target distribution.
pub struct Qwen35MtpMultimodalPrompt<'a> {
    pub input_ids: &'a [i32],
    pub embeddings: &'a Array,
    pub positions: [&'a [i32]; 3],
    pub visual_pos_mask: &'a [bool],
    pub deepstack: &'a [Array],
    pub continuation_delta: i32,
}

/// Constraint state that can be rewound after speculative exploration. Only emitted tokens remain
/// accepted after verification.
pub trait RewindableConstraintMask: ConstraintMask {
    /// Opaque checkpoint for the current committed state.
    fn checkpoint(&self) -> usize;
    /// Restore a checkpoint previously returned by [`Self::checkpoint`].
    fn rewind(&mut self, checkpoint: usize);
}

fn seq_rows(a: &Array, start: i32, len: i32) -> Result<Array> {
    let indices: Vec<i32> = (start..start + len).collect();
    Ok(a.take_axis(Array::from_slice(&indices, &[len]), 1)?)
}

/// Generate with Qwen3.8's native MTP predictor and exact target verification.
///
/// `num_draft` is the maximum speculative-token count per target pass. The caller has already
/// validated it against the loaded model's advertised MTP capability.
#[allow(clippy::too_many_arguments)]
pub fn generate_qwen35_mtp(
    model: &Qwen35Model,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    num_draft: usize,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn RewindableConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<(GenerationOutput, SpeculativeStats)> {
    let (output, stats, _) = generate_qwen35_mtp_inner(
        model,
        prompt_ids,
        config,
        num_draft,
        cancel,
        on_event,
        constraint,
        should_stop,
        None,
        None,
        false,
    )?;
    Ok((output, stats))
}

/// Synchronized two-phase Qwen3.8 MTP generation. The returned timer stays live until the provider
/// finishes detokenization and stream dispatch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_qwen35_mtp_with_timings(
    model: &Qwen35Model,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    num_draft: usize,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn RewindableConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<(TimedGenerationOutput, SpeculativeStats)> {
    let (output, stats, timer) = generate_qwen35_mtp_inner(
        model,
        prompt_ids,
        config,
        num_draft,
        cancel,
        on_event,
        constraint,
        should_stop,
        None,
        None,
        true,
    )?;
    Ok((
        TimedGenerationOutput {
            output,
            timer: timer.expect("timed MTP generation preserves its phase timer"),
        },
        stats,
    ))
}

/// Synchronized Qwen3.8 MTP generation from a fused image/video prompt.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_qwen35_mtp_multimodal_with_timings(
    model: &Qwen35Model,
    prompt: &Qwen35MtpMultimodalPrompt<'_>,
    config: &GenerationConfig,
    num_draft: usize,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn RewindableConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
    prefill_started: Instant,
) -> Result<(TimedGenerationOutput, SpeculativeStats)> {
    let (output, stats, timer) = generate_qwen35_mtp_inner(
        model,
        prompt.input_ids,
        config,
        num_draft,
        cancel,
        on_event,
        constraint,
        should_stop,
        Some(prompt),
        Some(prefill_started),
        true,
    )?;
    Ok((
        TimedGenerationOutput {
            output,
            timer: timer.expect("timed multimodal MTP preserves its phase timer"),
        },
        stats,
    ))
}

#[allow(clippy::too_many_arguments)]
fn generate_qwen35_mtp_inner(
    model: &Qwen35Model,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    num_draft: usize,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    mut constraint: Option<&mut dyn RewindableConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
    multimodal: Option<&Qwen35MtpMultimodalPrompt<'_>>,
    prefill_started: Option<Instant>,
    timed: bool,
) -> Result<(GenerationOutput, SpeculativeStats, Option<GenerationTimer>)> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate_qwen35_mtp: empty prompt".into()));
    }
    if !model.has_mtp() {
        return Err(Error::Msg(
            "generate_qwen35_mtp: model has no loaded MTP predictor".into(),
        ));
    }
    if num_draft == 0 {
        return Err(Error::Msg(
            "generate_qwen35_mtp: num_draft must be >= 1".into(),
        ));
    }

    let mut stats = SpeculativeStats::default();
    let mut generated = Vec::new();
    let mut finish = FinishReason::MaxTokens;
    if config.max_new_tokens == 0 {
        let mut timer = timed.then(|| {
            prefill_started.map_or_else(GenerationTimer::start, GenerationTimer::start_at)
        });
        if let Some(timer) = timer.as_mut() {
            timer.finish_prefill(std::iter::empty())?;
        }
        on_event(StreamEvent::Done {
            reason: finish,
            generated: 0,
        });
        return Ok((
            GenerationOutput {
                tokens: generated,
                finish_reason: finish,
            },
            stats,
            timer,
        ));
    }

    let mut rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let greedy = config.sampling.temperature <= 0.0;
    let mut target_cache = model.new_cache();
    // Validation, RNG setup, and empty-cache allocation are outside the measured backend phases.
    // Start immediately before the first target/predictor prompt-cache work.
    let mut timer = timed
        .then(|| prefill_started.map_or_else(GenerationTimer::start, GenerationTimer::start_at));
    let (prompt_hidden, prompt_logits) = match multimodal {
        Some(prompt) => model.prefill_hidden_and_last_logits_from_embeds_with_deepstack(
            prompt.embeddings,
            prompt.positions,
            &mut target_cache,
            prompt.visual_pos_mask,
            prompt.deepstack,
        )?,
        None => {
            model.prefill_hidden_and_last_logits(&input_ids(prompt_ids), &mut target_cache, 0)?
        }
    };
    stats.forwards += 1;

    // Seed MTP with shifted prompt pairs: embed(x[j+1]) + final-normalized target H[j], at
    // absolute positions 1..P-1. The next call pairs the first target-selected token with H[P-1].
    let prompt_len = prompt_ids.len() as i32;
    let mut mtp_cache = model
        .new_mtp_cache()
        .expect("has_mtp guarantees a predictor cache");
    let mtp_seed = if prompt_len > 1 {
        let shifted = match multimodal {
            Some(prompt) => seq_rows(prompt.embeddings, 1, prompt_len - 1)?,
            None => model.embed_input_ids(&input_ids(&prompt_ids[1..]))?,
        };
        let aligned = seq_rows(&prompt_hidden, 0, prompt_len - 1)?;
        let simple_positions;
        let positions = match multimodal {
            Some(prompt) => [
                &prompt.positions[0][1..],
                &prompt.positions[1][1..],
                &prompt.positions[2][1..],
            ],
            None => {
                simple_positions = (1..prompt_len).collect::<Vec<_>>();
                [
                    simple_positions.as_slice(),
                    simple_positions.as_slice(),
                    simple_positions.as_slice(),
                ]
            }
        };
        Some(model.mtp_warm_from_embeds(&shifted, &aligned, &mut mtp_cache, positions)?)
    } else {
        None
    };
    if let Some(timer) = timer.as_mut() {
        let mut arrays = vec![&prompt_hidden, &prompt_logits];
        if let Some(hidden) = mtp_seed.as_ref() {
            arrays.push(hidden);
        }
        timer.finish_prefill(arrays)?;
    }
    let mut last_target_hidden = seq_rows(&prompt_hidden, prompt_len - 1, 1)?;
    let first_logits = prompt_logits;
    let mut history = prompt_ids.to_vec();
    let first = {
        let mask = constraint.as_mut().map(|c| c.allowed());
        sample(&first_logits, &history, &config.sampling, &mut rng, mask)?
    };
    // The sample evaluated the target prefill; the predictor's warm-up graph is otherwise only
    // pulled by the first draft step, so force it here and release both prefills' transients.
    if let Some(hidden) = mtp_seed.as_ref() {
        eval([hidden])?;
    }
    drop(mtp_seed);
    let mut release = BufferRelease::after_prefill();
    if config.stop_tokens.contains(&first) {
        finish = FinishReason::StopToken;
        on_event(StreamEvent::Done {
            reason: finish,
            generated: 0,
        });
        return Ok((
            GenerationOutput {
                tokens: generated,
                finish_reason: finish,
            },
            stats,
            timer,
        ));
    }
    if let Some(c) = constraint.as_mut() {
        c.accept(first);
    }
    on_event(StreamEvent::Token { id: first, step: 0 });
    generated.push(first);
    history.push(first);
    if should_stop.is_some_and(|stop| stop()) {
        finish = FinishReason::Stopped;
    }
    let mut cur = first;
    let continuation_delta = multimodal.map_or(0, |prompt| prompt.continuation_delta);

    'outer: while generated.len() < config.max_new_tokens && finish != FinishReason::Stopped {
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }
        let remaining = config.max_new_tokens - generated.len();
        let k = num_draft.min(remaining.saturating_sub(1));
        let base_target = target_cache.offset();
        let target_base = target_cache.clone();
        let constraint_checkpoint = constraint.as_ref().map(|c| c.checkpoint());

        let mut drafts = Vec::with_capacity(k);
        let mut draft_dists = Vec::with_capacity(k);
        let mut draft_history = history.clone();
        let mut mtp_after_cur = None;
        if k > 0 {
            let (mut mtp_hidden, mut draft_logits) = model.mtp_step(
                &input_ids(&[cur]),
                &last_target_hidden,
                &mut mtp_cache,
                base_target + continuation_delta,
            )?;
            if timer.is_some() {
                eval([&mtp_hidden, &draft_logits])?;
            }
            mtp_after_cur = Some(mtp_cache.clone());

            for i in 0..k {
                if cancel.is_cancelled() {
                    stats.proposed += drafts.len();
                    if let (Some(c), Some(checkpoint)) =
                        (constraint.as_mut(), constraint_checkpoint)
                    {
                        c.rewind(checkpoint);
                    }
                    finish = FinishReason::Cancelled;
                    break 'outer;
                }
                if !greedy {
                    let mask = constraint.as_mut().map(|c| c.allowed());
                    draft_dists.push(shaped_candidates(
                        &draft_logits,
                        &draft_history,
                        &config.sampling,
                        mask,
                    )?);
                }
                let mask = constraint.as_mut().map(|c| c.allowed());
                let d = sample(
                    &draft_logits,
                    &draft_history,
                    &config.sampling,
                    &mut rng,
                    mask,
                )?;
                drafts.push(d);
                draft_history.push(d);
                if config.stop_tokens.contains(&d) {
                    break;
                }
                if let Some(c) = constraint.as_mut() {
                    c.accept(d);
                }
                if i + 1 < k {
                    let (h, q) = model.mtp_step(
                        &input_ids(&[d]),
                        &mtp_hidden,
                        &mut mtp_cache,
                        base_target + continuation_delta + 1 + i as i32,
                    )?;
                    if timer.is_some() {
                        eval([&h, &q])?;
                    }
                    mtp_hidden = h;
                    draft_logits = q;
                }
            }
        }
        stats.proposed += drafts.len();
        if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), constraint_checkpoint) {
            c.rewind(checkpoint);
        }

        // Verify current + all proposals in one target pass. The returned target hidden states are
        // also the authoritative inputs used to reconcile the predictor cache.
        let mut verify = Vec::with_capacity(1 + drafts.len());
        verify.push(cur);
        verify.extend_from_slice(&drafts);
        let (verify_hidden, target_logits) = model.hidden_and_logits(
            &input_ids(&verify),
            &mut target_cache,
            base_target + continuation_delta,
        )?;
        if timer.is_some() {
            eval([&verify_hidden, &target_logits])?;
        }
        stats.forwards += 1;
        let (committed, accepted) = if constraint.is_some() {
            decide_constrained(
                &target_logits,
                &drafts,
                &draft_dists,
                &history,
                config,
                &mut rng,
                greedy,
                constraint.as_deref_mut().expect("checked above"),
            )?
        } else if greedy {
            decide_greedy(&target_logits, &drafts, &history, config, &mut rng)?
        } else {
            decide_stochastic(
                &target_logits,
                &drafts,
                &draft_dists,
                &history,
                config,
                &mut rng,
            )?
        };
        if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), constraint_checkpoint) {
            c.rewind(checkpoint);
        }
        stats.accepted += accepted;

        // DeltaNet state is not invertible. A partial rejection restores the cloned base cache and
        // replays exactly current + the accepted prefix. Full acceptance keeps the trial cache.
        let kept_hidden = if accepted == drafts.len() {
            verify_hidden
        } else {
            target_cache = target_base;
            let kept = &verify[..1 + accepted];
            let (hidden, logits) = model.hidden_and_logits(
                &input_ids(kept),
                &mut target_cache,
                base_target + continuation_delta,
            )?;
            if timer.is_some() {
                eval([&hidden, &logits])?;
            }
            stats.forwards += 1;
            hidden
        };
        last_target_hidden = seq_rows(&kept_hidden, accepted as i32, 1)?;
        release.advance(committed.len());

        // Provisional recursive MTP hidden states never survive reconciliation. Restore the cache
        // after the target-selected current token, then replay accepted drafts paired with the
        // corresponding authoritative target predecessor states.
        if let Some(after_cur) = mtp_after_cur {
            mtp_cache = after_cur;
            for (i, &draft) in drafts[..accepted].iter().enumerate() {
                let predecessor = seq_rows(&kept_hidden, i as i32, 1)?;
                let (hidden, logits) = model.mtp_step(
                    &input_ids(&[draft]),
                    &predecessor,
                    &mut mtp_cache,
                    base_target + continuation_delta + 1 + i as i32,
                )?;
                if timer.is_some() {
                    eval([&hidden, &logits])?;
                }
            }
        }

        for &token in &committed {
            if config.stop_tokens.contains(&token) {
                finish = FinishReason::StopToken;
                break 'outer;
            }
            if let Some(c) = constraint.as_mut() {
                c.accept(token);
            }
            on_event(StreamEvent::Token {
                id: token,
                step: generated.len(),
            });
            generated.push(token);
            history.push(token);
            cur = token;
            if should_stop.is_some_and(|stop| stop()) {
                finish = FinishReason::Stopped;
                break 'outer;
            }
            if generated.len() >= config.max_new_tokens {
                finish = FinishReason::MaxTokens;
                break 'outer;
            }
        }
    }

    on_event(StreamEvent::Done {
        reason: finish,
        generated: generated.len(),
    });
    Ok((
        GenerationOutput {
            tokens: generated,
            finish_reason: finish,
        },
        stats,
        timer,
    ))
}

#[allow(clippy::too_many_arguments)]
fn decide_constrained(
    logits_all: &Array,
    drafts: &[i32],
    draft_dists: &[Vec<(i32, f32)>],
    history: &[i32],
    config: &GenerationConfig,
    rng: &mut SplitMix64,
    greedy: bool,
    constraint: &mut dyn RewindableConstraintMask,
) -> Result<(Vec<i32>, usize)> {
    let mut committed = Vec::with_capacity(drafts.len() + 1);
    let mut accepted = 0usize;
    let mut running_history = history.to_vec();

    for (i, &draft) in drafts.iter().enumerate() {
        let logits = logits_row(logits_all, i as i32)?;
        let selected = if greedy {
            let target = sample(
                &logits,
                &running_history,
                &config.sampling,
                rng,
                Some(constraint.allowed()),
            )?;
            if target == draft {
                Acceptance::Accepted(draft)
            } else {
                Acceptance::Rejected(target)
            }
        } else {
            let target = shaped_candidates(
                &logits,
                &running_history,
                &config.sampling,
                Some(constraint.allowed()),
            )?;
            accept_token(
                &target,
                &draft_dists[i],
                draft,
                rng.next_f32(),
                rng.next_f32(),
            )
        };
        let token = selected.token();
        committed.push(token);
        if selected.is_accepted() {
            accepted += 1;
        }
        if config.stop_tokens.contains(&token) {
            return Ok((committed, accepted));
        }
        constraint.accept(token);
        running_history.push(token);
        if !selected.is_accepted() {
            return Ok((committed, accepted));
        }
    }

    let logits = logits_row(logits_all, drafts.len() as i32)?;
    let mask = constraint.allowed();
    let bonus = if greedy {
        sample(&logits, &running_history, &config.sampling, rng, Some(mask))?
    } else {
        let target = shaped_candidates(&logits, &running_history, &config.sampling, Some(mask))?;
        sample_weighted(&target, rng.next_f32(), 0)
    };
    committed.push(bonus);
    Ok((committed, accepted))
}
