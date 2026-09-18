//! Native Qwen3.8 multi-token-prediction speculative decoding.
//!
//! The auxiliary MTP head proposes a configurable run of tokens and the Qwen target verifies the
//! whole run in one forward. Greedy decoding accepts the longest argmax-matching prefix; stochastic
//! decoding uses the shared distribution-preserving `p/q` acceptance rule. Qwen's hybrid target
//! cache contains non-invertible DeltaNet recurrent state, so a rejected suffix is recovered by
//! restoring a cheap reference-counted cache snapshot and replaying only the committed prefix.

use candle_core::Tensor;
use core_llm::speculative::{accept_token, sample_weighted, Acceptance};

use crate::decode::cancel::CancelFlag;
use crate::decode::speculative::{decide_greedy, decide_stochastic, SpeculativeStats};
use crate::decode::stream::{
    default_seed, ConstraintMask, Decode, FinishReason, GenerationConfig, GenerationOutput,
    StreamEvent,
};
use crate::error::{Error, Result};
use crate::models::{Qwen35Model, Qwen35Mtp};
use crate::primitives::input_ids;
use crate::primitives::sampler::{sample, shaped_candidates, SplitMix64, TokenRng};

/// Constraint state that can be rewound after speculative exploration. The committed state is
/// advanced only by tokens that are actually emitted.
pub trait RewindableConstraintMask: ConstraintMask {
    /// Opaque checkpoint for the current constraint state.
    fn checkpoint(&self) -> usize;
    /// Restore a checkpoint previously returned by [`Self::checkpoint`].
    fn rewind(&mut self, checkpoint: usize);
}

/// Fused Qwen3.8 multimodal prompt inputs for native MTP prefill.
pub struct Qwen35MtpMultimodalPrompt<'a> {
    /// Placeholder-expanded ids used for penalty history and usage alignment.
    pub input_ids: &'a [i32],
    /// Text embeddings with vision features spliced into placeholder rows.
    pub embeddings: &'a Tensor,
    /// Explicit interleaved M-RoPE temporal/height/width position rows.
    pub positions: [&'a [i32]; 3],
    /// Visual prompt mask used by target DeepStack fusion.
    pub visual_pos_mask: &'a [bool],
    /// Per-layer DeepStack features (empty for the Qwen3.8 tower).
    pub deepstack: &'a [Tensor],
    /// Position shift for post-prompt text tokens.
    pub continuation_delta: i32,
}

/// Generate with a checkpoint-native Qwen MTP predictor.
///
/// `num_draft` is an operational width, not the number of stored MTP layers. Qwen3.8 publishes one
/// layer and the upstream vLLM path cycles it autoregressively; the official recipe recommends
/// three drafts. The effective width is clamped to the remaining output budget each iteration.
#[allow(clippy::too_many_arguments)]
pub fn generate_qwen35_mtp(
    target: &Qwen35Model,
    mtp: &Qwen35Mtp,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    num_draft: u32,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn RewindableConstraintMask>,
) -> Result<(GenerationOutput, SpeculativeStats)> {
    generate_qwen35_mtp_inner(
        target, mtp, prompt_ids, None, config, num_draft, cancel, on_event, constraint, None,
    )
}

/// Timed-provider seam: invokes `on_prefill_complete` only after both target and MTP prompt caches
/// are populated. The callback may synchronize an asynchronous device before recording the split.
#[allow(clippy::too_many_arguments)]
pub fn generate_qwen35_mtp_timed(
    target: &Qwen35Model,
    mtp: &Qwen35Mtp,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    num_draft: u32,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn RewindableConstraintMask>,
    on_prefill_complete: &mut dyn FnMut() -> Result<()>,
) -> Result<(GenerationOutput, SpeculativeStats)> {
    generate_qwen35_mtp_inner(
        target,
        mtp,
        prompt_ids,
        None,
        config,
        num_draft,
        cancel,
        on_event,
        constraint,
        Some(on_prefill_complete),
    )
}

/// Multimodal twin of [`generate_qwen35_mtp`]. Both the target and predictor consume the same fused
/// vision embeddings and 3-D prompt positions before continuing with shifted one-dimensional text
/// positions.
#[allow(clippy::too_many_arguments)]
pub fn generate_qwen35_mtp_multimodal(
    target: &Qwen35Model,
    mtp: &Qwen35Mtp,
    prompt: Qwen35MtpMultimodalPrompt<'_>,
    config: &GenerationConfig,
    num_draft: u32,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn RewindableConstraintMask>,
) -> Result<(GenerationOutput, SpeculativeStats)> {
    generate_qwen35_mtp_inner(
        target,
        mtp,
        prompt.input_ids,
        Some(prompt),
        config,
        num_draft,
        cancel,
        on_event,
        constraint,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_qwen35_mtp_inner(
    target: &Qwen35Model,
    mtp: &Qwen35Mtp,
    prompt_ids: &[i32],
    multimodal: Option<Qwen35MtpMultimodalPrompt<'_>>,
    config: &GenerationConfig,
    num_draft: u32,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    mut constraint: Option<&mut dyn RewindableConstraintMask>,
    mut prefill_boundary: Option<&mut dyn FnMut() -> Result<()>>,
) -> Result<(GenerationOutput, SpeculativeStats)> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate_qwen35_mtp: empty prompt".into()));
    }
    if num_draft == 0 {
        return Err(Error::Msg(
            "generate_qwen35_mtp: num_draft must be at least one".into(),
        ));
    }

    let mut stats = SpeculativeStats::default();
    let mut generated = Vec::new();
    let mut finish = FinishReason::MaxTokens;
    if config.max_new_tokens == 0 {
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
        ));
    }

    let mut rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let greedy = config.sampling.temperature <= 0.0;
    let device = target.device();
    let mut target_cache = target.new_cache();
    let mut mtp_cache = mtp.new_cache();

    // Target prefill produces both the ordinary next-token logits and the final-normalized hidden
    // rows that train/infer the MTP pairing one position to the right.
    let (prompt_logits, prompt_hidden) = match multimodal.as_ref() {
        Some(prompt) => target.forward_from_embeds_deepstack_with_hidden(
            prompt.embeddings,
            prompt.positions,
            &mut target_cache,
            prompt.visual_pos_mask,
            prompt.deepstack,
        )?,
        None => {
            target.forward_with_hidden(&input_ids(prompt_ids, device)?, &mut target_cache, 0)?
        }
    };
    stats.forwards += 1;
    let prompt_len = prompt_ids.len();
    let logits_last = row3(&prompt_logits, prompt_len - 1)?;
    let mut previous_hidden = prompt_hidden.narrow(1, prompt_len - 1, 1)?;

    // Warm the MTP attention cache with the target-validated prompt pairs:
    // embed(token[j + 1]) is paired with target_hidden[j]. The first generated token is paired with
    // the last prompt hidden below, after ordinary target sampling.
    if prompt_len > 1 {
        let previous = prompt_hidden.narrow(1, 0, prompt_len - 1)?;
        match multimodal.as_ref() {
            Some(prompt) => {
                let shifted_embeddings = prompt.embeddings.narrow(1, 1, prompt_len - 1)?;
                let shifted_positions = [
                    &prompt.positions[0][1..],
                    &prompt.positions[1][1..],
                    &prompt.positions[2][1..],
                ];
                let _ = mtp.forward_embeddings_mrope(
                    &shifted_embeddings,
                    &previous,
                    shifted_positions,
                    &mut mtp_cache,
                )?;
            }
            None => {
                let _ = mtp.forward_sequence(&prompt_ids[1..], &previous, 1, &mut mtp_cache)?;
            }
        }
    }
    if let Some(boundary) = prefill_boundary.as_mut() {
        boundary()?;
    }

    let mut history = prompt_ids.to_vec();
    let first = {
        let mask = constraint.as_mut().map(|c| c.allowed());
        sample(&logits_last, &history, &config.sampling, &mut rng, mask)?
    };
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
        ));
    }
    on_event(StreamEvent::Token { id: first, step: 0 });
    if let Some(c) = constraint.as_mut() {
        c.accept(first);
    }
    generated.push(first);
    history.push(first);
    let mut cur = first;

    'outer: while generated.len() < config.max_new_tokens {
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }
        let remaining = config.max_new_tokens - generated.len();
        let requested = usize::try_from(num_draft).unwrap_or(usize::MAX);
        let k = requested.min(remaining.saturating_sub(1));
        let base_target = target_cache.offset();
        let continuation_delta = multimodal
            .as_ref()
            .map_or(0, |prompt| prompt.continuation_delta);
        let rope_position = base_target + continuation_delta;
        let constraint_checkpoint = constraint.as_ref().map(|c| c.checkpoint());

        // The first MTP step consumes `cur`, which is target-selected and therefore always valid.
        // Snapshot immediately afterwards; later draft feedback is provisional and is discarded
        // after verification even when its token is accepted, because replay must use target hidden
        // states rather than recursively drafted hidden states.
        let (mut draft_logits, mut feedback) =
            mtp.step(cur, &previous_hidden, 0, rope_position, &mut mtp_cache)?;
        let mtp_after_cur = mtp_cache.clone();
        let mut drafts = Vec::with_capacity(k);
        let mut draft_dists = Vec::with_capacity(k);
        let mut draft_history = history.clone();
        for step in 0..k {
            if cancel.is_cancelled() {
                stats.proposed += drafts.len();
                if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), constraint_checkpoint) {
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
            let draft = sample(
                &draft_logits,
                &draft_history,
                &config.sampling,
                &mut rng,
                mask,
            )?;
            drafts.push(draft);
            draft_history.push(draft);
            if !config.stop_tokens.contains(&draft) {
                if let Some(c) = constraint.as_mut() {
                    c.accept(draft);
                }
            } else {
                break;
            }
            if step + 1 < k {
                (draft_logits, feedback) = mtp.step(
                    draft,
                    &feedback,
                    step + 1,
                    rope_position + step as i32 + 1,
                    &mut mtp_cache,
                )?;
            }
        }
        if cancel.is_cancelled() {
            stats.proposed += drafts.len();
            if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), constraint_checkpoint) {
                c.rewind(checkpoint);
            }
            finish = FinishReason::Cancelled;
            break;
        }
        stats.proposed += drafts.len();
        if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), constraint_checkpoint) {
            c.rewind(checkpoint);
        }

        // Verify `[cur, drafts...]` in one target pass. The trial cache is committed only when all
        // drafts are accepted; otherwise the non-invertible hybrid state is restored and the kept
        // prefix is replayed from the snapshot.
        let mut verify = Vec::with_capacity(1 + drafts.len());
        verify.push(cur);
        verify.extend_from_slice(&drafts);
        let target_base = target_cache.clone();
        let mut trial_cache = target_base.clone();
        let (verify_logits, trial_hidden) = target.forward_with_hidden(
            &input_ids(&verify, device)?,
            &mut trial_cache,
            rope_position,
        )?;
        stats.forwards += 1;
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }

        let (committed, accepted) = if constraint.is_some() {
            decide_constrained(
                &verify_logits,
                &drafts,
                &draft_dists,
                &history,
                config,
                &mut rng,
                greedy,
                constraint.as_deref_mut().expect("checked above"),
            )?
        } else if greedy {
            decide_greedy(&verify_logits, &drafts, &history, config, &mut rng)?
        } else {
            decide_stochastic(
                &verify_logits,
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

        let keep_len = 1 + accepted;
        let kept_hidden = if accepted == drafts.len() {
            target_cache = trial_cache;
            trial_hidden
        } else {
            target_cache = target_base;
            let mut replay = Vec::with_capacity(keep_len);
            replay.push(cur);
            replay.extend_from_slice(&drafts[..accepted]);
            let (_, hidden) = target.forward_with_hidden(
                &input_ids(&replay, device)?,
                &mut target_cache,
                rope_position,
            )?;
            stats.forwards += 1;
            hidden
        };

        // Replace recursive draft state with target-confirmed state. `mtp_after_cur` includes the
        // always-valid current token; each accepted draft is replayed against the preceding target
        // hidden row. The bonus/correction stays unprocessed until the next iteration.
        mtp_cache = mtp_after_cur;
        if accepted > 0 {
            let preceding_hidden = kept_hidden.narrow(1, 0, accepted)?;
            let _ = mtp.forward_sequence(
                &drafts[..accepted],
                &preceding_hidden,
                rope_position + 1,
                &mut mtp_cache,
            )?;
        }
        previous_hidden = kept_hidden.narrow(1, keep_len - 1, 1)?;

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
            if cancel.is_cancelled() {
                finish = FinishReason::Cancelled;
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
    ))
}

fn row3(tensor: &Tensor, position: usize) -> Result<Tensor> {
    let (batch, _, vocab) = tensor.dims3()?;
    Ok(tensor.narrow(1, position, 1)?.reshape((batch, vocab))?)
}

#[allow(clippy::too_many_arguments)]
fn decide_constrained(
    logits_all: &Tensor,
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
        let logits = row3(logits_all, i)?;
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

    let logits = row3(logits_all, drafts.len())?;
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};
    use serde_json::json;

    use super::*;
    use crate::decode::generate_with;
    use crate::models::Qwen35Config;
    use crate::primitives::Weights;

    fn tensor(map: &mut HashMap<String, Tensor>, key: &str, data: Vec<f32>, dims: &[usize]) {
        map.insert(
            key.to_string(),
            Tensor::from_vec(data, dims.to_vec(), &Device::Cpu).unwrap(),
        );
    }

    fn zero(map: &mut HashMap<String, Tensor>, key: &str, dims: &[usize]) {
        tensor(map, key, vec![0.0; dims.iter().product()], dims);
    }

    fn fixture(aligned_predictor: bool) -> (Qwen35Model, Qwen35Mtp) {
        let cfg = Qwen35Config::from_json(&json!({
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 12,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 6,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "max_position_embeddings": 64,
                "tie_word_embeddings": false,
                "full_attention_interval": 1,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false
            }
        }))
        .unwrap();
        let (h, v, inter, nh, nkv, hd) = (8usize, 6usize, 12usize, 2usize, 1usize, 4usize);
        let mut map = HashMap::new();

        // Every token embeds to the same direction. With zero decoder projections, the target
        // residual path preserves that direction; the shared LM head therefore chooses token 1.
        let mut embeddings = vec![0.0f32; v * h];
        for row in embeddings.chunks_exact_mut(h) {
            row[0] = 1.0;
        }
        tensor(
            &mut map,
            "model.language_model.embed_tokens.weight",
            embeddings,
            &[v, h],
        );
        zero(&mut map, "model.language_model.norm.weight", &[h]);
        let mut lm_head = vec![0.0f32; v * h];
        lm_head[h] = 2.0; // token 1
        tensor(&mut map, "lm_head.weight", lm_head, &[v, h]);

        let lp = |suffix: &str| format!("model.language_model.layers.0.{suffix}");
        zero(&mut map, &lp("input_layernorm.weight"), &[h]);
        zero(&mut map, &lp("post_attention_layernorm.weight"), &[h]);
        zero(&mut map, &lp("self_attn.q_proj.weight"), &[nh * hd * 2, h]);
        zero(&mut map, &lp("self_attn.k_proj.weight"), &[nkv * hd, h]);
        zero(&mut map, &lp("self_attn.v_proj.weight"), &[nkv * hd, h]);
        zero(&mut map, &lp("self_attn.o_proj.weight"), &[h, nh * hd]);
        zero(&mut map, &lp("self_attn.q_norm.weight"), &[hd]);
        zero(&mut map, &lp("self_attn.k_norm.weight"), &[hd]);
        zero(&mut map, &lp("mlp.gate_proj.weight"), &[inter, h]);
        zero(&mut map, &lp("mlp.up_proj.weight"), &[inter, h]);
        zero(&mut map, &lp("mlp.down_proj.weight"), &[h, inter]);

        zero(&mut map, "mtp.pre_fc_norm_embedding.weight", &[h]);
        zero(&mut map, "mtp.pre_fc_norm_hidden.weight", &[h]);
        zero(&mut map, "mtp.norm.weight", &[h]);
        let mut fc = vec![0.0f32; h * h * 2];
        if aligned_predictor {
            for i in 0..h {
                fc[i * (2 * h) + i] = 1.0;
            }
        }
        tensor(&mut map, "mtp.fc.weight", fc, &[h, 2 * h]);
        let mp = |suffix: &str| format!("mtp.layers.0.{suffix}");
        zero(&mut map, &mp("input_layernorm.weight"), &[h]);
        zero(&mut map, &mp("post_attention_layernorm.weight"), &[h]);
        zero(&mut map, &mp("self_attn.q_proj.weight"), &[nh * hd * 2, h]);
        zero(&mut map, &mp("self_attn.k_proj.weight"), &[nkv * hd, h]);
        zero(&mut map, &mp("self_attn.v_proj.weight"), &[nkv * hd, h]);
        zero(&mut map, &mp("self_attn.o_proj.weight"), &[h, nh * hd]);
        zero(&mut map, &mp("self_attn.q_norm.weight"), &[hd]);
        zero(&mut map, &mp("self_attn.k_norm.weight"), &[hd]);
        zero(&mut map, &mp("mlp.gate_proj.weight"), &[inter, h]);
        zero(&mut map, &mp("mlp.up_proj.weight"), &[inter, h]);
        zero(&mut map, &mp("mlp.down_proj.weight"), &[h, inter]);

        let weights = Weights::from_map(map, Device::Cpu);
        assert!(Qwen35Mtp::complete_in(&weights, &cfg));
        let target = Qwen35Model::from_weights(&weights, "model.language_model", cfg).unwrap();
        let mtp = Qwen35Mtp::from_weights_with(&weights, &target, None).unwrap();
        (target, mtp)
    }

    fn config() -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: 7,
            sampling: Default::default(),
            seed: Some(7),
            stop_tokens: Vec::new(),
        }
    }

    #[test]
    fn rejected_drafts_restore_and_replay_to_target_only_greedy() {
        let (target, mtp) = fixture(false);
        let cfg = config();
        let cancel = CancelFlag::new();
        let baseline = generate_with(&target, &[2, 3], &cfg, &cancel, &mut |_| {}, None).unwrap();
        let (spec, stats) =
            generate_qwen35_mtp(&target, &mtp, &[2, 3], &cfg, 3, &cancel, &mut |_| {}, None)
                .unwrap();
        assert_eq!(spec.tokens, baseline.tokens);
        assert_eq!(spec.tokens, vec![1; cfg.max_new_tokens]);
        assert_eq!(stats.proposed, 12);
        assert_eq!(stats.accepted, 0);
        assert_eq!(
            stats.forwards, 12,
            "five rejected trials require five replay forwards"
        );
    }

    #[test]
    fn aligned_drafts_are_accepted() {
        let (target, mtp) = fixture(true);
        let cfg = config();
        let (out, stats) = generate_qwen35_mtp(
            &target,
            &mtp,
            &[2, 3],
            &cfg,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(out.tokens, vec![1; cfg.max_new_tokens]);
        assert_eq!(stats.proposed, 4);
        assert_eq!(stats.accepted, stats.proposed);
        assert_eq!(stats.forwards, 3);
    }

    #[test]
    fn stochastic_mtp_is_seed_deterministic_and_reports_bounded_acceptance() {
        let (target, mtp) = fixture(false);
        let mut cfg = config();
        cfg.sampling.temperature = 0.8;
        cfg.sampling.top_p = 0.9;
        cfg.sampling.top_k = 4;
        let run = || {
            generate_qwen35_mtp(
                &target,
                &mtp,
                &[2, 3],
                &cfg,
                3,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
            .unwrap()
        };
        let (first, first_stats) = run();
        let (second, second_stats) = run();
        assert_eq!(first.tokens, second.tokens);
        assert_eq!(first_stats, second_stats);
        assert!(first_stats.accepted <= first_stats.proposed);
        assert!(first_stats.proposed > 0);
    }

    #[test]
    fn equal_embedding_multimodal_prefill_matches_text_mtp() {
        let (target, mtp) = fixture(false);
        let cfg = config();
        let prompt = [2, 3];
        let embeds = target
            .embed_input_ids(&input_ids(&prompt, target.device()).unwrap())
            .unwrap();
        let positions = [0, 1];
        let (text, text_stats) = generate_qwen35_mtp(
            &target,
            &mtp,
            &prompt,
            &cfg,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        let (multimodal, multimodal_stats) = generate_qwen35_mtp_multimodal(
            &target,
            &mtp,
            Qwen35MtpMultimodalPrompt {
                input_ids: &prompt,
                embeddings: &embeds,
                positions: [&positions, &positions, &positions],
                visual_pos_mask: &[false, false],
                deepstack: &[],
                continuation_delta: 0,
            },
            &cfg,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(multimodal.tokens, text.tokens);
        assert_eq!(multimodal_stats, text_stats);
    }

    #[derive(Debug)]
    struct AuditConstraint {
        allowed: Vec<bool>,
        accepted: Vec<i32>,
        rewinds: usize,
    }

    impl AuditConstraint {
        fn new(vocab: usize) -> Self {
            let mut allowed = vec![false; vocab];
            allowed[0] = true;
            allowed[1] = true;
            Self {
                allowed,
                accepted: Vec::new(),
                rewinds: 0,
            }
        }
    }

    impl ConstraintMask for AuditConstraint {
        fn allowed(&mut self) -> &[bool] {
            &self.allowed
        }

        fn accept(&mut self, token: i32) {
            self.accepted.push(token);
        }
    }

    impl RewindableConstraintMask for AuditConstraint {
        fn checkpoint(&self) -> usize {
            self.accepted.len()
        }

        fn rewind(&mut self, checkpoint: usize) {
            self.accepted.truncate(checkpoint);
            self.rewinds += 1;
        }
    }

    #[test]
    fn constraint_state_discards_provisional_drafts() {
        let (target, mtp) = fixture(false);
        let mut constraint = AuditConstraint::new(6);
        let (out, stats) = generate_qwen35_mtp(
            &target,
            &mtp,
            &[2, 3],
            &config(),
            3,
            &CancelFlag::new(),
            &mut |_| {},
            Some(&mut constraint),
        )
        .unwrap();
        assert_eq!(stats.accepted, 0);
        assert_eq!(constraint.accepted, out.tokens);
        assert!(constraint.accepted.iter().all(|&token| token == 1));
        assert!(
            constraint.rewinds >= 2,
            "proposal and verification must both rewind"
        );
    }

    #[test]
    fn stop_and_cancellation_match_stream_contract() {
        let (target, mtp) = fixture(false);
        let mut stop_cfg = config();
        stop_cfg.stop_tokens = vec![1];
        let (stopped, stats) = generate_qwen35_mtp(
            &target,
            &mtp,
            &[2, 3],
            &stop_cfg,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert!(stopped.tokens.is_empty());
        assert_eq!(stopped.finish_reason, FinishReason::StopToken);
        assert_eq!(stats.forwards, 1);

        let pre_cancel = CancelFlag::new();
        pre_cancel.cancel();
        assert!(matches!(
            generate_qwen35_mtp(
                &target,
                &mtp,
                &[2, 3],
                &config(),
                3,
                &pre_cancel,
                &mut |_| {},
                None,
            ),
            Err(Error::Canceled)
        ));

        let mid_cancel = CancelFlag::new();
        let signal = mid_cancel.clone();
        let (cancelled, _) = generate_qwen35_mtp(
            &target,
            &mtp,
            &[2, 3],
            &config(),
            3,
            &mid_cancel,
            &mut |event| {
                if matches!(event, StreamEvent::Token { .. }) {
                    signal.cancel();
                }
            },
            None,
        )
        .unwrap();
        assert_eq!(cancelled.tokens, vec![1]);
        assert_eq!(cancelled.finish_reason, FinishReason::Cancelled);
    }
}
