use std::collections::HashMap;

use mlx_rs::Dtype;

use super::*;
use crate::config::{Architecture, ModelConfig};
use crate::decode::proposers::{DraftModelProposer, MtpProposer, NgramProposer};
use crate::decode::stream::generate_with;
use crate::models::qwen35::tests::{cfg_json, cfg_json_mtp, synthetic_weights};
use crate::models::Qwen35Config;
use crate::primitives::sampler::{host_transfers, HostTransfers};
use crate::primitives::Weights;

// ---- Fixtures: a tiny random llama-family decoder and the synthetic Qwen35 hybrid. ----

/// A tiny random llama-family decoder (vocab 24) whose greedy continuation of [`PROMPT`]
/// repeats its context, so prompt lookup proposes, accepts and rejects.
pub(crate) fn causal() -> CausalLm {
    tiny_llama(24)
}

/// The tiny random llama-family decoder over a `vocab`-entry vocabulary (hidden 16, two
/// layers), drawn from one fixed seed.
pub(crate) fn tiny_llama(vocab: i32) -> CausalLm {
    causal_model(vocab, 2)
}

/// The tiny llama-family decoder over `vocab` ids with its first `layers` layers. The weights
/// are drawn in a fixed order (embedding, output projection, then layer by layer), so a
/// shallower model is the deeper one with its later layers skipped — a draft that shares the
/// target's embedding, output projection and first layer (sc-24436).
pub(crate) fn causal_model(vocab: i32, layers: usize) -> CausalLm {
    synthetic_llama(16, layers, vocab)
}

/// A random llama-family decoder of width `hidden` (four query heads, two KV heads, a 2x MLP)
/// and `layers` layers over a `vocab`-entry vocabulary, drawn from one fixed seed in
/// [`causal_model`]'s order.
fn synthetic_llama(hidden: i32, layers: usize, vocab: i32) -> CausalLm {
    let cfg = ModelConfig {
        hidden_size: hidden,
        intermediate_size: 2 * hidden,
        num_layers: layers,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: hidden / 4,
        vocab_size: vocab,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        architecture: Architecture::Llama,
        max_position_embeddings: 0,
        quantization: None,
        moe: None,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        query_pre_attn_scalar: None,
        partial_rotary_factor: 1.0,
        mla: None,
        yarn: None,
        mrope_section: None,
        gemma4: None,
        activation_role: Default::default(),
    };
    let mut rng = SplitMix64::new(0x5EED_24434);
    let mut randn = |shape: &[i32]| {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.8).collect();
        Array::from_slice(&data, shape)
    };
    let (h, v, inter) = (cfg.hidden_size, cfg.vocab_size, cfg.intermediate_size);
    let (qd, kvd) = (
        cfg.num_heads * cfg.head_dim,
        cfg.num_kv_heads * cfg.head_dim,
    );
    let ones = |n: i32| Array::ones::<f32>(&[n]).unwrap();
    let mut m: HashMap<String, Array> = HashMap::new();
    m.insert("model.embed_tokens.weight".into(), randn(&[v, h]));
    m.insert("model.norm.weight".into(), ones(h));
    m.insert("lm_head.weight".into(), randn(&[v, h]));
    for i in 0..cfg.num_layers {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        m.insert(p("input_layernorm.weight"), ones(h));
        m.insert(p("post_attention_layernorm.weight"), ones(h));
        m.insert(p("self_attn.q_proj.weight"), randn(&[qd, h]));
        m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, h]));
        m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, h]));
        m.insert(p("self_attn.o_proj.weight"), randn(&[h, qd]));
        m.insert(p("mlp.gate_proj.weight"), randn(&[inter, h]));
        m.insert(p("mlp.up_proj.weight"), randn(&[inter, h]));
        m.insert(p("mlp.down_proj.weight"), randn(&[h, inter]));
    }
    CausalLm::from_weights(&Weights::from_map(m), "", cfg).unwrap()
}

/// The synthetic Qwen35 hybrid (vocab 50; three DeltaNet layers, one full-attention layer),
/// with or without its MTP head.
pub(crate) fn qwen35(mtp: bool) -> Qwen35Model {
    let cfg = Qwen35Config::from_json(&if mtp { cfg_json_mtp() } else { cfg_json() }).unwrap();
    Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg).unwrap()
}

fn greedy(max_new_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams::default(),
        seed: Some(7),
        stop_tokens: Vec::new(),
    }
}

#[test]
fn a_reused_sampler_resets_its_generated_only_presence_boundary() {
    let mut config = greedy(4);
    config.sampling.presence_penalty = 1.5;
    config.sampling.presence_penalty_generated_only = true;
    let mut sampler = MlxSampler::from_config(&config);
    let logits = Array::from_slice(&[2.0f32, 1.0, 0.0], &[1, 3]);
    for prompt_len in [3, 1, 3] {
        sampler.begin_generation(prompt_len).unwrap();
        let mut history = vec![0; prompt_len];
        assert_eq!(
            sampler
                .sample(&logits, &history, None)
                .unwrap()
                .resolve()
                .unwrap(),
            0
        );
        history.push(0);
        assert_eq!(
            sampler
                .sample(&logits, &history, None)
                .unwrap()
                .resolve()
                .unwrap(),
            1
        );
        assert_eq!(sampler.params().presence_penalty_prompt_len, prompt_len);
    }
    assert_eq!(config.sampling.presence_penalty_prompt_len, 0);
}

fn penalized(max_new_tokens: usize) -> GenerationConfig {
    let mut config = greedy(max_new_tokens);
    config.sampling.repetition_penalty = 1.3;
    config.sampling.repetition_context = 16;
    config.sampling.presence_penalty = 0.2;
    config
}

fn stochastic(max_new_tokens: usize) -> GenerationConfig {
    let mut config = greedy(max_new_tokens);
    config.sampling.temperature = 0.8;
    config.sampling.top_p = 0.9;
    config.sampling.top_k = 6;
    config
}

/// A deterministic constraint forbidding a fixed set of ids, with the checkpoint / rewind
/// the engine needs; records every accepted token and counts the engine's rewinds — all of
/// them, and those that actually moved the state back.
struct Forbid {
    allow: Vec<bool>,
    accepted: Vec<i32>,
    rewinds: usize,
    state_changing_rewinds: usize,
}

impl Forbid {
    fn new(vocab: usize, forbidden: &[usize]) -> Self {
        let mut allow = vec![true; vocab];
        for &f in forbidden {
            allow[f] = false;
        }
        Self {
            allow,
            accepted: Vec::new(),
            rewinds: 0,
            state_changing_rewinds: 0,
        }
    }
}

impl ConstraintMask for Forbid {
    fn allowed(&mut self) -> &[bool] {
        &self.allow
    }
    fn accept(&mut self, token: i32) {
        self.accepted.push(token);
    }
}

impl RewindableConstraintMask for Forbid {
    fn checkpoint(&self) -> usize {
        self.accepted.len()
    }
    fn rewind(&mut self, checkpoint: usize) {
        self.rewinds += 1;
        if checkpoint == self.accepted.len() {
            return;
        }
        self.state_changing_rewinds += 1;
        self.accepted.truncate(checkpoint);
    }
}

/// The plain, non-engine token-at-a-time loop ([`generate_with`]) — the reference every
/// engine route is held to.
fn plain<D: crate::decode::Decode>(
    model: &D,
    prompt: &[i32],
    config: &GenerationConfig,
    constraint: Option<&mut Forbid>,
) -> GenerationOutput {
    generate_with(
        model,
        prompt,
        config,
        &CancelFlag::new(),
        &mut |_| {},
        constraint.map(|c| c as &mut dyn ConstraintMask),
        None,
    )
    .unwrap()
}

fn engine<T, P>(
    target: &T,
    proposer: &mut P,
    prompt: &[i32],
    config: &GenerationConfig,
    drafts: usize,
    constraint: Option<&mut Forbid>,
) -> SpeculativeRun
where
    T: SpeculativeTarget,
    P: Proposer<T>,
{
    let mut events = Vec::new();
    let run = generate_speculative(
        target,
        proposer,
        SpeculativePrompt::Tokens(prompt),
        config,
        drafts,
        &CancelFlag::new(),
        &mut |e| events.push(e),
        EngineOptions {
            constraint: constraint.map(|c| c as &mut dyn RewindableConstraintMask),
            ..EngineOptions::default()
        },
    )
    .unwrap();
    // The event stream is the output: one Token per generated id, in order, then Done.
    let ids: Vec<i32> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::Token { id, .. } => Some(*id),
            StreamEvent::Done { .. } => None,
        })
        .collect();
    assert_eq!(ids, run.output.tokens, "streamed ids == returned tokens");
    assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
    run
}

/// A prompt whose continuation repeats context, so prompt lookup proposes.
const PROMPT: [i32; 12] = [3, 9, 4, 11, 3, 9, 4, 11, 3, 9, 4, 11];

/// The absolute accounting of a budget-bound run with no stop tokens: the first token comes
/// from the prefill, every verify step commits its accepted drafts plus exactly one bonus or
/// correction token, and every forward is the prefill, a verify step or a recovery replay.
fn assert_accounting(label: &str, run: &SpeculativeRun, config: &GenerationConfig) {
    assert!(
        config.stop_tokens.is_empty(),
        "{label}: accounting needs no stop tokens"
    );
    assert_eq!(run.output.finish_reason, FinishReason::MaxTokens, "{label}");
    let r = &run.report;
    assert_eq!(
        run.output.tokens.len() as u64,
        1 + r.verify_steps + r.accepted_tokens,
        "{label}: tokens = first + one per verify step + accepted drafts: {r:?}"
    );
    assert_eq!(r.prefill_forwards, 1, "{label}: one prefill forward: {r:?}");
    assert_eq!(
        r.target_forwards,
        r.prefill_forwards + r.verify_steps + r.replay_forwards + r.discarded_forwards,
        "{label}: forwards = prefill + verify steps + replays + discarded: {r:?}"
    );
}

/// Every sampler configuration the engine must honour exactly, with the sampler path the
/// report must name for it.
fn configs() -> Vec<(&'static str, GenerationConfig, bool, &'static str)> {
    vec![
        ("greedy", greedy(20), false, "device"),
        ("penalized", penalized(20), false, "host:penalty"),
        ("constrained", greedy(20), true, "host:constraint"),
        ("stochastic", stochastic(20), false, "device"),
        (
            "constrained stochastic",
            stochastic(20),
            true,
            "host:constraint",
        ),
    ]
}

/// `off` — the engine with [`NoProposer`] — against the plain, non-engine loop on one target:
/// identical tokens and finish reason for greedy, penalized, constrained and seeded stochastic
/// sampling, and a report naming the token-at-a-time path.
fn off_matches_plain<T: SpeculativeTarget + crate::decode::Decode>(target: &T, vocab: usize) {
    for (name, config, constrained, sampler) in configs() {
        let forbid = || Forbid::new(vocab, &[1, 9]);
        let (mut a, mut b) = (forbid(), forbid());
        let expected = plain(target, &PROMPT, &config, constrained.then_some(&mut a));
        let run = engine(
            target,
            &mut NoProposer,
            &PROMPT,
            &config,
            4,
            constrained.then_some(&mut b),
        );
        assert_eq!(
            run.output.tokens, expected.tokens,
            "{name}: off != plain loop"
        );
        assert_eq!(run.output.finish_reason, expected.finish_reason, "{name}");
        assert_eq!(run.output.tokens.len(), 20, "{name}: runs to the budget");
        if constrained {
            assert_eq!(
                b.accepted, a.accepted,
                "{name}: constraint advanced identically"
            );
            assert!(!run.output.tokens.iter().any(|t| *t == 1 || *t == 9));
        }
        let report = &run.report;
        assert_eq!(report.path, "step_model", "{name}");
        assert_eq!(report.proposer, ProposerKind::None, "{name}");
        assert_eq!(report.draft_tokens, None, "{name}: no proposer, no depth");
        assert_eq!(
            report.verify_steps, 19,
            "{name}: one verify per token after the first"
        );
        assert_eq!(report.target_forwards, 20, "{name}: prefill + one per step");
        assert_eq!((report.proposed_tokens, report.accepted_tokens), (0, 0));
        assert_eq!(report.mean_accepted_length(), None, "{name}");
        assert_eq!(report.sampler, sampler, "{name}");
        // Device draws pipeline every step after step 1 (token 0 is read back before step 1
        // is enqueued, sc-24446); host draws never do.
        let pipelined = if sampler == "device" { 18 } else { 0 };
        assert_eq!(run.stats.pipelined, pipelined, "{name}");
        assert_eq!(
            run.stats.discarded, 0,
            "{name}: a budget end enqueues nothing past it"
        );
    }
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn off_is_the_plain_loop_on_the_causal_and_hybrid_targets() {
    off_matches_plain(&causal(), 24);
    off_matches_plain(&qwen35(false), 50);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::off_is_the_plain_loop_on_the_causal_and_hybrid_targets");
}

/// AC1: greedy prompt lookup emits exactly the plain loop's tokens — plain, penalized and
/// constrained — on the Causal target (direct truncation) and the Qwen35 hybrid (its DeltaNet
/// checkpoint ring, sc-24435: direct, never a replay forward), while actually proposing,
/// accepting and rejecting.
/// The production decode records no cache ownership events (sc-20669 evidence is opt-in): a
/// run past a block growth — the engine's pipelined loop, then prompt lookup's verify and
/// rollback — leaves the caller's cache with none, while the same run on a cache asked to
/// record ([`KvCache::record_events`], what only a campaign observer does) records them.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_production_decode_records_no_cache_events() {
    let model = causal();
    let run = |record: bool, drafts: usize| {
        let mut cache = model.new_cache();
        if record {
            cache.record_events();
        }
        let logits = SpeculativeTarget::forward(
            &model,
            &mut cache,
            &input_ids(&PROMPT),
            0,
            LogitsScope::Last,
            false,
        )
        .unwrap()
        .logits;
        let mut proposer = NgramProposer::default();
        let mut plain = NoProposer;
        let proposer: &mut dyn Proposer<CausalLm> = if drafts > 0 {
            &mut proposer
        } else {
            &mut plain
        };
        generate_speculative(
            &model,
            proposer,
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits,
                hidden: None,
                history: &PROMPT,
                position_delta: 0,
            },
            &greedy(300),
            drafts,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions::default(),
        )
        .unwrap();
        assert!(cache.offset() > 256, "the run crosses a block growth");
        cache.events().len()
    };
    for drafts in [0, 3] {
        assert_eq!(run(false, drafts), 0, "drafts {drafts}");
        assert!(run(true, drafts) > 0, "drafts {drafts}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_production_decode_records_no_cache_events");
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn greedy_prompt_lookup_is_the_plain_loop_and_recovers_both_ways() {
    let causal = causal();
    let hybrid = qwen35(false);
    for (name, config, constrained, _) in configs()
        .into_iter()
        .filter(|c| !c.0.contains("stochastic"))
    {
        let forbid = |vocab| Forbid::new(vocab, &[1, 9]);

        let (mut a, mut b) = (forbid(24), forbid(24));
        let expected = plain(&causal, &PROMPT, &config, constrained.then_some(&mut a));
        let run = engine(
            &causal,
            &mut NgramProposer::default(),
            &PROMPT,
            &config,
            4,
            constrained.then_some(&mut b),
        );
        assert_eq!(run.output.tokens, expected.tokens, "causal {name}");
        assert_eq!(b.accepted, a.accepted, "causal {name}: constraint");
        assert_accounting(&format!("causal {name}"), &run, &config);
        let s = run.stats;
        assert!(
            s.proposed > 0 && s.accepted > 0,
            "causal {name}: lookup ran: {s:?}"
        );
        assert_eq!(s.replays, 0, "causal {name}: truncation never replays");
        assert_eq!(run.report.proposer, ProposerKind::PromptLookup);

        let (mut a, mut b) = (forbid(50), forbid(50));
        let expected = plain(&hybrid, &PROMPT, &config, constrained.then_some(&mut a));
        let run = engine(
            &hybrid,
            &mut NgramProposer::default(),
            &PROMPT,
            &config,
            4,
            constrained.then_some(&mut b),
        );
        assert_eq!(run.output.tokens, expected.tokens, "hybrid {name}");
        assert_eq!(b.accepted, a.accepted, "hybrid {name}: constraint");
        assert_accounting(&format!("hybrid {name}"), &run, &config);
        let s = run.stats;
        // The penalized hybrid continuation never repeats its context, so it proposes nothing
        // (its exactness with drafts is the Causal run's above).
        if name != "penalized" {
            assert!(
                s.proposed > 0 && s.accepted > 0,
                "hybrid {name}: lookup ran: {s:?}"
            );
        }
        assert_eq!(s.replays, 0, "hybrid {name}: the ring never replays");
        assert_eq!(run.report.replay_forwards, 0, "hybrid {name}");
    }
    // The plain greedy fixture exercises every recovery kind at least once.
    let run = engine(
        &causal,
        &mut NgramProposer::default(),
        &PROMPT,
        &greedy(24),
        4,
        None,
    );
    assert!(run.stats.direct_rollbacks > 0, "{:?}", run.stats);
    let run = engine(
        &hybrid,
        &mut NgramProposer::default(),
        &PROMPT,
        &greedy(24),
        4,
        None,
    );
    assert!(
        run.stats.direct_rollbacks > 0 && run.stats.replays == 0,
        "{:?}",
        run.stats
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::greedy_prompt_lookup_is_the_plain_loop_and_recovers_both_ways");
}

/// AC1: greedy MTP emits exactly the plain loop's tokens (plain, penalized, constrained) on
/// the Qwen35 MTP fixture, whose adversarial drafts force a partial rejection — recovered
/// through the checkpoint ring, with zero replay forwards (sc-24435).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn greedy_mtp_is_the_plain_loop() {
    let model = qwen35(true);
    for (name, config, constrained, _) in configs()
        .into_iter()
        .filter(|c| !c.0.contains("stochastic"))
    {
        let (mut a, mut b) = (Forbid::new(50, &[1, 9]), Forbid::new(50, &[1, 9]));
        let expected = plain(&model, &PROMPT, &config, constrained.then_some(&mut a));
        let run = engine(
            &model,
            &mut MtpProposer::new(),
            &PROMPT,
            &config,
            3,
            constrained.then_some(&mut b),
        );
        assert_eq!(run.output.tokens, expected.tokens, "{name}");
        assert_eq!(b.accepted, a.accepted, "{name}: constraint");
        assert_accounting(&format!("mtp {name}"), &run, &config);
        assert!(
            run.stats.proposed > 0 && run.stats.direct_rollbacks > 0,
            "{name}: {:?}",
            run.stats
        );
        assert_eq!(run.stats.replays, 0, "{name}: the ring never replays");
        assert_eq!(run.report.replay_forwards, 0, "{name}");
        assert_eq!(run.report.path, "mtp");
        assert_eq!(run.report.proposer, ProposerKind::Mtp);
        assert_eq!(run.report.draft_tokens, Some(3));
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::greedy_mtp_is_the_plain_loop");
}

/// The Qwen35 target under test: every forward is delegated to the model, then the attention
/// KV's block-buffer addresses are recorded (the in-place probe) — rolled back by `R`, the
/// checkpoint ring the target ships with or the step-start snapshot it replaced. A
/// [`LogitsScope::Last`] forward (the prefill, a no-draft step, a replay) runs outside any
/// checkpoint window: it must leave the cache recording nothing.
struct Observed<'m, R> {
    model: &'m Qwen35Model,
    addresses: std::cell::RefCell<Vec<Vec<usize>>>,
    rollback: fn(usize) -> R,
}

impl<'m, R> Observed<'m, R> {
    fn new(model: &'m Qwen35Model, rollback: fn(usize) -> R) -> Self {
        Self {
            model,
            addresses: Default::default(),
            rollback,
        }
    }
}

impl<R: CacheRollback<Qwen35Cache>> SpeculativeTarget for Observed<'_, R> {
    type Cache = Qwen35Cache;
    type Rollback = R;

    fn new_cache(&self) -> Qwen35Cache {
        self.model.new_cache()
    }

    fn cache_len(&self, cache: &Qwen35Cache) -> i32 {
        cache.offset()
    }

    fn rollback(&self, width: usize) -> R {
        (self.rollback)(width)
    }

    fn forward(
        &self,
        cache: &mut Qwen35Cache,
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        let out =
            SpeculativeTarget::forward(self.model, cache, ids, rope_offset, scope, want_hidden)?;
        if scope == LogitsScope::Last {
            assert_eq!(
                cache.checkpointed_tokens(),
                0,
                "a forward outside a verify step recorded checkpoints"
            );
        }
        self.addresses
            .borrow_mut()
            .push(cache.attn_buffer_addresses());
        Ok(out)
    }

    fn attention_label(&self) -> &'static str {
        "gqa"
    }
}

/// A proposer wrapper recording each step's `(proposed, accepted)` draft counts.
struct Tally<P> {
    inner: P,
    proposed: usize,
    steps: Vec<(usize, usize)>,
}

impl<P> Tally<P> {
    fn new(inner: P) -> Self {
        Self {
            inner,
            proposed: 0,
            steps: Vec::new(),
        }
    }

    /// Steps whose every proposed draft was accepted.
    fn full_acceptances(&self) -> usize {
        self.steps.iter().filter(|&&(p, a)| p > 0 && a == p).count()
    }

    /// Steps that rejected a proposed draft.
    fn partial_rejections(&self) -> usize {
        self.steps.iter().filter(|&&(p, a)| a < p).count()
    }
}

/// A Qwen35 proposer run against the observed target: the model is what it proposes from.
impl<'m, R, P> Proposer<Observed<'m, R>> for Tally<P>
where
    R: CacheRollback<Qwen35Cache>,
    P: Proposer<Qwen35Model>,
{
    fn kind(&self) -> ProposerKind {
        self.inner.kind()
    }

    fn wants_hidden(&self) -> bool {
        self.inner.wants_hidden()
    }

    fn warm(
        &mut self,
        target: &Observed<'m, R>,
        prompt: &[i32],
        hidden: Option<&Array>,
    ) -> Result<Option<Array>> {
        self.inner.warm(target.model, prompt, hidden)
    }

    fn propose(
        &mut self,
        target: &Observed<'m, R>,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        let proposal = self.inner.propose(target.model, ctx, sampler)?;
        self.proposed = proposal.drafts.len();
        Ok(proposal)
    }

    fn commit(
        &mut self,
        target: &Observed<'m, R>,
        cur: i32,
        accepted: &[i32],
        kept_hidden: Option<&Array>,
        position: i32,
    ) -> Result<()> {
        self.steps
            .push((std::mem::take(&mut self.proposed), accepted.len()));
        self.inner
            .commit(target.model, cur, accepted, kept_hidden, position)
    }
}

/// AC1 + AC2 (sc-24435): on the Qwen35 hybrid, prompt lookup and MTP recover every partial
/// rejection through the DeltaNet checkpoint ring — zero replay forwards, the greedy tokens
/// of the plain loop — and no step clones the target cache: across every forward after the
/// prefill (full-acceptance steps included) the attention KV block buffers keep their
/// addresses, i.e. every forward wrote them in place. The recurrence routes show the ring's
/// cost is confined to the verify steps: prefill and no-draft steps run the final-state
/// kernel, each verify step with drafts the checkpoint kernel.
///
/// The probe is shown to see a copy: under the step-start snapshot the ring replaced, the same
/// run clones the cache every verify step, the in-place write then lands in a fresh block,
/// and a partial rejection costs a replay forward — with the same tokens.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_hybrid_rolls_back_through_its_checkpoint_ring_without_replay_or_clone() {
    use crate::models::qwen35::counting_cache_clones;
    use crate::primitives::gated_delta::{recording_routes, Route};
    mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), || {
        let linear_layers = 3;
        for (label, model, mtp) in [
            ("prompt lookup", qwen35(false), false),
            ("mtp", qwen35(true), true),
        ] {
            let config = greedy(24);
            let expected = plain(&model, &PROMPT, &config, None);
            let ring = Observed::new(&model, CheckpointRingRollback::new);
            let run_with = |proposer: &mut dyn FnMut() -> SpeculativeRun| {
                crate::model_inference_device::record_model_inference_device();
                let ((run, clones), routes) = recording_routes(|| counting_cache_clones(proposer));
                (run, clones, routes)
            };
            let mut lookup = Tally::new(NgramProposer::default());
            let mut head = Tally::new(MtpProposer::new());
            let (run, clones, routes) = run_with(&mut || {
                crate::model_inference_device::record_model_inference_device();
                if mtp {
                    engine(&ring, &mut head, &PROMPT, &config, 3, None)
                } else {
                    engine(&ring, &mut lookup, &PROMPT, &config, 4, None)
                }
            });
            let (full, partial, steps) = if mtp {
                (
                    head.full_acceptances(),
                    head.partial_rejections(),
                    head.steps.clone(),
                )
            } else {
                (
                    lookup.full_acceptances(),
                    lookup.partial_rejections(),
                    lookup.steps.clone(),
                )
            };
            assert_eq!(run.output.tokens, expected.tokens, "{label}: != plain loop");
            assert_accounting(label, &run, &config);
            assert!(partial > 0, "{label}: a partial rejection ran: {steps:?}");
            assert_eq!(run.report.replay_forwards, 0, "{label}: {:?}", run.report);
            assert_eq!(run.stats.direct_rollbacks, partial, "{label}");
            assert_eq!(clones, 0, "{label}: the target cache is never cloned");
            let addresses = ring.addresses.borrow();
            assert_eq!(addresses.len() as u64, run.report.target_forwards);
            assert!(
                addresses.windows(2).all(|w| w[0] == w[1]),
                "{label}: the attention KV moved: {addresses:?}"
            );
            if !mtp {
                // The adversarial MTP head never has a draft accepted; the lookup fixture
                // covers the full-acceptance steps.
                assert!(full > 0, "{label}: a full acceptance ran: {steps:?}");
            }
            let drafted = steps.iter().filter(|&&(p, _)| p > 0).count();
            let count = |r: Route| routes.iter().filter(|&&x| x == r).count();
            assert_eq!(
                count(Route::CheckpointKernel),
                linear_layers * drafted,
                "{label}: {routes:?}"
            );
            assert_eq!(
                count(Route::Kernel),
                linear_layers * (1 + steps.len() - drafted),
                "{label}: prefill and no-draft steps keep the final-state kernel"
            );
            assert_eq!(count(Route::Ops) + count(Route::Chunked), 0, "{label}");
        }

        // A full acceptance closes the window: an oracle proposing the plain loop's own next
        // token (always accepted) every other step, so each full acceptance is followed by a
        // no-draft step, whose forward the observed target checks records nothing.
        struct Oracle {
            expected: Vec<i32>,
            steps: usize,
        }
        impl<T: SpeculativeTarget + ?Sized> Proposer<T> for Oracle {
            fn kind(&self) -> ProposerKind {
                ProposerKind::PromptLookup
            }
            fn warm(&mut self, _: &T, _: &[i32], _: Option<&Array>) -> Result<Option<Array>> {
                Ok(None)
            }
            fn propose(
                &mut self,
                _: &T,
                ctx: &ProposeContext<'_>,
                _: &mut DraftSampler<'_, '_>,
            ) -> Result<Proposal> {
                self.steps += 1;
                let next = self.expected.get(ctx.history.len() - PROMPT.len());
                Ok(Proposal {
                    drafts: next
                        .filter(|_| self.steps % 2 == 1)
                        .copied()
                        .into_iter()
                        .collect(),
                    dists: Vec::new(),
                })
            }
            fn commit(
                &mut self,
                _: &T,
                _: i32,
                _: &[i32],
                _: Option<&Array>,
                _: i32,
            ) -> Result<()> {
                Ok(())
            }
        }
        let model = qwen35(false);
        let config = greedy(12);
        let expected = plain(&model, &PROMPT, &config, None).tokens;
        let ring = Observed::new(&model, CheckpointRingRollback::new);
        let mut oracle = Tally::new(Oracle {
            expected: expected.clone(),
            steps: 0,
        });
        let run = engine(&ring, &mut oracle, &PROMPT, &config, 4, None);
        assert_eq!(run.output.tokens, expected, "oracle: != plain loop");
        let followed = oracle
            .steps
            .windows(2)
            .filter(|w| w[0] == (1, 1) && w[1].0 == 0)
            .count();
        assert!(
            followed > 0,
            "a no-draft step followed a full acceptance: {:?}",
            oracle.steps
        );

        // The contrast: the same lookup run on the step-start snapshot.
        let model = qwen35(false);
        let config = greedy(24);
        let snapshot = Observed::new(&model, |_| SnapshotRollback::<Qwen35Cache>::default());
        let (run, clones) = counting_cache_clones(|| {
            engine(
                &snapshot,
                &mut Tally::new(NgramProposer::default()),
                &PROMPT,
                &config,
                4,
                None,
            )
        });
        assert_eq!(
            run.output.tokens,
            plain(&model, &PROMPT, &config, None).tokens,
            "replay recovers the same tokens"
        );
        assert!(run.stats.replays > 0, "{:?}", run.stats);
        assert!(clones > 0, "the snapshot clones every verify step");
        let addresses = snapshot.addresses.borrow();
        assert!(
            addresses.windows(2).any(|w| w[0] != w[1]),
            "a held snapshot makes the in-place write copy its block"
        );
    });

    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::the_hybrid_rolls_back_through_its_checkpoint_ring_without_replay_or_clone");
}

/// The draft-model window through the [`CacheRollback`] seam (sc-24435; the shape a draft
/// proposer drives on its own cache): one `begin`, three separate single-token forwards, then
/// `recover` to each position from the step start (nothing kept) to all three kept. The
/// window is armed for `width + 1` tokens, so width 2 fits exactly these three forwards and
/// refuses a fourth (typed, the cache untouched). Every
/// recovery is direct, its DeltaNet state is bit-exact that of a cache that only ever saw the
/// kept tokens, the next forward's logits are too (the attention KV was truncated by offset),
/// and that forward records nothing — full acceptance included.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_ring_recovers_a_draft_window_of_single_token_forwards() {
    let model = qwen35(false);
    let step = |cache: &mut Qwen35Cache, ids: &[i32]| -> Vec<f32> {
        let offset = cache.offset();
        let out = SpeculativeTarget::forward(
            &model,
            cache,
            &input_ids(ids),
            offset,
            LogitsScope::Last,
            false,
        )
        .unwrap();
        out.logits
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    };
    let prefilled = || {
        let mut cache = model.new_cache();
        step(&mut cache, &PROMPT);
        cache
    };
    let start = PROMPT.len() as i32;
    let drafts = [4, 11, 7];
    for kept in 0..=drafts.len() {
        let mut cache = prefilled();
        let mut rollback = SpeculativeTarget::rollback(&model, 2);
        rollback.begin(&mut cache).unwrap();
        for &d in &drafts {
            step(&mut cache, &[d]);
        }
        // A fourth forward would outgrow what width 2 prices: refused, cache untouched.
        let before = cache.delta_states();
        let offset = cache.offset();
        let err = SpeculativeTarget::forward(
            &model,
            &mut cache,
            &input_ids(&[20]),
            offset,
            LogitsScope::Last,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::CheckpointWindowFull { max_tokens: 3, .. }),
            "{err}"
        );
        assert_eq!((cache.offset(), cache.delta_states()), (offset, before));
        let keep = start + kept as i32;
        assert_eq!(
            rollback.recover(&mut cache, keep).unwrap(),
            Rollback::Direct
        );
        assert_eq!(cache.offset(), keep);
        let mut reference = prefilled();
        for &d in &drafts[..kept] {
            step(&mut reference, &[d]);
        }
        assert_eq!(
            cache.delta_states(),
            reference.delta_states(),
            "kept {kept}: restored state"
        );
        assert_eq!(
            step(&mut cache, &[13]),
            step(&mut reference, &[13]),
            "kept {kept}: next logits"
        );
        assert_eq!(cache.checkpointed_tokens(), 0, "kept {kept}: window closed");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::the_ring_recovers_a_draft_window_of_single_token_forwards");
}

/// AC2 (engine half): the report names the proposer, its depth, the verify steps and accepted
/// drafts behind the mean accepted length, the recovery replays and the measured sampler.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_report_carries_the_measured_speculation() {
    let run = engine(
        &causal(),
        &mut NgramProposer::default(),
        &PROMPT,
        &greedy(24),
        4,
        None,
    );
    assert_accounting("causal lookup", &run, &greedy(24));
    assert!(run.report.accepted_tokens > 0, "{:?}", run.report);
    let (r, s) = (&run.report, run.stats);
    assert_eq!(r.path, "prompt_lookup");
    assert_eq!(r.proposer, ProposerKind::PromptLookup);
    assert_eq!(r.draft_tokens, Some(4));
    assert_eq!(
        r.sampler, "device",
        "plain greedy draws are device argmaxes"
    );
    assert_eq!(
        (r.kv_cache.as_str(), r.attention.as_str()),
        ("growing", "gqa")
    );
    assert_eq!(r.target_forwards, s.forwards as u64);
    assert_eq!(r.proposed_tokens, s.proposed as u64);
    assert_eq!(r.accepted_tokens, s.accepted as u64);
    assert_eq!(r.verify_steps, s.verify_steps as u64);
    assert!(
        r.verify_steps < 23,
        "speculation took fewer verify steps than tokens"
    );
    assert_eq!(
        r.mean_accepted_length(),
        Some(s.accepted as f64 / s.verify_steps as f64)
    );
    assert!(!r.cuda_graphs.enabled);
    assert_eq!(r.cuda_graphs.path, "none");
    // E3: MLX has no graph runner, so no step went through one — `none`, never `eager`.
    assert_eq!(r.graph_path, "none");
    // A speculative run never looks ahead, and the causal fixture runs no fused primitive.
    assert_eq!(r.discarded_forwards, 0);
    assert_eq!(r.fused_primitives.path, "none");
    assert!(
        r.fallbacks.is_empty(),
        "fallbacks are the provider's to add"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::the_report_carries_the_measured_speculation");
}

/// A stop token inside an accepted draft run ends the run exactly where the plain loop does,
/// and nothing past it is emitted.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_stop_token_inside_a_draft_run_ends_like_the_plain_loop() {
    let model = causal();
    let base = plain(&model, &PROMPT, &greedy(24), None).tokens;
    let mut config = greedy(24);
    config.stop_tokens = vec![base[7]];
    let expected = plain(&model, &PROMPT, &config, None);
    let run = engine(
        &model,
        &mut NgramProposer::default(),
        &PROMPT,
        &config,
        4,
        None,
    );
    assert_eq!(run.output.tokens, expected.tokens);
    assert_eq!(run.output.finish_reason, FinishReason::StopToken);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_stop_token_inside_a_draft_run_ends_like_the_plain_loop");
}

/// Seeded stochastic speculation is reproducible and stays within its budget; acceptance is
/// `core_llm`'s distribution-preserving rule (proven there).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn stochastic_speculation_is_seeded() {
    let model = causal();
    let a = engine(
        &model,
        &mut NgramProposer::default(),
        &PROMPT,
        &stochastic(20),
        4,
        None,
    );
    let b = engine(
        &model,
        &mut NgramProposer::default(),
        &PROMPT,
        &stochastic(20),
        4,
        None,
    );
    assert_eq!(a.output.tokens, b.output.tokens);
    assert_eq!(a.output.tokens.len(), 20);
    assert_accounting("stochastic lookup", &a, &stochastic(20));
    assert!(a.stats.accepted <= a.stats.proposed);
    assert!(
        a.report.sampler.starts_with("host:"),
        "{}",
        a.report.sampler
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::stochastic_speculation_is_seeded"
    );
}

/// The sampler is a seam: a caller-supplied [`TokenSampler`] draws every token.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_caller_sampler_draws_every_token() {
    struct Counting(MlxSampler, usize);
    impl TokenSampler for Counting {
        fn params(&self) -> &SamplingParams {
            self.0.params()
        }
        fn sample(&mut self, l: &Array, h: &[i32], m: Option<&[bool]>) -> Result<SampledToken> {
            self.1 += 1;
            self.0.sample(l, h, m)
        }
        fn argmax_rows(&mut self, l: &Array) -> Result<Vec<i32>> {
            self.1 += 1;
            self.0.argmax_rows(l)
        }
        fn distribution(
            &mut self,
            l: &Array,
            h: &[i32],
            m: Option<&[bool]>,
        ) -> Result<Vec<(i32, f32)>> {
            self.0.distribution(l, h, m)
        }
        fn uniform(&mut self) -> f32 {
            self.0.uniform()
        }
        fn path(&self) -> Option<SamplerPath> {
            self.0.path()
        }
    }
    let model = causal();
    let config = greedy(16);
    let mut sampler = Counting(MlxSampler::from_config(&config), 0);
    let run = generate_speculative(
        &model,
        &mut NgramProposer::default(),
        SpeculativePrompt::Tokens(&PROMPT),
        &config,
        4,
        &CancelFlag::new(),
        &mut |_| {},
        EngineOptions {
            sampler: Some(&mut sampler),
            ..EngineOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        run.output.tokens,
        plain(&model, &PROMPT, &config, None).tokens
    );
    assert_eq!(
        sampler.1 as u64,
        run.report.verify_steps + 1,
        "one decision per step"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_caller_sampler_draws_every_token"
    );
}

// ---- Draft-model speculation on the engine (epic sc-24432, story sc-24436). ----

/// Every greedy configuration through [`DraftModelProposer`] over `draft` against the plain
/// loop on `target`: identical tokens and constraint advance, a report naming `draft_model`.
/// Returns each configuration's counters.
fn draft_matches_plain<T, D>(
    label: &str,
    target: &T,
    draft: &D,
    vocab: usize,
) -> Vec<SpeculativeStats>
where
    T: SpeculativeTarget + crate::decode::Decode,
    D: SpeculativeTarget,
{
    let mut all = Vec::new();
    for (name, config, constrained, _) in configs()
        .into_iter()
        .filter(|c| !c.0.contains("stochastic"))
    {
        let (mut a, mut b) = (Forbid::new(vocab, &[1, 9]), Forbid::new(vocab, &[1, 9]));
        let expected = plain(target, &PROMPT, &config, constrained.then_some(&mut a));
        let mut proposer = DraftModelProposer::new(draft, 4);
        let run = engine(
            target,
            &mut proposer,
            &PROMPT,
            &config,
            4,
            constrained.then_some(&mut b),
        );
        assert_eq!(run.output.tokens, expected.tokens, "{label} {name}");
        assert_eq!(b.accepted, a.accepted, "{label} {name}: constraint");
        assert_eq!(run.report.path, "draft_model", "{label} {name}");
        assert_eq!(
            run.report.proposer,
            ProposerKind::DraftModel,
            "{label} {name}"
        );
        assert_eq!(run.report.draft_tokens, Some(4), "{label} {name}");
        assert!(run.stats.proposed > 0, "{label} {name}: the draft proposed");
        assert!(proposer.draft_forwards > 0, "{label} {name}");
        all.push(run.stats);
    }
    all
}

/// sc-24436 AC1 (engine half): a draft model proposes through the one engine loop and every
/// greedy configuration emits exactly the plain loop's tokens, whichever way each cache rolls
/// back — a truncating target with a truncating layer-skip draft (accepted and rejected
/// drafts), the hybrid target (its DeltaNet checkpoint ring, sc-24435: direct, never a
/// replay) with a softmax draft, a softmax target with the hybrid as its draft (the draft
/// recovers its own proposal window through its ring), and the hybrid drafting for an
/// identical hybrid (every draft accepted).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn greedy_draft_model_is_the_plain_loop_on_every_cache_pairing() {
    let stats = draft_matches_plain("causal/causal", &causal(), &causal_model(24, 1), 24);
    let (accepted, proposed) = stats
        .iter()
        .fold((0, 0), |(a, p), s| (a + s.accepted, p + s.proposed));
    assert!(
        accepted > 0 && accepted < proposed,
        "the layer-skip draft is accepted and rejected: {accepted} of {proposed}"
    );
    assert!(stats.iter().any(|s| s.direct_rollbacks > 0), "{stats:?}");

    let stats = draft_matches_plain("hybrid/causal", &qwen35(false), &causal_model(50, 1), 50);
    assert!(
        stats
            .iter()
            .all(|s| s.replays == 0 && s.direct_rollbacks > 0),
        "the hybrid target recovers through its ring: {stats:?}"
    );

    draft_matches_plain("causal/hybrid", &causal_model(50, 2), &qwen35(false), 50);

    let stats = draft_matches_plain("hybrid/hybrid", &qwen35(false), &qwen35(false), 50);
    assert!(
        stats.iter().all(|s| s.accepted == s.proposed),
        "a draft identical to its target is always accepted: {stats:?}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::greedy_draft_model_is_the_plain_loop_on_every_cache_pairing");
}

/// sc-24436 E1: a stochastic draft hands the engine the proposal distribution `q` each draft
/// was drawn from — one per draft, each containing its draft — so the exact rejection rule
/// compares the target's `p` against the draft's own `q`; seeded runs reproduce.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_stochastic_draft_model_reports_one_q_per_draft_and_is_seeded() {
    let target = causal();
    let draft = causal_model(24, 1);
    let config = stochastic(20);
    let mut proposer = DraftModelProposer::new(&draft, 4);
    Proposer::<CausalLm>::warm(&mut proposer, &target, &PROMPT, None).unwrap();
    let mut history = PROMPT.to_vec();
    history.push(5);
    let mut sampler = MlxSampler::from_config(&config);
    let mut draft_sampler = DraftSampler {
        config: &config,
        sampler: &mut sampler,
        constraint: None,
        advanced: false,
    };
    let ctx = ProposeContext {
        cur: 5,
        history: &history,
        previous_hidden: None,
        position: PROMPT.len() as i32,
        max_drafts: 4,
    };
    let proposal = proposer.propose(&target, &ctx, &mut draft_sampler).unwrap();
    assert_eq!(proposal.drafts.len(), 4);
    assert_eq!(
        proposal.dists.len(),
        proposal.drafts.len(),
        "one q per draft"
    );
    for (i, (token, q)) in proposal.drafts.iter().zip(&proposal.dists).enumerate() {
        assert!(
            q.iter().any(|&(t, w)| t == *token && w > 0.0),
            "{token} in {q:?}"
        );
        // Each `q` is the draft model's own shaped distribution at its position: the
        // sampler's distribution over a fresh draft prefill of everything before the draft.
        let mut before = history.clone();
        before.extend_from_slice(&proposal.drafts[..i]);
        let expected = MlxSampler::from_config(&config)
            .distribution(&last_logits(&draft, &before), &before, None)
            .unwrap();
        assert_distribution_eq(q, &expected, &format!("q of draft {i}"));
    }
    assert!(
        proposal.dists.iter().any(|q| q.len() > 1),
        "a stochastic draft is not a point mass: {:?}",
        proposal.dists
    );
    // The prompt, `cur` and all but the last draft are in the draft cache.
    assert_eq!(proposer.draft_forwards, 1 + 4);

    let run = |seed| {
        let mut config = stochastic(20);
        config.seed = Some(seed);
        engine(
            &target,
            &mut DraftModelProposer::new(&draft, 3),
            &PROMPT,
            &config,
            3,
            None,
        )
    };
    let (a, b) = (run(11), run(11));
    assert_eq!(a.output.tokens, b.output.tokens);
    assert_eq!(a.output.tokens.len(), 20);
    assert_eq!(a.report.proposer, ProposerKind::DraftModel);
    assert!(a.stats.proposed > 0 && a.stats.accepted <= a.stats.proposed);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_stochastic_draft_model_reports_one_q_per_draft_and_is_seeded");
}

/// The last-position logits of a fresh prefill of `ids` through `model`, `[1, vocab]`.
fn last_logits<M: SpeculativeTarget>(model: &M, ids: &[i32]) -> Array {
    let mut cache = model.new_cache();
    model
        .forward(&mut cache, &input_ids(ids), 0, LogitsScope::Last, false)
        .unwrap()
        .logits
}

fn host_row(logits: &Array) -> Vec<f32> {
    logits
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// Two `(token, weight)` sets name the same tokens with weights equal to within `1e-4`
/// (after normalizing each: `accept_token` normalizes, so only the shape matters).
fn assert_distribution_eq(got: &[(i32, f32)], want: &[(i32, f32)], label: &str) {
    let norm = |d: &[(i32, f32)]| {
        let total: f32 = d.iter().map(|&(_, w)| w).sum();
        let mut d: Vec<(i32, f32)> = d.iter().map(|&(t, w)| (t, w / total)).collect();
        d.sort_by_key(|&(t, _)| t);
        d
    };
    let (got, want) = (norm(got), norm(want));
    assert_eq!(
        got.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
        want.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
        "{label}: tokens {got:?} vs {want:?}"
    );
    for (&(t, a), &(_, b)) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-4, "{label}: token {t}: {a} vs {b}");
    }
}

/// A [`DraftModelProposer`] that, after every commit, holds the draft's state against the
/// committed sequence: the draft cache plus the queued committed tokens are exactly the
/// committed history, the queue is that history's tail, and the draft's logits for the next
/// position equal a fresh draft prefill of the history — whatever rollback strategy the
/// draft's cache recovers through. Counts the commits that kept some but not all of the
/// drafts the proposal fed (a partial acceptance past the step start) and those that kept
/// every fed draft.
struct CheckedDraft<'d, D: SpeculativeTarget> {
    inner: DraftModelProposer<'d, D>,
    draft: &'d D,
    history: Vec<i32>,
    partial: usize,
    full: usize,
}

impl<'d, D: SpeculativeTarget> CheckedDraft<'d, D>
where
    D::Cache: Clone,
{
    fn new(draft: &'d D, drafts: usize) -> Self {
        Self {
            inner: DraftModelProposer::new(draft, drafts),
            draft,
            history: Vec::new(),
            partial: 0,
            full: 0,
        }
    }

    fn check(&self) {
        let (cache, pending, _) = self.inner.state();
        let cache = cache.expect("warmed");
        let held = self.draft.cache_len(cache) as usize;
        assert_eq!(
            held + pending.len(),
            self.history.len(),
            "draft cache {held} + queued {pending:?} != committed {}",
            self.history.len()
        );
        assert_eq!(
            pending,
            &self.history[held..],
            "the queue is the history's tail"
        );
        let probe = 5;
        let mut tail = pending.to_vec();
        tail.push(probe);
        let mut resumed = cache.clone();
        let got = self
            .draft
            .forward(
                &mut resumed,
                &input_ids(&tail),
                held as i32,
                LogitsScope::Last,
                false,
            )
            .unwrap()
            .logits;
        let mut all = self.history.clone();
        all.push(probe);
        let (got, want) = (host_row(&got), host_row(&last_logits(self.draft, &all)));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-3,
            "draft logits after commit drift from a fresh prefill by {worst}"
        );
        crate::model_inference_device::record_model_inference_device();
        crate::model_inference_device::record_model_inference_device();
        crate::model_inference_device::record_model_inference_device();
    }
}

impl<T, D> Proposer<T> for CheckedDraft<'_, D>
where
    T: SpeculativeTarget,
    D: SpeculativeTarget,
    D::Cache: Clone,
{
    fn kind(&self) -> ProposerKind {
        ProposerKind::DraftModel
    }
    fn warm(&mut self, t: &T, prompt: &[i32], h: Option<&Array>) -> Result<Option<Array>> {
        self.history = prompt.to_vec();
        self.inner.warm(t, prompt, h)
    }
    fn propose(
        &mut self,
        t: &T,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        self.inner.propose(t, ctx, sampler)
    }
    fn commit(
        &mut self,
        t: &T,
        cur: i32,
        accepted: &[i32],
        h: Option<&Array>,
        position: i32,
    ) -> Result<()> {
        let (_, _, fed) = self.inner.state();
        match fed {
            Some(fed) if fed > 0 && accepted.len() >= fed => self.full += 1,
            Some(fed) if !accepted.is_empty() && accepted.len() < fed => self.partial += 1,
            _ => {}
        }
        self.inner.commit(t, cur, accepted, h, position)?;
        self.history.push(cur);
        self.history.extend_from_slice(accepted);
        self.check();
        Ok(())
    }
}

/// sc-24436: after every commit the draft's cache is the committed sequence — its length
/// plus the queued tokens is the history, and its next logits are a fresh prefill's — for a
/// draft that restores its step start (the Qwen35 hybrid) and one that truncates, under
/// greedy, penalized and constrained targets whose verify accepts some fed drafts and
/// rejects the rest, and ones that accept every fed draft. Generic over the draft's own
/// rollback strategy, so a strategy that keeps K − 1 single-token draft forwards between one
/// `begin` and `recover` (the DeltaNet checkpoint ring) is held to the same invariant.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_draft_cache_is_the_committed_sequence_after_every_commit() {
    fn run_checked<T, D>(label: &str, target: &T, draft: &D, vocab: usize) -> (usize, usize)
    where
        T: SpeculativeTarget + crate::decode::Decode,
        D: SpeculativeTarget,
        D::Cache: Clone,
    {
        let (mut partial, mut full) = (0, 0);
        for (name, config, constrained, _) in configs()
            .into_iter()
            .filter(|c| !c.0.contains("stochastic"))
        {
            let (mut a, mut b) = (Forbid::new(vocab, &[1, 9]), Forbid::new(vocab, &[1, 9]));
            let expected = plain(target, &PROMPT, &config, constrained.then_some(&mut a));
            let mut proposer = CheckedDraft::new(draft, 4);
            let run = engine(
                target,
                &mut proposer,
                &PROMPT,
                &config,
                4,
                constrained.then_some(&mut b),
            );
            assert_eq!(run.output.tokens, expected.tokens, "{label} {name}");
            partial += proposer.partial;
            full += proposer.full;
        }
        (partial, full)
    }
    // The hybrid draft (its checkpoint ring) for a softmax target that disagrees with it, and
    // for an identical hybrid target that keeps every fed draft.
    let (partial, _) = run_checked("causal/hybrid", &causal_model(50, 2), &qwen35(false), 50);
    let (_, full) = run_checked("hybrid/hybrid", &qwen35(false), &qwen35(false), 50);
    assert!(
        partial > 0 && full > 0,
        "hybrid draft: {partial} partial and {full} full keeps"
    );
    // A truncating draft (layer skip) for its deeper target.
    let (partial, full) = run_checked("causal/causal", &causal(), &causal_model(24, 1), 24);
    assert!(
        partial > 0 && full > 0,
        "causal draft: {partial} partial and {full} full keeps"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_draft_cache_is_the_committed_sequence_after_every_commit");
}

/// sc-24436: a stop token the draft proposes and the target accepts ends the run exactly as
/// the plain loop does, on the host draft path (a penalty, a near-zero temperature) and the
/// greedy one, for a truncating and a checkpoint-ring draft — and the draft's cache is
/// still the committed sequence after that commit (the proposal stopped drafting at the
/// stop without feeding it).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_draft_proposed_stop_token_accepted_by_the_target_ends_the_run() {
    fn check<T>(label: &str, target: &T)
    where
        T: SpeculativeTarget + crate::decode::Decode,
        T::Cache: Clone,
    {
        let mut cold = greedy(24);
        cold.sampling.temperature = 0.01;
        let mut light = greedy(24);
        light.sampling.presence_penalty = 0.01;
        for (name, mut config) in [("greedy", greedy(24)), ("penalized", light), ("cold", cold)] {
            // The draft is the target itself, so every draft — the stop included — is
            // accepted. The stream the stop is cut from: the plain loop's for a greedy
            // decision, the same seeded draft run's for a sampled one (speculation consumes
            // the seeded stream differently from the plain loop).
            // The draft's state is checked against the committed sequence after every commit,
            // the one that accepts the stop included.
            let draft_run = |config: &GenerationConfig| {
                engine(
                    target,
                    &mut CheckedDraft::new(target, 4),
                    &PROMPT,
                    config,
                    4,
                    None,
                )
            };
            let stream = if name == "cold" {
                draft_run(&config).output.tokens
            } else {
                plain(target, &PROMPT, &config, None).tokens
            };
            // A stop first appearing a few tokens in, so drafts carry it; it is not emitted.
            let end = (4..stream.len())
                .find(|&i| !stream[..i].contains(&stream[i]))
                .unwrap();
            config.stop_tokens = vec![stream[end]];
            let run = draft_run(&config);
            assert_eq!(run.output.tokens, stream[..end], "{label} {name}");
            assert_eq!(
                run.output.finish_reason,
                FinishReason::StopToken,
                "{label} {name}"
            );
            assert_eq!(run.stats.accepted, run.stats.proposed, "{label} {name}");
        }
    }
    check("causal", &causal());
    check("hybrid", &qwen35(false));
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_draft_proposed_stop_token_accepted_by_the_target_ends_the_run");
}

/// sc-24436: a draft whose padding differs from its target's proposes only its tokenizer's
/// ids, over logits the target's width, and greedy output is still the target's own.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_draft_padded_differently_proposes_only_tokenizer_ids() {
    let draft = causal_model(24, 1);
    let proposer = DraftModelProposer::new(&draft, 4).with_vocab(20, 28);
    let mut row = vec![0f32; 24];
    row[22] = 10.0;
    row[3] = 1.0;
    let shaped = host_row(&proposer.shaped(Array::from_slice(&row, &[1, 24])).unwrap());
    assert_eq!(shaped.len(), 28);
    assert_eq!(&shaped[..20], &row[..20]);
    assert!(shaped[20..].iter().all(|x| *x == f32::NEG_INFINITY));
    // Same width, proposable = the whole width: untouched.
    let untouched = DraftModelProposer::new(&draft, 4).with_vocab(24, 24);
    assert_eq!(
        host_row(&untouched.shaped(Array::from_slice(&row, &[1, 24])).unwrap()),
        row
    );
    // Through the engine: the target scores 24 ids, the draft proposes from its first 20.
    let target = causal();
    for config in [greedy(20), penalized(20)] {
        let expected = plain(&target, &PROMPT, &config, None);
        let mut proposer = DraftModelProposer::new(&draft, 4).with_vocab(20, 24);
        let run = engine(&target, &mut proposer, &PROMPT, &config, 4, None);
        assert_eq!(run.output.tokens, expected.tokens);
        assert!(run.stats.proposed > 0);
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_draft_padded_differently_proposes_only_tokenizer_ids");
}

/// The constraint is rewound only when something advanced it: never on the `off` loop, and on
/// a speculative run only after a proposal that sampled drafts (MTP) and after a decision over
/// drafts — so every rewind the engine issues moves the state back. (A `JsonMask` rewind
/// rebuilds and replays the grammar, so a redundant one per step made every constrained MLX
/// request quadratic.)
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_constraint_is_rewound_only_when_something_advanced_it() {
    let causal = causal();
    let mut forbid = Forbid::new(24, &[1, 9]);
    let run = engine(
        &causal,
        &mut NoProposer,
        &PROMPT,
        &greedy(20),
        4,
        Some(&mut forbid),
    );
    assert_eq!(run.output.tokens.len(), 20);
    assert_eq!(
        (forbid.rewinds, forbid.state_changing_rewinds),
        (0, 0),
        "off never rewinds"
    );

    // Prompt lookup never samples its drafts: only the decision over drafts rewinds.
    let mut forbid = Forbid::new(24, &[1, 9]);
    let run = engine(
        &causal,
        &mut NgramProposer::default(),
        &PROMPT,
        &greedy(20),
        4,
        Some(&mut forbid),
    );
    assert!(run.stats.proposed > 0, "{:?}", run.stats);
    assert!(forbid.rewinds > 0, "lookup rewinds its decisions");
    assert!(
        (forbid.rewinds as u64) < run.report.verify_steps,
        "no rewind on a step without drafts: {} rewinds over {} steps",
        forbid.rewinds,
        run.report.verify_steps
    );
    assert_eq!(
        forbid.rewinds, forbid.state_changing_rewinds,
        "lookup: every rewind moves the state"
    );

    // MTP samples its drafts: the proposal and the decision each rewind what they advanced.
    let mut forbid = Forbid::new(50, &[1, 9]);
    let run = engine(
        &qwen35(true),
        &mut MtpProposer::new(),
        &PROMPT,
        &greedy(20),
        3,
        Some(&mut forbid),
    );
    assert!(run.stats.proposed > 0, "{:?}", run.stats);
    assert!(forbid.rewinds > 0);
    assert_eq!(
        forbid.rewinds, forbid.state_changing_rewinds,
        "mtp: every rewind moves the state"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::the_constraint_is_rewound_only_when_something_advanced_it");
}

/// A cancel that trips during the prefill (here: while the proposer warms) ends the run before
/// its first draw — zero tokens, `Cancelled` — as the plain loop's per-step check does.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_cancel_during_prefill_emits_no_token() {
    struct CancelOnWarm(CancelFlag);
    impl<T: SpeculativeTarget + ?Sized> Proposer<T> for CancelOnWarm {
        fn kind(&self) -> ProposerKind {
            ProposerKind::None
        }
        fn warm(&mut self, _: &T, _: &[i32], _: Option<&Array>) -> Result<Option<Array>> {
            self.0.cancel();
            Ok(None)
        }
        fn propose(
            &mut self,
            _: &T,
            _: &ProposeContext<'_>,
            _: &mut DraftSampler<'_, '_>,
        ) -> Result<Proposal> {
            Ok(Proposal::default())
        }
        fn commit(&mut self, _: &T, _: i32, _: &[i32], _: Option<&Array>, _: i32) -> Result<()> {
            Ok(())
        }
    }
    let cancel = CancelFlag::new();
    let mut events = Vec::new();
    let run = generate_speculative(
        &causal(),
        &mut CancelOnWarm(cancel.clone()),
        SpeculativePrompt::Tokens(&PROMPT),
        &greedy(8),
        0,
        &cancel,
        &mut |e| events.push(e),
        EngineOptions::default(),
    )
    .unwrap();
    assert!(run.output.tokens.is_empty(), "{:?}", run.output.tokens);
    assert_eq!(run.output.finish_reason, FinishReason::Cancelled);
    assert_eq!(
        events,
        vec![StreamEvent::Done {
            reason: FinishReason::Cancelled,
            generated: 0
        }]
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_cancel_during_prefill_emits_no_token");
}

/// The sampler seam carries device-resident tokens: a sampler returning every plain-greedy draw
/// as an unevaluated on-device argmax drives the `off` loop and prompt lookup to exactly the
/// plain loop's tokens (the next forward consumes the array; the host id serves the rest).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_device_resident_sampler_drives_the_loop() {
    struct DeviceGreedy(MlxSampler, usize);
    impl TokenSampler for DeviceGreedy {
        fn params(&self) -> &SamplingParams {
            self.0.params()
        }
        fn sample(&mut self, l: &Array, h: &[i32], m: Option<&[bool]>) -> Result<SampledToken> {
            if m.is_some() || !self.0.params().is_plain_greedy() {
                return self.0.sample(l, h, m);
            }
            self.1 += 1;
            let id = mlx_rs::ops::indexing::argmax(&l.reshape(&[-1])?, None)?;
            Ok(SampledToken::Device(id))
        }
        fn argmax_rows(&mut self, l: &Array) -> Result<Vec<i32>> {
            self.0.argmax_rows(l)
        }
        fn distribution(
            &mut self,
            l: &Array,
            h: &[i32],
            m: Option<&[bool]>,
        ) -> Result<Vec<(i32, f32)>> {
            self.0.distribution(l, h, m)
        }
        fn uniform(&mut self) -> f32 {
            self.0.uniform()
        }
        fn path(&self) -> Option<SamplerPath> {
            Some(SamplerPath::Device)
        }
    }
    let model = causal();
    let config = greedy(16);
    let expected = plain(&model, &PROMPT, &config, None).tokens;
    for (label, drafts) in [("off", 0usize), ("prompt lookup", 4)] {
        let mut sampler = DeviceGreedy(MlxSampler::from_config(&config), 0);
        let mut proposer = NgramProposer::default();
        let run = generate_speculative(
            &model,
            &mut proposer,
            SpeculativePrompt::Tokens(&PROMPT),
            &config,
            drafts,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                sampler: Some(&mut sampler),
                ..EngineOptions::default()
            },
        )
        .unwrap();
        assert_eq!(run.output.tokens, expected, "{label}");
        assert!(sampler.1 > 0, "{label}: device tokens were drawn");
        assert_eq!(run.report.sampler, "device", "{label}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_device_resident_sampler_drives_the_loop");
}

/// Hand-built `[1, n, vocab]` logits, one row per slice.
fn rows(rows: &[&[f32]]) -> Array {
    let vocab = rows[0].len() as i32;
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Array::from_slice(&flat, &[1, rows.len() as i32, vocab])
}

/// The decision's bonus row sees the accepted drafts through the constraint: a stateful
/// constraint forbidding the last accepted token keeps the bonus off draft 0 even though draft
/// 0 is row 1's argmax.
#[test]
fn the_bonus_row_sees_accepted_drafts_through_the_constraint() {
    struct ForbidLast(Vec<bool>, Vec<i32>);
    impl ConstraintMask for ForbidLast {
        fn allowed(&mut self) -> &[bool] {
            self.0.iter_mut().for_each(|a| *a = true);
            if let Some(&last) = self.1.last() {
                self.0[last as usize] = false;
            }
            &self.0
        }
        fn accept(&mut self, token: i32) {
            self.1.push(token);
        }
    }
    impl RewindableConstraintMask for ForbidLast {
        fn checkpoint(&self) -> usize {
            self.1.len()
        }
        fn rewind(&mut self, checkpoint: usize) {
            self.1.truncate(checkpoint);
        }
    }
    let logits = rows(&[&[0.0, 0.0, 5.0, 0.0, 0.0], &[0.0, 0.0, 5.0, 4.0, 0.0]]);
    let config = greedy(4);
    let mut sampler = MlxSampler::from_config(&config);
    let mut constraint = ForbidLast(vec![true; 5], Vec::new());
    let decision = decide(
        &logits,
        &[2],
        &[],
        &[0, 1],
        &config,
        &mut sampler,
        Some(&mut constraint),
    )
    .unwrap();
    assert_eq!(decision.accepted, 1);
    assert_eq!(decision.committed, vec![2, 3], "the bonus avoids draft 0");
}

/// The decision's bonus row sees the accepted drafts through the penalty window: a presence
/// penalty on draft 0 — absent from the history — flips row 1's argmax only when the running
/// history carries the accepted draft.
#[test]
fn the_bonus_row_sees_accepted_drafts_through_the_penalty() {
    let logits = rows(&[&[0.0, 0.0, 5.0, 0.0, 0.0], &[0.0, 0.0, 5.0, 4.5, 0.0]]);
    let mut config = greedy(4);
    config.sampling.presence_penalty = 1.0;
    let mut sampler = MlxSampler::from_config(&config);
    let decision = decide(&logits, &[2], &[], &[0, 1], &config, &mut sampler, None).unwrap();
    assert_eq!(decision.accepted, 1);
    assert_eq!(
        decision.committed,
        vec![2, 3],
        "the penalty sees the accepted draft"
    );
}

/// E1 through the engine's own `decide`: over many seeds, the first committed token of a
/// stochastic verify step is distributed as the target's shaped distribution `p`, whether the
/// draft came with a point mass (n-gram) or the non-degenerate `q` it was drawn from (a draft
/// model / MTP head) — a chi-square test (ported from Candle's engine).
///
/// The rejection rule preserves `p` under *either* proposal distribution (a point mass is
/// exact conditionally on the draft), so the chi-square alone cannot tell whether `decide`
/// used the draft's real `q`. The acceptance rate can: with `q` it is `Σ min(p, q)`, with a
/// point mass `Σ q·p` — both pinned here, so dropping the reported `q` turns this red.
#[test]
fn stochastic_decision_preserves_the_target_distribution_chi_square() {
    let vocab = 5usize;
    let p_logits = [1.0f32, 2.2, 0.3, 1.7, -0.5];
    let config = GenerationConfig {
        max_new_tokens: 2,
        sampling: SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            ..SamplingParams::default()
        },
        seed: Some(0),
        stop_tokens: Vec::new(),
    };
    // Rows: position 0 (the draft's) and the bonus row (uniform, irrelevant to token 0).
    let logits = rows(&[&p_logits, &[0.0; 5]]);
    eval([&logits]).unwrap();
    let q: Vec<(i32, f32)> = vec![(0, 0.4), (1, 0.1), (2, 0.2), (3, 0.1), (4, 0.2)];
    let draw = |u: f32| {
        let mut target = u;
        for &(t, w) in &q {
            target -= w;
            if target <= 0.0 {
                return t;
            }
        }
        q[q.len() - 1].0
    };
    let max = p_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = p_logits.iter().map(|&x| f64::from(x - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    let n = 20_000usize;
    let p: Vec<f64> = weights.iter().map(|w| w / total).collect();
    let q_of = |t: usize| f64::from(q[t].1);
    let rate_with_q: f64 = (0..vocab).map(|t| p[t].min(q_of(t))).sum();
    let rate_point_mass: f64 = (0..vocab).map(|t| p[t] * q_of(t)).sum();
    let point_mass = |t: i32| vec![(t, 1.0f32)];
    let draft_q = |_: i32| q.clone();
    type DistOf<'a> = &'a dyn Fn(i32) -> Vec<(i32, f32)>;
    let cases: [(&str, DistOf<'_>, f64); 2] = [
        ("point mass", &point_mass, rate_point_mass),
        ("draft q", &draft_q, rate_with_q),
    ];
    for (name, dist_of, expected_rate) in cases {
        let mut proposals = SplitMix64::new(0x5eed);
        let mut sampler = MlxSampler::new(config.sampling, 0x24434);
        let mut counts = vec![0u64; vocab];
        let mut accepted = 0usize;
        for _ in 0..n {
            let proposed = draw(proposals.next_f32());
            let decision = decide(
                &logits,
                &[proposed],
                &[dist_of(proposed)],
                &[],
                &config,
                &mut sampler,
                None,
            )
            .unwrap();
            counts[decision.committed[0] as usize] += 1;
            accepted += decision.accepted;
        }
        let mut chi2 = 0.0f64;
        for (t, &c) in counts.iter().enumerate() {
            let expected = n as f64 * weights[t] / total;
            chi2 += (c as f64 - expected).powi(2) / expected;
        }
        // 4 degrees of freedom: chi-square 99.9 % critical value 18.47.
        assert!(
            chi2 < 18.47,
            "{name}: chi-square {chi2:.2} over counts {counts:?}"
        );
        // Binomial acceptance count within 5 standard deviations of its expectation.
        let mean = n as f64 * expected_rate;
        let sd = (n as f64 * expected_rate * (1.0 - expected_rate)).sqrt();
        assert!(
            (accepted as f64 - mean).abs() < 5.0 * sd,
            "{name}: {accepted} accepted of {n}, expected {mean:.0} ± {sd:.0}"
        );
    }
    assert!(
        rate_with_q - rate_point_mass > 0.2,
        "the rates are distinguishable"
    );
}

/// `StepTarget` runs any `Decode` model through the engine's `off` loop from a caller-prefilled
/// cache, token-for-token the plain `generate_from_prefill` loop, and refuses a draft verify.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_step_target_is_the_plain_prefilled_loop() {
    let model = causal();
    for (name, config, _, _) in configs()
        .into_iter()
        .filter(|c| !c.0.contains("constrained"))
    {
        let prefill = || {
            let mut cache = crate::decode::Decode::make_cache(&model);
            let logits =
                crate::decode::Decode::step(&model, &input_ids(&PROMPT), cache.as_mut(), 0)
                    .unwrap();
            (cache, logits)
        };
        let (mut cache, logits) = prefill();
        let expected = crate::decode::generate_from_prefill(
            &model,
            cache.as_mut(),
            logits,
            PROMPT.to_vec(),
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
        )
        .unwrap();
        let (mut cache, logits) = prefill();
        let run = generate_speculative(
            &StepTarget(&model),
            &mut NoProposer,
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits,
                hidden: None,
                history: &PROMPT,
                position_delta: 0,
            },
            &config,
            0,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions::default(),
        )
        .unwrap();
        assert_eq!(run.output.tokens, expected.tokens, "{name}");
        assert_eq!(run.output.finish_reason, expected.finish_reason, "{name}");
        assert_eq!(run.report.path, "step_model", "{name}");
        assert_accounting(name, &run, &config);
    }
    let mut cache = crate::decode::Decode::make_cache(&model);
    let err = StepTarget(&model)
        .forward(&mut cache, &input_ids(&[1, 2]), 0, LogitsScope::All, false)
        .unwrap_err();
    assert!(err.to_string().contains("cannot verify drafts"), "{err}");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_step_target_is_the_plain_prefilled_loop");
}

// ---- Pipelining and the device sampler (story sc-24439). ----

/// Temperature 0.7 / top-p 0.9 — the device sampler's request.
fn top_p(max_new_tokens: usize) -> GenerationConfig {
    let mut config = greedy(max_new_tokens);
    config.sampling.temperature = 0.7;
    config.sampling.top_p = 0.9;
    config
}

/// Stop the caller's way after this many tokens: `false` through the stop predicate, `true` by
/// tripping the cancel flag from the event sink.
type StopAfter = Option<(usize, bool)>;

/// An `off` run with `pipelining`, stopping the caller's way when asked ([`StopAfter`]).
fn off_run<T: SpeculativeTarget>(
    target: &T,
    config: &GenerationConfig,
    pipelining: Pipelining,
    stop_after: StopAfter,
) -> SpeculativeRun {
    off_run_from(
        target,
        SpeculativePrompt::Tokens(&PROMPT),
        config,
        pipelining,
        stop_after,
    )
}

/// [`off_run`] from `prompt` (a caller-prefilled cache, say).
fn off_run_from<T: SpeculativeTarget>(
    target: &T,
    prompt: SpeculativePrompt<'_, T::Cache>,
    config: &GenerationConfig,
    pipelining: Pipelining,
    stop_after: StopAfter,
) -> SpeculativeRun {
    let cancel = CancelFlag::new();
    let emitted = std::cell::Cell::new(0usize);
    let predicate = || matches!(stop_after, Some((n, false)) if emitted.get() >= n);
    let mut events = Vec::new();
    let run = generate_speculative(
        target,
        &mut NoProposer,
        prompt,
        config,
        0,
        &cancel,
        &mut |e| {
            if matches!(e, StreamEvent::Token { .. }) {
                emitted.set(emitted.get() + 1);
                if matches!(stop_after, Some((n, true)) if emitted.get() >= n) {
                    cancel.cancel();
                }
            }
            events.push(e);
        },
        EngineOptions {
            should_stop: Some(&predicate),
            pipelining,
            ..EngineOptions::default()
        },
    )
    .unwrap();
    let ids: Vec<i32> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::Token { id, .. } => Some(*id),
            StreamEvent::Done { .. } => None,
        })
        .collect();
    assert_eq!(ids, run.output.tokens, "nothing past the end is emitted");
    run
}

/// AC1: pipelining is invisible in the output. On the Causal and Qwen35 targets, greedy and
/// seeded device-sampled runs emit exactly the unpipelined run's tokens — to the budget, and
/// ending on a stop token, the caller's stop predicate and a cancel, where the enqueued
/// look-ahead is discarded unread (one forward, never emitted).
fn pipelining_is_invisible_on<T: SpeculativeTarget + crate::decode::Decode>(target: &T) {
    for (name, config) in [("greedy", greedy(20)), ("top-p", top_p(20))] {
        let off = off_run(target, &config, Pipelining::Off, None);
        let on = off_run(target, &config, Pipelining::Auto, None);
        assert_eq!(on.output.tokens, off.output.tokens, "{name}");
        assert_eq!(on.output.tokens.len(), 20, "{name}");
        assert_eq!((on.stats.pipelined, off.stats.pipelined), (18, 0), "{name}");
        assert_eq!(on.stats.discarded, 0, "{name}");
        assert_eq!(
            on.report.target_forwards, off.report.target_forwards,
            "{name}"
        );
        if name == "greedy" {
            assert_eq!(
                on.output.tokens,
                plain(target, &PROMPT, &config, None).tokens
            );
        }

        let mut stopping = config.clone();
        stopping.stop_tokens = vec![on.output.tokens[7]];
        let first_stop = on
            .output
            .tokens
            .iter()
            .position(|t| *t == on.output.tokens[7]);
        let ends: [(&str, &GenerationConfig, StopAfter); 3] = [
            ("stop token", &stopping, None),
            ("stop predicate", &config, Some((5, false))),
            ("cancel", &config, Some((5, true))),
        ];
        for (end, config, stop_after) in ends {
            let off = off_run(target, config, Pipelining::Off, stop_after);
            let on = off_run(target, config, Pipelining::Auto, stop_after);
            let label = format!("{name} / {end}");
            assert_eq!(on.output.tokens, off.output.tokens, "{label}");
            assert_eq!(on.output.finish_reason, off.output.finish_reason, "{label}");
            let (len, reason) = match end {
                "stop token" => (first_stop.unwrap(), FinishReason::StopToken),
                "stop predicate" => (5, FinishReason::Stopped),
                _ => (5, FinishReason::Cancelled),
            };
            assert_eq!(
                (on.output.tokens.len(), on.output.finish_reason),
                (len, reason),
                "{label}"
            );
            assert_eq!(
                on.stats.discarded, 1,
                "{label}: the look-ahead is discarded"
            );
            // E3: the report counts the discarded look-ahead, so the documented forward
            // invariant holds on every end — pipelined or not.
            assert_eq!(on.report.discarded_forwards, 1, "{label}");
            assert_eq!(off.report.discarded_forwards, 0, "{label}");
            for (run, which) in [(&on, "pipelined"), (&off, "unpipelined")] {
                let r = &run.report;
                assert_eq!(
                    r.target_forwards,
                    r.prefill_forwards + r.verify_steps + r.replay_forwards + r.discarded_forwards,
                    "{label} ({which}): forwards = prefill + verify + replay + discarded: {r:?}"
                );
            }
            assert_eq!(
                on.report.target_forwards,
                off.report.target_forwards + 1,
                "{label}: the discarded look-ahead is a forward that ran"
            );
            assert_eq!(on.report.verify_steps, off.report.verify_steps, "{label}");
        }
    }
}

#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn pipelining_is_invisible_on_the_causal_and_hybrid_targets() {
    pipelining_is_invisible_on(&causal());
    pipelining_is_invisible_on(&qwen35(false));
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::pipelining_is_invisible_on_the_causal_and_hybrid_targets");
}

/// sc-24446 (E5): the process switches the campaign isolates pipelining and the device sampler
/// with (`MLX_LLM_PIPELINING`, `MLX_LLM_DEVICE_SAMPLER`; here their thread-scoped layer) turn
/// each off for a request that would otherwise take it, and the greedy tokens do not move.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_pipelining_and_device_sampler_switches_turn_each_path_off() {
    use crate::switches::{DEVICE_SAMPLER, PIPELINING};
    fn check<T: SpeculativeTarget>(label: &str, model: &T) {
        let run = || off_run(model, &greedy(12), Pipelining::Auto, None);
        let on = run();
        assert_eq!(on.stats.pipelined, 10, "{label}: pipelined when allowed");
        assert_eq!(on.report.sampler, "device", "{label}");
        let unpipelined = PIPELINING.scoped(false, run);
        assert_eq!(unpipelined.stats.pipelined, 0, "{label}: the switch is off");
        assert_eq!(unpipelined.output.tokens, on.output.tokens, "{label}");
        let host = DEVICE_SAMPLER.scoped(false, run);
        assert_eq!(host.report.sampler, "host:reference", "{label}");
        assert_eq!(host.stats.pipelined, 0, "{label}: a host draw is read back");
        assert_eq!(host.output.tokens, on.output.tokens, "{label}");
    }
    check("causal", &causal());
    check("qwen35", &qwen35(false));
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::the_pipelining_and_device_sampler_switches_turn_each_path_off");
}

/// AC1 (speculative half): a run with a proposer is never pipelined — the proposer reads the
/// committed token on the host before it drafts — so prompt lookup and MTP report no pipelined
/// steps while the greedy output stays the plain loop's.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn speculative_runs_are_not_pipelined() {
    let causal = causal();
    let lookup = engine(
        &causal,
        &mut NgramProposer::default(),
        &PROMPT,
        &greedy(20),
        4,
        None,
    );
    let hybrid = qwen35(true);
    let mtp = engine(
        &hybrid,
        &mut MtpProposer::new(),
        &PROMPT,
        &greedy(20),
        2,
        None,
    );
    for (label, run) in [("prompt lookup", &lookup), ("mtp", &mtp)] {
        assert_eq!(run.stats.pipelined, 0, "{label}");
        assert!(run.stats.verify_steps > 0, "{label}");
    }
    assert_eq!(
        lookup.output.tokens,
        plain(&causal, &PROMPT, &greedy(20), None).tokens
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::speculative_runs_are_not_pipelined"
    );
}

/// AC2: a temperature 0.7 / top-p 0.9 request draws on the device (`sampler = device`) and
/// moves exactly one token id per step to the host — counted — pipelined or not; a greedy
/// request likewise; a penalized request copies the whole row every step.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_top_p_request_samples_on_the_device_one_id_per_step() {
    let model = causal();
    for pipelining in [Pipelining::Auto, Pipelining::Off] {
        for (name, config) in [("top-p", top_p(16)), ("greedy", greedy(16))] {
            let before = host_transfers();
            let run = off_run(&model, &config, pipelining, None);
            let moved = host_transfers().since(before);
            assert_eq!(run.report.sampler, "device", "{name} {pipelining:?}");
            assert_eq!(run.output.tokens.len(), 16);
            assert_eq!(
                moved,
                HostTransfers {
                    reads: 16,
                    elements: 16
                },
                "{name} {pipelining:?}: one id per step"
            );
        }
    }
    let before = host_transfers();
    let run = off_run(&model, &penalized(16), Pipelining::Auto, None);
    let moved = host_transfers().since(before);
    assert_eq!(run.report.sampler, "host:penalty");
    assert_eq!(
        moved.elements,
        16 * 24,
        "a penalized draw copies the vocab row"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_top_p_request_samples_on_the_device_one_id_per_step");
}

/// AC3: a repetition-penalty request samples on the host, says so with its reason, and is not
/// pipelined (its draw reads the history the unread token would extend); a constrained one
/// likewise with its own reason.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_penalized_or_constrained_request_reports_the_host_path_and_is_not_pipelined() {
    let model = causal();
    let mut repetition = top_p(12);
    repetition.sampling.repetition_penalty = 1.2;
    repetition.sampling.repetition_context = 16;
    let run = off_run(&model, &repetition, Pipelining::Auto, None);
    assert_eq!(run.report.sampler, "host:penalty");
    assert_eq!(run.stats.pipelined, 0);
    let mut forbid = Forbid::new(24, &[1, 9]);
    let run = engine(
        &model,
        &mut NoProposer,
        &PROMPT,
        &top_p(12),
        0,
        Some(&mut forbid),
    );
    assert_eq!(run.report.sampler, "host:constraint");
    assert_eq!(run.stats.pipelined, 0);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_penalized_or_constrained_request_reports_the_host_path_and_is_not_pipelined");
}

/// A target that records, at each forward, how many host reads the sampling seam had made
/// since the run began — when each step was enqueued relative to the token reads.
struct ReadsAtForward<'a, T> {
    inner: &'a T,
    start: HostTransfers,
    reads: std::cell::RefCell<Vec<u64>>,
}

impl<T: SpeculativeTarget> SpeculativeTarget for ReadsAtForward<'_, T> {
    type Cache = T::Cache;
    type Rollback = T::Rollback;

    fn new_cache(&self) -> T::Cache {
        self.inner.new_cache()
    }

    fn cache_len(&self, cache: &T::Cache) -> i32 {
        self.inner.cache_len(cache)
    }

    fn rollback(&self, width: usize) -> T::Rollback {
        self.inner.rollback(width)
    }

    fn forward(
        &self,
        cache: &mut T::Cache,
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        let reads = host_transfers().since(self.start).reads;
        self.reads.borrow_mut().push(reads);
        self.inner
            .forward(cache, ids, rope_offset, scope, want_hidden)
    }

    fn attention_label(&self) -> &'static str {
        self.inner.attention_label()
    }
}

/// The pipelining is real: step `t + 1`'s forward is enqueued before token `t` is read back.
/// With the prefill first, the `k`-th forward (`k >= 2`) runs after `k - 1` token reads under
/// `Auto` — one behind the unpipelined loop, whose `k`-th forward follows `k` reads. Step 1 is
/// the exception (sc-24446): token 0 is read back before it is enqueued, as unpipelined.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_pipelined_step_is_enqueued_before_the_previous_token_is_read() {
    let model = causal();
    for (pipelining, expected) in [
        (Pipelining::Auto, vec![0, 1, 1, 2, 3, 4, 5, 6]),
        (Pipelining::Off, vec![0, 1, 2, 3, 4, 5, 6, 7]),
    ] {
        let target = ReadsAtForward {
            inner: &model,
            start: host_transfers(),
            reads: Default::default(),
        };
        let run = off_run(&target, &top_p(8), pipelining, None);
        assert_eq!(run.output.tokens.len(), 8);
        assert_eq!(target.reads.into_inner(), expected, "{pipelining:?}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_pipelined_step_is_enqueued_before_the_previous_token_is_read");
}

/// `src` without its `#[cfg(...test...)] mod` blocks — the production code, wherever a test
/// module sits in the file (a `cfg(test)` module can precede production items).
fn without_test_modules(src: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim_start();
        if line.starts_with("#[cfg(") && line.contains("test") {
            // The item the attribute gates: past any further attributes and doc comments.
            let mut j = i + 1;
            while j < lines.len() {
                let next = lines[j].trim_start();
                if next.starts_with("#[") || next.starts_with("///") || next.starts_with("//!") {
                    j += 1;
                } else {
                    break;
                }
            }
            let item = lines.get(j).map_or("", |l| l.trim_start());
            let is_mod =
                item.starts_with("mod ") || (item.starts_with("pub") && item.contains(" mod "));
            if is_mod && item.trim_end().ends_with('{') {
                let mut depth = 0i64;
                let mut k = j;
                loop {
                    depth += lines[k].matches('{').count() as i64;
                    depth -= lines[k].matches('}').count() as i64;
                    k += 1;
                    if depth <= 0 || k >= lines.len() {
                        break;
                    }
                }
                i = k;
                continue;
            }
        }
        out.push_str(lines[i]);
        out.push('\n');
        i += 1;
    }
    out
}

/// Every device-to-host read a decode loop makes goes through the counted sampling seam
/// ([`SampledToken::resolve`], the sampler's row copy), so the AC2 counter sees them all: no
/// loop in `decode/` reads an array back itself. A source scan of the non-test code — every
/// line outside a `cfg(test)` module, before and after one.
#[test]
fn decode_loops_make_no_uncounted_host_read() {
    let sources = [
        ("engine.rs", include_str!("../engine.rs")),
        ("stream.rs", include_str!("../stream.rs")),
        ("prefix.rs", include_str!("../prefix.rs")),
        ("batch.rs", include_str!("../batch.rs")),
        ("continuous.rs", include_str!("../continuous.rs")),
        ("speculative.rs", include_str!("../speculative.rs")),
        ("proposers.rs", include_str!("../proposers.rs")),
    ];
    let host_reads = [
        ".item::<",
        ".item(",
        ".try_item",
        "as_slice::<",
        "try_as_slice",
        "as_slice_unchecked",
    ];
    for (name, src) in sources {
        let code = without_test_modules(src);
        // The scan covers the production code past the first `cfg(test)` item — the pipelined
        // loop after engine.rs's test counters, the MTP proposer after proposers.rs's test
        // accessor — and drops the test modules.
        let sentinel = match name {
            "engine.rs" => Some("fn pipelined_steps<"),
            "proposers.rs" => Some("pub struct MtpProposer<"),
            _ => None,
        };
        if let Some(sentinel) = sentinel {
            assert!(code.contains(sentinel), "{name}: {sentinel} not scanned");
        }
        assert!(!code.contains("fn decode_loops_make_no_uncounted_host_read"));
        for read in host_reads {
            assert!(
                !code.contains(read),
                "decode/{name} reads an array back outside the counted seam (`{read}`)"
            );
        }
    }
}

/// A pipelined run that ends with a discarded look-ahead reports the committed cache length —
/// the unpipelined run's — and a caller continuing its prefilled cache from there (truncating
/// the look-ahead row) gets a cold forward's logits.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_pipelined_end_reports_the_committed_cache_length() {
    let model = causal();
    let prefilled_run = |config: &GenerationConfig, pipelining, stop_after| {
        let mut cache = model.new_cache();
        let logits = SpeculativeTarget::forward(
            &model,
            &mut cache,
            &input_ids(&PROMPT),
            0,
            LogitsScope::Last,
            false,
        )
        .unwrap()
        .logits;
        let run = off_run_from(
            &model,
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits,
                hidden: None,
                history: &PROMPT,
                position_delta: 0,
            },
            config,
            pipelining,
            stop_after,
        );
        (run, cache)
    };
    let free = off_run(&model, &top_p(20), Pipelining::Off, None)
        .output
        .tokens;
    let mut stopping = top_p(20);
    stopping.stop_tokens = vec![free[7]];
    let ends: [(&str, &GenerationConfig, StopAfter); 4] = [
        ("stop token", &stopping, None),
        ("stop predicate", &top_p(20), Some((5, false))),
        ("cancel", &top_p(20), Some((5, true))),
        ("budget", &top_p(6), None),
    ];
    for (end, config, stop_after) in ends {
        let (off, off_cache) = prefilled_run(config, Pipelining::Off, stop_after);
        let (on, mut on_cache) = prefilled_run(config, Pipelining::Auto, stop_after);
        assert_eq!(on.output.tokens, off.output.tokens, "{end}");
        let tokens = &on.output.tokens;
        let fed = if on.output.finish_reason == FinishReason::StopToken {
            tokens.len()
        } else {
            tokens.len() - 1
        };
        let committed = PROMPT.len() as i32 + fed as i32;
        assert_eq!(off.committed_cache_len, committed, "{end}");
        assert_eq!(
            off_cache.offset(),
            committed,
            "{end}: off leaves the committed rows"
        );
        assert_eq!(on.committed_cache_len, committed, "{end}");
        let discarded = if end == "budget" { 0 } else { 1 };
        assert_eq!(on.stats.discarded, discarded, "{end}");
        assert_eq!(on_cache.offset(), committed + discarded as i32, "{end}");

        // Continue from the committed length with the next input: the last emitted token, or
        // the stop token that ended the run.
        let next = if fed == tokens.len() {
            stopping.stop_tokens[0]
        } else {
            tokens[fed]
        };
        on_cache.truncate(on.committed_cache_len).unwrap();
        let warm = model
            .decode_logits(&input_ids(&[next]), &mut on_cache, on.committed_cache_len)
            .unwrap();
        let mut sequence = PROMPT.to_vec();
        sequence.extend_from_slice(&tokens[..fed]);
        sequence.push(next);
        let mut cold_cache = model.new_cache();
        let cold = model
            .decode_logits(&input_ids(&sequence), &mut cold_cache, 0)
            .unwrap();
        let host = crate::primitives::kv_cache::testing::host;
        let (warm, cold) = (host(&warm), host(&cold));
        let drift = warm
            .iter()
            .zip(&cold)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(drift < 1e-4, "{end}: continuation drift {drift}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_pipelined_end_reports_the_committed_cache_length");
}

/// A seeded device-sampled run is reproducible, and a different seed draws differently: the
/// seed still drives the device draws (the documented change is only that they are not the
/// host sampler's per-seed draws).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn seeded_device_sampling_is_reproducible() {
    let model = causal();
    let a = off_run(&model, &top_p(24), Pipelining::Auto, None);
    let b = off_run(&model, &top_p(24), Pipelining::Auto, None);
    assert_eq!(a.output.tokens, b.output.tokens);
    let mut other = top_p(24);
    other.seed = Some(8);
    let c = off_run(&model, &other, Pipelining::Auto, None);
    assert_ne!(a.output.tokens, c.output.tokens);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::seeded_device_sampling_is_reproducible");
}

/// Pipelining overlap microbenchmark (manual, `--ignored`; prints, asserts nothing timed): a
/// synthetic ~0.25 GB decoder (hidden 1024, 6 layers, vocab 32k), 128 tokens, pipelining off
/// and on alternated seven times each, greedy and top-p; the median decode rate of each. The
/// output is identical either way (asserted); the rate shows the host work the look-ahead
/// hides behind the device. Run it with `--release` for representative host overhead.
#[test]
#[ignore = "manual timing microbenchmark"]
fn pipelining_overlap_microbench() {
    let model = synthetic_llama(1024, 6, 32_000);
    for (name, config) in [("greedy", greedy(128)), ("top-p", top_p(128))] {
        let mut tokens = None;
        let mut rates = [Vec::new(), Vec::new()];
        off_run(&model, &config, Pipelining::Auto, None); // warm the kernels
        for _ in 0..7 {
            for (slot, pipelining) in [Pipelining::Off, Pipelining::Auto].into_iter().enumerate() {
                let started = Instant::now();
                let run = off_run(&model, &config, pipelining, None);
                let secs = started.elapsed().as_secs_f64();
                rates[slot].push(run.output.tokens.len() as f64 / secs);
                let first = tokens.get_or_insert_with(|| run.output.tokens.clone());
                assert_eq!(first, &run.output.tokens, "{name}");
            }
        }
        for rates in &mut rates {
            rates.sort_by(f64::total_cmp);
        }
        eprintln!(
            "{name:>6}: off {:.1} tok/s, pipelined {:.1} tok/s (medians of 7)",
            rates[0][3], rates[1][3]
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::pipelining_overlap_microbench");
}

// ---- `auto`'s acceptance monitor and the first-token dispatch (sc-24446). ----

/// A target that records, at each forward, how many drawn tokens the decode loops had handed
/// to the device ([`dispatch_token`]) since the run began, and whether it asked for hidden
/// rows.
struct DispatchesAtForward<'a, T> {
    inner: &'a T,
    start: u64,
    dispatched: std::cell::RefCell<Vec<u64>>,
    hidden: std::cell::RefCell<Vec<bool>>,
}

impl<'a, T> DispatchesAtForward<'a, T> {
    fn new(inner: &'a T) -> Self {
        Self {
            inner,
            start: token_dispatches(),
            dispatched: Default::default(),
            hidden: Default::default(),
        }
    }
}

impl<T: SpeculativeTarget> SpeculativeTarget for DispatchesAtForward<'_, T> {
    type Cache = T::Cache;
    type Rollback = T::Rollback;

    fn new_cache(&self) -> T::Cache {
        self.inner.new_cache()
    }

    fn cache_len(&self, cache: &T::Cache) -> i32 {
        self.inner.cache_len(cache)
    }

    fn rollback(&self, width: usize) -> T::Rollback {
        self.inner.rollback(width)
    }

    fn forward(
        &self,
        cache: &mut T::Cache,
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        self.dispatched
            .borrow_mut()
            .push(token_dispatches() - self.start);
        self.hidden.borrow_mut().push(want_hidden);
        self.inner
            .forward(cache, ids, rope_offset, scope, want_hidden)
    }

    fn attention_label(&self) -> &'static str {
        self.inner.attention_label()
    }
}

/// Token 0 is handed to the device before step 1 is enqueued behind it, so its read-back
/// waits for the prefill and its draw only: at the `k`-th forward (`k >= 1`) `k` tokens have
/// been dispatched — token 0 included — where the pre-fix loop had dispatched `k - 1` (token
/// 0 then waited for step 1's whole forward). Every later step is still enqueued before the
/// previous token is read back ([`a_pipelined_step_is_enqueued_before_the_previous_token_is_read`]),
/// so the overlap is kept; the unpipelined loop dispatches nothing ahead of its reads.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_first_token_is_dispatched_before_step_one_is_enqueued() {
    let model = causal();
    for (pipelining, expected) in [
        (Pipelining::Auto, vec![0, 1, 2, 3, 4, 5, 6, 7]),
        (Pipelining::Off, vec![0; 8]),
    ] {
        let target = DispatchesAtForward::new(&model);
        let run = off_run(&target, &top_p(8), pipelining, None);
        assert_eq!(run.output.tokens.len(), 8);
        assert_eq!(target.dispatched.into_inner(), expected, "{pipelining:?}");
        let again = off_run(&model, &top_p(8), pipelining, None);
        assert_eq!(again.output.tokens, run.output.tokens, "{pipelining:?}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::the_first_token_is_dispatched_before_step_one_is_enqueued");
}

/// A target that records, at each forward, how many tokens the run had delivered to its event
/// sink (`emitted`, which the sink advances) — when each step was enqueued relative to the
/// stream.
struct EmittedAtForward<'a, T> {
    inner: &'a T,
    emitted: &'a std::cell::Cell<usize>,
    at: std::cell::RefCell<Vec<usize>>,
}

impl<T: SpeculativeTarget> SpeculativeTarget for EmittedAtForward<'_, T> {
    type Cache = T::Cache;
    type Rollback = T::Rollback;

    fn new_cache(&self) -> T::Cache {
        self.inner.new_cache()
    }

    fn cache_len(&self, cache: &T::Cache) -> i32 {
        self.inner.cache_len(cache)
    }

    fn rollback(&self, width: usize) -> T::Rollback {
        self.inner.rollback(width)
    }

    fn forward(
        &self,
        cache: &mut T::Cache,
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        self.at.borrow_mut().push(self.emitted.get());
        self.inner
            .forward(cache, ids, rope_offset, scope, want_hidden)
    }

    fn attention_label(&self) -> &'static str {
        self.inner.attention_label()
    }
}

/// sc-24446 (pipelined TTFT): the first token reaches the stream before step 1 is built or
/// enqueued — its delivery never waits for step 1's host-side graph build and dispatch — and
/// from step 2 on every step is still enqueued before the previous token is delivered (the
/// overlap). Greedy and seeded output is the unpipelined run's.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_first_token_is_delivered_before_step_one_is_enqueued() {
    for (name, config) in [("greedy", greedy(8)), ("top_p", top_p(8))] {
        for (pipelining, expected) in [
            (Pipelining::Auto, vec![0, 1, 1, 2, 3, 4, 5, 6]),
            (Pipelining::Off, vec![0, 1, 2, 3, 4, 5, 6, 7]),
        ] {
            let model = causal();
            let emitted = std::cell::Cell::new(0usize);
            let target = EmittedAtForward {
                inner: &model,
                emitted: &emitted,
                at: Default::default(),
            };
            let mut tokens = Vec::new();
            let run = generate_speculative(
                &target,
                &mut NoProposer,
                SpeculativePrompt::Tokens(&PROMPT),
                &config,
                0,
                &CancelFlag::new(),
                &mut |e| {
                    if let StreamEvent::Token { id, .. } = e {
                        emitted.set(emitted.get() + 1);
                        tokens.push(id);
                    }
                },
                EngineOptions {
                    pipelining,
                    ..EngineOptions::default()
                },
            )
            .unwrap();
            assert_eq!(tokens, run.output.tokens, "{name} {pipelining:?}");
            assert_eq!(
                target.at.into_inner(),
                expected,
                "{name} {pipelining:?}: tokens delivered at each forward"
            );
            let off = off_run(&model, &config, Pipelining::Off, None);
            assert_eq!(
                run.output.tokens, off.output.tokens,
                "{name} {pipelining:?}"
            );
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::the_first_token_is_delivered_before_step_one_is_enqueued");
}

/// sc-24446 (pipelined TTFT behind held tokens): when the caller holds its first tokens back
/// ([`EngineOptions::delivered`] — Qwen3's opening `<think>` emits nothing), its first
/// *delivered* token is read back before the step after it is enqueued, as token 0 is; the
/// look-ahead resumes once the caller has delivered, or after [`FIRST_DELIVERY_HOLD_STEPS`]
/// held tokens. Greedy and seeded output is the unpipelined run's.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_first_delivered_token_is_read_before_a_look_ahead_is_enqueued() {
    assert_eq!(
        FIRST_DELIVERY_HOLD_STEPS, 8,
        "the never-delivered row below assumes 8"
    );
    // (tokens the caller holds back before its first delivery, tokens delivered at each
    // forward of a 12-token run). The prefill is forward 0.
    let never = usize::MAX;
    let cases: [(Option<usize>, Pipelining, Vec<usize>); 6] = [
        (
            None,
            Pipelining::Auto,
            vec![0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ),
        (
            Some(0),
            Pipelining::Auto,
            vec![0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ),
        (
            Some(1),
            Pipelining::Auto,
            vec![0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ),
        (
            Some(3),
            Pipelining::Auto,
            vec![0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10],
        ),
        (
            Some(never),
            Pipelining::Auto,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10],
        ),
        (
            Some(1),
            Pipelining::Off,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ),
    ];
    for (name, config) in [("greedy", greedy(12)), ("top_p", top_p(12))] {
        let model = causal();
        let off = off_run(&model, &config, Pipelining::Off, None);
        for (held, pipelining, expected) in &cases {
            let emitted = std::cell::Cell::new(0usize);
            let target = EmittedAtForward {
                inner: &model,
                emitted: &emitted,
                at: Default::default(),
            };
            let delivered = || held.is_some_and(|held| emitted.get() > held);
            let run = generate_speculative(
                &target,
                &mut NoProposer,
                SpeculativePrompt::Tokens(&PROMPT),
                &config,
                0,
                &CancelFlag::new(),
                &mut |e| {
                    if let StreamEvent::Token { .. } = e {
                        emitted.set(emitted.get() + 1);
                    }
                },
                EngineOptions {
                    pipelining: *pipelining,
                    delivered: held.map(|_| &delivered as &dyn Fn() -> bool),
                    ..EngineOptions::default()
                },
            )
            .unwrap();
            assert_eq!(
                target.at.into_inner(),
                *expected,
                "{name} held {held:?} {pipelining:?}: tokens emitted at each forward"
            );
            assert_eq!(
                run.output.tokens, off.output.tokens,
                "{name} held {held:?} {pipelining:?}"
            );
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::the_first_delivered_token_is_read_before_a_look_ahead_is_enqueued");
}

/// sc-24446: a pipelined run never enqueues a forward past its budget, and a run that ends
/// where the loop could not foresee it (a stop token, a caller stop, a cancel) waits for its
/// discarded look-ahead before returning — no forward it enqueued is left in flight to be
/// charged to the next request's prefill. Every draw past the first is one forward, plus the
/// discarded look-ahead.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_pipelined_run_returns_with_no_look_ahead_in_flight() {
    let model = causal();
    let free = off_run(&model, &top_p(20), Pipelining::Off, None)
        .output
        .tokens;
    let mut stopping = top_p(20);
    stopping.stop_tokens = vec![free[7]];
    let ends: [(&str, GenerationConfig, StopAfter); 4] = [
        ("budget", top_p(6), None),
        ("stop token", stopping, None),
        ("stop predicate", top_p(20), Some((5, false))),
        ("cancel", top_p(20), Some((5, true))),
    ];
    for (end, config, stop_after) in ends {
        let target = DispatchesAtForward::new(&model);
        let drained = drained_tokens();
        let run = off_run(&target, &config, Pipelining::Auto, stop_after);
        let forwards = target.dispatched.into_inner().len();
        let draws = run.output.tokens.len()
            + usize::from(run.output.finish_reason == FinishReason::StopToken);
        let discarded = usize::from(end != "budget");
        assert_eq!(run.stats.discarded, discarded, "{end}");
        assert_eq!(
            forwards,
            draws + discarded,
            "{end}: no forward past the end"
        );
        assert_eq!(run.stats.forwards, forwards, "{end}");
        assert_eq!(
            drained_tokens() - drained,
            discarded as u64,
            "{end}: the discarded look-ahead was waited for before returning"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_pipelined_run_returns_with_no_look_ahead_in_flight");
}

/// What a [`Scripted`] proposer drafts every step.
#[derive(Clone, Copy, Debug)]
enum Script {
    /// The plain loop's own continuation: always accepted.
    Right,
    /// A token the greedy target never picks: always rejected.
    Wrong,
    /// Nothing: every step is an ordinary single-token draw.
    Empty,
    /// [`Script::Right`] for the first `n` proposals, [`Script::Wrong`] after: a late loser.
    RightFor(usize),
    /// The `n`-th proposal's (1-based) first `f(n)` drafts are right, the rest wrong.
    Pattern(fn(usize) -> usize),
}

/// A test proposer of a chosen kind drafting up to `max_drafts` tokens every step by its
/// [`Script`] against `expected` (the plain loop's tokens after `prompt_len` prompt ids).
/// Counts its proposals and commits.
struct Scripted {
    expected: Vec<i32>,
    prompt_len: usize,
    script: Script,
    kind: ProposerKind,
    vocab: i32,
    wants_hidden: bool,
    proposals: usize,
    commits: usize,
    /// Every [`Proposer::catch_up`]: the tokens, the preceding rows' count and the position.
    caught_up: Vec<(Vec<i32>, Option<i32>, i32)>,
}

impl Scripted {
    fn new(expected: &[i32], script: Script, kind: ProposerKind, vocab: i32) -> Self {
        Self {
            expected: expected.to_vec(),
            prompt_len: PROMPT.len(),
            script,
            kind,
            vocab,
            wants_hidden: false,
            proposals: 0,
            commits: 0,
            caught_up: Vec::new(),
        }
    }
}

impl<T: SpeculativeTarget + ?Sized> Proposer<T> for Scripted {
    fn kind(&self) -> ProposerKind {
        self.kind
    }
    fn wants_hidden(&self) -> bool {
        self.wants_hidden
    }
    fn warm(&mut self, _: &T, _: &[i32], _: Option<&Array>) -> Result<Option<Array>> {
        Ok(None)
    }
    fn propose(
        &mut self,
        _: &T,
        ctx: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        self.proposals += 1;
        let at = ctx.history.len() - self.prompt_len;
        let drafts = match self.script {
            Script::Empty => Vec::new(),
            script => self.expected[at.min(self.expected.len())..]
                .iter()
                .take(ctx.max_drafts)
                .enumerate()
                .map(|(i, &t)| match script {
                    Script::Right => t,
                    Script::RightFor(n) if self.proposals <= n => t,
                    Script::Pattern(f) if i < f(self.proposals) => t,
                    _ => (t + 1) % self.vocab,
                })
                .collect(),
        };
        Ok(Proposal {
            drafts,
            dists: Vec::new(),
        })
    }
    fn commit(&mut self, _: &T, _: i32, _: &[i32], _: Option<&Array>, _: i32) -> Result<()> {
        self.commits += 1;
        Ok(())
    }
    fn catch_up(
        &mut self,
        _: &T,
        tokens: &[i32],
        preceding_hidden: Option<&Array>,
        position: i32,
    ) -> Result<()> {
        let rows = preceding_hidden.map(|h| h.shape()[1]);
        self.caught_up.push((tokens.to_vec(), rows, position));
        Ok(())
    }
}

/// One engine run from `prompt` requesting `mode`, with `pipelining`.
fn monitored_from<T, P>(
    target: &T,
    proposer: &mut P,
    prompt: &[i32],
    config: &GenerationConfig,
    drafts: usize,
    mode: core_llm::Speculative,
    pipelining: Pipelining,
) -> SpeculativeRun
where
    T: SpeculativeTarget,
    P: Proposer<T> + ?Sized,
{
    // Untimed: the static thresholds decide, no plain probe (the timed monitor has its own
    // tests below, on a deterministic clock).
    core_llm::with_decode_clock(None, || {
        run_streamed(target, proposer, prompt, config, drafts, mode, pipelining)
    })
}

/// One engine run, its streamed ids checked against the returned tokens.
fn run_streamed<T, P>(
    target: &T,
    proposer: &mut P,
    prompt: &[i32],
    config: &GenerationConfig,
    drafts: usize,
    mode: core_llm::Speculative,
    pipelining: Pipelining,
) -> SpeculativeRun
where
    T: SpeculativeTarget,
    P: Proposer<T> + ?Sized,
{
    let mut ids = Vec::new();
    let run = generate_speculative(
        target,
        proposer,
        SpeculativePrompt::Tokens(prompt),
        config,
        drafts,
        &CancelFlag::new(),
        &mut |e| {
            if let StreamEvent::Token { id, .. } = e {
                ids.push(id);
            }
        },
        EngineOptions {
            speculative_mode: mode,
            pipelining,
            ..EngineOptions::default()
        },
    )
    .unwrap();
    assert_eq!(ids, run.output.tokens, "streamed ids == returned tokens");
    assert_demotion_agrees(&run);
    run
}

/// The report's two demotion fields agree: `speculative_demoted_at` is set exactly when the
/// monitor's last decision demoted (sc-24446).
fn assert_demotion_agrees(run: &SpeculativeRun) {
    let r = &run.report;
    assert_eq!(
        r.speculative_demoted_at.is_some(),
        r.speculative_monitor.is_some_and(|d| d.demoted),
        "demoted_at vs monitor: {r:?}"
    );
}

/// sc-24446: a run whose budget, or a stop token, ends it on the very step that closes the
/// demoting window still reports where it was demoted — `speculative_demoted_at` agrees with
/// the monitor's demoting decision — pipelined (lookup) or not (MTP kind).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_demotion_on_the_runs_last_step_is_reported() {
    let model = causal();
    for (pipelining, kind) in [
        (Pipelining::Auto, ProposerKind::PromptLookup),
        (Pipelining::Off, ProposerKind::Mtp),
    ] {
        let label = format!("{pipelining:?} {kind:?}");
        let config = greedy(DEMOTED_AT as usize);
        let expected = plain(&model, &PROMPT, &config, None).tokens;
        let mut wrong = Scripted::new(&expected, Script::Wrong, kind, 24);
        let run = monitored(&model, &mut wrong, &config, 4, AUTO, pipelining);
        assert_eq!(run.output.tokens, expected, "{label}");
        assert_eq!(run.output.finish_reason, FinishReason::MaxTokens, "{label}");
        assert!(
            run.report.speculative_monitor.is_some_and(|d| d.demoted),
            "{label}: the last window demoted"
        );
        assert_eq!(
            run.report.speculative_demoted_at,
            Some(DEMOTED_AT),
            "{label}"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_demotion_on_the_runs_last_step_is_reported");
}

/// A deterministic decode clock (sc-24446): time moves only when [`Timed`] runs a forward.
#[derive(Default)]
struct StepClock(std::cell::Cell<std::time::Duration>);

impl core_llm::DecodeClock for StepClock {
    fn now(&self) -> std::time::Duration {
        self.0.get()
    }
}

/// A target whose every forward advances `clock`: a single-token forward by `plain`, a wider
/// one by `ratio × plain` — a verify cost `r = ratio`.
struct Timed<'a, T> {
    inner: &'a T,
    clock: std::rc::Rc<StepClock>,
    plain: std::time::Duration,
    ratio: f64,
}

impl<T: SpeculativeTarget> SpeculativeTarget for Timed<'_, T> {
    type Cache = T::Cache;
    type Rollback = T::Rollback;

    fn new_cache(&self) -> T::Cache {
        self.inner.new_cache()
    }

    fn cache_len(&self, cache: &T::Cache) -> i32 {
        self.inner.cache_len(cache)
    }

    fn rollback(&self, width: usize) -> T::Rollback {
        self.inner.rollback(width)
    }

    fn forward(
        &self,
        cache: &mut T::Cache,
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        let cost = if ids.size() == 1 {
            self.plain
        } else {
            self.plain.mul_f64(self.ratio)
        };
        self.clock.0.set(self.clock.0.get() + cost);
        self.inner
            .forward(cache, ids, rope_offset, scope, want_hidden)
    }

    fn attention_label(&self) -> &'static str {
        self.inner.attention_label()
    }
}

/// One `auto` engine run of `target` timed by a [`StepClock`] at verify cost `ratio`.
fn timed_run<T, P>(
    target: &T,
    proposer: &mut P,
    config: &GenerationConfig,
    drafts: usize,
    ratio: f64,
    pipelining: Pipelining,
) -> SpeculativeRun
where
    T: SpeculativeTarget,
    P: for<'t> Proposer<Timed<'t, T>> + ?Sized,
{
    let clock = std::rc::Rc::new(StepClock::default());
    let timed = Timed {
        inner: target,
        clock: clock.clone(),
        plain: std::time::Duration::from_millis(10),
        ratio,
    };
    core_llm::with_decode_clock(Some(clock), || {
        run_streamed(&timed, proposer, &PROMPT, config, drafts, AUTO, pipelining)
    })
}

/// [`monitored_from`] from [`PROMPT`].
fn monitored<T, P>(
    target: &T,
    proposer: &mut P,
    config: &GenerationConfig,
    drafts: usize,
    mode: core_llm::Speculative,
    pipelining: Pipelining,
) -> SpeculativeRun
where
    T: SpeculativeTarget,
    P: Proposer<T> + ?Sized,
{
    monitored_from(target, proposer, &PROMPT, config, drafts, mode, pipelining)
}

const AUTO: core_llm::Speculative = core_llm::Speculative::Auto;
const WINDOW: usize = core_llm::ACCEPTANCE_PROBE_VERIFIES as usize;

/// The token count at which a never-accepting proposer is demoted: the first token plus one
/// committed token per probe-window verify step.
const DEMOTED_AT: u64 = 1 + WINDOW as u64;

/// sc-24446: under `auto`, prompt lookup below its break-even on the pipelinable MLX path is
/// demoted at the end of its first window — never proposed to or committed to again — and the
/// rest of the run is the pipelined plain loop (one handoff step, then every remaining token
/// pipelined); the output is the plain greedy loop's, the report records the demotion, and
/// the forward accounting holds — on the causal and the hybrid (checkpoint-ring) targets.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn auto_demotes_a_losing_proposer_to_the_pipelined_plain_loop() {
    fn check<T: SpeculativeTarget + crate::decode::Decode>(label: &str, target: &T, vocab: i32) {
        let config = greedy(40);
        let expected = plain(target, &PROMPT, &config, None).tokens;
        let mut wrong = Scripted::new(&expected, Script::Wrong, ProposerKind::PromptLookup, vocab);
        let run = monitored(target, &mut wrong, &config, 4, AUTO, Pipelining::Auto);
        assert_eq!(
            run.output.tokens, expected,
            "{label}: demotion changed the output"
        );
        assert_eq!(
            run.report.speculative_demoted_at,
            Some(DEMOTED_AT),
            "{label}"
        );
        assert_eq!(
            (wrong.proposals, wrong.commits),
            (WINDOW, WINDOW),
            "{label}: the proposer is not driven after the demotion"
        );
        // After the handoff step, every remaining token but the last is a pipelined
        // look-ahead.
        assert_eq!(
            run.stats.pipelined,
            40 - DEMOTED_AT as usize - 1,
            "{label}: the rest is pipelined"
        );
        assert_eq!(run.report.proposer, ProposerKind::PromptLookup, "{label}");
        assert_eq!(run.report.accepted_tokens, 0, "{label}");
        assert_eq!(run.report.proposed_tokens, 4 * WINDOW as u64, "{label}");
        assert_accounting(label, &run, &config);
    }
    check("causal", &causal(), 24);
    check("qwen35", &qwen35(false), 50);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::auto_demotes_a_losing_proposer_to_the_pipelined_plain_loop");
}

/// E5 (sc-24446 review): prompt lookup is demoted only where the demoted request would run
/// MLX's pipelined loop — the regime its break-even was measured in. Where the plain loop is
/// unpipelined anyway (`Pipelining::Off`, the pipelining switch off, a history-reading
/// sampler) the same losing lookup runs to the end; an MTP-kind proposer is demoted there
/// all the same (its cost is its own forwards) and the rest is single-token verify steps.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn lookup_is_demoted_only_where_the_plain_loop_is_pipelined() {
    let model = causal();
    let config = greedy(40);
    let mut penalized = penalized(40);
    penalized.sampling.temperature = 0.0;
    type Run<'a> = Box<dyn Fn(&mut Scripted) -> SpeculativeRun + 'a>;
    let unpipelined: [(&str, &GenerationConfig, Run<'_>); 3] = [
        (
            "pipelining off",
            &config,
            Box::new(|p| monitored(&model, p, &config, 4, AUTO, Pipelining::Off)),
        ),
        (
            "switch off",
            &config,
            Box::new(|p| {
                crate::switches::PIPELINING.scoped(false, || {
                    monitored(&model, p, &config, 4, AUTO, Pipelining::Auto)
                })
            }),
        ),
        (
            "penalized",
            &penalized,
            Box::new(|p| monitored(&model, p, &penalized, 4, AUTO, Pipelining::Auto)),
        ),
    ];
    for (label, config, run) in unpipelined {
        let expected = plain(&model, &PROMPT, config, None).tokens;
        let mut lookup = Scripted::new(&expected, Script::Wrong, ProposerKind::PromptLookup, 24);
        let kept = run(&mut lookup);
        assert_eq!(kept.output.tokens, expected, "{label}");
        assert_eq!(
            kept.report.speculative_demoted_at, None,
            "{label}: lookup kept"
        );
        assert_eq!(
            lookup.proposals as u64,
            kept.report.verify_steps - 1,
            "{label}"
        );
        let mut head = Scripted::new(&expected, Script::Wrong, ProposerKind::Mtp, 24);
        let demoted = run(&mut head);
        assert_eq!(demoted.output.tokens, expected, "{label}");
        assert_eq!(
            demoted.report.speculative_demoted_at,
            Some(DEMOTED_AT),
            "{label}"
        );
        assert_eq!(
            demoted.stats.pipelined, 0,
            "{label}: nothing to pipeline into"
        );
        assert_eq!((head.proposals, head.commits), (WINDOW, WINDOW), "{label}");
        assert_accounting(label, &demoted, config);
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::lookup_is_demoted_only_where_the_plain_loop_is_pipelined");
}

/// A demoted run stops asking the target for hidden rows: the prefill and the first window's
/// verify steps do (the proposer wants them), no forward after the demotion does — pipelined
/// or not.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_demoted_run_asks_for_no_hidden_rows() {
    let model = qwen35(false);
    let config = greedy(40);
    let expected = plain(&model, &PROMPT, &config, None).tokens;
    for pipelining in [Pipelining::Auto, Pipelining::Off] {
        let target = DispatchesAtForward::new(&model);
        let mut wrong = Scripted::new(&expected, Script::Wrong, ProposerKind::Mtp, 50);
        wrong.wants_hidden = true;
        let run = monitored(&target, &mut wrong, &config, 2, AUTO, pipelining);
        assert_eq!(run.output.tokens, expected, "{pipelining:?}");
        assert_eq!(run.report.speculative_demoted_at, Some(DEMOTED_AT));
        let hidden = target.hidden.into_inner();
        let window = 1 + WINDOW;
        assert!(
            hidden[..window].iter().all(|&h| h),
            "{pipelining:?}: {hidden:?}"
        );
        assert!(
            hidden[window..].iter().all(|&h| !h),
            "{pipelining:?}: {hidden:?}"
        );
        assert_eq!(hidden.len() as u64, run.report.target_forwards);
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_demoted_run_asks_for_no_hidden_rows");
}

/// The edges of the handoff: a budget one token past the demotion point (the handoff step
/// draws the last token, nothing is enqueued after it), and a stop token drawn by the handoff
/// step itself (the run ends there; on the pipelined path the look-ahead enqueued behind it
/// is discarded unread) — pipelined (lookup) and unpipelined (MTP kind), each the plain
/// loop's output.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_handoff_honours_the_budget_and_a_stop_token_it_draws() {
    let model = causal();
    for (pipelining, kind) in [
        (Pipelining::Auto, ProposerKind::PromptLookup),
        (Pipelining::Off, ProposerKind::Mtp),
    ] {
        let label = format!("{pipelining:?} {kind:?}");
        // Budget = demotion point + 1.
        let config = greedy(DEMOTED_AT as usize + 1);
        let expected = plain(&model, &PROMPT, &config, None).tokens;
        let mut wrong = Scripted::new(&expected, Script::Wrong, kind, 24);
        let run = monitored(&model, &mut wrong, &config, 4, AUTO, pipelining);
        assert_eq!(run.output.tokens, expected, "{label}");
        assert_eq!(run.output.finish_reason, FinishReason::MaxTokens, "{label}");
        assert_eq!(
            run.report.speculative_demoted_at,
            Some(DEMOTED_AT),
            "{label}"
        );
        assert_eq!(run.stats.pipelined, 0, "{label}: nothing past the budget");
        assert_eq!(run.stats.discarded, 0, "{label}");
        assert_accounting(&label, &run, &config);

        // A stop token first drawn by the handoff step (token index `DEMOTED_AT`): a prompt
        // whose plain continuation draws a fresh token there.
        let (prompt, stop) = handoff_stop_prompt(&model);
        let mut stopping = greedy(40);
        stopping.stop_tokens = vec![stop];
        let expected = plain(&model, &prompt, &stopping, None);
        assert_eq!(
            expected.tokens.len(),
            DEMOTED_AT as usize,
            "{label}: fixture premise"
        );
        let mut wrong = Scripted::new(&expected.tokens, Script::Wrong, kind, 24);
        wrong.prompt_len = prompt.len();
        let run = monitored_from(&model, &mut wrong, &prompt, &stopping, 4, AUTO, pipelining);
        assert_eq!(run.output.tokens, expected.tokens, "{label}");
        assert_eq!(run.output.finish_reason, FinishReason::StopToken, "{label}");
        assert_eq!(
            run.report.speculative_demoted_at,
            Some(DEMOTED_AT),
            "{label}"
        );
        let discarded = usize::from(pipelining == Pipelining::Auto);
        assert_eq!(run.stats.discarded, discarded, "{label}: the look-ahead");
        let r = &run.report;
        assert_eq!(
            r.target_forwards,
            r.prefill_forwards + r.verify_steps + r.replay_forwards + r.discarded_forwards,
            "{label}: {r:?}"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::the_handoff_honours_the_budget_and_a_stop_token_it_draws");
}

/// A prompt (and the token) whose plain greedy continuation on `model` first draws that token
/// at index [`DEMOTED_AT`] — the handoff step's draw.
fn handoff_stop_prompt(model: &CausalLm) -> (Vec<i32>, i32) {
    let config = greedy(DEMOTED_AT as usize + 1);
    for seed in 0..64i32 {
        let prompt: Vec<i32> = (0..12)
            .map(|i| (i * 7 + seed * 5 + i * i * seed) % 24)
            .collect();
        let tokens = plain(model, &prompt, &config, None).tokens;
        let at = DEMOTED_AT as usize;
        if !tokens[..at].contains(&tokens[at]) {
            return (prompt, tokens[at]);
        }
    }
    panic!("no fixture prompt draws a fresh token at the handoff step");
}

/// A seeded stochastic run across a demotion draws exactly the plain seeded run's tokens: a
/// proposer that never drafts makes every verify step the ordinary draw, the demotion hands
/// the same sampler (and its stream) to the plain loop — pipelined (lookup) or not (MTP kind).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_seeded_stochastic_run_across_a_demotion_is_the_plain_seeded_run() {
    let model = causal();
    for (pipelining, kind) in [
        (Pipelining::Auto, ProposerKind::PromptLookup),
        (Pipelining::Off, ProposerKind::Mtp),
    ] {
        let config = top_p(40);
        let reference = off_run(&model, &config, pipelining, None).output.tokens;
        let mut empty = Scripted::new(&[], Script::Empty, kind, 24);
        let run = monitored(&model, &mut empty, &config, 4, AUTO, pipelining);
        assert_eq!(run.output.tokens, reference, "{pipelining:?} {kind:?}");
        assert_eq!(run.report.speculative_demoted_at, Some(DEMOTED_AT));
        assert_eq!(run.report.sampler, "device");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_seeded_stochastic_run_across_a_demotion_is_the_plain_seeded_run");
}

/// A proposer at or above its break-even is never demoted under `auto`: it proposes to the
/// end, nothing is pipelined, and the output is still the plain loop's.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn auto_keeps_a_proposer_that_pays_for_itself() {
    let model = causal();
    let config = greedy(120);
    let expected = plain(&model, &PROMPT, &config, None).tokens;
    let mut right = Scripted::new(&expected, Script::Right, ProposerKind::PromptLookup, 24);
    let run = monitored(&model, &mut right, &config, 4, AUTO, Pipelining::Auto);
    assert_eq!(run.output.tokens, expected);
    assert_eq!(run.report.speculative_demoted_at, None);
    assert_eq!(run.stats.pipelined, 0);
    assert!(
        run.report.verify_steps > WINDOW as u64,
        "the run outlasts the first window: {:?}",
        run.report
    );
    // Every step proposes but one the budget clamped to no drafts.
    assert!(right.proposals as u64 + 1 >= run.report.verify_steps);
    assert_accounting("paying", &run, &config);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::auto_keeps_a_proposer_that_pays_for_itself");
}

/// sc-24446 (rolling windows): a lookup that pays for its first window and then stops
/// accepting is demoted at the end of its first failing window — not at the first window (it
/// passed), and not never (the one-shot decision kept it) — the rest handed to the pipelined
/// plain loop, and the output still the plain greedy loop's.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_late_loser_is_demoted_at_its_first_failing_window() {
    let model = causal();
    let config = greedy(120);
    let expected = plain(&model, &PROMPT, &config, None).tokens;
    let script = Script::RightFor(WINDOW);
    let mut late = Scripted::new(&expected, script, ProposerKind::PromptLookup, 24);
    let run = monitored(&model, &mut late, &config, 4, AUTO, Pipelining::Auto);
    assert_eq!(run.output.tokens, expected, "demotion changed the output");
    // The first window commits 5 tokens per step (4 accepted + the bonus), the failing one 1.
    let demoted_at = 1 + 5 * WINDOW + WINDOW;
    assert_eq!(run.report.speculative_demoted_at, Some(demoted_at as u64));
    assert_eq!((late.proposals, late.commits), (2 * WINDOW, 2 * WINDOW));
    assert_eq!(run.report.accepted_tokens, 4 * WINDOW as u64);
    // After the handoff step, every remaining token but the last is a pipelined look-ahead.
    assert_eq!(run.stats.pipelined, 120 - demoted_at - 1);
    assert_accounting("late loser", &run, &config);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_late_loser_is_demoted_at_its_first_failing_window");
}

/// The plain probe's length: its steps run without a proposal before the first window.
const PROBE: usize = core_llm::PLAIN_PROBE_MAX_STEPS as usize;

/// sc-24446 (cost-aware) on MLX's unpipelined path: at the same scale of acceptance, a
/// request whose verify steps are dear (r = 2.0, mal 0.75: gain 0.875, a loser but not a
/// clear one) is demoted on its measured gain at the end of its first window, one whose verify steps are cheap
/// (Qwen3.8-like: r = 1.5, mal 1.25) keeps its proposer — both after the plain probe, the
/// proposer caught up on the probed tokens, the plain loop's tokens throughout. Prompt lookup
/// (no static threshold here) is judged the same way.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_timed_auto_weighs_the_requests_own_verify_cost() {
    let model = qwen35(false);
    let config = greedy(96);
    let expected = plain(&model, &PROMPT, &config, None).tokens;
    let bonsai = Script::Pattern(|n| if n % 4 == 1 { 3 } else { 0 });
    let mut dear = Scripted::new(&expected, bonsai, ProposerKind::Mtp, 50);
    dear.wants_hidden = true;
    let run = timed_run(&model, &mut dear, &config, 3, 2.0, Pipelining::Off);
    assert_eq!(
        run.output.tokens, expected,
        "the probe and the demotion are output-neutral"
    );
    let demoted_at = 1 + PROBE + WINDOW + 12;
    assert_eq!(run.report.speculative_demoted_at, Some(demoted_at as u64));
    assert_eq!((dear.proposals, dear.commits), (WINDOW, PROBE + WINDOW));
    let d = run.report.speculative_monitor.expect("a judged window");
    assert_eq!(
        (d.basis, d.demoted, d.window),
        (core_llm::DemotionBasis::Measured, true, 1)
    );
    assert!((d.verify_cost_ratio().unwrap() - 2.0).abs() < 1e-9, "{d:?}");
    assert_eq!(
        dear.caught_up,
        vec![(
            expected[..PROBE].to_vec(),
            Some(PROBE as i32),
            PROMPT.len() as i32
        )]
    );
    assert_accounting("dear", &run, &config);

    let qwen = Script::Pattern(|n| if n % 4 == 1 { 2 } else { 1 });
    let mut cheap = Scripted::new(&expected, qwen, ProposerKind::Mtp, 50);
    cheap.wants_hidden = true;
    let run = timed_run(&model, &mut cheap, &config, 3, 1.5, Pipelining::Off);
    assert_eq!(run.output.tokens, expected);
    assert_eq!(run.report.speculative_demoted_at, None);
    let d = run.report.speculative_monitor.expect("a judged window");
    assert_eq!(
        (d.basis, d.demoted),
        (core_llm::DemotionBasis::Measured, false)
    );

    // Unpipelined lookup, never demoted by a static threshold, is demoted by its cost (gain
    // 1 / 1.15 = 0.87: at its window's end).
    let causal = causal();
    let config = greedy(40);
    let expected = plain(&causal, &PROMPT, &config, None).tokens;
    let mut wrong = Scripted::new(&expected, Script::Wrong, ProposerKind::PromptLookup, 24);
    let run = timed_run(&causal, &mut wrong, &config, 4, 1.15, Pipelining::Off);
    assert_eq!(run.output.tokens, expected);
    assert_eq!(
        run.report.speculative_demoted_at,
        Some((1 + PROBE + WINDOW) as u64)
    );
    assert_eq!(
        run.report.speculative_monitor.unwrap().basis,
        core_llm::DemotionBasis::Measured
    );
    assert_eq!(
        wrong.caught_up,
        vec![(expected[..PROBE].to_vec(), None, PROMPT.len() as i32)]
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_timed_auto_weighs_the_requests_own_verify_cost");
}

/// sc-24446 (cuda-campaign-b4): on the deterministic decode clock a clear loser — measured
/// gain below [`core_llm::CLEAR_LOSS_GAIN`] (a lookup accepting nothing at r = 2.4, gain
/// 0.42; an MTP-kind proposer at mal 0.75, r = 2.6, gain 0.67) — is demoted by the end of the
/// probe and its first window: at the first step its measurement holds
/// [`core_llm::CLEAR_LOSS_MIN_TIMED_STEPS`] timed steps past its shape's warm-up, or — the
/// MTP-kind one, too lumpy (3 drafts every fourth step) for the optimistic gain to call
/// early — at its first window's end. A winner (every
/// draft accepted, r = 1.5) is never demoted. Pipelined and not, the plain loop's tokens.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_timed_clear_loser_is_demoted_within_its_first_window_and_a_winner_never() {
    let first =
        core_llm::SHAPE_WARMUP_STEPS as usize + core_llm::CLEAR_LOSS_MIN_TIMED_STEPS as usize;
    let model = causal();
    let config = greedy(64);
    let expected = plain(&model, &PROMPT, &config, None).tokens;
    for pipelining in [Pipelining::Auto, Pipelining::Off] {
        let mut wrong = Scripted::new(&expected, Script::Wrong, ProposerKind::PromptLookup, 24);
        let run = timed_run(&model, &mut wrong, &config, 4, 2.4, pipelining);
        assert_eq!(run.output.tokens, expected, "{pipelining:?}");
        let at = run.report.speculative_demoted_at.expect("demoted") as usize;
        assert_eq!(at, 1 + PROBE + first, "{pipelining:?}");
        assert!(
            at <= 1 + PROBE + WINDOW,
            "{pipelining:?}: within the first window"
        );
        let d = run.report.speculative_monitor.unwrap();
        assert_eq!(
            (d.basis, d.demoted, d.window),
            (core_llm::DemotionBasis::Measured, true, 1),
            "{pipelining:?}"
        );
        assert!(d.gain().unwrap() < core_llm::CLEAR_LOSS_GAIN, "{d:?}");
        assert_eq!(wrong.proposals, first, "{pipelining:?}");
    }

    let hybrid = qwen35(false);
    let config = greedy(96);
    let expected = plain(&hybrid, &PROMPT, &config, None).tokens;
    let mut head = Scripted::new(
        &expected,
        Script::Pattern(|n| if n % 4 == 1 { 3 } else { 0 }),
        ProposerKind::Mtp,
        50,
    );
    head.wants_hidden = true;
    let run = timed_run(&hybrid, &mut head, &config, 3, 2.6, Pipelining::Off);
    assert_eq!(run.output.tokens, expected);
    let at = run.report.speculative_demoted_at.expect("demoted") as usize;
    assert!(
        at <= 1 + PROBE + WINDOW + 12,
        "within the first window: {at}"
    );
    assert_eq!(head.proposals, WINDOW);

    // Qwen3.8-like: mal 1.0 at r = 1.71 (the dearest measured depth-3 MTP verify), gain 1.17.
    let mut right = Scripted::new(
        &expected,
        Script::Pattern(|n| if n % 2 == 1 { 2 } else { 0 }),
        ProposerKind::Mtp,
        50,
    );
    right.wants_hidden = true;
    let run = timed_run(&hybrid, &mut right, &config, 3, 1.71, Pipelining::Off);
    assert_eq!(run.output.tokens, expected);
    assert_eq!(run.report.speculative_demoted_at, None);
    let d = run.report.speculative_monitor.expect("a judged window");
    assert_eq!(
        (d.basis, d.demoted),
        (core_llm::DemotionBasis::Measured, false)
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_timed_clear_loser_is_demoted_within_its_first_window_and_a_winner_never");
}

/// On MLX's pipelined path a measured loss demotes and hands the rest to the pipelined loop
/// (a lookup accepting one draft per step at r = 2.4 — above the static 0.4), while the static
/// threshold still demotes where the measured gain does not (accepting nothing at r = 1.0) —
/// the plain loop's tokens either way.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_timed_pipelined_run_demotes_on_a_measured_loss_or_its_static_threshold() {
    let model = causal();
    let config = greedy(48);
    let expected = plain(&model, &PROMPT, &config, None).tokens;
    for (script, ratio, basis, demoted_at) in [
        (
            Script::Pattern(|_| 1),
            2.4,
            core_llm::DemotionBasis::Measured,
            1 + PROBE + 2 * WINDOW,
        ),
        (
            Script::Wrong,
            1.0,
            core_llm::DemotionBasis::Static,
            1 + PROBE + WINDOW,
        ),
    ] {
        let mut lookup = Scripted::new(&expected, script, ProposerKind::PromptLookup, 24);
        let run = timed_run(&model, &mut lookup, &config, 4, ratio, Pipelining::Auto);
        assert_eq!(run.output.tokens, expected, "{ratio}");
        assert_eq!(
            run.report.speculative_demoted_at,
            Some(demoted_at as u64),
            "{ratio}"
        );
        let d = run.report.speculative_monitor.unwrap();
        assert_eq!((d.basis, d.demoted), (basis, true), "{ratio}");
        assert_eq!(
            run.stats.pipelined,
            48 - demoted_at - 1,
            "{ratio}: the rest is pipelined"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_timed_pipelined_run_demotes_on_a_measured_loss_or_its_static_threshold");
}

/// A clock that moves a fixed tick at every reading: every timed step measures the same
/// (`r = 1`), whatever the target.
#[derive(Default)]
struct TickClock(std::cell::Cell<std::time::Duration>);

impl core_llm::DecodeClock for TickClock {
    fn now(&self) -> std::time::Duration {
        let now = self.0.get() + std::time::Duration::from_millis(5);
        self.0.set(now);
        now
    }
}

/// The real MTP head through a timed run (the head is bound to its own model, so a
/// [`TickClock`] times it: every step the same, `r = 1`): the plain probe, the head caught up
/// on the probed tokens, its proposals — the plain loop's tokens throughout. Accepting nothing
/// at `r = 1` breaks even (measured gain 1.0), so on the unpipelined path the measured gain
/// keeps the head where the static 0.725 would have demoted it.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_timed_mtp_head_catches_up_after_the_probe() {
    let model = qwen35(true);
    let config = greedy(48);
    let expected = plain(&model, &PROMPT, &config, None).tokens;
    let clock = std::rc::Rc::new(TickClock::default());
    let run = core_llm::with_decode_clock(Some(clock), || {
        run_streamed(
            &model,
            &mut MtpProposer::new(),
            &PROMPT,
            &config,
            2,
            AUTO,
            Pipelining::Off,
        )
    });
    assert_eq!(run.output.tokens, expected);
    assert_eq!(
        run.report.accepted_tokens, 0,
        "fixture premise: the random head never pays"
    );
    assert_eq!(run.report.speculative_demoted_at, None);
    let d = run.report.speculative_monitor.unwrap();
    assert_eq!(
        (d.basis, d.demoted),
        (core_llm::DemotionBasis::Measured, false)
    );
    assert!(d.mean_accepted_length() < 0.725);
    // Every step after the probe proposed its two drafts (but the budget-clamped last two).
    let speculative = run.report.verify_steps - PROBE as u64;
    assert_eq!(
        run.report.proposed_tokens,
        2 * (speculative - 2) + 1,
        "{:?}",
        run.report
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_timed_mtp_head_catches_up_after_the_probe");
}

/// `MtpProposer::catch_up` leaves the head where proposing on each probed token would have
/// (its draft step for `cur`, then a commit without accepted drafts): the next proposal draws
/// the same drafts from the same distributions.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_mtp_head_caught_up_on_plain_steps_matches_one_stepped_through_them() {
    let model = qwen35(true);
    let config = top_p(8);
    let mut cache = model.new_cache();
    let out = SpeculativeTarget::forward(
        &model,
        &mut cache,
        &input_ids(&PROMPT),
        0,
        LogitsScope::Last,
        true,
    )
    .unwrap();
    let prompt_hidden = out.hidden.unwrap();
    let start = PROMPT.len() as i32;
    let heads = || {
        let mut head = MtpProposer::new();
        head.warm(&model, &PROMPT, Some(&prompt_hidden)).unwrap();
        head
    };
    let (mut caught, mut stepped, mut skipped) = (heads(), heads(), heads());
    let mut previous = last_row(&prompt_hidden).unwrap();
    let (mut tokens, mut rows) = (Vec::new(), Vec::new());
    let mut history = PROMPT.to_vec();
    let mut draw = MlxSampler::from_config(&config);
    for (i, token) in [5, 9, 2, 7].into_iter().enumerate() {
        let position = start + i as i32;
        let ctx = ProposeContext {
            cur: token,
            history: &history,
            previous_hidden: Some(&previous),
            position,
            max_drafts: 1,
        };
        let mut sampler = DraftSampler {
            config: &config,
            sampler: &mut draw,
            constraint: None,
            advanced: false,
        };
        stepped.propose(&model, &ctx, &mut sampler).unwrap();
        let step = SpeculativeTarget::forward(
            &model,
            &mut cache,
            &input_ids(&[token]),
            position,
            LogitsScope::Last,
            true,
        )
        .unwrap();
        let hidden = step.hidden.unwrap();
        stepped
            .commit(&model, token, &[], Some(&hidden), position + 1)
            .unwrap();
        tokens.push(token);
        rows.push(previous.clone());
        previous = last_row(&hidden).unwrap();
        history.push(token);
    }
    let rows: Vec<&Array> = rows.iter().collect();
    let preceding = mlx_rs::ops::concatenate_axis(&rows, 1).unwrap();
    caught
        .catch_up(&model, &tokens, Some(&preceding), start)
        .unwrap();
    let next = |head: &mut MtpProposer<'_>| {
        let ctx = ProposeContext {
            cur: 4,
            history: &history,
            previous_hidden: Some(&previous),
            position: start + tokens.len() as i32,
            max_drafts: 3,
        };
        let mut draw = MlxSampler::from_config(&config);
        let mut sampler = DraftSampler {
            config: &config,
            sampler: &mut draw,
            constraint: None,
            advanced: false,
        };
        head.propose(&model, &ctx, &mut sampler).unwrap()
    };
    let (a, b, c) = (next(&mut caught), next(&mut stepped), next(&mut skipped));
    let close = |x: &Proposal, y: &Proposal| {
        x.drafts == y.drafts
            && x.dists.len() == y.dists.len()
            && x.dists.iter().zip(&y.dists).all(|(p, q)| {
                p.len() == q.len()
                    && p.iter()
                        .zip(q)
                        .all(|(&(t, w), &(u, v))| t == u && (w - v).abs() < 1e-4)
            })
    };
    assert!(
        close(&a, &b),
        "caught-up head drifts: {:?} vs {:?}",
        a.drafts,
        b.drafts
    );
    assert!(
        !close(&c, &b),
        "fixture premise: the probed tokens change the head's state"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::the_mtp_head_caught_up_on_plain_steps_matches_one_stepped_through_them");
}

/// An explicit `{proposer, depth}` is the caller's choice: no monitor, so a losing proposer
/// runs to the end of the request — every step proposes — and nothing is demoted.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn an_explicit_proposer_request_is_never_demoted() {
    use core_llm::{Speculative, SpeculativeProposer};
    let model = causal();
    let config = greedy(40);
    let expected = plain(&model, &PROMPT, &config, None).tokens;
    for (kind, proposer) in [
        (
            ProposerKind::PromptLookup,
            SpeculativeProposer::PromptLookup,
        ),
        (ProposerKind::Mtp, SpeculativeProposer::Mtp),
    ] {
        let explicit = Speculative::proposer(proposer, 4);
        let mut wrong = Scripted::new(&expected, Script::Wrong, kind, 24);
        let run = monitored(&model, &mut wrong, &config, 4, explicit, Pipelining::Auto);
        assert_eq!(run.output.tokens, expected);
        assert_eq!(run.report.speculative_demoted_at, None, "{kind:?}");
        assert_eq!(run.stats.pipelined, 0);
        // Every step but the budget-clamped last one proposes.
        assert_eq!(wrong.proposals as u64, run.report.verify_steps - 1);
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::an_explicit_proposer_request_is_never_demoted");
}

/// The MTP head (a proposer that wants the target's hidden rows) demotes the same way on the
/// hybrid: the head is not committed to after the demotion and the rest is the pipelined
/// plain loop, with the plain greedy output.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_demoted_mtp_head_hands_the_hybrid_to_the_pipelined_loop() {
    let model = qwen35(true);
    let config = greedy(40);
    let expected = plain(&model, &PROMPT, &config, None).tokens;
    let run = monitored(
        &model,
        &mut MtpProposer::new(),
        &config,
        2,
        AUTO,
        Pipelining::Auto,
    );
    assert_eq!(run.output.tokens, expected);
    assert_eq!(
        run.report.accepted_tokens, 0,
        "fixture premise: the random head never pays"
    );
    assert_eq!(run.report.speculative_demoted_at, Some(DEMOTED_AT));
    assert_eq!(run.stats.pipelined, 40 - DEMOTED_AT as usize - 1);
    assert_eq!(run.report.proposer, ProposerKind::Mtp);
    assert_accounting("mtp", &run, &config);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tests::a_demoted_mtp_head_hands_the_hybrid_to_the_pipelined_loop");
}
