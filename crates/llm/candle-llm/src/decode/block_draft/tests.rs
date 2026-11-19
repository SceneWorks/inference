//! The Candle block-draft proposer's model-inference proofs (SW-7197): a tiny random draft of
//! each format — DSpine and DFlash2 — over the tiny `qwen3_5` target through the engine's linear
//! verifier, the MLX proofs' fixtures and checks. Every test that runs learned forwards is ignored
//! in ordinary CI and registered for the explicit model-proof workflow (the CPU by default, the
//! CUDA device in a `cuda` build, the CUDA-graph runner on); the block-length rule is plain
//! arithmetic.

use std::collections::HashMap;

use candle_core::{DType, Device};
use core_llm::{BlockDraftFormat, BlockDraftPrecision};

use super::*;
use crate::decode::cancel::CancelFlag;
use crate::decode::engine::{
    generate_speculative, generate_speculative_monitored, RewindableConstraintMask,
    SpeculativePrompt,
};
use crate::decode::graph::GraphRunner;
use crate::decode::step::generate_step;
use crate::decode::stream::{ConstraintMask, FinishReason, GenerationConfig, GenerationOutput};
use crate::decode::SpeculativeRun;
use crate::models::block_draft::tests::{
    proof_device, tiny_config, tiny_draft_on, tiny_target_on, TINY_VOCAB,
};
use crate::models::DraftSource;
use crate::primitives::decode_cache::DecodeCache;
use crate::primitives::sampler::{shaped_candidates, SamplingParams};

/// Each format's tiny draft at `precision` (`None`: dense) on `device`, labelled.
fn tiny_drafts(
    precision: Option<BlockDraftPrecision>,
    device: &Device,
) -> Vec<(BlockDraftFormat, BlockDraft)> {
    BlockDraftFormat::ALL
        .into_iter()
        .map(|f| (f, tiny_draft_on(tiny_config(f), precision, device)))
        .collect()
}

/// A prompt longer than the tiny drafts' 8-row window.
const PROMPT: [i32; 12] = [3, 9, 4, 11, 3, 9, 4, 11, 3, 9, 4, 11];

/// A constraint forbidding fixed ids, rewindable, recording what it accepted.
struct Forbid {
    allow: Vec<bool>,
    accepted: Vec<i32>,
}

impl Forbid {
    fn new(forbidden: &[usize]) -> Self {
        let mut allow = vec![true; TINY_VOCAB];
        for &f in forbidden {
            allow[f] = false;
        }
        Self {
            allow,
            accepted: Vec::new(),
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
        self.accepted.truncate(checkpoint);
    }
}

fn config(max_new_tokens: usize, seed: u64) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams::default(),
        seed: Some(seed),
        stop_tokens: Vec::new(),
    }
}

fn penalized(max_new_tokens: usize) -> GenerationConfig {
    let mut c = config(max_new_tokens, 7);
    c.sampling.repetition_penalty = 1.3;
    c.sampling.repetition_context = 16;
    c
}

/// The plain token-at-a-time loop on the bare target — the reference.
fn plain(
    model: &Qwen35Model,
    prompt: &[i32],
    config: &GenerationConfig,
    constraint: Option<&mut Forbid>,
) -> GenerationOutput {
    generate_step(
        model,
        prompt,
        config,
        &CancelFlag::new(),
        &mut |_| {},
        constraint.map(|c| c as &mut dyn ConstraintMask),
    )
    .unwrap()
    .0
}

/// The engine over the tapped target (through the graph runner, which replays its verify steps
/// on CUDA and passes through elsewhere) with `proposer`, from a request-sized cache prefilled
/// here and returned.
fn run_block(
    target: &TappedQwen35<'_>,
    proposer: &mut BlockDraftProposer<'_>,
    prompt: &[i32],
    config: &GenerationConfig,
    depth: usize,
    constraint: Option<&mut Forbid>,
) -> (SpeculativeRun, Qwen35Cache) {
    run_block_tree(
        target,
        proposer,
        prompt,
        config,
        depth,
        0,
        constraint,
        &CancelFlag::new(),
    )
}

/// [`run_block`] under a tree request of `tree_rows` candidate rows (`0`: linear), with a cancel
/// flag; the cache holds the tree's extra rows (SW-7198).
#[allow(clippy::too_many_arguments)]
fn run_block_tree<P: Proposer + ?Sized>(
    target: &TappedQwen35<'_>,
    proposer: &mut P,
    prompt: &[i32],
    config: &GenerationConfig,
    depth: usize,
    tree_rows: usize,
    constraint: Option<&mut Forbid>,
    cancel: &CancelFlag,
) -> (SpeculativeRun, Qwen35Cache) {
    let stepper = GraphRunner::new(target);
    let extra = crate::decode::tree_extra_positions(
        depth,
        core_llm::tree_workspace_rows(ProposerKind::BlockDraft, tree_rows),
    );
    let mut cache = stepper
        .new_cache_for(prompt.len() + config.max_new_tokens + extra, depth)
        .unwrap();
    let out = stepper
        .forward_step(
            &mut cache,
            StepRequest::last(prompt).with_hidden(true).as_prefill(),
        )
        .unwrap();
    let run = generate_speculative_monitored(
        &stepper,
        proposer,
        SpeculativePrompt::Prefilled {
            cache: &mut cache,
            logits: out.logits,
            hidden: out.hidden,
            history: prompt,
            position_delta: 0,
            warm_proposer: true,
        },
        config,
        depth,
        cancel,
        &mut |_| {},
        constraint.map(|c| c as &mut dyn RewindableConstraintMask),
        None,
        None,
        core_llm::Speculative::Off,
        tree_rows,
    )
    .unwrap();
    (run, cache)
}

/// AC1 (greedy): for each format, at every draft precision, depth and greedy configuration —
/// plain, penalized, constrained — the block draft commits exactly the plain loop's tokens (the
/// constraint sees the same accepted sequence), proposes, accepts and rejects, rolls the hybrid
/// back through its checkpoint ring without a replay, and names the proposer; on CUDA its verify
/// steps replay as captured graphs.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn greedy_block_draft_is_the_plain_loop() {
    let (_guard, device) = proof_device();
    let model = tiny_target_on(&device, true);
    let mut totals: HashMap<&str, (u64, u64, u64)> = HashMap::new();
    for precision in [
        None,
        Some(BlockDraftPrecision::Q8),
        Some(BlockDraftPrecision::Q4),
    ] {
        for (format, draft) in tiny_drafts(precision, &device) {
            let target = TappedQwen35::for_draft(&model, &draft);
            for depth in [1usize, 3, 5] {
                for (name, cfg, constrained) in [
                    ("greedy", config(24, 7), false),
                    ("penalized", penalized(24), false),
                    ("constrained", config(24, 7), true),
                ] {
                    let label = format!("{format:?} {precision:?} depth {depth} {name}");
                    let (mut a, mut b) = (Forbid::new(&[1, 9]), Forbid::new(&[1, 9]));
                    let expected = plain(&model, &PROMPT, &cfg, constrained.then_some(&mut a));
                    let mut proposer = BlockDraftProposer::new(&draft, &model);
                    let (run, _) = run_block(
                        &target,
                        &mut proposer,
                        &PROMPT,
                        &cfg,
                        depth,
                        constrained.then_some(&mut b),
                    );
                    assert_eq!(run.output.tokens, expected.tokens, "{label}");
                    assert_eq!(b.accepted, a.accepted, "{label}: constraint");
                    let r = &run.record;
                    assert_eq!(r.proposer, ProposerKind::BlockDraft, "{label}");
                    assert_eq!(r.path.label(), "block_draft", "{label}");
                    assert_eq!(r.drafts, depth as u32, "{label}");
                    assert_eq!(r.replay_forwards, 0, "{label}: the ring never replays");
                    assert_eq!(
                        run.output.tokens.len() as u64,
                        1 + r.verify_steps + r.accepted_tokens,
                        "{label}: one committed token per pass plus the accepted drafts"
                    );
                    assert!(proposer.draft_forwards > 0, "{label}");
                    assert!(proposer
                        .candidates()
                        .iter()
                        .all(|row| row.len() == 4 && row.windows(2).all(|w| w[0] < w[1])));
                    if device.is_cuda() && name == "greedy" {
                        assert!(
                            r.cuda_graphs.replayed > 0,
                            "{label}: the verify steps replay ({:?})",
                            r.cuda_graphs
                        );
                    }
                    let t = totals.entry(format.label()).or_default();
                    t.0 += r.proposed_tokens;
                    t.1 += r.accepted_tokens;
                    t.2 += r.direct_rollbacks;
                }
            }
        }
    }
    assert_eq!(totals.len(), 2);
    for (format, (proposed, accepted, rejected)) in totals {
        assert!(
            proposed > 0 && accepted > 0 && rejected > 0,
            "{format}: {proposed} {accepted} {rejected}"
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::greedy_block_draft_is_the_plain_loop");
}

/// SW-7197 (the DSpine greedy defect): a greedy block draft drafts exactly the same with the
/// CUDA-graph runner on as with it off — the same tokens, and the same target passes, proposed and
/// accepted counts (the draft runs eager either way, so a difference means a replay wrote memory
/// the draft reads). The 27B defect: the cold request captured its verify step while the target's
/// quantized matmuls read the shared, grows-only Q8_1 scratch workspace; a later eager call (the
/// draft's first wide commit) grew it and freed the slice the graph kept, and every replay then
/// wrote its activations over whatever the allocator put there next. The fixture follows that
/// order: a Q8_0 target (its verify graph runs the quantized kernels), a Q8_0 draft that taps all
/// four target layers (its input projection reads 4 × 64 columns, twice any target projection),
/// a two-token prompt (the warm-up projects two rows) and depth 5, so the first commits after the
/// capture run the request's widest quantized matmul. Graphs on runs first. The direct proof of the
/// mechanism is `candle-gen`'s `quantized_workspace_graph`; this one holds the product path. A
/// `cuda` build only: without one there is no graph runner and the two runs are the same run.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_greedy_block_draft_drafts_the_same_with_cuda_graphs_on_and_off() {
    let (_guard, device) = proof_device();
    let model = crate::models::block_draft::tests::tiny_target_quantized_on(
        &device,
        true,
        Some(crate::primitives::projection::QuantSpec::q8()),
    );
    let prompt = [3, 9];
    let cfg = config(40, 7);
    for format in BlockDraftFormat::ALL {
        let mut draft_cfg = tiny_config(format);
        draft_cfg.target_layer_ids = vec![0, 1, 2, 3];
        let draft = tiny_draft_on(draft_cfg, Some(BlockDraftPrecision::Q8), &device);
        let target = TappedQwen35::for_draft(&model, &draft);
        let run = |graphs: bool| {
            let _scope = crate::decode::cuda_graphs_scope(Some(graphs));
            let mut proposer = BlockDraftProposer::new(&draft, &model);
            run_block(&target, &mut proposer, &prompt, &cfg, 5, None).0
        };
        let on = run(true);
        let off = run(false);
        let label = format!("{format:?}");
        assert_eq!(on.output.tokens, off.output.tokens, "{label}: the tokens");
        let counts = |r: &SpeculativeRun| {
            (
                r.record.verify_steps,
                r.record.proposed_tokens,
                r.record.accepted_tokens,
            )
        };
        assert_eq!(
            counts(&on),
            counts(&off),
            "{label}: (target passes, proposed, accepted) with graphs on against off"
        );
        assert!(on.record.proposed_tokens > 0, "{label}: the draft proposes");
        assert_eq!(off.record.cuda_graphs.replayed, 0, "{label}: graphs off");
        assert!(
            on.record.cuda_graphs.replayed > 0,
            "{label}: the verify steps replay ({:?})",
            on.record.cuda_graphs
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::a_greedy_block_draft_drafts_the_same_with_cuda_graphs_on_and_off");
}

/// A non-causal draft runs its trained block whatever the depth, shortened only at the token
/// budget's end; a causal one runs the anchor and the rows it proposes.
#[test]
fn a_non_causal_draft_runs_its_trained_block_and_a_causal_one_its_drafts() {
    assert_eq!(block_len(false, 8, 4, 100), 8);
    assert_eq!(block_len(false, 8, 1, 100), 8);
    assert_eq!(block_len(false, 8, 7, 7), 8);
    assert_eq!(block_len(false, 8, 2, 2), 3);
    assert_eq!(block_len(false, 8, 4, 5), 6);
    assert_eq!(block_len(true, 8, 4, 100), 5);
    assert_eq!(block_len(true, 8, 1, 1), 2);
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The tiny target's tapped rows of `ids` for `draft`, from a fresh cache.
fn taps(model: &Qwen35Model, draft: &BlockDraft, ids: &[i32]) -> Tensor {
    let mut cache = StepModel::new_cache(model);
    model
        .forward_tapped(
            &input_ids(ids, model.device()).unwrap(),
            &mut cache,
            0,
            &draft.config().target_layer_ids,
            None,
            crate::decode::step::LogitsScope::Last,
        )
        .unwrap()
        .1
}

/// E2 (block conditioning) and the disk load: a non-causal DFlash2 draft's first rows, scored
/// alone, are exactly those rows of its whole-block proposal, and change when the block is
/// shorter (so the proposer runs the trained block); a causal DSpine draft's first rows are the
/// same in a shorter block. A draft loaded group by group from a BF16 checkpoint on disk proposes
/// exactly what the same weights built in memory propose.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_non_causal_draft_scores_its_rows_over_the_whole_block() {
    let (_guard, device) = proof_device();
    let model = tiny_target_on(&device, false);
    let tol = if device.is_cpu() { 1e-3 } else { 5e-2 };
    for format in BlockDraftFormat::ALL {
        let cfg = tiny_config(format);
        let stored = crate::models::block_draft::tests::tiny_tensors(&cfg);
        let dir = tempfile::tempdir().unwrap();
        let bf16: HashMap<String, Tensor> = stored
            .iter()
            .map(|(k, t)| {
                let t = match t.dtype() {
                    DType::F32 => t.to_dtype(DType::BF16).unwrap(),
                    _ => t.clone(),
                };
                (k.clone(), t)
            })
            .collect();
        candle_core::safetensors::save(&bf16, dir.path().join("model.safetensors")).unwrap();
        let q8 = BlockDraftPrecision::Q8;
        let dtype = crate::device::compute_dtype(&device);
        let from_disk = BlockDraft::load(dir.path(), cfg.clone(), q8, &device, dtype).unwrap();
        let in_memory = BlockDraft::from_source(
            &DraftSource::Map(bf16),
            cfg.clone(),
            Some(q8),
            &device,
            dtype,
        )
        .unwrap();
        let draft = tiny_draft_on(cfg.clone(), None, &device);
        let anchor = 5;
        let embed = |len: usize| {
            let mut block = vec![anchor];
            block.resize(len, cfg.mask_token_id as i32);
            model
                .embed_input_ids(&input_ids(&block, &device).unwrap())
                .unwrap()
        };
        let propose = |draft: &BlockDraft, len: usize, n: usize| {
            let mut context = draft.new_context(0);
            draft
                .append_context(&mut context, &taps(&model, draft, &PROMPT))
                .unwrap();
            let (cand, table) = draft
                .propose(&context, &embed(len), n, anchor, &model)
                .unwrap();
            (cand.concat(), table)
        };
        let full = cfg.block_size;
        assert_eq!(
            propose(&from_disk, full, full - 1),
            propose(&in_memory, full, full - 1),
            "{format:?}: the disk load"
        );
        let (whole_cand, whole_table) = propose(&draft, full, full - 1);
        let k = whole_cand.len() / (full - 1);
        let (first_cand, first_table) = (&whole_cand[..2 * k], &whole_table[..2 * k * k]);
        let (short_cand, short_table) = propose(&draft, 3, 2);
        let diff = worst(&short_table, first_table);
        if draft.causal() {
            assert_eq!(
                short_cand, first_cand,
                "{format:?}: causal rows ignore later rows"
            );
            assert!(
                diff <= tol,
                "{format:?}: a shorter causal block moved {diff}"
            );
        } else {
            let (cand, table) = propose(&draft, full, 2);
            assert_eq!(cand, first_cand, "{format:?}: candidates of the first rows");
            let same = worst(&table, first_table);
            assert!(
                same <= tol,
                "{format:?}: the first rows' scores moved {same}"
            );
            assert!(
                short_cand != first_cand || diff > tol,
                "{format:?}: a non-causal draft's rows see the whole block"
            );
        }
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::a_non_causal_draft_scores_its_rows_over_the_whole_block");
}

/// AC1 / E2 (rollback state): after a run with partial and full acceptances, the hybrid cache
/// holds exactly the committed sequence — its next-token logits are a fresh prefill's — and the
/// draft's context is the context a fresh draft builds from the committed rows: never a rejected
/// draft's row, and never more than the window's 7 rows (E5).
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_rolled_back_state_is_the_committed_sequence() {
    let (_guard, device) = proof_device();
    let model = tiny_target_on(&device, false);
    let cfg = config(20, 7);
    let tol = if device.is_cpu() { 1e-3 } else { 1.5e-1 };
    for (format, draft) in tiny_drafts(Some(BlockDraftPrecision::Q8), &device) {
        let target = TappedQwen35::for_draft(&model, &draft);
        for depth in [2usize, 5] {
            let label = format!("{format:?} depth {depth}");
            let mut proposer = BlockDraftProposer::new(&draft, &model);
            let (run, mut cache) = run_block(&target, &mut proposer, &PROMPT, &cfg, depth, None);
            assert_eq!(run.output.finish_reason, FinishReason::MaxTokens);
            assert!(run.stats.direct_rollbacks > 0, "{label}: a rejection");
            let mut committed = PROMPT.to_vec();
            committed.extend(&run.output.tokens);
            let fed = cache.offset() as usize;
            assert_eq!(
                fed,
                committed.len() - 1,
                "{label}: all but the last token fed"
            );

            let last = *committed.last().unwrap();
            let next = model
                .forward_step(&mut cache, StepRequest::last(&[last]))
                .unwrap()
                .logits;
            let mut fresh = StepModel::new_cache(&model);
            let want = model
                .forward_step(&mut fresh, StepRequest::last(&committed))
                .unwrap()
                .logits;
            let host = |t: &Tensor| t.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
            let d = worst(
                &host(&next).to_vec1::<f32>().unwrap(),
                &host(&want).to_vec1::<f32>().unwrap(),
            );
            assert!(d <= tol, "{label}: target logits moved {d}");

            let context = proposer.context().unwrap();
            assert_eq!(context.next_position() as usize, fed, "{label}");
            assert_eq!(context.rows(), 7, "{label}: the window's rows");
            let mut rebuilt = draft.new_context(0);
            draft
                .append_context(&mut rebuilt, &taps(&model, &draft, &committed[..fed]))
                .unwrap();
            assert_eq!(rebuilt.next_position(), context.next_position());
            for (i, (got, want)) in context.tensors().zip(rebuilt.tensors()).enumerate() {
                let d = worst(
                    &host(got).to_vec1::<f32>().unwrap(),
                    &host(want).to_vec1::<f32>().unwrap(),
                );
                assert!(d <= tol, "{label}: context tensor {i} moved {d}");
            }
        }
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::the_rolled_back_state_is_the_committed_sequence");
}

/// The largest absolute difference of `a` and `b` over the larger of 1 and `b`'s largest
/// magnitude, after checking both are finite.
fn relative(what: &str, a: &Tensor, b: &Tensor) -> f32 {
    let host = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let (a, b) = (host(a), host(b));
    assert!(
        a.iter().chain(&b).all(|x| x.is_finite()),
        "{what}: a non-finite element"
    );
    let scale = b.iter().fold(1f32, |m, x| m.max(x.abs()));
    worst(&a, &b) / scale
}

/// SW-7197 (the DSpine greedy defect, the cold-prompt half): a real request's first proposal reads
/// a context built from the prompt's rows in one cold prefill — the target's tapped rows of a
/// many-token prefill step, appended to the draft in one many-row forward — while every later row
/// comes from a verify step of at most `K + 1` rows. For each format at each precision, with a
/// window wide enough to keep a 24-token prompt (so the warm-up runs the draft's projections over
/// more than 16 rows): the cold prefill's tapped rows are the rows verify-sized steps tap; the
/// context appended at once — its first row scaled to an attention-sink magnitude — is the context
/// appended in verify-sized chunks, every element finite; and the proposals read from the two
/// contexts agree.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_cold_prompt_context_is_the_context_verify_steps_build() {
    let (_guard, device) = proof_device();
    let model = tiny_target_on(&device, false);
    let tol = if device.is_cpu() { 1e-3 } else { 1e-1 };
    let prompt: Vec<i32> = (0..24)
        .map(|i| (i * 7 + 3) % (TINY_VOCAB as i32 - 1))
        .collect();
    let chunks = [4usize, 5, 1, 3, 5, 2, 4];
    assert_eq!(chunks.iter().sum::<usize>(), prompt.len());
    for precision in [
        None,
        Some(BlockDraftPrecision::Q8),
        Some(BlockDraftPrecision::Q4),
    ] {
        for format in BlockDraftFormat::ALL {
            let label = format!("{format:?} {precision:?}");
            let mut cfg = tiny_config(format);
            cfg.sliding_window = Some(64);
            let draft = tiny_draft_on(cfg.clone(), precision, &device);
            let layers = cfg.target_layer_ids.clone();
            let keep = cfg.context_rows();

            // The target: one cold prefill against a prefill of the first chunk, then verify-sized
            // steps (the engine's cache, device positions on CUDA).
            let mut cold_cache = model.new_cache_for(prompt.len() + 8, 5).unwrap();
            let cold = model
                .forward_step_tapped(
                    &mut cold_cache,
                    StepRequest::last(&prompt).with_hidden(true).as_prefill(),
                    Some((&layers, keep)),
                )
                .unwrap()
                .hidden
                .unwrap();
            let mut cache = model.new_cache_for(prompt.len() + 8, 5).unwrap();
            let mut rows = Vec::new();
            let mut at = 0;
            for (i, &len) in chunks.iter().enumerate() {
                let ids = &prompt[at..at + len];
                let request = if i == 0 {
                    StepRequest::last(ids).as_prefill()
                } else {
                    StepRequest::all(ids)
                };
                rows.push(
                    model
                        .forward_step_tapped(
                            &mut cache,
                            request.with_hidden(true),
                            Some((&layers, keep)),
                        )
                        .unwrap()
                        .hidden
                        .unwrap(),
                );
                at += len;
            }
            let stepped = Tensor::cat(&rows, 1).unwrap();
            assert_eq!(cold.dims(), stepped.dims(), "{label}");
            let d = relative(&label, &cold, &stepped);
            assert!(
                d <= tol,
                "{label}: the cold prefill's tapped rows moved {d}"
            );

            // The draft: the same rows, the first scaled to an attention-sink magnitude, appended
            // at once and in verify-sized chunks.
            let mut scale = vec![1f32; prompt.len()];
            scale[0] = 512.0;
            let scale = Tensor::from_vec(scale, (1, prompt.len(), 1), &device)
                .and_then(|s| s.to_dtype(cold.dtype()))
                .unwrap();
            let taps = cold.broadcast_mul(&scale).unwrap();
            let mut at_once = draft.new_context(0);
            draft.append_context(&mut at_once, &taps).unwrap();
            let mut chunked = draft.new_context(0);
            let mut at = 0;
            for &len in &chunks {
                draft
                    .append_context(&mut chunked, &taps.narrow(1, at, len).unwrap())
                    .unwrap();
                at += len;
            }
            assert_eq!(
                at_once.rows(),
                prompt.len(),
                "{label}: the window keeps the prompt"
            );
            assert_eq!(at_once.next_position(), chunked.next_position(), "{label}");
            for (i, (a, b)) in at_once.tensors().zip(chunked.tensors()).enumerate() {
                let d = relative(&format!("{label}: context tensor {i}"), a, b);
                assert!(d <= tol, "{label}: context tensor {i} moved {d}");
            }
            let anchor = 5;
            let depth = 3;
            let mut block = vec![anchor];
            block.resize(
                block_len(draft.causal(), cfg.block_size, depth, 100),
                cfg.mask_token_id as i32,
            );
            let embeds = model
                .embed_input_ids(&input_ids(&block, &device).unwrap())
                .unwrap();
            let (cand_a, table_a) = draft
                .propose(&at_once, &embeds, depth, anchor, &model)
                .unwrap();
            let (cand_b, table_b) = draft
                .propose(&chunked, &embeds, depth, anchor, &model)
                .unwrap();
            assert!(
                table_a.iter().chain(&table_b).all(|x| x.is_finite()),
                "{label}: a non-finite score"
            );
            let scale = table_b.iter().fold(1f32, |m, x| m.max(x.abs()));
            let d = worst(&table_a, &table_b) / scale;
            assert!(d <= tol, "{label}: the proposal's scores moved {d}");
            if device.is_cpu() {
                assert_eq!(cand_a, cand_b, "{label}: the candidates");
            }
        }
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::a_cold_prompt_context_is_the_context_verify_steps_build");
}

/// The exact target distribution of a token after `history`, under the request's knobs and mask.
fn target_dist(
    model: &Qwen35Model,
    history: &[i32],
    params: &SamplingParams,
    mask: Option<&[bool]>,
) -> Vec<(i32, f64)> {
    let mut cache = StepModel::new_cache(model);
    let logits = model
        .forward_step(&mut cache, StepRequest::last(history))
        .unwrap()
        .logits;
    let dist = shaped_candidates(&logits, history, params, mask).unwrap();
    let total: f64 = dist.iter().map(|&(_, w)| f64::from(w)).sum();
    dist.into_iter()
        .filter(|&(_, w)| w > 0.0)
        .map(|(t, w)| (t, f64::from(w) / total))
        .collect()
}

/// AC1 (sampled): for each format, the block draft's committed tokens follow the target's
/// distribution exactly — the joint law of the first three tokens over seeded runs, of which the
/// second and third come from one two-draft verify step, matches the target's own chain rule by a
/// chi-square test (cells expected fewer than 5 times pooled), with the constraint mask and
/// without it. The target's head is random and the draft scores 32 candidates per row, so its
/// drafts overlap the target's top-4 often enough to be accepted.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn sampled_block_draft_keeps_the_target_distribution() {
    let (_guard, device) = proof_device();
    let model = tiny_target_on(&device, true);
    for format in BlockDraftFormat::ALL {
        let mut cfg = tiny_config(format);
        cfg.head_topk = 32;
        let draft = tiny_draft_on(cfg, Some(BlockDraftPrecision::Q8), &device);
        let target = TappedQwen35::for_draft(&model, &draft);
        let prompt = &PROMPT[..6];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 4,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        for forbidden in [&[][..], &[1usize, 9][..]] {
            let mask = Forbid::new(forbidden).allow;
            let masked = (!forbidden.is_empty()).then_some(mask.as_slice());
            let mut expected: HashMap<[i32; 3], f64> = HashMap::new();
            for (x0, p0) in target_dist(&model, prompt, &params, masked) {
                let h0: Vec<i32> = prompt.iter().copied().chain([x0]).collect();
                for (x1, p1) in target_dist(&model, &h0, &params, masked) {
                    let h1: Vec<i32> = h0.iter().copied().chain([x1]).collect();
                    for (x2, p2) in target_dist(&model, &h1, &params, masked) {
                        expected.insert([x0, x1, x2], p0 * p1 * p2);
                    }
                }
            }
            let n = 2000u64;
            let mut counts: HashMap<[i32; 3], u64> = HashMap::new();
            let (mut proposed, mut accepted) = (0u64, 0u64);
            for seed in 0..n {
                let mut cfg = config(4, seed);
                cfg.sampling = params;
                let mut constraint = Forbid::new(forbidden);
                let mut proposer = BlockDraftProposer::new(&draft, &model);
                let (run, _) = run_block(
                    &target,
                    &mut proposer,
                    prompt,
                    &cfg,
                    2,
                    (!forbidden.is_empty()).then_some(&mut constraint),
                );
                let t = &run.output.tokens;
                assert!(t.iter().all(|&x| !forbidden.contains(&(x as usize))));
                *counts.entry([t[0], t[1], t[2]]).or_default() += 1;
                proposed += run.record.proposed_tokens;
                accepted += run.record.accepted_tokens;
            }
            assert!(
                proposed > 0 && accepted > 0,
                "{format:?}: {proposed} proposed, {accepted} accepted"
            );
            for key in counts.keys() {
                assert!(
                    expected.contains_key(key),
                    "{key:?} has target probability 0"
                );
            }
            let (mut chi2, mut cells, mut pooled_e, mut pooled_o) = (0.0f64, 0usize, 0.0, 0.0);
            for (key, p) in &expected {
                let e = p * n as f64;
                let o = *counts.get(key).unwrap_or(&0) as f64;
                if e < 5.0 {
                    pooled_e += e;
                    pooled_o += o;
                } else {
                    chi2 += (o - e).powi(2) / e;
                    cells += 1;
                }
            }
            if pooled_e > 0.0 {
                chi2 += (pooled_o - pooled_e).powi(2) / pooled_e.max(1e-9);
                cells += 1;
            }
            // The 99.9 % critical value of chi-square with `cells - 1` degrees of freedom
            // (Wilson–Hilferty).
            let k = (cells - 1) as f64;
            let critical = k * (1.0 - 2.0 / (9.0 * k) + 3.09 * (2.0 / (9.0 * k)).sqrt()).powi(3);
            assert!(
                chi2 < critical,
                "{format:?} forbidden {forbidden:?}: chi-square {chi2:.1} over {cells} cells \
                 (critical {critical:.1})"
            );
        }
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::sampled_block_draft_keeps_the_target_distribution");
}

// ---- Image prompts (SW-7198). ----

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// The largest elementwise difference of two contexts' tensors, after checking they hold the
/// same rows from the same position.
fn context_moved(got: &BlockDraftContext, want: &BlockDraftContext, label: &str) -> f32 {
    assert_eq!(got.next_position(), want.next_position(), "{label}");
    assert_eq!(got.rows(), want.rows(), "{label}");
    got.tensors()
        .zip(want.tensors())
        .map(|(a, b)| worst(&host_f32(a), &host_f32(b)))
        .fold(0.0, f32::max)
}

/// The image placeholder id of the multimodal fixture (any non-mask id of the tiny vocabulary).
const IMAGE: i32 = 7;

/// A fused image + text prompt over the tiny target, prepared as the provider prepares one
/// (MLX's SW-7195 fixture): `[3, 9, image × 9, 4, 11, 3]` with a `1 × 6 × 6` image grid merged
/// by 2 — nine visual rows on a 3 × 3 M-RoPE grid, so the text after the image sits 6 positions
/// below its sequence index — the visual rows' embeddings shifted as encoder features would be,
/// and one DeepStack feature set fused after layer 0. Longer than the tiny drafts' 8-row window,
/// whose last 7 rows include image rows.
struct ImagePrompt {
    ids: Vec<i32>,
    embeds: Tensor,
    positions: [Vec<i32>; 3],
    visual: Vec<bool>,
    deepstack: Vec<Tensor>,
    delta: i32,
}

impl ImagePrompt {
    fn new(model: &Qwen35Model) -> Self {
        let mut ids = vec![3, 9];
        ids.extend([IMAGE; 9]);
        ids.extend([4, 11, 3]);
        let (t, h, w, delta) = model.mrope_positions(&ids, &[[1, 6, 6]], IMAGE, 2).unwrap();
        assert_eq!(delta, -6, "the 3 × 3 grid spans 3 positions, not 9");
        let visual: Vec<bool> = ids.iter().map(|&id| id == IMAGE).collect();
        let device = model.device();
        let embeds = model
            .embed_input_ids(&input_ids(&ids, device).unwrap())
            .unwrap();
        let (_, s, hidden) = embeds.dims3().unwrap();
        let shift: Vec<f32> = (0..s * hidden)
            .map(|i| {
                let (row, col) = (i / hidden, i % hidden);
                if visual[row] {
                    0.25 + ((row * 7 + col) % 13) as f32 * 0.02
                } else {
                    0.0
                }
            })
            .collect();
        let shift = Tensor::from_vec(shift, (1, s, hidden), device)
            .and_then(|t| t.to_dtype(embeds.dtype()))
            .unwrap();
        let embeds = (embeds + shift).unwrap();
        let feature: Vec<f32> = (0..9 * hidden)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let deepstack = vec![Tensor::from_vec(feature, (9, hidden), device).unwrap()];
        Self {
            ids,
            embeds,
            positions: [t, h, w],
            visual,
            deepstack,
            delta,
        }
    }

    fn positions(&self) -> [&[i32]; 3] {
        [&self.positions[0], &self.positions[1], &self.positions[2]]
    }

    /// The tapped multimodal prefill of the whole prompt into `cache` (left at the prompt's RoPE
    /// shift, as the provider leaves it): the last logits and the tapped rows (the last `keep`).
    fn prefill(
        &self,
        model: &Qwen35Model,
        layers: &[usize],
        keep: Option<usize>,
        cache: &mut Qwen35Cache,
    ) -> (Tensor, Tensor) {
        let out = model
            .prefill_tapped_from_embeds_deepstack(
                &self.embeds,
                self.positions(),
                cache,
                &self.visual,
                &self.deepstack,
                layers,
                keep,
            )
            .unwrap();
        cache.set_rope_delta(self.delta);
        out
    }

    /// The prefix-cache shape: the first `m` rows (the image among them) prefilled into `cache`,
    /// their taps dropped as a restored prefix's are, then the text suffix fed through the tapped
    /// target at its M-RoPE text positions. Returns the last logits and the suffix's taps.
    fn prefill_split(
        &self,
        target: &TappedQwen35<'_>,
        m: usize,
        cache: &mut Qwen35Cache,
    ) -> (Tensor, Tensor) {
        let head = |a: &[i32]| a[..m].to_vec();
        let (t, h, w) = (
            head(&self.positions[0]),
            head(&self.positions[1]),
            head(&self.positions[2]),
        );
        target
            .model
            .prefill_tapped_from_embeds_deepstack(
                &self.embeds.narrow(1, 0, m).unwrap(),
                [&t, &h, &w],
                cache,
                &self.visual[..m],
                &self.deepstack,
                &target.layers,
                target.keep,
            )
            .unwrap();
        cache.set_rope_delta(self.delta);
        assert!(self.visual[m..].iter().all(|v| !v), "the suffix is text");
        assert_eq!(
            self.positions[0][m],
            m as i32 + self.delta,
            "the suffix's text positions"
        );
        let out = target
            .forward_step(cache, StepRequest::last(&self.ids[m..]).with_hidden(true))
            .unwrap();
        (out.logits, out.hidden.unwrap())
    }
}

/// The plain token-at-a-time greedy loop over the fused prompt — the multimodal prefill, then one
/// token per step at its M-RoPE text position — the reference an image request is held to.
fn plain_image(
    model: &Qwen35Model,
    prompt: &ImagePrompt,
    max_new_tokens: usize,
    allow: Option<&[bool]>,
) -> Vec<i32> {
    let mut cache = model
        .new_cache_for(prompt.ids.len() + max_new_tokens, 1)
        .unwrap();
    let mut logits = model
        .decode_logits_from_embeds_deepstack(
            &prompt.embeds,
            prompt.positions(),
            &mut cache,
            &prompt.visual,
            &prompt.deepstack,
        )
        .unwrap();
    cache.set_rope_delta(prompt.delta);
    let mut out = Vec::new();
    loop {
        let row = host_f32(&logits);
        let token = row
            .iter()
            .enumerate()
            .filter(|(i, _)| allow.is_none_or(|a| a[*i]))
            .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                if v > best.1 {
                    (i, v)
                } else {
                    best
                }
            })
            .0 as i32;
        out.push(token);
        if out.len() == max_new_tokens {
            return out;
        }
        logits = model
            .forward_step(&mut cache, StepRequest::last(&[token]))
            .unwrap()
            .logits;
    }
}

/// The engine over the tapped target (through the graph runner) from an already-prefilled
/// multimodal cache at the prompt's RoPE shift.
#[allow(clippy::too_many_arguments)]
fn run_image_block(
    target: &TappedQwen35<'_>,
    proposer: &mut BlockDraftProposer<'_>,
    prompt: &ImagePrompt,
    cache: &mut Qwen35Cache,
    logits: Tensor,
    taps: Tensor,
    config: &GenerationConfig,
    depth: usize,
    constraint: Option<&mut Forbid>,
) -> SpeculativeRun {
    proposer.set_rope_delta(prompt.delta);
    let stepper = GraphRunner::new(target);
    generate_speculative(
        &stepper,
        proposer,
        SpeculativePrompt::Prefilled {
            cache,
            logits,
            hidden: Some(taps),
            history: &prompt.ids,
            position_delta: prompt.delta,
            warm_proposer: true,
        },
        config,
        depth,
        &CancelFlag::new(),
        &mut |_| {},
        constraint.map(|c| c as &mut dyn RewindableConstraintMask),
    )
    .unwrap()
}

/// The context a fresh draft builds from the target's own rows for `committed[..fed]` — the fused
/// prompt under its M-RoPE positions, then the continuation at its text positions — fed from
/// sequence position `from` on (`0`: cold; the prefix-cache boundary: a resumed request).
fn rebuilt_image_context(
    model: &Qwen35Model,
    draft: &BlockDraft,
    prompt: &ImagePrompt,
    committed: &[i32],
    fed: usize,
    from: usize,
) -> BlockDraftContext {
    let layers = &draft.config().target_layer_ids;
    let p = prompt.ids.len();
    let mut cache = StepModel::new_cache(model);
    let (_, taps) = prompt.prefill(model, layers, None, &mut cache);
    let mut all = vec![taps];
    if fed > p {
        let (_, rows) = model
            .forward_tapped(
                &input_ids(&committed[p..fed], model.device()).unwrap(),
                &mut cache,
                p as i32 + prompt.delta,
                layers,
                None,
                crate::decode::step::LogitsScope::Last,
            )
            .unwrap();
        all.push(rows);
    }
    let taps = Tensor::cat(&all, 1).unwrap();
    let n = taps.dim(1).unwrap();
    let mut context = draft.new_context(from as i32);
    draft
        .append_context(&mut context, &taps.narrow(1, from, n - from).unwrap())
        .unwrap();
    context
}

/// SW-7198 AC1 (image): an image + text request runs each draft format — DSpine and DFlash2,
/// dense, 8-bit and 4-bit, at several depths, greedy and constrained — and commits exactly the
/// plain loop's tokens over the same fused prompt (the constraint sees the same accepted
/// sequence): every committed token is one the target verified, and the hybrid rolls back without
/// a replay; on CUDA the verify steps replay as captured graphs.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn greedy_image_block_draft_is_the_plain_loop() {
    let (_guard, device) = proof_device();
    let model = tiny_target_on(&device, true);
    let prompt = ImagePrompt::new(&model);
    let mut totals: HashMap<&str, (u64, u64)> = HashMap::new();
    for precision in [
        None,
        Some(BlockDraftPrecision::Q8),
        Some(BlockDraftPrecision::Q4),
    ] {
        for (format, draft) in tiny_drafts(precision, &device) {
            let target = TappedQwen35::for_draft(&model, &draft);
            for depth in [1usize, 3, 5] {
                for constrained in [false, true] {
                    let label =
                        format!("{format:?} {precision:?} depth {depth} constrained {constrained}");
                    let mut constraint = Forbid::new(&[1, 9]);
                    let allow = constraint.allow.clone();
                    let expected =
                        plain_image(&model, &prompt, 24, constrained.then_some(allow.as_slice()));
                    let mut cache = model.new_cache_for(prompt.ids.len() + 24, depth).unwrap();
                    let (logits, taps) =
                        prompt.prefill(&model, &target.layers, target.keep, &mut cache);
                    assert_eq!(
                        taps.dim(1).unwrap(),
                        7,
                        "{label}: the window's rows, image rows among them"
                    );
                    let mut proposer = BlockDraftProposer::new(&draft, &model);
                    let run = run_image_block(
                        &target,
                        &mut proposer,
                        &prompt,
                        &mut cache,
                        logits,
                        taps,
                        &config(24, 7),
                        depth,
                        constrained.then_some(&mut constraint),
                    );
                    let first_difference = run
                        .output
                        .tokens
                        .iter()
                        .zip(&expected)
                        .position(|(x, y)| x != y);
                    assert_eq!(first_difference, None, "{label}: first differing position");
                    assert_eq!(run.output.tokens, expected, "{label}");
                    if constrained {
                        assert_eq!(constraint.accepted, expected, "{label}: constraint");
                    }
                    let r = &run.record;
                    assert_eq!(r.proposer, ProposerKind::BlockDraft, "{label}");
                    assert_eq!(r.replay_forwards, 0, "{label}: the ring never replays");
                    assert_eq!(
                        run.output.tokens.len() as u64,
                        1 + r.verify_steps + r.accepted_tokens,
                        "{label}: one committed token per pass plus the accepted drafts"
                    );
                    assert!(proposer.draft_forwards > 0, "{label}");
                    if device.is_cuda() && !constrained {
                        assert!(
                            r.cuda_graphs.replayed > 0,
                            "{label}: the verify steps replay ({:?})",
                            r.cuda_graphs
                        );
                    }
                    let t = totals.entry(format.label()).or_default();
                    t.0 += r.proposed_tokens;
                    t.1 += r.accepted_tokens;
                }
            }
        }
    }
    assert_eq!(totals.len(), 2);
    for (format, (proposed, accepted)) in totals {
        println!("{format}: {proposed} proposed, {accepted} accepted on the image prompt");
        assert!(proposed > 0, "{format}: the draft proposed");
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::greedy_image_block_draft_is_the_plain_loop");
}

/// SW-7198 AC1 (image positions): the draft's context follows the target's positions on an image
/// prompt.
///
/// * Cold: the warm-up reads the target's rows at the image positions — the fused rows under
///   M-RoPE and DeepStack, not the placeholder ids re-embedded — and the context's next position
///   is the prompt's sequence length (the draft's own RoPE runs over sequence indices, as the
///   references' draft cache offset does). After a run, the context is exactly the one a fresh
///   draft builds from the target's rows for the committed sequence, and the target's cache is
///   the committed sequence's.
/// * Prefix-cache shape: with the image inside a restored prefix the target feeds only the text
///   suffix at its M-RoPE text positions; the context starts at the boundary (the uncached
///   suffix), its positions stay sequence indices, the run is still the plain loop's, and its
///   final context is the fresh rebuild from the boundary.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn image_draft_context_follows_the_target_positions() {
    let (_guard, device) = proof_device();
    let model = tiny_target_on(&device, true);
    let prompt = ImagePrompt::new(&model);
    let p = prompt.ids.len();
    let tol = if device.is_cpu() { 1e-3 } else { 1.5e-1 };
    let max_new = 16;
    for (format, draft) in tiny_drafts(Some(BlockDraftPrecision::Q8), &device) {
        let target = TappedQwen35::for_draft(&model, &draft);
        let expected = plain_image(&model, &prompt, max_new, None);

        // The warm-up context: image rows read from the fused prefill.
        let mut cache = StepModel::new_cache(&model);
        let (_, fused) = prompt.prefill(&model, &target.layers, target.keep, &mut cache);
        let mut warmed = BlockDraftProposer::new(&draft, &model);
        warmed.warm(&prompt.ids, Some(&fused)).unwrap();
        let context = warmed.context().unwrap();
        assert_eq!(context.next_position() as usize, p, "{format:?}");
        assert_eq!(
            context.rows(),
            7,
            "{format:?}: image rows count against the window"
        );
        let mut placeholder = draft.new_context(0);
        draft
            .append_context(&mut placeholder, &taps(&model, &draft, &prompt.ids))
            .unwrap();
        let moved = context_moved(context, &placeholder, &format!("{format:?} placeholder"));
        assert!(
            moved > 1e-2,
            "{format:?}: the context reads the fused image rows, not re-embedded placeholders ({moved})"
        );

        // Cold run, then the prefix-cache shape with the boundary after the image.
        let boundary = 11;
        for from in [0usize, boundary] {
            let label = format!("{format:?} from {from}");
            let mut cache = model.new_cache_for(p + max_new, 3).unwrap();
            let (logits, taps) = if from == 0 {
                prompt.prefill(&model, &target.layers, target.keep, &mut cache)
            } else {
                prompt.prefill_split(&target, from, &mut cache)
            };
            if from > 0 {
                let mut resumed = BlockDraftProposer::new(&draft, &model);
                resumed.warm(&prompt.ids, Some(&taps)).unwrap();
                let context = resumed.context().unwrap();
                assert_eq!(context.next_position() as usize, p, "{label}");
                assert_eq!(
                    context.rows(),
                    p - from,
                    "{label}: the uncached suffix only"
                );
            }
            let mut proposer = BlockDraftProposer::new(&draft, &model);
            let run = run_image_block(
                &target,
                &mut proposer,
                &prompt,
                &mut cache,
                logits,
                taps,
                &config(max_new, 7),
                3,
                None,
            );
            assert_eq!(run.output.tokens, expected, "{label}: the plain loop");
            let fed = cache.offset() as usize;
            let mut committed = prompt.ids.clone();
            committed.extend(&run.output.tokens);
            assert_eq!(
                fed,
                committed.len() - 1,
                "{label}: all but the last token fed"
            );

            // Target: the next step on the rolled-back cache is the plain loop's next step.
            let last = *committed.last().unwrap();
            let next = model
                .forward_step(&mut cache, StepRequest::last(&[last]))
                .unwrap()
                .logits;
            let mut fresh = StepModel::new_cache(&model);
            model
                .decode_logits_from_embeds_deepstack(
                    &prompt.embeds,
                    prompt.positions(),
                    &mut fresh,
                    &prompt.visual,
                    &prompt.deepstack,
                )
                .unwrap();
            fresh.set_rope_delta(prompt.delta);
            if fed > p {
                model
                    .forward_step(&mut fresh, StepRequest::last(&committed[p..fed]))
                    .unwrap();
            }
            let want = model
                .forward_step(&mut fresh, StepRequest::last(&[last]))
                .unwrap()
                .logits;
            let d = worst(&host_f32(&next), &host_f32(&want));
            assert!(d <= tol, "{label}: target logits moved {d}");

            // Draft: positions are sequence indices, and the context is the fresh rebuild.
            let context = proposer.context().unwrap();
            assert_eq!(context.next_position() as usize, fed, "{label}");
            let rebuilt = rebuilt_image_context(&model, &draft, &prompt, &committed, fed, from);
            let d = context_moved(context, &rebuilt, &label);
            assert!(d <= tol, "{label}: the context moved {d} from the rebuild");
        }
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::image_draft_context_follows_the_target_positions");
}

// ---- Tree verification (SW-7198). ----

/// The tree report of a block-draft tree run: present, verifying under the bound with no linear
/// fallback, and counting every pass's rows and accepted tokens and the tree passes' own (AC2).
fn tree_report(run: &SpeculativeRun, label: &str) -> core_llm::TreeVerifyReport {
    let r = &run.record;
    assert_eq!(r.tree_fallback, None, "{label}");
    let tree = r.tree_verify.expect("a tree run reports its tree");
    assert_eq!(tree.passes, r.verify_steps, "{label}: every pass counted");
    assert_eq!(tree.accepted_tokens, r.accepted_tokens, "{label}");
    assert!(
        tree.rows_verified >= tree.passes,
        "{label}: anchors counted"
    );
    assert!(
        tree.max_rows_per_pass <= 1 + u64::from(tree.max_rows),
        "{label}: at most the anchor and the bound per pass"
    );
    assert!(tree.tree_passes <= tree.passes, "{label}");
    assert!(tree.tree_rows_verified <= tree.rows_verified, "{label}");
    assert!(tree.tree_accepted_tokens <= tree.accepted_tokens, "{label}");
    if tree.tree_passes > 0 {
        assert!(
            tree.tree_rows_verified > tree.tree_passes,
            "{label}: a tree pass feeds its anchor and at least one candidate"
        );
        assert!(tree.rows_per_tree_pass().is_some() && tree.accepted_per_tree_pass().is_some());
    }
    // Every tree pass runs eager through the graph runner, counted with its reason.
    assert!(
        r.cuda_graphs.eager >= tree.tree_passes,
        "{label}: {}",
        r.cuda_graphs.describe()
    );
    tree
}

/// SW-7198 AC2 (greedy): with tree verification on, for each format at every draft precision
/// (dense, Q8, Q4), each greedy configuration — plain, penalized, constrained — and two depths,
/// the block draft's candidate trees commit exactly the plain loop's tokens (the constraint sees
/// the same accepted sequence); the runs verify branching trees in one forward each, accept and
/// reject candidates, and report rows and accepted tokens per tree pass.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn greedy_block_draft_tree_is_the_plain_loop() {
    let (_guard, device) = proof_device();
    let model = tiny_target_on(&device, true);
    let mut totals: HashMap<&str, (u64, u64, u64)> = HashMap::new();
    for precision in [
        None,
        Some(BlockDraftPrecision::Q8),
        Some(BlockDraftPrecision::Q4),
    ] {
        for (format, draft) in tiny_drafts(precision, &device) {
            let target = TappedQwen35::for_draft(&model, &draft);
            for depth in [2usize, 5] {
                for (name, cfg, constrained) in [
                    ("greedy", config(24, 7), false),
                    ("penalized", penalized(24), false),
                    ("constrained", config(24, 7), true),
                ] {
                    let label = format!("{format:?} {precision:?} depth {depth} {name} tree");
                    let (mut a, mut b) = (Forbid::new(&[1, 9]), Forbid::new(&[1, 9]));
                    let expected = plain(&model, &PROMPT, &cfg, constrained.then_some(&mut a));
                    let mut proposer = BlockDraftProposer::new(&draft, &model);
                    let (run, _) = run_block_tree(
                        &target,
                        &mut proposer,
                        &PROMPT,
                        &cfg,
                        depth,
                        16,
                        constrained.then_some(&mut b),
                        &CancelFlag::new(),
                    );
                    assert_eq!(run.output.tokens, expected.tokens, "{label}");
                    assert_eq!(b.accepted, a.accepted, "{label}: constraint");
                    assert_eq!(run.record.proposer, ProposerKind::BlockDraft, "{label}");
                    assert_eq!(run.record.replay_forwards, 0, "{label}");
                    assert!(
                        proposer.draft_forwards > 0
                            && proposer.draft_forwards <= run.record.verify_steps,
                        "{label}: at most one draft forward per target pass"
                    );
                    let tree = tree_report(&run, &label);
                    println!(
                        "BLOCK_DRAFT_TREE {label}: passes={} tree_passes={} rows_per_pass={:.2} \
                         accepted_per_pass={:.2} rows_per_tree_pass={:.2} \
                         accepted_per_tree_pass={:.2} graphs={}",
                        tree.passes,
                        tree.tree_passes,
                        tree.rows_per_pass().unwrap_or(0.0),
                        tree.accepted_per_pass().unwrap_or(0.0),
                        tree.rows_per_tree_pass().unwrap_or(0.0),
                        tree.accepted_per_tree_pass().unwrap_or(0.0),
                        run.record.cuda_graphs.describe(),
                    );
                    let t = totals.entry(format.label()).or_default();
                    t.0 += tree.tree_passes;
                    t.1 += run.record.proposed_tokens;
                    t.2 += run.record.accepted_tokens;
                }
            }
        }
    }
    assert_eq!(totals.len(), 2);
    for (format, (tree_passes, proposed, accepted)) in totals {
        assert!(tree_passes > 0, "{format}: verified branching trees");
        assert!(
            accepted > 0 && proposed > accepted,
            "{format}: {proposed} proposed, {accepted} accepted"
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::greedy_block_draft_tree_is_the_plain_loop");
}

/// SW-7198 AC2 (state): after greedy and sampled tree runs with accepted and rejected branches,
/// the hybrid cache holds exactly the committed sequence and the draft's context is exactly the
/// context a fresh draft builds from the committed rows — a draft that only ever saw the accepted
/// path: no rejected sibling's tapped row entered it, for either format.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn block_draft_tree_context_is_the_accepted_path() {
    let (_guard, device) = proof_device();
    let model = tiny_target_on(&device, true);
    let tol = if device.is_cpu() { 1e-3 } else { 1.5e-1 };
    let mut sampled = config(20, 11);
    sampled.sampling = SamplingParams {
        temperature: 1.0,
        top_k: 4,
        top_p: 1.0,
        ..SamplingParams::default()
    };
    for (format, draft) in tiny_drafts(Some(BlockDraftPrecision::Q8), &device) {
        let target = TappedQwen35::for_draft(&model, &draft);
        for (mode, cfg) in [("greedy", config(20, 7)), ("sampled", sampled.clone())] {
            for depth in [2usize, 5] {
                let label = format!("{format:?} {mode} depth {depth} tree");
                let mut proposer = BlockDraftProposer::new(&draft, &model);
                let (run, mut cache) = run_block_tree(
                    &target,
                    &mut proposer,
                    &PROMPT,
                    &cfg,
                    depth,
                    16,
                    None,
                    &CancelFlag::new(),
                );
                let tree = tree_report(&run, &label);
                assert!(tree.tree_passes > 0, "{label}: branching trees verified");
                assert!(
                    run.record.proposed_tokens > run.record.accepted_tokens,
                    "{label}: rejected candidates"
                );
                let fed = cache.offset() as usize;
                let mut committed = PROMPT.to_vec();
                committed.extend(&run.output.tokens);
                assert_eq!(fed, committed.len() - 1, "{label}");

                // Target: the next step on the committed cache is a fresh prefill's.
                let last = *committed.last().unwrap();
                let next = model
                    .forward_step(&mut cache, StepRequest::last(&[last]))
                    .unwrap()
                    .logits;
                let mut fresh = StepModel::new_cache(&model);
                let want = model
                    .forward_step(&mut fresh, StepRequest::last(&committed))
                    .unwrap()
                    .logits;
                let d = worst(&host_f32(&next), &host_f32(&want));
                assert!(d <= tol, "{label}: target logits moved {d}");

                // Draft: the context of a draft that only ever saw the accepted path.
                let context = proposer.context().unwrap();
                assert_eq!(context.next_position() as usize, fed, "{label}");
                let mut rebuilt = draft.new_context(0);
                draft
                    .append_context(&mut rebuilt, &taps(&model, &draft, &committed[..fed]))
                    .unwrap();
                let d = context_moved(context, &rebuilt, &label);
                assert!(d <= tol, "{label}: the context moved {d} from the rebuild");
            }
        }
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::block_draft_tree_context_is_the_accepted_path");
}

/// SW-7198 AC2 (sampled): through the real tree verify, each format's candidate trees — one
/// sampled child per node drawn from its own `q`, then the most probable other tokens as point
/// masses — commit tokens that follow the target's distribution exactly: the joint law of the
/// first three tokens over seeded runs (the second and third from one tree pass of depth 2)
/// matches the target's chain rule by a chi-square test, plain, under the constraint mask, and
/// with repetition and presence penalties (each branch row scored against its own ancestor path).
/// The target and the drafts lean on the same few ids ([`tiny_target_biased_on`],
/// [`tiny_draft_damped_on`]), so most committed tokens are accepted drafts and the test is
/// sensitive to the `q` each was drawn from; the constrained arm forbids two of the leaned-on ids.
///
/// [`tiny_target_biased_on`]: crate::models::block_draft::tests::tiny_target_biased_on
/// [`tiny_draft_damped_on`]: crate::models::block_draft::tests::tiny_draft_damped_on
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn sampled_block_draft_tree_keeps_the_target_distribution() {
    let (_guard, device) = proof_device();
    // A target whose draft agrees with it often (both put most mass on the same few ids), so most
    // committed tokens are accepted drafts — whose law depends on the `q` each was drawn from —
    // rather than residual draws.
    let model = crate::models::block_draft::tests::tiny_target_biased_on(&device);
    let plain_params = SamplingParams {
        temperature: 1.0,
        top_k: 6,
        top_p: 1.0,
        ..SamplingParams::default()
    };
    let penalized_params = SamplingParams {
        repetition_penalty: 1.6,
        repetition_context: 16,
        presence_penalty: 0.6,
        ..plain_params
    };
    for format in BlockDraftFormat::ALL {
        let mut cfg = tiny_config(format);
        cfg.head_topk = 32;
        let draft = crate::models::block_draft::tests::tiny_draft_damped_on(
            cfg,
            Some(BlockDraftPrecision::Q8),
            &device,
        );
        let target = TappedQwen35::for_draft(&model, &draft);
        let prompt = &PROMPT[..6];
        for (forbidden, params) in [
            (&[][..], plain_params),
            (&[1usize, 9][..], plain_params),
            (&[][..], penalized_params),
        ] {
            // The constraint matters: without it the target puts real mass on the forbidden ids.
            let unmasked: f64 = target_dist(&model, prompt, &params, None)
                .iter()
                .filter(|&&(t, _)| forbidden.contains(&(t as usize)))
                .map(|&(_, p)| p)
                .sum();
            assert!(
                forbidden.is_empty() || unmasked > 0.1,
                "{format:?}: the forbidden ids carry {unmasked:.3} of the unmasked mass"
            );
            let mask = Forbid::new(forbidden).allow;
            let masked = (!forbidden.is_empty()).then_some(mask.as_slice());
            let mut expected: HashMap<[i32; 3], f64> = HashMap::new();
            for (x0, p0) in target_dist(&model, prompt, &params, masked) {
                let h0: Vec<i32> = prompt.iter().copied().chain([x0]).collect();
                for (x1, p1) in target_dist(&model, &h0, &params, masked) {
                    let h1: Vec<i32> = h0.iter().copied().chain([x1]).collect();
                    for (x2, p2) in target_dist(&model, &h1, &params, masked) {
                        expected.insert([x0, x1, x2], p0 * p1 * p2);
                    }
                }
            }
            let n = 2000u64;
            let mut counts: HashMap<[i32; 3], u64> = HashMap::new();
            let (mut tree_passes, mut proposed, mut accepted) = (0u64, 0u64, 0u64);
            let mut tree_accepted = 0u64;
            for seed in 0..n {
                let mut cfg = config(4, seed);
                cfg.sampling = params;
                let mut constraint = Forbid::new(forbidden);
                let mut proposer = BlockDraftProposer::new(&draft, &model);
                let (run, _) = run_block_tree(
                    &target,
                    &mut proposer,
                    prompt,
                    &cfg,
                    2,
                    16,
                    (!forbidden.is_empty()).then_some(&mut constraint),
                    &CancelFlag::new(),
                );
                let t = &run.output.tokens;
                assert!(t.iter().all(|&x| !forbidden.contains(&(x as usize))));
                *counts.entry([t[0], t[1], t[2]]).or_default() += 1;
                let report = tree_report(&run, format.label());
                tree_passes += report.tree_passes;
                tree_accepted += report.tree_accepted_tokens;
                proposed += run.record.proposed_tokens;
                accepted += run.record.accepted_tokens;
            }
            assert!(tree_passes > 0, "{format:?}: branching trees verified");
            // Most committed tokens are accepted drafts (measured 1.57–1.81 of the 2 a depth-2 pass
            // can accept), so a stored `q` other than the one a draft was drawn from moves the
            // law (a square-rooted `q` measured chi-square 670 against a critical 87).
            let accepted_per_tree_pass = tree_accepted as f64 / tree_passes as f64;
            assert!(
                accepted_per_tree_pass >= 1.0,
                "{format:?} forbidden {forbidden:?}: {accepted_per_tree_pass:.2} accepted per tree pass"
            );
            assert!(
                accepted > 0 && proposed > accepted,
                "{format:?}: {proposed} proposed, {accepted} accepted"
            );
            for key in counts.keys() {
                assert!(
                    expected.contains_key(key),
                    "{key:?} has target probability 0"
                );
            }
            let (mut chi2, mut cells, mut pooled_e, mut pooled_o) = (0.0f64, 0usize, 0.0, 0.0);
            for (key, p) in &expected {
                let e = p * n as f64;
                let o = *counts.get(key).unwrap_or(&0) as f64;
                if e < 5.0 {
                    pooled_e += e;
                    pooled_o += o;
                } else {
                    chi2 += (o - e).powi(2) / e;
                    cells += 1;
                }
            }
            if pooled_e > 0.0 {
                chi2 += (pooled_o - pooled_e).powi(2) / pooled_e.max(1e-9);
                cells += 1;
            }
            // The 99.9 % critical value of chi-square with `cells - 1` degrees of freedom
            // (Wilson–Hilferty).
            let k = (cells - 1) as f64;
            let critical = k * (1.0 - 2.0 / (9.0 * k) + 3.09 * (2.0 / (9.0 * k)).sqrt()).powi(3);
            println!(
                "BLOCK_DRAFT_TREE_SAMPLED {format:?} forbidden {forbidden:?} penalties {}: \
                 chi-square {chi2:.1} over {cells} cells (critical {critical:.1}); {tree_passes} \
                 tree passes, {accepted_per_tree_pass:.2} accepted per tree pass, \
                 {accepted}/{proposed} accepted",
                params.repetition_penalty != 1.0
            );
            assert!(
                chi2 < critical,
                "{format:?} forbidden {forbidden:?}: chi-square {chi2:.1} over {cells} cells \
                 (critical {critical:.1})"
            );
        }
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::sampled_block_draft_tree_keeps_the_target_distribution");
}

/// A block-draft proposer that cancels the request as it hands over its tree, keeping the tree.
struct CancelOnTree<'d, 'c> {
    inner: BlockDraftProposer<'d>,
    cancel: &'c CancelFlag,
    tree: Option<DraftTree>,
}

impl Proposer for CancelOnTree<'_, '_> {
    fn kind(&self) -> ProposerKind {
        ProposerKind::BlockDraft
    }
    fn wants_hidden(&self) -> bool {
        true
    }
    fn warm(&mut self, prompt: &[i32], hidden: Option<&Tensor>) -> Result<()> {
        self.inner.warm(prompt, hidden)
    }
    fn propose(
        &mut self,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        unreachable!("a tree run asks for a tree")
    }
    fn propose_tree(
        &mut self,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Option<DraftTree>> {
        let tree = self.inner.propose_tree(ctx, sampler)?;
        self.cancel.cancel();
        self.tree.clone_from(&tree);
        Ok(tree)
    }
    fn commit(
        &mut self,
        cur: i32,
        accepted: &[i32],
        hidden: Option<&Tensor>,
        position: i32,
    ) -> Result<()> {
        self.inner.commit(cur, accepted, hidden, position)
    }
}

/// The hybrid cache's comparable state: every layer's keys/values and recurrent and conv states.
fn cache_states(cache: &Qwen35Cache) -> Vec<Vec<f32>> {
    cache.prefix_snapshot().unwrap().host_tensors()
}

/// SW-7198 AC2 (abort / cancel): a request cancelled as its block-draft tree is handed over stops
/// before any tree row reaches the target cache, with the draft's context still the warm-up's;
/// and that tree, verified on the prefilled cache — every tree row's taps returned — and aborted,
/// returns the target cache to the step start bit for bit while the draft context, which only a
/// commit writes, is untouched.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn an_aborted_or_cancelled_block_draft_tree_leaves_both_caches_clean() {
    let (_guard, device) = proof_device();
    let model = tiny_target_on(&device, true);
    let depth = 4;
    for (format, draft) in tiny_drafts(Some(BlockDraftPrecision::Q8), &device) {
        let target = TappedQwen35::for_draft(&model, &draft);
        let label = format!("{format:?}");
        let capacity = PROMPT.len() + 8 + crate::decode::tree_extra_positions(depth, 16);
        let prefill = |cache: &mut Qwen35Cache| {
            target
                .forward_step(
                    cache,
                    StepRequest::last(&PROMPT).with_hidden(true).as_prefill(),
                )
                .unwrap()
        };
        let mut prefilled = target.new_cache_for(capacity, depth).unwrap();
        let out = prefill(&mut prefilled);
        let mut warmed = BlockDraftProposer::new(&draft, &model);
        warmed.warm(&PROMPT, out.hidden.as_ref()).unwrap();
        let warm: Vec<Vec<f32>> = warmed.context().unwrap().tensors().map(host_f32).collect();
        let start = cache_states(&prefilled);

        // Cancel while the tree is handed over.
        let cancel = CancelFlag::new();
        let mut proposer = CancelOnTree {
            inner: BlockDraftProposer::new(&draft, &model),
            cancel: &cancel,
            tree: None,
        };
        let (run, cache) = run_block_tree(
            &target,
            &mut proposer,
            &PROMPT,
            &config(8, 7),
            depth,
            16,
            None,
            &cancel,
        );
        assert_eq!(run.output.finish_reason, FinishReason::Cancelled, "{label}");
        assert_eq!(
            run.output.tokens.len(),
            1,
            "{label}: the prefill's token only"
        );
        assert_eq!(cache_states(&cache), start, "{label}: no tree row written");
        let context: Vec<Vec<f32>> = proposer
            .inner
            .context()
            .unwrap()
            .tensors()
            .map(host_f32)
            .collect();
        assert_eq!(context, warm, "{label}: the draft context is the warm-up's");
        let tree = proposer.tree.expect("the proposer built a tree");
        assert!(!tree.is_chain(), "{label}: a branching tree");

        // Verify that tree on the prefilled cache, then abort it.
        let mut cache = target.new_cache_for(capacity, depth).unwrap();
        let out = prefill(&mut cache);
        let logits = host_f32(&out.logits);
        let anchor =
            (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b }) as i32;
        let base = cache.offset();
        let verified = GraphRunner::new(&target)
            .forward_tree(
                &mut cache,
                &TreeVerify {
                    tree: &tree,
                    anchor,
                    want_hidden: true,
                },
            )
            .unwrap();
        assert_eq!(
            verified.step.hidden.as_ref().unwrap().dims(),
            &[1, tree.rows(), draft.config().tap_width()],
            "{label}: every tree row's taps"
        );
        DecodeCache::abort_tree(&mut cache, base).unwrap();
        assert_eq!(cache.offset(), base, "{label}");
        assert_eq!(
            cache_states(&cache),
            start,
            "{label}: the aborted tree left nothing"
        );
        let context: Vec<Vec<f32>> = warmed.context().unwrap().tensors().map(host_f32).collect();
        assert_eq!(context, warm, "{label}: an abort never reaches the draft");
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::an_aborted_or_cancelled_block_draft_tree_leaves_both_caches_clean");
}
