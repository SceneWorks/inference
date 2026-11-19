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
use crate::decode::engine::{generate_speculative, RewindableConstraintMask, SpeculativePrompt};
use crate::decode::graph::GraphRunner;
use crate::decode::step::generate_step;
use crate::decode::stream::{ConstraintMask, FinishReason, GenerationConfig, GenerationOutput};
use crate::decode::SpeculativeRun;
use crate::models::block_draft::tests::{
    proof_device, tiny_config, tiny_draft_on, tiny_target_on, TINY_VOCAB,
};
use crate::models::DraftSource;
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
    let stepper = GraphRunner::new(target);
    let mut cache = stepper
        .new_cache_for(prompt.len() + config.max_new_tokens, depth)
        .unwrap();
    let out = stepper
        .forward_step(
            &mut cache,
            StepRequest::last(prompt).with_hidden(true).as_prefill(),
        )
        .unwrap();
    let run = generate_speculative(
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
        &CancelFlag::new(),
        &mut |_| {},
        constraint.map(|c| c as &mut dyn RewindableConstraintMask),
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
