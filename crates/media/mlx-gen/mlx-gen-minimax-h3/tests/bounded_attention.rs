//! sc-18661 — **rung 3 reachability and declaration guards**, on the committed DiT fixture.
//!
//! Runs in CI: no snapshot, no tier directory, no `#[ignore]`. The measured half lives in
//! `tests/bounded_attention_real.rs`; what is asserted here is the half a null measurement does not
//! excuse — that the seam is real, that it reaches the model's own attention call, and that the
//! declaration cannot drift away from the mechanism in either direction.
//!
//! # Declaration is not enforcement is not reachability
//!
//! Three distinct claims, and this epic has shipped a contract that satisfied one while failing the
//! others more than once. They are separated deliberately below:
//!
//! | claim | asserted by |
//! |---|---|
//! | the kernel exists and is correct | `mlx_gen::attention`'s own unit tests |
//! | the plan **reaches** MiniMax-H3's attention call | [`a_bounded_plan_reaches_the_real_dit_attention_call`] |
//! | the plan reaches it through the **whole** block stack | [`a_bounded_plan_threads_the_whole_block_stack`] |
//! | the contract's disposition matches the mechanism | [`the_rung_three_declaration_cannot_drift_from_the_mechanism`] |
//!
//! The reachability tests read `dit::layers::attention_probe`, which records what the shared planner
//! decided at the live shape *inside* the forward. Without it every assertion here would be a false
//! green: "bounded output == unbounded output" is trivially true when the bounding never engaged.

mod common;

use mlx_rs::{Array, Dtype};

use mlx_gen::attention::{AttentionBudget, BoundedAttention};
use mlx_gen::gen_core::{
    LoadShape, LoadSpec, MemoryParameterRanges, MemoryStrategy, MemoryStrategySupport,
    WeightsSource,
};
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::dit::layers::{attention_probe, planned_attention_chunks, DitAttention};
use mlx_gen_minimax_h3::dit::{MiniMaxH3DitConfig, MmRope};

use common::{dit_fixture_config, DIT_FIXTURE};

fn model_weights() -> Weights {
    let mut w = Weights::from_file(DIT_FIXTURE).unwrap();
    for prefix in ["src.", "in.", "out.", "layout."] {
        w.remove_prefix(prefix);
    }
    w
}

fn rope_of(cfg: &MiniMaxH3DitConfig) -> MmRope {
    MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap()
}

/// Peak-relative max-abs difference between two arrays, both read at f32.
///
/// Relative, and specifically **not** a norm, a cosine or a checksum: all three are dominated by the
/// bulk of an already-agreeing tensor and would pass on a chunk boundary that dropped rows entirely.
fn relative_max_abs(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let delta: f32 = a
        .subtract(&b)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item();
    let scale: f32 = a.abs().unwrap().max(None).unwrap().item();
    assert!(scale > 1e-6, "the reference arm is degenerate ({scale})");
    delta / scale
}

/// The **query-row** axis's agreement bound at the fixture geometry.
///
/// `mlx_gen::attention`'s own documented tolerance on Metal: query-row chunking narrows the GEMM's
/// `M` and lands on a different reduced-precision specialization (the sc-2338 parity class), so it is
/// tolerance-equivalent rather than exact. Measured `5.9e-4` here.
///
/// **The head axis is held to `0.0` instead**, not to this. It is exact whenever the budget admits one
/// whole head, which is the configuration these tests drive — see [`mlx_gen::attention::AttentionChunkAxis`]
/// for the condition and for the trap: at half that budget the head axis falls back to query rows
/// inside each chunk and reports the query axis's `5.9e-4` under the head axis's name.
const FIXTURE_TOLERANCE: f32 = 2e-3;

/// **AC5 — the chunked path is reachable in the real attention call.**
///
/// Not "a bounded kernel exists" and not "a contract declares a rung": the fixture's own
/// `DitAttention`, loaded from the committed checkpoint bytes, driven through
/// `forward_bounded`, with the probe showing the split actually happened at the live shape.
///
/// Both axes, because rung 3 has two and a seam wired for one of them would leave the other
/// unreachable. The head axis is pinned to **exactly** zero and the query-row axis to
/// [`FIXTURE_TOLERANCE`] — the numerical contracts genuinely differ, and asserting the same bound on
/// both would throw away the stronger claim.
#[test]
fn a_bounded_plan_reaches_the_real_dit_attention_call() {
    let f = Weights::from_file(DIT_FIXTURE).unwrap();
    let cfg = dit_fixture_config();
    let mut w = model_weights();
    let attn =
        DitAttention::from_weights(&mut w, "transformer_blocks.0.attn", &cfg, Dtype::Float32)
            .unwrap();
    let rope = rope_of(&cfg);
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let x = f.require("in.attn.hidden").unwrap();
    let seq = x.shape()[1];
    let heads = cfg.num_attention_heads;

    attention_probe::reset();
    let unbounded = attn.forward(x, Some((&rope, &tables))).unwrap();
    assert_eq!(
        (
            attention_probe::last_head_chunks(),
            attention_probe::last_query_chunks(),
            attention_probe::last_seq()
        ),
        (1, 1, seq as usize),
        "the default forward must still be one un-chunked fused call over the whole sequence"
    );

    // **Exactly one head's score domain**, `B · Sq²`. That is the budget at which the head axis runs
    // one head per chunk with the query axis left whole — the configuration whose numerical claim is
    // about heads alone — while the same budget still forces the query-row axis to split, so the two
    // arms are a like-for-like comparison rather than two different bounds.
    //
    // Half of it would ALSO make the head arm fall back to query rows inside each head chunk, and the
    // head arm's number would then be the query axis's in disguise. Measured: that mistake reports
    // 5.9e-4 for "the head axis" at this geometry.
    let budget = AttentionBudget::from_score_elements((seq * seq) as u64, true);

    attention_probe::reset();
    let by_heads = attn
        .forward_bounded(x, Some((&rope, &tables)), BoundedAttention::heads(budget))
        .unwrap();
    let head_chunks = attention_probe::last_head_chunks();
    assert_eq!(
        (head_chunks, attention_probe::last_query_chunks()),
        (heads as usize, 1),
        "the head axis must have split into one head per chunk with the query axis whole inside \
         the real attention call"
    );

    attention_probe::reset();
    let by_rows = attn
        .forward_bounded(
            x,
            Some((&rope, &tables)),
            BoundedAttention::query_rows(budget),
        )
        .unwrap();
    assert!(
        attention_probe::last_query_chunks() > 1,
        "the query-row axis did not split inside the real attention call — the equivalence below \
         would be vacuous"
    );
    assert_eq!(
        attention_probe::last_head_chunks(),
        1,
        "the query-row axis must not split heads"
    );

    // Both axes must agree with the unbounded call to within the documented Metal parity class.
    let head_delta = relative_max_abs(&unbounded, &by_heads);
    let row_delta = relative_max_abs(&unbounded, &by_rows);
    eprintln!(
        "[fixture seq {seq} heads {heads}] head axis rel|Δ| {head_delta:.3e}, query axis rel|Δ| \
         {row_delta:.3e} (bound {FIXTURE_TOLERANCE:e})"
    );
    assert_eq!(
        head_delta, 0.0,
        "a one-head-per-chunk split with the query axis whole must reconstruct the unbounded \
         attention EXACTLY; {head_delta:e} means either a wiring error or a silent query-row \
         fallback (check the probe's query count, which this test pins at 1)"
    );
    assert!(
        row_delta < FIXTURE_TOLERANCE,
        "query-row chunking moved the attention output by {row_delta:e} relative — that is a \
         wiring error, not the reduced-precision drift a changed Metal specialization produces"
    );

    // **The tolerance is not vacuous.** A genuinely different attention — the same weights and the
    // same input with the rotary dropped, which `tests/dit_parity.rs` independently shows this
    // fixture is sensitive to — must blow far past the bound. Without this, a comparison that
    // accidentally compared an array with itself would read as agreement.
    //
    // Deliberately not "add a constant to `x`": the affine q/k RMSNorm largely absorbs one, and it
    // measured 8.6e-2 — above the bound but only by 43x, which is a weak floor for a metric whose
    // agreement arm sits at 5.9e-4.
    let no_rope = attn.forward(x, None).unwrap();
    let moved_delta = relative_max_abs(&unbounded, &no_rope);
    eprintln!("[fixture mutation check] rope dropped: rel|Δ| {moved_delta:.3e}");
    assert!(
        moved_delta > 100.0 * FIXTURE_TOLERANCE,
        "mutation check: dropping the rotary moved the metric only {moved_delta:e} — the \
         comparison above cannot distinguish agreement from a degenerate metric"
    );

    // **The probe is not a second implementation.** What it reports must be what the shared planner
    // says, computed from the same public arithmetic the kernel consumes.
    assert_eq!(
        planned_attention_chunks(BoundedAttention::heads(budget), 1, heads, seq),
        (head_chunks, 1)
    );
}

/// The plan must reach **every** block, not just the one a unit test happens to drive.
///
/// A seam threaded into `DitAttention` but dropped at `DitBlock` or `MiniMaxH3Dit` would pass the
/// test above and bound nothing in a render. Driven through both modulation arms — `Cached`, the
/// shipped precompute-and-evict path, and `Temb`, the resident one — because a rung reachable on
/// only one of them is selectable on a request routed to the other.
#[test]
fn a_bounded_plan_threads_the_whole_block_stack() {
    use mlx_gen_minimax_h3::dit::DitBlock;

    let f = Weights::from_file(DIT_FIXTURE).unwrap();
    let cfg = dit_fixture_config();
    let mut w = model_weights();
    let block =
        DitBlock::from_weights(&mut w, "transformer_blocks.0", &cfg, Dtype::Float32).unwrap();
    let rope = rope_of(&cfg);
    let tables = rope
        .tables(f.require("layout.position_ids").unwrap())
        .unwrap();
    let x = f.require("in.block.hidden").unwrap();
    let temb = f.require("in.temb").unwrap();
    let adaln_indices = f
        .require("layout.adaln_indices")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let seq = x.shape()[1];
    let budget = AttentionBudget::from_score_elements((seq * seq) as u64, true);
    let bounded = BoundedAttention::heads(budget);

    let modulation = block.modulation(temb).unwrap();
    for (label, bounded_out, plain) in [
        (
            "cached modulation",
            block
                .forward_bounded(x, &modulation, &adaln_indices, &rope, &tables, bounded)
                .unwrap(),
            block
                .forward(x, &modulation, &adaln_indices, &rope, &tables)
                .unwrap(),
        ),
        (
            "resident temb",
            block
                .forward_with_temb_bounded(x, temb, &adaln_indices, &rope, &tables, bounded)
                .unwrap(),
            block
                .forward_with_temb(x, temb, &adaln_indices, &rope, &tables)
                .unwrap(),
        ),
    ] {
        // The bounded arm is re-run under the probe so the engagement claim is about THIS call.
        attention_probe::reset();
        let _ = match label {
            "cached modulation" => block
                .forward_bounded(x, &modulation, &adaln_indices, &rope, &tables, bounded)
                .unwrap(),
            _ => block
                .forward_with_temb_bounded(x, temb, &adaln_indices, &rope, &tables, bounded)
                .unwrap(),
        };
        assert_eq!(
            attention_probe::last_head_chunks(),
            cfg.num_attention_heads as usize,
            "{label}: the plan did not reach the block's attention call"
        );
        let delta = relative_max_abs(&plain, &bounded_out);
        eprintln!("[fixture block, {label}] head axis rel|Δ| {delta:.3e}");
        assert_eq!(
            delta, 0.0,
            "{label}: a bit-exact attention split must leave a whole block's output unchanged, \
             moved {delta:e} relative"
        );
    }
}

/// **The declaration cannot drift from the mechanism, in either direction.**
///
/// Written against `conformance_errors` — the gate's own predicate — and driven by *mutating* the
/// shipped contract, because an assertion about the shipped value alone is a false green: rung 3 is
/// `StructurallyNotApplicable` today, so "if Implemented then a range is declared" holds vacuously
/// and would keep holding with the rule deleted.
///
/// The three mutations are the three ways this declaration can go wrong:
///
/// 1. `Implemented` with no `attention_chunk_sizes` — a lever with an empty domain (AC3);
/// 2. `StructurallyNotApplicable` with an empty reason — an inapplicability claim justified by
///    nothing, which is the inert-guard defect this epic keeps shipping;
/// 3. `StructurallyNotApplicable` while `lifecycle.attention_chunking` is set — a contract that says
///    both "this rung cannot apply" and "here is its implementation hook".
#[test]
fn the_rung_three_declaration_cannot_drift_from_the_mechanism() {
    // The catalog-conformance spec: a directory that does not exist, so nothing is resolved and the
    // contract states architecture facts only.
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
        .with_load_shape(LoadShape::DeferredMaterialization);
    let shipped = mlx_gen_minimax_h3::memory_strategy::weights_free_contract(&spec).unwrap();
    assert!(
        shipped.conformance_errors().is_empty(),
        "the shipped contract must conform: {:?}",
        shipped.conformance_errors()
    );

    let capability = shipped
        .capability(MemoryStrategy::BoundedAttention)
        .expect("rung 3 is declared");
    let MemoryStrategySupport::StructurallyNotApplicable { reason } = &capability.support else {
        panic!(
            "sc-18661 measured rung 3 inapplicable on this backend; a change of verdict must come \
             with its own measurement and must update this test, got {:?}",
            capability.support
        );
    };
    // The reason must carry the measurement, not a sentiment. `4·B·H·S·D` is the streaming-kernel
    // prediction the whole verdict rests on; a reason that stopped naming it would no longer let a
    // reader check the claim.
    assert!(
        reason.contains("4·B·H·S·D"),
        "the inapplicability reason must name the measured streaming prediction, got: {reason}"
    );
    assert!(
        !shipped.lifecycle.attention_chunking,
        "an inapplicable rung must not declare its implementation hook"
    );

    // (1) Implemented with an empty domain.
    let mut mutated = shipped.clone();
    let slot = mutated
        .strategies
        .iter_mut()
        .find(|c| c.strategy == MemoryStrategy::BoundedAttention)
        .unwrap();
    slot.support = MemoryStrategySupport::Implemented;
    slot.parameters = MemoryParameterRanges::default();
    mutated.lifecycle.attention_chunking = true;
    let errors = mutated.conformance_errors();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("attention chunk-size candidates")),
        "an implemented rung 3 with no declared chunk sizes must be refused, got {errors:?}"
    );

    // (2) Inapplicable for no stated reason.
    let mut mutated = shipped.clone();
    mutated
        .strategies
        .iter_mut()
        .find(|c| c.strategy == MemoryStrategy::BoundedAttention)
        .unwrap()
        .support = MemoryStrategySupport::StructurallyNotApplicable {
        reason: "   ".to_owned(),
    };
    let errors = mutated.conformance_errors();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("StructurallyNotApplicable without a reason")),
        "an unjustified inapplicability claim must be refused, got {errors:?}"
    );

    // (3) Inapplicable while the hook is declared.
    let mut mutated = shipped.clone();
    mutated.lifecycle.attention_chunking = true;
    let errors = mutated.conformance_errors();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("BoundedAttention cannot be StructurallyNotApplicable")),
        "declaring the hook under an inapplicable rung must be refused, got {errors:?}"
    );
}
