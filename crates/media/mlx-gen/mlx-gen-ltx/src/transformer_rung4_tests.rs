//! **Rung 4 loader identity and windowed/unwindowed bit-identity** (sc-18797, epic 18755 R9).
//!
//! A private submodule of [`crate::transformer`] rather than a `tests/` file, because the
//! observation AC2 demands is of an *internal loader* — it needs `Stream`, `StreamArgs`, `AvBlocks`
//! and `AvBlock`'s fields, none of which are public and none of which should become public to be
//! testable.
//!
//! # Why a synthetic checkpoint, and what it is allowed to prove
//!
//! The real AvDiT trunk is ~22 B parameters, and the resident/streamed A/B has to build the SAME
//! model twice in one process, which no real tier permits. So these build a dimensionally consistent
//! **tiny** AvDiT — every tensor derived from one [`LtxConfig`] — and write it to a real
//! `.safetensors` on disk, so [`LtxBlockStream`] reopens a genuine file through the production path
//! rather than a stub.
//!
//! Stated rather than implied:
//!
//! - This **does** prove loader identity, window arithmetic, the per-window drain, budget replay and
//!   bit-identity of the two block orders. All of those are shape-independent.
//! - This does **not** prove reference parity, peak residency in bytes, or that the rung is worth its
//!   latency. Those belong to `av_dit_parity` (real weights) and the epic's terminal evidence.

use super::*;
use crate::block_stream::{
    block_stream_diagnostics, reset_block_stream_diagnostics, BlockStreamDiagnostics,
    LtxBlockStream,
};
use mlx_rs::ops::array_eq;
use std::collections::HashMap;

const VIDEO_HEADS: i32 = 2;
const VIDEO_HEAD_DIM: i32 = 12;
const AUDIO_HEADS: i32 = 2;
const AUDIO_HEAD_DIM: i32 = 4;
const CHANNELS: i32 = 4;
const FFN_MULT: i32 = 2;
const N_LAYERS: i32 = 4;
/// The adaLN row count an AV block's `scale_shift_table` carries (`v_sst` is `(9, inner)`), which is
/// also the adaLN embedding coefficient the AV path must run at.
const ADALN_ROWS: i32 = 9;

fn tiny_cfg() -> LtxConfig {
    let mut cfg = LtxConfig::video_only_defaults();
    cfg.num_attention_heads = VIDEO_HEADS;
    cfg.attention_head_dim = VIDEO_HEAD_DIM;
    cfg.audio_num_attention_heads = AUDIO_HEADS;
    cfg.audio_attention_head_dim = AUDIO_HEAD_DIM;
    cfg.in_channels = CHANNELS;
    cfg.out_channels = CHANNELS;
    cfg.audio_in_channels = CHANNELS;
    cfg.audio_out_channels = CHANNELS;
    cfg.num_layers = N_LAYERS;
    cfg.cross_attention_dim = VIDEO_HEADS * VIDEO_HEAD_DIM;
    cfg.audio_cross_attention_dim = AUDIO_HEADS * AUDIO_HEAD_DIM;
    // The AV path is the gated one: `AvBlock` reads 9 adaLN rows, so the embedding coefficient must
    // be 9 or `ada_values` reshapes the timestep projection against the wrong row count.
    cfg.apply_gated_attention = true;
    cfg.adaln_embedding_coefficient = ADALN_ROWS;
    cfg.use_keyframes_abs_pos_embedding = false;
    cfg
}

/// Deterministic, non-degenerate values. A constant fill would make the bit-identity assertion pass
/// for the wrong reason: any two block orders agree when every weight is the same number.
fn tensor(shape: &[i32], seed: i32) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i + seed * 17) as f32) * 0.37).sin() * 0.5)
        .collect();
    Array::from_slice(&data, shape)
}

struct Builder {
    map: HashMap<String, Array>,
    seed: i32,
}

impl Builder {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            seed: 1,
        }
    }

    fn put(&mut self, key: &str, shape: &[i32]) {
        self.map.insert(key.to_owned(), tensor(shape, self.seed));
        self.seed += 1;
    }

    /// `[out, in]` weight plus `[out]` bias — the layout `Linear::load` reads.
    fn linear(&mut self, prefix: &str, out: i32, inp: i32) {
        self.put(&format!("{prefix}.weight"), &[out, inp]);
        self.put(&format!("{prefix}.bias"), &[out]);
    }

    /// `emb.timestep_embedder.linear1/linear2` + `linear`. `hidden` is the embedded width the silu
    /// chain carries; `out` is the scale-shift width the caller reshapes against its adaLN table.
    fn adaln(&mut self, prefix: &str, hidden: i32, out: i32) {
        self.linear(
            &format!("{prefix}.emb.timestep_embedder.linear1"),
            hidden,
            256,
        );
        self.linear(
            &format!("{prefix}.emb.timestep_embedder.linear2"),
            hidden,
            hidden,
        );
        self.linear(&format!("{prefix}.linear"), out, hidden);
    }

    fn attention(&mut self, prefix: &str, q_in: i32, kv_in: i32, inner: i32, out: i32, heads: i32) {
        self.linear(&format!("{prefix}.to_q"), inner, q_in);
        self.linear(&format!("{prefix}.to_k"), inner, kv_in);
        self.linear(&format!("{prefix}.to_v"), inner, kv_in);
        self.linear(&format!("{prefix}.to_out"), out, inner);
        // `Attention` uses `QkNormSpec::full_dim_pre_split`, so the q/k RMSNorm scales are the FULL
        // projection width and are applied before the head split — not `inner / heads`.
        self.put(&format!("{prefix}.q_norm.weight"), &[inner]);
        self.put(&format!("{prefix}.k_norm.weight"), &[inner]);
        self.linear(&format!("{prefix}.to_gate_logits"), heads, q_in);
    }
}

/// Every tensor `AvDiT::from_weights` reads, at the tiny config's dims.
fn tiny_weight_map(cfg: &LtxConfig) -> HashMap<String, Array> {
    let vi = cfg.inner_dim();
    let ai = cfg.audio_inner_dim();
    let vctx = cfg.cross_attention_dim;
    let actx = cfg.audio_cross_attention_dim;
    let mut b = Builder::new();

    // Video stream globals.
    b.linear("patchify_proj", vi, cfg.in_channels);
    b.adaln("adaln_single", vi, ADALN_ROWS * vi);
    b.adaln("prompt_adaln_single", vi, 2 * vi);
    b.adaln("av_ca_video_scale_shift_adaln_single", vi, 4 * vi);
    b.adaln("av_ca_a2v_gate_adaln_single", vi, vi);
    b.put("scale_shift_table", &[2, vi]);
    b.linear("proj_out", cfg.out_channels, vi);

    // Audio stream globals.
    b.linear("audio_patchify_proj", ai, cfg.audio_in_channels);
    b.adaln("audio_adaln_single", ai, ADALN_ROWS * ai);
    b.adaln("audio_prompt_adaln_single", ai, 2 * ai);
    b.adaln("av_ca_audio_scale_shift_adaln_single", ai, 4 * ai);
    b.adaln("av_ca_v2a_gate_adaln_single", ai, ai);
    b.put("audio_scale_shift_table", &[2, ai]);
    b.linear("audio_proj_out", cfg.audio_out_channels, ai);

    for i in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{i}");
        // Video half.
        b.attention(
            &format!("{p}.attn1"),
            vi,
            vi,
            vi,
            vi,
            cfg.num_attention_heads,
        );
        b.attention(
            &format!("{p}.attn2"),
            vi,
            vctx,
            vi,
            vi,
            cfg.num_attention_heads,
        );
        b.linear(&format!("{p}.ff.proj_in"), FFN_MULT * vi, vi);
        b.linear(&format!("{p}.ff.proj_out"), vi, FFN_MULT * vi);
        b.put(&format!("{p}.scale_shift_table"), &[ADALN_ROWS, vi]);
        b.put(&format!("{p}.prompt_scale_shift_table"), &[2, vi]);
        // Audio half.
        b.attention(
            &format!("{p}.audio_attn1"),
            ai,
            ai,
            ai,
            ai,
            cfg.audio_num_attention_heads,
        );
        b.attention(
            &format!("{p}.audio_attn2"),
            ai,
            actx,
            ai,
            ai,
            cfg.audio_num_attention_heads,
        );
        b.linear(&format!("{p}.audio_ff.proj_in"), FFN_MULT * ai, ai);
        b.linear(&format!("{p}.audio_ff.proj_out"), ai, FFN_MULT * ai);
        b.put(&format!("{p}.audio_scale_shift_table"), &[ADALN_ROWS, ai]);
        b.put(&format!("{p}.audio_prompt_scale_shift_table"), &[2, ai]);
        // Cross-modal: a2v queries video and writes video; v2a queries audio and writes audio. Both
        // run at the AUDIO inner dim, which is why the two differ from every other attention here.
        b.attention(
            &format!("{p}.audio_to_video_attn"),
            vi,
            ai,
            ai,
            vi,
            cfg.audio_num_attention_heads,
        );
        b.attention(
            &format!("{p}.video_to_audio_attn"),
            ai,
            vi,
            ai,
            ai,
            cfg.audio_num_attention_heads,
        );
        b.put(&format!("{p}.scale_shift_table_a2v_ca_audio"), &[5, ai]);
        b.put(&format!("{p}.scale_shift_table_a2v_ca_video"), &[5, vi]);
    }
    b.map
}

struct Fixture {
    _dir: tempfile::TempDir,
    component: std::path::PathBuf,
    cfg: LtxConfig,
    prec: Precision,
}

fn fixture() -> Fixture {
    let cfg = tiny_cfg();
    let dir = tempfile::tempdir().expect("tempdir");
    let component = dir.path().join("transformer.safetensors");
    let map = tiny_weight_map(&cfg);
    Array::save_safetensors(
        map.iter().map(|(k, v)| (k.as_str(), v)),
        None,
        component.to_str().expect("utf-8 path"),
    )
    .expect("write synthetic component");
    Fixture {
        _dir: dir,
        component,
        cfg,
        prec: Precision::quant_f32(8, 64),
    }
}

struct Inputs {
    v_latent: Array,
    v_ts: Array,
    v_ctx: Array,
    v_pos: Array,
    a_latent: Array,
    a_ts: Array,
    a_ctx: Array,
    a_pos: Array,
}

fn inputs(cfg: &LtxConfig) -> Inputs {
    let (b, sv, sa, txt) = (1i32, 6i32, 4i32, 3i32);
    let pos = |axes: i32, s: i32| -> Array {
        let data: Vec<f32> = (0..(b * axes * s * 2)).map(|i| (i % s) as f32).collect();
        Array::from_slice(&data, &[b, axes, s, 2])
    };
    Inputs {
        v_latent: tensor(&[b, sv, cfg.in_channels], 91),
        v_ts: tensor(&[b, sv], 92),
        v_ctx: tensor(&[b, txt, cfg.cross_attention_dim], 93),
        v_pos: pos(3, sv),
        a_latent: tensor(&[b, sa, cfg.audio_in_channels], 94),
        a_ts: tensor(&[b, sa], 95),
        a_ctx: tensor(&[b, txt, cfg.audio_cross_attention_dim], 96),
        a_pos: pos(1, sa),
    }
}

fn run(dit: &AvDiT, i: &Inputs) -> (Array, Array) {
    dit.forward(
        &i.v_latent,
        &i.v_ts,
        &i.v_ctx,
        None,
        &i.v_pos,
        &i.a_latent,
        &i.a_ts,
        &i.a_ctx,
        None,
        &i.a_pos,
        None,
        None,
    )
    .expect("joint forward")
}

fn run_video_only(dit: &AvDiT, i: &Inputs) -> Array {
    dit.forward_video_only(&i.v_latent, &i.v_ts, &i.v_ctx, None, &i.v_pos, None, None)
        .expect("video-only forward")
}

fn resident(f: &Fixture) -> AvDiT {
    let w = Weights::from_file(&f.component).expect("open component");
    AvDiT::from_weights(&w, &f.cfg, f.prec).expect("resident AvDiT")
}

fn streamed(f: &Fixture, window: usize) -> AvDiT {
    let w = Weights::from_file(&f.component).expect("open component");
    let stream =
        LtxBlockStream::new(&f.component, f.cfg.clone(), f.prec, &[]).expect("construct stream");
    let dit = AvDiT::from_weights_streamed(&w, &f.cfg, f.prec, stream).expect("streamed AvDiT");
    dit.set_transformer_window(window).expect("select window");
    dit
}

fn bits(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::transforms::eval([&a]).unwrap();
    a.as_slice::<f32>().to_vec()
}

/// **AC2 — loader identity.** The failure this catches is invisible in output: a "streamed" load that
/// quietly kept the resident stack renders byte-identical frames while bounding nothing.
///
/// The observation is therefore neither on the output nor on the request flag. It is on the loader
/// itself: `block_stream`'s counters increment inside `LtxBlockStream::open` and `::materialize`,
/// *after* the component file was parsed and *after* the exact `AvBlock` was assembled. A resident
/// forward cannot move them, because a resident `AvDiT` holds no `LtxBlockStream` to call.
#[test]
fn only_the_streamed_loader_opens_windows_and_materializes_blocks() {
    let f = fixture();
    let i = inputs(&f.cfg);
    let n_blocks = f.cfg.num_layers as usize;

    let resident_dit = resident(&f);
    assert!(!resident_dit.is_block_streamed());
    assert_eq!(
        resident_dit.resident_blocks(),
        n_blocks,
        "a resident stack holds every block"
    );
    reset_block_stream_diagnostics();
    let _ = run(&resident_dit, &i);
    assert_eq!(
        block_stream_diagnostics(),
        BlockStreamDiagnostics::default(),
        "the resident forward must not open a window or materialize a block — if this counts \
         anything, `Resident` is silently paying rung 4's cost"
    );

    let streamed_dit = streamed(&f, 1);
    assert!(streamed_dit.is_block_streamed());
    assert_eq!(
        streamed_dit.resident_blocks(),
        0,
        "a deferred stack must hold ZERO blocks — that is the entire rung"
    );
    assert_eq!(
        streamed_dit.num_blocks(),
        n_blocks,
        "...while still RUNNING every block"
    );
    let plan = streamed_dit.block_plan();
    assert!(
        plan.is_bounded(),
        "premise: window 1 over {n_blocks} blocks must bound something"
    );
    assert_eq!(plan.window_count(), n_blocks);

    reset_block_stream_diagnostics();
    let _ = run(&streamed_dit, &i);
    let observed = block_stream_diagnostics();
    assert_eq!(
        observed.window_reopens, n_blocks as u64,
        "the driver must open one FRESH view per window; a view reused across windows keeps every \
         materialized buffer alive and the bound silently does not hold"
    );
    assert_eq!(
        observed.block_materializations, n_blocks as u64,
        "every block must be rebuilt exactly once per forward"
    );
}

/// The selected window must be the executed one. A window accepted and then ignored makes every
/// calibration row describe a run that never happened.
#[test]
fn the_selected_window_is_the_executed_one() {
    let f = fixture();
    let i = inputs(&f.cfg);
    let n_blocks = f.cfg.num_layers as usize;
    for window in [1usize, 2, 4] {
        let dit = streamed(&f, window);
        reset_block_stream_diagnostics();
        let _ = run(&dit, &i);
        let observed = block_stream_diagnostics();
        assert_eq!(
            observed.window_reopens,
            n_blocks.div_ceil(window) as u64,
            "window {window} must walk {} windows",
            n_blocks.div_ceil(window)
        );
        assert_eq!(observed.block_materializations, n_blocks as u64);
    }
}

/// **AC3 — windowed output is bit-identical to unwindowed.** Exact equality, not a tolerance:
/// windowing changes only WHEN a block's weights are materialized, never the arithmetic, so any
/// difference at all is a real defect rather than accumulated float noise.
#[test]
fn windowed_output_is_bit_identical_to_unwindowed() {
    let f = fixture();
    let i = inputs(&f.cfg);
    let (rv, ra) = run(&resident(&f), &i);
    let (rv_bits, ra_bits) = (bits(&rv), bits(&ra));

    // Non-degeneracy: a forward returning all zeros (or all NaN) would make "identical" vacuously
    // true for every window.
    assert!(
        rv_bits.iter().any(|v| v.is_finite() && *v != 0.0),
        "premise: the resident video velocity must be non-degenerate"
    );
    assert!(
        ra_bits.iter().any(|v| v.is_finite() && *v != 0.0),
        "premise: the resident audio velocity must be non-degenerate"
    );

    for window in [1usize, 2, 3, 4] {
        let (sv, sa) = run(&streamed(&f, window), &i);
        assert!(
            array_eq(&sv, &rv, None).unwrap().item::<bool>(),
            "window {window}: video velocity is not bit-identical to the resident stack"
        );
        assert!(
            array_eq(&sa, &ra, None).unwrap().item::<bool>(),
            "window {window}: audio velocity is not bit-identical to the resident stack"
        );
    }
}

/// DFR temporal rounds use the video-only reduction, so rung 4 must stream that path too rather
/// than accepting the memory contract and then iterating only a resident block vector.
#[test]
fn windowed_video_only_output_is_bit_identical_to_unwindowed() {
    let f = fixture();
    let i = inputs(&f.cfg);
    let resident = run_video_only(&resident(&f), &i);
    let resident_bits = bits(&resident);
    assert!(
        resident_bits.iter().any(|v| v.is_finite() && *v != 0.0),
        "premise: the resident video-only velocity must be non-degenerate"
    );

    for window in [1usize, 2, 3, 4] {
        let streamed = run_video_only(&streamed(&f, window), &i);
        assert!(
            array_eq(&streamed, &resident, None).unwrap().item::<bool>(),
            "window {window}: video-only velocity is not bit-identical to the resident stack"
        );
    }
}

/// A ragged tail must not drop blocks. Window 3 over 4 blocks leaves a 1-block tail; losing it would
/// silently skip a layer. The bit-identity test would catch that too — this one names the mechanism
/// so the diagnosis is not "output differs somewhere".
#[test]
fn a_ragged_tail_still_materializes_every_block() {
    let f = fixture();
    let i = inputs(&f.cfg);
    let dit = streamed(&f, 3);
    assert_eq!(dit.block_plan().window_count(), 2, "4 blocks at window 3");
    reset_block_stream_diagnostics();
    let _ = run(&dit, &i);
    assert_eq!(
        block_stream_diagnostics().block_materializations,
        f.cfg.num_layers as u64
    );
}

/// Rung 3 must reach the streamed path. The budget is recorded on the stream and replayed onto every
/// materialized block; without the replay a rung-3 + rung-4 composition — the one the cost-order
/// default actually produces — runs unbounded attention with identical output.
#[test]
fn the_attention_budget_is_replayed_onto_every_streamed_block() {
    let f = fixture();
    let budget = mlx_gen::attention::AttentionBudget::CONSTRAINED;

    let mut streamed_dit = streamed(&f, 1);
    assert!(
        streamed_dit.attention_budget().is_unbounded(),
        "premise: the load default is unbounded"
    );
    streamed_dit.set_attention_budget(budget);
    assert_eq!(
        streamed_dit.attention_budget(),
        budget,
        "a streamed stack has no blocks to write to, so the budget must live on the stream and be \
         replayed per window"
    );

    // And the replay must reach a materialized block, not merely the stream's own field.
    let AvBlocks::Streamed(stream) = &streamed_dit.blocks else {
        panic!("streamed fixture");
    };
    let mut view = stream.open().expect("open view");
    let block = stream
        .materialize(&mut view, 0)
        .expect("materialize block 0");
    assert_eq!(
        block.attention_budget(),
        budget,
        "the block a window rebuilds must carry the selected budget"
    );

    let mut resident_dit = resident(&f);
    resident_dit.set_attention_budget(budget);
    assert_eq!(resident_dit.attention_budget(), budget);
}

/// All six attentions in an AV block move together. Bounding only the video half would leave the
/// audio branch and both cross-modal attentions unbounded while the contract claimed rung 3.
#[test]
fn every_attention_in_a_block_carries_the_selected_budget() {
    let f = fixture();
    let w = Weights::from_file(&f.component).unwrap();
    let mut block = AvBlock::load(&w, "transformer_blocks.0", &f.cfg, f.prec).expect("load block");
    let budget = mlx_gen::attention::AttentionBudget::CONSTRAINED;
    block.set_attention_budget(budget);
    for (name, attn) in [
        ("attn1", &block.attn1),
        ("attn2", &block.attn2),
        ("audio_attn1", &block.a_attn1),
        ("audio_attn2", &block.a_attn2),
        ("audio_to_video_attn", &block.a2v),
        ("video_to_audio_attn", &block.v2a),
    ] {
        assert_eq!(
            attn.attn_budget, budget,
            "{name} did not receive the budget"
        );
    }
}

/// A window may not be selected on a resident stack: silently accepting one is how a calibration
/// record comes to describe a bound that was never applied.
#[test]
fn a_window_cannot_be_selected_on_a_resident_stack() {
    let f = fixture();
    let error = resident(&f)
        .set_transformer_window(1)
        .expect_err("a resident stack must refuse a window");
    assert!(
        error.to_string().contains("deferred"),
        "the refusal must name the required load shape, got: {error}"
    );
}

/// A stream whose depth disagrees with the config is refused at construction: a plan built from a
/// desynchronized depth silently skips or repeats layers.
#[test]
fn a_desynchronized_block_depth_is_refused() {
    let f = fixture();
    let w = Weights::from_file(&f.component).unwrap();
    let mut wrong = f.cfg.clone();
    wrong.num_layers = f.cfg.num_layers + 1;
    let stream = LtxBlockStream::new(&f.component, wrong, f.prec, &[]).unwrap();
    // `AvDiT` is deliberately not `Debug` (it would print the whole trunk), so match rather than
    // `expect_err`.
    let error = match AvDiT::from_weights_streamed(&w, &f.cfg, f.prec, stream) {
        Ok(_) => panic!("a depth mismatch must be refused"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("block stream declares"),
        "got: {error}"
    );
}
