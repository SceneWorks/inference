//! sc-17152 — **the two measured facts the MiniMax-H3 duration bound rests on**, on Metal.
//!
//! `#[ignore]`d: each allocates multi-GB arrays at the shipped DiT widths. They need **no weights**
//! — the questions are about MLX's kernels at this model's shapes, not about the checkpoint — so
//! they run without `MINIMAX_H3_SNAPSHOT`:
//!
//! ```sh
//! cargo test -p mlx-gen-minimax-h3 --test integration sequence_cost_real:: -- --ignored --nocapture \
//!   --test-threads=1
//! ```
//!
//! # What each one settles
//!
//! | test | question | why it cannot be inherited |
//! |---|---|---|
//! | [`the_feed_forward_projection_is_exact_above_the_writable_bound`] | does `matmul` write correctly past `i32::MAX` **elements**? | sc-12748 measured `conv3d` / `pad` / `concatenate` / `reshape` / `as_slice` on this pin. **`matmul` was not on that list**, and the tensor that crosses here is a matmul output |
//! | [`the_fused_attention_streams_the_scores_at_the_minimax_h3_sequence`] | does MLX's fused SDPA materialize `[B, H, S, S]`? | `mlx_gen::attention` measured "no" at the Z-Image shape — f32, 30 heads, `Sq = 16416`. This model is bf16, **56** heads and `Sq` up to **104 030**, six times longer, and the fused kernel's dispatch is shape- and dtype-conditional |
//! | [`bounded_attention_at_the_minimax_h3_shape_is_exact_and_costs_peak_in_isolation`] | what would sc-18661's rung-3 chunking do here? | hands sc-18661 measured numbers at this family's shape rather than Z-Image's |
//!
//! # Probe rules
//!
//! Inherited from `mlx-gen/tests/mlx_write_bound_probe.rs`, because they are what stopped the
//! sc-12291 corruption hiding: **position-dependent** data (`i % 251`, exact in bf16 and non-constant
//! along every axis, so a scrambled index lands on a different value with probability ~250/251);
//! compare only **sub-bound slices**, never a reduction *over* the oversized tensor — a checksum of a
//! corrupt tensor is itself an over-bound read and silently agrees; and a single `eval` before the
//! readback, because MLX is lazy and an un-forced graph measures nothing.
//!
//! Each geometry **brackets** the bound with a real lattice rung on either side, so a probe that
//! measured nothing (an allocation that silently shrank, a shape that never crossed) fails rather
//! than passing.

use std::time::Instant;

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::matmul;
use mlx_rs::{Array, Dtype};

use mlx_gen::attention::{sdpa_budgeted_bhsd, AttentionBudget, AttentionPlan};
use mlx_gen_minimax_h3::cost::{DitSequenceCost, MAX_WRITABLE_ELEMS};
use mlx_gen_minimax_h3::tensor::slice_axis;
use mlx_gen_minimax_h3::MiniMaxH3DitConfig;

/// Packed sequence length of a 768×1344 / 243-frame request — `2.106e9` feed-forward elements,
/// **0.981×** the bound. The largest writable rung on the default canvas.
const SEQ_UNDER: i64 = 73_450;
/// Packed sequence length of a 768×1344 / 260-frame request — `2.252e9` feed-forward elements,
/// **1.049×** the bound. The first rung over it.
const SEQ_OVER: i64 = 78_546;

/// Position-dependent value for flat index `i`. `0..=250`, exact in bfloat16 (8 explicit mantissa
/// bits carry every integer below 256), and non-constant along every axis.
fn pv(i: i64) -> f32 {
    (i % 251) as f32
}

/// Peak MLX bytes a closure moves, with the process-wide cumulative counter reset first.
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, u64) {
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let out = f();
    (out, mlx_rs::memory::get_peak_memory() as u64)
}

/// **The `i32::MAX` correctness gate.**
///
/// The widest tensor one MiniMax-H3 DiT forward writes is `ff.net.0.proj`'s output,
/// `[1, S, 2 · ffn_dim]` = `[1, S, 28672]`, and on the default 768×1344 canvas it crosses
/// `i32::MAX` elements between 243 and 260 frames — **inside the advertised 5–15 s envelope**. That
/// is a `matmul`, and `matmul` is the one op in the forward that sc-12748's write-bound sweep did
/// not cover.
///
/// The weight is a **selection matrix**: `W[o, c] = 1` iff `c == o % hidden_size`, so
/// `y[0, s, o] = x[0, s, o % hidden_size]` exactly — every one of the 2.25 G output elements has a
/// predicted value, and each is a sum with a single non-zero term, so bf16 rounding cannot blur the
/// comparison. Positions are sampled in **both** the sub-bound and the over-bound regions of the
/// output, and the under-bound geometry runs first so a corruption below the bound fails the premise
/// rather than being read as the finding.
#[test]
#[ignore = "sc-17152: allocates ~6 GB of MLX arrays at the shipped DiT widths; run with --ignored on Metal"]
fn the_feed_forward_projection_is_exact_above_the_writable_bound() {
    let cfg = MiniMaxH3DitConfig::default();
    let hidden = i64::from(cfg.hidden_size); // 5376
    let out = 2 * i64::from(cfg.ffn_dim); // 28672 — `[value | gate]`, both halves in one tensor

    // The selection weight, `[28672, 5376]` — 308 MB at bf16.
    let mut wbuf = vec![0f32; (out * hidden) as usize];
    for o in 0..out {
        wbuf[(o * hidden + o % hidden) as usize] = 1.0;
    }
    let weight = Array::from_slice(&wbuf, &[out as i32, hidden as i32])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    weight.eval().unwrap();
    drop(wbuf);

    let run = |seq: i64| -> (i64, i64, i64) {
        let xhost: Vec<f32> = (0..seq * hidden).map(pv).collect();
        let x = Array::from_slice(&xhost, &[1, seq as i32, hidden as i32])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        x.eval().unwrap();
        drop(xhost);

        let y = matmul(&x, weight.t()).unwrap();
        y.eval().unwrap();
        assert_eq!(
            y.shape(),
            &[1, seq as i32, out as i32],
            "matmul output shape"
        );
        let elems = seq * out;

        // Read back only sub-bound WINDOWS of the output, but windows whose SOURCE offsets straddle
        // the bound — that is the read path under test. 64 rows is 1.8 M elements, 0.001× the bound.
        let window = 64i64;
        let mut first_bad = -1i64;
        let mut checked = 0i64;
        for row0 in [0i64, seq / 2, seq - window] {
            let slice = slice_axis(&y, 1, row0 as i32, (row0 + window) as i32)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            slice.eval().unwrap();
            let host = slice.as_slice::<f32>();
            for r in [0i64, window / 2, window - 1] {
                let s = row0 + r;
                for o in [0i64, 1, 5375, 5376, 14335, 14336, 28671] {
                    let got = host[(r * out + o) as usize];
                    let want = pv(s * hidden + o % hidden);
                    checked += 1;
                    if (got - want).abs() > 1e-3 && first_bad < 0 {
                        first_bad = s * out + o;
                    }
                }
            }
        }
        (elems, first_bad, checked)
    };

    let (under_elems, under_bad, under_checked) = run(SEQ_UNDER);
    eprintln!(
        "[ff.proj S={SEQ_UNDER} below] out_elems={under_elems} (={:.4}x 2^31)  checked={under_checked}  \
         first_bad_offset={under_bad}",
        under_elems as f64 / MAX_WRITABLE_ELEMS as f64
    );
    let (over_elems, over_bad, over_checked) = run(SEQ_OVER);
    eprintln!(
        "[ff.proj S={SEQ_OVER} above] out_elems={over_elems} (={:.4}x 2^31)  checked={over_checked}  \
         first_bad_offset={over_bad}",
        over_elems as f64 / MAX_WRITABLE_ELEMS as f64
    );

    // The premise: the two geometries really do bracket the bound, and they are the real lattice
    // rungs the cost model names rather than convenient synthetic sizes.
    assert!(
        under_elems < MAX_WRITABLE_ELEMS && over_elems > MAX_WRITABLE_ELEMS,
        "the probe must bracket 2^31: {under_elems} / {over_elems}"
    );
    assert_eq!(
        DitSequenceCost::new(SEQ_UNDER, &cfg)
            .unwrap()
            .ff_proj_elements,
        under_elems,
        "the probe must run the geometry the cost model bounds"
    );
    assert!(!DitSequenceCost::new(SEQ_UNDER, &cfg)
        .unwrap()
        .over_writable_bound());
    assert!(DitSequenceCost::new(SEQ_OVER, &cfg)
        .unwrap()
        .over_writable_bound());
    assert!(
        under_checked >= 60 && over_checked >= 60,
        "positions sampled"
    );

    assert!(
        under_bad < 0,
        "the matmul BELOW the bound is already wrong at offset {under_bad} — the probe's premise is \
         broken, not MLX's over-bound behaviour"
    );
    // The finding. A regression here — a future MLX bump re-introducing an i32 output offset in the
    // GEMM — makes the widest tensor of every long/large MiniMax-H3 request silently wrong, and the
    // 243-frame cap in `cost::largest_writable_frame_count` becomes an enforcement rather than a
    // characterization.
    assert!(
        over_bad < 0,
        "the SwiGLU projection matmul CORRUPTED above i32::MAX (first bad flat offset {over_bad}). \
         Every 768x1344 request past 243 frames writes a silently wrong feed-forward — cap \
         hardMaxDuration at 10.125 s and re-open sc-17152"
    );
}

/// **Does MLX's fused SDPA materialize the score tensor at *this* model's sequence?**
///
/// `mlx_gen::attention`'s contract says no, measured at the Z-Image DiT shape (f32, `H = 30`,
/// `Sq = 16416`). MiniMax-H3 is bf16 with **56** heads at up to `Sq = 104 030`, and MLX's fused
/// kernel selection is conditional on dtype and shape, so the contract is re-measured here rather
/// than assumed.
///
/// The discriminator is the **scaling**, not a single number: a streaming kernel's peak is
/// `4 · B·H·S·D · sizeof` — q, k, v and the output — which is **linear** in `S`, while a
/// materialized `[B, H, S, S]` score tensor is **quadratic**. At `S = 19 574` (576×320 at the
/// lattice top) those differ by 38×, and at `S = 104 030` by 203× — far outside any measurement
/// noise. The ladder runs ascending so a materializing kernel is caught at the cheap end rather than
/// by an allocation the machine cannot serve.
#[test]
#[ignore = "sc-17152: allocates ~6 GB of MLX arrays; run with --ignored on Metal"]
fn the_fused_attention_streams_the_scores_at_the_minimax_h3_sequence() {
    let cfg = MiniMaxH3DitConfig::default();
    let (h, d) = (cfg.num_attention_heads, cfg.attention_head_dim);
    let scale = 1.0 / (d as f32).sqrt();

    // 576x320 @ 124f and @ 345f, 1344x768 @ 124f and @ 345f — the four corners of the envelope.
    let mut previous: Option<(i64, u64)> = None;
    for seq in [7_138i64, 19_574, 37_774, 104_030] {
        let qkv_bytes = 4 * i64::from(h) * seq * i64::from(d) * 2;
        let score_bytes = i64::from(h) * seq * seq * 2;

        // q/k/v are built and forced BEFORE the measurement window opens. Generating them inside it
        // would charge the peak for the f32 random draws they are cast down from — three arrays at
        // twice the bf16 width — and that transient alone would exceed the streaming prediction,
        // making a streaming kernel indistinguishable from a materializing one.
        let shape = [1, h, seq as i32, d];
        let mk = |s: u64| {
            let a = mlx_rs::random::normal::<f32>(
                &shape,
                None,
                None,
                Some(&mlx_rs::random::key(s).unwrap()),
            )
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
            mlx_rs::transforms::eval([&a]).unwrap();
            a
        };
        let (q, k, v) = (mk(1), mk(2), mk(3));
        mlx_rs::memory::clear_cache();

        // The measured window holds the attention call and NOTHING else. An f32 cast or an `abs()`
        // over the whole output inside it would each add another full-size transient and put the
        // peak at exactly 2x the prediction — measured, and indistinguishable at a glance from a
        // kernel that was materializing something. The "it really ran" evidence is taken afterwards,
        // from a slice.
        let ((out, wall), peak) = peak_of(|| {
            let started = Instant::now();
            let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None).unwrap();
            // MLX is lazy: without this the timer measures graph construction and reports ~0, and
            // the peak counter never sees the kernel at all.
            mlx_rs::transforms::eval([&o]).unwrap();
            (o, started.elapsed())
        });
        let sum: f32 = slice_axis(&out, 2, 0, 8)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap()
            .item();

        eprintln!(
            "[sdpa S={seq}] peak {:.3} GB  (q/k/v/out predict {:.3} GB, materialized scores would \
             add {:.1} GB)  {:.3} s  sum|out|={sum:.4e}",
            peak as f64 / 1e9,
            qkv_bytes as f64 / 1e9,
            score_bytes as f64 / 1e9,
            wall.as_secs_f64()
        );
        assert!(sum.is_finite() && sum > 0.0, "the attention really ran");

        // The bound that discriminates. The prediction is exact — three resident inputs and one
        // output — so 1.35x is slack for allocator granularity, not for a hidden tensor: at the
        // smallest S here the score tensor alone is 14x the whole prediction, and at the largest,
        // 203x.
        assert!(
            (peak as i64) < qkv_bytes * 27 / 20,
            "S={seq}: peak {peak} B is over twice the {qkv_bytes} B q/k/v/out prediction — the \
             fused kernel appears to be materializing the {score_bytes} B score tensor. \
             MiniMax-H3 would then be over i32::MAX in the ATTENTION at every legal geometry and \
             sc-18661's bounded attention becomes a correctness fix, not a memory one"
        );
        // The anti-false-green guard, and it is tight on purpose. Without the `eval` inside the
        // window MLX would build the graph and run nothing, leaving the peak at the three resident
        // inputs — 0.75x the prediction — which this rejects. A looser bound would let a
        // never-executed kernel report "streams the scores".
        assert!(
            (peak as i64) > qkv_bytes * 9 / 10,
            "S={seq}: peak {peak} B is below 0.9x the {qkv_bytes} B q/k/v/out prediction — the \
             output was never materialized, so this measured graph construction, not the kernel"
        );

        // ...and the scaling really is linear, not quadratic. Checked against the previous rung so a
        // single flat number cannot pass by accident.
        if let Some((prev_seq, prev_peak)) = previous {
            let growth = peak as f64 / prev_peak as f64;
            let linear = seq as f64 / prev_seq as f64;
            eprintln!("      growth {growth:.2}x for a {linear:.2}x longer sequence");
            assert!(
                growth < linear * 1.35,
                "S {prev_seq} -> {seq}: peak grew {growth:.2}x against a {linear:.2}x sequence — \
                 that is quadratic, i.e. a materialized score tensor"
            );
        }
        previous = Some((seq, peak));
    }
}

/// **What sc-18661's rung-3 chunking would do at this family's shape.**
///
/// Two things this hands over, neither inferable from Z-Image's numbers:
///
/// 1. **exactness** — query-row chunking is mathematically equivalent but not bit-identical (it
///    changes the GEMM's `M`), so the agreement is asserted as a **relative max-abs difference**.
///    A norm, cosine or checksum comparison is blind here: all three are dominated by the bulk of an
///    already-agreeing tensor and would pass on a chunk boundary that dropped rows entirely;
/// 2. **cost in isolation** — with q/k/v materialized and pinned alive there is no upstream graph
///    for `eval_per_chunk` to cut, so chunking can only add transients. `mlx_gen::attention`
///    measured +50% peak at the Z-Image shape; this reports the number at 56 heads.
///
/// The isolated figure is deliberately **not** a prediction of the in-forward result — the whole
/// MLX-side saving comes from the graph cut, which only exists inside a real DiT forward. That
/// measurement is sc-18661's to make; this is the shape and the exactness it can build on.
#[test]
#[ignore = "sc-17152: allocates ~2 GB of MLX arrays; run with --ignored on Metal"]
fn bounded_attention_at_the_minimax_h3_shape_is_exact_and_costs_peak_in_isolation() {
    let cfg = MiniMaxH3DitConfig::default();
    let (h, d) = (cfg.num_attention_heads, cfg.attention_head_dim);
    let scale = 1.0 / (d as f32).sqrt();
    // 576x320 at the lattice top — the largest sequence that leaves room for both arms at once.
    let seq = 19_574i64;

    let shape = [1, h, seq as i32, d];
    let mk = |s: u64| {
        mlx_rs::random::normal::<f32>(&shape, None, None, Some(&mlx_rs::random::key(s).unwrap()))
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    };
    let (q, k, v) = (mk(11), mk(12), mk(13));
    mlx_rs::transforms::eval([&q, &k, &v]).unwrap();

    let arm = |plan: AttentionPlan<'_>| -> (Array, u64, f64) {
        let ((out, wall), peak) = peak_of(|| {
            let started = Instant::now();
            let o = sdpa_budgeted_bhsd(&q, &k, &v, scale, None, plan).unwrap();
            mlx_rs::transforms::eval([&o]).unwrap();
            (o, started.elapsed().as_secs_f64())
        });
        (out, peak, wall)
    };

    // **A warm-up arm before either measured one.** q/k/v are cast down from f32 random draws, and
    // on the first call after that those f32 sources are still resident — measured, and it inflates
    // whichever arm runs first by ~0.56 GB. Running one unmeasured arm settles the allocator so the
    // two measured arms start from the same state. Without it this test reported **+0.2%** when it
    // ran first in the binary and **+50.3%** under a filter: the same two arms, with the ordering
    // deciding the answer.
    let (warm, warm_peak, _) = arm(AttentionPlan::UNBOUNDED);
    drop(warm);
    mlx_rs::memory::clear_cache();

    let (unbounded, un_peak, un_wall) = arm(AttentionPlan::UNBOUNDED);
    let (chunked, ch_peak, ch_wall) = arm(AttentionPlan::budgeted(AttentionBudget::CONSTRAINED));

    // The settled baseline must be the q/k/v/out prediction, or the comparison below is against a
    // contaminated reference rather than against the unbounded kernel.
    let predicted = 4i64 * i64::from(h) * seq * i64::from(d) * 2;
    eprintln!(
        "[rung-3 warm-up] first arm {:.3} GB, settled unbounded arm {:.3} GB (q/k/v/out predicts \
         {:.3} GB)",
        warm_peak as f64 / 1e9,
        un_peak as f64 / 1e9,
        predicted as f64 / 1e9
    );
    assert!(
        (un_peak as i64) < predicted * 11 / 10,
        "the settled unbounded arm peaked at {un_peak} B against a {predicted} B prediction — the \
         baseline is still contaminated, so the chunked comparison would be against the wrong \
         reference"
    );

    // The chunking must actually have engaged, or this compares the fast path with itself.
    let block = AttentionBudget::CONSTRAINED.query_block(1, h, seq as i32, seq as i32);
    eprintln!(
        "[rung-3 S={seq} H={h}] query block {block} rows ({} chunks); unbounded {:.3} GB / \
         {un_wall:.3} s, chunked {:.3} GB / {ch_wall:.3} s ({:+.1}% peak)",
        (seq as i32 + block.max(1) - 1) / block.max(1),
        un_peak as f64 / 1e9,
        ch_peak as f64 / 1e9,
        (ch_peak as f64 / un_peak as f64 - 1.0) * 100.0
    );
    assert!(
        block < seq as i32,
        "the 64 Mi budget must chunk a {seq}-row sequence, or this arm is the fast path"
    );

    // **Relative max-abs difference**, not a norm or a checksum. Both arms are cast to f32 for the
    // comparison so the difference is not quantized to bf16's ~3 decimal digits and hidden.
    let a = unbounded.as_dtype(Dtype::Float32).unwrap();
    let b = chunked.as_dtype(Dtype::Float32).unwrap();
    let max_abs: f32 = a
        .subtract(&b)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item();
    let scale_of: f32 = a.abs().unwrap().max(None).unwrap().item();
    let relative = max_abs / scale_of;
    eprintln!(
        "      relative max-abs difference {relative:.3e} (unbounded max |out| {scale_of:.4})"
    );
    assert!(
        scale_of > 1e-3,
        "the reference arm is degenerate ({scale_of})"
    );
    assert!(
        relative < 1e-2,
        "query-row chunking changed the attention output by {relative:.3e} relative — that is a \
         wiring error, not the ULP drift narrowing the GEMM's M dimension can produce"
    );
}
