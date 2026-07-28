//! Krea Realtime 14B causal AR forward — **vendored torch golden** (sc-15064, S16).
//!
//! Defence-in-depth on top of the torch-free S3 invariants in `causal_forward.rs`. Those invariants
//! (cached-incremental == full-masked-recompute, single-chunk == stock `forward_packed`, a
//! hand-derived [4,4] mask matrix) pin our causal path against *itself* and against the already
//! validated Wan attention — they are discriminating, but every one of them is derived from **our
//! reading** of the reference. This file adds the missing EXTERNAL anchor: numbers dumped from the
//! actual reference `CausalWanModel`, so a mis-*interpretation* of the reference (as opposed to a
//! mis-implementation of our interpretation) fails a test.
//!
//! Golden: `tests/fixtures/causal_ar_golden.safetensors` (~600 KB), produced by
//! `tools/dump_causal_ar_golden.py` from
//! `krea/krea-realtime-video` @ `6b5d204f9d14c3c3a59608e49d1da7fad90daf8d`,
//! `transformer/causal_model.py::CausalWanModel` (Apache-2.0). The generator carries the full
//! provenance record and the exact list of reference patches it applies (portability / precision /
//! geometry — none of them touch the mask rule or the RoPE offset). `../NOTICE` carries the
//! attribution. Three AR chunks of two latent frames each (24 tokens total) at a 2-layer, dim-64
//! geometry: small, OOM-safe, and deep enough that the last chunk sits four latent frames into the
//! clip.
//!
//! Three tests:
//!
//!   * [`mask_rule_matches_reference_get_sdpa_mask`] — our [`build_block_causal_mask`] must reproduce
//!     the reference's own `get_sdpa_mask` allow/deny matrix, entry for entry. The direct mask-RULE
//!     anchor: `get_sdpa_mask` is fully parameterised upstream, so it is dumped completely unpatched.
//!   * [`multi_chunk_causal_forward_matches_torch_golden`] — the ≥2-chunk numeric anchor. Chunk 0
//!     fills the KV cache; chunks 1 and 2 run at `current_start = block_size` / `2·block_size` (the
//!     path where a mask-interpretation or RoPE-`start_frame` error shows up) and all three must
//!     match the reference velocities. Note the limit of this test taken alone: it catches a mask
//!     rule error only insofar as that error changes the *effective attention set*. In this geometry
//!     each AR chunk is exactly one block, so `block_causal_mask` returns `None` and no mask is ever
//!     materialised in the numeric path. A rule change that makes the mask non-`None` does surface
//!     here (making the intra-block attention causal instead of bidirectional moves `mean_rel` to
//!     1.05e-1), but a rule change that merely widens the allowed set *ahead* of the current block is
//!     structurally invisible: in the incremental AR path no future-block key is ever present in
//!     `kv_positions`. Concretely, an off-by-one in the block boundary (`causal.rs`, `end_of_block`
//!     returning `(q / bs + 1) * bs + 1`) leaves every chunk here bit-identical and is caught **only**
//!     by test (1), at 16/576 mismatched mask entries. The two tests are complementary, not
//!     redundant: (1) is the standalone mask-RULE anchor, this one is the end-to-end numeric anchor.
//!   * [`wrong_rope_offset_fails_the_torch_golden`] — the NEGATIVE CONTROL. A golden test that would
//!     pass against a wrong implementation is worthless, so this reproduces the deepest chunk with
//!     the causal-RoPE `start_frame` deliberately zeroed and asserts the golden comparison goes RED,
//!     with a wide measured margin over the correct run.
//!
//! Tolerance: the reference runs in **f32**; our transformer is cast to **bf16** on load
//! (`convert::TRANSFORMER_DTYPE`), as it is in production. The floor here is therefore bf16
//! activation precision over a 2-layer trunk, not an f32-vs-f32 comparison, and the gate is stated
//! relative to the reference signal (`mean_rel` / `peak_rel`, the metrics
//! `mlx-gen-sana/tests/transformer_parity.rs` uses). Measured on this fixture: a correct forward sits
//! at `mean_rel` 1.7e-3 / 2.2e-3 / 2.0e-3 (peak_rel ≤ 2.3e-3) across the three chunks, while the
//! wrong-RoPE-offset control sits at `mean_rel` 2.9e-2 — 14x higher. The gate is set at `8e-3`,
//! ~4x above the observed bf16 floor and ~3.5x below the control, so it separates the two cleanly.

use std::collections::HashMap;

use mlx_gen::weights::Weights;
use mlx_gen_krea_realtime::{
    build_block_causal_mask, load_krea_realtime_transformer, CausalKreaTransformer,
    KreaRealtimeConfig,
};
use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/causal_ar_golden.safetensors"
);

/// Geometry of the golden — must stay in lockstep with `tools/dump_causal_ar_golden.py`.
const LATENT_HW: i32 = 4;
const FRAMES_PER_BLOCK: usize = 2;
const FRAME_SEQ_LENGTH: usize = 4; // (4/2)·(4/2) patch tokens per latent frame
const BLOCK_TOKENS: usize = FRAMES_PER_BLOCK * FRAME_SEQ_LENGTH; // 8
const NUM_CHUNKS: usize = 3;
const TOTAL_TOKENS: usize = NUM_CHUNKS * BLOCK_TOKENS; // 24

/// `mean_rel` gate — see the module docs for how it was derived from the measured bf16 floor and the
/// negative control. The negative control asserts a wrong implementation lands *above* this.
const MEAN_REL_GATE: f32 = 8e-3;
/// `peak_rel` gate: a single worst-case element is allowed several bf16 ulps of the peak signal.
const PEAK_REL_GATE: f32 = 1e-2;

fn f32a(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32a(got), f32a(want)).unwrap()).unwrap();
    let denom = max_op(abs(f32a(want)).unwrap(), None)
        .unwrap()
        .item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32a(got), f32a(want)).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(f32a(want)).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

/// The tiny Krea config the golden was generated at (`local_attn_size = -1`, `sink_size = 0` — the
/// shipped checkpoint's values).
fn golden_cfg() -> KreaRealtimeConfig {
    let mut c = KreaRealtimeConfig::krea_realtime_14b();
    c.wan.dim = 64;
    c.wan.num_heads = 2; // head_dim 32
    c.wan.num_layers = 2;
    c.wan.ffn_dim = 128;
    c.wan.freq_dim = 32;
    c.wan.text_dim = 32;
    c.wan.text_len = 8;
    c.wan.in_dim = 16;
    c.wan.out_dim = 16;
    c.wan.patch_size = (1, 2, 2);
    c.ar.num_frames_per_block = FRAMES_PER_BLOCK;
    c.ar.frame_seq_length = FRAME_SEQ_LENGTH;
    c.ar.local_attn_size = -1;
    c.ar.sink_size = 0;
    c.ar.seq_length = 100_000; // global read window keeps the whole history
    c
}

/// The golden's `weights.<state_dict key>` tensors as the native (pre-sanitize) Wan inventory the
/// loader consumes. The reference `CausalWanModel` state-dict names ARE the native Wan names, so this
/// is a prefix strip and nothing else — no renaming, no reshaping, so the golden cannot be quietly
/// "made to fit" on the way in.
fn native_map(golden: &Weights) -> HashMap<String, Array> {
    let mut map = HashMap::new();
    for key in golden.keys() {
        if let Some(rest) = key.strip_prefix("weights.") {
            map.insert(rest.to_string(), golden.require(key).unwrap().clone());
        }
    }
    assert!(!map.is_empty(), "golden carries no `weights.` tensors");
    map
}

fn setup(
    golden: &Weights,
) -> (
    CausalKreaTransformer,
    Vec<(Array, Array)>,
    KreaRealtimeConfig,
) {
    let cfg = golden_cfg();
    let dit = load_krea_realtime_transformer(native_map(golden), &cfg).expect("load golden DiT");
    let causal = CausalKreaTransformer::new(dit, &cfg);
    let ctx_in = f32a(golden.require("input.context").expect("context"));
    let ctx = causal.inner().embed_text(&ctx_in).expect("embed text");
    let cross_kv = causal.prepare_cross_kv(&ctx).expect("cross kv");
    (causal, cross_kv, cfg)
}

/// (1) Mask-RULE anchor: our additive block-causal mask must have exactly the reference
/// `get_sdpa_mask` allow/deny pattern — `mask = (kv_idx < ends[q_idx]) | (q_idx == kv_idx)`.
#[test]
fn mask_rule_matches_reference_get_sdpa_mask() {
    let golden = Weights::from_file(GOLDEN).expect("load causal AR golden");
    let want = f32a(golden.require("output.sdpa_mask").expect("sdpa mask")); // 1.0 allow / 0.0 deny
    assert_eq!(want.shape(), &[TOTAL_TOKENS as i32, TOTAL_TOKENS as i32]);

    let pos: Vec<i64> = (0..TOTAL_TOKENS as i64).collect();
    let got = f32a(&build_block_causal_mask(&pos, &pos, BLOCK_TOKENS).unwrap());

    let want_v: Vec<f32> = want.as_slice::<f32>().to_vec();
    let got_v: Vec<f32> = got.as_slice::<f32>().to_vec();
    let mut mismatches = 0usize;
    let mut allowed = 0usize;
    for (i, (w, g)) in want_v.iter().zip(got_v.iter()).enumerate() {
        let ref_allows = *w > 0.5;
        let we_allow = g.is_finite(); // 0.0 allowed, -inf masked
        if ref_allows {
            allowed += 1;
        }
        if ref_allows != we_allow {
            if mismatches < 8 {
                let (q, kv) = (i / TOTAL_TOKENS, i % TOTAL_TOKENS);
                eprintln!(
                    "mask mismatch at (q={q}, kv={kv}): reference={ref_allows} ours={we_allow}"
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "block-causal mask rule differs from the reference `get_sdpa_mask`"
    );

    // Sanity that the reference matrix really is block-causal and not degenerate: block b sees
    // (b+1)·BLOCK_TOKENS keys for each of its BLOCK_TOKENS queries.
    let expect_allowed: usize = (1..=NUM_CHUNKS)
        .map(|b| b * BLOCK_TOKENS * BLOCK_TOKENS)
        .sum();
    assert_eq!(allowed, expect_allowed);

    // NEGATIVE CONTROL on the rule itself: plain token-level causality (`kv <= q`) is a different,
    // plausible-looking misreading of "block causal" — it must NOT reproduce the reference matrix.
    let mut wrong = 0usize;
    for q in 0..TOTAL_TOKENS {
        for kv in 0..TOTAL_TOKENS {
            if (want_v[q * TOTAL_TOKENS + kv] > 0.5) != (kv <= q) {
                wrong += 1;
            }
        }
    }
    assert!(
        wrong > 0,
        "the reference mask is indistinguishable from plain token causality — the golden geometry \
         is not discriminating (needs block_size > 1 and ≥2 blocks)"
    );
    println!(
        "mask rule: {allowed} allowed entries; a token-causal misreading differs in {wrong} entries"
    );
}

/// (2) The ≥2-chunk numeric anchor against the reference `CausalWanModel._forward_inference` run with
/// a populated KV cache.
#[test]
fn multi_chunk_causal_forward_matches_torch_golden() {
    let golden = Weights::from_file(GOLDEN).expect("load causal AR golden");
    let (causal, cross_kv, cfg) = setup(&golden);
    let t = f32a(golden.require("input.timestep").unwrap()).item::<f32>();
    let shape = [
        cfg.wan.out_dim as i32,
        FRAMES_PER_BLOCK as i32,
        LATENT_HW,
        LATENT_HW,
    ];

    let mut cache = causal.new_cache();
    for i in 0..NUM_CHUNKS {
        let chunk = f32a(golden.require(&format!("input.latent_chunk{i}")).unwrap());
        let want = golden.require(&format!("output.chunk{i}")).unwrap();
        let start = i * BLOCK_TOKENS;
        let got = causal
            .forward_chunk(&chunk, t, &cross_kv, start, &mut cache)
            .unwrap_or_else(|e| panic!("chunk {i} (current_start = {start}): {e}"));
        assert_eq!(cache.stored_tokens(), start + BLOCK_TOKENS);
        assert_eq!(got.shape(), shape, "chunk {i} shape");

        let (m, p) = (mean_rel(&got, want), peak_rel(&got, want));
        println!(
            "krea causal AR torch golden: chunk{i} (start={start}) mean_rel={m:.6} peak_rel={p:.6}"
        );
        // A mask/RoPE-interpretation error lands ~an order of magnitude above this — see
        // `wrong_rope_offset_fails_the_torch_golden`, which measures the gap on this same fixture.
        assert!(
            m < MEAN_REL_GATE,
            "chunk{i} mean_rel {m} is far above the bf16-vs-f32 floor — that is an interpretation \
             error, not rounding"
        );
        assert!(
            p < PEAK_REL_GATE,
            "chunk{i} peak_rel {p} above the bf16 precision ceiling"
        );
    }
}

/// (3) NEGATIVE CONTROL — the deliverable that makes the golden worth having. Reproduce the DEEPEST
/// chunk with the causal-RoPE `start_frame` offset deliberately zeroed (the single most likely
/// reference-interpretation error: reading `causal_rope_apply` as stock `rope_apply`) and assert the
/// golden comparison goes RED, by a wide margin over the correct run.
#[test]
fn wrong_rope_offset_fails_the_torch_golden() {
    let golden = Weights::from_file(GOLDEN).expect("load causal AR golden");
    let (causal, cross_kv, cfg) = setup(&golden);
    let t = f32a(golden.require("input.timestep").unwrap()).item::<f32>();
    let chunks: Vec<Array> = (0..NUM_CHUNKS)
        .map(|i| f32a(golden.require(&format!("input.latent_chunk{i}")).unwrap()))
        .collect();
    let last = NUM_CHUNKS - 1;
    let start = last * BLOCK_TOKENS;
    let want = golden.require(&format!("output.chunk{last}")).unwrap();

    // Correct run, for the margin.
    let mut cache_ok = causal.new_cache();
    let mut good = None;
    for (i, chunk) in chunks.iter().enumerate() {
        good = Some(
            causal
                .forward_chunk(chunk, t, &cross_kv, i * BLOCK_TOKENS, &mut cache_ok)
                .expect("correct run"),
        );
    }
    let good_mean = mean_rel(good.as_ref().unwrap(), want);

    // Perturbed run: same weights, same inputs, same cache contents — the ONLY difference is the
    // causal RoPE frame offset for the final chunk (0 instead of `start / FRAME_SEQ_LENGTH`).
    let dit = causal.inner();
    let mut cache_w = causal.new_cache();
    for (i, chunk) in chunks.iter().take(last).enumerate() {
        causal
            .forward_chunk(chunk, t, &cross_kv, i * BLOCK_TOKENS, &mut cache_w)
            .expect("control prefix");
    }
    let (tok, grid) = dit.patch_embed_tokens(&chunks[last]).unwrap();
    let (cos_wrong, sin_wrong) = dit.prepare_rope(grid).unwrap(); // offset 0 — WRONG
    let (prev_kv, _) = cache_w.window_prev(BLOCK_TOKENS).unwrap();
    let (vel_wrong, _) = dit
        .forward_causal_chunk(&tok, t, &cross_kv, &cos_wrong, &sin_wrong, &prev_kv, None)
        .unwrap();
    let bad = mlx_gen_wan::patchify::unpatchify(
        &vel_wrong
            .reshape(&[BLOCK_TOKENS as i32, vel_wrong.shape()[2]])
            .unwrap(),
        grid,
        cfg.wan.out_dim,
        cfg.wan.patch_size,
    )
    .unwrap();
    let bad_mean = mean_rel(&bad, want);

    println!(
        "negative control (RoPE start_frame 0 instead of {}): mean_rel {bad_mean:.6} vs correct \
         {good_mean:.6}  ({:.1}x, gate {MEAN_REL_GATE:.1e})",
        start / FRAME_SEQ_LENGTH,
        bad_mean / good_mean.max(1e-12)
    );
    assert!(
        bad_mean > MEAN_REL_GATE,
        "the wrong-RoPE-offset run ({bad_mean}) still passes the golden's tolerance \
         ({MEAN_REL_GATE}) — the golden is NOT discriminating and must be tightened or regenerated \
         at a larger geometry"
    );
    assert!(
        bad_mean > 10.0 * good_mean,
        "wrong-offset divergence ({bad_mean}) is not clearly separated from the correct run \
         ({good_mean})"
    );
}
