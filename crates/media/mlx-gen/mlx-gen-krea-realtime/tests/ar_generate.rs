//! Krea Realtime 14B **Self-Forcing few-step AR chunk loop** verification (sc-8437, S4) — torch-free,
//! tiny random-weight fixtures (no real 28.58 GB checkpoint, OOM-safe).
//!
//! The scheduler math (shifted-sigma table + timestep→sigma lookup, the Euler `x0` step, the renoise)
//! is pinned by hand-computed unit tests in `src/scheduler.rs`; the loop structure (forward count =
//! chunks × steps, `is_final` once per chunk, seed determinism, loop-level KV threading) by
//! weight-free unit tests in `src/generate.rs`. This file drives the loop through the **real** S3
//! causal transformer on a tiny geometry to pin the end-to-end integration:
//!
//!   * correct output latent shape `[out_dim, num_frames, H, W]` and chunk count `= ceil(frames / fpb)`;
//!   * **determinism** — same seed ⇒ identical latents, a different seed ⇒ different;
//!   * **KV threading** — the persistent cache grows by exactly one chunk per chunk to
//!     `num_frames · frame_seq_length` (a fresh-per-chunk cache would end at one chunk's tokens, and
//!     `forward_chunk`'s `start == stored_tokens` precondition would reject a mis-threaded chunk);
//!   * a geometry mismatch and an empty-schedule/steps misuse are rejected.
//!
//! Latent-sequence *coherence* against the real weights is validated downstream (S6 decode / S13
//! real-weight e2e); S4 asserts schedule-correctness + loop structure + determinism.

use std::collections::HashMap;

use mlx_gen::{CancelFlag, Progress};
use mlx_gen_krea_realtime::{
    generate_i2v_latents, generate_latents, generate_latents_conditioned_into,
    generate_latents_into, generate_v2v_latents, load_krea_realtime_transformer, ArGenParams,
    CausalKreaTransformer, KreaRealtimeConfig, RefConditioning,
};
use mlx_rs::{Array, Dtype};

/// A never-cancelled flag for the deterministic / structure integration tests (per-step cancel is
/// exercised weight-free at the loop level in `src/generate.rs`; sc-8441 S8).
fn no_cancel() -> CancelFlag {
    CancelFlag::new()
}

/// A no-op progress sink for the integration tests that do not assert on progress.
fn sink() -> impl FnMut(Progress) {
    |_| {}
}

/// A tiny but structurally-complete Krea Realtime / Wan-2.1 geometry (dim 64, 2 heads → head_dim 32,
/// 2 layers), with a small AR block geometry: `num_frames_per_block = 2`, `frame_seq_length = 4`
/// (h·w = 2·2 for a 4×4 latent under patch 2×2) ⇒ block size 8 tokens; global attention.
fn tiny_cfg() -> KreaRealtimeConfig {
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
    c.ar.num_frames_per_block = 2;
    c.ar.frame_seq_length = 4; // h·w for a 4×4 latent
    c.ar.local_attn_size = -1; // global
    c.ar.sink_size = 0;
    c.ar.seq_length = 100_000; // global read window keeps the whole history
    c
}

/// Deterministic (seedless, reproducible) fill: `bias + scale · U(-0.5, 0.5)` via xorshift64. F16 to
/// match the native Krea checkpoint dtype (cast to bf16 on load).
fn det_fill(shape: &[i32], seed: u64, scale: f32, bias: f32) -> Array {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let mut s = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        data.push(bias + scale * (u as f32 - 0.5));
    }
    Array::from_slice(&data, shape)
        .as_dtype(Dtype::Float16)
        .unwrap()
}

/// Build the full **native** (pre-sanitize) Wan-2.1 transformer inventory for the tiny geometry with
/// deterministic random weights (mirrors `causal_forward.rs`).
fn native_random_map(cfg: &KreaRealtimeConfig) -> HashMap<String, Array> {
    let w = &cfg.wan;
    let dim = w.dim as i32;
    let ffn = w.ffn_dim as i32;
    let text_dim = w.text_dim as i32;
    let freq = w.freq_dim as i32;
    let (pt, ph, pw) = w.patch_size;
    let (pt, ph, pw) = (pt as i32, ph as i32, pw as i32);
    let head_out = w.out_dim as i32 * pt * ph * pw;

    let mut map: HashMap<String, Array> = HashMap::new();
    let mut seed = 1u64;
    let mut put =
        |map: &mut HashMap<String, Array>, name: &str, shape: &[i32], scale: f32, bias: f32| {
            seed += 1;
            map.insert(name.to_string(), det_fill(shape, seed, scale, bias));
        };

    put(
        &mut map,
        "patch_embedding.weight",
        &[dim, w.in_dim as i32, pt, ph, pw],
        0.1,
        0.0,
    );
    put(&mut map, "patch_embedding.bias", &[dim], 0.05, 0.0);
    put(
        &mut map,
        "text_embedding.0.weight",
        &[dim, text_dim],
        0.1,
        0.0,
    );
    put(&mut map, "text_embedding.0.bias", &[dim], 0.05, 0.0);
    put(&mut map, "text_embedding.2.weight", &[dim, dim], 0.1, 0.0);
    put(&mut map, "text_embedding.2.bias", &[dim], 0.05, 0.0);
    put(&mut map, "time_embedding.0.weight", &[dim, freq], 0.1, 0.0);
    put(&mut map, "time_embedding.0.bias", &[dim], 0.05, 0.0);
    put(&mut map, "time_embedding.2.weight", &[dim, dim], 0.1, 0.0);
    put(&mut map, "time_embedding.2.bias", &[dim], 0.05, 0.0);
    put(
        &mut map,
        "time_projection.1.weight",
        &[6 * dim, dim],
        0.1,
        0.0,
    );
    put(&mut map, "time_projection.1.bias", &[6 * dim], 0.05, 0.0);
    put(&mut map, "head.head.weight", &[head_out, dim], 0.1, 0.0);
    put(&mut map, "head.head.bias", &[head_out], 0.05, 0.0);
    put(&mut map, "head.modulation", &[1, 2, dim], 0.05, 0.0);
    put(&mut map, "freqs", &[1024, 32], 0.0, 0.0); // dropped on sanitize

    for i in 0..w.num_layers {
        let p = format!("blocks.{i}");
        put(
            &mut map,
            &format!("{p}.modulation"),
            &[1, 6, dim],
            0.05,
            0.0,
        );
        for attn in ["self_attn", "cross_attn"] {
            for proj in ["q", "k", "v", "o"] {
                put(
                    &mut map,
                    &format!("{p}.{attn}.{proj}.weight"),
                    &[dim, dim],
                    0.12,
                    0.0,
                );
                put(
                    &mut map,
                    &format!("{p}.{attn}.{proj}.bias"),
                    &[dim],
                    0.03,
                    0.0,
                );
            }
            put(
                &mut map,
                &format!("{p}.{attn}.norm_q.weight"),
                &[dim],
                0.1,
                1.0,
            );
            put(
                &mut map,
                &format!("{p}.{attn}.norm_k.weight"),
                &[dim],
                0.1,
                1.0,
            );
        }
        put(&mut map, &format!("{p}.norm3.weight"), &[dim], 0.1, 1.0);
        put(&mut map, &format!("{p}.norm3.bias"), &[dim], 0.03, 0.0);
        put(
            &mut map,
            &format!("{p}.ffn.0.weight"),
            &[ffn, dim],
            0.1,
            0.0,
        );
        put(&mut map, &format!("{p}.ffn.0.bias"), &[ffn], 0.03, 0.0);
        put(
            &mut map,
            &format!("{p}.ffn.2.weight"),
            &[dim, ffn],
            0.1,
            0.0,
        );
        put(&mut map, &format!("{p}.ffn.2.bias"), &[dim], 0.03, 0.0);
    }
    map
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let d = mlx_rs::ops::subtract(a, b).unwrap();
    mlx_rs::ops::max(mlx_rs::ops::abs(&d).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

/// Load the tiny DiT, wrap it causal, and build a `[text_len, text_dim]` (f32) UMT5-shaped context.
fn setup() -> (CausalKreaTransformer, KreaRealtimeConfig, Array) {
    let cfg = tiny_cfg();
    let dit = load_krea_realtime_transformer(native_random_map(&cfg), &cfg).expect("load tiny DiT");
    let causal = CausalKreaTransformer::new(dit, &cfg);
    let context = det_fill(
        &[cfg.wan.text_len as i32, cfg.wan.text_dim as i32],
        99,
        1.0,
        0.0,
    )
    .as_dtype(Dtype::Float32)
    .unwrap();
    (causal, cfg, context)
}

fn params(seed: u64, frames: usize, steps: Option<usize>) -> ArGenParams {
    ArGenParams {
        seed,
        steps,
        num_latent_frames: frames,
        latent_height: 4,
        latent_width: 4,
        fps: 16,
    }
}

/// Correct latent-sequence shape + seed determinism through the real causal forward.
#[test]
fn generate_latents_shape_and_determinism() {
    let (causal, cfg, ctx) = setup();

    // 4 latent frames, fpb 2 ⇒ 2 chunks.
    let a = generate_latents(
        &causal,
        &cfg,
        &ctx,
        &params(42, 4, None),
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert_eq!(
        a.shape(),
        &[cfg.wan.out_dim as i32, 4, 4, 4],
        "latent sequence is [out_dim, num_frames, H, W]"
    );
    assert_eq!(a.dtype(), Dtype::Float32);

    // Same seed ⇒ bit-identical.
    let b = generate_latents(
        &causal,
        &cfg,
        &ctx,
        &params(42, 4, None),
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert_eq!(max_abs_diff(&a, &b), 0.0, "same seed must be deterministic");

    // Different seed ⇒ different latents.
    let c = generate_latents(
        &causal,
        &cfg,
        &ctx,
        &params(43, 4, None),
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert!(
        max_abs_diff(&a, &c) > 0.0,
        "a different seed must change the latents"
    );
}

/// KV threading: the persistent cache grows by exactly one chunk per chunk, ending at
/// `num_frames · frame_seq_length` — both for a whole-block clip and a partial trailing block.
#[test]
fn generate_latents_threads_persistent_kv_cache() {
    let (causal, cfg, ctx) = setup();
    let fsl = cfg.ar.frame_seq_length;

    // 4 frames, fpb 2 ⇒ 2 full chunks.
    let mut cache = causal.new_cache();
    let out = generate_latents_into(
        &causal,
        &cfg,
        &ctx,
        &params(7, 4, None),
        &mut cache,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert_eq!(out.shape(), &[cfg.wan.out_dim as i32, 4, 4, 4]);
    assert_eq!(
        cache.stored_tokens(),
        4 * fsl,
        "cache must accumulate every chunk's committed tokens (KV threading)"
    );
    assert_eq!(cache.num_layers(), cfg.wan.num_layers);

    // 3 frames, fpb 2 ⇒ chunks [2, 1] (partial trailing block).
    let mut cache3 = causal.new_cache();
    let out3 = generate_latents_into(
        &causal,
        &cfg,
        &ctx,
        &params(7, 3, None),
        &mut cache3,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert_eq!(out3.shape(), &[cfg.wan.out_dim as i32, 3, 4, 4]);
    assert_eq!(cache3.stored_tokens(), 3 * fsl);

    // A non-empty cache is rejected (the driver owns the fresh-cache invariant).
    let err = generate_latents_into(
        &causal,
        &cfg,
        &ctx,
        &params(7, 4, None),
        &mut cache,
        &no_cancel(),
        &mut sink(),
    )
    .expect_err("a populated cache must be rejected");
    assert!(err.to_string().contains("empty"), "err: {err}");
}

/// A caller `steps` override changes the number of forwards and still produces the correct shape.
#[test]
fn generate_latents_honors_steps_override() {
    let (causal, cfg, ctx) = setup();
    let out = generate_latents(
        &causal,
        &cfg,
        &ctx,
        &params(11, 4, Some(3)),
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert_eq!(out.shape(), &[cfg.wan.out_dim as i32, 4, 4, 4]);

    // A different step count generally yields a different trajectory than the 5-step default.
    let def = generate_latents(
        &causal,
        &cfg,
        &ctx,
        &params(11, 4, None),
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert!(
        max_abs_diff(&out, &def) > 0.0,
        "a 3-step override should differ from the 5-step default"
    );

    // steps = 0 is rejected.
    assert!(generate_latents(
        &causal,
        &cfg,
        &ctx,
        &params(11, 4, Some(0)),
        &no_cancel(),
        &mut sink()
    )
    .is_err());
}

/// A latent geometry whose per-frame token count disagrees with `frame_seq_length` is rejected up
/// front (the S3 causal forward bakes `frame_seq_length` into its cache windowing / RoPE offset).
#[test]
fn generate_latents_rejects_geometry_mismatch() {
    let (causal, cfg, ctx) = setup();
    // 8×4 latent under patch 2×2 → (4·2) = 8 tokens/frame ≠ frame_seq_length 4.
    let bad = ArGenParams {
        seed: 1,
        steps: None,
        num_latent_frames: 2,
        latent_height: 8,
        latent_width: 4,
        fps: 16,
    };
    let err = generate_latents(&causal, &cfg, &ctx, &bad, &no_cancel(), &mut sink())
        .expect_err("mismatched per-frame token count must be rejected");
    assert!(err.to_string().contains("frame_seq_length"), "err: {err}");
}

/// S5 clean-context KV recompute (sc-8438), through the real tiny causal transformer. With recompute
/// **on** (the shipped default) the chunk's committed self-attention K/V is the clean-context rerun's
/// (on the clean `x0` at `context_noise`); with recompute **off** (the S4 baseline) it is the near-clean
/// final-denoise-step's. Both commit exactly once per chunk (identical committed token count), but the
/// committed K/V **differ** — proving the recompute genuinely commits the clean-context forward, not the
/// noisy final step. Same seed/model/context isolate the recompute as the only variable.
#[test]
fn recompute_commits_clean_context_kv_distinct_from_s4_baseline() {
    let (causal, cfg_on, ctx) = setup();
    assert!(
        cfg_on.ar.do_kv_recomp,
        "shipped reference default is recompute on"
    );
    let mut cfg_off = cfg_on.clone();
    cfg_off.ar.do_kv_recomp = false;

    let p = params(7, 2, None); // one chunk (num_frames == num_frames_per_block == 2)
    let fsl = cfg_on.ar.frame_seq_length;

    let mut cache_on = causal.new_cache();
    let _out_on = generate_latents_into(
        &causal,
        &cfg_on,
        &ctx,
        &p,
        &mut cache_on,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    let mut cache_off = causal.new_cache();
    let _out_off = generate_latents_into(
        &causal,
        &cfg_off,
        &ctx,
        &p,
        &mut cache_off,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();

    // Exactly one commit per chunk in both modes ⇒ identical committed token count.
    assert_eq!(cache_on.stored_tokens(), 2 * fsl);
    assert_eq!(
        cache_off.stored_tokens(),
        cache_on.stored_tokens(),
        "one commit per chunk regardless of recompute"
    );

    // The committed K differs: recompute-on stores the clean-context forward's K, recompute-off the
    // near-clean final-denoise-step's.
    let (k_on, _) = cache_on.layer_kv(0).expect("on cache populated");
    let (k_off, _) = cache_off.layer_kv(0).expect("off cache populated");
    assert_eq!(k_on.shape(), k_off.shape());
    assert!(
        max_abs_diff(k_on, k_off) > 0.0,
        "recompute must commit clean-context K/V distinct from the S4 near-clean-final-step K/V"
    );
}

/// Bounded (streaming) window (sc-8438): a clip several windows long must (a) generate the correct
/// latents, (b) keep physical KV storage bounded to the window (eviction — not the full clip), and
/// (c) read only the last `max_attention_size` tokens each chunk without OOB. The window unit is the
/// reference's `local_attn_size × frame_seq_length` (frames), and the AR loop drives the windowed read.
#[test]
fn bounded_window_long_clip_stays_bounded_no_oob() {
    let mut cfg = tiny_cfg();
    cfg.ar.local_attn_size = 3; // 3 frames ⇒ max_attention_size = 3 × frame_seq_length
    let fsl = cfg.ar.frame_seq_length; // 4
    let window = cfg.ar.max_attention_size();
    assert_eq!(
        window,
        3 * fsl,
        "reference unit: local_attn_size frames × frame_seq_length"
    );

    let dit = load_krea_realtime_transformer(native_random_map(&cfg), &cfg).expect("load tiny DiT");
    let causal = CausalKreaTransformer::new(dit, &cfg);
    let ctx = det_fill(
        &[cfg.wan.text_len as i32, cfg.wan.text_dim as i32],
        99,
        1.0,
        0.0,
    )
    .as_dtype(Dtype::Float32)
    .unwrap();

    // 8 frames, fpb 2 ⇒ 4 chunks; committed 8·fsl = 32 ≫ window 12.
    let mut cache = causal.new_cache();
    let out = generate_latents_into(
        &causal,
        &cfg,
        &ctx,
        &params(5, 8, None),
        &mut cache,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();

    assert_eq!(
        out.shape(),
        &[cfg.wan.out_dim as i32, 8, 4, 4],
        "correct latents despite windowed eviction"
    );
    assert_eq!(
        cache.stored_tokens(),
        8 * fsl,
        "global committed count keeps growing"
    );
    assert!(
        cache.retained_tokens() < 8 * fsl,
        "eviction must actually happen for a clip longer than the window"
    );
    assert!(
        cache.retained_tokens() <= window + cfg.ar.num_frames_per_block * fsl,
        "physical KV storage stays bounded to the window (+ at most one chunk): {} (clip {})",
        cache.retained_tokens(),
        8 * fsl
    );
}

// ── S7 i2v / v2v conditioning (sc-8440) ──────────────────────────────────────────────────────────

/// A clean (VAE-encoded) reference latent `[in_dim, frames, 4, 4]` (f32) — the stand-in for the still /
/// source clip the pipeline VAE-encodes before conditioning.
fn ref_latents(seed: u64, frames: i32) -> Array {
    det_fill(&[16, frames, 4, 4], seed, 1.0, 0.0)
        .as_dtype(Dtype::Float32)
        .unwrap()
}

fn frame_slice(latents: &Array, start: i32, count: i32) -> Array {
    let idx: Vec<i32> = (start..start + count).collect();
    latents
        .take_axis(Array::from_slice(&idx, &[count]), 1)
        .unwrap()
}

/// **i2v**: a VAE-encoded reference still **warms** the KV cache (its clean-context K/V is committed
/// before generation) and is prepended verbatim to the output, so the generated continuation differs
/// from a same-seed t2v run — the still actually influences generation (the first generated chunk
/// attends the warmed reference). Correct output shape; deterministic; a different still ⇒ a different
/// continuation.
#[test]
fn i2v_reference_warms_cache_and_conditions_generation() {
    let (causal, cfg, ctx) = setup();
    let reference = ref_latents(314, 1); // one still ⇒ F_ref = 1
    let reference_b = ref_latents(271, 1);
    let p = params(42, 2, None); // generate 2 frames

    let t2v = generate_latents(&causal, &cfg, &ctx, &p, &no_cancel(), &mut sink()).unwrap(); // [16, 2, 4, 4]
    let i2v = generate_i2v_latents(
        &causal,
        &cfg,
        &ctx,
        &p,
        &reference,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();

    // (b) shape: the reference frame is prepended to the generated continuation.
    assert_eq!(
        i2v.shape(),
        &[16, 3, 4, 4],
        "F_ref (1) + num_latent_frames (2)"
    );
    // The reference frame is copied verbatim into output frame 0.
    assert_eq!(
        max_abs_diff(&frame_slice(&i2v, 0, 1), &reference),
        0.0,
        "the reference still is prepended verbatim"
    );

    // (a) DISCRIMINATING: the generated continuation differs from a same-seed t2v run — the ONLY
    // difference between the two runs is the warmed reference cache, so a difference proves the first
    // generated chunk attends the reference (the still influences generation).
    let gen = frame_slice(&i2v, 1, 2);
    assert!(
        max_abs_diff(&gen, &t2v) > 1e-4,
        "the warmed reference must change the generated frames vs t2v"
    );

    // (c) determinism: same seed ⇒ identical i2v latents.
    let i2v_again = generate_i2v_latents(
        &causal,
        &cfg,
        &ctx,
        &p,
        &reference,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert_eq!(
        max_abs_diff(&i2v, &i2v_again),
        0.0,
        "same seed ⇒ identical i2v latents"
    );

    // A different reference ⇒ a different continuation (the still actually conditions generation).
    let i2v_b = generate_i2v_latents(
        &causal,
        &cfg,
        &ctx,
        &p,
        &reference_b,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert!(
        max_abs_diff(&gen, &frame_slice(&i2v_b, 1, 2)) > 1e-4,
        "a different reference still must change the generated continuation"
    );
}

/// **Cache-warm correctness**: the reference frames' clean-context K/V populate the cache *before*
/// generation — after `generate_latents_conditioned_into` the committed count is `(F_ref +
/// num_latent_frames) · frame_seq_length`, i.e. one commit per reference block (the warm) plus one per
/// generated block. Mirrors the S5 clean-context commit applied to the reference.
#[test]
fn i2v_cache_warm_commits_reference_before_generation() {
    let (causal, cfg, ctx) = setup();
    let reference = ref_latents(314, 1);
    let p = params(42, 2, None);
    let cond = RefConditioning {
        context_latents: Some(reference.clone()),
        source: None,
    };
    let mut cache = causal.new_cache();
    let out = generate_latents_conditioned_into(
        &causal,
        &cfg,
        &ctx,
        &p,
        &cond,
        &mut cache,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();

    // Reference block(s) committed by the warm + one commit per generated block.
    assert_eq!(
        cache.stored_tokens(),
        (1 + 2) * cfg.ar.frame_seq_length,
        "cache holds the warmed reference frame + the generated frames"
    );
    assert_eq!(out.shape(), &[16, 3, 4, 4]);
}

/// **v2v**: a VAE-encoded source clip drives a strength-controlled init. The source and the strength
/// both influence the output (different source ⇒ different output at a partial strength; strength=0 vs
/// strength=1 differ), the shape is correct, generation is deterministic, and at `strength = 1` the
/// source is fully washed out (source-independent — a full regeneration).
#[test]
fn v2v_source_and_strength_condition_generation() {
    let (causal, cfg, ctx) = setup();
    let source_a = ref_latents(11, 2); // one source frame per generated frame
    let source_b = ref_latents(22, 2);
    let p = params(5, 2, None);

    let v_a_half = generate_v2v_latents(
        &causal,
        &cfg,
        &ctx,
        &p,
        &source_a,
        0.5,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert_eq!(
        v_a_half.shape(),
        &[16, 2, 4, 4],
        "v2v output shape [C, num, H, W]"
    );

    // (a) a different source ⇒ a different output at a partial strength (the source conditions gen).
    let v_b_half = generate_v2v_latents(
        &causal,
        &cfg,
        &ctx,
        &p,
        &source_b,
        0.5,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert!(
        max_abs_diff(&v_a_half, &v_b_half) > 1e-4,
        "a different source must change v2v output at strength 0.5"
    );

    // (b) the strength lever has an effect: strength 0 (keep source) vs strength 1 (regenerate) differ.
    let v_a_0 = generate_v2v_latents(
        &causal,
        &cfg,
        &ctx,
        &p,
        &source_a,
        0.0,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    let v_a_1 = generate_v2v_latents(
        &causal,
        &cfg,
        &ctx,
        &p,
        &source_a,
        1.0,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert!(
        max_abs_diff(&v_a_0, &v_a_1) > 1e-4,
        "strength 0 vs strength 1 must differ"
    );

    // (c) determinism: same seed + strength ⇒ identical.
    let v_a_half_2 = generate_v2v_latents(
        &causal,
        &cfg,
        &ctx,
        &p,
        &source_a,
        0.5,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert_eq!(
        max_abs_diff(&v_a_half, &v_a_half_2),
        0.0,
        "same seed + strength ⇒ identical v2v latents"
    );

    // (d) strength = 1 washes out the source: the init is pure noise (σ ≈ 1), so different sources
    // yield the same output — a full regeneration that no longer tracks the source.
    let v_b_1 = generate_v2v_latents(
        &causal,
        &cfg,
        &ctx,
        &p,
        &source_b,
        1.0,
        &no_cancel(),
        &mut sink(),
    )
    .unwrap();
    assert_eq!(
        max_abs_diff(&v_a_1, &v_b_1),
        0.0,
        "strength = 1 is source-independent (the source is fully regenerated)"
    );

    // A geometry mismatch (source frame count ≠ num_latent_frames) is rejected.
    let bad_source = ref_latents(33, 3);
    assert!(
        generate_v2v_latents(
            &causal,
            &cfg,
            &ctx,
            &p,
            &bad_source,
            0.5,
            &no_cancel(),
            &mut sink()
        )
        .is_err(),
        "a source with the wrong frame count is rejected"
    );
}
