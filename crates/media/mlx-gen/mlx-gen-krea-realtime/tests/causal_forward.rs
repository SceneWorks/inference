//! Krea Realtime 14B **causal AR forward + KV cache** verification (sc-8436, S3) — torch-free, tiny
//! random-weight fixtures (no real 28.58 GB checkpoint, OOM-safe).
//!
//! The DoD is "the forward produces correct shapes + matches the reference on a fixture chunk". With
//! no torch env in this Rust repo (and `mlx-gen-wan` carries no vendored torch-golden harness for the
//! attention block — a causal golden is filed as a follow-up), the causal path is pinned by three
//! torch-free checks that make a broken mask / mis-offset RoPE / wrong cache slice FAIL a specific
//! assertion:
//!
//!   * **(A) KV-cache-correctness invariant** — an incremental cached forward (chunk 0 populates the
//!     cache, chunk 1 attends cached chunk 0 + itself) must equal a single full-sequence forward over
//!     `[chunk0 ‖ chunk1]` under the block-causal additive mask (full recompute). Validates the cache
//!     append/read AND the causal RoPE global-frame offset. A negative control (chunk 1 with a WRONG
//!     zero offset) must diverge far past tolerance, proving the check discriminates.
//!   * **(B) Single-chunk equivalence anchor** — one chunk with an empty cache and no mask (a single
//!     block is fully bidirectional) must equal the stock non-causal
//!     [`mlx_gen_wan::WanTransformer::forward_packed`] over the same chunk, anchoring the causal path to
//!     the already-validated Wan attention.
//!   * **(C) Explicit mask-rule unit test** lives in `src/causal.rs` (hand-derived [4,4] matrix); the
//!     RoPE tail-equivalence unit test lives in `mlx-gen-wan/src/rope.rs`.
//!
//! Shapes: a chunk forward returns `[out_dim, F_chunk, H, W]`; the cache grows by the chunk token count.

use std::collections::HashMap;

use mlx_gen_krea_realtime::{
    build_block_causal_mask, load_krea_realtime_transformer, CausalKreaTransformer,
    KreaRealtimeConfig,
};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

/// A tiny but structurally-complete Krea Realtime / Wan-2.1 geometry (dim 64, 2 heads → head_dim 32,
/// 2 layers), with a small AR block geometry: `num_frames_per_block = 2`, `frame_seq_length = 4`
/// (h·w = 2·2) ⇒ block size 8 tokens; global attention (`local_attn_size = -1`).
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
    c.ar.frame_seq_length = 4; // h·w
    c.ar.local_attn_size = -1; // global
    c.ar.sink_size = 0;
    c.ar.seq_length = 100_000; // global read window keeps the whole history
    c
}

/// Deterministic (seedless, reproducible) fill: `bias + scale · U(-0.5, 0.5)` via xorshift64, so the
/// tests carry no RNG-flakiness. F16 to match the native Krea checkpoint dtype (cast to bf16 on load).
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
        let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        data.push(bias + scale * (u as f32 - 0.5));
    }
    Array::from_slice(&data, shape)
        .as_dtype(Dtype::Float16)
        .unwrap()
}

/// Build the full **native** (pre-sanitize) Wan-2.1 transformer inventory for the tiny geometry with
/// deterministic random weights. Norm gains (`norm_q`/`norm_k`) sit near 1.0 so self-attention is
/// non-trivial (discriminating); every other weight is small so the residual stream stays bounded.
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
            // Norm gains near 1.0 ⇒ meaningful attention logits.
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

/// A deterministic f32 latent chunk `[C, F, H, W]`.
fn det_latent(c: i32, f: i32, h: i32, w: i32, seed: u64) -> Array {
    let n = (c * f * h * w) as usize;
    let mut s = seed.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(7);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        data.push((u as f32 - 0.5) * 2.0); // ~[-1, 1]
    }
    Array::from_slice(&data, &[c, f, h, w])
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let d = mlx_rs::ops::subtract(a, b).unwrap();
    let ad = mlx_rs::ops::abs(&d).unwrap();
    mlx_rs::ops::max(&ad, None).unwrap().item::<f32>()
}

fn max_abs(a: &Array) -> f32 {
    mlx_rs::ops::max(mlx_rs::ops::abs(a).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

/// Load the tiny DiT + wrap it in the causal transformer, and build the per-prompt cross-attention KV.
fn setup() -> (
    CausalKreaTransformer,
    Vec<(Array, Array)>,
    KreaRealtimeConfig,
) {
    let cfg = tiny_cfg();
    let dit = load_krea_realtime_transformer(native_random_map(&cfg), &cfg).expect("load tiny DiT");
    let causal = CausalKreaTransformer::new(dit, &cfg);
    // Text context: [text_len, text_dim] → embed → [1, text_len, dim] → per-block cross K/V.
    let t5 = det_latent(cfg.wan.text_len as i32, cfg.wan.text_dim as i32, 1, 1, 99)
        .reshape(&[cfg.wan.text_len as i32, cfg.wan.text_dim as i32])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let ctx = causal.inner().embed_text(&t5).expect("embed text");
    let cross_kv = causal.prepare_cross_kv(&ctx).expect("cross kv");
    (causal, cross_kv, cfg)
}

/// (A) Incremental cached AR forward == single full-sequence forward under the block-causal mask.
#[test]
fn cached_incremental_matches_full_recompute() {
    let (causal, cross_kv, cfg) = setup();
    let c = cfg.wan.in_dim as i32;
    let t = 500.0f32;

    // Two chunks of `num_frames_per_block` frames each; the full clip is their frame-axis concat.
    let fpb = cfg.ar.num_frames_per_block as i32;
    let chunk0 = det_latent(c, fpb, 4, 4, 11);
    let chunk1 = det_latent(c, fpb, 4, 4, 22);
    let full = concatenate_axis(&[&chunk0, &chunk1], 1).unwrap(); // [C, 2·fpb, 4, 4]

    // --- Full recompute: one pass over the whole clip under the [2-block] block-causal mask. ---
    let dit = causal.inner();
    let (tok_full, grid_full) = dit.patch_embed_tokens(&full).unwrap();
    let l = (grid_full.0 * grid_full.1 * grid_full.2) as i64;
    let (cos_f, sin_f) = dit.prepare_rope(grid_full).unwrap(); // zero offset over the whole clip
    let q_full: Vec<i64> = (0..l).collect();
    let mask_full = build_block_causal_mask(&q_full, &q_full, causal.block_size()).unwrap();
    let (vel_full, _) = dit
        .forward_causal_chunk(
            &tok_full,
            t,
            &cross_kv,
            &cos_f,
            &sin_f,
            &[],
            Some(&mask_full),
        )
        .unwrap();
    let op = vel_full.shape()[2];
    let vel_full_latent = mlx_gen_wan::patchify::unpatchify(
        &vel_full.reshape(&[l as i32, op]).unwrap(),
        grid_full,
        cfg.wan.out_dim,
        cfg.wan.patch_size,
    )
    .unwrap();

    // --- Incremental: chunk 0 fills the cache, chunk 1 reads it + itself. ---
    let mut cache = causal.new_cache();
    let s_chunk = cfg.ar.num_frames_per_block * cfg.ar.frame_seq_length;
    let out0 = causal
        .forward_chunk(&chunk0, t, &cross_kv, 0, &mut cache)
        .unwrap();
    assert_eq!(
        cache.stored_tokens(),
        s_chunk,
        "cache grows by one chunk after step 0"
    );
    let out1 = causal
        .forward_chunk(&chunk1, t, &cross_kv, s_chunk, &mut cache)
        .unwrap();
    assert_eq!(
        cache.stored_tokens(),
        2 * s_chunk,
        "cache grows by one chunk after step 1"
    );
    let vel_inc_latent = concatenate_axis(&[&out0, &out1], 1).unwrap();

    assert_eq!(vel_inc_latent.shape(), vel_full_latent.shape());
    let scale = max_abs(&vel_full_latent).max(1e-6);
    let d = max_abs_diff(&vel_full_latent, &vel_inc_latent);
    assert!(
        d < 5e-2 * scale.max(1.0) || d < 5e-2,
        "cached incremental must match full recompute: max|Δ|={d} (signal ~{scale})"
    );

    // --- Negative control: chunk 1 with a WRONG (zero) RoPE offset must diverge far past tolerance. ---
    let mut cache_w = causal.new_cache();
    let _ = causal
        .forward_chunk(&chunk0, t, &cross_kv, 0, &mut cache_w)
        .unwrap();
    let (tok1, grid1) = dit.patch_embed_tokens(&chunk1).unwrap();
    let (cos_wrong, sin_wrong) = dit.prepare_rope(grid1).unwrap(); // offset 0 (WRONG: should be start_frame=fpb)
    let (prev_kv, _) = cache_w.window_prev(s_chunk).unwrap();
    let (vel1_wrong, _) = dit
        .forward_causal_chunk(&tok1, t, &cross_kv, &cos_wrong, &sin_wrong, &prev_kv, None)
        .unwrap();
    let out1_wrong = mlx_gen_wan::patchify::unpatchify(
        &vel1_wrong
            .reshape(&[s_chunk as i32, vel1_wrong.shape()[2]])
            .unwrap(),
        grid1,
        cfg.wan.out_dim,
        cfg.wan.patch_size,
    )
    .unwrap();
    let d_wrong = max_abs_diff(&out1, &out1_wrong);
    assert!(
        d_wrong > 10.0 * d.max(1e-6),
        "a zero RoPE offset must diverge from the correct offset (control d_wrong={d_wrong} vs d={d})"
    );
}

/// (B) Single chunk, empty cache, no mask == stock non-causal Wan `forward_packed` (the anchor).
#[test]
fn single_chunk_equals_stock_wan_forward() {
    let (causal, cross_kv, cfg) = setup();
    let c = cfg.wan.in_dim as i32;
    let t = 500.0f32;
    let fpb = cfg.ar.num_frames_per_block as i32;
    let chunk = det_latent(c, fpb, 4, 4, 33);

    let dit = causal.inner();
    let (tok, grid) = dit.patch_embed_tokens(&chunk).unwrap();
    let (cos, sin) = dit.prepare_rope(grid).unwrap();

    // Stock non-causal Wan over the same tokens (full bidirectional).
    let vel_stock = dit.forward_packed(&tok, t, &cross_kv, &cos, &sin).unwrap();

    // Causal path, empty cache + no mask (one block is fully bidirectional).
    let (vel_causal, new_kv) = dit
        .forward_causal_chunk(&tok, t, &cross_kv, &cos, &sin, &[], None)
        .unwrap();

    let d = max_abs_diff(&vel_stock, &vel_causal);
    assert!(
        d < 1e-4,
        "single-block causal must equal stock Wan forward_packed: max|Δ|={d}"
    );

    // The cache returns one (k, v) per layer, each [B, n, S, head_dim].
    assert_eq!(new_kv.len(), cfg.wan.num_layers);
    let s = (grid.0 * grid.1 * grid.2) as i32;
    let n = cfg.wan.num_heads as i32;
    let hd = (cfg.wan.dim / cfg.wan.num_heads) as i32;
    for (k, v) in &new_kv {
        assert_eq!(k.shape(), &[1, n, s, hd]);
        assert_eq!(v.shape(), &[1, n, s, hd]);
    }
}

/// Shape sanity: a chunk forward returns `[out_dim, F_chunk, H, W]` and the cache grows per chunk.
#[test]
fn forward_chunk_shapes_and_cache_growth() {
    let (causal, cross_kv, cfg) = setup();
    let c = cfg.wan.in_dim as i32;
    let fpb = cfg.ar.num_frames_per_block as i32;
    let mut cache = causal.new_cache();
    assert!(cache.is_empty());
    let s_chunk = cfg.ar.num_frames_per_block * cfg.ar.frame_seq_length;

    for step in 0..3 {
        let chunk = det_latent(c, fpb, 4, 4, 100 + step as u64);
        let out = causal
            .forward_chunk(&chunk, 500.0, &cross_kv, step * s_chunk, &mut cache)
            .unwrap();
        assert_eq!(out.shape(), &[cfg.wan.out_dim as i32, fpb, 4, 4]);
        assert_eq!(cache.stored_tokens(), (step + 1) * s_chunk);
    }
    assert_eq!(cache.num_layers(), cfg.wan.num_layers);
}

/// A mis-positioned chunk (cache/start disagreement, or non-frame-aligned start) is rejected.
#[test]
fn forward_chunk_rejects_bad_start() {
    let (causal, cross_kv, cfg) = setup();
    let c = cfg.wan.in_dim as i32;
    let fpb = cfg.ar.num_frames_per_block as i32;
    let chunk = det_latent(c, fpb, 4, 4, 55);
    let mut cache = causal.new_cache();
    // Start token disagrees with the (empty) cache.
    let err = causal
        .forward_chunk(&chunk, 500.0, &cross_kv, 8, &mut cache)
        .expect_err("start must equal cache stored tokens");
    assert!(err.to_string().contains("stored"), "err: {err}");
}
