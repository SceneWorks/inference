//! Krea Realtime 14B **text-to-video pipeline** e2e verification (sc-8439, S6) — torch-free, tiny
//! random-weight fixtures (no real 28.58 GB checkpoint, OOM-safe).
//!
//! Drives the full non-gated pipeline seam end-to-end on a tiny config — **tiny UMT5 + tiny DiT + tiny
//! z16 VAE** — to pin the S6 integration the registered `Generator` runs on real weights:
//!
//!   * `prompt-context → [`generate_latents`] → z16 Wan VAE decode → assembled clip` produces a
//!     [`GenerationOutput::Video`] of the **correct shape** (frames = 4·num_latent_frames; each frame
//!     8·latent × 8·latent), and is **deterministic** given a seed (a different seed differs);
//!   * the reused z16 Wan VAE decode is **T → 4·T temporal, ×8 spatial** on a small latent;
//!   * the tiny **UMT5** text encoder feeds a real context into the pipeline (the `prompt → context`
//!     seam the real path runs through `encode_prompt`).
//!
//! The tiny VAE is the committed z16 fixture (`tests/fixtures/wan_z16_vae_tiny.safetensors`, dim=4 /
//! z16, mirrored from `mlx-gen-wan`'s `dump_s2_fixtures.py`). Pixel *content* is meaningless on random
//! weights — the gates are shape + determinism, exactly like the S4 AR-loop gates. The real-weight
//! watchable-clip e2e is the gated S6 remainder (overlaps S13).

use std::collections::HashMap;

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, GenerationOutput, Progress};
use mlx_gen_krea_realtime::{
    generate_t2v_from_components, ArGenParams, CausalKreaTransformer, KreaRealtimeConfig,
};
use mlx_gen_wan::WanVae;
use mlx_rs::{Array, Dtype};

// ── Tiny geometry (mirrors tests/ar_generate.rs) ────────────────────────────────────────────────

/// A tiny but structurally-complete Krea Realtime / Wan-2.1 geometry: dim 64, 2 heads, 2 DiT layers;
/// tiny UMT5 (text_dim 32, 2 T5 layers, 2 heads, 8 buckets); AR block geometry num_frames_per_block 2,
/// frame_seq_length 4 (2·2 tokens for a 4×4 latent under patch 2×2). Global attention.
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
    c.wan.out_dim = 16; // z16 latent — matches the z16 VAE fixture
    c.wan.patch_size = (1, 2, 2);
    // Tiny UMT5 dims.
    c.wan.t5_dim_attn = 32;
    c.wan.t5_num_heads = 2; // t5 head_dim 16
    c.wan.t5_num_layers = 2;
    c.wan.t5_num_buckets = 8;
    // Tiny AR block geometry.
    c.ar.num_frames_per_block = 2;
    c.ar.frame_seq_length = 4; // 2·2 tokens for a 4×4 latent
    c.ar.local_attn_size = -1; // global
    c.ar.sink_size = 0;
    c.ar.seq_length = 100_000;
    c
}

/// Deterministic fill: `bias + scale · U(-0.5, 0.5)` via xorshift64, at a chosen dtype.
fn det_fill(shape: &[i32], seed: u64, scale: f32, bias: f32, dtype: Dtype) -> Array {
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
    Array::from_slice(&data, shape).as_dtype(dtype).unwrap()
}

/// Full **native** (pre-sanitize) Wan-2.1 transformer inventory for the tiny geometry (F16, mirrors
/// ar_generate.rs / causal_forward.rs).
fn native_dit_map(cfg: &KreaRealtimeConfig) -> HashMap<String, Array> {
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
    // (name, shape, scale, bias) → F16 deterministic fill (native Krea dtype; cast to bf16 on load).
    let mut put =
        |map: &mut HashMap<String, Array>, name: &str, shape: &[i32], scale: f32, bias: f32| {
            seed += 1;
            map.insert(
                name.to_string(),
                det_fill(shape, seed, scale, bias, Dtype::Float16),
            );
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

/// A tiny UMT5-XXL text encoder (MLX-layout keys), dense random weights matching the real dtype
/// convention (bf16 projections/embedding; f32 norms/pos-bias). `t5_ffn` rides on the weights.
fn tiny_umt5(cfg: &KreaRealtimeConfig) -> Weights {
    let w = &cfg.wan;
    let text_dim = w.text_dim as i32;
    let dim_attn = w.t5_dim_attn as i32;
    let t5_ffn = 64i32;
    let vocab = 64i32;
    let mut map: HashMap<String, Array> = HashMap::new();
    let mut seed = 500u64;
    let mut put = |map: &mut HashMap<String, Array>, name: &str, shape: &[i32], dtype: Dtype| {
        seed += 1;
        // Norms centered at 1.0 (rms-norm gamma); projections/embedding centered at 0.
        let bias = if name.contains("norm") && !name.contains("pos") {
            1.0
        } else {
            0.0
        };
        map.insert(name.to_string(), det_fill(shape, seed, 0.1, bias, dtype));
    };

    put(
        &mut map,
        "token_embedding.weight",
        &[vocab, text_dim],
        Dtype::Bfloat16,
    );
    put(&mut map, "norm.weight", &[text_dim], Dtype::Float32);
    for i in 0..w.t5_num_layers {
        let p = format!("blocks.{i}");
        put(
            &mut map,
            &format!("{p}.norm1.weight"),
            &[text_dim],
            Dtype::Float32,
        );
        put(
            &mut map,
            &format!("{p}.norm2.weight"),
            &[text_dim],
            Dtype::Float32,
        );
        // q/k/v: [dim_attn, text_dim]; o: [text_dim, dim_attn].
        put(
            &mut map,
            &format!("{p}.attn.q.weight"),
            &[dim_attn, text_dim],
            Dtype::Bfloat16,
        );
        put(
            &mut map,
            &format!("{p}.attn.k.weight"),
            &[dim_attn, text_dim],
            Dtype::Bfloat16,
        );
        put(
            &mut map,
            &format!("{p}.attn.v.weight"),
            &[dim_attn, text_dim],
            Dtype::Bfloat16,
        );
        put(
            &mut map,
            &format!("{p}.attn.o.weight"),
            &[text_dim, dim_attn],
            Dtype::Bfloat16,
        );
        put(
            &mut map,
            &format!("{p}.ffn.gate_proj.weight"),
            &[t5_ffn, text_dim],
            Dtype::Bfloat16,
        );
        put(
            &mut map,
            &format!("{p}.ffn.fc1.weight"),
            &[t5_ffn, text_dim],
            Dtype::Bfloat16,
        );
        put(
            &mut map,
            &format!("{p}.ffn.fc2.weight"),
            &[text_dim, t5_ffn],
            Dtype::Bfloat16,
        );
        put(
            &mut map,
            &format!("{p}.pos_embedding.embedding.weight"),
            &[w.t5_num_buckets as i32, w.t5_num_heads as i32],
            Dtype::Float32,
        );
    }
    Weights::from_map(map)
}

fn tiny_vae() -> WanVae {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/wan_z16_vae_tiny.safetensors"
    );
    let w = Weights::from_file(path).expect("read tiny z16 VAE fixture");
    WanVae::from_weights(&w).expect("build tiny WanVae")
}

/// Encode a short prompt-token sequence through the tiny UMT5 → context `[L, text_dim]` (f32) — the
/// `prompt → context` seam the real `encode_prompt` runs through the tokenizer.
fn tiny_context(cfg: &KreaRealtimeConfig) -> Array {
    use mlx_gen_wan::Umt5Encoder;
    let enc = Umt5Encoder::from_weights(&tiny_umt5(cfg), &cfg.wan).expect("build tiny UMT5");
    let l = 5i32;
    let ids = Array::from_slice(&[1i32, 2, 3, 4, 5], &[1, l]);
    let mask = Array::from_slice(&[1i32, 1, 1, 1, 1], &[1, l]);
    let hidden = enc.forward(&ids, &mask).expect("umt5 forward"); // [1, L, text_dim]
    let dim = hidden.shape()[2];
    hidden
        .reshape(&[l, dim])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
}

fn params(seed: u64, frames: usize) -> ArGenParams {
    ArGenParams {
        seed,
        steps: None,
        num_latent_frames: frames,
        latent_height: 4,
        latent_width: 4,
        fps: 16,
    }
}

fn video_frames(out: &GenerationOutput) -> &Vec<mlx_gen::Image> {
    match out {
        GenerationOutput::Video { frames, .. } => frames,
        other => panic!("expected Video output, got {other:?}"),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────────────────────────

/// Full pipeline e2e: tiny UMT5 context → tiny DiT AR latents → tiny z16 VAE decode → assembled clip
/// of the correct shape (frames = 4·num_latent_frames; each frame 8·latent × 8·latent).
#[test]
fn t2v_pipeline_produces_a_clip_of_correct_shape() {
    let cfg = tiny_cfg();
    let dit = mlx_gen_krea_realtime::load_krea_realtime_transformer(native_dit_map(&cfg), &cfg)
        .expect("load tiny DiT");
    let transformer = CausalKreaTransformer::new(dit, &cfg);
    let vae = tiny_vae();
    let context = tiny_context(&cfg);
    let cancel = CancelFlag::default();
    let mut saw_decoding = false;
    let mut on_progress = |p: Progress| {
        if p == Progress::Decoding {
            saw_decoding = true;
        }
    };

    // 4 latent frames, fpb 2 ⇒ 2 chunks. z16 VAE: T→4·T temporal, ×8 spatial.
    let out = generate_t2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &params(42, 4),
        None,
        &cancel,
        &mut on_progress,
    )
    .expect("t2v pipeline");

    let frames = video_frames(&out);
    assert_eq!(frames.len(), 4 * 4, "4·num_latent_frames output frames");
    for f in frames {
        assert_eq!(
            (f.width, f.height),
            (32, 32),
            "8× spatial upsample of a 4×4 latent"
        );
        assert_eq!(f.pixels.len(), 32 * 32 * 3);
    }
    match &out {
        GenerationOutput::Video { fps, audio, .. } => {
            assert_eq!(*fps, 16);
            assert!(audio.is_none(), "t2v has no audio track");
        }
        _ => unreachable!(),
    }
    assert!(
        saw_decoding,
        "the pipeline must emit Progress::Decoding before the VAE decode"
    );
}

/// Determinism: identical seed ⇒ byte-identical clip; a different seed ⇒ a different clip.
#[test]
fn t2v_pipeline_is_seed_deterministic() {
    let cfg = tiny_cfg();
    let dit = mlx_gen_krea_realtime::load_krea_realtime_transformer(native_dit_map(&cfg), &cfg)
        .expect("load tiny DiT");
    let transformer = CausalKreaTransformer::new(dit, &cfg);
    let vae = tiny_vae();
    let context = tiny_context(&cfg);
    let cancel = CancelFlag::default();
    let mut noop = |_: Progress| {};

    let run = |seed: u64,
               t: &CausalKreaTransformer,
               v: &WanVae,
               ctx: &Array,
               c: &CancelFlag,
               np: &mut dyn FnMut(Progress)| {
        let out =
            generate_t2v_from_components(t, &cfg, v, ctx, &params(seed, 2), None, c, np).unwrap();
        match out {
            GenerationOutput::Video { frames, .. } => frames,
            _ => unreachable!(),
        }
    };

    let a = run(7, &transformer, &vae, &context, &cancel, &mut noop);
    let b = run(7, &transformer, &vae, &context, &cancel, &mut noop);
    let c = run(8, &transformer, &vae, &context, &cancel, &mut noop);

    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    for (fa, fb) in a.iter().zip(&b) {
        assert_eq!(fa.pixels, fb.pixels, "same seed must be byte-identical");
    }
    assert!(
        a.iter().zip(&c).any(|(fa, fc)| fa.pixels != fc.pixels),
        "a different seed must change the clip"
    );
}

/// The reused z16 Wan VAE decode is **T → 4·T temporal, ×8 spatial** on a small latent, and the
/// pipeline's decode+assemble seam surfaces exactly that geometry as a clip.
#[test]
fn vae_decode_is_4x_temporal_8x_spatial() {
    use mlx_gen_krea_realtime::decode_latents_to_video;
    let vae = tiny_vae();
    let cancel = CancelFlag::default();
    // A [z16, T_lat=3, 5, 6] latent → [4·3, 8·5, 8·6, 3] = 12 frames of 48×40.
    let latents = det_fill(&[16, 3, 5, 6], 123, 1.0, 0.0, Dtype::Float32);
    let out = decode_latents_to_video(&vae, &latents, 24, None, &cancel).unwrap();
    let frames = match &out {
        GenerationOutput::Video { frames, fps, .. } => {
            assert_eq!(*fps, 24);
            frames
        }
        _ => unreachable!(),
    };
    assert_eq!(frames.len(), 4 * 3, "T → 4·T temporal upsample");
    assert_eq!(
        (frames[0].width, frames[0].height),
        (6 * 8, 5 * 8),
        "×8 spatial upsample (W=6·8, H=5·8)"
    );
}
