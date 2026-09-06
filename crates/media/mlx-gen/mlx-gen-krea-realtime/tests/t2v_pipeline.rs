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

use mlx_gen::gen_core::MemoryPhase;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, GenerationOutput, Image, Progress};
use mlx_gen_krea_realtime::{
    generate_i2v_from_components, generate_t2v_from_components, generate_v2v_from_components,
    ArGenParams, CausalKreaTransformer, KreaRealtimeConfig,
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
        memory: Default::default(),
    }
}

fn video_frames(out: &GenerationOutput) -> &Vec<mlx_gen::Image> {
    match out {
        GenerationOutput::Video { frames, .. } => frames,
        other => panic!("expected Video output, got {other:?}"),
    }
}

/// A tiny-config `params` with a calibration fault selected for `phase`, authorized or not.
fn fault_params(seed: u64, frames: usize, phase: MemoryPhase, authorized: bool) -> ArGenParams {
    let mut params = params(seed, frames);
    params.memory.calibration_error_phase = Some(phase);
    params.memory.calibration_fault_harness_authorized = authorized;
    params
}

/// sc-22738 (coordinator decision): the authorized Denoise fault refuses at the denoise phase's
/// **EXIT** — the AR loop has run to completion (its per-step `Progress::Step` events were emitted)
/// and its latents were materialized — but before the decode boundary is reached, so no
/// `Progress::Decoding` is ever emitted and the VAE never runs.
///
/// This is the behavioural discriminator for the convention: under the previous entry convention the
/// fault fired before the first AR step, so `seen` was empty. Hoisting the fault back above
/// `generate_latents` makes the `Progress::Step` assertion fail.
#[test]
fn the_authorized_denoise_fault_refuses_after_the_ar_denoise_has_run() {
    let cfg = tiny_cfg();
    let dit = mlx_gen_krea_realtime::load_krea_realtime_transformer(native_dit_map(&cfg), &cfg)
        .expect("load tiny DiT");
    let transformer = CausalKreaTransformer::new(dit, &cfg);
    let vae = tiny_vae();
    let context = tiny_context(&cfg);
    let cancel = CancelFlag::default();
    let mut seen: Vec<Progress> = Vec::new();
    let mut on_progress = |p: Progress| seen.push(p);

    let error = generate_t2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &fault_params(42, 4, MemoryPhase::Denoise, true),
        None,
        None,
        &cancel,
        &mut on_progress,
    )
    .expect_err("an authorized Denoise fault must refuse");

    let error = error.to_string();
    assert!(error.contains("Denoise"), "{error}");
    assert!(error.contains("krea_realtime_14b"), "{error}");
    assert!(
        seen.iter().any(|p| matches!(p, Progress::Step { .. })),
        "the AR denoise must have run before the fault: {seen:?}"
    );
    assert!(
        !seen.contains(&Progress::Decoding),
        "the fault must precede the decode boundary: {seen:?}"
    );
}

/// sc-22738 (coordinator decision): the authorized Decode fault refuses at the decode phase's
/// **EXIT** — the z16 VAE decode has actually run and its frames were read back to host RGB — and
/// only then is the fault returned.
///
/// Proving "the decode ran" needs a witness, because a refused generation returns no frames. The
/// witness is `decode_to_frames`' own entry cancellation check (`mlx-gen-wan/src/pipeline.rs`): the
/// second half of this test cancels from inside the progress callback on `Progress::Decoding`, which
/// the seam emits AFTER its own last cancel check and immediately BEFORE calling the decode. So
///
///   * under the exit convention control reaches into the decode, which observes the freshly-set
///     flag and returns `Canceled` — the calibration fault is never reached;
///   * under the entry convention the fault fires first and the error names `Decode` instead.
///
/// Together with the first half (the fault does fire, after `Progress::Decoding`) and the source-text
/// ordering test in `t2v.rs`, that pins fault-after-decode rather than fault-before-decode.
#[test]
fn the_authorized_decode_fault_refuses_after_the_vae_decode_has_run() {
    let cfg = tiny_cfg();
    let dit = mlx_gen_krea_realtime::load_krea_realtime_transformer(native_dit_map(&cfg), &cfg)
        .expect("load tiny DiT");
    let transformer = CausalKreaTransformer::new(dit, &cfg);
    let vae = tiny_vae();
    let context = tiny_context(&cfg);

    // Half 1: the fault fires, after the denoise ran and the decode boundary was reached.
    let cancel = CancelFlag::default();
    let mut seen: Vec<Progress> = Vec::new();
    let error = generate_t2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &fault_params(42, 4, MemoryPhase::Decode, true),
        None,
        None,
        &cancel,
        &mut |p: Progress| seen.push(p),
    )
    .expect_err("an authorized Decode fault must refuse");
    let error = error.to_string();
    assert!(error.contains("Decode"), "{error}");
    assert!(error.contains("krea_realtime_14b"), "{error}");
    assert!(
        seen.iter().any(|p| matches!(p, Progress::Step { .. })),
        "the denoise must have completed before the decode boundary: {seen:?}"
    );
    assert!(
        seen.contains(&Progress::Decoding),
        "the decode boundary must have been reached: {seen:?}"
    );

    // Half 2: the decode really is entered before the fault. Cancelling on `Progress::Decoding`
    // lands the flag after the seam's last cancel check and before its decode call, so the decode's
    // own entry check is the first thing that can observe it.
    let cancel = CancelFlag::default();
    let error = generate_t2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &fault_params(42, 4, MemoryPhase::Decode, true),
        None,
        None,
        &cancel,
        &mut |p: Progress| {
            if p == Progress::Decoding {
                cancel.cancel();
            }
        },
    )
    .expect_err("the cancelled decode must refuse");
    assert!(
        matches!(error, mlx_gen::Error::Canceled),
        "the VAE decode must have been entered and observed the cancellation, but the error was \
         `{error}` — a calibration-fault error here means the fault fired BEFORE the decode (the \
         entry convention)"
    );
}

/// sc-22738: a phase selection WITHOUT the harness authorization is inert — the clip renders.
#[test]
fn an_unauthorized_phase_selection_renders_the_clip_normally() {
    let cfg = tiny_cfg();
    let dit = mlx_gen_krea_realtime::load_krea_realtime_transformer(native_dit_map(&cfg), &cfg)
        .expect("load tiny DiT");
    let transformer = CausalKreaTransformer::new(dit, &cfg);
    let vae = tiny_vae();
    let context = tiny_context(&cfg);
    let cancel = CancelFlag::default();
    let mut noop = |_: Progress| {};

    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        let out = generate_t2v_from_components(
            &transformer,
            &cfg,
            &vae,
            &context,
            &fault_params(42, 2, phase, false),
            None,
            None,
            &cancel,
            &mut noop,
        )
        .unwrap_or_else(|error| panic!("{phase:?} must render normally: {error}"));
        assert_eq!(video_frames(&out).len(), 4 * 2, "{phase:?}");
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
        None, // no trim: assert the raw 4·num_latent decode
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
            generate_t2v_from_components(t, &cfg, v, ctx, &params(seed, 2), None, None, c, np)
                .unwrap();
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
    let out = decode_latents_to_video(&vae, &latents, 24, None, None, &cancel).unwrap();
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

/// The z16 decode over-delivers (latent count is `(f−1)/4+1` but the decode is `4·T_lat`), so a
/// requested output count not ≡ 1 (mod 4) must be trimmed back to exactly that count by dropping the
/// **leading** excess — never inventing frames when the request exceeds the decoded count.
#[test]
fn decode_trims_leading_frames_to_requested_count() {
    use mlx_gen_krea_realtime::decode_latents_to_video;
    let vae = tiny_vae();
    let cancel = CancelFlag::default();
    // [z16, T_lat=4, 4, 4] → 4·4 = 16 decoded frames. Request 13 (NOT a multiple of 4, < 16) → trim 3.
    let latents = det_fill(&[16, 4, 4, 4], 99, 1.0, 0.0, Dtype::Float32);
    let full = decode_latents_to_video(&vae, &latents, 16, None, None, &cancel).unwrap();
    let trimmed = decode_latents_to_video(&vae, &latents, 16, Some(13), None, &cancel).unwrap();
    let full = video_frames(&full);
    let trimmed = video_frames(&trimmed);
    assert_eq!(full.len(), 16, "raw z16 decode is 4·T_lat frames");
    assert_eq!(trimmed.len(), 13, "trimmed to exactly the requested count");
    // The kept frames are the LAST 13 (the leading 3 dropped), byte-identical to the untrimmed tail —
    // a discriminating check that the trim removes leading, not trailing, frames.
    for (i, tf) in trimmed.iter().enumerate() {
        assert_eq!(
            tf.pixels,
            full[i + 3].pixels,
            "trim drops the leading over-delivery, keeps the trailing requested frames"
        );
    }
    // A request ≥ the decoded count never invents frames — returns the full decode.
    let over = decode_latents_to_video(&vae, &latents, 16, Some(999), None, &cancel).unwrap();
    assert_eq!(
        video_frames(&over).len(),
        16,
        "over-request returns all decoded frames, no padding"
    );
}

/// End-to-end plumbing: `generate_t2v_from_components` threads the requested output count into the decode
/// trim, so the assembled clip is exactly the requested (non-multiple-of-4) frame count the real
/// `generate_t2v` passes down from `job.num_frames`.
#[test]
fn t2v_pipeline_trims_to_requested_frame_count() {
    let cfg = tiny_cfg();
    let dit = mlx_gen_krea_realtime::load_krea_realtime_transformer(native_dit_map(&cfg), &cfg)
        .expect("load tiny DiT");
    let transformer = CausalKreaTransformer::new(dit, &cfg);
    let vae = tiny_vae();
    let context = tiny_context(&cfg);
    let cancel = CancelFlag::default();
    let mut noop = |_: Progress| {};

    // 4 latent frames → 16 decoded frames; request 13 output frames (not ≡ 1 mod 4) → trimmed to 13.
    let out = generate_t2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &params(42, 4),
        Some(13),
        None,
        &cancel,
        &mut noop,
    )
    .expect("t2v pipeline");
    assert_eq!(
        video_frames(&out).len(),
        13,
        "the pipeline returns exactly the requested output frame count"
    );
}

// ── S7 i2v / v2v component pipeline (sc-8440) ─────────────────────────────────────────────────────

/// A 32×32 RGB8 image (the reference still is 8× the 4×4 latent) with a deterministic pixel pattern so
/// the VAE encode is non-trivial and seed-distinct.
fn tiny_image(seed: u64) -> Image {
    let (w, h) = (32u32, 32u32);
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(7);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h * 3) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        pixels.push((s >> 24) as u8);
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// A `frames`-long 32×32 clip (`frames = 4·(num_latent − 1) + 1` so the encode yields `num_latent`
/// latent frames).
fn tiny_clip(seed: u64, frames: usize) -> Vec<Image> {
    (0..frames)
        .map(|i| tiny_image(seed.wrapping_add(i as u64 * 101)))
        .collect()
}

/// **i2v** component pipeline: VAE-encode a reference still → warm the cache → generate → decode. The
/// clip has the correct shape (`4·(F_ref + num_latent)` frames of 32×32), the reference still changes
/// the decoded clip vs a same-seed t2v clip (the still influences generation), and the run is
/// deterministic.
#[test]
fn i2v_pipeline_conditions_on_the_reference_and_has_correct_shape() {
    let cfg = tiny_cfg();
    let dit = mlx_gen_krea_realtime::load_krea_realtime_transformer(native_dit_map(&cfg), &cfg)
        .expect("load tiny DiT");
    let transformer = CausalKreaTransformer::new(dit, &cfg);
    let vae = tiny_vae();
    let context = tiny_context(&cfg);
    let cancel = CancelFlag::default();
    let mut noop = |_: Progress| {};

    let reference = tiny_image(1234);
    // Generate 2 frames on top of the 1 reference latent ⇒ 3 latent frames ⇒ 12 decoded frames.
    let i2v = generate_i2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &reference,
        &params(42, 2),
        None,
        None,
        &cancel,
        &mut noop,
    )
    .expect("i2v pipeline");
    let i2v_frames = video_frames(&i2v);
    assert_eq!(
        i2v_frames.len(),
        4 * 3,
        "4·(F_ref + num_latent_frames) = 4·3 output frames"
    );
    for f in i2v_frames {
        assert_eq!((f.width, f.height), (32, 32));
    }

    // Discriminating: the reference still changes the decoded clip vs a same-seed t2v run of the SAME
    // total latent length (3 frames), so the difference is the reference conditioning, not the length.
    let t2v = generate_t2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &params(42, 3),
        None,
        None,
        &cancel,
        &mut noop,
    )
    .expect("t2v pipeline");
    let t2v_frames = video_frames(&t2v);
    assert_eq!(t2v_frames.len(), 4 * 3);
    assert!(
        i2v_frames
            .iter()
            .zip(t2v_frames)
            .any(|(a, b)| a.pixels != b.pixels),
        "the reference still must change the decoded clip vs t2v"
    );

    // Determinism.
    let i2v_again = generate_i2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &reference,
        &params(42, 2),
        None,
        None,
        &cancel,
        &mut noop,
    )
    .expect("i2v pipeline again");
    for (a, b) in i2v_frames.iter().zip(video_frames(&i2v_again)) {
        assert_eq!(a.pixels, b.pixels, "same seed ⇒ byte-identical i2v clip");
    }
}

/// **v2v** component pipeline: VAE-encode a source clip → strength-controlled generate → decode. The
/// clip has the correct shape, the source and the strength both influence the decoded clip
/// (strength=0 vs strength=1 differ), and the run is deterministic.
#[test]
fn v2v_pipeline_conditions_on_source_and_strength() {
    let cfg = tiny_cfg();
    let dit = mlx_gen_krea_realtime::load_krea_realtime_transformer(native_dit_map(&cfg), &cfg)
        .expect("load tiny DiT");
    let transformer = CausalKreaTransformer::new(dit, &cfg);
    let vae = tiny_vae();
    let context = tiny_context(&cfg);
    let cancel = CancelFlag::default();

    // 5-frame source clip ⇒ (5−1)/4+1 = 2 latent frames ⇒ params.num_latent_frames = 2.
    let source_a = tiny_clip(9000, 5);
    let source_b = tiny_clip(4000, 5);

    let run = |frames: &[Image], strength: f32| {
        generate_v2v_from_components(
            &transformer,
            &cfg,
            &vae,
            &context,
            frames,
            strength,
            &params(7, 2),
            None,
            None,
            &cancel,
            &mut |_: Progress| {},
        )
        .expect("v2v pipeline")
    };

    let a_half = run(&source_a, 0.5);
    let a_half_frames = video_frames(&a_half);
    assert_eq!(
        a_half_frames.len(),
        4 * 2,
        "4·num_latent_frames output frames"
    );
    for f in a_half_frames {
        assert_eq!((f.width, f.height), (32, 32));
    }

    // Different source ⇒ different clip at strength 0.5.
    let b_half = run(&source_b, 0.5);
    assert!(
        a_half_frames
            .iter()
            .zip(video_frames(&b_half))
            .any(|(x, y)| x.pixels != y.pixels),
        "a different source must change the v2v clip"
    );

    // strength=0 vs strength=1 differ (the strength lever has an effect on the decoded clip).
    let a_0 = run(&source_a, 0.0);
    let a_1 = run(&source_a, 1.0);
    assert!(
        video_frames(&a_0)
            .iter()
            .zip(video_frames(&a_1))
            .any(|(x, y)| x.pixels != y.pixels),
        "strength 0 vs strength 1 must change the decoded clip"
    );

    // Determinism.
    let a_half_2 = run(&source_a, 0.5);
    for (x, y) in a_half_frames.iter().zip(video_frames(&a_half_2)) {
        assert_eq!(
            x.pixels, y.pixels,
            "same seed + strength ⇒ byte-identical v2v clip"
        );
    }
}
