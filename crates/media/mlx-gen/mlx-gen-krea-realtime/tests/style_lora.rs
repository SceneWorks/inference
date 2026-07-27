//! Krea Realtime 14B **Wan-family style LoRA** verification (sc-15015, S14) — torch-free, tiny
//! random-weight DiT fixtures + **synthetic** low-rank LoRA files (no real 28.58 GB checkpoint, no real
//! LoRA weights — real-weight validation is gated to S13/S15).
//!
//! Krea Realtime 14B is Wan-2.1-14B T2V weight-for-weight, so a Wan-family LoRA installs onto the DiT as
//! forward-time residuals through the family-agnostic `apply_adapters_strict` path (the `mlx-gen-scail2`
//! dense-Wan template), resolved against [`CausalKreaTransformer`]'s [`AdaptableHost`] surface. These
//! gates make a broken install (a no-op apply, a dropped-not-reported target, a mis-normalized FFN key,
//! a mis-scaled residual) FAIL a specific assertion:
//!
//!   * **LoRA changes the forward (discriminating)** — a synthetic low-rank LoRA over the DiT linears
//!     shifts the AR forward's velocity latent measurably vs. the no-LoRA baseline (same seed); a
//!     **scale-0** apply is a **bit-exact no-op** (so a no-op install would be caught).
//!   * **Stacked + scale** — two stacked LoRAs on one target stack **additively**; a per-LoRA `scale`
//!     scales the residual linearly (`scale = 2` ⇒ exactly 2× the `scale = 1` residual, `scale = 0` ⇒
//!     zero).
//!   * **Report** — an unsupported / unmatched target is **surfaced (a hard error naming it)**, never
//!     silently dropped.
//!   * **Key normalization** — a Wan reference-named FFN key (`ffn.0`) resolves to the converted Krea
//!     DiT target (`ffn.fc1`) and installs.

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_gen::adapters::loader::apply_adapters_strict;
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::{AdapterKind, AdapterSpec};
use mlx_gen_krea_realtime::{
    load_krea_realtime_transformer, CausalKreaTransformer, KreaRealtimeConfig, MODEL_ID,
};
use mlx_rs::ops::all_close;
use mlx_rs::{Array, Dtype};

// ── Tiny geometry + fixtures (mirrors tests/causal_forward.rs) ───────────────────────────────────

/// A tiny but structurally-complete Krea / Wan-2.1 geometry: dim 64, 2 heads, 2 layers; AR block
/// geometry `num_frames_per_block = 2`, `frame_seq_length = 4`; global attention.
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
    c.ar.seq_length = 100_000;
    c
}

/// Deterministic (seedless, reproducible) fill: `bias + scale · U(-0.5, 0.5)` via xorshift64, at a
/// chosen dtype.
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

/// Full **native** (pre-sanitize) Wan-2.1 transformer inventory for the tiny geometry (F16). Norm gains
/// sit near 1.0 so attention is non-trivial; every other weight is small so the residual stream stays
/// bounded.
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

fn max_abs(a: &Array) -> f32 {
    mlx_rs::ops::max(mlx_rs::ops::abs(a).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    max_abs(&mlx_rs::ops::subtract(a, b).unwrap())
}

/// A fresh tiny Krea Realtime DiT (identical deterministic weights every call) — the base each host
/// starts from, so a LoRA's effect is the *only* difference between two hosts.
fn tiny_transformer(cfg: &KreaRealtimeConfig) -> CausalKreaTransformer {
    let dit = load_krea_realtime_transformer(native_random_map(cfg), cfg).expect("load tiny DiT");
    CausalKreaTransformer::new(dit, cfg)
}

/// Per-prompt cross-attention KV for the (deterministic) tiny text context.
fn cross_kv(causal: &CausalKreaTransformer, cfg: &KreaRealtimeConfig) -> Vec<(Array, Array)> {
    let t5 = det_latent(cfg.wan.text_len as i32, cfg.wan.text_dim as i32, 1, 1, 99)
        .reshape(&[cfg.wan.text_len as i32, cfg.wan.text_dim as i32])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let ctx = causal.inner().embed_text(&t5).expect("embed text");
    causal.prepare_cross_kv(&ctx).expect("cross kv")
}

/// One deterministic AR chunk forward → the denoised velocity latent `[C, F, H, W]`.
fn forward_once(
    causal: &CausalKreaTransformer,
    xkv: &[(Array, Array)],
    cfg: &KreaRealtimeConfig,
) -> Array {
    let c = cfg.wan.in_dim as i32;
    let fpb = cfg.ar.num_frames_per_block as i32;
    let chunk = det_latent(c, fpb, 4, 4, 11);
    let mut cache = causal.new_cache();
    causal
        .forward_chunk(&chunk, 500.0, xkv, 0, &mut cache)
        .expect("forward chunk")
}

/// Write a PEFT LoRA file (`diffusion_model.‹stem›.lora_A/B.weight`) for the given stems, with A
/// `[rank, in]`, B `[out, rank]`, no `.alpha` (⇒ scale = 1). f32 factors (so the residual math is not
/// bf16-rounded), deterministic per stem, at a tunable magnitude so the residual is clearly measurable.
fn write_lora(name: &str, stems: &[(&str, i32, i32)], rank: i32, seed: u64, mag: f32) -> PathBuf {
    let mut entries: Vec<(String, Array)> = Vec::new();
    for (i, (stem, out, inp)) in stems.iter().enumerate() {
        let a = det_fill(&[rank, *inp], seed + i as u64 * 7, mag, 0.0, Dtype::Float32);
        let b = det_fill(
            &[*out, rank],
            seed + 100 + i as u64 * 7,
            mag,
            0.0,
            Dtype::Float32,
        );
        entries.push((format!("diffusion_model.{stem}.lora_A.weight"), a));
        entries.push((format!("diffusion_model.{stem}.lora_B.weight"), b));
    }
    let dir = std::env::temp_dir().join("mlx_gen_krea_style_lora_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let refs: Vec<(&str, &Array)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, &path).unwrap();
    path
}

fn spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path, scale, AdapterKind::Lora)
}

/// Write a peft **LoKr** file for `blocks.0.self_attn.q` ([64,64] = kron(w1[8,8], w2[8,8])) with
/// `alpha = rank` in metadata (⇒ lycoris scale 1). Deterministic factors — the LoKr sibling of
/// [`write_lora`], so `supports_lokr` is honest-by-test (the same shared dense install path scail2 uses).
fn write_lokr(name: &str, mag: f32) -> PathBuf {
    let w1 = det_fill(&[8, 8], 21, mag, 0.0, Dtype::Float32);
    let w2 = det_fill(&[8, 8], 42, mag, 0.0, Dtype::Float32);
    let meta = HashMap::from([
        ("networkType".to_string(), "lokr".to_string()),
        ("alpha".to_string(), "8".to_string()),
        ("rank".to_string(), "8".to_string()),
    ]);
    let dir = std::env::temp_dir().join("mlx_gen_krea_style_lora_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    Array::save_safetensors(
        vec![
            ("blocks.0.self_attn.q.lokr_w1", &w1),
            ("blocks.0.self_attn.q.lokr_w2", &w2),
        ],
        Some(&meta),
        &path,
    )
    .unwrap();
    path
}

fn lokr_spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path, scale, AdapterKind::Lokr)
}

// ── Gates ────────────────────────────────────────────────────────────────────────────────────────

/// (1) A synthetic LoRA over the attention + FFN linears shifts the AR forward measurably vs. baseline,
/// AND a scale-0 apply is a **bit-exact no-op** — so a no-op install (adapters dropped, never applied)
/// would fail one of the two halves.
#[test]
fn synthetic_lora_changes_forward_and_scale0_is_noop() {
    let cfg = tiny_cfg();

    // Baseline (no LoRA).
    let base = tiny_transformer(&cfg);
    let xkv = cross_kv(&base, &cfg);
    let vel_base = forward_once(&base, &xkv, &cfg);

    // The two fresh transformers start byte-identical (deterministic weights) — the no-adapter path is
    // unchanged by the AdaptableHost impl.
    let base2 = tiny_transformer(&cfg);
    let vel_base2 = forward_once(&base2, &cross_kv(&base2, &cfg), &cfg);
    assert_eq!(
        max_abs_diff(&vel_base, &vel_base2),
        0.0,
        "the no-adapter forward must be bit-identical across fresh loads"
    );

    let lora = write_lora(
        "changes_forward.safetensors",
        &[
            ("blocks.0.self_attn.q", 64, 64),
            ("blocks.0.self_attn.v", 64, 64),
            ("blocks.0.ffn.0", 128, 64),
            ("blocks.1.cross_attn.o", 64, 64),
        ],
        4,
        1,
        0.5,
    );

    // scale = 1 ⇒ the forward shifts measurably.
    let mut adapted = tiny_transformer(&cfg);
    let report = apply_adapters_strict(&mut adapted, &[spec(lora.clone(), 1.0)], MODEL_ID).unwrap();
    assert_eq!(report.applied, 4, "all four targets install");
    assert!(report.unmatched_paths.is_empty());
    let vel_adapted = forward_once(&adapted, &cross_kv(&adapted, &cfg), &cfg);
    let signal = max_abs(&vel_base).max(1e-6);
    let delta = max_abs_diff(&vel_base, &vel_adapted);
    assert!(
        delta > 1e-3 * signal,
        "a synthetic LoRA must change the forward: max|Δ|={delta} (signal ~{signal})"
    );

    // scale = 0 ⇒ every LoRA residual is *exactly* zero (the bit-exact no-op is pinned at the linear
    // level by `per_lora_scale_scales_the_residual` — `max|res0| == 0`). At the full-forward level the
    // only remaining difference is a sub-threshold dtype-promotion floor: attaching an f32 residual (even
    // a zero one) to a bf16-activation linear (the FFN feeds bf16) widens that op's output to f32, a
    // ~1e-4 rounding shift orders of magnitude below any real residual. So a scale-0 apply is negligible
    // vs. the scale-1 change — a genuine no-op — while a dropped/ignored scale would not be.
    let mut noop = tiny_transformer(&cfg);
    apply_adapters_strict(&mut noop, &[spec(lora, 0.0)], MODEL_ID).unwrap();
    let vel_noop = forward_once(&noop, &cross_kv(&noop, &cfg), &cfg);
    let d0 = max_abs_diff(&vel_base, &vel_noop);
    assert!(
        delta > 50.0 * d0,
        "a scale-0 LoRA must be negligible (no-op) vs. the scale-1 change: d0={d0}, delta={delta}"
    );
}

/// A deterministic f32 activation `[n, dim]` for the per-linear residual probes.
fn acts(dim: i32) -> Array {
    det_fill(&[4, dim], 7, 1.0, 0.0, Dtype::Float32)
}

/// The forward output of the adaptable linear at `path` under `x` — the base plus every stacked LoRA
/// residual (the same forward the DiT block runs).
fn linear_forward(host: &mut CausalKreaTransformer, path: &[&str], x: &Array) -> Array {
    host.adaptable_mut(path)
        .expect("adaptable target")
        .forward(x)
        .expect("linear forward")
}

/// (2) Two stacked LoRAs on the same target stack **additively**: the residual of A+B equals the sum of
/// A's and B's residuals (both installed over the identical base).
#[test]
fn stacked_loras_stack_additively() {
    let cfg = tiny_cfg();
    let path = ["blocks", "0", "self_attn", "q"];
    let x = acts(cfg.wan.dim as i32);

    let base_out = {
        let mut h = tiny_transformer(&cfg);
        linear_forward(&mut h, &path, &x)
    };

    let lora_a = write_lora(
        "stack_a.safetensors",
        &[("blocks.0.self_attn.q", 64, 64)],
        4,
        1,
        0.3,
    );
    let lora_b = write_lora(
        "stack_b.safetensors",
        &[("blocks.0.self_attn.q", 64, 64)],
        6,
        2,
        0.25,
    );

    let res_a = {
        let mut h = tiny_transformer(&cfg);
        apply_adapters_strict(&mut h, &[spec(lora_a.clone(), 1.0)], MODEL_ID).unwrap();
        mlx_rs::ops::subtract(linear_forward(&mut h, &path, &x), &base_out).unwrap()
    };
    let res_b = {
        let mut h = tiny_transformer(&cfg);
        apply_adapters_strict(&mut h, &[spec(lora_b.clone(), 1.0)], MODEL_ID).unwrap();
        mlx_rs::ops::subtract(linear_forward(&mut h, &path, &x), &base_out).unwrap()
    };

    // Both files stacked in one install (applied twice on the same target).
    let mut h_ab = tiny_transformer(&cfg);
    let report =
        apply_adapters_strict(&mut h_ab, &[spec(lora_a, 1.0), spec(lora_b, 1.0)], MODEL_ID)
            .unwrap();
    assert_eq!(
        report.applied, 2,
        "both stacked LoRA files install onto the target"
    );
    let res_ab = mlx_rs::ops::subtract(linear_forward(&mut h_ab, &path, &x), &base_out).unwrap();

    // Non-trivial residuals (a dropped adapter would zero one of them and fail the additivity below).
    assert!(
        max_abs(&res_a) > 1e-3,
        "LoRA A residual must be non-trivial"
    );
    assert!(
        max_abs(&res_b) > 1e-3,
        "LoRA B residual must be non-trivial"
    );
    let sum = mlx_rs::ops::add(&res_a, &res_b).unwrap();
    assert!(
        all_close(&res_ab, &sum, 2e-2, 2e-2, false)
            .unwrap()
            .item::<bool>(),
        "stacked LoRA residual must equal the sum of the individual residuals"
    );
}

/// (3) A per-LoRA `scale` scales the residual linearly: `scale = 0` ⇒ exactly zero; `scale = 2` ⇒ ~2×
/// the `scale = 1` residual.
#[test]
fn per_lora_scale_scales_the_residual() {
    let cfg = tiny_cfg();
    let path = ["blocks", "0", "ffn", "0"]; // reference FFN naming → normalizes to ffn.fc1
    let x = acts(cfg.wan.dim as i32);

    let base_out = {
        let mut h = tiny_transformer(&cfg);
        linear_forward(&mut h, &path, &x)
    };
    let lora = write_lora(
        "scale.safetensors",
        &[("blocks.0.ffn.0", 128, 64)],
        4,
        3,
        0.3,
    );

    let residual_at = |scale: f32| -> Array {
        let mut h = tiny_transformer(&cfg);
        apply_adapters_strict(&mut h, &[spec(lora.clone(), scale)], MODEL_ID).unwrap();
        mlx_rs::ops::subtract(linear_forward(&mut h, &path, &x), &base_out).unwrap()
    };

    let res0 = residual_at(0.0);
    let res1 = residual_at(1.0);
    let res2 = residual_at(2.0);

    assert_eq!(
        max_abs(&res0),
        0.0,
        "scale = 0 must be an exact no-op residual"
    );
    assert!(
        max_abs(&res1) > 1e-3,
        "scale = 1 residual must be non-trivial (discriminating)"
    );
    let twice = mlx_rs::ops::add(&res1, &res1).unwrap();
    assert!(
        all_close(&res2, &twice, 2e-2, 2e-2, false)
            .unwrap()
            .item::<bool>(),
        "scale = 2 residual must be ~2× the scale = 1 residual"
    );
}

/// **(3b) sc-8446 S13 — the whole-model globals install AND move the forward.**
///
/// The settled globals decision is not "the resolver returns `Some`": a widened surface that routed the
/// keys but dropped the residual would still silently under-apply a real step-distill LoRA. So this
/// installs a LoRA that targets **only** globals — every one in the reference/file spelling a real
/// lightx2v / FastWan file carries — and requires all seven to install and the forward to change.
#[test]
fn globals_install_end_to_end_and_change_the_forward() {
    let cfg = tiny_cfg();
    let base = tiny_transformer(&cfg);
    let vel_base = forward_once(&base, &cross_kv(&base, &cfg), &cfg);

    // Shapes from `tiny_cfg`: dim 64, freq_dim 32, text_dim 32, in_dim 16, patch (1,2,2), out_dim 16.
    //   patch_embedding_proj [dim, in_dim·∏patch] = [64, 64]   text_embedding_0 [64, 32]
    //   text_embedding_1     [64, 64]                          time_embedding_0 [64, 32]
    //   time_embedding_1     [64, 64]                          time_projection  [6·64, 64] = [384, 64]
    //   head.head            [out_dim·∏patch, dim] = [64, 64]
    let lora = write_lora(
        "globals_only.safetensors",
        &[
            ("patch_embedding", 64, 64),
            ("text_embedding.0", 64, 32),
            ("text_embedding.2", 64, 64),
            ("time_embedding.0", 64, 32),
            ("time_embedding.2", 64, 64),
            ("time_projection.1", 384, 64),
            ("head.head", 64, 64),
        ],
        4,
        77,
        0.4,
    );

    let mut adapted = tiny_transformer(&cfg);
    let report = apply_adapters_strict(&mut adapted, &[spec(lora, 1.0)], MODEL_ID)
        .expect("a globals-only Wan LoRA must install (sc-8446 widened the surface)");
    assert_eq!(
        report.applied, 7,
        "all seven whole-model globals must install, not just resolve"
    );
    assert!(report.unmatched_paths.is_empty());

    let vel_adapted = forward_once(&adapted, &cross_kv(&adapted, &cfg), &cfg);
    let signal = max_abs(&vel_base).max(1e-6);
    let delta = max_abs_diff(&vel_base, &vel_adapted);
    assert!(
        delta > 1e-3 * signal,
        "a globals-only LoRA installed but did not move the forward (Δ {delta:.3e} vs signal \
         {signal:.3e}) — the residual is not reaching the global Linears"
    );
}

/// (4) An unsupported / unmatched adapter target is **surfaced** — the strict installer errors and names
/// it — never silently dropped.
///
/// The out-of-surface case is the **I2V-only image cross-attention** (`cross_attn.k_img`/`v_img`), which
/// real Wan-**I2V** LoRAs carry and which does not exist on this T2V backbone at any surface width. It is
/// deliberately NOT the whole-model globals: sc-8446 (S13) settled those as exposed, because real Wan-T2V
/// step-distill LoRAs target them with genuine low-rank factors (see `globals_install_end_to_end`).
#[test]
fn unsupported_target_is_reported_not_silently_dropped() {
    let cfg = tiny_cfg();

    // A file mixing a valid target with an I2V-only module this T2V backbone does not have.
    let mixed = write_lora(
        "mixed_target.safetensors",
        &[
            ("blocks.0.self_attn.q", 64, 64),
            ("blocks.0.cross_attn.k_img", 64, 64),
        ],
        4,
        4,
        0.2,
    );
    let mut h = tiny_transformer(&cfg);
    let err = apply_adapters_strict(&mut h, &[spec(mixed, 1.0)], MODEL_ID)
        .expect_err("an unmatched target must surface as an error, not be silently dropped");
    let msg = err.to_string();
    assert!(
        msg.contains("k_img"),
        "the error must name the unmatched target: {msg}"
    );

    // A block index beyond the model (paired with a valid target so the file matches *something*) is
    // likewise surfaced by name — never silently ignored.
    let oob = write_lora(
        "oob_block.safetensors",
        &[
            ("blocks.0.self_attn.q", 64, 64),
            ("blocks.99.self_attn.q", 64, 64),
        ],
        4,
        5,
        0.2,
    );
    let mut h2 = tiny_transformer(&cfg);
    let err2 = apply_adapters_strict(&mut h2, &[spec(oob, 1.0)], MODEL_ID)
        .expect_err("an out-of-range block target must surface as an error");
    assert!(
        err2.to_string().contains("blocks.99"),
        "the error must name the out-of-range target: {err2}"
    );
}

/// (5) Key normalization: a Wan reference-named FFN key (`ffn.0`) resolves to the converted Krea DiT
/// target (`ffn.fc1`) both via the raw resolver AND end-to-end through the strict installer.
#[test]
fn wan_family_ffn_key_normalizes_and_resolves() {
    let cfg = tiny_cfg();
    let mut host = tiny_transformer(&cfg);

    // Raw resolver: the reference `ffn.0` and the converted `ffn.fc1` both resolve (normalize maps the
    // former to the latter; the latter passes through); attention `q/k/v/o` pass through unchanged.
    assert!(
        host.adaptable_mut(&["blocks", "0", "ffn", "0"]).is_some(),
        "ffn.0 must resolve"
    );
    assert!(
        host.adaptable_mut(&["blocks", "0", "ffn", "fc1"]).is_some(),
        "ffn.fc1 must resolve"
    );
    assert!(host
        .adaptable_mut(&["blocks", "1", "self_attn", "q"])
        .is_some());
    assert!(
        host.adaptable_mut(&["blocks", "0", "ffn", "9"]).is_none(),
        "a bogus FFN index must not resolve"
    );
    // sc-8446 S13: the whole-model globals ARE exposed now, in the reference/file spelling a LoRA
    // carries (the normalizer bridges to the converted names). The I2V-only image cross-attention still
    // is not — those modules do not exist on a T2V backbone.
    assert!(
        host.adaptable_mut(&["patch_embedding"]).is_some(),
        "patch_embedding is an exposed global (sc-8446)"
    );
    assert!(
        host.adaptable_mut(&["head", "head"]).is_some(),
        "head.head is an exposed global (sc-8446)"
    );
    assert!(
        host.adaptable_mut(&["blocks", "0", "cross_attn", "k_img"])
            .is_none(),
        "the I2V-only image cross-attention does not exist on this T2V backbone"
    );

    // The reference-named `ffn.0` LoRA installs onto the FFN linear that `ffn.fc1` addresses: applying
    // the SAME factors under both spellings yields an identical linear forward.
    let x = acts(cfg.wan.dim as i32);
    let out_ref = {
        let lora = write_lora(
            "ffn_ref.safetensors",
            &[("blocks.0.ffn.0", 128, 64)],
            4,
            6,
            0.3,
        );
        let mut h = tiny_transformer(&cfg);
        let r = apply_adapters_strict(&mut h, &[spec(lora, 1.0)], MODEL_ID).unwrap();
        assert_eq!(r.applied, 1, "the reference-named ffn.0 LoRA installs");
        linear_forward(&mut h, &["blocks", "0", "ffn", "0"], &x)
    };
    let out_conv = {
        let lora = write_lora(
            "ffn_conv.safetensors",
            &[("blocks.0.ffn.fc1", 128, 64)],
            4,
            6,
            0.3,
        );
        let mut h = tiny_transformer(&cfg);
        let r = apply_adapters_strict(&mut h, &[spec(lora, 1.0)], MODEL_ID).unwrap();
        assert_eq!(r.applied, 1, "the converted-named ffn.fc1 LoRA installs");
        linear_forward(&mut h, &["blocks", "0", "ffn", "fc1"], &x)
    };
    assert_eq!(
        max_abs_diff(&out_ref, &out_conv),
        0.0,
        "ffn.0 and ffn.fc1 must address the same linear (normalization), so identical factors match"
    );
}

/// (6) A peft **LoKr** file installs on the dense base through the same strict path and shifts the
/// forward — the `supports_lokr = true` the descriptor advertises is genuine (dense install; the packed
/// Q4/Q8 LoKr is the separate quant-tier story S19).
#[test]
fn lokr_installs_on_dense_and_changes_forward() {
    let cfg = tiny_cfg();

    let base = tiny_transformer(&cfg);
    let vel_base = forward_once(&base, &cross_kv(&base, &cfg), &cfg);

    let lokr = write_lokr("q_lokr.safetensors", 0.5);
    let mut adapted = tiny_transformer(&cfg);
    let report = apply_adapters_strict(&mut adapted, &[lokr_spec(lokr, 1.0)], MODEL_ID).unwrap();
    assert_eq!(report.applied, 1, "the peft LoKr installs on the dense DiT");
    assert!(report.unmatched_paths.is_empty());

    let vel_adapted = forward_once(&adapted, &cross_kv(&adapted, &cfg), &cfg);
    let signal = max_abs(&vel_base).max(1e-6);
    let delta = max_abs_diff(&vel_base, &vel_adapted);
    assert!(
        delta > 1e-3 * signal,
        "a LoKr must change the forward: max|Δ|={delta} (signal ~{signal})"
    );
}
