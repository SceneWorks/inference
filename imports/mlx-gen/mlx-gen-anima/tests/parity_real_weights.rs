//! Anima MLX-port parity goldens (sc-10524, epic 10512) — the **real-weights** stages. `#[ignore]`d +
//! weights-gated (they need the licensed `circlestone-labs/Anima` snapshot in the HF cache + Metal), so
//! they **never run in CI**. Run locally with:
//!   cargo test -p mlx-gen-anima --release --test parity_real_weights -- --ignored --nocapture
//!
//! Each test reads a committed golden JSON (computed by the `tests/fixtures/gen_anima_*.py` generators
//! from the diffusers 0.39.0 reference; Apache-2.0) — **no Python at test time** — and runs the MLX port
//! on the single-file checkpoint, then compares a committed deterministic subsample + summary stats.
//!
//!   * **Stage 2** — Qwen3-0.6B `last_hidden_state` AFTER the attention-mask multiply. bf16 both sides.
//!   * **Stage 3** — `AnimaTextConditioner` output `(1, 512, 1024)`, right-padded after masking. fp32.
//!   * **Stage 4** — Cosmos DiT forward: one block, then all 28. fp32. Localizes adaLN-LoRA / RoPE drift.
//!   * **Stage 7** — end-to-end MLX vs diffusers for ALL THREE variants: identical injected init latent +
//!     deterministic Euler + identical schedule (fp32), comparing the final latent AND the decoded image.
//!
//! Stages 3, 4 & 7 feed a DETERMINISTIC input (bit-identical to the Python generators' `lcg_fill` /
//! `gauss_fill`), so no large input tensor is committed and the golden isolates the component's math
//! (fp32) rather than bf16 quantization. Stage 2's input is the deterministic Qwen2 token ids.

use std::path::{Path, PathBuf};

use mlx_rs::ops::{concatenate_axis, multiply};
use mlx_rs::{Array, Dtype};
use serde_json::Value;

use mlx_gen::image::decoded_to_image;
use mlx_gen::weights::Weights;
use mlx_gen::WeightsSource;
use mlx_gen_anima::conditioner::AnimaTextConditioner;
use mlx_gen_anima::config::{
    ConditionerConfig, DitConfig, Qwen3Config, Variant, QWEN_PAD_TOKEN_ID,
};
use mlx_gen_anima::pipeline::AnimaPipeline;
use mlx_gen_anima::text_encoder::AnimaQwen3;
use mlx_gen_anima::tokenizer::AnimaTokenizers;
use mlx_gen_anima::transformer::CosmosDiT;

// ------------------------------------------------------------------------------------------------
// Shared helpers.
// ------------------------------------------------------------------------------------------------

/// Glob the Anima snapshot's `split_files/` dir from the HF cache (no hardcoded sha). Mirrors
/// `tests/real_weights.rs`.
fn split_files() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--circlestone-labs--Anima/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let p = e.path().join("split_files");
            p.join("diffusion_models").is_dir().then_some(p)
        })
}

fn dit_file(split: &Path) -> PathBuf {
    split
        .join("diffusion_models")
        .join(Variant::Base.dit_filename())
}

fn load_golden(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// Portable LCG in [-1, 1) — **bit-identical** to the Python generator's `lcg_fill` (pure integer
/// recurrence + f64->f32 cast). Feeds the stage-3/4 deterministic synthetic inputs.
fn lcg_fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed & 0x7fff_ffff;
    (0..n)
        .map(|_| {
            s = (s.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fff_ffff;
            (s as f64 / 2147483647.0 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn i64s(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// Flatten a Rust output to f64 and assert its shape + element count against a golden summary.
fn flatten_checked(got: &Array, g: &Value, label: &str) -> Vec<f64> {
    let want_shape: Vec<i32> = g["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(got.shape(), &want_shape[..], "{label}: shape");
    let flat: Vec<f64> = got
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|&x| x as f64)
        .collect();
    let count = g["count"].as_u64().unwrap() as usize;
    assert_eq!(flat.len(), count, "{label}: element count");
    flat
}

/// The **aggregate-stats parity gate** (the real structural-correctness assertion): recompute
/// mean/std/l2 on the Rust output and require they match the golden within `stat_rtol`. A structural
/// port bug (wrong sign / dropped mask / mis-scaled RoPE) shifts these by orders of magnitude; bf16 /
/// fp32 reduced-precision does not. Returns the flattened output for optional elementwise checks.
fn assert_stats(got: &Array, g: &Value, label: &str, stat_rtol: f64) -> Vec<f64> {
    let flat = flatten_checked(got, g, label);
    let count = flat.len();
    let mean = flat.iter().sum::<f64>() / count as f64;
    let var = flat.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count as f64;
    let std = var.sqrt();
    let l2 = flat.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let (gmean, gstd, gl2) = (
        g["mean"].as_f64().unwrap(),
        g["std"].as_f64().unwrap(),
        g["l2"].as_f64().unwrap(),
    );
    let rel = |a: f64, b: f64| (a - b).abs() / (b.abs().max(1e-6));
    println!(
        "[{label}] stats: mean {mean:.5} (g {gmean:.5}), std {std:.5} (g {gstd:.5}), l2 {l2:.4} (g {gl2:.4})"
    );
    // mean can be ~0 (relative error explodes), so gate it absolutely against the std scale.
    assert!(
        (mean - gmean).abs() < stat_rtol * gstd.abs().max(1e-3),
        "{label}: mean drift {mean} vs {gmean}"
    );
    assert!(
        rel(std, gstd) < stat_rtol,
        "{label}: std drift {std} vs {gstd}"
    );
    assert!(rel(l2, gl2) < stat_rtol, "{label}: l2 drift {l2} vs {gl2}");
    flat
}

/// Aggregate stats only (shape/count/mean/std/l2). Used where the committed summary carries no
/// elementwise samples — e.g. stage-3's FULL `[1,512,1024]` (its samples live in the active region;
/// the full summary exists to verify the right-padding via count/std/l2).
fn assert_stats_only(got: &Array, g: &Value, label: &str, stat_rtol: f64) {
    let _ = assert_stats(got, g, label, stat_rtol);
}

/// Relative-L2 over the committed sample set: `‖got−want‖₂ / ‖want‖₂`. This is the robust
/// "no quality regression" metric for a **bf16 cross-backend** comparison — unlike a per-element
/// bound it is not dominated by a single small-magnitude residual whose bf16 rounding differs between
/// torch-CPU and MLX-Metal (a correct-port artifact of catastrophic cancellation, not a bug). The
/// aggregate-stats gate (`assert_stats`) plus this relative-L2 together pin structural correctness.
fn assert_sampled_rel_l2(
    got: &Array,
    g: &Value,
    label: &str,
    stat_rtol: f64,
    max_rel_l2: f64,
) -> f64 {
    let flat = assert_stats(got, g, label, stat_rtol);
    let idx = i64s(&g["sample_indices"]);
    let vals: Vec<f64> = g["sample_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    assert_eq!(
        idx.len(),
        vals.len(),
        "{label}: index/value length mismatch"
    );
    assert!(
        idx.len() >= 16,
        "{label}: too few golden samples ({}) — asserts nothing",
        idx.len()
    );
    let mut num = 0f64;
    let mut den = 0f64;
    for (&i, &want) in idx.iter().zip(&vals) {
        let d = flat[i as usize] - want;
        num += d * d;
        den += want * want;
    }
    let rel_l2 = (num / den.max(1e-12)).sqrt();
    println!(
        "[{label}] sampled rel-L2 = {rel_l2:.4e} over {} samples (bound {max_rel_l2:.1e})",
        idx.len()
    );
    assert!(
        rel_l2 < max_rel_l2,
        "{label}: sampled rel-L2 {rel_l2:.4e} exceeds {max_rel_l2:.1e} — a real port bug, not bf16 noise"
    );
    rel_l2
}

/// Full comparison: the aggregate-stats gate PLUS every committed sample index. `stat_rtol` gates the
/// aggregate (tight — the real parity gate); `sample_atol`/`sample_rtol` gate individual elements
/// (looser for bf16 peak elements, which diverge a few % between torch-CPU and MLX-Metal even on a
/// correct port — the aggregate proves correctness, the samples localize).
fn assert_matches_summary(
    got: &Array,
    g: &Value,
    label: &str,
    stat_rtol: f64,
    sample_atol: f64,
    sample_rtol: f64,
) {
    let flat = assert_stats(got, g, label, stat_rtol);
    let idx = i64s(&g["sample_indices"]);
    let vals: Vec<f64> = g["sample_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    assert_eq!(
        idx.len(),
        vals.len(),
        "{label}: golden index/value length mismatch"
    );
    assert!(
        idx.len() >= 16,
        "{label}: too few golden samples ({}) — a short/empty loop asserts nothing",
        idx.len()
    );
    let mut max_abs = 0f64;
    for (&i, &want) in idx.iter().zip(&vals) {
        let got = flat[i as usize];
        let d = (got - want).abs();
        max_abs = max_abs.max(d);
        assert!(
            d <= sample_atol + sample_rtol * want.abs(),
            "{label}: sample[{i}] = {got:.6}, want {want:.6} (|d| {d:.6} > {})",
            sample_atol + sample_rtol * want.abs()
        );
    }
    println!(
        "[{label}] max sample |Δ| = {max_abs:.6} over {} samples",
        idx.len()
    );
}

// ------------------------------------------------------------------------------------------------
// Stage 2 — Qwen3-0.6B last_hidden_state after the mask multiply (bf16).
// ------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot + Metal"]
fn stage2_qwen3_last_hidden_state() {
    let split = split_files().expect("Anima snapshot");
    let g = load_golden("qwen3_hidden_golden.json");
    let want_ids = i64s(&g["meta"]["qwen_ids"]);
    let real = g["meta"]["real_tokens"].as_u64().unwrap() as usize;
    let pad_k = g["meta"]["padded_tokens"].as_u64().unwrap() as i32;

    let te_w = Weights::from_file(split.join("text_encoders/qwen_3_06b_base.safetensors")).unwrap();
    let te = AnimaQwen3::from_weights(&te_w, "model", &Qwen3Config::anima()).unwrap();
    let tk = AnimaTokenizers::load().unwrap();
    let (real_ids, real_mask) = tk
        .encode_qwen(g["meta"]["prompt"].as_str().unwrap())
        .unwrap();
    // The Rust tokenizer must reproduce the golden's REAL tokens (the [0:real] prefix).
    let got_real: Vec<i64> = real_ids
        .as_slice::<i32>()
        .iter()
        .map(|&x| x as i64)
        .collect();
    assert_eq!(
        &got_real[..],
        &want_ids[..real],
        "stage2: real Qwen ids drifted from the golden"
    );

    // Explicitly right-pad with `pad_k` Qwen2-pad tokens at mask 0. The golden's mask carries real 0s, so
    // the mask-multiply below has actual rows to zero — an all-ones batch-1 mask made the trap a no-op and
    // dropping the multiply then changed nothing (sc-10524 review).
    let pad_ids = Array::from_slice(&vec![QWEN_PAD_TOKEN_ID; pad_k as usize], &[1, pad_k]);
    let pad_mask = Array::from_slice(&vec![0i32; pad_k as usize], &[1, pad_k]);
    let ids = concatenate_axis(&[&real_ids, &pad_ids], 1).unwrap();
    let mask = concatenate_axis(&[&real_mask, &pad_mask], 1).unwrap();
    let got_ids: Vec<i64> = ids.as_slice::<i32>().iter().map(|&x| x as i64).collect();
    assert_eq!(
        got_ids, want_ids,
        "stage2: padded Qwen ids must match the golden"
    );

    let hidden = te.forward(&ids, &mask).unwrap(); // [1, S, 1024] bf16
                                                   // the mask-multiply trap — NON-TRIVIAL now: it zeros the `pad_k` padded rows.
    let m = mask
        .as_dtype(hidden.dtype())
        .unwrap()
        .expand_dims(2)
        .unwrap();
    let hidden = multiply(&hidden, &m).unwrap();
    // The padded rows [real:] must be EXACTLY zero after the multiply — drop the multiply and they stay
    // nonzero (the causal tower still computes them), so the stats/rel-L2 below diverge. This is the
    // assertion that makes the mask-multiply trap catch a regression.
    let hf = hidden.as_dtype(Dtype::Float32).unwrap();
    let pad_abs_max = hf.as_slice::<f32>()[real * 1024..]
        .iter()
        .fold(0f32, |m, &v| m.max(v.abs()));
    println!("[stage2] pad rows [{real}:] abs-max = {pad_abs_max:.2e}");
    assert!(
        pad_abs_max < 1e-4,
        "stage2: mask-multiply must zero the {pad_k} padded rows (got abs-max {pad_abs_max:.2e})"
    );
    // bf16 tower vs a torch bf16 reference. Gate on (a) the aggregate stats (mean/std/l2 match to <0.3%
    // — the structural-correctness gate; a real port bug moves these by orders of magnitude) and (b) the
    // sampled relative-L2 (~1e-2 — the robust no-quality-regression metric). A per-element bound is the
    // WRONG metric here: bf16 elements that are small residuals of large cancellations diverge a few %
    // in absolute terms between torch-CPU and MLX-Metal even on a correct port (the fp32 stages 3 & 4,
    // which share rms_norm/attention/linear, pass a per-element 1e-2 — proving the math is right).
    assert_sampled_rel_l2(&hidden, &g["last_hidden_state"], "stage2_qwen3", 1e-2, 3e-2);
}

// ------------------------------------------------------------------------------------------------
// Stage 3 — AnimaTextConditioner output (1, 512, 1024), right-padded after masking (fp32).
// ------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot + Metal"]
fn stage3_conditioner_output() {
    let split = split_files().expect("Anima snapshot");
    let g = load_golden("conditioner_golden.json");
    let st = g["meta"]["st"].as_u64().unwrap() as i32;
    let t5_ids: Vec<i32> = i64s(&g["meta"]["t5_ids"])
        .iter()
        .map(|&x| x as i32)
        .collect();
    let src_shape: Vec<i32> = i64s(&g["meta"]["lcg"]["source_shape"])
        .iter()
        .map(|&x| x as i32)
        .collect();

    let w = Weights::from_file(dit_file(&split)).unwrap();
    let cond =
        AnimaTextConditioner::from_weights(&w, "net.llm_adapter", ConditionerConfig::anima())
            .unwrap();

    let n: usize = src_shape.iter().product::<i32>() as usize;
    let source = Array::from_slice(&lcg_fill(n, 3), &src_shape[..]); // fp32
    let target = Array::from_slice(&t5_ids, &[1, st]);
    let out = cond.forward(&source, &target, Dtype::Float32).unwrap();
    assert_eq!(
        out.shape(),
        &[1, 512, 1024],
        "stage3: must right-pad to 512 tokens"
    );

    // The right-padded rows [st:512] must be exactly the zero pad the DiT expects.
    let flat = out.as_slice::<f32>();
    let pad_from = (st as usize) * 1024;
    let pad_abs_max = flat[pad_from..].iter().fold(0f32, |m, &v| m.max(v.abs()));
    println!("[stage3] pad rows [{st}:512] abs-max = {pad_abs_max:.2e}");
    assert!(
        pad_abs_max < 1e-4,
        "stage3: rows past the real tokens must be zero padding"
    );

    // Aggregate stats over the full [1,512,1024] verify the padding; sample values assert the ACTIVE
    // conditioned region (rows [0:st]).
    assert_stats_only(&out, &g["full"], "stage3_full", 1e-2);
    let rows: Vec<i32> = (0..st).collect();
    let active = out.take_axis(Array::from_slice(&rows, &[st]), 1).unwrap();
    // fp32 both sides -> the conditioner isolates cleanly: tight aggregate + tight per-element.
    assert_matches_summary(&active, &g["active"], "stage3_active", 1e-2, 5e-3, 1e-2);
}

// ------------------------------------------------------------------------------------------------
// Stage 4 — Cosmos DiT forward: one block, then all 28 (fp32). adaLN-LoRA + NTK 3D RoPE hot spots.
// ------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot + Metal"]
fn stage4_dit_forward_block0_and_full() {
    let split = split_files().expect("Anima snapshot");
    let g = load_golden("dit_forward_golden.json");
    let lat_shape: Vec<i32> = i64s(&g["meta"]["latent_shape"])
        .iter()
        .map(|&x| x as i32)
        .collect();
    let enc_shape: Vec<i32> = i64s(&g["meta"]["encoder_shape"])
        .iter()
        .map(|&x| x as i32)
        .collect();
    let sigma_v = g["meta"]["lcg"]["sigma"].as_f64().unwrap() as f32;

    let w = Weights::from_file(dit_file(&split)).unwrap();
    let dit = CosmosDiT::from_weights(&w, "net", DitConfig::anima()).unwrap();

    let latent = Array::from_slice(
        &lcg_fill(lat_shape.iter().product::<i32>() as usize, 1),
        &lat_shape[..],
    );
    let encoder = Array::from_slice(
        &lcg_fill(enc_shape.iter().product::<i32>() as usize, 2),
        &enc_shape[..],
    );
    let sigma = Array::from_slice(&[sigma_v], &[1]);

    // one block (localizes adaLN-LoRA modulation + NTK-scaled 3D RoPE to a single block). fp32 both
    // sides -> ~1e-3 Metal-vs-CPU; a 1e-2 gate cleanly separates correct-port noise from a real bug.
    let block0 = dit
        .forward_hidden(&latent, &sigma, &encoder, Dtype::Float32, Some(1))
        .unwrap();
    assert_matches_summary(&block0, &g["block0"], "stage4_block0", 1e-2, 5e-3, 1e-2);

    // all 28 blocks + norm_out + proj_out + unpatchify (the final velocity).
    let full = dit
        .forward(&latent, &sigma, &encoder, Dtype::Float32)
        .unwrap();
    assert_matches_summary(&full, &g["full"], "stage4_full", 1e-2, 5e-3, 1e-2);
}

// ------------------------------------------------------------------------------------------------
// Stage 7 — end-to-end MLX vs diffusers for ALL THREE variants (fp32). The chaos objection is
// neutralized by injecting the IDENTICAL initial latent (a deterministic Gaussian, bit-reproduced
// from the Python generator's Box-Muller + LCG) and running DETERMINISTIC Euler over the identical
// sigma schedule on both sides — so residual drift is Metal-vs-MPS float error, not chaos. Compares the
// final latent (pre-VAE) and the decoded image. PNGs are written to $ANIMA_STAGE7_OUT for a visual look.
// ------------------------------------------------------------------------------------------------

/// Deterministic Gaussian init via LCG uniforms -> Box-Muller — **bit-identical** to the Python
/// generator's `gauss_fill` (same 31-bit LCG recurrence, `u=(s+0.5)/2^31`, f64 transcendentals -> f32).
fn gauss_fill(n: usize, seed: u64) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let mut s = seed & 0x7fff_ffff;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut next = || {
        s = (s.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fff_ffff;
        (s as f64 + 0.5) / 2147483648.0
    };
    let mut i = 0;
    while i < n {
        let u1 = next();
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        out[i] = (r * (two_pi * u2).cos()) as f32;
        i += 1;
        if i < n {
            out[i] = (r * (two_pi * u2).sin()) as f32;
            i += 1;
        }
    }
    out
}

fn save_png(pixels: &[u8], w: u32, h: u32, path: &std::path::Path) {
    if let Some(buf) = image::RgbImage::from_raw(w, h, pixels.to_vec()) {
        let _ = buf.save(path);
    }
}

/// Compare a decoded RGB image (HWC uint8) against the golden image summary: per-channel mean/std
/// (perceptual global match) + a sampled pixel MAE (local match). Prints the deltas so a real
/// divergence is legible.
fn assert_image_matches(pixels: &[u8], g: &Value, label: &str, mean_tol: f64, sample_mae_tol: f64) {
    let count = g["count"].as_u64().unwrap() as usize;
    assert_eq!(pixels.len(), count, "{label}: pixel count");
    let gpm: Vec<f64> = g["per_channel_mean"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    for (c, &gm) in gpm.iter().enumerate() {
        let vals: Vec<f64> = pixels
            .iter()
            .skip(c)
            .step_by(3)
            .map(|&p| p as f64)
            .collect();
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        println!(
            "[{label}] channel {c} mean {m:.2} (g {gm:.2}, Δ {:.2})",
            (m - gm).abs()
        );
        assert!(
            (m - gm).abs() < mean_tol,
            "{label}: channel {c} mean drift {m:.2} vs {gm:.2}"
        );
    }
    let idx = i64s(&g["sample_indices"]);
    let vals: Vec<f64> = g["sample_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    assert!(idx.len() >= 16, "{label}: too few golden pixel samples");
    let (mut sae, mut max_abs) = (0f64, 0f64);
    for (&i, &want) in idx.iter().zip(&vals) {
        let d = (pixels[i as usize] as f64 - want).abs();
        sae += d;
        max_abs = max_abs.max(d);
    }
    let mae = sae / idx.len() as f64;
    println!(
        "[{label}] sampled pixel MAE = {mae:.2} levels, max = {max_abs:.0} over {} samples",
        idx.len()
    );
    assert!(
        mae < sample_mae_tol,
        "{label}: sampled pixel MAE {mae:.2} exceeds {sample_mae_tol} levels"
    );
}

/// Full-tensor relative-L2 `‖a−b‖₂ / ‖b‖₂` over every element (both flattened to f64). Used to measure
/// the DIRECT bf16-vs-fp32 conditioning offset and the bf16-vs-fp32 final-latent delta (sc-10577).
fn rel_l2_full(a: &Array, b: &Array) -> f64 {
    let fa: Vec<f64> = a
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|&x| x as f64)
        .collect();
    let fb: Vec<f64> = b
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|&x| x as f64)
        .collect();
    assert_eq!(fa.len(), fb.len(), "rel_l2_full: shape mismatch");
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in fa.iter().zip(&fb) {
        num += (x - y) * (x - y);
        den += y * y;
    }
    (num / den.max(1e-12)).sqrt()
}

/// Sampled relative-L2 of `got` against a golden `final_latent` summary's committed sample set — the
/// SAME metric `assert_sampled_rel_l2` gates, but returned as a bare number (no bound) so the sc-10577
/// isolation test can report bf16-TE and fp32-TE residuals side by side.
fn sampled_rel_l2_value(got: &Array, g: &Value) -> f64 {
    let flat: Vec<f64> = got
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|&x| x as f64)
        .collect();
    let idx = i64s(&g["sample_indices"]);
    let vals: Vec<f64> = g["sample_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let (mut num, mut den) = (0f64, 0f64);
    for (&i, &want) in idx.iter().zip(&vals) {
        let d = flat[i as usize] - want;
        num += d * d;
        den += want * want;
    }
    (num / den.max(1e-12)).sqrt()
}

// ------------------------------------------------------------------------------------------------
// sc-10577 — ISOLATE the bf16-conditioning offset (fp32-TE reference variant).
//
// sc-10524's stage-7 golden left the ~7.8e-2 base final-latent residual attributed to accumulation by
// INFERENCE (step-1 rel-L2 ~2e-5 ≪ final, super-linear growth). This test MEASURES the bf16-conditioning
// contribution directly: it runs the identical injected-init + deterministic-Euler + schedule stage-7
// denoise (DiT fp32) TWICE per variant — once with the shipped bf16-weight conditioning, once with an
// fp32-upcast TE + conditioner (mirroring the diffusers reference's `.float()`, via
// `loader::load_conditioning_at_dtype`) — and compares each final latent to the SAME fp32 reference
// golden. The drop from the bf16-TE residual to the fp32-TE residual IS the bf16-conditioning offset.
//
// It also reports the DIRECT conditioning offset (rel-L2 between the bf16 and fp32 conditioner outputs,
// pre-DiT) — the pure bf16-vs-fp32 weight-precision difference before any trajectory amplification.
// ------------------------------------------------------------------------------------------------
#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot + Metal; SLOW (real 2B denoise x2 per variant)"]
fn stage7_bf16_conditioning_offset_sc10577() {
    let split = split_files().expect("Anima snapshot");
    let g = load_golden("e2e_golden.json");
    let prompt = g["meta"]["prompt"].as_str().unwrap();
    let negative = g["meta"]["negative"].as_str().unwrap();
    let init_shape: Vec<i32> = i64s(&g["meta"]["init"]["shape"])
        .iter()
        .map(|&x| x as i32)
        .collect();
    let seed = g["meta"]["init"]["seed"].as_u64().unwrap();
    let n: usize = init_shape.iter().product::<i32>() as usize;
    let init = Array::from_slice(&gauss_fill(n, seed), &init_shape[..]);
    let source = WeightsSource::Dir(split.clone());

    let variants = g["variants"].as_object().expect("variants object");
    println!(
        "\n=== sc-10577 bf16-conditioning offset isolation (fp32-TE vs bf16-TE stage-7 residual) ==="
    );
    // Collect every variant's numbers and print them BEFORE asserting, so the full measurement is
    // legible even if a bound fires.
    let mut rows: Vec<(String, f64, f64, f64, f64)> = Vec::new();
    for (id, v) in variants {
        let variant = Variant::from_id(id).unwrap_or_else(|| panic!("unknown variant {id}"));
        let steps = v["steps"].as_u64().unwrap() as usize;
        let guidance = v["guidance"].as_f64().unwrap() as f32;
        let pipeline = AnimaPipeline::from_source(&source, variant).unwrap();

        // Shipped bf16-weight conditioning.
        let cond_bf16 = pipeline.encode_prompt(prompt).unwrap();
        let uncond_bf16 = variant
            .uses_cfg()
            .then(|| pipeline.encode_prompt(negative).unwrap());

        // fp32-upcast reference conditioning (TE + conditioner weights → fp32, mirroring `.float()`).
        let (te32, cond_mod32) =
            mlx_gen_anima::loader::load_conditioning_at_dtype(&source, variant, Dtype::Float32)
                .unwrap();
        let cond_fp32 = pipeline
            .encode_prompt_with(prompt, &te32, &cond_mod32)
            .unwrap();
        let uncond_fp32 = variant.uses_cfg().then(|| {
            pipeline
                .encode_prompt_with(negative, &te32, &cond_mod32)
                .unwrap()
        });

        // (A) DIRECT conditioning offset — the pure bf16-vs-fp32 conditioner-output difference (pre-DiT).
        let cond_offset = rel_l2_full(&cond_bf16, &cond_fp32);

        // (B) End-to-end: identical injected init + Euler + schedule, DiT fp32, conditioning bf16 vs fp32.
        let latent_bf16 = pipeline
            .denoise_from_latent_with_conditioning(
                &init,
                &cond_bf16,
                uncond_bf16.as_ref(),
                steps,
                guidance,
                "euler",
                Dtype::Float32,
            )
            .unwrap();
        let latent_fp32 = pipeline
            .denoise_from_latent_with_conditioning(
                &init,
                &cond_fp32,
                uncond_fp32.as_ref(),
                steps,
                guidance,
                "euler",
                Dtype::Float32,
            )
            .unwrap();

        // Residual of each MLX run vs the fp32 diffusers reference golden (the committed sample set).
        let rel_bf16 = sampled_rel_l2_value(&latent_bf16, &v["final_latent"]);
        let rel_fp32 = sampled_rel_l2_value(&latent_fp32, &v["final_latent"]);
        // Fraction of the bf16-TE residual that matching the reference's conditioning precision removed
        // (NEGATIVE ⇒ it made the residual WORSE), and the direct bf16-vs-fp32 final-latent delta.
        let removed = rel_bf16 - rel_fp32;
        let pct = 100.0 * removed / rel_bf16.max(1e-12);
        let latent_delta = rel_l2_full(&latent_bf16, &latent_fp32);

        println!(
            "[sc10577 {id}] cond-offset(bf16 vs fp32) = {cond_offset:.3e} | final rel-L2 vs fp32-ref: \
             bf16-TE = {rel_bf16:.4e}, fp32-TE = {rel_fp32:.4e} | removed = {removed:.4e} ({pct:.1}% of the residual) | \
             bf16-vs-fp32 latent Δ = {latent_delta:.3e}"
        );

        rows.push((id.clone(), cond_offset, rel_bf16, rel_fp32, latent_delta));
        mlx_rs::memory::clear_cache();
    }

    // The MEASURED conclusion (do not tune toward a predetermined answer — sc-10577):
    //   * The DIRECT bf16-vs-fp32 conditioner-output offset is TINY (~1.3e-3 across all variants), and
    //   * matching the reference's conditioning precision changes the stage-7 residual by only ~±10% —
    //     it never collapses it: base 7.8e-2 → 8.5e-2 (+9%), aesthetic 8.0e-2 → 8.9e-2 (+11%), turbo
    //     3.4e-2 → 3.0e-2 (−10%). fp32-TE HURTS base/aesthetic slightly and HELPS turbo slightly — the
    //     ~3e-2 conditioning-propagation perturbation is roughly orthogonal to the MLX-vs-reference gap.
    // ⇒ the bf16-conditioning offset is NOT the dominant term in the ~7.8e-2 residual; the residual is
    //   dominated by cross-backend (Metal-vs-MPS) DiT/VAE float accumulation, independent of the
    //   conditioning precision. This CONFIRMS (and, via the mixed-sign near-null result, strengthens)
    //   sc-10524's accumulation-dominated inference. The bounds below LOCK that finding: they fire if a
    //   future change ever makes the conditioning precision the dominant term (flipping the conclusion).
    for (id, cond_offset, rel_bf16, rel_fp32, _delta) in &rows {
        // The raw conditioning offset is small — the bf16 conditioner is close to the fp32 one.
        assert!(
            *cond_offset < 5e-3,
            "sc10577 {id}: direct bf16-vs-fp32 conditioning offset {cond_offset:.3e} unexpectedly large"
        );
        // Both MLX runs land in the reference ballpark (structural tripwire — the 1.2e-1 stage-7 uses).
        assert!(
            *rel_bf16 < 1.2e-1 && *rel_fp32 < 1.2e-1,
            "sc10577 {id}: a final rel-L2 (bf16 {rel_bf16:.3e} / fp32 {rel_fp32:.3e}) blew the 1.2e-1 tripwire"
        );
        // THE FINDING: fp32-TE removes < half the residual ⇒ bf16-conditioning is NOT dominant. (If a
        // future change made conditioning dominant, fp32-TE would collapse the residual and this fires.)
        let removed_frac = (rel_bf16 - rel_fp32) / rel_bf16.max(1e-12);
        assert!(
            removed_frac < 0.5,
            "sc10577 {id}: fp32-TE removed {:.0}% of the residual — bf16-conditioning became dominant; \
             re-examine the accumulation-vs-conditioning conclusion",
            100.0 * removed_frac
        );
    }
    println!(
        "CONCLUSION: bf16-conditioning is NOT the dominant term — the fp32-TE variant changes the \
         stage-7 residual by only ~±10% (base/aesthetic +9..11%, turbo -10%; never collapses it), and \
         the direct conditioning offset is ~1.3e-3. The ~7.8e-2 residual is cross-backend \
         accumulation-dominated (confirms sc-10524)."
    );
    println!("=== end sc-10577 ===\n");
}

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot + Metal; SLOW (real 2B denoise x3 variants)"]
fn stage7_e2e_all_variants() {
    let split = split_files().expect("Anima snapshot");
    let g = load_golden("e2e_golden.json");
    let prompt = g["meta"]["prompt"].as_str().unwrap();
    let negative = g["meta"]["negative"].as_str().unwrap();
    let init_shape: Vec<i32> = i64s(&g["meta"]["init"]["shape"])
        .iter()
        .map(|&x| x as i32)
        .collect();
    let seed = g["meta"]["init"]["seed"].as_u64().unwrap();
    let n: usize = init_shape.iter().product::<i32>() as usize;
    let init = Array::from_slice(&gauss_fill(n, seed), &init_shape[..]);

    let out_dir = std::path::PathBuf::from(
        std::env::var("ANIMA_STAGE7_OUT").unwrap_or_else(|_| "/tmp/anima_sc10524_stage7".into()),
    );
    std::fs::create_dir_all(&out_dir).ok();

    let variants = g["variants"].as_object().expect("variants object");
    assert_eq!(
        variants.len(),
        3,
        "expected all three variants in the golden"
    );
    for (id, v) in variants {
        let variant = Variant::from_id(id).unwrap_or_else(|| panic!("unknown variant {id}"));
        let steps = v["steps"].as_u64().unwrap() as usize;
        let guidance = v["guidance"].as_f64().unwrap() as f32;
        let pipeline =
            AnimaPipeline::from_source(&WeightsSource::Dir(split.clone()), variant).unwrap();

        // Identical injected init + deterministic Euler + identical schedule, fp32 both sides. Capture the
        // latent after 1 and 5 Euler steps so the accumulation-vs-bias question is TESTABLE, not asserted
        // by prose (sc-10524 review).
        let (latent, caps) = pipeline
            .denoise_from_latent_capture(
                &init,
                prompt,
                negative,
                variant,
                steps,
                guidance,
                "euler",
                Dtype::Float32,
                &[1, 5],
            )
            .unwrap();

        // Intermediate-step parity — makes the accumulation-vs-bias question TESTABLE, not asserted by
        // prose (sc-10524 review). Two effects could inflate the final residual:
        //   (1) a FIXED bias — the MLX Qwen3+conditioner encode is bf16-locked (text_encoder.rs forces
        //       Bfloat16; the `dtype` arg reaches only `dit.forward`) while this reference runs fp32, so a
        //       bf16-conditioning offset enters v at every step. A large bias shows up as a step-1 rel-L2
        //       that is already a big fraction of the final; and
        //   (2) diffuse float ACCUMULATION over the 30 (10 for turbo) Euler steps, amplified in the stiff
        //       low-σ endgame, which grows super-linearly with step count.
        // MEASURED (this checkpoint): step-1 rel-L2 ~2e-5, step-5 ~2e-4 (turbo 3e-3), final ~3–8e-2 — the
        // early deltas are 3–4 orders of magnitude BELOW the final and grow super-linearly, so the residual
        // is accumulation-dominated (endgame amplification); the bf16-conditioning offset is present but a
        // MINOR per-step contributor, not a dominant bias. This is now DIRECTLY MEASURED, not just inferred:
        // sc-10577's `stage7_bf16_conditioning_offset_sc10577` (this file) reruns the identical stage-7
        // denoise with an fp32-upcast TE + conditioner (via `loader::load_conditioning_at_dtype`) and finds
        // the direct bf16-vs-fp32 conditioner-output offset is only ~1.4e-3 and that matching the reference's
        // conditioning precision does NOT reduce the final residual (base 7.8e-2 → 8.5e-2) — so the residual
        // is cross-backend accumulation-dominated, NOT bf16-conditioning-dominated. See that test for the
        // full per-variant numbers.
        //
        // The intermediate bounds must actually LOCK the measured value — the whole reason we capture the
        // step latents is to distinguish accumulation from a fixed bias, and the final's loose structural
        // tripwire (rel-L2 1.2e-1) would let a ~12% bias sail through here, defeating that (sc-10524
        // review). So each step is gated a few× above its MEASURED value, not at the final's bound:
        //   step-1: rel-L2 1e-3 (≈50× over the measured 2e-5; ~120× tighter than the old 1.2e-1);
        //   step-5: rel-L2 1e-2 (covers the turbo-variant worst case ~3e-3 with headroom; ~12× tighter).
        // Aggregate stats are locked at 5e-3 (>2.5× over the final's measured <0.2%; steps are closer to
        // the reference than the final, so this holds with margin). If a real conditioning bias ever
        // emerged, step-1 would blow past 1e-3 here long before the final tripwire noticed.
        let steps_g = v["step_latents"].as_object().expect("step_latents object");
        assert_eq!(
            caps.len(),
            steps_g.len(),
            "stage7_{id}: captured step count"
        );
        let mut step_rel: Vec<(usize, f64)> = Vec::new();
        for (k, arr) in &caps {
            let g_step = &steps_g[&k.to_string()];
            let rel_bound = match k {
                1 => 1e-3,
                _ => 1e-2, // step-5 (and any later capture): covers the turbo ~3e-3 worst case
            };
            let r = assert_sampled_rel_l2(
                arr,
                g_step,
                &format!("stage7_{id}_step{k}"),
                5e-3,
                rel_bound,
            );
            step_rel.push((*k, r));
        }

        // Final-latent parity (pre-VAE). The distribution/trajectory match is the parity signal: with an
        // identical injected init + deterministic Euler + identical schedule, the aggregate stats
        // (mean/std/l2) match to <0.2% (tight gate). The element-wise relative-L2 carries float
        // accumulation (amplified in the low-σ endgame; see the step-1/step-5 deltas above) plus a minor
        // bf16-conditioning offset — the per-step DiT parity is 1e-2 (stage 4), so this is accumulation,
        // not a structural bug; the `max_rel_l2` is a generous structural-bug tripwire (a wrong sign / CFG
        // / RoPE diverges by >>0.5). The IMAGE below is the real visual-indistinguishability gate.
        // Measured (fp32, this checkpoint): aggregate mean/std/l2 within <0.2%; sampled rel-L2 3.4e-2
        // (turbo) .. 8.0e-2 (base/aesthetic). Bounds: aggregate 2e-2 (tight); rel-L2 1.2e-1 (structural
        // tripwire — a wrong sign/CFG/RoPE blows past this AND the image gate by orders of magnitude).
        let final_rel = assert_sampled_rel_l2(
            &latent,
            &v["final_latent"],
            &format!("stage7_{id}_latent"),
            2e-2,
            1.2e-1,
        );
        println!(
            "[stage7_{id}] rel-L2 by step {:?} -> final {final_rel:.4e} \
             (early deltas <<final and super-linear => accumulation/endgame-amplification, not a fixed bias)",
            step_rel
                .iter()
                .map(|(k, r)| format!("{k}:{r:.4e}"))
                .collect::<Vec<_>>()
        );

        // Decoded-image parity (the "visually indistinguishable" check): MLX QwenVae.decode applies the
        // same latent*std+mean de-norm the reference does, so the same latent decodes to the same image.
        let decoded = pipeline.components().vae.decode(&latent).unwrap();
        let img = decoded_to_image(&decoded).unwrap();
        save_png(
            &img.pixels,
            img.width,
            img.height,
            &out_dir.join(format!("{id}_mlx.png")),
        );
        // The visual-indistinguishability gate. Measured: per-channel mean Δ <=0.58 levels, sampled
        // pixel MAE 0.50 (turbo) .. 3.77 (aesthetic, whose faint background text amplifies sub-pixel
        // shifts). Bounds 2.0 / 7.0 leave margin for Metal run-to-run nondeterminism; a structural port
        // bug produces a visibly different image (MAE >> 20). PNG pairs verified visually indistinguishable.
        assert_image_matches(
            &img.pixels,
            &v["image"],
            &format!("stage7_{id}_image"),
            2.0,
            7.0,
        );

        mlx_rs::memory::clear_cache();
    }
}
