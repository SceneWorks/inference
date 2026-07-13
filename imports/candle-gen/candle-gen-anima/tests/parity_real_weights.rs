//! Anima candle-port parity goldens (sc-10525, epic 10512) — the **real-weights** stages, validated on
//! candle's **CPU** backend (no NVIDIA GPU required). `#[ignore]`d + weights-gated (they need the
//! licensed `circlestone-labs/Anima` single-file snapshot in the HF cache), so they never run in CI.
//! Run locally with:
//!   cargo test -p candle-gen-anima --release --test parity_real_weights -- --ignored --nocapture
//!
//! Each test reads a committed golden JSON (computed by the MLX lane's diffusers 0.39.0 generators —
//! framework-INDEPENDENT numbers) and runs the candle port on the single-file checkpoint on CPU, then
//! compares the aggregate stats (mean/std/l2 — the structural-correctness gate a real port bug moves by
//! orders of magnitude) + a deterministic sub-sample.
//!
//!   * **Stage 2** — Qwen3-0.6B `last_hidden_state` AFTER the attention-mask multiply (GQA 16/8 +
//!     the 6-pad-row mask-multiply trap: 18 real + 6 right-pad rows at mask 0, so the multiply zeros
//!     real padded rows — the pad-check bites, not a vacuous empty slice). f32 candle vs bf16 torch.
//!   * **Stage 3** — `AnimaTextConditioner` output `[1, 512, 1024]`, right-padded after masking. f32 both.
//!   * **Stage 4** — Cosmos DiT full forward (velocity `[1, 16, 1, 8, 12]` — NON-SQUARE post-patch grid,
//!     so an h/w RoPE axis swap is detectable). f32 both. Exercises adaLN-LoRA modulation, NTK-scaled
//!     3-axis RoPE, the 17-ch mask concat, patch/unpatch.
//!
//! Stages 3 & 4 feed DETERMINISTIC `lcg_fill` inputs (bit-identical to the Python generators), so the
//! golden isolates the component's fp32 math rather than bf16 quantization.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec, WeightsSource};
use serde_json::Value;

use candle_gen_anima::adapters::apply_anima_adapters;
use candle_gen_anima::conditioner::AnimaTextConditioner;
use candle_gen_anima::config::{ConditionerConfig, DitConfig, Qwen3Config, Variant};
use candle_gen_anima::loader::detect_dit_prefix;
use candle_gen_anima::text_encoder::AnimaQwen3;
use candle_gen_anima::tokenizer::AnimaTokenizers;
use candle_gen_anima::transformer::CosmosDiT;
use candle_gen_anima::AnimaComponents;

// ------------------------------------------------------------------------------------------------
// Shared helpers.
// ------------------------------------------------------------------------------------------------

/// Glob the Anima snapshot's `split_files/` dir from the HF cache (no hardcoded sha).
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

/// Portable LCG in [-1, 1) — **bit-identical** to the Python generator's `lcg_fill`.
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

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn flatten_f32(t: &Tensor) -> Vec<f64> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(|x| x as f64)
        .collect()
}

/// The aggregate-stats parity gate: recompute mean/std/l2 and require they match the golden within
/// `stat_rtol` (a real port bug — wrong sign / dropped mask / mis-scaled RoPE — shifts these by orders
/// of magnitude; bf16/fp32 reduced precision does not). Also prints the sampled values for inspection.
fn assert_stats(got: &Tensor, g: &Value, label: &str, stat_rtol: f64) -> Vec<f64> {
    let want_shape: Vec<usize> = i64s(&g["shape"]).iter().map(|&x| x as usize).collect();
    assert_eq!(got.dims(), &want_shape[..], "{label}: shape");
    let flat = flatten_f32(got);
    let count = g["count"].as_u64().unwrap() as usize;
    assert_eq!(flat.len(), count, "{label}: element count");

    let mean = flat.iter().sum::<f64>() / count as f64;
    let var = flat.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count as f64;
    let std = var.sqrt();
    let l2 = flat.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let (gmean, gstd, gl2) = (
        g["mean"].as_f64().unwrap(),
        g["std"].as_f64().unwrap(),
        g["l2"].as_f64().unwrap(),
    );
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-6);
    println!(
        "[{label}] mean {mean:.5} (g {gmean:.5}), std {std:.5} (g {gstd:.5}), l2 {l2:.4} (g {gl2:.4})"
    );
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

/// Sampled relative-L2 over the golden's `sample_indices` — the robust no-quality-regression metric.
fn assert_sampled_rel_l2(flat: &[f64], g: &Value, label: &str, tol: f64) {
    let idx = i64s(&g["sample_indices"]);
    let want = f64s(&g["sample_values"]);
    assert_eq!(idx.len(), want.len(), "{label}: sample len");
    let mut num = 0.0;
    let mut den = 0.0;
    for (k, &i) in idx.iter().enumerate() {
        let got = flat[i as usize];
        num += (got - want[k]).powi(2);
        den += want[k].powi(2);
    }
    let rel_l2 = (num / den.max(1e-12)).sqrt();
    println!("[{label}] sampled rel-L2 = {rel_l2:.4e} (n={})", idx.len());
    assert!(
        rel_l2 < tol,
        "{label}: sampled rel-L2 {rel_l2} exceeds {tol}"
    );
}

fn cpu_vb(path: &Path) -> candle_gen::candle_nn::VarBuilder<'static> {
    let files = [path.to_path_buf()];
    candle_gen::mmap_var_builder(&files, DType::F32, &Device::Cpu).expect("mmap var builder")
}

// ------------------------------------------------------------------------------------------------
// Stage 2 — Qwen3-0.6B last_hidden_state AFTER the mask-multiply (GQA 16/8 + 6 padded rows).
// ------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the licensed Anima snapshot in the HF cache"]
fn stage2_qwen3_last_hidden_state() {
    let Some(split) = split_files() else {
        eprintln!("stage2: Anima snapshot absent -> skip");
        return;
    };
    let g = load_golden("qwen3_hidden_golden.json");
    let want_ids = i64s(&g["meta"]["qwen_ids"]);
    // Format-robust: newer goldens (sc-10524 review) right-pad the input with K pad rows at mask 0 and
    // carry `real_tokens`/`padded_tokens`; older committed goldens are the real prompt only (K=0).
    let real = g["meta"]["real_tokens"]
        .as_u64()
        .map(|x| x as usize)
        .unwrap_or(want_ids.len());
    let pad_k = g["meta"]["padded_tokens"]
        .as_u64()
        .map(|x| x as usize)
        .unwrap_or(0);

    let vb = cpu_vb(&split.join("text_encoders/qwen_3_06b_base.safetensors"));
    let te = AnimaQwen3::new(&vb.pp("model"), &Qwen3Config::anima()).unwrap();
    let tk = AnimaTokenizers::load().unwrap();
    let (real_ids, real_mask) = tk
        .encode_qwen(g["meta"]["prompt"].as_str().unwrap())
        .unwrap();
    let real_ids_i64: Vec<i64> = real_ids.iter().map(|&x| x as i64).collect();
    assert_eq!(
        &real_ids_i64[..],
        &want_ids[..real],
        "stage2: real Qwen ids drifted"
    );

    // Right-pad with `pad_k` Qwen2-pad tokens at mask 0 (the mask-multiply trap gets real rows to zero).
    const PAD_ID: i32 = 151643;
    let mut ids = real_ids.clone();
    let mut mask = real_mask.clone();
    ids.extend(std::iter::repeat_n(PAD_ID, pad_k));
    mask.extend(std::iter::repeat_n(0i32, pad_k));
    let ids_i64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    assert_eq!(
        ids_i64, want_ids,
        "stage2: padded Qwen ids must match the golden"
    );

    let s = ids.len();
    let ids_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
    let input_ids = Tensor::from_vec(ids_u32, (1, s), &Device::Cpu).unwrap();
    let hidden = te.forward(&input_ids, DType::F32).unwrap(); // [1, S, 1024]

    // The mask-multiply trap (now non-trivial): zero the `pad_k` padded rows.
    let mask_f: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
    let m = Tensor::from_vec(mask_f, (1, s, 1), &Device::Cpu).unwrap();
    let hidden = hidden.broadcast_mul(&m).unwrap();

    // Padded rows [real:] must be EXACTLY zero after the multiply.
    let flat = flatten_f32(&hidden);
    let pad_abs_max = flat[real * 1024..]
        .iter()
        .fold(0f64, |m, &v| m.max(v.abs()));
    println!("[stage2] pad rows [{real}:] abs-max = {pad_abs_max:.2e}");
    assert!(
        pad_abs_max < 1e-4,
        "stage2: mask-multiply must zero the {pad_k} padded rows"
    );

    // f32 candle vs bf16 torch reference: gate on aggregate stats (tight) + a looser sampled rel-L2.
    let flat = assert_stats(&hidden, &g["last_hidden_state"], "stage2_qwen3", 2e-2);
    assert_sampled_rel_l2(&flat, &g["last_hidden_state"], "stage2_qwen3", 5e-2);
}

// ------------------------------------------------------------------------------------------------
// Stage 3 — AnimaTextConditioner output, right-padded to 512 (fp32 both sides).
// ------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the licensed Anima snapshot in the HF cache"]
fn stage3_conditioner_output() {
    let Some(split) = split_files() else {
        eprintln!("stage3: Anima snapshot absent -> skip");
        return;
    };
    let g = load_golden("conditioner_golden.json");
    let st = g["meta"]["st"].as_u64().unwrap() as usize;
    let t5_ids: Vec<u32> = i64s(&g["meta"]["t5_ids"])
        .iter()
        .map(|&x| x as u32)
        .collect();
    let src_shape: Vec<usize> = i64s(&g["meta"]["lcg"]["source_shape"])
        .iter()
        .map(|&x| x as usize)
        .collect();

    let dit = dit_file(&split);
    let prefix = detect_dit_prefix(&dit).unwrap();
    let vb = cpu_vb(&dit);
    let cond = AnimaTextConditioner::new(
        &vb.pp(&prefix).pp("llm_adapter"),
        ConditionerConfig::anima(),
    )
    .unwrap();

    let n: usize = src_shape.iter().product();
    let source = Tensor::from_vec(
        lcg_fill(n, 3),
        (src_shape[0], src_shape[1], src_shape[2]),
        &Device::Cpu,
    )
    .unwrap();
    let target = Tensor::from_vec(t5_ids, (1, st), &Device::Cpu).unwrap();
    let out = cond.forward(&source, &target, DType::F32).unwrap();
    assert_eq!(
        out.dims(),
        &[1, 512, 1024],
        "stage3: must right-pad to 512 tokens"
    );

    // Right-padded rows [st:512] must be exactly the zero pad the DiT expects.
    let flat = flatten_f32(&out);
    let pad_abs_max = flat[st * 1024..].iter().fold(0f64, |m, &v| m.max(v.abs()));
    println!("[stage3] pad rows [{st}:512] abs-max = {pad_abs_max:.2e}");
    assert!(
        pad_abs_max < 1e-4,
        "stage3: rows past the real tokens must be zero padding"
    );

    assert_stats(&out, &g["full"], "stage3_full", 1e-2);
    let active = out.narrow(1, 0, st).unwrap();
    let flat_active = assert_stats(&active, &g["active"], "stage3_active", 1e-2);
    assert_sampled_rel_l2(&flat_active, &g["active"], "stage3_active", 1e-2);
}

// ------------------------------------------------------------------------------------------------
// Stage 4 — Cosmos DiT full forward: the final velocity (fp32 both sides).
// ------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the licensed Anima snapshot in the HF cache"]
fn stage4_dit_full_velocity() {
    let Some(split) = split_files() else {
        eprintln!("stage4: Anima snapshot absent -> skip");
        return;
    };
    let g = load_golden("dit_forward_golden.json");
    let lat_shape: Vec<usize> = i64s(&g["meta"]["latent_shape"])
        .iter()
        .map(|&x| x as usize)
        .collect();
    let enc_shape: Vec<usize> = i64s(&g["meta"]["encoder_shape"])
        .iter()
        .map(|&x| x as usize)
        .collect();
    let sigma = g["meta"]["lcg"]["sigma"].as_f64().unwrap() as f32;

    let dit_path = dit_file(&split);
    let prefix = detect_dit_prefix(&dit_path).unwrap();
    let vb = cpu_vb(&dit_path);
    let dit = CosmosDiT::new(&vb.pp(&prefix), DitConfig::anima()).unwrap();

    let ln: usize = lat_shape.iter().product();
    let en: usize = enc_shape.iter().product();
    let latent = Tensor::from_vec(
        lcg_fill(ln, 1),
        (
            lat_shape[0],
            lat_shape[1],
            lat_shape[2],
            lat_shape[3],
            lat_shape[4],
        ),
        &Device::Cpu,
    )
    .unwrap();
    let encoder = Tensor::from_vec(
        lcg_fill(en, 2),
        (enc_shape[0], enc_shape[1], enc_shape[2]),
        &Device::Cpu,
    )
    .unwrap();
    let s = Tensor::from_vec(vec![sigma], (1,), &Device::Cpu).unwrap();

    let v = dit.forward(&latent, &s, &encoder, DType::F32).unwrap();
    let flat = assert_stats(&v, &g["full"], "stage4_dit_full", 1e-2);
    assert_sampled_rel_l2(&flat, &g["full"], "stage4_dit_full", 2e-2);
}

// ------------------------------------------------------------------------------------------------
// Loader — detect_dit_prefix over the real base checkpoint (proves the anchor + prefix split).
// ------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the licensed Anima snapshot in the HF cache"]
fn loader_detects_base_prefix_and_assembles() {
    let Some(split) = split_files() else {
        eprintln!("loader: Anima snapshot absent -> skip");
        return;
    };
    // The base cut roots at `net`; assembling the full component set proves every key resolves.
    let prefix = detect_dit_prefix(&dit_file(&split)).unwrap();
    assert_eq!(prefix, "net", "base variant roots at `net`");
    let comps = AnimaComponents::load(&WeightsSource::Dir(split), Variant::Base, &Device::Cpu, &[])
        .unwrap();
    assert_eq!(comps.dtype, DType::F32, "CPU compute dtype is f32");
    let _ = comps.dit.config();
}

// ------------------------------------------------------------------------------------------------
// LoRA — the real Anima-Official-LoRAs files: 508 (448 DiT + 60 conditioner) vs 448 DiT-only, folded
// onto the real base DiT+conditioner, on CPU. Weight-level property; no GPU. (sc-10525 / MLX sc-10521.)
// ------------------------------------------------------------------------------------------------

use std::collections::HashMap;

/// Glob one Anima-Official-LoRAs file from the HF cache (no hardcoded sha).
fn lora_file(name: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--circlestone-labs--Anima-Official-LoRAs/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let p = e.path().join(name);
            p.is_file().then_some(p)
        })
}

/// Load the real base DiT single-file into a CPU `HashMap` (f32 for the fold), like the loader's
/// adapter path.
fn base_map(dit: &Path) -> HashMap<String, candle_gen::candle_core::Tensor> {
    candle_gen::candle_core::safetensors::load(dit, &Device::Cpu).unwrap()
}

fn spec(path: PathBuf) -> AdapterSpec {
    AdapterSpec::new(path, 1.0, AdapterKind::Lora)
}

#[test]
#[ignore = "needs the licensed Anima snapshot + Anima-Official-LoRAs in the HF cache"]
fn lora_turbo_508_and_style_448_fold_onto_the_real_base() {
    let Some(split) = split_files() else {
        eprintln!("lora: Anima snapshot absent -> skip");
        return;
    };
    let (Some(turbo), Some(style)) = (
        lora_file("anima-turbo-lora-v0.2.safetensors"),
        lora_file("anima-greg-rutkowski-style.safetensors"),
    ) else {
        eprintln!("lora: Anima-Official-LoRAs absent -> skip");
        return;
    };
    let dit = dit_file(&split);
    let prefix = detect_dit_prefix(&dit).unwrap();

    // Snapshot a DiT target and a conditioner target to prove per-class fold behavior.
    let dit_key = format!("{prefix}.blocks.0.self_attn.q_proj.weight");
    let cond_key = format!("{prefix}.llm_adapter.blocks.0.self_attn.q_proj.weight");
    let max_abs = |t: &candle_gen::candle_core::Tensor| {
        t.to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    let diff_max = |a: &candle_gen::candle_core::Tensor, b: &candle_gen::candle_core::Tensor| {
        (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };

    // --- anima-turbo-lora-v0.2: 508 targets (448 DiT + 60 conditioner). ---
    let mut base = base_map(&dit);
    let before_dit = base[&dit_key].clone();
    let before_cond = base[&cond_key].clone();
    let report = apply_anima_adapters(&mut base, &prefix, &[spec(turbo.clone())]).unwrap();
    assert_eq!(
        report.merged, 508,
        "anima-turbo-lora-v0.2 must fold 508 targets (448 DiT + 60 conditioner), got {}",
        report.merged
    );
    // The DiT target changed (non-zero delta); the conditioner target is unchanged (all 60 conditioner
    // lora_B are zero-init in this file, so B·A ≡ 0 — inert, but the count above proves it still ROUTED).
    assert!(
        diff_max(&base[&dit_key], &before_dit) > 1e-6,
        "DiT target must change"
    );
    assert!(
        diff_max(&base[&cond_key], &before_cond) < 1e-6,
        "conditioner target has zero lora_B → unchanged (but must still be routed in the 508 count)"
    );
    println!(
        "[lora_turbo] 508 routed; DiT Δmax {:.3e}, cond Δmax {:.3e} (zero-B)",
        diff_max(&base[&dit_key], &before_dit),
        diff_max(&base[&cond_key], &before_cond)
    );
    drop(base);

    // --- anima-greg-rutkowski-style: 448 DiT-only, zero conditioner. ---
    let mut base = base_map(&dit);
    let report = apply_anima_adapters(&mut base, &prefix, &[spec(style)]).unwrap();
    assert_eq!(
        report.merged, 448,
        "greg-rutkowski is 448 DiT-only, got {}",
        report.merged
    );
    drop(base);

    // --- Mutation (sc-10274): a DiT-only base (llm_adapter modules dropped) must REJECT the turbo
    // LoRA's 60 conditioner targets, not silently drop them. ---
    let mut base = base_map(&dit);
    base.retain(|k, _| !k.contains("llm_adapter"));
    let err = apply_anima_adapters(&mut base, &prefix, &[spec(turbo)])
        .expect_err("a DiT-only base must reject the 60 conditioner targets");
    assert!(
        err.to_string().contains("did not route"),
        "expected unrouted error, got: {err}"
    );
    let _ = max_abs; // (helper kept for symmetry with the stats readouts)
}
