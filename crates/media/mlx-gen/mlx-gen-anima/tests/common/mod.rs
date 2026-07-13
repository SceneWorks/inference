//! Shared real-weights fixtures + LoRA math/synthesis helpers for the mlx-gen-anima integration
//! tests (sc-10521). Included by both `tests/lora_injection.rs` and `tests/scale_convention.rs`.
//!
//! ## Why two binaries share this module
//! The scale-convention checks capture a target's base weight, run the heavy 508-target injection,
//! then re-read the same target's forward. That capture → heavy-inject → re-read pattern strands a
//! lazily-held array once enough prior injections have churned mlx-rs's single Metal default stream
//! ("There is no Stream(gpu, N)"). Running it 6th in the original 7-test `lora_injection` binary
//! reproduced exactly that panic (it passes in isolation). So the scale tests live in their own
//! process (`tests/scale_convention.rs`), precisely as sc-10515 moved `velocity_convention` out of
//! `real_weights.rs`. Keeping the fixtures here — instead of duplicating a 35-line safetensors writer
//! and the base-weight/rel-err machinery into both binaries — means the two can't silently drift.
//!
//! Each binary uses a subset of these helpers, so `dead_code` is allowed at the module level (a
//! `tests/common/mod.rs` is compiled into every including binary, and libtest would otherwise warn on
//! the unused remainder).
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use mlx_rs::ops::{add, matmul, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen::WeightsSource;

use mlx_gen_anima::config::Variant;
use mlx_gen_anima::loader::AnimaComponents;

// -------------------------------------------------------------------------------------------------
// Snapshot fixtures (glob the HF cache; no hardcoded sha)
// -------------------------------------------------------------------------------------------------

/// Glob the Anima base snapshot's `split_files/` dir (DiT checkpoints + VAE + TE).
pub fn split_files() -> Option<PathBuf> {
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

/// Glob the official-LoRAs snapshot dir.
pub fn lora_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--circlestone-labs--Anima-Official-LoRAs/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let p = e.path();
            p.join("anima-turbo-lora-v0.2.safetensors")
                .is_file()
                .then_some(p)
        })
}

pub fn turbo_lora() -> PathBuf {
    lora_dir()
        .expect("Anima LoRA snapshot")
        .join("anima-turbo-lora-v0.2.safetensors")
}

pub fn style_lora() -> PathBuf {
    lora_dir()
        .expect("Anima LoRA snapshot")
        .join("anima-greg-rutkowski-style.safetensors")
}

pub fn load_base() -> AnimaComponents {
    let split = split_files().expect("Anima base snapshot");
    AnimaComponents::load(&WeightsSource::Dir(split), Variant::Base).expect("load base components")
}

pub fn lora_spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path, scale, AdapterKind::Lora)
}

// -------------------------------------------------------------------------------------------------
// Numeric helpers
// -------------------------------------------------------------------------------------------------

pub fn randn(shape: &[i32]) -> Array {
    random::normal::<f32>(shape, None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

/// L2 norm of `a` in f32 (via `Σ a·a`), avoiding any method-name ambiguity.
pub fn l2(a: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    multiply(&a, &a)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>()
        .sqrt()
}

/// `||got − want|| / ||want||`, in f32 — a scale-free relative error for bf16 comparisons.
pub fn rel_err(got: &Array, want: &Array) -> f32 {
    let diff = subtract(
        got.as_dtype(Dtype::Float32).unwrap(),
        want.as_dtype(Dtype::Float32).unwrap(),
    )
    .unwrap();
    l2(&diff) / l2(want).max(1e-12)
}

// -------------------------------------------------------------------------------------------------
// Scale-convention machinery (base-weight capture + injected forward vs `base + scale·B·A`)
// -------------------------------------------------------------------------------------------------

/// A target linear plus its raw-file LoRA key: `(route_into_conditioner, dotted_path, lora_key)`.
pub type Target = (bool, &'static [&'static str], &'static str);

pub const DIT_Q: Target = (
    false,
    &["blocks", "0", "self_attn", "q_proj"],
    "diffusion_model.blocks.0.self_attn.q_proj",
);
pub const DIT_ADALN: Target = (
    false,
    &["blocks", "0", "adaln_modulation_self_attn", "2"],
    "diffusion_model.blocks.0.adaln_modulation_self_attn.2",
);
pub const COND_Q: Target = (
    true,
    &["blocks", "0", "self_attn", "q_proj"],
    "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj",
);

/// The base `[out,in]` weight of a target BEFORE injection. Force-EVALUATED here (not left lazy): the
/// heavy 508-target `apply_anima_adapters` that runs between capture and use recreates the default
/// Metal stream, which would strand a lazily-held base weight ("no Stream(gpu, N)"). Materializing it
/// now pins the value.
pub fn base_weight(c: &mut AnimaComponents, t: &Target) -> Array {
    let (route_cond, path, _) = *t;
    let lin = if route_cond {
        c.conditioner.adaptable_mut(path)
    } else {
        c.dit.adaptable_mut(path)
    };
    let w = lin
        .expect("target linear")
        .dense_weight()
        .expect("dense base")
        .0
        .clone();
    mlx_rs::transforms::eval([&w]).unwrap();
    w
}

/// After injection, the relative error of the target's forward against `base(x) + scale·x·(B·A)ᵀ`
/// computed from the raw file factors — i.e. how close the effective merged weight is to `W + scale·B·A`
/// (no alpha/rank fold). `lw` is the already-loaded LoRA weights (loaded once per test).
pub fn injected_rel_err(
    c: &mut AnimaComponents,
    t: &Target,
    lw: &Weights,
    w_base: &Array,
    scale: f32,
) -> f32 {
    let (route_cond, path, key) = *t;
    let a_raw = lw.require(&format!("{key}.lora_A.weight")).unwrap(); // [r, in]
    let b_raw = lw.require(&format!("{key}.lora_B.weight")).unwrap(); // [out, r]
    let lin = if route_cond {
        c.conditioner.adaptable_mut(path)
    } else {
        c.dit.adaptable_mut(path)
    }
    .unwrap();
    let x = randn(&[1, 8, w_base.shape()[1]]);
    let got = lin.forward(&x).unwrap();
    // want = x·Wᵀ + scale·((x·Aᵀ)·Bᵀ)  (A[r,in]→Aᵀ[in,r]; B[out,r]→Bᵀ[r,out]).
    let base_out = matmul(&x, w_base.t()).unwrap();
    let resid = matmul(matmul(&x, a_raw.t()).unwrap(), b_raw.t()).unwrap();
    let resid = multiply(
        &resid,
        Array::from_f32(scale).as_dtype(resid.dtype()).unwrap(),
    )
    .unwrap();
    let want = add(&base_out, &resid).unwrap();
    rel_err(&got, &want)
}

// -------------------------------------------------------------------------------------------------
// Synthetic adapter files (raw safetensors bytes — no mlx involvement, so the file-build stays off
// the GPU and can't trip mlx-rs's shared-Metal-stream lifecycle the way constructing + evaluating a
// second batch of mlx arrays in a real-weights test binary would).
// -------------------------------------------------------------------------------------------------

/// Write a minimal valid F32 safetensors from `(name, shape)` entries with deterministic, non-trivial
/// factor values scaled by `amp` (never all-zero) and a string-valued `__metadata__` map (plus
/// `format: pt`).
pub fn write_raw_safetensors(
    path: &Path,
    entries: &[(&str, &[usize])],
    meta: &[(&str, &str)],
    amp: f32,
) {
    let mut data: Vec<u8> = Vec::new();
    let mut header = String::from("{");
    header.push_str("\"__metadata__\":{\"format\":\"pt\"");
    for (k, v) in meta {
        header.push_str(&format!(",\"{k}\":\"{v}\""));
    }
    header.push('}');
    for (i, (name, shape)) in entries.iter().enumerate() {
        let n: usize = shape.iter().product();
        let start = data.len();
        for j in 0..n {
            // deterministic bounded pseudo-values in ~[-amp/2, amp/2], never all-zero.
            let v = (((i * 131 + j * 17 + 7) % 101) as f32 / 101.0 - 0.5) * amp;
            data.extend_from_slice(&v.to_le_bytes());
        }
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            ",\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{dims}],\"data_offsets\":[{start},{}]}}",
            data.len()
        ));
    }
    header.push('}');
    let header_bytes = header.into_bytes();
    let mut buf = (header_bytes.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&data);
    std::fs::write(path, buf).unwrap();
}

/// Synthesize a peft-format LoKr `.safetensors` (`networkType=lokr`, `rank`/`alpha` metadata, full
/// Kronecker factors `lokr_w1`/`lokr_w2`) targeting one DiT and one conditioner module, and return its
/// path. `alpha == rank` ⇒ scale 1.0 (PEFT). No official Anima LoKr exists, so this is a hand-built
/// LoKr proving the path end to end; `a·b == out`, `c·d == in` per target.
pub fn synth_lokr() -> PathBuf {
    let dir = std::env::temp_dir().join("anima_sc10521_lokr");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("anima_synth_lokr.safetensors");
    // DiT blocks.1.self_attn.k_proj: W [2048,2048]; w1 [64,64] ⊗ w2 [32,32] → [2048,2048].
    // Conditioner llm_adapter.blocks.1.self_attn.k_proj: W [1024,1024]; w1 [32,32] ⊗ w2 [32,32].
    let entries: &[(&str, &[usize])] = &[
        ("blocks.1.self_attn.k_proj.lokr_w1", &[64, 64]),
        ("blocks.1.self_attn.k_proj.lokr_w2", &[32, 32]),
        ("llm_adapter.blocks.1.self_attn.k_proj.lokr_w1", &[32, 32]),
        ("llm_adapter.blocks.1.self_attn.k_proj.lokr_w2", &[32, 32]),
    ];
    write_raw_safetensors(
        &path,
        entries,
        &[("networkType", "lokr"), ("rank", "32"), ("alpha", "32")],
        0.04,
    );
    path
}

/// Synthesize a peft-format **LoRA** with a deliberately **NON-ZERO** `lora_B` targeting one
/// conditioner module (`diffusion_model.llm_adapter.blocks.0.self_attn.q_proj`), and return its path.
/// Metadata is `{"format":"pt"}` only — no `.alpha` tensor and no `lora_alpha`/`alpha_pattern` blob —
/// so the loader takes the no-fold branch (α = rank ⇒ scale 1.0), identical to the shipped Anima
/// LoRAs. `amp = 0.2` makes `|B·A·x|` a sizeable fraction (~half) of `|W·x|` so the scale-1.0 vs
/// scale-0.5 residuals are cleanly separable (measured: `rel_err` vs the wrong half-scale ≈ 0.1–0.2,
/// vs the matching scale ≈ 0) — this is the fixture that makes the conditioner scale assertion
/// NON-vacuous (unlike the turbo LoRA, whose 60 conditioner `lora_B` are all zero-initialized, so its
/// conditioner leg could not distinguish any scale from any other). q_proj is `[1024, 1024]` (16 heads
/// × 64 head_dim); rank 8.
pub fn synth_conditioner_lora() -> PathBuf {
    let dir = std::env::temp_dir().join("anima_sc10521_cond_lora");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("anima_synth_conditioner_lora.safetensors");
    let entries: &[(&str, &[usize])] = &[
        (
            "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight",
            &[8, 1024],
        ),
        (
            "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_B.weight",
            &[1024, 8],
        ),
    ];
    write_raw_safetensors(&path, entries, &[], 0.2);
    path
}
