//! LoRA/LoKr injection tests for mlx-gen-anima (sc-10521). All `#[ignore]`d and real-weights-gated —
//! they need BOTH the `circlestone-labs/Anima` base snapshot (DiT checkpoints) and the
//! `circlestone-labs/Anima-Official-LoRAs` snapshot in the HF cache, plus Metal. Run with:
//!   cargo test -p mlx-gen-anima --release --test lora_injection -- --ignored --nocapture
//!
//! CI runs NONE of these (no weights); the injection *math* (PEFT LoRA / LoKr install, alpha/rank
//! fold, stacking) is covered in CI by the shared core `src/adapters/loader.rs` unit tests, and the
//! anima capability advertisement (`supports_lora`/`supports_lokr`) by `model.rs::descriptors_surface`.
//!
//! ## Why these tests, and not the story's "turbo reproduction"
//! The story proposed proving injection + scale by reproducing the merged `anima-turbo-v1.0` checkpoint
//! from `anima-base-v1.0` + `anima-turbo-lora-v0.2`. **That premise is false against the shipped
//! weights** (measured, [`turbo_checkpoint_is_not_base_plus_lora`]): the LoRA weight delta is
//! ORTHOGONAL (|cos| < 0.05) to `anima-turbo − anima-base`, `|anima-turbo − anima-base|` is ~10× the
//! LoRA delta's magnitude, and the conditioner is byte-identical between base and turbo while the LoRA
//! carries 60 conditioner deltas. `anima-turbo-v1.0` is an independent fine-tune, NOT this LoRA merged
//! onto base — so no scale reproduces it. These tests therefore validate the two silent-failure classes
//! the reproduction test was meant to catch, DIRECTLY and more tightly:
//!   1. `llm_adapter` injection actually happens (count 508/448 + the DiT-only-host mutation), and
//!   2. the α = r ⇒ scale 1.0 convention (weight-level: injected forward == base + B·A, + the
//!      halve-scale mutation).

use std::path::PathBuf;

use mlx_rs::ops::{add, matmul, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::adapters::loader::{apply_adapter_specs_autoprefix, apply_adapters_strict};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen::WeightsSource;

use mlx_gen_anima::apply_anima_adapters;
use mlx_gen_anima::config::Variant;
use mlx_gen_anima::loader::AnimaComponents;

// -------------------------------------------------------------------------------------------------
// Fixtures
// -------------------------------------------------------------------------------------------------

/// Glob the Anima base snapshot's `split_files/` dir (DiT checkpoints + VAE + TE).
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

/// Glob the official-LoRAs snapshot dir.
fn lora_dir() -> Option<PathBuf> {
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

fn turbo_lora() -> PathBuf {
    lora_dir()
        .expect("Anima LoRA snapshot")
        .join("anima-turbo-lora-v0.2.safetensors")
}
fn style_lora() -> PathBuf {
    lora_dir()
        .expect("Anima LoRA snapshot")
        .join("anima-greg-rutkowski-style.safetensors")
}

fn load_base() -> AnimaComponents {
    let split = split_files().expect("Anima base snapshot");
    AnimaComponents::load(&WeightsSource::Dir(split), Variant::Base).expect("load base components")
}

fn lora_spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path, scale, AdapterKind::Lora)
}

fn randn(shape: &[i32]) -> Array {
    random::normal::<f32>(shape, None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

/// L2 norm of `a` in f32 (via `Σ a·a`), avoiding any method-name ambiguity.
fn l2(a: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    multiply(&a, &a)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>()
        .sqrt()
}

/// `||got − want|| / ||want||`, in f32 — a scale-free relative error for bf16 comparisons.
fn rel_err(got: &Array, want: &Array) -> f32 {
    let diff = subtract(
        got.as_dtype(Dtype::Float32).unwrap(),
        want.as_dtype(Dtype::Float32).unwrap(),
    )
    .unwrap();
    l2(&diff) / l2(want).max(1e-12)
}

// -------------------------------------------------------------------------------------------------
// 1. Injected-target COUNT (correctness bar #1) — the sc-10274 partial-injection guard.
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn turbo_lora_injects_508_style_injects_448() {
    // Turbo LoRA = 448 DiT (28×16) + 60 conditioner (6×10) = 508 targets.
    let mut c = load_base();
    let report = apply_anima_adapters(
        &mut c.dit,
        &mut c.conditioner,
        &[lora_spec(turbo_lora(), 1.0)],
    )
    .expect("apply turbo LoRA");
    assert_eq!(
        report.applied, 508,
        "turbo LoRA must inject all 448 DiT + 60 adapter targets"
    );
    assert!(
        report.unmatched_paths.is_empty(),
        "no target may be unmatched: {:?}",
        report.unmatched_paths
    );

    // Style LoRA = 448 DiT-only, zero conditioner targets.
    let mut c2 = load_base();
    let report2 = apply_anima_adapters(
        &mut c2.dit,
        &mut c2.conditioner,
        &[lora_spec(style_lora(), 1.0)],
    )
    .expect("apply style LoRA");
    assert_eq!(
        report2.applied, 448,
        "DiT-only style LoRA must inject exactly 448 targets"
    );
    assert!(report2.unmatched_paths.is_empty());
}

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn adaln_modulation_pairs_are_injected() {
    // The three adaLN-modulation down/up pairs (`.1`/`.2`) are the ones most likely to be skipped.
    // Assert each is actually adapted after injecting the turbo LoRA.
    let mut c = load_base();
    apply_anima_adapters(
        &mut c.dit,
        &mut c.conditioner,
        &[lora_spec(turbo_lora(), 1.0)],
    )
    .unwrap();
    for adaln in [
        "adaln_modulation_self_attn",
        "adaln_modulation_cross_attn",
        "adaln_modulation_mlp",
    ] {
        for updown in ["1", "2"] {
            let lin = c
                .dit
                .adaptable_mut(&["blocks", "3", adaln, updown])
                .unwrap_or_else(|| panic!("no adaptable linear at blocks.3.{adaln}.{updown}"));
            assert_eq!(
                lin.adapters().len(),
                1,
                "blocks.3.{adaln}.{updown} not adapted"
            );
        }
    }
}

// -------------------------------------------------------------------------------------------------
// 2. MUTATION: a DiT-only injection walk (the sc-10274 bug) must FAIL LOUDLY, not partial-load.
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn mutation_dit_only_walk_drops_60_adapter_targets() {
    // Route the turbo LoRA through the DiT host ALONE (skipping the `llm_adapter` conditioner) — the
    // exact partial-injection regression sc-10274 was. The strict applier MUST error on the 60
    // unmatched conditioner targets; a lenient walk reports only 448 applied + 60 unmatched.
    let mut dit_only = load_base();
    let lenient =
        apply_adapter_specs_autoprefix(&mut dit_only.dit, &[lora_spec(turbo_lora(), 1.0)])
            .expect("lenient apply");
    assert_eq!(
        lenient.applied, 448,
        "DiT-only walk injects only the 448 DiT targets"
    );
    assert_eq!(
        lenient.unmatched_paths.len(),
        60,
        "the 60 llm_adapter targets are dropped"
    );

    let mut dit_only2 = load_base();
    let err = apply_adapters_strict(
        &mut dit_only2.dit,
        &[lora_spec(turbo_lora(), 1.0)],
        "anima-dit-only",
    )
    .expect_err("strict apply onto a DiT-only host must ERROR on the dropped adapter targets");
    let msg = err.to_string();
    assert!(
        msg.contains("matched no module"),
        "expected an unmatched-target error, got: {msg}"
    );
}

// -------------------------------------------------------------------------------------------------
// 3. SCALE convention (correctness bar #2): the injected effective weight is base + B·A at scale 1.0
//    (α = r ⇒ 1.0, NO fold — the file carries no alpha/rank). Proven at the weight level for a DiT
//    target, an adaLN pair, AND a conditioner target; plus the halve-scale mutation.
// -------------------------------------------------------------------------------------------------

/// A target linear plus its raw-file LoRA key: `(route_into_conditioner, dotted_path, lora_key)`.
type Target = (bool, &'static [&'static str], &'static str);

const DIT_Q: Target = (
    false,
    &["blocks", "0", "self_attn", "q_proj"],
    "diffusion_model.blocks.0.self_attn.q_proj",
);
const DIT_ADALN: Target = (
    false,
    &["blocks", "0", "adaln_modulation_self_attn", "2"],
    "diffusion_model.blocks.0.adaln_modulation_self_attn.2",
);
const COND_Q: Target = (
    true,
    &["blocks", "0", "self_attn", "q_proj"],
    "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj",
);

/// The base `[out,in]` weight of a target BEFORE injection. Force-EVALUATED here (not left lazy): the
/// heavy 508-target `apply_anima_adapters` that runs between capture and use recreates the default
/// Metal stream, which would strand a lazily-held base weight ("no Stream(gpu, N)"). Materializing it
/// now pins the value.
fn base_weight(c: &mut AnimaComponents, t: &Target) -> Array {
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
/// (no alpha/rank fold). `lw` is the already-loaded turbo LoRA weights (loaded once per test).
fn injected_rel_err(
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

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn scale_convention_effective_weight_is_base_plus_ba_at_scale_1() {
    // ONE model load, ONE injection: capture the three base weights, inject the turbo LoRA at scale
    // 1.0, then compare each target's forward to base + B·A. (Repeated `load_base()` in one binary
    // trips mlx-rs's shared-Metal-stream lifecycle.)
    let mut c = load_base();
    let wb_dit = base_weight(&mut c, &DIT_Q);
    let wb_adaln = base_weight(&mut c, &DIT_ADALN);
    let wb_cond = base_weight(&mut c, &COND_Q);
    let lw = Weights::from_file(turbo_lora()).unwrap();
    apply_anima_adapters(
        &mut c.dit,
        &mut c.conditioner,
        &[lora_spec(turbo_lora(), 1.0)],
    )
    .unwrap();

    let e_dit = injected_rel_err(&mut c, &DIT_Q, &lw, &wb_dit, 1.0);
    let e_adaln = injected_rel_err(&mut c, &DIT_ADALN, &lw, &wb_adaln, 1.0);
    let e_cond = injected_rel_err(&mut c, &COND_Q, &lw, &wb_cond, 1.0);
    println!("[scale=1.0] rel_err  DiT={e_dit:.2e}  adaLN={e_adaln:.2e}  conditioner={e_cond:.2e}");
    for (name, e) in [("DiT", e_dit), ("adaLN", e_adaln), ("conditioner", e_cond)] {
        assert!(
            e < 2e-2,
            "{name}: injected weight != base + B·A at scale 1.0 (rel_err {e:.3e})"
        );
    }
}

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn mutation_halved_scale_breaks_the_scale_1_equivalence() {
    // ONE load, ONE injection at scale 0.5. The forward must MATCH base + 0.5·B·A (rel_err small)...
    let mut c = load_base();
    let wb = base_weight(&mut c, &DIT_Q);
    let lw = Weights::from_file(turbo_lora()).unwrap();
    apply_anima_adapters(
        &mut c.dit,
        &mut c.conditioner,
        &[lora_spec(turbo_lora(), 0.5)],
    )
    .unwrap();

    let e_half_ref = injected_rel_err(&mut c, &DIT_Q, &lw, &wb, 0.5);
    assert!(
        e_half_ref < 2e-2,
        "scale enters linearly (base + 0.5·B·A), rel_err {e_half_ref:.3e}"
    );

    // ...and must therefore DIVERGE from the scale-1.0 reference (residual is half). This is the
    // halve-scale mutation: the α=r⇒1.0 equivalence test is scale-sensitive, so a wrong scale FAILS it.
    let e_vs_scale1 = injected_rel_err(&mut c, &DIT_Q, &lw, &wb, 1.0);
    println!("[mutation] scale-0.5 injection: rel_err vs 0.5·B·A={e_half_ref:.3e}  vs 1.0·B·A={e_vs_scale1:.3e}");
    assert!(
        e_vs_scale1 > 1e-3,
        "halving the scale did NOT change the forward — the scale test is insensitive!"
    );
}

// -------------------------------------------------------------------------------------------------
// 4. The turbo-reproduction reality check — encodes the finding that turbo ≠ base + LoRA.
// -------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn turbo_checkpoint_is_not_base_plus_lora() {
    // Cosine between the turbo LoRA's weight delta (B·A) and (W_turbo − W_base) for a representative
    // target. If anima-turbo-v1.0 were base + this LoRA, cos ≈ 1; measured cos ≈ 0 ⇒ independent
    // fine-tune. This documents WHY the story's reproduction test is invalid (not an injection/scale
    // bug). Also confirms the conditioner is byte-identical base↔turbo.
    let split = split_files().expect("Anima base snapshot");
    let base_w =
        Weights::from_file(split.join("diffusion_models/anima-base-v1.0.safetensors")).unwrap();
    let turbo_w =
        Weights::from_file(split.join("diffusion_models/anima-turbo-v1.0.safetensors")).unwrap();
    let lw = Weights::from_file(turbo_lora()).unwrap();

    let wb = base_w
        .require("net.blocks.0.self_attn.q_proj.weight")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let wt = turbo_w
        .require("model.diffusion_model.blocks.0.self_attn.q_proj.weight")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let a = lw
        .require("diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let b = lw
        .require("diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();

    let lora_delta = matmul(&b, &a).unwrap(); // [out,in]
    let turbo_delta = subtract(&wt, &wb).unwrap();
    let dot = multiply(&lora_delta, &turbo_delta)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>();
    let nl = l2(&lora_delta);
    let nt = l2(&turbo_delta);
    let cos = dot / (nl * nt).max(1e-12);
    println!("cos(B·A, W_turbo−W_base) = {cos:+.4}  |B·A|={nl:.5}  |turbo−base|={nt:.5}");
    assert!(cos.abs() < 0.2, "turbo checkpoint UNEXPECTEDLY reproduces base+LoRA (cos {cos:.3}) — re-examine the premise");

    // Conditioner unchanged base↔turbo (further proof turbo is not base + this LoRA, whose 60
    // conditioner deltas would have moved it).
    let cb = base_w
        .require("net.llm_adapter.blocks.0.self_attn.q_proj.weight")
        .unwrap();
    let ct = turbo_w
        .require("model.diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.weight")
        .unwrap();
    let cond_diff = l2(&subtract(
        ct.as_dtype(Dtype::Float32).unwrap(),
        cb.as_dtype(Dtype::Float32).unwrap(),
    )
    .unwrap());
    println!("||conditioner turbo−base|| = {cond_diff:.6e}");
    assert!(
        cond_diff < 1e-2,
        "conditioner differs base↔turbo (expected identical): {cond_diff:.3e}"
    );
}

// -------------------------------------------------------------------------------------------------
// 5. LoKr load + apply, and stacked LoRA + LoKr (mixed).
// -------------------------------------------------------------------------------------------------

/// Synthesize a peft-format LoKr `.safetensors` (`networkType=lokr`, `rank`/`alpha` metadata, full
/// Kronecker factors `lokr_w1`/`lokr_w2`) targeting one DiT and one conditioner module, and return its
/// path. `alpha == rank` ⇒ scale 1.0 (PEFT). No official Anima LoKr exists, so this is a hand-built
/// LoKr proving the path end to end; `a·b == out`, `c·d == in` per target.
///
/// The file is written as **raw safetensors bytes** (deterministic f32 factors), NOT via mlx arrays —
/// constructing + evaluating mlx arrays in a second real-weights test within the same binary trips
/// mlx-rs's shared-Metal-stream lifecycle ("no Stream(gpu, N)"), the same hazard that put
/// `velocity_convention` in its own binary. Raw bytes keep this file-build off the GPU entirely.
fn synth_lokr() -> PathBuf {
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
    );
    path
}

/// Write a minimal valid F32 safetensors from `(name, shape)` entries with deterministic, non-trivial
/// factor values and a string-valued `__metadata__` map (plus `format: pt`). No mlx involvement.
fn write_raw_safetensors(
    path: &std::path::Path,
    entries: &[(&str, &[usize])],
    meta: &[(&str, &str)],
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
            // deterministic bounded pseudo-values in ~[-0.02, 0.02], never all-zero.
            let v = (((i * 131 + j * 17 + 7) % 101) as f32 / 101.0 - 0.5) * 0.04;
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

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot"]
fn lokr_loads_and_applies_to_dit_and_conditioner() {
    let mut c = load_base();
    let spec = AdapterSpec::new(synth_lokr(), 1.0, AdapterKind::Lokr);
    let report = apply_anima_adapters(&mut c.dit, &mut c.conditioner, &[spec]).expect("apply LoKr");
    assert_eq!(
        report.applied, 2,
        "LoKr must apply to both its DiT and conditioner targets"
    );
    assert!(
        report.unmatched_paths.is_empty(),
        "unmatched: {:?}",
        report.unmatched_paths
    );
    // Both targets carry exactly one (LoKr) adapter, and the residual is non-trivial.
    assert_eq!(
        c.dit
            .adaptable_mut(&["blocks", "1", "self_attn", "k_proj"])
            .unwrap()
            .adapters()
            .len(),
        1
    );
    assert_eq!(
        c.conditioner
            .adaptable_mut(&["blocks", "1", "self_attn", "k_proj"])
            .unwrap()
            .adapters()
            .len(),
        1
    );
}

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn stacked_lora_plus_lokr_mixed() {
    // Apply the turbo LoRA (508 targets) AND the synthetic LoKr (2 targets) in one strict call.
    let mut c = load_base();
    let specs = vec![
        lora_spec(turbo_lora(), 1.0),
        AdapterSpec::new(synth_lokr(), 1.0, AdapterKind::Lokr),
    ];
    let report =
        apply_anima_adapters(&mut c.dit, &mut c.conditioner, &specs).expect("apply stacked");
    assert_eq!(report.applied, 510, "508 LoRA + 2 LoKr targets");
    // blocks.1.self_attn.k_proj is hit by BOTH the turbo LoRA and the LoKr → two stacked adapters.
    let lin = c
        .dit
        .adaptable_mut(&["blocks", "1", "self_attn", "k_proj"])
        .unwrap();
    assert_eq!(
        lin.adapters().len(),
        2,
        "mixed LoRA + LoKr must stack on the same module"
    );
    // A forward runs (both residuals compose over the shared base).
    let x = randn(&[1, 8, 2048]);
    assert_eq!(lin.forward(&x).unwrap().shape(), &[1, 8, 2048]);
}

// -------------------------------------------------------------------------------------------------
// 6. End-to-end: anima_base + turbo LoRA generates a coherent image (proves the injected LoRA runs
//    through the full pipeline). NOTE: not compared to anima-turbo-v1.0 — see
//    `turbo_checkpoint_is_not_base_plus_lora`; they are different models.
// -------------------------------------------------------------------------------------------------

fn grayscale_std(pixels: &[u8]) -> f32 {
    let gray: Vec<f32> = pixels
        .chunks(3)
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect();
    let mean = gray.iter().sum::<f32>() / gray.len() as f32;
    (gray.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / gray.len() as f32).sqrt()
}

#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots; SLOW (2B DiT denoise)"]
fn base_plus_turbo_lora_generates_coherent_image() {
    use mlx_gen::runtime::CancelFlag;
    use mlx_gen::Progress;
    use mlx_gen_anima::pipeline::{AnimaPipeline, GenOptions};

    let split = split_files().expect("Anima base snapshot");
    let mut pipeline =
        AnimaPipeline::from_source(&WeightsSource::Dir(split), Variant::Base).expect("pipeline");
    let report = pipeline
        .apply_adapters(&[lora_spec(turbo_lora(), 1.0)])
        .expect("apply turbo LoRA");
    assert_eq!(report.applied, 508);

    // Turbo regime (few-step, CFG-free) — the LoRA is a distillation adapter.
    let opts = GenOptions {
        width: 1024,
        height: 1024,
        steps: 10,
        guidance: 1.0,
        seed: 42,
        sampler: None,
    };
    let cancel = CancelFlag::default();
    let mut prog = |_p: Progress| {};
    let img = pipeline
        .generate(
            "an anime girl with silver hair, detailed illustration, masterpiece",
            "",
            Variant::Base,
            &opts,
            &cancel,
            &mut prog,
        )
        .expect("generate");
    let std = grayscale_std(&img.pixels);
    println!("[base+turbo-lora] grayscale std = {std:.2}");
    assert_eq!((img.width, img.height), (1024, 1024));
    assert!(
        std > 8.0,
        "base+turbo-LoRA output is near-blank (std {std:.2}) — injection likely broken"
    );
}
