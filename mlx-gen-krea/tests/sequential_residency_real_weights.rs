//! sc-11101 (epic 10834 Phase 1 fan-out): the `Sequential` component-residency A/B on real Krea 2
//! weights, across the WHOLE MLX family — `krea_2_turbo`, `krea_2_raw`, `krea_2_edit`,
//! `krea_2_turbo_control`.
//!
//! `#[ignore]`d — needs the real snapshots (env overrides, else the HF cache):
//!
//! - turbo — `SceneWorks/krea-2-turbo-mlx` (tier subdir `q8`/`q4`/`bf16`), env `KREA_TURBO_DIR`.
//! - raw — `SceneWorks/krea-2-raw-mlx` (`bf16`, quantized in place), env `KREA_RAW_DIR`.
//! - edit — dense `krea/Krea-2-Raw` + the `conradlocke/krea2-identity-edit` LoRA (`KREA_EDIT_DIR` /
//!   `KREA_EDIT_LORA`).
//! - control — dense base (`SceneWorks/krea-2-turbo-mlx/bf16`) + the
//!   `SceneWorks/krea2-pose-controlnet-beta` overlay (`KREA_CONTROL_DIR` / `KREA_CONTROL_OVERLAY`).
//!
//! Run e.g. `cargo test -p mlx-gen-krea --release --test sequential_residency_real_weights --
//! --ignored --nocapture`.
//!
//! Two claims (the SDXL/Z-Image/Qwen/Lens A/B): (1) `Sequential` peaks LOWER than `Resident` because
//! the Qwen3-VL-4B text phase is dropped (+ `clear_cache()`) before the 12B single-stream DiT + the
//! denoise activations materialize, and (2) the output is BYTE-IDENTICAL. Krea's TE (~4B) is SMALLER
//! than the DiT (the qwen-image pattern), so the win is the denoise-phase DiT+activations that
//! `Resident` stacks on the resident text phase.
//!
//! Quant note: turbo/raw/edit quantize BOTH the Qwen3-VL TE Linears AND the DiT — measure Q8/Q4.
//! Control is dense-bf16-only (the overlay is trained on the dense base), so it is measured at bf16;
//! its dropped text phase is the ~8 GB bf16 encoder, and the pose branch stays on the heavy side.

// Force-link the provider crate so its `inventory` generator registrations run.
use mlx_gen_krea as _;

use mlx_gen::{
    AdapterKind, AdapterSpec, Conditioning, ControlKind, GenerationOutput, GenerationRequest,
    Image, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

/// First snapshot dir under an HF-cache `models--…` entry.
fn hf_snapshot(model: &str) -> PathBuf {
    let snaps = home()
        .join(".cache/huggingface/hub")
        .join(model)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("HF cache snapshots dir for {model}: {}", snaps.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Which quant tier to load turbo/raw/edit at (Q8 default; `KREA_SEQ_Q4=1` for Q4). Returned as the
/// tier-subdir name too (turbo's turnkey is laid out `q8/`/`q4/`/`bf16/`).
fn seq_quant() -> Option<Quant> {
    if std::env::var("KREA_SEQ_Q4").is_ok() {
        Some(Quant::Q4)
    } else {
        Some(Quant::Q8)
    }
}

/// A fixed deterministic RGB source image (a smooth gradient). Quality is irrelevant — the A/B only
/// needs the SAME input for Resident and Sequential, so a reproducible image suffices for the
/// edit/control byte-identity checks.
fn fixed_image(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 255 / w.max(1)) as u8);
            pixels.push((y * 255 / h.max(1)) as u8);
            pixels.push(((x + y) * 127 / (w + h).max(1)) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Render one image under `spec`, measuring the process peak unified memory (`get_peak_memory`).
fn render_measured(spec: LoadSpec, model_id: &str, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let model = mlx_gen::load(model_id, &spec).unwrap_or_else(|e| panic!("load {model_id}: {e}"));
    reset_peak_memory();
    let out = model.generate(req, &mut |_| {}).expect("generate");
    let peak = get_peak_memory();
    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "expected a single image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    drop(model);
    clear_cache();
    (img.pixels, peak)
}

/// The shared A/B assertion: Sequential is byte-identical to Resident AND peaks lower. `base_spec` is a
/// closure so each policy build starts from a fresh `LoadSpec` (the `Sequential` run must not reuse the
/// `Resident` process's warm arrays).
fn assert_ab(
    label: &str,
    model_id: &str,
    req: &GenerationRequest,
    base_spec: impl Fn() -> LoadSpec,
) {
    let (px_res, peak_res) = render_measured(
        base_spec().with_offload_policy(OffloadPolicy::Resident),
        model_id,
        req,
    );
    let (px_seq, peak_seq) = render_measured(
        base_spec().with_offload_policy(OffloadPolicy::Sequential),
        model_id,
        req,
    );

    println!(
        "{label} {}x{} @ {} steps:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        req.width,
        req.height,
        req.steps.unwrap_or(0),
        peak_res as f64 / GIB,
        peak_seq as f64 / GIB,
        (peak_res.saturating_sub(peak_seq)) as f64 / GIB,
        100.0 * (peak_res.saturating_sub(peak_seq)) as f64 / peak_res as f64,
    );

    let diff = px_res.iter().zip(&px_seq).filter(|(a, b)| a != b).count();
    assert_eq!(
        px_res.len(),
        px_seq.len(),
        "{label}: image sizes differ ({} vs {})",
        px_res.len(),
        px_seq.len()
    );
    assert_eq!(
        diff,
        0,
        "{label}: Sequential changed the output — {diff}/{} bytes differ (must be byte-identical)",
        px_res.len()
    );
    assert!(
        peak_seq < peak_res,
        "{label}: Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the text-phase drop \
         did not reduce peak",
        peak_seq as f64 / GIB,
        peak_res as f64 / GIB,
    );
}

/// The turbo turnkey root for a tier — `KREA_TURBO_DIR` (a tier subdir) or the HF-cache
/// `SceneWorks/krea-2-turbo-mlx/<tier>`.
fn turbo_root(tier: &str) -> PathBuf {
    if let Ok(p) = std::env::var("KREA_TURBO_DIR") {
        return PathBuf::from(p);
    }
    hf_snapshot("models--SceneWorks--krea-2-turbo-mlx").join(tier)
}

#[test]
#[ignore = "needs SceneWorks/krea-2-turbo-mlx (KREA_TURBO_DIR or the HF cache)"]
fn turbo_sequential_bounds_peak_and_is_byte_identical() {
    // The turbo turnkey ships pre-packed q8/q4/bf16 subdirs, so point at the tier dir and pass the
    // matching quant (a no-op on the already-packed base; the F-076 check just confirms the tier).
    let tier = if std::env::var("KREA_SEQ_Q4").is_ok() {
        "q4"
    } else {
        "q8"
    };
    let root = turbo_root(tier);
    let size = env_u32("KREA_SEQ_SIZE", 768);
    let req = GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("KREA_SEQ_STEPS", 8)),
        ..Default::default()
    };
    assert_ab("krea_2_turbo", "krea_2_turbo", &req, || {
        LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(seq_quant().unwrap())
    });
}

#[test]
#[ignore = "needs SceneWorks/krea-2-raw-mlx (KREA_RAW_DIR or the HF cache)"]
fn raw_sequential_bounds_peak_and_is_byte_identical() {
    // The raw turnkey ships bf16 only; quantize it in place to the requested tier. Full CFG + a
    // non-empty negative prompt exercises the two-forward joint CFG AND the negative-branch encode —
    // the stringent case for "encode both contexts, then drop the text phase" byte-identity.
    let root = std::env::var("KREA_RAW_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_snapshot("models--SceneWorks--krea-2-raw-mlx").join("bf16"));
    let size = env_u32("KREA_SEQ_SIZE", 768);
    let req = GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, low quality".into()),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("KREA_RAW_STEPS", 12)),
        guidance: Some(env_f32("KREA_RAW_GUIDANCE", 3.5)),
        ..Default::default()
    };
    assert_ab("krea_2_raw", "krea_2_raw", &req, || {
        LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(seq_quant().unwrap())
    });
}

#[test]
#[ignore = "needs dense krea/Krea-2-Raw + the identity-edit LoRA (KREA_EDIT_DIR / KREA_EDIT_LORA)"]
fn edit_sequential_bounds_peak_and_is_byte_identical() {
    // Edit grounds on the source image through the Qwen3-VL VISION tower — the text phase drops the
    // encoder AND the vision tower. Needs the DENSE Raw base (grounding uses `visual.*`) + the
    // community identity-edit LoRA on the DiT.
    let root = std::env::var("KREA_EDIT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_snapshot("models--krea--Krea-2-Raw"));
    let lora = std::env::var("KREA_EDIT_LORA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            hf_snapshot("models--conradlocke--krea2-identity-edit")
                .join("krea2_identity_edit_v1_1.safetensors")
        });
    let size = env_u32("KREA_SEQ_SIZE", 768);
    let req = GenerationRequest {
        prompt: "make the person smile, keep the identity".into(),
        negative_prompt: Some("blurry, distorted".into()),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("KREA_EDIT_STEPS", 8)),
        guidance: Some(env_f32("KREA_EDIT_GUIDANCE", 3.5)),
        conditioning: vec![Conditioning::Reference {
            image: fixed_image(512, 512),
            strength: None,
        }],
        ..Default::default()
    };
    assert_ab("krea_2_edit", "krea_2_edit", &req, || {
        LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_quant(seq_quant().unwrap())
            .with_adapters(vec![AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)])
    });
}

#[test]
#[ignore = "needs a dense Krea base + the pose overlay (KREA_CONTROL_DIR / KREA_CONTROL_OVERLAY)"]
fn control_sequential_bounds_peak_and_is_byte_identical() {
    // Pose control is dense bf16 only (the overlay is trained on the dense base) — measured at bf16.
    // The `Krea2ControlBranch` is a SECOND heavy component that stays on the heavy side; only the
    // Qwen3-VL text phase drops. Base = the turbo turnkey's dense bf16 subdir.
    let root = std::env::var("KREA_CONTROL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| turbo_root("bf16"));
    let overlay = std::env::var("KREA_CONTROL_OVERLAY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            hf_snapshot("models--SceneWorks--krea2-pose-controlnet-beta")
                .join("control_step5000.safetensors")
        });
    let size = env_u32("KREA_SEQ_SIZE", 768);
    let req = GenerationRequest {
        prompt: "a person standing in a studio, photograph".into(),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("KREA_CONTROL_STEPS", 8)),
        conditioning: vec![Conditioning::Control {
            image: fixed_image(512, 512),
            kind: ControlKind::Pose,
            scale: Some(0.6),
        }],
        ..Default::default()
    };
    // Dense bf16 (no quant override — control rejects it).
    assert_ab("krea_2_turbo_control", "krea_2_turbo_control", &req, || {
        LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_control(WeightsSource::File(overlay.clone()))
    });
}
