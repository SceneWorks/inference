//! sc-11000 / sc-11006 (epic 10834 Phase 1 fan-out): the `Sequential` component-residency A/B on real
//! Qwen-Image weights, across the whole family — T2I (`qwen_image`, sc-11000), **Edit**
//! (`qwen_image_edit`) and **Control** (`qwen_image_control`), the two sc-11006 fan-out engines.
//!
//! `#[ignore]`d — needs the real snapshots (env overrides, else the HF cache). Run:
//!   cargo test -p mlx-gen-qwen-image --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Two claims per engine (same as the SDXL/Z-Image A/B): (1) `Sequential` peaks LOWER than `Resident`
//! because the Qwen2.5-VL text (T2I/Control) or vision-language (Edit) encoder is dropped (+
//! `clear_cache()`) before the DiT materializes, and (2) the output is BYTE-IDENTICAL. Qwen-Image's
//! ~15 GB encoder is comparable to the ~20 GB DiT, so this is the biggest image-lane saving (36→20 GB,
//! fits a 32 GB Mac). A repeat-job check confirms nothing stays resident across jobs.
//!
//! Snapshot env overrides: T2I `QWEN_IMAGE_SNAPSHOT`; Edit `QWEN_IMAGE_EDIT_SNAPSHOT`; Control base
//! `QWEN_CONTROL_BASE_SNAPSHOT` + branch `QWEN_CONTROL_WEIGHTS`. Defaults pick the SceneWorks q8
//! re-host tiers in the HF cache. Set `QWEN_SEQ_Q8=1` for the T2I Q8 case; `QWEN_SEQ_STEPS`/
//! `QWEN_SEQ_SIZE` tune all three.

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn snapshot() -> PathBuf {
    let p = std::env::var("QWEN_IMAGE_SNAPSHOT").unwrap_or_else(|_| panic!("set QWEN_IMAGE_SNAPSHOT to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"));
    PathBuf::from(p)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn probe_request() -> GenerationRequest {
    // A fixed seed makes the byte-identity assertion meaningful; quality is irrelevant here
    // (Resident vs Sequential, not a golden). Qwen-Image is true-CFG — the default (unset) sampler is
    // the production flow-match path with a negative branch, exercising two encode_prompt calls.
    let size = env_u32("QWEN_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("QWEN_SEQ_STEPS", 8)),
        ..Default::default()
    }
}

fn base_spec() -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if std::env::var("QWEN_SEQ_Q8").is_ok() {
        spec = spec.with_quant(Quant::Q8);
    }
    spec
}

fn render_measured(policy: OffloadPolicy, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let spec = base_spec().with_offload_policy(policy);
    let model = mlx_gen_qwen_image::provider_registry()
        .unwrap()
        .load("qwen_image", &spec)
        .expect("load qwen_image");
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
    let Image { pixels, .. } = img;
    drop(model);
    clear_cache();
    (pixels, peak)
}

#[test]
#[ignore = "needs a real Qwen/Qwen-Image snapshot (QWEN_IMAGE_SNAPSHOT or the HF cache)"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let req = probe_request();
    let (pixels_resident, peak_resident) = render_measured(OffloadPolicy::Resident, &req);
    let (pixels_sequential, peak_sequential) = render_measured(OffloadPolicy::Sequential, &req);

    println!(
        "Qwen-Image {}x{} @ {} steps{}:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        req.width,
        req.height,
        req.steps.unwrap(),
        if std::env::var("QWEN_SEQ_Q8").is_ok() { " (Q8)" } else { "" },
        peak_resident as f64 / GIB,
        peak_sequential as f64 / GIB,
        (peak_resident.saturating_sub(peak_sequential)) as f64 / GIB,
        100.0 * (peak_resident.saturating_sub(peak_sequential)) as f64 / peak_resident as f64,
    );

    let diff = pixels_resident
        .iter()
        .zip(&pixels_sequential)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff,
        0,
        "Sequential residency changed the output: {diff}/{} bytes differ (must be byte-identical)",
        pixels_resident.len()
    );
    assert!(
        peak_sequential < peak_resident,
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the text-encoder drop did not \
         reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

#[test]
#[ignore = "needs a real Qwen/Qwen-Image snapshot (QWEN_IMAGE_SNAPSHOT or the HF cache)"]
fn sequential_repeat_job_stays_bounded() {
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, &req);
    println!(
        "Qwen-Image Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
        peak1 as f64 / GIB,
        peak2 as f64 / GIB,
    );
    let slop = peak1 / 10;
    assert!(
        peak2 <= peak1 + slop,
        "repeat Sequential job peaked higher ({:.3} vs {:.3} GiB) — a component stayed resident",
        peak2 as f64 / GIB,
        peak1 as f64 / GIB,
    );
}

// ---------------------------------------------------------------------------------------------------
// sc-11006 fan-out: the same A/B for the Edit + Control engines. Shared harness below.
// ---------------------------------------------------------------------------------------------------

/// Resolve a SceneWorks q8 re-host tier in the HF cache: the first snapshot dir, then its `q8/` tier
/// subdir if present (the turnkeys nest tiers), else the snapshot root (a flat component layout).
fn tier_snapshot(repo: &str) -> PathBuf {
    let home = std::env::var("MLX_GEN_MODELS_ROOT").expect("set MLX_GEN_MODELS_ROOT to the explicit models root (holds models--*/snapshots); inference never self-fetches or derives a cache location (epic 13657)");
    let snaps = PathBuf::from(home).join(repo).join("snapshots");
    let snap = std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("HF cache snapshots dir for {repo}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .unwrap_or_else(|| panic!("a snapshot dir for {repo}"));
    let q8 = snap.join("q8");
    if q8.is_dir() {
        q8
    } else {
        snap
    }
}

/// A deterministic synthetic RGB image (a fixed gradient) — a stand-in reference / pose skeleton so
/// the byte-identity claim is meaningful without shipping a fixture (quality is irrelevant here).
fn synthetic_image(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: (0..(w * h * 3)).map(|i| (i % 256) as u8).collect(),
    }
}

/// Load `engine_id` from `spec` (already carrying its offload policy), reset the MLX peak counter,
/// render one image, and return its pixels + the measured peak (mirrors [`render_measured`]).
fn render_with(engine_id: &str, spec: &LoadSpec, req: &GenerationRequest) -> (Vec<u8>, usize) {
    let model = mlx_gen_qwen_image::provider_registry()
        .unwrap()
        .load(engine_id, spec)
        .unwrap_or_else(|e| panic!("load {engine_id}: {e:?}"));
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
    let Image { pixels, .. } = img;
    drop(model);
    clear_cache();
    (pixels, peak)
}

/// The two-claim A/B for one engine: `Sequential` peaks below `Resident`, and byte-identical output.
/// `spec_for` builds the [`LoadSpec`] for a given policy (so per-engine weights/control are captured).
fn assert_sequential_ab(
    label: &str,
    engine: &str,
    spec_for: impl Fn(OffloadPolicy) -> LoadSpec,
    req: &GenerationRequest,
) {
    let (px_res, peak_res) = render_with(engine, &spec_for(OffloadPolicy::Resident), req);
    let (px_seq, peak_seq) = render_with(engine, &spec_for(OffloadPolicy::Sequential), req);
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
        diff,
        0,
        "{label}: Sequential residency changed the output: {diff}/{} bytes differ (must be byte-identical)",
        px_res.len()
    );
    assert!(
        peak_seq < peak_res,
        "{label}: Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the encoder drop did \
         not reduce peak",
        peak_seq as f64 / GIB,
        peak_res as f64 / GIB,
    );
}

// --- Edit (`qwen_image_edit`) ----------------------------------------------------------------------

/// Base `Qwen-Image-Edit` snapshot dir (env override, else the SceneWorks q8 re-host in the HF cache).
fn edit_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_EDIT_SNAPSHOT") {
        return PathBuf::from(p);
    }
    tier_snapshot("models--SceneWorks--qwen-image-edit-2511-mlx")
}

fn edit_spec(policy: OffloadPolicy) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(edit_snapshot())).with_offload_policy(policy)
}

fn edit_probe_request() -> GenerationRequest {
    // A single `Reference` image drives the dual-latent edit; the default (unset) sampler is true-CFG,
    // exercising both LM encodes over the shared vision-tower output.
    let size = env_u32("QWEN_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "make the sky a dramatic sunset".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(1234),
        steps: Some(env_u32("QWEN_SEQ_STEPS", 8)),
        conditioning: vec![Conditioning::Reference {
            image: synthetic_image(size, size),
            strength: None,
        }],
        ..Default::default()
    }
}

#[test]
#[ignore = "needs a real Qwen-Image-Edit snapshot (QWEN_IMAGE_EDIT_SNAPSHOT or the HF cache)"]
fn edit_sequential_bounds_peak_and_is_byte_identical() {
    // The Edit wrinkle vs T2I: the dropped component is the Qwen2.5-VL *vision-language* encoder, and
    // the drop must land after BOTH the vision-tower pass over the reference and the LM pass over the
    // prompts. Byte-identity here proves the reference VAE-encode (moved after the drop) is unaffected.
    assert_sequential_ab(
        "Qwen-Image-Edit",
        "qwen_image_edit",
        edit_spec,
        &edit_probe_request(),
    );
}

// --- Control (`qwen_image_control`) ----------------------------------------------------------------

/// Base `Qwen-Image` snapshot dir for the control variant (env override, else the SceneWorks q8 base).
fn control_base_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_CONTROL_BASE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    tier_snapshot("models--SceneWorks--qwen-image-mlx")
}

/// The Fun-Controlnet-Union checkpoint (env `QWEN_CONTROL_WEIGHTS`, else the SceneWorks re-host q8
/// single-file tier in the HF cache).
fn control_source() -> WeightsSource {
    if let Ok(p) = std::env::var("QWEN_CONTROL_WEIGHTS") {
        return WeightsSource::File(PathBuf::from(p));
    }
    let tier = tier_snapshot("models--SceneWorks--qwen-image-2512-fun-controlnet-union");
    let file = std::fs::read_dir(&tier)
        .expect("control tier dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .expect("a control .safetensors");
    WeightsSource::File(file)
}

fn control_spec(policy: OffloadPolicy) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(control_base_snapshot()))
        .with_control(control_source())
        .with_offload_policy(policy)
}

fn control_probe_request() -> GenerationRequest {
    let size = env_u32("QWEN_SEQ_SIZE", 768);
    GenerationRequest {
        prompt: "a person standing, photorealistic".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(1234),
        steps: Some(env_u32("QWEN_SEQ_STEPS", 8)),
        conditioning: vec![Conditioning::Control {
            image: synthetic_image(size, size),
            kind: ControlKind::Pose,
            scale: Some(1.0),
        }],
        ..Default::default()
    }
}

#[test]
#[ignore = "needs the control base snapshot + Fun-Controlnet checkpoint (QWEN_CONTROL_* or the HF cache)"]
fn control_sequential_bounds_peak_and_is_byte_identical() {
    // The Control audit: the VACE control branch is an EXTRA heavy component (loaded + quantized with
    // the base DiT), so it stays on the heavy side of the split. Byte-identity here proves the pose
    // VAE-encode (moved after the text-encoder drop) is unaffected; the peak drop is the ~15 GB TE.
    assert_sequential_ab(
        "Qwen-Image-Control",
        "qwen_image_control",
        control_spec,
        &control_probe_request(),
    );
}
