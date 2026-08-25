//! Krea Realtime 14B **real** Wan-family style-LoRA validation (sc-8446, S13).
//!
//! `tests/style_lora.rs` proves the install mechanics on synthetic low-rank files. This proves the
//! thing that could not be proven synthetically: that **published** Wan-2.1-14B LoRA files resolve
//! against this DiT's [`AdaptableHost`] surface, install through `LoadSpec::adapters` →
//! `apply_adapters_strict_with_diff_patch`, and **measurably change the render** — without tripping
//! the strict installer's unmatched-target hard error.
//!
//! ## The two real classes, and why the globals surface was widened (the S13 decision)
//!
//! Surveyed from the published safetensors headers of real Wan-2.1-14B LoRA files:
//!
//! | class | example | low-rank targets |
//! |---|---|---|
//! | plain style | `shauray/Origami_WanLora`, `motimalu/wan-flat-color-v2` | 400 per-block stems only |
//! | step-distill | lightx2v `Wan2.1-T2V-14B` cfg-step-distill v2, `FastWan` T2V-14B (both headers read) | the same 400 **plus 6 of the 7 whole-model globals** |
//! | I2V-family | `Remade-AI/Squish` | per-block **plus `cross_attn.k_img`/`v_img`** |
//!
//! The step-distill class is what settled the decision: those globals carry genuine `lora_down`/
//! `lora_up` factors, so **soft-skipping** them would have silently installed a step-distill LoRA with
//! its text/time/output projections missing — a wrong render that reports success. Widening the
//! surface applies them instead (see `causal::krea_adaptable_paths`). The I2V-only image cross-attention
//! stays unexposed and still errors loudly: those modules do not exist on a T2V backbone at any surface
//! width.
//!
//! **Six, not seven.** `patch_embedding` ships a `.diff_b` bias delta with **no** low-rank pair, so a
//! real step-distill file installs **406** targets against the 407-wide surface — asserted below, not
//! inferred. sc-15326 completed that surface: the remaining 647 keys (447 `.diff_b` + 200 norm
//! `.diff`) now land through the diff-patch-aware path on every tier, and any genuinely unsupported
//! target is returned to the provider instead of being dropped silently.
//!
//! ```text
//! KREA_REALTIME_SNAPSHOT_DIR=~/.cache/krea-realtime-mlx-snapshot/q4 \
//! KREA_STYLE_LORA=~/.cache/wan-loras/origami/origami_000000500.safetensors \
//!   cargo test -p mlx-gen-krea-realtime --test integration style_lora_real_weights:: -- --ignored --nocapture
//! ```
//!
//! Env: `KREA_REALTIME_SNAPSHOT_DIR`, `KREA_STYLE_LORA` (a real Wan style LoRA `.safetensors`),
//! `KREA_DISTILL_LORA` (optional — a real Wan step-distill LoRA that targets the globals),
//! `KREA_LORA_W`/`_H`/`_FRAMES` (default 832×480, 13 frames — small: this is an A/B, not a showreel).

use std::path::PathBuf;

use mlx_gen::{
    AdapterKind, AdapterSpec, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec,
    WeightsSource,
};
use mlx_gen_krea_realtime::MODEL_ID;

/// The full adaptable surface: `num_layers × 10` per-block Linears + the 7 whole-model globals.
const ADAPTABLE_SURFACE_WIDTH: usize = 40 * 10 + 7;
/// What a canonical plain style LoRA resolves: the per-block Linears only.
const STYLE_LORA_TARGETS: usize = 400;
/// What a real step-distill file resolves: 400 per-block + the **6** globals that ship a low-rank pair.
/// `patch_embedding` ships `.diff_b` only, so it is exposed but unmatched — hence 406, not 407.
const STEP_DISTILL_TARGETS: usize = 406;

/// Install `lora` onto a freshly-loaded real DiT and return the strict installer's own report. This is
/// the only place the **resolved target count** is observable: the provider swallows it, and a render
/// A/B can only show *that* something changed, never that the file resolved the targets we believe it
/// has. A LoRA that silently resolved half its targets would still move the pixels.
fn install_count(lora: &std::path::Path) -> usize {
    use mlx_gen::adapters::loader::apply_adapters_strict;
    use mlx_gen_krea_realtime::{
        load_krea_realtime_transformer_with_quant, CausalKreaTransformer, KreaRealtimeConfig,
    };
    use mlx_rs::Array;

    let root = snapshot_dir();
    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let w =
        mlx_gen::weights::Weights::from_file(root.join("dit.safetensors")).expect("open the DiT");
    let raw: std::collections::HashMap<String, Array> = w
        .keys()
        .map(|k| (k.to_string(), w.get(k).expect("listed key").clone()))
        .collect();
    let (dit, _) = load_krea_realtime_transformer_with_quant(raw, &cfg).expect("load the DiT");
    let mut host = CausalKreaTransformer::new(dit, &cfg);
    let report = apply_adapters_strict(
        &mut host,
        &[AdapterSpec::new(lora.to_path_buf(), 1.0, AdapterKind::Lora)],
        MODEL_ID,
    )
    .expect("the strict installer must accept this file");
    assert!(
        report.unmatched_paths.is_empty(),
        "unmatched targets: {:?}",
        report.unmatched_paths
    );
    report.applied
}

fn snapshot_dir() -> PathBuf {
    std::env::var("KREA_REALTIME_SNAPSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/krea-realtime-mlx-convert")
        })
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn require_lora(var: &str) -> PathBuf {
    let p = PathBuf::from(
        std::env::var(var).unwrap_or_else(|_| panic!("set {var} to a real Wan LoRA .safetensors")),
    );
    assert!(
        p.is_file(),
        "{var} does not point at a file: {}",
        p.display()
    );
    p
}

fn request(w: usize, h: usize, frames: usize) -> GenerationRequest {
    GenerationRequest {
        prompt: "a paper crane folding itself on a wooden desk, warm light".into(),
        width: w as u32,
        height: h as u32,
        frames: Some(frames as u32),
        seed: Some(1234),
        fps: Some(24),
        sampler: Some("self_forcing".into()),
        ..Default::default()
    }
}

/// Product-contract proof for the report path: actual real-weight adapter application must traverse
/// registry load → each advertised Generator route → the common reported finish seam → the public
/// accessor. The non-ignored tiny-host companion in `pipeline` gates the same funnel in CI; this
/// hardware-gated test proves the complete snapshot/runtime route without a test backend.
#[test]
#[ignore = "real snapshot + real Wan step-distill LoRA; run with --ignored on macOS"]
fn registry_t2v_i2v_v2v_routes_publish_the_real_adapter_report() {
    let root = snapshot_dir();
    assert!(
        root.join("dit.safetensors").exists() || root.join("transformer").is_dir(),
        "no Krea Realtime DiT at {} — set KREA_REALTIME_SNAPSHOT_DIR",
        root.display()
    );
    let adapter = require_lora("KREA_DISTILL_LORA");
    let image = Image {
        width: 16,
        height: 16,
        pixels: vec![127; 16 * 16 * 3],
    };
    let source = vec![image.clone(); 5];
    let requests = [
        ("t2v", request(16, 16, 5)),
        (
            "i2v",
            GenerationRequest {
                conditioning: vec![Conditioning::Reference {
                    image: image.clone(),
                    strength: None,
                }],
                ..request(16, 16, 5)
            },
        ),
        (
            "v2v",
            GenerationRequest {
                conditioning: vec![Conditioning::VideoClip {
                    frames: source,
                    frame_idx: 0,
                    strength: 0.5,
                }],
                ..request(16, 16, 5)
            },
        ),
    ];

    for (route, req) in requests {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        spec.adapters = vec![AdapterSpec::new(adapter.clone(), 1.0, AdapterKind::Lora)];
        let provider = mlx_gen_krea_realtime::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap_or_else(|e| panic!("{route}: registry load must succeed: {e}"));
        assert!(
            provider.adapter_apply_reports().is_empty(),
            "{route}: reports describe completed generation, never an unrun provider"
        );
        provider
            .generate(&req, &mut |_| {})
            .unwrap_or_else(|e| panic!("{route}: generation must succeed: {e}"));
        let reports = provider.adapter_apply_reports();
        assert_eq!(reports.len(), 1, "{route}: one selected adapter");
        assert_eq!(
            reports[0].adapter_path, adapter,
            "{route}: file attribution"
        );
        assert!(
            reports[0].applied > 0,
            "{route}: the real engine must report material adapter targets"
        );
        assert!(
            reports[0].skipped.is_empty(),
            "{route}: the real T2V step-distill file must fully land: {:?}",
            reports[0].skipped
        );
    }
}

/// Render `req` with the given adapters and return the decoded frames.
///
/// Each call loads a COMPLETE copy of the 14B stack, and the callers drive several per process:
/// `real_wan_style_lora_loads_and_changes_the_render` does four (three renders plus
/// [`install_count`]), `real_wan_step_distill_lora_installs_over_the_widened_globals` three (two
/// renders plus [`install_count`]). So the per-arm memory line below is not decoration. sc-17355's open question is whether the arms accumulate: run 30869410054 died in the
/// `lora@1.0` arm on a Metal command-buffer cascade, and two local runs died in that same arm on a
/// jetsam SIGKILL, which are two different deaths in one place. Whether the entering `active`
/// figure climbs arm-over-arm is what separates "the second load starts from a dirty baseline" from
/// "the box was simply busy", and it is the number nobody has. It is printed rather than asserted
/// on purpose: a threshold here would be an invented constant, and sc-17355 is explicit that one
/// failure in two runs does not yet justify changing anything.
///
/// `reset_peak_memory()` is a PROCESS-GLOBAL MLX mutation, not a per-test one, so this is only
/// sound while each test here owns its process. The real-weight lane guarantees that by invoking
/// `cargo test … "$name" -- --exact` once per test rather than selecting both in one run — if that
/// ever collapses into a single invocation, libtest's default thread pool would run the two tests
/// concurrently and each would rebase the other's high-water mid-render. `test_ci_workflow_policy`
/// pins the per-test invocation for exactly this reason.
fn render(adapters: Vec<AdapterSpec>, req: &GenerationRequest, label: &str) -> Vec<Image> {
    use mlx_rs::memory::{get_active_memory, get_peak_memory, reset_peak_memory};

    let root = snapshot_dir();
    assert!(
        root.join("dit.safetensors").exists() || root.join("transformer").is_dir(),
        "no Krea Realtime DiT at {} — set KREA_REALTIME_SNAPSHOT_DIR",
        root.display()
    );
    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    spec.adapters = adapters;
    // Rebase the high-water to whatever this arm INHERITED, so the peak below is this arm's own
    // allocation rather than a running maximum that only the first arm can ever set.
    let entering = get_active_memory();
    reset_peak_memory();
    let t0 = std::time::Instant::now();
    let gen = mlx_gen_krea_realtime::provider_registry()
        .unwrap()
        .load(MODEL_ID, &spec)
        .unwrap_or_else(|e| panic!("{label}: load must succeed: {e}"));
    let out = gen
        .generate(req, &mut |_| {})
        .unwrap_or_else(|e| panic!("{label}: generate must succeed: {e}"));
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    println!(
        "  {label}: rendered in {:.1?} (mlx active on entry {:.2} GiB, peak {:.2} GiB, active on \
         exit {:.2} GiB)",
        t0.elapsed(),
        entering as f64 / GIB,
        get_peak_memory() as f64 / GIB,
        get_active_memory() as f64 / GIB,
    );
    match out {
        GenerationOutput::Video { frames, .. } => frames,
        other => panic!("{label}: expected a Video output, got {other:?}"),
    }
}

/// Mean absolute per-byte difference between two clips of the same geometry, in 0..255 units.
fn mean_abs_delta(a: &[Image], b: &[Image]) -> f64 {
    assert_eq!(a.len(), b.len(), "clip lengths differ");
    let mut total = 0.0f64;
    let mut n = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.pixels.len(), y.pixels.len(), "frame buffer sizes differ");
        total += x
            .pixels
            .iter()
            .zip(y.pixels.iter())
            .map(|(&p, &q)| (p as f64 - q as f64).abs())
            .sum::<f64>();
        n += x.pixels.len();
    }
    total / n as f64
}

/// **(a) loads, (b) measurably changes the output, (c) no strict-apply hard error** — the three things
/// story sc-8446 asks of a real Wan style LoRA, on one seed so the only difference is the adapter.
///
/// Discriminating in both directions: the same LoRA at **scale 0** must leave the render essentially
/// where the baseline is, so a "the renders differ" pass cannot be produced by nondeterminism alone —
/// if it could, the scale-0 arm would differ by the same amount and the ordering assertion fails.
#[test]
#[ignore = "real snapshot + a real Wan LoRA; run with --ignored on macOS (see module doc)"]
fn real_wan_style_lora_loads_and_changes_the_render() {
    let w = env_usize("KREA_LORA_W", 832);
    let h = env_usize("KREA_LORA_H", 480);
    let frames = env_usize("KREA_LORA_FRAMES", 13);
    let lora = require_lora("KREA_STYLE_LORA");
    let lora_path = lora.clone();
    let req = request(w, h, frames);

    let base = render(Vec::new(), &req, "baseline");
    let with_lora = render(
        vec![AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)],
        &req,
        "lora@1.0",
    );
    let zeroed = render(
        vec![AdapterSpec::new(lora, 0.0, AdapterKind::Lora)],
        &req,
        "lora@0.0",
    );

    let applied = mean_abs_delta(&base, &with_lora);
    let noop = mean_abs_delta(&base, &zeroed);
    println!("  mean |Δ| vs baseline: lora@1.0 = {applied:.3}, lora@0.0 = {noop:.3} (0..255)");

    // The count, not just the effect: a canonical style LoRA resolves exactly the per-block surface.
    let n = install_count(&lora_path);
    println!("  installed targets: {n} (surface width {ADAPTABLE_SURFACE_WIDTH})");
    assert_eq!(
        n, STYLE_LORA_TARGETS,
        "a plain Wan style LoRA must resolve exactly the {STYLE_LORA_TARGETS} per-block Linears"
    );

    assert!(
        applied > 1.0,
        "the LoRA did not measurably change the render (mean |Δ| {applied:.3}) — it loaded but is \
         inert"
    );
    assert!(
        applied > noop * 4.0,
        "the LoRA's effect ({applied:.3}) is not clearly above the scale-0 floor ({noop:.3}) — the \
         'it changed the output' claim would not be attributable to the adapter"
    );
}

/// The **step-distill** class: a real Wan-T2V LoRA that also targets the seven whole-model globals must
/// now install rather than hard-error. This is the S13 globals decision under test on a real file, so a
/// regression that re-narrows the surface fails here with the installer's own unmatched-target message.
///
/// Optional: skipped when `KREA_DISTILL_LORA` is unset, because it needs a ~1.2 GB second LoRA.
#[test]
#[ignore = "real snapshot + a real Wan step-distill LoRA; run with --ignored on macOS"]
fn real_wan_step_distill_lora_installs_over_the_widened_globals() {
    let Ok(path) = std::env::var("KREA_DISTILL_LORA") else {
        eprintln!("skip: set KREA_DISTILL_LORA to a real Wan T2V step-distill LoRA .safetensors");
        return;
    };
    let path = PathBuf::from(path);
    assert!(path.is_file(), "KREA_DISTILL_LORA: {}", path.display());

    let w = env_usize("KREA_LORA_W", 832);
    let h = env_usize("KREA_LORA_H", 480);
    let frames = env_usize("KREA_LORA_FRAMES", 13);
    let req = request(w, h, frames);

    let base = render(Vec::new(), &req, "baseline");
    let distilled = render(
        vec![AdapterSpec::new(path.clone(), 1.0, AdapterKind::Lora)],
        &req,
        "step-distill",
    );
    let delta = mean_abs_delta(&base, &distilled);
    println!("  step-distill mean |Δ| vs baseline: {delta:.3} (0..255)");
    assert!(
        delta > 1.0,
        "the step-distill LoRA installed but changed nothing (mean |Δ| {delta:.3})"
    );

    // **The regression gate on the sc-8446 globals decision.** 406 = 400 per-block + 6 globals. On the
    // pre-widening per-block-only surface this file did not install at all — `apply_adapters_strict`
    // hard-errored on the unmatched globals — so a count of 400 here means the widening was reverted,
    // and 407 would mean `patch_embedding` grew a low-rank pair (i.e. a different file).
    let n = install_count(&path);
    println!("  installed targets: {n} (surface width {ADAPTABLE_SURFACE_WIDTH})");
    assert_eq!(
        n, STEP_DISTILL_TARGETS,
        "a step-distill file must resolve {STEP_DISTILL_TARGETS} targets (400 per-block + 6 globals); \
         {STYLE_LORA_TARGETS} would mean the globals surface was re-narrowed"
    );
    assert_eq!(
        ADAPTABLE_SURFACE_WIDTH - n,
        1,
        "exactly one exposed global (patch_embedding, `.diff_b`-only) goes unmatched by this file"
    );
}
