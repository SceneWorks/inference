//! sc-2528: end-to-end Qwen-Image LoRA/LoKr adapter consumption against real weights.
//!
//! `#[ignore]`d — needs the real `Qwen/Qwen-Image` snapshot in the HF cache (env
//! `MLX_GEN_QWEN_SNAPSHOT`) and the adapter goldens from `tools/dump_qwen_adapter_golden.py`
//! (gitignored, local). Run:
//!   python3 crates/media/mlx-gen/tools/verify_adapter_parity_artifacts.py
//!   cargo test -p mlx-gen-qwen-image --release --test adapter_real_weights -- --ignored --nocapture --test-threads=1
//!
//! Gates: (1) the key→module map resolves the FULL fork `QwenLoRAMapping` surface (60 blocks x
//! modulation + attention + img/txt MLP) against the real module tree; (2) the public
//! `load(spec.with_adapters(…)).generate()` render matches the fork's LoRA *and* LoKr golden
//! (px>8); (3) a scale-0 adapter is a bit-exact no-op.

use std::path::PathBuf;

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_qwen_image::{apply_qwen_adapters, decoded_to_image, loader};
use mlx_rs::ops::array_eq;
use mlx_rs::Array;

fn snapshot() -> PathBuf {
    let p = std::env::var("MLX_GEN_QWEN_SNAPSHOT").unwrap_or_else(|_| panic!("set MLX_GEN_QWEN_SNAPSHOT to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"));
    PathBuf::from(p)
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden")
}

/// Locate `filename` in the PINNED lightx2v snapshot dir named by `env`, else in the first cached
/// `snapshots/<hash>/` under `MLX_GEN_MODELS_ROOT`.
///
/// `env` wins when set, and that is the CI contract rather than a convenience. The lane derives
/// `<MLX_GEN_MODELS_ROOT>/<repo_dir>/snapshots/<rev>` from the revision pinned in
/// `release/real-weight-models.toml`, exports it as `QWEN_LIGHTNING_SNAPSHOT` /
/// `QWEN_EDIT_LIGHTNING_SNAPSHOT` through `$GITHUB_ENV`, and verifies THAT dir with
/// `ensure_model_snapshot.py`. Without the override a lane can verify the pin and then load a
/// different revision that happens to be cached beside it (sc-17284) — the same hazard
/// `lightning_render_real_weights::lightx2v_file` documents, resolved the same way, so both steps
/// of one job resolve the same LoRA by the same rule. Set-but-missing is a hard error, never a
/// silent fall back to an unpinned sibling.
///
/// Unpinned (local dev, variable unset) the candidates are scanned and SORTED, so a box holding two
/// revisions at least resolves the same one run to run — but that is NOT the pin. Callers print the
/// resolved path, which is the only record of WHICH file a measurement came from.
fn hf_cache_file(env: &str, repo_dir: &str, filename: &str) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var(env) {
        let p = PathBuf::from(dir).join(filename);
        assert!(p.exists(), "{env} is set but {} is missing", p.display());
        return Some(p);
    }
    let home = std::env::var("MLX_GEN_MODELS_ROOT").ok()?;
    let snaps = PathBuf::from(home).join(format!("{repo_dir}/snapshots"));
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join(filename))
        .filter(|p| p.exists())
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

/// The lightx2v Lightning LoRAs to probe — the 8-step T2I and Edit-2511 variants (the 4-step
/// variants share the identical per-block key structure). `(label, pinned-snapshot-dir variable,
/// HF-cache repo dir, filename)`; both variables are exported by the lane from the manifest pin.
const LIGHTNING_LORAS: [(&str, &str, &str, &str); 2] = [
    (
        "qwen-image-lightning-8step",
        "QWEN_LIGHTNING_SNAPSHOT",
        "models--lightx2v--Qwen-Image-Lightning",
        "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors",
    ),
    (
        "qwen-image-edit-2511-lightning-8step",
        "QWEN_EDIT_LIGHTNING_SNAPSHOT",
        "models--lightx2v--Qwen-Image-Edit-2511-Lightning",
        "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    ),
];

/// Resolve every entry of `LIGHTNING_LORAS`, dropping any that is neither pinned nor cached. The
/// caller asserts the FULL set came back, so an absent file names itself instead of quietly
/// shrinking the gate to whatever happened to resolve.
fn lightning_loras() -> Vec<(&'static str, PathBuf)> {
    LIGHTNING_LORAS
        .iter()
        .filter_map(|(label, env, repo_dir, filename)| {
            hf_cache_file(env, repo_dir, filename).map(|p| (*label, p))
        })
        .collect()
}

/// Qwen-Image's transformer depth, and the per-block adapter targets the host map reaches — the
/// `60 x 14 = 840` surface `routing_map_covers_full_fork_surface` gates directly.
const QWEN_BLOCKS: usize = 60;
const QWEN_MODULES_PER_BLOCK: usize = 14;

/// The 12 per-block module classes a published lightx2v Qwen-Image Lightning file trains. This is
/// the FILE's surface, measured from its safetensors header — deliberately not the host's 14. The
/// two the host reaches and no Lightning release trains are `img_mod.1` and `txt_mod.1`; the test
/// prints them as the untouched remainder rather than asserting their absence twice.
const LIGHTNING_TRAINED_MODULES: [&str; 12] = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
];

/// sc-2909: the acceleration-LoRA **viability** gate. The real lightx2v Lightning LoRAs (T2I
/// `Qwen-Image-Lightning` + `Qwen-Image-Edit-2511-Lightning`) load through `apply_qwen_adapters`
/// with **zero silent drops**: every module the FILE addresses resolves on the real 60-block tree.
/// The merge math itself is the sc-2528 seam (already proven bit-exact); this confirms the Lightning
/// files address modules the host map reaches, and pins WHICH ones.
///
/// sc-17518 — this asserted `applied == 840` from sc-2909 until 2026-08. That number is the HOST
/// map's target count (`60 x 14`, what `routing_map_covers_full_fork_surface` gates), never a
/// measured application: the published files carry `60 x 12 = 720`, with `unmatched_paths` empty.
///
/// Evidence, all gathered against the pinned revisions in `release/real-weight-models.toml` — and
/// the resolution below now HONORS that pin (`hf_cache_file` prefers the lane-exported
/// `QWEN_LIGHTNING_SNAPSHOT` / `QWEN_EDIT_LIGHTNING_SNAPSHOT`), so the numbers keep describing the
/// file the assertion measures even on a box that later caches a second revision beside it:
///
/// * `Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors` @ `e74da8d4` has 2160 tensors —
///   60 blocks x 12 modules x {`lora_down.weight`, `lora_up.weight`, `alpha`}. No modulation stems.
/// * That filename has exactly ONE version in the repo's history: its LFS oid
///   `fe04c3ec1dd46f361fe29010027da4168c439f93143879731ca8d407935279c8` is identical at the pin and
///   at `598f480d` (2025-08-12), the only commit that ever uploaded it. There is no earlier revision
///   an `840` could have been measured against.
/// * Every other variant in the repo carries the same 12 classes, back to the initial content
///   commit `0aa0a344` (2025-08-09, `8steps-V1.0`): 4-step V1.0/V2.0, 8-step V1.1/V2.0, both the
///   bf16 and fp32 spellings, and the Edit siblings. Modulation has never been trained.
/// * `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors` @ `d74eba14` is the same shape:
///   2160 tensors, 60 x 12. The loop below reached it for the first time in sc-17518 — the old
///   assertion panicked on the first LoRA and it had never been measured.
///
/// So the gate asserts the file's own surface: per-CLASS coverage (a raw count cannot tell 60x12
/// from 40x18), all 60 blocks, exactly one adapter per target, and nothing unmatched.
#[test]
#[ignore = "needs real Qwen-Image weights + the cached lightx2v Lightning LoRAs"]
fn lightning_loras_apply_cleanly() {
    let loras = lightning_loras();
    // Not a floor: sc-17518 requires BOTH published surfaces to be measured, and a `!is_empty()`
    // check would report `1 passed` off the T2I file alone if the Edit-2511 one were absent.
    let resolved: Vec<&str> = loras.iter().map(|(label, _)| *label).collect();
    let expected: Vec<&str> = LIGHTNING_LORAS.iter().map(|(label, ..)| *label).collect();
    assert_eq!(
        resolved, expected,
        "every pinned Lightning LoRA must be measured, but only {resolved:?} resolved — set \
         QWEN_LIGHTNING_SNAPSHOT / QWEN_EDIT_LIGHTNING_SNAPSHOT to the snapshot dirs for the \
         revisions pinned in release/real-weight-models.toml, or cache both repos under \
         MLX_GEN_MODELS_ROOT",
    );
    let expected_applied = QWEN_BLOCKS * LIGHTNING_TRAINED_MODULES.len();
    for (label, path) in loras {
        // Fresh transformer per LoRA so each report reflects that file alone (no stacking).
        let mut t = loader::load_transformer(&snapshot()).unwrap();
        // The host's whole adapter surface, captured BEFORE anything is installed. The applied
        // count below is a fraction of this, not the whole of it, and holding both is what makes
        // the 720/840 gap a measurement rather than a bare constant.
        let surface = t.adaptable_paths();
        assert_eq!(
            surface.len(),
            QWEN_BLOCKS * QWEN_MODULES_PER_BLOCK,
            "host adapter surface drifted from {QWEN_BLOCKS} blocks x {QWEN_MODULES_PER_BLOCK}"
        );
        let report = apply_qwen_adapters(
            &mut t,
            &[AdapterSpec::new(path.clone(), 1.0, AdapterKind::Lora)],
        )
        .unwrap_or_else(|e| panic!("{label} ({}) failed to apply: {e}", path.display()));
        assert!(
            report.unmatched_paths.is_empty(),
            "{label}: {} unmatched target(s): {:?}",
            report.unmatched_paths.len(),
            report.unmatched_paths
        );

        // Which of the host's targets actually carry an adapter, grouped by per-block module class.
        let mut adapted: std::collections::BTreeMap<String, usize> = Default::default();
        let mut untouched: std::collections::BTreeSet<String> = Default::default();
        for p in &surface {
            let segs: Vec<&str> = p.split('.').collect();
            let installed = AdaptableHost::adaptable_mut(&mut t, &segs)
                .unwrap_or_else(|| panic!("{label}: enumerated {p} does not resolve"))
                .adapters()
                .len();
            // `transformer_blocks.<i>.<module>` → `<module>`.
            let module = p.splitn(3, '.').nth(2).unwrap().to_string();
            match installed {
                0 => {
                    untouched.insert(module);
                }
                1 => *adapted.entry(module).or_default() += 1,
                n => panic!("{label}: {p} carries {n} adapters from a single file"),
            }
        }
        println!(
            "{label} ({}): applied {} of {} host targets over {} module class(es); untouched {:?}; unmatched {:?}",
            path.display(),
            report.applied,
            surface.len(),
            adapted.len(),
            untouched,
            report.unmatched_paths,
        );

        let classes: Vec<&str> = adapted.keys().map(String::as_str).collect();
        // `adapted` is a BTreeMap, so `classes` is sorted; sort the expectation rather than relying
        // on the constant's hand-maintained declaration order.
        let mut expected_classes = LIGHTNING_TRAINED_MODULES;
        expected_classes.sort_unstable();
        assert_eq!(
            classes,
            expected_classes.to_vec(),
            "{label}: the trained module classes drifted from the pinned lightx2v surface"
        );
        for (module, blocks) in &adapted {
            assert_eq!(
                *blocks, QWEN_BLOCKS,
                "{label}: {module} adapted on {blocks} of {QWEN_BLOCKS} blocks"
            );
        }
        assert_eq!(
            report.applied,
            expected_applied,
            "{label}: expected {expected_applied} per-block modules ({QWEN_BLOCKS} blocks x {}) — \
             the host reaches {QWEN_MODULES_PER_BLOCK} per block, but no published lightx2v \
             Lightning file trains img_mod.1/txt_mod.1 (sc-17518)",
            LIGHTNING_TRAINED_MODULES.len(),
        );
    }
}

/// (1) The top-level `AdaptableHost` resolves every fork `QwenLoRAMapping` target (all per-block:
/// the modulation linears, joint attention, and the two stream MLPs; no globals) across the real
/// 60-block tree, and rejects off-surface paths.
#[test]
#[ignore = "needs real Qwen-Image weights"]
fn routing_map_covers_full_fork_surface() {
    let mut t = loader::load_transformer(&snapshot()).unwrap();
    let resolves = |t: &mut _, p: &str| -> bool {
        let segs: Vec<&str> = p.split('.').collect();
        AdaptableHost::adaptable_mut(t, &segs).is_some()
    };

    let targets = [
        "img_mod.1",
        "txt_mod.1",
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
        "txt_mlp.net.0.proj",
        "txt_mlp.net.2",
    ];
    for i in 0..60 {
        for tgt in targets {
            let p = format!("transformer_blocks.{i}.{tgt}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for p in [
        "transformer_blocks.60.attn.to_q",    // out of range
        "transformer_blocks.0.attn.to_out",   // missing .0
        "transformer_blocks.0.attn.add_q",    // internal name, not the file's add_q_proj
        "transformer_blocks.0.img_mlp.net.1", // gelu slot
        "img_in",                             // not a trained target
    ] {
        assert!(!resolves(&mut t, p), "expected {p} NOT to resolve");
    }
    println!("routing map covers the full fork QwenLoRAMapping surface (60 blocks x 14 targets)");
}

fn render(adapter: Option<(&str, AdapterKind, f32)>, golden_kind: &str) -> Vec<u8> {
    let g = Weights::from_file(golden_dir().join(format!("qwen_{golden_kind}_golden.safetensors")))
        .unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("width").unwrap().parse().unwrap();
    let h: u32 = g.metadata("height").unwrap().parse().unwrap();

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if let Some((file, kind, scale)) = adapter {
        spec = spec.with_adapters(vec![AdapterSpec {
            path: golden_dir().join(file),
            scale,
            kind,
            pass_scales: None,
            moe_expert: None,
        }]);
    }
    let generator = mlx_gen_qwen_image::provider_registry()
        .unwrap()
        .load("qwen_image", &spec)
        .unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap().pixels,
        other => panic!("expected Images, got {other:?}"),
    }
}

/// Count RGB8 pixels differing by >8 between two buffers.
fn px_gt8(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count()
}

/// Count elements where the Rust adapter residual differs from the fork adapter residual by >8.
fn residual_px_gt8(adapted: &[u8], base: &[u8], fork_adapted: &[u8], fork_base: &[u8]) -> usize {
    assert_eq!(adapted.len(), base.len());
    assert_eq!(adapted.len(), fork_adapted.len());
    assert_eq!(adapted.len(), fork_base.len());
    (0..adapted.len())
        .filter(|&index| {
            let rust_residual = adapted[index] as i32 - base[index] as i32;
            let fork_residual = fork_adapted[index] as i32 - fork_base[index] as i32;
            (rust_residual - fork_residual).abs() > 8
        })
        .count()
}

fn matching_fork_base(kind: &str) -> Vec<u8> {
    let ag =
        Weights::from_file(golden_dir().join(format!("qwen_{kind}_golden.safetensors"))).unwrap();
    let bg = Weights::from_file(golden_dir().join("qwen_image_golden.safetensors"))
        .expect("base golden qwen_image_golden.safetensors (dump it at the adapter config)");
    for k in [
        "prompt",
        "seed",
        "steps",
        "width",
        "height",
        "guidance",
        "reference_mflux_repository",
        "reference_mflux_revision",
        "reference_model_repository",
        "reference_model_revision",
        "reference_model_ref",
        "reference_model_subdirectory",
        "reference_model_path",
        "reference_model_inventory_sha256",
        "reference_provenance_sha256",
        "reference_runtime",
    ] {
        assert!(
            ag.metadata(k).is_some(),
            "adapter golden missing metadata {k}"
        );
        assert!(bg.metadata(k).is_some(), "base golden missing metadata {k}");
        assert_eq!(
            ag.metadata(k),
            bg.metadata(k),
            "base golden {k} != adapter golden {k} — regenerate qwen_image_golden.safetensors at the adapter config"
        );
    }
    decoded_to_image(bg.require("decoded").unwrap())
        .unwrap()
        .pixels
}

/// The base (no-adapter) render's px>8 vs the fork base golden, at the SAME config as the `kind`
/// adapter golden — the inherited bf16 toolchain drift floor the adapter render sits on. The base
/// golden (`qwen_image_golden.safetensors`) MUST be dumped at the adapter golden's
/// (seed, steps, size); a mismatch is a hard error (it would yield a bogus floor).
fn base_floor_px(kind: &str, pixels: &[u8]) -> usize {
    px_gt8(pixels, &matching_fork_base(kind))
}

fn print_residual_diagnostic(kind: &str, my_kind: AdapterKind) {
    let adapted = render(
        Some((&format!("qwen_{kind}_adapter.safetensors"), my_kind, 1.0)),
        kind,
    );
    let base = render(None, kind);
    let ag =
        Weights::from_file(golden_dir().join(format!("qwen_{kind}_golden.safetensors"))).unwrap();
    let fork_adapted = decoded_to_image(ag.require("decoded").unwrap())
        .unwrap()
        .pixels;
    let fork_base = matching_fork_base(kind);
    let residual_differ = residual_px_gt8(&adapted, &base, &fork_adapted, &fork_base);
    // A dropped adapter has a zero Rust residual, so its error against the frozen-fork residual is
    // exactly the fork adapter-vs-base delta.
    let zero_residual_control = px_gt8(&fork_adapted, &fork_base);
    println!(
        "SC15505_RESULT qwen_{kind} residual_samples_gt8={residual_differ} zero_residual_samples_gt8={zero_residual_control} rgb_samples={}",
        adapted.len()
    );
}

/// Pre-tightening diagnostic. The source/env-bound recorder runs this exact test first; a cap may
/// be locked only after the adapted residual is measured below the zero-residual control.
#[test]
#[ignore = "needs real Qwen-Image weights + adapter & base goldens (same config)"]
fn residual_mutation_diagnostic() {
    // Cargo prints `test <name> ... ` without a newline before test output.
    println!();
    print_residual_diagnostic("lora", AdapterKind::Lora);
    print_residual_diagnostic("lokr", AdapterKind::Lokr);
}

struct AdapterEffectMeasurement {
    effect_samples_gt8: usize,
    scale_zero_byte_differences: usize,
    applied: usize,
    unmatched: usize,
    rgb_samples: usize,
}

fn measure_adapter_effect(
    kind: &str,
    my_kind: AdapterKind,
    active: &[u8],
    base: &[u8],
) -> AdapterEffectMeasurement {
    let adapter_file = format!("qwen_{kind}_adapter.safetensors");
    let zero = render(Some((&adapter_file, my_kind, 0.0)), kind);
    let effect_samples_gt8 = px_gt8(active, base);
    let scale_zero_byte_differences = base.iter().zip(&zero).filter(|(a, b)| a != b).count();

    let mut transformer = loader::load_transformer(&snapshot()).unwrap();
    let report = apply_qwen_adapters(
        &mut transformer,
        &[AdapterSpec::new(
            golden_dir().join(adapter_file),
            1.0,
            my_kind,
        )],
    )
    .unwrap();
    let expected_applied = match kind {
        "lora" => 24,
        "lokr" => 21,
        _ => panic!("unsupported adapter golden kind {kind}"),
    };
    assert_eq!(
        report.applied, expected_applied,
        "{kind}: applied module count drift"
    );
    assert!(
        report.unmatched_paths.is_empty(),
        "{kind}: unmatched adapter paths: {:?}",
        report.unmatched_paths
    );
    assert_eq!(
        scale_zero_byte_differences, 0,
        "{kind}: scale-zero adapter must remain a bit-exact no-op"
    );
    AdapterEffectMeasurement {
        effect_samples_gt8,
        scale_zero_byte_differences,
        applied: report.applied,
        unmatched: report.unmatched_paths.len(),
        rgb_samples: active.len(),
    }
}

fn print_adapter_effect_diagnostic(kind: &str, my_kind: AdapterKind) {
    let adapter_file = format!("qwen_{kind}_adapter.safetensors");
    let active = render(Some((&adapter_file, my_kind, 1.0)), kind);
    let base = render(None, kind);
    let measurement = measure_adapter_effect(kind, my_kind, &active, &base);
    println!(
        "SC15505_RESULT qwen_{kind} effect_samples_gt8={} scale_zero_byte_differences={} applied={} unmatched={} rgb_samples={}",
        measurement.effect_samples_gt8,
        measurement.scale_zero_byte_differences,
        measurement.applied,
        measurement.unmatched,
        measurement.rgb_samples,
    );
}

/// Pre-tightening same-runtime effect diagnostic. A fixed nonzero minimum is locked only after
/// these active-vs-base counts are captured by the source/env-bound recorder.
#[test]
#[ignore = "needs real Qwen-Image weights + adapter & base goldens (same config)"]
fn adapter_effect_diagnostic() {
    // Cargo prints `test <name> ... ` without a newline before test output.
    println!();
    print_adapter_effect_diagnostic("lora", AdapterKind::Lora);
    print_adapter_effect_diagnostic("lokr", AdapterKind::Lokr);
}

fn assert_matches_golden(kind: &str, my_kind: AdapterKind) {
    let pixels = render(
        Some((&format!("qwen_{kind}_adapter.safetensors"), my_kind, 1.0)),
        kind,
    );
    let g =
        Weights::from_file(golden_dir().join(format!("qwen_{kind}_golden.safetensors"))).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = px_gt8(&pixels, &gimg.pixels);
    let no_adapter = render(None, kind);

    // Original floor-relative fork gate: this constrains the adapted render to the inherited
    // same-config Rust-base-vs-fork-base drift plus explicit headroom. It is intentionally separate
    // from the same-runtime adapter-effect oracle because Qwen's direct and residual fork metrics
    // cannot distinguish an active adapter from a dropped one.
    let base = base_floor_px(kind, &no_adapter);
    let cap = base * 2 + pixels.len() / 200;
    let effect = measure_adapter_effect(kind, my_kind, &pixels, &no_adapter);
    let minimum_samples_gt8 = match kind {
        "lora" => 22_399,
        "lokr" => 19_704,
        _ => panic!("unsupported adapter golden kind {kind}"),
    };
    let pct = |n: usize| n as f64 / pixels.len() as f64 * 100.0;
    println!(
        "✓ qwen {kind} adapter render: {differ} px>8 ({:.4}%); base floor {base} ({:.4}%); cap {cap} ({:.4}%)",
        pct(differ),
        pct(base),
        pct(cap),
    );
    println!(
        "SC15505_RESULT qwen_{kind} samples_gt8={differ} base_floor={base} cap={cap} effect_samples_gt8={} minimum_samples_gt8={minimum_samples_gt8} scale_zero_byte_differences={} applied={} unmatched={} rgb_samples={}",
        effect.effect_samples_gt8,
        effect.scale_zero_byte_differences,
        effect.applied,
        effect.unmatched,
        effect.rgb_samples,
    );
    assert!(
        differ <= cap,
        "qwen {kind} adapter render diverges beyond the base floor: {differ} px ({:.3}%) > cap {cap} px (base {base})",
        pct(differ),
    );
    assert!(
        effect.effect_samples_gt8 >= minimum_samples_gt8,
        "qwen {kind} adapter effect fell below the locked same-runtime floor: {} < {minimum_samples_gt8}",
        effect.effect_samples_gt8,
    );
}

#[test]
#[ignore = "needs real Qwen-Image weights + adapter & base goldens (same config)"]
fn lora_render_matches_fork_golden() {
    assert_matches_golden("lora", AdapterKind::Lora);
}

#[test]
#[ignore = "needs real Qwen-Image weights + adapter & base goldens (same config)"]
fn lokr_render_matches_fork_golden() {
    assert_matches_golden("lokr", AdapterKind::Lokr);
}

/// Diagnostic (per the divergence-is-not-rounding rule): the Rust base render (no adapter) vs the
/// fork base golden at the adapter golden's config — the floor the LoRA/LoKr px>8 numbers inherit
/// and the `assert_matches_golden` cap is derived from. Needs `qwen_image_golden.safetensors`
/// dumped at the adapter config (`tools/dump_qwen_image_golden.py`; default 512²).
#[test]
#[ignore = "needs real Qwen-Image weights + base golden at the adapter config"]
fn base_render_drift_attributes_adapter_gap() {
    // The inherited bf16 toolchain floor (also folded into assert_matches_golden's cap).
    println!(
        "qwen base (no adapter) vs fork base: {} px>8",
        base_floor_px("lora", &render(None, "lora"))
    );
}

#[test]
#[ignore = "needs real Qwen-Image weights + adapter golden"]
fn scale_zero_adapter_is_noop() {
    let base = render(None, "lora");
    let zero = render(
        Some(("qwen_lora_adapter.safetensors", AdapterKind::Lora, 0.0)),
        "lora",
    );
    let differ = base.iter().zip(&zero).filter(|(a, b)| a != b).count();
    println!("✓ qwen scale-0 adapter no-op: {differ} px differ from the no-adapter render");
    assert_eq!(differ, 0, "scale-0 adapter must be a bit-exact no-op");
}

/// The single installed LoRA's `(a, b)` arrays, or panic.
fn lora_arrays(adapters: &[Adapter]) -> (Array, Array) {
    match adapters {
        [Adapter::Lora { a, b, .. }] => (a.clone(), b.clone()),
        _ => panic!("expected exactly one LoRA adapter, got {}", adapters.len()),
    }
}

/// sc-2618: a kohya `lora_unet_` file resolves the SAME modules and installs the byte-identical
/// adapter as the equivalent PEFT file, on the REAL Qwen 60-block tree. Drift guard (every kohya
/// path resolves), collision-free flattening, and a fused/off-surface key errors loudly.
#[test]
#[ignore = "needs real Qwen-Image weights"]
fn kohya_matches_peft_on_real_tree() {
    let none = None as Option<&std::collections::HashMap<String, String>>;
    let mut probe = loader::load_transformer(&snapshot()).unwrap();
    let paths = probe.adaptable_paths();
    assert!(!paths.is_empty(), "no kohya targets enumerated");
    for p in &paths {
        let segs: Vec<&str> = p.split('.').collect();
        assert!(
            AdaptableHost::adaptable_mut(&mut probe, &segs).is_some(),
            "drift: enumerated {p} does not resolve via adaptable_mut"
        );
    }
    let flat: std::collections::BTreeSet<String> =
        paths.iter().map(|p| p.replace('.', "_")).collect();
    assert_eq!(
        flat.len(),
        paths.len(),
        "two paths collide when flattened to a kohya stem"
    );

    // One on-disk spelling per module (drop a `.0` alias when its bare sibling is enumerated; a
    // no-op for Qwen, which has no aliases).
    let targets: Vec<String> = paths
        .iter()
        .filter(|p| match p.strip_suffix(".0") {
            Some(base) => !paths.iter().any(|q| q.as_str() == base),
            None => true,
        })
        .cloned()
        .collect();

    let r = 2i32;
    let mut kohya: Vec<(String, Array)> = Vec::new();
    let mut peft: Vec<(String, Array)> = Vec::new();
    for p in &targets {
        let segs: Vec<&str> = p.split('.').collect();
        let shape = AdaptableHost::adaptable_mut(&mut probe, &segs)
            .unwrap()
            .base_shape();
        let (out, inp) = (shape[0], shape[1]);
        let a = Array::from_slice(
            &(0..r * inp)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.001)
                .collect::<Vec<_>>(),
            &[r, inp],
        );
        let b = Array::from_slice(
            &(0..out * r)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.001)
                .collect::<Vec<_>>(),
            &[out, r],
        );
        let alpha = Array::from_slice(&[4.0f32], &[1]);
        let stem = p.replace('.', "_");
        kohya.push((format!("lora_unet_{stem}.lora_down.weight"), a.clone()));
        kohya.push((format!("lora_unet_{stem}.lora_up.weight"), b.clone()));
        kohya.push((format!("lora_unet_{stem}.alpha"), alpha.clone()));
        peft.push((format!("transformer.{p}.lora_A.weight"), a));
        peft.push((format!("transformer.{p}.lora_B.weight"), b));
        peft.push((format!("transformer.{p}.alpha"), alpha));
    }
    let dir = std::env::temp_dir().join(format!("mlx_gen_qwen_kohya_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (kpath, ppath) = (dir.join("kohya.safetensors"), dir.join("peft.safetensors"));
    Array::save_safetensors(
        kohya
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        none,
        &kpath,
    )
    .unwrap();
    Array::save_safetensors(
        peft.iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        none,
        &ppath,
    )
    .unwrap();

    let mut tk = loader::load_transformer(&snapshot()).unwrap();
    let rk = apply_qwen_adapters(
        &mut tk,
        &[AdapterSpec {
            path: kpath,
            scale: 0.8,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .unwrap();
    assert_eq!(rk.applied, targets.len(), "kohya: not all targets applied");
    assert!(
        rk.unmatched_paths.is_empty(),
        "kohya unmatched: {:?}",
        rk.unmatched_paths
    );

    let mut tp = loader::load_transformer(&snapshot()).unwrap();
    apply_qwen_adapters(
        &mut tp,
        &[AdapterSpec {
            path: ppath,
            scale: 0.8,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .unwrap();

    for p in &targets {
        let segs: Vec<&str> = p.split('.').collect();
        let (ka, kb) = lora_arrays(
            AdaptableHost::adaptable_mut(&mut tk, &segs)
                .unwrap()
                .adapters(),
        );
        let (pa, pb) = lora_arrays(
            AdaptableHost::adaptable_mut(&mut tp, &segs)
                .unwrap()
                .adapters(),
        );
        assert!(
            array_eq(&ka, &pa, false).unwrap().item::<bool>()
                && array_eq(&kb, &pb, false).unwrap().item::<bool>(),
            "kohya and peft installed different adapters at {p}"
        );
    }
    println!(
        "✓ kohya ≡ peft across {} Qwen modules (byte-identical adapters)",
        targets.len()
    );

    // A fused/off-surface kohya key (Qwen has no fused `attn.qkv`) is surfaced and errors.
    let small = Array::from_slice(&[0.01f32], &[1, 1]);
    let opath = dir.join("kohya_offsurface.safetensors");
    Array::save_safetensors(
        vec![
            (
                "lora_unet_transformer_blocks_0_attn_qkv.lora_down.weight",
                &small,
            ),
            (
                "lora_unet_transformer_blocks_0_attn_qkv.lora_up.weight",
                &small,
            ),
        ],
        none,
        &opath,
    )
    .unwrap();
    let mut to = loader::load_transformer(&snapshot()).unwrap();
    assert!(
        apply_qwen_adapters(
            &mut to,
            &[AdapterSpec {
                path: opath,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .is_err(),
        "an off-surface kohya key must error, not silently drop"
    );
}
