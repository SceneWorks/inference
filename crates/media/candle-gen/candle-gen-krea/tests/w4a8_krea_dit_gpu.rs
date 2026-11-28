//! **SW-6135 — the Krea 2 Turbo W4A8 tier, end to end, on real weights** (feature SW-F287).
//!
//! The tier is built offline from the bf16 master by `examples/krea-w4a8-pack.rs`
//! (`candle_gen_krea::w4a8_tier::build_w4a8_tier`) and every render loads through a production
//! `LoadSpec` entry point — the route a SceneWorks install takes:
//!
//! * a native single-file tier (W4A8, NVFP4) through [`candle_gen_krea::load_from_native_dit_file`]:
//!   `LoadSpec { weights: File(native DiT), components[BASE_SNAPSHOT_COMPONENT]: Dir(bf16 base) }`,
//!   prepared (`LoadSpec::prepare_file_sources`) and built by the registered provider, which reads
//!   the pinned file batch through `LoadSpec::read_files_unchanged` (the validated-content receipt
//!   these proofs register);
//! * a snapshot tier (bf16, q8, q4) through the `krea_2_turbo` registry provider with
//!   `LoadSpec { weights: Dir(snapshot) }`.
//!
//! Two ignored proofs:
//!
//! 1. [`w4a8_krea_dit_gpu_fixed_seed_render_vs_bf16`] — fixed-seed 1024² 8-step Turbo renders on
//!    the bf16 snapshot and on the W4A8 tier (plus, when their fixtures are set, the q8 / q4
//!    snapshot tiers, the Kitchen NVFP4 single file and the blanket W4A8 measurement arm), each in
//!    its own load. Records image PSNR vs the bf16 render, s/step (median inter-step interval) and
//!    the device peak over the idle baseline, writes every PNG for visual inspection, and asserts
//!    the W4A8 render clears the declared floor. The W4A8 tier renders twice: resident, and through
//!    the advertised `StagedResidency` shape (an adapter-free Sequential + DeferredMaterialization
//!    File load with a staged request), whose DiT must load resident inside the sequential heavy
//!    phase rather than being routed into native block streaming (PR #1526 review); when the
//!    Kitchen NVFP4 fixture is set it renders the same two ways. Each staged render must match its
//!    resident twin near-exactly.
//! 2. [`w4a8_krea_dit_gpu_additive_lora_matches_q8_class`] — a real Krea 2 LoRA installs additively
//!    over the W4A8 trunk onto exactly the targets it resolves on the q8 trunk, and the adapted
//!    W4A8 render lands in the same result class as the adapted q8 render.
//!
//! ```text
//! KREA_TURBO_BF16_DIR=E:\huggingface\hub\models--SceneWorks--krea-2-turbo-mlx\snapshots\<rev>\bf16
//! KREA_TURBO_W4A8_FILE=<artifacts>\krea2_turbo_w4a8.safetensors
//! W4A8_KREA_OUT_DIR=<artifacts>\renders            (never on E:)
//! optional: KREA_TURBO_Q8_DIR, KREA_TURBO_Q4_DIR, KREAMANIA_VARIANT7 (NVFP4 file),
//!           KREA_TURBO_W4A8_BLANKET_FILE; KREA_2_LORA for proof 2 (with KREA_TURBO_Q8_DIR)
//! CUDA_VISIBLE_DEVICES=0 cargo test -p candle-gen-krea --release --features cuda --test integration \
//!   w4a8_krea_dit_gpu:: -- --ignored --nocapture --test-threads 1
//! ```
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, GenerationMemory, GenerationOutput, GenerationRequest, Image,
    LoadShape, LoadSpec, OffloadPolicy, Progress, WeightsSource,
};
use candle_gen::testkit::{save_image_png, VramProbe};
use candle_gen_krea::loader::Weights;
use candle_gen_krea::native_mapping::DeclaredLogicalShapes;
use candle_gen_krea::{Krea2Config, Krea2Transformer};

const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";
const SEED: u64 = 0x5_6135;
const EDGE: u32 = 1024;
const STEPS: u32 = 8;

/// The W4A8 render must reach at least this image PSNR vs the bf16 render. W4A8 is a lossy,
/// opt-in tier (epic E2); the bar is the one the catalog already ships beside it — the MLX q4
/// tier, whose image PSNR vs bf16 is ~22.7 dB on Krea 2 Turbo (sc-9601). A W4A8 render below it
/// would be a tier that is worse than the 4-bit tier it is offered next to.
const W4A8_PSNR_FLOOR_DB: f64 = 22.0;

/// A staged load runs the same resident DiT as the resident load (only the text encoder's lifetime
/// differs), so its render must match the resident render of the same tier near-exactly; the
/// W4A8 pair is bit-identical on sm_120. 40 dB leaves room only for kernel-order noise.
const STAGED_PARITY_FLOOR_DB: f64 = 40.0;

/// Exactly the projections `W4A8Policy::Mixed` packs on Krea 2 Turbo: the six normalized-input
/// leaves of single-stream blocks 4..=26.
const MIXED_PACKED_PROJECTIONS: usize = 23 * 6;

fn required(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(name)
            .unwrap_or_else(|_| panic!("this explicitly selected real-weight test needs {name}")),
    )
}

fn optional(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

fn out_dir() -> PathBuf {
    let dir = required("W4A8_KREA_OUT_DIR");
    assert!(
        !dir.to_string_lossy().to_ascii_uppercase().starts_with("E:"),
        "W4A8_KREA_OUT_DIR must not be the read-only model drive"
    );
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[derive(Clone, Copy)]
enum Tier<'a> {
    /// A self-describing snapshot root (bf16, or the MLX-packed q8/q4 turnkey).
    Snapshot(&'a Path),
    /// A native single-file DiT over the bf16 base snapshot (W4A8, NVFP4).
    Native { base: &'a Path, dit: &'a Path },
    /// [`Tier::Native`] loaded Sequential + DeferredMaterialization: the staged-residency shape.
    NativeStaged { base: &'a Path, dit: &'a Path },
}

struct Measured {
    image: Image,
    sec_per_step: f64,
    peak_gb: f64,
    /// `generate` start to the end of conditioning (the generator loads lazily, so this is the
    /// component load plus the prompt encode), or to the first step when no conditioning boundary
    /// is reported.
    load_s: f64,
}

fn request(edge: u32, steps: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.to_owned(),
        width: edge,
        height: edge,
        steps: Some(steps),
        seed: Some(SEED),
        ..Default::default()
    }
}

/// One production load + render in its own scope (the generator drops before the next tier
/// loads), with the device peak over the idle baseline and the median inter-step interval.
fn render(tier: Tier<'_>, adapters: &[AdapterSpec], req: &GenerationRequest) -> Measured {
    let mut probe = VramProbe::start_rendered();
    let phase = probe.phase();
    let generator = match tier {
        Tier::Snapshot(root) => candle_gen_krea::provider_registry()
            .expect("the Krea provider registry")
            .load(
                "krea_2_turbo",
                &LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
                    .with_adapters(adapters.to_vec()),
            ),
        Tier::Native { base, dit } => candle_gen_krea::load_from_native_dit_file(
            dit,
            base,
            adapters,
            candle_gen_krea::descriptor(),
        ),
        Tier::NativeStaged { base, dit } => {
            let mut spec = LoadSpec::new(WeightsSource::File(dit.to_path_buf()))
                .with_component(
                    candle_gen::gen_core::BASE_SNAPSHOT_COMPONENT,
                    WeightsSource::Dir(base.to_path_buf()),
                )
                .with_adapters(adapters.to_vec())
                .with_offload_policy(OffloadPolicy::Sequential)
                .with_load_shape(LoadShape::DeferredMaterialization);
            spec.prepare_file_sources()
                .and_then(|()| candle_gen_krea::provider_registry()?.load("krea_2_turbo", &spec))
        }
    }
    .expect("the tier loads through its production LoadSpec route");
    probe.end_load(phase);

    let mut steps: Vec<Instant> = Vec::new();
    let mut conditioned: Option<Instant> = None;
    let phase = probe.phase();
    let started = Instant::now();
    let output = generator
        .generate(req, &mut |p| match p {
            Progress::Step { .. } => steps.push(Instant::now()),
            Progress::ConditioningComplete => {
                conditioned.get_or_insert_with(Instant::now);
            }
            _ => {}
        })
        .expect("the tier renders");
    probe.end_gen(phase);
    drop(generator);
    let GenerationOutput::Images(images) = output else {
        panic!("expected image output")
    };
    assert_eq!(images.len(), 1, "one image per request");
    assert_eq!(steps.len(), STEPS as usize, "every denoising step ran");

    let load_s = conditioned
        .or_else(|| steps.first().copied())
        .map_or(f64::NAN, |t| (t - started).as_secs_f64());
    let mut intervals: Vec<f64> = steps
        .windows(2)
        .map(|w| (w[1] - w[0]).as_secs_f64())
        .collect();
    intervals.sort_by(f64::total_cmp);
    let sec_per_step = intervals
        .get(intervals.len() / 2)
        .copied()
        .unwrap_or(f64::NAN);
    let report = probe.report().assert_trustworthy(96.0);
    Measured {
        image: images.into_iter().next().expect("one image"),
        sec_per_step,
        peak_gb: report.peak_gb,
        load_s,
    }
}

fn psnr(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len(), "renders share geometry");
    let mse = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum::<f64>()
        / a.pixels.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }
}

/// The tier file is what the builder wrote: native W4A8, exactly the mixed-policy projection set,
/// every one delegated with its five tensors, and a key surface the architecture accounts for.
fn assert_tier_structure(dit: &Path, cfg: &Krea2Config, expected_projections: Option<usize>) {
    let weights = Weights::from_native_file_for(
        dit,
        &Device::Cpu,
        DType::BF16,
        DeclaredLogicalShapes::FromConfig(cfg),
    )
    .expect("the W4A8 tier plans");
    assert!(
        weights.is_native_w4a8(),
        "{} is not a W4A8 tier",
        dit.display()
    );
    let delegated = weights.w4a8_delegated_keys().count();
    assert_eq!(delegated % 5, 0);
    if let Some(expected) = expected_projections {
        assert_eq!(delegated / 5, expected, "packed projection census");
    }
    candle_gen_krea::convert::validate_native_transformer(&weights, cfg)
        .expect("the tier's tensor surface is exactly the architecture plus its W4A8 operands");
    println!(
        "[sw-6135] {}: {} W4A8 projections, {:.2} GB packed operands",
        dit.display(),
        delegated / 5,
        weights.w4a8_resident_bytes() as f64 / 1e9
    );
}

/// **AC: Krea 2 Turbo loads the W4A8 tier and renders a fixed-seed 1024² image; PSNR vs the bf16
/// render is recorded next to the q4 / q8 / NVFP4 numbers.**
#[test]
#[ignore = "real-weight GPU test: needs the Krea 2 Turbo bf16 snapshot, a built W4A8 tier and a CUDA sm_80+ device"]
fn w4a8_krea_dit_gpu_fixed_seed_render_vs_bf16() {
    let bf16 = required("KREA_TURBO_BF16_DIR");
    let w4a8 = required("KREA_TURBO_W4A8_FILE");
    let out = out_dir();
    let cfg = Krea2Config::from_snapshot(&bf16).expect("bf16 snapshot config");
    assert_tier_structure(&w4a8, &cfg, Some(MIXED_PACKED_PROJECTIONS));
    let blanket = optional("KREA_TURBO_W4A8_BLANKET_FILE");
    if let Some(blanket) = &blanket {
        assert_tier_structure(blanket, &cfg, None);
    }

    let req = request(EDGE, STEPS);
    let reference = render(Tier::Snapshot(&bf16), &[], &req);
    save_image_png(&out.join("bf16.png"), &reference.image);

    let mut rows: Vec<(String, Measured)> = Vec::new();
    rows.push((
        "w4a8 (mixed)".into(),
        render(
            Tier::Native {
                base: &bf16,
                dit: &w4a8,
            },
            &[],
            &req,
        ),
    ));
    // PR #1526 review: the same W4A8 tier through the advertised StagedResidency shape. Before the
    // fix this adapter-free Sequential + Deferred File load was implicitly armed for native block
    // streaming and refused after conditioning; its DiT must now load resident in the heavy phase.
    let staged_req = GenerationRequest {
        memory: Some(GenerationMemory {
            stage_residency: true,
            ..Default::default()
        }),
        ..request(EDGE, STEPS)
    };
    rows.push((
        "w4a8 (mixed, staged)".into(),
        render(
            Tier::NativeStaged {
                base: &bf16,
                dit: &w4a8,
            },
            &[],
            &staged_req,
        ),
    ));
    if let Some(blanket) = &blanket {
        rows.push((
            "w4a8 (blanket)".into(),
            render(
                Tier::Native {
                    base: &bf16,
                    dit: blanket,
                },
                &[],
                &req,
            ),
        ));
    }
    for (label, var) in [("q8", "KREA_TURBO_Q8_DIR"), ("q4", "KREA_TURBO_Q4_DIR")] {
        if let Some(root) = optional(var) {
            rows.push((label.into(), render(Tier::Snapshot(&root), &[], &req)));
        }
    }
    if let Some(nvfp4) = optional("KREAMANIA_VARIANT7") {
        rows.push((
            "nvfp4 (Kitchen)".into(),
            render(
                Tier::Native {
                    base: &bf16,
                    dit: &nvfp4,
                },
                &[],
                &req,
            ),
        ));
        // The same staged-residency shape for the Kitchen NVFP4 tier (PR #1526 review): its DiT
        // must load resident in the sequential heavy phase, never implicitly block-streamed.
        rows.push((
            "nvfp4 (Kitchen, staged)".into(),
            render(
                Tier::NativeStaged {
                    base: &bf16,
                    dit: &nvfp4,
                },
                &[],
                &staged_req,
            ),
        ));
    }

    println!(
        "\n[sw-6135] Krea 2 Turbo, {EDGE}x{EDGE}, {STEPS} steps, seed {SEED:#x}, sm_120 (RTX PRO 6000 Blackwell Max-Q, GPU 0)"
    );
    println!(
        "| tier | PSNR vs bf16 (dB) | s/step | peak VRAM over idle (GB) | load + encode (s) |"
    );
    println!("|---|---:|---:|---:|---:|");
    println!(
        "| bf16 | — | {:.3} | {:.1} | {:.1} |",
        reference.sec_per_step, reference.peak_gb, reference.load_s
    );
    for (label, m) in &rows {
        let file = format!("{}.png", label.replace([' ', '(', ')'], "_"));
        save_image_png(&out.join(file), &m.image);
        println!(
            "| {label} | {:.2} | {:.3} | {:.1} | {:.1} |",
            psnr(&m.image, &reference.image),
            m.sec_per_step,
            m.peak_gb,
            m.load_s
        );
    }

    let w4a8_psnr = psnr(&rows[0].1.image, &reference.image);
    assert!(
        w4a8_psnr >= W4A8_PSNR_FLOOR_DB,
        "the W4A8 render is {w4a8_psnr:.2} dB from bf16, below the {W4A8_PSNR_FLOOR_DB} dB floor"
    );
    let staged_psnr = psnr(&rows[1].1.image, &reference.image);
    let w4a8_parity = psnr(&rows[1].1.image, &rows[0].1.image);
    println!("[sw-6135] staged W4A8 vs resident W4A8: {w4a8_parity:.2} dB");
    assert!(
        w4a8_parity >= STAGED_PARITY_FLOOR_DB,
        "the staged W4A8 render is {w4a8_parity:.2} dB from the resident W4A8 render"
    );
    assert!(
        staged_psnr >= W4A8_PSNR_FLOOR_DB,
        "the staged W4A8 render is {staged_psnr:.2} dB from bf16, below the {W4A8_PSNR_FLOOR_DB} dB floor"
    );
    let by_label = |label: &str| rows.iter().find(|(l, _)| l == label).map(|(_, m)| m);
    if let (Some(resident), Some(staged)) = (
        by_label("nvfp4 (Kitchen)"),
        by_label("nvfp4 (Kitchen, staged)"),
    ) {
        let parity = psnr(&staged.image, &resident.image);
        println!("[sw-6135] staged NVFP4 vs resident NVFP4: {parity:.2} dB");
        assert!(
            parity >= STAGED_PARITY_FLOOR_DB,
            "the staged NVFP4 render is {parity:.2} dB from the resident NVFP4 render"
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED w4a8_krea_dit_gpu::w4a8_krea_dit_gpu_fixed_seed_render_vs_bf16");
}

fn lora(path: &Path) -> Vec<AdapterSpec> {
    vec![AdapterSpec::new(path.to_path_buf(), 1.0, AdapterKind::Lora)]
}

/// Install the LoRA on one trunk and return `(applied, skipped_keys, skipped_targets)`.
fn install_census(
    weights: &Weights,
    cfg: &Krea2Config,
    adapters: &[AdapterSpec],
) -> (usize, usize, usize) {
    let mut trunk = Krea2Transformer::load(weights, cfg).expect("trunk builds");
    let report = candle_gen_krea::install_additive(&mut trunk, adapters, 0)
        .expect("the LoRA installs additively");
    (
        report.applied,
        report.skipped_keys,
        report.skipped_targets.len(),
    )
}

/// **AC: an additive LoRA applies on the W4A8 tier with the same result class as on q8.**
///
/// "Same result class", stated as three checks: (1) the install resolves exactly the targets it
/// resolves on q8 (the same applied / skipped census), so no projection was dropped because it is
/// packed; (2) on both tiers the adapter visibly moves the render; (3) the adapted W4A8 render is
/// closer to the adapted q8 render than to its own un-adapted render, i.e. the LoRA's effect, not
/// the tier, decides what the W4A8 image looks like.
#[test]
#[ignore = "real-weight GPU test: needs the Krea 2 Turbo bf16 + q8 snapshots, a built W4A8 tier, a Krea 2 LoRA and a CUDA sm_80+ device"]
fn w4a8_krea_dit_gpu_additive_lora_matches_q8_class() {
    let bf16 = required("KREA_TURBO_BF16_DIR");
    let q8 = required("KREA_TURBO_Q8_DIR");
    let w4a8 = required("KREA_TURBO_W4A8_FILE");
    let lora_path = required("KREA_2_LORA");
    let out = out_dir();
    let cfg = Krea2Config::from_snapshot(&bf16).expect("bf16 snapshot config");
    let adapters = lora(&lora_path);
    let device = Device::new_cuda(0).expect("CUDA device 0");

    let q8_census = {
        let w = Weights::from_dir(&q8.join("transformer"), &device, DType::BF16).unwrap();
        install_census(&w, &cfg, &adapters)
    };
    let w4a8_census = {
        let w = Weights::from_native_file_for(
            &w4a8,
            &device,
            DType::BF16,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )
        .unwrap()
        .with_int8_context(candle_gen::quant::Int8Context::new(&device).unwrap());
        install_census(&w, &cfg, &adapters)
    };
    println!("[sw-6135] LoRA install census (applied, skipped keys, skipped targets): q8 {q8_census:?}, w4a8 {w4a8_census:?}");
    assert_eq!(
        w4a8_census, q8_census,
        "the W4A8 trunk hosts exactly the q8 trunk's targets"
    );
    // Equal census is the claim; a target the DiT surface does not carry (the darkbrush LoRA's
    // `text_fusion.projector`, which no Krea tier exposes to residuals) skips identically on both.
    assert!(q8_census.0 > 0 && q8_census.1 == 0);

    let req = request(EDGE, STEPS);
    let q8_base = render(Tier::Snapshot(&q8), &[], &req);
    let q8_lora = render(Tier::Snapshot(&q8), &adapters, &req);
    let w4_base = render(
        Tier::Native {
            base: &bf16,
            dit: &w4a8,
        },
        &[],
        &req,
    );
    let w4_lora = render(
        Tier::Native {
            base: &bf16,
            dit: &w4a8,
        },
        &adapters,
        &req,
    );
    for (name, m) in [
        ("q8_base", &q8_base),
        ("q8_lora", &q8_lora),
        ("w4a8_base", &w4_base),
        ("w4a8_lora", &w4_lora),
    ] {
        save_image_png(&out.join(format!("{name}.png")), &m.image);
    }
    let q8_effect = psnr(&q8_lora.image, &q8_base.image);
    let w4_effect = psnr(&w4_lora.image, &w4_base.image);
    let adapted_cross = psnr(&w4_lora.image, &q8_lora.image);
    let base_cross = psnr(&w4_base.image, &q8_base.image);
    println!(
        "[sw-6135] LoRA effect PSNR(adapted, base): q8 {q8_effect:.2} dB, w4a8 {w4_effect:.2} dB | \
         PSNR(w4a8, q8): adapted {adapted_cross:.2} dB, base {base_cross:.2} dB | s/step w4a8+LoRA {:.3} vs w4a8 {:.3}",
        w4_lora.sec_per_step, w4_base.sec_per_step
    );
    assert!(
        q8_effect < 30.0 && w4_effect < 30.0,
        "the LoRA visibly moves both renders"
    );
    assert!(
        adapted_cross > w4_effect,
        "the adapted W4A8 render must sit closer to the adapted q8 render ({adapted_cross:.2} dB) \
         than to its own base ({w4_effect:.2} dB)"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED w4a8_krea_dit_gpu::w4a8_krea_dit_gpu_additive_lora_matches_q8_class");
}
