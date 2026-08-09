//! SC-15998 — real-weight proof that phase-level staged residency and transformer-block deferred
//! materialization are independent axes.
//!
//! The probe runs the same request twice on one loaded generator for every composition, recording:
//! per-phase/request active-memory peaks, first/warm latency, retained active bytes between requests,
//! and per-request component-load events. `Resident + DeferredMaterialization` is the decisive arm:
//! it must execute rung 4 without any phase-level reload while retaining a warm generator.
//!
//! ```text
//! MLX_GEN_ZIMAGE_SNAPSHOT=<q4/q8/bf16 tier dir> ZIMAGE_SIZE=512 ZIMAGE_STEPS=1 \
//!   cargo test -p mlx-gen-z-image --release --test deferred_materialization_real_weights \
//!   -- --ignored --nocapture --test-threads=1
//! ```

mod common;

use common::tier_snapshot as snapshot;
use mlx_gen::gen_core::{GenerationMemory, TransformerComponent};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadPhase, LoadShape, LoadSpec, OffloadPolicy,
    Progress, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
use std::time::Instant;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Clone, Copy)]
struct Arm {
    label: &'static str,
    offload: OffloadPolicy,
    load_shape: LoadShape,
    component: Option<TransformerComponent>,
}

const ARMS: &[Arm] = &[
    Arm {
        label: "resident eager",
        offload: OffloadPolicy::Resident,
        load_shape: LoadShape::EagerMaterialization,
        component: None,
    },
    Arm {
        label: "rung 1 only",
        offload: OffloadPolicy::Sequential,
        load_shape: LoadShape::EagerMaterialization,
        component: None,
    },
    Arm {
        label: "deferred load only",
        offload: OffloadPolicy::Resident,
        load_shape: LoadShape::DeferredMaterialization,
        component: None,
    },
    Arm {
        label: "rung 4 Dit",
        offload: OffloadPolicy::Resident,
        load_shape: LoadShape::DeferredMaterialization,
        component: Some(TransformerComponent::Dit),
    },
    Arm {
        label: "rung 4 TextEncoder",
        offload: OffloadPolicy::Resident,
        load_shape: LoadShape::DeferredMaterialization,
        component: Some(TransformerComponent::TextEncoder),
    },
    Arm {
        label: "rung 4 Both",
        offload: OffloadPolicy::Resident,
        load_shape: LoadShape::DeferredMaterialization,
        component: Some(TransformerComponent::Both),
    },
    Arm {
        label: "rungs 1+4 Both",
        offload: OffloadPolicy::Sequential,
        load_shape: LoadShape::DeferredMaterialization,
        component: Some(TransformerComponent::Both),
    },
];

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn tier_from_path() -> Option<Quant> {
    match snapshot()
        .file_name()
        .map(|name| name.to_string_lossy())
        .as_deref()
    {
        Some("q4") => Some(Quant::Q4),
        Some("q8") => Some(Quant::Q8),
        _ => None,
    }
}

fn spec(arm: Arm) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()))
        .with_offload_policy(arm.offload)
        .with_load_shape(arm.load_shape);
    if let Some(quant) = tier_from_path() {
        spec = spec.with_quant(quant);
    }
    spec
}

fn request(component: Option<TransformerComponent>, stage_residency: bool) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: env_u32("ZIMAGE_SIZE", 512),
        height: env_u32("ZIMAGE_SIZE", 512),
        count: 1,
        seed: Some(1234),
        steps: Some(env_u32("ZIMAGE_STEPS", 1)),
        memory: (component.is_some() || stage_residency).then(|| GenerationMemory {
            stage_residency,
            stream_transformer_blocks: component.is_some(),
            transformer_window_size: component.map(|_| 1),
            transformer_window_component: component,
            ..Default::default()
        }),
        ..Default::default()
    }
}

struct Run {
    conditioning: usize,
    denoise: usize,
    decode: usize,
    wall_secs: f64,
    retained_after: usize,
    text_loads: usize,
    heavy_loads: usize,
    image: Image,
}

impl Run {
    fn request_peak(&self) -> usize {
        self.conditioning.max(self.denoise).max(self.decode)
    }
}

fn run_once(generator: &dyn mlx_gen::Generator, request: &GenerationRequest) -> Run {
    clear_cache();
    reset_peak_memory();
    let start = Instant::now();
    let mut conditioning = 0usize;
    let mut denoise = 0usize;
    let mut text_loads = 0usize;
    let mut heavy_loads = 0usize;
    let mut on_progress = |progress: Progress| match progress {
        Progress::Loading(LoadPhase::TextEncoder) => text_loads += 1,
        Progress::Loading(LoadPhase::Renderer) => heavy_loads += 1,
        Progress::Step { current: 1, .. } => {
            conditioning = get_peak_memory();
            reset_peak_memory();
        }
        Progress::Decoding if denoise == 0 => {
            denoise = get_peak_memory();
            reset_peak_memory();
        }
        _ => {}
    };
    let output = generator
        .generate(request, &mut on_progress)
        .expect("generation");
    let wall_secs = start.elapsed().as_secs_f64();
    let decode = get_peak_memory();
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected images, got {other:?}"),
    };
    clear_cache();
    let retained_after = get_active_memory();
    assert!(
        conditioning > 0 && denoise > 0 && decode > 0,
        "a phase boundary was not sampled"
    );
    Run {
        conditioning,
        denoise,
        decode,
        wall_secs,
        retained_after,
        text_loads,
        heavy_loads,
        image,
    }
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / GIB
}

fn image_delta(left: &Image, right: &Image) -> u8 {
    assert_eq!(
        (left.width, left.height, left.pixels.len()),
        (right.width, right.height, right.pixels.len())
    );
    left.pixels
        .iter()
        .zip(&right.pixels)
        .fold(0u8, |max, (a, b)| max.max(a.abs_diff(*b)))
}

#[test]
#[ignore = "needs a real Z-Image snapshot + Apple/Metal GPU"]
fn deferred_materialization_is_independent_from_staged_residency() {
    let req = request(None, false);
    println!(
        "\nSC-15998 decoupled residency/materialization A/B — {}x{} @ {} step(s)",
        req.width,
        req.height,
        req.steps.unwrap_or(1)
    );
    println!(
        "  {:<22} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9} {:>9} {:>7}",
        "composition",
        "cond GiB",
        "dit GiB",
        "vae GiB",
        "req GiB",
        "cold sec",
        "warm sec",
        "retained",
        "loads",
    );

    let mut baseline_image = None;
    let mut baseline_peak = None;
    let mut deferred_baseline_peak = None;
    let mut rung4_dit_peak = None;
    let mut rung4_te_peak = None;
    let mut rung4_both_peak = None;
    let mut rung4_both_retained = None;

    for arm in ARMS {
        let generator = mlx_gen_z_image::load(&spec(*arm)).expect("load z_image_turbo");
        let req = request(
            arm.component,
            matches!(arm.offload, OffloadPolicy::Sequential),
        );
        let cold = run_once(generator.as_ref(), &req);
        let warm = run_once(generator.as_ref(), &req);
        println!(
            "  {:<22} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>9.3} {:>9.3} {:>8.3}G {:>3}+{:>3}",
            arm.label,
            gib(warm.conditioning),
            gib(warm.denoise),
            gib(warm.decode),
            gib(warm.request_peak()),
            cold.wall_secs,
            warm.wall_secs,
            gib(warm.retained_after),
            warm.text_loads,
            warm.heavy_loads,
        );

        if matches!(arm.offload, OffloadPolicy::Resident) {
            assert_eq!(
                (warm.text_loads, warm.heavy_loads),
                (0, 0),
                "{} reloaded a phase component instead of preserving the warm generator",
                arm.label
            );
        } else {
            assert_eq!(
                (warm.text_loads, warm.heavy_loads),
                (1, 1),
                "{} did not exercise staged per-request component loads",
                arm.label
            );
        }

        match baseline_image.as_ref() {
            None => {
                baseline_peak = Some(warm.request_peak());
                baseline_image = Some(warm.image.clone());
            }
            Some(image) => assert_eq!(
                image_delta(image, &warm.image),
                0,
                "{} changed the rendered image",
                arm.label
            ),
        }
        match arm.label {
            "deferred load only" => deferred_baseline_peak = Some(warm.request_peak()),
            "rung 4 Dit" => rung4_dit_peak = Some(warm.request_peak()),
            "rung 4 TextEncoder" => rung4_te_peak = Some(warm.request_peak()),
            "rung 4 Both" => {
                rung4_both_peak = Some(warm.request_peak());
                rung4_both_retained = Some(warm.retained_after);
            }
            _ => {}
        }
        drop(generator);
        clear_cache();
    }

    let baseline_peak = baseline_peak.expect("resident baseline");
    let print_cut = |label: &str, comparison: usize, peak: usize| {
        println!(
            "  => {label} request peak: {:.3} -> {:.3} GiB ({:.1}% cut)",
            gib(comparison),
            gib(peak),
            100.0 * (1.0 - peak as f64 / comparison as f64)
        );
    };
    let deferred_baseline_peak = deferred_baseline_peak.expect("deferred load baseline");
    print_cut("deferred load shape", baseline_peak, deferred_baseline_peak);
    print_cut(
        "rung 4 Dit vs deferred load",
        deferred_baseline_peak,
        rung4_dit_peak.expect("Dit arm"),
    );
    print_cut(
        "rung 4 TextEncoder vs deferred load",
        deferred_baseline_peak,
        rung4_te_peak.expect("TextEncoder arm"),
    );
    print_cut(
        "rung 4 Both vs deferred load",
        deferred_baseline_peak,
        rung4_both_peak.expect("Both arm"),
    );

    assert!(
        rung4_dit_peak.expect("Dit arm") <= deferred_baseline_peak,
        "bounded DiT window made the deferred-load request peak worse"
    );

    // The fresh-generator arms above are not enough: a TextEncoder-scoped request used to execute
    // the excluded DiT through its lazy resident stack, retaining ~3 GiB. A later DiT/Both request
    // then streamed another copy while the unchanged global load-shape flag still passed the guard.
    // Exercise that exact warm-cache transition on one generator.
    let mixed_arm = Arm {
        label: "mixed TE -> Both",
        offload: OffloadPolicy::Resident,
        load_shape: LoadShape::DeferredMaterialization,
        component: None,
    };
    let generator = mlx_gen_z_image::load(&spec(mixed_arm)).expect("load mixed-scope generator");
    let text_encoder_only = run_once(
        generator.as_ref(),
        &request(Some(TransformerComponent::TextEncoder), false),
    );
    let both_after_text_encoder = run_once(
        generator.as_ref(),
        &request(Some(TransformerComponent::Both), false),
    );
    let clean_both_peak = rung4_both_peak.expect("Both arm");
    let clean_both_retained = rung4_both_retained.expect("Both retained");
    let tolerance = 256 * 1024 * 1024;
    println!(
        "  {:<22} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>9} {:>9.3} {:>8.3}G {:>3}+{:>3}",
        mixed_arm.label,
        gib(both_after_text_encoder.conditioning),
        gib(both_after_text_encoder.denoise),
        gib(both_after_text_encoder.decode),
        gib(both_after_text_encoder.request_peak()),
        "-",
        both_after_text_encoder.wall_secs,
        gib(both_after_text_encoder.retained_after),
        both_after_text_encoder.text_loads,
        both_after_text_encoder.heavy_loads,
    );
    assert_eq!(
        (
            text_encoder_only.text_loads,
            text_encoder_only.heavy_loads,
            both_after_text_encoder.text_loads,
            both_after_text_encoder.heavy_loads,
        ),
        (0, 0, 0, 0),
        "mixed scopes reloaded a phase component"
    );
    assert!(
        text_encoder_only.retained_after <= clean_both_retained + tolerance,
        "TextEncoder-only request bulk-materialized and retained the excluded DiT: {:.3} GiB vs \
         {:.3} GiB clean Both",
        gib(text_encoder_only.retained_after),
        gib(clean_both_retained)
    );
    assert!(
        both_after_text_encoder.request_peak() <= clean_both_peak + tolerance,
        "Both after TextEncoder streamed over retained weights: {:.3} GiB vs {:.3} GiB clean Both",
        gib(both_after_text_encoder.request_peak()),
        gib(clean_both_peak)
    );
    assert_eq!(
        image_delta(
            baseline_image.as_ref().expect("resident baseline image"),
            &both_after_text_encoder.image,
        ),
        0,
        "mixed-scope transition changed the rendered image"
    );
}
