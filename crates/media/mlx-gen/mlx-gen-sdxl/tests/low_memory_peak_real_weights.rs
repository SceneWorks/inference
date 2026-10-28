//! sc-26268 bullet 1, real weights (Apple/Metal, opt-in): the **staged q4 512² request peak** for
//! `sdxl` and `illustrious_xl_v2`, and the memory probe that **the U-Net is not resident during the
//! decode** of a staged request.
//!
//! This file deliberately uses only API that also exists on the base commit
//! (`provider_registry`, `GenerationMemory::stage_residency`, `Progress`, MLX's own memory
//! counters), so the identical harness measures the *before* row on the base and the *after* row
//! here. On the base, `staged_decode_does_not_hold_the_unet` is expected to FAIL — that failure is
//! the before evidence for the defect this story fixes.
//!
//! Measurement discipline (as `memory_ladder_real_weights.rs`): MLX's ACTIVE-byte accounting, a
//! fresh generator per row, `reset_peak_memory` after the load so the row is the request.
//!
//! | env var | meaning |
//! |---|---|
//! | `SDXL_LADDER_SDXL` | `SceneWorks/sdxl-base-mlx` snapshot root (holds `q4/`) |
//! | `SDXL_LADDER_ILLUSTRIOUS_XL_V2` | `SceneWorks/illustrious-xl-v2-mlx` snapshot root (holds `q4/`) |
//! | `SDXL_LOWMEM_SIZE` | output edge, default 512 |
//! | `SDXL_LOWMEM_STEPS` | denoise steps, default 8 |
//!
//! Running a test in this file without its entry's env var is a **failure that names the var**,
//! never a silent pass: the tests are `#[ignore]`d precisely so that only a deliberate run reaches
//! them.

use std::path::PathBuf;

use mlx_gen::gen_core::{GenerationMemory, GenerationOutput, GenerationRequest, Progress};
use mlx_gen::{LoadSpec, OffloadPolicy, Quant, WeightsSource};
use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The two entries this story's AC names.
const ENTRIES: &[(&str, &str)] = &[
    ("sdxl", "SDXL_LADDER_SDXL"),
    ("illustrious_xl_v2", "SDXL_LADDER_ILLUSTRIOUS_XL_V2"),
];

fn q4_dir(var: &str) -> PathBuf {
    let root = std::env::var(var).unwrap_or_else(|_| {
        panic!(
            "set {var} to the entry's snapshot root (the directory holding `q4/`); inference never \
             self-fetches or derives a cache location (epic 13657)"
        )
    });
    let dir = PathBuf::from(root).join("q4");
    assert!(dir.is_dir(), "{var}: no q4/ tier under {}", dir.display());
    dir
}

fn env_u32(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{var} must be an integer"))
        })
        .unwrap_or(default)
}

pub(crate) fn q4_spec(dir: &std::path::Path) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
        .with_offload_policy(OffloadPolicy::Sequential);
    spec.quantize = Some(Quant::Q4);
    spec
}

pub(crate) fn staged_request(memory: GenerationMemory) -> GenerationRequest {
    let edge = env_u32("SDXL_LOWMEM_SIZE", 512);
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        negative_prompt: Some("blurry, lowres".into()),
        width: edge,
        height: edge,
        count: 1,
        steps: Some(env_u32("SDXL_LOWMEM_STEPS", 8)),
        guidance: Some(7.0),
        seed: Some(1234),
        memory: Some(GenerationMemory {
            stage_residency: true,
            ..memory
        }),
        ..Default::default()
    }
}

/// One measured staged request on a fresh generator.
pub(crate) struct Row {
    pub(crate) peak_gib: f64,
    /// ACTIVE bytes when the last denoise step reports (the U-Net is still alive).
    pub(crate) active_at_last_step: usize,
    /// ACTIVE bytes when the decode begins.
    pub(crate) active_at_decode: usize,
    /// The U-Net's resident bytes as the loaded contract prices them.
    pub(crate) unet_bytes: u64,
    pub(crate) pixels: Vec<u8>,
}

#[track_caller]
pub(crate) fn measure(spec: &LoadSpec, req: &GenerationRequest) -> Row {
    let registry = mlx_gen_sdxl::provider_registry().expect("provider registry");
    let model = registry.load("sdxl", spec).expect("load sdxl");
    let unet_bytes = model
        .memory_strategy_contract()
        .expect("sdxl publishes a memory contract")
        .asset_facts
        .transformer_bytes;
    clear_cache();
    reset_peak_memory();
    let mut active_at_last_step = 0;
    let mut active_at_decode = 0;
    let out = (
        crate::common::record_model_inference_device(),
        model.generate(req, &mut |progress: Progress| match progress {
            Progress::Step { current, total } if current == total => {
                active_at_last_step = get_active_memory();
            }
            Progress::Decoding if active_at_decode == 0 => {
                active_at_decode = get_active_memory();
            }
            _ => {}
        }),
    )
        .1
        .expect("generate must succeed");
    let peak = get_peak_memory();
    let pixels = match out {
        GenerationOutput::Images(images) => images.first().expect("one image").pixels.clone(),
        other => panic!("expected images, got {other:?}"),
    };
    drop(model);
    clear_cache();
    Row {
        peak_gib: peak as f64 / GIB,
        active_at_last_step,
        active_at_decode,
        unet_bytes,
        pixels,
    }
}

/// The staged q4 request peak per entry — the before/after number the AC asks the PR to report.
/// Report-only: the comparison is across commits, so there is nothing to assert in one run beyond
/// valid accounting.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "real weights + Apple GPU: set SDXL_LADDER_SDXL and SDXL_LADDER_ILLUSTRIOUS_XL_V2"]
fn q4_staged_request_peak() {
    for (entry, var) in ENTRIES {
        let spec = q4_spec(&q4_dir(var));
        let row = measure(&spec, &staged_request(GenerationMemory::default()));
        assert!(row.peak_gib.is_finite() && row.peak_gib > 0.0);
        eprintln!(
            "[sc-26268 peak] {entry} q4 {}² staged: request peak {:.4} GiB | active at last step \
             {:.4} GiB, at decode {:.4} GiB | contract U-Net {:.4} GiB",
            env_u32("SDXL_LOWMEM_SIZE", 512),
            row.peak_gib,
            row.active_at_last_step as f64 / GIB,
            row.active_at_decode as f64 / GIB,
            row.unet_bytes as f64 / GIB,
        );
    }

    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED low_memory_peak_real_weights::q4_staged_request_peak"
    );
}

/// The AC's memory probe on real weights: by the time a staged request's decode starts, at least
/// 90% of the U-Net's resident bytes have been released.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "real weights + Apple GPU: set SDXL_LADDER_SDXL and SDXL_LADDER_ILLUSTRIOUS_XL_V2"]
fn staged_decode_does_not_hold_the_unet() {
    for (entry, var) in ENTRIES {
        let spec = q4_spec(&q4_dir(var));
        let row = measure(&spec, &staged_request(GenerationMemory::default()));
        assert!(
            row.active_at_last_step > 0 && row.active_at_decode > 0,
            "{entry}: no probe read"
        );
        assert!(row.unet_bytes > 0, "{entry}: the contract priced no U-Net");
        let released = row.active_at_last_step.saturating_sub(row.active_at_decode) as u64;
        assert!(
            released * 10 >= row.unet_bytes * 9,
            "{entry}: the staged decode still holds the U-Net — released {:.4} GiB of a {:.4} GiB \
             U-Net between the last denoise step ({:.4} GiB active) and the decode ({:.4} GiB)",
            released as f64 / GIB,
            row.unet_bytes as f64 / GIB,
            row.active_at_last_step as f64 / GIB,
            row.active_at_decode as f64 / GIB,
        );
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED low_memory_peak_real_weights::staged_decode_does_not_hold_the_unet");
}
