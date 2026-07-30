//! The **canonical** SANA render: the shipped macOS bundle, default settings, nothing overridden.
//!
//! ```text
//! cargo run --release -p runtime-macos --features media --example sana_canonical -- <snapshot_dir>
//!     [--out FILE] [--size N] [--steps N] [--seed N] [--prompt TEXT]
//! ```
//!
//! # Why this is the reference and the iOS examples are not
//!
//! Every SANA render measured during the iOS work went through `mlx-gen-ios-catalog` — a narrow
//! composition root built for a memory-capped device, driven under `OffloadPolicy::Sequential` with
//! a tiled DC-AE decode. That is the right configuration *for a phone* and the wrong one to judge
//! quality against, because both of those choices change the output:
//!
//! * `Sequential` is transparent (proved byte-identical in `sequential_residency_real_weights`),
//! * but the **tiled decode is not** — DC-AE's `SanaMultiscaleLinearAttention` normalizes by
//!   `1/(Σ+eps)` over whatever spatial extent it is given, so a tile carries a different denominator
//!   than the whole image and no overlap repairs it.
//!
//! This example takes the path a SceneWorks consumer actually takes on a Mac: `runtime-macos`'s
//! `provider_registry()`, `OffloadPolicy::Resident`, whole-image decode. That is the quality gate —
//! the image the model is capable of, with nothing traded away for a memory bound.
//!
//! Compare anything against it with `mlx-gen-ios-catalog`'s `compare_renders`.
//!
//! # It will not be byte-identical to a device render, and that is expected twice over
//!
//! The tiled decode differs by construction (above), and separately **generation is not
//! bit-deterministic across platforms**: the same prompt, seed, steps and tile size on macOS and
//! iOS differ by mean |Δ| 0.90/255 (measured), because two different Metal GPUs select different
//! kernels. Perceptually identical, numerically not. Neither is a defect; both are worth knowing
//! before treating any cross-platform comparison as a pass/fail.

use std::path::PathBuf;
use std::time::Instant;

use runtime_macos::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Progress, WeightsSource,
};

/// The base (non-Sprint) SANA id, as registered by `mlx-gen-catalog`.
const MODEL_ID: &str = "sana_1600m";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().ok_or(
        "usage: sana_canonical <snapshot_dir> [--out FILE] [--size N] [--steps N] [--seed N] \
         [--prompt TEXT]",
    )?;

    let mut out = "sana-canonical.png".to_string();
    let mut size = 1024u32;
    let mut steps = 4u32;
    let mut seed = 0u64;
    let mut prompt = "a lighthouse on a rocky coast at dawn".to_string();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--out" => out = args.next().ok_or("--out needs a value")?,
            "--size" => size = args.next().ok_or("--size needs a value")?.parse()?,
            "--steps" => steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--seed" => seed = args.next().ok_or("--seed needs a value")?.parse()?,
            "--prompt" => prompt = args.next().ok_or("--prompt needs a value")?,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    // Refuse to run with the tiling override set. This example's entire purpose is to be the
    // untiled reference, and inheriting a stray env var from a measurement shell would silently
    // produce a tiled "canonical" image — the exact confusion it exists to prevent.
    if let Ok(v) = std::env::var("MLX_GEN_SANA_DECODE_TILE") {
        return Err(format!(
            "MLX_GEN_SANA_DECODE_TILE={v} is set. The canonical render must be whole-image; \
             unset it (`env -u MLX_GEN_SANA_DECODE_TILE ...`) and re-run."
        )
        .into());
    }

    println!(
        "canonical SANA render (runtime-macos, Resident, whole-image decode)\n  \
         {size}px, {steps} steps, seed {seed}\n  \"{prompt}\""
    );

    // Through the SHIPPED bundle's validated catalog, not a direct loader: this is the composition
    // a SceneWorks consumer resolves the provider through, so catalog wiring is part of the gate.
    let catalog = runtime_macos::catalog()?;
    let registry = catalog.media();
    let spec = LoadSpec {
        offload_policy: OffloadPolicy::Resident,
        ..LoadSpec::new(WeightsSource::Dir(PathBuf::from(&dir)))
    };
    let started = Instant::now();
    let generator = registry.load(MODEL_ID, &spec)?;

    let request = GenerationRequest {
        prompt,
        width: size,
        height: size,
        count: 1,
        steps: Some(steps),
        seed: Some(seed),
        ..Default::default()
    };
    let mut noop = |_: Progress| {};
    let image = match generator.generate(&request, &mut noop)? {
        GenerationOutput::Images(mut v) if !v.is_empty() => v.remove(0),
        _ => return Err("generator returned no image".into()),
    };

    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(image.width, image.height, image.pixels.clone())
            .ok_or("pixel buffer does not match dimensions")?;
    buf.save(&out)?;

    println!(
        "  wrote {out} ({}x{}) in {:.1}s",
        image.width,
        image.height,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
