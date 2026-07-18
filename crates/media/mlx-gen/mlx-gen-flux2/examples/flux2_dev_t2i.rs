//! Manual FLUX.2-**dev** txt2img smoke — prove the 32B dev model generates on a 64 GB Apple-Silicon
//! box via the Q8 + `OffloadPolicy::Sequential` fit path (dev is ~112 GB bf16; Q8 halves it and
//! Sequential drops the Mistral-3 text encoder before the transformer denoise, so peak ≈ the
//! transformer working set, not the sum). NOT a CI test (a 32B model on Metal + licensed weights).
//!
//! Env overrides: FLUX2_DEV_SNAPSHOT (weights dir), FLUX2_DEV_PROMPT, FLUX2_DEV_OUT,
//! FLUX2_DEV_STEPS, FLUX2_DEV_GUIDANCE, FLUX2_DEV_SEED, FLUX2_DEV_W, FLUX2_DEV_H, FLUX2_DEV_QUANT
//! (`q8`|`q4`|`bf16`), FLUX2_DEV_RESIDENT (set to keep everything co-resident instead of Sequential).
//!
//! Run (reuses the warm workspace target):
//!   FLUX2_DEV_SNAPSHOT=~/Models/aether/flux2-dev \
//!     cargo run --release --example flux2_dev_t2i -p mlx-gen-flux2 -- 2>&1
use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::gen_core::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_gen::media::Image;
use mlx_gen_flux2::load_dev;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn save_png(img: &Image, path: &str) {
    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(img.width, img.height, img.pixels.clone())
            .expect("output image buffer");
    buf.save(path)
        .unwrap_or_else(|e| panic!("save {path}: {e}"));
}

fn main() {
    let snapshot = env_or("FLUX2_DEV_SNAPSHOT", "");
    assert!(
        !snapshot.is_empty(),
        "set FLUX2_DEV_SNAPSHOT to the flux2-dev weights directory"
    );
    let prompt = env_or(
        "FLUX2_DEV_PROMPT",
        "A photorealistic close-up portrait of a weathered lighthouse keeper in his sixties, \
         salt-and-pepper beard, deep wrinkles, wearing a wet yellow raincoat, storm light, \
         shot on 85mm, shallow depth of field, natural skin texture",
    );
    let out_path = env_or("FLUX2_DEV_OUT", "/tmp/flux2_dev_proof.png");
    let steps: u32 = env_or("FLUX2_DEV_STEPS", "20").parse().expect("steps");
    let seed: u64 = env_or("FLUX2_DEV_SEED", "0").parse().expect("seed");
    let width: u32 = env_or("FLUX2_DEV_W", "1024").parse().expect("width");
    let height: u32 = env_or("FLUX2_DEV_H", "1024").parse().expect("height");
    let guidance: Option<f32> = std::env::var("FLUX2_DEV_GUIDANCE")
        .ok()
        .map(|g| g.parse().expect("guidance"));

    // The fit path for a 64 GB box: Q8 (unless overridden) + Sequential residency.
    let quant = match env_or("FLUX2_DEV_QUANT", "q8").to_lowercase().as_str() {
        "q8" => Some(Quant::Q8),
        "q4" => Some(Quant::Q4),
        "bf16" | "none" | "" => None,
        other => panic!("FLUX2_DEV_QUANT must be q8|q4|bf16 (got {other})"),
    };
    let residency = if std::env::var("FLUX2_DEV_RESIDENT").is_ok() {
        OffloadPolicy::Resident
    } else {
        OffloadPolicy::Sequential
    };

    let mut spec =
        LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot))).with_offload_policy(residency);
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }

    eprintln!(
        "[flux2_dev] loading from {snapshot}\n            quant={quant:?} residency={residency:?}"
    );
    let t_load = Instant::now();
    let generator = load_dev(&spec).expect("load flux2_dev generator");
    eprintln!(
        "[flux2_dev] loaded generator in {:.1}s",
        t_load.elapsed().as_secs_f32()
    );

    let request = GenerationRequest {
        prompt: prompt.clone(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        guidance,
        cancel: CancelFlag::new(),
        ..Default::default()
    };
    generator.validate(&request).expect("validate request");

    eprintln!(
        "[flux2_dev] '{prompt}'\n            ({width}x{height}, {steps} steps, guidance={guidance:?}, seed={seed})"
    );
    let t_gen = Instant::now();
    let mut ticks = 0u32;
    let output = generator
        .generate(&request, &mut |_p| {
            ticks += 1;
            eprint!(".");
        })
        .expect("generate");
    let dt = t_gen.elapsed().as_secs_f32();
    eprintln!();

    let out = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        _ => panic!("flux2_dev returned non-image output"),
    };
    let mn = *out.pixels.iter().min().unwrap();
    let mx = *out.pixels.iter().max().unwrap();
    assert!(
        mx > mn,
        "degenerate (constant) output — pipeline produced flat pixels"
    );
    save_png(&out, &out_path);
    eprintln!(
        "[flux2_dev] wrote {out_path} ({}x{}) in {dt:.1}s generate ({} progress ticks, {:.1}s/step avg)",
        out.width,
        out.height,
        ticks,
        dt / steps.max(1) as f32
    );
}
