//! Real-weight smoke for the Krea 2 dual-conditioned image-edit path through the PRODUCTION generator
//! seam (epic 10871 P2.3 + P3.1, sc-10881/sc-10882).
//!
//! Loads the cached Krea 2 Raw diffusers snapshot + the community `krea2-identity-edit` LoRA via the
//! `krea_2_edit` generator (`model::load_edit` → `Box<dyn Generator>`), then runs a dual-conditioned edit
//! the way the SceneWorks worker does: a single `Conditioning::Reference` (the source) passed through
//! `Generator::generate`, which the `generate_impl` edit branch routes to `generate_edit_with_progress`
//! (in-context VAE tokens + Qwen3-VL grounding). This is a MANUAL on-Metal validation (a 12.9B model),
//! NOT a CI test. Paths default to the local HF cache; override via env (`KREA_SNAPSHOT`,
//! `KREA_EDIT_LORA`, `KREA_EDIT_SOURCE`, `KREA_EDIT_INSTRUCTION`, `KREA_EDIT_OUT`, `KREA_EDIT_STEPS`,
//! `KREA_EDIT_GUIDANCE`). Two-reference: set `KREA_EDIT_SOURCE_B` (scene = SOURCE, person = SOURCE_B) →
//! a `Conditioning::MultiReference`. R5 ablation: `KREA_EDIT_LORA=none` loads WITHOUT the identity LoRA
//! (dual conditioning present but untrained/inert) — used for the epic-10871 P4.2 dual-vs-inert delta.
//!
//! Run: `cargo run --release --example krea_edit_smoke -p mlx-gen-krea`

use std::path::PathBuf;

use mlx_gen::gen_core::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen::media::Image;
use mlx_gen::{AdapterKind, AdapterSpec};
use mlx_gen_krea::model::load_edit;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn load_rgb(path: &str) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open source image {path}: {e}"))
        .to_rgb8();
    let (width, height) = img.dimensions();
    Image {
        width,
        height,
        pixels: img.into_raw(),
    }
}

fn save_png(img: &Image, path: &str) {
    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(img.width, img.height, img.pixels.clone())
            .expect("output image buffer");
    buf.save(path)
        .unwrap_or_else(|e| panic!("save {path}: {e}"));
}

fn main() {
    let snapshot = env_or(
        "KREA_SNAPSHOT",
        "/Users/michael/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494",
    );
    let lora = env_or(
        "KREA_EDIT_LORA",
        "/Users/michael/.cache/huggingface/hub/models--conradlocke--krea2-identity-edit/snapshots/8f3856364fcee7db52116f72558fce0c233eaac4/krea2_identity_edit_v1_1_r128.safetensors",
    );
    let source = env_or(
        "KREA_EDIT_SOURCE",
        "/Users/michael/.cache/huggingface/hub/models--conradlocke--krea2-identity-edit/snapshots/8f3856364fcee7db52116f72558fce0c233eaac4/showcase/release_1.png",
    );
    let instruction = env_or(
        "KREA_EDIT_INSTRUCTION",
        "change the background to a snowy mountain landscape",
    );
    let out_path = env_or("KREA_EDIT_OUT", "/tmp/krea_edit_out.png");
    let steps: u32 = env_or("KREA_EDIT_STEPS", "16").parse().expect("steps");
    let guidance: f32 = env_or("KREA_EDIT_GUIDANCE", "3.0")
        .parse()
        .expect("guidance");

    // Build the LoadSpec the way the worker does: snapshot dir + the edit LoRA as an adapter, then load
    // the `krea_2_edit` generator (the production Generator seam, not the direct pipeline method).
    // `KREA_EDIT_LORA=none` (or empty) → load WITHOUT the identity LoRA: the R5 ablation. The engine
    // still runs the full dual conditioning (VAE in-context tokens + Qwen3-VL grounding), but with the
    // conditioning inert the base is off-distribution — what the worker R5 gate blocks (epic 10871 P4.2).
    let base = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let no_lora = lora.trim().is_empty() || lora.trim().eq_ignore_ascii_case("none");
    let spec = if no_lora {
        eprintln!(
            "[smoke] NO edit LoRA (R5 ablation — dual conditioning present but untrained/inert)"
        );
        base
    } else {
        eprintln!("[smoke] edit LoRA {lora}");
        base.with_adapters(vec![AdapterSpec::new(
            PathBuf::from(&lora),
            1.0,
            AdapterKind::Lora,
        )])
    };
    eprintln!("[smoke] loading krea_2_edit generator from {snapshot}");
    let generator = load_edit(&spec).expect("load krea_2_edit generator");

    let src = load_rgb(&source);
    let (sw, sh) = (src.width, src.height);

    // Two-reference edit (epic 10871 P1.3): with a second source (`KREA_EDIT_SOURCE_B`) the worker sends
    // a `Conditioning::MultiReference` in FIXED order — scene = image 1, person = image 2. Without it,
    // the single-source path (one `Conditioning::Reference`) — both route through the same `krea_2_edit`
    // Generator seam and on to `generate_edit_with_progress`.
    let source_b = std::env::var("KREA_EDIT_SOURCE_B")
        .ok()
        .filter(|b| !b.trim().is_empty());
    let conditioning = match &source_b {
        Some(b) => {
            let person = load_rgb(b);
            eprintln!(
                "[smoke] scene {sw}x{sh} + person {}x{} → MultiReference (scene, person) → edit '{instruction}' ({steps} steps, g={guidance})",
                person.width, person.height
            );
            vec![Conditioning::MultiReference {
                images: vec![src, person],
            }]
        }
        None => {
            eprintln!(
                "[smoke] source {sw}x{sh} → edit '{instruction}' ({steps} steps, g={guidance}) via Generator::generate"
            );
            vec![Conditioning::Reference {
                image: src,
                strength: None,
            }]
        }
    };

    // The worker's exact request shape; `generate_impl` sees the `krea_2_edit` descriptor and routes the
    // Reference / MultiReference source(s) to the edit entrypoint.
    let request = GenerationRequest {
        prompt: instruction.clone(),
        negative_prompt: Some(String::new()),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(guidance),
        conditioning,
        cancel: CancelFlag::new(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, &mut |_| {})
        .expect("generate edit");
    let out = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("edit produced one image"),
        _ => panic!("edit generator returned non-image output"),
    };

    // Basic sanity: a non-degenerate (non-constant) image.
    let mn = *out.pixels.iter().min().unwrap();
    let mx = *out.pixels.iter().max().unwrap();
    let mean: f64 = out.pixels.iter().map(|&p| p as f64).sum::<f64>() / out.pixels.len() as f64;
    eprintln!(
        "[smoke] output {}x{} px range [{mn},{mx}] mean {mean:.1}",
        out.width, out.height
    );
    assert!(mx > mn, "degenerate (constant) output image");
    save_png(&out, &out_path);
    eprintln!("[smoke] wrote {out_path}");
}
