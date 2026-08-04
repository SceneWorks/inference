//! sc-16462: side-by-side evidence for packing Chroma's T5-XXL + FLUX.1 VAE **at the selected tier**.
//!
//! The shipped q4/q8 tiers pack only the DiT; T5 and the VAE ride along at bf16. On a q4 tier that is
//! above-tier residency — the exact thing `config/tier-integrity.jsonc` exists to eliminate, and the
//! reason `chroma1_{base,hd,flash}` carry six exception rows while `flux1_*` (same T5-XXL, packed by
//! `mlx_gen_flux::convert::prequantize_turnkey`) carry none.
//!
//! This harness renders the SAME prompts and seeds through the shipped tier and through candidates
//! whose ONLY difference is the auxiliary width, and writes both the PNGs and the numbers. The
//! transformer bytes are held fixed by [`mlx_gen_chroma::convert::repack_auxiliaries`] and asserted
//! byte-identical here, so any pixel difference is attributable to the auxiliaries alone.
//!
//! It deliberately does NOT assert a pixel-identity threshold against the bf16-auxiliary render.
//! Packing a text encoder to the selected tier is *supposed* to move the conditioning — a q4 render
//! is a q4 render end to end, which is exactly what the user asked for when they chose the tier. A
//! pixel-identity gate against a bf16-conditioned reference would only ever be satisfiable by NOT
//! doing the thing the story asks for; that is the trap sc-16462 spent two days in. What IS asserted
//! is that the render stays coherent (the packed-surface completeness check) and that the
//! auxiliaries actually shrink.
//!
//! The candidate width is not selectable here either — `repack_auxiliaries` derives it from the
//! baseline's own transformer — so pointing this at the q4 tier compares bf16 vs Q4 auxiliaries, and
//! pointing it at the q8 tier compares bf16 vs Q8.
//!
//! Run (≈15 GB baseline tier, Apple Silicon + Metal):
//!   SC16462_BASELINE=<shipped q4 tier dir> SC16462_OUT=<scratch dir> \
//!     cargo test -p mlx-gen-chroma --release --test auxiliary_tier_comparison \
//!       -- --ignored --nocapture compare_auxiliary_widths

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use std::path::{Path, PathBuf};

/// The two prompts carried over from the sc-16462 hosted calibration, so numbers here line up with
/// the numbers already recorded on the story.
const PROMPTS: &[(&str, u64)] = &[
    ("a photograph of an astronaut riding a horse", 42),
    (
        "a highly detailed portrait of an elderly woman, soft window light",
        7,
    ),
];

fn model_id() -> String {
    std::env::var("SC16462_MODEL").unwrap_or_else(|_| "chroma1_base".into())
}

fn size() -> u32 {
    std::env::var("SC16462_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512)
}

fn steps() -> u32 {
    std::env::var("SC16462_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20)
}

fn save_png(img: &Image, path: &Path) {
    image::save_buffer(
        path,
        &img.pixels,
        img.width,
        img.height,
        image::ColorType::Rgb8,
    )
    .expect("write png");
}

/// Sum of every tensor byte under a component dir — the isolated on-disk residency the ledger's
/// `costBytesByTier` records.
fn component_bytes(dir: &Path) -> u64 {
    let mut total = 0;
    for entry in std::fs::read_dir(dir).expect("read component dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            total += std::fs::metadata(&path).expect("stat shard").len();
        }
    }
    total
}

fn render_all(root: &Path, id: &str) -> Vec<Image> {
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
    let generator = mlx_gen_chroma::provider_registry()
        .unwrap()
        .load(id, &spec)
        .expect("tier loads");
    PROMPTS
        .iter()
        .map(|(prompt, seed)| {
            let req = GenerationRequest {
                prompt: (*prompt).into(),
                width: size(),
                height: size(),
                count: 1,
                seed: Some(*seed),
                steps: Some(steps()),
                ..Default::default()
            };
            match generator
                .generate(&req, &mut |_| {})
                .expect("generate succeeds")
            {
                GenerationOutput::Images(mut v) => v.pop().expect("one image"),
                other => panic!("expected Images, got {other:?}"),
            }
        })
        .collect()
}

/// Cosine similarity and mean absolute pixel error against the reference render.
fn metrics(reference: &Image, candidate: &Image) -> (f64, f64) {
    assert_eq!(reference.pixels.len(), candidate.pixels.len(), "size");
    let (mut dot, mut na, mut nb, mut abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in reference.pixels.iter().zip(&candidate.pixels) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        abs += (a - b).abs();
    }
    (
        dot / (na.sqrt() * nb.sqrt()),
        abs / reference.pixels.len() as f64,
    )
}

fn pixel_std(img: &Image) -> f64 {
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / img.pixels.len() as f64;
    (img.pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / img.pixels.len() as f64)
        .sqrt()
}

#[test]
#[ignore = "needs the shipped Chroma tier + Apple Silicon MLX; set SC16462_BASELINE/SC16462_OUT"]
fn compare_auxiliary_widths() {
    let baseline = PathBuf::from(
        std::env::var("SC16462_BASELINE").expect("SC16462_BASELINE = shipped q4/q8 tier dir"),
    );
    let out = PathBuf::from(std::env::var("SC16462_OUT").expect("SC16462_OUT = scratch dir"));
    std::fs::create_dir_all(&out).expect("create scratch");
    let id = model_id();

    let baseline_t5 = component_bytes(&baseline.join("text_encoder"));
    let baseline_vae = component_bytes(&baseline.join("vae"));
    println!(
        "baseline {} @ {}x{} / {} steps\n  text_encoder {:.3} GB (bf16)\n  vae {:.3} GB (bf16)",
        id,
        size(),
        size(),
        steps(),
        baseline_t5 as f64 / 1e9,
        baseline_vae as f64 / 1e9
    );

    let reference = render_all(&baseline, &id);
    for (i, img) in reference.iter().enumerate() {
        save_png(img, &out.join(format!("aux-bf16-prompt{i}.png")));
    }

    let transformer_rel = Path::new("transformer").join("diffusion_pytorch_model.safetensors");
    {
        let candidate = out.join("tier-aux-at-tier");
        if candidate.exists() {
            std::fs::remove_dir_all(&candidate).expect("clear stale candidate");
        }
        mlx_gen_chroma::convert::repack_auxiliaries(&baseline, &candidate)
            .expect("repack auxiliaries");
        let auxiliary_bits = mlx_gen::quant::packed_quant_bits_at(&candidate.join("text_encoder"))
            .expect("read packed width")
            .expect("auxiliaries are packed");

        // The whole comparison rests on this: only the auxiliaries moved.
        assert_eq!(
            std::fs::read(baseline.join(&transformer_rel)).expect("baseline transformer"),
            std::fs::read(candidate.join(&transformer_rel)).expect("candidate transformer"),
            "Q{auxiliary_bits} candidate transformer differs from the shipped transformer; the \
             auxiliary comparison would not be isolated"
        );

        let t5 = component_bytes(&candidate.join("text_encoder"));
        let vae = component_bytes(&candidate.join("vae"));
        let images = render_all(&candidate, &id);
        for (i, img) in images.iter().enumerate() {
            save_png(
                img,
                &out.join(format!("aux-q{auxiliary_bits}-prompt{i}.png")),
            );
        }

        println!(
            "\nauxiliaries @ Q{auxiliary_bits} (group {}/{}):",
            mlx_gen_chroma::convert::T5_GROUP_SIZE,
            64,
        );
        println!(
            "  text_encoder {:.3} GB ({:.1}% of bf16, -{:.3} GB)",
            t5 as f64 / 1e9,
            100.0 * t5 as f64 / baseline_t5 as f64,
            (baseline_t5 - t5) as f64 / 1e9
        );
        println!(
            "  vae          {:.3} GB ({:.1}% of bf16, -{:.3} GB)",
            vae as f64 / 1e9,
            100.0 * vae as f64 / baseline_vae as f64,
            (baseline_vae - vae) as f64 / 1e9
        );
        for (i, (reference, candidate)) in reference.iter().zip(&images).enumerate() {
            let (cosine, mae) = metrics(reference, candidate);
            let std = pixel_std(candidate);
            println!(
                "  prompt{i}: cosine {cosine:.6}  MAE {mae:.4}  px-std {std:.1}  \"{}\"",
                PROMPTS[i].0
            );
            // Completeness, not identity: a missed packed site loads u32 codes as dense floats and
            // collapses the render. That is the failure this harness must catch.
            assert!(
                std > 20.0,
                "Q{auxiliary_bits} prompt{i} render is degenerate (px-std {std:.1}) — a packed site \
                 is being read as dense floats"
            );
        }
        assert!(
            t5 < baseline_t5 && vae < baseline_vae,
            "Q{auxiliary_bits} auxiliaries did not shrink; the tier-integrity rows cannot be removed"
        );
    }
    println!("\nPNGs written to {}", out.display());
}
