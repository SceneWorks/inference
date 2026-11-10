//! SW-7090: MiniMax-H3-Image (`minimax_h3_image`) on real weights — the DiT's tensor map, its
//! first-step parity with diffusers, and a 2048² text-to-image render.
//!
//! All `#[ignore]`d: they need `MiniMaxAI/MiniMax-H3-Image` (`MINIMAX_H3_IMAGE_SNAPSHOT`, holding
//! `transformer_diffusers/`, `vae_diffusers/` and `tokenizer/`), and the render also needs
//! MiniMax-H3's `text_encoder/` (`MINIMAX_H3_SNAPSHOT`). The `minimax-h3-image` real-weights profile
//! runs them on the `mlx-minimax-h3-image` job. Each asserts on its inputs rather than skipping, so
//! a missing snapshot cannot read as a pass.

use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::denoise::schedule::SigmaSchedule;
use mlx_gen_minimax_h3::dit::config::MiniMaxH3ImageDitConfig;
use mlx_gen_minimax_h3::h3_image::{
    ImageDenoiser, ImageLayout, MiniMaxH3ImageDit, IMAGE_DIT_COMPONENT, PUBLISHED_IMAGE_DIT_TENSORS,
};
use mlx_gen_minimax_h3::model::TEXT_ENCODER_COMPONENT;

/// The committed producer whose output the parity gate accepts.
const DIT_PRODUCER: &[u8] = include_bytes!("../../tools/dump_minimax_h3_image_dit_real.py");

/// Relative L2 bound for a bf16 50-block DiT against the diffusers bf16 forward (CPU torch). The
/// two run the same mixed precision but accumulate in different orders across 50 residual
/// blocks; the cosine bound below is the shape check, this one the magnitude check.
const DIT_REL_L2_TOL: f32 = 5e-2;
/// Minimum cosine similarity of each step's velocity against diffusers.
const DIT_MIN_COSINE: f32 = 0.995;

fn image_snapshot() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_IMAGE_SNAPSHOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_SNAPSHOT must point at a MiniMaxAI/MiniMax-H3-Image snapshot root. This \
         test is #[ignore]d and asserts rather than skips."
    );
    let path = PathBuf::from(raw);
    assert!(
        path.join(IMAGE_DIT_COMPONENT).join("config.json").is_file(),
        "MINIMAX_H3_IMAGE_SNAPSHOT={} has no {IMAGE_DIT_COMPONENT}/config.json",
        path.display()
    );
    path
}

fn rel_l2(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let diff = mlx_rs::ops::subtract(&got, want).unwrap();
    let n = |x: &Array| -> f32 {
        mlx_rs::ops::multiply(x, x)
            .unwrap()
            .sum(None)
            .unwrap()
            .sqrt()
            .unwrap()
            .item()
    };
    n(&diff) / n(want)
}

fn cosine(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let dot: f32 = mlx_rs::ops::multiply(&got, want)
        .unwrap()
        .sum(None)
        .unwrap()
        .item();
    let n = |x: &Array| -> f32 {
        mlx_rs::ops::multiply(x, x)
            .unwrap()
            .sum(None)
            .unwrap()
            .sqrt()
            .unwrap()
            .item()
    };
    dot / (n(&got) * n(want))
}

/// AC2: the declared name set is EXACTLY the published `transformer_diffusers` index — 644
/// tensors, no audio projection, `pdd_proj_out.{0..6}` — read from the shard index alone.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT); index only"]
fn h3_image_dit_names_match_the_published_index() {
    let root = image_snapshot();
    let dir = root.join(IMAGE_DIT_COMPONENT);
    let cfg = MiniMaxH3ImageDitConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(cfg, MiniMaxH3ImageDitConfig::default());
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("diffusion_pytorch_model.safetensors.index.json"))
            .unwrap(),
    )
    .unwrap();
    let mut published: Vec<String> = index["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    published.sort();
    let mut declared = MiniMaxH3ImageDit::names(&cfg);
    declared.sort();
    assert_eq!(published.len(), PUBLISHED_IMAGE_DIT_TENSORS);
    assert_eq!(declared, published, "declared vs published tensor names");
    assert!(!published.iter().any(|n| n.contains("audio")));
    println!("MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_dit_names_match_the_published_index");
}

/// AC1 / AC2: the MLX DiT's step-0 and step-3 velocities against diffusers'
/// `MiniMaxH3ImageTransformer2DModel` on the same inputs (`tools/dump_minimax_h3_image_dit_real.py`,
/// hash-pinned). Step 3 runs through head 3 at `t = 0.25`, so a head or timestep mis-binding fails
/// it even when step 0 passes; the two heads are also shown to give different outputs on the
/// same rows.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot + a DiT reference (MINIMAX_H3_IMAGE_DIT_REFERENCE); ~40 GB resident"]
fn h3_image_dit_first_step_matches_diffusers() {
    use sha2::{Digest, Sha256};

    let root = image_snapshot();
    let reference_path = std::env::var("MINIMAX_H3_IMAGE_DIT_REFERENCE").unwrap_or_default();
    assert!(
        !reference_path.is_empty(),
        "MINIMAX_H3_IMAGE_DIT_REFERENCE must point at the output of \
         tools/dump_minimax_h3_image_dit_real.py; this test asserts rather than skips."
    );
    let r = Weights::from_file(&reference_path)
        .unwrap_or_else(|e| panic!("load {reference_path}: {e}"));
    assert_eq!(
        r.metadata("reference").unwrap_or_default(),
        "diffusers.MiniMaxH3ImageTransformer2DModel"
    );
    assert_eq!(
        r.metadata("producer_sha256").unwrap_or_default(),
        format!("{:x}", Sha256::digest(DIT_PRODUCER)),
        "the reference was produced by a different dump_minimax_h3_image_dit_real.py than the \
         committed one; re-produce it"
    );
    let num_text: i32 = r.metadata("num_text").unwrap().parse().unwrap();
    let (lh, lw) = {
        let hw = r.metadata("latent_hw").unwrap();
        let (h, w) = hw.split_once('x').unwrap();
        (h.parse::<i32>().unwrap(), w.parse::<i32>().unwrap())
    };
    let rows = r.require("in.rows").unwrap().clone();
    let context = r
        .require("in.context")
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    let dit = MiniMaxH3ImageDit::load(&root, Dtype::Bfloat16).unwrap();
    let layout = ImageLayout::t2i(num_text, lh, lw, dit.config().base.patch_size).unwrap();
    let denoiser = ImageDenoiser::new(
        dit,
        layout,
        &context,
        SigmaSchedule::h3_image_distilled().unwrap(),
    )
    .unwrap();
    assert!(
        denoiser.released_adaln_bytes() > 0,
        "the AdaLN projections were evicted"
    );
    let mut results = Vec::new();
    for step in [0usize, 3] {
        let got = denoiser.velocity(step, &rows).unwrap();
        mlx_rs::transforms::eval([&got]).unwrap();
        let want = r.require(&format!("out.step{step}")).unwrap();
        assert_eq!(got.shape(), want.shape(), "step {step} shape");
        results.push((step, rel_l2(&got, want), cosine(&got, want), got));
    }
    let head_gap = rel_l2(
        &results[0].3,
        &results[1].3.as_dtype(Dtype::Float32).unwrap(),
    );
    for (step, rel, cos, _) in &results {
        println!(
            "h3-image DiT step {step} vs diffusers: rel-L2 {rel:.3e} (bound {DIT_REL_L2_TOL:.0e}), \
             cosine {cos:.6} (floor {DIT_MIN_COSINE})"
        );
    }
    println!("step-0 vs step-3 velocity rel-L2 gap {head_gap:.3e}");
    for (step, rel, cos, _) in &results {
        assert!(
            *rel < DIT_REL_L2_TOL,
            "step {step} diverges: rel-L2 {rel:.3e}"
        );
        assert!(*cos > DIT_MIN_COSINE, "step {step} cosine {cos:.6}");
    }
    assert!(
        head_gap > DIT_REL_L2_TOL,
        "steps 0 and 3 gave near-identical velocities ({head_gap:.3e}); the step index is not \
         reaching the head/timestep selection"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_dit_first_step_matches_diffusers"
    );
}

/// AC1: the registered generator renders a 2048² bf16 image end to end — MiniMax-H3's text encoder
/// staged as the `text_encoder` component, six steps, the tiled 2-D VAE decode. Writes the PNG to
/// `MINIMAX_H3_IMAGE_RENDER_OUT` when set.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/); full 2048² render"]
fn h3_image_t2i_renders_2048_bf16() {
    let root = image_snapshot();
    let h3 = PathBuf::from(std::env::var("MINIMAX_H3_SNAPSHOT").unwrap_or_default());
    assert!(
        h3.join("text_encoder").join("config.json").is_file(),
        "MINIMAX_H3_SNAPSHOT={} must hold MiniMax-H3's text_encoder/ (the byte-identical copy \
         MiniMax-H3-Image conditions on)",
        h3.display()
    );
    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.to_owned(),
        WeightsSource::Dir(h3.join("text_encoder")),
    );
    let registry = mlx_gen_minimax_h3::h3_image::register_providers(
        mlx_gen::gen_core::ProviderRegistryBuilder::new(),
    )
    .build()
    .unwrap();
    let registration = registry
        .generators()
        .find(|g| (g.descriptor)().id == "minimax_h3_image")
        .expect("minimax_h3_image is registered");
    let generator = (registration.load)(&spec).unwrap();
    let req = GenerationRequest {
        prompt: "A lighthouse on a rocky coast at dusk, warm light in the lamp room, waves \
                 breaking on the rocks, detailed photograph"
            .into(),
        width: 2048,
        height: 2048,
        count: 1,
        seed: Some(7090),
        ..Default::default()
    };
    generator.validate(&req).unwrap();
    let started = std::time::Instant::now();
    let mut steps = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let mlx_gen::gen_core::Progress::Step { current, .. } = p {
                steps = current;
            }
        })
        .unwrap();
    let GenerationOutput::Images(images) = out else {
        panic!("minimax_h3_image must return images");
    };
    assert_eq!(images.len(), 1);
    let image = &images[0];
    assert_eq!((image.width, image.height), (2048, 2048));
    assert_eq!(steps, 6, "the fixed sampler runs six steps");
    let n = image.pixels.len() as f64;
    let mean = image.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = image
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    println!(
        "minimax_h3_image 2048² bf16 render in {:.1} s: pixel mean {mean:.2}, std {:.2}",
        started.elapsed().as_secs_f64(),
        var.sqrt()
    );
    assert!(
        var.sqrt() > 10.0,
        "the render is near-constant (std {:.2})",
        var.sqrt()
    );
    if let Ok(out) = std::env::var("MINIMAX_H3_IMAGE_RENDER_OUT") {
        if !out.is_empty() {
            image::save_buffer(
                &out,
                &image.pixels,
                image.width,
                image.height,
                image::ColorType::Rgb8,
            )
            .unwrap();
            println!("wrote {out}");
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_t2i_renders_2048_bf16");
}
