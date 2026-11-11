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
    ImageDenoiser, ImageDitTrace, ImageLayout, MiniMaxH3ImageDit, IMAGE_DIT_COMPONENT,
    PUBLISHED_IMAGE_DIT_TENSORS,
};
use mlx_gen_minimax_h3::model::TEXT_ENCODER_COMPONENT;
use mlx_gen_minimax_h3::tensor::slice_axis;

/// The committed producer whose output the parity gate accepts.
const DIT_PRODUCER: &[u8] = include_bytes!("../../tools/dump_minimax_h3_image_dit_real.py");

/// The family DiT parity bounds (requirement E9), shared with the Candle gate: a bf16 50-block DiT
/// against the diffusers forward (CPU torch, float32 activations over the same bf16-stored weights).
use mlx_gen::gen_core::h3_image::{
    DIT_PARITY_MAX_REL_L2 as DIT_REL_L2_TOL, DIT_PARITY_MIN_COSINE as DIT_MIN_COSINE,
};

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

/// `max|got - want| / max|want|` — the peak-relative error, reported next to rel-L2 per stage.
fn rel_peak(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let diff = mlx_rs::ops::abs(mlx_rs::ops::subtract(&got, want).unwrap()).unwrap();
    let peak: f32 = mlx_rs::ops::abs(want).unwrap().max(None).unwrap().item();
    let d: f32 = diff.max(None).unwrap().item();
    d / peak.max(1e-12)
}

/// One diagnostic line: a named stage's rel-L2, peak-relative error and cosine against the
/// reference's own tensor. Printed, never asserted — the bounds stay on the velocity alone.
fn stage_line(step: usize, stage: &str, got: &Array, want: &Array) -> String {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    assert_eq!(got.shape(), want.shape(), "step {step} {stage} shape");
    format!(
        "h3-image DiT step {step} stage {stage:<16} rel-L2 {:.3e} peak-rel {:.3e} cosine {:.6}",
        rel_l2(&got, want),
        rel_peak(&got, want),
        cosine(&got, want)
    )
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
/// `MiniMaxH3ImageTransformer2DModel` (`tools/dump_minimax_h3_image_dit_real.py`, hash-pinned,
/// float32 activations over the bf16 weights). Step 0 is asserted on the noise rows. Step 3 is
/// asserted on the sampler's own `x_3` (`diag.traj.rows3`, reached by the reference from those
/// noise rows), through head 3 at `t = 0.25`, so a head or timestep mis-binding fails it even when
/// step 0 passes; the two heads are also shown to give different outputs on the same rows.
///
/// Before asserting, every step prints its error at each stage the reference recorded (`temb`,
/// `norm_out` shift/scale, block 0's AdaLN tables, the residual stream after the `diag_layers`,
/// `norm_out`'s output). Also printed, never asserted: step 3 on the step-0 noise rows (an input
/// no sampler shows the model at `t = 0.25`), and both asserted velocities against the
/// reference's own bf16-compute forward (`diag.bf16.*`).
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
    assert_eq!(
        r.metadata("compute_dtype").unwrap_or_default(),
        "float32",
        "the asserted reference must be computed with float32 activations"
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
    let want_t: Vec<f32> = r
        .metadata("timesteps")
        .unwrap()
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    let sched_t = denoiser.schedule().timesteps();
    assert_eq!(
        want_t,
        vec![sched_t[0], sched_t[3]],
        "the timesteps must be the released sampler's float32 values"
    );
    let diag_layers: Vec<usize> = r
        .metadata("diag_layers")
        .expect("the reference records its diag_layers")
        .split(',')
        .map(|k| k.parse().unwrap())
        .collect();
    let row_block = |table: &Array, g: i32, rows: i32| -> Array {
        slice_axis(table, 0, g * rows, (g + 1) * rows).unwrap()
    };
    let mut results = Vec::new();
    for step in [0usize, 3] {
        let mut trace = ImageDitTrace::new(&diag_layers);
        let got = denoiser.velocity_traced(step, &rows, &mut trace).unwrap();
        mlx_rs::transforms::eval([&got]).unwrap();
        let want = r.require(&format!("out.step{step}")).unwrap();
        assert_eq!(got.shape(), want.shape(), "step {step} shape");

        let d = |name: &str| r.require(&format!("diag.step{step}.{name}")).unwrap();
        let g = denoiser
            .adaln_cache()
            .schedule()
            .global_timestep_index(step, 0)
            .unwrap();
        let mods = denoiser.adaln_cache().modulation(0).unwrap();
        let block0 = mlx_rs::ops::stack_axis(
            &mods
                .tables()
                .map(|t| row_block(t, g, 3))
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();
        let norm_out = denoiser.norm_out_modulation();
        let mut lines = vec![
            stage_line(step, "temb", &row_block(denoiser.temb(), g, 1), d("temb")),
            stage_line(
                step,
                "norm_out.shift",
                &row_block(&norm_out.shift, g, 1),
                d("norm_out.shift"),
            ),
            stage_line(
                step,
                "norm_out.scale",
                &row_block(&norm_out.scale, g, 1),
                d("norm_out.scale"),
            ),
            stage_line(step, "block0.adaln", &block0, d("block0.adaln")),
        ];
        for (layer, hidden) in &trace.hidden {
            lines.push(stage_line(
                step,
                &format!("hidden.block{layer}"),
                hidden,
                d(&format!("hidden.block{layer}")),
            ));
        }
        lines.push(stage_line(
            step,
            "pre_head",
            trace.pre_head.as_ref().expect("the trace records norm_out"),
            d("pre_head"),
        ));
        lines.push(stage_line(step, "velocity", &got, want));
        for line in lines {
            println!("{line}");
        }
        results.push((step, rel_l2(&got, want), cosine(&got, want), got));
    }
    // Asserted step 3: the sampler's own x_3, reached by the reference from the same noise rows.
    let traj_rows = r.require("diag.traj.rows3").unwrap().clone();
    let traj = denoiser.velocity(3, &traj_rows).unwrap();
    mlx_rs::transforms::eval([&traj]).unwrap();
    let traj_want = r.require("diag.traj.out3").unwrap();
    assert_eq!(
        traj.shape(),
        traj_want.shape(),
        "on-trajectory step 3 shape"
    );
    println!("{}", stage_line(3, "on-trajectory", &traj, traj_want));
    // The reference's own rounding: the same two asserted inputs at bf16 compute.
    let bf16_step0 = r.require("diag.bf16.step0").unwrap();
    let bf16_traj = r.require("diag.bf16.traj.out3").unwrap();
    println!(
        "{}",
        stage_line(
            0,
            "ref bf16 vs f32",
            bf16_step0,
            r.require("out.step0").unwrap()
        )
    );
    println!("{}", stage_line(3, "ref bf16 vs f32", bf16_traj, traj_want));
    println!(
        "{}",
        stage_line(0, "vs ref bf16", &results[0].3, bf16_step0)
    );
    println!("{}", stage_line(3, "traj vs ref bf16", &traj, bf16_traj));
    let head_gap = rel_l2(
        &results[0].3,
        &results[1].3.as_dtype(Dtype::Float32).unwrap(),
    );
    let (off_rel, off_cos) = (results[1].1, results[1].2);
    let asserted = [
        ("step 0", results[0].1, results[0].2),
        (
            "step 3 on-trajectory",
            rel_l2(&traj, traj_want),
            cosine(&traj, traj_want),
        ),
    ];
    for (what, rel, cos) in &asserted {
        println!(
            "h3-image DiT {what} vs diffusers: rel-L2 {rel:.3e} (bound {DIT_REL_L2_TOL:.0e}), \
             cosine {cos:.6} (floor {DIT_MIN_COSINE})"
        );
    }
    println!(
        "h3-image DiT step 3 off-trajectory (step-0 noise rows at t = 0.25; diagnostic, not \
         asserted): rel-L2 {off_rel:.3e}, cosine {off_cos:.6}"
    );
    println!("step-0 vs step-3 velocity rel-L2 gap {head_gap:.3e}");
    for (what, rel, cos) in &asserted {
        assert!(*rel < DIT_REL_L2_TOL, "{what} diverges: rel-L2 {rel:.3e}");
        assert!(*cos > DIT_MIN_COSINE, "{what} cosine {cos:.6}");
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
