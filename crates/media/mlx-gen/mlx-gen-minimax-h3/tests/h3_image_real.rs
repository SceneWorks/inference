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

// ── SW-7092: edit (1-16 reference images) ────────────────────────────────────────────────────────

/// The committed producer whose output the edit gates accept.
const EDIT_PRODUCER: &[u8] = include_bytes!("../../tools/dump_minimax_h3_image_edit_real.py");

/// Peak-relative bound for the f32 VAE encode against diffusers' float32 encode (the video VAE's
/// real-weight encode bound on MLX).
const EDIT_ENCODE_TOL: f32 = 1e-2;

/// The edit reference, refused unless the committed producer made it.
fn edit_reference() -> Weights {
    use sha2::{Digest, Sha256};
    let path = std::env::var("MINIMAX_H3_IMAGE_EDIT_REFERENCE").unwrap_or_default();
    assert!(
        !path.is_empty(),
        "MINIMAX_H3_IMAGE_EDIT_REFERENCE must point at the output of \
         tools/dump_minimax_h3_image_edit_real.py; this test asserts rather than skips."
    );
    let r = Weights::from_file(&path).unwrap_or_else(|e| panic!("load {path}: {e}"));
    assert_eq!(
        r.metadata("producer_sha256").unwrap_or_default(),
        format!("{:x}", Sha256::digest(EDIT_PRODUCER)),
        "the reference was produced by a different dump_minimax_h3_image_edit_real.py than the \
         committed one; re-produce it"
    );
    assert_eq!(r.metadata("num_references").unwrap_or_default(), "3");
    r
}

fn host_f32(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let (g, w) = (host_f32(got), host_f32(want));
    assert_eq!(g.len(), w.len());
    let peak = w.iter().fold(0f32, |m, v| m.max(v.abs()));
    g.iter()
        .zip(&w)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / peak
}

/// AC2: the `edit` presentation this crate builds with the snapshot's own tokenizer — a
/// `Picture i: ` label and a vision block per reference, then the prompt, no chat template — is
/// the processor template's, id for id and tag for tag, as diffusers'
/// `MiniMaxH3ImageEditTextEncoderStep` builds it.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT (tokenizer/) + MINIMAX_H3_IMAGE_EDIT_REFERENCE; tokenizer only"]
fn h3_image_edit_presentation_matches_the_processor_template() {
    let root = image_snapshot();
    let r = edit_reference();
    let counts: Vec<usize> = r
        .metadata("vision_tokens")
        .unwrap()
        .split(',')
        .map(|n| n.parse().unwrap())
        .collect();
    let prompt = r.metadata("prompt").unwrap();
    let tok = mlx_gen_minimax_h3::text_encoder::MiniMaxH3Tokenizer::from_snapshot(&root).unwrap();
    let (ids, tags) = tok.image_edit_ids(prompt, &counts).unwrap();
    let want_ids: Vec<i32> = host_f32(r.require("in.token_ids").unwrap())
        .into_iter()
        .map(|v| v as i32)
        .collect();
    let want_tags: Vec<i32> = host_f32(r.require("in.token_tags").unwrap())
        .into_iter()
        .map(|v| v as i32)
        .collect();
    println!(
        "edit presentation: {} rows ({} vision blocks of {counts:?} pads)",
        ids.len(),
        counts.len()
    );
    assert_eq!(
        ids, want_ids,
        "token ids differ from the reference presentation"
    );
    assert_eq!(
        tags, want_tags,
        "row tags differ from the reference presentation"
    );
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_edit_presentation_matches_the_processor_template"
    );
}

/// AC1 (reference latents): the f32 image VAE encode of the three reference images — one wider
/// than a 256 px tile, one with off-grid pixel and latent sizes — matches diffusers'
/// `AutoencoderKLMiniMaxH3Image.encode` posterior mean, and the released latent recipe (sample
/// with the reference's own draw, float16 round, patch crop, normalize) matches its latent.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT (vae_diffusers/) + MINIMAX_H3_IMAGE_EDIT_REFERENCE"]
fn h3_image_edit_reference_latents_match_diffusers() {
    use mlx_gen_minimax_h3::h3_image::reference_vae_pixels;
    use mlx_gen_minimax_h3::image_vae::MiniMaxH3ImageVaeEncoder;

    let root = image_snapshot();
    let r = edit_reference();
    let encoder = MiniMaxH3ImageVaeEncoder::load(&root).unwrap();
    let mut worst = (0f32, 0f32);
    for i in 0..3 {
        let raw = r.require(&format!("ref{i}.image")).unwrap();
        let (h, w) = (raw.shape()[0] as u32, raw.shape()[1] as u32);
        let image = mlx_gen::media::Image {
            width: w,
            height: h,
            pixels: host_f32(raw).into_iter().map(|v| v as u8).collect(),
        };
        let posterior = encoder
            .encode(&reference_vae_pixels(&image).unwrap())
            .unwrap();
        let want_mean = r.require(&format!("ref{i}.mean")).unwrap();
        let s = want_mean.shape().to_vec();
        let mean = posterior.mean().reshape(&s).unwrap();
        let noise = r
            .require(&format!("ref{i}.posterior_noise"))
            .unwrap()
            .reshape(&[s[0], s[1], 1, s[2], s[3]])
            .unwrap();
        let latent = encoder
            .reference_latent(&posterior, &noise, (2, 2))
            .unwrap();
        mlx_rs::transforms::eval([&mean, &latent]).unwrap();
        let want_latent = r.require(&format!("ref{i}.latent")).unwrap();
        assert_eq!(
            latent.shape(),
            want_latent.shape(),
            "reference {i} latent shape"
        );
        let (pm, pl) = (peak_rel(&mean, want_mean), peak_rel(&latent, want_latent));
        println!(
            "reference {i} ({w}x{h} px): posterior mean peak-rel {pm:.3e}, latent peak-rel {pl:.3e} \
             (bound {EDIT_ENCODE_TOL:.0e})"
        );
        worst = (worst.0.max(pm), worst.1.max(pl));
    }
    assert!(
        worst.0 < EDIT_ENCODE_TOL,
        "posterior mean diverges: {:.3e}",
        worst.0
    );
    assert!(
        worst.1 < EDIT_ENCODE_TOL,
        "reference latent diverges: {:.3e}",
        worst.1
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_edit_reference_latents_match_diffusers"
    );
}

/// AC1: a 3-reference edit's step-0 (head 0) and step-3 (head 3) DiT outputs match diffusers'
/// `MiniMaxH3ImageTransformer2DModel` over `[text | 3 references | 512² image]`, computed by the
/// reference with float32 activations over the bf16 weights (`compute_dtype`), as the `text2image`
/// gate's reference is. The reference rows are built here — `0.999` noising, per-reference
/// patchify, packed in front — from the reference's latents and draws, and must equal the
/// producer's rows before the DiT runs; the layout (each reference on its own grid at its own
/// rotary slot) and the held `0.999` row class are this crate's.
///
/// Step 0 is asserted on the packed rows. Step 3 is asserted on the sampler's own `x_3`
/// (`diag.traj.rows3`: the generated rows walked through steps 0-2 by the reference, the reference
/// rows unchanged), through head 3 at `t = 0.25`. Printed, never asserted: every stage the
/// reference recorded (both timestep rows of `temb`, `norm_out`, block 0's AdaLN tables, the
/// residual stream after the `diag_layers`, `norm_out`'s output), the velocity error over the
/// reference rows and over the generated rows separately, step 3 on the step-0 rows, and both
/// asserted velocities against the reference's own bf16-compute forward (`diag.bf16.*`).
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_IMAGE_EDIT_REFERENCE; ~40 GB resident"]
fn h3_image_edit_first_step_matches_diffusers() {
    use mlx_gen_minimax_h3::h3_image::reference_condition_rows;

    let root = image_snapshot();
    let r = edit_reference();
    assert_eq!(
        r.metadata("compute_dtype").unwrap_or_default(),
        "float32",
        "the asserted edit reference must be computed with float32 activations"
    );
    let tags: Vec<i32> = host_f32(r.require("in.token_tags").unwrap())
        .into_iter()
        .map(|v| v as i32)
        .collect();
    let (lh, lw) = {
        let hw = r.metadata("latent_hw").unwrap();
        let (h, w) = hw.split_once('x').unwrap();
        (h.parse::<i32>().unwrap(), w.parse::<i32>().unwrap())
    };
    let latents: Vec<Array> = (0..3)
        .map(|i| r.require(&format!("ref{i}.latent")).unwrap().clone())
        .collect();
    let aug: Vec<Array> = (0..3)
        .map(|i| r.require(&format!("ref{i}.aug_noise")).unwrap().clone())
        .collect();
    let condition = reference_condition_rows(&latents, &aug, [1, 2, 2]).unwrap();
    let rows = mlx_rs::ops::concatenate_axis(
        &[condition, r.require("in.target_rows").unwrap().clone()],
        1,
    )
    .unwrap();
    let want_rows = r.require("in.rows").unwrap();
    let rows_gap = peak_rel(&rows, want_rows);
    assert!(
        rows_gap < 1e-6,
        "packed reference rows differ: {rows_gap:.3e}"
    );
    let grids: Vec<(i32, i32)> = latents
        .iter()
        .map(|l| (l.shape()[2], l.shape()[3]))
        .collect();
    let context = r
        .require("in.context")
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    let dit = MiniMaxH3ImageDit::load(&root, Dtype::Bfloat16).unwrap();
    let layout = ImageLayout::edit(&tags, &grids, lh, lw, dit.config().base.patch_size).unwrap();
    assert_eq!(
        layout.num_reference_rows().to_string(),
        r.metadata("num_reference_rows").unwrap()
    );
    let num_ref = layout.num_reference_rows();
    let num_packed = layout.num_packed_image_rows();
    let denoiser = ImageDenoiser::new(
        dit,
        layout,
        &context,
        SigmaSchedule::h3_image_distilled().unwrap(),
    )
    .unwrap();
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
    // The reference's tables hold one row block per DISTINCT timestep of the step, ascending: the
    // step's timestep (row class 0), then the reference rows' 0.999 (row class 1).
    let blocks = |table: &Array, step: usize, rows: i32| -> Array {
        let schedule = denoiser.adaln_cache().schedule();
        let parts: Vec<Array> = (0..2)
            .map(|class| {
                let g = schedule.global_timestep_index(step, class).unwrap();
                slice_axis(table, 0, g * rows, (g + 1) * rows).unwrap()
            })
            .collect();
        mlx_rs::ops::concatenate_axis(&parts, 0).unwrap()
    };
    let regions = |step: usize, what: &str, got: &Array, want: &Array| {
        for (name, lo, hi) in [
            ("reference rows", 0, num_ref),
            ("generated rows", num_ref, num_packed),
        ] {
            println!(
                "{}",
                stage_line(
                    step,
                    &format!("{what} {name}"),
                    &slice_axis(got, 1, lo, hi).unwrap(),
                    &slice_axis(want, 1, lo, hi).unwrap(),
                )
            );
        }
    };
    let mut results = Vec::new();
    for step in [0usize, 3] {
        let mut trace = ImageDitTrace::new(&diag_layers);
        let got = denoiser.velocity_traced(step, &rows, &mut trace).unwrap();
        mlx_rs::transforms::eval([&got]).unwrap();
        let want = r.require(&format!("out.step{step}")).unwrap();
        assert_eq!(got.shape(), want.shape(), "step {step} shape");

        let d = |name: &str| r.require(&format!("diag.step{step}.{name}")).unwrap();
        let mods = denoiser.adaln_cache().modulation(0).unwrap();
        let block0 = mlx_rs::ops::stack_axis(
            &mods
                .tables()
                .map(|t| blocks(t, step, 3))
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();
        let norm_out = denoiser.norm_out_modulation();
        let mut lines = vec![
            stage_line(step, "temb", &blocks(denoiser.temb(), step, 1), d("temb")),
            stage_line(
                step,
                "norm_out.shift",
                &blocks(&norm_out.shift, step, 1),
                d("norm_out.shift"),
            ),
            stage_line(
                step,
                "norm_out.scale",
                &blocks(&norm_out.scale, step, 1),
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
        regions(step, "velocity", &got, want);
        results.push((step, rel_l2(&got, want), cosine(&got, want), got));
    }
    // Asserted step 3: the sampler's own x_3, reached by the reference from the same rows.
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
    regions(3, "on-trajectory", &traj, traj_want);
    // The reference's own rounding: the same two asserted inputs at bf16 compute.
    let want0 = r.require("out.step0").unwrap();
    let bf16_step0 = r.require("diag.bf16.step0").unwrap();
    let bf16_traj = r.require("diag.bf16.traj.out3").unwrap();
    println!("{}", stage_line(0, "ref bf16 vs f32", bf16_step0, want0));
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
            "h3-image edit DiT {what} vs diffusers: rel-L2 {rel:.3e} (bound \
             {DIT_REL_L2_TOL:.0e}), cosine {cos:.6} (floor {DIT_MIN_COSINE})"
        );
    }
    println!(
        "h3-image edit DiT step 3 off-trajectory (step-0 rows at t = 0.25; diagnostic, not \
         asserted): rel-L2 {off_rel:.3e}, cosine {off_cos:.6}"
    );
    println!("step-0 vs step-3 velocity rel-L2 gap {head_gap:.3e}");
    for (what, rel, cos) in &asserted {
        assert!(*rel < DIT_REL_L2_TOL, "{what} diverges: rel-L2 {rel:.3e}");
        assert!(*cos > DIT_MIN_COSINE, "{what} cosine {cos:.6}");
    }
    assert!(
        head_gap > DIT_REL_L2_TOL,
        "steps 0 and 3 are indistinguishable"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_edit_first_step_matches_diffusers"
    );
}

/// The registered generator runs a 3-reference `edit` end to end at a requested 1024x768 —
/// vision tower + grounded conditioner, the reference VAE encodes, six steps over the packed
/// `[text | references | image]` sequence, the tiled decode — and returns the requested size.
/// Writes the PNG to `MINIMAX_H3_IMAGE_EDIT_RENDER_OUT` when set.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT + MINIMAX_H3_SNAPSHOT (text_encoder/ shards 1-12 and 14); full edit render"]
fn h3_image_edit_renders_three_references_bf16() {
    let root = image_snapshot();
    let h3 = PathBuf::from(std::env::var("MINIMAX_H3_SNAPSHOT").unwrap_or_default());
    assert!(
        h3.join("text_encoder")
            .join("model-00014-of-00014.safetensors")
            .is_file(),
        "MINIMAX_H3_SNAPSHOT={} must hold MiniMax-H3's text_encoder/ including shard 14 (the \
         vision tower)",
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
    let reference = |w: u32, h: u32, k: u32| mlx_gen::media::Image {
        width: w,
        height: h,
        pixels: (0..w * h)
            .flat_map(|p| {
                let (x, y) = (p % w, p / w);
                [
                    ((x * 255) / w) as u8,
                    ((y * 255) / h) as u8,
                    (((x / (8 + k)) + (y / (8 + k))) % 2 * 200) as u8,
                ]
            })
            .collect(),
    };
    let req = GenerationRequest {
        prompt: "Place the checkered tile from <Picture 1> on the floor of the room in \
                 <Picture 2>, with the colours of <Picture 3>"
            .into(),
        width: 1024,
        height: 768,
        count: 1,
        seed: Some(7092),
        conditioning: vec![
            mlx_gen::gen_core::Conditioning::Reference {
                image: reference(640, 480, 0),
                strength: None,
            },
            mlx_gen::gen_core::Conditioning::MultiReference {
                images: vec![reference(512, 768, 3), reference(1200, 1100, 6)],
            },
        ],
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
    assert_eq!(
        (image.width, image.height),
        (1024, 768),
        "the size is the request's"
    );
    assert_eq!(steps, 6);
    let n = image.pixels.len() as f64;
    let mean = image.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let std = (image
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    println!(
        "minimax_h3_image 3-reference edit at 1024x768 in {:.1} s: pixel mean {mean:.2}, std {std:.2}",
        started.elapsed().as_secs_f64()
    );
    assert!(std > 10.0, "the render is near-constant (std {std:.2})");
    if let Ok(out) = std::env::var("MINIMAX_H3_IMAGE_EDIT_RENDER_OUT") {
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
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_edit_renders_three_references_bf16"
    );
}

// ---- SW-7093: the q8 / q4 DiT tiers ---------------------------------------------------------

/// Where the converter wrote the tiers: `<root>/{q8,q4}/transformer_diffusers/` plus
/// `<root>/{q8,q4}/pdd_fused_heads.safetensors` (`MINIMAX_H3_IMAGE_TIERS_ROOT`).
fn tiers_root() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_IMAGE_TIERS_ROOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_TIERS_ROOT must point at the converter's output (the \
         `minimax_h3_prequant` example with MINIMAX_H3_COMPONENT=image_dit). This test is \
         #[ignore]d and asserts rather than skips."
    );
    PathBuf::from(raw)
}

/// Audit a converted tier's index: the tier carries exactly the published 644 tensor names plus a
/// `.scales` / `.biases` pair for every packed one; the six `pdd_proj_out.{i}` heads and the I/O
/// projections are dense float32 / unpacked; the blocks are packed; the heads file is copied
/// byte for byte.
fn audit_tier(label: &str, bits: i32) {
    let tier = tiers_root().join(label);
    let dir = tier.join(IMAGE_DIT_COMPONENT);
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["quantization"]["bits"], bits, "{label} marker bits");
    assert_eq!(cfg["quantization"]["group_size"], 64);
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("diffusion_pytorch_model.safetensors.index.json"))
            .unwrap(),
    )
    .unwrap();
    let keys: std::collections::BTreeSet<String> = index["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    let published: std::collections::BTreeSet<String> = MiniMaxH3ImageDit::names(
        &MiniMaxH3ImageDitConfig::from_diffusers_json(
            &std::fs::read_to_string(dir.join("config.json")).unwrap(),
        )
        .unwrap(),
    )
    .into_iter()
    .collect();
    assert_eq!(published.len(), PUBLISHED_IMAGE_DIT_TENSORS);
    let unpacked: std::collections::BTreeSet<String> = keys
        .iter()
        .filter(|k| !k.ends_with(".scales") && !k.ends_with(".biases"))
        .cloned()
        .collect();
    assert_eq!(
        unpacked, published,
        "{label}: tensor names vs the published set"
    );
    for i in 0..6 {
        assert!(
            !keys.contains(&format!("pdd_proj_out.{i}.scales")),
            "{label}: pdd_proj_out.{i} was packed"
        );
    }
    for dense in [
        "proj_in",
        "context_embedder",
        "norm_out.linear",
        "time_embedder.linear_1",
    ] {
        assert!(
            !keys.contains(&format!("{dense}.scales")),
            "{label}: {dense} was packed"
        );
    }
    assert!(
        keys.contains("transformer_blocks.0.attn.to_q.scales"),
        "{label}: blocks not packed"
    );
    assert!(keys.contains("transformer_blocks.49.adaln_proj.linear.scales"));
    let w = Weights::from_dir(&dir).unwrap();
    for i in 0..6 {
        assert_eq!(
            w.require(&format!("pdd_proj_out.{i}.weight"))
                .unwrap()
                .dtype(),
            Dtype::Float32,
            "{label}: head {i} is dense float32"
        );
    }
    let heads = image_snapshot().join("pdd_fused_heads.safetensors");
    if heads.is_file() {
        assert_eq!(
            std::fs::read(tier.join("pdd_fused_heads.safetensors")).unwrap(),
            std::fs::read(&heads).unwrap(),
            "{label}: pdd_fused_heads.safetensors must be a byte copy"
        );
    }
}

/// Audit the converted tier, then render 2048² through the registered generator at it. The text encoder is MiniMax-H3's
/// own tier artifact from `SceneWorks/minimax-h3-mlx` (`MINIMAX_H3_TIER_SNAPSHOT/{tier}/text_encoder`,
/// AC2) -- never a copy under the image repository.
fn render_tier(label: &str, quant: mlx_gen::gen_core::Quant) {
    // AC1 (converter): the tier keeps the heads dense and packs the blocks.
    audit_tier(label, quant.bits());
    let root = image_snapshot();
    let te = PathBuf::from(std::env::var("MINIMAX_H3_TIER_SNAPSHOT").unwrap_or_default())
        .join(label)
        .join("text_encoder");
    assert!(
        te.join("config.json").is_file(),
        "MINIMAX_H3_TIER_SNAPSHOT must hold MiniMax-H3's {label}/text_encoder/ (found no {})",
        te.join("config.json").display()
    );
    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    spec.quantize = Some(quant);
    spec.components.insert(
        IMAGE_DIT_COMPONENT.to_owned(),
        WeightsSource::Dir(tiers_root().join(label).join(IMAGE_DIT_COMPONENT)),
    );
    spec.components
        .insert(TEXT_ENCODER_COMPONENT.to_owned(), WeightsSource::Dir(te));
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
        seed: Some(7093),
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
    let image = &images[0];
    assert_eq!((image.width, image.height), (2048, 2048));
    assert_eq!(steps, 6);
    let n = image.pixels.len() as f64;
    let mean = image.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let std = (image
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    println!(
        "minimax_h3_image 2048² {label} render in {:.1} s: pixel mean {mean:.2}, std {std:.2}",
        started.elapsed().as_secs_f64()
    );
    assert!(
        std > 10.0,
        "the {label} render is near-constant (std {std:.2})"
    );
    if let Ok(out) = std::env::var("MINIMAX_H3_IMAGE_RENDER_OUT") {
        if !out.is_empty() {
            image::save_buffer(
                format!("{out}-{label}.png"),
                &image.pixels,
                image.width,
                image.height,
                image::ColorType::Rgb8,
            )
            .unwrap();
        }
    }
    crate::model_inference_device::record_model_inference_device();
}

/// AC1: a 2048² render at the q8 tier.
#[test]
#[ignore = "needs the converted q8 tier, MINIMAX_H3_IMAGE_SNAPSHOT and MiniMax-H3's q8 text encoder (MINIMAX_H3_TIER_SNAPSHOT); full 2048² render"]
fn h3_image_t2i_renders_2048_q8() {
    render_tier("q8", mlx_gen::gen_core::Quant::Q8);
    println!("MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_t2i_renders_2048_q8");
}

/// AC1: a 2048² render at the q4 tier.
#[test]
#[ignore = "needs the converted q4 tier, MINIMAX_H3_IMAGE_SNAPSHOT and MiniMax-H3's q4 text encoder (MINIMAX_H3_TIER_SNAPSHOT); full 2048² render"]
fn h3_image_t2i_renders_2048_q4() {
    render_tier("q4", mlx_gen::gen_core::Quant::Q4);
    println!("MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_t2i_renders_2048_q4");
}

/// The q4 tier's load spec: the converted DiT and MiniMax-H3's own q4 text encoder, at `shape`.
fn q4_spec(shape: mlx_gen::gen_core::LoadShape) -> LoadSpec {
    let te = PathBuf::from(std::env::var("MINIMAX_H3_TIER_SNAPSHOT").unwrap_or_default())
        .join("q4")
        .join("text_encoder");
    assert!(
        te.join("config.json").is_file(),
        "MINIMAX_H3_TIER_SNAPSHOT must hold MiniMax-H3's q4/text_encoder/ (found no {})",
        te.join("config.json").display()
    );
    let mut spec = LoadSpec::new(WeightsSource::Dir(image_snapshot()));
    spec.quantize = Some(mlx_gen::gen_core::Quant::Q4);
    spec.load_shape = shape;
    spec.components.insert(
        IMAGE_DIT_COMPONENT.to_owned(),
        WeightsSource::Dir(tiers_root().join("q4").join(IMAGE_DIT_COMPONENT)),
    );
    spec.components
        .insert(TEXT_ENCODER_COMPONENT.to_owned(), WeightsSource::Dir(te));
    spec
}

/// SW-7094: a windowed (rung 4) image DiT reproduces the resident one. One step at 1024² through
/// `ImageDenoiser::new` and through `ImageDenoiser::new_windowed` (window 1) on the real q4 tier,
/// with identical synthetic inputs; H3's block stream is bit-identical on every tier, so the two
/// velocities must be too.
#[test]
#[ignore = "needs the converted q4 tier (MINIMAX_H3_IMAGE_TIERS_ROOT); loads the q4 image DiT twice"]
fn h3_image_streamed_dit_matches_resident_q4() {
    let dir = tiers_root().join("q4").join(IMAGE_DIT_COMPONENT);
    let cfg = MiniMaxH3ImageDitConfig::default();
    let layout = ImageLayout::t2i(256, 64, 64, cfg.base.patch_size).unwrap();
    let key = |seed| mlx_rs::random::key(seed).unwrap();
    let context =
        mlx_rs::random::normal::<f32>(&[1, 256, cfg.base.text_dim], None, None, Some(&key(5)))
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
    let rows = mlx_rs::random::normal::<f32>(
        &[
            1,
            layout.num_packed_image_rows(),
            cfg.base.video_patch_dim(),
        ],
        None,
        None,
        Some(&key(6)),
    )
    .unwrap();
    mlx_rs::transforms::eval([&context, &rows]).unwrap();
    let schedule = || SigmaSchedule::h3_image_distilled().unwrap();
    let resident = {
        let dit = MiniMaxH3ImageDit::load_dir(&dir, Dtype::Bfloat16).unwrap();
        let d = ImageDenoiser::new(dit, layout.clone(), &context, schedule()).unwrap();
        let v = d.velocity(0, &rows).unwrap();
        mlx_rs::transforms::eval([&v]).unwrap();
        v
    };
    mlx_rs::memory::clear_cache();
    let streamed = {
        let dit = MiniMaxH3ImageDit::load_dir_deferred(&dir, Dtype::Bfloat16).unwrap();
        assert!(dit.is_deferred() && dit.blocks().is_empty());
        let d = ImageDenoiser::new_windowed(
            dit,
            layout.clone(),
            &context,
            schedule(),
            1,
            mlx_gen::CancelFlag::default(),
        )
        .unwrap();
        assert_eq!(d.block_window().unwrap().window_count(), 50);
        assert_eq!(d.released_adaln_bytes(), 0);
        let v = d.velocity(0, &rows).unwrap();
        mlx_rs::transforms::eval([&v]).unwrap();
        v
    };
    let diff = mlx_gen::testing::max_abs_diff_f32(&resident, &streamed);
    println!("minimax_h3_image q4 streamed vs resident step 0: max|Δ| {diff:e}");
    assert_eq!(
        diff, 0.0,
        "the windowed image DiT moved the velocity by {diff:e} — H3's block stream is bit-identical"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_streamed_dit_matches_resident_q4"
    );
}

/// SW-7094 AC2: a 16 MP (4096²) q4 request that does NOT fit resident under the admitted budget is
/// admitted on rung 4 by `h3_image::admission::admit` and completes through the streamed route.
///
/// The budget is set to 1.25x the streamed estimate, which the test asserts is below the staged
/// estimate — so the resident and staged rungs are refused and the walk must stream. The render
/// runs with the generation memory the contract derives for that selection, on a
/// `DeferredMaterialization` load, and its MLX peak must stay inside the budget. The per-phase
/// estimates are printed (derived, not recorded as anchors).
#[test]
#[ignore = "needs the converted q4 tier, MINIMAX_H3_IMAGE_SNAPSHOT and MiniMax-H3's q4 text encoder (MINIMAX_H3_TIER_SNAPSHOT); 16 MP streamed render"]
fn h3_image_t2i_renders_16mp_streamed_q4() {
    use mlx_gen::gen_core::{LoadShape, MemoryBudget, MemoryStrategy};
    use mlx_gen_minimax_h3::h3_image::admission::{
        admit, estimate, ladder, ImageMemoryFacts, ImageRequestShape,
    };
    let spec = q4_spec(LoadShape::DeferredMaterialization);
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
    let contract = generator.memory_strategy_contract().unwrap().clone();
    let tier = mlx_gen::request_scope::loaded_tier(&spec);
    let facts = ImageMemoryFacts::resolve(&spec).unwrap();
    let shape = ImageRequestShape {
        prompt_tokens: Some(64),
        ..ImageRequestShape::text_to_image(4096, 4096)
    };
    let at = |strategy| {
        let selection = ladder(&contract, tier)
            .into_iter()
            .find(|s| s.strategy == strategy)
            .unwrap();
        estimate(&contract, &facts, &shape, &selection).unwrap()
    };
    let staged = at(MemoryStrategy::StagedResidency);
    let streamed = at(MemoryStrategy::BoundedTransformerResidency);
    let total = streamed.peak_bytes() + streamed.peak_bytes() / 4;
    assert!(
        total < staged.peak_bytes(),
        "the budget {total} must refuse the staged rung ({} B) for this to test streaming",
        staged.peak_bytes()
    );
    let admission = admit(
        &contract,
        &facts,
        &shape,
        tier,
        MemoryBudget {
            total_bytes: total,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
    )
    .unwrap();
    assert_eq!(
        admission.estimate.selection.strategy,
        MemoryStrategy::BoundedTransformerResidency
    );
    for e in admission.rejected.iter().chain([&admission.estimate]) {
        println!(
            "estimate {:?}: peak {:.2} GB ({:?} binds) — {}",
            e.selection.strategy,
            e.peak_bytes() as f64 / 1e9,
            e.binding_phase(),
            e.phases
                .iter()
                .map(|p| format!(
                    "{:?} {:.2} GB GPU + {:.3} GB host",
                    p.phase,
                    p.device_bytes as f64 / 1e9,
                    p.host_bytes as f64 / 1e9
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let req = GenerationRequest {
        prompt: "A lighthouse on a rocky coast at dusk, warm light in the lamp room, waves \
                 breaking on the rocks, detailed photograph"
            .into(),
        width: 4096,
        height: 4096,
        count: 1,
        seed: Some(7094),
        memory: contract.generation_memory(&admission.estimate.selection),
        ..Default::default()
    };
    assert!(req.memory.is_some_and(|m| m.stream_transformer_blocks));
    generator.validate(&req).unwrap();
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let started = std::time::Instant::now();
    let mut steps = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let mlx_gen::gen_core::Progress::Step { current, .. } = p {
                steps = current;
            }
        })
        .unwrap();
    let peak = mlx_rs::memory::get_peak_memory() as u64;
    let GenerationOutput::Images(images) = out else {
        panic!("minimax_h3_image must return images");
    };
    let image = &images[0];
    assert_eq!((image.width, image.height), (4096, 4096));
    assert_eq!(steps, 6);
    let n = image.pixels.len() as f64;
    let mean = image.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let std = (image
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    println!(
        "minimax_h3_image 4096² q4 streamed render in {:.1} s: MLX peak {:.2} GB against a {:.2} GB \
         budget (streamed estimate {:.2} GB); pixel mean {mean:.2}, std {std:.2}",
        started.elapsed().as_secs_f64(),
        peak as f64 / 1e9,
        total as f64 / 1e9,
        streamed.peak_bytes() as f64 / 1e9
    );
    assert!(
        std > 10.0,
        "the 16 MP render is near-constant (std {std:.2})"
    );
    assert!(
        peak <= total,
        "the streamed render peaked at {peak} B, above the {total} B budget it was admitted under"
    );
    if let Ok(out) = std::env::var("MINIMAX_H3_IMAGE_RENDER_OUT") {
        if !out.is_empty() {
            image::save_buffer(
                format!("{out}-16mp-streamed-q4.png"),
                &image.pixels,
                image.width,
                image.height,
                image::ColorType::Rgb8,
            )
            .unwrap();
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED h3_image_real::h3_image_t2i_renders_16mp_streamed_q4");
}
