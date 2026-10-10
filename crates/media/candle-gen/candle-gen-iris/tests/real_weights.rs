//! Real-weight checks against the pinned `speridlabs/iris-3b` + `Qwen/Qwen3-VL-4B-Instruct`
//! snapshots on the build's device (`cuda:0` under `--features cuda`). `#[ignore]`d: they need
//! ~20 GB of weights and a GPU. Inputs come from the environment (no machine path is baked in):
//!
//! * `IRIS_WEIGHTS_DIR` — the `speridlabs/iris-3b` snapshot root (generation backbone).
//! * `IRIS_TEXT_ENCODER_DIR` — the `Qwen/Qwen3-VL-4B-Instruct` snapshot.
//!
//! The upstream reference for the parity tests is COMMITTED: the MLX twin's
//! `tests/fixtures/iris_real_golden_256.safetensors` (~4.4 MB), the compact form
//! `crates/media/mlx-gen/tools/dump_iris_realweight.py` writes beside its machine-local full golden —
//! so the CUDA dispatch lane needs no machine-local file and no network at test time.
//! * `IRIS_OUT` — a directory the provider render writes its PNG into.
//! * `IRIS_SIZE` (default 1024) / `IRIS_STEPS` (default: the release's 100) — the render geometry.
//! * `IRIS_VRAM_PROBE=1` — sample device VRAM with `nvidia-smi` across load and generate.
//!
//! `cargo test -p candle-gen-iris --release --features cuda --test integration -- --ignored
//! real_weights:: --test-threads 1`

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::iris::{
    GenerationParams, IrisConfig, PROMPT_PREFIX, PROMPT_SUFFIX, TEXT_ENCODER_COMPONENT,
};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, PreviewSink, Progress, WeightsSource,
};
use candle_gen::testkit::{probe_gpu, used_mib, VramProbe};
use candle_gen_iris::{
    denoise, encode, load_backbone, Conditioning, IrisTextEncoder, TextBatch, TextConditioning,
};

use crate::common::{assert_close, errors, fixture, fixtures, host_i32, Fixture};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .map(|s| {
            s.parse()
                .unwrap_or_else(|_| panic!("{name}={s} is not a number"))
        })
        .unwrap_or(default)
}

/// The committed real-weight golden with its conditioning re-expanded to the full window: the
/// compact file keeps only the real (mask = 1) rows, in bf16 — lossless, because the release tower
/// computes in bf16 and the pad rows are zero (both asserted by the producer) — so zero-padding the
/// rows back and widening to f32 restores exactly the tensors upstream returned.
fn real_golden() -> Fixture {
    let mut golden = fixture("iris_real_golden_256.safetensors");
    for key in ["cond", "null"] {
        let mask = host_i32(golden.require(&format!("{key}/mask")));
        let real = mask.iter().filter(|m| **m == 1).count();
        assert!(
            mask[..real].iter().all(|m| *m == 1),
            "{key}/mask must be a prefix of real rows"
        );
        let rows = golden
            .require(&format!("{key}/embeddings"))
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(
            rows.dim(0).unwrap(),
            real,
            "{key}: committed rows = real tokens"
        );
        let full = rows.pad_with_zeros(0, 0, mask.len() - real).unwrap();
        golden.tensors.insert(format!("{key}/embeddings"), full);
    }
    golden
}

/// The pinned tokenizer reproduces upstream's separately-tokenized prefix / caption / suffix ids
/// (the MLX twin's pinned id battery).
#[test]
#[ignore = "needs the Qwen/Qwen3-VL-4B-Instruct snapshot (IRIS_TEXT_ENCODER_DIR)"]
fn real_tokenizer_matches_the_pinned_ids() {
    let tokenizer = TextTokenizer::from_file(
        env_dir("IRIS_TEXT_ENCODER_DIR").join("tokenizer.json"),
        TokenizerConfig {
            max_length: usize::MAX,
            pad_token_id: 0,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .unwrap();
    let pinned: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("iris_tokenizer_ids.json")).unwrap(),
    )
    .unwrap();
    let ids = |v: &serde_json::Value| -> Vec<i32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect()
    };
    assert_eq!(
        tokenizer.encode_ids(PROMPT_PREFIX, false).unwrap(),
        ids(&pinned["prefix_ids"])
    );
    assert_eq!(
        tokenizer.encode_ids(PROMPT_SUFFIX, false).unwrap(),
        ids(&pinned["suffix_ids"])
    );
    for case in pinned["captions"].as_array().unwrap() {
        let prompt = case["prompt"].as_str().unwrap();
        assert_eq!(
            tokenizer.encode_ids(prompt, false).unwrap(),
            ids(&case["ids"]),
            "caption {prompt:?}"
        );
    }
}

/// Real weights vs the upstream CPU reference (`tools/dump_iris_realweight.py`, committed as
/// [`real_golden`]) on the build's device: the 12-layer conditioning (prompt and CFG null) and
/// one backbone + pixel-head forward at the release's bf16 policy (or FP32 with
/// `IRIS_REAL_COMPUTE=fp32`). The bf16 bound is upstream's OWN bf16-autocast distance from its FP32
/// forward on the same inputs (the MLX twin's measured mean 3.7e-3, max 0.118 of peak).
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR)"]
fn real_conditioning_and_forward_match_upstream() {
    let backbone = env_dir("IRIS_WEIGHTS_DIR");
    let golden = real_golden();
    let config = IrisConfig::from_dir(&backbone).unwrap();
    config.validate_supported().unwrap();
    let device = candle_gen::default_device().unwrap();
    let te = IrisTextEncoder::load(
        &env_dir("IRIS_TEXT_ENCODER_DIR"),
        &config.text_encoder,
        &device,
    )
    .unwrap();
    let params = GenerationParams::resolve(
        &GenerationRequest {
            prompt: golden.meta("prompt").to_owned(),
            steps: Some(golden.meta("steps").parse().unwrap()),
            guidance: Some(golden.meta("cfg_scale").parse().unwrap()),
            seed: Some(0),
            width: 256,
            height: 256,
            ..Default::default()
        },
        0,
        &config,
    )
    .unwrap();
    let conditioning = encode(&te, &params).unwrap();
    drop(te);
    assert_eq!(
        conditioning.cond[0].mask,
        host_i32(golden.require("cond/mask"))
    );
    // bf16 tower (f32 on the Candle CPU lane) vs the torch-CPU bf16 oracle: bf16-ulp scale.
    assert_close(
        "real cond/embeddings",
        &conditioning.cond[0].states.squeeze(0).unwrap(),
        golden.require("cond/embeddings"),
        2e-2,
    );
    let uncond = conditioning.uncond.as_ref().unwrap();
    assert_close(
        "real null/embeddings",
        &uncond.states.squeeze(0).unwrap(),
        golden.require("null/embeddings"),
        2e-2,
    );

    let fp32 = std::env::var("IRIS_REAL_COMPUTE").is_ok_and(|v| v == "fp32");
    let compute = if fp32 { DType::F32 } else { DType::BF16 };
    let dit = load_backbone(&backbone, &config, compute, &device).unwrap();
    let mask = vec![conditioning.cond[0].mask.clone()];
    let velocity = dit
        .forward(
            &golden.require("forward/x").to_device(&device).unwrap(),
            &golden.require("forward/t").to_device(&device).unwrap(),
            &TextBatch {
                states: &conditioning.cond[0].states,
                mask: &mask,
            },
        )
        .unwrap();
    let (max_abs, peak, mean) = errors(&velocity, golden.require("forward/velocity"));
    eprintln!(
        "[[IRIS_CANDLE]] real forward/velocity ({compute:?}): max|Δ|={max_abs:.3e} \
         mean|Δ|={mean:.3e} peak={peak:.3e}"
    );
    let (mean_tol, max_tol) = if fp32 { (1e-3, 5e-2) } else { (5e-3, 0.15) };
    assert!(
        mean <= mean_tol * peak,
        "mean |Δ| {mean} > {mean_tol} × {peak}"
    );
    assert!(
        max_abs <= max_tol * peak,
        "max |Δ| {max_abs} > {max_tol} × {peak}"
    );
}

/// The structural gate on the REAL 3B backbone: FP32 on the Candle CPU device (true f32), fed the
/// oracle's own conditioning, against upstream's FP32 CPU forward — and the whole 20-step CFG-3
/// generate from the oracle's noise — isolating the backbone + pixel head + solver from bf16
/// rounding. The MLX twin's bounds.
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR)"]
fn real_backbone_matches_upstream_in_fp32_on_cpu() {
    let backbone = env_dir("IRIS_WEIGHTS_DIR");
    let golden = real_golden();
    let config = IrisConfig::from_dir(&backbone).unwrap();
    let cpu = Device::Cpu;
    let dit = load_backbone(&backbone, &config, DType::F32, &cpu).unwrap();
    let text = |key: &str| TextConditioning {
        states: golden
            .require(&format!("{key}/embeddings"))
            .unsqueeze(0)
            .unwrap(),
        mask: host_i32(golden.require(&format!("{key}/mask"))),
        truncated_tokens: 0,
    };
    let cond = text("cond");
    let mask = vec![cond.mask.clone()];
    let start = std::time::Instant::now();
    let velocity = dit
        .forward(
            golden.require("forward/x"),
            golden.require("forward/t"),
            &TextBatch {
                states: &cond.states,
                mask: &mask,
            },
        )
        .unwrap();
    eprintln!(
        "[[IRIS_CANDLE]] cpu fp32 forward: {:.1}s",
        start.elapsed().as_secs_f32()
    );
    assert_close(
        "real forward/velocity (fp32, cpu, oracle conditioning)",
        &velocity,
        golden.require("forward/velocity"),
        1e-4,
    );

    let conditioning = Conditioning {
        cond: vec![cond],
        uncond: Some(text("null")),
        warnings: Vec::new(),
    };
    let params = GenerationParams::resolve(
        &GenerationRequest {
            prompt: golden.meta("prompt").to_owned(),
            steps: Some(golden.meta("steps").parse().unwrap()),
            guidance: Some(golden.meta("cfg_scale").parse().unwrap()),
            seed: Some(0),
            width: 256,
            height: 256,
            ..Default::default()
        },
        0,
        &config,
    )
    .unwrap();
    let start = std::time::Instant::now();
    let image = denoise(
        &dit,
        &config.flow,
        &conditioning,
        golden.require("generate/noise"),
        &params,
        &CancelFlag::new(),
        |_| {},
        &PreviewSink::default(),
    )
    .unwrap();
    eprintln!(
        "[[IRIS_CANDLE]] cpu fp32 generate: {:.1}s",
        start.elapsed().as_secs_f32()
    );
    // MLX twin: mean |Δ| 1.6e-4 on [−1, 1]; the max (≈0.1) is a handful of pixels where 20 CFG-3
    // steps amplify f32 summation-order differences. Gate the mean tightly and bound the max.
    let (max_abs, _, mean) = errors(&image, golden.require("generate/image"));
    eprintln!(
        "[[IRIS_CANDLE]] real generate/image (fp32, cpu): max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e}"
    );
    assert!(mean <= 1e-3, "mean |Δ| {mean}");
    assert!(max_abs <= 0.25, "max |Δ| {max_abs}");
}

/// The production path end to end on the build's device: catalog load from the two task
/// resources, default controls (CFG 3, empty negative), progress, one PNG.
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR) and IRIS_OUT"]
fn provider_renders_a_real_image() {
    let mut spec = LoadSpec::new(WeightsSource::Dir(env_dir("IRIS_WEIGHTS_DIR")));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(env_dir("IRIS_TEXT_ENCODER_DIR")),
    );
    let size = env_u32("IRIS_SIZE", 1024);
    let steps = env_u32("IRIS_STEPS", 100);
    // Report-only: a host without a readable `nvidia-smi` renders without the VRAM line rather
    // than failing the render it was asked to prove.
    let probe_vram =
        std::env::var("IRIS_VRAM_PROBE").is_ok_and(|v| v == "1") && used_mib(probe_gpu()).is_some();
    if !probe_vram {
        eprintln!(
            "[[IRIS_CANDLE]] vram probe off (IRIS_VRAM_PROBE unset or nvidia-smi unreadable)"
        );
    }
    let mut probe = probe_vram.then(VramProbe::start_rendered);

    let start = std::time::Instant::now();
    let load_phase = probe.as_ref().map(VramProbe::phase);
    let g = candle_gen_iris::provider_registry()
        .unwrap()
        .load("iris_3b", &spec)
        .unwrap();
    if let (Some(p), Some(phase)) = (probe.as_mut(), load_phase) {
        p.end_load(phase);
    }
    eprintln!(
        "[[IRIS_CANDLE]] device={} load={:.1}s",
        if cfg!(feature = "cuda") {
            "cuda:0"
        } else {
            "cpu"
        },
        start.elapsed().as_secs_f32()
    );
    let req = GenerationRequest {
        prompt: "a red fox sleeping in fresh snow, golden hour".into(),
        width: size,
        height: size,
        steps: Some(steps),
        seed: Some(0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let gen_phase = probe.as_ref().map(VramProbe::phase);
    let mut last = (0, 0);
    let out = g
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                last = (current, total);
            }
        })
        .unwrap();
    if let (Some(p), Some(phase)) = (probe.as_mut(), gen_phase) {
        p.end_gen(phase);
    }
    let secs = start.elapsed().as_secs_f32();
    eprintln!(
        "[[IRIS_CANDLE]] generate {size}x{size} steps={steps} cfg=3: {secs:.1}s ({:.2}s/step)",
        secs / steps as f32
    );
    if let Some(p) = &probe {
        eprintln!("[[IRIS_CANDLE]] vram {}", p.report());
    }
    assert_eq!(last, (steps, steps));
    let GenerationOutput::Images(images) = out else {
        panic!("expected images")
    };
    let img = &images[0];
    assert_eq!((img.width, img.height), (size, size));
    let path = env_dir("IRIS_OUT").join(format!("iris_3b_candle_{size}_{steps}.png"));
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .unwrap()
        .save(&path)
        .unwrap();
    let mean = img.pixels.iter().map(|&p| p as f32).sum::<f32>() / img.pixels.len() as f32;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f32 - mean).powi(2))
        .sum::<f32>()
        / img.pixels.len() as f32;
    eprintln!(
        "[[IRIS_CANDLE]] wrote {} (mean pixel {mean:.1}, std {:.1})",
        path.display(),
        var.sqrt()
    );
    assert!(mean > 5.0 && mean < 250.0, "degenerate image (mean {mean})");
    assert!(var.sqrt() > 5.0, "flat image (std {})", var.sqrt());
}

/// sc-25681: the production path with every non-default upstream control at once — a two-prompt
/// batch, order 1 (`euler`), shift 3, CFG 4.5 gated to `cfg_interval` (0.05, 0.95), a negative
/// prompt, a portrait canvas, and the per-step preview sink — one PNG per prompt.
/// `IRIS_CONTROLS_SIZE` (default 512) is the short side; `IRIS_CONTROLS_STEPS` (default 30) the step
/// count.
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR) and IRIS_OUT"]
fn provider_renders_with_non_default_controls() {
    use std::sync::{Arc, Mutex};

    let mut spec = LoadSpec::new(WeightsSource::Dir(env_dir("IRIS_WEIGHTS_DIR")));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(env_dir("IRIS_TEXT_ENCODER_DIR")),
    );
    let short: u32 = std::env::var("IRIS_CONTROLS_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let steps: u32 = std::env::var("IRIS_CONTROLS_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let (width, height) = (short, short / 16 * 20);
    let start = std::time::Instant::now();
    let g = candle_gen_iris::provider_registry()
        .unwrap()
        .load("iris_3b", &spec)
        .unwrap();
    eprintln!(
        "[[IRIS_CONTROLS]] load: {:.1}s",
        start.elapsed().as_secs_f32()
    );
    let frames = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&frames);
    let req = GenerationRequest {
        prompt_batch: vec![
            "a red fox sleeping in fresh snow, golden hour".into(),
            "a lighthouse on a rocky coast at dusk, oil painting".into(),
        ],
        width,
        height,
        steps: Some(steps),
        sampler: Some("euler".into()),
        scheduler_shift: Some(3.0),
        guidance: Some(4.5),
        cfg_interval: Some((0.05, 0.95)),
        negative_prompt: Some("blurry, low quality".into()),
        seed: Some(7),
        preview: PreviewSink::new(move |frame| {
            sink.lock().unwrap().push((
                frame.current,
                frame.total,
                frame.image.width,
                frame.image.height,
            ))
        }),
        ..Default::default()
    };
    g.validate(&req).unwrap();
    let start = std::time::Instant::now();
    let mut last = (0, 0);
    let out = g
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                last = (current, total);
            }
        })
        .unwrap();
    let secs = start.elapsed().as_secs_f32();
    eprintln!(
        "[[IRIS_CONTROLS]] generate 2 x {width}x{height} steps={steps} euler shift=3 cfg=4.5 \
         interval=(0.05,0.95): {secs:.1}s ({:.2}s/step)",
        secs / steps as f32
    );
    assert_eq!(last, (steps, steps));
    let frames = frames.lock().unwrap().clone();
    assert_eq!(frames.len(), steps as usize, "one preview frame per step");
    assert_eq!(frames[0].2, width / 16);
    assert_eq!(frames[0].3, height / 16);
    let GenerationOutput::Images(images) = out else {
        panic!("expected images")
    };
    assert_eq!(images.len(), 2, "one image per prompt of the batch");
    for (i, img) in images.iter().enumerate() {
        assert_eq!((img.width, img.height), (width, height));
        let path = env_dir("IRIS_OUT").join(format!("iris_3b_candle_controls_{i}.png"));
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .unwrap()
            .save(&path)
            .unwrap();
        let mean = img.pixels.iter().map(|&p| p as f32).sum::<f32>() / img.pixels.len() as f32;
        let var = img
            .pixels
            .iter()
            .map(|&p| (p as f32 - mean).powi(2))
            .sum::<f32>()
            / img.pixels.len() as f32;
        eprintln!(
            "[[IRIS_CONTROLS]] wrote {} (mean pixel {mean:.1}, std {:.1})",
            path.display(),
            var.sqrt()
        );
        assert!(
            mean > 5.0 && mean < 250.0,
            "degenerate image {i} (mean {mean})"
        );
        assert!(var.sqrt() > 5.0, "flat image {i} (std {})", var.sqrt());
    }
    assert_ne!(
        images[0].pixels, images[1].pixels,
        "the two prompts rendered the same image"
    );
}
