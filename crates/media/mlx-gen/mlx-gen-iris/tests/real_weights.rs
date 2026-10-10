//! Real-weight checks against the pinned `speridlabs/iris-3b` + `Qwen/Qwen3-VL-4B-Instruct`
//! snapshots. `#[ignore]`d: they need ~20 GB of weights and the GPU. Inputs come from the
//! environment (no machine path is baked in):
//!
//! * `IRIS_WEIGHTS_DIR` — the `speridlabs/iris-3b` snapshot root (generation backbone).
//! * `IRIS_TEXT_ENCODER_DIR` — the `Qwen/Qwen3-VL-4B-Instruct` snapshot.
//! * `IRIS_REAL_GOLDEN` — the machine-local upstream reference from `tools/dump_iris_realweight.py`
//!   (parity tests only).
//! * `IRIS_OUT` — a directory the provider render writes its PNG into (render test only).
//!
//! Run each under an external memory guard:
//! `cargo test -p mlx-gen-iris --release --test integration -- --ignored real_weights:: --test-threads 1`

use std::path::PathBuf;

use mlx_gen::gen_core::iris::{GenerationParams, IrisConfig, TEXT_ENCODER_COMPONENT};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};
use mlx_gen_iris::{
    denoise, encode, load_backbone, Conditioning, IrisTextEncoder, TextBatch, TextConditioning,
};
use mlx_rs::Dtype;

use crate::common::{assert_close, errors, fixtures, host_i32};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

/// The pinned tokenizer reproduces upstream's separately-tokenized prefix / caption / suffix ids.
#[test]
#[ignore = "needs the Qwen/Qwen3-VL-4B-Instruct snapshot (IRIS_TEXT_ENCODER_DIR)"]
fn real_tokenizer_matches_the_pinned_ids() {
    let tokenizer = mlx_gen::tokenizer::TextTokenizer::from_file(
        env_dir("IRIS_TEXT_ENCODER_DIR").join("tokenizer.json"),
        mlx_gen::tokenizer::TokenizerConfig {
            max_length: usize::MAX,
            pad_token_id: 0,
            chat_template: mlx_gen::tokenizer::ChatTemplate::None,
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
    use mlx_gen::gen_core::iris::{PROMPT_PREFIX, PROMPT_SUFFIX};
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

/// Real weights vs the upstream CPU reference: the 12-layer conditioning (prompt and CFG null) and
/// one backbone + pixel-head forward at the release's bf16 policy (or FP32 with
/// `IRIS_REAL_COMPUTE=fp32`). The bf16 bound is upstream's OWN bf16-autocast distance from its FP32
/// forward on the same inputs (measured mean 3.7e-3, max 0.118 of peak).
#[test]
#[ignore = "needs the real weights + IRIS_REAL_GOLDEN (tools/dump_iris_realweight.py)"]
fn real_conditioning_and_forward_match_upstream() {
    let backbone = env_dir("IRIS_WEIGHTS_DIR");
    let golden = Weights::from_file(env_dir("IRIS_REAL_GOLDEN")).unwrap();
    let config = IrisConfig::from_dir(&backbone).unwrap();
    config.validate_supported().unwrap();
    let prompt = golden.metadata("prompt").unwrap().to_owned();

    let te =
        IrisTextEncoder::load(&env_dir("IRIS_TEXT_ENCODER_DIR"), &config.text_encoder).unwrap();
    let params = GenerationParams {
        steps: golden.metadata("steps").unwrap().parse().unwrap(),
        cfg_scale: golden.metadata("cfg_scale").unwrap().parse().unwrap(),
        negative_prompt: String::new(),
        seed: 0,
        width: 256,
        height: 256,
    };
    let conditioning = encode(&te, &prompt, &params).unwrap();
    drop(te);
    assert_eq!(
        conditioning.cond.mask,
        host_i32(golden.require("cond/mask").unwrap())
    );
    // bf16 tower on both sides (MLX GPU vs torch CPU): bf16-ulp scale.
    assert_close(
        "real cond/embeddings",
        &conditioning.cond.states.squeeze_axes(&[0]).unwrap(),
        golden.require("cond/embeddings").unwrap(),
        2e-2,
    );
    let uncond = conditioning.uncond.as_ref().unwrap();
    assert_close(
        "real null/embeddings",
        &uncond.states.squeeze_axes(&[0]).unwrap(),
        golden.require("null/embeddings").unwrap(),
        2e-2,
    );

    // `IRIS_REAL_COMPUTE=fp32` runs the backbone in FP32 (Metal's f32 GEMM is itself reduced
    // precision, ~1e-3 relative) to separate structure from bf16 rounding; the default is the
    // release's bf16 autocast policy.
    let fp32 = std::env::var("IRIS_REAL_COMPUTE").is_ok_and(|v| v == "fp32");
    let compute = if fp32 {
        Dtype::Float32
    } else {
        Dtype::Bfloat16
    };
    let dit = load_backbone(&backbone, &config, compute).unwrap();
    let mask = vec![conditioning.cond.mask.clone()];
    let velocity = dit
        .forward(
            golden.require("forward/x").unwrap(),
            golden.require("forward/t").unwrap(),
            &TextBatch {
                states: &conditioning.cond.states,
                mask: &mask,
            },
        )
        .unwrap();
    let (max_abs, peak, mean) = errors(&velocity, golden.require("forward/velocity").unwrap());
    eprintln!(
        "real forward/velocity ({compute:?}): max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e} \
         peak={peak:.3e}"
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

    // No image-level assertion here: at the 256² / 20-step reference geometry the released model
    // renders texture noise in EVERY implementation (upstream FP32, upstream bf16 autocast, this
    // port), a chaotic regime where bf16 rounding flips the trajectory. The trajectory is gated in
    // FP32 by `real_backbone_matches_upstream_in_fp32_on_cpu`, and a real render at a
    // supported size by `provider_renders_a_real_image`.
}

/// The structural gate on the REAL 3B backbone: FP32 on the MLX CPU stream (true f32), fed the
/// oracle's own conditioning, against upstream's FP32 CPU forward — isolating the backbone + pixel
/// head from bf16 and Metal-GEMM rounding.
#[test]
#[ignore = "needs the real weights + IRIS_REAL_GOLDEN (tools/dump_iris_realweight.py)"]
fn real_backbone_matches_upstream_in_fp32_on_cpu() {
    let backbone = env_dir("IRIS_WEIGHTS_DIR");
    let golden = Weights::from_file(env_dir("IRIS_REAL_GOLDEN")).unwrap();
    let config = IrisConfig::from_dir(&backbone).unwrap();
    crate::common::on_cpu(|| {
        let dit = load_backbone(&backbone, &config, Dtype::Float32).unwrap();
        let states = golden
            .require("cond/embeddings")
            .unwrap()
            .expand_dims(0)
            .unwrap();
        let mask = vec![host_i32(golden.require("cond/mask").unwrap())];
        let start = std::time::Instant::now();
        let velocity = dit
            .forward(
                golden.require("forward/x").unwrap(),
                golden.require("forward/t").unwrap(),
                &TextBatch {
                    states: &states,
                    mask: &mask,
                },
            )
            .unwrap();
        mlx_rs::transforms::eval([&velocity]).unwrap();
        eprintln!("cpu fp32 forward: {:.1}s", start.elapsed().as_secs_f32());
        assert_close(
            "real forward/velocity (fp32, cpu, oracle conditioning)",
            &velocity,
            golden.require("forward/velocity").unwrap(),
            1e-4,
        );

        // The whole 20-step CFG-3 generate on the same footing (oracle conditioning + noise).
        let text = |key: &str| TextConditioning {
            states: golden
                .require(&format!("{key}/embeddings"))
                .unwrap()
                .expand_dims(0)
                .unwrap(),
            mask: host_i32(golden.require(&format!("{key}/mask")).unwrap()),
            truncated_tokens: 0,
        };
        let conditioning = Conditioning {
            cond: text("cond"),
            uncond: Some(text("null")),
        };
        let params = GenerationParams {
            steps: golden.metadata("steps").unwrap().parse().unwrap(),
            cfg_scale: golden.metadata("cfg_scale").unwrap().parse().unwrap(),
            negative_prompt: String::new(),
            seed: 0,
            width: 256,
            height: 256,
        };
        let start = std::time::Instant::now();
        let image = denoise(
            &dit,
            &config.flow,
            &conditioning,
            golden.require("generate/noise").unwrap(),
            &params,
            &CancelFlag::new(),
            |_| {},
        )
        .unwrap();
        eprintln!("cpu fp32 generate: {:.1}s", start.elapsed().as_secs_f32());
        // Measured: mean |Δ| 1.6e-4 on [−1, 1]; the max (≈0.1) is a handful of pixels where 20
        // CFG-3 steps amplify f32 summation-order differences. Gate the mean tightly and bound the
        // max.
        let (max_abs, _, mean) = errors(&image, golden.require("generate/image").unwrap());
        eprintln!("real generate/image (fp32, cpu): max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e}");
        assert!(mean <= 1e-3, "mean |Δ| {mean}");
        assert!(max_abs <= 0.25, "max |Δ| {max_abs}");
    });
}

/// The production path end to end: catalog load from the two task resources, default controls
/// (100 steps, CFG 3, empty negative), progress, one PNG.
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR) and IRIS_OUT"]
fn provider_renders_a_real_image() {
    let mut spec = LoadSpec::new(WeightsSource::Dir(env_dir("IRIS_WEIGHTS_DIR")));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(env_dir("IRIS_TEXT_ENCODER_DIR")),
    );
    let size: u32 = std::env::var("IRIS_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let start = std::time::Instant::now();
    let g = mlx_gen_iris::provider_registry()
        .unwrap()
        .load("iris_3b", &spec)
        .unwrap();
    eprintln!("load: {:.1}s", start.elapsed().as_secs_f32());
    let req = GenerationRequest {
        prompt: "a red fox sleeping in fresh snow, golden hour".into(),
        width: size,
        height: size,
        seed: Some(0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let mut last = (0, 0);
    let out = g
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                last = (current, total);
            }
        })
        .unwrap();
    eprintln!(
        "generate {size}² (default 100 steps, CFG 3): {:.1}s",
        start.elapsed().as_secs_f32()
    );
    assert_eq!(last, (100, 100));
    let GenerationOutput::Images(images) = out else {
        panic!("expected images")
    };
    let img = &images[0];
    let path = env_dir("IRIS_OUT").join(format!("iris_3b_{size}.png"));
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .unwrap()
        .save(&path)
        .unwrap();
    let mean = img.pixels.iter().map(|&p| p as f32).sum::<f32>() / img.pixels.len() as f32;
    eprintln!("wrote {} (mean pixel {mean:.1})", path.display());
    assert!(mean > 5.0 && mean < 250.0, "degenerate image (mean {mean})");
}
