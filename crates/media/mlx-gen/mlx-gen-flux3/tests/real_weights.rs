//! Real-weight checks against the pinned `black-forest-labs/flux-3-image-klein` package.
//! `#[ignore]`d: they need ~20 GB of weights and the GPU. Inputs come from the environment (no
//! machine path is baked in):
//!
//! * `FLUX3_PACKAGE_DIR` — the pinned package snapshot root (`config.json`, `model.safetensors`,
//!   `ae.safetensors`, `text_encoder/`).
//! * `FLUX3_REAL_GOLDEN` — the upstream CPU reference from `tools/dump_flux3_realweight.py`
//!   (default: the committed `tests/fixtures/flux3_real_golden_64.safetensors`).
//! * `FLUX3_OUT` — a directory the render writes its PNG + record into (render case only).
//! * `FLUX3_REAL_SIZE` / `FLUX3_REAL_STEPS` / `FLUX3_REAL_PROMPT` — the render geometry
//!   (defaults: the release's 1024², 16 steps, the golden's prompt).
//! * `FLUX3_REAL_COMPUTE=fp32` — run the DiT in FP32 instead of the release's bf16 policy.
//!
//! Run each under an external memory guard:
//! `cargo test -p mlx-gen-flux3 --release --test integration -- --ignored real_weights:: --test-threads 1`

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::gen_core::flux3::{text_ids, Flux3Config, GenerationParams};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};
use mlx_gen_flux3::{denoise_flow_map, load_autoencoder, load_dit, Flux3TextEncoder, SampleTokens};
use mlx_rs::{Array, Dtype};

use crate::common::{assert_close, errors, fixtures, host_f32, host_i32, nchw_to_nhwc};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn package() -> PathBuf {
    env_dir("FLUX3_PACKAGE_DIR")
}

fn golden() -> Weights {
    let path = std::env::var("FLUX3_REAL_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fixtures().join("flux3_real_golden_64.safetensors"));
    Weights::from_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn meta_f32(g: &Weights, key: &str) -> f32 {
    g.metadata(key)
        .unwrap_or_else(|| panic!("golden metadata {key}"))
        .parse()
        .unwrap()
}

fn compute() -> Dtype {
    if std::env::var("FLUX3_REAL_COMPUTE").is_ok_and(|v| v == "fp32") {
        Dtype::Float32
    } else {
        Dtype::Bfloat16
    }
}

fn ids_of(a: &Array) -> Vec<[f32; 4]> {
    host_f32(a)
        .chunks(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

/// The pinned package parses, validates and agrees with the source constants.
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR)"]
fn real_package_config_is_the_release() {
    let config = Flux3Config::from_dir(&package()).unwrap();
    config.validate_supported().unwrap();
    assert_eq!(config.model, "flux-3-image-klein");
    assert_eq!((config.dit.hidden, config.dit.heads), (3072, 24));
    assert_eq!((config.dit.depth, config.dit.depth_single), (5, 28));
    assert!(config.dit.flow_map);
    assert_eq!(config.text_encoder.window_max_length(), 4096);
    assert_eq!(config.sampler.steps, 16);
}

/// The pinned tokenizer + the bf16 tower reproduce upstream's window and conditioning. The window
/// is exact; the tower is bf16 on both sides (MLX GPU vs torch CPU): bf16-ulp scale, 4e-2 of peak.
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR) + the real golden"]
fn real_text_encoder_matches_upstream() {
    let config = Flux3Config::from_dir(&package()).unwrap();
    let g = golden();
    let te = Flux3TextEncoder::load(
        &package().join(&config.text_encoder.path),
        &config.text_encoder,
        config.dit.text_in_channels,
    )
    .unwrap();
    let prompt = g.metadata("prompt").unwrap().to_owned();
    let out = te.encode(&prompt).unwrap();
    assert_eq!(
        out.window.input_ids,
        host_i32(g.require("text/input_ids").unwrap())
    );
    assert_eq!(out.window.mask, host_i32(g.require("text/mask").unwrap()));
    assert_eq!(
        out.window.positions,
        host_i32(g.require("text/positions").unwrap())
    );
    assert_eq!(out.window.length() % 80, 0);
    assert_close(
        "real text/tokens",
        &out.tokens,
        g.require("text/tokens").unwrap(),
        4e-2,
    );
}

/// One flow-map forward of the real DiT on the golden's inputs. Under bf16 the bound is 1.5× the
/// distance upstream's OWN bf16 autocast lands from its f32 forward on the same input (recorded
/// in the golden); under `FLUX3_REAL_COMPUTE=fp32` it is 1e-2 of peak (Metal's f32 GEMM is itself
/// reduced precision, ~1e-3 relative, over 33 blocks).
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR) + the real golden"]
fn real_dit_forward_matches_upstream() {
    let config = Flux3Config::from_dir(&package()).unwrap();
    let g = golden();
    let compute = compute();
    let dit = load_dit(&package().join("model.safetensors"), &config, compute).unwrap();
    let tokens = g
        .require("text/tokens")
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let text_ids = text_ids(tokens.shape()[0] as usize);
    let image_ids = ids_of(g.require("dit/image_ids").unwrap());
    let out = dit
        .forward(&[SampleTokens {
            text: &tokens,
            text_ids: &text_ids,
            image: g.require("dit/x").unwrap(),
            image_ids: &image_ids,
            image_cond: None,
            timestep: host_f32(g.require("dit/t").unwrap())[0],
            target_timestep: Some(host_f32(g.require("dit/t_ref").unwrap())[0]),
        }])
        .unwrap();
    if compute == Dtype::Float32 {
        assert_close(
            "real dit/velocity (fp32)",
            &out,
            g.require("dit/velocity").unwrap(),
            1e-2,
        );
    } else {
        let bound = (1.5 * meta_f32(&g, "dit_bf16_distance")).max(2e-2);
        let (max_abs, peak, mean) = errors(&out, g.require("dit/velocity").unwrap());
        eprintln!("real dit/velocity (bf16): max|d|={max_abs:.3e} mean|d|={mean:.3e} peak={peak:.3e} bound={bound:.3e} of peak");
        assert!(
            max_abs <= bound * peak.max(1.0),
            "bf16 forward outside upstream's own bf16 distance"
        );
        assert_close(
            "real dit/velocity_bf16",
            &out,
            g.require("dit/velocity_bf16").unwrap(),
            bound,
        );
    }
}

/// The real autoencoder decodes the golden's random tokens (4×4) to upstream's f32 image.
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR) + the real golden"]
fn real_autoencoder_decode_matches_upstream() {
    let g = golden();
    let ae = load_autoencoder(&package().join("ae.safetensors")).unwrap();
    assert_eq!(ae.latent_channels(), 128);
    let decoded = ae
        .decode(g.require("ae/tokens_rand").unwrap(), 4, 4)
        .unwrap();
    assert_close(
        "real ae/decoded",
        &decoded,
        &nchw_to_nhwc(g.require("ae/decoded").unwrap()),
        5e-3,
    );
}

/// The 16-step flow-map loop at 64×64 from the golden's noise, with the real text encoder and
/// DiT, against upstream's f32 loop: under bf16 the bound is 1.5× upstream's own bf16-loop
/// distance; the decoded image follows within the same bound.
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR) + the real golden"]
fn real_e2e_64_matches_upstream() {
    let config = Flux3Config::from_dir(&package()).unwrap();
    let g = golden();
    let compute = compute();
    let te = Flux3TextEncoder::load(
        &package().join(&config.text_encoder.path),
        &config.text_encoder,
        config.dit.text_in_channels,
    )
    .unwrap();
    let text = te.encode(g.metadata("prompt").unwrap()).unwrap();
    drop(te);
    let dit = load_dit(&package().join("model.safetensors"), &config, compute).unwrap();
    let schedule = host_f32(g.require("e2e/schedule").unwrap());
    let ids = mlx_gen_flux3::image_ids(4, 4, config.dit.image_rope_scale as f32);
    let tokens = denoise_flow_map(
        &dit,
        &text,
        g.require("e2e/noise").unwrap(),
        &ids,
        &schedule,
        &CancelFlag::new(),
        |_| {},
    )
    .unwrap();
    let bound = if compute == Dtype::Float32 {
        2e-2
    } else {
        (1.5 * meta_f32(&g, "e2e_bf16_distance")).max(5e-2)
    };
    assert_close(
        "real e2e/tokens",
        &tokens,
        g.require("e2e/tokens").unwrap(),
        bound,
    );
    drop(dit);
    let ae = load_autoencoder(&package().join("ae.safetensors")).unwrap();
    let image = ae.decode(&tokens, 4, 4).unwrap();
    assert_close(
        "real e2e/image",
        &image,
        &nchw_to_nhwc(g.require("e2e/image").unwrap()),
        bound,
    );
}

/// The ordinary catalog route on the real package: `provider_registry().load("flux3_klein")` →
/// one text-to-image render at the release geometry → a PNG and a JSON record in `FLUX3_OUT`.
/// Asserts the progress contract and a non-degenerate image (no constant frame).
#[test]
#[ignore = "needs the pinned package (FLUX3_PACKAGE_DIR), FLUX3_OUT and the GPU"]
fn real_distilled_text_to_image_renders() {
    let out_dir = env_dir("FLUX3_OUT");
    std::fs::create_dir_all(&out_dir).unwrap();
    let size: u32 = std::env::var("FLUX3_REAL_SIZE")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(1024);
    let steps: u32 = std::env::var("FLUX3_REAL_STEPS")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(16);
    let prompt = std::env::var("FLUX3_REAL_PROMPT").unwrap_or_else(|_| {
        "a red fox sitting in fresh snow at golden hour, soft light, photograph".to_owned()
    });
    let mut spec = LoadSpec::new(WeightsSource::Dir(package()));
    if compute() == Dtype::Float32 {
        spec.precision = mlx_gen::Precision::Fp32;
    }
    let config = Flux3Config::from_dir(&package()).unwrap();
    let request = GenerationRequest {
        prompt: prompt.clone(),
        width: size,
        height: size,
        steps: Some(steps),
        seed: Some(0),
        ..Default::default()
    };
    let params = GenerationParams::resolve(&request, 0, &config).unwrap();
    let t_load = Instant::now();
    let g = mlx_gen_flux3::provider_registry()
        .unwrap()
        .load("flux3_klein", &spec)
        .unwrap();
    let load_seconds = t_load.elapsed().as_secs_f64();
    let mut events = Vec::new();
    let t_render = Instant::now();
    let out = g
        .generate(&request, &mut |p| events.push(format!("{p:?}")))
        .unwrap();
    let render_seconds = t_render.elapsed().as_secs_f64();
    let expected: Vec<String> = (1..=steps)
        .map(|i| format!("Step {{ current: {i}, total: {steps} }}"))
        .chain(std::iter::once("Decoding".to_owned()))
        .collect();
    assert_eq!(events, expected);
    let GenerationOutput::Images(images) = out else {
        panic!("expected images")
    };
    assert_eq!(images.len(), 1);
    let image = &images[0];
    assert_eq!((image.width, image.height), (size, size));
    let n = image.pixels.len() as f64;
    let mean = image.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = image
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    assert!(
        var.sqrt() > 10.0,
        "a constant frame is not a render (std {})",
        var.sqrt()
    );
    let png = out_dir.join(format!("flux3_klein_{size}_{steps}steps_seed0.png"));
    image::save_buffer(
        &png,
        &image.pixels,
        image.width,
        image.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    let record = serde_json::json!({
        "model": "flux3_klein",
        "package_revision": mlx_gen::gen_core::flux3::DISTILLED_WEIGHTS_REVISION,
        "upstream_commit": mlx_gen::gen_core::flux3::UPSTREAM_CODE_REVISION,
        "prompt": prompt,
        "width": size,
        "height": size,
        "steps": params.steps,
        "alpha": params.alpha,
        "seed": params.seed,
        "guidance": serde_json::Value::Null,
        "compute": format!("{:?}", compute()),
        "load_seconds": load_seconds,
        "render_seconds": render_seconds,
        "pixel_std": var.sqrt(),
        "png": png.file_name().unwrap().to_string_lossy(),
    });
    std::fs::write(
        out_dir.join("flux3_klein_render.json"),
        serde_json::to_string_pretty(&record).unwrap() + "\n",
    )
    .unwrap();
    eprintln!(
        "rendered {} in {render_seconds:.1}s (load {load_seconds:.1}s)",
        png.display()
    );
    let _ = Progress::Decoding;
}
