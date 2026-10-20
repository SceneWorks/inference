//! Real-weight Supra2-IMG parity and renders on Candle (sc-24214). `#[ignore]`d; every weight input
//! is an explicit path (nothing is downloaded or derived from a cache location):
//!
//! ```sh
//! SUPRA2_BUNDLE=<converted bundle dir>            # gen_core::supra2::convert_pinned_checkpoint output
//! SUPRA2_T5_SNAPSHOT=<google/flan-t5-base @ 7bcac572…>
//! SUPRA2_VAE_SNAPSHOT=<stabilityai/sd-vae-ft-mse @ 31f26fde…>
//! SUPRA2_OUTPUT_DIR=<where PNGs are written>      # optional
//! SUPRA2_TEST_DEVICE=cpu                          # optional: force the CPU on an accelerator build
//! cargo test --release -p candle-gen-supra2 [--features cuda] --test integration real_weights:: -- --ignored --nocapture
//! ```
//!
//! The reference is the committed `tests/fixtures/supra2_real_golden.safetensors`: the MLX twin's
//! torch CPU float32 dump (`mlx-gen/tools/dump_supra2_reference.py real`, frozen upstream
//! `inference.py` classes fed identical token ids, contexts and initial noise) repacked verbatim by
//! the same tool's `pack` mode, so runners without a torch environment hold Candle to the same
//! numbers MLX was held to.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::supra2::{
    image_seed, initial_noise, BundleManifest, BUNDLE_MANIFEST, DENOISER_FILE, PROVIDER_ID,
    TEXT_ENCODER_COMPONENT, VAE_COMPONENT,
};
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};
use candle_gen_supra2::Supra2Image;

use crate::support::{diff_stats, host, report};

/// The device under test: the build's default (`cuda:0` under `--features cuda`), or the CPU when
/// `SUPRA2_TEST_DEVICE=cpu`.
fn device() -> Device {
    if std::env::var("SUPRA2_TEST_DEVICE").as_deref() == Ok("cpu") {
        Device::Cpu
    } else {
        candle_gen::default_device().unwrap()
    }
}

/// Declared `(max_rel, mean_rel)` bound (relative to the reference abs-max).
///
/// * CPU — exact f32, the structural gate; the MLX CPU device's bounds, which Candle CPU meets with
///   the same margin.
/// * CUDA — f32 end to end with candle's reduced-precision (TF32) GEMM left **off**, so the gap is
///   reduction order (cuBLAS tiling, device softmax / norm reductions), compounded over 12 T5
///   layers, 14 DiT blocks, up to 50 Euler steps and the VAE's 30+ convolutions. Declared at the
///   bound in each call; the measured values are printed by every `PARITY` line.
fn tol(device: &Device, cpu: (f32, f32), cuda: (f32, f32)) -> (f32, f32) {
    if device.is_cpu() {
        cpu
    } else {
        cuda
    }
}

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

fn spec_for(bundle: &Path) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(bundle.to_path_buf()))
        .with_component(
            TEXT_ENCODER_COMPONENT,
            WeightsSource::Dir(env_path("SUPRA2_T5_SNAPSHOT")),
        )
        .with_component(
            VAE_COMPONENT,
            WeightsSource::Dir(env_path("SUPRA2_VAE_SNAPSHOT")),
        )
}

fn golden() -> HashMap<String, Tensor> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/supra2_real_golden.safetensors");
    candle_gen::candle_core::safetensors::load(path, &Device::Cpu).unwrap()
}

fn tokenizer_cases() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contracts/gen-core/tests/fixtures/supra2/tokenizer_cases.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn pixel_report(stage: &str, images: &[Image], torch_nchw: &[f32], max_mean: f32) {
    let mut got = Vec::new();
    for img in images {
        // HWC → CHW to match the torch tensor.
        let (w, h) = (img.width as usize, img.height as usize);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    got.push(img.pixels[(y * w + x) * 3 + c] as f32);
                }
            }
        }
    }
    let (max_abs, mean_abs, _) = diff_stats(&got, torch_nchw);
    eprintln!("PIXELS {stage}: mean |d| {mean_abs:.4} / 255 (<= {max_mean}), max |d| {max_abs}");
    assert!(mean_abs <= max_mean, "{stage}: mean pixel diff {mean_abs}");
}

fn save(images: &[Image], stem: &str) {
    let Ok(dir) = std::env::var("SUPRA2_OUTPUT_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (i, img) in images.iter().enumerate() {
        let path = dir.join(format!("{stem}_{i}.png"));
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .unwrap()
            .save(&path)
            .unwrap();
        eprintln!("wrote {}", path.display());
    }
}

/// Text encoder (every tokenizer case, incl. the truncated one), stored conditioning, one CFG DiT
/// evaluation and the VAE decode, each fed the reference's own inputs.
#[test]
#[ignore = "needs real Supra2 weights"]
fn components_match_torch_reference() {
    let device = device();
    eprintln!("device: {device:?}");
    let model =
        Supra2Image::load_on_device(&spec_for(&env_path("SUPRA2_BUNDLE")), device.clone()).unwrap();
    let g = golden();
    let t5_tol = tol(&device, (2e-5, 2e-6), (1e-3, 1e-4));
    let dit_tol = tol(&device, (2e-5, 2e-6), (1e-3, 1e-4));
    let vae_tol = tol(&device, (2e-4, 2e-5), (5e-3, 5e-4));
    let cases = tokenizer_cases();
    let ctx_dim = model.config().ctx_dim;
    for (name, case) in cases.as_object().unwrap() {
        let (got, tokens) = model
            .encode_prompt(case["prompt"].as_str().unwrap())
            .unwrap();
        let want_mask = host(&g[&format!("t5.mask.{name}")]);
        assert_eq!(got.mask, want_mask, "{name}");
        assert_eq!(tokens.truncated, name == "long");
        let real = want_mask.iter().filter(|&&m| m == 1.0).count();
        let ours = host(&got.context);
        let want = host(&g[&format!("t5.ctx.{name}")]);
        // Flan-T5 f32, 12 layers: every real-token row against torch's masked full-length encode.
        report(
            &format!("t5.{name} ({real} real tokens)"),
            &ours[..real * ctx_dim],
            &want[..real * ctx_dim],
            t5_tol,
        );
        // Padded rows are written as zeros (inert under the DiT's key mask).
        assert!(ours[real * ctx_dim..].iter().all(|&v| v == 0.0), "{name}");
    }
    let stored = model
        .stored_unconditional()
        .expect("pinned checkpoint stores uncond");
    assert_eq!(host(&stored.context), host(&g["t5.uncond_text"]));
    assert_eq!(stored.mask, host(&g["t5.uncond_mask"]));

    // One batched CFG evaluation at t = 0.3 with the reference's own context.
    let dit = model.denoiser();
    let z = g["dit.z"].to_device(&device).unwrap();
    let t = host(&g["dit.t"])[0];
    let ctx = Tensor::cat(
        &[
            &g["t5.ctx.short"]
                .unsqueeze(0)
                .unwrap()
                .to_device(&device)
                .unwrap(),
            &stored.context,
        ],
        0,
    )
    .unwrap();
    let mut mask01 = host(&g["t5.mask.short"]);
    mask01.extend(stored.mask.iter());
    let mask = candle_gen_supra2::SupraDit::additive_mask(&mask01, 2, 128, &device).unwrap();
    let z2 = Tensor::cat(&[&z, &z], 0).unwrap();
    let v = host(
        &dit.forward(&z2, &dit.timestep_features(&[t, t]).unwrap(), &ctx, &mask)
            .unwrap(),
    );
    let half = v.len() / 2;
    report("dit.v_cond", &v[..half], &host(&g["dit.v_cond"]), dit_tol);
    report(
        "dit.v_uncond",
        &v[half..],
        &host(&g["dit.v_uncond"]),
        dit_tol,
    );

    // `latent` is a denoised (unscaled) Supra2 latent; `decode_latent` owns the `/ 0.18215`.
    for (stage, latent, decoded) in [
        ("vae.decode", "vae.latent", "vae.decoded"),
        (
            "vae.decode_random",
            "vae.random_latent",
            "vae.random_decoded",
        ),
    ] {
        let got = model
            .decode_latent(&g[latent].to_device(&device).unwrap())
            .unwrap();
        report(stage, &host(&got), &host(&g[decoded]), vae_tol);
    }
}

/// A bundle view without the stored conditioning, to exercise upstream's empty-string fallback on
/// the real checkpoint: the denoiser is hard-linked (copied if the filesystem refuses) and the
/// manifest drops its conditioning entry.
fn fallback_bundle(bundle: &Path) -> tempfile::TempDir {
    let dir = tempfile::Builder::new()
        .prefix(&format!("supra2-fallback-{}-", std::process::id()))
        .tempdir()
        .unwrap();
    let mut manifest: BundleManifest =
        serde_json::from_slice(&std::fs::read(bundle.join(BUNDLE_MANIFEST)).unwrap()).unwrap();
    manifest.conditioning = None;
    let (from, to) = (bundle.join(DENOISER_FILE), dir.path().join(DENOISER_FILE));
    if std::fs::hard_link(&from, &to).is_err() {
        std::fs::copy(&from, &to).unwrap();
    }
    std::fs::write(
        dir.path().join(BUNDLE_MANIFEST),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    dir
}

/// Whole-pipeline trajectories from the shared seed-0 initial noise (2 images): stored-uncond CFG
/// at 8 and 50 steps, fallback-uncond CFG, and guidance off — latents and decoded pixels vs torch.
#[test]
#[ignore = "needs real Supra2 weights"]
fn trajectories_match_torch_reference() {
    let device = device();
    eprintln!("device: {device:?}");
    let bundle = env_path("SUPRA2_BUNDLE");
    let g = golden();
    let z0 = host(&g["traj.z0"]);
    let ours: Vec<f32> = (0..2)
        .flat_map(|i| initial_noise(image_seed(0, i), 4096))
        .collect();
    assert_eq!(
        z0, ours,
        "reference z0 must be gen_core::supra2::initial_noise(seed 0, images 0..2)"
    );

    let stored = Supra2Image::load_on_device(&spec_for(&bundle), device.clone()).unwrap();
    let fb_dir = fallback_bundle(&bundle);
    let fallback = Supra2Image::load_on_device(&spec_for(fb_dir.path()), device.clone()).unwrap();
    assert!(fallback.stored_unconditional().is_none());
    let prompt = "a sea jellyfish floating in the pitch-black ocean depths";
    let latent_tol = tol(&device, (5e-5, 5e-6), (5e-3, 5e-4));
    let max_px = if device.is_cpu() { 0.01 } else { 0.1 };
    let cases: [(&str, &Supra2Image, u32, f32); 4] = [
        ("cfg3_steps8_stored", &stored, 8, 3.0),
        ("cfg3_steps8_fallback", &fallback, 8, 3.0),
        ("cfg1_steps8", &stored, 8, 1.0),
        ("cfg3_steps50_stored", &stored, 50, 3.0),
    ];
    for (name, model, steps, guidance) in cases {
        let request = GenerationRequest {
            prompt: prompt.into(),
            width: 256,
            height: 256,
            count: 2,
            steps: Some(steps),
            guidance: Some(guidance),
            seed: Some(0),
            ..Default::default()
        };
        let start = Instant::now();
        let out = model
            .generate_with(&request, Some(&z0), &mut |_| {})
            .unwrap();
        eprintln!("{name}: {:.2}s", start.elapsed().as_secs_f32());
        assert_eq!(
            out.used_stored_unconditional,
            guidance > 1.0 && name.ends_with("stored")
        );
        report(
            &format!("latent.{name}"),
            &out.latents,
            &host(&g[&format!("traj.{name}.latent")]),
            latent_tol,
        );
        pixel_report(
            name,
            &out.images,
            &host(&g[&format!("traj.{name}.pixels")]),
            max_px,
        );
        save(&out.images, &format!("candle_{name}"));
    }
}

/// The SceneWorks worker path: an explicit registry composed from `register_providers`, loaded by
/// engine id, default request (seed 0, CFG 3.0, 50 steps), two individual images — on the build's
/// default device (`cuda:0` under `--features cuda`).
#[test]
#[ignore = "needs real Supra2 weights"]
fn registry_renders_published_defaults() {
    let registry = candle_gen_supra2::provider_registry().unwrap();
    let start = Instant::now();
    let generator = registry
        .load(PROVIDER_ID, &spec_for(&env_path("SUPRA2_BUNDLE")))
        .unwrap();
    let load_s = start.elapsed().as_secs_f32();
    let request = GenerationRequest {
        prompt: "a sea jellyfish floating in the pitch-black ocean depths".into(),
        width: 256,
        height: 256,
        count: 2,
        ..Default::default()
    };
    generator.validate(&request).unwrap();
    let mut steps = 0;
    let start = Instant::now();
    let GenerationOutput::Images(images) = generator
        .generate(&request, &mut |p| {
            if matches!(p, Progress::Step { .. }) {
                steps += 1;
            }
        })
        .unwrap()
    else {
        panic!("expected images");
    };
    eprintln!(
        "registry render: load {load_s:.2}s, generate {:.2}s for 2 images x 50 steps",
        start.elapsed().as_secs_f32()
    );
    assert_eq!(steps, 50);
    assert_eq!(images.len(), 2);
    assert!(images.iter().all(|i| i.width == 256 && i.height == 256));
    assert_ne!(
        images[0].pixels, images[1].pixels,
        "seed + index gives distinct images"
    );
    save(&images, "candle_registry_seed0_cfg3_steps50");

    // Repeatable seeds: the same request reproduces image 0 bit-for-bit.
    let again = match generator
        .generate(
            &GenerationRequest {
                count: 1,
                ..request.clone()
            },
            &mut |_| {},
        )
        .unwrap()
    {
        GenerationOutput::Images(images) => images,
        _ => panic!("expected images"),
    };
    assert_eq!(again[0].pixels, images[0].pixels);

    // Cancellation surfaces as the typed error.
    // (A fresh flag: a cloned request shares its original's.)
    let cancelled = GenerationRequest {
        cancel: candle_gen::gen_core::CancelFlag::new(),
        ..request.clone()
    };
    cancelled.cancel.cancel();
    assert!(matches!(
        generator.generate(&cancelled, &mut |_| {}),
        Err(candle_gen::gen_core::Error::Canceled)
    ));

    // Off-interface requests are refused rather than silently coerced.
    for bad in [
        GenerationRequest {
            width: 512,
            height: 512,
            ..request.clone()
        },
        GenerationRequest {
            negative_prompt: Some("blurry".into()),
            ..request.clone()
        },
    ] {
        assert!(generator.validate(&bad).is_err());
    }
}
