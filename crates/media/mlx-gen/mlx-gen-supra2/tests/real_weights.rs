//! Real-weight Supra2-IMG parity and renders (sc-24213). `#[ignore]`d; every weight input is an
//! explicit path (nothing is downloaded or derived from a cache location):
//!
//! ```sh
//! SUPRA2_BUNDLE=<converted bundle dir>            # gen_core::supra2::convert_pinned_checkpoint output
//! SUPRA2_T5_SNAPSHOT=<google/flan-t5-base @ 7bcac572…>
//! SUPRA2_VAE_SNAPSHOT=<stabilityai/sd-vae-ft-mse @ 31f26fde…>
//! SUPRA2_OUTPUT_DIR=<where PNGs are written>      # optional
//! SUPRA2_TEST_DEVICE=cpu                          # optional: exact-f32 structural run on the MLX CPU device
//! cargo test --release -p mlx-gen-supra2 --test integration real_weights:: -- --ignored --nocapture
//! ```
//!
//! The reference is the committed `candle-gen-supra2/tests/fixtures/supra2_real_golden.safetensors`
//! (shared with the Candle twin): `tools/dump_supra2_reference.py real`'s torch CPU float32 runs of
//! the frozen upstream `inference.py` classes fed the identical tensors (token ids, context, initial
//! noise), repacked verbatim by the same tool's `pack` mode under `t5.` / `dit.` / `vae.` / `traj.`
//! prefixes — so RNG differences never masquerade as model error, and a runner without a torch
//! environment (the `mlx-supra2` real-weights lane) holds MLX to the same numbers.

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_gen::gen_core::supra2::{
    image_seed, initial_noise, PROVIDER_ID, TEXT_ENCODER_COMPONENT, VAE_COMPONENT,
};
use mlx_gen::gen_core::ProviderRegistryBuilder;
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_supra2::{Supra2Image, SupraDit};

use crate::common::{diff_stats, host};

fn on_cpu() -> bool {
    std::env::var("SUPRA2_TEST_DEVICE").as_deref() == Ok("cpu")
}

/// `SUPRA2_TEST_DEVICE=cpu` runs on the MLX CPU device (exact-f32 structural check); default GPU.
fn select_device() {
    if on_cpu() {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    }
}

/// Declared `(max_rel, mean_rel)` bound (relative to the reference abs-max) for the selected device.
/// CPU is exact f32 — the structural gate, measured ≤ 1e-4 relative at every stage. Metal's f32
/// GEMM accumulates in reduced precision, so its bounds carry that measured floor (max_rel: T5
/// 8.4e-3, DiT 4.4e-3, VAE 7.9e-2; mean_rel ≤ 1.4e-3).
fn tol(cpu: (f32, f32), gpu: (f32, f32)) -> (f32, f32) {
    if on_cpu() {
        cpu
    } else {
        gpu
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

/// The committed torch golden (see the module docs), keyed `t5.*` / `dit.*` / `vae.*` / `traj.*`.
fn golden() -> Weights {
    Weights::from_file(
        Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../candle-gen/candle-gen-supra2/tests/fixtures/supra2_real_golden.safetensors",
        ),
    )
    .unwrap()
}

fn tokenizer_cases() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contracts/gen-core/tests/fixtures/supra2/tokenizer_cases.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn report(stage: &str, got: &[f32], want: &[f32], max_rel: f32, mean_rel: f32) {
    let (max_abs, mean_abs, ref_max) = diff_stats(got, want);
    eprintln!(
        "PARITY {stage}: max_abs {max_abs:.3e} mean_abs {mean_abs:.3e} ref_absmax {ref_max:.3e} \
         (max_rel {:.3e} ≤ {max_rel:.0e}, mean_rel {:.3e} ≤ {mean_rel:.0e})",
        max_abs / ref_max,
        mean_abs / ref_max
    );
    assert!(
        max_abs <= max_rel * ref_max,
        "{stage}: max_abs {max_abs} ref {ref_max}"
    );
    assert!(
        mean_abs <= mean_rel * ref_max,
        "{stage}: mean_abs {mean_abs} ref {ref_max}"
    );
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
    eprintln!("PIXELS {stage}: mean |Δ| {mean_abs:.3} / 255 (≤ {max_mean}), max |Δ| {max_abs}");
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

/// Text encoder, stored conditioning, one CFG DiT evaluation and the VAE decode, each fed the
/// reference's own inputs.
#[test]
#[ignore = "needs real Supra2 weights"]
fn components_match_torch_reference() {
    select_device();
    let model = Supra2Image::load(&spec_for(&env_path("SUPRA2_BUNDLE"))).unwrap();
    let g = golden();
    let cases = tokenizer_cases();
    let t5_tol = tol((2e-5, 2e-6), (2e-2, 3e-3));
    let dit_tol = tol((2e-5, 2e-6), (1e-2, 2e-3));
    let vae_tol = tol((2e-4, 2e-5), (1.5e-1, 3e-3));
    for (name, case) in cases.as_object().unwrap() {
        let (got, tokens) = model
            .encode_prompt(case["prompt"].as_str().unwrap())
            .unwrap();
        assert_eq!(
            got.mask,
            host(g.require(&format!("t5.mask.{name}")).unwrap()),
            "{name}"
        );
        assert_eq!(tokens.truncated, name == "long");
        // Flan-T5 f32, 12 layers.
        report(
            &format!("t5.{name}"),
            &host(&got.context),
            &host(g.require(&format!("t5.ctx.{name}")).unwrap()),
            t5_tol.0,
            t5_tol.1,
        );
    }
    let stored = model
        .stored_unconditional()
        .expect("pinned checkpoint stores uncond");
    assert_eq!(
        host(&stored.context),
        host(g.require("t5.uncond_text").unwrap())
    );
    assert_eq!(stored.mask, host(g.require("t5.uncond_mask").unwrap()));

    let z = g.require("dit.z").unwrap();
    let t = host(g.require("dit.t").unwrap())[0];
    let ctx = mlx_rs::ops::concatenate_axis(
        &[
            &g.require("t5.ctx.short").unwrap().expand_dims(0).unwrap(),
            &stored.context,
        ],
        0,
    )
    .unwrap();
    let mut mask01 = host(g.require("t5.mask.short").unwrap());
    mask01.extend(stored.mask.iter());
    let mask = SupraDit::additive_mask(&mask01, 2, 128).unwrap();
    let z2 = mlx_rs::ops::concatenate_axis(&[z, z], 0).unwrap();
    let dit = model.denoiser();
    let v = dit
        .forward(&z2, &dit.timestep_features(&[t, t]), &ctx, &mask)
        .unwrap();
    let v = host(&v);
    let half = v.len() / 2;
    let v_cond = host(g.require("dit.v_cond").unwrap());
    let v_uncond = host(g.require("dit.v_uncond").unwrap());
    report("dit.v_cond", &v[..half], &v_cond, dit_tol.0, dit_tol.1);
    report("dit.v_uncond", &v[half..], &v_uncond, dit_tol.0, dit_tol.1);

    // `latent` is a denoised (unscaled) Supra2 latent; the VAE owns the `/ 0.18215`.
    for (stage, latent, decoded) in [
        ("vae.decode", "vae.latent", "vae.decoded"),
        (
            "vae.decode_random",
            "vae.random_latent",
            "vae.random_decoded",
        ),
    ] {
        let latent = g
            .require(latent)
            .unwrap()
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap();
        let got = model
            .vae()
            .decode(&latent)
            .unwrap()
            .transpose_axes(&[0, 3, 1, 2])
            .unwrap();
        let want = host(g.require(decoded).unwrap());
        report(stage, &host(&got), &want, vae_tol.0, vae_tol.1);
    }
}

/// Whole-pipeline trajectories from the shared seed-0 initial noise (2 images): stored-uncond CFG
/// at 8 and 50 steps, fallback-uncond CFG, and guidance off — latents and decoded pixels vs torch.
#[test]
#[ignore = "needs real Supra2 weights"]
fn trajectories_match_torch_reference() {
    select_device();
    let bundle = env_path("SUPRA2_BUNDLE");
    let g = golden();
    let z0 = host(g.require("traj.z0").unwrap());
    let ours: Vec<f32> = (0..2)
        .flat_map(|i| initial_noise(image_seed(0, i), 4096))
        .collect();
    assert_eq!(
        z0, ours,
        "reference z0 must be gen_core::supra2::initial_noise(seed 0, images 0..2)"
    );

    let stored = Supra2Image::load(&spec_for(&bundle)).unwrap();
    // Upstream's empty-string fallback on the real checkpoint (the pinned bundle always stores an
    // unconditional embedding, so only the test seam can drop it).
    let fallback = Supra2Image::load(&spec_for(&bundle))
        .unwrap()
        .without_stored_unconditional();
    assert!(stored.stored_unconditional().is_some());
    assert!(fallback.stored_unconditional().is_none());
    let prompt = "a sea jellyfish floating in the pitch-black ocean depths";
    // Latent `(max_rel, mean_rel)` and mean |Δpixel| (of 255) bounds. CPU measured ≤ 5.9e-6 /
    // 2.2e-7 with max |Δpixel| 1; Metal measured ≤ 3.2e-2 / 5.8e-4 with mean |Δpixel| ≤ 0.24.
    let (max_rel, mean_rel) = tol((5e-5, 5e-6), (6e-2, 2e-3));
    let max_px = if on_cpu() { 0.01 } else { 0.5 };
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
            &host(g.require(&format!("traj.{name}.latent")).unwrap()),
            max_rel,
            mean_rel,
        );
        pixel_report(
            name,
            &out.images,
            &host(g.require(&format!("traj.{name}.pixels")).unwrap()),
            max_px,
        );
        save(&out.images, &format!("mlx_{name}"));
    }
}

/// The SceneWorks worker path: an explicit registry composed from `register_providers`, loaded by
/// engine id, default request (seed 0, CFG 3.0, 50 steps), two individual images.
#[test]
#[ignore = "needs real Supra2 weights"]
fn registry_renders_published_defaults() {
    select_device();
    let registry = mlx_gen_supra2::register_providers(ProviderRegistryBuilder::new())
        .build()
        .unwrap();
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
            if matches!(p, mlx_gen::Progress::Step { .. }) {
                steps += 1;
            }
        })
        .unwrap()
    else {
        panic!("expected images");
    };
    eprintln!(
        "registry render: load {load_s:.2}s, generate {:.2}s for 2 images × 50 steps",
        start.elapsed().as_secs_f32()
    );
    assert_eq!(steps, 50);
    assert_eq!(images.len(), 2);
    assert!(images.iter().all(|i| i.width == 256 && i.height == 256));
    assert_ne!(
        images[0].pixels, images[1].pixels,
        "seed + index gives distinct images"
    );
    save(&images, "registry_seed0_cfg3_steps50");

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
        cancel: mlx_gen::CancelFlag::new(),
        ..request.clone()
    };
    cancelled.cancel.cancel();
    assert!(matches!(
        generator.generate(&cancelled, &mut |_| {}),
        Err(mlx_gen::gen_core::Error::Canceled)
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
