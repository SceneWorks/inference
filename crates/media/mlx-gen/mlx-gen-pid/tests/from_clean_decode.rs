//! sc-7843 e2e: real `from_clean` PiD decodes in MLX — `CaptionEncoder` + `PidDecoder.decode` on the
//! real `clean_latent`s (QwenImage_VAE_2d encode, dumped by `tools/dump_pid_clean_latent.py`) for each
//! runA sample. Saves the SR PNG for a visual/coherence check against the CUDA reference
//! `*__pid_4step__4096.png`.
//!
//! `#[ignore]`d (needs the converted qwenimage checkpoint + gemma-2-2b-it + the clean_latent dumps).
//! Decodes the small (512→2048²) latent by default; set `PID_DECODE_NATIVE=1` for the full 1024→4096².
//!
//! ```sh
//! cargo test -p mlx-gen-pid --release --test integration from_clean_decode:: -- --ignored --nocapture
//! ```

use mlx_gen::decoder::LatentDecoder;
use mlx_gen::testing::{env_or, save_nchw_signed_unit_png};
use mlx_gen::weights::Weights;
use mlx_gen_pid::{
    CaptionEncoder, Gemma2, Gemma2Config, PidConfig, PidDecoder, PidNet, Sampler, SamplerConfig,
};
use mlx_rs::ops::{max, mean, min};
use mlx_rs::Dtype;

/// Each runA `from_clean` sample: name (→ `clean_latent_<name>.safetensors` + `mlx_<name>_*.png`) and
/// the **PiD decode caption** (the short text condition fed to PiD, NOT the longer Qwen-Image
/// generation prompt that produced the input image).
const SAMPLES: &[(&str, &str)] = &[
    (
        "landscape",
        "a mountain valley landscape at golden hour with a winding river and pine forest",
    ),
    (
        "portrait",
        "a close-up portrait photograph of an elderly fisherman, weathered skin, detailed eyes and beard",
    ),
    (
        "text_storefront",
        "a vintage bookstore storefront, hand-painted sign reading CORNER BOOKS, RARE & USED, red brick",
    ),
];

fn gemma_snapshot() -> String {
    env_or("PID_GEMMA_DIR", &{
        let home = std::env::var("MLX_GEN_MODELS_ROOT").expect("set MLX_GEN_MODELS_ROOT to the explicit models root (holds models--*/snapshots); inference never self-fetches or derives a cache location (epic 13657)");
        let base = format!("{home}/models--Efficient-Large-Model--gemma-2-2b-it/snapshots");
        std::fs::read_dir(&base)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|d| d.is_dir())
            .unwrap()
            .to_string_lossy()
            .into_owned()
    })
}

#[test]
#[ignore = "needs qwenimage ckpt + gemma + clean_latent dumps"]
fn from_clean_runa() {
    // --- checkpoint + gemma caption encoder, loaded once (Array handles are refcounted, so
    //     rebuilding PidNet per sample below is cheap) ---
    let ckpt = env_or(
        "PID_QWEN_SAFETENSORS",
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pid/qwenimage_2kto4k.safetensors"
        ),
    );
    let w = Weights::from_file(&ckpt).unwrap();

    let snap = gemma_snapshot();
    let gw = Weights::from_file(format!("{snap}/gemma-2-2b-it.safetensors")).unwrap();
    let gemma = Gemma2::from_weights(&gw, "model.", &Gemma2Config::gemma_2_2b()).unwrap();
    let enc = CaptionEncoder::new(gemma, format!("{snap}/tokenizer.json")).unwrap();

    let key = if std::env::var("PID_DECODE_NATIVE").is_ok() {
        "clean_latent_native"
    } else {
        "clean_latent_small"
    };
    // Restrict to one sample with PID_SAMPLE=<name> (default: all three).
    let only = std::env::var("PID_SAMPLE").ok();

    for (name, caption) in SAMPLES {
        if only.as_deref().is_some_and(|o| o != *name) {
            continue;
        }
        let latent_file = format!(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../tools/golden/pid/clean_latent_{}.safetensors"
            ),
            name
        );
        let latents = Weights::from_file(&latent_file).unwrap();
        let latent = latents
            .require(key)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        // bf16 decode — the reference's inference dtype + the dtype the LQ-adapter convs expect.
        let caption_embs = enc
            .encode(caption)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let decoder = PidDecoder::new(
            PidNet::from_weights(&w, "", &PidConfig::sr4x()).unwrap(),
            Sampler::new(&SamplerConfig::distill_4step()),
            caption_embs,
            0.0,  // degrade σ (clean-latent decode)
            4,    // scale
            8,    // vae_compression
            1234, // seed
        );
        let (th, tw) = decoder.target_hw(&latent);
        eprintln!(
            "[{name}] clean_latent[{key}] {:?} -> {th}x{tw} ...",
            latent.shape()
        );
        let out = decoder.decode(&latent).unwrap();
        assert_eq!(out.shape()[2], th);
        assert_eq!(out.shape()[3], tw);

        let lo = min(&out, None).unwrap().item::<f32>();
        let hi = max(&out, None).unwrap().item::<f32>();
        let mu = mean(&out, None).unwrap().item::<f32>();
        eprintln!(
            "[{name}] decoded {:?} min={lo:.3} max={hi:.3} mean={mu:.3}",
            out.shape()
        );
        assert!(lo.is_finite() && hi.is_finite(), "non-finite output");
        assert!(lo >= -1.001 && hi <= 1.001, "out of [-1,1]");
        assert!(hi - lo > 0.2, "degenerate (flat) output");

        let png = format!(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../tools/golden/pid/mlx_{}_{}.png"
            ),
            name, th
        );
        save_nchw_signed_unit_png(&out, &png);
        eprintln!("[{name}] wrote {png}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED from_clean_decode::from_clean_runa");
}
