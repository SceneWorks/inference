//! sc-7843 Run B (`from_ldm`) self-consistent validation — the PiD decode half.
//!
//! Reads the captured Qwen-Image trajectory latents dumped by `mlx-gen-qwen-image`'s
//! `dump_runb_latents` test (`tools/golden/pid/runb_<label>.safetensors`: the unpacked latent, its
//! flow-match σ, and the Qwen-VAE 1024² decode), and for each one runs the PiD 4-step decode to
//! 4096² with that σ fed to the sigma-aware LQ gate (the whole point of `from_ldm`: PiD decoding a
//! *partially-denoised* captured `x_t`, not just a clean VAE latent). Saves, per label, the Qwen-VAE
//! 1024² baseline and the PiD 4096² — the "fair pair" for Run B (same latent, two decoders).
//!
//! This is a *self-consistent* MLX reproduction, NOT a parity check against the CUDA reference PNGs:
//! the latents come from an MLX Qwen-Image generation (MLX PRNG ≠ torch), so the night-market image
//! differs from the reference. It validates the `from_ldm` σ-aware decode path end-to-end.
//!
//! `#[ignore]`d (needs the PiD qwenimage ckpt + gemma + the `dump_runb_latents` outputs). Run:
//! ```sh
//! cargo test -p mlx-gen-pid --release --test from_ldm_decode -- --ignored --nocapture
//! ```

use mlx_gen::decoder::LatentDecoder;
use mlx_gen::weights::Weights;
use mlx_gen_pid::{
    CaptionEncoder, Gemma2, Gemma2Config, PidConfig, PidDecoder, PidNet, Sampler, SamplerConfig,
};
use mlx_rs::ops::{max, mean, min};
use mlx_rs::{Array, Dtype};

/// Run B drives the LDM generation AND the PiD caption from the same single prompt.
const CAPTION: &str = "a highly detailed photograph of a bustling night market food stall, glowing paper lanterns, a neon sign reading OPEN, steam rising from a wok, fresh vegetables, reflections on wet cobblestones, shallow depth of field, 35mm";

/// Captured trajectory steps (+ final x0), matching `--save_xt_steps 44 46 48` over 50 LDM steps.
const LABELS: &[&str] = &["44xt", "46xt", "48xt", "x0"];

fn env_or(name: &str, default: String) -> String {
    std::env::var(name).unwrap_or(default)
}

fn gemma_snapshot() -> String {
    env_or("PID_GEMMA_DIR", {
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

/// `[1,3,H,W]` in [-1,1] → RGB8 PNG. (`as_slice` is physical-order, so reshape after the transpose
/// to force a logical-order copy — see `from_clean_decode::save_png`.)
fn save_png(out: &Array, path: &str) {
    let sh = out.shape();
    let (h, w) = (sh[2], sh[3]);
    let hwc = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[3, h, w])
        .unwrap()
        .transpose_axes(&[1, 2, 0])
        .unwrap()
        .reshape(&[h * w * 3])
        .unwrap();
    let buf: Vec<u8> = hwc
        .as_slice::<f32>()
        .iter()
        .map(|x| (((x + 1.0) * 127.5).clamp(0.0, 255.0)) as u8)
        .collect();
    image::save_buffer(path, &buf, w as u32, h as u32, image::ColorType::Rgb8).unwrap();
}

#[test]
#[ignore = "needs qwenimage ckpt + gemma + dump_runb_latents outputs"]
fn from_ldm_runb() {
    let ckpt = env_or(
        "PID_QWEN_SAFETENSORS",
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pid/qwenimage_2kto4k.safetensors"
        )
        .to_string(),
    );
    let w = Weights::from_file(&ckpt).unwrap();

    let snap = gemma_snapshot();
    let gw = Weights::from_file(format!("{snap}/gemma-2-2b-it.safetensors")).unwrap();
    let gemma = Gemma2::from_weights(&gw, "model.", &Gemma2Config::gemma_2_2b()).unwrap();
    let enc = CaptionEncoder::new(gemma, format!("{snap}/tokenizer.json")).unwrap();
    // Single prompt for the whole run → caption embeds computed once.
    let caption_embs = enc
        .encode(CAPTION)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden/pid");

    for label in LABELS {
        let f = Weights::from_file(format!("{dir}/runb_{label}.safetensors")).unwrap();
        let latent = f
            .require("latent")
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let sigma = f.require("sigma").unwrap().as_slice::<f32>()[0];

        // Render the Qwen-VAE 1024² baseline (the "fair pair" left half).
        let vae_img = mlx_gen::image::decoded_to_image(
            &f.require("vae_decoded")
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap(),
        )
        .unwrap();
        image::save_buffer(
            format!("{dir}/qwenvae_runb_{label}.png"),
            &vae_img.pixels,
            vae_img.width,
            vae_img.height,
            image::ColorType::Rgb8,
        )
        .unwrap();

        // PiD 4096² decode with the captured step's σ feeding the sigma-aware LQ gate.
        let decoder = PidDecoder::new(
            PidNet::from_weights(&w, "", &PidConfig::sr4x()).unwrap(),
            Sampler::new(&SamplerConfig::distill_4step()),
            caption_embs.clone(),
            sigma, // degrade σ = the flow-match σ of this captured x_t
            4,     // scale
            8,     // vae_compression
            1234,  // seed
        );
        let (th, tw) = decoder.target_hw(&latent);
        eprintln!(
            "[{label}] σ={sigma:.4} latent {:?} -> {th}x{tw} ...",
            latent.shape()
        );
        let out = decoder.decode(&latent).unwrap();

        let lo = min(&out, None).unwrap().item::<f32>();
        let hi = max(&out, None).unwrap().item::<f32>();
        let mu = mean(&out, None).unwrap().item::<f32>();
        eprintln!(
            "[{label}] decoded {:?} min={lo:.3} max={hi:.3} mean={mu:.3}",
            out.shape()
        );
        assert!(lo.is_finite() && hi.is_finite(), "non-finite output");
        assert!(hi - lo > 0.2, "degenerate (flat) output");

        save_png(&out, &format!("{dir}/mlx_runb_{label}_4096.png"));
        eprintln!("[{label}] wrote mlx_runb_{label}_4096.png + qwenvae_runb_{label}.png");
    }
}
