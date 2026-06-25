//! sc-7843 e2e: a real `from_clean` PiD decode in MLX — `CaptionEncoder` + `PidDecoder.decode` on a
//! real `clean_latent` (QwenImage_VAE_2d encode, dumped by `tools/dump_pid_clean_latent.py`). Saves
//! the SR PNG for a visual/coherence check against the CUDA reference `*__pid_4step__4096.png`.
//!
//! `#[ignore]`d (needs the converted qwenimage checkpoint + gemma-2-2b-it + the clean_latent dump).
//! Defaults decode the small (256→1024²) latent; set `PID_DECODE_NATIVE=1` for the full 1024→4096².
//!
//! ```sh
//! cargo test -p mlx-gen-pid --release --test from_clean_decode -- --ignored --nocapture
//! ```

use mlx_gen::decoder::LatentDecoder;
use mlx_gen::weights::Weights;
use mlx_gen_pid::{
    CaptionEncoder, Gemma2, Gemma2Config, PidConfig, PidDecoder, PidNet, Sampler, SamplerConfig,
};
use mlx_rs::ops::{max, mean, min};
use mlx_rs::{Array, Dtype};

const CAPTION: &str =
    "a mountain valley landscape at golden hour with a winding river and pine forest";

fn env_or(name: &str, default: String) -> String {
    std::env::var(name).unwrap_or(default)
}

fn gemma_snapshot() -> String {
    env_or("PID_GEMMA_DIR", {
        let home = std::env::var("HOME").unwrap();
        let base = format!(
            "{home}/.cache/huggingface/hub/models--Efficient-Large-Model--gemma-2-2b-it/snapshots"
        );
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

/// `[1,3,H,W]` in [-1,1] → RGB8 PNG.
fn save_png(out: &Array, path: &str) {
    let sh = out.shape();
    let (h, w) = (sh[2], sh[3]);
    let chw = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[3, h, w])
        .unwrap();
    let hwc = chw.transpose_axes(&[1, 2, 0]).unwrap(); // [H,W,3]
    let v: Vec<f32> = hwc.as_slice::<f32>().to_vec();
    let buf: Vec<u8> = v
        .iter()
        .map(|x| (((x + 1.0) * 127.5).clamp(0.0, 255.0)) as u8)
        .collect();
    image::save_buffer(path, &buf, w as u32, h as u32, image::ColorType::Rgb8).unwrap();
}

#[test]
#[ignore = "needs qwenimage ckpt + gemma + clean_latent dump"]
fn from_clean_landscape() {
    // --- PiD net (real qwenimage student) ---
    let ckpt = env_or(
        "PID_QWEN_SAFETENSORS",
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pid/qwenimage_2kto4k.safetensors"
        )
        .to_string(),
    );
    let net =
        PidNet::from_weights(&Weights::from_file(&ckpt).unwrap(), "", &PidConfig::sr4x()).unwrap();

    // --- caption embeds (real gemma + Chi-prompt) ---
    let snap = gemma_snapshot();
    let gw = Weights::from_file(format!("{snap}/gemma-2-2b-it.safetensors")).unwrap();
    let gemma = Gemma2::from_weights(&gw, "model.", &Gemma2Config::gemma_2_2b()).unwrap();
    let enc = CaptionEncoder::new(gemma, format!("{snap}/tokenizer.json")).unwrap();
    // Run the decode in bf16 (the reference's inference dtype + the dtype the LQ-adapter convs expect).
    let caption_embs = enc.encode(CAPTION).unwrap().as_dtype(Dtype::Bfloat16).unwrap();
    eprintln!("caption_embs {:?}", caption_embs.shape());

    // --- clean latent (dumped from the QwenImage_VAE_2d encode) ---
    let latents = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tools/golden/pid/clean_latent_landscape.safetensors"
    ))
    .unwrap();
    let native = std::env::var("PID_DECODE_NATIVE").is_ok();
    let key = if native {
        "clean_latent_native"
    } else {
        "clean_latent_small"
    };
    let latent = latents
        .require(key)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    eprintln!("clean_latent[{key}] {:?}", latent.shape());

    // --- decode (σ=0 clean-latent path, 4×) ---
    let decoder = PidDecoder::new(
        net,
        Sampler::new(&SamplerConfig::distill_4step()),
        caption_embs,
        0.0,  // degrade σ
        4,    // scale
        8,    // vae_compression
        1234, // seed
    );
    let (th, tw) = decoder.target_hw(&latent);
    eprintln!("decoding -> {th}x{tw} ...");
    let out = decoder.decode(&latent).unwrap();
    assert_eq!(out.shape()[2], th);
    assert_eq!(out.shape()[3], tw);

    // coherence: finite, in-range, non-degenerate
    let lo = min(&out, None).unwrap().item::<f32>();
    let hi = max(&out, None).unwrap().item::<f32>();
    let mu = mean(&out, None).unwrap().item::<f32>();
    eprintln!(
        "decoded {:?}  min={lo:.3} max={hi:.3} mean={mu:.3}",
        out.shape()
    );
    assert!(lo.is_finite() && hi.is_finite(), "non-finite output");
    assert!(lo >= -1.001 && hi <= 1.001, "out of [-1,1]");
    assert!(hi - lo > 0.2, "degenerate (flat) output");

    let png = format!(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/golden/pid/mlx_landscape_{}.png"
        ),
        th
    );
    save_png(&out, &png);
    eprintln!("wrote {png}");
}
