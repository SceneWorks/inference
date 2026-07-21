//! GPU A/B validation for the NVIDIA PiD `sdxl` decoder (epic 7840 / sc-7853): render once through the
//! native SDXL VAE and once through PiD (`use_pid`), write both PNGs. Confirms the SDXL de-norm seam
//! (PiD gets the 0.13025-normalized latent, not the de-scaled raw latent).
//!
//! `cargo run -p candle-gen-sdxl --example sdxl-pid-ab --features cuda --release -- [model] [base] [pid] [gemma] [W] [H] [seed]`

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};

fn mean(px: &[u8]) -> f64 {
    if px.is_empty() {
        0.0
    } else {
        px.iter().map(|&p| p as f64).sum::<f64>() / px.len() as f64
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).cloned().unwrap_or_else(|| "sdxl".into());
    // Explicit passed-in local paths (positional arg, else env var) — inference never self-fetches or
    // derives an HF-cache location (epic 13657).
    let base = a.get(2).cloned().or_else(|| std::env::var("SDXL_BASE_SNAPSHOT").ok())
        .ok_or("pass the base snapshot as arg 2 or set SDXL_BASE_SNAPSHOT to a stabilityai/stable-diffusion-xl-base-1.0 snapshot dir")?;
    let pid = a
        .get(3)
        .cloned()
        .or_else(|| std::env::var("PID_SDXL_FILE").ok())
        .ok_or(
            "pass the pid-sdxl file as arg 3 or set PID_SDXL_FILE to pid_sdxl_2kto4k.safetensors",
        )?;
    let gemma = a.get(4).cloned().or_else(|| std::env::var("PID_GEMMA_DIR").ok())
        .ok_or("pass the gemma dir as arg 4 or set PID_GEMMA_DIR to a google/gemma-2-2b-it snapshot dir")?;
    let w: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(512);
    let h: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(512);
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(7);
    let prompt = "a red fox sitting in a snowy pine forest at dawn, sharp fur detail".to_string();

    eprintln!("model {model}\nbase  {base}\npid   {pid}\ngemma {gemma}");
    // epic 13657 / sc-13663: the CLIP tokenizers + fp16-fix VAE are passed-in components (env-pointed
    // local dirs: SDXL_TOKENIZER_CLIP_L_DIR / SDXL_TOKENIZER_CLIP_BIGG_DIR / SDXL_VAE_FP16_FIX_DIR).
    let component = |env: &str| -> Result<WeightsSource, Box<dyn std::error::Error>> {
        let dir =
            std::env::var(env).map_err(|_| format!("set {env} to the SDXL component's dir"))?;
        Ok(WeightsSource::Dir(dir.into()))
    };
    let spec = LoadSpec::new(WeightsSource::Dir(base.into()))
        .with_pid(
            WeightsSource::File(pid.into()),
            WeightsSource::Dir(gemma.into()),
        )
        .with_component("tokenizer_clip_l", component("SDXL_TOKENIZER_CLIP_L_DIR")?)
        .with_component(
            "tokenizer_clip_bigg",
            component("SDXL_TOKENIZER_CLIP_BIGG_DIR")?,
        )
        .with_component("vae_fp16_fix", component("SDXL_VAE_FP16_FIX_DIR")?);
    let gen = candle_gen_sdxl::provider_registry()?.load(&model, &spec)?;
    let mut op = |p: Progress| match p {
        Progress::Step { current, total } => eprintln!("  step {current}/{total}"),
        Progress::Decoding => eprintln!("  decoding…"),
        Progress::Loading(phase) => eprintln!("  loading {phase:?}"),
    };
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        count: 1,
        seed: Some(seed),
        ..Default::default()
    };

    eprintln!("[A] VAE {w}x{h}…");
    let GenerationOutput::Images(mut i) = gen.generate(&req, &mut op)? else {
        return Err("images".into());
    };
    let v = i.pop().ok_or("no vae img")?;
    eprintln!("  VAE {}x{} mean={:.1}", v.width, v.height, mean(&v.pixels));
    image::RgbImage::from_raw(v.width, v.height, v.pixels.clone())
        .ok_or("buf")?
        .save(format!("{model}_pid_vae.png"))?;

    eprintln!("[B] PiD (expect {}x{})…", w * 4, h * 4);
    let preq = GenerationRequest {
        use_pid: true,
        ..req
    };
    let GenerationOutput::Images(mut i) = gen.generate(&preq, &mut op)? else {
        return Err("images".into());
    };
    let p = i.pop().ok_or("no pid img")?;
    eprintln!("  PiD {}x{} mean={:.1}", p.width, p.height, mean(&p.pixels));
    image::RgbImage::from_raw(p.width, p.height, p.pixels.clone())
        .ok_or("buf")?
        .save(format!("{model}_pid_pid.png"))?;

    assert_eq!(p.width, v.width * 4, "PiD must be 4× the VAE width");
    let (lo, hi) = p
        .pixels
        .iter()
        .fold((255u8, 0u8), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(hi - lo > 32, "degenerate color range {lo}..{hi}");
    eprintln!("OK — wrote {model}_pid_vae.png + {model}_pid_pid.png");
    Ok(())
}
