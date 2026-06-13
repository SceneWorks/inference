//! sc-5115 — end-to-end Lens-Turbo T2I parity vs the vendor `LensPipeline`.
//!
//! Runs the **full** candle pipeline — the [`LensTokenizer`] (harmony render) → the gpt-oss encoder
//! (capture + `txt_offset` slice) → the DiT denoise (turbo schedule + norm-rescaled CFG) → the Flux.2
//! VAE decode — on the **same injected initial latents** the torch golden used, and compares against
//! the reference's final latents + decoded image.
//!
//! The e2e is **cross-build** (candle-CUDA vs torch, both bf16): per-step bf16 op-order diverges and
//! accumulates over 48 DiT blocks × 4 steps, so the gate is **structural** (cosine) + coherence, not
//! bit-exact. Injecting the reference's starting noise removes the only un-reproducible source (the
//! RNG); a wrong wiring (channel packing, offset slice, CFG, timestep convention, …) would collapse
//! the cosine. The tokenizer is validated *inside* the e2e: the candle render (with the golden's
//! date) must reproduce the golden's `input_ids` byte-for-byte.
//!
//! Heavy + machine-specific (loads the full ~48 GB bf16 pipeline + needs the GPU), so it is **gated**:
//!   LENS_SNAPSHOT_DIR — the `microsoft/Lens-Turbo` snapshot root (tokenizer/ text_encoder/ …)
//!   LENS_E2E_GOLDENS  — lens_e2e_golden.safetensors (default: .scratch/lens-e2e-goldens/…)
//! Run with the `cuda` feature:
//!   cargo test -p candle-gen-lens --features cuda --test e2e_parity -- --nocapture

use candle_gen::candle_core::{DType, Tensor};
use candle_gen_lens::text::LensTokenizer;
use candle_gen_lens::LensGenerator;

type AnyErr = Box<dyn std::error::Error>;

const PROMPT: &str = "a red fox sitting in a snowy forest at sunrise, photorealistic";
const NEGATIVE: &str = "";
const LATENT_H: usize = 32; // 512 / 16
const LATENT_W: usize = 32;
const NUM_STEPS: usize = 4;
const GUIDANCE: f32 = 1.0;

fn cosine(a: &Tensor, b: &Tensor) -> Result<f32, AnyErr> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shape mismatch in cosine");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    Ok((dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32)
}

fn peak_rel(a: &Tensor, b: &Tensor) -> Result<f32, AnyErr> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_diff = 0f64;
    let mut max_b = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        max_diff = max_diff.max((*x - *y).abs() as f64);
        max_b = max_b.max((*y).abs() as f64);
    }
    Ok((max_diff / max_b.max(1e-12)) as f32)
}

fn std_of(t: &Tensor) -> Result<f32, AnyErr> {
    let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
    Ok(var.sqrt())
}

#[test]
fn lens_e2e_matches_reference() -> Result<(), AnyErr> {
    let root = match std::env::var("LENS_SNAPSHOT_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: set LENS_SNAPSHOT_DIR to the Lens-Turbo snapshot root");
            return Ok(());
        }
    };
    let goldens_path = std::env::var("LENS_E2E_GOLDENS")
        .unwrap_or_else(|_| ".scratch/lens-e2e-goldens/lens_e2e_golden.safetensors".to_string());
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!(
            "SKIP: goldens not found at {goldens_path} (run scripts/dump_lens_e2e_golden.py)"
        );
        return Ok(());
    }

    let device = candle_gen::default_device()?;
    eprintln!("device: {device:?}");
    let g = candle_gen::candle_core::safetensors::load(&goldens_path, &device)?;

    let date = String::from_utf8(g["date_utf8"].to_dtype(DType::U8)?.to_vec1::<u8>()?)?;
    eprintln!("date={date}");

    // 1. Tokenizer cross-check (inside the e2e): the candle harmony render with the golden's date must
    //    reproduce the golden's input_ids exactly — otherwise the encoder sees a different sequence.
    let tok =
        LensTokenizer::from_file(std::path::Path::new(&root).join("tokenizer/tokenizer.json"))?;
    let got_ids = tok.encode(PROMPT, &date)?;
    let want_ids = g["input_ids"]
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    assert_eq!(
        got_ids,
        want_ids,
        "candle tokenizer ids differ from the golden (len {} vs {})",
        got_ids.len(),
        want_ids.len()
    );

    // 2. Load the full pipeline (bf16) and run the real path with the injected latents.
    eprintln!("loading Lens pipeline (encoder MXFP4→bf16 + DiT bf16 + VAE f32)…");
    let gen = LensGenerator::for_parity(&root)?;
    let init = g["init_latents"].to_dtype(DType::F32)?;

    eprintln!("denoising {NUM_STEPS} steps @ latent {LATENT_H}x{LATENT_W}…");
    let (latents, decoded) = gen.denoise_for_parity(
        PROMPT, NEGATIVE, &date, &init, LATENT_H, LATENT_W, NUM_STEPS, GUIDANCE,
    )?;

    // 3. Final latents (the tightest e2e signal: encoder + DiT + scheduler + CFG, pre-VAE).
    let lat_cos = cosine(&latents, &g["final_latents"])?;
    let lat_pr = peak_rel(&latents, &g["final_latents"])?;
    eprintln!("final latents: cosine={lat_cos:.5}  peak_rel={lat_pr:.3e}");

    // 4. Decoded image (full e2e incl. the VAE shim); both in [-1, 1].
    let img_cos = cosine(&decoded, &g["image"])?;
    let img_pr = peak_rel(&decoded, &g["image"])?;
    let (got_std, want_std) = (std_of(&decoded)?, std_of(&g["image"])?);
    eprintln!(
        "image: cosine={img_cos:.5}  peak_rel={img_pr:.3e}  std got={got_std:.4} / ref={want_std:.4}"
    );

    // Structural gates (cross-build, not bit-exact).
    assert!(
        lat_cos > 0.90,
        "final-latent cosine {lat_cos:.5} ≤ 0.90 — wiring divergence"
    );
    assert!(
        img_cos > 0.90,
        "decoded-image cosine {img_cos:.5} ≤ 0.90 — wiring/VAE divergence"
    );
    assert!(
        got_std > 0.5 * want_std,
        "decoded image near-flat (std {got_std:.4} vs ref {want_std:.4}) — not a coherent render"
    );
    eprintln!("ALL PASS");
    Ok(())
}
