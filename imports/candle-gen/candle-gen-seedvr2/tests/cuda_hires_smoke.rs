//! High-resolution CUDA regression for the SeedVR2 image path (sc-10083).
//!
//! Guards the "2× upscale of a large image renders pure black" regression: candle's CUDA conv2d
//! silently fills an oversized im2col buffer with uninitialised memory (`f32::MAX` → casts to 0 →
//! black), and the still-image VAE conv path (batch == 1) had no chunking to keep it under budget, so
//! any output ≥ ~1536² came out black. This exercises the exact production sizes — 1536² (a single
//! pass at the VAE decode cap) and 2048² (the spatial-tiled path, whose tiles are themselves 1536²) —
//! and asserts the output is a real, non-black, structurally-faithful upscale.
//!
//! `#[ignore]` by default (needs weights + a CUDA build). Run on the Blackwell box with:
//! ```text
//! set SEEDVR2_CKPT=D:\sceneworks-seedvr2-validate\ckpt
//! set SEEDVR2_DTYPE=bf16
//! cargo test -p candle-gen-seedvr2 --features cuda --release --test cuda_hires_smoke -- --ignored --nocapture
//! ```

use candle_gen::candle_core::DType;
use candle_gen::gen_core::{imageops, Image};
use candle_gen_seedvr2::config::DitConfig;
use candle_gen_seedvr2::pipeline::Seedvr2Pipeline;

const DIT_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";

/// A deterministic structured LR image (gradients + checkerboard + rings) — real detail to upscale.
fn synth_lr(side: usize) -> Image {
    let mut pixels = vec![0u8; side * side * 3];
    for y in 0..side {
        for x in 0..side {
            let i = (y * side + x) * 3;
            let check = (((x / 12) + (y / 12)) % 2) as u8 * 90;
            let cx = side as f32 / 2.0;
            let dr = (((x as f32 - cx).powi(2) + (y as f32 - cx).powi(2)).sqrt() * 0.18).sin();
            pixels[i] = (x * 255 / side) as u8;
            pixels[i + 1] = (40 + check as usize).min(255) as u8;
            pixels[i + 2] = (((dr + 1.0) * 0.5) * 255.0) as u8;
        }
    }
    Image {
        width: side as u32,
        height: side as u32,
        pixels,
    }
}

fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va.sqrt() * vb.sqrt()).max(1e-12)
}

#[test]
#[ignore = "needs SEEDVR2_CKPT weights + a CUDA build"]
fn cuda_hires_upscale_not_black() {
    let ckpt = match std::env::var("SEEDVR2_CKPT") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP: set SEEDVR2_CKPT to a numz/SeedVR2_comfyUI checkpoint dir");
            return;
        }
    };
    let dtype = match std::env::var("SEEDVR2_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        _ => DType::F32,
    };
    let device = candle_gen::default_device().expect("device");
    let cfg = DitConfig::seedvr2_3b();
    let pipe = Seedvr2Pipeline::load(&ckpt, DIT_FILE, &cfg, dtype, &device).expect("load pipeline");
    eprintln!("[hires-smoke] device={device:?} dtype={dtype:?}");

    // 1536² = single-pass at the VAE decode cap; 2048² = the spatial-tiled path (1536² tiles). Both
    // regressed to pure black before sc-10083; both must now be real, non-black, faithful upscales.
    for (src, tgt) in [(768usize, 1536usize), (1024, 2048)] {
        let lr = synth_lr(src);
        let out = pipe.generate(&lr, tgt, tgt, 42, 0.0).expect("generate");
        assert_eq!((out.width, out.height), (tgt as u32, tgt as u32));

        let black = out.pixels.iter().filter(|&&v| v == 0).count() as f64 / out.pixels.len() as f64;
        let mx = *out.pixels.iter().max().unwrap();
        let mean = out.pixels.iter().map(|&v| v as f64).sum::<f64>() / out.pixels.len() as f64;

        // structural faithfulness vs a bicubic baseline (a corrupt/blank decode destroys correlation).
        let base = imageops::resize_bicubic_u8(&lr.pixels, src, src, tgt, tgt).unwrap();
        let out_f: Vec<f32> = out.pixels.iter().map(|&v| v as f32).collect();
        let corr = pearson(&out_f, &base);
        eprintln!(
            "[hires-smoke] {src}->{tgt}: max={mx} mean={mean:.1} black_frac={black:.4} corr_vs_bicubic={corr:.4}"
        );

        assert!(
            mx > 0 && black < 0.5,
            "{tgt}² upscale is (mostly) black (max={mx}, black_frac={black:.4}) — sc-10083 regression: \
             the still-image VAE conv path is corrupting its oversized im2col buffer again"
        );
        assert!(
            corr > 0.7,
            "{tgt}² upscale not structurally faithful (corr={corr:.4})"
        );
    }
}
