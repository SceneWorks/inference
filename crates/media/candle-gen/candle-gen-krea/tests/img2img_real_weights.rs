//! sc-10134 (epic 8588 slice A) — candle Krea 2 **Turbo img2img** (reference-guided latent-init)
//! end-to-end real-weight validation, the Windows/CUDA twin of mlx-gen-krea's img2img spike
//! (`tests/img2img_spike_real_weights.rs`, mlx A0/A1). Drives the engine pipeline directly
//! (`load_components` + `load_vae_encoder` + `render_img2img`) rather than the `Generator` contract so
//! the engine capability is exercised in isolation.
//!
//! The AC (mlx A0): reference-guided output is coherent across the strength range AND monotone in
//! reference fidelity — a higher strength starts the denoise later, so the output stays CLOSER to the
//! reference (mean-abs-error to the reference FALLS as strength rises; the fork's convention, the inverse
//! of SDXL's). `#[ignore]` — needs a real Krea 2 **Turbo** snapshot, a source image, and a CUDA GPU:
//! ```sh
//! KREA_TURBO_DIR=D:\models\Krea-2-Turbo \
//! KREA_IMG2IMG_SOURCE=D:\fixtures\photo.png \
//!   cargo test -p candle-gen-krea --release --features cuda --test img2img_real_weights -- --ignored --nocapture
//! ```
//! `KREA_IMG2IMG_SIZE=WxH` (multiples of 16) overrides the target resolution; else the reference's size
//! rounded down to a multiple of 16. `KREA_IMG2IMG_STEPS` overrides the ~8-step Turbo budget.
//! `KREA_IMG2IMG_PROMPT` overrides the prompt. Sources may be binary P6 `.ppm` or `.png`.

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::{GenerationRequest, Image};
use candle_gen_krea::pipeline::{load_components, render_img2img};
use candle_gen_krea::vae::load_vae_encoder;

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) — a coherent natural image has a broad
/// histogram AND spatial smoothness; pure noise has a high adjacent Δ and flat std.
fn image_stats(px: &[u8], w: u32) -> (f32, usize, f32) {
    let n = px.len() as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let mut seen = [false; 256];
    for &v in px {
        seen[v as usize] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count();
    let stride = (w * 3) as usize;
    let (mut adj_sum, mut adj_n) = (0f64, 0u64);
    for (i, &v) in px.iter().enumerate() {
        if i >= 3 && i % stride >= 3 {
            adj_sum += (v as i32 - px[i - 3] as i32).unsigned_abs() as f64;
            adj_n += 1;
        }
    }
    (
        var.sqrt() as f32,
        distinct,
        (adj_sum / adj_n.max(1) as f64) as f32,
    )
}

fn is_coherent(img: &Image) -> bool {
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    std > 10.0 && distinct > 24 && adj < 60.0
}

/// Mean absolute pixel error between an output and a `target`-sized reference (both RGB8, same dims).
fn mae(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "MAE inputs must be the same size");
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.len() as f32
}

/// Minimal binary-PPM (P6) reader → an [`Image`] (RGB8), matching the edit smoke fixture format.
fn read_ppm(path: &PathBuf) -> Image {
    let bytes = std::fs::read(path).expect("read source PPM");
    assert_eq!(&bytes[0..2], b"P6", "expected a binary P6 PPM");
    let mut nums = Vec::with_capacity(3);
    let mut cur = String::new();
    let mut consumed = 2usize;
    for b in bytes[2..].iter().copied() {
        consumed += 1;
        if b.is_ascii_whitespace() {
            if !cur.is_empty() {
                nums.push(cur.parse::<usize>().unwrap());
                cur.clear();
                if nums.len() == 3 {
                    break;
                }
            }
        } else {
            cur.push(b as char);
        }
    }
    let (w, h) = (nums[0], nums[1]);
    let pixels = bytes[consumed..consumed + w * h * 3].to_vec();
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// Decode a source reference from either a binary P6 `.ppm` or a `.png` into an RGB8 [`Image`].
fn read_source(path: &PathBuf) -> Image {
    let is_png = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("png"));
    if !is_png {
        return read_ppm(path);
    }
    let rgb = image::open(path).expect("decode PNG source").to_rgb8();
    let (width, height) = rgb.dimensions();
    Image {
        width,
        height,
        pixels: rgb.into_raw(),
    }
}

/// LANCZOS-resize a reference to `(w, h)` as RGB8 — the reference-side twin of the engine's img2img
/// preprocess, so the MAE comparison is against the reference at the output resolution.
fn resize_ref(im: &Image, w: u32, h: u32) -> Vec<u8> {
    if (im.width, im.height) == (w, h) {
        return im.pixels.clone();
    }
    let f = resize_lanczos_u8(
        &im.pixels,
        im.height as usize,
        im.width as usize,
        h as usize,
        w as usize,
    )
    .expect("resize reference");
    f.iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect()
}

fn save(img: &Image, name: &str) {
    let dir = std::env::temp_dir().join("krea_img2img_smoke");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    eprintln!("  saved {}", path.display());
}

/// The engine img2img AC on the real GPU: VAE-encode a source, denoise from the strength-blended init, and
/// (a) render a coherent image at every strength, and (b) show monotone reference fidelity — MAE→reference
/// falls as strength rises (higher strength = later start = closer to the reference; a broken start-step /
/// blend / schedule-slice would break the ordering or yield noise).
#[test]
#[ignore = "needs KREA_TURBO_DIR + KREA_IMG2IMG_SOURCE; --features cuda"]
fn img2img_is_coherent_and_monotone_in_reference_fidelity() {
    let (Ok(root), Ok(source)) = (
        std::env::var("KREA_TURBO_DIR"),
        std::env::var("KREA_IMG2IMG_SOURCE"),
    ) else {
        eprintln!("skipping: set KREA_TURBO_DIR + KREA_IMG2IMG_SOURCE");
        return;
    };
    let root = PathBuf::from(root);
    let device = candle_gen::default_device().expect("device");

    let reference = read_source(&PathBuf::from(&source));
    // Target: KREA_IMG2IMG_SIZE=WxH (multiples of 16) or the reference size rounded down to a multiple.
    let (w, h) = std::env::var("KREA_IMG2IMG_SIZE")
        .ok()
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
        })
        .unwrap_or((reference.width / 16 * 16, reference.height / 16 * 16));
    assert!(w >= 256 && h >= 256, "target {w}x{h} too small");
    let steps: u32 = std::env::var("KREA_IMG2IMG_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let prompt = std::env::var("KREA_IMG2IMG_PROMPT")
        .unwrap_or_else(|_| "a highly detailed photograph, sharp focus, natural light".into());

    let t_load = Instant::now();
    let comps = load_components(&root, &device, &[], None).expect("load Krea Turbo components");
    let vae_encoder = load_vae_encoder(&root, &device).expect("load VAE encoder");
    let load_s = t_load.elapsed().as_secs_f32();

    let ref_at_target = resize_ref(&reference, w, h);

    // Sweep strength (the A0 workable band). MAE→reference must be non-increasing as strength rises.
    let strengths = [0.35f32, 0.6, 0.85];
    let mut prev_mae = f32::INFINITY;
    for &s in &strengths {
        let req = GenerationRequest {
            prompt: prompt.clone(),
            width: w,
            height: h,
            count: 1,
            seed: Some(0),
            steps: Some(steps),
            ..Default::default()
        };
        let t_gen = Instant::now();
        let imgs = render_img2img(
            &comps,
            &vae_encoder,
            &req,
            &reference,
            Some(s),
            &device,
            &mut |_| {},
        )
        .expect("render_img2img");
        let gen_s = t_gen.elapsed().as_secs_f32();

        assert_eq!(imgs.len(), 1);
        let img = &imgs[0];
        assert_eq!((img.width, img.height), (w, h), "output dims");
        let (std, distinct, adj) = image_stats(&img.pixels, img.width);
        let m = mae(&img.pixels, &ref_at_target);
        eprintln!(
            "[krea img2img {w}x{h} strength={s} steps={steps}] load {load_s:.1}s · render {gen_s:.1}s · \
             std={std:.1} distinct={distinct} adjΔ={adj:.1} MAE→ref={m:.2} coherent={}",
            is_coherent(img)
        );
        save(img, &format!("img2img_{w}x{h}_s{}", (s * 100.0) as u32));
        assert!(
            is_coherent(img),
            "strength {s}: img2img render must be coherent, not noise (std={std:.1} distinct={distinct} adjΔ={adj:.1})"
        );
        // Monotone reference fidelity: a higher strength must not move FARTHER from the reference. Small
        // slack for sampler noise between adjacent rungs.
        assert!(
            m <= prev_mae + 2.0,
            "MAE→ref must be non-increasing as strength rises (strength {s}: MAE {m:.2} > prev {prev_mae:.2})"
        );
        prev_mae = m;
    }
}
