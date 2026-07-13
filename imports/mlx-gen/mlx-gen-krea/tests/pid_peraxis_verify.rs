//! Verify the per-axis pixel-pos clamp fixes BOTH orientations: tall 720×1280 (height OOD → was the
//! original bottom cast) and wide 1280×720 (height in-range → the aspect-preserving version wrongly
//! cast it). Renders each with the fix on (default, now per-axis) and measures bottom-vs-top
//! greenness. Success: both bottoms neutral. Baselines from prior runs — tall absolute +34.17, wide
//! aspect-preserving (old fix) +13.86, wide absolute −1.97.
//!
//! ```sh
//! cargo test -p mlx-gen-krea --release --test pid_peraxis_verify -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_krea::load;

fn snapshot(model_dir: &str) -> PathBuf {
    let base = PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub")
        .join(model_dir)
        .join("snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("missing HF cache: {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .unwrap_or_else(|| panic!("no snapshot dir under {}", base.display()))
}

fn band_greenness(img: &Image, r0: u32, r1: u32) -> f64 {
    let w = img.width;
    let (mut sum, mut n) = (0f64, 0u64);
    for row in r0..r1 {
        for col in 0..w {
            let i = ((row * w + col) * 3) as usize;
            sum +=
                img.pixels[i + 1] as f64 - 0.5 * (img.pixels[i] as f64 + img.pixels[i + 2] as f64);
            n += 1;
        }
    }
    sum / n.max(1) as f64
}

fn save_png(img: &Image, path: &str) {
    image::save_buffer(
        path,
        &img.pixels,
        img.width,
        img.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
}

fn render(width: u32, height: u32) -> Image {
    let spec = LoadSpec::new(WeightsSource::Dir(
        snapshot("models--SceneWorks--krea-2-turbo-mlx").join("q4"),
    ))
    .with_pid(
        WeightsSource::File(
            snapshot("models--SceneWorks--pid-qwenimage").join("pid_qwenimage_2kto4k.safetensors"),
        ),
        WeightsSource::Dir(snapshot("models--SceneWorks--gemma-2-2b-it")),
    );
    let model = load(&spec).expect("load Krea 2 Turbo Q4 + PiD");
    let req = GenerationRequest {
        prompt:
            "An old barn full of rectangular bails of hay. The floor is weathered wooden slats \
                 with loose hay laying around. Golden hour light streams in."
                .into(),
        width,
        height,
        count: 1,
        seed: Some(333_941_069),
        steps: Some(8),
        use_pid: true,
        ..Default::default()
    };
    match model.generate(&req, &mut |_| {}).expect("pid generate") {
        GenerationOutput::Images(v) => v.into_iter().next().unwrap(),
        _ => panic!("expected images"),
    }
}

fn check(label: &str, width: u32, height: u32) -> f64 {
    let img = render(width, height);
    let band = (img.height / 10).max(1);
    let top = band_greenness(&img, 0, band);
    let bottom = band_greenness(&img, img.height - band, img.height);
    let excess = bottom - top;
    eprintln!(
        "[{label}] native {width}x{height} -> {}x{}  top={top:+.2} bottom={bottom:+.2} bottom-top={excess:+.2}",
        img.width, img.height
    );
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden/pid");
    let _ = std::fs::create_dir_all(out);
    save_png(&img, &format!("{out}/krea_peraxis_{label}.png"));
    excess
}

#[test]
#[ignore = "needs Krea Turbo Q4 + pid-qwenimage + gemma-2-2b-it (GPU, ~minutes)"]
fn per_axis_clamp_fixes_both_orientations() {
    let tall = check("tall_720x1280", 720, 1280);
    let wide = check("wide_1280x720", 1280, 720);
    eprintln!("\n=== per-axis fix: tall bottom-top={tall:+.2}  wide bottom-top={wide:+.2} (both should be ~neutral) ===");
    assert!(tall.abs() < 5.0, "tall still cast: bottom-top={tall}");
    assert!(wide.abs() < 5.0, "wide cast: bottom-top={wide}");
}
