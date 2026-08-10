//! F3 fixed-seed native-vs-Wan-2.1 decoder characterization through the public Krea generator.
//!
//! This is a standalone executable rather than a Rust test because MLX's default Metal stream is
//! process-thread-local: libtest runs test bodies on a worker thread, while the production worker and
//! this executable initialize and use MLX on their main inference thread.
//!
//! The denoiser, prompt, geometry, steps, and seed are identical; only
//! `LoadSpec.components["vae"]` changes. Both outputs must remain coherent and non-degenerate, while
//! a measurable pixel delta confirms the request actually reached the alternate terminal decoder.
//!
//! ```sh
//! KREA_TURBO_DIR=/path/to/krea-2-turbo-mlx/q4 \
//! WAN21_VAE_FILE=/path/to/krea-realtime-14b-mlx/q4/vae.safetensors \
//! cargo run -p mlx-gen-krea --release --example alternate_decoder_characterization
//! ```
//!
//! # Force the production bounded-decode seam
//!
//! ```sh
//! KREA_AB_SIZE=768 KREA_AB_TILED=1 \
//! cargo run -p mlx-gen-krea --release --example alternate_decoder_characterization
//! ```
//!
//! This selects the Krea memory ladder's real-weight-verified 512 px tile edge and 64 px overlap.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::GenerationMemory;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource, VAE_COMPONENT};
use mlx_gen_krea::load;

const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";
const SEED: u64 = 7;
const TILE_EDGE: u32 = 512;
const TILE_OVERLAP: u32 = 64;

fn render(spec: &LoadSpec, size: u32, tiled: bool) -> mlx_gen::media::Image {
    let generator = load(spec).expect("load Krea 2 Turbo");
    let output = generator
        .generate(
            &GenerationRequest {
                prompt: PROMPT.to_owned(),
                width: size,
                height: size,
                count: 1,
                seed: Some(SEED),
                steps: Some(8),
                memory: tiled.then_some(GenerationMemory {
                    tile_vae_decode: true,
                    decode_tile_edge: Some(TILE_EDGE),
                    decode_overlap: Some(TILE_OVERLAP),
                    ..Default::default()
                }),
                ..Default::default()
            },
            &mut |_| {},
        )
        .expect("fixed-seed generation");
    match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one output image"),
        _ => panic!("Krea returned non-image output"),
    }
}

fn stats(pixels: &[u8], width: u32) -> (f64, usize, f64) {
    let mean = pixels.iter().map(|&value| f64::from(value)).sum::<f64>() / pixels.len() as f64;
    let variance = pixels
        .iter()
        .map(|&value| (f64::from(value) - mean).powi(2))
        .sum::<f64>()
        / pixels.len() as f64;
    let mut levels = [false; 256];
    for &value in pixels {
        levels[value as usize] = true;
    }
    let stride = width as usize * 3;
    let mut adjacent = 0_u64;
    let mut samples = 0_u64;
    for (index, &value) in pixels.iter().enumerate() {
        if index >= 3 && index % stride >= 3 {
            adjacent += i32::from(value).abs_diff(i32::from(pixels[index - 3])) as u64;
            samples += 1;
        }
    }
    (
        variance.sqrt(),
        levels.into_iter().filter(|present| *present).count(),
        adjacent as f64 / samples.max(1) as f64,
    )
}

fn save(image: &mlx_gen::media::Image, name: &str) {
    let path = Path::new("/tmp/krea_alternate_decoder_ab").join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    image::save_buffer(
        &path,
        &image.pixels,
        image.width,
        image.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    eprintln!("saved {}", path.display());
}

fn main() {
    let base = PathBuf::from(std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR"));
    let donor = PathBuf::from(std::env::var("WAN21_VAE_FILE").expect("set WAN21_VAE_FILE"));
    let size = std::env::var("KREA_AB_SIZE")
        .ok()
        .map(|value| value.parse::<u32>().expect("KREA_AB_SIZE must be a u32"))
        .unwrap_or(512);
    let tiled = std::env::var("KREA_AB_TILED").as_deref() == Ok("1");
    if tiled {
        assert!(
            size > TILE_EDGE,
            "KREA_AB_SIZE={size} must exceed the {TILE_EDGE}px tile edge to force multiple tiles"
        );
    }
    eprintln!(
        "configuration geometry={size}x{size} tiled={tiled} tile_edge={} overlap={}",
        tiled.then_some(TILE_EDGE).unwrap_or(0),
        tiled.then_some(TILE_OVERLAP).unwrap_or(0)
    );
    let native_spec = LoadSpec::new(WeightsSource::Dir(base)).with_quant(Quant::Q4);

    let native = render(&native_spec, size, tiled);
    mlx_rs::memory::clear_cache();
    let alternate = render(
        &native_spec.with_component(VAE_COMPONENT, WeightsSource::File(donor)),
        size,
        tiled,
    );

    assert_eq!(
        (alternate.width, alternate.height),
        (native.width, native.height)
    );
    let native_stats = stats(&native.pixels, native.width);
    let alternate_stats = stats(&alternate.pixels, alternate.width);
    let changed = native
        .pixels
        .iter()
        .zip(&alternate.pixels)
        .filter(|(left, right)| left != right)
        .count();
    let mean_abs_delta = native
        .pixels
        .iter()
        .zip(&alternate.pixels)
        .map(|(&left, &right)| f64::from(left.abs_diff(right)))
        .sum::<f64>()
        / native.pixels.len() as f64;
    eprintln!(
        "seed={SEED} native(std={:.2}, levels={}, adjacent={:.2}) \
         wan(std={:.2}, levels={}, adjacent={:.2}) changed={:.2}% mean_abs_delta={mean_abs_delta:.3}",
        native_stats.0,
        native_stats.1,
        native_stats.2,
        alternate_stats.0,
        alternate_stats.1,
        alternate_stats.2,
        changed as f64 * 100.0 / native.pixels.len() as f64,
    );
    for (label, (std, levels, adjacent)) in [("native", native_stats), ("Wan 2.1", alternate_stats)]
    {
        assert!(std > 10.0, "{label} output histogram is degenerate: {std}");
        assert!(levels > 24, "{label} output has only {levels} levels");
        assert!(
            adjacent < 60.0,
            "{label} output resembles noise: {adjacent}"
        );
    }
    assert!(
        changed > native.pixels.len() / 100,
        "alternate decoder was inert"
    );
    assert!(
        mean_abs_delta > 0.5,
        "alternate decoder delta is negligible"
    );
    let mode = if tiled { "tiled" } else { "untiled" };
    save(&native, &format!("native-{mode}-{size}.png"));
    save(&alternate, &format!("wan21-{mode}-{size}.png"));
}
