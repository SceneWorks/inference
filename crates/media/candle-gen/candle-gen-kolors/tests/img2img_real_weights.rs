//! sc-18476 real-weight CUDA proof for the registered Kolors source-image img2img path.
//!
//! Native leading-Euler and curated-Euler conditioned renders are each compared with a text-only
//! render at the same seed. Byte identity would expose the exact regression this story closes:
//! accepting `Reference` while silently taking the unconditioned T2I path. A native-vs-curated
//! comparison also proves the new curated img2img branch ran rather than aliasing the default lane.

use std::path::PathBuf;

use candle_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};

fn required_path(name: &str) -> PathBuf {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"));
    let path = PathBuf::from(value);
    assert!(path.exists(), "{name} does not exist: {}", path.display());
    path
}

fn structured_reference(size: u32) -> Image {
    let mut pixels = vec![0u8; (size * size * 3) as usize];
    for y in 0..size {
        for x in 0..size {
            let offset = ((y * size + x) * 3) as usize;
            let upper = y < size / 2;
            pixels[offset] = if upper { 45 } else { 48 };
            pixels[offset + 1] = if upper { 120 } else { 150 };
            pixels[offset + 2] = if upper { 210 } else { 72 };
            let dx = x as i64 - (size * 3 / 4) as i64;
            let dy = y as i64 - (size / 4) as i64;
            if dx * dx + dy * dy < (size / 10) as i64 * (size / 10) as i64 {
                pixels[offset] = 245;
                pixels[offset + 1] = 185;
                pixels[offset + 2] = 35;
            }
        }
    }
    Image {
        width: size,
        height: size,
        pixels,
    }
}

fn one_image(output: GenerationOutput) -> Image {
    let GenerationOutput::Images(mut images) = output else {
        panic!("expected image output");
    };
    assert_eq!(images.len(), 1);
    images.pop().expect("one image")
}

fn standard_deviation(pixels: &[u8]) -> f64 {
    let mean = pixels.iter().map(|&pixel| pixel as f64).sum::<f64>() / pixels.len() as f64;
    let variance = pixels
        .iter()
        .map(|&pixel| (pixel as f64 - mean).powi(2))
        .sum::<f64>()
        / pixels.len() as f64;
    variance.sqrt()
}

fn mean_abs_delta(left: &[u8], right: &[u8]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / left.len() as f64
}

fn save(image: &Image, name: &str) {
    let dir = required_path("KOLORS_IMG2IMG_ARTIFACT_DIR");
    std::fs::create_dir_all(&dir).expect("create artifact directory");
    let path = dir.join(name);
    image::save_buffer(
        &path,
        &image.pixels,
        image.width,
        image.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap_or_else(|error| panic!("save {}: {error}", path.display()));
    eprintln!("saved {}", path.display());
}

#[test]
#[ignore = "needs KOLORS_IMG2IMG_SNAPSHOT + CUDA; run explicitly with --features cuda --ignored"]
fn registered_reference_img2img_uses_the_vae_init_and_strength_tail() {
    const SIZE: u32 = 512;
    const STEPS: u32 = 10;
    const STRENGTH: f32 = 0.8;

    let generator = candle_gen_kolors::provider_registry()
        .expect("Kolors registry")
        .load(
            candle_gen_kolors::MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(required_path("KOLORS_IMG2IMG_SNAPSHOT"))),
        )
        .expect("load Kolors");
    let reference = structured_reference(SIZE);
    let base = GenerationRequest {
        prompt: "transform the flat landscape into intricate stained-glass artwork while \
                 preserving the rolling green hills and golden sun"
            .into(),
        negative_prompt: Some("blurry, flat, watermark, text".into()),
        width: SIZE,
        height: SIZE,
        count: 1,
        seed: Some(18_476),
        steps: Some(STEPS),
        guidance: Some(5.0),
        ..Default::default()
    };

    let text_only = one_image(
        generator
            .generate(&base, &mut |_| {})
            .expect("text-only baseline"),
    );
    let mut conditioned = base.clone();
    conditioned.conditioning = vec![Conditioning::Reference {
        image: reference.clone(),
        strength: Some(STRENGTH),
    }];
    let mut denoise_steps = 0u32;
    let img2img = one_image(
        generator
            .generate(&conditioned, &mut |progress| {
                if matches!(progress, Progress::Step { .. }) {
                    denoise_steps += 1;
                }
            })
            .expect("reference-conditioned render"),
    );

    // Exercise the newly wired curated img2img branch as a distinct real-weight path. `euler` is a
    // curated solver name (unlike the native `euler_discrete` default), so this request must enter
    // `CuratedSetup::new_img2img` and denoise the same strength-selected schedule tail.
    let mut curated_request = base;
    curated_request.sampler = Some("euler".into());
    curated_request.conditioning = vec![Conditioning::Reference {
        image: reference.clone(),
        strength: Some(STRENGTH),
    }];
    let mut curated_steps = 0u32;
    let curated_img2img = one_image(
        generator
            .generate(&curated_request, &mut |progress| {
                if matches!(progress, Progress::Step { .. }) {
                    curated_steps += 1;
                }
            })
            .expect("curated reference-conditioned render"),
    );

    let expected_steps = (STEPS as f32 * STRENGTH).floor() as u32;
    assert_eq!(
        denoise_steps, expected_steps,
        "strength-selected schedule tail"
    );
    assert_eq!(
        curated_steps, expected_steps,
        "curated strength-selected schedule tail"
    );
    let spread = standard_deviation(&img2img.pixels);
    let t2i_delta = mean_abs_delta(&text_only.pixels, &img2img.pixels);
    let reference_delta = mean_abs_delta(&reference.pixels, &img2img.pixels);
    let curated_spread = standard_deviation(&curated_img2img.pixels);
    let curated_t2i_delta = mean_abs_delta(&text_only.pixels, &curated_img2img.pixels);
    let curated_reference_delta = mean_abs_delta(&reference.pixels, &curated_img2img.pixels);
    let lane_delta = mean_abs_delta(&img2img.pixels, &curated_img2img.pixels);
    eprintln!(
        "Kolors img2img: std={spread:.2}, mean |Δ vs T2I|={t2i_delta:.2}, \
         mean |Δ vs reference|={reference_delta:.2}"
    );
    eprintln!(
        "Kolors curated img2img: std={curated_spread:.2}, mean abs vs T2I=\
         {curated_t2i_delta:.2}, mean abs vs reference={curated_reference_delta:.2}, \
         mean abs vs native={lane_delta:.2}"
    );
    assert!(spread > 8.0, "conditioned output is near-flat: {spread:.2}");
    assert!(
        t2i_delta > 1.0,
        "conditioned output is indistinguishable from the unconditioned route: {t2i_delta:.3}"
    );
    assert!(
        reference_delta > 5.0,
        "conditioned output did not materially edit the source image: {reference_delta:.3}"
    );
    assert!(
        curated_spread > 8.0,
        "curated conditioned output is near-flat: {curated_spread:.2}"
    );
    assert!(
        curated_t2i_delta > 1.0,
        "curated conditioned output is indistinguishable from T2I: {curated_t2i_delta:.3}"
    );
    assert!(
        curated_reference_delta > 5.0,
        "curated output did not materially edit the source: {curated_reference_delta:.3}"
    );
    assert!(
        lane_delta > 0.1,
        "curated and native paths produced indistinguishable outputs: {lane_delta:.3}"
    );

    save(&reference, "kolors_reference.png");
    save(&text_only, "kolors_text_only.png");
    save(&img2img, "kolors_img2img.png");
    save(&curated_img2img, "kolors_img2img_curated_euler.png");
}
