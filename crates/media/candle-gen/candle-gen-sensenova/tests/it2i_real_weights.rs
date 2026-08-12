//! sc-18476 real-weight CUDA proof for the registered SenseNova singular- and multi-reference paths.
//!
//! The same prompt and seed are rendered without a reference, with two materially different singular
//! references, with one reference at two true-CFG values, and with two references. Same-shape A/B
//! divergence proves pixels (not merely marker/cache geometry) affect output; the other comparisons
//! prove true-CFG and each registered request shape remain live.

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
            pixels[offset] = if upper { 35 } else { 52 };
            pixels[offset + 1] = if upper { 115 } else { 145 };
            pixels[offset + 2] = if upper { 205 } else { 68 };
            let dx = x as i64 - (size * 3 / 4) as i64;
            let dy = y as i64 - (size / 4) as i64;
            if dx * dx + dy * dy < (size / 10) as i64 * (size / 10) as i64 {
                pixels[offset] = 250;
                pixels[offset + 1] = 190;
                pixels[offset + 2] = 30;
            }
        }
    }
    Image {
        width: size,
        height: size,
        pixels,
    }
}

fn secondary_reference(size: u32) -> Image {
    let mut pixels = vec![0u8; (size * size * 3) as usize];
    for y in 0..size {
        for x in 0..size {
            let offset = ((y * size + x) * 3) as usize;
            let left = x < size / 2;
            pixels[offset] = if left { 195 } else { 42 };
            pixels[offset + 1] = if left { 58 } else { 188 };
            pixels[offset + 2] = if left { 82 } else { 125 };
            if x.abs_diff(y) < size / 24 {
                pixels[offset] = 245;
                pixels[offset + 1] = 235;
                pixels[offset + 2] = 48;
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
    let dir = required_path("SENSENOVA_IT2I_ARTIFACT_DIR");
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
#[ignore = "needs SENSENOVA_IT2I_SNAPSHOT + CUDA; run explicitly with --features cuda --ignored"]
fn registered_reference_it2i_consumes_image_and_true_cfg() {
    const SIZE: u32 = 512;
    const STEPS: u32 = 4;

    let generator = candle_gen_sensenova::provider_registry()
        .expect("SenseNova registry")
        .load(
            candle_gen_sensenova::MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(required_path("SENSENOVA_IT2I_SNAPSHOT"))),
        )
        .expect("load SenseNova");
    let reference = structured_reference(SIZE);
    let secondary = secondary_reference(SIZE);
    let base = GenerationRequest {
        prompt: "transform the landscape into stained glass and add a small red fox".into(),
        width: SIZE,
        height: SIZE,
        count: 1,
        seed: Some(18_476),
        steps: Some(STEPS),
        guidance: Some(4.0),
        ..Default::default()
    };

    let text_only = one_image(
        generator
            .generate(&base, &mut |_| {})
            .expect("text-only baseline"),
    );
    let mut conditioned = base.clone();
    conditioned.true_cfg = Some(1.5);
    conditioned.conditioning = vec![Conditioning::Reference {
        image: reference.clone(),
        strength: None,
    }];
    let mut denoise_steps = 0u32;
    let it2i = one_image(
        generator
            .generate(&conditioned, &mut |progress| {
                if matches!(progress, Progress::Step { .. }) {
                    denoise_steps += 1;
                }
            })
            .expect("reference-conditioned render"),
    );

    let mut secondary_request = conditioned.clone();
    secondary_request.conditioning = vec![Conditioning::Reference {
        image: secondary.clone(),
        strength: None,
    }];
    let secondary_it2i = one_image(
        generator
            .generate(&secondary_request, &mut |_| {})
            .expect("same-shape secondary-reference render"),
    );

    let mut true_cfg_one_request = conditioned.clone();
    true_cfg_one_request.true_cfg = Some(1.0);
    let true_cfg_one_it2i = one_image(
        generator
            .generate(&true_cfg_one_request, &mut |_| {})
            .expect("same-reference true-CFG 1.0 render"),
    );

    // Exercise MultiReference independently of the singular shape. The second, visually disjoint
    // image makes a collapsed implementation (one that keeps only the first element) detectable.
    let mut multi_request = base;
    multi_request.true_cfg = Some(1.5);
    multi_request.conditioning = vec![Conditioning::MultiReference {
        images: vec![reference.clone(), secondary.clone()],
    }];
    let mut multi_steps = 0u32;
    let multi_it2i = one_image(
        generator
            .generate(&multi_request, &mut |progress| {
                if matches!(progress, Progress::Step { .. }) {
                    multi_steps += 1;
                }
            })
            .expect("multi-reference-conditioned render"),
    );

    assert_eq!(denoise_steps, STEPS);
    assert_eq!(multi_steps, STEPS);
    let spread = standard_deviation(&it2i.pixels);
    let delta = mean_abs_delta(&text_only.pixels, &it2i.pixels);
    let multi_spread = standard_deviation(&multi_it2i.pixels);
    let multi_t2i_delta = mean_abs_delta(&text_only.pixels, &multi_it2i.pixels);
    let multi_vs_single = mean_abs_delta(&it2i.pixels, &multi_it2i.pixels);
    let singular_pixel_delta = mean_abs_delta(&it2i.pixels, &secondary_it2i.pixels);
    let true_cfg_delta = mean_abs_delta(&it2i.pixels, &true_cfg_one_it2i.pixels);
    eprintln!("SenseNova it2i: std={spread:.2}, mean |Δ vs T2I|={delta:.2}");
    assert!(spread > 8.0, "conditioned output is near-flat: {spread:.2}");
    assert!(
        delta > 1.0,
        "conditioned output is indistinguishable from the unconditioned route: {delta:.3}"
    );
    assert!(
        singular_pixel_delta > 0.1,
        "same-shape singular renders ignored materially different reference pixels: \
         {singular_pixel_delta:.3}"
    );
    assert!(
        true_cfg_delta > 0.1,
        "identical-reference renders ignored true-CFG 1.0 vs 1.5: {true_cfg_delta:.3}"
    );
    eprintln!(
        "SenseNova multi-reference it2i: std={multi_spread:.2}, mean abs vs T2I=\
         {multi_t2i_delta:.2}, mean abs vs singular={multi_vs_single:.2}"
    );
    assert!(
        multi_spread > 8.0,
        "multi-reference output is near-flat: {multi_spread:.2}"
    );
    assert!(
        multi_t2i_delta > 1.0,
        "multi-reference output is indistinguishable from T2I: {multi_t2i_delta:.3}"
    );
    assert!(
        multi_vs_single > 0.1,
        "multi-reference output ignored its additional image: {multi_vs_single:.3}"
    );

    save(&reference, "sensenova_reference.png");
    save(&secondary, "sensenova_reference_secondary.png");
    save(&text_only, "sensenova_text_only.png");
    save(&it2i, "sensenova_it2i.png");
    save(&secondary_it2i, "sensenova_it2i_secondary_reference.png");
    save(&true_cfg_one_it2i, "sensenova_it2i_true_cfg_1_0.png");
    save(&multi_it2i, "sensenova_it2i_multi_reference.png");
}
