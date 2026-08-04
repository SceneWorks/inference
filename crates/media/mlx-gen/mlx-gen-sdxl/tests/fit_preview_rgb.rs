//! Reproducible real-weight producer for the SDXL-family four-channel latent-to-RGB preview fit.
//!
//! The fit corpus deliberately spans warm/cool palettes, indoor/outdoor lighting, people, objects,
//! landscapes, and illustration. Four prompt/seed renders determine the ordinary-least-squares
//! transform; two disjoint renders are a holdout and never contribute to the coefficients. Each
//! target is the average RGB colour of the decoded 8x8 pixel block represented by one NHWC latent.
//!
//! ```sh
//! SDXL_SNAPSHOT=/path/to/sdxl cargo test -p mlx-gen-sdxl --release \
//!   --test fit_preview_rgb -- --ignored --nocapture
//! ```
//!
//! Set `SDXL_PREVIEW_ARTIFACT_DIR` to retain decoded corpus images, latent-resolution fitted
//! previews, and the complete numerical report.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mlx_gen::{CancelFlag, PreviewSink};
use mlx_gen_sdxl::config::DiffusionConfig;
use mlx_gen_sdxl::sampler::AncestralEuler;
use mlx_gen_sdxl::{
    denoise, denoise_with_preview, encode_conditioning, load_text_encoder_1_dtype,
    load_text_encoder_2_dtype, load_tokenizer, load_unet_dtype, load_vae, seeded_prior,
    text_time_ids, Denoiser, EulerSampler,
};
use mlx_rs::ops::{add, maximum, mean_axes, minimum, multiply};
use mlx_rs::Dtype;

use mlx_gen::array::scalar;

const FIT_CORPUS: [(&str, u64); 4] = [
    (
        "a cinematic photograph of a chef cooking at a night market, orange lantern light, teal neon reflections on wet pavement, rising steam",
        1663301,
    ),
    (
        "a wide alpine landscape at noon, snow covered mountains, deep blue sky, dark evergreen forest, crisp cold sunlight",
        1663302,
    ),
    (
        "a studio portrait of an elderly woman wearing a vivid red scarf against a neutral gray background, soft window light, natural skin tones",
        1663303,
    ),
    (
        "a colorful editorial illustration of tropical fruit and green leaves on a pale pink table, flat graphic shapes, saturated palette",
        1663304,
    ),
];

const HOLDOUT_CORPUS: [(&str, u64); 2] = [
    (
        "a quiet modern library interior at dusk, rows of books, warm reading lamps, cool blue light through tall windows",
        1663391,
    ),
    (
        "macro photograph of purple wildflowers after rain, bright green stems, silvery droplets, dark blurred background",
        1663392,
    ),
];

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const STEPS: usize = 12;
const GUIDANCE: f32 = 5.0;
const CHANNELS: usize = 4;
const DIM: usize = CHANNELS + 1;
// Corpus-specific guardrails derived after measuring this producer, not borrowed from a different
// latent family. The weakest holdout channel measured 0.84844 and overall measured 0.86065; these
// floors leave room for small backend/weight-file numerical variation while rejecting a materially
// degraded fit.
const FIT_CHANNEL_R2_FLOOR: f64 = 0.88;
const HOLDOUT_CHANNEL_R2_FLOOR: f64 = 0.80;
const HOLDOUT_OVERALL_R2_FLOOR: f64 = 0.84;

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("SDXL_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set SDXL_SNAPSHOT to the required real-weight SDXL snapshot directory")
    }))
}

struct Pair {
    latent: Vec<f64>,
    rgb: Vec<f64>,
    n: usize,
    decoded: mlx_gen::Image,
}

struct Components {
    tokenizer: mlx_gen_sdxl::ClipBpeTokenizer,
    te1: mlx_gen_sdxl::ClipTextEncoder,
    te2: mlx_gen_sdxl::ClipTextEncoder,
    unet: mlx_gen_sdxl::UNet2DConditionModel,
    vae: mlx_gen_sdxl::Autoencoder,
    base_sampler: EulerSampler,
}

impl Components {
    fn load(root: &Path) -> Self {
        let dtype = Dtype::Float16;
        Self {
            tokenizer: load_tokenizer(root).unwrap(),
            te1: load_text_encoder_1_dtype(root, dtype).unwrap(),
            te2: load_text_encoder_2_dtype(root, dtype).unwrap(),
            unet: load_unet_dtype(root, dtype).unwrap(),
            vae: load_vae(root).unwrap(),
            base_sampler: EulerSampler::new_with_dtype(&DiffusionConfig::sdxl_base(), false, dtype)
                .unwrap(),
        }
    }

    fn render_pair(&self, prompt: &str, seed: u64) -> Pair {
        self.render_pair_impl(prompt, seed, false)
    }

    fn render_pair_impl(&self, prompt: &str, seed: u64, direct_preview_api: bool) -> Pair {
        let tokens = self.tokenizer.tokenize_batch(prompt, Some("")).unwrap();
        let (conditioning, pooled) = encode_conditioning(&self.te1, &self.te2, &tokens).unwrap();
        let time_ids = text_time_ids(2);
        let sampler =
            AncestralEuler::new(&self.base_sampler, STEPS, self.base_sampler.max_time()).unwrap();
        let init = seeded_prior(&self.base_sampler, seed, WIDTH, HEIGHT).unwrap();
        let denoiser = Denoiser::new(&self.unet, &sampler);
        let latents = if direct_preview_api {
            denoise_with_preview(
                &denoiser,
                init,
                &conditioning,
                &pooled,
                &time_ids,
                GUIDANCE,
                &CancelFlag::default(),
                &mut |_| {},
                &PreviewSink::default(),
            )
        } else {
            denoise(
                &denoiser,
                init,
                &conditioning,
                &pooled,
                &time_ids,
                GUIDANCE,
                &CancelFlag::default(),
                &mut |_| {},
            )
        }
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
        let decoded_tensor = self
            .vae
            .decode(&latents)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let decoded = mlx_gen_sdxl::decoded_to_image(&decoded_tensor).unwrap();

        let lh = (HEIGHT / 8) as i32;
        let lw = (WIDTH / 8) as i32;
        let x = latents.reshape(&[lh * lw, 4]).unwrap();

        let half = scalar(0.5);
        let y = add(multiply(&decoded_tensor, &half).unwrap(), &half).unwrap();
        let y = minimum(maximum(&y, scalar(0.0)).unwrap(), scalar(1.0)).unwrap();
        let y = y.reshape(&[1, lh, 8, lw, 8, 3]).unwrap();
        let y = mean_axes(&y, &[2, 4], false)
            .unwrap()
            .reshape(&[lh * lw, 3])
            .unwrap();

        Pair {
            latent: x.as_slice::<f32>().iter().map(|&v| v as f64).collect(),
            rgb: y.as_slice::<f32>().iter().map(|&v| v as f64).collect(),
            n: (lh * lw) as usize,
            decoded,
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn solve(mut a: [[f64; DIM]; DIM], mut b: [[f64; 3]; DIM]) -> [[f64; 3]; DIM] {
    for col in 0..DIM {
        let pivot = (col..DIM)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        let divisor = a[col][col];
        assert!(divisor.abs() > 1e-12, "singular fit at column {col}");
        for k in col..DIM {
            a[col][k] /= divisor;
        }
        for k in 0..3 {
            b[col][k] /= divisor;
        }
        for row in 0..DIM {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for k in col..DIM {
                a[row][k] -= factor * a[col][k];
            }
            for k in 0..3 {
                b[row][k] -= factor * b[col][k];
            }
        }
    }
    b
}

fn fit(pairs: &[Pair]) -> [[f64; 3]; DIM] {
    let mut xtx = [[0.0; DIM]; DIM];
    let mut xty = [[0.0; 3]; DIM];
    for pair in pairs {
        for sample in 0..pair.n {
            let mut row = [1.0; DIM];
            row[..CHANNELS]
                .copy_from_slice(&pair.latent[sample * CHANNELS..(sample + 1) * CHANNELS]);
            let y = &pair.rgb[sample * 3..sample * 3 + 3];
            for a in 0..DIM {
                for b in 0..DIM {
                    xtx[a][b] += row[a] * row[b];
                }
                for channel in 0..3 {
                    xty[a][channel] += row[a] * y[channel];
                }
            }
        }
    }
    solve(xtx, xty)
}

fn r2(pairs: &[Pair], coef: &[[f64; 3]; DIM]) -> ([f64; 3], f64) {
    let mut count = 0usize;
    let mut sum = [0.0; 3];
    for pair in pairs {
        count += pair.n;
        for y in pair.rgb.chunks_exact(3) {
            for channel in 0..3 {
                sum[channel] += y[channel];
            }
        }
    }
    let means = sum.map(|value| value / count as f64);
    let mut residual = [0.0; 3];
    let mut total = [0.0; 3];
    for pair in pairs {
        for sample in 0..pair.n {
            let x = &pair.latent[sample * CHANNELS..(sample + 1) * CHANNELS];
            for channel in 0..3 {
                let prediction = coef[CHANNELS][channel]
                    + x.iter()
                        .enumerate()
                        .map(|(i, value)| value * coef[i][channel])
                        .sum::<f64>();
                residual[channel] += (pair.rgb[sample * 3 + channel] - prediction).powi(2);
                total[channel] += (pair.rgb[sample * 3 + channel] - means[channel]).powi(2);
            }
        }
    }
    let per_channel = std::array::from_fn(|channel| 1.0 - residual[channel] / total[channel]);
    let overall = 1.0 - residual.iter().sum::<f64>() / total.iter().sum::<f64>();
    (per_channel, overall)
}

fn save_pair_artifacts(dir: &Path, split: &str, index: usize, pair: &Pair, coef: &[[f64; 3]; DIM]) {
    std::fs::create_dir_all(dir).unwrap();
    image::save_buffer(
        dir.join(format!("{split}_{index}_decoded.png")),
        &pair.decoded.pixels,
        pair.decoded.width,
        pair.decoded.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
    let pixels: Vec<u8> = pair
        .latent
        .chunks_exact(CHANNELS)
        .flat_map(|x| {
            (0..3).map(move |channel| {
                let value = coef[CHANNELS][channel]
                    + x.iter()
                        .enumerate()
                        .map(|(i, value)| value * coef[i][channel])
                        .sum::<f64>();
                (value.clamp(0.0, 1.0) * 255.0).round() as u8
            })
        })
        .collect();
    image::save_buffer(
        dir.join(format!("{split}_{index}_projected.png")),
        &pixels,
        WIDTH / 8,
        HEIGHT / 8,
        image::ColorType::Rgb8,
    )
    .unwrap();
}

#[test]
#[ignore = "needs real SDXL weights and a Metal-capable macOS host"]
fn fit_preview_rgb() {
    let components = Components::load(&snapshot());
    let mut fit_pairs = Vec::new();
    for (index, (prompt, seed)) in FIT_CORPUS.iter().enumerate() {
        eprintln!("rendering fit {index} seed {seed}: {prompt}");
        let pair = components.render_pair(prompt, *seed);
        if index == 0 {
            eprintln!("replaying fit 0 through explicit inert preview API for byte identity");
            let direct = components.render_pair_impl(prompt, *seed, true);
            assert_eq!(
                pair.latent, direct.latent,
                "fixed-seed legacy wrapper and explicit inert-preview route changed latent bytes"
            );
            assert_eq!(
                pair.decoded, direct.decoded,
                "fixed-seed legacy wrapper and explicit inert-preview route changed output bytes"
            );
        }
        fit_pairs.push(pair);
    }
    let coef = fit(&fit_pairs);

    let mut holdout_pairs = Vec::new();
    for (index, (prompt, seed)) in HOLDOUT_CORPUS.iter().enumerate() {
        eprintln!("rendering holdout {index} seed {seed}: {prompt}");
        holdout_pairs.push(components.render_pair(prompt, *seed));
    }

    let (fit_channels, fit_overall) = r2(&fit_pairs, &coef);
    let (holdout_channels, holdout_overall) = r2(&holdout_pairs, &coef);
    let mut report = String::new();
    writeln!(report, "SDXL preview OLS corpus: 4 fit + 2 holdout renders, 512x512, 12-step ancestral Euler, CFG 5.0").unwrap();
    writeln!(report, "fit seeds: {:?}", FIT_CORPUS.map(|(_, seed)| seed)).unwrap();
    writeln!(
        report,
        "holdout seeds: {:?}",
        HOLDOUT_CORPUS.map(|(_, seed)| seed)
    )
    .unwrap();
    writeln!(
        report,
        "fixed-seed inert PreviewSink identity: exact final latent and RGB8 bytes"
    )
    .unwrap();
    writeln!(
        report,
        "fit R2 RGB={fit_channels:?} overall={fit_overall:.8}"
    )
    .unwrap();
    writeln!(
        report,
        "holdout R2 RGB={holdout_channels:?} overall={holdout_overall:.8}"
    )
    .unwrap();
    writeln!(report, "const RGB_FACTORS: [[f32; 3]; 4] = [").unwrap();
    for row in &coef[..CHANNELS] {
        writeln!(report, "    [{:.9}, {:.9}, {:.9}],", row[0], row[1], row[2]).unwrap();
    }
    writeln!(
        report,
        "];\nconst RGB_BIAS: [f32; 3] = [{:.9}, {:.9}, {:.9}];",
        coef[4][0], coef[4][1], coef[4][2]
    )
    .unwrap();
    println!("{report}");

    assert!(fit_overall.is_finite() && holdout_overall.is_finite());
    assert!(
        fit_channels
            .into_iter()
            .all(|value| value >= FIT_CHANNEL_R2_FLOOR),
        "fit channel R² fell below {FIT_CHANNEL_R2_FLOOR}: {fit_channels:?}"
    );
    assert!(
        holdout_channels
            .into_iter()
            .all(|value| value >= HOLDOUT_CHANNEL_R2_FLOOR),
        "holdout channel R² fell below {HOLDOUT_CHANNEL_R2_FLOOR}: {holdout_channels:?}"
    );
    assert!(
        holdout_overall >= HOLDOUT_OVERALL_R2_FLOOR,
        "holdout overall R² {holdout_overall} fell below {HOLDOUT_OVERALL_R2_FLOOR}"
    );

    if let Ok(path) = std::env::var("SDXL_PREVIEW_ARTIFACT_DIR") {
        let dir = PathBuf::from(path);
        for (index, pair) in fit_pairs.iter().enumerate() {
            save_pair_artifacts(&dir, "fit", index, pair, &coef);
        }
        for (index, pair) in holdout_pairs.iter().enumerate() {
            save_pair_artifacts(&dir, "holdout", index, pair, &coef);
        }
        std::fs::write(dir.join("fit_report.txt"), report).unwrap();
    }
}
