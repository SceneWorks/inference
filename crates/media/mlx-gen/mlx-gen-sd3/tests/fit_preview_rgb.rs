//! Reproducible real-weight producer for SD3.5's native 16-channel latent preview fit.
//!
//! Four diverse prompt/seed renders determine the OLS transform. Two disjoint renders are holdout
//! evidence and never contribute to its coefficients. Targets are 8x8-average-pooled native VAE
//! decodes. Text conditioning is materialized and its encoders dropped before the MMDiT/VAE load,
//! matching the provider's sequential residency discipline. The fixed-seed replay also proves the
//! legacy public wrapper and explicit inert-preview API produce exact final f32 latents and RGB8.
//!
//! ```sh
//! SD3_PREVIEW_SNAPSHOT=/path/to/stable-diffusion-3.5-large \
//! SD3_PREVIEW_ARTIFACT_DIR=/path/to/artifacts \
//!   cargo test --release -p mlx-gen-sd3 --test fit_preview_rgb -- --ignored --nocapture
//! ```

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mlx_gen::array::scalar;
use mlx_gen::{CancelFlag, FlowMatchEuler, PreviewSink};
use mlx_rs::ops::{add, maximum, mean_axes, minimum, multiply};
use mlx_rs::Dtype;

use mlx_gen_sd3::loader;
use mlx_gen_sd3::pipeline;
use mlx_gen_sd3::{Sd3Conditioning, Sd3Variant};

const FIT_CORPUS: [(&str, u64); 4] = [
    (
        "a chef cooking at a night market, orange lanterns, teal reflections on wet pavement, rising steam",
        1663401,
    ),
    (
        "snow covered alpine mountains at noon, deep blue sky, dark evergreen forest, crisp cold sunlight",
        1663402,
    ),
    (
        "studio portrait of an elderly woman wearing a vivid red scarf, neutral gray background, natural skin tones",
        1663403,
    ),
    (
        "editorial illustration of tropical fruit and green leaves on a pale pink table, flat saturated shapes",
        1663404,
    ),
];
const HOLDOUT_CORPUS: [(&str, u64); 2] = [
    (
        "modern library at dusk, warm reading lamps and cool blue light through tall windows",
        1663491,
    ),
    (
        "macro photograph of purple wildflowers after rain, bright green stems and silver droplets",
        1663492,
    ),
];

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const STEPS: usize = 8;
const GUIDANCE: f32 = 3.5;
const CHANNELS: usize = 16;
const DIM: usize = CHANNELS + 1;

// Corpus-specific guardrails are updated to the measured values produced by this test before the
// fitted constants are committed. They must reject a materially degraded or incorrectly laid-out
// fit while allowing small backend/weight-file numerical variation.
const FIT_CHANNEL_R2_FLOOR: f64 = 0.95;
const HOLDOUT_CHANNEL_R2_FLOOR: f64 = 0.84;
const HOLDOUT_OVERALL_R2_FLOOR: f64 = 0.88;

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("SD3_PREVIEW_SNAPSHOT")
            .unwrap_or_else(|_| panic!("set SD3_PREVIEW_SNAPSHOT to real SD3.5-Large weights")),
    )
}

struct Pair {
    latent: Vec<f64>,
    rgb: Vec<f64>,
    n: usize,
    decoded: mlx_gen::Image,
}

struct Heavy {
    transformer: mlx_gen_sd3::Sd3Transformer,
    vae: mlx_gen_z_image::vae::Vae,
    uncond: Sd3Conditioning,
}

impl Heavy {
    fn render_pair(&self, cond: &Sd3Conditioning, seed: u64) -> Pair {
        self.render_pair_impl(cond, seed, false)
    }

    fn render_pair_impl(
        &self,
        cond: &Sd3Conditioning,
        seed: u64,
        explicit_inert_preview: bool,
    ) -> Pair {
        let scheduler = FlowMatchEuler::for_static_shift(STEPS, pipeline::SCHEDULE_SHIFT);
        let init = pipeline::create_noise(seed, WIDTH, HEIGHT).unwrap();
        let cancel = CancelFlag::default();
        let mut progress = |_| {};
        let latents = if explicit_inert_preview {
            pipeline::denoise_cfg_with_preview(
                &self.transformer,
                &scheduler,
                None,
                seed,
                init,
                cond,
                Some(&self.uncond),
                GUIDANCE,
                &cancel,
                &mut progress,
                &PreviewSink::default(),
            )
        } else {
            pipeline::denoise_cfg(
                &self.transformer,
                &scheduler,
                None,
                seed,
                init,
                cond,
                Some(&self.uncond),
                GUIDANCE,
                &cancel,
                &mut progress,
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
        let decoded = mlx_gen::image::decoded_to_image(&decoded_tensor).unwrap();

        let lh = (HEIGHT / 8) as i32;
        let lw = (WIDTH / 8) as i32;
        let x = latents
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap()
            .reshape(&[lh * lw, CHANNELS as i32])
            .unwrap();
        let decoded4 = decoded_tensor.squeeze_axes(&[2]).unwrap();
        let y = add(multiply(&decoded4, scalar(0.5)).unwrap(), scalar(0.5)).unwrap();
        let y = minimum(maximum(&y, scalar(0.0)).unwrap(), scalar(1.0)).unwrap();
        let y = mean_axes(y.reshape(&[1, 3, lh, 8, lw, 8]).unwrap(), &[3, 5], false)
            .unwrap()
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap()
            .reshape(&[lh * lw, 3])
            .unwrap();

        Pair {
            latent: x
                .as_slice::<f32>()
                .iter()
                .map(|&value| value as f64)
                .collect(),
            rgb: y
                .as_slice::<f32>()
                .iter()
                .map(|&value| value as f64)
                .collect(),
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
        for channel in 0..3 {
            b[col][channel] /= divisor;
        }
        for row in 0..DIM {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for k in col..DIM {
                a[row][k] -= factor * a[col][k];
            }
            for channel in 0..3 {
                b[row][channel] -= factor * b[col][channel];
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
    let count = pairs.iter().map(|pair| pair.n).sum::<usize>();
    let mut sum = [0.0; 3];
    for pair in pairs {
        for y in pair.rgb.chunks_exact(3) {
            for channel in 0..3 {
                sum[channel] += y[channel];
            }
        }
    }
    let mean = sum.map(|value| value / count as f64);
    let mut residual = [0.0; 3];
    let mut total = [0.0; 3];
    for pair in pairs {
        for sample in 0..pair.n {
            let x = &pair.latent[sample * CHANNELS..(sample + 1) * CHANNELS];
            for channel in 0..3 {
                let prediction = coef[CHANNELS][channel]
                    + x.iter()
                        .enumerate()
                        .map(|(index, value)| value * coef[index][channel])
                        .sum::<f64>();
                let target = pair.rgb[sample * 3 + channel];
                residual[channel] += (target - prediction).powi(2);
                total[channel] += (target - mean[channel]).powi(2);
            }
        }
    }
    let channels = std::array::from_fn(|channel| 1.0 - residual[channel] / total[channel]);
    let overall = 1.0 - residual.iter().sum::<f64>() / total.iter().sum::<f64>();
    (channels, overall)
}

fn save_artifacts(dir: &Path, split: &str, index: usize, pair: &Pair, coef: &[[f64; 3]; DIM]) {
    std::fs::create_dir_all(dir).unwrap();
    image::save_buffer(
        dir.join(format!("{split}_{index}_decoded.png")),
        &pair.decoded.pixels,
        pair.decoded.width,
        pair.decoded.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
    let projected: Vec<u8> = pair
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
        &projected,
        WIDTH / 8,
        HEIGHT / 8,
        image::ColorType::Rgb8,
    )
    .unwrap();
    let mut comparison = vec![0u8; (WIDTH * 2 * HEIGHT * 3) as usize];
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            let left = (y * WIDTH as usize + x) * 3;
            let dst = (y * WIDTH as usize * 2 + x) * 3;
            comparison[dst..dst + 3].copy_from_slice(&pair.decoded.pixels[left..left + 3]);
            let src = ((y / 8) * (WIDTH as usize / 8) + x / 8) * 3;
            let right = dst + WIDTH as usize * 3;
            comparison[right..right + 3].copy_from_slice(&projected[src..src + 3]);
        }
    }
    image::save_buffer(
        dir.join(format!("{split}_{index}_comparison.png")),
        &comparison,
        WIDTH * 2,
        HEIGHT,
        image::ColorType::Rgb8,
    )
    .unwrap();
}

#[test]
#[ignore = "needs real SD3.5-Large weights and a Metal-capable macOS host"]
fn fit_preview_rgb() {
    let root = snapshot();
    let clip_tokenizer = loader::load_clip_tokenizer(&root).unwrap();
    let clip_pad = loader::load_clip_pad_ids(&root).unwrap();
    let t5_tokenizer = loader::load_t5_tokenizer(&root).unwrap();
    let encoders = loader::load_text_encoders(&root).unwrap();
    let prompts: Vec<_> = FIT_CORPUS
        .iter()
        .chain(HOLDOUT_CORPUS.iter())
        .map(|(prompt, _)| {
            pipeline::encode_prompt(&encoders, &clip_tokenizer, clip_pad, &t5_tokenizer, prompt)
                .unwrap()
        })
        .collect();
    let uncond =
        pipeline::encode_prompt(&encoders, &clip_tokenizer, clip_pad, &t5_tokenizer, "").unwrap();
    let mut arrays = Vec::with_capacity(prompts.len() * 2 + 2);
    for cond in &prompts {
        arrays.push(&cond.context);
        arrays.push(&cond.pooled);
    }
    arrays.push(&uncond.context);
    arrays.push(&uncond.pooled);
    mlx_rs::transforms::eval(arrays).unwrap();
    drop(encoders);
    mlx_rs::memory::clear_cache();

    let heavy = Heavy {
        transformer: loader::load_transformer(&root, &Sd3Variant::Large.arch()).unwrap(),
        vae: loader::load_vae(&root).unwrap(),
        uncond,
    };
    let fit_pairs: Vec<_> = FIT_CORPUS
        .iter()
        .enumerate()
        .map(|(index, (prompt, seed))| {
            eprintln!("rendering fit {index} seed {seed}: {prompt}");
            heavy.render_pair(&prompts[index], *seed)
        })
        .collect();
    let inert = heavy.render_pair_impl(&prompts[0], FIT_CORPUS[0].1, true);
    assert_eq!(
        fit_pairs[0].latent, inert.latent,
        "inert preview changed final latent bytes"
    );
    assert_eq!(
        fit_pairs[0].decoded, inert.decoded,
        "inert preview changed final RGB8 bytes"
    );
    let holdout_pairs: Vec<_> = HOLDOUT_CORPUS
        .iter()
        .enumerate()
        .map(|(index, (prompt, seed))| {
            eprintln!("rendering holdout {index} seed {seed}: {prompt}");
            heavy.render_pair(&prompts[FIT_CORPUS.len() + index], *seed)
        })
        .collect();

    let coef = fit(&fit_pairs);
    let (fit_channels, fit_overall) = r2(&fit_pairs, &coef);
    let (holdout_channels, holdout_overall) = r2(&holdout_pairs, &coef);
    let mut report = String::new();
    writeln!(report, "SD3.5-Large preview OLS: 4 fit + 2 disjoint holdout, 256x256, 8-step static-shift-3 flow Euler, CFG 3.5").unwrap();
    writeln!(report, "fit seeds: {:?}", FIT_CORPUS.map(|(_, seed)| seed)).unwrap();
    writeln!(
        report,
        "holdout seeds: {:?}",
        HOLDOUT_CORPUS.map(|(_, seed)| seed)
    )
    .unwrap();
    writeln!(
        report,
        "fixed-seed inert PreviewSink identity: exact final latent f32 and RGB8 bytes"
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
    writeln!(report, "const RGB_FACTORS: [[f32; 3]; 16] = [").unwrap();
    for row in &coef[..CHANNELS] {
        writeln!(report, "    [{:.9}, {:.9}, {:.9}],", row[0], row[1], row[2]).unwrap();
    }
    writeln!(
        report,
        "];\nconst RGB_BIAS: [f32; 3] = [{:.9}, {:.9}, {:.9}];",
        coef[CHANNELS][0], coef[CHANNELS][1], coef[CHANNELS][2]
    )
    .unwrap();
    println!("{report}");

    assert!(fit_channels
        .iter()
        .all(|value| *value >= FIT_CHANNEL_R2_FLOOR));
    assert!(holdout_channels
        .iter()
        .all(|value| *value >= HOLDOUT_CHANNEL_R2_FLOOR));
    assert!(holdout_overall >= HOLDOUT_OVERALL_R2_FLOOR);
    if let Ok(path) = std::env::var("SD3_PREVIEW_ARTIFACT_DIR") {
        let dir = PathBuf::from(path);
        for (index, pair) in fit_pairs.iter().enumerate() {
            save_artifacts(&dir, "fit", index, pair, &coef);
        }
        for (index, pair) in holdout_pairs.iter().enumerate() {
            save_artifacts(&dir, "holdout", index, pair, &coef);
        }
        std::fs::write(dir.join("fit_report.txt"), report).unwrap();
    }
}
