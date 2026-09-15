//! Reproducible real-weight producer for the FLUX.2-family true 32-channel latent preview fit.
//!
//! ```sh
//! MLX_GEN_FLUX2_SNAPSHOT=/path/to/snapshot cargo test --release -p mlx-gen-flux2 \
//!   --test integration -- fit_preview_rgb:: --ignored --nocapture
//! ```
//!
//! Eight diverse prompt/seed renders determine the OLS transform. Four disjoint renders are holdout
//! evidence and never contribute to its coefficients. Targets are average-pooled VAE decodes at
//! the raw latent's 8x spatial scale. Set `FLUX2_PREVIEW_ARTIFACT_DIR` to retain the corpus and
//! numerical report.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mlx_gen::image::decoded_to_image;
use mlx_gen::{
    run_flow_sampler, run_flow_sampler_with_latent_hook, CancelFlag, Progress, TimestepConvention,
};
use mlx_gen_flux2::{
    create_noise, load_text_encoder, load_tokenizer, load_transformer, load_vae, prepare_grid_ids,
    prepare_text_ids, schedule, Flux2Vae,
};
use mlx_rs::ops::{add, maximum, mean_axes, minimum, multiply};
use mlx_rs::Dtype;

use mlx_gen::array::scalar;

const FIT_CORPUS: [(&str, u64); 8] = [
    (
        "a chef at a night market, orange lanterns, teal reflections, rising steam",
        1663001,
    ),
    (
        "snowy alpine mountains under a deep blue sky, dark evergreen forest",
        1663002,
    ),
    (
        "studio portrait of an elderly woman wearing a vivid red scarf, gray background",
        1663003,
    ),
    (
        "graphic illustration of tropical fruit and green leaves on a pale pink table",
        1663004,
    ),
    (
        "a bright yellow tram crossing a rainy city street at blue hour",
        1663005,
    ),
    (
        "turquoise ocean waves breaking beside white cliffs in midday sun",
        1663006,
    ),
    (
        "a cozy reading room with walnut shelves, amber lamps, and a blue armchair",
        1663007,
    ),
    (
        "a black cat in a lush garden of white and magenta flowers",
        1663008,
    ),
];
const HOLDOUT_CORPUS: [(&str, u64); 4] = [
    (
        "modern library at dusk, warm lamps and cool blue window light",
        1663091,
    ),
    (
        "macro photograph of purple wildflowers after rain, bright green stems",
        1663092,
    ),
    (
        "a golden retriever running on a pale beach beneath a cloudy sky",
        1663093,
    ),
    (
        "a glossy red sports car parked under neon signs on a wet night",
        1663094,
    ),
];

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const STEPS: usize = 8;
const CHANNELS: usize = 32;
const DIM: usize = CHANNELS + 1;
// Corpus-specific guardrails measured from this producer. The weakest fit channel is 0.74224; the
// weakest disjoint-holdout channel is 0.64247 and holdout overall is 0.69504. These floors allow
// small backend precision drift while rejecting a materially degraded or accidentally packed fit.
const FIT_CHANNEL_R2_FLOOR: f64 = 0.70;
const HOLDOUT_CHANNEL_R2_FLOOR: f64 = 0.58;
const HOLDOUT_OVERALL_R2_FLOOR: f64 = 0.65;

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("MLX_GEN_FLUX2_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set MLX_GEN_FLUX2_SNAPSHOT to the required real FLUX.2 snapshot")
    }))
}

struct Pair {
    latent: Vec<f64>,
    rgb: Vec<f64>,
    n: usize,
    decoded: mlx_gen::Image,
}

struct Components {
    tokenizer: mlx_gen::tokenizer::TextTokenizer,
    text_encoder: mlx_gen_flux2::Qwen3TextEncoder,
    transformer: mlx_gen_flux2::Flux2Transformer,
    vae: Flux2Vae,
}

impl Components {
    fn load(root: &Path) -> Self {
        Self {
            tokenizer: load_tokenizer(root).unwrap(),
            text_encoder: load_text_encoder(root).unwrap(),
            transformer: load_transformer(root).unwrap(),
            vae: load_vae(root).unwrap(),
        }
    }

    fn render_pair(&self, prompt: &str, seed: u64) -> Pair {
        self.render_pair_impl(prompt, seed, false)
    }

    fn render_pair_impl(&self, prompt: &str, seed: u64, explicit_inert_preview: bool) -> Pair {
        let tokens = self.tokenizer.tokenize(prompt).unwrap();
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tokens);
        let embeds = self
            .text_encoder
            .prompt_embeds(&input_ids, &attention_mask)
            .unwrap();
        let text_ids = prepare_text_ids(embeds.shape()[1] as usize);
        let latent_h = (HEIGHT / 16) as usize;
        let latent_w = (WIDTH / 16) as usize;
        let latent_ids = prepare_grid_ids(latent_h, latent_w, 0);
        let sched = schedule(STEPS, WIDTH, HEIGHT);
        let init = create_noise(seed, WIDTH, HEIGHT, 128).unwrap();
        let cancel = CancelFlag::default();
        let mut progress = |_progress: Progress| {};
        let predict = |latents: &mlx_rs::Array, sigma: f32| {
            self.transformer
                .forward(latents, &embeds, &latent_ids, &text_ids, sigma * 1000.0)
        };
        let packed_tokens = if explicit_inert_preview {
            run_flow_sampler_with_latent_hook(
                None,
                TimestepConvention::Sigma,
                &sched.sigmas,
                init,
                seed,
                &cancel,
                &mut progress,
                |_, _| {},
                predict,
            )
        } else {
            run_flow_sampler(
                None,
                TimestepConvention::Sigma,
                &sched.sigmas,
                init,
                seed,
                &cancel,
                &mut progress,
                predict,
            )
        }
        .unwrap();
        let packed = packed_tokens
            .reshape(&[1, latent_h as i32, latent_w as i32, 128])
            .unwrap();
        let raw = self
            .vae
            .unpack_flux_packed_latents(&packed)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let decoded_tensor = self
            .vae
            .decode(&raw)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let decoded =
            decoded_to_image(&decoded_tensor.transpose_axes(&[0, 3, 1, 2]).unwrap()).unwrap();

        let lh = (HEIGHT / 8) as i32;
        let lw = (WIDTH / 8) as i32;
        let x = raw.reshape(&[lh * lw, CHANNELS as i32]).unwrap();
        let half = scalar(0.5);
        let y = add(multiply(&decoded_tensor, &half).unwrap(), &half).unwrap();
        let y = minimum(maximum(&y, scalar(0.0)).unwrap(), scalar(1.0)).unwrap();
        let y = mean_axes(y.reshape(&[1, lh, 8, lw, 8, 3]).unwrap(), &[2, 4], false)
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

    // Decoded target beside an 8× nearest-neighbour enlargement of the latent projection.
    let mut comparison = vec![0u8; (WIDTH * 2 * HEIGHT * 3) as usize];
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            let dst_left = (y * (WIDTH as usize * 2) + x) * 3;
            let src_left = (y * WIDTH as usize + x) * 3;
            comparison[dst_left..dst_left + 3]
                .copy_from_slice(&pair.decoded.pixels[src_left..src_left + 3]);
            let dst_right = (y * (WIDTH as usize * 2) + WIDTH as usize + x) * 3;
            let src_right = ((y / 8) * (WIDTH as usize / 8) + (x / 8)) * 3;
            comparison[dst_right..dst_right + 3]
                .copy_from_slice(&projected[src_right..src_right + 3]);
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
#[ignore = "needs real FLUX.2-klein-9b weights and a Metal-capable macOS host"]
fn fit_preview_rgb() {
    mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    let components = Components::load(&snapshot());
    let fit_pairs: Vec<Pair> = FIT_CORPUS
        .iter()
        .enumerate()
        .map(|(index, (prompt, seed))| {
            eprintln!("rendering fit {index} seed {seed}: {prompt}");
            components.render_pair(prompt, *seed)
        })
        .collect();
    let coef = fit(&fit_pairs);
    let inert_replay = components.render_pair_impl(FIT_CORPUS[0].0, FIT_CORPUS[0].1, true);
    assert_eq!(
        fit_pairs[0].latent, inert_replay.latent,
        "fixed-seed legacy and explicit inert-preview routes changed final latent bytes"
    );
    assert_eq!(
        fit_pairs[0].decoded, inert_replay.decoded,
        "fixed-seed legacy and explicit inert-preview routes changed final RGB8 bytes"
    );
    let holdout_pairs: Vec<Pair> = HOLDOUT_CORPUS
        .iter()
        .enumerate()
        .map(|(index, (prompt, seed))| {
            eprintln!("rendering holdout {index} seed {seed}: {prompt}");
            components.render_pair(prompt, *seed)
        })
        .collect();
    let (fit_channels, fit_overall) = r2(&fit_pairs, &coef);
    let (holdout_channels, holdout_overall) = r2(&holdout_pairs, &coef);

    let mut report = String::new();
    writeln!(
        report,
        "FLUX.2 preview OLS corpus: 8 fit + 4 disjoint holdout renders, 256x256, 8-step flow Euler"
    )
    .unwrap();
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
    writeln!(report, "const RGB_FACTORS: [[f32; 3]; 32] = [").unwrap();
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

    assert!(fit_overall.is_finite() && holdout_overall.is_finite());
    assert!(
        fit_channels
            .iter()
            .all(|value| *value >= FIT_CHANNEL_R2_FLOOR),
        "fit channel R² fell below {FIT_CHANNEL_R2_FLOOR}: {fit_channels:?}"
    );
    assert!(
        holdout_channels
            .iter()
            .all(|value| *value >= HOLDOUT_CHANNEL_R2_FLOOR),
        "holdout channel R² fell below {HOLDOUT_CHANNEL_R2_FLOOR}: {holdout_channels:?}"
    );
    assert!(
        holdout_overall >= HOLDOUT_OVERALL_R2_FLOOR,
        "holdout overall R² {holdout_overall} fell below {HOLDOUT_OVERALL_R2_FLOOR}"
    );

    if let Ok(path) = std::env::var("FLUX2_PREVIEW_ARTIFACT_DIR") {
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
