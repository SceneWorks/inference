//! Reproducible real-weight producer for the native FLUX.1 16-channel latent preview fit.
//!
//! Four diverse prompt/seed renders determine the OLS transform. Two disjoint renders are holdout
//! evidence and never contribute to its coefficients. Targets are 8×8-average-pooled native VAE
//! decodes. The fixed-seed replay also proves the legacy sampler wrapper and explicit inert latent
//! hook produce byte-identical final f32 latents and RGB8 output.
//!
//! ```sh
//! FLUX1_PREVIEW_SNAPSHOT=/path/to/flux1-dev/q4 \
//! FLUX1_PREVIEW_ARTIFACT_DIR=/path/to/artifacts \
//!   cargo test --release -p mlx-gen-flux --test integration fit_preview_rgb:: -- --ignored --nocapture
//! ```

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mlx_gen::array::scalar;
use mlx_gen::image::decoded_to_image;
use mlx_gen::{
    run_flow_sampler, run_flow_sampler_with_latent_hook, CancelFlag, Progress, TimestepConvention,
};
use mlx_gen_flux::{
    build_linear_sigmas, create_noise, load_clip_encoder, load_clip_tokenizer, load_t5_encoder,
    load_t5_tokenizer, load_transformer, load_vae, unpack_latents, FluxTextEncoders, FluxVariant,
};
use mlx_rs::ops::{add, maximum, mean_axes, minimum, multiply};
use mlx_rs::Dtype;

const FIT_CORPUS: [(&str, u64); 4] = [
    (
        "a chef at a night market, orange lanterns, teal reflections on wet pavement, rising steam",
        1663201,
    ),
    (
        "snowy alpine mountains at noon under a deep blue sky, dark evergreen forest",
        1663202,
    ),
    (
        "studio portrait of an elderly woman wearing a vivid red scarf, neutral gray background",
        1663203,
    ),
    (
        "graphic illustration of tropical fruit and green leaves on a pale pink table",
        1663204,
    ),
];
const HOLDOUT_CORPUS: [(&str, u64); 2] = [
    (
        "modern library at dusk, warm reading lamps and cool blue light through tall windows",
        1663291,
    ),
    (
        "macro photograph of purple wildflowers after rain, bright green stems and silver droplets",
        1663292,
    ),
];

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const STEPS: usize = 8;
const GUIDANCE: f32 = 3.5;
const CHANNELS: usize = 16;
const DIM: usize = CHANNELS + 1;

// Corpus-specific guardrails measured from this producer. The weakest fit channel is 0.97910; the
// weakest disjoint-holdout channel is 0.89133 and holdout overall is 0.92176. These floors permit
// minor backend drift while rejecting a materially degraded or incorrectly unpacked fit.
const FIT_CHANNEL_R2_FLOOR: f64 = 0.95;
const HOLDOUT_CHANNEL_R2_FLOOR: f64 = 0.85;
const HOLDOUT_OVERALL_R2_FLOOR: f64 = 0.88;

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("FLUX1_PREVIEW_SNAPSHOT")
            .unwrap_or_else(|_| panic!("set FLUX1_PREVIEW_SNAPSHOT to a real FLUX.1-dev snapshot")),
    )
}

struct Pair {
    latent: Vec<f64>,
    rgb: Vec<f64>,
    n: usize,
    decoded: mlx_gen::Image,
}

struct Components {
    t5_tokenizer: mlx_gen::tokenizer::TextTokenizer,
    clip_tokenizer: mlx_gen::tokenizer::TextTokenizer,
    text: FluxTextEncoders,
    transformer: mlx_gen_flux::FluxTransformer,
    vae: mlx_gen_z_image::vae::Vae,
}

impl Components {
    fn load(root: &Path) -> Self {
        let vae = load_vae(root).unwrap();
        Self {
            t5_tokenizer: load_t5_tokenizer(root, FluxVariant::Dev).unwrap(),
            clip_tokenizer: load_clip_tokenizer().unwrap(),
            text: FluxTextEncoders {
                t5: load_t5_encoder(root).unwrap(),
                clip: load_clip_encoder(root).unwrap(),
            },
            transformer: load_transformer(root, FluxVariant::Dev).unwrap(),
            vae,
        }
    }

    fn render_pair(&self, prompt: &str, seed: u64) -> Pair {
        self.render_pair_impl(prompt, seed, false)
    }

    fn render_pair_impl(&self, prompt: &str, seed: u64, explicit_inert_hook: bool) -> Pair {
        let (t5_ids, _) =
            mlx_gen::tokenizer::to_arrays(&self.t5_tokenizer.tokenize(prompt).unwrap());
        let (clip_ids, _) =
            mlx_gen::tokenizer::to_arrays(&self.clip_tokenizer.tokenize(prompt).unwrap());
        let (prompt_embeds, pooled) = self.text.encode(&t5_ids, &clip_ids).unwrap();
        let sigmas = build_linear_sigmas(STEPS, WIDTH, HEIGHT, true).unwrap();
        let init = create_noise(seed, WIDTH, HEIGHT).unwrap();
        let cancel = CancelFlag::default();
        let mut progress = |_progress: Progress| {};
        let predict = |latents: &mlx_rs::Array, sigma: f32| {
            self.transformer.forward(
                latents,
                &prompt_embeds,
                &pooled,
                sigma,
                GUIDANCE,
                WIDTH,
                HEIGHT,
            )
        };
        let packed = if explicit_inert_hook {
            run_flow_sampler_with_latent_hook(
                None,
                TimestepConvention::Sigma,
                &sigmas,
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
                &sigmas,
                init,
                seed,
                &cancel,
                &mut progress,
                predict,
            )
        }
        .unwrap();
        let raw = unpack_latents(&packed, WIDTH, HEIGHT)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let decoded_tensor = self
            .vae
            .decode(&raw)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let decoded = decoded_to_image(&decoded_tensor).unwrap();

        let lh = (HEIGHT / 8) as i32;
        let lw = (WIDTH / 8) as i32;
        let x = raw
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
    let mut comparison = vec![0u8; (WIDTH * 2 * HEIGHT * 3) as usize];
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            let left = (y * WIDTH as usize + x) * 3;
            let dst = (y * (WIDTH as usize * 2) + x) * 3;
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
#[ignore = "needs real FLUX.1-dev weights and a Metal-capable macOS host"]
fn fit_preview_rgb() {
    let components = Components::load(&snapshot());
    let fit_pairs: Vec<_> = FIT_CORPUS
        .iter()
        .enumerate()
        .map(|(index, (prompt, seed))| {
            eprintln!("rendering fit {index} seed {seed}: {prompt}");
            components.render_pair(prompt, *seed)
        })
        .collect();
    let coef = fit(&fit_pairs);
    let inert = components.render_pair_impl(FIT_CORPUS[0].0, FIT_CORPUS[0].1, true);
    assert_eq!(
        fit_pairs[0].latent, inert.latent,
        "inert hook changed final latent bytes"
    );
    assert_eq!(
        fit_pairs[0].decoded, inert.decoded,
        "inert hook changed final RGB8 bytes"
    );
    let holdout_pairs: Vec<_> = HOLDOUT_CORPUS
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
        "FLUX.1-dev Q4 preview OLS: 4 fit + 2 disjoint holdout, 256x256, 8-step flow Euler"
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
        "fixed-seed inert hook identity: exact final latent f32 and RGB8 bytes"
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
    if let Ok(path) = std::env::var("FLUX1_PREVIEW_ARTIFACT_DIR") {
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
