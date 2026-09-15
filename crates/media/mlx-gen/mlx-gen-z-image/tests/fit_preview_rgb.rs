//! Reproducible real-weight producer for the Z-Image 16-channel latent-to-RGB preview fit.
//!
//! Four diverse prompt/seed renders determine the ordinary-least-squares transform. Two disjoint
//! renders are holdout evidence and never contribute to the coefficients. Each target is the average
//! RGB colour of the decoded 8×8 pixel block represented by one native Z-Image latent position.
//!
//! ```sh
//! MLX_GEN_ZIMAGE_SNAPSHOT=/path/to/z-image-turbo-mlx/bf16 \
//! ZIMAGE_PREVIEW_ARTIFACT_DIR=/path/to/artifacts \
//!   cargo test --release -p mlx-gen-z-image --test integration fit_preview_rgb:: -- --ignored --nocapture
//! ```

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mlx_gen::array::{host_i32, scalar};
use mlx_gen::{CancelFlag, FlowMatchEuler, PreviewSink};
use mlx_gen_z_image::{
    create_noise, decoded_to_image, denoise_with_progress, denoise_with_progress_and_preview,
    load_text_encoder, load_tokenizer, load_transformer, load_vae, slice_valid, unpack_latents,
};
use mlx_rs::ops::{add, maximum, mean_axes, minimum, multiply};
use mlx_rs::Dtype;

const FIT_CORPUS: [(&str, u64); 4] = [
    (
        "a chef at a night market, orange lanterns, teal reflections on wet pavement, rising steam",
        1663101,
    ),
    (
        "snowy alpine mountains at noon under a deep blue sky, dark evergreen forest",
        1663102,
    ),
    (
        "studio portrait of an elderly woman wearing a vivid red scarf, neutral gray background",
        1663103,
    ),
    (
        "graphic illustration of tropical fruit and green leaves on a pale pink table",
        1663104,
    ),
];
const HOLDOUT_CORPUS: [(&str, u64); 2] = [
    (
        "modern library at dusk, warm reading lamps and cool blue light through tall windows",
        1663191,
    ),
    (
        "macro photograph of purple wildflowers after rain, bright green stems and silver droplets",
        1663192,
    ),
];

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const STEPS: usize = 8;
const SCHEDULE_SHIFT: f32 = 3.0;
const CHANNELS: usize = 16;
const DIM: usize = CHANNELS + 1;

// Corpus-specific guardrails measured from this producer. The weakest fit channel is 0.97883; the
// weakest disjoint-holdout channel is 0.89464 and holdout overall is 0.92827. These floors allow
// small backend precision drift while rejecting a materially degraded or incorrectly ordered fit.
const FIT_CHANNEL_R2_FLOOR: f64 = 0.95;
const HOLDOUT_CHANNEL_R2_FLOOR: f64 = 0.86;
const HOLDOUT_OVERALL_R2_FLOOR: f64 = 0.90;

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("MLX_GEN_ZIMAGE_SNAPSHOT").unwrap_or_else(|_| {
            panic!(
                "set MLX_GEN_ZIMAGE_SNAPSHOT to the required SceneWorks/z-image-turbo-mlx tier dir"
            )
        }),
    )
}

struct Pair {
    latent: Vec<f64>,
    rgb: Vec<f64>,
    n: usize,
    decoded: mlx_gen::Image,
}

struct Components {
    tokenizer: mlx_gen::tokenizer::TextTokenizer,
    text_encoder: mlx_gen_z_image::text_encoder::TextEncoder,
    transformer: mlx_gen_z_image::ZImageTransformer,
    vae: mlx_gen_z_image::vae::Vae,
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

    fn encode(&self, prompt: &str) -> mlx_rs::Array {
        let tokens = self.tokenizer.tokenize(prompt).unwrap();
        let (ids, mask) = mlx_gen::tokenizer::to_arrays(&tokens);
        let valid: i32 = host_i32(&mask).unwrap().iter().sum();
        let encoded = self.text_encoder.forward(&ids, &mask).unwrap();
        slice_valid(&encoded, valid)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    }

    fn render_pair(&self, prompt: &str, seed: u64) -> Pair {
        self.render_pair_impl(prompt, seed, false)
    }

    fn render_pair_impl(&self, prompt: &str, seed: u64, explicit_inert: bool) -> Pair {
        let cap = self.encode(prompt);
        let scheduler = FlowMatchEuler::for_static_shift(STEPS, SCHEDULE_SHIFT);
        let init = create_noise(seed, WIDTH, HEIGHT)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let cancel = CancelFlag::default();
        let native = if explicit_inert {
            denoise_with_progress_and_preview(
                &self.transformer,
                &scheduler,
                None,
                seed,
                init,
                &cap,
                0,
                mlx_gen::attention::AttentionBudget::UNBOUNDED,
                None,
                &cancel,
                &PreviewSink::default(),
                &mut |_| {},
            )
        } else {
            denoise_with_progress(
                &self.transformer,
                &scheduler,
                None,
                seed,
                init,
                &cap,
                0,
                mlx_gen::attention::AttentionBudget::UNBOUNDED,
                None,
                &cancel,
                &mut |_| {},
            )
        }
        .unwrap();
        let unpacked = unpack_latents(&native)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let decoded_tensor = self
            .vae
            .decode(&unpacked)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let decoded = decoded_to_image(&decoded_tensor).unwrap();

        let lh = (HEIGHT / 8) as i32;
        let lw = (WIDTH / 8) as i32;
        let x = unpacked
            .reshape(&[CHANNELS as i32, lh * lw])
            .unwrap()
            .transpose_axes(&[1, 0])
            .unwrap()
            // `as_slice` exposes physical storage for a transpose view. Flattening materializes the
            // logical `[position, channel]` order before host-side OLS accumulation.
            .reshape(&[lh * lw * CHANNELS as i32])
            .unwrap();
        let decoded4 = decoded_tensor.squeeze_axes(&[2]).unwrap();
        let half = scalar(0.5);
        let y = add(multiply(&decoded4, &half).unwrap(), &half).unwrap();
        let y = minimum(maximum(&y, scalar(0.0)).unwrap(), scalar(1.0)).unwrap();
        let y = mean_axes(y.reshape(&[1, 3, lh, 8, lw, 8]).unwrap(), &[3, 5], false)
            .unwrap()
            .reshape(&[3, lh * lw])
            .unwrap()
            .transpose_axes(&[1, 0])
            .unwrap()
            .reshape(&[lh * lw * 3])
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
            let left = (y * WIDTH as usize * 2 + x) * 3;
            let decoded = (y * WIDTH as usize + x) * 3;
            comparison[left..left + 3].copy_from_slice(&pair.decoded.pixels[decoded..decoded + 3]);
            let right = (y * WIDTH as usize * 2 + WIDTH as usize + x) * 3;
            let projected_pixel = ((y / 8) * (WIDTH as usize / 8) + x / 8) * 3;
            comparison[right..right + 3]
                .copy_from_slice(&projected[projected_pixel..projected_pixel + 3]);
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
#[ignore = "needs real Z-Image-Turbo weights and a Metal-capable macOS host"]
fn fit_preview_rgb() {
    mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
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
        "inert preview changed latent bytes"
    );
    assert_eq!(
        fit_pairs[0].decoded, inert.decoded,
        "inert preview changed RGB8 bytes"
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
    writeln!(report, "Z-Image preview OLS corpus: 4 fit + 2 disjoint holdout renders, 256x256, 8-step static-shift-3 flow Euler").unwrap();
    writeln!(report, "snapshot: SceneWorks/z-image-turbo-mlx revision bb2bc9893b3c49ae96c813350775f791a2e8bc80 tier bf16").unwrap();
    writeln!(report, "transformer/model.safetensors: 12309874234 bytes, SHA256 b7e2a1579aad3e0044cbf4863b15aadb535c9a311a90a06349f0790d2043da33").unwrap();
    writeln!(report, "text_encoder/model.safetensors: 8044981933 bytes, SHA256 7a9d609e583a82be1dad882544bb2366c1b35ea22864cb2312fb307589d66deb").unwrap();
    writeln!(report, "vae/model.safetensors: 167666968 bytes, SHA256 0fbab8b661f6ee6af81c88a6eb1501ec1f7b4b8fe4ad29803507ebe0cf863810").unwrap();
    writeln!(report, "tokenizer/tokenizer.json: 11422654 bytes, SHA256 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4").unwrap();
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

    if let Ok(path) = std::env::var("ZIMAGE_PREVIEW_ARTIFACT_DIR") {
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
