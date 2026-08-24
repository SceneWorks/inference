//! Reproducible real-weight producer for SANA's native 32-channel DC-AE latent preview fit.
//!
//! Four diverse prompt/seed renders determine the OLS transform. Two disjoint renders are holdout
//! evidence and never contribute to its coefficients. Targets are native DC-AE decodes average-
//! pooled by the actual `SPATIAL_SCALE = 32`. Text conditioning is materialized and its encoder
//! dropped before the Linear-DiT/DC-AE load, matching sequential residency. A fixed-seed replay also
//! proves the legacy wrapper and explicit inert-preview API produce exact final f32 latents and RGB8.
//!
//! ```sh
//! SANA_PREVIEW_VARIANT=base \
//! SANA_PREVIEW_SNAPSHOT=/path/to/sana/tier \
//! SANA_PREVIEW_ARTIFACT_DIR=/path/to/artifacts \
//!   cargo test --release -p mlx-gen-sana --test integration fit_preview_rgb:: -- --ignored --nocapture
//! ```

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mlx_gen::array::scalar;
use mlx_gen::image::decoded_to_image;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, FlowMatchEuler, PreviewSink};
use mlx_rs::ops::{add, divide, maximum, mean_axes, minimum, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen_sana::pipeline::{self, SPATIAL_SCALE};
use mlx_gen_sana::{
    DcAeConfig, DcAeDecoder, SanaTextEncoder, SanaTransformer, SanaTransformerConfig,
};

const FIT_CORPUS: [(&str, u64); 4] = [
    (
        "a chef cooking at a night market, orange lanterns, teal reflections on wet pavement, rising steam",
        1663501,
    ),
    (
        "snow covered alpine mountains at noon, deep blue sky, dark evergreen forest, crisp cold sunlight",
        1663502,
    ),
    (
        "studio portrait of an elderly woman wearing a vivid red scarf, neutral gray background, natural skin tones",
        1663503,
    ),
    (
        "editorial illustration of tropical fruit and green leaves on a pale pink table, flat saturated shapes",
        1663504,
    ),
];
const HOLDOUT_CORPUS: [(&str, u64); 2] = [
    (
        "modern library at dusk, warm reading lamps and cool blue light through tall windows",
        1663591,
    ),
    (
        "macro photograph of purple wildflowers after rain, bright green stems and silver droplets",
        1663592,
    ),
];

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const BASE_STEPS: usize = 8;
const SPRINT_STEPS: usize = 4;
const GUIDANCE: f32 = 4.5;
const CHANNELS: usize = 32;
const DIM: usize = CHANNELS + 1;

const FIT_CHANNEL_R2_FLOOR: f64 = 0.92;
const HOLDOUT_CHANNEL_R2_FLOOR: f64 = 0.86;
const HOLDOUT_OVERALL_R2_FLOOR: f64 = 0.88;
const PER_IMAGE_CHANNEL_R2_FLOOR: f64 = 0.70;
const PER_IMAGE_OVERALL_R2_FLOOR: f64 = 0.85;
const PER_IMAGE_TARGET_VARIANCE_FLOOR: f64 = 0.005;

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("SANA_PREVIEW_SNAPSHOT")
            .unwrap_or_else(|_| panic!("set SANA_PREVIEW_SNAPSHOT to a real SANA tier")),
    )
}

struct Pair {
    latent: Vec<f64>,
    rgb: Vec<f64>,
    n: usize,
    decoded: mlx_gen::Image,
}

struct Heavy {
    transformer: SanaTransformer,
    decoder: DcAeDecoder,
    dc_ae: DcAeConfig,
    variant: Variant,
    uncond: Option<(mlx_rs::Array, mlx_rs::Array)>,
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    Base,
    Sprint,
}

impl Variant {
    fn from_env() -> Self {
        match std::env::var("SANA_PREVIEW_VARIANT").as_deref() {
            Ok("base") => Self::Base,
            Ok("sprint") => Self::Sprint,
            _ => panic!("set SANA_PREVIEW_VARIANT to base or sprint"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Sprint => "sprint",
        }
    }

    fn steps(self) -> usize {
        match self {
            Self::Base => BASE_STEPS,
            Self::Sprint => SPRINT_STEPS,
        }
    }
}

impl Heavy {
    fn render_pair(&self, cond: &(mlx_rs::Array, mlx_rs::Array), seed: u64) -> Pair {
        self.render_pair_impl(cond, seed, false)
    }

    fn render_pair_impl(
        &self,
        cond: &(mlx_rs::Array, mlx_rs::Array),
        seed: u64,
        explicit_inert_preview: bool,
    ) -> Pair {
        let init = pipeline::create_noise(seed, WIDTH, HEIGHT).unwrap();
        let cancel = CancelFlag::default();
        let mut progress = |_| {};
        let latents = match self.variant {
            Variant::Base => {
                let scheduler = FlowMatchEuler::for_static_shift(
                    self.variant.steps(),
                    pipeline::SCHEDULE_SHIFT,
                );
                let uncond = self.uncond.as_ref().unwrap();
                if explicit_inert_preview {
                    pipeline::denoise_cfg_with_preview(
                        &self.transformer,
                        &scheduler,
                        None,
                        0,
                        seed,
                        init,
                        &cond.0,
                        Some(&cond.1),
                        Some(&uncond.0),
                        Some(&uncond.1),
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
                        0,
                        seed,
                        init,
                        &cond.0,
                        Some(&cond.1),
                        Some(&uncond.0),
                        Some(&uncond.1),
                        GUIDANCE,
                        &cancel,
                        &mut progress,
                    )
                }
            }
            Variant::Sprint => {
                let scheduler = mlx_gen_sana::ScmScheduler::new(self.variant.steps());
                if explicit_inert_preview {
                    pipeline::denoise_sprint_with_preview(
                        &self.transformer,
                        &scheduler,
                        seed,
                        init,
                        &cond.0,
                        Some(&cond.1),
                        GUIDANCE,
                        0.1,
                        &cancel,
                        &mut progress,
                        &PreviewSink::default(),
                    )
                } else {
                    pipeline::denoise_sprint(
                        &self.transformer,
                        &scheduler,
                        seed,
                        init,
                        &cond.0,
                        Some(&cond.1),
                        GUIDANCE,
                        0.1,
                        &cancel,
                        &mut progress,
                    )
                }
            }
        }
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
        let unscaled = divide(
            &latents,
            Array::from_slice(&[self.dc_ae.scaling_factor], &[1]),
        )
        .unwrap();
        let decoded_nhwc = self
            .decoder
            .decode(&unscaled, &cancel)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let decoded =
            decoded_to_image(&decoded_nhwc.transpose_axes(&[0, 3, 1, 2]).unwrap()).unwrap();

        let lh = (HEIGHT / SPATIAL_SCALE) as i32;
        let lw = (WIDTH / SPATIAL_SCALE) as i32;
        let x = latents
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap()
            .reshape(&[lh * lw, CHANNELS as i32])
            .unwrap();
        let y = add(multiply(&decoded_nhwc, scalar(0.5)).unwrap(), scalar(0.5)).unwrap();
        let y = minimum(maximum(&y, scalar(0.0)).unwrap(), scalar(1.0)).unwrap();
        let y = mean_axes(
            y.reshape(&[1, lh, SPATIAL_SCALE as i32, lw, SPATIAL_SCALE as i32, 3])
                .unwrap(),
            &[2, 4],
            false,
        )
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

/// Compute standard R² separately for one image: target SST is centered around this image's own
/// channel means, while SSE uses the raw prediction residual. This prevents a fit that only
/// distinguishes prompt-level palettes from passing while producing spatially flat previews and
/// also penalizes incorrect image-level palette means.
fn spatial_metrics(pair: &Pair, coef: &[[f64; 3]; DIM]) -> ([f64; 3], [f64; 3], f64) {
    let mut sum = [0.0; 3];
    for y in pair.rgb.chunks_exact(3) {
        for channel in 0..3 {
            sum[channel] += y[channel];
        }
    }
    let mean = sum.map(|value| value / pair.n as f64);
    let mut residual = [0.0; 3];
    let mut total = [0.0; 3];
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
    let variance = total.map(|value| value / pair.n as f64);
    let channels = std::array::from_fn(|channel| 1.0 - residual[channel] / total[channel]);
    let overall = 1.0 - residual.iter().sum::<f64>() / total.iter().sum::<f64>();
    (variance, channels, overall)
}

fn projected_pixels(pair: &Pair, coef: &[[f64; 3]; DIM]) -> Vec<u8> {
    let mut nchw = vec![0.0_f32; pair.n * CHANNELS];
    for sample in 0..pair.n {
        for channel in 0..CHANNELS {
            nchw[channel * pair.n + sample] = pair.latent[sample * CHANNELS + channel] as f32;
        }
    }
    let latent_height = HEIGHT / SPATIAL_SCALE;
    let latent_width = WIDTH / SPATIAL_SCALE;
    let latents = Array::from_slice(
        &nchw,
        &[
            1,
            CHANNELS as i32,
            latent_height as i32,
            latent_width as i32,
        ],
    );
    let factors: Vec<[f32; 3]> = coef[..CHANNELS]
        .iter()
        .map(|row| row.map(|value| value as f32))
        .collect();
    let bias = coef[CHANNELS].map(|value| value as f32);
    mlx_gen::preview::project_latents(&latents, &factors, bias)
        .unwrap()
        .pixels
}

fn analytic_projected_pixels(pair: &Pair, coef: &[[f64; 3]; DIM]) -> Vec<u8> {
    pair.latent
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
        .collect()
}

fn analytic_projector_max_delta(pair: &Pair, coef: &[[f64; 3]; DIM]) -> u8 {
    projected_pixels(pair, coef)
        .iter()
        .zip(analytic_projected_pixels(pair, coef))
        .map(|(production, analytic)| production.abs_diff(analytic))
        .max()
        .unwrap_or(0)
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
    let projected = projected_pixels(pair, coef);
    assert!(
        analytic_projector_max_delta(pair, coef) <= 1,
        "shared f32 production projector diverges by more than one RGB8 level from the independent f64 fit formula at {split} image {index}"
    );
    image::save_buffer(
        dir.join(format!("{split}_{index}_projected.png")),
        &projected,
        WIDTH / SPATIAL_SCALE,
        HEIGHT / SPATIAL_SCALE,
        image::ColorType::Rgb8,
    )
    .unwrap();
    let mut comparison = vec![0u8; (WIDTH * 2 * HEIGHT * 3) as usize];
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            let left = (y * WIDTH as usize + x) * 3;
            let dst = (y * WIDTH as usize * 2 + x) * 3;
            comparison[dst..dst + 3].copy_from_slice(&pair.decoded.pixels[left..left + 3]);
            let src = ((y / SPATIAL_SCALE as usize) * (WIDTH as usize / SPATIAL_SCALE as usize)
                + x / SPATIAL_SCALE as usize)
                * 3;
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
#[ignore = "needs real SANA/DC-AE weights and a Metal-capable macOS host"]
fn fit_preview_rgb() {
    let root = snapshot();
    let variant = Variant::from_env();
    let text_encoder = SanaTextEncoder::from_snapshot(root.join("text_encoder")).unwrap();
    let prompts: Vec<_> = FIT_CORPUS
        .iter()
        .chain(HOLDOUT_CORPUS.iter())
        .map(|(prompt, _)| text_encoder.encode_with_mask(prompt).unwrap())
        .collect();
    let uncond =
        matches!(variant, Variant::Base).then(|| text_encoder.encode_with_mask("").unwrap());
    let mut arrays = Vec::with_capacity(prompts.len() * 2 + 2);
    for (cond, mask) in &prompts {
        arrays.push(cond);
        arrays.push(mask);
    }
    if let Some((uncond, mask)) = &uncond {
        arrays.push(uncond);
        arrays.push(mask);
    }
    mlx_rs::transforms::eval(arrays).unwrap();
    drop(text_encoder);
    mlx_rs::memory::clear_cache();

    let transformer_config = match variant {
        Variant::Base => SanaTransformerConfig::sana_1600m(),
        Variant::Sprint => SanaTransformerConfig::sana_sprint_1600m(),
    };
    let transformer = SanaTransformer::from_weights(
        &Weights::from_dir(root.join("transformer")).unwrap(),
        transformer_config,
    )
    .unwrap();
    let dc_ae = DcAeConfig::sana_f32c32();
    let decoder =
        DcAeDecoder::from_weights(&Weights::from_dir(root.join("vae")).unwrap(), dc_ae.clone())
            .unwrap();
    let heavy = Heavy {
        transformer,
        decoder,
        dc_ae,
        variant,
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
    writeln!(
        report,
        "SANA {} preview OLS: 4 fit + 2 disjoint holdout, 256x256, native 8x8 latent, {} steps, guidance 4.5",
        variant.name(),
        variant.steps()
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
        "DC-AE identity proof: Base and Sprint files are both 1,249,044,836 bytes but differ: Base SHA256 15a4b09e56d95b768a0ec9da50b702e21d920333fc9b3480d66bb5c7fad9d87f; Sprint SHA256 dfd991d1b54ffabf22745c5885589d8f2a7bc59930d95d92bd741c4fc64454bb; constants are not shared"
    )
    .unwrap();
    writeln!(
        report,
        "official config proof: Base dc-ae-f32c32-sana-1.0 config revision d1b54936033cd7d45410ecadd692c5c502a19a38 SHA256 4e5669aa03caa77615b73b21c9347c384cb11124a9590683e32a510df0176eb4; Sprint Sana_Sprint revision 19683c58b7ea290e55cedd8950ae1d86ada7ef96 config SHA256 ba6f3d3e44d75d44fdd3760097c069173b5b925e6d14604d5d3582628d09cca6; consumed architecture/scaling fields match, model metadata differs (Sana 1.0 vs 1.1)"
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
    writeln!(
        report,
        "spatial diagnostic: each image uses its own target-centered SST and raw prediction SSE, so prompt-level palette means cannot satisfy these R2 checks and palette bias remains penalized"
    )
    .unwrap();
    for (split, pairs) in [("fit", &fit_pairs), ("holdout", &holdout_pairs)] {
        for (index, pair) in pairs.iter().enumerate() {
            let (variance, channels, overall) = spatial_metrics(pair, &coef);
            writeln!(
                report,
                "{split}[{index}] target variance RGB={variance:?}; per-image spatial R2 RGB={channels:?} overall={overall:.8}; producer artifact uses exact shared production-projector bytes; independent f64 audit max RGB8 delta={}",
                analytic_projector_max_delta(pair, &coef)
            )
            .unwrap();
            assert!(variance
                .iter()
                .all(|value| *value >= PER_IMAGE_TARGET_VARIANCE_FLOOR));
            assert!(channels
                .iter()
                .all(|value| *value >= PER_IMAGE_CHANNEL_R2_FLOOR));
            assert!(overall >= PER_IMAGE_OVERALL_R2_FLOOR);
            assert!(analytic_projector_max_delta(pair, &coef) <= 1);
        }
    }
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

    assert!(fit_channels
        .iter()
        .all(|value| *value >= FIT_CHANNEL_R2_FLOOR));
    assert!(holdout_channels
        .iter()
        .all(|value| *value >= HOLDOUT_CHANNEL_R2_FLOOR));
    assert!(holdout_overall >= HOLDOUT_OVERALL_R2_FLOOR);
    if let Ok(path) = std::env::var("SANA_PREVIEW_ARTIFACT_DIR") {
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
