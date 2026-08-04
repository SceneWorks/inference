//! sc-16960 — the **producer** for SenseNova-U1's preview RGB fit (epic 16948, Tier 2).
//!
//! This is the only new fit in the candle epic. Every other family maps onto a latent space epic
//! 16624 already fitted on MLX; SenseNova does not, so this file measures one on real weights.
//!
//! ## What is being fitted, and why it is not a VAE latent space
//!
//! The epic scoped this story as "SenseNova-U1 has its own VAE". It does not have one at all — see
//! `crate`'s own module docs ("there is no separate VAE or text encoder"),
//! [`the_snapshot_ships_no_autoencoder`], and `config.json`, which carries no autoencoder section.
//! SenseNova-U1 denoises in **pixel space**: the running state of `T2iModel::denoise` is the image
//! itself in the model's `[-1, 1]` space, and the "decode" is the affine map `tensor_to_image`
//! applies (`x·0.5 + 0.5`, clamped).
//!
//! So the fit measured here is over a **three-channel** space, which is on its own enough to rule it
//! out of every epic-16624 reuse (those are 4-, 16- and 32-channel VAE latents).
//!
//! ## The corpus and the split — stated exactly, because the bar is a holdout bar
//!
//! [`FIT_CORPUS`] is **four** diverse prompt/seed renders and [`HOLDOUT_CORPUS`] is **two** further
//! renders with different prompts *and* different seeds. The holdout renders are produced after the
//! coefficients are solved and never contribute to them — the split is by whole render, never a
//! random subsample of one render's pixels, which would leak the render's own palette into both
//! halves. That is the epic-16624 standard, and the standard that rejected LTX (fit .984 / holdout
//! .619), Mage (.938 / .806) and Mochi (.847 / .807).
//!
//! **Holdout R² ≥ 0.88 is the go/no-go bar.** [`FIT_OVERALL_R2_FLOOR`] and
//! [`HOLDOUT_OVERALL_R2_FLOOR`] are separate constants for separate splits, and the report labels
//! which is which.
//!
//! ```sh
//! SENSENOVA_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--SceneWorks--sensenova-u1-8b-mlx\snapshots\<rev>\q8 \
//! SENSENOVA_PREVIEW_ARTIFACT_DIR=E:\out\sc-16960 \
//!   cargo test -p candle-gen-sensenova --release --features cuda --test fit_preview_rgb \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::{CancelFlag, Image, PreviewFrame, PreviewSink, Progress};
use candle_gen::preview::PreviewHook;
use candle_gen_sensenova::{load_understanding, tensor_to_image, T2iOptions};
use sha2::{Digest, Sha256};

/// Four diverse prompt/seed renders whose pooled pixels determine the coefficients.
const FIT_CORPUS: [(&str, u64); 4] = [
    (
        "a chef cooking at a night market, orange lanterns, teal reflections on wet pavement, rising steam",
        1696001,
    ),
    (
        "snow covered alpine mountains at noon, deep blue sky, dark evergreen forest, crisp cold sunlight",
        1696002,
    ),
    (
        "studio portrait of an elderly woman wearing a vivid red scarf, neutral gray background, natural skin tones",
        1696003,
    ),
    (
        "editorial illustration of tropical fruit and green leaves on a pale pink table, flat saturated shapes",
        1696004,
    ),
];

/// Two renders the fit never sees — different prompts *and* different seeds, held out whole.
const HOLDOUT_CORPUS: [(&str, u64); 2] = [
    (
        "modern library at dusk, warm reading lamps and cool blue light through tall windows",
        1696091,
    ),
    (
        "macro photograph of purple wildflowers after rain, bright green stems and silver droplets",
        1696092,
    ),
];

/// SenseNova's model-space channel count. Three, because it denoises in pixel space.
const CHANNELS: usize = 3;
/// Design matrix width: one column per channel plus the intercept.
const DIM: usize = CHANNELS + 1;

/// Fit-split floors. Separate constants for a separate split — the confusion sc-16954 was bounced for.
const FIT_CHANNEL_R2_FLOOR: f64 = 0.99;
const FIT_OVERALL_R2_FLOOR: f64 = 0.99;

/// **The epic's go/no-go bar**, on the *holdout* split alone.
const HOLDOUT_CHANNEL_R2_FLOOR: f64 = 0.88;
const HOLDOUT_OVERALL_R2_FLOOR: f64 = 0.88;

/// Per-image spatial floors: each image's own target-centered SST against its raw prediction SSE, so
/// a fit that only distinguished prompt-level palettes could not pass.
const PER_IMAGE_OVERALL_R2_FLOOR: f64 = 0.98;
/// Each pooled target must carry real spatial structure, or the per-image R² above is measured
/// against nothing.
const PER_IMAGE_TARGET_VARIANCE_FLOOR: f64 = 0.002;

fn required_path(name: &str) -> PathBuf {
    let value = std::env::var(name).unwrap_or_else(|_| {
        panic!("{name} must be set — this row measures a fit and cannot be skipped into a pass")
    });
    let path = PathBuf::from(value);
    assert!(
        path.exists(),
        "{name} points at {} — not found",
        path.display()
    );
    path
}

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| {
            value.parse().unwrap_or_else(|_| {
                panic!("{name} must parse as a positive integer, got {value:?}")
            })
        })
        .unwrap_or(fallback)
}

fn env_f32(name: &str, fallback: f32) -> f32 {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must parse as a float, got {value:?}"))
        })
        .unwrap_or(fallback)
}

fn artifact_dir() -> PathBuf {
    required_path("SENSENOVA_PREVIEW_ARTIFACT_DIR")
}

fn sha256_of(path: &Path) -> String {
    let mut file = std::fs::File::open(path).expect("open for hashing");
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash");
    format!("{:x}", hasher.finalize())
}

/// The `*.safetensors` shards under a SenseNova snapshot tier, excluding the optional distill LoRA —
/// the same selection `backbone_files` makes.
fn shards(root: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(root)
        .expect("read the snapshot dir")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .filter(|path| {
            path.file_name().and_then(|n| n.to_str()) != Some("distill_lora.safetensors")
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no shards under {}", root.display());
    files
}

/// The tensor names in one safetensors shard, read from its **header alone**.
///
/// The shipped SenseNova tiers are single 12–35 GB containers, so a `deserialize` of the whole file
/// is not an option here; the header is the leading `u64` length plus that many bytes of JSON, whose
/// keys are the tensor names.
fn shard_tensor_names(path: &Path) -> Vec<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect("open a shard");
    let mut length = [0u8; 8];
    file.read_exact(&mut length)
        .expect("read the header length");
    let mut header = vec![0u8; u64::from_le_bytes(length) as usize];
    file.read_exact(&mut header).expect("read the header");
    let json: serde_json::Value = serde_json::from_slice(&header).expect("parse the header");
    json.as_object()
        .expect("the safetensors header is a JSON object")
        .keys()
        .filter(|key| *key != "__metadata__")
        .cloned()
        .collect()
}

/// One render's pooled `(state, decode)` pair plus the finished RGB8 image.
struct Pair {
    /// Sample-major model-space values, `n · CHANNELS`.
    state: Vec<f64>,
    /// Sample-major decoded values in `[0, 1]`, `n · 3`.
    rgb: Vec<f64>,
    n: usize,
    decoded: Image,
}

/// Flatten `[1, C, h, w]` to sample-major `n · C` f64.
fn sample_major(tensor: &Tensor) -> (Vec<f64>, usize) {
    let dims = tensor.dims().to_vec();
    let (channels, h, w) = (dims[1], dims[2], dims[3]);
    let values = tensor
        .permute((0, 2, 3, 1))
        .expect("to sample-major")
        .contiguous()
        .expect("contiguous")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to vec");
    assert_eq!(values.len(), h * w * channels);
    (values.into_iter().map(f64::from).collect(), h * w)
}

/// Build the `(state, decode)` pair from one finished render.
///
/// The predictor is the `cell`-pooled model-space state; the target is the `cell`-pooled **clamped**
/// decode. Clamp-then-pool, in that order, because that is the order the finished image is produced
/// in — and the clamp is the only non-linearity in an otherwise exactly affine path, so getting the
/// order wrong would move the residual this fit is measuring.
fn pair_from(final_state: &Tensor, cell: usize) -> Pair {
    let f32_state = final_state.to_dtype(DType::F32).expect("f32");
    let pooled_state = f32_state.avg_pool2d(cell).expect("pool the state");
    let decoded = f32_state
        .affine(0.5, 0.5)
        .expect("model space to [0,1]")
        .clamp(0f32, 1f32)
        .expect("clamp")
        .avg_pool2d(cell)
        .expect("pool the decode");

    let (state, n) = sample_major(&pooled_state);
    let (rgb, rgb_n) = sample_major(&decoded);
    assert_eq!(n, rgb_n);
    Pair {
        state,
        rgb,
        n,
        decoded: tensor_to_image(final_state).expect("decode to RGB8"),
    }
}

#[allow(clippy::needless_range_loop)]
fn solve(mut a: [[f64; DIM]; DIM], mut b: [[f64; 3]; DIM]) -> [[f64; 3]; DIM] {
    for col in 0..DIM {
        let pivot = (col..DIM)
            .max_by(|&i, &j| {
                a[i][col]
                    .abs()
                    .partial_cmp(&a[j][col].abs())
                    .expect("finite")
            })
            .expect("a pivot");
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
                .copy_from_slice(&pair.state[sample * CHANNELS..(sample + 1) * CHANNELS]);
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

fn predict(x: &[f64], coef: &[[f64; 3]; DIM], channel: usize) -> f64 {
    coef[CHANNELS][channel]
        + x.iter()
            .enumerate()
            .map(|(index, value)| value * coef[index][channel])
            .sum::<f64>()
}

/// Standard R² over a split: SSE from the raw prediction residual, SST about the split's own means.
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
            let x = &pair.state[sample * CHANNELS..(sample + 1) * CHANNELS];
            for channel in 0..3 {
                let error = pair.rgb[sample * 3 + channel] - predict(x, coef, channel);
                residual[channel] += error * error;
                let centered = pair.rgb[sample * 3 + channel] - mean[channel];
                total[channel] += centered * centered;
            }
        }
    }
    let channels = std::array::from_fn(|channel| 1.0 - residual[channel] / total[channel]);
    let overall = 1.0 - residual.iter().sum::<f64>() / total.iter().sum::<f64>();
    (channels, overall)
}

/// Per-image R²: SST centered on **this image's own** channel means, so a fit that only separated
/// prompt-level palettes could not pass, and target variance so the statistic is not measured against
/// a flat frame.
fn spatial_metrics(pair: &Pair, coef: &[[f64; 3]; DIM]) -> ([f64; 3], f64) {
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
        let x = &pair.state[sample * CHANNELS..(sample + 1) * CHANNELS];
        for channel in 0..3 {
            let error = pair.rgb[sample * 3 + channel] - predict(x, coef, channel);
            residual[channel] += error * error;
            let centered = pair.rgb[sample * 3 + channel] - mean[channel];
            total[channel] += centered * centered;
        }
    }
    let variance = total.map(|value| value / pair.n as f64);
    let overall = 1.0 - residual.iter().sum::<f64>() / total.iter().sum::<f64>();
    (variance, overall)
}

/// The largest distance between any solved coefficient and the analytic decode transform
/// `x·0.5 + 0.5` the model itself applies.
fn analytic_max_deviation(coef: &[[f64; 3]; DIM]) -> f64 {
    let mut worst: f64 = 0.0;
    for (row, factors) in coef[..CHANNELS].iter().enumerate() {
        for (column, value) in factors.iter().enumerate() {
            let expected = if row == column { 0.5 } else { 0.0 };
            worst = worst.max((value - expected).abs());
        }
    }
    for value in coef[CHANNELS] {
        worst = worst.max((value - 0.5).abs());
    }
    worst
}

/// The **shipped** projector's bytes for one render, so the constants block below is validated
/// through the code that will use it rather than through a second implementation of the same maths.
fn shipped_projection(final_state: &Tensor, cell: usize) -> Image {
    candle_gen_sensenova::preview::project_running_image(final_state, cell)
        .expect("the shipped projector must accept the running state")
}

/// The independent f64 evaluation of the *solved* coefficients, for the same pooled samples.
fn analytic_projection(pair: &Pair, coef: &[[f64; 3]; DIM]) -> Vec<u8> {
    pair.state
        .chunks_exact(CHANNELS)
        .flat_map(|x| {
            (0..3).map(move |channel| {
                (predict(x, coef, channel).clamp(0.0, 1.0) * 255.0).round() as u8
            })
        })
        .collect()
}

fn save_png(pixels: &[u8], width: u32, height: u32, name: &str) {
    let dir = artifact_dir();
    std::fs::create_dir_all(&dir).expect("create the artifact dir");
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(&path, pixels, width, height, image::ExtendedColorType::Rgb8)
        .expect("save a PNG");
    eprintln!("  saved {}", path.display());
}

fn inert_hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::new(sink, |_: &Tensor| {
        panic!("an inert preview sink must not invoke projection")
    })
}

fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
    (sink, frames)
}

/// **SenseNova-U1 ships no autoencoder** — the structural fact that makes this a Tier 2 fit rather
/// than a Tier 1 reuse, proved against the snapshot rather than asserted in prose.
///
/// Three things are checked, because any one alone is weak: no `vae/` component directory beside the
/// checkpoint, no `vae`/autoencoder section in `config.json`, and no autoencoder-shaped tensor key in
/// the shard headers. If any of them were false, the reuse question epic 16948 asks every wiring
/// story would have been live here too.
#[test]
#[ignore = "needs a real SenseNova-U1 snapshot (set SENSENOVA_PREVIEW_SNAPSHOT)"]
fn the_snapshot_ships_no_autoencoder() {
    let root = required_path("SENSENOVA_PREVIEW_SNAPSHOT");

    for component in ["vae", "autoencoder", "first_stage_model"] {
        assert!(
            !root.join(component).exists(),
            "{} exists — this snapshot has an autoencoder and the Tier 2 premise is wrong",
            root.join(component).display()
        );
    }

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("config.json")).expect("config"))
            .expect("parse config.json");
    for key in ["vae", "vae_config", "autoencoder", "autoencoder_config"] {
        assert!(
            config.get(key).is_none(),
            "config.json declares {key:?} — this checkpoint has an autoencoder"
        );
    }

    let files = shards(&root);
    let mut prefixes: BTreeSet<String> = BTreeSet::new();
    let mut total = 0usize;
    for file in &files {
        for name in shard_tensor_names(file) {
            total += 1;
            let prefix = name.split('.').next().unwrap_or(&name).to_string();
            prefixes.insert(prefix);
            let lowered = name.to_ascii_lowercase();
            for needle in ["vae", "autoencoder", "first_stage"] {
                assert!(
                    !lowered.contains(needle),
                    "tensor {name} looks like an autoencoder weight"
                );
            }
        }
    }
    eprintln!("checkpoint: {} shard(s), {total} tensors", files.len());
    eprintln!("top-level prefixes: {prefixes:?}");
    // The whole checkpoint is the dual-path backbone plus the FM head: no third subtree.
    assert!(
        prefixes.iter().all(|prefix| matches!(
            prefix.as_str(),
            "language_model" | "fm_modules" | "vision_model"
        )),
        "unexpected top-level tensor subtree in {prefixes:?} — the fit's premise is that this \
         checkpoint holds only the backbone and the flow-matching head"
    );

    for file in &files {
        eprintln!(
            "provenance: {} — {} bytes, SHA-256 {}",
            file.display(),
            std::fs::metadata(file).expect("metadata").len(),
            sha256_of(file)
        );
    }
}

/// **The fit.** Four renders solve it, two disjoint renders measure it, and both R²s are reported
/// separately and labelled.
#[test]
#[ignore = "needs a real SenseNova-U1 snapshot + a CUDA GPU; run with --features cuda --ignored"]
fn fit_preview_rgb() {
    let root = required_path("SENSENOVA_PREVIEW_SNAPSHOT");
    let size = env_usize("SENSENOVA_PREVIEW_SIZE", 512);
    let steps = env_usize("SENSENOVA_PREVIEW_STEPS", 8);
    let guidance = env_f32("SENSENOVA_PREVIEW_GUIDANCE", 4.0);

    let (model, tokenizer) = load_understanding(&root).expect("load the SenseNova-U1 checkpoint");
    let cell = model.cell();
    eprintln!(
        "token cell {cell}px ⇒ a {size}² render previews at {}²",
        size / cell
    );

    let cancel = CancelFlag::default();
    let mut progress = |_: Progress| {};
    let inert = PreviewSink::default();
    let hook = inert_hook(&inert);
    let mut render = |prompt: &str, seed: u64| -> Tensor {
        let opts = T2iOptions {
            cfg_scale: guidance,
            num_steps: steps,
            timestep_shift: 3.0,
            seed,
            ..Default::default()
        };
        model
            .generate(
                &tokenizer,
                prompt,
                size,
                size,
                &opts,
                &cancel,
                &mut progress,
                &hook,
            )
            .unwrap_or_else(|e| panic!("render seed {seed}: {e}"))
    };

    let fit_states: Vec<Tensor> = FIT_CORPUS
        .iter()
        .enumerate()
        .map(|(index, (prompt, seed))| {
            eprintln!("rendering fit {index} seed {seed}: {prompt}");
            render(prompt, *seed)
        })
        .collect();
    let fit_pairs: Vec<Pair> = fit_states
        .iter()
        .map(|state| pair_from(state, cell))
        .collect();

    // Solve on the fit split alone, THEN render the holdout — so nothing about the holdout can have
    // reached the coefficients even by accident.
    let coef = fit(&fit_pairs);

    let holdout_states: Vec<Tensor> = HOLDOUT_CORPUS
        .iter()
        .enumerate()
        .map(|(index, (prompt, seed))| {
            eprintln!("rendering holdout {index} seed {seed}: {prompt}");
            render(prompt, *seed)
        })
        .collect();
    let holdout_pairs: Vec<Pair> = holdout_states
        .iter()
        .map(|state| pair_from(state, cell))
        .collect();

    let (fit_channels, fit_overall) = r2(&fit_pairs, &coef);
    let (holdout_channels, holdout_overall) = r2(&holdout_pairs, &coef);

    let mut report = String::new();
    writeln!(
        report,
        "SenseNova-U1 preview OLS (sc-16960): {} fit + {} DISJOINT holdout renders, {size}², \
         {steps} flow-match Euler steps, guidance {guidance}, token cell {cell} ⇒ {}² frames",
        FIT_CORPUS.len(),
        HOLDOUT_CORPUS.len(),
        size / cell
    )
    .unwrap();
    writeln!(
        report,
        "split: whole renders, never a pixel subsample — fit seeds {:?}, holdout seeds {:?}, and \
         the holdout prompts are disjoint from the fit prompts",
        FIT_CORPUS.map(|(_, seed)| seed),
        HOLDOUT_CORPUS.map(|(_, seed)| seed)
    )
    .unwrap();
    writeln!(
        report,
        "latent space: THREE-channel pixel space — SenseNova-U1 ships no VAE, so this could not be \
         one of the seven epic-16624 spaces and had to be measured"
    )
    .unwrap();
    writeln!(
        report,
        "FIT      R2 (R,G,B) = {fit_channels:?}  overall = {fit_overall:.8}"
    )
    .unwrap();
    writeln!(
        report,
        "HOLDOUT  R2 (R,G,B) = {holdout_channels:?}  overall = {holdout_overall:.8}"
    )
    .unwrap();
    writeln!(
        report,
        "go/no-go: the epic's bar is HOLDOUT overall R2 >= {HOLDOUT_OVERALL_R2_FLOOR} \
         (LTX .984 fit/.619 holdout, Mage .938/.806, Mochi .847/.807 were rejected on it); \
         measured holdout overall = {holdout_overall:.8}"
    )
    .unwrap();
    writeln!(
        report,
        "max |coefficient − analytic x·0.5+0.5| = {:.10} (the residual is the decode's clamp)",
        analytic_max_deviation(&coef)
    )
    .unwrap();

    for (split, pairs) in [("fit", &fit_pairs), ("holdout", &holdout_pairs)] {
        for (index, pair) in pairs.iter().enumerate() {
            let (variance, overall) = spatial_metrics(pair, &coef);
            writeln!(
                report,
                "{split}[{index}] n = {} samples; target variance RGB = {variance:?}; \
                 per-image spatial R2 overall = {overall:.8}",
                pair.n
            )
            .unwrap();
            assert!(
                variance
                    .iter()
                    .all(|value| *value >= PER_IMAGE_TARGET_VARIANCE_FLOOR),
                "{split}[{index}] pooled target is too flat to measure against: {variance:?}"
            );
            assert!(
                overall >= PER_IMAGE_OVERALL_R2_FLOOR,
                "{split}[{index}] per-image spatial R2 {overall:.8} below \
                 {PER_IMAGE_OVERALL_R2_FLOOR}"
            );
        }
    }

    writeln!(report, "const RGB_FACTORS: [[f32; 3]; {CHANNELS}] = [").unwrap();
    for row in &coef[..CHANNELS] {
        writeln!(report, "    [{:.9}, {:.9}, {:.9}],", row[0], row[1], row[2]).unwrap();
    }
    writeln!(
        report,
        "];\nconst RGB_BIAS: [f32; 3] = [{:.9}, {:.9}, {:.9}];",
        coef[CHANNELS][0], coef[CHANNELS][1], coef[CHANNELS][2]
    )
    .unwrap();

    // The shipped projector must agree with the independent f64 evaluation of the SOLVED
    // coefficients to within one RGB8 level — the check that binds the constants block above to the
    // code that will consume it. (It is run against the COMMITTED constants, so it also fails loudly
    // if the block below is transcribed wrongly.)
    for (split, pairs, states) in [
        ("fit", &fit_pairs, &fit_states),
        ("holdout", &holdout_pairs, &holdout_states),
    ] {
        for (index, (pair, state)) in pairs.iter().zip(states.iter()).enumerate() {
            let shipped = shipped_projection(state, cell);
            let analytic = analytic_projection(pair, &coef);
            assert_eq!(shipped.pixels.len(), analytic.len());
            let worst = shipped
                .pixels
                .iter()
                .zip(&analytic)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            writeln!(
                report,
                "{split}[{index}] shipped projector vs independent f64 evaluation of the solved \
                 coefficients: max RGB8 delta = {worst}"
            )
            .unwrap();
            assert!(
                worst <= 1,
                "{split}[{index}]: the shipped projector diverges from the solved fit by {worst} \
                 RGB8 levels — the committed constants block does not match this measurement"
            );
            save_png(
                &pair.decoded.pixels,
                pair.decoded.width,
                pair.decoded.height,
                &format!("fit_{split}_{index}_decoded"),
            );
            save_png(
                &shipped.pixels,
                shipped.width,
                shipped.height,
                &format!("fit_{split}_{index}_projected"),
            );
        }
    }

    // An inert sink is byte-identical to a live one at the same seed, on this very producer's path.
    let (live_sink, frames) = collecting_sink();
    let live_hook = PreviewHook::new(&live_sink, |state: &Tensor| {
        candle_gen_sensenova::preview::project_running_image(state, cell)
    });
    let opts = T2iOptions {
        cfg_scale: guidance,
        num_steps: steps,
        timestep_shift: 3.0,
        seed: FIT_CORPUS[0].1,
        ..Default::default()
    };
    let live = model
        .generate(
            &tokenizer,
            FIT_CORPUS[0].0,
            size,
            size,
            &opts,
            &cancel,
            &mut progress,
            &live_hook,
        )
        .expect("live-sink replay");
    let flat = |t: &Tensor| {
        t.flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("vec")
    };
    assert_eq!(
        flat(&fit_states[0]),
        flat(&live),
        "an active preview sink must not change one f32 of the final state at the same seed"
    );
    let emitted = candle_gen::lock_recover(&frames).len();
    writeln!(
        report,
        "inert-vs-live replay at seed {}: final f32 state identical; the live render emitted \
         {emitted} frames over {steps} steps",
        FIT_CORPUS[0].1
    )
    .unwrap();
    assert_eq!(
        emitted, steps,
        "the bespoke loop must emit exactly one frame per outer step"
    );

    println!("{report}");
    let dir = artifact_dir();
    std::fs::create_dir_all(&dir).expect("create the artifact dir");
    std::fs::write(dir.join("sc-16960-fit-report.txt"), &report).expect("write the fit report");

    // Fit and holdout are asserted against SEPARATE floors, and the holdout floor is the epic's bar.
    assert!(
        fit_channels.iter().all(|v| *v >= FIT_CHANNEL_R2_FLOOR),
        "fit per-channel R2 {fit_channels:?} below {FIT_CHANNEL_R2_FLOOR}"
    );
    assert!(
        fit_overall >= FIT_OVERALL_R2_FLOOR,
        "fit overall R2 {fit_overall:.8} below {FIT_OVERALL_R2_FLOOR}"
    );
    assert!(
        holdout_channels
            .iter()
            .all(|v| *v >= HOLDOUT_CHANNEL_R2_FLOOR),
        "HOLDOUT per-channel R2 {holdout_channels:?} below the epic bar {HOLDOUT_CHANNEL_R2_FLOOR}"
    );
    assert!(
        holdout_overall >= HOLDOUT_OVERALL_R2_FLOOR,
        "HOLDOUT overall R2 {holdout_overall:.8} below the epic bar {HOLDOUT_OVERALL_R2_FLOOR} — \
         this is a NO-GO and no fit may ship"
    );
}
