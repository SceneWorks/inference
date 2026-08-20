//! sc-17218 one-time real-weight CUDA acceptance for Boogu previews.
//!
//! This file is deliberately `#[ignore]`d and is not wired into CI. Run it once on a CUDA host with
//! the three deployed snapshot directories, retain the generated strips/metrics, and record the run in
//! `docs/migration/evidence/sc-17218-boogu-candle-preview.md`.
//!
//! ```text
//! set BOOGU_BASE_SNAPSHOT=...\base-q4
//! set BOOGU_TURBO_SNAPSHOT=...\turbo-q4
//! set BOOGU_EDIT_SNAPSHOT=...\edit-q4
//! set BOOGU_EDIT_REFERENCE=...\boogu-turbo-native-final.png
//! set BOOGU_PREVIEW_OUT=...\sc-17218
//! cargo test -p candle-gen-boogu --features cuda --release --test preview_real_weights \
//!   -- --ignored --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink,
    Progress, WeightsSource,
};
use image::{imageops, RgbImage};

const PROMPT: &str =
    "a small red sailboat crossing a deep blue mountain lake at golden hour, cinematic photograph";
const EDIT_PROMPT: &str =
    "turn the scene into a snowy winter morning while preserving the sailboat";
const SEED: u64 = 17_218;

// The reused FLUX.1 fit has in-sample R² 0.98224, so no preview can exceed its correlation ceiling
// sqrt(0.98224) ~= 0.9911. Each floor is a lane-specific fraction of that ceiling, then checked against
// the retained 512² Q4 measurement with roughly three correlation points of headroom: Base default
// 96.9% of ceiling (0.960 vs +0.9890 measured), Base Heun 95.9% (0.950 vs +0.9824), Turbo native
// 95.3% (0.945 vs +0.9772), and Edit default 94.8% (0.940 vs +0.9706). The different fractions encode
// how much trajectory each schedule leaves to its unpreviewed terminal advancement.
const BASE_MIN_R_LAST: f64 = 0.96;
const BASE_HEUN_MIN_R_LAST: f64 = 0.95;
const TURBO_MIN_R_LAST: f64 = 0.945;
const EDIT_MIN_R_LAST: f64 = 0.94;

#[derive(Clone, Copy)]
struct Lane<'a> {
    label: &'a str,
    id: &'a str,
    snapshot_var: &'a str,
    sampler: Option<&'a str>,
    steps: u32,
    min_r_last: f64,
    prove_inert_identity: bool,
}

struct Metrics {
    correlations: Vec<f64>,
    distances: Vec<f64>,
    movement: Vec<f64>,
}

fn required_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name} for the one-time sc-17218 CUDA acceptance run"))
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn one_image(output: GenerationOutput) -> Image {
    let GenerationOutput::Images(mut images) = output else {
        panic!("Boogu must return images")
    };
    assert_eq!(images.len(), 1);
    images.pop().unwrap()
}

fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
    (sink, frames)
}

fn read_reference(path: &Path) -> Image {
    let rgb = image::open(path)
        .unwrap_or_else(|error| panic!("open reference {}: {error}", path.display()))
        .to_rgb8();
    Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    }
}

fn request(lane: Lane<'_>, size: u32, conditioning: Vec<Conditioning>) -> GenerationRequest {
    GenerationRequest {
        prompt: if lane.id == candle_gen_boogu::BOOGU_IMAGE_EDIT_ID {
            EDIT_PROMPT.into()
        } else {
            PROMPT.into()
        },
        width: size,
        height: size,
        count: 1,
        seed: Some(SEED),
        steps: Some(lane.steps),
        sampler: lane.sampler.map(str::to_string),
        conditioning,
        ..GenerationRequest::default()
    }
}

fn rgb(image: &Image) -> RgbImage {
    RgbImage::from_raw(image.width, image.height, image.pixels.clone()).unwrap()
}

fn resized_pixels(image: &Image, width: u32, height: u32) -> Vec<u8> {
    imageops::resize(&rgb(image), width, height, imageops::FilterType::Triangle).into_raw()
}

fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn correlation(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let mean_a = a.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let mean_b = b.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let (mut covariance, mut variance_a, mut variance_b) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let x = f64::from(x) - mean_a;
        let y = f64::from(y) - mean_b;
        covariance += x * y;
        variance_a += x * x;
        variance_b += y * y;
    }
    covariance / (variance_a * variance_b).sqrt()
}

fn analyze(label: &str, frames: &[PreviewFrame], final_image: &Image, steps: u32) -> Metrics {
    assert_eq!(
        frames
            .iter()
            .map(|frame| (frame.current, frame.total))
            .collect::<Vec<_>>(),
        (1..=steps)
            .map(|current| (current, steps))
            .collect::<Vec<_>>(),
        "{label}: exactly one frame per outer step"
    );
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (final_image.width / 8, final_image.height / 8),
            "{label}: preview must use native VAE latent resolution"
        );
    }

    let movement = frames
        .windows(2)
        .map(|pair| mean_abs_delta(&pair[0].image.pixels, &pair[1].image.pixels))
        .collect::<Vec<_>>();
    let target = resized_pixels(final_image, final_image.width / 8, final_image.height / 8);
    let distances = frames
        .iter()
        .map(|frame| mean_abs_delta(&frame.image.pixels, &target))
        .collect::<Vec<_>>();
    let coarse_target = resized_pixels(final_image, 16, 16);
    let correlations = frames
        .iter()
        .map(|frame| correlation(&resized_pixels(&frame.image, 16, 16), &coarse_target))
        .collect::<Vec<_>>();

    for (index, frame) in frames.iter().enumerate() {
        eprintln!(
            "  {label} {:>2}/{steps}: distance {:>6.2}, correlation {:+.4}",
            frame.current, distances[index], correlations[index]
        );
        if let Some(delta) = movement.get(index) {
            eprintln!("                     next-frame mean |delta| {delta:.3}");
        }
    }

    Metrics {
        correlations,
        distances,
        movement,
    }
}

fn save_artifacts(label: &str, frames: &[PreviewFrame], final_image: &Image, metrics: &Metrics) {
    let dir = required_path("BOOGU_PREVIEW_OUT");
    std::fs::create_dir_all(&dir).unwrap();

    rgb(final_image)
        .save(dir.join(format!("{label}-final.png")))
        .unwrap();

    let tile = 256u32;
    let mut strip = RgbImage::new(tile * (frames.len() as u32 + 1), tile);
    for (index, frame) in frames.iter().enumerate() {
        let upscaled = imageops::resize(
            &rgb(&frame.image),
            tile,
            tile,
            imageops::FilterType::Nearest,
        );
        imageops::replace(&mut strip, &upscaled, i64::from(index as u32 * tile), 0);
    }
    let final_tile = imageops::resize(
        &rgb(final_image),
        tile,
        tile,
        imageops::FilterType::Triangle,
    );
    imageops::replace(
        &mut strip,
        &final_tile,
        i64::from(frames.len() as u32 * tile),
        0,
    );
    strip.save(dir.join(format!("{label}-strip.png"))).unwrap();

    let mut table = String::from(
        "frame\tcorrelation_to_final\tmean_abs_distance_to_final\tmean_abs_movement_to_next\n",
    );
    for index in 0..frames.len() {
        let movement = metrics
            .movement
            .get(index)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_default();
        table.push_str(&format!(
            "{}\t{:.6}\t{:.6}\t{}\n",
            index + 1,
            metrics.correlations[index],
            metrics.distances[index],
            movement
        ));
    }
    std::fs::write(dir.join(format!("{label}-metrics.tsv")), table).unwrap();
}

fn run_lane(lane: Lane<'_>, conditioning: Vec<Conditioning>) -> (usize, Vec<PreviewFrame>) {
    let size = env_u32("BOOGU_PREVIEW_SIZE", 512);
    eprintln!(
        "-- {}: {}x{}, {} steps, sampler {:?}",
        lane.label, size, size, lane.steps, lane.sampler
    );
    let spec = LoadSpec::new(WeightsSource::Dir(required_path(lane.snapshot_var)));
    let generator = candle_gen_boogu::provider_registry()
        .unwrap()
        .load(lane.id, &spec)
        .unwrap_or_else(|error| panic!("load {}: {error}", lane.id));

    let base_request = request(lane, size, conditioning);
    let inert = lane.prove_inert_identity.then(|| {
        one_image(
            generator
                .generate(&base_request, &mut |_: Progress| {})
                .unwrap_or_else(|error| panic!("{} inert render: {error}", lane.label)),
        )
    });

    let (sink, frames) = collecting_sink();
    let mut live_request = base_request;
    live_request.preview = sink;
    let mut events = 0usize;
    let live = one_image(
        generator
            .generate(&live_request, &mut |progress| {
                if matches!(progress, Progress::Step { .. }) {
                    events += 1;
                }
            })
            .unwrap_or_else(|error| panic!("{} live render: {error}", lane.label)),
    );

    if let Some(inert) = inert {
        assert_eq!(
            inert.pixels, live.pixels,
            "{}: a live preview sink changed the seeded render",
            lane.label
        );
    }

    let frames = candle_gen::lock_recover(&frames).clone();
    let metrics = analyze(lane.label, &frames, &live, lane.steps);
    save_artifacts(lane.label, &frames, &live, &metrics);

    assert!(
        metrics.movement.iter().all(|movement| *movement > 0.1),
        "{}: consecutive preview frames must differ: {:?}",
        lane.label,
        metrics.movement
    );
    assert!(
        metrics.distances.windows(2).all(|pair| pair[1] < pair[0]),
        "{}: every frame must approach the finished image: {:?}",
        lane.label,
        metrics.distances
    );
    assert!(
        metrics
            .correlations
            .windows(2)
            .all(|pair| pair[1] > pair[0]),
        "{}: resemblance must rise at every step: {:?}",
        lane.label,
        metrics.correlations
    );
    let first = metrics.correlations[0];
    let last = *metrics.correlations.last().unwrap();
    assert!(
        last > lane.min_r_last,
        "{}: terminal correlation {last:+.4} must clear its measured floor {:+.4}",
        lane.label,
        lane.min_r_last
    );
    assert!(
        last - first > 0.15,
        "{}: the strip must visibly develop ({first:+.4} -> {last:+.4})",
        lane.label
    );

    (events, frames)
}

fn base_lane(label: &'static str, sampler: Option<&'static str>, steps: u32) -> Lane<'static> {
    Lane {
        label,
        id: candle_gen_boogu::BOOGU_IMAGE_ID,
        snapshot_var: "BOOGU_BASE_SNAPSHOT",
        sampler,
        steps,
        min_r_last: BASE_MIN_R_LAST,
        prove_inert_identity: false,
    }
}

#[test]
#[ignore = "one-time CUDA acceptance only; not a CI job"]
fn base_default_lane_emits_a_converging_strip() {
    run_lane(base_lane("boogu-base-default", None, 8), Vec::new());
}

#[test]
#[ignore = "one-time CUDA acceptance only; not a CI job"]
fn base_heun_emits_once_per_outer_step() {
    let mut lane = base_lane("boogu-base-heun", Some("heun"), 4);
    lane.min_r_last = BASE_HEUN_MIN_R_LAST;
    let (events, frames) = run_lane(lane, Vec::new());
    assert!(
        events > lane.steps as usize,
        "Heun must evaluate more than once per outer step ({events} events for {} steps)",
        lane.steps
    );
    assert_eq!(frames.len(), lane.steps as usize);
}

#[test]
#[ignore = "one-time CUDA acceptance only; not a CI job"]
fn turbo_native_lane_is_decorative_and_converges() {
    run_lane(
        Lane {
            label: "boogu-turbo-native",
            id: candle_gen_boogu::BOOGU_IMAGE_TURBO_ID,
            snapshot_var: "BOOGU_TURBO_SNAPSHOT",
            sampler: None,
            steps: 4,
            min_r_last: TURBO_MIN_R_LAST,
            prove_inert_identity: true,
        },
        Vec::new(),
    );
}

#[test]
#[ignore = "one-time CUDA acceptance only; not a CI job"]
fn edit_default_lane_emits_a_converging_strip() {
    let reference = read_reference(&required_path("BOOGU_EDIT_REFERENCE"));
    run_lane(
        Lane {
            label: "boogu-edit-default",
            id: candle_gen_boogu::BOOGU_IMAGE_EDIT_ID,
            snapshot_var: "BOOGU_EDIT_SNAPSHOT",
            sampler: None,
            steps: 8,
            min_r_last: EDIT_MIN_R_LAST,
            prove_inert_identity: false,
        },
        vec![Conditioning::Reference {
            image: reference,
            strength: None,
        }],
    );
}
