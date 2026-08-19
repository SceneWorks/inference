//! **SC-15794**: rung 4's text-encoder scope on real weights — identity, and the bound it claims.
//!
//! Two things must both hold, and either alone is a false green:
//!
//! 1. **Identity.** The streamed conditioning tensor must equal the resident one. A rung is
//!    quality-preserving by definition, and `cap_feats` is what the whole denoise conditions on.
//! 2. **The bound is real.** A rung-4 implementation can look completely correct while saving
//!    nothing — the DiT twin measured 238.4 MiB vs 8.0 MiB purely on whether the carried state was
//!    materialized before the window dropped. So identity is asserted *alongside* a measured peak,
//!    and a mutation check proves the measurement can tell the two apart.
//!
//! ```sh
//! MLX_GEN_ZIMAGE_SNAPSHOT=<tier dir, e.g. …/q4> \
//!   cargo test -p mlx-gen-z-image --release --test text_encoder_window_real_weights \
//!   -- --ignored --nocapture --test-threads=1
//! ```

mod common;

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, WeightsSource};
use mlx_gen_z_image::text_encoder::{TextEncoder, ZTextEncoderConfig};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use mlx_rs::Array;

use common::tier_snapshot as snapshot;

const PROMPT: &str = "A photorealistic portrait of an elderly lighthouse keeper on a weathered \
     stone pier at dawn, heavy wool coat soaked with sea spray, low coastal fog behind him";

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// `(input_ids, attention_mask)` for the production prompt, from the tier's own tokenizer.
fn inputs(tier: &std::path::Path) -> (Array, Array) {
    let tokenizer = mlx_gen_z_image::load_tokenizer(tier).expect("load tokenizer");
    let tokens = tokenizer.tokenize(PROMPT).expect("tokenize");
    mlx_gen::tokenizer::to_arrays(&tokens)
}

fn te_dir(tier: &std::path::Path) -> std::path::PathBuf {
    tier.join("text_encoder")
}

fn streamable(weights: &Weights, dir: &std::path::Path, cfg: &ZTextEncoderConfig) -> TextEncoder {
    let source = WeightsSource::Dir(dir.to_path_buf());
    let validated = mlx_gen_z_image::ENCODER_CONTRACT
        .validate_source(&source)
        .expect("exact Z-Image text-encoder contract");
    TextEncoder::from_validated_streamable_source(weights, validated, "model", cfg)
        .expect("streamed encoder")
}

/// Max absolute difference over the two conditioning tensors, cast to f32.
fn max_abs_delta(a: &Array, b: &Array) -> f32 {
    assert_eq!(a.shape(), b.shape(), "conditioning shape changed");
    let n: i32 = a.shape().iter().product();
    let a = a
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap();
    let b = b
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap();
    a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// **Identity.** The streamed encoder must produce the same `cap_feats` as the resident one.
///
/// Asserted **exact**, not within a tolerance: streaming changes *when* a layer's weights exist, not
/// what arithmetic runs on them, and every layer is quantized identically to its resident twin. A
/// non-zero delta here means the streamed and resident stacks are not the same computation — a
/// tolerance would let that hide.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn the_streamed_encoder_matches_the_resident_encoder_exactly() {
    let tier = snapshot();
    let dir = te_dir(&tier);
    let cfg = ZTextEncoderConfig::z_image();
    let (input_ids, attention_mask) = inputs(&tier);

    let w = Weights::from_dir(&dir).expect("weights");
    let resident = TextEncoder::from_weights(&w, "model", &cfg).expect("resident encoder");
    let resident_out = resident
        .forward(&input_ids, &attention_mask)
        .expect("resident forward");
    mlx_rs::transforms::eval([&resident_out]).expect("eval");
    drop(resident);

    let streamed = streamable(&w, &dir, &cfg);
    assert!(streamed.is_streamable());
    drop(w);

    let cancel = CancelFlag::default();
    for window in [1usize, 4, 36] {
        let out = streamed
            .forward_windowed(&input_ids, &attention_mask, window, &cancel)
            .expect("windowed forward");
        mlx_rs::transforms::eval([&out]).expect("eval");
        let delta = max_abs_delta(&resident_out, &out);
        println!("SC-15794 identity at window={window}: max|Δ| {delta:e}");
        assert_eq!(
            delta, 0.0,
            "window={window}: streamed conditioning differs from resident by {delta:e} — the two \
             stacks are not the same computation"
        );
    }
}

/// A streamable encoder holds **no resident layers** — that is where its saving comes from. So a
/// plain, unscoped [`TextEncoder::forward`] on one must still run all 36 layers via the stream.
///
/// This is the highest-severity failure mode in the whole change: an empty stack would return the bare
/// token embedding as `cap_feats`, and the DiT would happily denoise it into a plausible-looking image
/// that ignored the prompt. No memory assertion and no compile error would catch it — only this.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn the_unscoped_forward_on_a_streamable_encoder_still_runs_every_layer() {
    let tier = snapshot();
    let dir = te_dir(&tier);
    let cfg = ZTextEncoderConfig::z_image();
    let (input_ids, attention_mask) = inputs(&tier);

    let w = Weights::from_dir(&dir).expect("weights");
    let resident = TextEncoder::from_weights(&w, "model", &cfg).expect("resident encoder");
    let expected = resident
        .forward(&input_ids, &attention_mask)
        .expect("resident forward");
    mlx_rs::transforms::eval([&expected]).expect("eval");
    drop(resident);

    let streamed = streamable(&w, &dir, &cfg);
    drop(w);
    let got = streamed
        .forward(&input_ids, &attention_mask)
        .expect("unscoped forward on a streamable encoder");
    mlx_rs::transforms::eval([&got]).expect("eval");

    let delta = max_abs_delta(&expected, &got);
    println!("SC-15794 unscoped forward on a streamable encoder: max|Δ| {delta:e}");
    assert_eq!(
        delta, 0.0,
        "an unscoped forward on a streamable encoder diverged by {delta:e} — if it returned the bare \
         embedding this is the whole conditioning signal being silently dropped"
    );

    // Mutation guard on the assertion above: the metric must be able to SEE a dropped stack. A
    // zero-layer encoder returns exactly what an empty-stack forward would have — the bare embedding —
    // and it must land far from the correct conditioning, or the equality above proves nothing.
    let mut zero_layers = ZTextEncoderConfig::z_image();
    zero_layers.n_layers = 0;
    let w2 = Weights::from_dir(te_dir(&tier)).expect("weights");
    let bare = TextEncoder::from_weights(&w2, "model", &zero_layers).expect("zero-layer encoder");
    let embedding_only = bare
        .forward(&input_ids, &attention_mask)
        .expect("bare forward");
    mlx_rs::transforms::eval([&embedding_only]).expect("eval");
    let dropped = max_abs_delta(&expected, &embedding_only);
    println!("SC-15794 mutation control (stack dropped entirely): max|Δ| {dropped:e}");
    assert!(
        dropped > 1e-3,
        "the metric cannot distinguish a full encode from the bare embedding ({dropped:e}) — the \
         assertion above would pass even with every layer skipped"
    );
}

/// The **policy gate**: only the `Sequential` loader builds a streamable encoder.
///
/// Under `Resident` the generator holds one encoder across every request, so a streamed stack would
/// re-materialize all 36 layers on every encode and bound nothing — measured ~2x slower per encode,
/// paid per dataset item in the trainer's caching loop and twice per generation on the CFG paths.
/// Making the streamable loader unconditional is an easy and completely silent regression: nothing
/// about the output changes, so only a test that looks at *which loader ran* can catch it.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn only_the_sequential_loader_builds_a_streamable_encoder() {
    let tier = snapshot();
    let resident = mlx_gen_z_image::load_text_encoder(&tier).expect("resident loader");
    assert!(
        !resident.is_streamable(),
        "the default (Resident) loader built a streamable encoder — every Resident generation and \
         every training encode would re-materialize the whole stack from disk for no bound"
    );
    let streamed = mlx_gen_z_image::load_text_encoder_streamable(&tier).expect("streamable loader");
    assert!(
        streamed.is_streamable(),
        "the Sequential loader did not build a streamable encoder — rung 4's text-encoder scope \
         would silently never engage"
    );
}

/// **The bound.** Peak active bytes across the conditioning forward, by window size.
///
/// The resident arm holds all 36 layers; each streamed arm holds `window` at a time. A rung-4
/// implementation that is correct but saves nothing passes the identity test above and fails here.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn the_window_bounds_the_conditioning_peak() {
    let tier = snapshot();
    let dir = te_dir(&tier);
    let cfg = ZTextEncoderConfig::z_image();
    let (input_ids, attention_mask) = inputs(&tier);
    let cancel = CancelFlag::default();

    let w = Weights::from_dir(&dir).expect("weights");
    let resident = TextEncoder::from_weights(&w, "model", &cfg).expect("resident encoder");
    clear_cache();
    reset_peak_memory();
    let out = resident
        .forward(&input_ids, &attention_mask)
        .expect("resident forward");
    mlx_rs::transforms::eval([&out]).expect("eval");
    let resident_peak = get_peak_memory() as u64;
    drop(out);
    drop(resident);
    clear_cache();

    let streamed = streamable(&w, &dir, &cfg);
    drop(w);

    println!(
        "\nSC-15794 conditioning peak by window ({}):",
        tier.display()
    );
    println!("  {:<10} {:>12} {:>10}", "arm", "peak", "vs resident");
    println!(
        "  {:<10} {:>11.3}G {:>10}",
        "resident",
        gib(resident_peak),
        "—"
    );
    let mut peaks = Vec::new();
    for window in [36usize, 8, 4, 2, 1] {
        clear_cache();
        reset_peak_memory();
        let out = streamed
            .forward_windowed(&input_ids, &attention_mask, window, &cancel)
            .expect("windowed forward");
        mlx_rs::transforms::eval([&out]).expect("eval");
        let peak = get_peak_memory() as u64;
        drop(out);
        clear_cache();
        let delta = 1.0 - peak as f64 / resident_peak.max(1) as f64;
        println!(
            "  w={:<8} {:>11.3}G {:>9.1}%",
            window,
            gib(peak),
            delta * 100.0
        );
        peaks.push((window, peak));
    }

    // The bound must actually bind: the tightest window has to beat the resident stack by a margin far
    // outside measurement noise. The measured weight share of this encoder is 58-91% depending on
    // tier, so a real stream moves the peak by tens of percent, not single digits.
    let (_, tightest) = *peaks.last().expect("a window arm");
    let saving = 1.0 - tightest as f64 / resident_peak.max(1) as f64;
    assert!(
        saving > 0.20,
        "window=1 peaked at {:.3} GiB against a resident {:.3} GiB ({:.1}%) — the stream is not \
         bounding anything. A rung-4 stack that produces correct output while freeing nothing is the \
         documented failure mode (the DiT twin: 238.4 MiB vs 8.0 MiB on the materialize guard alone).",
        gib(tightest),
        gib(resident_peak),
        saving * 100.0
    );

    // Monotone in the window: a tighter window must never peak higher than a looser one. This is what
    // distinguishes a real bound from a measurement that happens to land low once.
    for pair in peaks.windows(2) {
        let ((wa, pa), (wb, pb)) = (pair[0], pair[1]);
        assert!(
            pb <= pa + (pa / 20),
            "window={wb} peaked at {:.3} GiB, above window={wa}'s {:.3} GiB — the window is not \
             what is bounding the peak",
            gib(pb),
            gib(pa)
        );
    }
}

/// Cancellation lands at a window boundary and surfaces as the **typed** error.
///
/// A stringified cancellation reports a cancelled render as a FAILED job (sc-4481), and the window
/// boundary is the only place inside a conditioning forward where a cancel can land at all.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn a_cancel_stops_the_streamed_encoder_at_a_window_boundary() {
    let tier = snapshot();
    let dir = te_dir(&tier);
    let cfg = ZTextEncoderConfig::z_image();
    let (input_ids, attention_mask) = inputs(&tier);

    let w = Weights::from_dir(&dir).expect("weights");
    let streamed = streamable(&w, &dir, &cfg);
    drop(w);

    let cancel = CancelFlag::default();
    cancel.cancel();
    let err = streamed
        .forward_windowed(&input_ids, &attention_mask, 1, &cancel)
        .expect_err("a cancelled flag must stop the streamed encoder");
    assert!(
        matches!(err, mlx_gen::Error::Canceled),
        "expected the typed Error::Canceled, got {err}"
    );

    // An un-tripped flag changes nothing.
    let live = CancelFlag::default();
    assert!(streamed
        .forward_windowed(&input_ids, &attention_mask, 1, &live)
        .is_ok());
}

/// An encoder built without a re-openable source must **refuse** the rung rather than silently
/// running resident. A caller that selected rung 4 and got resident behaviour would record a verified
/// strategy that never executed.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn a_non_streamable_encoder_refuses_the_rung_instead_of_running_resident() {
    let tier = snapshot();
    let cfg = ZTextEncoderConfig::z_image();
    let (input_ids, attention_mask) = inputs(&tier);
    let w = Weights::from_dir(te_dir(&tier)).expect("weights");
    let resident = TextEncoder::from_weights(&w, "model", &cfg).expect("resident encoder");
    assert!(!resident.is_streamable());
    let err = resident
        .forward_windowed(&input_ids, &attention_mask, 1, &CancelFlag::default())
        .expect_err("a resident encoder must not silently satisfy the rung");
    assert!(
        format!("{err}").contains("re-openable"),
        "the refusal must name the reason, got: {err}"
    );
}
