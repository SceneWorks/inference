//! SC-15800: real-weight Lens text-encoder identity, window-bound, policy, and prompt-length evidence.
//!
//! Run one tier at a time so process-level MLX peaks cannot leak across tiers:
//!
//! ```sh
//! LENS_DIR=<SceneWorks/lens-mlx/q4 or lens-turbo-mlx/bf16> \
//!   cargo test -p mlx-gen-lens --release --test text_encoder_window_real_weights \
//!   -- --ignored --nocapture --test-threads=1
//! ```

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Quant, WeightsSource};
use mlx_gen_lens::config::GptOssConfig;
use mlx_gen_lens::text::LensTokenizer;
use mlx_gen_lens::text_encoder::encoder::{LensTextEncoder, DEFAULT_SELECTED_LAYERS};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use mlx_rs::ops::{abs, max, multiply, subtract, sum};
use mlx_rs::{Array, Dtype};

const DATE: &str = "2026-07-31";
const PROMPT: &str = "A weathered lighthouse keeper standing on a stone pier at dawn, sea spray, \
    low coastal fog, documentary photograph";
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn snapshot() -> std::path::PathBuf {
    let path = std::path::PathBuf::from(
        std::env::var("LENS_DIR").expect("set LENS_DIR to an explicit local Lens tier directory"),
    );
    assert!(
        path.is_dir(),
        "Lens tier does not exist: {}",
        path.display()
    );
    path
}

fn tier(root: &std::path::Path) -> Option<Quant> {
    match root.file_name().and_then(|name| name.to_str()) {
        Some("q4") => Some(Quant::Q4),
        Some("q8") => Some(Quant::Q8),
        _ => None,
    }
}

fn input_ids(root: &std::path::Path, prompt: &str) -> Array {
    let tokenizer = LensTokenizer::from_file(root.join("tokenizer/tokenizer.json"))
        .expect("load Lens tokenizer");
    let tokens = tokenizer.encode(prompt, DATE).expect("tokenize prompt");
    Array::from_slice(&tokens.ids, &[1, tokens.ids.len() as i32])
}

fn resident(root: &std::path::Path) -> LensTextEncoder {
    LensTextEncoder::from_weights_quant(
        Weights::from_dir(root.join("text_encoder")).expect("load text-encoder weights"),
        &GptOssConfig::lens(),
        Dtype::Bfloat16,
        tier(root),
    )
    .expect("build resident encoder")
}

fn streamed(root: &std::path::Path) -> LensTextEncoder {
    let dir = root.join("text_encoder");
    LensTextEncoder::from_streamable_source(
        Weights::from_dir(&dir).expect("open text-encoder weights"),
        WeightsSource::Dir(dir),
        &GptOssConfig::lens(),
        Dtype::Bfloat16,
        DEFAULT_SELECTED_LAYERS.to_vec(),
        tier(root),
    )
    .expect("build streamable encoder")
}

fn eval_all(values: &[Array]) {
    mlx_rs::transforms::eval(values.iter()).expect("evaluate conditioning");
}

fn max_abs_delta(a: &[Array], b: &[Array]) -> f32 {
    assert_eq!(a.len(), b.len(), "captured layer count changed");
    a.iter().zip(b).fold(0.0_f32, |worst, (a, b)| {
        assert_eq!(a.shape(), b.shape(), "captured layer shape changed");
        let a = a.as_dtype(Dtype::Float32).unwrap();
        let b = b.as_dtype(Dtype::Float32).unwrap();
        a.as_slice::<f32>()
            .iter()
            .zip(b.as_slice::<f32>())
            .fold(worst, |m, (x, y)| m.max((x - y).abs()))
    })
}

/// The established Lens BF16 encoder metrics. Dense resident and streamed execution schedule the
/// same operations differently on Metal, so absolute equality is not a valid cross-schedule gate
/// for the model's very wide activation range (see `encoder_parity.rs`).
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn cosine(got: &Array, want: &Array) -> f32 {
    let dot = sum(multiply(got, want).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let na = sum(multiply(got, got).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let nb = sum(multiply(want, want).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

fn assert_resident_parity(
    root: &std::path::Path,
    expected: &[Array],
    got: &[Array],
    window: usize,
) {
    let delta = max_abs_delta(expected, got);
    if tier(root).is_some() {
        println!(
            "SC-15800 Lens identity window={window}: max|delta|={delta:e} (exact quantized gate)"
        );
        assert_eq!(delta, 0.0, "window={window} changed quantized conditioning");
        return;
    }

    let mut worst_peak = 0.0_f32;
    let mut worst_cos = 1.0_f32;
    for (capture, (want, actual)) in expected.iter().zip(got).enumerate() {
        let pr = peak_rel(actual, want);
        let cos = cosine(actual, want);
        worst_peak = worst_peak.max(pr);
        worst_cos = worst_cos.min(cos);
        println!(
            "SC-15800 Lens BF16 window={window} capture={capture}: peak_rel={pr:.3e} cosine={cos:.7}"
        );
    }
    println!(
        "SC-15800 Lens BF16 window={window}: max|delta|={delta:e} worst_peak_rel={worst_peak:.3e} worst_cosine={worst_cos:.7}"
    );
    assert!(
        worst_cos > 0.995,
        "window={window} worst cosine {worst_cos:.7} is below the established Lens BF16 floor"
    );
    assert!(
        worst_peak < 0.15,
        "window={window} worst peak_rel {worst_peak:.3e} exceeds the established Lens BF16 floor"
    );
}

#[test]
#[ignore = "needs real Lens weights and Apple/Metal"]
fn streamed_conditioning_matches_resident_and_unscoped_runs_every_layer() {
    let root = snapshot();
    let ids = input_ids(&root, PROMPT);
    let resident = resident(&root);
    let expected = resident.encode(&ids, None).expect("resident encode");
    eval_all(&expected);
    drop(resident);
    clear_cache();

    let streamed = streamed(&root);
    assert!(streamed.is_streamable());
    for window in [1usize, 4, 24] {
        let got = streamed
            .encode_windowed(&ids, window, &CancelFlag::default())
            .expect("streamed encode");
        eval_all(&got);
        assert_resident_parity(&root, &expected, &got, window);

        if window == 1 {
            let repeated = streamed
                .encode_windowed(&ids, window, &CancelFlag::default())
                .expect("repeat streamed encode");
            eval_all(&repeated);
            assert_eq!(
                max_abs_delta(&got, &repeated),
                0.0,
                "the streamed BF16 schedule is not deterministic"
            );
        }
    }

    let unscoped = streamed
        .encode(&ids, None)
        .expect("unscoped streamed encode");
    eval_all(&unscoped);
    assert_resident_parity(&root, &expected, &unscoped, 24);

    // Zero-layer mutation control: reproduce the exact catastrophic empty-stack output (the bare
    // token embedding) and prove the metric above can see it. Without this, an equality check over a
    // accidentally inert prompt/metric could pass while every decoder layer was skipped.
    let bare_weights = Weights::from_dir(root.join("text_encoder")).expect("mutation weights");
    let bare = bare_weights
        .require("model.embed_tokens.weight")
        .expect("embedding")
        .as_dtype(Dtype::Bfloat16)
        .expect("embedding dtype")
        .take_axis(&ids, 0)
        .expect("embedding lookup");
    eval_all(std::slice::from_ref(&bare));
    let dropped = max_abs_delta(
        std::slice::from_ref(&expected[3]),
        std::slice::from_ref(&bare),
    );
    println!("SC-15800 zero-layer mutation control: max|delta|={dropped:e}");
    assert!(
        dropped > 1e-3,
        "the metric cannot distinguish a full encode from the bare embedding ({dropped:e})"
    );
}

#[test]
#[ignore = "needs real Lens weights and Apple/Metal"]
fn the_window_is_load_bearing_and_prompt_length_is_swept() {
    let root = snapshot();
    let streamed = streamed(&root);
    let cancel = CancelFlag::default();
    let prompts = [
        "red fox",
        PROMPT,
        "A carefully staged large-format editorial photograph of a red fox crossing a snow-covered \
         boreal clearing before sunrise, with windblown ice crystals, distant spruce silhouettes, \
         subtle tracks, layered blue fog, natural rim light, realistic fur detail, restrained color \
         grading, and a low camera angle that preserves the broad winter landscape",
    ];

    println!(
        "\nSC-15800 Lens conditioning peak sweep ({})",
        root.display()
    );
    println!("  {:>6} {:>7} {:>12}", "window", "tokens", "peak GiB");
    let mut tightest = None;
    let ids = input_ids(&root, PROMPT);
    for window in [24usize, 8, 4, 2, 1] {
        clear_cache();
        reset_peak_memory();
        let out = streamed
            .encode_windowed(&ids, window, &cancel)
            .expect("windowed encode");
        eval_all(&out);
        let peak = get_peak_memory() as u64;
        println!(
            "  {:>6} {:>7} {:>12.3}",
            window,
            ids.shape()[1],
            peak as f64 / GIB
        );
        if window == 24 {
            tightest = Some((peak, 0_u64));
        } else if window == 1 {
            tightest.as_mut().unwrap().1 = peak;
        }
        drop(out);
        clear_cache();
    }
    let (all_covering, one) = tightest.expect("window endpoints");
    assert!(
        one < all_covering.saturating_sub(all_covering / 10),
        "window=1 ({:.3} GiB) did not beat the all-covering mutation control ({:.3} GiB) by >10%; \
         the materialize/drain/release bound is not load-bearing",
        one as f64 / GIB,
        all_covering as f64 / GIB
    );

    // Lens does NOT pad to a fixed length: the harmony preamble is fixed, but prompt tokens remain
    // variable. Sweep all three lengths under the production-tight window and report the true peak.
    let mut token_counts = Vec::new();
    for prompt in prompts {
        let ids = input_ids(&root, prompt);
        clear_cache();
        reset_peak_memory();
        let out = streamed
            .encode_windowed(&ids, 1, &cancel)
            .expect("prompt-length encode");
        eval_all(&out);
        let peak = get_peak_memory() as u64;
        println!(
            "  {:>6} {:>7} {:>12.3}  prompt-length sweep",
            1,
            ids.shape()[1],
            peak as f64 / GIB
        );
        token_counts.push(ids.shape()[1]);
        drop(out);
        clear_cache();
    }
    assert!(
        token_counts.windows(2).all(|pair| pair[0] < pair[1]),
        "prompt sweep did not produce strictly increasing token lengths: {token_counts:?}"
    );
}

#[test]
#[ignore = "needs real Lens weights and Apple/Metal"]
fn resident_is_not_streamable_and_cancellation_stays_typed() {
    let root = snapshot();
    let ids = input_ids(&root, PROMPT);
    let resident = resident(&root);
    assert!(
        !resident.is_streamable(),
        "Resident must keep its warm stack"
    );
    let error = resident
        .encode_windowed(&ids, 1, &CancelFlag::default())
        .expect_err("Resident must refuse a selected stream");
    assert!(error.to_string().contains("Sequential loader"));
    drop(resident);
    clear_cache();

    let streamed = streamed(&root);
    let cancel = CancelFlag::default();
    cancel.cancel();
    let error = streamed
        .encode_windowed(&ids, 1, &cancel)
        .expect_err("cancelled stream must stop");
    assert!(matches!(error, mlx_gen::Error::Canceled));
}
