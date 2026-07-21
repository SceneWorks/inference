//! Real-weight conformance for the candle Synchformer visual encoder (sc-13438).
//!
//! ## What this gates on real weights
//!
//! Loads the pinned `hkchengrex/MMAudio` `ext_weights/synchformer_state_dict.pth` (~907 MB), builds
//! the MotionFormer visual encoder from its `vfeat_extractor.*` sub-tree, and asserts the ported
//! forward produces coherent sync features:
//!
//! - [`synchformer_features_shape_finite_deterministic`] — real frames → `(S, 8, 768)` features
//!   (segment-level, 768-d), every value finite, and **byte-identical run-to-run** (deterministic).
//!   A broken weight mapping (wrong key names / transposed Conv3d / mis-ordered divided attention)
//!   would surface here as a load error, a NaN, or a shape mismatch.
//! - [`synchformer_features_are_frame_varying`] — two clips with **different frame content** yield
//!   materially different features, and features vary **across segments** within a moving clip. This
//!   is the coherent-output check: an encoder that ignored its input (or collapsed to a constant)
//!   would fail. A static clip's segments are near-identical while a moving clip's are not.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test conformance -- --ignored --nocapture
//! ```
//! Set `SYNCHFORMER_SNAPSHOT` to a `synchformer_state_dict.pth` file (or a dir containing it under
//! `ext_weights/` or at its root), or leave unset to resolve the pinned checkpoint via the audio
//! lane's F-029 hub path (downloads ~907 MB into the ordinary HF cache on first run).

use candle_audio_mmaudio as sf;
use candle_audio_mmaudio::candle_audio::candle_core::Device;
use image::{Rgb, RgbImage};

/// Resolve the encoder from the required `SYNCHFORMER_SNAPSHOT` env path — inference never
/// self-fetches or derives a cache location (epic 13657).
fn load_encoder() -> sf::SynchformerVisualEncoder {
    let dev = Device::Cpu;
    let p = std::env::var("SYNCHFORMER_SNAPSHOT").expect(
        "set SYNCHFORMER_SNAPSHOT to a synchformer_state_dict.pth file or its snapshot dir",
    );
    let path = std::path::PathBuf::from(&p);
    if path.is_dir() {
        sf::load(&sf::gen_core::WeightsSource::Dir(path), &dev)
            .expect("load synchformer from SYNCHFORMER_SNAPSHOT dir")
    } else {
        sf::load_from_pth(&path, &dev).expect("load synchformer from SYNCHFORMER_SNAPSHOT file")
    }
}

/// A synthetic RGB frame: a moving gradient whose phase depends on `t` (frame index) and `clip`.
fn frame(t: usize, clip: u8) -> RgbImage {
    let (w, h) = (320u32, 240u32);
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 * 0.02 + t as f32 * 0.3).sin() * 127.0 + 128.0) as u8;
            let g = ((y as f32 * 0.02 + t as f32 * 0.2 + clip as f32).cos() * 127.0 + 128.0) as u8;
            let b = (((x + y) as f32 * 0.01 + t as f32 * 0.1).sin() * 127.0 + 128.0) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    img
}

fn moving_clip(n: usize, clip: u8) -> Vec<RgbImage> {
    (0..n).map(|t| frame(t, clip)).collect()
}

fn static_clip(n: usize, clip: u8) -> Vec<RgbImage> {
    (0..n).map(|_| frame(7, clip)).collect()
}

fn encode(enc: &sf::SynchformerVisualEncoder, frames: &[RgbImage]) -> Vec<f32> {
    let segs = sf::preprocess::frames_to_segments(frames, enc.device()).expect("segments");
    let segs = enc.prepare_input(&segs).expect("prepare input");
    let feats = enc.encode(&segs).expect("encode");
    let dims = feats.dims().to_vec();
    assert_eq!(dims.len(), 3, "features must be (S, t, 768), got {dims:?}");
    assert_eq!(dims[1], 8, "8 temporal tokens per segment");
    assert_eq!(dims[2], 768, "768-d sync features");
    feats.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
#[ignore = "downloads ~907MB synchformer_state_dict.pth; run explicitly with --ignored"]
fn synchformer_features_shape_finite_deterministic() {
    let enc = load_encoder();
    // 40 frames @ 25fps ≈ 1.6s → (40-16)/8+1 = 4 segments.
    let frames = moving_clip(40, 0);
    let a = encode(&enc, &frames);
    assert_eq!(a.len(), 4 * 8 * 768, "4 segments × 8 × 768");
    assert!(a.iter().all(|v| v.is_finite()), "all features finite");
    // Determinism: re-encode the same frames → identical bytes.
    let b = encode(&enc, &frames);
    assert_eq!(a, b, "encoder must be deterministic run-to-run");
    // Non-degenerate: features are not all (near) equal.
    let mean = a.iter().sum::<f32>() / a.len() as f32;
    let var = a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / a.len() as f32;
    assert!(
        var > 1e-6,
        "features must not be a constant vector (var={var})"
    );
    eprintln!(
        "synchformer real-weights: shape=(4,8,768) len={} mean={mean:.4} var={var:.4} min={:.3} max={:.3}",
        a.len(),
        a.iter().cloned().fold(f32::INFINITY, f32::min),
        a.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
}

#[test]
#[ignore = "downloads ~907MB synchformer_state_dict.pth; run explicitly with --ignored"]
fn synchformer_features_are_frame_varying() {
    let enc = load_encoder();
    // Two clips with genuinely different frame content.
    let clip0 = encode(&enc, &moving_clip(24, 0)); // 2 segments
    let clip1 = encode(&enc, &moving_clip(24, 1));
    let max_diff = clip0
        .iter()
        .zip(&clip1)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "different frame content must produce different features (max|Δ|={max_diff})"
    );

    // Within a MOVING clip, consecutive segments differ; within a STATIC clip they are ~identical.
    let moving = encode(&enc, &moving_clip(24, 0));
    let seg = 8 * 768;
    let moving_seg_diff = moving[0..seg]
        .iter()
        .zip(&moving[seg..2 * seg])
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let stat = encode(&enc, &static_clip(24, 0));
    let static_seg_diff = stat[0..seg]
        .iter()
        .zip(&stat[seg..2 * seg])
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "synchformer frame-variance: cross-clip max|Δ|={max_diff:.4}, moving inter-segment max|Δ|={moving_seg_diff:.4}, static inter-segment max|Δ|={static_seg_diff:.5}"
    );
    assert!(
        moving_seg_diff > static_seg_diff,
        "a moving clip's segments must vary more than a static clip's (moving={moving_seg_diff}, static={static_seg_diff})"
    );
}
