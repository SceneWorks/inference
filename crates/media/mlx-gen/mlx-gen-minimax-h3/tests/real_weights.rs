//! sc-17140: real-weight smokes for the MiniMax-H3 video VAE decode.
//!
//! These are `#[ignore]`d — they need the ~9.7 GB `vae/` component of a `MiniMaxAI/MiniMax-H3`
//! snapshot and Metal. Point `MINIMAX_H3_SNAPSHOT` at the snapshot root (the directory holding
//! `vae/`) and run:
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<root> cargo test -p mlx-gen-minimax-h3 --test real_weights -- --ignored --nocapture
//! ```
//!
//! **A skipped run must not look like a passing one.** An `#[ignore]`d test that returns early
//! when its input is missing prints `ok` in 0.00s, which reads exactly like success. Every test
//! here therefore *asserts* on the snapshot (see `common::snapshot`) and then asserts on evidence
//! that the model actually executed — the loaded tensor count, the decoded shape, and the fact
//! that the output is finite and non-constant. None of those can hold unless real weights were
//! read and a real decode ran.

mod common;

use std::time::Instant;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::{MiniMaxH3VaeConfig, MiniMaxH3VideoVae};

use common::{snapshot, std_dev};

/// Total tensors in the published `vae/` component.
const PUBLISHED_VAE_TENSORS: usize = 703;
/// Of those, the encode half this crate does not port: 116 encoder + 2 `quant_conv`.
const ENCODE_HALF_TENSORS: usize = 118;

/// The declared decode-path key set must be EXACTLY the published checkpoint's, minus the encode
/// half. Reads only the shard index, so it costs no weight I/O.
///
/// This is the exhaustive-mapping proof against the real model rather than against the tiny
/// fixture: a tensor the loader never reads would decode to something plausible but wrong.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); dev box / macos-mlx only"]
fn declared_tensor_names_match_the_published_checkpoint() {
    let root = snapshot();
    let index = root
        .join("vae")
        .join("diffusion_pytorch_model.safetensors.index.json");
    let text = std::fs::read_to_string(&index)
        .unwrap_or_else(|e| panic!("reading {}: {e}", index.display()));
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    let map = json
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .expect("index.json has a weight_map");

    let published: std::collections::BTreeSet<String> = map.keys().cloned().collect();
    assert_eq!(
        published.len(),
        PUBLISHED_VAE_TENSORS,
        "published vae/ tensor count changed"
    );

    let cfg = MiniMaxH3VaeConfig::from_diffusers_json(
        &std::fs::read_to_string(root.join("vae").join("config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(cfg, MiniMaxH3VaeConfig::default(), "shipped config drifted");

    let declared: std::collections::BTreeSet<String> =
        MiniMaxH3VideoVae::tensor_names(&cfg).into_iter().collect();

    let missing: Vec<&String> = declared.difference(&published).collect();
    assert!(
        missing.is_empty(),
        "loader requires tensors the checkpoint does not have: {missing:?}"
    );

    let unconsumed: Vec<&String> = published
        .difference(&declared)
        .filter(|k| !k.starts_with("encoder.") && !k.starts_with("quant_conv."))
        .collect();
    assert!(
        unconsumed.is_empty(),
        "checkpoint tensors outside the encode half that the decode path never reads: \
         {unconsumed:?}"
    );

    assert_eq!(declared.len(), PUBLISHED_VAE_TENSORS - ENCODE_HALF_TENSORS);
    println!(
        "declared {} decode tensors; {} published; {} encode-half deliberately unported",
        declared.len(),
        published.len(),
        ENCODE_HALF_TENSORS
    );
}

/// Load the real 36-layer decoder and decode a small latent end to end.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; ~10 GB resident"]
fn real_weight_decode_produces_a_plausible_video() {
    let root = snapshot();

    let started = Instant::now();
    let dir = root.join("vae");
    let mut w = Weights::from_dir(&dir).unwrap();
    let loaded = w.len();
    assert_eq!(
        loaded, PUBLISHED_VAE_TENSORS,
        "expected the full published vae/ tensor set"
    );

    let cfg = MiniMaxH3VaeConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    // bf16 is the checkpoint's own dtype; loading f32 would double an already-10 GB decoder.
    let vae = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Bfloat16).unwrap();
    let load_ms = started.elapsed().as_millis();

    // Shape sanity straight off the real weights: 2048 dim, 36 blocks, 24 latent channels.
    assert_eq!(cfg.dim(), 2048);
    assert_eq!(cfg.num_layers, 36);
    assert_eq!(cfg.rope_apply_dim(), 48);
    assert_eq!(vae.geometry().tokens_chunk_size, 5);
    assert_eq!(vae.geometry().token_overlap, 2);
    assert_eq!(vae.geometry().frame_overlap, 5);

    // 7 temporal tokens -> one chunk -> 22 frames; 4x4 latent -> 64x64 pixels.
    let (tokens, lat_h, lat_w) = (7, 4, 4);
    let n = 24 * tokens * lat_h * lat_w;
    let values: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin() * 0.8).collect();
    let latent = Array::from_slice(&values, &[1, 24, tokens, lat_h, lat_w]);

    let decode_started = Instant::now();
    let video = vae.decode(&latent).unwrap();

    assert_eq!(
        video.shape(),
        &[1, 3, 22, lat_h * 16, lat_w * 16],
        "7 tokens should decode to 22 frames at 16x spatial / 4x temporal"
    );

    let video = video.as_dtype(Dtype::Float32).unwrap();
    // MLX is LAZY: `decode` above only builds the graph. Pulling a scalar back to the host is what
    // forces all 36 blocks to run, so the timer has to span this too — timing the call alone would
    // report ~0 ms and prove nothing. NaN/Inf both propagate through the sum, so this doubles as
    // the finiteness check (mlx-rs has no `is_finite` op).
    let checksum: f32 = video.sum(None).unwrap().item();
    let decode_ms = decode_started.elapsed().as_millis();
    assert!(checksum.is_finite(), "real-weight decode produced NaN/Inf");

    // A stub, an all-zero load or a silently-unwired decoder yields a constant frame.
    let spread = std_dev(&video);
    assert!(
        spread > 1e-3,
        "decoded video is ~constant (std {spread:.3e}); the decoder did not really run"
    );
    let max: f32 = video.abs().unwrap().max(None).unwrap().item();
    assert!(
        (1e-3..1e3).contains(&max),
        "decoded pixel magnitude {max:.3e} is implausible for a VAE output"
    );

    // Receipt. The evidence that this really executed is the 703-tensor load, the shape checks
    // against the real 2048-dim / 36-layer geometry, and a finite non-constant decode in a
    // plausible pixel range — not the wall-clock, which mmap and lazy evaluation both flatter.
    println!(
        "REAL-WEIGHT SMOKE: loaded {loaded} tensors ({load_ms} ms to map + build), decoded {:?} \
         in {decode_ms} ms (std {spread:.4}, max |px| {max:.4}, checksum {checksum:.3e})",
        video.shape()
    );
}

/// The multi-chunk path runs on real weights: 12 tokens span two chunks joined by the 5-frame
/// cross-fade, so this exercises the blend and the chunk-advance arithmetic that the single-chunk
/// smoke above never reaches.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; ~10 GB resident"]
fn real_weight_multi_chunk_decode_blends_the_seam() {
    let root = snapshot();
    let dir = root.join("vae");
    let mut w = Weights::from_dir(&dir).unwrap();
    let cfg = MiniMaxH3VaeConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let vae = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Bfloat16).unwrap();

    let (lat_h, lat_w) = (4, 4);
    let n = 24 * 12 * lat_h * lat_w;
    let values: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.29).cos() * 0.7).collect();
    let latent = Array::from_slice(&values, &[1, 24, 12, lat_h, lat_w]);

    let video = vae.decode(&latent).unwrap();
    // 12 tokens -> 2 chunks -> 17·2 + 5 = 39 frames.
    assert_eq!(video.shape(), &[1, 3, 39, lat_h * 16, lat_w * 16]);

    let video = video.as_dtype(Dtype::Float32).unwrap();
    let checksum: f32 = video.sum(None).unwrap().item();
    assert!(checksum.is_finite(), "multi-chunk decode produced NaN/Inf");
    let spread = std_dev(&video);
    assert!(spread > 1e-3, "multi-chunk decode is ~constant");

    println!(
        "REAL-WEIGHT SMOKE: 12 tokens -> {:?} across 2 blended chunks (std {spread:.4})",
        video.shape()
    );
}
