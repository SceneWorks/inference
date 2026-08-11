//! sc-17140 / sc-17141: real-weight smokes for the MiniMax-H3 VAE decode paths.
//!
//! These are `#[ignore]`d — they need the ~9.7 GB `vae/` and ~0.6 GB `audio_vae/` components of a
//! `MiniMaxAI/MiniMax-H3` snapshot and Metal. Point `MINIMAX_H3_SNAPSHOT` at the snapshot root (the
//! directory holding `vae/`) and run:
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
use mlx_gen_minimax_h3::{
    kaiser_sinc_filter1d, MiniMaxH3AudioVae, MiniMaxH3AudioVaeConfig, MiniMaxH3VaeConfig,
    MiniMaxH3VideoVae,
};

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

// =============================================================================================
// sc-17141 — audio VAE
// =============================================================================================

/// Total tensors in the published `audio_vae/` component (encode + decode).
const PUBLISHED_AUDIO_TENSORS: usize = 1087;
/// Of those, the decode half this crate ports.
const AUDIO_DECODE_TENSORS: usize = 914;

/// The reference's own `from_pretrained` inputs, under `FL2VA/audio_vae/`.
fn audio_source_dir(root: &std::path::Path) -> std::path::PathBuf {
    let dir = root.join("FL2VA").join("audio_vae");
    for name in ["config.json", "config.yaml", "metadata.json"] {
        assert!(
            dir.join(name).is_file(),
            "MINIMAX_H3_SNAPSHOT is missing FL2VA/audio_vae/{name}. The root `audio_vae/` dir \
             cannot substitute: it is the diffusers repackaging and ships none of the three \
             `source_*_path` documents the reference loads."
        );
    }
    dir
}

fn read(path: std::path::PathBuf) -> String {
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// The published documents must parse to exactly [`MiniMaxH3AudioVaeConfig::default`], and the
/// diffusers-repackaged root config must agree with them.
///
/// This is the check that the constructor kwargs really come from `metadata.json` / `config.yaml`
/// rather than from a hardcoded table that happens to match: change either file and this fails.
/// It reads no weights.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); dev box / macos-mlx only"]
fn published_audio_configs_reproduce_the_declared_geometry() {
    let root = snapshot();
    let dir = audio_source_dir(&root);

    let cfg = MiniMaxH3AudioVaeConfig::from_source_files(
        &read(dir.join("config.json")),
        &read(dir.join("config.yaml")),
        &read(dir.join("metadata.json")),
    )
    .unwrap();
    assert_eq!(
        cfg,
        MiniMaxH3AudioVaeConfig::default(),
        "the shipped audio-VAE config drifted from the crate's declared default"
    );

    // The repackaged root config declares a subset of the same architecture.
    let repackaged = root.join("audio_vae").join("config.json");
    assert!(
        repackaged.is_file(),
        "no audio_vae/config.json in the snapshot"
    );
    cfg.cross_check_diffusers_json(&read(repackaged)).unwrap();

    // The published envelope, straight off the files.
    assert_eq!(cfg.sample_rate, 32_000);
    assert_eq!(cfg.output_channels, 2);
    assert_eq!(cfg.latent_channels, 32);
    assert_eq!(cfg.hop_length(), 800);
    assert_eq!(cfg.token_rate_hz(), 40.0);
    assert_eq!(cfg.bigvgan.num_mels, 2048);

    println!(
        "AUDIO CONFIG: sr {} · {} ch · {} latent ch · hop {} ({} Hz tokens) · decoder_dim {} · \
         num_mels {} · {} stages · {} AMP blocks",
        cfg.sample_rate,
        cfg.output_channels,
        cfg.latent_channels,
        cfg.hop_length(),
        cfg.token_rate_hz(),
        cfg.decoder_dim,
        cfg.bigvgan.num_mels,
        cfg.bigvgan.num_upsamples(),
        cfg.bigvgan.num_upsamples() * cfg.bigvgan.num_kernels(),
    );
}

/// The declared decode-path key set must be EXACTLY the published checkpoint's, minus the encode
/// half — asserted against BOTH published weight files, whose tensor names must also agree with
/// each other. Reads only the safetensors headers, so it costs no weight I/O.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); dev box / macos-mlx only"]
fn declared_audio_tensor_names_match_the_published_checkpoint() {
    let root = snapshot();
    let repackaged = Weights::from_dir(root.join("audio_vae")).unwrap();
    let source = Weights::from_dir(audio_source_dir(&root)).unwrap();

    let published: std::collections::BTreeSet<String> =
        repackaged.keys().map(str::to_string).collect();
    let source_keys: std::collections::BTreeSet<String> =
        source.keys().map(str::to_string).collect();
    assert_eq!(
        published.len(),
        PUBLISHED_AUDIO_TENSORS,
        "published audio_vae/ tensor count changed"
    );
    // The two published layouts are the same weights under the same names — which is why the
    // decode path can load either.
    assert_eq!(
        published, source_keys,
        "FL2VA/audio_vae/model.safetensors and audio_vae/diffusion_pytorch_model.safetensors \
         disagree on tensor names"
    );

    let declared: std::collections::BTreeSet<String> =
        MiniMaxH3AudioVae::tensor_names(&MiniMaxH3AudioVaeConfig::default())
            .into_iter()
            .collect();
    let missing: Vec<&String> = declared.difference(&published).collect();
    assert!(
        missing.is_empty(),
        "loader requires tensors the checkpoint does not have: {missing:?}"
    );

    // Everything outside the encode half must be consumed.
    let unconsumed: Vec<&String> = published
        .difference(&declared)
        .filter(|k| {
            !k.starts_with("encoder.")
                && !k.starts_with("mean_proj.")
                && !k.starts_with("logs_proj.")
                && !k.starts_with("pre_block.")
        })
        .collect();
    assert!(
        unconsumed.is_empty(),
        "checkpoint tensors outside the encode half that the decode path never reads: \
         {unconsumed:?}"
    );
    assert_eq!(declared.len(), AUDIO_DECODE_TENSORS);
    println!(
        "declared {} decode tensors; {} published; {} encode-half deliberately unported",
        declared.len(),
        published.len(),
        published.len() - declared.len()
    );
}

/// Load the real 605 MB audio VAE and decode a stereo latent end to end.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; ~0.6 GB resident"]
fn real_weight_audio_decode_produces_a_plausible_stereo_track() {
    let root = snapshot();
    let dir = audio_source_dir(&root);

    let started = Instant::now();
    let mut w = Weights::from_dir(root.join("audio_vae")).unwrap();
    let loaded = w.len();
    assert_eq!(
        loaded, PUBLISHED_AUDIO_TENSORS,
        "expected the full published audio_vae/ tensor set"
    );

    let cfg = MiniMaxH3AudioVaeConfig::from_source_files(
        &read(dir.join("config.json")),
        &read(dir.join("config.yaml")),
        &read(dir.join("metadata.json")),
    )
    .unwrap();
    // The checkpoint is f32 and it is only 605 MB, so there is nothing to gain from casting.
    let vae = MiniMaxH3AudioVae::from_weights(&mut w, &cfg, Dtype::Float32).unwrap();
    let load_ms = started.elapsed().as_millis();

    // Shape sanity straight off the real weights.
    assert_eq!(cfg.bigvgan.upsample_initial_channel, 1024);
    assert_eq!(cfg.bigvgan.num_mels, 2048);
    assert_eq!(cfg.bigvgan.stage_out_channels(6), 8);

    // The stored Kaiser-sinc buffers must be the ones `kaiser_sinc_filter1d` derives. The loader
    // uses the checkpoint's, so without this the derivation is never held against the real model.
    let derived = kaiser_sinc_filter1d(0.25, 0.3, 12).unwrap();
    for key in [
        "decoder.activation_post.upsample.filter",
        "decoder.resblocks.0.activations.0.downsample.lowpass.filter",
        "decoder.resblocks.20.activations.5.upsample.filter",
    ] {
        let stored = w.require(key).unwrap();
        assert_eq!(stored.shape(), &[1, 1, 12], "{key}");
        let err: f32 = mlx_rs::ops::subtract(&derived, stored)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        assert!(
            err < 1e-6,
            "{key}: the shipped filter differs from kaiser_sinc_filter1d(0.25, 0.3, 12) by {err:.3e}"
        );
    }

    // 20 tokens at 40 Hz = 0.5 s of 32 kHz stereo. The two channels are drawn INDEPENDENTLY, so a
    // mono-duplicating decode cannot pass the channel-gap check below.
    let tokens = 20;
    let n = 2 * 32 * tokens;
    let values: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.37).sin() * 0.8 + ((i as f32) * 0.13).cos() * 0.2)
        .collect();
    let latent = Array::from_slice(&values, &[1, 2, 32, tokens]);

    let decode_started = Instant::now();
    let track = vae.decode_audio_track(&latent).unwrap();
    let decode_ms = decode_started.elapsed().as_millis();

    // `decode_audio_track` pulls the samples back to the host, so this timing already spans the
    // forced evaluation — MLX is lazy and timing a graph build alone would report ~0 ms.
    assert_eq!(track.sample_rate, 32_000);
    assert_eq!(track.channels, 2);
    assert!(track.stems.is_empty());
    assert_eq!(track.samples.len(), (tokens * 800 * 2) as usize);
    assert!(
        track.samples.iter().all(|s| s.is_finite()),
        "real-weight audio decode produced NaN/Inf"
    );

    // A stub, an all-zero load or a silently-unwired decoder yields silence or a constant.
    let pcm = Array::from_slice(&track.samples, &[track.samples.len() as i32]);
    let spread = std_dev(&pcm);
    assert!(
        spread > 1e-4,
        "decoded audio is ~constant (std {spread:.3e}); the decoder did not really run"
    );
    let peak: f32 = pcm.abs().unwrap().max(None).unwrap().item();
    assert!(
        (1e-3..=1.0).contains(&peak),
        "decoded peak {peak:.3e} is implausible for a bounded vocoder output"
    );

    // The two channels must be genuinely different — the acceptance criterion a lazy test fails.
    let left: Vec<f32> = track.samples.iter().step_by(2).copied().collect();
    let right: Vec<f32> = track.samples.iter().skip(1).step_by(2).copied().collect();
    let gap = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        gap / peak > 1e-2,
        "the two decoded channels are near-identical (rel gap {:.3e}); the decode is \
         mono-duplicating",
        gap / peak
    );

    // Receipt. The evidence this really executed is the 1087-tensor load, the 2048-wide geometry,
    // the shipped Kaiser buffers matching the derivation, and 32000 finite non-constant samples
    // per channel in a plausible range — none of which hold without real weights and a real decode.
    println!(
        "REAL-WEIGHT AUDIO SMOKE: loaded {loaded} tensors ({load_ms} ms to map + build), decoded \
         {tokens} tokens -> {} interleaved samples ({:.3} s stereo @ {} Hz) in {decode_ms} ms \
         (std {spread:.4}, peak {peak:.4}, L-vs-R gap {:.4})",
        track.samples.len(),
        left.len() as f32 / track.sample_rate as f32,
        track.sample_rate,
        gap / peak,
    );
}
