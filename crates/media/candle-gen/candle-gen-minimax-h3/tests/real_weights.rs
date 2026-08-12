//! sc-17154: real-weight smokes for the candle MiniMax-H3 VAE decode paths.
//!
//! These are `#[ignore]`d — they need the ~9.7 GB `vae/` and ~0.6 GB `audio_vae/` components of a
//! `MiniMaxAI/MiniMax-H3` snapshot. Point `MINIMAX_H3_SNAPSHOT` at the snapshot root (the directory
//! holding `vae/`) and run:
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<root> \
//!   cargo test -p candle-gen-minimax-h3 --test real_weights -- --ignored --nocapture
//! ```
//!
//! On the self-hosted Windows CUDA runner (sc-18677) the snapshot is provisioned at
//! `E:\huggingface\hub\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765`.
//! Everything loads at **f32** so the numbers are comparable with the committed-fixture lane; that
//! is ~19.4 GB resident for the video decoder and ~1.2 GB for the audio one. Two of the tests below
//! read only safetensors headers and cost no weight I/O at all.
//!
//! **A skipped run must not look like a passing one.** An `#[ignore]`d test that returns early when
//! its input is missing prints `ok` in 0.00s, which reads exactly like success. Every test here
//! therefore *asserts* on the snapshot (see `common::snapshot`) and then asserts on evidence that
//! the model actually executed — the published tensor count, the decoded shape, and the fact that
//! the output is finite and non-constant. None of those can hold unless real weights were read.

mod common;

use std::collections::BTreeSet;

use common::{flat, read_snapshot_file, rel, safetensors_keys, snapshot, std_dev};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::loader::sorted_safetensors;
use candle_gen_minimax_h3::{
    kaiser_sinc_filter1d, MiniMaxH3AudioVae, MiniMaxH3AudioVaeConfig, MiniMaxH3VaeConfig,
    MiniMaxH3VideoVae,
};

/// Total tensors in the published `vae/` component.
const VAE_TENSOR_COUNT: usize = 703;
/// Of those, the encode half this crate does not port: 116 encoder + 2 `quant_conv`.
const VAE_ENCODE_TENSORS: usize = 118;
/// Total tensors in the published `audio_vae/` component (encode + decode).
const AUDIO_TENSOR_COUNT: usize = 1087;
/// Of those, the decode half this crate ports.
const AUDIO_DECODE_TENSORS: usize = 914;

/// Where real-weight decodes run. On a `--features cuda` build this is the GPU; on the CPU lane it
/// is the CPU, and either way the dtype is f32 so the residuals are comparable with the
/// committed-fixture suite.
fn device() -> Device {
    candle_gen::default_device().expect("a candle device")
}

fn shard_keys(dir: &std::path::Path, label: &str) -> BTreeSet<String> {
    let files = sorted_safetensors(dir, label).expect("safetensors shards");
    assert!(!files.is_empty(), "{}: no shards", dir.display());
    let mut keys = BTreeSet::new();
    for file in &files {
        for k in safetensors_keys(file) {
            assert!(keys.insert(k.clone()), "duplicate tensor {k} across shards");
        }
    }
    keys
}

// =============================================================================================
// Video VAE
// =============================================================================================

/// The declared decode-path key set must be EXACTLY the published checkpoint's, minus the encode
/// half. Reads only the safetensors headers, so it costs no weight I/O.
///
/// This is the exhaustive-mapping proof against the real model rather than against the tiny
/// fixture: a tensor the loader never reads would decode to something plausible but wrong.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT)"]
fn declared_tensor_names_match_the_published_checkpoint() {
    let root = snapshot();
    let published = shard_keys(&root.join("vae"), "minimax-h3 vae");
    assert_eq!(
        published.len(),
        VAE_TENSOR_COUNT,
        "the published vae/ component should carry {VAE_TENSOR_COUNT} tensors"
    );

    let encode: BTreeSet<&String> = published
        .iter()
        .filter(|k| k.starts_with("encoder.") || k.starts_with("quant_conv."))
        .collect();
    assert_eq!(encode.len(), VAE_ENCODE_TENSORS);

    let decode: BTreeSet<String> = published
        .iter()
        .filter(|k| !encode.contains(k))
        .cloned()
        .collect();

    let cfg = MiniMaxH3VaeConfig::from_diffusers_json(&read_snapshot_file(
        &root.join("vae").join("config.json"),
    ))
    .expect("parse vae/config.json");
    assert_eq!(cfg, MiniMaxH3VaeConfig::default(), "the shipped geometry");

    let declared: BTreeSet<String> = MiniMaxH3VideoVae::tensor_names(&cfg).into_iter().collect();
    assert_eq!(
        declared, decode,
        "the declared decode key set must be exactly the published one minus the encode half"
    );
    println!(
        "video vae: {} published tensors, {} encode-half, {} declared and consumed",
        published.len(),
        encode.len(),
        declared.len()
    );
}

/// Load the real 36-layer decoder and decode a small latent end to end.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); ~19.4 GB resident at f32"]
fn real_weight_decode_produces_a_plausible_video() {
    let root = snapshot();
    let dev = device();
    let vae = MiniMaxH3VideoVae::load(&root, &dev, DType::F32).expect("load the real video VAE");
    let cfg = vae.config().clone();
    assert_eq!(cfg.num_layers, 36, "the real 36-layer transformer decoder");
    assert_eq!(cfg.dim(), 2048);

    // 5 temporal tokens of a 2x3 latent grid -> 17 frames at 32x48 px.
    let (t, h, w) = (5usize, 2usize, 3usize);
    let n = cfg.latent_channels * t * h * w;
    let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin() * 0.8).collect();
    let latent = Tensor::from_vec(vals, (1, cfg.latent_channels, t, h, w), &dev).expect("latent");

    let video = vae.decode(&latent).expect("decode");
    assert_eq!(
        video.dims(),
        &[1, 3, 17, h * 16, w * 16],
        "16x spatial / 4x temporal upsampling with the clip_length-17 chunk plan"
    );

    let px = flat(&video);
    assert!(
        px.iter().all(|v| v.is_finite()),
        "decode produced non-finite pixels"
    );
    let spread = std_dev(&video);
    println!(
        "real video decode: {:?}, std {spread:.4}, |max| {:.4}",
        video.dims(),
        px.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    );
    assert!(
        spread > 1e-3,
        "the decode is ~constant ({spread:.3e}); real weights would not produce that"
    );
}

/// **The gate a self-generated checksum cannot be.** Decode the *same latent* the official
/// diffusers `AutoencoderKLMiniMaxH3` decoded, from the *same published bytes*, and compare
/// numerically.
///
/// Why the smoke above cannot do this job: it asserts `std` and finiteness, which any
/// plausible-but-wrong port also satisfies — the sc-18740 gate/value half-swap changed neither.
/// Reproducing an independent implementation's output on real weights is the only assertion that
/// can catch a layout error, because layout errors are invisible to shape, magnitude and checksum.
///
/// Generate the reference with the MLX lane's `tools/dump_minimax_h3_video_vae_real.py` (a few
/// hundred KB, deliberately not committed) and point `MINIMAX_H3_VIDEO_VAE_REFERENCE` at it. Like
/// every test in this file it **asserts** rather than skipping, so a missing reference cannot read
/// as a pass.
#[test]
#[ignore = "needs a real snapshot + a reference decode (MINIMAX_H3_VIDEO_VAE_REFERENCE)"]
fn real_weight_decode_matches_the_official_diffusers_vae() {
    let root = snapshot();
    let raw = std::env::var("MINIMAX_H3_VIDEO_VAE_REFERENCE").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_VIDEO_VAE_REFERENCE must point at a reference decode produced by \
         tools/dump_minimax_h3_video_vae_real.py. This test asserts rather than skips so a missing \
         reference cannot be mistaken for a pass."
    );
    let reference = common::Golden::load(&raw);
    for key in ["in.latent", "out.video"] {
        assert!(
            reference.has(key),
            "{raw} has no `{key}`; it is not a MiniMax-H3 real-weight reference decode"
        );
    }

    let dev = device();
    let vae = MiniMaxH3VideoVae::load(&root, &dev, DType::F32).expect("load the real video VAE");
    let latent = reference
        .tensor("in.latent")
        .to_device(&dev)
        .expect("latent");
    let want = reference.tensor("out.video");
    let got = vae
        .decode(&latent)
        .expect("decode")
        .to_device(&Device::Cpu)
        .expect("to cpu");

    assert_eq!(got.dims(), want.dims(), "decoded shape");
    let (peak, mean) = rel(&got, &want);
    let cos = common::cosine(&got, &want);
    println!(
        "real-weight decode vs diffusers AutoencoderKLMiniMaxH3: peak rel {peak:.3e}, mean rel \
         {mean:.3e}, cosine {cos:.6}"
    );
    // The reference runs bf16 or f32 depending on how it was dumped; 2e-2 is the bar the MLX lane
    // uses for the same comparison, and the sc-18740 half-swap sits at 0.86-0.99 — two orders
    // clear of it either way.
    assert!(
        peak < 2e-2,
        "the candle decode diverges from the official diffusers VAE by {peak:.3e}; a gated-FFN \
         half-swap or a QKV mis-split lands here (sc-18740)"
    );
    assert!(
        std_dev(&want) > 1e-3,
        "the reference decode is ~constant; it would gate nothing"
    );
}

/// The multi-chunk path runs on real weights: 12 tokens span two chunks joined by the 5-frame
/// cross-fade, so this exercises the blend and the chunk-advance arithmetic that the single-chunk
/// smoke above never reaches.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); ~19.4 GB resident at f32"]
fn real_weight_multi_chunk_decode_blends_the_seam() {
    let root = snapshot();
    let dev = device();
    let vae = MiniMaxH3VideoVae::load(&root, &dev, DType::F32).expect("load the real video VAE");
    let g = vae.geometry();
    assert_eq!(g.tokens_chunk_size, 5);
    assert_eq!(g.frame_overlap, 5, "the cross-faded seam");

    let (t, h, w) = (12usize, 2usize, 2usize);
    let cfg = vae.config();
    let n = cfg.latent_channels * t * h * w;
    let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.21).cos() * 0.7).collect();
    let latent = Tensor::from_vec(vals, (1, cfg.latent_channels, t, h, w), &dev).expect("latent");

    let video = vae.decode(&latent).expect("decode");
    assert_eq!(
        video.dims(),
        &[1, 3, 39, h * 16, w * 16],
        "12 tokens decode to 39 frames across two chunks"
    );
    assert!(flat(&video).iter().all(|v| v.is_finite()));

    // A cross-faded seam must not be a visible discontinuity: the frame-to-frame delta across the
    // seam should be the same order as elsewhere. Frames 17..22 carry the blend.
    let frame = |i: usize| flat(&video.narrow(2, i, 1).expect("frame"));
    let delta = |a: usize, b: usize| {
        frame(a)
            .iter()
            .zip(frame(b).iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let seam = delta(16, 17).max(delta(21, 22));
    let interior = delta(5, 6).max(delta(30, 31));
    println!("real multi-chunk decode: seam delta {seam:.4}, interior delta {interior:.4}");
    assert!(
        seam < interior * 10.0,
        "the chunk seam ({seam:.4}) is an order louder than ordinary frame-to-frame motion \
         ({interior:.4}); the cross-fade is mis-aligned"
    );
}

// =============================================================================================
// Audio VAE
// =============================================================================================

/// The reference's own `from_pretrained` inputs, under `FL2VA/audio_vae/`.
fn audio_source_dir(root: &std::path::Path) -> std::path::PathBuf {
    let dir = root.join("FL2VA").join("audio_vae");
    assert!(
        dir.join("config.json").is_file(),
        "{} has no config.json; the snapshot is missing the FL2VA audio_vae source bundle",
        dir.display()
    );
    dir
}

/// The published documents must parse to exactly [`MiniMaxH3AudioVaeConfig::default`], and the
/// diffusers-repackaged root config must agree with them.
///
/// This is the check that the constructor kwargs really come from `metadata.json` / `config.yaml`
/// rather than from a hardcoded table that happens to match: change either file and this fails. It
/// reads no weights.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT)"]
fn published_audio_configs_reproduce_the_declared_geometry() {
    let root = snapshot();
    let src = audio_source_dir(&root);
    let cfg = MiniMaxH3AudioVaeConfig::from_source_files(
        &read_snapshot_file(&src.join("config.json")),
        &read_snapshot_file(&src.join("config.yaml")),
        &read_snapshot_file(&src.join("metadata.json")),
    )
    .expect("parse the FL2VA audio source triple");
    assert_eq!(
        cfg,
        MiniMaxH3AudioVaeConfig::default(),
        "the published documents must reproduce the declared config exactly"
    );

    cfg.cross_check_diffusers_json(&read_snapshot_file(
        &root.join("audio_vae").join("config.json"),
    ))
    .expect("the repackaged root config must agree with the source triple");

    println!(
        "audio vae: sample_rate {}, latent_dim {}, hop {}, {} upsample stages",
        cfg.sample_rate,
        cfg.bigvgan.num_mels,
        cfg.hop_length(),
        cfg.bigvgan.num_upsamples()
    );
}

/// The declared decode-path key set must be EXACTLY the published checkpoint's, minus the encode
/// half — asserted against BOTH published weight files, whose tensor names must also agree with
/// each other. Reads only the safetensors headers, so it costs no weight I/O.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT)"]
fn declared_audio_tensor_names_match_the_published_checkpoint() {
    let root = snapshot();
    let repackaged = shard_keys(&root.join("audio_vae"), "minimax-h3 audio_vae");
    let source = shard_keys(&audio_source_dir(&root), "minimax-h3 FL2VA audio_vae");
    assert_eq!(
        repackaged, source,
        "the two published audio VAE forms must carry identical tensor names (the conversion is \
         an identity — sc-18740's audit result)"
    );
    assert_eq!(repackaged.len(), AUDIO_TENSOR_COUNT);

    let declared: BTreeSet<String> =
        MiniMaxH3AudioVae::tensor_names(&MiniMaxH3AudioVaeConfig::default())
            .into_iter()
            .collect();
    assert_eq!(declared.len(), AUDIO_DECODE_TENSORS);
    assert!(
        declared.is_subset(&repackaged),
        "declared audio tensors absent from the checkpoint: {:?}",
        declared.difference(&repackaged).collect::<Vec<_>>()
    );

    // The remainder is the encode half, and it is exactly what the module docs claim.
    let unread: Vec<&String> = repackaged.difference(&declared).collect();
    assert_eq!(unread.len(), AUDIO_TENSOR_COUNT - AUDIO_DECODE_TENSORS);
    assert!(
        unread.iter().all(|k| k.starts_with("encoder.")
            || k.starts_with("mean_proj.")
            || k.starts_with("logs_proj.")
            || k.starts_with("pre_block.")),
        "an unread tensor is outside the encode half: {unread:?}"
    );
    println!(
        "audio vae: {} published tensors, {} declared and consumed, {} encode-half",
        repackaged.len(),
        declared.len(),
        unread.len()
    );
}

/// Load the real ~605 MB audio VAE and decode a stereo latent end to end.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); ~1.2 GB resident at f32"]
fn real_weight_audio_decode_produces_a_plausible_stereo_track() {
    let root = snapshot();
    let dev = device();
    let cfg = MiniMaxH3AudioVaeConfig::default();
    let files =
        sorted_safetensors(&root.join("audio_vae"), "minimax-h3 audio_vae").expect("audio shards");
    let w = candle_gen::Weights::from_files_filtered(
        &files,
        &dev,
        DType::F32,
        &["dec_in_proj.", "decoder."],
    )
    .expect("load the audio decode half");
    let vae = MiniMaxH3AudioVae::from_weights(&w, &cfg, &dev, DType::F32)
        .expect("build the real audio VAE");

    // 40 latent tokens = 1.0 s of 32 kHz stereo.
    let tokens = 40usize;
    let n = 2 * cfg.latent_channels * tokens;
    let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).sin() * 0.9).collect();
    let z = Tensor::from_vec(vals, (1, 2, cfg.latent_channels, tokens), &dev).expect("latent");

    let track = vae.decode_audio_track(&z).expect("decode_audio_track");
    assert_eq!(track.sample_rate, 32_000);
    assert_eq!(track.channels, 2);
    assert!(track.stems.is_empty());
    assert_eq!(
        track.samples.len(),
        tokens * 800 * 2,
        "40 tokens at 40 Hz = 1 s of interleaved stereo at 32 kHz"
    );
    assert!(track.samples.iter().all(|s| s.is_finite()));
    assert!(track.samples.iter().all(|s| (-1.0..=1.0).contains(s)));

    let mean = track.samples.iter().sum::<f32>() / track.samples.len() as f32;
    let spread = (track
        .samples
        .iter()
        .map(|s| (s - mean) * (s - mean))
        .sum::<f32>()
        / track.samples.len() as f32)
        .sqrt();
    let saturated = track
        .samples
        .iter()
        .filter(|s| s.abs() >= 1.0 - 1e-6)
        .count() as f32
        / track.samples.len() as f32;
    println!(
        "real audio decode: {} samples, std {spread:.4}, {:.2}% clamped",
        track.samples.len(),
        saturated * 100.0
    );
    assert!(spread > 1e-3, "the waveform is ~silent ({spread:.3e})");

    // The two channels must be genuinely different — a mono-duplicating port would pass every
    // shape and finiteness assertion above.
    let gap = (0..tokens * 800)
        .map(|t| (track.samples[2 * t] - track.samples[2 * t + 1]).abs())
        .fold(0.0f32, f32::max);
    println!("  L-vs-R peak gap {gap:.4}");
    assert!(
        gap > 1e-3,
        "the two decoded channels are near-identical ({gap:.3e})"
    );
}

/// The Kaiser-sinc taps the port DERIVES must reproduce the ones the real checkpoint STORES.
///
/// The loader reads the stored buffers, so without this the derivation would only ever be checked
/// against the tiny fixture. Reads a handful of 12-float buffers, so it costs no meaningful I/O.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT)"]
fn stored_kaiser_filters_match_the_derivation_on_real_weights() {
    let root = snapshot();
    let files =
        sorted_safetensors(&root.join("audio_vae"), "minimax-h3 audio_vae").expect("audio shards");
    let w = candle_gen::Weights::from_files_filtered(
        &files,
        &Device::Cpu,
        DType::F32,
        &[
            "decoder.activation_post.",
            "decoder.resblocks.0.activations.0.",
        ],
    )
    .expect("load the stored filters");

    let derived = kaiser_sinc_filter1d(0.25, 0.3, 12, &Device::Cpu).expect("derive");
    let mut checked = 0usize;
    for key in [
        "decoder.activation_post.upsample.filter",
        "decoder.activation_post.downsample.lowpass.filter",
        "decoder.resblocks.0.activations.0.upsample.filter",
        "decoder.resblocks.0.activations.0.downsample.lowpass.filter",
    ] {
        let stored = w.require(key).expect(key);
        assert_eq!(stored.dims(), &[1, 1, 12], "{key} is not a 12-tap buffer");
        let (peak, _) = rel(&derived, &stored);
        println!("  {key}: peak rel {peak:.3e}");
        assert!(
            peak < 1e-5,
            "{key} disagrees with `kaiser_sinc_filter1d` by {peak:.3e}; the derivation and the \
             shipped buffers describe different filters"
        );
        checked += 1;
    }
    assert_eq!(checked, 4, "every filter must have been read");
}
