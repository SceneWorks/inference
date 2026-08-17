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
use mlx_gen_minimax_h3::spatial_tiling::{
    SpatialTiling, TilePlan, TILE_SAMPLE_MIN_OVERLAP, TILE_SAMPLE_MIN_SIZE,
};
use mlx_gen_minimax_h3::{
    kaiser_sinc_filter1d, DitBlock, MiniMaxH3AudioVae, MiniMaxH3AudioVaeConfig, MiniMaxH3DitConfig,
    MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer, MiniMaxH3VaeConfig,
    MiniMaxH3VideoVae, MmRope, TokenRefiner, APPLIES_CHAT_TEMPLATE, LM_PREFIX,
    MINIMAX_ADDED_SPECIALS, VISION_PREFIX,
};

use common::{cosine, l2_norm, rel, snapshot, std_dev};

/// Total tensors in the published `vae/` component.
const PUBLISHED_VAE_TENSORS: usize = 703;
/// Of those, the encode half: 116 `encoder.*` + 2 `quant_conv.*`. Unported until sc-17148, which
/// `fl2va` forced — a keyframe is conditioned through the VAE as well as the vision tower.
const ENCODE_HALF_TENSORS: usize = 118;

/// The declared key set must be EXACTLY the published checkpoint's — **both halves**, since
/// sc-17148. Reads only the shard index, so it costs no weight I/O.
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

    let unconsumed: Vec<&String> = published.difference(&declared).collect();
    assert!(
        unconsumed.is_empty(),
        "published tensors the loader never reads: {unconsumed:?}"
    );

    // The mapping is now the WHOLE file, not the decode half with a documented omission.
    assert_eq!(declared.len(), PUBLISHED_VAE_TENSORS);
    let encode_half: Vec<&String> = declared
        .iter()
        .filter(|k| k.starts_with("encoder.") || k.starts_with("quant_conv."))
        .collect();
    assert_eq!(
        encode_half.len(),
        ENCODE_HALF_TENSORS,
        "the encode half must be exactly the 118 keys the published index carries"
    );
    println!(
        "declared {} tensors ({} decode + {} encode); {} published",
        declared.len(),
        declared.len() - ENCODE_HALF_TENSORS,
        ENCODE_HALF_TENSORS,
        published.len()
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
    // bf16 is a DOWNCAST here, not the checkpoint's own dtype: `vae/` ships F32 (703 of 703
    // tensors, 9.70 GiB, measured from the published headers), so this halves it rather than
    // matching it. f32 would be a no-op cast holding the full 9.70 GiB — which is what the parity
    // gate below deliberately does — not the doubling this comment used to assert.
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

/// **The gate sc-17140 was missing** (sc-18740): decode the *same latent* the official diffusers
/// `AutoencoderKLMiniMaxH3` decoded, from the *same published bytes*, and compare numerically.
///
/// Why the smoke above cannot do this job: it asserts `std`, `max |px|` and a checksum that were
/// recorded **from this port**. Those re-derive whatever the port does — the gate/value half-swap
/// changed every one of them and the test would simply have been written against the new values.
/// Reproducing an independent implementation's output on real weights is the only assertion that
/// can catch a layout error, because layout errors are invisible to shape, magnitude and checksum.
///
/// Generate the reference with `tools/dump_minimax_h3_video_vae_real.py` (a few hundred KB,
/// deliberately not committed) and point `MINIMAX_H3_VIDEO_VAE_REFERENCE` at it. Like every test in
/// this file it **asserts** rather than skipping, so a missing reference cannot read as a pass.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot + a reference decode (MINIMAX_H3_VIDEO_VAE_REFERENCE)"]
fn real_weight_decode_matches_the_official_diffusers_vae() {
    let root = snapshot();
    let reference_path = std::env::var("MINIMAX_H3_VIDEO_VAE_REFERENCE").unwrap_or_default();
    assert!(
        !reference_path.is_empty(),
        "MINIMAX_H3_VIDEO_VAE_REFERENCE must point at the output of \
         tools/dump_minimax_h3_video_vae_real.py. This test is #[ignore]d and asserts rather than \
         skips so a missing reference cannot be mistaken for a pass."
    );
    let r = Weights::from_file(&reference_path)
        .unwrap_or_else(|e| panic!("load {reference_path}: {e}"));
    assert_eq!(
        r.metadata("reference").unwrap_or_default(),
        "diffusers.AutoencoderKLMiniMaxH3",
        "the reference must come from the official converted-checkpoint class"
    );

    let dir = root.join("vae");
    let mut w = Weights::from_dir(&dir).unwrap();
    let cfg = MiniMaxH3VaeConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    // f32 to keep the comparison about layout rather than about bf16 rounding. That is free, not
    // costly: `vae/` is already F32 on disk (703 of 703 tensors, 9.70 GiB), so this is a no-op cast
    // holding ~9.70 GiB — NOT the ~20 GB this comment used to claim, which came from reading the
    // component as bf16 and doubling it.
    let vae = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Float32).unwrap();

    let latent = r.require("in.latent").unwrap();
    let want = r.require("out.video").unwrap();
    let got = vae
        .decode(latent)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    assert_eq!(got.shape(), want.shape(), "decoded shape");

    let diff = mlx_rs::ops::subtract(&got, want).unwrap().abs().unwrap();
    let peak: f32 = want.abs().unwrap().max(None).unwrap().item();
    let rel_max: f32 = diff.max(None).unwrap().item::<f32>() / peak;
    let cos = {
        let dot: f32 = mlx_rs::ops::multiply(&got, want)
            .unwrap()
            .sum(None)
            .unwrap()
            .item();
        let a: f32 = got
            .square()
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>()
            .sqrt();
        let b: f32 = want
            .square()
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>()
            .sqrt();
        dot / (a * b)
    };
    let (n_got, n_want) = (
        got.square()
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>()
            .sqrt(),
        want.square()
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>()
            .sqrt(),
    );
    println!(
        "REAL-WEIGHT PARITY vs {} {}: rel-max-abs={rel_max:.3e} cosine={cos:.6} \
         ||port||={n_got:.4} ||reference||={n_want:.4}",
        r.metadata("reference").unwrap_or_default(),
        r.metadata("reference_version").unwrap_or_default(),
    );

    // 1e-2 is this crate's house tolerance; MLX's reduced-precision Metal f32 matmul over a
    // 36-layer / 2048-dim stack is the floor. The sc-18740 half-swap sits at 0.86-0.99 here, i.e.
    // roughly two orders of magnitude above it.
    assert!(
        rel_max < 1e-2,
        "real-weight decode differs from the official implementation by {rel_max:.3e} \
         (cosine {cos:.6}). Norms are {n_got:.4} vs {n_want:.4} — note how little those move, \
         which is why no magnitude or checksum assertion can gate this."
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
// sc-19445 — the video VAE ENCODE half
//
// sc-17148 landed the MLX encoder with no real-weight gate at all: the only thing validating it
// was `tests/fixtures/video_vae_encode.safetensors`, and **that fixture structurally cannot
// express a level-indexing error.** Its geometry is
//
//     block_out_channels          (32, 32, 32, 64)
//     spatial_downsample_factors  (1, 2, 2, 1)
//     temporal_downsample_factors (1, 2, 2, 1)      <- IDENTICAL to the spatial list
//     norm_num_groups             32 == the width at levels 0-2  <- instance norm on 3 of 4 levels
//
// against the shipped
//
//     block_out_channels          (128, 256, 256, 512, 512, 1024)
//     spatial_downsample_factors  (2, 2, 2, 2, 1, 1)
//     temporal_downsample_factors (1, 2, 2, 1, 1, 1)  <- DIFFERENT from the spatial list
//     norm_num_groups             32 over 128..1024
//
// Three consequences, and each is a defect the fixture reports green:
//
//  * the toy's two factor lists agree, so `encoder_level_has_downsampler`'s product predicate,
//    the spatial factor alone and the temporal factor alone are the SAME function on it. On the
//    shipped config they disagree at levels 0 and 3. That is the mutation
//    `real_weight_encode_matches_the_official_diffusers_vae` is demonstrated against below.
//  * toy levels 1 and 2 are indistinguishable — same in/out width, same factors, same downsampler,
//    same absence of `conv_shortcut` — so swapping the CONFIG level between them is bit-inert
//    there and is a load failure on the shipped stack, where level 1 changes width and level 2
//    does not.
//  * `groups == channels` at levels 0-2 makes the frame-isolated GroupNorm an instance norm there.
//    Only level 3 groups at all, and only 2 channels to a group against the shipped 4 to 32, so
//    the grouping is very nearly unobservable at fixture scale.
//
// The two tests below are the two halves the fixture leaves open: the published *shapes* against a
// table derived from the config, and the published *values* against an independent implementation.
// =============================================================================================

/// The bound `real_weight_encode_matches_the_official_diffusers_vae` gates on, **derived from this
/// lane's own measured floor** rather than inherited from the decode lane or from candle.
///
/// Measured on `nax-macos` (Mac17,6, Metal) against
/// `tools/dump_minimax_h3_video_vae_encode_real.py` @ snapshot `939557dc`, diffusers 0.40.0.dev0,
/// all six (probe, moment) pairs:
///
/// ```text
/// keyframe.mean  4.930e-4     keyframe.std  1.385e-3
/// clip.mean      5.580e-4     clip.std      1.514e-3
/// multiclip.mean 6.351e-4     multiclip.std 1.789e-3   <- the floor
/// ```
///
/// so the floor is **1.789e-3** and this bound is ~5.6x it.
///
/// **It has to be measured here and cannot be borrowed.** MLX's reduced-precision Metal f32 matmul
/// over the six-level 128..1024 conv stack sits two orders above candle's f32 CPU path for the
/// identical comparison — `REAL_WEIGHT_ENCODE_TOL` is 2e-4 there — which is the same split the
/// decode lane shows (~4.96e-3 on MLX against ~4.73e-6 on candle). Taking candle's 2e-4 would make
/// this test fail on clean code; taking a round house value would state a convention in the voice
/// of a derivation.
///
/// **The exposure sits on `std`, and predictably so.** `std` is `exp(0.5 · logvar)`, so an
/// absolute wobble in `logvar` returns as a *relative* one here, magnified by the exponential.
/// Every one of the three worst entries above is a `std`; every `mean` is ~3x tighter.
///
/// The headroom over 1.789e-3 covers a device change — `rw-mage` is on both Macs and only one of
/// them has been measured — not a defect. Every defect class this epic has actually shipped is
/// orders clear of it: the sc-18740 gate/value half-swap at 0.86-0.99, a symmetric downsampler pad
/// at 1.8, a global GroupNorm at 1.6, a front pad at 6.9e-1, and an untiled encode at **9.430e-2**
/// — that last one re-measured on this lane, below, as the discrimination check.
///
/// **WHAT 1e-2 DOES NOT BOUND.** Every class in that list is a LAYOUT or STRUCTURE defect, and each
/// clears the bound by one to two orders; for those, 1e-2 is loose in the right direction. It is
/// not a bound on SCALE or PRECISION defects at all. The metric is relative, so a uniform gain
/// error anywhere up to ±1% sits under it by construction no matter what it does to the image —
/// which is precisely the shape of the 2% scale drift this story demonstrates against (it fires at
/// only 1.956e-2, ~2x the bound, while cosine reads exactly 1.0000000). Read 1e-2 as "the layout is
/// right", never as "the numbers are right to 1%".
const REAL_WEIGHT_ENCODE_TOL: f32 = 1e-2;

/// `name -> shape` over the `vae/` shard headers, read without any weight I/O.
///
/// Shards are enumerated from the published `index.json` weight map rather than by globbing the
/// directory: an AppleDouble `._`-prefixed sidecar matches `*.safetensors` and is not one.
fn vae_shard_shapes(dir: &std::path::Path) -> std::collections::BTreeMap<String, Vec<i32>> {
    let index = dir.join("diffusion_pytorch_model.safetensors.index.json");
    let text = std::fs::read_to_string(&index)
        .unwrap_or_else(|e| panic!("reading {}: {e}", index.display()));
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    let shards: std::collections::BTreeSet<String> = json
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .expect("index.json has a weight_map")
        .values()
        .filter_map(|v| v.as_str().map(str::to_owned))
        .collect();
    assert!(!shards.is_empty(), "the weight map names no shard");

    let mut out = std::collections::BTreeMap::new();
    for shard in &shards {
        let path = dir.join(shard);
        let mut f =
            std::fs::File::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let mut len = [0u8; 8];
        std::io::Read::read_exact(&mut f, &mut len).expect("safetensors header length");
        let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
        std::io::Read::read_exact(&mut f, &mut header).expect("safetensors header");
        let header: serde_json::Value =
            serde_json::from_slice(&header).expect("safetensors header json");
        for (k, v) in header.as_object().expect("header object") {
            if k == "__metadata__" {
                continue;
            }
            let shape: Vec<i32> = v["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|d| d.as_i64().expect("dim") as i32)
                .collect();
            assert!(
                out.insert(k.clone(), shape).is_none(),
                "{k} appears in more than one shard"
            );
        }
    }
    out
}

/// **Every declared encoder tensor's SHAPE, derived from the config and checked against the
/// published shard headers** — plus the level structure the toy fixture has no room to carry.
///
/// `declared_tensor_names_match_the_published_checkpoint` above is necessary and not sufficient: a
/// port can name all 118 encode keys correctly and still read them at the wrong *level*, because
/// `Weights` hands back whatever is stored under a name and neither the key-set proof nor the
/// fixture can see a 512-wide conv where the checkpoint has 1024.
///
/// **The table is DERIVED, not transcribed.** Every expectation comes from
/// `MiniMaxH3VaeConfig`'s own accessors — `encoder_level_in_channels`, `block_out_channels`,
/// `encoder_level_has_downsampler` — so it is a statement about the port's geometry that the
/// published bytes then judge. A table copied out of the headers would agree with them by
/// construction and prove nothing.
///
/// Reads headers only: no weight I/O, no Metal.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); shard headers only, no weight I/O"]
fn declared_encoder_shapes_match_the_published_checkpoint() {
    let root = snapshot();
    let dir = root.join("vae");
    let cfg = MiniMaxH3VaeConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let shapes = vae_shard_shapes(&dir);
    assert_eq!(
        shapes.len(),
        PUBLISHED_VAE_TENSORS,
        "published vae/ tensor count changed"
    );

    // ---- the derived table -------------------------------------------------------------------
    let mut want: std::collections::BTreeMap<String, Vec<i32>> = Default::default();
    let c0 = cfg.block_out_channels[0];
    want.insert(
        "encoder.conv_in.weight".into(),
        vec![c0, cfg.in_channels, 3, 3, 3],
    );
    want.insert("encoder.conv_in.bias".into(), vec![c0]);
    for level in 0..cfg.num_encoder_levels() {
        let out = cfg.block_out_channels[level];
        for j in 0..cfg.layers_per_block {
            let inp = if j == 0 {
                cfg.encoder_level_in_channels(level)
            } else {
                out
            };
            let p = format!("encoder.down_blocks.{level}.resnets.{j}");
            want.insert(format!("{p}.norm1.weight"), vec![inp]);
            want.insert(format!("{p}.norm1.bias"), vec![inp]);
            want.insert(format!("{p}.conv1.weight"), vec![out, inp, 3, 3, 3]);
            want.insert(format!("{p}.conv1.bias"), vec![out]);
            want.insert(format!("{p}.norm2.weight"), vec![out]);
            want.insert(format!("{p}.norm2.bias"), vec![out]);
            want.insert(format!("{p}.conv2.weight"), vec![out, out, 3, 3, 3]);
            want.insert(format!("{p}.conv2.bias"), vec![out]);
            // The residual projection exists exactly where the width changes — the same predicate
            // `ResnetBlock3d::from_weights` uses to decide whether to load one.
            if inp != out {
                want.insert(format!("{p}.conv_shortcut.weight"), vec![out, inp, 1, 1, 1]);
                want.insert(format!("{p}.conv_shortcut.bias"), vec![out]);
            }
        }
        if cfg.encoder_level_has_downsampler(level) {
            let p = format!("encoder.down_blocks.{level}.downsamplers.0.conv");
            want.insert(format!("{p}.weight"), vec![out, out, 3, 3, 3]);
            want.insert(format!("{p}.bias"), vec![out]);
        }
    }
    let last = *cfg.block_out_channels.last().expect("at least one level");
    want.insert("encoder.norm_out.weight".into(), vec![last]);
    want.insert("encoder.norm_out.bias".into(), vec![last]);
    // `conv_out` and `quant_conv` are twice the latent width: they carry mean AND logvar.
    let params = 2 * cfg.latent_channels;
    want.insert(
        "encoder.conv_out.weight".into(),
        vec![params, last, 3, 3, 3],
    );
    want.insert("encoder.conv_out.bias".into(), vec![params]);
    want.insert("quant_conv.weight".into(), vec![params, params, 1, 1, 1]);
    want.insert("quant_conv.bias".into(), vec![params]);

    assert_eq!(
        want.len(),
        ENCODE_HALF_TENSORS,
        "the derived table must cover all {ENCODE_HALF_TENSORS} encode tensors"
    );
    // …and it must describe exactly the tensors the loader declares, or the table is checking a
    // different model than the one that runs.
    let declared: std::collections::BTreeSet<String> = MiniMaxH3VideoVae::tensor_names(&cfg)
        .into_iter()
        .filter(|k| k.starts_with("encoder.") || k.starts_with("quant_conv."))
        .collect();
    assert_eq!(
        want.keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        declared,
        "the derived shape table and the declared encode key set name different tensors"
    );

    for (key, expected) in &want {
        let got = shapes
            .get(key)
            .unwrap_or_else(|| panic!("the published vae/ has no `{key}`"));
        assert_eq!(
            got, expected,
            "{key}: published shape vs the port's geometry"
        );
    }

    // ---- the level structure the 4-level toy cannot carry ------------------------------------
    // Asserted against the PUBLISHED index, not against the config that produced the table above,
    // so this is an independent statement rather than a restatement of it.
    let published: std::collections::BTreeSet<&String> = shapes.keys().collect();
    let has = |k: &str| published.contains(&k.to_string());
    let any = |needle: &str| published.iter().any(|k| k.contains(needle));

    for level in 0..4 {
        assert!(
            has(&format!(
                "encoder.down_blocks.{level}.downsamplers.0.conv.weight"
            )),
            "level {level} must carry a downsampler in the published checkpoint"
        );
    }
    for level in [4, 5] {
        assert!(
            !any(&format!("down_blocks.{level}.downsamplers")),
            "level {level} must carry NO downsampler; applying one where none exists is the \
             sc-19445 defect class"
        );
    }
    for level in [1, 3, 5] {
        assert!(
            has(&format!(
                "encoder.down_blocks.{level}.resnets.0.conv_shortcut.weight"
            )),
            "level {level} changes width and must carry a residual projection"
        );
    }
    for level in [0, 2, 4] {
        assert!(
            !any(&format!(
                "encoder.down_blocks.{level}.resnets.0.conv_shortcut"
            )),
            "level {level} keeps its width and must carry no residual projection"
        );
    }
    // The port's own predicates must agree with the checkpoint on both counts, per level. This is
    // the half that turns the structural facts above into a gate on the LOADER rather than on the
    // bytes: `encoder_level_has_downsampler` is what decides whether the forward pass strides.
    for level in 0..cfg.num_encoder_levels() {
        assert_eq!(
            cfg.encoder_level_has_downsampler(level),
            any(&format!("down_blocks.{level}.downsamplers")),
            "level {level}: the port and the checkpoint disagree about a downsampler"
        );
        assert_eq!(
            cfg.encoder_level_in_channels(level) != cfg.block_out_channels[level],
            any(&format!(
                "encoder.down_blocks.{level}.resnets.0.conv_shortcut"
            )),
            "level {level}: the port and the checkpoint disagree about a residual projection"
        );
    }

    // Every 3x3x3 kernel really is 3x3x3 — a 1x3x3 would make the whole temporal-causality
    // discussion vacuous — and `conv_shortcut` really is the pointwise 1x1x1.
    let kernel = |k: &str| shapes[k][2..].to_vec();
    assert_eq!(kernel("encoder.conv_in.weight"), vec![3, 3, 3]);
    assert_eq!(
        kernel("encoder.down_blocks.0.downsamplers.0.conv.weight"),
        vec![3, 3, 3]
    );
    assert_eq!(
        kernel("encoder.down_blocks.1.resnets.0.conv_shortcut.weight"),
        vec![1, 1, 1]
    );

    println!(
        "REAL-WEIGHT ENCODE SHAPE TABLE: all {} derived encode shapes match the published \
         checkpoint; block_out_channels {:?}, spatial {:?}, temporal {:?}; downsamplers on levels \
         {:?}, residual projections on levels {:?}",
        want.len(),
        cfg.block_out_channels,
        cfg.spatial_downsample_factors,
        cfg.temporal_downsample_factors,
        (0..cfg.num_encoder_levels())
            .filter(|l| cfg.encoder_level_has_downsampler(*l))
            .collect::<Vec<_>>(),
        (0..cfg.num_encoder_levels())
            .filter(|l| cfg.encoder_level_in_channels(*l) != cfg.block_out_channels[*l])
            .collect::<Vec<_>>(),
    );
}

/// **The gate sc-17148 shipped without.** Encode the *same pixels* the official diffusers
/// `AutoencoderKLMiniMaxH3` encoded, from the *same published bytes*, and compare numerically.
///
/// Why the committed fixture cannot do this job: see the section header above. The decisive one is
/// that the toy's spatial and temporal factor lists are *the same list*, so a level predicate that
/// reads the wrong one of them is the identity there and mis-strides the real stack.
///
/// Three probes, each reaching a path the other two leave dark (the generator derives all three
/// from the shipped config and refuses to write a probe that would be inert):
///
/// | probe | frames | what only it reaches |
/// |---|---|---|
/// | `keyframe` | 1 | the single-frame short circuit, and the shipped 256/64 spatial tiling at 320 px |
/// | `clip` | 17 | the temporal strides and the `token_drop` tail trim, untiled |
/// | `multiclip` | 20 | the frame-repeat pad to a multiple of `clip_length`, and the clip-by-clip concatenation |
///
/// 20 rather than any other ragged count, because `token_drop` otherwise *hides* the pad: only the
/// first `tokens_chunk_size - token_drop` latent frames of the final clip survive, and those reach
/// back over clip-local pixels `0 ..= pad_reach` only. At 25 frames, repeating the FIRST frame
/// instead of the last leaves the encode bit-identical — measured on this epic, not reasoned. The
/// assertion below re-derives `pad_reach` from the shipped config so it cannot silently stop
/// holding.
///
/// **Gated on relative max-abs-diff.** Not norm, not cosine, not a checksum: all three are printed
/// beside it and none of them is asserted on, because all three were blind to real defects in this
/// family seven times. sc-18740 shipped a functionally wrong VAE green at cosine 0.73-0.78 with
/// norms 89 vs 85.
///
/// Generate the reference with `tools/dump_minimax_h3_video_vae_encode_real.py` (~5.7 MB,
/// deliberately not committed) and point `MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE` at it. Like every
/// test in this file it **asserts** rather than skipping, so a missing reference cannot read as a
/// pass.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot + a reference encode (MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE); ~9.7 GiB resident"]
fn real_weight_encode_matches_the_official_diffusers_vae() {
    let root = snapshot();
    let reference_path = std::env::var("MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE").unwrap_or_default();
    assert!(
        !reference_path.is_empty(),
        "MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE must point at the output of \
         tools/dump_minimax_h3_video_vae_encode_real.py. This test is #[ignore]d and asserts \
         rather than skips so a missing reference cannot be mistaken for a pass."
    );
    let r = Weights::from_file(&reference_path)
        .unwrap_or_else(|e| panic!("load {reference_path}: {e}"));
    // layout.rs Rule 3: only a golden built from the CONVERTED checkpoint can validate a loader
    // that reads the converted checkpoint. A golden dumped from the MiniMax source modules would
    // share this loader's key names and disagree with the shipped bytes — sc-18740's mechanism.
    assert_eq!(
        r.metadata("provenance").unwrap_or_default(),
        "converted-checkpoint",
        "the reference must come from `AutoencoderKLMiniMaxH3`, not the MiniMax source modules"
    );
    assert_eq!(
        r.metadata("reference").unwrap_or_default(),
        "diffusers.AutoencoderKLMiniMaxH3"
    );
    assert_eq!(r.metadata("half").unwrap_or_default(), "encode");
    // The reference must have run the SHIPPED tile geometry, or the keyframe comparison below is
    // about some other tiling than the one production uses.
    assert_eq!(
        r.metadata("tile_sample_min_size").unwrap_or_default(),
        TILE_SAMPLE_MIN_SIZE.to_string(),
        "the reference tiled at a different size than this port does"
    );
    assert_eq!(
        r.metadata("tile_sample_min_overlap").unwrap_or_default(),
        TILE_SAMPLE_MIN_OVERLAP.to_string()
    );

    let dir = root.join("vae");
    let mut w = Weights::from_dir(&dir).unwrap();
    let cfg = MiniMaxH3VaeConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    // f32 to keep the comparison about layout rather than about bf16 rounding. `vae/` is already
    // F32 on disk (703 of 703 tensors, 9.70 GiB), so this is a no-op cast, not a doubling.
    let t0 = Instant::now();
    let vae = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Float32).unwrap();
    let load_ms = t0.elapsed().as_millis();
    assert!(
        vae.can_encode(),
        "the published vae/ always ships the encode half"
    );
    // The real six-level stack, not the four-level toy this test exists to get past.
    assert_eq!(cfg.num_encoder_levels(), 6);
    assert_eq!(cfg.block_out_channels, vec![128, 256, 256, 512, 512, 1024]);
    assert_ne!(
        cfg.spatial_downsample_factors, cfg.temporal_downsample_factors,
        "the shipped factor lists DIFFER; that difference is what makes a level-indexing error \
         expressible here and inert on the fixture, so a config where they agreed would silently \
         demote this test to a second copy of the fixture lane"
    );

    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    for name in ["keyframe", "clip", "multiclip"] {
        let pixels = r.require(&format!("in.{name}.pixels")).unwrap();
        let posterior = vae.encode(pixels).expect("real-weight encode");

        for (label, got) in [("mean", posterior.mean()), ("std", posterior.std())] {
            let want = r.require(&format!("out.{name}.{label}")).unwrap();
            let got = got.as_dtype(Dtype::Float32).unwrap();
            assert_eq!(got.shape(), want.shape(), "{name}.{label}: posterior shape");

            let (peak, mean_rel) = rel(&got, want);
            let cos = cosine(&got, want);
            let (n_got, n_want) = (l2_norm(&got), l2_norm(want));
            println!(
                "REAL-WEIGHT ENCODE PARITY {name}.{label} vs {} {}: rel-max-abs={peak:.3e} \
                 rel-mean={mean_rel:.3e} cosine={cos:.7} ||port||={n_got:.4} \
                 ||reference||={n_want:.4}",
                r.metadata("reference").unwrap_or_default(),
                r.metadata("reference_version").unwrap_or_default(),
            );
            // **The gate.** `cos` and the two norms above are printed and never asserted on: a
            // gate/value half-swap holds cosine at 0.73-0.78 and moves the norm by 4%, and cosine
            // is scale-invariant besides.
            assert!(
                peak < REAL_WEIGHT_ENCODE_TOL,
                "the MLX encode of `{name}.{label}` diverges from the official diffusers VAE by \
                 {peak:.3e} (bound {REAL_WEIGHT_ENCODE_TOL:.0e}). A mis-levelled downsampler, a \
                 symmetric pad where the reference is asymmetric, a global rather than \
                 frame-isolated GroupNorm, or an untiled encode all land here — note cosine \
                 {cos:.7} and norms {n_got:.4} vs {n_want:.4}, which barely move."
            );
            if peak > worst {
                worst = peak;
                worst_at = format!("{name}.{label}");
            }
            assert!(
                std_dev(want) > 1e-3,
                "{name}.{label}: the reference posterior is ~constant and would gate nothing"
            );
        }
    }

    // ---- the shape facts the values alone cannot pin -----------------------------------------
    // A keyframe is ONE latent frame: the `T == 1` short circuit is live on real weights, not only
    // in the fixture. Padding a keyframe up to `clip_length` instead would yield `5 - token_drop`.
    assert_eq!(
        r.require("out.keyframe.mean").unwrap().shape()[2],
        1,
        "a keyframe must encode to exactly one latent frame"
    );
    assert!(r.require("out.clip.mean").unwrap().shape()[2] > 1);

    // The multiclip probe has to actually be ragged, and its repeated frames have to be
    // observable, or the two branches it exists to gate are never entered and it is a third copy
    // of `clip`. Both facts are re-derived from the shipped config here rather than restated.
    let frames = r.require("in.multiclip.pixels").unwrap().shape()[2];
    let clip = cfg.clip_length;
    assert_ne!(
        frames % clip,
        0,
        "the multiclip probe is {frames} frames, a whole number of {clip}-frame clips; it never \
         reaches the frame-repeat pad"
    );
    let clips = frames.div_euclid(clip) + 1;
    assert!(
        clips > 1,
        "the multiclip probe pads to {clips} clip(s); it never reaches the concatenation"
    );
    let per_clip = clip.div_euclid(cfg.patch_size_t) + i32::from(clip % cfg.patch_size_t != 0);
    let pad_reach = (per_clip - cfg.token_drop - 1) * cfg.patch_size_t;
    let tail_real = frames % clip;
    assert!(
        tail_real <= pad_reach,
        "the multiclip probe leaves {tail_real} real frames in its final clip, past the \
         {pad_reach}-pixel reach of the latents that survive token_drop; every repeated frame \
         would land inside the dropped tail and the probe could not tell a last-frame repeat from \
         any other"
    );
    assert_eq!(
        r.require("out.multiclip.mean").unwrap().shape()[2],
        clips * per_clip - cfg.token_drop,
        "a {frames}-frame encode is {clips} clips of {per_clip} latent frames less the {} dropped \
         once at the tail",
        cfg.token_drop
    );

    // ---- the discrimination check ------------------------------------------------------------
    // **Can this comparison fail at all?** The keyframe probe is 320 px, above the shipped 256 px
    // tile, so the reference genuinely tiled it. If an UNTILED encode of the same pixels agreed
    // with the tiled one to within the bound, then agreeing with the reference would say nothing
    // about whether this port tiles — the gate would be measurably inert while reporting green.
    let keyframe = r.require("in.keyframe.pixels").unwrap();
    assert!(
        keyframe.shape()[3] > TILE_SAMPLE_MIN_SIZE,
        "the keyframe probe is {} px and must exceed the {TILE_SAMPLE_MIN_SIZE} px tile",
        keyframe.shape()[3]
    );
    let tiled = vae.encode_clip(keyframe).unwrap();
    let untiled = vae.encode_clip_untiled(keyframe).unwrap();
    assert_eq!(
        tiled.shape(),
        untiled.shape(),
        "tiling must not change the latent SHAPE — a shape-blind gate would be the only one to \
         notice if it did"
    );
    let (untiled_delta, _) = rel(&untiled, &tiled);
    // Measured 9.430e-2 here: 9.4x the 1e-2 bound and 53x the 1.789e-3 floor. But the criterion
    // asserted below is 5x the BOUND, i.e. 5.0e-2, so the margin this control actually runs on is
    // **1.89x** — not the 9.4x the raw measurement suggests. Stated plainly because it is thin: a
    // port that silently stopped tiling collapses this to ~0 and is a red, which is the point, but
    // a device whose reductions halved the measurement would flake the control instead. If the
    // second Mac lands materially below 9.4e-2, re-derive this criterion rather than loosening it.
    assert!(
        untiled_delta > 5.0 * REAL_WEIGHT_ENCODE_TOL,
        "an untiled encode of a {} px canvas agrees with the tiled one to {untiled_delta:.3e}, \
         within 5x the {REAL_WEIGHT_ENCODE_TOL:.0e} bound; the reference comparison above cannot \
         be said to gate the tiling at all",
        keyframe.shape()[3]
    );

    println!(
        "REAL-WEIGHT ENCODE PARITY: worst rel-max-abs {worst:.3e} at {worst_at} over 6 (probe, \
         moment) pairs, bound {REAL_WEIGHT_ENCODE_TOL:.0e}; untiled-vs-tiled keyframe control \
         {untiled_delta:.3e} ({load_ms} ms to map + build {PUBLISHED_VAE_TENSORS} f32 tensors); \
         {frames}-frame ragged probe -> {clips} clips of {per_clip} latent frames less \
         {} dropped, pad_reach {pad_reach}",
        cfg.token_drop
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

// ── sc-17143: the Qwen3-VL-32B context extraction ────────────────────────────────────────────────
//
// The text encoder is 66.7 GB, so these smokes deliberately do NOT hold the whole component
// resident. They are split by what each can prove for the least memory:
//
// * the key-set / trim evidence needs only the shard index (no weight I/O at all);
// * the `<d>` derivation needs only the two tokenizer files;
// * the forward smoke loads the first two shards (9.8 GB) and runs a real forward at the REAL
//   published width — 5120 hidden, head_dim 128, GQA 64/8, FFN 25600 — with a reduced tap. Width,
//   not depth, is where the block math is exercised; depth is covered exactly by the committed
//   fixture and by the index test below.

/// Total tensors in the published `text_encoder/` component.
const PUBLISHED_TE_TENSORS: usize = 1058;
/// Of those: the text tower, the vision tower, and `lm_head`.
const TE_LANGUAGE_MODEL_TENSORS: usize = 706;
const TE_VISION_TENSORS: usize = 351;

/// Per-layer leaf names of one Qwen3 decoder layer, as published.
const LAYER_LEAVES: [&str; 11] = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
];

fn te_weight_map(root: &std::path::Path) -> serde_json::Map<String, serde_json::Value> {
    let index = root
        .join("text_encoder")
        .join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index)
        .unwrap_or_else(|e| panic!("reading {}: {e}", index.display()));
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json.get("weight_map")
        .and_then(serde_json::Value::as_object)
        .expect("index.json has a weight_map")
        .clone()
}

/// Every tensor the layer-50 tap needs must exist in the published checkpoint, and the tensors it
/// does NOT need must be identifiable as a contiguous trimmable tail.
///
/// This is the evidence sc-17139's hosting decision rests on, measured rather than asserted: the
/// tap reads 551 tensors; decoder layers 50-63 (154 tensors), `lm_head` and the final `norm` are
/// never read and account for **15.209 GB** of the 66.715 GB component. Reads only the shard index,
/// so it costs no weight I/O.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); dev box / macos-mlx only"]
fn te_layer_50_tap_is_exhaustive_and_the_tail_is_trimmable() {
    let root = snapshot();
    let map = te_weight_map(&root);
    assert_eq!(
        map.len(),
        PUBLISHED_TE_TENSORS,
        "published text_encoder tensor count changed"
    );

    let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
    assert_eq!(cfg.layers_to_run(), 50);
    assert_eq!(cfg.num_layers, 64);

    // 1. Everything the tap reads is present.
    let mut required = vec![format!("{LM_PREFIX}.embed_tokens.weight")];
    for i in 0..cfg.layers_to_run() {
        for leaf in LAYER_LEAVES {
            required.push(format!("{LM_PREFIX}.layers.{i}.{leaf}"));
        }
    }
    assert_eq!(required.len(), 1 + 50 * 11, "551 tensors feed the tap");
    let missing: Vec<&String> = required.iter().filter(|k| !map.contains_key(*k)).collect();
    assert!(
        missing.is_empty(),
        "missing from the checkpoint: {missing:?}"
    );

    // 2. The trimmable tail exists, is exactly layers 50..64 + lm_head + norm, and is NOT read.
    let mut tail: Vec<String> = Vec::new();
    for i in cfg.layers_to_run()..cfg.num_layers as usize {
        for leaf in LAYER_LEAVES {
            let k = format!("{LM_PREFIX}.layers.{i}.{leaf}");
            assert!(map.contains_key(&k), "{k} should exist in the checkpoint");
            tail.push(k);
        }
    }
    assert_eq!(tail.len(), 14 * 11, "154 tensors in layers 50-63");
    assert!(map.contains_key("lm_head.weight"));
    assert!(map.contains_key(&format!("{LM_PREFIX}.norm.weight")));
    for k in &tail {
        assert!(!required.contains(k), "{k} must not be read by the tap");
    }

    // 3. The tower split, and the vision half that fl2va / Ref2VA need.
    let lm = map.keys().filter(|k| k.starts_with(LM_PREFIX)).count();
    let visual = map.keys().filter(|k| k.starts_with(VISION_PREFIX)).count();
    assert_eq!(lm, TE_LANGUAGE_MODEL_TENSORS);
    assert_eq!(visual, TE_VISION_TENSORS);
    assert_eq!(lm + visual + 1, PUBLISHED_TE_TENSORS, "+1 for lm_head");

    // 4. The trim arithmetic, in bytes, derived ENTIRELY from `cfg`'s dims and checked against a
    //    literal. It does NOT read `metadata.total_size` from the index (66714780128 at snapshot
    //    939557dc) — nothing here cross-checks the config against the published byte count.
    let hidden = cfg.hidden_size as u64;
    let inter = cfg.intermediate_size as u64;
    let vocab = cfg.vocab_size as u64;
    let per_layer = 2
        * (hidden // input_layernorm + post_attention_layernorm
        + hidden
        + (cfg.num_heads * cfg.head_dim) as u64 * hidden        // q_proj
        + 2 * (cfg.num_kv_heads * cfg.head_dim) as u64 * hidden // k_proj + v_proj
        + hidden * (cfg.num_heads * cfg.head_dim) as u64        // o_proj
        + 3 * inter * hidden                                    // gate + up + down
        + 2 * cfg.head_dim as u64); // q_norm + k_norm
    let trimmable = 14 * per_layer + 2 * vocab * hidden + 2 * hidden; // layers + lm_head + norm
    let gb = trimmable as f64 / 1e9;
    assert!(
        (gb - 15.209).abs() < 0.01,
        "trimmable tail computed as {gb:.3} GB, expected 15.209 GB"
    );

    println!(
        "REAL-WEIGHT TE INDEX SMOKE: {} published tensors ({lm} language_model / {visual} visual / \
         1 lm_head); the layer-{} tap reads {} of them; layers 50-63 + lm_head + norm = {} tensors \
         = {gb:.3} GB are never read and can be trimmed on upload",
        map.len(),
        cfg.select_hidden,
        required.len(),
        tail.len() + 2,
    );
}

/// `<d>` resolves to 151669 through the REAL shipped files — the id `transformers` assigns — and a
/// bare `tokenizer.json` would silently BPE-split it instead.
///
/// This is the acceptance item that cannot be checked from the tensor index at all: the token is
/// declared only in `tokenizer_config.json`, and its id comes from that array's ORDER.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); dev box / macos-mlx only"]
fn real_tokenizer_resolves_the_minimax_special_tokens() {
    let root = snapshot();
    let tok = MiniMaxH3Tokenizer::from_snapshot(&root).expect("load the shipped tokenizer");

    let specials = tok.specials();
    assert_eq!(specials.dialogue_open(), Some(151669), "<d>");
    assert_eq!(specials.dialogue_close(), Some(151670), "</d>");
    for (i, name) in MINIMAX_ADDED_SPECIALS.iter().enumerate() {
        assert_eq!(specials.get(name), Some(151669 + i as i32), "{name}");
    }
    // The upstream Qwen specials keep their vocabulary ids.
    assert_eq!(specials.get("<|im_start|>"), Some(151644));
    assert_eq!(specials.get("<|image_pad|>"), Some(151655));

    // `<d>` must encode to ONE id, not the `['<d', '>']` split a bare tokenizer.json produces.
    let ids = tok.encode_raw("<d>").expect("encode <d>");
    assert_eq!(
        ids,
        vec![151669],
        "<d> must be a single registered special token, not a BPE split"
    );

    // **sc-18741, against the REAL tokenizer.** The presentation is the prompt verbatim: the
    // port's ids must equal `tokenizer(prompt, add_special_tokens=False)` exactly, and must not be
    // the chat-template render — with or without a compensating 3-token slice.
    const { assert!(!APPLIES_CHAT_TEMPLATE) };
    const PROMPT: &str = "a red fox leaps over a mossy log at dawn";
    const CUE: [i32; 5] = [151645, 198, 151644, 77091, 198];

    let got = tok.ids(PROMPT).expect("presentation ids");
    assert_eq!(got, tok.encode_raw(PROMPT).unwrap(), "the prompt, verbatim");
    assert!(
        got.iter().all(|id| *id < 151643),
        "the presentation must contain no special tokens at all, got {got:?}"
    );

    // Reconstruct exactly what sc-17143 fed the DiT and prove the port no longer produces it.
    let templated = tok
        .encode_raw(&format!(
            "<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
        ))
        .unwrap();
    assert_eq!(&templated[..3], &[151644, 872, 198], "the template prefix");
    let shipped: Vec<i32> = templated[3..].to_vec();
    assert_ne!(got, shipped, "the port still emits sc-17143's presentation");
    assert_eq!(
        shipped.len(),
        got.len() + 5,
        "sc-17143's slice removed the prefix but never the generation cue"
    );
    assert_eq!(&shipped[shipped.len() - 5..], &CUE);

    // The head-corruption mode: the 3-token slice lands on the prefix boundary only when the
    // tokenizer does not merge the template's trailing newline into the prompt. It does merge for
    // a whitespace-leading prompt, and then a real prompt token is destroyed as well.
    const WS_PROMPT: &str = "\nleading newline";
    let ws_ref = tok.ids(WS_PROMPT).unwrap();
    let ws_shipped: Vec<i32> = tok
        .encode_raw(&format!(
            "<|im_start|>user\n{WS_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
        ))
        .unwrap()[3..]
        .to_vec();
    assert_ne!(
        &ws_shipped[..ws_ref.len().min(ws_shipped.len())],
        &ws_ref[..],
        "expected sc-17143 to also lose a leading token for a whitespace-leading prompt"
    );

    println!(
        "REAL-WEIGHT TOKENIZER SMOKE: <d>={:?} </d>={:?}; the seven MiniMax specials resolve to \
         151669-151675 from tokenizer_config.json alone (they appear in no vocabulary file). \
         Presentation for {PROMPT:?}: {} ids (reference) vs {} ids under sc-17143 — the extra 5 \
         are the generation cue {CUE:?}. Whitespace-leading prompt {WS_PROMPT:?}: reference \
         {ws_ref:?} vs sc-17143 {ws_shipped:?}, which also loses a real prompt token.",
        specials.dialogue_open().unwrap(),
        specials.dialogue_close().unwrap(),
        got.len(),
        shipped.len(),
    );
}

/// **The gate sc-17143 was missing** (sc-18741): run the FULL 50-layer tap on the real 62 GB
/// `text_encoder/` over the same prompt the official conditioner encoded, and compare numerically.
///
/// This is the only assertion that can catch a presentation defect. The committed fixture pins the
/// tensor math at tiny dims and the tokenizer smoke pins the ids, but neither runs the real stack
/// end to end, and sc-17143's context was a plausible, finite, non-constant tensor of the wrong
/// length carrying the wrong rows — every self-generated check it had passed.
///
/// Generate the reference with `tools/dump_minimax_h3_te_real.py` (~220 KB, deliberately not
/// committed) and point `MINIMAX_H3_TE_REFERENCE` at it. **Run this in its own process**: the
/// conditioner is ~62 GB on the Python side and does not release inside a process on MPS, and this
/// side maps shards 1-12 whole — **58.568 GB**, of which the tap's own parameters (`embed_tokens`
/// plus layers 0-49) are 50.316 GB. Both figures are byte sums over the published shard headers at
/// snapshot `939557dc`. Asserts rather than skips, like everything else here.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot + a reference context (MINIMAX_H3_TE_REFERENCE); maps 58.6 GB"]
fn real_weight_te_context_matches_the_official_conditioner() {
    let root = snapshot();
    let reference_path = std::env::var("MINIMAX_H3_TE_REFERENCE").unwrap_or_default();
    assert!(
        !reference_path.is_empty(),
        "MINIMAX_H3_TE_REFERENCE must point at the output of tools/dump_minimax_h3_te_real.py. \
         This test is #[ignore]d and asserts rather than skips so a missing reference cannot be \
         mistaken for a pass."
    );
    let r = Weights::from_file(&reference_path)
        .unwrap_or_else(|e| panic!("load {reference_path}: {e}"));
    assert_eq!(
        r.metadata("reference").unwrap_or_default(),
        "diffusers.MiniMaxH3TextEncoderStep",
        "the reference must come from the official conditioner"
    );
    assert_eq!(
        r.metadata("applies_chat_template").unwrap_or_default(),
        "false"
    );
    let prompt = r.metadata("prompt").unwrap_or_default().to_string();

    // The port's own tokenization of that prompt must equal the reference's ids BEFORE any
    // forward runs — a context that matched numerically but on different ids would be luck.
    let tok = MiniMaxH3Tokenizer::from_snapshot(&root).expect("tokenizer");
    let want_ids = r.require("in.input_ids").unwrap();
    let got_ids = tok.ids(&prompt).expect("presentation ids");
    assert_eq!(
        got_ids,
        want_ids.as_slice::<i32>().to_vec(),
        "the port's presentation differs from the official conditioner's for {prompt:?}"
    );

    let dir = root.join("text_encoder");
    let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
    assert_eq!(cfg.select_hidden, 50);

    // Shards 1-12, 58.568 GB, rather than `Weights::from_dir`'s all 14 (66.715 GB). The range
    // mirrors production's `TE_SHARDS` (`1..=12`, `src/text_encoder/mod.rs`) rather than deriving a
    // minimum: layers 0-49 and `embed_tokens` in fact fit in shards 1-11 (53.692 GB), and shard 12
    // holds only layers 53-58. `from_file` panics on a missing shard, so this is what goes red if
    // the manifest row that feeds it ever stops fetching one of the twelve. Shards 13-14 hold only
    // the never-executed tail, plus — in 14 — the vision tower `t2va` does not use.
    let t0 = Instant::now();
    let mut w = Weights::empty();
    for i in 1..=12 {
        let shard = format!("model-{i:05}-of-00014.safetensors");
        let part =
            Weights::from_file(dir.join(&shard)).unwrap_or_else(|e| panic!("load {shard}: {e}"));
        let keys: Vec<String> = part.keys().map(str::to_owned).collect();
        for k in keys {
            let t = part.require(&k).unwrap().clone();
            w.insert(k, t);
        }
    }
    let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg)
        .expect("build the encoder at the full published depth");
    assert_eq!(te.num_loaded_layers(), 50, "the full layer-50 tap must run");
    let load_ms = t0.elapsed().as_millis();

    let (ids, mask) = tok.encode_prompt(&prompt).expect("encode");
    let seq = ids.shape()[1];
    let t1 = Instant::now();
    let got = te
        .forward(&ids, &mask)
        .expect("real-weight forward")
        .as_dtype(Dtype::Float32)
        .unwrap();
    let want = r.require("out.context").unwrap();

    assert_eq!(
        got.shape(),
        &[1, seq, cfg.hidden_size],
        "one context row per presentation token"
    );
    assert_eq!(got.shape(), want.shape(), "context shape vs the reference");

    let diff = mlx_rs::ops::subtract(&got, want).unwrap().abs().unwrap();
    let peak: f32 = want.abs().unwrap().max(None).unwrap().item();
    let rel_max: f32 = diff.max(None).unwrap().item::<f32>() / peak;
    // Reported, never asserted on (sc-19505). A `fwd_ms > 0` gate here only ever said "the clock
    // ticked": it cannot fail except on a broken timer, and it certainly cannot say the forward was
    // evaluated — `rel_max` above already forced the graph through `.item()`, and the `rel_max`
    // gate below is what actually holds this test up. This is the same call
    // `real_weight_dit_block_runs_one_forward` already made for its own `fwd_ms`.
    let fwd_ms = t1.elapsed().as_millis();

    println!(
        "REAL-WEIGHT TE PARITY vs {} ({}): prompt {prompt:?} -> {seq} ids -> context {:?}; \
         rel-max-abs={rel_max:.3e} ({load_ms} ms to map 12 shards + build 50 layers, {fwd_ms} ms \
         to forward)",
        r.metadata("reference").unwrap_or_default(),
        r.metadata("reference_version").unwrap_or_default(),
        got.shape(),
    );

    // The reference is bf16 (the checkpoint's own dtype) so the floor is bf16 round-off through a
    // 50-layer stack, not f32 round-off; 5e-2 is the house value for a bf16 real-weight comparison.
    assert!(
        rel_max < 5e-2,
        "real-weight context differs from the official conditioner by {rel_max:.3e}"
    );
}

/// Load the first two shards (9.8 GB of the 66.7 GB component) and run a REAL forward at the
/// published width — 5120 hidden, `head_dim` 128, GQA 64 query / 8 kv, FFN 25600 — over a real
/// tokenized prompt, with a reduced 4-layer tap.
///
/// Depth is already pinned exactly by the committed fixture and by the index test above; what only
/// real weights can prove is that the published key layout, bf16 dtypes and non-square projections
/// load and execute. The timer spans the `.item()` that forces evaluation — MLX is lazy, and
/// sc-17140's first attempt reported 0.0 s for a full decode because the timer stopped before
/// anything had been computed.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; ~10 GB resident"]
fn real_weight_te_forward_runs_at_the_published_geometry() {
    let root = snapshot();

    // Only the shards the reduced tap needs — never the whole 66.7 GB component.
    let dir = root.join("text_encoder");
    let t0 = Instant::now();
    let mut w = Weights::empty();
    for shard in [
        "model-00001-of-00014.safetensors",
        "model-00002-of-00014.safetensors",
    ] {
        let part =
            Weights::from_file(dir.join(shard)).unwrap_or_else(|e| panic!("load {shard}: {e}"));
        let keys: Vec<String> = part.keys().map(str::to_owned).collect();
        for k in keys {
            let t = part.require(&k).unwrap().clone();
            w.insert(k, t);
        }
    }
    let loaded = w.len();
    let load_ms = t0.elapsed().as_millis();
    assert!(
        loaded > 40,
        "only {loaded} tensors loaded from the first two shards"
    );

    // The published geometry, with the tap reduced to what these shards carry.
    let mut cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
    cfg.select_hidden = 4;
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.head_dim, 128);
    assert_ne!(cfg.head_dim, cfg.hidden_size / cfg.num_heads);

    let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg)
        .expect("build the encoder from the published weights");
    assert_eq!(te.num_loaded_layers(), 4);

    // A real prompt through the real tokenizer.
    let tok = MiniMaxH3Tokenizer::from_snapshot(&root).expect("tokenizer");
    let (ids, mask) = tok
        .encode_prompt("a cinematic wide shot of a starship bridge, amber console glow")
        .expect("encode");
    let seq = ids.shape()[1];
    assert!(seq > 4, "the probe prompt should be several tokens long");

    let t1 = Instant::now();
    let ctx = te.forward(&ids, &mask).expect("real-weight forward");
    // Force evaluation INSIDE the timer — MLX is lazy, so a timer that stops before an `.item()`
    // measures graph construction, not compute.
    let spread = std_dev(&ctx);
    let peak: f32 = ctx.abs().unwrap().max(None).unwrap().item();
    let fwd_ms = t1.elapsed().as_millis();

    assert_eq!(
        ctx.shape(),
        &[1, seq, cfg.hidden_size],
        "context is [1, seq, 5120] — one row per presentation token, none sliced (sc-18741)"
    );
    assert!(
        peak.is_finite() && peak > 0.0,
        "context peak {peak} is not finite/positive"
    );
    assert!(spread > 1e-3, "context is ~constant (std {spread:.3e})");
    // `fwd_ms` is reported, never asserted on (sc-19505) — see the note at the parity smoke above.
    // What it was reaching for, "the forward really was evaluated", is already held by `spread` and
    // `peak`: both come from `.item()`, which forces the graph, and both are numeric properties of
    // the computed context rather than of the clock.

    println!(
        "REAL-WEIGHT TE SMOKE: loaded {loaded} tensors in {load_ms} ms; 4 layers at the published \
         5120/128/64-8/25600 geometry; prompt -> {seq} ids -> context {:?} in {fwd_ms} ms \
         (std {spread:.4}, peak {peak:.4})",
        ctx.shape(),
    );
}

// ---------------------------------------------------------------------------------------------
// sc-17144 — the DiT block stack
// ---------------------------------------------------------------------------------------------

/// Total tensors in the published `transformer/` component: 50 blocks × 12, the refiner's 21, and
/// 17 input/output/timestep tensors this story does not port.
const PUBLISHED_DIT_TENSORS: usize = 638;

/// The shard `transformer_blocks.0` lives in — 4.5 GiB of the 62 GB component. A single-block
/// smoke has no reason to materialize the other thirteen.
const DIT_FIRST_SHARD: &str = "diffusion_pytorch_model-00001-of-00014.safetensors";

/// The declared block-stack key set must be EXACTLY the published checkpoint's, minus the
/// input/output layers sc-17147 owns. Reads only the shard index, so it costs no weight I/O.
///
/// This is the exhaustive-mapping proof against the real 33 B model rather than against the tiny
/// fixture. A tensor the loader never reads would compute something plausible and wrong.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); dev box / macos-mlx only"]
fn declared_dit_tensor_names_match_the_published_checkpoint() {
    let root = snapshot();
    let dir = root.join("transformer");
    let index = dir.join("diffusion_pytorch_model.safetensors.index.json");
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
        PUBLISHED_DIT_TENSORS,
        "published transformer/ tensor count changed"
    );

    let cfg = MiniMaxH3DitConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        cfg,
        MiniMaxH3DitConfig::default(),
        "the shipped transformer config drifted from this crate's defaults"
    );

    let mut declared: std::collections::BTreeSet<String> = (0..cfg.num_layers)
        .flat_map(|i| DitBlock::names(&format!("transformer_blocks.{i}")))
        .collect();
    declared.extend(TokenRefiner::names("token_refiner", &cfg));

    let missing: Vec<&String> = declared.difference(&published).collect();
    assert!(
        missing.is_empty(),
        "the block stack requires tensors the checkpoint does not have: {missing:?}"
    );

    // Everything else must be an input/output/timestep tensor — i.e. sc-17147's, not silently
    // dropped block-stack weights.
    let unconsumed: Vec<&String> = published
        .difference(&declared)
        .filter(|k| k.starts_with("transformer_blocks.") || k.starts_with("token_refiner."))
        .collect();
    assert!(
        unconsumed.is_empty(),
        "block-stack tensors the loader never reads: {unconsumed:?}"
    );

    assert_eq!(declared.len(), 50 * 12 + 21);
    let outer: Vec<&String> = published.difference(&declared).collect();
    assert_eq!(
        outer.len(),
        17,
        "the 17 input/output/timestep tensors belong to sc-17147: {outer:?}"
    );
    println!(
        "declared {} block-stack tensors of {} published; {} input/output tensors deliberately \
         unported (sc-17147)",
        declared.len(),
        published.len(),
        outer.len()
    );
}

/// Load the real `transformer_blocks.0` and run one forward at the published 5376/56×128 geometry.
///
/// **A skipped run must not look like a passing one.** `snapshot()` asserts rather than returning
/// early, and every assertion below is on evidence only a real run produces: the published shard's
/// tensor count, the block's real `[7168, 5376]` / `[28672, 5376]` / `[96768, 2688]` shapes, qk-norm
/// weights that are near 1 but NOT the all-ones a fabricated tensor would be, and a finite,
/// non-constant output of the right shape.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT) + Metal; ~4.5 GB resident"]
fn real_weight_dit_block_runs_one_forward() {
    let root = snapshot();
    let dir = root.join("transformer");

    let cfg = MiniMaxH3DitConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(cfg.hidden_size, 5376);
    assert_eq!(
        cfg.inner_dim(),
        7168,
        "attention is wider than the residual stream"
    );
    assert_eq!(cfg.rotary_dim(), 96, "96 of 128 head channels rotate");

    let started = Instant::now();
    // Only the shard holding block 0 — the full component is 62 GB and one block needs 4.5.
    let mut w = Weights::from_file(dir.join(DIT_FIRST_SHARD)).unwrap();
    let shard_tensors = w.len();
    assert_eq!(
        shard_tensors, 64,
        "expected the published first shard's tensor set"
    );

    // bf16 is the checkpoint's own dtype for the block stack.
    let block =
        DitBlock::from_weights(&mut w, "transformer_blocks.0", &cfg, Dtype::Bfloat16).unwrap();
    let load_ms = started.elapsed().as_millis();

    // Evidence the tensors are the real trained ones rather than anything synthesized: the qk-norm
    // weights are a learned scatter about 1, not the all-ones an untrained `nn.RMSNorm` holds.
    let nq = Weights::from_file(dir.join(DIT_FIRST_SHARD))
        .unwrap()
        .require("transformer_blocks.0.attn.norm_q.weight")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    assert_eq!(nq.shape(), &[cfg.attention_head_dim], "qk-norm is per head");
    let nq_spread = std_dev(&nq);
    assert!(
        nq_spread > 1e-4,
        "norm_q.weight is constant (std {nq_spread:.3e}) — these are not trained weights"
    );

    // A packed sequence with all three modalities at two timesteps, so the AdaLN row addressing is
    // exercised: 8 text rows, 8 audio rows, 48 video rows.
    let seq = 64i32;
    let ids: Vec<f32> = (0..seq)
        .flat_map(|i| [i as f32, (i % 7) as f32, (i % 5) as f32])
        .collect();
    let position_ids = Array::from_slice(&ids, &[seq, 3]);
    let rope = MmRope::new(cfg.rope_freq_dim, cfg.rope_theta).unwrap();
    let tables = rope.tables(&position_ids).unwrap();
    assert_eq!(tables.cos.shape(), &[seq, 96]);

    let tags: Vec<i32> = (0..seq)
        .map(|i| {
            if i < 8 {
                1
            } else if i < 16 {
                2
            } else {
                0
            }
        })
        .collect();
    let steps: Vec<i32> = (0..seq).map(|i| i32::from(i >= 16)).collect();
    let adaln: Vec<i32> = steps
        .iter()
        .zip(&tags)
        .map(|(s, t)| s * mlx_gen_minimax_h3::MODALITY_NUM + t)
        .collect();
    let adaln_indices = Array::from_slice(&adaln, &[seq]);

    let hidden: Vec<f32> = (0..seq * cfg.hidden_size)
        .map(|i| (i as f32 * 0.013).sin() * 0.5)
        .collect();
    let hidden = Array::from_slice(&hidden, &[1, seq, cfg.hidden_size])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let temb: Vec<f32> = (0..2 * cfg.time_embed_dim)
        .map(|i| (i as f32 * 0.021).cos() * 0.3)
        .collect();
    let temb = Array::from_slice(&temb, &[2, cfg.time_embed_dim])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    let t1 = Instant::now();
    let out = block
        .forward_with_temb(&hidden, &temb, &adaln_indices, &rope, &tables)
        .expect("real-weight block forward");
    // Force evaluation INSIDE the timer — MLX is lazy, so a timer that stops before an `.item()`
    // measures graph construction, not compute. An earlier story's timer read 0.0 s for a full
    // 36-layer decode for exactly this reason.
    let spread = std_dev(&out);
    let peak: f32 = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item();
    let fwd_ms = t1.elapsed().as_millis();

    assert_eq!(out.shape(), &[1, seq, cfg.hidden_size]);
    assert!(
        peak.is_finite() && peak > 0.0,
        "block output peak {peak} is not finite/positive"
    );
    assert!(
        spread > 1e-3,
        "block output is ~constant (std {spread:.3e})"
    );
    // The block is not an identity on real weights.
    let moved = std_dev(
        &mlx_rs::ops::subtract(
            out.as_dtype(Dtype::Float32).unwrap(),
            hidden.as_dtype(Dtype::Float32).unwrap(),
        )
        .unwrap(),
    );
    assert!(
        moved > 1e-3,
        "the real block was an identity (residual std {moved:.3e})"
    );
    // The evaluation is forced by the `std_dev`/`.item()` calls above, which sit INSIDE the timer;
    // `fwd_ms` is therefore a real compute measurement rather than graph-construction time. It is
    // reported, not asserted on — a wall-clock lower bound proves nothing on its own, and the
    // evidence that the block really ran is the shape, finiteness and non-constant residual.
    println!(
        "REAL-WEIGHT DiT BLOCK SMOKE: loaded {shard_tensors} tensors from {DIT_FIRST_SHARD} and \
         built transformer_blocks.0 in {load_ms} ms at the published 5376 / 56x128 / ffn 14336 / \
         adaln 96768x2688 geometry; norm_q std {nq_spread:.4}; {seq}-row packed sequence \
         (3 modalities, 2 timesteps) -> {:?} in {fwd_ms} ms (std {spread:.4}, peak {peak:.4}, \
         residual std {moved:.4})",
        out.shape(),
    );
}

// ---------------------------------------------------------------------------------------------
// Spatial tiling on real weights (sc-18786)
// ---------------------------------------------------------------------------------------------

/// **The gate for sc-18786.** `AutoencoderKLMiniMaxH3` ships with `use_tiling = True` (256 px
/// tiles, 64 px minimum overlap) for both halves, and upstream states the consequence outright:
/// MiniMax-H3 was released with tiling enabled and *the released frames are the blended-tile ones,
/// so disabling tiling changes the output*. sc-17140 decoded the whole canvas in one pass.
///
/// # Why the sc-18740 reference could not catch this
///
/// `tools/dump_minimax_h3_video_vae_real.py` decodes a 4x4 latent to 64x64 px and calls
/// `disable_tiling()` first, because at that canvas the two paths are bit-identical anyway. It is
/// inert at its own geometry by construction, so its 4.338e-3 residual is genuine but blind here.
///
/// This test uses a second reference (`tools/dump_minimax_h3_video_vae_tiling.py`) at a **512x320**
/// canvas — 3 tile rows x 2 tile columns, non-square so a transposed plan is not accidentally
/// correct, with a genuine interior row that a 2x2 grid never exercises.
///
/// # The separation this gates
///
/// The generator measured the reference's own tiled-vs-untiled delta at this canvas and recorded it
/// in the file: **rel-max-abs 6.470e-1**. That is the size of the defect — a 65 % error, not a
/// rounding difference — and it is asserted below rather than trusted, so a reference regenerated
/// at an inert canvas fails loudly instead of passing vacuously.
///
/// # Reference validity
///
/// The reference is torch, so the MLX `i32` write cap (`gen-core::tiling::MAX_WRITABLE_ELEMS`) does
/// not bind it. It does not bind *this* side's untiled comparison decode either: MiniMax-H3's
/// decoder is a ViT that runs everything at latent resolution and unpatchifies once at the end, so
/// its widest full-resolution write is 3 channels per output voxel. `heads · tokens²` looks like
/// the binding term and is not one — that is the score matrix, and MLX's fused
/// `scaled_dot_product_attention` streams it rather than writing it, the same distinction
/// `mlx_gen_minimax_h3::cost` draws for the DiT.
#[test]
#[ignore = "needs a real snapshot + a tiling reference (MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE)"]
fn real_weight_tiled_decode_matches_the_official_diffusers_vae() {
    let root = snapshot();
    let path = std::env::var("MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE").unwrap_or_default();
    assert!(
        !path.is_empty(),
        "MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE must point at the output of \
         tools/dump_minimax_h3_video_vae_tiling.py. This test is #[ignore]d and asserts rather \
         than skips so a missing reference cannot be mistaken for a pass."
    );
    let r = Weights::from_file(&path).unwrap_or_else(|e| panic!("load {path}: {e}"));
    assert_eq!(
        r.metadata("reference").unwrap_or_default(),
        "diffusers.AutoencoderKLMiniMaxH3",
        "the reference must come from the official converted-checkpoint class"
    );

    // The reference records what the shipped model actually does. Pin it here rather than trusting
    // this crate's own constants, which is the only way the port's defaults are gated against the
    // model instead of against themselves.
    let shipped = r.metadata("shipped_tiling").unwrap_or_default();
    for expected in [
        "\"use_tiling\": true",
        "\"tile_sample_min_height\": 256",
        "\"tile_sample_min_width\": 256",
        "\"tile_sample_min_overlap_height\": 64",
        "\"tile_sample_min_overlap_width\": 64",
    ] {
        assert!(
            shipped.contains(expected),
            "the shipped VAE no longer reports {expected}; recorded defaults were {shipped}"
        );
    }
    let defaults = SpatialTiling::default();
    assert!(defaults.enabled, "this port must tile by default too");
    assert_eq!((defaults.tile_height, defaults.tile_width), (256, 256));
    assert_eq!((defaults.overlap_height, defaults.overlap_width), (64, 64));

    let dir = root.join("vae");
    let mut w = Weights::from_dir(&dir).unwrap();
    let cfg = MiniMaxH3VaeConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let vae = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Float32).unwrap();

    let latent = r.require("in.latent").unwrap();
    let want = r.require("out.video.tiled").unwrap();
    let want_untiled = r.require("out.video.untiled").unwrap();

    // The canvas must genuinely tile in BOTH axes, or this test proves nothing at all.
    let ls = latent.shape();
    let (lat_h, lat_w) = (ls[3], ls[4]);
    let ratio = cfg.patch_size;
    let rows = TilePlan::split(lat_h * ratio, 256, 64, ratio).unwrap();
    let cols = TilePlan::split(lat_w * ratio, 256, 64, ratio).unwrap();
    assert!(
        rows.len() > 1 && cols.len() > 1,
        "the reference canvas is {}x{} px = {} rows x {} cols; it does not cross a tile boundary \
         on both axes and cannot gate spatial tiling",
        lat_h * ratio,
        lat_w * ratio,
        rows.len(),
        cols.len()
    );
    // …and against the reference's own recorded plan, so the derivation is gated rather than the
    // grid merely being non-trivial.
    let plan = r.metadata("tile_plan").unwrap_or_default();
    assert!(
        plan.contains(&format!("{:?}", rows.starts))
            && plan.contains(&format!("{:?}", cols.starts)),
        "the port's tile starts (rows {:?}, cols {:?}) are not the reference's: {plan}",
        rows.starts,
        cols.starts
    );

    let got = vae
        .decode(latent)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    assert_eq!(got.shape(), want.shape(), "decoded shape");

    let rel_max = |a: &Array, b: &Array| -> f32 {
        let d: f32 = mlx_rs::ops::subtract(a, b)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item();
        d / b.abs().unwrap().max(None).unwrap().item::<f32>()
    };

    // **The pre-fix behaviour, measured on this same port.** sc-17140 decoded the whole canvas in
    // one pass, which is exactly what `disable_tiling()` selects. Running it here turns "the new
    // code is necessary" from a claim into an assertion: if tiling were a no-op, or if the tiled
    // and untiled paths converged, `before` would pass the same gate `after` does and this test
    // would be proving nothing.
    let before = {
        let mut off = vae.clone();
        off.disable_tiling();
        rel_max(
            &off.decode(latent)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap(),
            want,
        )
    };

    let against_tiled = rel_max(&got, want);
    let against_untiled = rel_max(&got, want_untiled);
    let reference_separation = rel_max(want_untiled, want);
    println!(
        "REAL-WEIGHT TILED PARITY ({} rows x {} cols at {}x{} px): BEFORE (untiled, sc-17140) \
         = {before:.3e} -> AFTER (tiled) = {against_tiled:.3e}, both vs the TILED reference. \
         Our tiled decode vs the UNTILED reference = {against_untiled:.3e}; the reference's own \
         tiled/untiled separation = {reference_separation:.3e}",
        rows.len(),
        cols.len(),
        lat_h * ratio,
        lat_w * ratio,
    );

    // (0) The single-pass decode this story replaced **fails** this gate. Without this the whole
    // test could pass on a canvas where tiling happened not to matter.
    assert!(
        before > 1e-2,
        "an UNTILED decode is within {before:.3e} of the tiled reference, so this canvas cannot \
         distinguish the pre-sc-18786 behaviour from the fix"
    );
    assert!(
        against_tiled < before / 10.0,
        "tiling only improved the residual from {before:.3e} to {against_tiled:.3e}; that is not \
         the reference's geometry"
    );

    // (1) The canvas actually separates the two implementations. Without this the rest is vacuous.
    assert!(
        reference_separation > 1e-2,
        "the reference's own tiled and untiled decodes differ by only {reference_separation:.3e} \
         at this canvas; regenerate the reference at a canvas that crosses a tile boundary"
    );
    // (2) We match the TILED reference — the released frames.
    assert!(
        against_tiled < 1e-2,
        "the tiled decode differs from the official implementation by {against_tiled:.3e}; \
         gate on rel-max-abs, never on norm or cosine (sc-18740)"
    );
    // (3) …and are decisively closer to it than to the untiled one, so a regression back to a
    // single-pass decode fails here rather than merely loosening a tolerance.
    assert!(
        against_untiled > 1e-2,
        "the decode is within {against_untiled:.3e} of the UNTILED reference too, so this test \
         cannot tell the two paths apart"
    );
    assert!(
        std_dev(want) > 1e-3,
        "the reference decode is ~constant; it would gate nothing"
    );
}

/// The mirror assertion, on real weights: **below one tile the tiled and untiled paths agree
/// exactly**, which is what keeps sc-17140's and sc-18740's sub-tile fixtures valid across this
/// change. The reference asserts the same thing on its side (`subtile_delta_max_abs == 0`).
#[test]
#[ignore = "needs a real snapshot + a tiling reference (MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE)"]
fn real_weight_tiling_is_inert_below_one_tile() {
    let root = snapshot();
    let path = std::env::var("MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE").unwrap_or_default();
    assert!(
        !path.is_empty(),
        "MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE must point at the output of \
         tools/dump_minimax_h3_video_vae_tiling.py"
    );
    let r = Weights::from_file(&path).unwrap_or_else(|e| panic!("load {path}: {e}"));
    assert_eq!(
        r.metadata("subtile_delta_max_abs").unwrap_or_default(),
        "0.000000e+00",
        "the reference itself no longer finds tiling inert below one tile"
    );

    let dir = root.join("vae");
    let mut w = Weights::from_dir(&dir).unwrap();
    let cfg = MiniMaxH3VaeConfig::from_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let vae = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Float32).unwrap();

    let latent = r.require("in.latent.subtile").unwrap();
    let want = r.require("out.video.subtile.tiled").unwrap();
    let s = latent.shape();
    assert!(
        s[3] * cfg.patch_size <= 256 && s[4] * cfg.patch_size <= 256,
        "the sub-tile control is not below one tile"
    );

    let tiled = vae
        .decode(latent)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let untiled = {
        let mut off = vae.clone();
        off.disable_tiling();
        off.decode(latent)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
    };
    let delta: f32 = mlx_rs::ops::subtract(&tiled, &untiled)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item();
    assert_eq!(
        delta, 0.0,
        "below one tile the two paths must be BIT-identical, got max|delta| {delta:.3e}"
    );

    let d: f32 = mlx_rs::ops::subtract(&tiled, want)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item();
    let rel = d / want.abs().unwrap().max(None).unwrap().item::<f32>();
    println!("REAL-WEIGHT SUB-TILE PARITY: rel-max-abs={rel:.3e} (tiling inert, delta 0)");
    assert!(rel < 1e-2, "the sub-tile decode diverges by {rel:.3e}");
}
