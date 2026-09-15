//! sc-17148: video-VAE **encode** parity — the 3-D causal CNN half, against the official
//! diffusers `AutoencoderKLMiniMaxH3`.
//!
//! Fixture `tests/fixtures/video_vae_encode.safetensors` ←
//! `tools/dump_minimax_h3_video_vae_encode.py`, which documents at length why the encode half has
//! its own file rather than sharing the decode goldens': the geometries cannot match (the encoder
//! must not crop), and the decode fixture's bytes are shared verbatim with
//! `candle-gen-minimax-h3`, whose cross-backend gate digests them.
//!
//! `fl2va` conditions a keyframe through **both** the text encoder's vision tower and this VAE, so
//! sc-17140's decode-only port is not sufficient for it. The four conventions this file exists to
//! gate are set out in [`mlx_gen_minimax_h3::vae_encoder`]'s module docs; each has a parity golden
//! here and, where the convention could be silently inert, a probe that makes the wrong
//! implementation measurably different rather than merely unproven.
//!
//! # What "unproven" would look like here
//!
//! Three of the four conventions are invisible under the obvious test:
//!
//! | convention | wrong version is indistinguishable when… |
//! |---|---|
//! | frame-isolated GroupNorm | `T == 1` — the isolated and global reductions are the same tensor |
//! | tiled encode | the canvas is smaller than one tile — the plan degenerates to one span |
//! | single-frame short circuit | the input already has ≥ 17 frames |
//!
//! So every golden below is dumped at `T = 5` or `T = 17` with a canvas that really does span two
//! tiles, and the degenerate cases are asserted to be degenerate rather than assumed to be.

use crate::common;

use common::{
    assert_parity, encode_fixture_config, encode_fixture_tiles, rel, std_dev, ENCODE_FIXTURE,
};

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::{
    DiagonalGaussian, FrameIsolatedGroupNorm, MiniMaxH3VideoVae, VideoEncoder3d,
    ENCODER_TILING_IS_ON_BY_DEFAULT, TILE_SAMPLE_MIN_OVERLAP, TILE_SAMPLE_MIN_SIZE,
};

/// Peak-relative tolerance, the mlx-gen house value. See `video_vae_parity.rs` for why 1e-2.
const TOL: f32 = 1e-2;

/// A mutation must move the output by at least this much to count as gated.
const MUTATION_FLOOR: f32 = 1e-2;

fn fixture() -> Weights {
    Weights::from_file(ENCODE_FIXTURE).unwrap()
}

fn model_weights() -> Weights {
    let mut w = fixture();
    for prefix in ["src.", "in.", "out.", "const."] {
        w.remove_prefix(prefix);
    }
    w
}

fn vae(token_drop: i32) -> MiniMaxH3VideoVae {
    let mut w = model_weights();
    MiniMaxH3VideoVae::from_weights(&mut w, &encode_fixture_config(token_drop), Dtype::Float32)
        .unwrap()
}

fn get(f: &Weights, key: &str) -> Array {
    f.require(key)
        .unwrap_or_else(|_| panic!("fixture is missing `{key}`; re-run the dump script"))
        .clone()
}

// ---------------------------------------------------------------------------------------------
// Parity
// ---------------------------------------------------------------------------------------------

/// The bare encoder stack: `conv_in → down_blocks → norm_out → silu → conv_out`, at `T = 5` so
/// the frame-isolated GroupNorm and the temporal strides are both live.
#[test]
fn encoder_stack_matches_the_reference() {
    let f = fixture();
    let cfg = encode_fixture_config(3);
    let w = model_weights();
    let enc = VideoEncoder3d::from_weights(&w, "encoder", &cfg, Dtype::Float32).unwrap();

    let pixels = get(&f, "in.encoder.pixels");
    let want = get(&f, "out.encoder.params");
    assert!(
        pixels.shape()[2] > 1,
        "the golden must have a temporal axis"
    );

    let got = enc.forward(&pixels).unwrap();
    assert_eq!(got.shape(), want.shape());
    let (peak, mean) = rel(&got, &want);
    println!(
        "encoder stack: peak-rel {peak:.3e} mean-rel {mean:.3e} shape {:?}",
        got.shape()
    );
    assert_parity(&got, &want, TOL, "encoder stack");
    assert!(std_dev(&want) > 1e-4, "a ~constant golden is a false green");
}

/// The spatial compression really is `patch_size`, with **no cropping**. A downsampler that
/// strides time but not space still convolves a 3-wide kernel with no spatial padding and would
/// crop two columns per level — producing a plausible latent of the wrong size.
#[test]
fn the_encoder_halves_rather_than_crops() {
    let f = fixture();
    let cfg = encode_fixture_config(3);
    let pixels = get(&f, "in.encoder.pixels");
    let want = get(&f, "out.encoder.params");
    let (h, w) = (pixels.shape()[3], pixels.shape()[4]);
    assert_eq!(
        want.shape()[3],
        h / cfg.patch_size,
        "latent height must be pixel height / {}",
        cfg.patch_size
    );
    assert_eq!(want.shape()[4], w / cfg.patch_size);
    // …and the posterior carries mean AND logvar, so it is twice the latent width.
    assert_eq!(want.shape()[1], 2 * cfg.latent_channels);
}

/// `_encode_clip` untiled — the encoder followed by `quant_conv`.
#[test]
fn encode_clip_matches_the_reference_untiled() {
    let f = fixture();
    let v = vae(3);
    let pixels = get(&f, "in.encode_clip.pixels");
    let want = get(&f, "out.encode_clip.params");

    // A tile larger than the canvas degenerates to a single span — that IS the untiled path.
    let got = v.encode_clip_tiled(&pixels, 4096, 64).unwrap();
    let (peak, mean) = rel(&got, &want);
    println!("encode_clip untiled: peak-rel {peak:.3e} mean-rel {mean:.3e}");
    assert_parity(&got, &want, TOL, "encode_clip untiled");

    // The shipped 256 px tile is also larger than this fixture canvas, so the production entry
    // point must agree with the untiled result exactly rather than approximately.
    let production = v.encode_clip(&pixels).unwrap();
    assert!(pixels.shape()[3] < TILE_SAMPLE_MIN_SIZE);
    assert_parity(
        &production,
        &want,
        TOL,
        "encode_clip at the shipped tile size",
    );
}

/// **The tiled encode**, at the fixture's shrunk tile geometry — the blend, the round-robin slack
/// distribution and the latent-space overlaps all at once.
///
/// This is the convention most likely to ship unimplemented, because the shipped VAE turns tiling
/// on in `__init__` rather than through `enable_tiling()`, so a port written from the class body
/// alone would never see it.
#[test]
fn tiled_encode_matches_the_reference() {
    let f = fixture();
    let v = vae(3);
    let (tile, overlap) = encode_fixture_tiles(&f);
    let pixels = get(&f, "in.encode_clip.pixels");
    let want = get(&f, "out.encode_clip_tiled.params");
    let untiled = get(&f, "out.encode_clip.params");

    assert!(
        pixels.shape()[3] > tile,
        "the canvas must span more than one {tile}px tile or this test proves nothing"
    );

    let got = v.encode_clip_tiled(&pixels, tile, overlap).unwrap();
    assert_eq!(got.shape(), want.shape());
    let (peak, mean) = rel(&got, &want);
    println!("encode_clip tiled ({tile}/{overlap}): peak-rel {peak:.3e} mean-rel {mean:.3e}");
    assert_parity(&got, &want, TOL, "encode_clip tiled");

    // **The tiled result must DIFFER from the untiled one**, or the golden is inert and would
    // pass against a port that ignores tiling entirely.
    let (tile_delta, _) = rel(&want, &untiled);
    println!("tiled vs untiled: peak-rel {tile_delta:.3e}");
    assert!(
        tile_delta > MUTATION_FLOOR,
        "the tiled and untiled encodes agree to {tile_delta:.3e}; this fixture cannot gate the \
         tile blend"
    );
    const { assert!(ENCODER_TILING_IS_ON_BY_DEFAULT) };
}

/// **The keyframe path.** A single frame short-circuits the temporal chunking and produces
/// exactly ONE latent frame.
///
/// Padding it up to `clip_length` by repetition instead — the obvious implementation — would run
/// the temporal path over 17 copies of the image and return `17 / 4 - 3` latent frames, a
/// plausible tensor of the wrong shape carrying conditioning the model was never trained on.
#[test]
fn a_keyframe_encodes_to_exactly_one_latent_frame() {
    let f = fixture();
    let v = vae(3);
    let pixels = get(&f, "in.encode_single.pixels");
    assert_eq!(pixels.shape()[2], 1, "a keyframe is one frame");

    let posterior = v.encode(&pixels).unwrap();
    let want_mean = get(&f, "out.encode_single.mean");
    let want_std = get(&f, "out.encode_single.std");

    assert_eq!(
        posterior.mean().shape()[2],
        1,
        "a keyframe must encode to ONE latent frame, got {:?}",
        posterior.mean().shape()
    );
    let (peak, mean) = rel(posterior.mean(), &want_mean);
    println!("keyframe posterior mean: peak-rel {peak:.3e} mean-rel {mean:.3e}");
    assert_parity(posterior.mean(), &want_mean, TOL, "keyframe posterior mean");
    assert_parity(posterior.std(), &want_std, TOL, "keyframe posterior std");

    // The std is a real spread, not a degenerate zero that would make the seed irrelevant.
    assert!(
        std_dev(&want_std) > 1e-6,
        "a constant posterior std would make the sample seed inert"
    );
}

/// The chunked video path — 17 frames is exactly one clip, and `token_drop` trims the tail. Kept
/// alongside the keyframe golden so the two paths gate each other rather than only one being
/// exercised.
#[test]
fn chunked_encode_matches_the_reference() {
    let f = fixture();
    let v = vae(3);
    let pixels = get(&f, "in.encode_chunked.pixels");
    let want = get(&f, "out.encode_chunked.mean");
    assert_eq!(pixels.shape()[2], v.config().clip_length);

    let posterior = v.encode(&pixels).unwrap();
    assert_eq!(posterior.mean().shape(), want.shape());
    let (peak, mean) = rel(posterior.mean(), &want);
    println!(
        "chunked encode ({} frames -> {} latent): peak-rel {peak:.3e} mean-rel {mean:.3e}",
        pixels.shape()[2],
        want.shape()[2]
    );
    assert_parity(posterior.mean(), &want, TOL, "chunked encode");

    // The two paths really are different: one frame gives one latent frame, 17 give more than
    // one, and neither is the other's shape.
    let single = v.encode(&get(&f, "in.encode_single.pixels")).unwrap();
    assert_eq!(single.mean().shape()[2], 1);
    assert!(posterior.mean().shape()[2] > 1);
}

// ---------------------------------------------------------------------------------------------
// The conventions, probed where parity alone would be blind
// ---------------------------------------------------------------------------------------------

/// **Frame-isolated GroupNorm is not the same as a global one — and the difference is exactly
/// zero at `T = 1`.**
///
/// This is why the parity goldens above are dumped at `T = 5` and `T = 17`. Written as a direct
/// comparison rather than through the encoder, so it measures the convention itself.
#[test]
fn groupnorm_is_frame_isolated() {
    let cfg = encode_fixture_config(3);
    let channels = cfg.norm_num_groups;
    let mut w = Weights::empty();
    w.insert(
        "n.weight",
        Array::from_slice(&vec![1.0f32; channels as usize], &[channels]),
    );
    w.insert(
        "n.bias",
        Array::from_slice(&vec![0.0f32; channels as usize], &[channels]),
    );
    let norm =
        FrameIsolatedGroupNorm::from_weights(&w, "n", Dtype::Float32, cfg.norm_num_groups, 1e-6)
            .unwrap();

    // Two frames with WILDLY different scales. A global reduction shares one mean/variance across
    // both and squashes the second; the isolated one normalizes each on its own.
    let t = 2;
    let (h, wd) = (2, 2);
    let n = (t * h * wd * channels) as usize;
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let frame = i / ((h * wd * channels) as usize);
            let base = (i % 7) as f32;
            if frame == 0 {
                base
            } else {
                base * 100.0 + 500.0
            }
        })
        .collect();
    let x = Array::from_slice(&vals, &[1, t, h, wd, channels]);
    let out = norm.forward(&x).unwrap();
    assert_eq!(out.shape(), x.shape());

    // Each frame, alone, must normalize to the SAME tensor it does inside the pair — that is
    // precisely what isolation means.
    for frame in 0..t {
        let one = mlx_gen_minimax_h3::tensor::slice_axis(&x, 1, frame, frame + 1).unwrap();
        let alone = norm.forward(&one).unwrap();
        let inside = mlx_gen_minimax_h3::tensor::slice_axis(&out, 1, frame, frame + 1).unwrap();
        let (peak, _) = rel(&alone, &inside);
        assert!(
            peak < 1e-4,
            "frame {frame} normalized differently in isolation ({peak:.3e}); the statistics are \
             leaking across time"
        );
    }

    // …and a GLOBAL reduction over the same tensor is measurably different, so "isolated" is a
    // real choice rather than a distinction without a difference.
    let folded = x.reshape(&[1, t * h * wd, channels]).unwrap();
    let global = mlx_gen::nn::group_norm(
        &folded,
        &Array::from_slice(&vec![1.0f32; channels as usize], &[channels]),
        &Array::from_slice(&vec![0.0f32; channels as usize], &[channels]),
        channels,
        1e-6,
    )
    .unwrap()
    .reshape(&[1, t, h, wd, channels])
    .unwrap();
    let (delta, _) = rel(&global, &out);
    println!("frame-isolated vs global GroupNorm: peak-rel {delta:.3e}");
    assert!(
        delta > MUTATION_FLOOR,
        "a global GroupNorm gives the same answer ({delta:.3e}); this probe cannot gate isolation"
    );
}

// ---------------------------------------------------------------------------------------------
// Mutation probes — every parity assertion above, proved non-inert
// ---------------------------------------------------------------------------------------------

/// Perturb one encoder tensor and confirm the encode moves. Each row proves the corresponding
/// golden actually depends on that tensor — most importantly `conv_shortcut`, which a loader could
/// omit entirely and still produce a plausible latent.
#[test]
fn encoder_weights_are_all_load_bearing() {
    let f = fixture();
    let cfg = encode_fixture_config(3);
    let pixels = get(&f, "in.encode_clip.pixels");
    let baseline = vae(3).encode_clip_tiled(&pixels, 4096, 64).unwrap();
    assert_parity(
        &baseline,
        &get(&f, "out.encode_clip.params"),
        TOL,
        "baseline before mutation",
    );

    let probes = [
        "encoder.conv_in.weight",
        "encoder.down_blocks.0.resnets.0.conv1.weight",
        "encoder.down_blocks.1.downsamplers.0.conv.weight",
        "encoder.down_blocks.3.resnets.0.conv_shortcut.weight",
        "encoder.norm_out.weight",
        "encoder.conv_out.bias",
        "quant_conv.weight",
    ];
    for key in probes {
        let mut w = model_weights();
        let original = w
            .require(key)
            .unwrap_or_else(|_| panic!("fixture has no `{key}`"))
            .clone();
        // A multiplicative perturbation, so a zero-valued tensor cannot absorb it silently.
        w.insert(
            key,
            original
                .multiply(Array::from_f32(1.25))
                .unwrap()
                .add(Array::from_f32(0.05))
                .unwrap(),
        );
        let mutated = MiniMaxH3VideoVae::from_weights(&mut w, &cfg, Dtype::Float32)
            .unwrap()
            .encode_clip_tiled(&pixels, 4096, 64)
            .unwrap();
        let (peak, _) = rel(&mutated, &baseline);
        println!("mutate {key}: peak-rel {peak:.3e}");
        assert!(
            peak > MUTATION_FLOOR,
            "perturbing `{key}` moved the encode by only {peak:.3e} — the golden does not gate it"
        );
    }
}

/// The tiled golden depends on the tile geometry: encoding at a different tile size must give a
/// different answer, or the "tiled" test is silently running the untiled path.
#[test]
fn the_tile_geometry_changes_the_result() {
    let f = fixture();
    let v = vae(3);
    let (tile, overlap) = encode_fixture_tiles(&f);
    let pixels = get(&f, "in.encode_clip.pixels");

    let at_fixture = v.encode_clip_tiled(&pixels, tile, overlap).unwrap();
    let untiled = v.encode_clip_tiled(&pixels, 4096, overlap).unwrap();
    let (delta, _) = rel(&at_fixture, &untiled);
    assert!(
        delta > MUTATION_FLOOR,
        "the tile size is inert ({delta:.3e})"
    );

    // A different tile SIZE is a different plan and so a different result.
    let narrower = tile * 3 / 4;
    let at_narrower = v.encode_clip_tiled(&pixels, narrower, overlap).unwrap();
    let (ndelta, _) = rel(&at_narrower, &at_fixture);
    println!("tile {tile}/{overlap} vs {narrower}/{overlap}: peak-rel {ndelta:.3e}");
    assert!(
        ndelta > MUTATION_FLOOR,
        "narrowing the tile to {narrower}px changed nothing ({ndelta:.3e})"
    );

    // **Raising `min_overlap` is NOT automatically a different plan**, and that is a property of
    // the reference worth pinning rather than a gap in the probe above. `_split_tiles` first picks
    // the smallest tile count that can cover the axis at `min_overlap`, then distributes ALL the
    // remaining slack over the overlaps in whole compression-ratio steps — so the realized overlap
    // is usually already larger than the minimum, and asking for more changes nothing until the
    // tile count itself has to grow. Here 4 and 8 both realize an overlap of 8.
    let wider = v.encode_clip_tiled(&pixels, tile, overlap * 2).unwrap();
    let (odelta, _) = rel(&wider, &at_fixture);
    println!(
        "tile {tile}/{overlap} vs {tile}/{}: peak-rel {odelta:.3e} (slack already absorbed it)",
        overlap * 2
    );
    assert!(
        odelta < 1e-6,
        "a min_overlap the slack has already absorbed must give the identical plan, got \
         {odelta:.3e}"
    );
}

/// Malformed encode inputs are typed errors, not silent reshapes.
///
/// # Each rejection is asserted by MESSAGE, not by `is_err()` (sc-19488)
///
/// Every input below would fail *somewhere* with the guard deleted — a rank-1 tensor has no `s[1]`
/// to read, 4 channels meets a 3-channel `CausalConv3d`, and a 0-frame tensor reflect-pads off an
/// axis of length 0. So `assert!(..is_err())` passes here whether or not `encode`'s entry guards
/// exist, which makes it inert: it is satisfied by an error that has nothing to do with the
/// property under test. That is not hypothetical on this crate — the sibling tile probe in
/// `video_vae_parity.rs` survived its own mutation for exactly this reason, failing downstream with
/// "cannot reflect-pad 1/1 off an axis of length 1".
///
/// Naming the guard's own message is what makes each of these bite: delete the matching arm of
/// `MiniMaxH3VideoVae::encode` and the input still errors, but with a different string, and the
/// assertion goes red.
#[test]
fn encode_rejects_malformed_input() {
    let v = vae(3);
    let ok = Array::from_slice(&vec![0.5f32; 3 * 8 * 8], &[1, 3, 1, 8, 8]);
    v.encode(&ok).unwrap();

    // Wrong rank.
    let msg = v
        .encode(&Array::from_slice(&[1.0f32], &[1]))
        .expect_err("a rank-1 tensor is not [B, 3, T, H, W]")
        .to_string();
    assert!(
        // Full prefix, not the bare shape tail — `src/pipeline.rs`'s decode guard spells the same
        // `expected [B, 3, T, H, W]`, so the short form would go inert if encode ever delegated
        // through it (sc-19488).
        msg.contains("minimax-h3 vae encode: expected [B, 3, T, H, W]"),
        "the rank guard must be what rejects this, not a downstream shape fault: {msg}"
    );

    // Wrong channel count — 4 channels is a plausible RGBA mistake.
    let rgba = Array::from_slice(&vec![0.5f32; 4 * 8 * 8], &[1, 4, 1, 8, 8]);
    let msg = v
        .encode(&rgba)
        .expect_err("an RGBA-shaped input must be rejected rather than convolved")
        .to_string();
    // The substring is deliberately the guard's FULL prefix, not the bare phrase "input channels".
    // Measured (sc-19488): with this guard disabled the RGBA tensor reaches the encoder stack and
    // mlx rejects it with `[conv] Expect the input channels in the input and weight array to match`
    // — which CONTAINS "input channels", so the shorter substring stayed green under its own
    // mutation and was still inert. Matching `minimax-h3 vae encode:` pins it to this crate's guard.
    assert!(
        msg.contains("minimax-h3 vae encode: expected 3 input channels"),
        "the channel guard must be what rejects this, not the 3-channel conv failing later: {msg}"
    );

    // Zero frames.
    let empty = Array::from_slice(&Vec::<f32>::new(), &[1, 3, 0, 8, 8]);
    let msg = v
        .encode(&empty)
        .expect_err("a zero-frame clip must be refused")
        .to_string();
    assert!(
        msg.contains("at least one frame"),
        "the frame-count guard must be what rejects this, not the reflect pad: {msg}"
    );
}

/// The posterior sample is `mean + std · noise`, and the noise is genuinely load-bearing — a
/// sampler that quietly returned the mean would pass any test that only inspects shapes.
#[test]
fn the_posterior_sample_depends_on_its_noise() {
    let f = fixture();
    let v = vae(3);
    let posterior = v.encode(&get(&f, "in.encode_single.pixels")).unwrap();
    let shape = posterior.mean().shape().to_vec();
    let n: i32 = shape.iter().product();

    let zeros = mlx_rs::ops::zeros_dtype(&shape, Dtype::Float32).unwrap();
    assert_parity(
        &posterior.sample_with(&zeros).unwrap(),
        posterior.mean(),
        1e-6,
        "zero noise samples the mean",
    );

    let ones: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let noise = Array::from_slice(&ones, &shape);
    let sampled = posterior.sample_with(&noise).unwrap();
    let (delta, _) = rel(&sampled, posterior.mean());
    println!("posterior sample vs mean: peak-rel {delta:.3e}");
    assert!(
        delta > MUTATION_FLOOR,
        "the posterior noise is inert ({delta:.3e}); std must be a real spread"
    );

    // A shape mismatch is rejected rather than broadcast. Asserted by message (sc-19488): with the
    // guard deleted, `[1, 1, 1, 1, 1]` would BROADCAST cleanly against the mean on some shapes and
    // fail on others, so `is_err()` alone reports the guard's absence inconsistently. The asserted
    // text is the FULL unique prefix: `does not match the mean` alone is emitted by three guards
    // (this one, the audio encoder's, and the candle twin), so the short form would not identify
    // which one fired.
    let wrong = mlx_rs::ops::zeros_dtype(&[1, 1, 1, 1, 1], Dtype::Float32).unwrap();
    let msg = posterior
        .sample_with(&wrong)
        .expect_err("noise of the wrong shape must be refused, not broadcast")
        .to_string();
    assert!(
        msg.contains("minimax-h3 encoder: posterior noise"),
        "the VIDEO encoder's own noise-shape guard must be what rejects this: {msg}"
    );
    let _ = DiagonalGaussian::from_parameters(&Array::from_slice(&[0.0f32, 0.0], &[1, 2, 1, 1, 1]))
        .unwrap();
}

/// The shipped tile constants are the reference's, and the fixture's are deliberately NOT.
#[test]
fn the_shipped_tile_geometry_is_pinned() {
    assert_eq!(TILE_SAMPLE_MIN_SIZE, 256);
    assert_eq!(TILE_SAMPLE_MIN_OVERLAP, 64);
    let (tile, overlap) = encode_fixture_tiles(&fixture());
    assert!(
        tile < TILE_SAMPLE_MIN_SIZE && overlap < TILE_SAMPLE_MIN_OVERLAP,
        "the fixture tiles at {tile}/{overlap}; it must be SMALLER than the shipped \
         {TILE_SAMPLE_MIN_SIZE}/{TILE_SAMPLE_MIN_OVERLAP} so a committable canvas spans more than \
         one tile"
    );
}
