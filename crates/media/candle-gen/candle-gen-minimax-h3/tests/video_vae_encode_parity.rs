//! sc-19008: candle video-VAE **encode** parity — the 3-D causal CNN half, against the official
//! diffusers `AutoencoderKLMiniMaxH3`.
//!
//! Fixture `tests/fixtures/video_vae_encode.safetensors` ← the MLX lane's
//! `tools/dump_minimax_h3_video_vae_encode.py`, copied byte-for-byte (`cross_backend.rs` asserts
//! that). Its goldens are the **official diffusers class's own outputs**, so every number below is
//! a measurement against the reference implementation rather than against another port.
//!
//! `fl2va` conditions a keyframe through **both** the text encoder's vision tower and this VAE, so
//! sc-17154's decode-only port is not sufficient for it. The four conventions this file exists to
//! gate are set out in [`candle_gen_minimax_h3::vae_encoder`]'s module docs; each has a parity
//! golden here and, where the convention could be silently inert, a probe that makes the wrong
//! implementation measurably different rather than merely unproven.
//!
//! # What "unproven" would look like here
//!
//! Four of the conventions are invisible under the obvious test:
//!
//! | convention | wrong version is indistinguishable when… |
//! |---|---|
//! | frame-isolated GroupNorm | `T == 1` — the isolated and global reductions are the same tensor |
//! | tiled encode | the canvas is smaller than one tile — the plan degenerates to one span |
//! | single-frame short circuit | the input already has ≥ 17 frames |
//! | asymmetric downsampler pad | the shape alone — a symmetric pad gives the SAME output size |
//!
//! So every golden below is dumped at `T = 5` or `T = 17` with a canvas that really does span two
//! tiles, and the degenerate cases are asserted to be degenerate rather than assumed to be.
//!
//! # Tolerance, and what it can resolve
//!
//! [`TOL`] is **1e-5 peak-relative**, three orders TIGHTER than the MLX sibling's 1e-2 house value
//! and set from *this* lane's own measured floor, exactly as the decode suite's is. That is not a
//! stricter standard applied to the same hardware — MLX pays for Metal's reduced-precision f32
//! matmul and measures a ~1e-3 residual, while this lane runs f32 on the CPU and measures
//! **4.8e-7 … 1.5e-6** across the goldens in this file (each test prints its own), so the bound
//! sits ~7x above the worst of them. Keeping 1e-2 here would leave a gate four orders above its
//! own noise floor, which is a gate in name only.
//!
//! The encode residual is ~5x the decode suite's 2.1e-7 … 3.7e-7 rather than equal to it: this
//! path is twelve stacked convolutions whose reduction order differs from torch's, where the
//! decoder is a matmul stack. The bound is set from the number this lane actually measures, not
//! copied from the sibling suite.
//!
//! On the **real** 118-tensor checkpoint the same comparison measures at worst **1.730e-5** on the
//! CPU and **1.848e-5** on Metal (`real_weights.rs::real_weight_encode_matches_the_official_diffusers_vae`,
//! bound `REAL_WEIGHT_ENCODE_TOL`, derived there from that arm's own floor across both devices,
//! sc-19455) — a different bound because that comparison spans six levels and 1024-wide channels,
//! not because a different standard applies.
//!
//! **Every assertion is on the peak relative max-abs-diff.** Not norm, not cosine, not a checksum:
//! sc-18740's gate/value half-swap sat at cosine 0.73-0.78 with output norms 89 → 85, and a
//! `flip_sin_to_cos` error in this family moves output by 9.999e-1 at *unchanged* norm. Only the
//! relative max-abs-diff moves for every defect class this epic has actually shipped.

mod common;

use std::collections::HashMap;

use common::{
    assert_parity, encode_fixture_config, encode_fixture_tiles, flat, rel, std_dev, weights,
    Golden, ENCODE_FIXTURE, FIXTURE,
};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen_minimax_h3::{
    reflect_pad_axis, CausalConv3d, DiagonalGaussian, FrameIsolatedGroupNorm, MiniMaxH3VideoVae,
    VideoEncoder3d, ENCODER_TEMPORAL_PADDING, ENCODER_TILING_IS_ON_BY_DEFAULT,
    TILE_SAMPLE_MIN_OVERLAP, TILE_SAMPLE_MIN_SIZE,
};

/// Peak-relative tolerance. See the module docs: set from this lane's measured 4.8e-7 … 1.5e-6
/// floor, not inherited from the MLX sibling's 1e-2 nor from the decode suite's own residual.
const TOL: f32 = 1e-5;

/// A mutation must move the encode by at least this much to count as gated. Three orders above
/// [`TOL`], so "the golden depends on this tensor" is never a marginal call.
const MUTATION_FLOOR: f32 = 1e-2;

/// The bar for the four tensors a **downstream normalization largely removes** — see
/// [`ABSORBED_BY_THE_NEXT_NORM`]. Still 10× [`TOL`] and ~70× this lane's measured 1.4e-6 parity
/// residual, so the goldens really do gate them (they measure 2.2e-4 … 6.7e-3); they simply cannot
/// clear [`MUTATION_FLOOR`], and pretending otherwise would mean either dropping the probes or
/// lowering the bar for all sixteen.
const NORMALIZED_MUTATION_FLOOR: f32 = 1e-4;

/// The tensors whose perturbation the next `GroupNorm` largely cancels, so they are held to
/// [`NORMALIZED_MUTATION_FLOOR`] instead of [`MUTATION_FLOOR`]. Enumerated rather than inferred, so
/// a change that quiets some *other* tensor fails this suite instead of being waved through.
///
/// Three are biases feeding straight into a norm, and one is the last conv before `norm_out`. The
/// effect is **amplified by the fixture's geometry**: it sets `norm_num_groups = 32` with 32-wide
/// blocks, so each group holds ONE channel and the GroupNorm degenerates to an instance norm,
/// which removes a per-channel offset exactly. The shipped model runs 32 groups over 128…1024
/// channels — 4 to 32 channels per group — where the same offset survives. These four are
/// therefore *quieter in the fixture than on real weights*, which is a property of the fixture and
/// not of the port.
const ABSORBED_BY_THE_NEXT_NORM: [&str; 4] = [
    "encoder.down_blocks.2.downsamplers.0.conv.bias",
    "encoder.down_blocks.3.resnets.0.conv2.weight",
    "encoder.down_blocks.3.resnets.0.conv_shortcut.bias",
    "quant_conv.bias",
];

fn fixture() -> Golden {
    Golden::load(ENCODE_FIXTURE)
}

fn model_map(f: &Golden) -> HashMap<String, Tensor> {
    f.model_map(&["src.", "in.", "out.", "const."])
}

fn vae_from(map: HashMap<String, Tensor>, token_drop: i32) -> MiniMaxH3VideoVae {
    MiniMaxH3VideoVae::from_weights(
        &weights(map),
        &encode_fixture_config(token_drop),
        &Device::Cpu,
        DType::F32,
    )
    .expect("build the fixture VAE")
}

fn vae(f: &Golden, token_drop: i32) -> MiniMaxH3VideoVae {
    vae_from(model_map(f), token_drop)
}

// ---------------------------------------------------------------------------------------------
// Rule 3: what produced this golden
// ---------------------------------------------------------------------------------------------

/// **The sc-18740 methodology gate.** A fixture generated from the MiniMax reference modules and
/// renamed onto the published key names carries the *source* layout under *published* names; a
/// loader that reads it the source way agrees with it and is wrong on the real checkpoint.
///
/// So this suite refuses a fixture that cannot show it came from the converted-checkpoint path.
/// `candle_core::safetensors` drops `__metadata__`, which is why `common::Golden` parses the
/// container header by hand.
#[test]
fn fixture_provenance_records_the_converted_path() {
    let f = fixture();
    assert_eq!(
        f.meta("provenance"),
        Some("converted-checkpoint"),
        "this golden must come from the CONVERTED checkpoint layout (`AutoencoderKLMiniMaxH3`), \
         not from the MiniMax reference modules under a rename table — see \
         candle_gen_minimax_h3::layout Rule 3"
    );
    assert_eq!(
        f.meta("reference"),
        Some("diffusers.AutoencoderKLMiniMaxH3")
    );
    assert_eq!(f.meta("half"), Some("encode"));
    let version = f
        .meta("reference_version")
        .expect("the fixture must record which diffusers produced it");
    assert!(
        version.starts_with("0.40"),
        "unexpected reference version {version}"
    );
    // The generator recorded how far the tiled encode moved from the untiled one. If that ever
    // regenerates near zero the tiled golden is inert, and this suite would silently stop gating
    // the blend.
    let blend: f32 = f
        .meta("encode_tile_blend_rel")
        .expect("the fixture records its own tile-blend separation")
        .parse()
        .expect("a float");
    println!("fixture: tiled vs untiled separation {blend:.3e} (recorded by the generator)");
    assert!(blend > MUTATION_FLOOR, "the tiled golden is inert");
    // The compression ratios the generator asserted against the running reference.
    assert_eq!(f.meta("spatial_compression_ratio"), Some("4"));
    assert_eq!(f.meta("temporal_compression_ratio"), Some("4"));
    let cfg = encode_fixture_config(3);
    assert_eq!(cfg.patch_size, 4, "the port must use the same geometry");
    assert_eq!(cfg.patch_size_t, 4);
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
    let w = weights(model_map(&f));
    let enc = VideoEncoder3d::from_weights(&w, "encoder", &cfg, DType::F32).expect("encoder");

    let pixels = f.tensor("in.encoder.pixels");
    let want = f.tensor("out.encoder.params");
    assert!(pixels.dims()[2] > 1, "the golden must have a temporal axis");

    let got = enc.forward(&pixels).expect("encoder forward");
    assert_eq!(got.dims(), want.dims());
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
    let pixels = f.shape("in.encoder.pixels");
    let want = f.shape("out.encoder.params");
    assert_eq!(
        want[3],
        pixels[3] / cfg.patch_size,
        "latent height must be pixel height / {}",
        cfg.patch_size
    );
    assert_eq!(want[4], pixels[4] / cfg.patch_size);
    // …and the posterior carries mean AND logvar, so it is twice the latent width.
    assert_eq!(want[1], 2 * cfg.latent_channels);
}

/// `_encode_clip` untiled — the encoder followed by `quant_conv`.
#[test]
fn encode_clip_matches_the_reference_untiled() {
    let f = fixture();
    let v = vae(&f, 3);
    let pixels = f.tensor("in.encode_clip.pixels");
    let want = f.tensor("out.encode_clip.params");

    // A tile larger than the canvas degenerates to a single span — that IS the untiled path.
    let got = v.encode_clip_tiled(&pixels, 4096, 64).expect("encode_clip");
    assert_parity(&got, &want, TOL, "encode_clip untiled");

    // The shipped 256 px tile is also larger than this fixture canvas, so the production entry
    // point must agree with the untiled result exactly rather than approximately.
    assert!(f.shape("in.encode_clip.pixels")[3] < TILE_SAMPLE_MIN_SIZE);
    let production = v.encode_clip(&pixels).expect("encode_clip");
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
    let v = vae(&f, 3);
    let (tile, overlap) = encode_fixture_tiles(&f);
    let pixels = f.tensor("in.encode_clip.pixels");
    let want = f.tensor("out.encode_clip_tiled.params");
    let untiled = f.tensor("out.encode_clip.params");

    assert!(
        f.shape("in.encode_clip.pixels")[3] > tile,
        "the canvas must span more than one {tile}px tile or this test proves nothing"
    );

    let got = v
        .encode_clip_tiled(&pixels, tile, overlap)
        .expect("tiled encode");
    assert_eq!(got.dims(), want.dims());
    assert_parity(
        &got,
        &want,
        TOL,
        &format!("encode_clip tiled {tile}/{overlap}"),
    );

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
/// the temporal path over 17 copies of the image and return several latent frames, a plausible
/// tensor of the wrong shape carrying conditioning the model was never trained on.
#[test]
fn a_keyframe_encodes_to_exactly_one_latent_frame() {
    let f = fixture();
    let v = vae(&f, 3);
    let pixels = f.tensor("in.encode_single.pixels");
    assert_eq!(pixels.dims()[2], 1, "a keyframe is one frame");

    let posterior = v.encode(&pixels).expect("encode");
    let want_mean = f.tensor("out.encode_single.mean");
    let want_std = f.tensor("out.encode_single.std");

    assert_eq!(
        posterior.mean().dims()[2],
        1,
        "a keyframe must encode to ONE latent frame, got {:?}",
        posterior.mean().dims()
    );
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
    let v = vae(&f, 3);
    let pixels = f.tensor("in.encode_chunked.pixels");
    let want = f.tensor("out.encode_chunked.mean");
    assert_eq!(pixels.dims()[2] as i32, v.config().clip_length);

    let posterior = v.encode(&pixels).expect("encode");
    assert_eq!(posterior.mean().dims(), want.dims());
    assert_parity(posterior.mean(), &want, TOL, "chunked encode");

    // The two paths really are different: one frame gives one latent frame, 17 give more than
    // one, and neither is the other's shape.
    let single = v
        .encode(&f.tensor("in.encode_single.pixels"))
        .expect("encode");
    assert_eq!(single.mean().dims()[2], 1);
    assert!(posterior.mean().dims()[2] > 1);
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
    let dev = Device::Cpu;
    let cfg = encode_fixture_config(3);
    let channels = cfg.norm_num_groups;
    let mut map = HashMap::new();
    map.insert(
        "n.weight".to_string(),
        Tensor::from_vec(vec![1.0f32; channels], channels, &dev).unwrap(),
    );
    map.insert(
        "n.bias".to_string(),
        Tensor::zeros(channels, DType::F32, &dev).unwrap(),
    );
    let norm = FrameIsolatedGroupNorm::from_weights(&weights(map), "n", DType::F32, channels, 1e-6)
        .expect("group norm");

    // Two frames with WILDLY different scales. A global reduction shares one mean/variance across
    // both and squashes the second; the isolated one normalizes each on its own.
    let (t, h, w) = (2usize, 2usize, 2usize);
    let n = t * h * w * channels;
    // NCTHW: index = ((c · T + f) · H + y) · W + x.
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let frame = (i / (h * w)) % t;
            let base = (i % 7) as f32;
            if frame == 0 {
                base
            } else {
                base * 100.0 + 500.0
            }
        })
        .collect();
    let x = Tensor::from_vec(vals, (1, channels, t, h, w), &dev).unwrap();
    let out = norm.forward(&x).expect("forward");
    assert_eq!(out.dims(), x.dims());

    // Each frame, alone, must normalize to the SAME tensor it does inside the pair — that is
    // precisely what isolation means.
    for frame in 0..t {
        let one = x.narrow(2, frame, 1).unwrap().contiguous().unwrap();
        let alone = norm.forward(&one).expect("forward");
        let inside = out.narrow(2, frame, 1).unwrap().contiguous().unwrap();
        let (peak, _) = rel(&alone, &inside);
        assert!(
            peak < 1e-5,
            "frame {frame} normalized differently in isolation ({peak:.3e}); the statistics are \
             leaking across time"
        );
    }

    // …and a GLOBAL reduction over the same tensor is measurably different, so "isolated" is a
    // real choice rather than a distinction without a difference. Computed here directly: one
    // mean/variance per (batch, group) over the WHOLE (C/G, T, H, W) volume.
    let flatx = flat(&x);
    let mean = flatx.iter().sum::<f32>() / flatx.len() as f32;
    let var = flatx.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / flatx.len() as f32;
    let global: Vec<f32> = flatx
        .iter()
        .map(|v| (v - mean) / (var + 1e-6f32).sqrt())
        .collect();
    let global = Tensor::from_vec(global, x.dims(), &dev).unwrap();
    let (delta, _) = rel(&global, &out);
    println!("frame-isolated vs global GroupNorm: peak-rel {delta:.3e}");
    assert!(
        delta > MUTATION_FLOOR,
        "a global GroupNorm gives the same answer ({delta:.3e}); this probe cannot gate isolation"
    );
}

/// **The downsampler's reflect pad is bottom/right only, and a symmetric one is the same SHAPE.**
///
/// This is the convention no shape check can see: both pads give `ceil(size / 2)` output on an
/// even canvas, so only the values differ. Fed through the fixture's own stride-2 downsampler
/// weight so the separation is measured on real convolution output rather than on the pad alone.
#[test]
fn the_downsampler_pad_is_asymmetric_and_that_is_not_free() {
    let f = fixture();
    let w = weights(model_map(&f));
    // The stride-(2,2,2) downsampler at level 1 of the fixture geometry.
    let conv = CausalConv3d::from_weights(
        &w,
        "encoder.down_blocks.1.downsamplers.0.conv",
        DType::F32,
        (2, 2, 2),
        0,
        ENCODER_TEMPORAL_PADDING,
    )
    .expect("downsampler");

    let (b, c, t, h, wd) = (1usize, 32usize, 5usize, 8usize, 8usize);
    let vals: Vec<f32> = (0..b * c * t * h * wd)
        .map(|i| ((i as f32) * 0.29).sin())
        .collect();
    let x = Tensor::from_vec(vals, (b, c, t, h, wd), &Device::Cpu).unwrap();

    let asym = {
        let p = reflect_pad_axis(&x, 3, 0, 1).unwrap();
        let p = reflect_pad_axis(&p, 4, 0, 1).unwrap();
        conv.forward(&p).expect("asymmetric")
    };
    let sym = {
        let p = reflect_pad_axis(&x, 3, 1, 1).unwrap();
        let p = reflect_pad_axis(&p, 4, 1, 1).unwrap();
        conv.forward(&p).expect("symmetric")
    };

    // The asymmetric pad is what gives exactly `ceil(size / 2)`.
    assert_eq!(asym.dims()[3], h.div_ceil(2));
    assert_eq!(asym.dims()[4], wd.div_ceil(2));
    assert_eq!(
        sym.dims(),
        asym.dims(),
        "a symmetric pad produces the SAME shape on an even canvas — shape cannot gate this"
    );
    let (delta, _) = rel(&sym, &asym);
    println!("symmetric vs asymmetric downsampler pad: peak-rel {delta:.3e} (shapes identical)");
    assert!(
        delta > MUTATION_FLOOR,
        "a symmetric pad gives the same answer ({delta:.3e}); the parity goldens above cannot be \
         said to gate the half-pixel shift"
    );
}

/// **What this suite cannot resolve**, pinned with numbers rather than left to be assumed.
///
/// The encoder's epsilon is 1e-6 and the decoder's is 1e-5, and `vae/config.json` ships both under
/// different keys (`norm_eps` / `decoder_norm_eps`). Using one for the other is a pure numeric
/// change: no shape moves, no tensor goes unread, and no key-mapping proof can see it.
///
/// Measured here, the substitution moves the encode by **~2.1e-6** — above this lane's ~1.4e-6
/// parity residual but *below* the [`TOL`] the goldens gate at, so **the parity assertions above
/// would stay green on a port that made it**. The assertion is deliberately in that direction: it
/// is not a wish that the gate stay weak, it is a pin on a claim, and if the residual ever falls
/// far enough that parity *can* resolve the swap, this fails and the docs get rewritten rather
/// than silently left standing.
///
/// The class is covered **structurally** instead, by `config`'s `shipped_encoder_geometry_is_pinned`
/// (the two epsilons are asserted distinct and each pinned to its own value) and
/// `a_config_missing_an_encoder_key_is_rejected` (a config without `norm_eps` is an error, not a
/// silently-defaulted encoder).
#[test]
fn the_parity_bound_cannot_resolve_an_epsilon_substitution() {
    let f = fixture();
    let pixels = f.tensor("in.encode_clip.pixels");
    let cfg = encode_fixture_config(3);
    assert_eq!(cfg.encoder_norm_eps, 1e-6, "the ENCODER's eps");
    assert_eq!(cfg.norm_eps, 1e-5, "the DECODER's eps");

    let map = model_map(&f);
    let baseline = vae_from(map.clone(), 3)
        .encode_clip_tiled(&pixels, 4096, 64)
        .expect("baseline");

    let mut wrong = cfg.clone();
    wrong.encoder_norm_eps = cfg.norm_eps;
    let displaced =
        MiniMaxH3VideoVae::from_weights(&weights(map), &wrong, &Device::Cpu, DType::F32)
            .expect("vae")
            .encode_clip_tiled(&pixels, 4096, 64)
            .expect("displaced");

    let (delta, _) = rel(&displaced, &baseline);
    let (residual, _) = rel(&baseline, &f.tensor("out.encode_clip.params"));
    println!(
        "BLIND SPOT: encoder eps 1e-6 -> 1e-5 moves the encode by {delta:.3e}; this lane's \
         residual against the reference is {residual:.3e} and the parity bound is {TOL:.0e} — the \
         substitution is {:.0}x SMALLER than the bound",
        TOL / delta
    );
    assert!(
        delta > residual,
        "the substitution ({delta:.3e}) is below this lane's own parity residual \
         ({residual:.3e}); it would not even be a distinguishable computation"
    );
    assert!(
        delta < TOL,
        "an epsilon substitution ({delta:.3e}) now clears the parity bound ({TOL:.0e}). That is \
         an improvement, but this file's docs claim the goldens cannot resolve it — re-state them \
         before re-greening this."
    );
}

// ---------------------------------------------------------------------------------------------
// Mutation probes — every parity assertion above, proved non-inert
// ---------------------------------------------------------------------------------------------

/// Perturb one encoder tensor **at a time** and confirm the encode moves. Each row proves the
/// corresponding golden actually depends on that tensor — most importantly `conv_shortcut`, which
/// a loader could omit entirely and still produce a plausible latent.
///
/// Mutating the whole set at once would prove only that the set is load-bearing, not each member.
#[test]
fn encoder_weights_are_all_load_bearing() {
    let f = fixture();
    let pixels = f.tensor("in.encode_clip.pixels");
    let baseline = vae(&f, 3)
        .encode_clip_tiled(&pixels, 4096, 64)
        .expect("baseline");
    assert_parity(
        &baseline,
        &f.tensor("out.encode_clip.params"),
        TOL,
        "baseline before mutation",
    );

    let probes = [
        "encoder.conv_in.weight",
        "encoder.conv_in.bias",
        "encoder.down_blocks.0.resnets.0.conv1.weight",
        "encoder.down_blocks.0.resnets.0.norm1.weight",
        "encoder.down_blocks.0.resnets.0.norm2.bias",
        "encoder.down_blocks.1.downsamplers.0.conv.weight",
        "encoder.down_blocks.2.downsamplers.0.conv.bias",
        "encoder.down_blocks.3.resnets.0.conv2.weight",
        "encoder.down_blocks.3.resnets.0.conv_shortcut.weight",
        "encoder.down_blocks.3.resnets.0.conv_shortcut.bias",
        "encoder.norm_out.weight",
        "encoder.norm_out.bias",
        "encoder.conv_out.weight",
        "encoder.conv_out.bias",
        "quant_conv.weight",
        "quant_conv.bias",
    ];
    for key in probes {
        let mut map = model_map(&f);
        let original = map
            .get(key)
            .unwrap_or_else(|| panic!("fixture has no `{key}`"))
            .clone();
        // A multiplicative perturbation PLUS an offset, so neither a zero-valued tensor nor a
        // scale-invariant downstream norm can absorb it silently.
        map.insert(
            key.to_string(),
            ((original * 1.25).unwrap() + 0.05).unwrap(),
        );
        let mutated = vae_from(map, 3)
            .encode_clip_tiled(&pixels, 4096, 64)
            .expect("mutated encode");
        let (peak, _) = rel(&mutated, &baseline);
        let absorbed = ABSORBED_BY_THE_NEXT_NORM.contains(&key);
        let floor = if absorbed {
            NORMALIZED_MUTATION_FLOOR
        } else {
            MUTATION_FLOOR
        };
        println!(
            "mutate {key}: peak-rel {peak:.3e} (floor {floor:.0e}{})",
            if absorbed { ", norm-absorbed" } else { "" }
        );
        assert!(
            peak > floor,
            "perturbing `{key}` moved the encode by only {peak:.3e} — the golden does not gate it"
        );
        // The lower tier must be earned, not claimed: a tensor listed as norm-absorbed that in
        // fact moves the encode loudly is a stale exemption and has to go back to the main floor.
        if absorbed {
            assert!(
                peak < MUTATION_FLOOR,
                "`{key}` moves the encode by {peak:.3e}, which clears the ordinary \
                 {MUTATION_FLOOR:.0e} floor; drop it from ABSORBED_BY_THE_NEXT_NORM"
            );
        }
    }
    assert_eq!(probes.len(), 16, "every module kind must have a probe");
    for key in ABSORBED_BY_THE_NEXT_NORM {
        assert!(
            probes.contains(&key),
            "`{key}` is exempted but never probed"
        );
    }
    // The lower tier is still an order above the bound this suite gates parity at, so every one of
    // the sixteen is a divergence this file would actually catch.
    const { assert!(NORMALIZED_MUTATION_FLOOR >= TOL * 10.0) };
    const { assert!(MUTATION_FLOOR > NORMALIZED_MUTATION_FLOOR) };
}

/// The tiled golden depends on the tile geometry: encoding at a different tile size must give a
/// different answer, or the "tiled" test is silently running the untiled path.
#[test]
fn the_tile_geometry_changes_the_result() {
    let f = fixture();
    let v = vae(&f, 3);
    let (tile, overlap) = encode_fixture_tiles(&f);
    let pixels = f.tensor("in.encode_clip.pixels");

    let at_fixture = v.encode_clip_tiled(&pixels, tile, overlap).expect("tiled");
    let untiled = v
        .encode_clip_tiled(&pixels, 4096, overlap)
        .expect("untiled");
    let (delta, _) = rel(&at_fixture, &untiled);
    assert!(
        delta > MUTATION_FLOOR,
        "the tile size is inert ({delta:.3e})"
    );

    // A different tile SIZE is a different plan and so a different result.
    let narrower = tile * 3 / 4;
    let at_narrower = v
        .encode_clip_tiled(&pixels, narrower, overlap)
        .expect("narrower");
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
    // tile count itself has to grow.
    let wider = v
        .encode_clip_tiled(&pixels, tile, overlap * 2)
        .expect("wider overlap");
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

// ---------------------------------------------------------------------------------------------
// Surface
// ---------------------------------------------------------------------------------------------

/// Malformed encode inputs are typed errors, not silent reshapes — **on both tiling settings**,
/// which are two different code paths.
///
/// With tiling on, `encode_clip` goes through `encode_clip_tiled_2d`, which checks the rank up
/// front. With tiling off it goes through `encode_clip_untiled`, which — unlike
/// `decode_clip_untiled` — has **no rank check of its own**, so the rejection comes from
/// `reflect_pad_axis` inside the first causal conv instead. That is still a typed error naming what
/// is wrong, and never a silent wrong answer, so the untiled path does not need a check added; it
/// needs covering, which is what the second half of this test does.
#[test]
fn encode_rejects_malformed_input() {
    use candle_gen_minimax_h3::spatial_tiling::SpatialTiling;
    let f = fixture();
    let v = vae(&f, 3);
    let dev = Device::Cpu;
    let ok = Tensor::zeros((1, 3, 1, 8, 8), DType::F32, &dev).unwrap();
    v.encode(&ok).expect("a well-formed keyframe encodes");

    // Wrong rank.
    let rank1 = Tensor::zeros(1, DType::F32, &dev).unwrap();
    assert!(v.encode(&rank1).is_err());
    assert!(v.encode_clip(&rank1).is_err());
    // Wrong channel count — 4 channels is a plausible RGBA mistake.
    let rgba = Tensor::zeros((1, 4, 1, 8, 8), DType::F32, &dev).unwrap();
    assert!(
        v.encode(&rgba).is_err(),
        "an RGBA-shaped input must be rejected rather than convolved"
    );
    // Zero frames.
    let empty = Tensor::zeros((1, 3, 0, 8, 8), DType::F32, &dev).unwrap();
    assert!(v.encode(&empty).is_err());

    // …and with tiling DISABLED, where `encode_clip` reaches `encode_clip_untiled` and there is no
    // up-front rank check to catch it. The rejection must still be typed and must still name the
    // axis it could not pad, rather than reshaping the tensor into something plausible.
    let off = vae(&f, 3).with_tiling(SpatialTiling::disabled());
    assert!(!off.tiling().enabled);
    off.encode(&ok)
        .expect("a well-formed keyframe still encodes");
    let msg = off
        .encode_clip(&rank1)
        .expect_err("an untiled encode of a rank-1 tensor must be refused")
        .to_string();
    assert!(
        msg.contains("reflect pad axis 3 is outside a rank-1 tensor"),
        "the untiled path must reject it by name, got: {msg}"
    );
    assert!(off.encode(&rgba).is_err());
    assert!(off.encode(&empty).is_err());
}

/// A VAE built from a weight map **without** the encode half decodes but cannot encode, and says
/// so by name rather than panicking or returning noise. The committed decode fixture is exactly
/// such a map, so this is the shape a decode-only caller really holds.
#[test]
fn a_decode_only_weight_map_reports_the_missing_encode_half() {
    let decode = Golden::load(FIXTURE);
    let v = MiniMaxH3VideoVae::from_weights(
        &weights(decode.model_map(&["src.", "in.", "out.", "const."])),
        &common::fixture_config(3),
        &Device::Cpu,
        DType::F32,
    )
    .expect("the decode fixture builds a decode-only VAE");
    assert!(!v.can_encode());

    let pixels = Tensor::zeros((1, 3, 1, 8, 8), DType::F32, &Device::Cpu).unwrap();
    for e in [
        v.encode(&pixels).err(),
        v.encode_clip(&pixels).err(),
        v.encode_clip_tiled(&pixels, 16, 4).err(),
    ] {
        let msg = e
            .expect("encoding without the encode half must fail")
            .to_string();
        assert!(
            msg.contains("encode half") && msg.contains("quant_conv"),
            "the error must name what is missing, got: {msg}"
        );
    }

    // …and the encode fixture's VAE, which does carry it, encodes.
    assert!(vae(&fixture(), 3).can_encode());
}

/// The posterior sample is `mean + std · noise`, and the noise is genuinely load-bearing — a
/// sampler that quietly returned the mean would pass any test that only inspects shapes.
#[test]
fn the_posterior_sample_depends_on_its_noise() {
    let f = fixture();
    let v = vae(&f, 3);
    let posterior = v
        .encode(&f.tensor("in.encode_single.pixels"))
        .expect("encode");
    let shape = posterior.mean().dims().to_vec();
    let n: usize = shape.iter().product();

    let zeros = Tensor::zeros(shape.clone(), DType::F32, &Device::Cpu).unwrap();
    assert_parity(
        &posterior.sample_with(&zeros).expect("sample"),
        posterior.mean(),
        1e-6,
        "zero noise samples the mean",
    );

    let alt: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let noise = Tensor::from_vec(alt, shape, &Device::Cpu).unwrap();
    let sampled = posterior.sample_with(&noise).expect("sample");
    let (delta, _) = rel(&sampled, posterior.mean());
    println!("posterior sample vs mean: peak-rel {delta:.3e}");
    assert!(
        delta > MUTATION_FLOOR,
        "the posterior noise is inert ({delta:.3e}); std must be a real spread"
    );

    // A shape mismatch is rejected rather than broadcast.
    let wrong = Tensor::zeros((1, 1, 1, 1, 1), DType::F32, &Device::Cpu).unwrap();
    assert!(posterior.sample_with(&wrong).is_err());
    let params = Tensor::zeros((1, 2, 1, 1, 1), DType::F32, &Device::Cpu).unwrap();
    DiagonalGaussian::from_parameters(&params).expect("a minimal posterior");
}

/// `normalize` is the exact inverse of `denormalize`. `fl2va` encodes a keyframe and then
/// normalizes it into the DiT's latent space; a port that reached for the usual scalar
/// `scaling_factor` — which this VAE does not have — would round-trip wrong.
#[test]
fn normalize_inverts_denormalize_over_the_real_statistics() {
    let f = fixture();
    let v = vae(&f, 3);
    let cfg = v.config().clone();
    let n = cfg.latent_channels * 2 * 3 * 3;
    let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.41).sin() * 1.7).collect();
    let z = Tensor::from_vec(vals, (1, cfg.latent_channels, 2, 3, 3), &Device::Cpu).unwrap();

    let round_trip = v
        .normalize(&v.denormalize(&z).expect("denorm"))
        .expect("norm");
    assert_parity(&round_trip, &z, 1e-6, "normalize ∘ denormalize");

    // And it is not the identity: the statistics really are applied.
    let (delta, _) = rel(&v.denormalize(&z).expect("denorm"), &z);
    println!("denormalize moves the latent by peak-rel {delta:.3e}");
    assert!(
        delta > MUTATION_FLOOR,
        "the 24-entry statistics are inert ({delta:.3e})"
    );
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

/// **The `enable_tiling` / `disable_tiling` knobs reach the ENCODE half**, not only the decode one.
///
/// Upstream has a single `use_tiling` and a single set of `tile_sample_min_*` shared by both
/// halves, so a caller that retunes tiling for decode retunes the `fl2va` conditioning encode with
/// it. The decode twin of this test lives in `video_vae_parity.rs`, whose fixture carries no
/// encoder — so this is the only place the encode side of that sharing is observable, and without
/// it `encode_clip` could keep reading the shipped constants directly while `decode_clip` follows
/// the knobs, which compiles and leaves every other assertion in both suites passing.
#[test]
fn the_tiling_knobs_select_the_paths_they_name_on_the_encode_half() {
    use candle_gen_minimax_h3::spatial_tiling::SpatialTiling;
    let f = fixture();
    let pixels = f.tensor("in.encode_clip.pixels");
    let (tile, overlap) = encode_fixture_tiles(&f);
    let base = vae(&f, 3);

    let untiled = flat(&base.encode_clip_untiled(&pixels).expect("untiled"));
    let tiled = flat(
        &base
            .encode_clip_tiled(&pixels, tile, overlap)
            .expect("tiled"),
    );
    assert_ne!(
        tiled, untiled,
        "the two reference paths must differ or this test proves nothing"
    );

    // `enable_tiling` at the fixture geometry routes `encode_clip` to the tiled path, exactly.
    let mut on = vae(&f, 3);
    on.enable_tiling(Some(tile), Some(tile), Some(overlap), Some(overlap));
    assert_eq!(flat(&on.encode_clip(&pixels).expect("on")), tiled);

    // `disable_tiling` routes it to the untiled path even at a geometry that would tile.
    let mut off = vae(&f, 3);
    off.enable_tiling(Some(tile), Some(tile), Some(overlap), Some(overlap));
    off.disable_tiling();
    assert!(!off.tiling().enabled);
    assert_eq!(flat(&off.encode_clip(&pixels).expect("off")), untiled);

    // …and the builder form agrees with the setters.
    let built = vae(&f, 3).with_tiling(SpatialTiling::disabled());
    assert_eq!(flat(&built.encode_clip(&pixels).expect("built")), untiled);
    let built_on = vae(&f, 3).with_tiling(SpatialTiling::square(tile, overlap));
    assert_eq!(
        flat(&built_on.encode_clip(&pixels).expect("built_on")),
        tiled
    );

    // The shipped default tiles at 256/64, which this sub-tile fixture canvas cannot span — so the
    // default must agree with the untiled path, exactly as it does for decode.
    assert!(f.shape("in.encode_clip.pixels")[3] < TILE_SAMPLE_MIN_SIZE);
    assert_eq!(flat(&base.encode_clip(&pixels).expect("default")), untiled);
}

/// **The latent-alignment guard is reachable from the ENCODE half's public API**, not merely from
/// decode's — the twin of `video_vae_parity.rs::a_tile_geometry_the_latent_grid_cannot_express_is_
/// refused`.
///
/// This is a **new error surface on this side**. The encode lane used to carry its own `TilePlan`
/// with no alignment clause, and `encode_clip` used to pass the shipped constants rather than
/// `self.tiling`; sharing `spatial_tiling::TilePlan::split` and reading the knobs is what puts
/// `enable_tiling` in front of this guard for encode. The decode twin gates only its own half — the
/// decode fixture carries no encoder — so without this test the clause could be neutered on the
/// encode path alone and every other assertion in both suites would still pass.
///
/// The encode fixture's ratio is **4**, not the decode fixture's 2, so the geometries refused here
/// are the ones 4 refuses.
#[test]
fn an_encode_tile_geometry_the_latent_grid_cannot_express_is_refused() {
    let f = fixture();
    let pixels = f.tensor("in.encode_clip.pixels");
    let ratio = encode_fixture_config(3).patch_size;
    assert_eq!(
        ratio, 4,
        "this test's odd tile sizes are only misaligned at ratio 4"
    );

    // A misaligned TILE, at an aligned overlap.
    let err = vae(&f, 3)
        .encode_clip_tiled(&pixels, 6, 4)
        .expect_err("a 6 px tile is 1.5 latent cells and must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains(&format!("{ratio}x spatial compression ratio")),
        "{msg}"
    );
    assert!(
        msg.contains("silently truncated"),
        "the error must say WHY: {msg}"
    );

    // A misaligned OVERLAP, at an aligned tile. Asserted on the message, not merely on `is_err`:
    // a misaligned plan that gets built usually *does* fail somewhere downstream, so a bare
    // `is_err()` here passes with the clause deleted and gates nothing (measured — it survived the
    // mutation).
    let msg = vae(&f, 3)
        .encode_clip_tiled(&pixels, 8, 2)
        .expect_err("a 2 px overlap is half a latent cell and must be refused")
        .to_string();
    assert!(
        msg.contains(&format!("{ratio}x spatial compression ratio")),
        "the overlap must be refused BY THE GUARD, not blow up downstream: {msg}"
    );

    // …and it reaches through `enable_tiling` / `encode_clip`, which is the surface a caller uses,
    // and the surface `fl2va` keyframe conditioning goes through.
    let mut misaligned = vae(&f, 3);
    misaligned.enable_tiling(Some(3), Some(3), Some(2), Some(2));
    let msg = misaligned
        .encode_clip(&pixels)
        .expect_err("enable_tiling must not be able to smuggle a misaligned plan into encode")
        .to_string();
    assert!(
        msg.contains(&format!("{ratio}x spatial compression ratio")),
        "{msg}"
    );

    // **The guard binds BEFORE the sub-tile short circuit.** `TilePlan::split` returns one span
    // covering the axis whenever `tile_size >= length`, so on a canvas no larger than one tile a
    // misaligned geometry never reaches the division it would corrupt — it is refused anyway. That
    // is a behaviour change on this public surface rather than pure strengthening, and it is
    // deliberate: it is what decode has done since sc-18786.
    let (h, w) = (
        f.shape("in.encode_clip.pixels")[3],
        f.shape("in.encode_clip.pixels")[4],
    );
    let sub_tile = h.max(w) * ratio + 1;
    assert!(
        sub_tile > h.max(w) && !sub_tile.is_multiple_of(ratio),
        "the probe must be both misaligned and larger than the {h}x{w} canvas"
    );
    let msg = vae(&f, 3)
        .encode_clip_tiled(&pixels, sub_tile, 0)
        .expect_err("a misaligned tile is refused even where tiling would be inert")
        .to_string();
    assert!(
        msg.contains(&format!("{ratio}x spatial compression ratio")),
        "a {sub_tile} px tile over a {h}x{w} canvas must be refused by the guard, not \
         short-circuited past it: {msg}"
    );

    // The aligned geometry still encodes, so the guard refuses the misalignment and nothing else.
    let (tile, overlap) = encode_fixture_tiles(&f);
    assert!(vae(&f, 3).encode_clip_tiled(&pixels, tile, overlap).is_ok());
}
