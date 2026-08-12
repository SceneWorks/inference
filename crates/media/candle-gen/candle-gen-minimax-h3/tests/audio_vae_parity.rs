//! sc-17154: candle audio-VAE decode parity against the Apache-2.0 reference implementation.
//!
//! Fixture `tests/fixtures/audio_vae_decode.safetensors` ← the MLX lane's
//! `tools/dump_minimax_h3_audio_vae.py`, which imports the reference bundle shipped INSIDE the
//! MiniMax-H3 snapshot (`FL2VA/audio_vae/{dac_audio_vae,dac_bigvgan,dac_activations,
//! dac_alias_free_*}.py`) and runs its real `DacAudioVAE.decode` at tiny dims. A genuine
//! independent-graph parity check, not a re-derivation from config: the fixture's decode-half
//! tensor NAMES are byte-identical to the published checkpoint's 914 (see `tests/real_weights.rs`).
//!
//! The fixture stores the reference's **NCL** tensors, and this port is NCL throughout, so unlike
//! the MLX twin nothing here transposes on the way in. That is the largest deliberate
//! implementation difference between the two lanes and is what makes their agreement
//! (`cross_backend.rs`) evidence about the model rather than about one shared array layout.
//!
//! ## Why the sub-fixtures exist
//!
//! `SnakeBeta` and the Kaiser-sinc resamplers carry this port's numerical risk — a periodic
//! activation and a filtered resampler are both easy to get subtly wrong in a way an end-to-end
//! tolerance hides, and the decoder applies 127 of them. Each therefore has its own golden at its
//! own tolerance, above the whole-model one.
//!
//! ## Tolerances
//!
//! Every bound here is set from the **measured** residual on this lane, not inherited from the MLX
//! sibling's. candle runs f32 on the CPU, where MLX pays for Metal's reduced-precision f32 matmul;
//! the MLX suite therefore documents 1.2e-3 for one AMP block and 2.7e-3 … 4.4e-3 for the whole
//! decode, while this one measures four to five orders tighter. Reusing its 2e-2 here would have
//! left every gate far above its own noise floor. Each test prints its residual, so the real margin
//! stays auditable rather than implied.

mod common;

use std::collections::BTreeSet;

use common::{
    assert_parity, audio_fixture_config, flat, rel, std_dev, weights, Golden, AUDIO_FIXTURE,
};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen_minimax_h3::audio_vae::{AmpBlock1, BigVgan};
use candle_gen_minimax_h3::{
    kaiser_sinc_filter1d, Activation1d, LowPassFilter1d, MiniMaxH3AudioVae,
    MiniMaxH3AudioVaeConfig, SnakeBeta, UpSample1d, AUDIO_VAE_IS_UNCONVERTED,
};

/// Whole-decoder parity. Measured **2.5e-6 … 3.3e-6** through `BigVGAN`, `decode`, `decode_stereo`
/// and the interleave — seven convolutional upsample stages and 127 anti-aliased activations of
/// accumulated f32 round-off. ~30× headroom.
const TOL: f32 = 1e-4;
/// One AMP block, measured **2.0e-7**: multi-channel convolutions but no 21-block amplification.
const BLOCK_TOL: f32 = 1e-5;
/// Sub-module parity for the pieces that are elementwise or single-channel — the Kaiser filter,
/// `SnakeBeta`, the resamplers, `Activation1d`. Those run no matmul, so candle reproduces the
/// reference to **0 … 3.1e-7** and a loose bound here would hide exactly the errors these fixtures
/// exist to catch.
const UNIT_TOL: f32 = 1e-5;
/// Mutation and "this knob is load-bearing" probes must clear the whole-decoder gate by two orders,
/// or "the output moved" could be numerical jitter rather than a wiring difference.
const MUTATION_FLOOR: f32 = 1e-2;

/// The reference-side extras plus the standalone AMP block's own copy of its weights.
const NON_MODEL_PREFIXES: [&str; 4] = ["in.", "out.", "const.", "amp."];

fn fixture() -> Golden {
    Golden::load(AUDIO_FIXTURE)
}

fn model_map(f: &Golden) -> std::collections::HashMap<String, Tensor> {
    f.model_map(&NON_MODEL_PREFIXES)
}

fn vae_from(map: std::collections::HashMap<String, Tensor>) -> MiniMaxH3AudioVae {
    vae_with(map, &audio_fixture_config())
}

fn vae_with(
    map: std::collections::HashMap<String, Tensor>,
    cfg: &MiniMaxH3AudioVaeConfig,
) -> MiniMaxH3AudioVae {
    MiniMaxH3AudioVae::from_weights(&weights(map), cfg, &Device::Cpu, DType::F32)
        .expect("build the audio VAE")
}

fn vae(f: &Golden) -> MiniMaxH3AudioVae {
    vae_from(model_map(f))
}

// ---------------------------------------------------------------------------------------------
// Kaiser-sinc filter derivation — targeted fixture #1
// ---------------------------------------------------------------------------------------------

/// The four dumped cases walk the whole branch table of
/// `dac_alias_free_filter.py::kaiser_sinc_filter1d`: the shipped 12-tap / ratio-2 filter
/// (`A > 50`), a wider ratio, an ODD kernel size (a different `time` grid), and a short kernel
/// whose attenuation estimate falls under 21 so `beta` is zero.
///
/// A port that hardcoded the shipped filter, or implemented only the `A > 50` branch, or used the
/// even-kernel `time` grid unconditionally, fails here while still passing an end-to-end golden
/// (the checkpoint ships its filters as buffers, so the model never calls this on the load path).
#[test]
fn kaiser_sinc_filter_matches_the_reference() {
    let f = fixture();
    for tag in ["r2k12", "r4k24", "odd11", "beta0"] {
        let params = f.f32(&format!("const.kaiser.{tag}"));
        let (cutoff, half_width, ksize) = (params[0] as f64, params[1] as f64, params[2] as usize);
        let got = kaiser_sinc_filter1d(cutoff, half_width, ksize, &Device::Cpu).expect("filter");
        assert_parity(
            &got,
            &f.tensor(&format!("out.kaiser.{tag}")),
            UNIT_TOL,
            &format!("kaiser_sinc_filter1d({tag})"),
        );
    }

    // The four cases really are different filters, so passing all four is not one assertion in
    // disguise.
    assert_ne!(f.shape("out.kaiser.r2k12"), f.shape("out.kaiser.odd11"));
}

/// The taps the checkpoint STORES are reproducible from the derivation. The loader reads the stored
/// buffers (that is what the reference's `register_buffer` + strict load does), so without this the
/// derivation would be untested against the model's own filters.
#[test]
fn stored_filters_match_the_derivation() {
    let f = fixture();
    let derived = kaiser_sinc_filter1d(0.5 / 2.0, 0.6 / 2.0, 12, &Device::Cpu).expect("filter");
    for key in [
        "const.resample.up_filter",
        "const.resample.down_filter",
        "decoder.activation_post.upsample.filter",
        "decoder.activation_post.downsample.lowpass.filter",
        "decoder.resblocks.0.activations.0.upsample.filter",
        "decoder.resblocks.20.activations.5.downsample.lowpass.filter",
    ] {
        assert_parity(&derived, &f.tensor(key), UNIT_TOL, key);
    }
}

// ---------------------------------------------------------------------------------------------
// Alias-free resamplers — targeted fixture #2
// ---------------------------------------------------------------------------------------------

/// `UpSample1d`, `DownSample1d` and the round trip the anti-aliased activation performs.
///
/// The upsample is where the reference's asymmetric bookkeeping lives: replicate-pad by
/// `taps/ratio − 1`, zero-insert transposed convolution scaled by `ratio`, then trim
/// `pad·ratio + (taps − ratio)/2` from the left and `pad·ratio + (taps − ratio + 1)/2` from the
/// right. Those two trims differ by one for an odd `taps − ratio`, and swapping them shifts the
/// whole waveform by a sample.
#[test]
fn resamplers_match_the_reference() {
    let f = fixture();
    let filter = f.tensor("const.resample.up_filter");
    let up = UpSample1d::from_filter(filter.clone(), 2).expect("up");
    let down = LowPassFilter1d::from_filter(filter, 2).expect("down");
    let x = f.tensor("in.resample.x");

    let got_up = up.forward(&x).expect("upsample");
    assert_parity(
        &got_up,
        &f.tensor("out.resample.up"),
        UNIT_TOL,
        "UpSample1d",
    );

    let got_down = down.forward(&x).expect("downsample");
    assert_parity(
        &got_down,
        &f.tensor("out.resample.down"),
        UNIT_TOL,
        "DownSample1d",
    );

    let round_trip = down.forward(&got_up).expect("round trip");
    assert_parity(
        &round_trip,
        &f.tensor("out.resample.up_down"),
        UNIT_TOL,
        "UpSample1d -> DownSample1d",
    );
}

/// A shifted or mis-scaled upsample would still round-trip to the right SHAPE, so pin the actual
/// samples: the reference's `x = ratio · conv_transpose(...)` scaling and its trim offsets.
#[test]
fn upsample_matches_the_reference_sample_for_sample() {
    let f = fixture();
    let filter = f.tensor("const.resample.up_filter");
    let up = UpSample1d::from_filter(filter, 2).expect("up");
    let got = flat(&up.forward(&f.tensor("in.resample.x")).expect("upsample"));
    let want = flat(&f.tensor("out.resample.up"));
    assert_eq!(got.len(), want.len());
    // A one-sample shift is the failure mode the tolerance alone would not name.
    let shifted: f32 = got
        .iter()
        .skip(3)
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let aligned: f32 = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  upsample aligned {aligned:.3e}, shifted-by-one-frame {shifted:.3e}");
    assert!(aligned < 1e-5, "upsample differs by {aligned:.3e}");
    assert!(
        shifted > aligned * 100.0,
        "a shifted comparison is just as good ({shifted:.3e}); the test would not detect an \
         off-by-one"
    );
}

// ---------------------------------------------------------------------------------------------
// SnakeBeta — targeted fixture #3
// ---------------------------------------------------------------------------------------------

/// `x + sin²(α·x) / (β + 1e-9)` with log-scale α/β, and the proof that the log scale matters.
#[test]
fn snakebeta_matches_the_reference() {
    let f = fixture();
    let x = f.tensor("in.snake.x");
    let alpha = f.tensor("in.snake.alpha");
    let beta = f.tensor("in.snake.beta");

    for (tag, logscale) in [("log", true), ("linear", false)] {
        let got = SnakeBeta::new(alpha.clone(), beta.clone(), logscale)
            .expect("snakebeta")
            .forward(&x)
            .expect("forward");
        assert_parity(
            &got,
            &f.tensor(&format!("out.snake.{tag}")),
            UNIT_TOL,
            &format!("SnakeBeta(alpha_logscale={logscale})"),
        );
    }

    // `snake_logscale = true` appears in NO config file — it is selected by the reference's
    // `sample_rate` branch. If the two modes agreed, the fixture could not police it.
    let (gap, _) = rel(&f.tensor("out.snake.log"), &f.tensor("out.snake.linear"));
    println!("  log vs linear snake scale: peak rel {gap:.3e}");
    assert!(
        gap > MUTATION_FLOOR,
        "the two snake scales agree (rel {gap:.3e}); the golden cannot pin snake_logscale"
    );
}

/// The composed `Activation1d`: 2× up → SnakeBeta → 2× down.
#[test]
fn activation1d_matches_the_reference() {
    let f = fixture();
    let filter = f.tensor("const.resample.up_filter");
    let act = Activation1d::new(
        SnakeBeta::new(f.tensor("in.act1d.alpha"), f.tensor("in.act1d.beta"), true)
            .expect("snakebeta"),
        UpSample1d::from_filter(filter.clone(), 2).expect("up"),
        LowPassFilter1d::from_filter(filter, 2).expect("down"),
    );
    let got = act.forward(&f.tensor("in.act1d.x")).expect("forward");
    assert_parity(&got, &f.tensor("out.act1d.y"), UNIT_TOL, "Activation1d");
}

// ---------------------------------------------------------------------------------------------
// Blocks and the whole decoder
// ---------------------------------------------------------------------------------------------

/// One `AMPBlock1` in isolation: three dilated conv pairs, six anti-aliased activations paired
/// `activations[::2]` / `activations[1::2]`, residual after each pair.
#[test]
fn amp_block_matches_the_reference() {
    let f = fixture();
    let block = AmpBlock1::from_weights(
        &weights(f.prefixed_map("amp.")),
        "amp",
        7,
        &[1, 3, 5],
        true,
        DType::F32,
    )
    .expect("amp block");
    let got = block.forward(&f.tensor("in.amp.x")).expect("forward");
    assert_parity(&got, &f.tensor("out.amp.y"), BLOCK_TOL, "AMPBlock1");
}

/// The activation pairing is observable: swapping `activations.0` with `activations.1` moves the
/// block's output. A port that paired them `0,1,2` / `3,4,5` loads every tensor and looks correct.
#[test]
fn amp_block_activation_pairing_is_load_bearing() {
    let f = fixture();
    let x = f.tensor("in.amp.x");
    let baseline = AmpBlock1::from_weights(
        &weights(f.prefixed_map("amp.")),
        "amp",
        7,
        &[1, 3, 5],
        true,
        DType::F32,
    )
    .expect("amp block")
    .forward(&x)
    .expect("forward");

    let mut map = f.prefixed_map("amp.");
    for leaf in ["act.alpha", "act.beta"] {
        let a = map[&format!("amp.activations.0.{leaf}")].clone();
        let b = map[&format!("amp.activations.1.{leaf}")].clone();
        map.insert(format!("amp.activations.0.{leaf}"), b);
        map.insert(format!("amp.activations.1.{leaf}"), a);
    }
    let swapped = AmpBlock1::from_weights(&weights(map), "amp", 7, &[1, 3, 5], true, DType::F32)
        .expect("amp block")
        .forward(&x)
        .expect("forward");
    let (peak, _) = rel(&swapped, &baseline);
    println!("  swapping activations 0<->1 moved the block by {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "swapping two activations did not change the block (rel {peak:.3e}); the pairing is not \
         being exercised"
    );
}

/// The BigVGAN vocoder alone: `conv_pre`, 7 upsample stages each averaging 3 AMP blocks,
/// `activation_post`, `conv_post`, and the final CLAMP (not tanh).
#[test]
fn bigvgan_matches_the_reference() {
    let f = fixture();
    let cfg = audio_fixture_config();
    let decoder = BigVgan::from_weights(&weights(model_map(&f)), "decoder", &cfg, DType::F32)
        .expect("vocoder");
    let got = decoder.forward(&f.tensor("in.bigvgan.x")).expect("forward");
    // 4 tokens x the real 800x hop.
    assert_eq!(got.dims(), &[1, 1, 3200]);
    assert_parity(&got, &f.tensor("out.bigvgan.y"), TOL, "BigVGAN");
}

/// `DacAudioVAE.decode` = `dec_in_proj` → BigVGAN. Reference-exact: no de-normalization.
#[test]
fn decode_matches_the_reference() {
    let f = fixture();
    let got = vae(&f).decode(&f.tensor("in.decode.z")).expect("decode");
    assert_eq!(
        got.dims(),
        &[1, 1, 3200],
        "4 tokens at 40 Hz -> 0.1 s @ 32 kHz"
    );
    assert_parity(
        &got,
        &f.tensor("out.decode.audio"),
        TOL,
        "DacAudioVAE.decode",
    );
}

/// De-normalization, stereo folding and the interleave, each against its own golden.
#[test]
fn stereo_decode_matches_the_reference() {
    let f = fixture();
    let vae = vae(&f);
    let z = f.tensor("in.stereo.z");

    let denorm = vae.denormalize(&z).expect("denormalize");
    assert_parity(
        &denorm,
        &f.tensor("out.stereo.denorm"),
        UNIT_TOL,
        "latent de-normalization",
    );

    let stereo = vae.decode_stereo(&denorm).expect("decode_stereo");
    assert_eq!(stereo.dims(), &[1, 2, 3200]);
    assert_parity(&stereo, &f.tensor("out.stereo.audio"), TOL, "decode_stereo");

    let track = vae.decode_audio_track(&z).expect("audio track");
    let got = Tensor::from_vec(track.samples.clone(), track.samples.len(), &Device::Cpu)
        .expect("track tensor");
    assert_parity(
        &got,
        &f.tensor("out.stereo.interleaved"),
        TOL,
        "AudioTrack interleave",
    );
}

/// The `gen-core` output contract: 32 kHz, 2 channels, interleaved f32, no stems.
#[test]
fn audio_track_is_the_gen_core_contract() {
    let f = fixture();
    let track = vae(&f)
        .decode_audio_track(&f.tensor("in.stereo.z"))
        .expect("audio track");
    assert_eq!(track.sample_rate, 32_000);
    assert_eq!(track.channels, 2);
    assert!(track.stems.is_empty(), "this model emits a mix, not stems");
    // 4 latent tokens · 800 samples · 2 channels.
    assert_eq!(track.samples.len(), 4 * 800 * 2);
    assert!(track.samples.iter().all(|s| s.is_finite()));
    assert!(track.samples.iter().all(|s| (-1.0..=1.0).contains(s)));
}

/// The interleave is `L0, R0, L1, R1, …` — and the two channels are GENUINELY different, so a
/// mono-duplicating port cannot pass.
#[test]
fn stereo_channels_are_independent_and_interleaved() {
    let f = fixture();
    let vae = vae(&f);
    let z = f.tensor("in.stereo.z");
    let stereo = vae
        .decode_stereo(&vae.denormalize(&z).expect("denormalize"))
        .expect("decode_stereo");
    let planar = flat(&stereo);
    let (left, right) = planar.split_at(3200);

    let gap = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let peak = right.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("  L-vs-R peak gap {:.3e} (peak |R| {peak:.3e})", gap / peak);
    assert!(
        gap / peak > MUTATION_FLOOR,
        "the two decoded channels are near-identical; a mono-duplicating port would pass"
    );

    let track = vae.decode_audio_track(&z).expect("audio track");
    for t in 0..3200 {
        assert_eq!(
            track.samples[2 * t],
            left[t],
            "even samples are the left channel"
        );
        assert_eq!(
            track.samples[2 * t + 1],
            right[t],
            "odd samples are the right channel"
        );
    }

    // Swapping the two input channels swaps the output channels — the fold really is per-channel
    // and not, say, a reshape that mixes them.
    let idx = Tensor::from_vec(vec![1u32, 0], 2, &Device::Cpu).expect("idx");
    let swapped_z = z.index_select(&idx, 1).expect("swap channels");
    let swapped = flat(
        &vae.decode_stereo(&vae.denormalize(&swapped_z).expect("denormalize"))
            .expect("decode_stereo"),
    );
    let (sl, sr) = swapped.split_at(3200);
    assert_eq!(sl, right, "swapping inputs must swap outputs exactly");
    assert_eq!(sr, left);
}

/// A latent packed `[B, latent_channels, output_channels, T]` — the plausible transposition — is
/// rejected rather than decoded into nonsense, and so is a batch the interleave cannot represent.
#[test]
fn mis_packed_latents_are_rejected() {
    let f = fixture();
    let vae = vae(&f);
    let z = f.tensor("in.stereo.z");
    let transposed = z.permute((0, 2, 1, 3)).expect("transpose");
    assert!(vae.decode_stereo(&transposed).is_err());
    // The mono entry point rejects a stereo-ranked latent too.
    assert!(vae.decode(&z).is_err());
    // B > 1 cannot be interleaved into one track.
    let two = Tensor::cat(&[&z, &z], 0).expect("batch of two");
    assert!(vae.decode_audio_track(&two).is_err());
    assert!(
        vae.decode_stereo(&two).is_ok(),
        "but decode_stereo batches fine"
    );
}

// ---------------------------------------------------------------------------------------------
// Weight mapping
// ---------------------------------------------------------------------------------------------

/// The declared name list must be exactly the checkpoint's, and the load must require all of it.
/// A silently unmapped tensor still produces plausible-looking audio.
#[test]
fn weight_mapping_is_exhaustive() {
    let f = fixture();
    let cfg = audio_fixture_config();
    let present: BTreeSet<String> = model_map(&f).keys().cloned().collect();
    assert_eq!(
        present.len(),
        914,
        "the fixture carries the decode half only"
    );

    let declared: BTreeSet<String> = MiniMaxH3AudioVae::tensor_names(&cfg).into_iter().collect();
    assert_eq!(
        declared, present,
        "declared tensor names differ from the checkpoint's"
    );

    let _vae = vae(&f);
    let mut short = model_map(&f);
    short.remove("decoder.resblocks.13.convs2.1.weight_v");
    assert!(
        MiniMaxH3AudioVae::from_weights(&weights(short), &cfg, &Device::Cpu, DType::F32).is_err(),
        "dropping a declared tensor must be a load error, not a silent default"
    );
}

/// A checkpoint whose `dec_in_proj` disagrees with the config is a loud error, not a silent
/// re-interpretation.
#[test]
fn a_mismatched_checkpoint_is_rejected() {
    let f = fixture();
    let mut cfg = audio_fixture_config();
    cfg.latent_channels = 16;
    cfg.latents_mean.truncate(16);
    cfg.latents_std.truncate(16);
    assert!(MiniMaxH3AudioVae::from_weights(
        &weights(model_map(&f)),
        &cfg,
        &Device::Cpu,
        DType::F32
    )
    .is_err());
}

// ---------------------------------------------------------------------------------------------
// False-green guards
// ---------------------------------------------------------------------------------------------

/// A constant, all-zero or fully-saturated golden is a false green.
///
/// `SnakeBeta` initializes `alpha`/`beta` to ONES, which under the log scale means `exp(1)` for
/// both and makes the two parameters indistinguishable; the decoder's final CLAMP means a golden
/// dumped at a large scale would saturate and be reproduced by any port that also saturates.
#[test]
fn fixture_is_not_degenerate() {
    let f = fixture();
    for key in [
        "out.decode.audio",
        "out.bigvgan.y",
        "out.stereo.audio",
        "out.amp.y",
        "out.act1d.y",
        "out.snake.log",
        "out.resample.up",
    ] {
        assert!(
            std_dev(&f.tensor(key)) > 1e-4,
            "{key} is ~constant; a constant golden is a false green"
        );
    }

    // alpha and beta must differ, or the golden cannot tell them apart. (`SnakeBeta` initializes
    // BOTH to ones, which under the log scale makes them literally interchangeable.)
    for key in [
        "decoder.activation_post",
        "decoder.resblocks.0.activations.0",
        "decoder.resblocks.10.activations.3",
    ] {
        let a = f.tensor(&format!("{key}.act.alpha"));
        let b = f.tensor(&format!("{key}.act.beta"));
        let (gap, _) = rel(&a, &b);
        assert!(gap > 1e-2, "{key}: alpha and beta are indistinguishable");
        // `activation_post` is a single channel at this width (128 / 2^7 = 1), so per-channel
        // spread is only meaningful for the wider blocks.
        if a.elem_count() > 1 {
            assert!(
                std_dev(&a) > 1e-3,
                "{key}: alpha is ~constant across channels"
            );
        }
    }

    // The decode must not sit on the clamp.
    let audio = flat(&f.tensor("out.stereo.audio"));
    let saturated = audio.iter().filter(|s| s.abs() >= 1.0 - 1e-6).count();
    let fraction = saturated as f32 / audio.len() as f32;
    println!("  golden audio saturation {:.3}%", fraction * 100.0);
    assert!(
        fraction < 0.05,
        "{:.1}% of the golden is clamped at ±1; any port that saturates would reproduce it",
        fraction * 100.0
    );
}

/// Mutation check: perturbing any single weight must move the decode. If a tensor can be changed
/// without changing the output, it is not wired into the graph and the parity test is not
/// covering it.
#[test]
fn every_weight_is_load_bearing() {
    let f = fixture();
    let z = f.tensor("in.decode.z");
    let baseline = vae(&f).decode(&z).expect("baseline");

    // One representative of every distinct tensor role in the decode path — including both halves
    // of a weight-norm pair (a port that read only `weight_v` would decode plausibly), the stored
    // Kaiser filters, and both ends of the 7-stage stack.
    let probes: [&str; 17] = [
        "dec_in_proj.weight",
        "dec_in_proj.bias",
        "decoder.conv_pre.weight_g",
        "decoder.conv_pre.weight_v",
        "decoder.conv_pre.bias",
        "decoder.ups.0.0.weight_g",
        "decoder.ups.0.0.weight_v",
        "decoder.ups.0.0.bias",
        "decoder.ups.6.0.weight_v",
        "decoder.resblocks.0.convs1.0.weight_v",
        "decoder.resblocks.0.convs2.2.bias",
        "decoder.resblocks.0.activations.0.act.alpha",
        "decoder.resblocks.0.activations.1.act.beta",
        // A mid-stack block (stage 4) and the last one (stage 6), so the probe set spans the
        // whole 7-stage stack rather than only its ends.
        "decoder.resblocks.12.convs1.1.weight_g",
        "decoder.resblocks.20.convs1.1.weight_v",
        "decoder.activation_post.act.alpha",
        "decoder.conv_post.weight_v",
    ];

    for key in probes {
        let mut map = model_map(&f);
        let original = map[key].clone();
        // Scale AND shift: the shift makes an all-zero tensor observable, the scale keeps the
        // perturbation proportionate for tensors that are already large.
        let bumped = ((original * 1.3).expect("scale") + 0.2).expect("shift");
        map.insert(key.to_string(), bumped);
        let mutated = vae_from(map).decode(&z).expect("mutated decode");
        let (peak, _) = rel(&mutated, &baseline);
        println!("  {key}: peak rel {peak:.3e} (floor {MUTATION_FLOOR:.1e})");
        assert!(
            peak > MUTATION_FLOOR,
            "perturbing {key} moved the decode by only {peak:.3e}, under the \
             {MUTATION_FLOOR:.1e} floor — it is not wired into the graph"
        );
    }

    // The stored Kaiser filters are read, not re-derived, so they must be load-bearing too.
    let mut map = model_map(&f);
    for suffix in ["upsample.filter", "downsample.lowpass.filter"] {
        let key = format!("decoder.resblocks.0.activations.0.{suffix}");
        let flat_filter = kaiser_sinc_filter1d(0.4, 0.2, 12, &Device::Cpu).expect("filter");
        map.insert(key, flat_filter);
    }
    let mutated = vae_from(map).decode(&z).expect("mutated decode");
    let (peak, _) = rel(&mutated, &baseline);
    println!("  stored Kaiser filters: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "replacing a stored resampler filter did not change the decode ({peak:.3e}); the buffers \
         are not being used"
    );
}

/// The three `sample_rate`-branch knobs that appear in no config file and leave no tensor behind
/// must each change the decode — otherwise nothing in this crate would notice a port that took
/// BigVGAN's upstream defaults.
#[test]
fn config_only_knobs_are_load_bearing() {
    let f = fixture();
    let z = f.tensor("in.decode.z");
    let baseline = vae(&f).decode(&z).expect("baseline");

    let build =
        |cfg: &MiniMaxH3AudioVaeConfig| vae_with(model_map(&f), cfg).decode(&z).expect("decode");

    // snake_logscale = false: alpha/beta used raw instead of exp(·).
    let mut cfg = audio_fixture_config();
    cfg.bigvgan.snake_logscale = false;
    let (peak, _) = rel(&build(&cfg), &baseline);
    println!("  snake_logscale=false: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "snake_logscale is inert ({peak:.3e})"
    );

    // use_tanh_at_final = true: tanh instead of the hard clamp.
    let mut cfg = audio_fixture_config();
    cfg.bigvgan.use_tanh_at_final = true;
    let (peak, _) = rel(&build(&cfg), &baseline);
    println!("  use_tanh_at_final=true: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "use_tanh_at_final is inert ({peak:.3e})"
    );

    // use_bias_at_final = true would demand a `conv_post.bias` the checkpoint does not have.
    let mut cfg = audio_fixture_config();
    cfg.bigvgan.use_bias_at_final = true;
    assert!(MiniMaxH3AudioVae::from_weights(
        &weights(model_map(&f)),
        &cfg,
        &Device::Cpu,
        DType::F32
    )
    .is_err());
}

/// De-normalization is APPLIED, and applied per channel — the pipeline path must differ from the
/// reference-exact one, and the closed form must hold element-wise.
#[test]
fn latent_denormalization_is_applied_per_channel() {
    let f = fixture();
    let vae = vae(&f);
    let z = f.tensor("in.stereo.z");

    let denorm = vae.denormalize(&z).expect("denormalize");
    let (peak, _) = rel(&denorm, &z);
    assert!(peak > 1e-2, "de-normalization was a no-op ({peak:.3e})");

    let raw = flat(&z);
    let got = flat(&denorm);
    let cfg = MiniMaxH3AudioVaeConfig::default();
    // [1, 2, 32, T] with T = 4: element (0, 0, c, t) is at index c·4 + t.
    for c in [0usize, 5, 31] {
        let idx = c * 4 + 2;
        let expect = raw[idx] * cfg.latents_std[c] + cfg.latents_mean[c];
        assert!(
            (got[idx] - expect).abs() < 1e-5,
            "channel {c}: expected z·std + mean = {expect}, got {}",
            got[idx]
        );
    }

    // Skipping de-normalization changes the audio.
    let with = vae.decode_audio_track(&z).expect("track");
    let without = vae.decode_stereo(&z).expect("raw decode");
    let a = Tensor::from_vec(with.samples.clone(), with.samples.len(), &Device::Cpu)
        .expect("track tensor");
    let b = without
        .reshape((2, 3200))
        .expect("reshape")
        .t()
        .expect("transpose")
        .contiguous()
        .expect("contiguous")
        .reshape(6400)
        .expect("flatten");
    let (peak, _) = rel(&a, &b);
    println!("  denormalized vs raw decode: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "decode ignored latents_mean/std ({peak:.3e})"
    );
}

// ---------------------------------------------------------------------------------------------
// sc-18740 — the audio VAE's audit result, pinned
// ---------------------------------------------------------------------------------------------

/// **The audio VAE carries no conversion transform, and has no fused gated projection to
/// mis-read.**
///
/// sc-18740 asked for this to be confirmed by evidence rather than assumed. Three independent
/// checks, all against upstream sources:
///
/// 1. `convert_minimax_h3_to_diffusers.py::convert_audio_vae` states "the mapping is an identity:
///    `AutoencoderKLMiniMaxH3Audio` reproduces the original module tree name for name". It renames
///    nothing, transforms nothing, and raises `KeyError` on any key the freshly-built model does
///    not declare — so a silent addition is impossible, not merely unlikely.
/// 2. `AutoencoderKLMiniMaxH3Audio` contains no `SwiGLU` / `GEGLU` / `FeedForward` / `chunk(2, …)`
///    construct anywhere. There is no gated projection in the architecture at all.
/// 3. Of the 1087 published tensors, the 14 with a 2:1 out:in ratio — the shape signature a fused
///    gate would have — are all rank-3 `ConvTranspose1d` / `Conv1d` kernels
///    (`decoder.ups.N.0.weight_v` etc.), where the ratio is the stage's channel doubling. A fused
///    gated projection is rank-2 `[2·inner, dim]`. None exists here.
///
/// This test pins the *consequence*: nothing in the audio decode path splits a projection into
/// halves, so the ordering hazard `crate::layout` describes cannot apply. If a gated block is ever
/// added, this fails and whoever adds it has to route it through `layout::split_gate_value`.
#[test]
fn audio_decode_path_has_no_fused_gated_projection() {
    const {
        assert!(
            AUDIO_VAE_IS_UNCONVERTED,
            "the audio VAE's conversion is an identity mapping"
        )
    };

    let f = fixture();
    let cfg = audio_fixture_config();
    let names = MiniMaxH3AudioVae::tensor_names(&cfg);

    for name in &names {
        if !f.has(name) {
            continue;
        }
        let shape = f.shape(name);
        if shape.len() == 2 && shape[0] == 2 * shape[1] {
            panic!(
                "{name} is a rank-2 [2*inner, dim] tensor — the fused-gate shape signature. If the \
                 audio decoder has grown a gated projection it must read its halves through \
                 candle_gen_minimax_h3::layout::split_gate_value, not an ad-hoc slice (sc-18740)."
            );
        }
    }
    println!(
        "AUDIO VAE AUDIT: {} declared decode tensors, none a rank-2 fused gated projection; the \
         official conversion carries every audio key over unchanged",
        names.len()
    );
}
