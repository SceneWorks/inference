//! sc-17141: audio-VAE decode parity against the Apache-2.0 reference implementation.
//!
//! Fixture `tests/fixtures/audio_vae_decode.safetensors` ← `tools/dump_minimax_h3_audio_vae.py`,
//! which imports the reference bundle shipped INSIDE the MiniMax-H3 snapshot
//! (`FL2VA/audio_vae/{dac_audio_vae,dac_bigvgan,dac_activations,dac_alias_free_*}.py`) and runs its
//! real `DacAudioVAE.decode` at tiny dims. A genuine independent-graph parity check, not a
//! re-derivation from config: the fixture's decode-half tensor NAMES are byte-identical to the
//! published checkpoint's 914 (see `tests/real_weights.rs`).
//!
//! ## Why the sub-fixtures exist
//!
//! `SnakeBeta` and the Kaiser-sinc resamplers carry this port's numerical risk — a periodic
//! activation and a filtered resampler are both easy to get subtly wrong in a way an end-to-end
//! tolerance hides, and the decoder applies 127 of them. Each therefore has its own golden at its
//! own tolerance, above the whole-model one.
//!
//! ## Tolerances — three tiers, and where each floor comes from
//!
//! | tier | gate | observed | floor |
//! |---|---|---|---|
//! | Kaiser filter, `SnakeBeta`, the resamplers, `Activation1d` | 1e-5 | ~1e-7 | f32 round-off |
//! | one `AMPBlock1` | 1e-2 | 1.2e-3 | MLX reduced-precision matmul |
//! | `BigVGAN`, `decode`, `decode_stereo` | 2e-2 | 2.7e-3 … 4.4e-3 | the same, over 21 blocks |
//!
//! The first tier is elementwise or single-channel, so MLX runs no matmul and reproduces the
//! reference essentially exactly. That is why it gets a bound three orders tighter than the house
//! value: a loose bound there would hide exactly what these fixtures exist to catch.
//!
//! The other two inherit MLX's **reduced-precision f32 matmul on Metal** — a bare 6→6 `conv1d`
//! already disagrees with torch by ~4e-4 relative where a 1→1 one agrees to 1e-7 — and thirty-odd
//! such convolutions with a periodic activation between them land the decode at ~4e-3. That is a
//! platform floor, not a port defect. Every test prints its residual so the real margin stays
//! auditable, and the mutation table clears the 2e-2 gate by 4× to 60×.
//!
//! **MLX's `conv_transpose1d` would have put even that out of reach.** It is *worse* than reduced
//! precision on Metal — a 12-tap depthwise upsample of a unit impulse comes back f16-quantized
//! (`0.4432098` → `0.4431152`) — and this decoder applies 127 of them plus 7 wide upsample stages.
//! The port therefore runs every transposed convolution as zero-insert + `conv1d`
//! (`alias_free::transposed_conv1d`), which is exact.
//! `upsample_matches_the_reference_sample_for_sample` at 1e-5 is what catches a regression back to
//! the built-in.

mod common;

use common::{audio_fixture_config, to_nlc, AUDIO_FIXTURE};

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::audio_vae::{AmpBlock1, BigVgan};
use mlx_gen_minimax_h3::{
    kaiser_sinc_filter1d, LowPassFilter1d, MiniMaxH3AudioVae, MiniMaxH3AudioVaeConfig, SnakeBeta,
    UpSample1d,
};

/// Whole-decoder parity. Observed residual is 4.4e-3 — see the module docs for where that comes
/// from and why it cannot be tightened.
const TOL: f32 = 2e-2;
/// One AMP block. Its convolutions are multi-channel, so it inherits the same reduced-precision
/// floor as the whole decoder (measured 1.2e-3) without the 21-block amplification.
const BLOCK_TOL: f32 = 1e-2;
/// Sub-module parity for the pieces that are elementwise or single-channel — the Kaiser filter,
/// `SnakeBeta`, the resamplers, `Activation1d`. Those run no matmul, so MLX reproduces the
/// reference to ~1e-7 and a loose bound here would hide exactly the errors these fixtures exist
/// to catch.
const UNIT_TOL: f32 = 1e-5;
/// Mutation and "this knob is load-bearing" probes must clear the whole-decoder gate, or "the
/// output moved" could be numerical jitter rather than a wiring difference.
const MUTATION_FLOOR: f32 = 2e-2;

fn fixture() -> Weights {
    Weights::from_file(AUDIO_FIXTURE).unwrap()
}

/// The fixture minus the reference-side extras — i.e. exactly the model weights in the published
/// naming, which is what the loader consumes for real weights.
fn model_weights() -> Weights {
    let mut w = fixture();
    for prefix in ["in.", "out.", "const.", "amp."] {
        w.remove_prefix(prefix);
    }
    w
}

/// Just the standalone AMP block's tensors, still under their `amp.` prefix.
fn amp_weights() -> Weights {
    let f = fixture();
    let mut map = std::collections::HashMap::new();
    for key in f.keys().map(str::to_string).collect::<Vec<_>>() {
        if key.starts_with("amp.") {
            map.insert(key.clone(), f.require(&key).unwrap().clone());
        }
    }
    Weights::from_map(map)
}

fn vae() -> MiniMaxH3AudioVae {
    let mut w = model_weights();
    MiniMaxH3AudioVae::from_weights(&mut w, &audio_fixture_config(), Dtype::Float32).unwrap()
}

/// `(max|a−b| / peak|b|, mean|a−b| / mean|b|)`.
fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let a = a
        .as_dtype(Dtype::Float32)
        .unwrap()
        .flatten(None, None)
        .unwrap();
    let b = b
        .as_dtype(Dtype::Float32)
        .unwrap()
        .flatten(None, None)
        .unwrap();
    let diff = mlx_rs::ops::subtract(&a, &b).unwrap().abs().unwrap();
    let bb = b.abs().unwrap();
    let peak: f32 = bb.max(None).unwrap().item();
    let mean: f32 = bb.mean(None).unwrap().item();
    let max_d: f32 = diff.max(None).unwrap().item();
    let mean_d: f32 = diff.mean(None).unwrap().item();
    (max_d / peak.max(1e-12), mean_d / mean.max(1e-12))
}

fn assert_parity(got: &Array, want: &Array, tol: f32, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (peak, mean) = rel(got, want);
    println!("  {what}: peak rel {peak:.3e} (mean {mean:.3e}, tol {tol:.1e})");
    assert!(
        peak < tol,
        "{what}: peak-relative error {peak:.3e} (mean {mean:.3e}) exceeds {tol:.1e}"
    );
}

fn std_dev(x: &Array) -> f32 {
    let x = x.as_dtype(Dtype::Float32).unwrap();
    let mean: f32 = x.mean(None).unwrap().item();
    let centered = mlx_rs::ops::subtract(&x, Array::from_f32(mean)).unwrap();
    let var: f32 = centered.square().unwrap().mean(None).unwrap().item();
    var.sqrt()
}

/// Materialize a (possibly transposed) array's logical order — `as_slice` returns the PHYSICAL
/// buffer, so a strided view would be read in the wrong order.
fn values(x: &Array) -> Vec<f32> {
    let n = x.size() as i32;
    x.as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

// ---------------------------------------------------------------------------------------------
// Kaiser-sinc filter derivation — targeted fixture #1
// ---------------------------------------------------------------------------------------------

/// The four dumped cases walk the whole branch table of
/// `dac_alias_free_filter.py::kaiser_sinc_filter1d`: the shipped 12-tap / ratio-2 filter (`A > 50`),
/// a wider ratio, an ODD kernel size (a different `time` grid), and a short kernel whose
/// attenuation estimate falls under 21 so `beta` is zero.
///
/// A port that hardcoded the shipped filter, or implemented only the `A > 50` branch, or used the
/// even-kernel `time` grid unconditionally, fails here while still passing an end-to-end golden
/// (the checkpoint ships its filters as buffers, so the model never calls this on the load path).
#[test]
fn kaiser_sinc_filter_matches_the_reference() {
    let f = fixture();
    for tag in ["r2k12", "r4k24", "odd11", "beta0"] {
        let params = values(f.require(&format!("const.kaiser.{tag}")).unwrap());
        let (cutoff, half_width, ksize) = (params[0] as f64, params[1] as f64, params[2] as i32);
        let got = kaiser_sinc_filter1d(cutoff, half_width, ksize).unwrap();
        let want = f.require(&format!("out.kaiser.{tag}")).unwrap();
        assert_parity(
            &got,
            want,
            UNIT_TOL,
            &format!("kaiser_sinc_filter1d({tag})"),
        );
    }

    // The four cases really are different filters, so passing all four is not one assertion in
    // disguise.
    let r2 = f.require("out.kaiser.r2k12").unwrap();
    let odd = f.require("out.kaiser.odd11").unwrap();
    assert_ne!(r2.shape(), odd.shape());
}

/// The taps the checkpoint STORES are reproducible from the derivation. The loader reads the stored
/// buffers (that is what the reference's `register_buffer` + strict load does), so without this the
/// derivation would be untested against the model's own filters.
#[test]
fn stored_filters_match_the_derivation() {
    let f = fixture();
    let derived = kaiser_sinc_filter1d(0.5 / 2.0, 0.6 / 2.0, 12).unwrap();
    for key in [
        "const.resample.up_filter",
        "const.resample.down_filter",
        "decoder.activation_post.upsample.filter",
        "decoder.activation_post.downsample.lowpass.filter",
        "decoder.resblocks.0.activations.0.upsample.filter",
        "decoder.resblocks.20.activations.5.downsample.lowpass.filter",
    ] {
        assert_parity(&derived, f.require(key).unwrap(), UNIT_TOL, key);
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
    let filter = f.require("const.resample.up_filter").unwrap().clone();
    let up = UpSample1d::from_filter(filter.clone(), 2).unwrap();
    let down = LowPassFilter1d::from_filter(filter, 2).unwrap();
    let x = to_nlc(f.require("in.resample.x").unwrap());

    let got_up = up.forward(&x).unwrap();
    assert_parity(
        &got_up,
        &to_nlc(f.require("out.resample.up").unwrap()),
        UNIT_TOL,
        "UpSample1d",
    );

    let got_down = down.forward(&x).unwrap();
    assert_parity(
        &got_down,
        &to_nlc(f.require("out.resample.down").unwrap()),
        UNIT_TOL,
        "DownSample1d",
    );

    let round_trip = down.forward(&got_up).unwrap();
    assert_parity(
        &round_trip,
        &to_nlc(f.require("out.resample.up_down").unwrap()),
        UNIT_TOL,
        "UpSample1d -> DownSample1d",
    );
}

/// A shifted or mis-scaled upsample would still round-trip to the right SHAPE, so pin the actual
/// samples: the reference's `x = ratio · conv_transpose(...)` scaling and its trim offsets.
#[test]
fn upsample_matches_the_reference_sample_for_sample() {
    let f = fixture();
    let filter = f.require("const.resample.up_filter").unwrap().clone();
    let up = UpSample1d::from_filter(filter, 2).unwrap();
    let got = values(
        &up.forward(&to_nlc(f.require("in.resample.x").unwrap()))
            .unwrap(),
    );
    let want = values(&to_nlc(f.require("out.resample.up").unwrap()));
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
        "a channel-shifted comparison is just as good ({shifted:.3e}); the test would not \
         detect an off-by-one"
    );
}

// ---------------------------------------------------------------------------------------------
// SnakeBeta — targeted fixture #3
// ---------------------------------------------------------------------------------------------

/// `x + sin²(α·x) / (β + 1e-9)` with log-scale α/β, and the proof that the log scale matters.
#[test]
fn snakebeta_matches_the_reference() {
    let f = fixture();
    let x = to_nlc(f.require("in.snake.x").unwrap());
    let alpha = f.require("in.snake.alpha").unwrap().clone();
    let beta = f.require("in.snake.beta").unwrap().clone();

    for (tag, logscale) in [("log", true), ("linear", false)] {
        let got = SnakeBeta::new(alpha.clone(), beta.clone(), logscale)
            .unwrap()
            .forward(&x)
            .unwrap();
        assert_parity(
            &got,
            &to_nlc(f.require(&format!("out.snake.{tag}")).unwrap()),
            UNIT_TOL,
            &format!("SnakeBeta(alpha_logscale={logscale})"),
        );
    }

    // `snake_logscale = true` appears in NO config file — it is selected by the reference's
    // `sample_rate` branch. If the two modes agreed, the fixture could not police it.
    let (gap, _) = rel(
        f.require("out.snake.log").unwrap(),
        f.require("out.snake.linear").unwrap(),
    );
    assert!(
        gap > MUTATION_FLOOR,
        "the two snake scales agree (rel {gap:.3e}); the golden cannot pin snake_logscale"
    );
}

/// The composed `Activation1d`: 2× up → SnakeBeta → 2× down.
#[test]
fn activation1d_matches_the_reference() {
    let f = fixture();
    let filter = f.require("const.resample.up_filter").unwrap().clone();
    let act = mlx_gen_minimax_h3::Activation1d::new(
        SnakeBeta::new(
            f.require("in.act1d.alpha").unwrap().clone(),
            f.require("in.act1d.beta").unwrap().clone(),
            true,
        )
        .unwrap(),
        UpSample1d::from_filter(filter.clone(), 2).unwrap(),
        LowPassFilter1d::from_filter(filter, 2).unwrap(),
    );
    let got = act
        .forward(&to_nlc(f.require("in.act1d.x").unwrap()))
        .unwrap();
    assert_parity(
        &got,
        &to_nlc(f.require("out.act1d.y").unwrap()),
        UNIT_TOL,
        "Activation1d",
    );
}

// ---------------------------------------------------------------------------------------------
// Blocks and the whole decoder
// ---------------------------------------------------------------------------------------------

/// One `AMPBlock1` in isolation: three dilated conv pairs, six anti-aliased activations paired
/// `activations[::2]` / `activations[1::2]`, residual after each pair.
#[test]
fn amp_block_matches_the_reference() {
    let f = fixture();
    let mut w = amp_weights();
    let block =
        AmpBlock1::from_weights(&mut w, "amp", 7, &[1, 3, 5], true, Dtype::Float32).unwrap();
    let got = block
        .forward(&to_nlc(f.require("in.amp.x").unwrap()))
        .unwrap();
    assert_parity(
        &got,
        &to_nlc(f.require("out.amp.y").unwrap()),
        BLOCK_TOL,
        "AMPBlock1",
    );
}

/// The activation pairing is observable: swapping `activations.0` with `activations.1` moves the
/// block's output. A port that paired them `0,1,2` / `3,4,5` loads every tensor and looks correct.
#[test]
fn amp_block_activation_pairing_is_load_bearing() {
    let f = fixture();
    let x = to_nlc(f.require("in.amp.x").unwrap());
    let mut w = amp_weights();
    let baseline = AmpBlock1::from_weights(&mut w, "amp", 7, &[1, 3, 5], true, Dtype::Float32)
        .unwrap()
        .forward(&x)
        .unwrap();

    let mut w = amp_weights();
    for leaf in ["act.alpha", "act.beta"] {
        let a = w
            .require(&format!("amp.activations.0.{leaf}"))
            .unwrap()
            .clone();
        let b = w
            .require(&format!("amp.activations.1.{leaf}"))
            .unwrap()
            .clone();
        w.insert(format!("amp.activations.0.{leaf}"), b);
        w.insert(format!("amp.activations.1.{leaf}"), a);
    }
    let swapped = AmpBlock1::from_weights(&mut w, "amp", 7, &[1, 3, 5], true, Dtype::Float32)
        .unwrap()
        .forward(&x)
        .unwrap();
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
    let mut w = model_weights();
    let cfg = audio_fixture_config();
    let decoder = BigVgan::from_weights(&mut w, "decoder", &cfg, Dtype::Float32).unwrap();
    let got = decoder
        .forward(&to_nlc(f.require("in.bigvgan.x").unwrap()))
        .unwrap();
    let want = to_nlc(f.require("out.bigvgan.y").unwrap());
    // 4 tokens x the real 800x hop.
    assert_eq!(got.shape(), &[1, 3200, 1]);
    assert_parity(&got, &want, TOL, "BigVGAN");
}

/// `DacAudioVAE.decode` = `dec_in_proj` → BigVGAN. Reference-exact: no de-normalization.
#[test]
fn decode_matches_the_reference() {
    let f = fixture();
    let got = vae().decode(f.require("in.decode.z").unwrap()).unwrap();
    let want = f.require("out.decode.audio").unwrap();
    assert_eq!(
        got.shape(),
        &[1, 1, 3200],
        "4 tokens at 40 Hz -> 0.1 s @ 32 kHz"
    );
    assert_parity(&got, want, TOL, "DacAudioVAE.decode");
}

/// De-normalization, stereo folding and the interleave, each against its own golden.
#[test]
fn stereo_decode_matches_the_reference() {
    let f = fixture();
    let vae = vae();
    let z = f.require("in.stereo.z").unwrap();

    let denorm = vae.denormalize(z).unwrap();
    assert_parity(
        &denorm,
        f.require("out.stereo.denorm").unwrap(),
        UNIT_TOL,
        "latent de-normalization",
    );

    let stereo = vae.decode_stereo(&denorm).unwrap();
    assert_eq!(stereo.shape(), &[1, 2, 3200]);
    assert_parity(
        &stereo,
        f.require("out.stereo.audio").unwrap(),
        TOL,
        "decode_stereo",
    );

    let track = vae.decode_audio_track(z).unwrap();
    let got = Array::from_slice(&track.samples, &[track.samples.len() as i32]);
    assert_parity(
        &got,
        f.require("out.stereo.interleaved").unwrap(),
        TOL,
        "AudioTrack interleave",
    );
}

/// The `gen-core` output contract: 32 kHz, 2 channels, interleaved f32, no stems.
#[test]
fn audio_track_is_the_gen_core_contract() {
    let f = fixture();
    let track = vae()
        .decode_audio_track(f.require("in.stereo.z").unwrap())
        .unwrap();
    assert_eq!(track.sample_rate, 32_000);
    assert_eq!(track.channels, 2);
    assert!(track.stems.is_empty(), "this model emits a mix, not stems");
    // 4 latent tokens · 800 samples · 2 channels.
    assert_eq!(track.samples.len(), 4 * 800 * 2);
    assert!(track.samples.iter().all(|s| s.is_finite()));
    assert!(track.samples.iter().all(|s| (-1.0..=1.0).contains(s)));
}

/// The interleave is `L0, R0, L1, R1, …` — and the two channels are GENUINELY different, so a
/// mono-duplicating port cannot pass. This is the check the story calls out as easy to fake.
#[test]
fn stereo_channels_are_independent_and_interleaved() {
    let f = fixture();
    let vae = vae();
    let z = f.require("in.stereo.z").unwrap();
    let stereo = vae.decode_stereo(&vae.denormalize(z).unwrap()).unwrap();
    let planar = values(&stereo);
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

    let track = vae.decode_audio_track(z).unwrap();
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
    let swapped_z = z.take_axis(Array::from_slice(&[1, 0], &[2]), 1).unwrap();
    let swapped = values(
        &vae.decode_stereo(&vae.denormalize(&swapped_z).unwrap())
            .unwrap(),
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
    let vae = vae();
    let z = f.require("in.stereo.z").unwrap();
    let transposed = z.transpose_axes(&[0, 2, 1, 3]).unwrap();
    assert!(vae.decode_stereo(&transposed).is_err());
    // The mono entry point rejects a stereo-ranked latent too.
    assert!(vae.decode(z).is_err());
    // B > 1 cannot be interleaved into one track.
    let two = mlx_rs::ops::concatenate_axis(&[z, z], 0).unwrap();
    assert!(vae.decode_audio_track(&two).is_err());
    assert!(
        vae.decode_stereo(&two).is_ok(),
        "but decode_stereo batches fine"
    );
}

// ---------------------------------------------------------------------------------------------
// Weight mapping
// ---------------------------------------------------------------------------------------------

/// Every model tensor must be consumed, and the declared name list must be exactly the
/// checkpoint's. A silently unmapped tensor still produces plausible-looking audio.
#[test]
fn weight_mapping_is_exhaustive() {
    let cfg = audio_fixture_config();
    let mut w = model_weights();
    let before: std::collections::BTreeSet<String> = w.keys().map(str::to_string).collect();
    assert_eq!(
        before.len(),
        914,
        "the fixture carries the decode half only"
    );

    let _vae = MiniMaxH3AudioVae::from_weights(&mut w, &cfg, Dtype::Float32).unwrap();
    w.remove_accessed();
    let leftover: Vec<&str> = w.keys().collect();
    assert!(
        leftover.is_empty(),
        "these checkpoint tensors were never read: {leftover:?}"
    );

    let declared: std::collections::BTreeSet<String> =
        MiniMaxH3AudioVae::tensor_names(&cfg).into_iter().collect();
    assert_eq!(
        declared, before,
        "declared tensor names differ from the checkpoint's"
    );
}

/// A checkpoint whose `dec_in_proj` disagrees with the config is a loud error, not a silent
/// re-interpretation.
#[test]
fn a_mismatched_checkpoint_is_rejected() {
    let mut cfg = audio_fixture_config();
    cfg.latent_channels = 16;
    cfg.latents_mean.truncate(16);
    cfg.latents_std.truncate(16);
    let mut w = model_weights();
    assert!(MiniMaxH3AudioVae::from_weights(&mut w, &cfg, Dtype::Float32).is_err());
}

// ---------------------------------------------------------------------------------------------
// False-green guards
// ---------------------------------------------------------------------------------------------

/// A constant, all-zero or fully-saturated golden is a false green.
///
/// `SnakeBeta` initializes `alpha`/`beta` to ONES, which under the log scale means `exp(1)` for
/// both and makes the two parameters indistinguishable; the decoder's final CLAMP means a golden
/// dumped at a large scale would saturate and be reproduced by any port that also saturates. The
/// generator re-randomizes and checks both; this is the in-repo half of that contract.
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
        let t = f.require(key).unwrap();
        assert!(
            std_dev(t) > 1e-4,
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
        let a = f.require(&format!("{key}.act.alpha")).unwrap();
        let b = f.require(&format!("{key}.act.beta")).unwrap();
        let (gap, _) = rel(a, b);
        assert!(gap > 1e-2, "{key}: alpha and beta are indistinguishable");
        // `activation_post` is a single channel at this width (128 / 2^7 = 1), so per-channel
        // spread is only meaningful for the wider blocks.
        if a.size() > 1 {
            assert!(
                std_dev(a) > 1e-3,
                "{key}: alpha is ~constant across channels"
            );
        }
    }

    // The decode must not sit on the clamp.
    let audio = values(f.require("out.stereo.audio").unwrap());
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
    let cfg = audio_fixture_config();
    let z = f.require("in.decode.z").unwrap();
    let baseline = vae().decode(z).unwrap();

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
        let mut w = model_weights();
        let original = w.require(key).unwrap().clone();
        // Scale AND shift: the shift makes an all-zero tensor observable, the scale keeps the
        // perturbation proportionate for tensors that are already large.
        let bumped = mlx_rs::ops::add(
            mlx_rs::ops::multiply(&original, Array::from_f32(1.3)).unwrap(),
            Array::from_f32(0.2),
        )
        .unwrap();
        w.insert(key, bumped);
        let mutated = MiniMaxH3AudioVae::from_weights(&mut w, &cfg, Dtype::Float32)
            .unwrap()
            .decode(z)
            .unwrap();
        let (peak, _) = rel(&mutated, &baseline);
        println!("  {key}: peak rel {peak:.3e} (floor {MUTATION_FLOOR:.1e})");
        assert!(
            peak > MUTATION_FLOOR,
            "perturbing {key} moved the decode by only {peak:.3e}, under the \
             {MUTATION_FLOOR:.1e} floor — it is not wired into the graph"
        );
    }

    // The stored Kaiser filters are read, not re-derived, so they must be load-bearing too.
    let mut w = model_weights();
    for suffix in ["upsample.filter", "downsample.lowpass.filter"] {
        let key = format!("decoder.resblocks.0.activations.0.{suffix}");
        let flat = kaiser_sinc_filter1d(0.4, 0.2, 12).unwrap();
        w.insert(&key, flat);
    }
    let mutated = MiniMaxH3AudioVae::from_weights(&mut w, &cfg, Dtype::Float32)
        .unwrap()
        .decode(z)
        .unwrap();
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
    let z = f.require("in.decode.z").unwrap();
    let baseline = vae().decode(z).unwrap();

    let build = |cfg: &MiniMaxH3AudioVaeConfig| {
        let mut w = model_weights();
        MiniMaxH3AudioVae::from_weights(&mut w, cfg, Dtype::Float32)
            .unwrap()
            .decode(z)
            .unwrap()
    };

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
    let mut w = model_weights();
    assert!(MiniMaxH3AudioVae::from_weights(&mut w, &cfg, Dtype::Float32).is_err());
}

/// De-normalization is APPLIED, and applied per channel — the pipeline path must differ from the
/// reference-exact one, and the closed form must hold element-wise.
#[test]
fn latent_denormalization_is_applied_per_channel() {
    let f = fixture();
    let vae = vae();
    let z = f.require("in.stereo.z").unwrap();

    let denorm = vae.denormalize(z).unwrap();
    let (peak, _) = rel(&denorm, z);
    assert!(peak > 1e-2, "de-normalization was a no-op ({peak:.3e})");

    let raw = values(z);
    let got = values(&denorm);
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
    let with = vae.decode_audio_track(z).unwrap();
    let without = vae.decode_stereo(z).unwrap();
    let a = Array::from_slice(&with.samples, &[with.samples.len() as i32]);
    let b = without
        .reshape(&[2, 3200])
        .unwrap()
        .transpose_axes(&[1, 0])
        .unwrap()
        .reshape(&[6400])
        .unwrap();
    let (peak, _) = rel(&a, &b);
    println!("  denormalized vs raw decode: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "decode ignored latents_mean/std ({peak:.3e})"
    );
}
