//! **Reference parity** for the post-process (sc-19378): the `save_audio` limiter (clamp and
//! rescale) plus `replace_low_freq_with_energy_matched` against the upstream Python on a real
//! two-stem render — weights-free, so it runs on every CI lane.
//!
//! `scripts/reference/yue_vocos_reference.py` decodes two synthetic (stdlib, licence-free) clips
//! through the upstream codec and both Vocos decoders, then runs the upstream post-process in
//! memory. `tests/fixtures/vocos_splice_reference.safetensors` commits its inputs — each track's
//! 16 kHz codec waveform (`*_wave16 [8000]`) and 44.1 kHz Vocos waveform (`*_wave44 [22050]`) — and
//! its outputs for both limiter modes (`{clamp,rescale}_{mix,vocal,inst} [22050]`). The clips are
//! loud: the 16 kHz vocal stem and both mixes exceed 0.99, so the clamp, the rescale and
//! `lfilter`'s `±1` clamp all act.
//!
//! Each comparison asserts a **magnitude** bound — `max|Δ| / max|ref|` — never a cosine alone. The
//! bound sits well above the measured noise floor (recorded beside it) and well below what a real
//! defect produces; the goldens are then **mutated** (and the port run with a wrong crossover and
//! the wrong limiter) and the same check must fail.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_audio_yue::candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio_yue::splice::{
    post_process, replace_low_freq_with_energy_matched, Limiter, TrackPair,
};

/// Measured max relative difference vs the torch/torchaudio CPU reference: 2.6e-7 (clamp mix),
/// 2.5e-7 (rescale mix), 0 on the limited stems — f32 rounding in the resampler's and the IIR's
/// accumulation order. A 0.1 % gain error measures ~1e-3, a one-sample shift ~1e-1.
const MAX_REL: f64 = 1e-5;

fn fixture() -> HashMap<String, Tensor> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vocos_splice_reference.safetensors");
    candle_audio_yue::candle_audio::candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

fn flat(fx: &HashMap<String, Tensor>, name: &str) -> Vec<f32> {
    fx[name]
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// `max|got − want| / max|want|`; a length mismatch is an infinite error.
fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    let peak = want.iter().fold(0f64, |m, &v| m.max(v.abs() as f64));
    let diff = got
        .iter()
        .zip(want)
        .fold(0f64, |m, (&a, &b)| m.max((a as f64 - b as f64).abs()));
    diff / peak
}

fn pairs(fx: &HashMap<String, Tensor>) -> (TrackPair, TrackPair) {
    (
        TrackPair {
            vocals: flat(fx, "vocal_wave16"),
            instrumental: flat(fx, "inst_wave16"),
        },
        TrackPair {
            vocals: flat(fx, "vocal_wave44"),
            instrumental: flat(fx, "inst_wave44"),
        },
    )
}

#[test]
fn post_process_matches_the_reference_in_both_limiter_modes() {
    let fx = fixture();
    let (codec, vocoder) = pairs(&fx);
    // The fixture is non-trivial: the limits and the splice all act on it.
    let peak = |v: &[f32]| v.iter().fold(0f32, |m, x| m.max(x.abs()));
    assert!(
        peak(&codec.vocals) > 0.99,
        "a 16 kHz stem exceeds the clamp"
    );
    let raw_mix: Vec<f32> = vocoder
        .vocals
        .iter()
        .zip(&vocoder.instrumental)
        .map(|(a, b)| a + b)
        .collect();
    assert!(peak(&raw_mix) > 0.99, "the vocoder mix exceeds the limit");

    for (limiter, mode) in [(Limiter::Clamp, "clamp"), (Limiter::Rescale, "rescale")] {
        let out = post_process(&codec, &vocoder, limiter);
        // AC 3: mix and both stems at 44.1 kHz (882 samples per 50 Hz codec frame).
        assert_eq!(out.mix.len(), 25 * 882);
        for (got, part) in [
            (&out.mix, "mix"),
            (&out.vocals, "vocal"),
            (&out.instrumental, "inst"),
        ] {
            let want = flat(&fx, &format!("{mode}_{part}"));
            let rel = max_rel(got, &want);
            println!("{mode} {part}: max|Δ|/max|ref| = {rel:.3e} (bound {MAX_REL:.0e})");
            assert!(rel <= MAX_REL, "{mode} {part} diverges: {rel:.3e}");
        }
        // The splice actually changed the mix: it is not the limited vocoder mix.
        let limited: Vec<f32> = candle_audio_yue::splice::limit(&raw_mix, limiter);
        assert!(
            max_rel(&out.mix, &limited) > 1e-2,
            "{mode}: splice was a no-op"
        );
    }
}

#[test]
fn mutated_goldens_and_wrong_ports_fail_the_same_check() {
    let fx = fixture();
    let (codec, vocoder) = pairs(&fx);
    let want = flat(&fx, "clamp_mix");
    let out = post_process(&codec, &vocoder, Limiter::Clamp);
    assert!(max_rel(&out.mix, &want) <= MAX_REL);

    let mut gained = want.clone();
    gained.iter_mut().for_each(|v| *v *= 1.001);
    let mut shifted = want.clone();
    shifted.rotate_right(1);
    let wrong_limiter = post_process(&codec, &vocoder, Limiter::Rescale).mix;
    // The crossover moved 5 % (and the energy match then scales a different band).
    let recons: Vec<f32> = candle_audio_yue::splice::limit(&codec.vocals, Limiter::Clamp)
        .iter()
        .zip(candle_audio_yue::splice::limit(
            &codec.instrumental,
            Limiter::Clamp,
        ))
        .map(|(a, b)| a + b)
        .collect();
    let mix44: Vec<f32> = candle_audio_yue::splice::limit(
        &vocoder
            .instrumental
            .iter()
            .zip(&vocoder.vocals)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>(),
        Limiter::Clamp,
    );
    let right_cutoff =
        replace_low_freq_with_energy_matched(&recons, 16_000, &mix44, 44_100, 5_500.0);
    assert!(max_rel(&right_cutoff, &want) <= MAX_REL);
    let wrong_cutoff =
        replace_low_freq_with_energy_matched(&recons, 16_000, &mix44, 44_100, 5_225.0);

    for (label, rel) in [
        ("golden ×1.001", max_rel(&out.mix, &gained)),
        ("golden shifted one sample", max_rel(&out.mix, &shifted)),
        ("rescale instead of clamp", max_rel(&wrong_limiter, &want)),
        ("crossover 5225 Hz", max_rel(&wrong_cutoff, &want)),
    ] {
        println!("mutation {label}: {rel:.3e}");
        assert!(rel > MAX_REL, "mutation '{label}' passed ({rel:.3e})");
    }
}
