//! **Reference parity** for the native Vocos upsamplers (sc-19378): the vocal (`decoder_131000`)
//! and instrumental (`decoder_151000`) decoders against the upstream PyTorch `VocosDecoder`, on the
//! embedding upstream feeds them — `SoundStream.get_embed(codes)` (`vocoder.process_audio`) — and
//! the whole codec → Vocos → post-process chain against the upstream mix.
//!
//! `scripts/reference/yue_vocos_reference.py` encodes two synthetic (stdlib, licence-free) clips
//! with the upstream codec at 4 kb/s and commits, per track, the 8-codebook grid (`*_codes [8, 25]`)
//! and the reference 44.1 kHz Vocos waveform (`*_wave44 [22050]`), plus the post-processed outputs
//! (float32, CPU) — `tests/fixtures/vocos_splice_reference.safetensors`. The embedding is rebuilt
//! here through the native `get_embed` (bit-exact to upstream, `tests/xcodec_parity.rs`).
//!
//! Each comparison asserts a **magnitude** bound — `max|Δ| / max|ref|` — never a cosine alone. The
//! bounds sit well above the measured CPU noise floor (recorded beside each constant) and well
//! below what a real defect produces; mutated goldens — and each track run through the *other*
//! track's decoder — must fail the same check.
//!
//! ```text
//! YUE_XCODEC_SNAPSHOT=/path/to/xcodec-mini-infer \
//!   cargo test --locked -p candle-audio-yue --test vocos_parity -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use candle_audio_yue::candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio_yue::codec::{XcodecDecoder, CHECKPOINT};
use candle_audio_yue::gen_core::OutputLimiter;
use candle_audio_yue::splice::{post_process, TrackPair};
use candle_audio_yue::tokens::CodecFrames;
use candle_audio_yue::vocoder::{Track, VocosUpsampler, SAMPLES_PER_FRAME};

/// The 44.1 kHz stems through the embed conv, 8 ConvNeXt blocks, the 3530-wide head, `exp`/`cos`/
/// `sin` and a 3528-point inverse DFT. Measured max relative difference on CPU/f32: 1.07e-5 (vocal)
/// and 3.3e-6 (instrumental). That is the f32 noise floor — the torch f32 reference itself differs
/// from the same model run in float64 by 9.6e-6 / 3.1e-6 — so the bound sits just above it, while
/// a 0.1 % gain error (1e-3), a one-sample shift (~2e-1) or the other track's decoder (~1) land far
/// outside it.
const WAVE_MAX_REL: f64 = 2e-5;
/// The full chain (native codec + Vocos + post-process) vs the upstream outputs: the stems' noise
/// floor carried through the limiter and the splice; measured 6.1e-6 (clamp mix) / 5.9e-6
/// (rescale mix), the stems as above.
const MIX_MAX_REL: f64 = 2e-5;

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

fn grid(fx: &HashMap<String, Tensor>, name: &str) -> CodecFrames {
    CodecFrames {
        codebooks: fx[name]
            .to_dtype(DType::I64)
            .unwrap()
            .to_vec2::<i64>()
            .unwrap()
            .into_iter()
            .map(|row| row.into_iter().map(|c| c as u32).collect())
            .collect(),
    }
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

#[test]
#[ignore = "real weights: set YUE_XCODEC_SNAPSHOT to the staged xcodec-mini-infer snapshot; run with --ignored"]
fn vocos_decoders_and_the_post_processed_mix_match_the_reference() {
    let snapshot = PathBuf::from(
        std::env::var("YUE_XCODEC_SNAPSHOT")
            .expect("set YUE_XCODEC_SNAPSHOT to the xcodec-mini-infer snapshot dir"),
    );
    // CPU, not `default_device()`: the reference is torch-CPU f32, and a GPU-vs-CPU rounding gate
    // would be a red nobody can reproduce. The Metal/CUDA lanes compile the same code path.
    let device = Device::Cpu;
    let fx = fixture();
    let codec = XcodecDecoder::load(&snapshot.join(CHECKPOINT), &device).expect("load xcodec");
    let vocos = VocosUpsampler::load(&snapshot, &device).expect("load both Vocos decoders");

    let mut native = HashMap::new();
    for (track, name) in [(Track::Vocals, "vocal"), (Track::Instrumental, "inst")] {
        let frames = grid(&fx, &format!("{name}_codes"));
        let embed = codec.get_embed(&frames).expect("get_embed");
        let want = flat(&fx, &format!("{name}_wave44"));
        assert_eq!(want.len(), frames.frames() * SAMPLES_PER_FRAME);

        // AC 1 + AC 3 — each decoder's 44.1 kHz stem.
        let got = vocos.decoder(track).forward(&embed).expect("vocos");
        let rel = max_rel(&got, &want);
        println!("{name} vocos: max|Δ|/max|ref| = {rel:.3e} (bound {WAVE_MAX_REL:.0e})");
        assert!(rel <= WAVE_MAX_REL, "{name} vocos diverges: {rel:.3e}");

        // Mutations: gain, shift, and the other track's checkpoint.
        let mut gained = want.clone();
        gained.iter_mut().for_each(|v| *v *= 1.001);
        let mut shifted = want.clone();
        shifted.rotate_right(1);
        let other = match track {
            Track::Vocals => Track::Instrumental,
            Track::Instrumental => Track::Vocals,
        };
        let swapped = vocos.decoder(other).forward(&embed).unwrap();
        for (label, m) in [
            ("×1.001", max_rel(&got, &gained)),
            ("shift 1", max_rel(&got, &shifted)),
            ("other decoder", max_rel(&swapped, &want)),
        ] {
            println!("{name} mutation {label}: {m:.3e}");
            assert!(
                m > WAVE_MAX_REL,
                "{name} mutation '{label}' passed ({m:.3e})"
            );
        }

        let wave16 = codec
            .decode_embedding(&embed, &|| false)
            .unwrap()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        native.insert(name, (wave16, got));
    }

    // AC 2 + AC 3 on the native chain: codec + Vocos + post-process vs the upstream outputs.
    let codec_rate = TrackPair {
        vocals: native["vocal"].0.clone(),
        instrumental: native["inst"].0.clone(),
    };
    let vocoder_rate = TrackPair {
        vocals: native["vocal"].1.clone(),
        instrumental: native["inst"].1.clone(),
    };
    for (limiter, mode) in [
        (OutputLimiter::Clamp, "clamp"),
        (OutputLimiter::Rescale, "rescale"),
    ] {
        let out = post_process(&codec_rate, &vocoder_rate, limiter).unwrap();
        for (got, part) in [
            (&out.mix, "mix"),
            (&out.vocals, "vocal"),
            (&out.instrumental, "inst"),
        ] {
            let rel = max_rel(got, &flat(&fx, &format!("{mode}_{part}")));
            println!("chain {mode} {part}: {rel:.3e} (bound {MIX_MAX_REL:.0e})");
            assert!(
                rel <= MIX_MAX_REL,
                "chain {mode} {part} diverges: {rel:.3e}"
            );
        }
    }
}
