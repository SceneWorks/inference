//! **Reference parity** for the native ICL reference encoder (sc-19379): reference audio → the
//! prompt's codec token block, token for token against YuE-v1's own Python.
//!
//! `scripts/reference/yue_icl_reference.py` runs `infer.py`'s `load_audio_mono` + `encode_audio`
//! (extracted verbatim) and its ICL windowing/wrapping lines on synthetic clips and commits
//! `tests/fixtures/yue_icl_reference.json`: per mode the windowed codebook-0 ids and the full
//! first-segment stage-1 prompt, plus the SHA-256 of each clip's int16 PCM. The clips are built
//! from integer-exact recipes this file mirrors (`synth_*`), so the fixture carries no audio; the
//! test first proves its PCM is the reference's byte for byte.
//!
//! * single-track — 44.1 kHz **stereo** (channel mean + the 441:160 torchaudio resample), window
//!   0.5–3.5 s → 150 ids at 50 tok/s;
//! * dual-track — 48 kHz vocal + instrumental stems (3:1 resample), window 1.0–2.5 s → 150 ids
//!   interleaved at 100 tok/s.
//!
//! Both clips are a non-whole number of 20 ms frames at 16 kHz, so `SoundStream.encode` takes its
//! acoustic re-encode-on-padded-input branch (202 / 151 codec frames), the common case for real
//! clips.
//!
//! The ids are compared **exactly** (the AC is token for token), and so is the whole first-segment
//! prompt the block lands in; a mutated golden (one id changed) must fail the same comparisons.
//!
//! **Exactness, measured (CPU f32, 2026-09-24).** 150/150 single and 150/150 dual ids equal the
//! reference. On whole-frame versions of these clips the pre-quantization `fc_prior` embedding
//! differs from torch's by max|Δ|/max|ref| 5e-5 … 1.1e-4 (HuBERT layer mean 1.4e-5, DAC encoder
//! 1.3e-5, resampled wave ≤ 2 f32 ulp) — the
//! codec's own float sensitivity: torch itself, fed the native resampler's wave (≤ 2 ulp away) or
//! a ±1-ulp jittered one, moves by 4e-5 … 9.6e-5. A token can therefore only differ from the
//! reference where two codewords are within that noise of equidistant; the fixture records each
//! track's smallest codeword margin (`d₂ − d₁`, reference 0.19 … 0.49 on `‖x‖² ≈ 2e4`), which the
//! test prints beside the native encoder's own (they agree to ≤ 0.05). The chunked encoder
//! (50-frame chunks, several seams per clip) must reproduce the same ids.
//!
//! `resampler_matches_torchaudio` (weights-free, runs everywhere) holds the ported torchaudio
//! `Resample` to the reference on a short clip at four source rates; the token test above can only
//! be exact if that front end is.
//!
//! It also encodes upstream's own example reference, `yue/prompt_egs/pop.00001.mp3` and its
//! `.Vocals` / `.Instrumental` stems (single and dual, 0–30 s = 1 500 frames — past the default
//! chunking and HuBERT query blocking), from the PCM the producer decoded into
//! `$YUE_REF_DIR/sceneworks-derived/` (SHA-256-checked; the test fails if it is missing).
//!
//! ```text
//! YUE_XCODEC_SNAPSHOT=/path/to/xcodec-mini-infer YUE_REF_DIR=~/.cache/sceneworks-yue-ref \
//!   cargo test --locked -p candle-audio-yue --test icl_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use candle_audio_yue::candle_audio::candle_core::{Device, Tensor};
use candle_audio_yue::codec::CHECKPOINT;
use candle_audio_yue::config::{IclReference, IclTracks};
use candle_audio_yue::gen_core::{AudioTrack, CancelFlag};
use candle_audio_yue::icl::{IclEncoder, IclPromptCodes, XcodecEncoder};
use candle_audio_yue::tokenizer::{MmTokenizer, PromptInput, PromptTokenizer, YuePromptBuilder};
use serde_json::Value;

fn fixture() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/yue_icl_reference.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

// ---- integer-exact synthetic clips (mirror `synth_*` in scripts/reference/yue_icl_reference.py)

fn lcg(state: u32) -> u32 {
    state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

fn to_i16(x: f64) -> i16 {
    (x * 32767.0 + 0.5).floor().clamp(-32768.0, 32767.0) as i16
}

fn noise(state: u32) -> f64 {
    f64::from(state >> 8) / 16_777_216.0 - 0.5
}

fn tau() -> f64 {
    2.0 * std::f64::consts::PI
}

fn synth_single(rate: u32, frames: u32) -> Vec<i16> {
    let mut out = Vec::new();
    let mut state = 12_345u32;
    for i in 0..frames {
        let t = f64::from(i) / f64::from(rate);
        state = lcg(state);
        let n = noise(state);
        let env = (-4.0 * (t % 0.5)).exp();
        let hat = (-40.0 * (t % 0.25)).exp();
        let chord = (0.0
            + (tau() * 196.0 * t).sin()
            + (tau() * 246.94 * t).sin()
            + (tau() * 293.66 * t).sin())
            / 3.0;
        let left = 0.5 * env * chord + 0.15 * hat * n;
        let ph = tau() * 392.0 * t + 4.0 * (tau() * 5.5 * t).sin();
        let right = 0.3 * (ph.sin() + 0.5 * (2.0 * ph).sin() + 0.25 * (3.0 * ph).sin());
        out.push(to_i16(left));
        out.push(to_i16(right));
    }
    out
}

fn synth_vocals(rate: u32, frames: u32) -> Vec<i16> {
    (0..frames)
        .map(|i| {
            let t = f64::from(i) / f64::from(rate);
            let f0 = 220.0 + 110.0 * (t / 3.0);
            let ph = tau() * f0 * t + 3.0 * (tau() * 6.0 * t).sin();
            let gate = 0.5 - 0.5 * (tau() * 2.0 * t).cos();
            let v =
                ph.sin() + 0.6 * (2.0 * ph).sin() + 0.4 * (3.0 * ph).sin() + 0.2 * (5.0 * ph).sin();
            to_i16(0.35 * gate * v)
        })
        .collect()
}

fn synth_instrumental(rate: u32, frames: u32) -> Vec<i16> {
    let mut out = Vec::new();
    let mut state = 777u32;
    for i in 0..frames {
        let t = f64::from(i) / f64::from(rate);
        state = lcg(state);
        let n = noise(state);
        let beat = t % 0.5;
        let kick = (-30.0 * beat).exp() * (tau() * 60.0 * beat).sin();
        let snare = (-25.0 * ((t + 0.25) % 0.5)).exp() * n;
        let bass = (tau() * 55.0 * t).sin();
        let chord = (0.0
            + (tau() * 261.63 * t).sin()
            + (tau() * 329.63 * t).sin()
            + (tau() * 392.0 * t).sin())
            / 3.0;
        out.push(to_i16(0.4 * kick + 0.3 * snare + 0.25 * bass + 0.2 * chord));
    }
    out
}

fn synth_resample_input(n: u32) -> Vec<i16> {
    let mut state = 4242u32;
    (0..n)
        .map(|i| {
            state = lcg(state);
            to_i16(0.4 * noise(state) + 0.5 * (0.05 * f64::from(i)).sin())
        })
        .collect()
}

fn sha256_hex(pcm: &[i16]) -> String {
    use sha2::{Digest, Sha256};
    let bytes: Vec<u8> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `torchaudio.load` of a PCM-16 WAV: `int16 / 32768` float32, interleaved.
fn track(pcm: &[i16], rate: u32, channels: u16) -> AudioTrack {
    AudioTrack {
        samples: pcm.iter().map(|&v| f32::from(v) / 32768.0).collect(),
        sample_rate: rate,
        channels,
        stems: Vec::new(),
    }
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn u32_of(v: &Value) -> u32 {
    v.as_u64().unwrap() as u32
}

fn f32_of(v: &Value) -> f32 {
    v.as_f64().unwrap() as f32
}

#[test]
fn synthetic_clips_reproduce_the_reference_pcm() {
    let fx = fixture();
    let (s, d) = (&fx["single"], &fx["dual"]);
    assert_eq!(
        sha256_hex(&synth_single(u32_of(&s["rate"]), u32_of(&s["frames"]))),
        s["pcm_sha256"].as_str().unwrap()
    );
    assert_eq!(
        sha256_hex(&synth_vocals(u32_of(&d["rate"]), u32_of(&d["frames"]))),
        d["vocals_pcm_sha256"].as_str().unwrap()
    );
    assert_eq!(
        sha256_hex(&synth_instrumental(
            u32_of(&d["rate"]),
            u32_of(&d["frames"])
        )),
        d["instrumental_pcm_sha256"].as_str().unwrap()
    );
    // The fingerprint discriminates: one sample off by one LSB is a different clip.
    let mut pcm = synth_vocals(u32_of(&d["rate"]), u32_of(&d["frames"]));
    pcm[1000] = pcm[1000].wrapping_add(1);
    assert_ne!(sha256_hex(&pcm), d["vocals_pcm_sha256"].as_str().unwrap());
}

/// Per-stage bounds (`max|Δ| / max|ref|`) for the tiny encode-half fixture — see
/// [`tiny_encode_half_matches_the_reference_stage_by_stage`] for the measured floors.
const TINY_MAX_REL: [(&str, f64); 4] = [
    ("hubert_mean", 1e-5),
    ("semantic", 1e-5),
    ("acoustic", 1e-5),
    ("fc_prior", 1e-5),
];

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length");
    let peak = want.iter().fold(0f64, |m, &v| m.max(f64::from(v).abs()));
    let diff = got.iter().zip(want).fold(0f64, |m, (&a, &b)| {
        m.max((f64::from(a) - f64::from(b)).abs())
    });
    diff / peak
}

/// The whole encode half — HuBERT (feature encoder, GroupNorm, positional conv, post-LN layers,
/// the 13-state mean), the RepCodec semantic encoder, the DAC encoder on its padded re-encode
/// branch, `fc_prior`, the codebook-0 nearest codeword — against upstream's own classes and
/// `SoundStream.encode`, at small random widths, weights-free (`yue_icl_reference.py --tiny`).
/// Measured on CPU f32: HuBERT mean 1.2e-6, semantic 9.2e-7, acoustic 1.2e-6, `fc_prior` 8.4e-7
/// relative (bounds 1e-5, ~8× headroom); codes 51/51 exact.
#[test]
fn tiny_encode_half_matches_the_reference_stage_by_stage() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/yue_icl_tiny_reference.safetensors");
    let fx = candle_audio_yue::candle_audio::candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap();
    let enc = XcodecEncoder::load(&path, &Device::Cpu).expect("load the tiny checkpoint");
    let input = flat(&fx["ref.input"]);
    let stages = enc.encode_stages(&input, &|| false).unwrap().unwrap();
    let got = [
        ("hubert_mean", &stages.hubert_mean),
        ("semantic", &stages.semantic),
        ("acoustic", &stages.acoustic),
        ("fc_prior", &stages.prior),
    ];
    for ((name, t), (bound_name, bound)) in got.into_iter().zip(TINY_MAX_REL) {
        assert_eq!(name, bound_name);
        let want = &fx[&format!("ref.{name}")];
        assert_eq!(&t.dims()[1..], want.dims(), "{name} shape");
        let rel = max_rel(&flat(t), &flat(want));
        println!("{name}: max|Δ|/max|ref| = {rel:.3e} (bound {bound:.0e})");
        assert!(rel <= bound, "{name} diverges: {rel:.3e}");
    }
    let want_codes: Vec<u32> = fx["ref.codes"]
        .to_vec1::<i64>()
        .unwrap()
        .into_iter()
        .map(|c| c as u32)
        .collect();
    let codes = enc.encode_codes(&input, &|| false).unwrap().unwrap();
    assert_eq!(codes, want_codes, "codes");
    let mut bad = want_codes.clone();
    bad[7] = (bad[7] + 1) % 1024;
    assert_ne!(codes, bad, "a mutated code golden passed");
}

/// torchaudio's resampler, measured against the native port: max |Δ| 6.0e-8 … 1.2e-7 over the four
/// source rates (≤ 1 f32 ulp at these amplitudes — the float64-accumulated port against torch's
/// float32 conv). The bound admits that rounding and nothing else: a 1e-5 nudge of one golden
/// sample, a one-sample shift, or a Kaiser-window resampler (the shared `candle_audio::dsp` one,
/// 3e-2 … 7e-2 here) all land far outside it.
const RESAMPLE_MAX_ABS: f32 = 5e-7;

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    got.iter()
        .zip(want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
}

#[test]
fn resampler_matches_torchaudio() {
    use candle_audio_yue::icl::resample_to_16k;
    let fx = fixture();
    let r = &fx["resample"];
    let pcm = synth_resample_input(u32_of(&r["input_samples"]));
    assert_eq!(sha256_hex(&pcm), r["input_pcm_sha256"].as_str().unwrap());
    let x: Vec<f32> = pcm.iter().map(|&v| f32::from(v) / 32768.0).collect();
    for case in r["cases"].as_array().unwrap() {
        let rate = u32_of(&case["rate"]);
        let want: Vec<f32> = case["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let got = resample_to_16k(&x, rate).unwrap();
        let d = max_abs(&got, &want);
        println!("{rate} Hz → 16 kHz: {} samples, max |Δ| {d:.3e}", got.len());
        assert!(d <= RESAMPLE_MAX_ABS, "{rate} Hz: max |Δ| {d:.3e}");

        let mut nudged = want.clone();
        nudged[want.len() / 3] += 1e-5;
        assert!(
            max_abs(&got, &nudged) > RESAMPLE_MAX_ABS,
            "{rate}: nudge passed"
        );
        let mut shifted = want.clone();
        shifted.rotate_right(1);
        assert!(
            max_abs(&got, &shifted) > RESAMPLE_MAX_ABS,
            "{rate}: shift passed"
        );
        let kaiser = candle_audio_yue::candle_audio::dsp::resample(&x, rate, 16_000, 1).unwrap();
        let n = kaiser.len().min(want.len());
        let dk = max_abs(&kaiser[..n], &want[..n]);
        println!("  shared Kaiser resampler instead: max |Δ| {dk:.3e}");
        assert!(dk > RESAMPLE_MAX_ABS, "{rate}: the Kaiser resampler passed");
    }
}

/// The codec frame count and the smallest `d₂ − d₁` over a track's frames under the native
/// encoder.
fn native_frames_and_margin(
    enc: &XcodecEncoder,
    t: &AudioTrack,
    codebook: &Codebook,
) -> (usize, f64) {
    use candle_audio_yue::icl::{downmix, resample_to_16k};
    let wave = resample_to_16k(&downmix(t).unwrap(), t.sample_rate).unwrap();
    let e = enc.prior_embedding(&wave, &|| false).unwrap().unwrap();
    let x: Vec<Vec<f32>> = e.squeeze(0).unwrap().t().unwrap().to_vec2().unwrap();
    let margin = x
        .iter()
        .map(|row| {
            let mut d: Vec<f64> = codebook
                .0
                .iter()
                .map(|c| {
                    row.iter()
                        .zip(c)
                        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                        .sum()
                })
                .collect();
            d.sort_by(f64::total_cmp);
            d[1] - d[0]
        })
        .fold(f64::INFINITY, f64::min);
    (x.len(), margin)
}

struct Codebook(Vec<Vec<f32>>);

#[test]
#[ignore = "real weights: set YUE_XCODEC_SNAPSHOT to the staged xcodec-mini-infer snapshot; run with --ignored"]
fn icl_prompt_ids_match_the_reference_token_for_token() {
    let snapshot = PathBuf::from(
        std::env::var("YUE_XCODEC_SNAPSHOT")
            .expect("set YUE_XCODEC_SNAPSHOT to the xcodec-mini-infer snapshot dir"),
    );
    // CPU, not `default_device()`: the reference is torch-CPU f32 (see xcodec_parity.rs).
    let enc = XcodecEncoder::load(&snapshot.join(CHECKPOINT), &Device::Cpu).expect("load");
    // The same encoder evaluating its sample-rate stages 50 frames at a time: the 202- and
    // 151-frame clips then cross several chunk seams (the default chunk is longer than them).
    let chunked = XcodecEncoder::load(&snapshot.join(CHECKPOINT), &Device::Cpu)
        .expect("load")
        .with_chunk_frames(50);
    let builder = YuePromptBuilder::new(
        MmTokenizer::from_file(&snapshot.join("mm_tokenizer_v0.2_hf/tokenizer.json")).unwrap(),
    )
    .unwrap();
    let codebook = {
        let st = candle_audio_yue::candle_audio::candle_core::safetensors::load(
            snapshot.join(CHECKPOINT),
            &Device::Cpu,
        )
        .unwrap();
        Codebook(
            st["quantizer.vq.layers.0._codebook.embed"]
                .to_vec2()
                .unwrap(),
        )
    };
    let fx = fixture();
    let (genres, lyrics) = (
        fx["genres"].as_str().unwrap(),
        fx["lyrics"].as_str().unwrap(),
    );

    let s = &fx["single"];
    let single = track(
        &synth_single(u32_of(&s["rate"]), u32_of(&s["frames"])),
        u32_of(&s["rate"]),
        s["channels"].as_u64().unwrap() as u16,
    );
    let d = &fx["dual"];
    let (rate, frames) = (u32_of(&d["rate"]), u32_of(&d["frames"]));
    let vocals = track(&synth_vocals(rate, frames), rate, 1);
    let instrumental = track(&synth_instrumental(rate, frames), rate, 1);

    let cases = [
        (
            "single",
            s,
            IclTracks::Single(single.clone()),
            vec![("mix", &single, s["min_margin"].as_f64().unwrap())],
        ),
        (
            "dual",
            d,
            IclTracks::Dual {
                vocals: vocals.clone(),
                instrumental: instrumental.clone(),
            },
            vec![
                ("vocals", &vocals, d["min_margin_vocals"].as_f64().unwrap()),
                (
                    "instrumental",
                    &instrumental,
                    d["min_margin_instrumental"].as_f64().unwrap(),
                ),
            ],
        ),
    ];
    for (name, c, tracks, margins) in cases {
        for (label, t, ref_margin) in margins {
            let (frames, got) = native_frames_and_margin(&enc, t, &codebook);
            assert_eq!(
                frames,
                c["codec_frames"].as_u64().unwrap() as usize,
                "{name}/{label}"
            );
            println!(
                "{name}/{label}: smallest codeword margin native {got:.3e}, reference \
                 {ref_margin:.3e}"
            );
        }
        let reference = IclReference {
            tracks,
            start_secs: f32_of(&c["start"]),
            end_secs: f32_of(&c["end"]),
        };
        let codes = enc.encode(&reference, &CancelFlag::new()).expect("encode");
        let want = ids(&c["icl_ids"]);
        let diffs = codes.ids.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!(
            "{name}: {} ids (reference {}), {diffs} differ",
            codes.ids.len(),
            want.len()
        );
        assert_eq!(codes.ids, want, "{name}: windowed ICL ids");
        let chunked_codes = chunked
            .encode(&reference, &CancelFlag::new())
            .expect("chunked encode");
        assert_eq!(
            chunked_codes.ids, want,
            "{name}: windowed ICL ids, 50-frame chunks"
        );

        // The full first-segment prompt the ICL block lands in.
        let prompt = builder
            .build(&PromptInput {
                genres,
                lyrics,
                icl: Some(&codes),
            })
            .unwrap();
        assert_eq!(
            prompt.segments[0].ids,
            ids(&c["segment0_prompt_ids"]),
            "{name}: first-segment prompt"
        );

        // A mutated golden (one id moved to its neighbouring codeword) must fail.
        let mut bad = want.clone();
        bad[want.len() / 2] ^= 1;
        assert_ne!(codes.ids, bad, "{name}: a mutated golden passed");
        let mutated = IclPromptCodes { ids: bad };
        let bad_prompt = builder
            .build(&PromptInput {
                genres,
                lyrics,
                icl: Some(&mutated),
            })
            .unwrap();
        assert_ne!(bad_prompt.segments[0].ids, prompt.segments[0].ids);
    }

    // Upstream's own example reference: 30 s of real music (1 500 frames — past the default
    // 500-frame chunk and the 256-row HuBERT query block), single 0–30 s and dual stems 0–30 s.
    let pop = &fx["pop"];
    let ref_dir = PathBuf::from(std::env::var("YUE_REF_DIR").expect(
        "set YUE_REF_DIR to the YuE reference environment (holds sceneworks-derived/, written by \
         scripts/reference/yue_icl_reference.py)",
    ));
    let clip = |name: &str| -> AudioTrack {
        let meta = &pop["clips"][name];
        let path = ref_dir
            .join("sceneworks-derived")
            .join(format!("{name}.f32le"));
        let raw = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e} — run scripts/reference/yue_icl_reference.py to decode it",
                path.display()
            )
        });
        assert_eq!(
            {
                use sha2::{Digest, Sha256};
                Sha256::digest(&raw)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            },
            meta["pcm_sha256"].as_str().unwrap(),
            "{name}: decoded PCM differs from the reference's"
        );
        AudioTrack {
            samples: raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            sample_rate: u32_of(&meta["rate"]),
            channels: meta["channels"].as_u64().unwrap() as u16,
            stems: Vec::new(),
        }
    };
    let (start_secs, end_secs) = (f32_of(&pop["start"]), f32_of(&pop["end"]));
    let pop_cases = [
        (
            "pop single",
            IclTracks::Single(clip("pop.00001")),
            ids(&pop["single_icl_ids"]),
        ),
        (
            "pop dual",
            IclTracks::Dual {
                vocals: clip("pop.00001.Vocals"),
                instrumental: clip("pop.00001.Instrumental"),
            },
            ids(&pop["dual_icl_ids"]),
        ),
    ];
    for (name, tracks, want) in pop_cases {
        let reference = IclReference {
            tracks,
            start_secs,
            end_secs,
        };
        let got = enc.encode(&reference, &CancelFlag::new()).expect("encode");
        let diffs = got.ids.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!(
            "{name}: {} ids (reference {}), {diffs} differ",
            got.ids.len(),
            want.len()
        );
        assert_eq!(got.ids, want, "{name}: windowed ICL ids");
    }
}
