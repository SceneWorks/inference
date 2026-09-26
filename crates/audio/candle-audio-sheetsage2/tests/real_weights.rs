//! Real-weight parity of the native SheetSage2 / MERT-v2-FullSong port (sc-22996).
//!
//! `#[ignore]`d in ordinary runs (CI has no weights); under `--ignored` a missing environment
//! variable panics rather than silently passing. Nothing is downloaded.
//!
//! * `YUE2_HF_HUB` — a Hugging Face hub directory (holding `models--m-a-p--*/`) with the pinned
//!   `m-a-p/SheetSage2@eab522a…` and `m-a-p/MERT-v2-FullSong@d8ba1c7…` snapshots.
//! * `SHEETSAGE2_MODEL_INPUTS` — the sc-23003 fixtures directory holding the digest-pinned 24 kHz
//!   mono float32 model-input arrays (`run_experiment.py fixtures`); each is checked against
//!   `artifacts/fixtures.json` before use. They are fed as-is (never re-decoded).
//! * `SHEETSAGE2_PARITY_DUMPS` — the `native_parity/` directory written by
//!   `native_parity.py real` (per-layer reference hidden states; only the encoder test needs it).
//!
//! ```text
//! YUE2_HF_HUB=… SHEETSAGE2_MODEL_INPUTS=… SHEETSAGE2_PARITY_DUMPS=… \
//!   cargo test --release -p candle-audio-sheetsage2 --test real_weights -- --ignored \
//!   --nocapture --test-threads 1
//! ```
//!
//! CPU only, float32. Hidden states are compared by max-abs and relative error (max-abs over the
//! reference's max-abs); tokens and every symbolic artifact must match exactly.

use std::path::PathBuf;
use std::time::Instant;

use candle_audio_sheetsage2::candle_core::{Device, Tensor};
use candle_audio_sheetsage2::events::parse_tokens_txt;
use candle_audio_sheetsage2::provider::{live_models, SourceAudio, Transcriber};
use candle_audio_sheetsage2::review::{Readiness, ReviewArtifact, TranscriptionSettings};
use candle_audio_yue2::inventory;
use candle_audio_yue2::snapshot::resolve_closure;
use candle_audio_yue2::{Closure, SnapshotDirs};
use sha2::{Digest, Sha256};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("real-weight test run without {name} (see the module docs)")),
    )
}

fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../scripts/reference/sheetsage2/artifacts")
}

fn hub() -> SnapshotDirs {
    let hub = env_dir("YUE2_HF_HUB");
    inventory::REPOS
        .iter()
        .fold(SnapshotDirs::new(), |dirs, repo| {
            dirs.with(
                repo.id,
                hub.join(format!("models--{}", repo.id.replace('/', "--")))
                    .join("snapshots")
                    .join(repo.revision),
            )
        })
}

fn sha256_f32(samples: &[f32]) -> String {
    let mut h = Sha256::new();
    for s in samples {
        h.update(s.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// A digest-pinned model-input array (`nav_ssb`, `synth`, `synth_eb`).
fn model_input(name: &str) -> Vec<f32> {
    let fixtures: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts().join("fixtures.json")).unwrap()).unwrap();
    let entry = &fixtures[name]["model_input"];
    let path = env_dir("SHEETSAGE2_MODEL_INPUTS").join(entry["file"].as_str().unwrap());
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(samples.len() as u64, entry["samples"].as_u64().unwrap());
    assert_eq!(
        sha256_f32(&samples),
        entry["sha256"].as_str().unwrap(),
        "{name}"
    );
    samples
}

fn rss_mib() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<f64>()
        .unwrap()
        / 1024.0
}

fn load() -> Transcriber {
    let start = Instant::now();
    let closure = resolve_closure(Closure::Cover, &hub()).unwrap_or_else(|e| panic!("{e}"));
    let verified = start.elapsed();
    let transcriber = Transcriber::load(&closure, &Device::Cpu).unwrap();
    println!(
        "closure verified in {verified:.1?}, model loaded in {:.1?} ({:.2} GB parameters), RSS {:.0} MiB",
        start.elapsed() - verified,
        transcriber.model().parameter_bytes() as f64 / 1e9,
        rss_mib()
    );
    transcriber
}

fn error(ours: &Tensor, theirs: &Tensor) -> (f32, f32) {
    let a = ours.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = theirs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(a.len(), b.len());
    let max_abs = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    (max_abs, max_abs / scale)
}

/// Every MERT2 hidden state (mel, subsampler, 24 blocks), the layer mix, the decoder memory and
/// the first-step logits against upstream's torch CPU fp32 run on the same arrays.
///
/// Tolerances (relative = max-abs over the reference's max-abs), from the values this test
/// measured on aarch64 CPU (2026-09-26), with ~2x headroom:
///
/// | state | synth (measured) | nav_ssb (measured) | bound |
/// |---|---|---|---|
/// | mel | 4.0e-4 | 1.9e-5 | 1e-3 |
/// | subsampler + 24 blocks, worst | 5.8e-3 (block 15) | 5.1e-4 (block 15) | 1.5e-2 |
/// | decoder memory | 8.3e-3 | 4.2e-4 | 1.5e-2 |
/// | first-step logits | 4.2e-6 | 9.7e-7 | 1e-4 |
///
/// The synthetic clip is ~15x looser than the recording: its spectra have near-empty bands and an
/// exact-zero tail, where the float32 FFT noise floors of this port and torch's pocketfft differ in
/// dB. The drift stays below every greedy decision: the token tests below are exact.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB, SHEETSAGE2_MODEL_INPUTS, SHEETSAGE2_PARITY_DUMPS"]
fn encoder_states_match_the_reference() {
    let transcriber = load();
    let model = transcriber.model();
    let dumps = env_dir("SHEETSAGE2_PARITY_DUMPS");
    for name in ["synth", "nav_ssb"] {
        let reference = candle_audio_sheetsage2::candle_core::safetensors::load(
            dumps.join(format!("{name}.safetensors")),
            &Device::Cpu,
        )
        .unwrap();
        let samples = model_input(name);
        let start = Instant::now();
        let features = model.audio_features(&samples).unwrap();
        println!(
            "{name}: encoder {:.1?}, RSS {:.0} MiB",
            start.elapsed(),
            rss_mib()
        );
        let get = |k: &str| reference[k].unsqueeze(0).unwrap();
        let mut worst = 0.0f32;
        let (m_abs, m_rel) = error(&features.mel, &get("mel"));
        println!("  mel: max_abs {m_abs:.3e} rel {m_rel:.3e}");
        assert!(m_rel <= 1e-3, "{name} mel: relative {m_rel:.3e}");
        let mut states = vec![("input_hidden".to_string(), features.input_hidden.clone())];
        for (i, b) in features.blocks.iter().enumerate() {
            states.push((format!("block.{i}"), b.clone()));
        }
        states.push(("mixed".into(), features.mixed.clone()));
        states.push(("memory".into(), features.memory.clone()));
        for (key, ours) in &states {
            let (abs, rel) = error(ours, &get(key));
            println!("  {key}: max_abs {abs:.3e} rel {rel:.3e}");
            worst = worst.max(rel);
            assert!(rel <= 1.5e-2, "{name} {key}: relative {rel:.3e}");
        }
        // First decoder step over the prompt prefix: the last position's raw logits.
        let prefix: Vec<u32> = [1, 4, 5, 6, 7, 9, 11, 3].to_vec();
        let mut first = None;
        model
            .generate(&features.memory, &prefix, prefix.len() + 1, None, |s| {
                first.get_or_insert_with(|| s.logits.to_vec());
                Ok(())
            })
            .unwrap();
        let theirs = reference["prefix_logits"]
            .narrow(0, prefix.len() - 1, 1)
            .unwrap();
        let ours = Tensor::new(first.unwrap(), &Device::Cpu).unwrap();
        let (abs, rel) = error(&ours, &theirs);
        println!("  prefix logits: max_abs {abs:.3e} rel {rel:.3e}; worst hidden rel {worst:.3e}");
        assert!(rel <= 1e-4, "{name} prefix logits: relative {rel:.3e}");
    }
    let receipt = transcriber.unload();
    assert!(receipt.released);
}

fn committed_tokens(case: &str) -> String {
    std::fs::read_to_string(artifacts().join(case).join("tokens.txt")).unwrap()
}

fn check_case(
    transcriber: &Transcriber,
    case: &str,
    samples: Vec<f32>,
) -> candle_audio_sheetsage2::review::Transcription {
    let start = Instant::now();
    let rss_before = rss_mib();
    let t = transcriber
        .transcribe(
            &SourceAudio::mono(samples),
            &TranscriptionSettings::default(),
            |_| Ok(()),
        )
        .unwrap();
    let ours: Vec<Vec<u32>> = t
        .stitched
        .records
        .iter()
        .map(|r| r.tokens.clone())
        .collect();
    let theirs: Vec<Vec<u32>> = parse_tokens_txt(&committed_tokens(case))
        .unwrap()
        .into_iter()
        .map(|r| r.tokens)
        .collect();
    println!(
        "{case}: {} window(s), {} tokens in {:.1?} (RSS {:.0} → {:.0} MiB)",
        ours.len(),
        ours.iter().map(Vec::len).sum::<usize>(),
        start.elapsed(),
        rss_before,
        rss_mib()
    );
    println!(
        "  peak RSS so far {:.0} MiB",
        candle_audio::harness::peak_rss_bytes().unwrap_or(0) as f64 / (1024.0 * 1024.0)
    );
    for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        if a != b {
            let at = a
                .iter()
                .zip(b)
                .position(|(x, y)| x != y)
                .unwrap_or(a.len().min(b.len()));
            panic!(
                "{case} window {i}: first token difference at index {at}: ours {:?} theirs {:?}",
                a.get(at),
                b.get(at)
            );
        }
    }
    assert_eq!(ours.len(), theirs.len(), "{case}: window count");
    let committed_abc = std::fs::read_to_string(artifacts().join(case).join("score.abc")).unwrap();
    assert_eq!(
        t.full.abc.as_deref(),
        Some(committed_abc.as_str()),
        "{case}: score.abc"
    );
    t
}

/// Token-exact greedy decoding through the production provider on the digest-pinned arrays, and
/// byte-identical scores: the synthetic lead sheet, the public-domain recording (full and
/// melody-only), and the Eb-major clip (head-revision chord spelling). The real recording's review
/// carries the octave and harmony warnings, and its octave evidence reproduces sc-23003's numbers.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB, SHEETSAGE2_MODEL_INPUTS"]
fn transcriptions_match_the_committed_token_oracles() {
    let transcriber = load();
    check_case(&transcriber, "synth_full", model_input("synth"));
    check_case(&transcriber, "synth_eb_head", model_input("synth_eb"));
    let real = check_case(&transcriber, "real_full", model_input("nav_ssb"));
    let melody = std::fs::read_to_string(artifacts().join("real_melody/score.abc")).unwrap();
    assert_eq!(real.melody_abc.as_deref(), Some(melody.as_str()));
    let codes: Vec<&str> = real
        .review
        .warnings
        .iter()
        .map(|w| w.code.as_str())
        .collect();
    println!("real_full warnings: {codes:?}");
    assert!(codes.contains(&"harmony_collapsed"));
    assert!(codes.contains(&"octave_f0_half_evidence"));

    // The octave evidence equals sc-23003's committed check (same array, same notes).
    let evaluation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifacts().join("evaluation.json")).unwrap())
            .unwrap();
    let expected = &evaluation["real_clip_octave_check"];
    let octave = real.octave.as_ref().unwrap();
    assert_eq!(
        octave.notes.len() as u64,
        expected["notes_checked"].as_u64().unwrap()
    );
    assert_eq!(
        octave.f0_half_dominant() as u64,
        expected["notes_with_more_energy_at_f0_half"]
            .as_u64()
            .unwrap()
    );
    for (ours, theirs) in octave
        .notes
        .iter()
        .zip(expected["notes"].as_array().unwrap())
    {
        for (value, key) in [
            (ours.energy_f0_half, "energy_f0_half"),
            (ours.energy_f0, "energy_f0"),
            (ours.energy_2f0, "energy_2f0"),
        ] {
            let t = theirs[key].as_f64().unwrap();
            assert!(
                (value - t).abs() <= 1e-3 + 1e-4 * t.abs(),
                "{key}: {value} vs {t}"
            );
        }
    }

    // Persist, reopen, replay.
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("real");
    real.save(&dir, transcriber.model().tokenizer()).unwrap();
    let artifact = ReviewArtifact::open(&dir).unwrap();
    let report = artifact.replay().unwrap();
    println!("replayed {report:?}");
    assert_eq!(artifact.readiness(true), Readiness::Ready);
    let receipt = transcriber.unload();
    assert!(receipt.released);
}

/// The >300 s multi-window path: a 358.2 s array concatenated from the pinned arrays (recipe and
/// digest in `artifacts/long_multiwindow/case.json`), transcribed natively; both windows, the
/// 1,124-token overlap prefix and the stitched score must equal upstream's.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB, SHEETSAGE2_MODEL_INPUTS"]
fn multi_window_song_matches_upstream() {
    let case: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifacts().join("long_multiwindow/case.json")).unwrap(),
    )
    .unwrap();
    let mut samples = Vec::new();
    for part in case["input"]["recipe"]["order"].as_array().unwrap() {
        samples.extend(model_input(part.as_str().unwrap()));
    }
    assert_eq!(
        sha256_f32(&samples),
        case["input"]["sha256"].as_str().unwrap()
    );
    let transcriber = load();
    let t = check_case(&transcriber, "long_multiwindow", samples);
    assert_eq!(t.stitched.records[1].prefix_tokens, 1124);
    transcriber.unload();
}

/// Digital silence: upstream's tokens (a beat grid, no melody) are reproduced, and the review
/// refuses both cover modes where upstream silently returns a rest-only score.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB, SHEETSAGE2_MODEL_INPUTS"]
fn silence_is_refused_as_a_cover_source() {
    let transcriber = load();
    let t = check_case(&transcriber, "silence", vec![0.0; 12 * 24_000]);
    assert!(matches!(t.review.melody_cover, Readiness::Refused(_)));
    assert!(matches!(t.review.full_cover, Readiness::Refused(_)));
    transcriber.unload();
}

/// Unloading returns the weights' memory to the system, measured as process RSS.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB"]
fn unload_returns_the_weights_memory() {
    let baseline = rss_mib();
    let transcriber = load();
    let loaded = rss_mib();
    let bytes = transcriber.model().parameter_bytes();
    assert_eq!(live_models(), 1);
    let receipt = transcriber.unload();
    let after = rss_mib();
    println!(
        "RSS baseline {baseline:.0} MiB, loaded {loaded:.0} MiB, after unload {after:.0} MiB \
         ({:.2} GB parameters)",
        bytes as f64 / 1e9
    );
    assert!(receipt.released);
    assert_eq!(live_models(), 0);
    // At least 90 % of the parameter bytes left the process.
    assert!(
        loaded - after >= 0.9 * bytes as f64 / (1024.0 * 1024.0),
        "{loaded} → {after}"
    );
}
