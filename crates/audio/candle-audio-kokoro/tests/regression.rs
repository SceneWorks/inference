//! Kokoro-82M regression fixture (sc-12854) — the audio validation harness
//! (`candle_audio::harness`) run against the real provider, asserted against the committed
//! metric envelope + PCM repeatability hash in
//! `tests/fixtures/kokoro_82m_regression.json`.
//!
//! `#[ignore]`d and snapshot-gated exactly like `tests/conformance.rs`: set
//! `KOKORO_SNAPSHOT` to a `hexgrad/Kokoro-82M` snapshot dir, or leave it unset to resolve
//! the pinned snapshot through the audio lane's F-029 hub path.
//!
//! ```text
//! cargo test --locked -p candle-audio-kokoro --test regression -- --ignored --nocapture
//! ```
//!
//! What the fixture pins (numbers only — generated media stays out of git):
//!
//! - **Envelope drift fails**: every harness run's duration, integrated loudness (LUFS),
//!   true peak (dBTP), clipping count, sample rate, and channel count must sit inside the
//!   committed [`MetricEnvelope`] bands.
//! - **Repeatability**: all runs in the process must produce byte-identical PCM, and on the
//!   fixture's canonical platform (os/arch recorded in the fixture — the same class as the
//!   real-weights runner) the SHA-256 must equal the committed hash exactly. On other
//!   platforms the exact-hash check is skipped (Candle CPU kernels are not bit-identical
//!   across architectures) and the observed hash is printed instead; the envelope and
//!   intra-process repeatability still gate.
//!
//! [`MetricEnvelope`]: candle_audio::harness::MetricEnvelope

use std::path::PathBuf;

use candle_audio_kokoro::candle_audio::harness;
use candle_audio_kokoro::gen_core::{
    self, AudioParams, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};

const FIXTURE: &str = include_str!("fixtures/kokoro_82m_regression.json");

/// Resolve the snapshot from the required `KOKORO_SNAPSHOT` env (a passed-in `hexgrad/Kokoro-82M`
/// snapshot dir). Inference never self-fetches or derives a cache location (epic 13657).
fn snapshot() -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(std::env::var("KOKORO_SNAPSHOT").expect(
        "set KOKORO_SNAPSHOT to a hexgrad/Kokoro-82M snapshot dir (config.json + kokoro-v1_0.pth + voices/)",
    )))
}

/// Pull a required field out of the fixture JSON, with a path-labeled panic on schema drift.
fn field<'a>(v: &'a serde_json::Value, path: &[&str]) -> &'a serde_json::Value {
    let mut cur = v;
    for key in path {
        cur = cur
            .get(key)
            .unwrap_or_else(|| panic!("fixture is missing `{}`", path.join(".")));
    }
    cur
}

#[test]
#[ignore = "real weights: needs a hexgrad/Kokoro-82M snapshot (KOKORO_SNAPSHOT or network); run with --ignored"]
fn kokoro_regression_fixture() {
    let fx: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON parses");
    assert_eq!(
        field(&fx, &["model"]).as_str(),
        Some(candle_audio_kokoro::MODEL_ID),
        "fixture targets this provider"
    );
    let runs = field(&fx, &["runs"]).as_u64().expect("runs") as usize;
    let envelope = harness::MetricEnvelope {
        sample_rate: field(&fx, &["envelope", "sample_rate"])
            .as_u64()
            .expect("sample_rate") as u32,
        channels: field(&fx, &["envelope", "channels"])
            .as_u64()
            .expect("channels") as u16,
        min_duration_secs: field(&fx, &["envelope", "min_duration_secs"])
            .as_f64()
            .expect("min_duration_secs"),
        max_duration_secs: field(&fx, &["envelope", "max_duration_secs"])
            .as_f64()
            .expect("max_duration_secs"),
        min_integrated_lufs: field(&fx, &["envelope", "min_integrated_lufs"])
            .as_f64()
            .expect("min_integrated_lufs") as f32,
        max_integrated_lufs: field(&fx, &["envelope", "max_integrated_lufs"])
            .as_f64()
            .expect("max_integrated_lufs") as f32,
        max_true_peak_dbtp: field(&fx, &["envelope", "max_true_peak_dbtp"])
            .as_f64()
            .expect("max_true_peak_dbtp") as f32,
        max_clipped_samples: field(&fx, &["envelope", "max_clipped_samples"])
            .as_u64()
            .expect("max_clipped_samples") as usize,
    };

    // Load through the explicit registry, exactly like conformance.
    let spec = LoadSpec::new(snapshot());
    let generator = candle_audio_kokoro::provider_registry()
        .unwrap()
        .load(candle_audio_kokoro::MODEL_ID, &spec)
        .expect("kokoro_82m loads through the explicit registry");

    let req = GenerationRequest {
        prompt: field(&fx, &["script"]).as_str().expect("script").to_owned(),
        seed: Some(field(&fx, &["seed"]).as_u64().expect("seed")),
        audio: Some(AudioParams {
            voice: Some(field(&fx, &["voice"]).as_str().expect("voice").to_owned()),
            language: Some(
                field(&fx, &["language"])
                    .as_str()
                    .expect("language")
                    .to_owned(),
            ),
            ..Default::default()
        }),
        ..Default::default()
    };

    let report = harness::measure_generation(
        || match generator.generate(&req, &mut |_| {})? {
            GenerationOutput::Audio(track) => Ok(track),
            other => Err(gen_core::Error::Msg(format!(
                "expected GenerationOutput::Audio, got {other:?}"
            ))),
        },
        runs,
    )
    .expect("harness completes every run");

    // Envelope: any drift on any run fails, with every finding named.
    for (i, metrics) in report.runs.iter().enumerate() {
        let violations = envelope.violations(metrics);
        assert!(
            violations.is_empty(),
            "run {i} drifted outside the committed envelope:\n  {}",
            violations.join("\n  ")
        );
        // The deferred quality slots must stay honest until sc-12851 (CLAP) / sc-12850
        // (ASR) exist — a Some here means someone faked a score.
        assert_eq!(metrics.prompt_adherence, None, "CLAP not landed (sc-12851)");
        assert_eq!(metrics.lyric_alignment, None, "ASR not landed (sc-12850)");
    }

    // Intra-process repeatability: every seeded run must be byte-identical.
    let hash = report
        .repeatability_hash()
        .expect("seeded runs produce byte-identical PCM");

    // Exact committed hash on the fixture's canonical platform (the real-weights runner).
    let fx_os = field(&fx, &["repeatability", "os"]).as_str().expect("os");
    let fx_arch = field(&fx, &["repeatability", "arch"])
        .as_str()
        .expect("arch");
    let fx_hash = field(&fx, &["repeatability", "pcm_sha256"])
        .as_str()
        .expect("pcm_sha256");
    if std::env::consts::OS == fx_os && std::env::consts::ARCH == fx_arch {
        assert_eq!(
            hash, fx_hash,
            "PCM repeatability hash drifted from the committed fixture on the canonical \
             platform ({fx_os}/{fx_arch}) — the seeded output changed"
        );
    } else {
        println!(
            "note: platform {}/{} != fixture canonical {fx_os}/{fx_arch}; exact-hash check \
             skipped (observed hash {hash})",
            std::env::consts::OS,
            std::env::consts::ARCH,
        );
    }

    // The measured evidence, for envelope review and re-baselining.
    let first = &report.runs[0];
    println!(
        "kokoro_regression_fixture: {runs} runs | first-run latency {:.3}s, steady {:?}, \
         warmup overhead {:?} | peak RSS {} | duration {:.3}s @ {} Hz x{} | \
         {:.2} LUFS, {:.2} dBTP, {} clipped | hash {hash}",
        first.latency.as_secs_f64(),
        report.steady_latency(),
        report.warmup_overhead(),
        first
            .peak_rss_bytes
            .map_or("unavailable".to_owned(), |b| format!(
                "{:.1} MiB",
                b as f64 / (1024.0 * 1024.0)
            )),
        first.duration_secs,
        first.sample_rate,
        first.channels,
        first.integrated_lufs,
        first.true_peak_dbtp,
        first.clipped_samples,
    );
}
