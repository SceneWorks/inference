//! Audio validation & quality harness (sc-12854) — per-generation-run measurement plus the
//! regression-envelope machinery the per-model fixtures assert against.
//!
//! One harness invocation ([`measure_generation`]) drives a generator closure N times and
//! captures, per run, the full core metric set ([`MetricSet`]):
//!
//! - **latency** — wall time of the generate call ([`MetricSet::latency`]); the
//!   [`HarnessReport`] derives **warmup** (first-run vs steady-state) across runs.
//! - **peak memory** — the best-available process metric on macOS/Linux:
//!   `getrusage(RUSAGE_SELF).ru_maxrss` (see [`peak_rss_bytes`] for the exact semantics).
//! - **output duration / sample rate / channels** — straight off the
//!   [`gen_core::AudioTrack`].
//! - **clipping** — count of samples at/over full scale ([`count_clipped`]).
//! - **integrated loudness (LUFS) + true peak (dBTP)** — measured by
//!   [`gen_core::audio_dsp::measure_track_loudness`] (BS.1770-4 gated loudness, 4× polyphase
//!   true peak). The harness deliberately reuses the contract's DSP — it never reimplements
//!   the meters.
//! - **repeatability hash** — SHA-256 over the canonical PCM byte serialization
//!   ([`pcm_sha256`]), so seeded regression fixtures can pin byte-exact output.
//!
//! # Regression fixtures
//!
//! A per-model fixture commits a deterministic request (script/voice/seed) plus a
//! [`MetricEnvelope`] and the expected repeatability hash — numbers only, never generated
//! media. The fixture test regenerates, measures with this harness, and fails when any run
//! drifts outside the envelope ([`MetricEnvelope::violations`]). The first fixture is Kokoro
//! (`candle-audio-kokoro/tests/regression.rs`).
//!
//! # Deferred quality metrics
//!
//! [`MetricSet::prompt_adherence`] and [`MetricSet::lyric_alignment`] exist as slots but are
//! **deliberately unpopulated**: they require the CLAP audio-text embedding provider
//! (sc-12851) and the ASR lane (sc-12850), neither of which exists in the repository yet.
//! The harness never fakes them — see the field docs.

use std::time::{Duration, Instant};

use gen_core::audio_dsp::measure_track_loudness;
use gen_core::AudioTrack;
use sha2::{Digest, Sha256};

/// The full per-run metric set the harness measures for one generation run.
///
/// Constructed by [`MetricSet::measure`]; all fields are public so regression tests and
/// synthetic drift tests can build and inspect values directly.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricSet {
    /// Wall-clock time of the generate call this run measured.
    pub latency: Duration,
    /// Process peak RSS in bytes at measurement time, per [`peak_rss_bytes`] — `None` where
    /// the mechanism is unavailable (non-unix targets). NOTE: this is the process-lifetime
    /// high-water mark (monotonic, typically set by model load / the warmup run), not a
    /// per-run delta.
    pub peak_rss_bytes: Option<u64>,
    /// Output duration in seconds (`frames / sample_rate`).
    pub duration_secs: f64,
    /// Output sample rate, Hz.
    pub sample_rate: u32,
    /// Output channel count (interleaved layout).
    pub channels: u16,
    /// Number of samples at/over full scale (`|s| >= 1.0`) — see [`count_clipped`].
    pub clipped_samples: usize,
    /// BS.1770-4 gated integrated loudness, LUFS
    /// (from [`gen_core::audio_dsp::measure_track_loudness`]).
    pub integrated_lufs: f32,
    /// 4×-oversampled true peak, dBTP
    /// (from [`gen_core::audio_dsp::measure_track_loudness`]).
    pub true_peak_dbtp: f32,
    /// Lowercase-hex SHA-256 over the canonical PCM bytes — see [`pcm_sha256`].
    pub pcm_sha256: String,
    /// CLAP-based prompt-adherence score. **Deliberately `None` today**: computing it needs
    /// the CLAP audio-text embedding provider tracked as **sc-12851**, which does not exist
    /// in this repository yet. The slot is declared so fixture schemas and reports carry it
    /// from day one; the harness never fabricates a value.
    pub prompt_adherence: Option<f32>,
    /// ASR-based lyric/script alignment score. **Deliberately `None` today**: computing it
    /// needs the ASR lane tracked as **sc-12850**. Same contract as
    /// [`MetricSet::prompt_adherence`] — declared, never faked.
    pub lyric_alignment: Option<f32>,
}

impl MetricSet {
    /// Measure every core metric for one generated track, given the observed wall-clock
    /// `latency` of the generate call. Loudness/true-peak come from
    /// [`gen_core::audio_dsp::measure_track_loudness`], which also validates the track shape
    /// (non-zero rate/channels, whole frames).
    pub fn measure(track: &AudioTrack, latency: Duration) -> gen_core::Result<Self> {
        let loudness = measure_track_loudness(track)?;
        // measure_track_loudness has validated sample_rate > 0 and channels > 0.
        let frames = track.samples.len() / usize::from(track.channels);
        Ok(Self {
            latency,
            peak_rss_bytes: peak_rss_bytes(),
            duration_secs: frames as f64 / f64::from(track.sample_rate),
            sample_rate: track.sample_rate,
            channels: track.channels,
            clipped_samples: count_clipped(&track.samples),
            integrated_lufs: loudness.integrated_lufs,
            true_peak_dbtp: loudness.true_peak_dbtp,
            pcm_sha256: pcm_sha256(&track.samples),
            prompt_adherence: None, // requires CLAP — sc-12851
            lyric_alignment: None,  // requires ASR — sc-12850
        })
    }
}

/// Count samples at/over full scale: `|s| >= 1.0` (a sample exactly at ±1.0 already clips
/// once encoded to integer PCM). NaN never counts (`NaN >= 1.0` is false), matching the
/// track-level finiteness checks providers already assert.
pub fn count_clipped(samples: &[f32]) -> usize {
    samples.iter().filter(|s| s.abs() >= 1.0).count()
}

/// Repeatability hash: lowercase-hex SHA-256 over the canonical PCM byte serialization —
/// each `f32` sample in order, little-endian. Byte-exact regeneration (same model, request,
/// seed, platform) produces the same digest; any sample-level drift changes it.
pub fn pcm_sha256(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for s in samples {
        hasher.update(s.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Best-available process peak-memory metric: `getrusage(RUSAGE_SELF).ru_maxrss`, normalized
/// to bytes. macOS reports `ru_maxrss` in **bytes**; Linux (and other unices) report
/// **kilobytes**. This is the process-lifetime resident-set high-water mark — monotonic and
/// process-wide, so it captures the model + synthesis peak but cannot be reset between runs.
/// Returns `None` on non-unix targets or if the syscall fails.
#[cfg(unix)]
pub fn peak_rss_bytes() -> Option<u64> {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage fills the rusage struct we own; we only read it on rc == 0.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: rc == 0 means the kernel initialized the struct.
    let ru = unsafe { ru.assume_init() };
    let maxrss = u64::try_from(ru.ru_maxrss).ok()?;
    if cfg!(target_os = "macos") {
        Some(maxrss)
    } else {
        Some(maxrss.saturating_mul(1024))
    }
}

/// Non-unix fallback: no portable peak-RSS mechanism is wired here — reports `None`.
#[cfg(not(unix))]
pub fn peak_rss_bytes() -> Option<u64> {
    None
}

/// The measured runs of one harness invocation, with warmup/steady-state accessors.
#[derive(Debug, Clone, PartialEq)]
pub struct HarnessReport {
    /// Per-run metrics in execution order (`runs[0]` is the warmup run).
    pub runs: Vec<MetricSet>,
}

impl HarnessReport {
    /// The first (warmup) run's latency, or `None` if the report is empty.
    pub fn first_run_latency(&self) -> Option<Duration> {
        self.runs.first().map(|m| m.latency)
    }

    /// Steady-state latency: the **median** of every run after the first (robust to a
    /// single outlier). `None` unless at least two runs were measured.
    pub fn steady_latency(&self) -> Option<Duration> {
        let rest = self.runs.get(1..)?;
        if rest.is_empty() {
            return None;
        }
        let mut latencies: Vec<Duration> = rest.iter().map(|m| m.latency).collect();
        latencies.sort_unstable();
        let mid = latencies.len() / 2;
        if latencies.len() % 2 == 1 {
            Some(latencies[mid])
        } else {
            Some((latencies[mid - 1] + latencies[mid]) / 2)
        }
    }

    /// Warmup overhead: first-run latency minus steady-state latency (saturating at zero
    /// when the first run was not the slowest). `None` unless at least two runs.
    pub fn warmup_overhead(&self) -> Option<Duration> {
        Some(
            self.first_run_latency()?
                .saturating_sub(self.steady_latency()?),
        )
    }

    /// The repeatability hash **iff every run produced byte-identical PCM** — `Some(hex)`
    /// when all runs agree, `None` when the report is empty or any run diverged (a seeded
    /// determinism failure).
    pub fn repeatability_hash(&self) -> Option<&str> {
        let first = self.runs.first()?;
        self.runs
            .iter()
            .all(|m| m.pcm_sha256 == first.pcm_sha256)
            .then_some(first.pcm_sha256.as_str())
    }
}

/// Drive `generate` `runs` times (first run = warmup), measuring the full [`MetricSet`] per
/// run. The closure returns the produced [`AudioTrack`] (callers unwrap their provider's
/// `GenerationOutput::Audio` inside it, so load/registry plumbing stays with the caller).
pub fn measure_generation<F>(mut generate: F, runs: usize) -> gen_core::Result<HarnessReport>
where
    F: FnMut() -> gen_core::Result<AudioTrack>,
{
    if runs == 0 {
        return Err(gen_core::Error::Msg(
            "harness: at least one run is required".into(),
        ));
    }
    let mut measured = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let track = generate()?;
        let latency = start.elapsed();
        measured.push(MetricSet::measure(&track, latency)?);
    }
    Ok(HarnessReport { runs: measured })
}

/// The regression envelope a per-model fixture commits: bands for the deterministic seeded
/// request's metrics. [`MetricEnvelope::violations`] reports every metric outside its band —
/// fixtures assert the list is empty, so drift **fails**.
///
/// Prompt-adherence / lyric-alignment carry no envelope fields yet — they gain bands when
/// sc-12851 (CLAP) / sc-12850 (ASR) land and the slots start reporting real scores.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricEnvelope {
    /// Exact expected sample rate, Hz.
    pub sample_rate: u32,
    /// Exact expected channel count.
    pub channels: u16,
    /// Inclusive duration band, seconds.
    pub min_duration_secs: f64,
    /// Inclusive duration band, seconds.
    pub max_duration_secs: f64,
    /// Inclusive integrated-loudness band, LUFS.
    pub min_integrated_lufs: f32,
    /// Inclusive integrated-loudness band, LUFS.
    pub max_integrated_lufs: f32,
    /// Maximum allowed true peak, dBTP.
    pub max_true_peak_dbtp: f32,
    /// Maximum allowed clipped-sample count (0 = non-clipping).
    pub max_clipped_samples: usize,
}

impl MetricEnvelope {
    /// Every way `metrics` drifts outside this envelope, as human-readable findings — empty
    /// means the run conforms. Fixtures assert emptiness so any drift is a test failure.
    pub fn violations(&self, metrics: &MetricSet) -> Vec<String> {
        let mut out = Vec::new();
        if metrics.sample_rate != self.sample_rate {
            out.push(format!(
                "sample_rate {} != expected {}",
                metrics.sample_rate, self.sample_rate
            ));
        }
        if metrics.channels != self.channels {
            out.push(format!(
                "channels {} != expected {}",
                metrics.channels, self.channels
            ));
        }
        if metrics.duration_secs < self.min_duration_secs
            || metrics.duration_secs > self.max_duration_secs
        {
            out.push(format!(
                "duration {:.3}s outside [{:.3}, {:.3}]s",
                metrics.duration_secs, self.min_duration_secs, self.max_duration_secs
            ));
        }
        if metrics.integrated_lufs < self.min_integrated_lufs
            || metrics.integrated_lufs > self.max_integrated_lufs
        {
            out.push(format!(
                "integrated loudness {:.2} LUFS outside [{:.2}, {:.2}] LUFS",
                metrics.integrated_lufs, self.min_integrated_lufs, self.max_integrated_lufs
            ));
        }
        if metrics.true_peak_dbtp > self.max_true_peak_dbtp {
            out.push(format!(
                "true peak {:.2} dBTP over the {:.2} dBTP ceiling",
                metrics.true_peak_dbtp, self.max_true_peak_dbtp
            ));
        }
        if metrics.clipped_samples > self.max_clipped_samples {
            out.push(format!(
                "{} clipped samples over the {} budget",
                metrics.clipped_samples, self.max_clipped_samples
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 440 Hz mono sine at moderate level: 1 s @ 24 kHz, peak 0.25 (≈ -12 dBFS).
    fn sine_track() -> AudioTrack {
        let sample_rate = 24_000u32;
        let samples: Vec<f32> = (0..sample_rate)
            .map(|i| {
                0.25 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();
        AudioTrack {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    #[test]
    fn clipping_counter_counts_at_and_over_full_scale() {
        // At (±1.0) and over (±1.2) count; interior samples and NaN do not.
        let samples = [0.0, 0.5, -0.999, 1.0, -1.0, 1.2, -3.0, f32::NAN];
        assert_eq!(count_clipped(&samples), 4);
        assert_eq!(count_clipped(&[]), 0);
    }

    #[test]
    fn pcm_hash_is_stable_and_sensitive() {
        let track = sine_track();
        // Stability: hashing the same PCM twice is identical.
        let h1 = pcm_sha256(&track.samples);
        let h2 = pcm_sha256(&track.samples);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64, "lowercase-hex SHA-256");
        // Sensitivity: a single-ULP change to a single sample changes the digest.
        let mut drifted = track.samples.clone();
        drifted[12_000] = f32::from_bits(drifted[12_000].to_bits() ^ 1);
        assert_ne!(pcm_sha256(&drifted), h1);
    }

    #[test]
    fn measure_produces_full_metric_set_with_deferred_slots_empty() {
        let track = sine_track();
        let m = MetricSet::measure(&track, Duration::from_millis(125)).expect("measure");
        assert_eq!(m.latency, Duration::from_millis(125));
        assert_eq!(m.sample_rate, 24_000);
        assert_eq!(m.channels, 1);
        assert!((m.duration_secs - 1.0).abs() < 1e-9);
        assert_eq!(m.clipped_samples, 0);
        // The meters come from gen_core::audio_dsp: a -12 dBFS-peak sine is clearly audible
        // and clearly below full scale.
        assert!(m.integrated_lufs > -40.0 && m.integrated_lufs < -5.0);
        assert!(m.true_peak_dbtp < -6.0 && m.true_peak_dbtp > -20.0);
        assert_eq!(m.pcm_sha256, pcm_sha256(&track.samples));
        // Deferred slots exist and are honest: None until sc-12851 (CLAP) / sc-12850 (ASR).
        assert_eq!(m.prompt_adherence, None);
        assert_eq!(m.lyric_alignment, None);
    }

    #[cfg(unix)]
    #[test]
    fn peak_rss_reports_a_plausible_process_high_water_mark() {
        let peak = peak_rss_bytes().expect("getrusage works on unix");
        // Any live Rust test process has resident at least 1 MiB and (sanity ceiling)
        // under 1 TiB — catches unit mistakes (bytes-vs-KB) on both platforms.
        assert!(peak > 1 << 20, "peak {peak} bytes implausibly small");
        assert!(peak < 1 << 40, "peak {peak} bytes implausibly large");
    }

    /// The deliberate-drift proof on synthetic data: the envelope built around the measured
    /// sine passes, and every single drifted metric produces a named violation.
    #[test]
    fn envelope_passes_in_band_and_fails_each_drift() {
        let track = sine_track();
        let m = MetricSet::measure(&track, Duration::from_millis(1)).expect("measure");
        let envelope = MetricEnvelope {
            sample_rate: 24_000,
            channels: 1,
            min_duration_secs: 0.9,
            max_duration_secs: 1.1,
            min_integrated_lufs: m.integrated_lufs - 1.0,
            max_integrated_lufs: m.integrated_lufs + 1.0,
            max_true_peak_dbtp: m.true_peak_dbtp + 0.5,
            max_clipped_samples: 0,
        };
        assert!(envelope.violations(&m).is_empty(), "in-band run conforms");

        // Each drift axis, one at a time, must fail with a finding naming that metric.
        type Drift = (&'static str, fn(&mut MetricSet));
        let drifts: [Drift; 6] = [
            ("sample_rate", |m| m.sample_rate = 22_050),
            ("channels", |m| m.channels = 2),
            ("duration", |m| m.duration_secs = 2.5),
            ("loudness", |m| m.integrated_lufs += 3.0), // hotter than the band
            ("true peak", |m| m.true_peak_dbtp = 0.2),
            ("clipped", |m| m.clipped_samples = 17),
        ];
        for (name, drift) in drifts {
            let mut drifted = m.clone();
            drift(&mut drifted);
            let violations = envelope.violations(&drifted);
            assert_eq!(violations.len(), 1, "{name}: exactly one finding");
            assert!(
                violations[0].contains(name),
                "{name}: finding names the metric ({})",
                violations[0]
            );
        }
    }

    #[test]
    fn report_derives_warmup_and_repeatability() {
        let track = sine_track();
        let base = MetricSet::measure(&track, Duration::ZERO).expect("measure");
        let with_latency = |ms: u64| MetricSet {
            latency: Duration::from_millis(ms),
            ..base.clone()
        };
        // Warmup run at 900 ms, steady runs at 300/320/280 ms → median 300 ms.
        let report = HarnessReport {
            runs: vec![
                with_latency(900),
                with_latency(300),
                with_latency(320),
                with_latency(280),
            ],
        };
        assert_eq!(report.first_run_latency(), Some(Duration::from_millis(900)));
        assert_eq!(report.steady_latency(), Some(Duration::from_millis(300)));
        assert_eq!(report.warmup_overhead(), Some(Duration::from_millis(600)));
        // All four runs hashed identical PCM → the repeatability hash is reported.
        assert_eq!(report.repeatability_hash(), Some(base.pcm_sha256.as_str()));

        // A diverging run kills the repeatability hash (seeded determinism failure).
        let mut diverged = report.clone();
        diverged.runs[2].pcm_sha256 = "deadbeef".into();
        assert_eq!(diverged.repeatability_hash(), None);

        // Single-run and empty reports degrade honestly.
        let single = HarnessReport {
            runs: vec![with_latency(900)],
        };
        assert_eq!(single.steady_latency(), None);
        assert_eq!(single.warmup_overhead(), None);
        assert!(single.repeatability_hash().is_some());
        let empty = HarnessReport { runs: Vec::new() };
        assert_eq!(empty.first_run_latency(), None);
        assert_eq!(empty.repeatability_hash(), None);
    }

    #[test]
    fn measure_generation_runs_the_closure_and_rejects_zero_runs() {
        let track = sine_track();
        let mut calls = 0usize;
        let report = measure_generation(
            || {
                calls += 1;
                Ok(track.clone())
            },
            3,
        )
        .expect("harness runs");
        assert_eq!(calls, 3);
        assert_eq!(report.runs.len(), 3);
        assert!(report.repeatability_hash().is_some());

        assert!(measure_generation(|| Ok(sine_track()), 0).is_err());
    }
}
