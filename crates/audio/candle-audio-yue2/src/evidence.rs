//! The machine-readable summary line the real-weight harnesses print (sc-23002).
//!
//! `quality::tier_quality_against_the_f32_reference` (one line per measured configuration) and
//! `tests/engine_real_weights.rs`' registered-loader render each print
//!
//! ```text
//! YUE2_EVIDENCE_SUMMARY {"schema":"yue2-evidence-summary-v1","test":…,"device":…,"dtype":…}
//! ```
//!
//! — [`SUMMARY_PREFIX`] followed by one compact JSON object that runs to the end of the line. A
//! test's other output can share the line before the prefix (libtest prints a test's name and its
//! captured output on one line), so a scraper finds the prefix anywhere in a line and parses
//! everything after it. Every record carries [`SUMMARY_SCHEMA`], the test, the backend, device and
//! compute dtype, the random streams (epic E9), per-stage wall times, the host's peak resident set
//! where the platform reports one ([`peak_rss_bytes`]), each output's [`audio_record`] and the
//! truncation flags. Device-memory peaks are sampled outside the process (`nvidia-smi` in
//! `real-weights-yue.yml`), so they are not in the line.

use serde_json::{json, Value};

/// What starts the summary line's JSON (a trailing space separates it from the object).
pub const SUMMARY_PREFIX: &str = "YUE2_EVIDENCE_SUMMARY ";

/// The record's schema; a change to any field's meaning bumps it.
pub const SUMMARY_SCHEMA: &str = "yue2-evidence-summary-v1";

/// `record` (a JSON object) as one summary line: [`SUMMARY_PREFIX`] and the object, compact, with
/// [`SUMMARY_SCHEMA`] as its `schema`. Compact JSON escapes every control character inside a
/// string, so the line never breaks.
///
/// # Panics
///
/// If `record` is not a JSON object (a harness bug, not a runtime condition).
pub fn summary_line(record: Value) -> String {
    let Value::Object(mut map) = record else {
        panic!("a YuE2 evidence summary is a JSON object");
    };
    map.insert("schema".into(), json!(SUMMARY_SCHEMA));
    format!("{SUMMARY_PREFIX}{}", Value::Object(map))
}

/// Duration, loudness and repeatability hash of interleaved 48 kHz stereo `samples`: `seconds`
/// (frames / 48 kHz), `rms`, `peak_abs`, `samples` (the count) and `sha256` (over each `f32`
/// little-endian, [`candle_audio::harness::pcm_sha256`] — the same digest the other audio
/// harnesses record).
pub fn audio_record(samples: &[f32]) -> Value {
    let frames = samples.len() / crate::vae::AUDIO_CHANNELS;
    let (sum_sq, peak) = samples.iter().fold((0f64, 0f64), |(s, p), &v| {
        let v = v as f64;
        (s + v * v, p.max(v.abs()))
    });
    json!({
        "seconds": frames as f64 / crate::vae::SAMPLE_RATE as f64,
        "rms": (sum_sq / samples.len().max(1) as f64).sqrt(),
        "peak_abs": peak,
        "samples": samples.len(),
        "sha256": candle_audio::harness::pcm_sha256(samples),
    })
}

/// The process's peak resident set in bytes (`getrusage` `ru_maxrss`; process-lifetime, so a
/// later reading covers every earlier stage), or `None` where the platform reports none (Windows).
pub fn peak_rss_bytes() -> Option<u64> {
    candle_audio::harness::peak_rss_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scraper's view: the object after the prefix, wherever the prefix sits in the line.
    fn scrape(line: &str) -> Value {
        let at = line.find(SUMMARY_PREFIX).expect("the prefix");
        serde_json::from_str(&line[at + SUMMARY_PREFIX.len()..]).unwrap()
    }

    #[test]
    fn a_summary_is_one_scrapable_line() {
        let record = json!({
            "test": "t",
            "device": "metal",
            "note": "two\nlines",
            "seconds": {"load": 1.5},
        });
        let line = summary_line(record.clone());
        assert!(!line.contains('\n'), "{line}");
        assert!(line.starts_with(SUMMARY_PREFIX));
        let got = scrape(&format!("test quality::x ... {line}"));
        assert_eq!(got["schema"], SUMMARY_SCHEMA);
        assert_eq!(got["note"], "two\nlines");
        assert_eq!(got["seconds"], record["seconds"]);
    }

    #[test]
    fn an_audio_record_measures_interleaved_stereo() {
        // One second of 48 kHz stereo: L = 0.5, R = -0.5.
        let samples: Vec<f32> = (0..96_000)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let r = audio_record(&samples);
        assert_eq!(r["seconds"], 1.0);
        assert_eq!(r["rms"], 0.5);
        assert_eq!(r["peak_abs"], 0.5);
        assert_eq!(r["samples"], 96_000);
        assert_eq!(r["sha256"], candle_audio::harness::pcm_sha256(&samples));
        assert_eq!(audio_record(&[])["seconds"], 0.0);
    }
}
