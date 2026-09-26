//! The transcription review artifact: what a reviewer (and the cover path) needs, persisted so it
//! can be reviewed later and **replayed** — re-derived from the exact tokens, settings and model
//! input, byte for byte, without the model.
//!
//! A persisted artifact is only ever used through [`ReviewArtifact::open`], which verifies it
//! completely: every file against its recorded SHA-256, the source identity against the persisted
//! model input ([`SOURCE_AUDIO`]), the closure identity against the pinned inventory, and **every
//! manifest field** — review, cover readiness, warnings, ABC errors, note counts, octave evidence —
//! against what replay re-derives. A manifest edited by hand (a refusal flipped to ready, warnings
//! removed, a different source digest) is refused. The source `name`, `original_sha256` and
//! `conversion` are caller-supplied labels and cannot be re-derived.
//!
//! What the review adds to upstream's outputs, all visible and none silently "fixed":
//!
//! * **Empty melody is a refusal.** Upstream turns digital silence into a rest-only score with no
//!   warning. Here a transcription with no melody notes is marked not cover-ready for either mode,
//!   with the reason; the partial artifacts (beats, key, structure, the rest-only score) are kept.
//! * **ABC failures are recorded, not raised.** Each mode's `abc_error` is kept next to every other
//!   artifact, which is still written.
//! * **Octave evidence**: per-voice ranges, and the spectral f0/2 check of every vocal note against
//!   the source audio ([`crate::octave`]). A high vocal register or f0/2-dominant notes produce a
//!   warning; pitches are never shifted.
//! * **Harmony collapse**, **voice assignment** (melody only in `Ins`), **sparse melody**,
//!   recovered decodes and token-limit hits are warnings.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::events::tokens_txt;
use crate::exports::{export, Exports};
use crate::octave::OctaveEvidence;
use crate::pipeline::{Stitched, Stitcher};
use crate::tokenizer::Tokenizer;
use crate::Error;

/// Manifest file name.
pub const MANIFEST: &str = "transcription.json";
/// Manifest schema.
pub const SCHEMA: &str = "sceneworks-sheetsage2-transcription-v1";
/// The melody-only score's file name (the full score keeps upstream's `score.abc`).
pub const MELODY_SCORE: &str = "score_melody.abc";
/// The exact per-window tokens.
pub const TOKENS_JSON: &str = "tokens.json";
/// The exact model input, float32 little-endian, raw (so the source identity and the octave
/// evidence can be re-derived on replay).
pub const SOURCE_AUDIO: &str = "source.f32le";

/// Fewer melody notes than this is a sparse-melody warning.
pub const SPARSE_MELODY_NOTES: usize = 8;
/// A vocal median above this MIDI pitch (E5) is a high-register warning.
pub const HIGH_VOCAL_MEDIAN: f64 = 76.0;
/// A share of f0/2-dominant vocal notes above this (with enough notes checked) is a warning.
pub const OCTAVE_EVIDENCE_FRACTION: f64 = 0.5;
/// Minimum checked notes before the f0/2 share is reported as a warning.
pub const OCTAVE_EVIDENCE_MIN_NOTES: usize = 8;

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// What was transcribed.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceIdentity {
    /// SHA-256 of the exact 24 kHz mono float32 little-endian model input.
    pub sha256: String,
    /// Samples of the model input.
    pub samples: usize,
    /// Always 24,000.
    pub sample_rate: u32,
    /// Display name, if the caller gave one.
    pub name: Option<String>,
    /// SHA-256 of the original encoded file, if the caller supplied it.
    pub original_sha256: Option<String>,
    /// How the model input was derived from the caller's samples.
    pub conversion: String,
}

impl SourceIdentity {
    /// Seconds.
    pub fn duration(&self) -> f64 {
        self.samples as f64 / f64::from(self.sample_rate)
    }

    fn to_json(&self) -> Value {
        json!({
            "sha256": self.sha256,
            "samples": self.samples,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration(),
            "name": self.name,
            "original_sha256": self.original_sha256,
            "conversion": self.conversion,
        })
    }

    fn from_json(v: &Value) -> Result<Self, Error> {
        let s = |k: &str| v[k].as_str().map(str::to_string);
        Ok(Self {
            sha256: s("sha256").ok_or_else(|| bad("source.sha256"))?,
            samples: v["samples"].as_u64().ok_or_else(|| bad("source.samples"))? as usize,
            sample_rate: v["sample_rate"]
                .as_u64()
                .ok_or_else(|| bad("source.sample_rate"))? as u32,
            name: s("name"),
            original_sha256: s("original_sha256"),
            conversion: s("conversion").ok_or_else(|| bad("source.conversion"))?,
        })
    }
}

/// Transcription settings (everything that changes the output for the same source).
#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptionSettings {
    /// Task prompts (default: upstream's full set).
    pub prompts: Vec<String>,
    /// Overlap between consecutive 300 s windows, seconds.
    pub overlap_seconds: f64,
    /// Right-hand look-ahead of each non-final window, seconds.
    pub lookahead_seconds: f64,
    /// Crop the source to this many seconds before transcribing.
    pub max_seconds: Option<f64>,
}

impl Default for TranscriptionSettings {
    fn default() -> Self {
        Self {
            prompts: crate::tokenizer::FULL_TASK_PROMPTS
                .iter()
                .map(|p| p.to_string())
                .collect(),
            overlap_seconds: crate::pipeline::DEFAULT_OVERLAP_SECONDS,
            lookahead_seconds: crate::pipeline::DEFAULT_LOOKAHEAD_SECONDS,
            max_seconds: None,
        }
    }
}

impl TranscriptionSettings {
    fn to_json(&self) -> Value {
        json!({
            "prompts": self.prompts,
            "overlap_seconds": self.overlap_seconds,
            "lookahead_seconds": self.lookahead_seconds,
            "max_seconds": self.max_seconds,
            "preset": "default",
            "precision": "float32",
        })
    }

    fn from_json(v: &Value) -> Result<Self, Error> {
        Ok(Self {
            prompts: v["prompts"]
                .as_array()
                .ok_or_else(|| bad("settings.prompts"))?
                .iter()
                .map(|p| {
                    p.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| bad("settings.prompts"))
                })
                .collect::<Result<_, _>>()?,
            overlap_seconds: v["overlap_seconds"]
                .as_f64()
                .ok_or_else(|| bad("settings.overlap_seconds"))?,
            lookahead_seconds: v["lookahead_seconds"]
                .as_f64()
                .ok_or_else(|| bad("settings.lookahead_seconds"))?,
            max_seconds: v["max_seconds"].as_f64(),
        })
    }
}

/// Which pinned closure, code revision and vocabulary produced a transcription.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosureIdentity {
    /// `(repo, revision, weights sha256, config sha256)` of SheetSage2.
    pub sheetsage2: [String; 4],
    /// `(repo, revision, weights sha256, config sha256)` of MERT-v2-FullSong.
    pub mert: [String; 4],
    /// The upstream code revision the post-processing is ported from.
    pub ported_code_revision: String,
    /// The vocabulary fingerprint.
    pub tokenizer_fingerprint: String,
    /// The device the model ran on (`cpu`, `metal`, `cuda`).
    pub device: String,
}

impl ClosureIdentity {
    fn to_json(&self) -> Value {
        let repo = |r: &[String; 4]| json!({"repo": r[0], "revision": r[1], "weights_sha256": r[2], "config_sha256": r[3]});
        json!({
            "sheetsage2": repo(&self.sheetsage2),
            "mert_v2_fullsong": repo(&self.mert),
            "ported_code_revision": self.ported_code_revision,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "device": self.device,
            "licence": "CC-BY-NC-4.0 weights and derived code; noncommercial experimentation only",
        })
    }

    fn from_json(v: &Value) -> Result<Self, Error> {
        let repo = |r: &Value| -> Result<[String; 4], Error> {
            let s = |k: &str| {
                r[k].as_str()
                    .map(str::to_string)
                    .ok_or_else(|| bad("closure"))
            };
            Ok([
                s("repo")?,
                s("revision")?,
                s("weights_sha256")?,
                s("config_sha256")?,
            ])
        };
        let s = |k: &str| {
            v[k].as_str()
                .map(str::to_string)
                .ok_or_else(|| bad("closure"))
        };
        Ok(Self {
            sheetsage2: repo(&v["sheetsage2"])?,
            mert: repo(&v["mert_v2_fullsong"])?,
            ported_code_revision: s("ported_code_revision")?,
            tokenizer_fingerprint: s("tokenizer_fingerprint")?,
            device: s("device")?,
        })
    }
}

fn bad(field: &str) -> Error {
    Error::Replay(format!("manifest field `{field}` is missing or malformed"))
}

/// A review warning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Warning {
    /// Stable machine code.
    pub code: String,
    /// What the reviewer should know.
    pub message: String,
}

/// Summary of one melody voice.
#[derive(Clone, Debug, PartialEq)]
pub struct VoiceSummary {
    /// Notes.
    pub notes: usize,
    /// Lowest MIDI pitch.
    pub min_pitch: Option<i64>,
    /// Highest MIDI pitch.
    pub max_pitch: Option<i64>,
    /// Median MIDI pitch.
    pub median_pitch: Option<f64>,
}

/// Whether a cover mode can use this transcription.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readiness {
    /// Usable (after review).
    Ready,
    /// Refused, with the reason.
    Refused(String),
}

/// The reviewer's view of a transcription.
#[derive(Clone, Debug, PartialEq)]
pub struct Review {
    /// Vocal (`[0]`) and instrumental (`[1]`) melody summaries.
    pub voices: [VoiceSummary; 2],
    /// Distinct key-corrected chord labels.
    pub distinct_chords: Vec<String>,
    /// Distinct chord roots (pitch classes).
    pub distinct_chord_roots: usize,
    /// Keys, in order of appearance.
    pub keys: Vec<String>,
    /// Section labels, in order.
    pub sections: Vec<String>,
    /// Bars of the full score (0 when it failed).
    pub bars: usize,
    /// Warnings.
    pub warnings: Vec<Warning>,
    /// Notation diagnostics (padded bars, inferred meters, clipped overlaps).
    pub diagnostics: Vec<String>,
    /// `cot=melody` readiness.
    pub melody_cover: Readiness,
    /// `cot=full` readiness.
    pub full_cover: Readiness,
}

fn summarize(pitches: &[i64]) -> VoiceSummary {
    let mut sorted = pitches.to_vec();
    sorted.sort_unstable();
    let median = (!sorted.is_empty()).then(|| {
        let n = sorted.len();
        if n % 2 == 1 {
            sorted[n / 2] as f64
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) as f64 / 2.0
        }
    });
    VoiceSummary {
        notes: pitches.len(),
        min_pitch: sorted.first().copied(),
        max_pitch: sorted.last().copied(),
        median_pitch: median,
    }
}

impl Review {
    /// Assess a finished transcription.
    pub fn assess(
        stitched: &Stitched,
        full: &Exports,
        melody_abc_error: Option<&str>,
        octave: Option<&OctaveEvidence>,
    ) -> Self {
        let mut warnings = Vec::new();
        let mut warn = |code: &str, message: String| {
            warnings.push(Warning {
                code: code.to_string(),
                message,
            })
        };
        for w in &stitched.warnings {
            if w.contains("token limit") {
                warn("token_limit", w.clone());
            } else {
                warn(
                    "decode_recovered",
                    format!("strict decode failed; recovered in non-strict decode: {w}"),
                );
            }
        }
        let vocal: Vec<i64> = full
            .notes
            .iter()
            .filter(|n| n.track == 0)
            .map(|n| n.pitch)
            .collect();
        let ins: Vec<i64> = full
            .notes
            .iter()
            .filter(|n| n.track == 1)
            .map(|n| n.pitch)
            .collect();
        let voices = [summarize(&vocal), summarize(&ins)];
        let total = full.notes.len();

        let empty = total == 0;
        if empty {
            warn(
                "empty_melody",
                "no melody notes were transcribed (silence, or no melody the model could hear); \
                 upstream would still emit a rest-only score — it is kept for review but refused \
                 as a cover plan"
                    .into(),
            );
        } else if total < SPARSE_MELODY_NOTES {
            warn(
                "sparse_melody",
                format!("only {total} melody notes were transcribed; review before covering"),
            );
        }
        if vocal.is_empty() && !ins.is_empty() {
            warn(
                "melody_only_in_ins",
                "every melody note was assigned to the Ins (instrumental) voice and none to Vocal; \
                 YuE2 conditions the two voices separately — review the voice assignment"
                    .into(),
            );
        }
        if let Some(median) = voices[0].median_pitch {
            if median > HIGH_VOCAL_MEDIAN {
                warn(
                    "octave_high_register",
                    format!(
                        "the vocal line's median pitch is MIDI {median} (above E5); SheetSage2 can \
                         transcribe real vocals an octave high — check the register before covering"
                    ),
                );
            }
        }
        if let Some(evidence) = octave {
            if let Some(fraction) = evidence.fraction_f0_half_dominant() {
                if evidence.notes.len() >= OCTAVE_EVIDENCE_MIN_NOTES
                    && fraction > OCTAVE_EVIDENCE_FRACTION
                {
                    warn(
                        "octave_f0_half_evidence",
                        format!(
                            "{} of {} checked vocal notes have more energy one octave below the \
                             transcribed pitch (heuristic; accompaniment can cause it) — the vocal \
                             line may be written an octave high",
                            evidence.f0_half_dominant(),
                            evidence.notes.len()
                        ),
                    );
                }
            }
        }
        let mut distinct_chords: Vec<String> = Vec::new();
        let mut roots: Vec<i64> = Vec::new();
        for (_, _, label) in &full.chords {
            if label == "N" || label == "X" {
                continue;
            }
            if !distinct_chords.contains(label) {
                distinct_chords.push(label.clone());
            }
            if let Some(root) = label
                .split(':')
                .next()
                .and_then(|r| crate::chord_spelling::pitch_class_to_semitone(r).ok())
            {
                if !roots.contains(&root) {
                    roots.push(root);
                }
            }
        }
        let bars = full.score.as_ref().map_or(0, |s| s.measures.len());
        if roots.len() == 1 && bars >= 8 {
            warn(
                "harmony_collapsed",
                format!(
                    "every chord over {bars} bars has the same root ({}); a full-score cover would \
                     lock the accompaniment to it — prefer the melody-only cover or correct the \
                     chords",
                    distinct_chords.join(", ")
                ),
            );
        }
        if let Some(e) = &full.abc_error {
            warn(
                "abc_failed_full",
                format!("full score could not be built: {e}"),
            );
        }
        if let Some(e) = melody_abc_error {
            warn(
                "abc_failed_melody",
                format!("melody-only score could not be built: {e}"),
            );
        }
        let mut keys = Vec::new();
        for (_, _, k) in &full.keys {
            if keys.last() != Some(k) {
                keys.push(k.clone());
            }
        }
        let sections = full.structures.iter().map(|s| s.2.clone()).collect();
        let refusal = |error: Option<&str>| {
            if empty {
                Readiness::Refused("the transcription has no melody notes".into())
            } else if let Some(e) = error {
                Readiness::Refused(format!("the score could not be built: {e}"))
            } else {
                Readiness::Ready
            }
        };
        Self {
            melody_cover: refusal(melody_abc_error),
            full_cover: refusal(full.abc_error.as_deref()),
            voices,
            distinct_chords,
            distinct_chord_roots: roots.len(),
            keys,
            sections,
            bars,
            warnings,
            diagnostics: full.diagnostics.clone(),
        }
    }

    fn to_json(&self) -> Value {
        let voice = |v: &VoiceSummary| {
            json!({"notes": v.notes, "min_pitch": v.min_pitch, "max_pitch": v.max_pitch,
                   "median_pitch": v.median_pitch})
        };
        let readiness = |r: &Readiness| match r {
            Readiness::Ready => json!({"ready": true}),
            Readiness::Refused(reason) => json!({"ready": false, "reason": reason}),
        };
        json!({
            "voices": {"vocal": voice(&self.voices[0]), "instrumental": voice(&self.voices[1])},
            "distinct_chords": self.distinct_chords,
            "distinct_chord_roots": self.distinct_chord_roots,
            "keys": self.keys,
            "sections": self.sections,
            "bars": self.bars,
            "warnings": self.warnings.iter().map(|w| json!({"code": w.code, "message": w.message})).collect::<Vec<_>>(),
            "diagnostics": self.diagnostics,
            "cover": {"melody": readiness(&self.melody_cover), "full": readiness(&self.full_cover)},
        })
    }
}

fn octave_json(o: &OctaveEvidence) -> Value {
    json!({
        "method": "spectral peak within ±3% of f0/2, f0 and 2·f0 over each vocal note's interior \
                   (start+50 ms .. end−30 ms, ≥2048 samples), Hann window, 65,536-point FFT; \
                   evidence, not ground truth",
        "transcribed_midi_range": o.transcribed_midi_range.map(|(a, b)| vec![a, b]),
        "notes_checked": o.notes.len(),
        "notes_with_more_energy_at_f0_half": o.f0_half_dominant(),
        "fraction_f0_half_dominant": o.fraction_f0_half_dominant(),
        "notes": o.notes.iter().map(|n| json!({
            "start": n.start, "midi": n.midi, "energy_f0_half": n.energy_f0_half,
            "energy_f0": n.energy_f0, "energy_2f0": n.energy_2f0,
        })).collect::<Vec<_>>(),
    })
}

/// A finished transcription: identity, settings, the model input, exact tokens, every export, both
/// scores, octave evidence and the review.
#[derive(Clone, Debug)]
pub struct Transcription {
    /// What was transcribed.
    pub source: SourceIdentity,
    /// How.
    pub settings: TranscriptionSettings,
    /// With which closure.
    pub closure: ClosureIdentity,
    /// The stitched timeline and exact window tokens.
    pub stitched: Stitched,
    /// The full export (upstream file set, `score.abc` with chord symbols).
    pub full: Exports,
    /// The melody-only score.
    pub melody_abc: Option<String>,
    /// Why the melody-only score could not be built.
    pub melody_abc_error: Option<String>,
    /// Spectral octave evidence (`None` when there is no vocal note).
    pub octave: Option<OctaveEvidence>,
    /// The review.
    pub review: Review,
    /// The exact model input (persisted as [`SOURCE_AUDIO`], so the source identity and the octave
    /// evidence are re-derivable on replay).
    audio: Vec<f32>,
}

fn samples_sha256(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for s in samples {
        hasher.update(s.to_le_bytes());
    }
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

impl Transcription {
    /// Build every output from a stitched timeline and the model input `audio` it was transcribed
    /// from. `audio` must be the source identity's samples (its SHA-256 and length are checked).
    pub fn build(
        source: SourceIdentity,
        settings: TranscriptionSettings,
        closure: ClosureIdentity,
        stitched: Stitched,
        audio: Vec<f32>,
    ) -> Result<Self, Error> {
        if audio.len() != source.samples || samples_sha256(&audio) != source.sha256 {
            return Err(Error::Request(
                "the audio is not the model input the source identity names".into(),
            ));
        }
        let full = export(&stitched.decoded, stitched.duration, false)?;
        let melody = export(&stitched.decoded, stitched.duration, true)?;
        let vocal: Vec<(f64, f64, i64)> = full
            .notes
            .iter()
            .filter(|n| n.track == 0)
            .map(|n| (n.start, n.end, n.pitch))
            .collect();
        let octave = (!vocal.is_empty())
            .then(|| crate::octave::octave_evidence(&audio, source.sample_rate, &vocal));
        let review = Review::assess(
            &stitched,
            &full,
            melody.abc_error.as_deref(),
            octave.as_ref(),
        );
        Ok(Self {
            source,
            settings,
            closure,
            stitched,
            full,
            melody_abc: melody.abc,
            melody_abc_error: melody.abc_error,
            octave,
            review,
            audio,
        })
    }

    /// The model input.
    pub fn audio(&self) -> &[f32] {
        &self.audio
    }

    fn files(&self, tokenizer: &Tokenizer) -> Result<BTreeMap<String, Vec<u8>>, Error> {
        let mut files = BTreeMap::new();
        for (path, text) in &self.full.texts {
            files.insert(path.clone(), text.clone().into_bytes());
        }
        for (path, bytes) in &self.full.midis {
            files.insert(path.clone(), bytes.clone());
        }
        if let Some(abc) = &self.melody_abc {
            files.insert(MELODY_SCORE.into(), abc.clone().into_bytes());
        }
        files.insert(
            "tokens.txt".into(),
            tokens_txt(tokenizer, &self.stitched.records)?.into_bytes(),
        );
        let windows: Vec<Value> = self
            .stitched
            .records
            .iter()
            .map(|r| {
                json!({
                    "window_index": r.index,
                    "start": r.window.start, "end": r.window.end,
                    "accept_start": r.window.accept_start, "accept_end": r.window.accept_end,
                    "prefix_end": r.window.prefix_end, "generation_stop": r.window.generation_stop,
                    "prefix_tokens": r.prefix_tokens, "tokens": r.tokens,
                })
            })
            .collect();
        files.insert(
            TOKENS_JSON.into(),
            serde_json::to_vec_pretty(&windows).expect("serializable"),
        );
        files.insert(
            SOURCE_AUDIO.into(),
            self.audio.iter().flat_map(|s| s.to_le_bytes()).collect(),
        );
        Ok(files)
    }

    /// The whole manifest. Replay rebuilds it with this same function and requires it to equal the
    /// persisted one field for field.
    fn manifest(&self, tokenizer: &Tokenizer, files: &BTreeMap<String, Vec<u8>>) -> Value {
        let digests: BTreeMap<&String, String> =
            files.iter().map(|(rel, bytes)| (rel, sha256_hex(bytes))).collect();
        json!({
            "schema": SCHEMA,
            "source": self.source.to_json(),
            "settings": self.settings.to_json(),
            "closure": self.closure.to_json(),
            "duration_seconds": self.stitched.duration,
            "window_seconds": tokenizer.audio_length_seconds(),
            "time_hz": tokenizer.time_hz(),
            "windows": self.stitched.records.len(),
            "events": self.full.events,
            "melody_notes": self.full.notes.len(),
            "vocal_notes": self.full.vocal_notes(),
            "instrumental_notes": self.full.instrumental_notes(),
            "abc_error": {"full": self.full.abc_error, "melody": self.melody_abc_error},
            "decode_warnings": self.stitched.warnings,
            "octave_evidence": self.octave.as_ref().map(octave_json),
            "review": self.review.to_json(),
            "artifacts": digests,
        })
    }

    /// Persist into `dir`, which must not exist yet or be empty (a review artifact is never
    /// overwritten in place). Every file is written first, the manifest (with each file's SHA-256)
    /// last, so a manifest means a complete artifact. Returns the manifest's SHA-256.
    pub fn save(&self, dir: &Path, tokenizer: &Tokenizer) -> Result<String, Error> {
        if dir.exists()
            && std::fs::read_dir(dir)
                .map_err(|e| Error::io(dir, e))?
                .next()
                .is_some()
        {
            return Err(Error::Request(format!(
                "{} is not empty; a review artifact is written to a fresh directory",
                dir.display()
            )));
        }
        let files = self.files(tokenizer)?;
        for (rel, bytes) in &files {
            let path = join(dir, rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
            }
            std::fs::write(&path, bytes).map_err(|e| Error::io(&path, e))?;
        }
        let bytes =
            serde_json::to_vec_pretty(&self.manifest(tokenizer, &files)).expect("serializable");
        let path = dir.join(MANIFEST);
        std::fs::write(&path, &bytes).map_err(|e| Error::io(&path, e))?;
        Ok(sha256_hex(&bytes))
    }
}

fn join(dir: &Path, rel: &str) -> PathBuf {
    rel.split('/')
        .fold(dir.to_path_buf(), |p, part| p.join(part))
}

/// A persisted review artifact that passed [`ReviewArtifact::open`]'s full verification.
///
/// Its fields are private and `open` is the only constructor, so holding one means the artifact
/// was replayed: every file hashes to its recorded digest, every derived manifest field (review,
/// cover readiness, warnings, ABC errors, note counts, decode warnings, octave evidence) equals
/// what this build re-derives from the persisted tokens and model input, the source identity is
/// the persisted model input's, and the closure identity is the pinned inventory's.
#[derive(Clone, Debug)]
pub struct ReviewArtifact {
    dir: PathBuf,
    manifest: Value,
    manifest_bytes: Vec<u8>,
    manifest_sha256: String,
    report: ReplayReport,
}

/// The outcome of a replay.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayReport {
    /// Artifacts re-derived and found byte-identical.
    pub artifacts_matched: usize,
    /// Windows replayed.
    pub windows: usize,
}

/// The closure identity every persisted artifact must carry: the inventory's pins for both
/// repositories (revision, weights SHA-256, `config.json` SHA-256), the ported code revision and
/// the checkpoint's tokenizer fingerprint.
fn check_pinned_closure(closure: &ClosureIdentity) -> Result<(), Error> {
    use candle_audio_yue2::inventory::ComponentId;
    let expect = |id: ComponentId| {
        let c = id.component();
        [
            c.repo.id.to_string(),
            c.repo.revision.to_string(),
            c.weights().expect("weights").sha256.to_string(),
            c.file("config.json").expect("config").sha256.to_string(),
        ]
    };
    if closure.sheetsage2 != expect(ComponentId::SheetSage2)
        || closure.mert != expect(ComponentId::MertV2FullSong)
    {
        return Err(Error::Replay(
            "the recorded closure identity is not the pinned cover closure".into(),
        ));
    }
    if closure.ported_code_revision != crate::PORTED_CODE_REVISION {
        return Err(Error::Replay(format!(
            "recorded with ported code revision {}, this build ports {}",
            closure.ported_code_revision,
            crate::PORTED_CODE_REVISION
        )));
    }
    if closure.tokenizer_fingerprint != PINNED_TOKENIZER_FINGERPRINT {
        return Err(Error::Replay(
            "the recorded tokenizer fingerprint is not the pinned checkpoint's".into(),
        ));
    }
    if !matches!(closure.device.as_str(), "cpu" | "metal" | "cuda") {
        return Err(Error::Replay(format!(
            "unknown device {:?}",
            closure.device
        )));
    }
    Ok(())
}

/// `tokenizer_fingerprint` of the pinned `m-a-p/SheetSage2` `config.json`.
pub const PINNED_TOKENIZER_FINGERPRINT: &str = "5ba3325af0344c7f";

impl ReviewArtifact {
    /// Open and verify `dir` (see the type docs): hash every listed file, then replay. Any
    /// difference — an altered file, a forged manifest field, a source or closure identity that
    /// does not hold — is an error.
    pub fn open(dir: &Path) -> Result<Self, Error> {
        let path = dir.join(MANIFEST);
        let bytes = std::fs::read(&path).map_err(|e| Error::io(&path, e))?;
        let manifest: Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Replay(format!("manifest is not JSON: {e}")))?;
        if manifest["schema"] != SCHEMA {
            return Err(Error::Replay(format!(
                "unsupported manifest schema {}",
                manifest["schema"]
            )));
        }
        let mut artifact = Self {
            dir: dir.to_path_buf(),
            manifest,
            manifest_sha256: sha256_hex(&bytes),
            manifest_bytes: bytes,
            report: ReplayReport {
                artifacts_matched: 0,
                windows: 0,
            },
        };
        artifact.report = artifact.replay()?;
        Ok(artifact)
    }

    /// The manifest (verified by [`ReviewArtifact::open`]).
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }

    /// SHA-256 of the manifest file (the artifact's identity for provenance).
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// What the verifying replay matched.
    pub fn report(&self) -> &ReplayReport {
        &self.report
    }

    /// The source identity.
    pub fn source(&self) -> Result<SourceIdentity, Error> {
        SourceIdentity::from_json(&self.manifest["source"])
    }

    /// The settings.
    pub fn settings(&self) -> Result<TranscriptionSettings, Error> {
        TranscriptionSettings::from_json(&self.manifest["settings"])
    }

    /// The closure identity.
    pub fn closure(&self) -> Result<ClosureIdentity, Error> {
        ClosureIdentity::from_json(&self.manifest["closure"])
    }

    /// Read an artifact file, re-hashing it against its recorded SHA-256 on every read (a file
    /// replaced after `open` is refused).
    pub fn read(&self, rel: &str) -> Result<Vec<u8>, Error> {
        let digest = self.manifest["artifacts"]
            .get(rel)
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Replay(format!("{rel} is not part of this artifact")))?;
        let path = join(&self.dir, rel);
        let bytes = std::fs::read(&path).map_err(|e| Error::Replay(format!("{rel}: {e}")))?;
        if sha256_hex(&bytes) != digest {
            return Err(Error::Replay(format!(
                "{rel} does not match its recorded SHA-256; the artifact was altered"
            )));
        }
        Ok(bytes)
    }

    /// Cover readiness of `melody_only` / full, as verified.
    pub fn readiness(&self, melody_only: bool) -> Readiness {
        let r = &self.manifest["review"]["cover"][if melody_only { "melody" } else { "full" }];
        if r["ready"].as_bool() == Some(true) {
            Readiness::Ready
        } else {
            Readiness::Refused(
                r["reason"]
                    .as_str()
                    .unwrap_or("not recorded as ready")
                    .to_string(),
            )
        }
    }

    /// The warnings, as verified.
    pub fn warnings(&self) -> Vec<Warning> {
        self.manifest["review"]["warnings"]
            .as_array()
            .map(|ws| {
                ws.iter()
                    .map(|w| Warning {
                        code: w["code"].as_str().unwrap_or_default().to_string(),
                        message: w["message"].as_str().unwrap_or_default().to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Re-derive the whole artifact — every file and every manifest field — from the persisted
    /// tokens, settings and model input (no model) and require equality with what is on disk. Also
    /// checks the source identity against the persisted model input and the closure identity
    /// against the pinned inventory. [`ReviewArtifact::open`] runs this; call it again to re-check.
    pub fn replay(&self) -> Result<ReplayReport, Error> {
        // Every file on disk still hashes to its recorded digest (the derived ones are then
        // compared through the manifest below).
        for rel in self.manifest["artifacts"]
            .as_object()
            .ok_or_else(|| bad("artifacts"))?
            .keys()
        {
            self.read(rel)?;
        }
        let settings = self.settings()?;
        let closure = self.closure()?;
        check_pinned_closure(&closure)?;
        let source = self.source()?;
        let raw = self.read(SOURCE_AUDIO)?;
        if raw.len() % 4 != 0 {
            return Err(Error::Replay(format!("{SOURCE_AUDIO} is not float32 samples")));
        }
        let audio: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if audio.len() != source.samples || samples_sha256(&audio) != source.sha256 {
            return Err(Error::Replay(
                "the recorded source identity is not the persisted model input's".into(),
            ));
        }
        let tokenizer = Tokenizer::new(
            self.manifest["window_seconds"]
                .as_f64()
                .ok_or_else(|| bad("window_seconds"))?,
            self.manifest["time_hz"].as_u64().ok_or_else(|| bad("time_hz"))? as u32,
            Some(&closure.tokenizer_fingerprint),
        )?;
        let windows: Vec<Value> = serde_json::from_slice(&self.read(TOKENS_JSON)?)
            .map_err(|e| Error::Replay(format!("tokens.json: {e}")))?;
        let tokens: Vec<Vec<u32>> = windows
            .iter()
            .map(|w| {
                w["tokens"]
                    .as_array()
                    .ok_or_else(|| bad("tokens"))?
                    .iter()
                    .map(|t| t.as_u64().map(|t| t as u32).ok_or_else(|| bad("tokens")))
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        let duration = source.duration();
        let prompts: Vec<&str> = settings.prompts.iter().map(String::as_str).collect();
        let stitcher = Stitcher::new(
            &tokenizer,
            &prompts,
            duration,
            settings.overlap_seconds,
            settings.lookahead_seconds,
            crate::pipeline::MAX_OUTPUT_SEQ_LEN,
        )?;
        let stitched = stitcher.replay(&tokens)?;
        let rebuilt = Transcription::build(source, settings, closure, stitched, audio)?;
        let files = rebuilt.files(&tokenizer)?;
        // Byte equality with the file on disk: the manifest is re-serialized exactly as `save`
        // writes it (serde_json's float parsing is not round-trip exact, so parsed values are
        // not compared).
        let expected =
            serde_json::to_vec_pretty(&rebuilt.manifest(&tokenizer, &files)).expect("serializable");
        if expected != self.manifest_bytes {
            let ours: Value = serde_json::from_slice(&expected).expect("just serialized");
            let field = ours
                .as_object()
                .and_then(|o| {
                    o.keys()
                        .chain(self.manifest.as_object().into_iter().flat_map(|m| m.keys()))
                        .find(|k| o.get(*k) != self.manifest.get(*k))
                        .cloned()
                })
                .unwrap_or_else(|| "(formatting)".into());
            return Err(Error::Replay(format!(
                "manifest field `{field}` differs from the replayed transcription"
            )));
        }
        Ok(ReplayReport {
            artifacts_matched: files.len(),
            windows: tokens.len(),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
