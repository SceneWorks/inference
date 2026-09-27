//! Zero-shot covers (story sc-22996): a reviewed score plus a target style and lyrics → a YuE2
//! [`SongRequest`] that plans from that score.
//!
//! This is upstream's documented cover workflow (`docs/covers.md` and the `yue2-music` skill at
//! YuE@92a73cc): the melody travels **symbolically**, as the two-voice ABC YuE2 plans in, and the
//! general checkpoint realizes it in the requested style. There is no audio conditioning of any kind
//! — no reference audio, no in-context-learning prompt (YuE1's ICL path in `candle-audio-yue` is not
//! involved), and [`SongRequest`] has no field that could carry one.
//!
//! * [`CoverMode::Melody`] → `cot = melody`: every chord symbol is removed first
//!   ([`abc::strip_chords`], which proves the kept melodies, meters and tempo are unchanged), so the
//!   accompaniment is free; optionally only one voice is kept ([`abc::KeepVoice`]).
//! * [`CoverMode::Full`] → `cot = full`: the score's harmony is supplied and must be present.
//!
//! The score must parse in the native dialect ([`abc::parse`]) and carry sounding notes. The lyrics
//! (source lyrics, or a translation together with the source it translates) are checked against the
//! score's sections and each other ([`lyrics`]); structural mismatches of a translation are refused,
//! estimates are warnings in the [`CoverReport`]. The request itself is validated by the protocol
//! ([`SongRequest::new`]) exactly like any other request.

pub mod abc;
pub mod lyrics;

use serde_json::{json, Value};

use crate::protocol::{
    CotMode, ProtocolError, SongRequest, SongRequestSpec, DEFAULT_ID, DEFAULT_SEED,
};
use abc::{KeepVoice, Score};
use lyrics::{estimate_syllables, is_non_vocal, score_sections, sections, LyricSection};

/// How much of the score the cover keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoverMode {
    /// Melody only (`cot = melody`), chord symbols removed — upstream's recommended cover route.
    Melody,
    /// The full score with its harmony (`cot = full`).
    Full,
}

impl CoverMode {
    /// The planning mode the request uses.
    pub fn cot(self) -> CotMode {
        match self {
            CoverMode::Melody => CotMode::Melody,
            CoverMode::Full => CotMode::Full,
        }
    }
}

/// The lyrics of a cover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverLyrics {
    /// The words to sing, with section tags.
    pub lyrics: String,
    /// When `lyrics` is a translation: the source lyrics it translates (section-aligned).
    pub translated_from: Option<String>,
}

impl CoverLyrics {
    /// Source-language lyrics.
    pub fn source(lyrics: impl Into<String>) -> Self {
        Self {
            lyrics: lyrics.into(),
            translated_from: None,
        }
    }

    /// A translation of `source`.
    pub fn translation(lyrics: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            lyrics: lyrics.into(),
            translated_from: Some(source.into()),
        }
    }
}

/// Everything a cover request is built from.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverSpec {
    /// Melody-only or full score.
    pub mode: CoverMode,
    /// The reviewed score (native two-voice dialect).
    pub score: String,
    /// Which melodies a melody-only cover keeps (ignored for [`CoverMode::Full`]).
    pub keep: KeepVoice,
    /// Target style.
    pub style: String,
    /// Lyrics.
    pub lyrics: CoverLyrics,
    /// Seed.
    pub seed: u64,
    /// CFG scale (`None` = the mode default).
    pub cfg_scale: Option<f64>,
    /// Filename-safe id.
    pub id: String,
}

impl CoverSpec {
    /// A spec with upstream's default seed and id.
    pub fn new(
        mode: CoverMode,
        score: impl Into<String>,
        style: impl Into<String>,
        lyrics: CoverLyrics,
    ) -> Self {
        Self {
            mode,
            score: score.into(),
            keep: KeepVoice::Both,
            style: style.into(),
            lyrics,
            seed: DEFAULT_SEED,
            cfg_scale: None,
            id: DEFAULT_ID.to_string(),
        }
    }
}

/// A cover that cannot be built.
#[derive(Debug, thiserror::Error)]
pub enum CoverError {
    /// The score is outside the native dialect or fails an invariant.
    #[error("cover score: {0}")]
    Score(#[from] abc::AbcError),
    /// The score or lyrics cannot serve the requested mode.
    #[error("cover refused: {0}")]
    Refused(String),
    /// The request is invalid under the protocol.
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
}

/// A cover warning (reviewable; not a refusal).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverWarning {
    /// Stable code.
    pub code: &'static str,
    /// Detail.
    pub message: String,
}

/// What was checked while building the cover.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverReport {
    /// Chord symbols removed (melody mode).
    pub chords_removed: usize,
    /// Vocal / instrumental sounding notes of the plan score.
    pub notes: [usize; 2],
    /// Bars.
    pub bars: usize,
    /// Tempo.
    pub bpm: i64,
    /// Nominal duration at the notated tempo, seconds.
    pub nominal_seconds: f64,
    /// The score's vocal sections, in order.
    pub score_sections: Vec<String>,
    /// The lyric sections, in order.
    pub lyric_sections: Vec<String>,
    /// Warnings.
    pub warnings: Vec<CoverWarning>,
}

impl CoverReport {
    /// A JSON record for provenance.
    pub fn to_json(&self) -> Value {
        json!({
            "chords_removed": self.chords_removed,
            "vocal_notes": self.notes[0],
            "instrumental_notes": self.notes[1],
            "bars": self.bars,
            "bpm": self.bpm,
            "nominal_seconds": self.nominal_seconds,
            "score_sections": self.score_sections,
            "lyric_sections": self.lyric_sections,
            "warnings": self.warnings.iter().map(|w| json!({"code": w.code, "message": w.message})).collect::<Vec<_>>(),
        })
    }
}

/// A built cover: the request (whose external ABC is the plan score) and the checks.
#[derive(Clone, Debug)]
pub struct PreparedCover {
    /// The validated request (`cot = melody` or `full`, `abc` = the plan score).
    pub request: SongRequest,
    /// The parsed plan score.
    pub score: Score,
    /// The checks.
    pub report: CoverReport,
}

fn labels(sections: &[LyricSection]) -> Vec<String> {
    sections.iter().map(|s| s.label.clone()).collect()
}

fn syllables(section: &LyricSection) -> usize {
    section.lines.iter().map(|l| estimate_syllables(l)).sum()
}

/// Build the cover request from `spec` (see the module docs for every check).
pub fn prepare_cover(spec: &CoverSpec) -> Result<PreparedCover, CoverError> {
    let source = abc::parse(&spec.score)?;
    let mut chords_removed = 0;
    let plan_text = match spec.mode {
        CoverMode::Melody => {
            let (text, removed) = abc::strip_chords(&spec.score, spec.keep)?;
            chords_removed = removed;
            text
        }
        CoverMode::Full => {
            if source.chord_count() == 0 {
                return Err(CoverError::Refused(
                    "a full-score cover supplies the score's harmony, but it has no chord \
                     symbols; use the melody-only cover"
                        .into(),
                ));
            }
            if spec.keep != KeepVoice::Both {
                return Err(CoverError::Refused(
                    "voice selection applies to melody-only covers".into(),
                ));
            }
            spec.score.clone()
        }
    };
    let score = abc::parse(&plan_text)?;
    let notes = [score.voices[0].notes.len(), score.voices[1].notes.len()];
    if notes == [0, 0] {
        return Err(CoverError::Refused(
            "the score has no sounding notes: there is no melody to cover".into(),
        ));
    }
    let lyric_sections = sections(&spec.lyrics.lyrics);
    if lyric_sections.iter().all(|s| s.lines.is_empty()) {
        return Err(CoverError::Refused(
            "the cover needs lyrics (with section tags)".into(),
        ));
    }
    let mut warnings = Vec::new();
    if let Some(source_lyrics) = &spec.lyrics.translated_from {
        let original = sections(source_lyrics);
        if labels(&original) != labels(&lyric_sections) {
            return Err(CoverError::Refused(format!(
                "the translation's sections {:?} are not aligned with the source lyrics' {:?}",
                labels(&lyric_sections),
                labels(&original)
            )));
        }
        for (ours, theirs) in lyric_sections.iter().zip(&original) {
            if ours.lines.len() != theirs.lines.len() {
                warnings.push(CoverWarning {
                    code: "translation_line_count",
                    message: format!(
                        "[{}]: {} lines, the source has {}",
                        ours.tag,
                        ours.lines.len(),
                        theirs.lines.len()
                    ),
                });
            }
            let (a, b) = (syllables(ours), syllables(theirs));
            if b > 0 && (a as f64 - b as f64).abs() / b as f64 > 0.3 {
                warnings.push(CoverWarning {
                    code: "translation_syllables",
                    message: format!(
                        "[{}]: ~{a} syllables against the source's ~{b}; match phrasing and \
                         syllable counts to the melody",
                        ours.tag
                    ),
                });
            }
        }
    }
    let score_vocal = score_sections(&score);
    // The lyric sections that are sung: the same filter `score_sections` applies to the score
    // (tagged, non-empty, not an intro / outro / instrumental-type label).
    let sung: Vec<&LyricSection> = lyric_sections
        .iter()
        .filter(|s| !s.label.is_empty() && !s.lines.is_empty() && !is_non_vocal(&s.label))
        .collect();
    let lyric_labels: Vec<String> = sung.iter().map(|s| s.label.clone()).collect();
    let score_labels: Vec<String> = score_vocal.iter().map(|s| s.label.clone()).collect();
    if notes[0] == 0 && spec.keep != KeepVoice::Ins {
        warnings.push(CoverWarning {
            code: "no_vocal_melody",
            message: "the score's Vocal voice has no notes; the lyrics have no melody to follow"
                .into(),
        });
    }
    if !score_labels.is_empty() && lyric_labels != score_labels {
        warnings.push(CoverWarning {
            code: "sections_misaligned",
            message: format!(
                "lyric sections {lyric_labels:?} do not follow the score's vocal sections \
                 {score_labels:?}; align the section tags and lyric order with the score"
            ),
        });
    }
    if lyric_labels.len() == score_vocal.len() {
        for (section, lyric) in score_vocal.iter().zip(&sung) {
            let s = syllables(lyric);
            if s > 0
                && (s as f64 > 1.5 * section.vocal_notes as f64
                    || (s as f64) < 0.5 * section.vocal_notes as f64)
            {
                warnings.push(CoverWarning {
                    code: "syllables_vs_notes",
                    message: format!(
                        "[{}]: ~{s} syllables for {} melody notes",
                        lyric.tag, section.vocal_notes
                    ),
                });
            }
        }
    }
    let request = SongRequest::new(SongRequestSpec {
        style: spec.style.clone(),
        lyrics: spec.lyrics.lyrics.clone(),
        cot: spec.mode.cot(),
        seed: spec.seed,
        abc: Some(plan_text),
        cfg_scale: spec.cfg_scale,
        id: spec.id.clone(),
    })?;
    let report = CoverReport {
        chords_removed,
        notes,
        bars: score.voices[0].bars.len(),
        bpm: score.bpm,
        nominal_seconds: score.nominal_seconds(),
        score_sections: score_labels,
        lyric_sections: lyric_labels,
        warnings,
    };
    Ok(PreparedCover {
        request,
        score,
        report,
    })
}

#[cfg(test)]
mod tests;
