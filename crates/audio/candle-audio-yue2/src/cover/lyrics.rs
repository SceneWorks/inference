//! Lyric sections and their alignment with a cover score.
//!
//! YuE2 has no note-level lyric alignment input: the lyrics are free text with section tags
//! (`[Verse]`, `[Chorus]`, …) and the score is the melody. Upstream's cover guidance is to "align
//! section tags and lyric order with the score" and, for a translation, to "match phrasing and
//! syllable counts to the melody". This module makes that alignment checkable: sections are parsed
//! from both, compared in order, and syllable estimates are compared with the melody's notes. A
//! structural mismatch between a translation and its source is refused; everything that is only an
//! estimate is a warning.

use super::abc::Score;

/// Score sections that carry no lyrics.
const NON_VOCAL_SECTIONS: [&str; 8] = [
    "intro",
    "outro",
    "interlude",
    "instrumental",
    "silence",
    "solo",
    "fade-out",
    "inst",
];

/// One `[Tag]` section of a lyric sheet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LyricSection {
    /// The tag as written.
    pub tag: String,
    /// Normalized label (lower case, trailing numbers dropped).
    pub label: String,
    /// Non-empty lyric lines.
    pub lines: Vec<String>,
}

/// Normalize a section name: lower case, whitespace-collapsed, trailing numbering dropped
/// (`Verse 2` → `verse`).
pub fn normalize_label(name: &str) -> String {
    let lower = name.trim().to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let trimmed: Vec<&str> = match words.split_last() {
        Some((last, rest)) if !rest.is_empty() && last.chars().all(|c| c.is_ascii_digit()) => {
            rest.to_vec()
        }
        _ => words,
    };
    trimmed
        .join(" ")
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .trim()
        .to_string()
}

/// Parse `[Tag]` sections. Lines before the first tag form an untagged section (`label` empty).
pub fn sections(lyrics: &str) -> Vec<LyricSection> {
    let mut out: Vec<LyricSection> = Vec::new();
    for raw in lyrics.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') && line.len() > 2 {
            let tag = line[1..line.len() - 1].trim().to_string();
            out.push(LyricSection {
                label: normalize_label(&tag),
                tag,
                lines: Vec::new(),
            });
            continue;
        }
        if out.is_empty() {
            out.push(LyricSection {
                tag: String::new(),
                label: String::new(),
                lines: Vec::new(),
            });
        }
        out.last_mut()
            .expect("non-empty")
            .lines
            .push(line.to_string());
    }
    out
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF   // Hiragana, Katakana
        | 0x3400..=0x4DBF // CJK Ext A
        | 0x4E00..=0x9FFF // CJK Unified
        | 0xAC00..=0xD7AF // Hangul syllables
        | 0xF900..=0xFAFF)
}

fn is_vowel(c: char) -> bool {
    "aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿœ".contains(c.to_ascii_lowercase())
}

/// A syllable estimate: one per CJK character, vowel groups per alphabetic word (at least one per
/// word, a final silent `e` discounted). A heuristic for warnings only.
pub fn estimate_syllables(line: &str) -> usize {
    let mut count = 0;
    for word in line.split(|c: char| !(c.is_alphabetic() || c == '\'')) {
        if word.is_empty() {
            continue;
        }
        let cjk = word.chars().filter(|c| is_cjk(*c)).count();
        if cjk > 0 {
            count += cjk;
            continue;
        }
        let chars: Vec<char> = word.chars().collect();
        let mut groups = 0;
        let mut previous = false;
        for &c in &chars {
            let v = is_vowel(c);
            if v && !previous {
                groups += 1;
            }
            previous = v;
        }
        if groups > 1
            && chars.len() > 2
            && chars[chars.len() - 1].eq_ignore_ascii_case(&'e')
            && !is_vowel(chars[chars.len() - 2])
        {
            groups -= 1;
        }
        count += groups.max(1);
    }
    count
}

/// A score section with lyric-bearing melody.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScoreSection {
    /// Normalized label.
    pub label: String,
    /// Vocal notes in it.
    pub vocal_notes: usize,
}

/// The score's vocal sections: `% label` spans holding at least one `Vocal` note, excluding
/// intro/outro/instrumental-type labels.
pub fn score_sections(score: &Score) -> Vec<ScoreSection> {
    let vocal = &score.voice("Vocal").notes;
    let mut out = Vec::new();
    for (i, (start, label)) in score.sections.iter().enumerate() {
        let end = score.sections.get(i + 1).map(|s| s.0);
        let notes = vocal
            .iter()
            .filter(|n| n.onset >= *start && end.is_none_or(|e| n.onset < e))
            .count();
        let label = normalize_label(label);
        if notes > 0 && !NON_VOCAL_SECTIONS.contains(&label.as_str()) {
            out.push(ScoreSection {
                label,
                vocal_notes: notes,
            });
        }
    }
    out
}
