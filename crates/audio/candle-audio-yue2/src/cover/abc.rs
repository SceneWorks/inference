//! The bounded two-voice ABC dialect YuE2 plans in — a native port of the parts of
//! `skills/yue2-music/scripts/abc_tools.py` (YuE@92a73cc, Apache-2.0) the cover path needs:
//! [`parse`] (fail-closed structural validation with sounding notes resolved through ties and
//! bar-scoped accidentals), [`strip_chords`] (with the optional single-voice selection) and
//! [`compare`] (the exact melody invariant `strip_chords` must keep).
//!
//! Additions for the cover path, outside upstream's helper: [`Score::sections`] records each
//! `% label` comment's onset, so lyrics can be aligned with the score's sections.

use std::collections::BTreeMap;
use std::fmt;

/// The two voices, in order.
pub const VOICES: [&str; 2] = ["Vocal", "Ins"];
const DURATIONS: [i64; 11] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48];
const QUALITIES: [&str; 15] = [
    "", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5", "sus4", "sus2", "6", "m6", "7sus4",
    "m(maj7)",
];
const VOICE_LINES: [&str; 2] = [
    "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"",
    "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"",
];

/// An unsupported token or a failed structural invariant (upstream `AbcError`).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct AbcError(pub String);

fn fail(condition: bool, message: impl FnOnce() -> String) -> Result<(), AbcError> {
    if condition {
        Err(AbcError(message()))
    } else {
        Ok(())
    }
}

/// An exact fraction of a quarter note (upstream uses `fractions.Fraction`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Frac {
    num: i64,
    den: i64,
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

impl Frac {
    /// `num / den`, reduced.
    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "zero denominator");
        let g = gcd(num, den).max(1);
        let sign = if den < 0 { -1 } else { 1 };
        Self {
            num: sign * num / g,
            den: sign * den / g,
        }
    }

    /// Zero.
    pub const ZERO: Frac = Frac { num: 0, den: 1 };

    /// The value as a float.
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl PartialOrd for Frac {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Frac {
    /// By value (denominators are kept positive).
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (i128::from(self.num) * i128::from(other.den))
            .cmp(&(i128::from(other.num) * i128::from(self.den)))
    }
}

impl std::ops::Add for Frac {
    type Output = Frac;
    fn add(self, o: Frac) -> Frac {
        Frac::new(self.num * o.den + o.num * self.den, self.den * o.den)
    }
}

impl std::ops::Mul<i64> for Frac {
    type Output = Frac;
    fn mul(self, k: i64) -> Frac {
        Frac::new(self.num * k, self.den)
    }
}

impl fmt::Debug for Frac {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Frac {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

fn natural(letter: char) -> i64 {
    match letter {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        _ => 11,
    }
}

fn key_signature(key: &str) -> Option<i64> {
    const MAJOR: [&str; 15] = [
        "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#",
    ];
    const MINOR: [&str; 15] = [
        "Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m",
        "A#m",
    ];
    MAJOR
        .iter()
        .position(|k| *k == key)
        .or_else(|| MINOR.iter().position(|k| *k == key))
        .map(|i| i as i64 - 7)
}

fn key_accidentals(key: &str) -> Result<BTreeMap<char, i64>, AbcError> {
    let count = key_signature(key).ok_or_else(|| {
        AbcError(format!(
            "Unsupported key {key:?}; use a standard major or minor K: field"
        ))
    })?;
    let mut out: BTreeMap<char, i64> = "CDEFGAB".chars().map(|c| (c, 0)).collect();
    let order = if count > 0 { "FCGDAEB" } else { "BEADGCF" };
    for letter in order.chars().take(count.unsigned_abs() as usize) {
        out.insert(letter, if count > 0 { 1 } else { -1 });
    }
    Ok(out)
}

fn meter_value(text: &str) -> Result<(i64, i64), AbcError> {
    let bad = || {
        AbcError(format!(
            "Unsupported meter {text:?}; write an explicit fraction"
        ))
    };
    let (n, d) = text.split_once('/').ok_or_else(bad)?;
    let positive = |s: &str| -> Option<i64> {
        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) && !s.starts_with('0'))
            .then(|| s.parse().ok())
            .flatten()
    };
    let (n, d) = (positive(n).ok_or_else(bad)?, positive(d).ok_or_else(bad)?);
    fail(d > 1024 || d & (d - 1) != 0, || {
        format!("Unsupported meter denominator {d}")
    })?;
    Ok((n, d))
}

fn is_pitch_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some('A'..='G'))
        && matches!(chars.as_str(), "" | "b" | "#" | "bb" | "##")
}

/// `CHORD.fullmatch`: root, one of the dialect's qualities, optional `/bass`.
pub fn is_supported_chord(chord: &str) -> bool {
    for root_len in [1usize, 2, 3] {
        if root_len > chord.len() || !chord.is_char_boundary(root_len) {
            continue;
        }
        let (root, rest) = chord.split_at(root_len);
        if !is_pitch_name(root) {
            continue;
        }
        for quality in QUALITIES {
            if let Some(after) = rest.strip_prefix(quality) {
                if after.is_empty() {
                    return true;
                }
                if let Some(bass) = after.strip_prefix('/') {
                    if is_pitch_name(bass) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// One token of a music line (upstream `TOKEN`).
#[derive(Clone, Debug, PartialEq, Eq)]
enum Token<'a> {
    Chord(&'a str),
    Key(&'a str),
    Note {
        acc: &'a str,
        note: char,
        octave: &'a str,
        duration: &'a str,
        tie: bool,
    },
}

/// `TOKEN.match(text, pos)`: the token at `pos` and its end.
fn token_at(text: &str, pos: usize) -> Option<(Token<'_>, usize)> {
    let rest = &text[pos..];
    if let Some(body) = rest.strip_prefix('"') {
        let end = body.find(['"', '\n'])?;
        if body.as_bytes()[end] == b'"' {
            return Some((Token::Chord(&body[..end]), pos + 1 + end + 1));
        }
        return None;
    }
    if let Some(body) = rest.strip_prefix("[K:") {
        let end = body.find([']', '\n'])?;
        if end > 0 && body.as_bytes()[end] == b']' {
            return Some((Token::Key(&body[..end]), pos + 3 + end + 1));
        }
        return None;
    }
    let bytes = rest.as_bytes();
    let acc_len = if rest.starts_with("^^") || rest.starts_with("__") {
        2
    } else if matches!(bytes.first(), Some(b'^' | b'_' | b'=')) {
        1
    } else {
        0
    };
    let note = *bytes.get(acc_len)?;
    if !matches!(note, b'A'..=b'G' | b'a'..=b'g' | b'z') {
        return None;
    }
    let mut j = acc_len + 1;
    let octave_start = j;
    while j < bytes.len() && matches!(bytes[j], b',' | b'\'') {
        j += 1;
    }
    let octave_end = j;
    while j < bytes.len() && bytes[j].is_ascii_digit() {
        j += 1;
    }
    let duration_end = j;
    let tie = j < bytes.len() && bytes[j] == b'-';
    if tie {
        j += 1;
    }
    Some((
        Token::Note {
            acc: &rest[..acc_len],
            note: note as char,
            octave: &rest[octave_start..octave_end],
            duration: &rest[octave_end..duration_end],
            tie,
        },
        pos + j,
    ))
}

/// A sounding note: onset and duration in quarter notes, MIDI pitch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoundingNote {
    /// Onset, quarter notes from the start.
    pub onset: Frac,
    /// MIDI pitch.
    pub pitch: i64,
    /// Length after merging ties, quarter notes.
    pub span: Frac,
}

/// One voice of a parsed score.
#[derive(Clone, Debug, PartialEq)]
pub struct Voice {
    meter: (i64, i64),
    key: String,
    time: Frac,
    /// Sounding notes.
    pub notes: Vec<SoundingNote>,
    /// Bars `(start, length, meter)`.
    pub bars: Vec<(Frac, Frac, (i64, i64))>,
    /// Chord symbols `(onset, text)`.
    pub chords: Vec<(Frac, String)>,
    /// Key timeline `(onset, key)`.
    pub keys: Vec<(Frac, String)>,
    pending: Option<(i64, i64)>,
}

impl Voice {
    fn new(meter: (i64, i64), key: &str) -> Self {
        Self {
            meter,
            key: key.to_string(),
            time: Frac::ZERO,
            notes: Vec::new(),
            bars: Vec::new(),
            chords: Vec::new(),
            keys: vec![(Frac::ZERO, key.to_string())],
            pending: None,
        }
    }

    /// Total length, quarter notes.
    pub fn length(&self) -> Frac {
        self.time
    }
}

/// A parsed, validated score.
#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    /// The text.
    pub text: String,
    /// `L:` unit.
    pub unit: Frac,
    /// `Q:1/4=` tempo.
    pub bpm: i64,
    /// `Vocal` and `Ins`.
    pub voices: [Voice; 2],
    /// `line index → voice index` of every music line.
    music_lines: BTreeMap<usize, usize>,
    /// `% label` section comments with their onsets (quarter notes).
    pub sections: Vec<(Frac, String)>,
}

impl Score {
    /// The voice named `name`.
    pub fn voice(&self, name: &str) -> &Voice {
        &self.voices[if name == "Vocal" { 0 } else { 1 }]
    }

    /// Nominal duration at the notated tempo, seconds.
    pub fn nominal_seconds(&self) -> f64 {
        self.voices[0].time.to_f64() * 60.0 / self.bpm as f64
    }

    /// Chord symbols in either voice.
    pub fn chord_count(&self) -> usize {
        self.voices.iter().map(|v| v.chords.len()).sum()
    }
}

fn parse_bar(body: &str, voice: &mut Voice, unit: Frac, context: &str) -> Result<(), AbcError> {
    let (n, d) = voice.meter;
    let length = Frac::new(4 * n, d);
    let start = voice.time;
    let mut offset = Frac::ZERO;
    let mut local: BTreeMap<char, i64> = BTreeMap::new();
    if body == "Z" {
        fail(voice.pending.is_some(), || {
            format!("{context}: tie enters a full-measure rest")
        })?;
        offset = length;
    } else {
        let mut cursor = 0;
        while cursor < body.len() {
            let c = body[cursor..].chars().next().expect("in bounds");
            if c.is_whitespace() {
                cursor += c.len_utf8();
                continue;
            }
            let (token, end) = token_at(body, cursor).ok_or_else(|| {
                let snippet: String = body[cursor..].chars().take(24).collect();
                AbcError(format!("{context}: unsupported token at {snippet:?}"))
            })?;
            cursor = end;
            fail(offset >= length, || {
                format!("{context}: event after the measure end")
            })?;
            match token {
                Token::Chord(chord) => {
                    fail(!is_supported_chord(chord), || {
                        format!("{context}: unsupported chord {chord:?}")
                    })?;
                    voice.chords.push((start + offset, chord.to_string()));
                }
                Token::Key(key) => {
                    key_accidentals(key)?;
                    voice.key = key.to_string();
                    voice.keys.push((start + offset, key.to_string()));
                    local.clear();
                }
                Token::Note {
                    acc,
                    note,
                    octave,
                    duration,
                    tie,
                } => {
                    let units: i64 = if duration.is_empty() {
                        1
                    } else {
                        duration.parse().unwrap_or(i64::MAX)
                    };
                    fail(!DURATIONS.contains(&units), || {
                        format!(
                            "{context}: unsupported duration {units}; split it into tied \
                             supported lengths"
                        )
                    })?;
                    let dur = unit * (units * 4);
                    fail(offset + dur > length, || {
                        format!("{context}: note/rest exceeds meter duration")
                    })?;
                    fail(octave.contains(',') && octave.contains('\''), || {
                        format!("{context}: mixed octave marks")
                    })?;
                    if note == 'z' {
                        fail(!acc.is_empty() || !octave.is_empty() || tie, || {
                            format!(
                                "{context}: a rest cannot have accidentals, octave marks or ties"
                            )
                        })?;
                        fail(voice.pending.is_some(), || {
                            format!("{context}: tie enters a rest")
                        })?;
                    } else {
                        let letter = note.to_ascii_uppercase();
                        let mut written =
                            60 + natural(letter) + if note.is_ascii_lowercase() { 12 } else { 0 };
                        written += 12
                            * (octave.matches('\'').count() as i64
                                - octave.matches(',').count() as i64);
                        let mut alteration = match local.get(&letter) {
                            Some(a) => *a,
                            None => key_accidentals(&voice.key)?[&letter],
                        };
                        if !acc.is_empty() {
                            alteration = match acc {
                                "=" => 0,
                                "_" => -1,
                                "__" => -2,
                                "^" => 1,
                                _ => 2,
                            };
                            local.insert(letter, alteration);
                        }
                        let mut pitch = written + alteration;
                        if let Some((old_pitch, old_written)) = voice.pending {
                            if acc.is_empty() && written == old_written {
                                pitch = old_pitch;
                            }
                            fail(pitch != old_pitch, || {
                                format!("{context}: tie changes pitch from {old_pitch} to {pitch}")
                            })?;
                            let last = voice.notes.last_mut().expect("a pending tie has a note");
                            last.span = last.span + dur;
                        } else {
                            fail(!(0..=127).contains(&pitch), || {
                                format!("{context}: pitch {pitch} is outside MIDI range")
                            })?;
                            voice.notes.push(SoundingNote {
                                onset: start + offset,
                                pitch,
                                span: dur,
                            });
                        }
                        voice.pending = tie.then_some((pitch, written));
                    }
                    offset = offset + dur;
                }
            }
        }
    }
    fail(offset != length, || {
        format!("{context}: duration {offset} quarter notes != meter duration {length}")
    })?;
    voice.bars.push((start, length, voice.meter));
    voice.time = voice.time + length;
    Ok(())
}

/// Upstream `parse`: fail closed on unsupported tokens; resolve sounding notes.
pub fn parse(text: &str) -> Result<Score, AbcError> {
    let lines: Vec<&str> = text.lines().collect();
    fail(lines.len() < 12, || {
        "Incomplete native two-voice ABC".into()
    })?;
    fail(lines[0..2] != ["X:1", "T:"], || {
        "Expected native X:1 and blank T: header".into()
    })?;
    fail(!lines[2].starts_with("M:"), || "Missing header M:".into())?;
    let meter = meter_value(&lines[2][2..])?;
    let unit_den = lines[3]
        .strip_prefix("L:1/")
        .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()) && !d.starts_with('0'))
        .and_then(|d| d.parse::<i64>().ok())
        .ok_or_else(|| AbcError("Expected L:1/<power of two>, usually L:1/32".into()))?;
    fail(unit_den > 1024 || unit_den & (unit_den - 1) != 0, || {
        "Unsupported L: denominator".into()
    })?;
    let unit = Frac::new(1, unit_den);
    let bpm = lines[4]
        .strip_prefix("Q:1/4=")
        .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()) && !d.starts_with('0'))
        .and_then(|d| d.parse::<i64>().ok())
        .ok_or_else(|| AbcError("Expected integer quarter-note tempo Q:1/4=<BPM>".into()))?;
    fail(lines[5..7] != VOICE_LINES, || {
        "Preserve native Vocal and Ins voice definitions".into()
    })?;
    fail(!lines[7].starts_with("K:"), || "Missing header K:".into())?;
    let key = &lines[7][2..];
    key_accidentals(key)?;
    let mut voices = [Voice::new(meter, key), Voice::new(meter, key)];
    let mut music_lines = BTreeMap::new();
    let mut sections = Vec::new();
    let mut cursor = 8;
    let mut group = 0;
    while cursor < lines.len() {
        while cursor < lines.len() && lines[cursor].starts_with("% ") {
            sections.push((voices[0].time, lines[cursor][2..].trim().to_string()));
            cursor += 1;
        }
        fail(cursor == lines.len(), || {
            "Dangling section comment without music".into()
        })?;
        group += 1;
        let mut counts = Vec::new();
        for (index, name) in VOICES.iter().enumerate() {
            let context = format!("group {group}, {name}");
            fail(
                cursor >= lines.len() || lines[cursor] != format!("V: {name}"),
                || format!("{context}: expected V: {name}"),
            )?;
            cursor += 1;
            let voice = &mut voices[index];
            let mut fields = Vec::new();
            while cursor < lines.len()
                && (lines[cursor].starts_with("M:") || lines[cursor].starts_with("K:"))
            {
                let (field, value) = lines[cursor].split_once(':').expect("prefix checked");
                fail(fields.contains(&field), || {
                    format!("{context}: duplicate {field}: field")
                })?;
                fields.push(field);
                if field == "M" {
                    voice.meter = meter_value(value)?;
                } else {
                    key_accidentals(value)?;
                    voice.key = value.to_string();
                    let at = voice.time;
                    voice.keys.push((at, value.to_string()));
                }
                cursor += 1;
            }
            fail(cursor >= lines.len(), || {
                format!("{context}: missing music line")
            })?;
            let line = lines[cursor];
            fail(!line.ends_with('|'), || {
                format!("{context}: music line must end with a plain barline")
            })?;
            music_lines.insert(cursor, index);
            cursor += 1;
            let mut bars = Vec::new();
            for bar in line[..line.len() - 1].split('|') {
                let bar = bar.trim();
                fail(bar.is_empty(), || {
                    format!("{context}: empty measure or unsupported double/repeat barline")
                })?;
                match bar.strip_prefix('Z') {
                    Some("") => bars.push("Z"),
                    Some(n @ ("2" | "3" | "4")) => {
                        bars.extend(std::iter::repeat_n("Z", n.parse().expect("digit")))
                    }
                    _ => bars.push(bar),
                }
            }
            fail(!(1..=4).contains(&bars.len()), || {
                format!("{context}: expected 1–4 measures after expanding Z rests")
            })?;
            counts.push(bars.len());
            for bar in bars {
                let bar_context = format!("{context}, bar {}", voice.bars.len() + 1);
                parse_bar(bar, voice, unit, &bar_context)?;
            }
        }
        fail(counts[0] != counts[1], || {
            format!("group {group}: voices have different measure counts")
        })?;
    }
    for (name, voice) in VOICES.iter().zip(&voices) {
        fail(voice.pending.is_some(), || {
            format!("{name}: unresolved tie at end of score")
        })?;
    }
    fail(!voices[1].chords.is_empty(), || {
        "Native chord symbols belong in Vocal, not Ins".into()
    })?;
    fail(voices[0].bars != voices[1].bars, || {
        "Voice meter/time grids differ".into()
    })?;
    fail(voices[0].keys != voices[1].keys, || {
        "Voice key-change timelines differ".into()
    })?;
    Ok(Score {
        text: text.to_string(),
        unit,
        bpm,
        voices,
        music_lines,
        sections,
    })
}

/// Upstream `compare`: tempo, bar grids and sounding notes of the named voices. Returns the
/// differences (empty = match).
pub fn compare(
    before: &Score,
    after: &Score,
    names: &[&str],
    allow_tempo_change: bool,
) -> Vec<String> {
    let mut differences = Vec::new();
    if before.bpm != after.bpm && !allow_tempo_change {
        differences.push("quarter-note tempo differs".to_string());
    }
    for name in names {
        let (a, b) = (before.voice(name), after.voice(name));
        if a.bars != b.bars {
            differences.push(format!("{name}: bar meter/time grid differs"));
        }
        if a.notes != b.notes {
            let common = a.notes.len().min(b.notes.len());
            let first = (0..common)
                .find(|&i| a.notes[i] != b.notes[i])
                .unwrap_or(common);
            differences.push(format!(
                "{name}: sounding notes differ starting at note {} (pitch, onset or duration)",
                first + 1
            ));
        }
    }
    differences
}

/// Which melody voices a chord-free cover keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeepVoice {
    /// Both melodies (upstream's default; instrumental themes and solos included).
    Both,
    /// Only `Vocal`; `Ins` becomes rests on the same grid.
    Vocal,
    /// Only `Ins`; `Vocal` becomes rests on the same grid.
    Ins,
}

impl KeepVoice {
    /// The kept voice names.
    pub fn names(self) -> &'static [&'static str] {
        match self {
            KeepVoice::Both => &VOICES,
            KeepVoice::Vocal => &["Vocal"],
            KeepVoice::Ins => &["Ins"],
        }
    }
}

/// Upstream `strip_chords`: remove every chord symbol from the music lines (header quotes are
/// untouched), optionally silence one voice, and prove the kept melodies, meters and tempo are
/// unchanged. Returns the new text and how many chord symbols were removed.
pub fn strip_chords(text: &str, keep: KeepVoice) -> Result<(String, usize), AbcError> {
    let source = parse(text)?;
    let mut removed = 0;
    let mut out_lines: Vec<String> = Vec::new();
    for (index, line) in text.split_inclusive('\n').enumerate() {
        let Some(&voice) = source.music_lines.get(&index) else {
            out_lines.push(line.to_string());
            continue;
        };
        let name = VOICES[voice];
        let mut result = String::new();
        let mut cursor = 0;
        while cursor < line.len() {
            match token_at(line, cursor) {
                Some((token, end)) => {
                    match token {
                        Token::Chord(_) => removed += 1,
                        Token::Note { duration, .. }
                            if keep != KeepVoice::Both && !keep.names().contains(&name) =>
                        {
                            result.push('z');
                            result.push_str(duration);
                        }
                        _ => result.push_str(&line[cursor..end]),
                    }
                    cursor = end;
                }
                None => {
                    let c = line[cursor..].chars().next().expect("in bounds");
                    result.push(c);
                    cursor += c.len_utf8();
                }
            }
        }
        out_lines.push(result);
    }
    let output = out_lines.concat();
    let result = parse(&output)?;
    let differences = compare(&source, &result, keep.names(), false);
    fail(!differences.is_empty(), || {
        format!("Chord removal changed melody: {differences:?}")
    })?;
    fail(result.chord_count() != 0, || {
        "Chord removal left a chord symbol".into()
    })?;
    if keep != KeepVoice::Both {
        fail(
            VOICES
                .iter()
                .any(|n| !keep.names().contains(n) && !result.voice(n).notes.is_empty()),
            || "Unselected voice was not silenced".into(),
        )?;
    }
    Ok((output, removed))
}
