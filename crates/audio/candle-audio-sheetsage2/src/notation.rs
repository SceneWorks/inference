//! Timed events → validated two-voice ABC (port of `notation_sheetsage2.py` at `4f89269`, which is
//! byte-identical to the release's).
//!
//! The dialect is the one YuE2 plans in: `X:1`, empty `T:`, `M:`, `L:1/n`, `Q:1/4=bpm`, the fixed
//! `V: Vocal …` / `V: Ins …` voice lines, `K:`, then groups of at most four bars per voice with
//! `% section` comments, chord symbols only in `Vocal`, inline `[K:…]` key changes, ties, and `Z`
//! whole-bar rests. [`score_to_abc`] validates its own output ([`validate_serialized_abc`]) before
//! returning it, exactly as upstream does, so a score that fails any structural invariant is an
//! error, never a silently wrong file.

use std::collections::HashMap;

use crate::midi::{Midi, Note};

/// Sub-beats per beat.
pub const SUBBEAT_DIVISION: usize = 4;
/// The two voices, in score order.
pub const VOICE_IDS: [&str; 2] = ["Vocal", "Ins"];

/// A deterministic reconstruction failure (upstream `AbcRebuildError` and its subclasses).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct AbcRebuildError(pub String);

fn err<T>(message: impl Into<String>) -> Result<T, AbcRebuildError> {
    Err(AbcRebuildError(message.into()))
}

/// One beat row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeatEvent {
    /// Seconds.
    pub time: f64,
    /// 1-based beat within the bar.
    pub beat_id: i64,
    /// Declared meter numerator.
    pub declared_numerator: i64,
    /// Meter denominator.
    pub denominator: i64,
    /// 1-based row number.
    pub line_no: usize,
}

/// One bar of the score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measure {
    /// Bar index.
    pub index: usize,
    /// First beat row.
    pub start_beat: usize,
    /// One past the last beat row.
    pub end_beat: usize,
    /// Beats actually in the bar.
    pub numerator: i64,
    /// Meter denominator.
    pub denominator: i64,
    /// A leading partial bar.
    pub pickup: bool,
    /// A trailing partial bar.
    pub partial: bool,
    /// Whether the meter was inferred rather than declared.
    pub inferred: bool,
    /// Notated numerator when it differs from the beat count.
    pub notated_numerator: Option<i64>,
    /// Notated denominator when it differs.
    pub notated_denominator: Option<i64>,
    /// Pad the bar with a leading (not trailing) rest.
    pub pad_before: bool,
}

impl Measure {
    /// First sub-beat.
    pub fn start_t(&self) -> usize {
        self.start_beat * SUBBEAT_DIVISION
    }
    /// One past the last sub-beat.
    pub fn end_t(&self) -> usize {
        self.end_beat * SUBBEAT_DIVISION
    }
    /// Notated numerator.
    pub fn abc_numerator(&self) -> i64 {
        match self.notated_numerator {
            Some(n) if n != 0 => n,
            _ => self.numerator,
        }
    }
    /// Notated denominator.
    pub fn abc_denominator(&self) -> i64 {
        match self.notated_denominator {
            Some(d) if d != 0 => d,
            _ => self.denominator,
        }
    }
}

/// The score on the sub-beat grid.
#[derive(Clone, Debug)]
pub struct RebuiltAbcScore {
    /// Beat rows.
    pub beats: Vec<BeatEvent>,
    /// Bars.
    pub measures: Vec<Measure>,
    /// Seconds of every sub-beat boundary.
    pub subbeat_times: Vec<f64>,
    /// Quarter-note position of every sub-beat boundary.
    pub subbeat_quarters: Vec<f64>,
    /// Meter denominator of every sub-beat.
    pub subbeat_denominators: Vec<i64>,
    /// ABC key of every sub-beat.
    pub key_arr: Vec<String>,
    /// Chord label of every sub-beat (`N` = none).
    pub chord_arr: Vec<String>,
    /// `(sub-beat, label)` section starts.
    pub structure_events: Vec<(usize, String)>,
    /// Per voice: `0` = rest, `pitch*2+2` = sustain, `pitch*2+3` = onset.
    pub voice_arrs: [Vec<i32>; 2],
    /// Notation diagnostics (padding, inferred meters).
    pub diagnostics: Vec<String>,
}

/// An interval row `(start, end, label)`.
pub type Interval = (f64, f64, String);

fn parse_beats(rows: &[[f64; 4]]) -> Result<Vec<BeatEvent>, AbcRebuildError> {
    let mut beats: Vec<BeatEvent> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let line_no = i + 1;
        let as_int = |v: f64| -> Result<i64, AbcRebuildError> {
            if v.fract() != 0.0 || !v.is_finite() {
                return err(format!("beats:{line_no}: invalid beat row"));
            }
            Ok(v as i64)
        };
        let beat = BeatEvent {
            time: row[0],
            beat_id: as_int(row[1])?,
            declared_numerator: as_int(row[2])?,
            denominator: as_int(row[3])?,
            line_no,
        };
        if beat.beat_id < 1 {
            return err(format!("beats:{line_no}: beat ID must be positive"));
        }
        if beat.declared_numerator < 1 {
            return err(format!("beats:{line_no}: meter numerator must be positive"));
        }
        if beat.denominator < 1 || beat.denominator & (beat.denominator - 1) != 0 {
            return err(format!(
                "beats:{line_no}: meter denominator must be a positive power of two"
            ));
        }
        if let Some(last) = beats.last() {
            if beat.time <= last.time {
                return err(format!(
                    "beats:{line_no}: beat times must be strictly increasing"
                ));
            }
        }
        beats.push(beat);
    }
    if beats.len() < 2 {
        return err("beats: at least two beat events are required");
    }
    Ok(beats)
}

fn parse_intervals(
    rows: &[Interval],
    what: &str,
    validate: impl Fn(&str) -> Result<String, AbcRebuildError>,
) -> Result<Vec<Interval>, AbcRebuildError> {
    let mut out = Vec::new();
    let mut previous_end: Option<f64> = None;
    for (i, (start, end, label)) in rows.iter().enumerate() {
        let line_no = i + 1;
        let label = label.trim();
        if end <= start {
            return err(format!("{what}s:{line_no}: {what} end must be after start"));
        }
        if let Some(prev) = previous_end {
            if *start < prev - 1e-6 {
                return err(format!("{what}s:{line_no}: overlapping {what} intervals"));
            }
        }
        out.push((*start, *end, validate(label)?));
        previous_end = Some(*end);
    }
    Ok(out)
}

fn mode_with_first_tiebreak(values: &[i64]) -> i64 {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for v in values {
        *counts.entry(*v).or_default() += 1;
    }
    let maximum = counts.values().copied().max().unwrap_or(0);
    *values
        .iter()
        .find(|v| counts[v] == maximum)
        .expect("non-empty")
}

/// Upstream `infer_measures` (meter-conflict policy `infer`).
pub fn infer_measures(beats: &[BeatEvent]) -> Result<(Vec<Measure>, Vec<String>), AbcRebuildError> {
    let downbeats: Vec<usize> = beats
        .iter()
        .enumerate()
        .filter(|(_, b)| b.beat_id == 1)
        .map(|(i, _)| i)
        .collect();
    if downbeats.is_empty() {
        return err("No downbeat (beat ID 1) exists in the beat lab");
    }
    let mut spans: Vec<(usize, usize, bool, bool)> = Vec::new();
    if downbeats[0] > 0 {
        spans.push((0, downbeats[0], true, false));
    }
    spans.extend(downbeats.windows(2).map(|w| (w[0], w[1], false, false)));
    if *downbeats.last().expect("non-empty") < beats.len() - 1 {
        spans.push((
            *downbeats.last().expect("non-empty"),
            beats.len() - 1,
            false,
            true,
        ));
    }
    if spans.is_empty() {
        return err("No positive-length measure exists between downbeats");
    }
    let mut diagnostics = Vec::new();
    let mut measures = Vec::new();
    for (index, &(start, end, pickup, partial)) in spans.iter().enumerate() {
        let events = &beats[start..end];
        let beat_count = events.len() as i64;
        if beat_count < 1 {
            return err(format!("Measure {index}: empty downbeat span"));
        }
        let ids: Vec<i64> = events.iter().map(|e| e.beat_id).collect();
        let expected: Vec<i64> = (ids[0]..ids[0] + beat_count).collect();
        if ids != expected {
            return err(format!(
                "Measure {index} (beat rows {}-{}): non-consecutive beat IDs {ids:?}",
                events[0].line_no,
                events[events.len() - 1].line_no
            ));
        }
        if !pickup && ids[0] != 1 {
            return err(format!(
                "Measure {index}: full measure does not start at beat ID 1"
            ));
        }
        let denominators: Vec<i64> = events.iter().map(|e| e.denominator).collect();
        let denominator = mode_with_first_tiebreak(&denominators);
        let declared: Vec<i64> = events.iter().map(|e| e.declared_numerator).collect();
        let declared_numerator = mode_with_first_tiebreak(&declared);
        let numerator_conflict = declared.iter().any(|&v| v != beat_count);
        let denominator_conflict = denominators.iter().any(|&v| v != denominator);
        let unique_declared = {
            let mut d = declared.clone();
            d.sort_unstable();
            d.dedup();
            d.len()
        };
        let pad_final_partial = partial
            && unique_declared == 1
            && !denominator_conflict
            && declared_numerator >= beat_count;
        let inferred = pickup || partial || numerator_conflict || denominator_conflict;
        if pad_final_partial && declared_numerator > beat_count {
            diagnostics.push(format!(
                "measure {index}: padded final {beat_count}/{denominator} span to declared \
                 {declared_numerator}/{denominator} with trailing rest"
            ));
        } else if numerator_conflict {
            diagnostics.push(format!(
                "measure {index}: inferred {beat_count}/{denominator} from downbeat span; \
                 declared numerators were {}",
                py_list(&declared)
            ));
        }
        if denominator_conflict {
            diagnostics.push(format!(
                "measure {index}: placed denominator {denominator} at the measure boundary; row \
                 declarations were {}",
                py_list(&denominators)
            ));
        }
        measures.push(Measure {
            index,
            start_beat: start,
            end_beat: end,
            numerator: beat_count,
            denominator,
            pickup,
            partial,
            inferred,
            notated_numerator: Some(if pad_final_partial {
                declared_numerator
            } else {
                beat_count
            }),
            notated_denominator: None,
            pad_before: false,
        });
    }
    if measures.len() >= 2 {
        let first = measures[0];
        let following = measures[1];
        let first_duration = first.numerator as f64 / first.denominator as f64;
        let following_duration =
            following.abc_numerator() as f64 / following.abc_denominator() as f64;
        if first_duration < following_duration {
            measures[0] = Measure {
                inferred: true,
                notated_numerator: Some(following.abc_numerator()),
                notated_denominator: Some(following.abc_denominator()),
                pad_before: true,
                ..first
            };
            diagnostics.push(format!(
                "measure 0: padded leading {}/{} span to {}/{} with preceding rest",
                first.numerator,
                first.denominator,
                following.abc_numerator(),
                following.abc_denominator()
            ));
        }
    }
    Ok((measures, diagnostics))
}

fn py_list(values: &[i64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// `np.linspace(start, end, num)` (endpoint included).
fn linspace(start: f64, end: f64, num: usize) -> Vec<f64> {
    let div = (num - 1) as f64;
    let step = (end - start) / div;
    let mut y: Vec<f64> = (0..num)
        .map(|i| {
            if step == 0.0 {
                (i as f64 / div) * (end - start) + start
            } else {
                i as f64 * step + start
            }
        })
        .collect();
    if num > 1 {
        y[num - 1] = end;
    }
    y
}

type Grid = (Vec<f64>, Vec<f64>, Vec<i64>);

fn build_grid(beats: &[BeatEvent], measures: &[Measure]) -> Result<Grid, AbcRebuildError> {
    let mut interval_denominators = vec![0i64; beats.len() - 1];
    for m in measures {
        for d in &mut interval_denominators[m.start_beat..m.end_beat] {
            *d = m.denominator;
        }
    }
    if interval_denominators.contains(&0) {
        return err("Downbeat spans do not cover every beat interval");
    }
    let mut times = Vec::new();
    let mut denominators = Vec::new();
    let mut quarters = vec![0.0f64];
    let mut current = 0.0f64;
    for index in 0..beats.len() - 1 {
        let denominator = interval_denominators[index];
        let grid = linspace(
            beats[index].time,
            beats[index + 1].time,
            SUBBEAT_DIVISION + 1,
        );
        times.extend_from_slice(&grid[..SUBBEAT_DIVISION]);
        denominators.extend(std::iter::repeat_n(denominator, SUBBEAT_DIVISION));
        let step = 4.0 / denominator as f64 / SUBBEAT_DIVISION as f64;
        for _ in 0..SUBBEAT_DIVISION {
            current += step;
            quarters.push(current);
        }
    }
    times.push(beats[beats.len() - 1].time);
    denominators.push(*interval_denominators.last().expect("non-empty"));
    Ok((times, quarters, denominators))
}

fn boundaries(times: &[f64]) -> Vec<f64> {
    times.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
}

fn searchsorted_left(sorted: &[f64], value: f64) -> usize {
    sorted.partition_point(|&b| b < value)
}

fn fill_intervals(
    rows: &[Interval],
    times: &[f64],
    default: &str,
    width: usize,
) -> Result<Vec<String>, AbcRebuildError> {
    let bounds = boundaries(times);
    let mut result = vec![truncate(default, width); times.len()];
    for (start, end, value) in rows {
        let last = result.len() - 1;
        let s = searchsorted_left(&bounds, *start).min(last);
        let e = searchsorted_left(&bounds, *end).min(last);
        if e <= s {
            return err(format!(
                "Interval {start:.6}-{end:.6} ({value}) is shorter than the ABC subbeat grid"
            ));
        }
        for slot in &mut result[s..e] {
            *slot = truncate(value, width);
        }
    }
    if result.len() > 1 {
        let n = result.len();
        result[n - 1] = result[n - 2].clone();
    }
    Ok(result)
}

/// numpy `<U{width}` storage truncates longer strings.
fn truncate(value: &str, width: usize) -> String {
    value.chars().take(width).collect()
}

fn notes_to_arr(notes: &[Note], times: &[f64], voice: &str) -> Result<Vec<i32>, AbcRebuildError> {
    let mut result = vec![0i32; times.len()];
    let bounds = boundaries(times);
    let mut sorted = notes.to_vec();
    sorted.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.end.total_cmp(&b.end))
            .then(a.pitch.cmp(&b.pitch))
    });
    for note in sorted {
        let last = result.len() - 1;
        let s = searchsorted_left(&bounds, note.start).min(last);
        let e = searchsorted_left(&bounds, note.end).min(last);
        if e <= s {
            return err(format!(
                "{voice}: MIDI note pitch={} at {:.6}-{:.6} cannot be represented on the decoded \
                 subbeat grid",
                note.pitch, note.start, note.end
            ));
        }
        if result[s..e].iter().any(|&v| v != 0) {
            return err(format!(
                "{voice}: overlapping quantized melody notes at subbeats {s}:{e}"
            ));
        }
        let sustain = i32::from(note.pitch) * 2 + 2;
        for slot in &mut result[s..e] {
            *slot = sustain;
        }
        result[s] = sustain + 1;
    }
    Ok(result)
}

const NATURAL_PITCH_CLASS: [(char, i64); 7] = [
    ('C', 0),
    ('D', 2),
    ('E', 4),
    ('F', 5),
    ('G', 7),
    ('A', 9),
    ('B', 11),
];
const LETTERS: [char; 7] = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
const SHARP_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];
const FLAT_NAMES: [&str; 12] = [
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
];

fn natural(letter: char) -> i64 {
    NATURAL_PITCH_CLASS
        .iter()
        .find(|(l, _)| *l == letter)
        .expect("validated letter")
        .1
}

/// `_ROOT_RE`: a letter plus up to two sharps or up to two flats.
fn root_match(root: &str) -> Option<(char, &str)> {
    let mut chars = root.chars();
    let letter = chars.next()?;
    if !LETTERS.contains(&letter) {
        return None;
    }
    let accidental = &root[1..];
    let ok = matches!(accidental, "" | "#" | "##" | "b" | "bb");
    ok.then_some((letter, accidental))
}

fn pitch_class(root: &str) -> Result<(i64, char, String), AbcRebuildError> {
    let (letter, accidental) = root_match(root)
        .ok_or_else(|| AbcRebuildError(format!("Invalid pitch spelling {root:?}")))?;
    let offset = accidental.matches('#').count() as i64 - accidental.matches('b').count() as i64;
    Ok((
        (natural(letter) + offset).rem_euclid(12),
        letter,
        accidental.to_string(),
    ))
}

fn portable_pitch_name(root: &str, preserve_double: bool) -> Result<String, AbcRebuildError> {
    let (pc, _, accidental) = pitch_class(root)?;
    if preserve_double || accidental.len() <= 1 {
        return Ok(root.to_string());
    }
    let names = if accidental.starts_with('#') {
        SHARP_NAMES
    } else {
        FLAT_NAMES
    };
    Ok(names[pc as usize].to_string())
}

fn bass_degree_to_pitch(root: &str, degree_text: &str) -> Result<String, AbcRebuildError> {
    if root_match(degree_text).is_some() {
        return portable_pitch_name(degree_text, true);
    }
    // _BASS_DEGREE_RE: (#{0,2}|b{0,2})([1-9]|1[0-3])
    let split = degree_text
        .find(|c: char| c.is_ascii_digit())
        .ok_or_else(|| AbcRebuildError(format!("Invalid chord bass degree {degree_text:?}")))?;
    let (degree_accidental, digits) = degree_text.split_at(split);
    let valid_accidental = matches!(degree_accidental, "" | "#" | "##" | "b" | "bb");
    let degree: i64 = match digits.parse::<i64>() {
        Ok(d) if valid_accidental && (1..=13).contains(&d) && !digits.starts_with('0') => d,
        _ => {
            return err(format!("Invalid chord bass degree {degree_text:?}"));
        }
    };
    let (root_pc, root_letter, root_accidental) = pitch_class(root)?;
    let scale = [0i64, 2, 4, 5, 7, 9, 11];
    let mut interval = scale[((degree - 1) % 7) as usize] + 12 * ((degree - 1) / 7);
    interval += degree_accidental.matches('#').count() as i64
        - degree_accidental.matches('b').count() as i64;
    let target_pc = (root_pc + interval).rem_euclid(12);
    let letter_index = (LETTERS
        .iter()
        .position(|&l| l == root_letter)
        .expect("letter") as i64
        + degree
        - 1)
    .rem_euclid(7);
    let target_letter = LETTERS[letter_index as usize];
    let difference = (target_pc - natural(target_letter) + 6).rem_euclid(12) - 6;
    if (-2..=2).contains(&difference) {
        let accidental = match difference {
            -2 => "bb",
            -1 => "b",
            0 => "",
            1 => "#",
            _ => "##",
        };
        return Ok(format!("{target_letter}{accidental}"));
    }
    let names = if format!("{root_accidental}{degree_accidental}").contains('#') {
        SHARP_NAMES
    } else {
        FLAT_NAMES
    };
    Ok(names[target_pc as usize].to_string())
}

fn quality_to_abc(quality: &str) -> Option<&'static str> {
    Some(match quality {
        "maj" => "",
        "min" => "m",
        "dim" => "dim",
        "aug" => "aug",
        "7" => "7",
        "maj7" => "maj7",
        "min7" => "m7",
        "dim7" => "dim7",
        "hdim7" => "m7b5",
        "sus4" => "sus4",
        "sus2" => "sus2",
        "maj6" => "6",
        "min6" => "m6",
        "sus4(b7)" => "7sus4",
        "minmaj7" => "m(maj7)",
        _ => return None,
    })
}

/// Upstream `chord_symbol_to_abc`: `Eb:maj7/7` → `Ebmaj7/D`; `N`/`X`/`?` → `None`.
pub fn chord_symbol_to_abc(chord: &str) -> Result<Option<String>, AbcRebuildError> {
    let chord = chord.trim();
    if matches!(chord, "N" | "X" | "?") {
        return Ok(None);
    }
    let (root, descriptor) = chord.split_once(':').ok_or_else(|| {
        AbcRebuildError(format!(
            "Chord {chord:?} is missing the ':' quality separator"
        ))
    })?;
    let (quality, bass) = match descriptor.split_once('/') {
        Some((q, b)) => (q, Some(b)),
        None => (descriptor, None),
    };
    let abc_quality = quality_to_abc(quality).ok_or_else(|| {
        AbcRebuildError(format!(
            "Unsupported chord quality {quality:?} in {chord:?}; refusing to rewrite it as major"
        ))
    })?;
    let mut text = portable_pitch_name(root, true)? + abc_quality;
    if let Some(bass) = bass.filter(|b| !b.is_empty()) {
        text.push('/');
        text.push_str(&bass_degree_to_pitch(root, bass)?);
    }
    Ok(Some(text))
}

fn key_signature_accidentals(key: &str) -> Option<i64> {
    Some(match key {
        "C" | "Am" => 0,
        "G" | "Em" => 1,
        "D" | "Bm" => 2,
        "A" | "F#m" => 3,
        "E" | "C#m" => 4,
        "B" | "G#m" => 5,
        "F#" | "D#m" => 6,
        "C#" | "A#m" => 7,
        "F" | "Dm" => -1,
        "Bb" | "Gm" => -2,
        "Eb" | "Cm" => -3,
        "Ab" | "Fm" => -4,
        "Db" | "Bbm" => -5,
        "Gb" | "Ebm" => -6,
        "Cb" | "Abm" => -7,
        _ => return None,
    })
}

/// Upstream `key_symbol_to_abc`: `Db:major` → `Db`, `C#:minor` → `C#m`.
pub fn key_symbol_to_abc(key: &str) -> Result<String, AbcRebuildError> {
    let key = key.trim();
    let (root, minor) = if let Some((root, mode)) = key.split_once(':') {
        match mode {
            "major" => (root, false),
            "minor" => (root, true),
            _ => return err(format!("Unsupported key mode {mode:?} in {key:?}")),
        }
    } else if let Some(root) = key.strip_suffix('m') {
        (root, true)
    } else {
        (key, false)
    };
    let (pc, _, accidental) = pitch_class(root)?;
    let suffix = if minor { "m" } else { "" };
    let candidate = portable_pitch_name(root, false)? + suffix;
    if key_signature_accidentals(&candidate).is_some() {
        return Ok(candidate);
    }
    let flat = accidental.contains('b');
    let names = if flat { FLAT_NAMES } else { SHARP_NAMES };
    let mut candidate = format!("{}{suffix}", names[pc as usize]);
    if key_signature_accidentals(&candidate).is_none() {
        let fallback = if flat { SHARP_NAMES } else { FLAT_NAMES };
        candidate = format!("{}{suffix}", fallback[pc as usize]);
    }
    if key_signature_accidentals(&candidate).is_none() {
        return err(format!("Cannot encode portable ABC key for {key:?}"));
    }
    Ok(candidate)
}

fn key_accidentals(key: &str) -> Result<[i64; 7], AbcRebuildError> {
    let count = key_signature_accidentals(key)
        .ok_or_else(|| AbcRebuildError(format!("Unsupported ABC key signature {key:?}")))?;
    let mut accidentals = [0i64; 7];
    let order: &[char] = if count > 0 {
        &['F', 'C', 'G', 'D', 'A', 'E', 'B']
    } else {
        &['B', 'E', 'A', 'D', 'G', 'C', 'F']
    };
    for letter in &order[..count.unsigned_abs() as usize] {
        let i = LETTERS.iter().position(|l| l == letter).expect("letter");
        accidentals[i] = if count > 0 { 1 } else { -1 };
    }
    Ok(accidentals)
}

fn key_relative_names(count: i64) -> Option<[&'static str; 12]> {
    Some(match count {
        7 => [
            "B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B",
        ],
        6 => [
            "B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "G##", "A#", "B",
        ],
        5 => [
            "B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B",
        ],
        4 => [
            "B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B",
        ],
        3 => [
            "B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B",
        ],
        2 => [
            "C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B",
        ],
        1 => [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ],
        0 => [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B",
        ],
        -1 => [
            "C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B",
        ],
        -2 => [
            "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
        ],
        -3 => [
            "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
        ],
        -4 => [
            "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
        ],
        -5 => [
            "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb",
        ],
        -6 => [
            "C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb",
        ],
        -7 => [
            "C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "Bbb", "Bb", "Cb",
        ],
        _ => return None,
    })
}

/// Upstream `note_to_abc`: key-relative spelling, writing only bar-state accidental changes.
fn note_to_abc(
    note: i32,
    key_acc: &[i64; 7],
    measure_acc: &mut HashMap<usize, i64>,
) -> Result<String, AbcRebuildError> {
    let count: i64 = key_acc.iter().sum();
    let names = key_relative_names(count).ok_or_else(|| {
        AbcRebuildError(format!(
            "Unsupported key signature accidental count {count}"
        ))
    })?;
    let name = names[note.rem_euclid(12) as usize];
    let letter = name.chars().next().expect("non-empty");
    let accidental_number: i64 = match &name[1..] {
        "" => 0,
        "#" => 1,
        "##" => 2,
        "b" => -1,
        _ => -2,
    };
    let mut octave = i64::from(note - 60).div_euclid(12);
    if note.rem_euclid(12) == 11 && accidental_number == -1 {
        octave += 1;
    } else if note.rem_euclid(12) == 0 && accidental_number == 1 {
        octave -= 1;
    }
    let scale_index = LETTERS.iter().position(|&l| l == letter).expect("letter");
    let current = *measure_acc
        .get(&scale_index)
        .unwrap_or(&key_acc[scale_index]);
    let mut text = String::new();
    if current != accidental_number {
        measure_acc.insert(scale_index, accidental_number);
        text.push_str(match accidental_number {
            -2 => "__",
            -1 => "_",
            0 => "=",
            1 => "^",
            _ => "^^",
        });
    }
    if octave > 0 {
        text.push(letter.to_ascii_lowercase());
        text.extend(std::iter::repeat_n('\'', (octave - 1) as usize));
    } else {
        text.push(letter);
        if octave < 0 {
            text.extend(std::iter::repeat_n(',', octave.unsigned_abs() as usize));
        }
    }
    Ok(text)
}

/// Upstream `build_rebuilt_abc_score_from_data`: the score on the sub-beat grid, from the notation
/// melody MIDI (read back through [`Midi::from_bytes`]), the beat rows and the interval rows.
pub fn build_score(
    melody_midi: &[u8],
    beats: &[[f64; 4]],
    chords: &[Interval],
    keys: &[Interval],
    structures: &[Interval],
    melody_only: bool,
) -> Result<RebuiltAbcScore, AbcRebuildError> {
    let beats = parse_beats(beats)?;
    let keys = parse_intervals(keys, "key", key_symbol_to_abc)?;
    if keys.is_empty() {
        return err("keys: at least one key interval is required");
    }
    let structures = parse_intervals(structures, "structure", |s| Ok(s.to_string()))?;
    let chords = if melody_only {
        Vec::new()
    } else {
        parse_intervals(chords, "chord", |c| {
            chord_symbol_to_abc(c)?;
            Ok(c.to_string())
        })?
    };
    let midi = Midi::from_bytes(melody_midi).map_err(|e| AbcRebuildError(e.to_string()))?;
    let (measures, diagnostics) = infer_measures(&beats)?;
    let (times, quarters, denominators) = build_grid(&beats, &measures)?;
    // _classify_melody_tracks
    let mut vocal = Vec::new();
    let mut ins = Vec::new();
    let mut unknown = Vec::new();
    for instrument in &midi.instruments {
        let name = instrument.name.trim().to_lowercase();
        if name.contains("vocal") {
            vocal.push(instrument);
        } else if name.contains("ins") || name.contains("instrument") {
            ins.push(instrument);
        } else if !instrument.notes.is_empty() {
            unknown.push(instrument);
        }
    }
    if !unknown.is_empty() {
        if vocal.is_empty() && ins.is_empty() && unknown.len() == 1 {
            ins.extend(unknown);
        } else {
            let names: Vec<String> = unknown
                .iter()
                .map(|i| {
                    if i.name.is_empty() {
                        "<unnamed>".into()
                    } else {
                        i.name.clone()
                    }
                })
                .collect();
            return err(format!(
                "Cannot map non-empty melody track(s) {names:?} to fixed Vocal/Ins voices"
            ));
        }
    }
    let collect = |list: &[&crate::midi::Instrument]| -> Vec<Note> {
        list.iter().flat_map(|i| i.notes.iter().copied()).collect()
    };
    let voice_arrs = [
        notes_to_arr(&collect(&vocal), &times, "Vocal")?,
        notes_to_arr(&collect(&ins), &times, "Ins")?,
    ];
    let key_arr = fill_intervals(&keys, &times, &keys[0].2, 16)?;
    let chord_arr = if melody_only {
        vec!["N".to_string(); times.len()]
    } else {
        fill_intervals(&chords, &times, "N", 64)?
    };
    let bounds = boundaries(&times);
    let structure_events = structures
        .iter()
        .map(|(start, _, label)| {
            (
                searchsorted_left(&bounds, *start).min(times.len() - 1),
                label.clone(),
            )
        })
        .collect();
    Ok(RebuiltAbcScore {
        beats,
        measures,
        subbeat_times: times,
        subbeat_quarters: quarters,
        subbeat_denominators: denominators,
        key_arr,
        chord_arr,
        structure_events,
        voice_arrs,
        diagnostics,
    })
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// Upstream `abc_unit_denominator`: `L:1/n` with `n` the lcm of every bar's sub-beat divisor.
pub fn abc_unit_denominator(score: &RebuiltAbcScore) -> Result<i64, AbcRebuildError> {
    let mut lcm = 1i64;
    for m in &score.measures {
        for d in [m.denominator, m.abc_denominator()] {
            let v = d * SUBBEAT_DIVISION as i64;
            lcm = lcm / gcd(lcm, v) * v;
        }
    }
    if lcm > 1024 {
        return err(format!(
            "Required ABC unit length 1/{lcm} is unreasonably small"
        ));
    }
    Ok(lcm)
}

fn measure_padding_units(m: &Measure, unit: i64) -> i64 {
    m.abc_numerator() * unit / m.abc_denominator() - m.numerator * unit / m.denominator
}

fn duration_units(
    score: &RebuiltAbcScore,
    start: usize,
    end: usize,
    unit: i64,
) -> Result<i64, AbcRebuildError> {
    let mut units = 0;
    for &denominator in &score.subbeat_denominators[start..end] {
        let divisor = denominator * SUBBEAT_DIVISION as i64;
        if unit % divisor != 0 {
            return err(format!(
                "ABC L:1/{unit} cannot express a 1/{divisor} subbeat exactly"
            ));
        }
        units += unit / divisor;
    }
    Ok(units)
}

/// Upstream `estimate_tempo` (quarter notes per minute over the whole grid).
pub fn estimate_tempo(score: &RebuiltAbcScore) -> Result<f64, AbcRebuildError> {
    let seconds = score.subbeat_times[score.subbeat_times.len() - 1] - score.subbeat_times[0];
    let quarters =
        score.subbeat_quarters[score.subbeat_quarters.len() - 1] - score.subbeat_quarters[0];
    if seconds <= 0.0 || quarters <= 0.0 {
        return err("Cannot estimate tempo from a zero-duration score");
    }
    Ok(quarters / seconds * 60.0)
}

fn continues_pitch(value: i32, next: i32) -> bool {
    value > 0 && next == (value / 2 - 1) * 2 + 2
}

fn same_note_segment(value: i32, next: i32) -> bool {
    if value == 0 {
        next == 0
    } else {
        next == (value / 2 - 1) * 2 + 2
    }
}

const SUPPORTED_DURATION_UNITS: [i64; 11] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48];

fn split_units(duration: i64) -> Result<Vec<i64>, AbcRebuildError> {
    if duration <= 0 {
        return err(format!("Cannot serialize non-positive duration {duration}"));
    }
    let mut result = Vec::new();
    let mut remaining = duration;
    while remaining != 0 {
        if SUPPORTED_DURATION_UNITS.contains(&remaining) {
            result.push(remaining);
            break;
        }
        let chunk = SUPPORTED_DURATION_UNITS
            .iter()
            .copied()
            .filter(|&v| v < remaining)
            .max()
            .ok_or_else(|| {
                AbcRebuildError(format!(
                    "Duration {duration} cannot be split into representable ABC values"
                ))
            })?;
        result.push(chunk);
        remaining -= chunk;
    }
    Ok(result)
}

fn render_duration_tokens(
    prefix: &str,
    note_text: &str,
    duration: i64,
    tie_out: bool,
) -> Result<Vec<String>, AbcRebuildError> {
    let chunks = split_units(duration)?;
    let n = chunks.len();
    Ok(chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| {
            let continues = note_text != "z" && (i + 1 < n || tie_out);
            format!(
                "{}{note_text}{}{}",
                if i == 0 { prefix } else { "" },
                if chunk == 1 {
                    String::new()
                } else {
                    chunk.to_string()
                },
                if continues { "-" } else { "" }
            )
        })
        .collect())
}

fn render_voice_measure(
    score: &RebuiltAbcScore,
    voice_index: usize,
    measure: &Measure,
    unit: i64,
) -> Result<String, AbcRebuildError> {
    let voice = &score.voice_arrs[voice_index];
    let show_chords = voice_index == 0;
    let mut measure_acc: HashMap<usize, i64> = HashMap::new();
    let mut current_key = score.key_arr[measure.start_t()].clone();
    let mut key_acc = key_accidentals(&current_key)?;
    let mut parts: Vec<String> = Vec::new();
    let padding = measure_padding_units(measure, unit);
    if padding < 0 {
        return err(format!(
            "Measure {}: notated meter is shorter than its decoded span",
            measure.index
        ));
    }
    let mut leading = if measure.pad_before { padding } else { 0 };
    let mut trailing = if measure.pad_before { 0 } else { padding };
    let (start_t, end_t) = (measure.start_t(), measure.end_t());
    let mut t = start_t;
    while t < end_t {
        let mut change_points = vec![end_t];
        if let Some(p) = (t + 1..end_t).find(|&p| !same_note_segment(voice[t], voice[p])) {
            change_points.push(p);
        }
        if let Some(p) = (t + 1..end_t).find(|&p| score.key_arr[p] != score.key_arr[p - 1]) {
            change_points.push(p);
        }
        if show_chords {
            if let Some(p) = (t + 1..end_t).find(|&p| score.chord_arr[p] != score.chord_arr[p - 1])
            {
                change_points.push(p);
            }
        }
        let next_t = *change_points.iter().min().expect("non-empty");
        let mut prefix = String::new();
        let key = &score.key_arr[t];
        if t > start_t && *key != current_key {
            current_key = key.clone();
            key_acc = key_accidentals(&current_key)?;
            measure_acc.clear();
            prefix.push_str(&format!("[K:{current_key}]"));
        }
        if show_chords && (t == start_t || score.chord_arr[t] != score.chord_arr[t - 1]) {
            if let Some(text) = chord_symbol_to_abc(&score.chord_arr[t])? {
                prefix.push_str(&format!("\"{text}\""));
            }
        }
        let value = voice[t];
        let note_text = if value == 0 {
            "z".to_string()
        } else {
            note_to_abc(value / 2 - 1, &key_acc, &mut measure_acc)?
        };
        let mut duration = duration_units(score, t, next_t, unit)?;
        if t == start_t && leading != 0 {
            if value == 0 && prefix.is_empty() {
                duration += leading;
            } else {
                parts.extend(render_duration_tokens("", "z", leading, false)?);
            }
            leading = 0;
        }
        if value == 0 && next_t == end_t && trailing != 0 {
            duration += trailing;
            trailing = 0;
        }
        if duration <= 0 {
            return err(format!(
                "Non-positive ABC duration at subbeats {t}:{next_t}"
            ));
        }
        let tie_out = value > 0 && next_t < voice.len() && continues_pitch(value, voice[next_t]);
        parts.extend(render_duration_tokens(
            &prefix, &note_text, duration, tie_out,
        )?);
        t = next_t;
    }
    if leading != 0 {
        return err(format!(
            "Measure {}: leading rest padding was not serialized",
            measure.index
        ));
    }
    if trailing != 0 {
        parts.extend(render_duration_tokens("", "z", trailing, false)?);
    }
    Ok(parts.concat())
}

/// One element of a serialized music line (upstream `_MUSIC_ELEMENT_RE`).
#[derive(Clone, Debug, PartialEq)]
enum Element {
    Quoted(String),
    Key(String),
    Note {
        note: String,
        duration: String,
        tie: bool,
    },
}

/// Emulates `_MUSIC_ELEMENT_RE.finditer`: `(start, end, element)` of every match, scanning left to
/// right and skipping characters where no alternative matches.
fn music_elements(text: &str) -> Vec<(usize, usize, Element)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        // 1. "…"
        if bytes[i] == b'"' {
            if let Some(close) = text[i + 1..].find('"') {
                let end = i + 1 + close + 1;
                out.push((i, end, Element::Quoted(text[i + 1..end - 1].to_string())));
                i = end;
                continue;
            }
        }
        // 2. [K:…]
        if text[i..].starts_with("[K:") {
            if let Some(close) = text[i + 3..].find(']') {
                if close > 0 {
                    let end = i + 3 + close + 1;
                    out.push((i, end, Element::Key(text[i + 3..end - 1].to_string())));
                    i = end;
                    continue;
                }
            }
        }
        // 3. [_=^]*[A-Ga-gz][,']*\d*-?
        let mut j = i;
        while j < bytes.len() && matches!(bytes[j], b'_' | b'=' | b'^') {
            j += 1;
        }
        if j < bytes.len() && matches!(bytes[j], b'A'..=b'G' | b'a'..=b'g' | b'z') {
            j += 1;
            while j < bytes.len() && matches!(bytes[j], b',' | b'\'') {
                j += 1;
            }
            let note_end = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let duration = text[note_end..j].to_string();
            let tie = j < bytes.len() && bytes[j] == b'-';
            if tie {
                j += 1;
            }
            out.push((
                i,
                j,
                Element::Note {
                    note: text[i..note_end].to_string(),
                    duration,
                    tie,
                },
            ));
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

fn is_compressible_full_rest(rendered: &str) -> bool {
    let mut cursor = 0;
    let mut saw_note = false;
    for (start, end, element) in music_elements(rendered) {
        if start != cursor {
            return false;
        }
        cursor = end;
        match element {
            Element::Quoted(_) | Element::Key(_) => return false,
            Element::Note { note, tie, .. } => {
                saw_note = true;
                if note != "z" || tie {
                    return false;
                }
            }
        }
    }
    saw_note && cursor == rendered.len()
}

fn render_voice_group(
    score: &RebuiltAbcScore,
    voice_index: usize,
    measures: &[Measure],
    unit: i64,
) -> Result<String, AbcRebuildError> {
    let rendered = measures
        .iter()
        .map(|m| render_voice_measure(score, voice_index, m, unit))
        .collect::<Result<Vec<_>, _>>()?;
    let mut parts = String::new();
    let mut index = 0;
    while index < rendered.len() {
        if !is_compressible_full_rest(&rendered[index]) {
            parts.push_str(&rendered[index]);
            parts.push('|');
            index += 1;
            continue;
        }
        let mut end = index + 1;
        while end < rendered.len() && is_compressible_full_rest(&rendered[end]) {
            end += 1;
        }
        let count = end - index;
        parts.push('Z');
        if count > 1 {
            parts.push_str(&count.to_string());
        }
        parts.push('|');
        index = end;
    }
    Ok(parts)
}

/// A group of up to four bars sharing one line per voice.
#[derive(Clone, Debug)]
struct MeasureGroup {
    measures: Vec<Measure>,
    structure_labels: Vec<String>,
    meter_changed: bool,
    key_changed: bool,
}

fn sanitize_label(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn measure_groups(score: &RebuiltAbcScore) -> Vec<MeasureGroup> {
    let first = score.measures[0];
    let mut active_meter = (first.abc_numerator(), first.abc_denominator());
    let mut active_key = score.key_arr[first.start_t()].clone();
    let mut active_structure = String::new();
    let mut groups: Vec<MeasureGroup> = Vec::new();
    for measure in &score.measures {
        let meter = (measure.abc_numerator(), measure.abc_denominator());
        let key = &score.key_arr[measure.start_t()];
        let meter_changed = meter != active_meter;
        let key_changed = *key != active_key;
        let mut labels = Vec::new();
        for (t, label) in &score.structure_events {
            if !(measure.start_t() <= *t && *t < measure.end_t()) {
                continue;
            }
            let clean = sanitize_label(label);
            if !clean.is_empty() && clean != active_structure {
                labels.push(clean.clone());
                active_structure = clean;
            }
        }
        let start_group = groups.is_empty()
            || groups.last().expect("non-empty").measures.len() >= 4
            || meter_changed
            || key_changed
            || !labels.is_empty();
        if start_group {
            groups.push(MeasureGroup {
                measures: vec![*measure],
                structure_labels: labels,
                meter_changed,
                key_changed,
            });
        } else {
            groups
                .last_mut()
                .expect("non-empty")
                .measures
                .push(*measure);
        }
        active_meter = meter;
        active_key = score.key_arr[measure.end_t() - 1].clone();
    }
    groups
}

/// Upstream `score_to_abc`: serialize and validate.
pub fn score_to_abc(score: &RebuiltAbcScore) -> Result<String, AbcRebuildError> {
    let unit = abc_unit_denominator(score)?;
    let first = score.measures[0];
    let first_key = &score.key_arr[first.start_t()];
    let tempo = estimate_tempo(score)?.round_ties_even() as i64;
    let mut lines = vec![
        "X:1".to_string(),
        "T:".to_string(),
        format!("M:{}/{}", first.abc_numerator(), first.abc_denominator()),
        format!("L:1/{unit}"),
        format!("Q:1/4={tempo}"),
        "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"".to_string(),
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"".to_string(),
        format!("K:{first_key}"),
    ];
    for group in measure_groups(score) {
        lines.extend(group.structure_labels.iter().map(|l| format!("% {l}")));
        let head = group.measures[0];
        for (voice_index, voice) in VOICE_IDS.iter().enumerate() {
            lines.push(format!("V: {voice}"));
            if group.meter_changed {
                lines.push(format!(
                    "M:{}/{}",
                    head.abc_numerator(),
                    head.abc_denominator()
                ));
            }
            if group.key_changed {
                lines.push(format!("K:{}", score.key_arr[head.start_t()]));
            }
            lines.push(render_voice_group(
                score,
                voice_index,
                &group.measures,
                unit,
            )?);
        }
    }
    let text = lines.join("\n") + "\n";
    validate_serialized_abc(&text, score)?;
    Ok(text)
}

type ParsedMeasure = (Vec<(i64, String)>, Vec<(i64, String)>);

fn parse_music_measure(
    line: &str,
    expected: i64,
    context: &str,
) -> Result<ParsedMeasure, AbcRebuildError> {
    if line == "Z" {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut position = 0i64;
    let mut cursor = 0usize;
    let mut quoted = Vec::new();
    let mut keys = Vec::new();
    for (start, end, element) in music_elements(line) {
        let gap = &line[cursor..start];
        if !gap.trim().is_empty() {
            return err(format!(
                "{context}: unsupported serialized ABC tokens {gap:?}"
            ));
        }
        cursor = end;
        match element {
            Element::Quoted(q) => quoted.push((position, q)),
            Element::Key(k) => keys.push((position, k)),
            Element::Note {
                note,
                duration,
                tie,
            } => {
                if tie && note == "z" {
                    return err(format!("{context}: a rest cannot be tied"));
                }
                let d: i64 = if duration.is_empty() {
                    1
                } else {
                    duration.parse().map_err(|_| {
                        AbcRebuildError(format!(
                            "{context}: duration {duration} is not parser-representable"
                        ))
                    })?
                };
                if !SUPPORTED_DURATION_UNITS.contains(&d) {
                    return err(format!(
                        "{context}: duration {d} is not parser-representable"
                    ));
                }
                position += d;
            }
        }
    }
    if !line[cursor..].trim().is_empty() {
        return err(format!(
            "{context}: unsupported serialized ABC tokens {:?}",
            &line[cursor..]
        ));
    }
    if position != expected {
        return err(format!(
            "{context}: duration {position} does not match meter duration {expected}"
        ));
    }
    // re.search(r"(^|[\s|])-[_=^A-Ga-g]", line)
    let bytes = line.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'-'
            && (i == 0 || bytes[i - 1].is_ascii_whitespace() || bytes[i - 1] == b'|')
            && i + 1 < bytes.len()
            && matches!(bytes[i + 1], b'_' | b'=' | b'^' | b'A'..=b'G' | b'a'..=b'g')
        {
            return err(format!("{context}: tie is written before its second note"));
        }
    }
    Ok((quoted, keys))
}

fn expected_measure_chords(
    score: &RebuiltAbcScore,
    m: &Measure,
    unit: i64,
) -> Result<Vec<(i64, String)>, AbcRebuildError> {
    let leading = if m.pad_before {
        measure_padding_units(m, unit)
    } else {
        0
    };
    let mut expected = Vec::new();
    for t in m.start_t()..m.end_t() {
        if t != m.start_t() && score.chord_arr[t] == score.chord_arr[t - 1] {
            continue;
        }
        let position = leading + duration_units(score, m.start_t(), t, unit)?;
        if let Some(text) = chord_symbol_to_abc(&score.chord_arr[t])? {
            expected.push((position, text));
        }
    }
    Ok(expected)
}

fn expected_measure_keys(
    score: &RebuiltAbcScore,
    m: &Measure,
    unit: i64,
) -> Result<Vec<(i64, String)>, AbcRebuildError> {
    let leading = if m.pad_before {
        measure_padding_units(m, unit)
    } else {
        0
    };
    let mut out = Vec::new();
    for t in m.start_t() + 1..m.end_t() {
        if score.key_arr[t] != score.key_arr[t - 1] {
            out.push((
                leading + duration_units(score, m.start_t(), t, unit)?,
                score.key_arr[t].clone(),
            ));
        }
    }
    Ok(out)
}

type VoiceGroup = (usize, Vec<(String, String)>, Vec<String>);

fn parse_voice_group(
    lines: &[&str],
    mut cursor: usize,
    voice: &str,
    group_index: usize,
) -> Result<VoiceGroup, AbcRebuildError> {
    let expected = format!("V: {voice}");
    if cursor >= lines.len() || lines[cursor] != expected {
        let observed = lines.get(cursor).copied().unwrap_or("<end>");
        return err(format!(
            "Group {group_index}: expected {expected}, got {observed:?}"
        ));
    }
    cursor += 1;
    let mut fields: Vec<(String, String)> = Vec::new();
    while cursor < lines.len()
        && (lines[cursor].starts_with("M:") || lines[cursor].starts_with("K:"))
    {
        let (name, value) = lines[cursor].split_once(':').expect("prefix checked");
        if fields.iter().any(|(n, _)| n == name) {
            return err(format!(
                "Group {group_index} {voice}: repeated {name}: field"
            ));
        }
        fields.push((name.to_string(), value.to_string()));
        cursor += 1;
    }
    if cursor >= lines.len() {
        return err(format!("Group {group_index} {voice}: missing music line"));
    }
    let music = lines[cursor];
    if ["V:", "M:", "K:", "%"].iter().any(|p| music.starts_with(p)) {
        return err(format!(
            "Group {group_index} {voice}: invalid music line {music:?}"
        ));
    }
    cursor += 1;
    let split: Vec<&str> = music.split('|').collect();
    if split.last().is_none_or(|l| !l.trim().is_empty()) {
        return err(format!(
            "Group {group_index} {voice}: music line must end with a barline"
        ));
    }
    let serialized: Vec<&str> = split[..split.len() - 1].iter().map(|b| b.trim()).collect();
    if serialized.iter().any(|b| b.is_empty()) {
        return err(format!(
            "Group {group_index} {voice}: empty serialized measure"
        ));
    }
    let mut bars = Vec::new();
    for bar in serialized {
        let rest_count = bar.strip_prefix('Z').filter(|rest| {
            rest.is_empty() || (rest.len() == 1 && matches!(rest.as_bytes()[0], b'1'..=b'4'))
        });
        match rest_count {
            None => bars.push(bar.to_string()),
            Some("1") => {
                return err(format!(
                    "Group {group_index} {voice}: Z1 must be written as Z"
                ))
            }
            Some(count) => {
                let n: usize = if count.is_empty() {
                    1
                } else {
                    count.parse().expect("digit")
                };
                bars.extend(std::iter::repeat_n("Z".to_string(), n));
            }
        }
    }
    if !(1..=4).contains(&bars.len()) {
        return err(format!(
            "Group {group_index} {voice}: expected 1-4 semantic measures"
        ));
    }
    Ok((cursor, fields, bars))
}

/// Upstream `validate_serialized_abc`: every structural invariant the score must satisfy before it
/// is written (fixed header and voices, meters, keys, groups, bar durations, chord and key
/// positions, chords only in `Vocal`).
pub fn validate_serialized_abc(text: &str, score: &RebuiltAbcScore) -> Result<(), AbcRebuildError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.first() != Some(&"X:1") {
        return err("ABC must start with X:1");
    }
    if lines.get(1) != Some(&"T:") {
        return err("ABC title must be fixed as empty T:");
    }
    if lines
        .iter()
        .any(|l| l.starts_with("%abc-") || l.starts_with("I:abc-creator"))
    {
        return err("ABC must not contain version or creator metadata");
    }
    if text.contains("%%MIDI gchordoff") {
        return err("ABC must not contain %%MIDI gchordoff");
    }
    if text.contains("% ss2") {
        return err("ABC must not contain % ss2 metadata");
    }
    let header_voices: Vec<&str> = lines
        .iter()
        .filter_map(|l| {
            if l.starts_with("V: Vocal ") {
                Some("Vocal")
            } else if l.starts_with("V: Ins ") {
                Some("Ins")
            } else {
                None
            }
        })
        .collect();
    if header_voices != VOICE_IDS {
        return err(format!(
            "Expected fixed Vocal/Ins voice definitions, got {header_voices:?}"
        ));
    }
    let first_ins = lines.iter().position(|l| l.starts_with("V: Ins "));
    let header_key_index = lines
        .iter()
        .enumerate()
        .position(|(i, l)| i > 0 && l.starts_with("K:") && first_ins.is_some_and(|h| h < i))
        .ok_or_else(|| AbcRebuildError("ABC header K: field is missing".into()))?;
    let first = score.measures[0];
    let expected_meter = format!("M:{}/{}", first.abc_numerator(), first.abc_denominator());
    let header_meters: Vec<&str> = lines[..=header_key_index]
        .iter()
        .copied()
        .filter(|l| l.starts_with("M:"))
        .collect();
    if header_meters != [expected_meter.as_str()] {
        return err(format!(
            "ABC header meters {header_meters:?} != [{expected_meter:?}]"
        ));
    }
    let expected_key = format!("K:{}", score.key_arr[first.start_t()]);
    let header_keys: Vec<&str> = lines[..=header_key_index]
        .iter()
        .copied()
        .filter(|l| l.starts_with("K:"))
        .collect();
    if header_keys != [expected_key.as_str()] {
        return err(format!(
            "ABC header keys {header_keys:?} != [{expected_key:?}]"
        ));
    }
    let unit = abc_unit_denominator(score)?;
    let mut cursor = header_key_index + 1;
    for (group_index, group) in measure_groups(score).iter().enumerate() {
        let mut labels = Vec::new();
        while cursor < lines.len() && lines[cursor].starts_with("% ") {
            labels.push(lines[cursor][2..].trim().to_string());
            cursor += 1;
        }
        if labels != group.structure_labels {
            return err(format!(
                "Group {group_index}: structure labels {labels:?} != {:?}",
                group.structure_labels
            ));
        }
        let (next, vocal_fields, vocal_bars) =
            parse_voice_group(&lines, cursor, "Vocal", group_index)?;
        let (next, ins_fields, ins_bars) = parse_voice_group(&lines, next, "Ins", group_index)?;
        cursor = next;
        if vocal_fields != ins_fields {
            return err(format!(
                "Group {group_index}: meter/key changes must be scoped to both voices"
            ));
        }
        let head = group.measures[0];
        let mut expected_fields = Vec::new();
        if group.meter_changed {
            expected_fields.push((
                "M".to_string(),
                format!("{}/{}", head.abc_numerator(), head.abc_denominator()),
            ));
        }
        if group.key_changed {
            expected_fields.push(("K".to_string(), score.key_arr[head.start_t()].clone()));
        }
        let mut sorted_actual = vocal_fields.clone();
        sorted_actual.sort();
        let mut sorted_expected = expected_fields.clone();
        sorted_expected.sort();
        if sorted_actual != sorted_expected {
            return err(format!(
                "Group {group_index}: fields {vocal_fields:?} != required changes {expected_fields:?}"
            ));
        }
        if vocal_bars.len() != group.measures.len() || ins_bars.len() != group.measures.len() {
            return err(format!(
                "Group {group_index}: both voices must contain {} measures",
                group.measures.len()
            ));
        }
        for (bar_index, m) in group.measures.iter().enumerate() {
            let expected_duration = m.abc_numerator() * unit / m.abc_denominator();
            for (voice, bars) in [("Vocal", &vocal_bars), ("Ins", &ins_bars)] {
                let (quoted, keys) = parse_music_measure(
                    &bars[bar_index],
                    expected_duration,
                    &format!("measure {} {voice}", m.index),
                )?;
                let expected_keys = expected_measure_keys(score, m, unit)?;
                if keys != expected_keys {
                    return err(format!(
                        "Measure {} {voice}: inline keys {keys:?} != {expected_keys:?}",
                        m.index
                    ));
                }
                if voice == "Vocal" {
                    let expected_chords = expected_measure_chords(score, m, unit)?;
                    if quoted != expected_chords {
                        return err(format!(
                            "Measure {}: chord symbols {quoted:?} != {expected_chords:?}",
                            m.index
                        ));
                    }
                } else if !quoted.is_empty() {
                    return err(format!("Measure {}: chords must only be in Vocal", m.index));
                }
            }
        }
    }
    if cursor != lines.len() {
        return err(format!(
            "Unexpected trailing ABC body lines: {:?}",
            &lines[cursor..(cursor + 5).min(lines.len())]
        ));
    }
    Ok(())
}

/// Upstream `generate_abc_from_data`: build, serialize and validate.
pub fn generate_abc(
    melody_midi: &[u8],
    beats: &[[f64; 4]],
    chords: &[Interval],
    keys: &[Interval],
    structures: &[Interval],
    melody_only: bool,
) -> Result<(String, RebuiltAbcScore), AbcRebuildError> {
    let score = build_score(melody_midi, beats, chords, keys, structures, melody_only)?;
    let text = score_to_abc(&score)?;
    Ok((text, score))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chord_symbols_follow_the_abc_dialect() {
        for (label, abc) in [
            ("C:maj", Some("C")),
            ("C:maj7/7", Some("Cmaj7/B")),
            ("C:maj7/5", Some("Cmaj7/G")),
            ("Eb:7/b7", Some("Eb7/Db")),
            ("A:min/b3", Some("Am/C")),
            ("F#:hdim7", Some("F#m7b5")),
            ("Bb:sus4(b7)", Some("Bb7sus4")),
            ("G:minmaj7", Some("Gm(maj7)")),
            ("N", None),
        ] {
            assert_eq!(
                chord_symbol_to_abc(label).unwrap().as_deref(),
                abc,
                "{label}"
            );
        }
        assert!(chord_symbol_to_abc("C:maj9").is_err());
    }

    #[test]
    fn keys_encode_to_portable_signatures() {
        assert_eq!(key_symbol_to_abc("Db:major").unwrap(), "Db");
        assert_eq!(key_symbol_to_abc("C#:minor").unwrap(), "C#m");
        assert_eq!(key_symbol_to_abc("Eb:major").unwrap(), "Eb");
        assert_eq!(key_symbol_to_abc("D#:minor").unwrap(), "D#m");
        assert!(key_symbol_to_abc("C:dorian").is_err());
    }

    #[test]
    fn durations_split_into_parser_values() {
        assert_eq!(split_units(5).unwrap(), vec![4, 1]);
        assert_eq!(split_units(20).unwrap(), vec![16, 4]);
        assert_eq!(split_units(48).unwrap(), vec![48]);
        assert!(split_units(0).is_err());
    }

    #[test]
    fn music_elements_match_the_upstream_regex() {
        let parsed = music_elements("\"C\"^f4-z2[K:Eb]_B,3|");
        let kinds: Vec<String> = parsed.iter().map(|(_, _, e)| format!("{e:?}")).collect();
        assert_eq!(kinds.len(), 5, "{kinds:?}");
        assert!(is_compressible_full_rest("z16"));
        assert!(!is_compressible_full_rest("z8c8"));
        assert!(!is_compressible_full_rest("\"C\"z16"));
    }
}
