//! Key-aware chord spelling — port of `chord_spelling_sheetsage2.py` at the **head** code revision
//! `m-a-p/SheetSage2@4f89269db831bdc1880124164a00d4f9385cd129` (identical to `55bfe14e…`, "Fix chord
//! spelling for non-C candidate roots").
//!
//! Why the head and not the release (`eab522a`, whose weights this crate loads — the weight LFS
//! object is identical at both revisions): the release spells every chord with sharps, so an
//! Eb-major recording gets `K:Eb` with `D#` / `G#` / `A#` chord symbols — an internally inconsistent
//! score that a `cot=full` cover would feed straight to YuE2. The head spells each chord against
//! its local key (`Eb`, `Ab`, `Bb`). The committed `synth_eb_head` oracle pins this choice; the
//! `synth_eb_release` oracle is deliberately *not* reproduced.

use crate::Error;

const KEY_MAP: [[&str; 12]; 7] = [
    [
        "C:major", "Db:major", "D:major", "Eb:major", "E:major", "F:major", "F#:major", "G:major",
        "Ab:major", "A:major", "Bb:major", "B:major",
    ],
    [
        "D:dorian",
        "Eb:dorian",
        "E:dorian",
        "F:dorian",
        "F#:dorian",
        "G:dorian",
        "G#:dorian",
        "A:dorian",
        "Bb:dorian",
        "B:dorian",
        "C:dorian",
        "C#:dorian",
    ],
    [
        "E:phrygian",
        "F:phrygian",
        "F#:phrygian",
        "G:phrygian",
        "G#:phrygian",
        "A:phrygian",
        "A#:phrygian",
        "B:phrygian",
        "C:phrygian",
        "C#:phrygian",
        "D:phrygian",
        "D#:phrygian",
    ],
    [
        "F:lydian",
        "Gb:lydian",
        "G:lydian",
        "Ab:lydian",
        "A:lydian",
        "Bb:lydian",
        "B:lydian",
        "C:lydian",
        "Db:lydian",
        "D:lydian",
        "Eb:lydian",
        "E:lydian",
    ],
    [
        "G:mixolydian",
        "Ab:mixolydian",
        "A:mixolydian",
        "Bb:mixolydian",
        "B:mixolydian",
        "C:mixolydian",
        "C#:mixolydian",
        "D:mixolydian",
        "Eb:mixolydian",
        "E:mixolydian",
        "F:mixolydian",
        "F#:mixolydian",
    ],
    [
        "A:minor", "Bb:minor", "B:minor", "C:minor", "C#:minor", "D:minor", "D#:minor", "E:minor",
        "F:minor", "F#:minor", "G:minor", "G#:minor",
    ],
    [
        "B:locrian",
        "C:locrian",
        "C#:locrian",
        "D:locrian",
        "D#:locrian",
        "E:locrian",
        "E#:locrian",
        "F#:locrian",
        "G:locrian",
        "G#:locrian",
        "A:locrian",
        "A#:locrian",
    ],
];

const MODE_NAMES: [&str; 7] = [
    "major",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "minor",
    "locrian",
];
const MODE_STARTS: [i64; 7] = [0, 2, 4, 5, 7, 9, 11];
const NOTE_NAMES: &[u8; 7] = b"CDEFGAB";
const CIRCLE_OF_FIFTH_INV: [i64; 7] = [1, 3, 5, 0, 2, 4, 6];
const MAJOR_SCALE_FIFTHS: [i64; 7] = [0, 2, 4, -1, 1, 3, 5];
const DEFAULT_SPELLING: [&str; 12] = [
    "1", "b2", "2", "b3", "3", "4", "b5", "5", "#5", "6", "b7", "7",
];

/// Chord-quality chroma templates (`QUALITIES`), `2` marking the root.
fn quality_chroma(quality: &str) -> Option<[u8; 12]> {
    Some(match quality {
        "maj" => [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "min" => [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "aug" => [2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "dim" => [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        "sus4" => [2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        "sus4(b7)" => [2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        "sus4(b7,9)" => [2, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        "sus2" => [2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "7" => [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "maj7" => [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "min7" => [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        "minmaj7" => [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        "maj6" => [2, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        "min6" => [2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        "9" => [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "maj9" => [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "min9" => [2, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        "7(b9)" => [2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "7(#9)" => [2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
        "maj6(9)" => [2, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        "min6(9)" => [2, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        "maj(9)" => [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "min(9)" => [2, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "maj(11)" => [2, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        "min(11)" => [2, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
        "11" => [2, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
        "maj9(11)" => [2, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        "min11" => [2, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        "13" => [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        "maj13" => [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        "min13" => [2, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        "dim7" => [2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "hdim7" => [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        _ => return None,
    })
}

/// `mir_eval.chord.pitch_class_to_semitone`: letter plus `#`/`b` offsets, wrapped to `0..12`.
pub fn pitch_class_to_semitone(name: &str) -> Result<i64, Error> {
    let mut chars = name.chars();
    let letter = chars
        .next()
        .ok_or_else(|| Error::Symbolic(format!("empty pitch class {name:?}")))?;
    let mut semitone: i64 = match letter {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return Err(Error::Symbolic(format!("invalid pitch class {name:?}"))),
    };
    for c in chars {
        match c {
            '#' => semitone += 1,
            'b' => semitone -= 1,
            _ => {
                return Err(Error::Symbolic(format!(
                    "pitch class improperly formed: {name}"
                )))
            }
        }
    }
    Ok(semitone.rem_euclid(12))
}

/// `scale_degree_to_tuple`: `"b3"` → `(2, -1)`.
fn scale_degree_to_tuple(degree: &str) -> (i64, i64) {
    let (offset, rest) = if degree.starts_with('#') {
        (degree.matches('#').count() as i64, degree.trim_matches('#'))
    } else if degree.starts_with('b') {
        (
            -(degree.matches('b').count() as i64),
            degree.trim_matches('b'),
        )
    } else {
        (0, degree)
    };
    (
        rest.parse::<i64>().expect("table degrees are numeric") - 1,
        offset,
    )
}

fn note_name_to_tuple(name: &str) -> (i64, i64) {
    let letter = NOTE_NAMES
        .iter()
        .position(|&c| c == name.as_bytes()[0])
        .expect("KEY_MAP tonics are letters") as i64;
    let offset = if name.ends_with('#') {
        name.matches('#').count() as i64
    } else if name.ends_with('b') {
        -(name.matches('b').count() as i64)
    } else {
        0
    };
    (letter, offset)
}

fn quality_spelling(quality: &str) -> Option<Vec<(i64, i64)>> {
    let chroma = quality_chroma(quality)?;
    let mut spelling = Vec::new();
    for (i, &value) in chroma.iter().enumerate() {
        if value == 0 {
            continue;
        }
        let degree = if quality.contains("#9") && i == 3 {
            "#2"
        } else if quality.contains("dim7") && i == 9 {
            "bb7"
        } else {
            DEFAULT_SPELLING[i]
        };
        spelling.push(scale_degree_to_tuple(degree));
    }
    Some(spelling)
}

/// `spell_chord_tones`: spell every tone of `quality` above `root` on the circle of fifths.
fn spell_chord_tones(root: (i64, i64), intervals: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let root_pos = CIRCLE_OF_FIFTH_INV[root.0 as usize] + root.1 * 7;
    intervals
        .iter()
        .map(|&(degree, offset)| {
            let letter = (root.0 + degree).rem_euclid(7);
            let tone_pos = root_pos + MAJOR_SCALE_FIFTHS[degree as usize] + offset * 7;
            (
                letter,
                (tone_pos - CIRCLE_OF_FIFTH_INV[letter as usize]).div_euclid(7),
            )
        })
        .collect()
}

/// `score_spelling_under_key` (the root counts twice).
fn score_spelling_under_key(spelling: &[(i64, i64)], key: (i64, i64)) -> f64 {
    let key_pos = CIRCLE_OF_FIFTH_INV[key.0 as usize] + key.1 * 7;
    let relative: Vec<i64> = spelling
        .iter()
        .map(|&(letter, offset)| {
            let pos = CIRCLE_OF_FIFTH_INV[letter.rem_euclid(7) as usize] + offset * 7;
            ((pos - (key_pos + 2)).abs() - 3).max(0)
        })
        .collect();
    relative.iter().sum::<i64>() as f64 + relative.first().copied().unwrap_or(0) as f64
}

/// `normalize_key_name`: the canonical tonal spelling of a `tonic:mode` key.
pub fn normalize_key_name(key: &str) -> Result<String, Error> {
    let (tonic, mode) = key
        .split_once(':')
        .ok_or_else(|| Error::Symbolic(format!("key {key:?} has no mode")))?;
    let mode_id = MODE_NAMES
        .iter()
        .position(|&m| m == mode)
        .ok_or_else(|| Error::Symbolic(format!("unknown key mode {mode:?}")))?;
    let scale = (pitch_class_to_semitone(tonic)? - MODE_STARTS[mode_id]).rem_euclid(12);
    Ok(KEY_MAP[mode_id][scale as usize].to_string())
}

/// `correct_chord_spelling`: respell `label`'s root against `key` (inversion and quality kept).
pub fn correct_chord_spelling(label: &str, key: &str) -> Result<String, Error> {
    if label == "N" || label == "X" {
        return Ok(label.to_string());
    }
    let (key_tonal, key_mode) = key
        .split_once(':')
        .ok_or_else(|| Error::Symbolic(format!("key {key:?} has no mode")))?;
    let mode_id = MODE_NAMES
        .iter()
        .position(|&m| m == key_mode)
        .ok_or_else(|| Error::Symbolic(format!("unknown key mode {key_mode:?}")))?;
    let scale = (pitch_class_to_semitone(key_tonal)? - MODE_STARTS[mode_id]).rem_euclid(12);
    let key_tonic = KEY_MAP[0][scale as usize]
        .split(':')
        .next()
        .expect("KEY_MAP entries have a tonic");
    let key_spelling = note_name_to_tuple(key_tonic);
    let (body, inversion) = match label.split_once('/') {
        Some((body, inversion)) => (body, inversion),
        None => (label, ""),
    };
    let (root, quality) = body
        .split_once(':')
        .ok_or_else(|| Error::Symbolic(format!("chord {label:?} has no quality")))?;
    let intervals = quality_spelling(quality)
        .ok_or_else(|| Error::Symbolic(format!("chord quality {quality:?} has no spelling")))?;
    let root_semitone = pitch_class_to_semitone(root)?;
    let offset_for = |letter: usize| -> Result<i64, Error> {
        let natural = pitch_class_to_semitone(&(NOTE_NAMES[letter] as char).to_string())?;
        Ok((root_semitone - natural + 6).rem_euclid(12) - 6)
    };
    let mut best = 0usize;
    let mut best_score = f64::INFINITY;
    for letter in 0..7 {
        let spelling = spell_chord_tones((letter as i64, offset_for(letter)?), &intervals);
        let score = score_spelling_under_key(&spelling, key_spelling);
        // np.argmin: the first minimum.
        if score < best_score {
            best_score = score;
            best = letter;
        }
    }
    let offset = offset_for(best)?;
    let mut root_name = (NOTE_NAMES[best] as char).to_string();
    if offset < 0 {
        root_name.extend(std::iter::repeat_n('b', offset.unsigned_abs() as usize));
    } else {
        root_name.extend(std::iter::repeat_n('#', offset as usize));
    }
    Ok(if inversion.is_empty() {
        format!("{root_name}:{quality}")
    } else {
        format!("{root_name}:{quality}/{inversion}")
    })
}

/// An interval row `(start, end, label)`.
pub type IntervalRow = (f64, f64, String);

/// `correct_chord_rows`: spell each chord against the key active at its midpoint (left-boundary tie
/// rule of `np.searchsorted`); without keys, rows are unchanged.
pub fn correct_chord_rows(
    chords: &[IntervalRow],
    keys: &[IntervalRow],
) -> Result<Vec<IntervalRow>, Error> {
    if keys.is_empty() {
        return Ok(chords.to_vec());
    }
    let boundaries: Vec<f64> = keys[..keys.len() - 1].iter().map(|k| k.1).collect();
    chords
        .iter()
        .map(|(start, end, label)| {
            let mid = (start + end) / 2.0;
            let index = boundaries.partition_point(|&b| b < mid);
            Ok((*start, *end, correct_chord_spelling(label, &keys[index].2)?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_normalize_to_the_lm_spelling() {
        assert_eq!(normalize_key_name("C#:major").unwrap(), "Db:major");
        assert_eq!(normalize_key_name("D#:minor").unwrap(), "D#:minor");
        assert_eq!(normalize_key_name("A#:minor").unwrap(), "Bb:minor");
        assert_eq!(normalize_key_name("D#:major").unwrap(), "Eb:major");
        assert!(normalize_key_name("C:blues").is_err());
    }

    /// The head-revision behaviour the `synth_eb_head` oracle shows: flat roots in a flat key,
    /// sharp in a sharp key, inversion and quality untouched.
    #[test]
    fn chords_are_spelled_against_the_local_key() {
        assert_eq!(
            correct_chord_spelling("D#:maj", "Eb:major").unwrap(),
            "Eb:maj"
        );
        assert_eq!(
            correct_chord_spelling("G#:maj", "Eb:major").unwrap(),
            "Ab:maj"
        );
        assert_eq!(
            correct_chord_spelling("A#:7/b7", "Eb:major").unwrap(),
            "Bb:7/b7"
        );
        assert_eq!(
            correct_chord_spelling("F#:min", "E:major").unwrap(),
            "F#:min"
        );
        assert_eq!(
            correct_chord_spelling("C:maj7/7", "C:major").unwrap(),
            "C:maj7/7"
        );
        assert_eq!(correct_chord_spelling("N", "C:major").unwrap(), "N");
    }
}
