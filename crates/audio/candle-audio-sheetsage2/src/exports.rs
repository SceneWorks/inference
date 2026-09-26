//! Stitched events → the upstream export set: `events.json` / `events.tsv`, LAB annotations,
//! melody / chord / transcription MIDI, `playback.json` and the (full or melody-only) ABC score (port
//! of `exports_sheetsage2.export_result` and `midi_sheetsage2.py` at `4f89269`).
//!
//! Every file is produced in memory and is byte-identical to upstream's for the same events; the
//! committed oracles under `scripts/reference/sheetsage2/artifacts/` pin that. When the ABC cannot
//! be built, the error is recorded (`abc_error`) and every other artifact is still produced — the
//! partial transcription stays reviewable.

use crate::chord_spelling::{correct_chord_rows, normalize_key_name, pitch_class_to_semitone};
use crate::events::{np_median, Decoded, Event};
use crate::midi::{Instrument, Midi, Note};
use crate::notation::{generate_abc, Interval, RebuiltAbcScore};
use crate::pyfmt::{rows_text, Cell, PyJson};
use crate::tokenizer::Field;
use crate::Error;

/// A melody note on the song timeline `(start, end, pitch, track)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SongNote {
    /// Start, seconds.
    pub start: f64,
    /// End, seconds.
    pub end: f64,
    /// MIDI pitch.
    pub pitch: i64,
    /// `0` = vocal, `1` = instrumental.
    pub track: i64,
}

fn cmp_note(a: &SongNote, b: &SongNote) -> std::cmp::Ordering {
    a.start
        .total_cmp(&b.start)
        .then(a.end.total_cmp(&b.end))
        .then(a.pitch.cmp(&b.pitch))
        .then(a.track.cmp(&b.track))
}

/// Everything one export produces.
#[derive(Clone, Debug)]
pub struct Exports {
    /// Text files by upstream relative path (`events.json`, `chord.lab`, `notation/song_beats.txt`,
    /// `score.abc`, `playback.json`, …), in upstream's write order.
    pub texts: Vec<(String, String)>,
    /// MIDI files by upstream relative path.
    pub midis: Vec<(String, Vec<u8>)>,
    /// The ABC score, when it could be built.
    pub abc: Option<String>,
    /// Why the ABC could not be built.
    pub abc_error: Option<String>,
    /// The rebuilt score (bars, grid) behind [`Exports::abc`].
    pub score: Option<RebuiltAbcScore>,
    /// Melody notes on the song timeline, sorted.
    pub notes: Vec<SongNote>,
    /// Key-corrected chord intervals.
    pub chords: Vec<Interval>,
    /// Normalized key intervals.
    pub keys: Vec<Interval>,
    /// Structure intervals.
    pub structures: Vec<Interval>,
    /// Beat rows `(time, beat, numerator, denominator)`, when the rhythm decoded.
    pub beats: Vec<[f64; 4]>,
    /// Notation / playback diagnostics.
    pub diagnostics: Vec<String>,
    /// Decoded events.
    pub events: usize,
}

impl Exports {
    /// The text file at `path`.
    pub fn text(&self, path: &str) -> Option<&str> {
        self.texts
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, t)| t.as_str())
    }

    /// The MIDI file at `path`.
    pub fn midi(&self, path: &str) -> Option<&[u8]> {
        self.midis
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, b)| b.as_slice())
    }

    /// Vocal-track notes.
    pub fn vocal_notes(&self) -> usize {
        self.notes.iter().filter(|n| n.track == 0).count()
    }

    /// Instrumental-track notes.
    pub fn instrumental_notes(&self) -> usize {
        self.notes.iter().filter(|n| n.track == 1).count()
    }
}

fn interval_rows(events: &[Event], field: Field, duration: f64) -> Vec<Interval> {
    let mut rows: Vec<Interval> = events
        .iter()
        .filter(|e| e.has(field))
        .map(|e| {
            let value = match field {
                Field::Chord => e.values.chord.clone(),
                Field::Key => e.values.key.clone(),
                Field::Structure => e.values.structure.clone(),
                _ => unreachable!("interval fields only"),
            };
            (e.time(), 0.0, value.expect("present field has a value"))
        })
        .collect();
    for i in 0..rows.len() {
        rows[i].1 = if i + 1 < rows.len() {
            rows[i + 1].0
        } else {
            duration
        };
    }
    rows.retain(|r| r.1 > r.0);
    rows
}

fn interval_cells(rows: &[Interval]) -> Vec<Vec<Cell>> {
    rows.iter()
        .map(|(a, b, v)| vec![Cell::Float(*a), Cell::Float(*b), Cell::Text(v.clone())])
        .collect()
}

fn beat_cells(rows: &[[f64; 4]]) -> Vec<Vec<Cell>> {
    rows.iter()
        .map(|r| {
            vec![
                Cell::Float(r[0]),
                Cell::Int(r[1] as i64),
                Cell::Int(r[2] as i64),
                Cell::Int(r[3] as i64),
            ]
        })
        .collect()
}

/// Upstream `rhythm_rows`: beat rows from the decoded eighth-note positions and running meter.
fn rhythm_rows(events: &[Event]) -> Result<Vec<[f64; 4]>, String> {
    let mut rows = Vec::new();
    let mut meter: Option<(u32, u32)> = None;
    for event in events {
        let Some(rhythm) = event.values.rhythm.as_ref() else {
            continue;
        };
        if let Some(m) = rhythm.meter() {
            meter = Some(m);
        }
        let (Some(eighth), Some((n, d))) = (rhythm.eighth_position(), meter) else {
            continue;
        };
        let numerator = u64::from(eighth) * u64::from(d);
        if numerator % 8 != 0 {
            return Err(format!(
                "Eighth position {eighth} is off the {n}/{d} beat grid"
            ));
        }
        let position = numerator / 8;
        if position >= u64::from(n) {
            return Err(format!(
                "Eighth position {eighth} is outside meter ({n}, {d})"
            ));
        }
        rows.push([
            event.time(),
            (position + 1) as f64,
            f64::from(n),
            f64::from(d),
        ]);
    }
    Ok(rows)
}

/// `_midi(notes)`: the `Vocal` and `Ins` melody instruments at resolution 960.
fn melody_midi(notes: &[SongNote]) -> Midi {
    let mut midi = Midi::new(960);
    for (track, name) in [(0, "Vocal"), (1, "Ins")] {
        midi.instruments.push(Instrument {
            program: 0,
            name: name.into(),
            notes: notes
                .iter()
                .filter(|n| n.track == track && n.end > n.start)
                .map(|n| Note {
                    velocity: 100,
                    pitch: n.pitch as u8,
                    start: n.start,
                    end: n.end,
                })
                .collect(),
        });
    }
    midi
}

/// `notation_notes`: a monophonic view per track (a note overlapping the next onset is clipped to
/// it), keeping the raw prediction separately.
fn notation_notes(notes: &[SongNote]) -> (Vec<SongNote>, Vec<String>) {
    let mut result = Vec::new();
    let mut diagnostics = Vec::new();
    for track in [0, 1] {
        let mut ordered: Vec<SongNote> =
            notes.iter().copied().filter(|n| n.track == track).collect();
        ordered.sort_by(|a, b| {
            a.start
                .total_cmp(&b.start)
                .then(a.pitch.cmp(&b.pitch))
                .then(a.end.total_cmp(&b.end))
        });
        for i in 0..ordered.len() {
            if i + 1 < ordered.len() && ordered[i].end > ordered[i + 1].start + 1e-6 {
                ordered[i].end = ordered[i + 1].start;
                diagnostics.push(format!(
                    "notation only: clipped track {track} note at {:.3} to next onset",
                    ordered[i].start
                ));
            }
            if ordered[i].end > ordered[i].start + 1e-6 {
                result.push(ordered[i]);
            }
        }
    }
    result.sort_by(cmp_note);
    (result, diagnostics)
}

/// `mir_eval.chord.encode(label, reduce_extended_chords=True)` for the SheetSage2 vocabulary
/// (any root spelling), as `chord_pitches` uses it: the bass below the chord, the chord tones above
/// MIDI 48.
pub fn chord_pitches(label: &str) -> Result<Vec<u8>, Error> {
    if matches!(label, "N" | "X" | "?") {
        return Ok(Vec::new());
    }
    let (root, descriptor) = label
        .split_once(':')
        .ok_or_else(|| Error::Symbolic(format!("chord {label:?} has no quality")))?;
    let (quality, bass) = match descriptor.split_once('/') {
        Some((q, b)) => (q, Some(b)),
        None => (descriptor, None),
    };
    let (quality, extensions) = match quality.split_once('(') {
        Some((q, rest)) => (q, rest.trim_end_matches(')')),
        None => (quality, ""),
    };
    let mut bitmap: [bool; 12] = match quality {
        "maj" => mask(&[0, 4, 7]),
        "min" => mask(&[0, 3, 7]),
        "dim" => mask(&[0, 3, 6]),
        "aug" => mask(&[0, 4, 8]),
        "maj7" => mask(&[0, 4, 7, 11]),
        "min7" => mask(&[0, 3, 7, 10]),
        "7" => mask(&[0, 4, 7, 10]),
        "hdim7" => mask(&[0, 3, 6, 10]),
        "dim7" => mask(&[0, 3, 6, 9]),
        "minmaj7" => mask(&[0, 3, 7, 11]),
        "sus2" => mask(&[0, 2, 7]),
        "sus4" => mask(&[0, 5, 7]),
        "maj6" => mask(&[0, 4, 7, 9]),
        "min6" => mask(&[0, 3, 7, 9]),
        other => {
            return Err(Error::Symbolic(format!(
                "chord quality {other:?} is outside the SheetSage2 vocabulary"
            )))
        }
    };
    for degree in extensions.split(',').filter(|d| !d.is_empty()) {
        if let Some(removed) = degree.strip_prefix('*') {
            bitmap[scale_degree_semitone(removed)?.rem_euclid(12) as usize] = false;
        } else {
            bitmap[scale_degree_semitone(degree)?.rem_euclid(12) as usize] = true;
        }
    }
    let bass = match bass {
        Some(b) => scale_degree_semitone(b)?.rem_euclid(12),
        None => 0,
    };
    bitmap[bass as usize] = true;
    let root = pitch_class_to_semitone(root)?;
    let mut pitches: Vec<u8> = vec![(36 + (root + bass) % 12) as u8];
    for (interval, &on) in bitmap.iter().enumerate() {
        if on {
            pitches.push((48 + root + interval as i64) as u8);
        }
    }
    pitches.sort_unstable();
    pitches.dedup();
    Ok(pitches)
}

fn mask(intervals: &[usize]) -> [bool; 12] {
    let mut bitmap = [false; 12];
    for &i in intervals {
        bitmap[i] = true;
    }
    bitmap
}

/// `mir_eval.chord.scale_degree_to_semitone` (degrees 1–13 with `#`/`b` prefixes).
fn scale_degree_semitone(degree: &str) -> Result<i64, Error> {
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
    let base = match rest {
        "1" => 0,
        "2" => 2,
        "3" => 4,
        "4" => 5,
        "5" => 7,
        "6" => 9,
        "7" => 11,
        "9" => 14,
        "11" => 17,
        "13" => 21,
        _ => {
            return Err(Error::Symbolic(format!(
                "scale degree improperly formed: {degree}"
            )))
        }
    };
    Ok(base + offset)
}

fn measure_map(score: &RebuiltAbcScore, duration: f64) -> PyJson {
    let mut rows = Vec::new();
    let mut position = 0.0f64;
    for m in &score.measures {
        let length = m.abc_numerator() as f64 / m.abc_denominator() as f64;
        let actual = m.numerator as f64 / m.denominator as f64;
        let start = score.beats[m.start_beat].time;
        let end = score.beats[m.end_beat].time;
        rows.push(PyJson::Dict(vec![
            ("index".into(), PyJson::Int(m.index as i64)),
            ("start".into(), PyJson::Float(start)),
            ("end".into(), PyJson::Float(end.min(duration))),
            ("score_start".into(), PyJson::Float(position)),
            ("score_end".into(), PyJson::Float(position + length)),
            (
                "leading_rest".into(),
                PyJson::Float(if m.pad_before { length - actual } else { 0.0 }),
            ),
            (
                "trailing_rest".into(),
                PyJson::Float(if m.pad_before { 0.0 } else { length - actual }),
            ),
        ]));
        position += length;
    }
    PyJson::List(rows)
}

/// `build_playback`: the transcription MIDI (melody + re-articulated chords), the chords-only MIDI
/// and `playback.json`, all derived from the serialized bytes exactly as upstream does.
fn build_playback(
    melody: &[u8],
    chord_rows: &[Interval],
    score: Option<&RebuiltAbcScore>,
    duration: f64,
) -> Result<(PyJson, Vec<u8>, Vec<u8>), Error> {
    let mut midi = Midi::from_bytes(melody)?;
    let mut chord_track = Instrument {
        program: 0,
        name: "Chords".into(),
        notes: Vec::new(),
    };
    let mut warnings = Vec::new();
    let downbeats: Vec<f64> = score
        .map(|s| {
            s.beats
                .iter()
                .filter(|b| b.beat_id == 1)
                .map(|b| b.time)
                .collect()
        })
        .unwrap_or_default();
    for (start, end, label) in chord_rows {
        let (start, end) = (start.max(0.0), end.min(duration));
        let pitches = match chord_pitches(label) {
            Ok(p) => p,
            Err(e) => {
                warnings.push(PyJson::Str(format!("Chord playback skipped {label}: {e}")));
                continue;
            }
        };
        let mut cuts = vec![start];
        cuts.extend(downbeats.iter().copied().filter(|&t| start < t && t < end));
        cuts.push(end);
        for w in cuts.windows(2) {
            if w[1] <= w[0] {
                continue;
            }
            for &pitch in &pitches {
                chord_track.notes.push(Note {
                    velocity: 48,
                    pitch,
                    start: w[0],
                    end: w[1],
                });
            }
        }
    }
    midi.instruments.push(chord_track.clone());
    let transcription = midi.to_bytes();
    let mut chords = Midi::new(midi.resolution);
    chords.instruments.push(chord_track);
    let chords = chords.to_bytes();
    let reloaded = Midi::from_bytes(&transcription)?;
    let tracks = reloaded
        .instruments
        .iter()
        .map(|i| {
            PyJson::Dict(vec![
                ("name".into(), PyJson::Str(i.name.clone())),
                ("program".into(), PyJson::Int(i64::from(i.program))),
                (
                    "notes".into(),
                    PyJson::List(
                        i.notes
                            .iter()
                            .map(|n| {
                                PyJson::Dict(vec![
                                    ("pitch".into(), PyJson::Int(i64::from(n.pitch))),
                                    ("start".into(), PyJson::Float(n.start)),
                                    ("end".into(), PyJson::Float(n.end)),
                                    ("velocity".into(), PyJson::Int(i64::from(n.velocity))),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ])
        })
        .collect();
    let data = PyJson::Dict(vec![
        ("version".into(), PyJson::Int(1)),
        ("duration".into(), PyJson::Float(duration)),
        ("midi".into(), PyJson::Str("transcription.mid".into())),
        ("tracks".into(), PyJson::List(tracks)),
        (
            "measures".into(),
            score.map_or(PyJson::List(Vec::new()), |s| measure_map(s, duration)),
        ),
        ("warnings".into(), PyJson::List(warnings)),
    ]);
    Ok((data, transcription, chords))
}

/// Upstream `export_result(decoded, tokenizer, duration=duration, melody_only=…)` for stitched
/// events. `melody_only` removes chords from the ABC and from playback only; every annotation is
/// unchanged.
pub fn export(decoded: &Decoded, duration: f64, melody_only: bool) -> Result<Exports, Error> {
    let events = &decoded.events;
    let mut texts: Vec<(String, String)> = Vec::new();
    let mut midis: Vec<(String, Vec<u8>)> = Vec::new();
    let mut notes = Vec::new();
    for event in events {
        let start = event.time();
        for note in event.values.melody.iter().flatten() {
            let end = duration.min(note.end_time.expect("stitched notes have an end"));
            if end > start {
                notes.push(SongNote {
                    start,
                    end,
                    pitch: note.pitch,
                    track: note.track,
                });
            }
        }
    }
    notes.sort_by(cmp_note);
    texts.push(("events.json".into(), decoded.to_json().dumps_indent(2)));
    let mut tsv = vec![vec![
        Cell::Text("time".into()),
        Cell::Text("global_subbeat".into()),
        Cell::Text("fields".into()),
    ]];
    for e in events {
        tsv.push(vec![
            Cell::Float(e.time()),
            Cell::Int(e.stitch.expect("stitched").global_subbeat),
            Cell::Text(e.field_text()),
        ]);
    }
    texts.push(("events.tsv".into(), rows_text(&tsv)));
    let melody = melody_midi(&notes).to_bytes();
    let note_cells = |track: Option<i64>| -> Vec<Vec<Cell>> {
        notes
            .iter()
            .filter(|n| track.is_none_or(|t| n.track == t))
            .map(|n| {
                let mut row = vec![Cell::Float(n.start), Cell::Float(n.end), Cell::Int(n.pitch)];
                if track.is_none() {
                    row.push(Cell::Int(n.track));
                }
                row
            })
            .collect()
    };
    texts.push(("melody_full.lab".into(), rows_text(&note_cells(None))));
    let mut track_midis = Vec::new();
    for (track, name) in [(0, "vocal"), (1, "instrumental")] {
        texts.push((
            format!("melody_{name}.lab"),
            rows_text(&note_cells(Some(track))),
        ));
        let selected: Vec<SongNote> = notes.iter().copied().filter(|n| n.track == track).collect();
        track_midis.push((
            format!("melody_{name}.mid"),
            melody_midi(&selected).to_bytes(),
        ));
    }
    let keys: Vec<Interval> = interval_rows(events, Field::Key, duration)
        .into_iter()
        .map(|(a, b, k)| Ok((a, b, normalize_key_name(&k)?)))
        .collect::<Result<_, Error>>()?;
    let chords = correct_chord_rows(&interval_rows(events, Field::Chord, duration), &keys)?;
    let structures = interval_rows(events, Field::Structure, duration);
    texts.push(("chord.lab".into(), rows_text(&interval_cells(&chords))));
    texts.push(("key.lab".into(), rows_text(&interval_cells(&keys))));
    texts.push((
        "structure.lab".into(),
        rows_text(&interval_cells(&structures)),
    ));
    let raw_rhythm: Vec<Vec<Cell>> = events
        .iter()
        .filter(|e| e.has(Field::Rhythm) || e.has(Field::Timestamp))
        .map(|e| {
            vec![
                Cell::Float(e.time()),
                Cell::Text(
                    e.values
                        .rhythm
                        .as_ref()
                        .map_or(PyJson::Dict(Vec::new()), |r| r.to_json())
                        .dumps(),
                ),
            ]
        })
        .collect();
    texts.push(("rhythm_events.lab".into(), rows_text(&raw_rhythm)));

    let mut diagnostics = Vec::new();
    let mut beats_out = Vec::new();
    let mut notation_midi = None;
    let abc_result: Result<(String, RebuiltAbcScore), String> = (|| {
        let beats = rhythm_rows(events)?;
        texts.push(("beat.lab".into(), rows_text(&beat_cells(&beats))));
        let downbeats: Vec<Vec<Cell>> = beats
            .iter()
            .filter(|r| r[1] == 1.0)
            .map(|r| vec![Cell::Float(r[0])])
            .collect();
        texts.push(("downbeat.lab".into(), rows_text(&downbeats)));
        beats_out = beats.clone();
        if beats.len() < 2 {
            return Err("At least two decoded beats are required for ABC".into());
        }
        let tail = &beats[beats.len().saturating_sub(9)..];
        let diffs: Vec<f64> = tail.windows(2).map(|w| w[1][0] - w[0][0]).collect();
        let period = np_median(&diffs);
        if period <= 0.0 {
            return Err("Decoded beats must increase in time".into());
        }
        let last_note_end = notes.iter().map(|n| n.end).fold(0.0f64, f64::max);
        let end = duration.max(if notes.is_empty() { 0.0 } else { last_note_end });
        let mut abc_beats = beats.clone();
        while abc_beats.last().expect("non-empty")[0] < end - 1e-6 {
            let prev = *abc_beats.last().expect("non-empty");
            abc_beats.push([
                prev[0] + period,
                ((prev[1] as i64) % (prev[2] as i64) + 1) as f64,
                prev[2],
                prev[3],
            ]);
        }
        texts.push((
            "notation/song_beats.txt".into(),
            rows_text(&beat_cells(&abc_beats)),
        ));
        let first = abc_beats[0][0];
        let last = abc_beats[abc_beats.len() - 1][0];
        let clip = |rows: Vec<Interval>| -> Vec<Interval> {
            rows.into_iter()
                .filter(|(a, b, _)| *b > first && *a < last)
                .map(|(a, b, v)| (first.max(a), last.min(b), v))
                .collect()
        };
        let raw_keys = interval_rows(events, Field::Key, duration);
        let notation_chords = clip(
            correct_chord_rows(&interval_rows(events, Field::Chord, duration), &raw_keys)
                .map_err(|e| e.to_string())?,
        );
        let notation_keys = clip(
            raw_keys
                .iter()
                .map(|(a, b, k)| Ok((*a, *b, normalize_key_name(k)?)))
                .collect::<Result<Vec<_>, Error>>()
                .map_err(|e| e.to_string())?,
        );
        let notation_structures = clip(interval_rows(events, Field::Structure, duration));
        texts.push((
            "notation/song_chords.txt".into(),
            rows_text(&interval_cells(&notation_chords)),
        ));
        texts.push((
            "notation/song_keys.txt".into(),
            rows_text(&interval_cells(&notation_keys)),
        ));
        texts.push((
            "notation/song_structures.txt".into(),
            rows_text(&interval_cells(&notation_structures)),
        ));
        if raw_keys.is_empty() {
            return Err("No key was decoded; cannot construct a keyed ABC score".into());
        }
        let (clean, adjustments) = notation_notes(&notes);
        diagnostics.extend(adjustments);
        let song_melody = melody_midi(&clean).to_bytes();
        notation_midi = Some(song_melody.clone());
        let (text, score) = generate_abc(
            &song_melody,
            &abc_beats,
            &notation_chords,
            &notation_keys,
            &notation_structures,
            melody_only,
        )
        .map_err(|e| e.0)?;
        diagnostics.extend(score.diagnostics.iter().cloned());
        Ok((text, score))
    })();
    let (abc, score, abc_error) = match abc_result {
        Ok((text, score)) => {
            texts.push(("score.abc".into(), text.clone()));
            (Some(text), Some(score), None)
        }
        Err(message) => (None, None, Some(message)),
    };
    let (playback, transcription, chord_midi) = build_playback(
        &melody,
        if melody_only { &[] } else { &chords },
        score.as_ref(),
        duration,
    )?;
    if let PyJson::Dict(items) = &playback {
        if let Some((_, PyJson::List(warnings))) = items.iter().find(|(k, _)| k == "warnings") {
            diagnostics.extend(warnings.iter().filter_map(|w| match w {
                PyJson::Str(s) => Some(s.clone()),
                _ => None,
            }));
        }
    }
    texts.push(("playback.json".into(), playback.dumps_indent(2)));
    midis.push(("melody.mid".into(), melody));
    midis.extend(track_midis);
    midis.push(("transcription.mid".into(), transcription));
    midis.push(("chords.mid".into(), chord_midi));
    if let Some(bytes) = notation_midi {
        midis.push(("notation/song_melody.mid".into(), bytes));
    }
    Ok(Exports {
        texts,
        midis,
        abc,
        abc_error,
        score,
        notes,
        chords,
        keys,
        structures,
        beats: beats_out,
        diagnostics,
        events: events.len(),
    })
}

#[cfg(test)]
mod tests;
