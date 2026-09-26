//! Byte parity of the whole symbolic path against upstream's committed outputs.
//!
//! Each case feeds the committed `tokens.txt` oracle (produced by upstream from the digest-pinned
//! model-input array) through the production symbolic path — [`Stitcher::replay`] (decode, overlap
//! prefixes, stitching) then [`export`] — and compares every committed file byte for byte.

use super::*;
use crate::events::parse_tokens_txt;
use crate::pipeline::{
    Stitcher, DEFAULT_LOOKAHEAD_SECONDS, DEFAULT_OVERLAP_SECONDS, MAX_OUTPUT_SEQ_LEN,
};
use crate::tokenizer::{Tokenizer, FULL_TASK_PROMPTS};

macro_rules! artifact {
    ($case:literal, $file:literal) => {
        (
            $file,
            &include_bytes!(concat!(
                "../../../../../scripts/reference/sheetsage2/artifacts/",
                $case,
                "/",
                $file
            ))[..],
        )
    };
}

struct Case {
    name: &'static str,
    duration: f64,
    melody_only: bool,
    tokens: &'static str,
    files: Vec<(&'static str, &'static [u8])>,
}

fn replay(case: &Case) -> Exports {
    let tokenizer = Tokenizer::new(300.0, 100, Some("5ba3325af0344c7f")).unwrap();
    let records = parse_tokens_txt(case.tokens).unwrap();
    let stitcher = Stitcher::new(
        &tokenizer,
        &FULL_TASK_PROMPTS,
        case.duration,
        DEFAULT_OVERLAP_SECONDS,
        DEFAULT_LOOKAHEAD_SECONDS,
        MAX_OUTPUT_SEQ_LEN,
    )
    .unwrap();
    let windows: Vec<Vec<u32>> = records.iter().map(|r| r.tokens.clone()).collect();
    let stitched = stitcher.replay(&windows).unwrap();
    assert!(
        stitched.warnings.is_empty(),
        "{}: {:?}",
        case.name,
        stitched.warnings
    );
    // The replayed plan and prefix lengths are the ones upstream recorded.
    for (ours, theirs) in stitched.records.iter().zip(&records) {
        assert_eq!(ours.prefix_tokens, theirs.prefix_tokens, "{}", case.name);
        assert_eq!(
            crate::events::tokens_txt(&tokenizer, std::slice::from_ref(ours)).unwrap(),
            crate::events::tokens_txt(&tokenizer, std::slice::from_ref(theirs)).unwrap(),
            "{}",
            case.name
        );
    }
    // `synth_eb_release` was written by the release code, whose `describe` printed the raw sharp
    // key (`<key_D#:major>`); the head code this port follows prints `<key_Eb:major>`. The token
    // ids are identical, so only that case's descriptions are exempt.
    if case.name != "synth_eb_release" {
        assert_eq!(
            crate::events::tokens_txt(&tokenizer, &stitched.records).unwrap(),
            case.tokens,
            "{}: tokens.txt",
            case.name
        );
    }
    export(&stitched.decoded, case.duration, case.melody_only).unwrap()
}

fn check(case: Case) {
    let exports = replay(&case);
    for (file, expected) in &case.files {
        let actual: &[u8] = if file.ends_with(".mid") {
            exports
                .midi(file)
                .unwrap_or_else(|| panic!("{}: no {file}", case.name))
        } else {
            exports
                .text(file)
                .unwrap_or_else(|| panic!("{}: no {file}", case.name))
                .as_bytes()
        };
        if actual != *expected {
            let a = String::from_utf8_lossy(actual);
            let e = String::from_utf8_lossy(expected);
            let line = a
                .lines()
                .zip(e.lines())
                .position(|(x, y)| x != y)
                .unwrap_or(0);
            panic!(
                "{}: {file} differs at line {line}:\n ours:   {:?}\n theirs: {:?}",
                case.name,
                a.lines().nth(line),
                e.lines().nth(line)
            );
        }
    }
}

/// Mutation that must fail: spell chords with the release's sharps (`correct_chord_rows` → identity),
/// or round MIDI ticks half-up.
#[test]
fn synth_full_matches_upstream_byte_for_byte() {
    check(Case {
        name: "synth_full",
        duration: 39.4,
        melody_only: false,
        tokens: include_str!(
            "../../../../../scripts/reference/sheetsage2/artifacts/synth_full/tokens.txt"
        ),
        files: vec![
            artifact!("synth_full", "score.abc"),
            artifact!("synth_full", "events.json"),
            artifact!("synth_full", "events.tsv"),
            artifact!("synth_full", "beat.lab"),
            artifact!("synth_full", "downbeat.lab"),
            artifact!("synth_full", "chord.lab"),
            artifact!("synth_full", "key.lab"),
            artifact!("synth_full", "structure.lab"),
            artifact!("synth_full", "melody_full.lab"),
            artifact!("synth_full", "melody_vocal.lab"),
            artifact!("synth_full", "melody_instrumental.lab"),
            artifact!("synth_full", "rhythm_events.lab"),
            artifact!("synth_full", "playback.json"),
            artifact!("synth_full", "notation/song_beats.txt"),
            artifact!("synth_full", "notation/song_chords.txt"),
            artifact!("synth_full", "notation/song_keys.txt"),
            artifact!("synth_full", "notation/song_structures.txt"),
            artifact!("synth_full", "melody.mid"),
            artifact!("synth_full", "melody_vocal.mid"),
            artifact!("synth_full", "melody_instrumental.mid"),
            artifact!("synth_full", "transcription.mid"),
            artifact!("synth_full", "chords.mid"),
            artifact!("synth_full", "notation/song_melody.mid"),
        ],
    });
}

#[test]
fn real_full_matches_upstream_byte_for_byte() {
    check(Case {
        name: "real_full",
        duration: 60.0,
        melody_only: false,
        tokens: include_str!(
            "../../../../../scripts/reference/sheetsage2/artifacts/real_full/tokens.txt"
        ),
        files: vec![
            artifact!("real_full", "score.abc"),
            artifact!("real_full", "events.json"),
            artifact!("real_full", "events.tsv"),
            artifact!("real_full", "beat.lab"),
            artifact!("real_full", "chord.lab"),
            artifact!("real_full", "key.lab"),
            artifact!("real_full", "structure.lab"),
            artifact!("real_full", "melody_vocal.lab"),
            artifact!("real_full", "rhythm_events.lab"),
            artifact!("real_full", "playback.json"),
            artifact!("real_full", "melody.mid"),
            artifact!("real_full", "transcription.mid"),
            artifact!("real_full", "chords.mid"),
        ],
    });
}

/// `melody_only`: no chord symbols, and the rest merging upstream does where a chord change had
/// split a rest (`…g4z2z2|` → `…g4z4|`).
#[test]
fn real_melody_only_matches_upstream_byte_for_byte() {
    check(Case {
        name: "real_melody",
        duration: 60.0,
        melody_only: true,
        tokens: include_str!(
            "../../../../../scripts/reference/sheetsage2/artifacts/real_melody/tokens.txt"
        ),
        files: vec![
            artifact!("real_melody", "score.abc"),
            artifact!("real_melody", "playback.json"),
            artifact!("real_melody", "transcription.mid"),
            artifact!("real_melody", "chords.mid"),
            artifact!("real_melody", "chord.lab"),
        ],
    });
}

/// Key-aware chord spelling of the head code revision: `K:Eb` with `Eb`/`Ab`/`Bb` symbols.
#[test]
fn flat_key_uses_head_revision_chord_spelling() {
    check(Case {
        name: "synth_eb_head",
        duration: 39.4,
        melody_only: false,
        tokens: include_str!(
            "../../../../../scripts/reference/sheetsage2/artifacts/synth_eb_head/tokens.txt"
        ),
        files: vec![
            artifact!("synth_eb_head", "score.abc"),
            artifact!("synth_eb_head", "chord.lab"),
            artifact!("synth_eb_head", "key.lab"),
        ],
    });
    // The release's sharp spelling is deliberately NOT what this port produces.
    let release = include_str!(
        "../../../../../scripts/reference/sheetsage2/artifacts/synth_eb_release/score.abc"
    );
    let ours = replay(&Case {
        name: "synth_eb_release",
        duration: 39.4,
        melody_only: false,
        tokens: include_str!(
            "../../../../../scripts/reference/sheetsage2/artifacts/synth_eb_release/tokens.txt"
        ),
        files: vec![],
    });
    assert_ne!(ours.abc.as_deref(), Some(release));
}

/// The >300 s multi-window path: two windows, the second conditioned on a 1,124-token overlap
/// prefix rebuilt from the events accepted from the first. `replay` refuses unless the natively
/// rebuilt prefix equals the one upstream generated from, token for token.
#[test]
fn multi_window_song_stitches_like_upstream() {
    check(Case {
        name: "long_multiwindow",
        duration: 358.2,
        melody_only: false,
        tokens: include_str!(
            "../../../../../scripts/reference/sheetsage2/artifacts/long_multiwindow/tokens.txt"
        ),
        files: vec![
            artifact!("long_multiwindow", "score.abc"),
            artifact!("long_multiwindow", "events.tsv"),
            artifact!("long_multiwindow", "beat.lab"),
            artifact!("long_multiwindow", "chord.lab"),
            artifact!("long_multiwindow", "key.lab"),
            artifact!("long_multiwindow", "structure.lab"),
            artifact!("long_multiwindow", "melody_full.lab"),
            artifact!("long_multiwindow", "rhythm_events.lab"),
            artifact!("long_multiwindow", "notation/song_beats.txt"),
            artifact!("long_multiwindow", "melody.mid"),
            artifact!("long_multiwindow", "transcription.mid"),
            artifact!("long_multiwindow", "notation/song_melody.mid"),
        ],
    });
}

/// Upstream turns 12 s of digital silence into a rest-only score with no warning; the native
/// export reproduces that score (the refusal lives in the review layer, which is tested there).
#[test]
fn silence_reproduces_upstreams_rest_only_score() {
    check(Case {
        name: "silence",
        duration: 12.0,
        melody_only: false,
        tokens: include_str!(
            "../../../../../scripts/reference/sheetsage2/artifacts/silence/tokens.txt"
        ),
        files: vec![
            artifact!("silence", "score.abc"),
            artifact!("silence", "melody_full.lab"),
            artifact!("silence", "beat.lab"),
        ],
    });
}

/// The native chord pitch sets equal `mir_eval.chord.encode(reduce_extended_chords=True)` for
/// every label of the vocabulary (`testdata/chord_pitches.json`, from `native_parity.py tables`).
#[test]
fn chord_pitches_match_mir_eval_for_the_whole_vocabulary() {
    let table: serde_json::Value =
        serde_json::from_str(include_str!("../../testdata/chord_pitches.json")).unwrap();
    let pitches = table["pitches"].as_object().unwrap();
    assert_eq!(pitches.len(), 360);
    for (label, expected) in pitches {
        let expected: Vec<u8> = expected
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(chord_pitches(label).unwrap(), expected, "{label}");
    }
    // Flat spellings (what key-aware spelling produces) resolve to the same pitch classes.
    assert_eq!(
        chord_pitches("Eb:maj").unwrap(),
        chord_pitches("D#:maj").unwrap()
    );
}
