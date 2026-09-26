use super::abc::{compare, parse, strip_chords, Frac, KeepVoice};
use super::lyrics::{estimate_syllables, normalize_label, sections};
use super::*;

/// A SheetSage2 transcription of the public-domain Navy Band recording (committed sc-23003
/// oracles): the full score with chord symbols, and upstream's melody-only rendering of the same
/// tokens.
const FULL: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/real_full/score.abc");
const MELODY: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/real_melody/score.abc");
const SILENCE: &str =
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/silence/score.abc");

const LYRICS: &str = "[Verse]\nO say can you see by the dawn's early light\nWhat so proudly we hailed at \
                      the twilight's last gleaming\nWhose broad stripes and bright stars through the \
                      perilous fight\nO'er the ramparts we watched were so gallantly streaming\n";
const STYLE: &str = "English, warm female vocal, gentle acoustic folk, fingerpicked guitar, 80 BPM";

fn small(body: &str) -> String {
    format!(
        "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=88\nV: Vocal clef=treble name=\"Vocal Melody\" \
         snm=\"Vocal\"\nV: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:C\n{body}"
    )
}

#[test]
fn the_dialect_parser_resolves_ties_and_bar_scoped_accidentals() {
    // The skill's own example: in C major, `^F32-|F8F24|` is a five-quarter F-sharp followed by a
    // three-quarter F-natural.
    let text = small("% verse\nV: Vocal\n\"C\"^F32-|F8F24|\nV: Ins\nZ2|\n");
    let score = parse(&text).unwrap();
    let notes = &score.voice("Vocal").notes;
    assert_eq!(notes.len(), 2);
    assert_eq!((notes[0].pitch, notes[0].span), (66, Frac::new(5, 1)));
    assert_eq!((notes[1].pitch, notes[1].onset), (65, Frac::new(5, 1)));
    assert_eq!(score.sections, vec![(Frac::ZERO, "verse".to_string())]);
    // Unsupported material is refused, never guessed.
    assert!(parse(&small("V: Vocal\n(3CDE24|\nV: Ins\nZ|\n")).is_err());
    assert!(
        parse(&small("V: Vocal\nC32|\nV: Ins\nZ2|\n")).is_err(),
        "measure counts differ"
    );
    assert!(
        parse(&small("V: Vocal\nC31|\nV: Ins\nZ|\n")).is_err(),
        "bar too short"
    );
    assert!(
        parse(&small("V: Vocal\nC32-|\nV: Ins\nZ|\n")).is_err(),
        "unresolved tie"
    );
    assert!(
        parse(&small("V: Vocal\nC32|\nV: Ins\n\"C\"C32|\n")).is_err(),
        "chords in Ins"
    );
}

/// A real SheetSage2 score parses in the YuE2 dialect, and stripping its chords reproduces the
/// sounding content of upstream's own melody-only rendering exactly.
///
/// Mutation that must fail: drop the tie merge in `parse_bar` (note counts change), or keep
/// chord symbols in `strip_chords`.
#[test]
fn stripping_a_transcribed_score_matches_the_melody_only_rendering() {
    let full = parse(FULL).unwrap();
    assert_eq!(full.voices[0].bars.len(), 27);
    // Ties rejoin every note the notation split across bars or into representable lengths: the
    // 76 transcribed vocal notes come back as 76 sounding notes.
    assert_eq!(full.voice("Vocal").notes.len(), 76);
    // `"C"g4e4c4-|"C"c2…` (bar 17): the tied C5 sounds for a quarter plus an eighth.
    assert!(full
        .voice("Vocal")
        .notes
        .iter()
        .any(|n| n.onset == Frac::new(50, 1) && n.pitch == 72 && n.span == Frac::new(3, 2)));
    assert!(full.chord_count() > 0);
    assert!(full.voice("Ins").notes.is_empty());
    let (stripped, removed) = strip_chords(FULL, KeepVoice::Both).unwrap();
    assert_eq!(removed, full.chord_count());
    let stripped = parse(&stripped).unwrap();
    assert_eq!(stripped.chord_count(), 0);
    let melody = parse(MELODY).unwrap();
    assert!(compare(&stripped, &melody, &super::abc::VOICES, false).is_empty());
    assert!(compare(&full, &melody, &super::abc::VOICES, false).is_empty());
}

#[test]
fn melody_cover_request_plans_from_the_chord_free_score() {
    let prepared = prepare_cover(&CoverSpec::new(
        CoverMode::Melody,
        FULL,
        STYLE,
        CoverLyrics::source(LYRICS),
    ))
    .unwrap();
    assert_eq!(prepared.request.cot(), CotMode::Melody);
    let abc = prepared.request.abc().unwrap();
    assert!(
        !abc.lines().skip(8).any(|l| l.contains('"')),
        "no chord symbols"
    );
    assert_eq!(prepared.score.chord_count(), 0);
    assert!(prepared.report.chords_removed > 0);
    assert_eq!(prepared.report.score_sections, vec!["verse"]);
    assert_eq!(prepared.report.lyric_sections, vec!["verse"]);
    assert!(
        !prepared
            .report
            .warnings
            .iter()
            .any(|w| w.code == "sections_misaligned"),
        "{:?}",
        prepared.report.warnings
    );
    // Symbolic conditioning only: the request carries exactly the protocol's fields, none of
    // which is audio (no reference audio / ICL prompt exists in the YuE2 request).
    let json = prepared.request.to_json();
    let keys: Vec<&str> = json
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let mut expected = SongRequest::FIELDS.to_vec();
    expected.sort_unstable();
    let mut keys_sorted = keys.clone();
    keys_sorted.sort_unstable();
    assert_eq!(keys_sorted, expected);
}

#[test]
fn full_cover_supplies_the_harmony_and_needs_it() {
    let prepared = prepare_cover(&CoverSpec::new(
        CoverMode::Full,
        FULL,
        STYLE,
        CoverLyrics::source(LYRICS),
    ))
    .unwrap();
    assert_eq!(prepared.request.cot(), CotMode::Full);
    assert_eq!(prepared.request.abc(), Some(FULL));
    let err = prepare_cover(&CoverSpec::new(
        CoverMode::Full,
        MELODY,
        STYLE,
        CoverLyrics::source(LYRICS),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("no chord symbols"), "{err}");
}

/// A rest-only score (what upstream makes of silence) is refused: there is no melody to cover.
///
/// Mutation that must fail: remove the `notes == [0, 0]` refusal.
#[test]
fn a_score_without_notes_is_refused() {
    let err = prepare_cover(&CoverSpec::new(
        CoverMode::Melody,
        SILENCE,
        STYLE,
        CoverLyrics::source(LYRICS),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("no sounding notes"), "{err}");
}

#[test]
fn translations_must_keep_the_source_sections() {
    let source =
        "[Verse]\nO say can you see\nBy the dawn's early light\n\n[Chorus]\nO say does that \
                  star\n";
    let aligned = "[Verse]\nOh dis peux-tu voir\nÀ la lueur de l'aube\n\n[Chorus]\nDis-moi si \
                   l'étoile\n";
    let prepared = prepare_cover(&CoverSpec::new(
        CoverMode::Melody,
        FULL,
        STYLE,
        CoverLyrics::translation(aligned, source),
    ))
    .unwrap();
    assert_eq!(prepared.request.lyrics(), aligned);
    // The score has one vocal section; the lyrics two: surfaced for review, not refused.
    assert!(prepared
        .report
        .warnings
        .iter()
        .any(|w| w.code == "sections_misaligned"));
    let reordered = "[Chorus]\nDis-moi si l'étoile\n\n[Verse]\nOh dis peux-tu voir\n";
    let err = prepare_cover(&CoverSpec::new(
        CoverMode::Melody,
        FULL,
        STYLE,
        CoverLyrics::translation(reordered, source),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("not aligned"), "{err}");
    assert!(prepare_cover(&CoverSpec::new(
        CoverMode::Melody,
        FULL,
        STYLE,
        CoverLyrics::source("   \n"),
    ))
    .is_err());
}

#[test]
fn keeping_one_voice_silences_the_other_on_the_same_grid() {
    let text = small("% verse\nV: Vocal\n\"C\"C32|\nV: Ins\nE16G16|\n");
    let (vocal_only, _) = strip_chords(&text, KeepVoice::Vocal).unwrap();
    let parsed = parse(&vocal_only).unwrap();
    assert!(parsed.voice("Ins").notes.is_empty());
    assert_eq!(parsed.voice("Vocal").notes.len(), 1);
    let prepared = prepare_cover(&CoverSpec {
        keep: KeepVoice::Ins,
        ..CoverSpec::new(
            CoverMode::Melody,
            text,
            STYLE,
            CoverLyrics::source("[Verse]\nla la\n"),
        )
    })
    .unwrap();
    assert_eq!(prepared.report.notes, [0, 2]);
}

#[test]
fn lyric_helpers() {
    assert_eq!(normalize_label("Verse 2"), "verse");
    assert_eq!(normalize_label(" Pre-Chorus "), "pre-chorus");
    assert_eq!(normalize_label("chorus2"), "chorus");
    let s = sections("intro words\n[Verse 1]\na\n\nb\n[Chorus]\nc\n");
    assert_eq!(s.len(), 3);
    assert_eq!(s[1].label, "verse");
    assert_eq!(s[1].lines, vec!["a", "b"]);
    assert_eq!(estimate_syllables("O say can you see"), 5);
    assert_eq!(estimate_syllables("你好世界"), 4);
    assert_eq!(estimate_syllables("the twilight's last gleaming"), 6);
}
