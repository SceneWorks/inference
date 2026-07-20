//! Text → Kokoro phonemes, pure Rust (sc-12836): the `misaki-rs` G2P engine (the embedded
//! misaki gold/silver lexicons + POS tagger — the same phonemization lineage Kokoro-82M was
//! trained with; MIT, **no espeak, no Python**) plus a post-processing pass that maps its raw
//! IPA output onto the exact US/GB phoneme alphabets misaki's reference `en.py` feeds Kokoro:
//!
//! - tied diphthongs collapse to the trained single-char forms (`e‍ɪ → A`, `a‍ɪ → I`,
//!   `a‍ʊ → W`, `ɔ‍ɪ → Y`, `o‍ʊ → O`, GB `ə‍ʊ → Q`), and any leftover U+200D tie is dropped;
//! - US: length marks (`ː`) are removed and the r-colored `ɚ`/`ɝ` expand to `əɹ`/`ɜɹ`
//!   (`US_VOCAB` carries neither), matching the reference gold entries
//!   (`fˈɑːks → fˈɑks`, `ˌo‍ʊvɚɹ → ˌOvəɹ`);
//! - GB: length marks are kept (`GB_VOCAB` has `ː`); flap/glottal (`ɾ`, `ʔ`) become `t`.
//!
//! Punctuation: misaki-rs blanks punctuation tokens (it emits `Some(" ")` for `.,;:` but
//! `Some("")` for `—`); Kokoro's vocab carries punctuation as prosody-bearing tokens (a `.`
//! is a pause, `—` id 9 a longer one), so this wrapper re-assembles the phoneme string from
//! the token stream and restores any single-char punctuation whose phoneme came back blank —
//! matching *both* blank forms, or the em-dash pause is silently dropped.
//!
//! Dashes: an intra-word compound hyphen (`text-to-speech`, `well-known`) should read as one
//! seamless phrase, while a standalone/interruption dash (`wait -- what`) is a pause. ASCII
//! `-` and en-dash `–` are absent from Kokoro's vocab, so a raw hyphen only ever degrades to
//! a token boundary; [`normalize_dashes`] promotes the *punctuation* dashes to `—` (a real
//! pause) and leaves compound hyphens to flow. Finally runs of whitespace collapse to a
//! single space: the vocab's space (id 16) is itself a pause token, so a hyphen that degrades
//! to a bare space would otherwise stack dead air (`text-to-speech` → three spaces around
//! `to`).
//!
//! OOV words cannot panic: misaki-rs falls back to letter-by-letter spelling, and anything
//! still outside the model vocab is silently dropped at id mapping
//! ([`crate::config::KokoroConfig::phonemes_to_ids`]), the reference `KModel.forward` filter.

use candle_audio::{AudioError, Result};
use misaki_rs::{Language, G2P};

/// Punctuation characters Kokoro's vocab carries (prosody-bearing pass-through tokens).
const KOKORO_PUNCT: &str = ";:,.!?—…\"()“”";

/// Which English variant a voice speaks (Kokoro voice-name prefix: `a…` American, `b…`
/// British).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnglishVariant {
    American,
    British,
}

/// The lazily-built G2P engine for one English variant.
pub struct KokoroG2p {
    variant: EnglishVariant,
    engine: G2P,
}

impl KokoroG2p {
    /// Build the engine (parses the embedded misaki lexicons — construct once and reuse).
    pub fn new(variant: EnglishVariant) -> Self {
        let lang = match variant {
            EnglishVariant::American => Language::EnglishUS,
            EnglishVariant::British => Language::EnglishGB,
        };
        Self {
            variant,
            engine: G2P::new(lang),
        }
    }

    /// Phonemize `text` into Kokoro's phoneme alphabet.
    pub fn phonemize(&self, text: &str) -> Result<String> {
        let normalized = normalize_dashes(text);
        let (_, tokens) = self
            .engine
            .g2p(&normalized)
            .map_err(|e| AudioError::Msg(format!("kokoro g2p: {e}")))?;
        // Re-assemble from tokens (reference KPipeline join), restoring the punctuation
        // misaki-rs blanks out. It returns `Some(" ")` for `.,;:` but `Some("")` for `—`, so
        // treat *any* blank phoneme on a single punctuation char as "restore it" — matching
        // only `Some(" ")` would silently drop the em-dash pause token.
        let mut joined = String::new();
        for tk in &tokens {
            let is_blank_punct = tk.text.chars().count() == 1
                && tk.text.chars().all(|c| KOKORO_PUNCT.contains(c))
                && tk.phonemes.as_deref().is_none_or(|p| p.trim().is_empty());
            let ps = if is_blank_punct {
                tk.text.clone()
            } else {
                tk.phonemes.clone().unwrap_or_default()
            };
            joined.push_str(&ps);
            joined.push_str(&tk.whitespace);
        }
        // Collapse whitespace runs so token boundaries are single spaces (see module docs:
        // each extra space is a pause token — dead air).
        Ok(post_process(
            &collapse_whitespace(joined.trim()),
            self.variant,
        ))
    }
}

/// Promote dashes used as *punctuation* to the em-dash Kokoro carries as a pause token
/// (id 9), while leaving intra-word compound hyphens (`text-to-speech`, `well-known`) alone
/// so they read as one seamless phrase. ASCII `-` and en-dash `–` are outside the vocab, so a
/// dash only becomes an audible pause once mapped to `—`; an untouched compound hyphen
/// instead degrades to a plain token boundary and flows.
fn normalize_dashes(text: &str) -> String {
    let mut s = text.replace('\u{2013}', "\u{2014}"); // en-dash → em-dash
    s = s.replace("--", "\u{2014}"); // ASCII double-hyphen → em-dash
    s = s.replace(" - ", " \u{2014} "); // spaced interruption dash → em-dash
    s
}

/// Collapse runs of whitespace to a single space (reference single-space token join).
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out
}

/// Map raw misaki-rs IPA onto the US/GB alphabets Kokoro was trained with (module docs).
fn post_process(phonemes: &str, variant: EnglishVariant) -> String {
    const ZWJ: char = '\u{200d}';
    let mut s = phonemes.to_string();
    // Tied diphthongs → trained single-char forms (both tied and bare orders).
    let tied: &[(&str, &str)] = &[
        ("e\u{200d}ɪ", "A"),
        ("a\u{200d}ɪ", "I"),
        ("a\u{200d}ʊ", "W"),
        ("ɔ\u{200d}ɪ", "Y"),
        ("o\u{200d}ʊ", "O"),
        (
            "ə\u{200d}ʊ",
            match variant {
                EnglishVariant::American => "O",
                EnglishVariant::British => "Q",
            },
        ),
    ];
    for (from, to) in tied {
        s = s.replace(from, to);
    }
    for (from, to) in [
        ("eɪ", "A"),
        ("aɪ", "I"),
        ("aʊ", "W"),
        ("ɔɪ", "Y"),
        ("oʊ", "O"),
    ] {
        s = s.replace(from, to);
    }
    s = s.replace(ZWJ, "");
    match variant {
        EnglishVariant::American => {
            s = s.replace('ː', "");
            s = s.replace("ɚɹ", "əɹ").replace('ɚ', "əɹ");
            s = s.replace("ɝɹ", "ɜɹ").replace('ɝ', "ɜɹ");
        }
        EnglishVariant::British => {
            s = s.replace("ɚɹ", "ə").replace('ɚ', "ə").replace('ɝ', "ɜː");
            s = s.replace(['ɾ', 'ʔ'], "t");
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn us_post_process_matches_reference_gold_forms() {
        // Raw misaki-rs lexicon forms → the hexgrad us_gold forms Kokoro was trained on.
        assert_eq!(
            post_process("bɹˈa\u{200d}ʊn", EnglishVariant::American),
            "bɹˈWn"
        );
        assert_eq!(post_process("fˈɑːks", EnglishVariant::American), "fˈɑks");
        assert_eq!(
            post_process("lˈe\u{200d}ɪzi", EnglishVariant::American),
            "lˈAzi"
        );
        assert_eq!(
            post_process("ˌo\u{200d}ʊvɚɹ", EnglishVariant::American),
            "ˌOvəɹ"
        );
        assert_eq!(post_process("wˈɜːld", EnglishVariant::American), "wˈɜld");
    }

    #[test]
    fn gb_post_process_keeps_length_and_uses_q() {
        assert_eq!(post_process("gˈə\u{200d}ʊ", EnglishVariant::British), "gˈQ");
        assert_eq!(post_process("fˈɑːks", EnglishVariant::British), "fˈɑːks");
    }

    // Engine-backed snapshots (embedded lexicons; offline but slower — they parse the misaki
    // data once per engine).
    #[test]
    fn phonemizes_the_reference_sentence_into_vocab_phonemes() {
        let g2p = KokoroG2p::new(EnglishVariant::American);
        let ps = g2p
            .phonemize("The quick brown fox jumps over the lazy dog.")
            .unwrap();
        // Content words arrive in trained gold form.
        for expect in ["kwˈɪk", "bɹˈWn", "fˈɑks", "Ovəɹ", "lˈAzi", "dˈɑɡ"] {
            assert!(ps.contains(expect), "phonemes {ps:?} missing {expect:?}");
        }
        // Sentence-final punctuation is restored (prosody-bearing token).
        assert!(
            ps.ends_with('.'),
            "phonemes {ps:?} must keep the final period"
        );
        // No ties / length marks / unknown-word markers survive for US output.
        assert!(!ps.contains('\u{200d}') && !ps.contains('ː'), "{ps:?}");
    }

    #[test]
    fn oov_words_degrade_gracefully_without_panicking() {
        let g2p = KokoroG2p::new(EnglishVariant::American);
        let ps = g2p.phonemize("Xylqarbz!").unwrap();
        // Whatever the fallback produced, it is a non-panicking string; unknown chars are
        // dropped later at id mapping.
        assert!(!ps.is_empty());
    }

    #[test]
    fn normalize_dashes_promotes_only_punctuation_dashes() {
        // Intra-word compound hyphen is left for misaki to split into a seamless boundary.
        assert_eq!(normalize_dashes("text-to-speech"), "text-to-speech");
        assert_eq!(normalize_dashes("well-known"), "well-known");
        // Standalone / doubled / en-dash forms become the em-dash pause token.
        assert_eq!(normalize_dashes("wait -- what"), "wait \u{2014} what");
        assert_eq!(normalize_dashes("5 - 10"), "5 \u{2014} 10");
        assert_eq!(normalize_dashes("a\u{2013}b"), "a\u{2014}b");
    }

    #[test]
    fn collapse_whitespace_squashes_runs() {
        assert_eq!(collapse_whitespace("a   b  c"), "a b c");
        assert_eq!(collapse_whitespace("a\tb\nc"), "a b c");
        assert_eq!(collapse_whitespace("solo"), "solo");
    }

    #[test]
    fn compound_hyphen_reads_seamlessly_like_spaces() {
        // The reported artifact: `text-to-speech` used to stack three spaces (dead air) around
        // `to`. It must now phonemize identically to the plain-spaced phrase, with no pause
        // token and no doubled space.
        let g2p = KokoroG2p::new(EnglishVariant::American);
        let hyphen = g2p.phonemize("text-to-speech").unwrap();
        let spaced = g2p.phonemize("text to speech").unwrap();
        assert_eq!(hyphen, spaced, "hyphen and spaced forms must match");
        assert!(!hyphen.contains('\u{2014}'), "no em-dash pause: {hyphen:?}");
        assert!(!hyphen.contains("  "), "no doubled space: {hyphen:?}");
    }

    #[test]
    fn interruption_and_typed_em_dashes_become_pause_tokens() {
        let g2p = KokoroG2p::new(EnglishVariant::American);
        // A doubled ASCII dash used as an interruption becomes the `—` pause token.
        assert!(
            g2p.phonemize("wait -- what").unwrap().contains('\u{2014}'),
            "double-hyphen must map to em-dash pause"
        );
        // A literal em-dash was previously dropped by the `Some(\" \")`-only guard; it must
        // now survive as the pause token.
        assert!(
            g2p.phonemize("stop \u{2014} now")
                .unwrap()
                .contains('\u{2014}'),
            "typed em-dash must be restored, not dropped"
        );
    }
}
