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
//! Punctuation: misaki-rs maps punctuation tokens to a bare space; Kokoro's vocab carries
//! punctuation as prosody-bearing tokens (a `.` is a pause), so this wrapper re-assembles the
//! phoneme string from the token stream and restores single-char punctuation the model knows.
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
        let (_, tokens) = self
            .engine
            .g2p(text)
            .map_err(|e| AudioError::Msg(format!("kokoro g2p: {e}")))?;
        // Re-assemble from tokens (reference KPipeline join), restoring punctuation misaki-rs
        // flattened to spaces.
        let mut joined = String::new();
        for tk in &tokens {
            let ps = match tk.phonemes.as_deref() {
                Some(" ") | None
                    if tk.text.chars().count() == 1
                        && tk.text.chars().all(|c| KOKORO_PUNCT.contains(c)) =>
                {
                    tk.text.clone()
                }
                Some(ps) => ps.to_string(),
                None => String::new(),
            };
            joined.push_str(&ps);
            joined.push_str(&tk.whitespace);
        }
        Ok(post_process(joined.trim(), self.variant))
    }
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
}
