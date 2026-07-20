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
//! ## Out-of-vocabulary tier (sc-13038)
//!
//! misaki's embedded lexicon covers common English, but with espeak off it degrades any word it
//! does not know — names, jargon, proper nouns — to **letter-by-letter spell-out** (each
//! character phonemized and space-joined), which sounds like reciting the spelling. This wrapper
//! inserts a middle tier before spell-out:
//!
//! ```text
//!   misaki lexicon  →  CMU Pronouncing Dictionary  →  letter spell-out
//! ```
//!
//! When misaki spelled a word out (a single all-alphabetic token whose phonemes still carry an
//! internal space), the word is looked up in the embedded [CMU dict](cmudict_fast) (135k words,
//! BSD-2-Clause; see `resources/cmudict.dict`). A hit's ARPAbet pronunciation is mapped to the
//! same raw misaki IPA the lexicon emits (`arpabet_to_ipa`) and then flows through the same
//! `post_process` as every other token, so a cmudict word and a lexicon word land in exactly
//! the same Kokoro alphabet. A miss (word in neither dictionary) keeps misaki's spell-out — the
//! graceful final fallback. cmudict is US English; GB voices reuse it (a real US-derived
//! pronunciation still beats spelling the word out) and its output is finalized by GB
//! `post_process` (r-colored vowels de-rhoticize, flap/glottal → `t`). Multilingual G2P is a
//! separate concern (sc-12848); this tier is English-only.
//!
//! The dict parses once, lazily, into a process-global (`cmudict`); if the embedded data ever
//! fails to parse the tier is simply inert and every word falls through to spell-out (no panic).
//! Anything still outside the model vocab is silently dropped at id mapping
//! ([`crate::config::KokoroConfig::phonemes_to_ids`]), the reference `KModel.forward` filter.

use std::str::FromStr;
use std::sync::OnceLock;

use candle_audio::{AudioError, Result};
use cmudict_fast::{Cmudict, Rule, Stress, Symbol};
use misaki_rs::{Language, MToken, G2P};

/// Punctuation characters Kokoro's vocab carries (prosody-bearing pass-through tokens).
const KOKORO_PUNCT: &str = ";:,.!?—…\"()“”";

/// The CMU Pronouncing Dictionary, vendored (BSD-2-Clause) and embedded so the OOV tier needs
/// no snapshot file, no network, and no Python — parsed once into the process-global [`cmudict`].
const CMUDICT_DATA: &str = include_str!("../resources/cmudict.dict");

/// The lazily-parsed process-global CMU dictionary shared by every [`KokoroG2p`]. `None` when the
/// embedded data fails to parse (the OOV tier then goes inert — words fall through to spell-out).
fn cmudict() -> Option<&'static Cmudict> {
    static DICT: OnceLock<Option<Cmudict>> = OnceLock::new();
    DICT.get_or_init(|| Cmudict::from_str(CMUDICT_DATA).ok())
        .as_ref()
}

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
    /// The CMU-dict OOV tier (`None` disables it — misaki lexicon → spell-out only).
    cmudict: Option<&'static Cmudict>,
}

impl KokoroG2p {
    /// Build the engine (parses the embedded misaki lexicons — construct once and reuse). The
    /// CMU-dict OOV tier is wired in from the process-global `cmudict`.
    pub fn new(variant: EnglishVariant) -> Self {
        Self::build(variant, cmudict())
    }

    /// Shared constructor: `cmudict` selects whether the OOV tier is active. `new` passes the
    /// global dict; tests pass `None` to isolate misaki's spell-out baseline (the discrimination
    /// seam that proves the cmudict tier is load-bearing).
    fn build(variant: EnglishVariant, cmudict: Option<&'static Cmudict>) -> Self {
        let lang = match variant {
            EnglishVariant::American => Language::EnglishUS,
            EnglishVariant::British => Language::EnglishGB,
        };
        Self {
            variant,
            engine: G2P::new(lang),
            cmudict,
        }
    }

    /// Phonemize `text` into Kokoro's phoneme alphabet.
    pub fn phonemize(&self, text: &str) -> Result<String> {
        let (_, tokens) = self
            .engine
            .g2p(text)
            .map_err(|e| AudioError::Msg(format!("kokoro g2p: {e}")))?;
        // Re-assemble from tokens (reference KPipeline join), restoring punctuation misaki-rs
        // flattened to spaces and upgrading any letter-spelled OOV word to its cmudict form.
        let mut joined = String::new();
        for tk in &tokens {
            let ps = if let Some(oov) = self.cmudict_override(tk) {
                oov
            } else {
                match tk.phonemes.as_deref() {
                    Some(" ") | None
                        if tk.text.chars().count() == 1
                            && tk.text.chars().all(|c| KOKORO_PUNCT.contains(c)) =>
                    {
                        tk.text.clone()
                    }
                    Some(ps) => ps.to_string(),
                    None => String::new(),
                }
            };
            joined.push_str(&ps);
            joined.push_str(&tk.whitespace);
        }
        Ok(post_process(joined.trim(), self.variant))
    }

    /// The CMU-dict OOV tier for one token: `Some(raw misaki IPA)` when `tk` is a word misaki
    /// spelled out letter-by-letter AND the CMU dict knows it, else `None` (keep misaki's own
    /// output). The returned IPA is intentionally un-`post_process`ed — it rejoins the token
    /// stream and is finalized alongside every other token by [`post_process`].
    fn cmudict_override(&self, tk: &MToken) -> Option<String> {
        let dict = self.cmudict?;
        let word = tk.text.as_str();
        // Only a single all-ASCII-alphabetic word (>1 char): this excludes numbers, hyphen /
        // underscore subtokens, contractions, and punctuation — the cases misaki already
        // resolves through its number/hyphen/rule paths, none of which we want to override.
        if word.chars().count() <= 1 || !word.chars().all(|c| c.is_ascii_alphabetic()) {
            return None;
        }
        // Only intercept misaki's letter spell-out. A lexicon or rule hit is one contiguous
        // phoneme run; the spell-out path is the only one that space-joins per-character
        // phonemes inside a single word — so an internal space is its signature. Leaving
        // contiguous output untouched is what makes misaki tier 1 (it always wins when it knows
        // the word).
        let spelled_out = tk
            .phonemes
            .as_deref()
            .is_some_and(|p| p.trim().contains(' '));
        if !spelled_out {
            return None;
        }
        let rule = dict.get(&word.to_ascii_lowercase())?.first()?;
        let ipa = arpabet_to_ipa(rule);
        (!ipa.is_empty()).then_some(ipa)
    }
}

/// Map a CMU-dict ARPAbet pronunciation onto the raw misaki IPA the embedded lexicon emits, so
/// the result flows through [`post_process`] identically to a lexicon word. Stress marks precede
/// the stressed vowel (`ˈ` primary, `ˌ` secondary), matching misaki's gold layout (`K W IH1 K →
/// kwˈɪk`); diphthongs are emitted in the bare two-char forms `post_process` collapses (`EY1 →
/// eɪ → A`), and r-colored `ER` maps to `ɚ`/`ɝ` for `post_process` to expand (US) or
/// de-rhoticize (GB). AH is `ə` unstressed / `ʌ` stressed, the misaki schwa split.
fn arpabet_to_ipa(rule: &Rule) -> String {
    let mut out = String::new();
    for sym in rule.pronunciation() {
        push_symbol(sym, &mut out);
    }
    out
}

/// Append one ARPAbet [`Symbol`] as raw misaki IPA (with a leading stress mark for stressed
/// vowels). Factored out so the mapping is one exhaustive `match` the compiler keeps total.
fn push_symbol(sym: &Symbol, out: &mut String) {
    use Symbol::*;
    // (IPA, stress-of-this-vowel). Consonants carry no stress.
    let (ipa, stress): (&str, Option<&Stress>) = match sym {
        AA(s) => ("ɑ", Some(s)),
        AE(s) => ("æ", Some(s)),
        AH(s) => (
            match s {
                Stress::None => "ə",
                _ => "ʌ",
            },
            Some(s),
        ),
        AO(s) => ("ɔ", Some(s)),
        AW(s) => ("aʊ", Some(s)),
        AY(s) => ("aɪ", Some(s)),
        EH(s) => ("ɛ", Some(s)),
        ER(s) => (
            match s {
                Stress::None => "ɚ",
                _ => "ɝ",
            },
            Some(s),
        ),
        EY(s) => ("eɪ", Some(s)),
        IH(s) => ("ɪ", Some(s)),
        IY(s) => ("i", Some(s)),
        OW(s) => ("oʊ", Some(s)),
        OY(s) => ("ɔɪ", Some(s)),
        UH(s) => ("ʊ", Some(s)),
        UW(s) => ("u", Some(s)),
        B => ("b", None),
        CH => ("ʧ", None),
        D => ("d", None),
        DH => ("ð", None),
        F => ("f", None),
        G => ("ɡ", None),
        HH => ("h", None),
        JH => ("ʤ", None),
        K => ("k", None),
        L => ("l", None),
        M => ("m", None),
        N => ("n", None),
        NG => ("ŋ", None),
        P => ("p", None),
        R => ("ɹ", None),
        S => ("s", None),
        SH => ("ʃ", None),
        T => ("t", None),
        TH => ("θ", None),
        V => ("v", None),
        W => ("w", None),
        Y => ("j", None),
        Z => ("z", None),
        ZH => ("ʒ", None),
    };
    match stress {
        Some(Stress::Primary) => out.push('ˈ'),
        Some(Stress::Secondary) => out.push('ˌ'),
        _ => {}
    }
    out.push_str(ipa);
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

    // ------------------------------------------------------------------------------------------
    // sc-13038 — the CMU-dict OOV tier (misaki lexicon → cmudict → letter spell-out).
    // ------------------------------------------------------------------------------------------

    /// A US-variant engine with the cmudict tier disabled — the spell-out baseline every
    /// discrimination test compares against. If the cmudict tier were removed from `phonemize`,
    /// `KokoroG2p::new(..)` would collapse onto exactly this behavior and the `assert_ne!`s below
    /// would fail — that is what makes these tests discriminate the tier rather than a constant.
    fn spell_out_only(variant: EnglishVariant) -> KokoroG2p {
        KokoroG2p::build(variant, None)
    }

    /// The embedded dict must actually parse — otherwise the tier is silently inert and every
    /// "improvement" test below would still pass on spell-out. This is the guard against a
    /// false-green whole feature.
    #[test]
    fn embedded_cmudict_parses_and_is_populated() {
        let dict = cmudict().expect("embedded resources/cmudict.dict must parse");
        // A staple word and a proper noun that misaki does not carry.
        assert!(dict.get("rust").is_some());
        assert!(dict.get("einstein").is_some());
        // A brand/jargon word genuinely outside cmudict (final-fallback fixture below).
        assert!(dict.get("anthropic").is_none());
    }

    /// The ARPAbet→IPA converter reproduces misaki's own gold forms for words BOTH dictionaries
    /// know — proving the cmudict tier lands in the exact same Kokoro alphabet as the lexicon
    /// tier (so mixing them in one utterance is seamless). These are the reference gold forms the
    /// engine snapshot above asserts misaki produces.
    #[test]
    fn arpabet_conversion_lands_in_the_misaki_gold_alphabet() {
        let dict = cmudict().unwrap();
        for (word, gold) in [
            ("fox", "fˈɑks"),   // F AA1 K S
            ("quick", "kwˈɪk"), // K W IH1 K
            ("lazy", "lˈAzi"),  // L EY1 Z IY0
            ("brown", "bɹˈWn"), // B R AW1 N
        ] {
            let rule = dict.get(word).unwrap().first().unwrap();
            let got = post_process(&arpabet_to_ipa(rule), EnglishVariant::American);
            assert_eq!(
                got, gold,
                "cmudict {word:?} → {got:?}, expected gold {gold:?}"
            );
        }
    }

    /// OOV proper nouns / jargon that misaki spells out get real cmudict pronunciations. Each
    /// assertion pins the SPECIFIC improved phonemes (not merely "non-empty") AND proves it
    /// differs from the spell-out baseline, so removing the cmudict tier fails the test. Every
    /// word here was confirmed to spell out under misaki alone (verified: the baseline is a
    /// spaced letter recitation).
    #[test]
    fn oov_names_and_jargon_get_real_cmudict_pronunciations() {
        let full = KokoroG2p::new(EnglishVariant::American);
        let base = spell_out_only(EnglishVariant::American);
        // (input, exact cmudict-derived US phonemes). e.g. Zuckerberg Z AH1 K ER0 B ER2 G →
        // z ˈʌ k ɚ b ˌɝ ɡ → (US post_process ɚ→əɹ, ɝ→ɜɹ) → zˈʌkəɹbˌɜɹɡ. Names, a brand, and
        // technical/math terms — the exact OOV classes the story calls out.
        for (word, expected) in [
            ("Zuckerberg", "zˈʌkəɹbˌɜɹɡ"), // surname
            ("Pixar", "pˈɪksɑɹ"),          // brand
            ("Fibonacci", "fˌɪbənˈɑʧi"),   // math term
            ("Chebyshev", "ʧˌɛbɪʃˈɛv"),    // math name
            ("Riemann", "ɹˈimən"),         // math name
            ("Hilbert", "hˈɪlbəɹt"),       // math name
        ] {
            let got = full.phonemize(word).unwrap();
            assert_eq!(got, expected, "cmudict tier: {word:?} → {got:?}");
            // The spell-out baseline is a different (letter-recited) string — the tier changed it.
            let spelled = base.phonemize(word).unwrap();
            assert_ne!(
                got, spelled,
                "{word:?}: cmudict output must differ from spell-out {spelled:?}"
            );
            // Real pronunciations are one contiguous run; spell-out recites letters with spaces.
            assert!(
                !got.contains(' '),
                "{word:?}: cmudict output {got:?} should not be spelled out"
            );
            assert!(
                spelled.contains(' '),
                "{word:?}: baseline {spelled:?} should be a spaced letter spell-out"
            );
        }
    }

    /// The tier is genuinely a tier: a word misaki DOES know is left exactly as misaki produced
    /// it (cmudict never overrides tier 1), even though cmudict also carries that word.
    #[test]
    fn known_lexicon_words_are_not_overridden_by_cmudict() {
        let full = KokoroG2p::new(EnglishVariant::American);
        let base = spell_out_only(EnglishVariant::American);
        // "quick"/"brown"/"lazy" are in misaki's lexicon — enabling cmudict must not change them.
        for word in ["quick", "brown", "lazy", "world"] {
            assert_eq!(
                full.phonemize(word).unwrap(),
                base.phonemize(word).unwrap(),
                "{word:?}: a known lexicon word must be identical with/without the cmudict tier"
            );
        }
    }

    /// Words OOV in BOTH dictionaries prove spell-out remains the graceful FINAL fallback: the
    /// cmudict tier is inert for them, so the full engine and the spell-out baseline agree, the
    /// output is a genuine (spaced) letter spell-out, and nothing panics or returns empty. These
    /// are words misaki spells out that cmudict also lacks (a real, coined brand, and nonsense).
    #[test]
    fn words_oov_in_both_dictionaries_still_spell_out() {
        let full = KokoroG2p::new(EnglishVariant::American);
        let base = spell_out_only(EnglishVariant::American);
        for word in ["Kubernetes", "numpy", "Xylqarbz"] {
            // Genuinely absent from cmudict — otherwise this would not exercise the final tier.
            assert!(
                cmudict().unwrap().get(&word.to_ascii_lowercase()).is_none(),
                "{word:?} must be OOV in cmudict for this fixture to test spell-out"
            );
            let got = full.phonemize(word).unwrap();
            assert!(!got.is_empty(), "{word:?}: spell-out must be non-empty");
            // The final fallback is a letter recitation — spaced, not one contiguous word.
            assert!(
                got.contains(' '),
                "{word:?}: OOV-in-both output {got:?} should be a spaced letter spell-out"
            );
            assert_eq!(
                got,
                base.phonemize(word).unwrap(),
                "{word:?}: cmudict miss must leave misaki's spell-out unchanged"
            );
        }
    }

    /// British voices reuse the (US) cmudict tier: the OOV word still gets a real pronunciation
    /// (not spell-out), finalized by GB `post_process` — so a word with r-colored vowels comes
    /// out non-rhotic, unlike the US form. Zuckerberg: US `zˈʌkəɹbˌɜɹɡ` (rhotic) vs GB
    /// `zˈʌkəbˌɜːɡ` (the `ɚ`/`ɝ` de-rhoticize to `ə`/`ɜː`), proving the tier flows through the
    /// same variant-specific `post_process` as every lexicon word.
    #[test]
    fn gb_variant_also_upgrades_oov_words_via_cmudict() {
        let gb = KokoroG2p::new(EnglishVariant::British);
        let gb_base = spell_out_only(EnglishVariant::British);
        let got = gb.phonemize("Zuckerberg").unwrap();
        assert_eq!(got, "zˈʌkəbˌɜːɡ", "GB cmudict tier: Zuckerberg → {got:?}");
        // Different from the GB spell-out baseline (so the tier fired) and non-rhotic, unlike US.
        assert_ne!(got, gb_base.phonemize("Zuckerberg").unwrap());
        assert!(
            !got.contains(' '),
            "GB cmudict output {got:?} must not be spelled out"
        );
        assert_ne!(
            got,
            KokoroG2p::new(EnglishVariant::American)
                .phonemize("Zuckerberg")
                .unwrap(),
            "GB output should de-rhoticize and differ from the US rhotic form"
        );
    }
}
