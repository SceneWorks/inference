//! Chatterbox text front-end (sc-13222) — a faithful port of `punc_norm` (`tts.py`) and the
//! `EnTokenizer` (`models/tokenizers/tokenizer.py`), the exact text pipeline the base English
//! Chatterbox model was trained with.
//!
//! Flow: `punc_norm` cleanup → replace ASCII space with the `[SPACE]` token string → BPE-encode
//! with the pinned `tokenizer.json` → prepend `start_text_token` (255) and append
//! `stop_text_token` (0) around the row (the SOT/EOT the T3 conditioning expects).

use candle_audio::{AudioError, Result};
use tokenizers::Tokenizer;

/// The token-string the reference substitutes for an ASCII space before BPE encoding.
pub const SPACE_TOKEN: &str = "[SPACE]";

/// Reference `punc_norm`: capitalize the first letter, collapse runs of whitespace, rewrite a
/// fixed set of uncommon/LLM punctuation, and guarantee a sentence-ending character. Byte-for-byte
/// the same rewrite table and ordering as `tts.py::punc_norm`.
pub fn punc_norm(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }

    // Capitalize the first letter if it is lowercase (ASCII-first-letter, matching Python's
    // `text[0].islower()` for the common Latin case).
    let mut s: String = {
        let mut chars = text.chars();
        match chars.next() {
            Some(first) if first.is_lowercase() => {
                first.to_uppercase().collect::<String>() + chars.as_str()
            }
            _ => text.to_string(),
        }
    };

    // Collapse all whitespace runs to single spaces (Python `" ".join(text.split())`).
    s = s.split_whitespace().collect::<Vec<_>>().join(" ");

    // Uncommon/LLM punctuation rewrites, in the reference order.
    const REPLACEMENTS: &[(&str, &str)] = &[
        ("...", ", "),
        ("\u{2026}", ", "), // …
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("\u{2014}", "-"), // —
        ("\u{2013}", "-"), // –
        (" ,", ","),
        ("\u{201c}", "\""), // “
        ("\u{201d}", "\""), // ”
        ("\u{2018}", "'"),  // ‘
        ("\u{2019}", "'"),  // ’
    ];
    for (from, to) in REPLACEMENTS {
        s = s.replace(from, to);
    }

    // Trailing spaces stripped, then guarantee a sentence-ender.
    let trimmed = s.trim_end_matches(' ');
    let mut out = trimmed.to_string();
    let ends_ok = out
        .chars()
        .last()
        .is_some_and(|c| matches!(c, '.' | '!' | '?' | '-' | ','));
    if !ends_ok {
        out.push('.');
    }
    out
}

/// The Chatterbox English tokenizer — a thin, faithful wrapper over the pinned `tokenizer.json`
/// BPE model that reproduces `EnTokenizer.encode` (space → `[SPACE]`, then BPE).
pub struct EnTokenizer {
    tokenizer: Tokenizer,
    start_text_token: u32,
    stop_text_token: u32,
}

impl EnTokenizer {
    /// Load from a `tokenizer.json` file. `start_text_token` / `stop_text_token` are the T3
    /// config's SOT/EOT ids (255 / 0 for the base English model).
    pub fn from_file(
        path: &std::path::Path,
        start_text_token: u32,
        stop_text_token: u32,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| AudioError::Msg(format!("chatterbox: load {}: {e}", path.display())))?;
        let vocab = tokenizer.get_vocab(true);
        // The reference asserts SOT/EOT are present; mirror that (a mismatched tokenizer.json is a
        // hard error, not a silent degrade).
        if !vocab.contains_key("[START]") || !vocab.contains_key("[STOP]") {
            return Err(AudioError::Msg(
                "chatterbox: tokenizer.json missing [START]/[STOP] special tokens".into(),
            ));
        }
        Ok(Self {
            tokenizer,
            start_text_token,
            stop_text_token,
        })
    }

    /// Reference `EnTokenizer.encode`: replace ASCII spaces with `[SPACE]`, then BPE-encode
    /// (special tokens NOT added by the tokenizer itself — the SOT/EOT are added by
    /// [`Self::text_to_tokens`]).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let replaced = text.replace(' ', SPACE_TOKEN);
        let encoding = self
            .tokenizer
            .encode(replaced, false)
            .map_err(|e| AudioError::Msg(format!("chatterbox: tokenizer encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The full text-token row the T3 conditioning consumes: `punc_norm` → encode → SOT-prefixed,
    /// EOT-suffixed ids (the reference `text_to_tokens` plus the `generate()` SOT/EOT wrap).
    pub fn text_to_tokens(&self, text: &str) -> Result<Vec<u32>> {
        let normed = punc_norm(text);
        let mut ids = Vec::new();
        ids.push(self.start_text_token);
        ids.extend(self.encode(&normed)?);
        ids.push(self.stop_text_token);
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punc_norm_capitalizes_and_adds_terminator() {
        assert_eq!(punc_norm("hello world"), "Hello world.");
        // Already capitalized + already terminated is untouched (besides whitespace collapse).
        assert_eq!(punc_norm("Hi there!"), "Hi there!");
        // Empty → the reference placeholder.
        assert_eq!(punc_norm(""), "You need to add some text for me to talk.");
    }

    #[test]
    fn punc_norm_collapses_whitespace_and_rewrites_punc() {
        assert_eq!(punc_norm("a    b\tc"), "A b c.");
        // Whitespace is collapsed BEFORE the punctuation rewrite (reference order), so "..." → ", "
        // legitimately leaves the following space in place — a double space, faithfully.
        assert_eq!(punc_norm("wait... really"), "Wait,  really.");
        // ":" → "," (no trailing space added) but ";" → ", " (adds one) — faithful to the table.
        assert_eq!(punc_norm("one:two;three"), "One,two, three.");
        // Smart quotes normalized.
        assert_eq!(punc_norm("\u{201c}hi\u{201d}"), "\"hi\".");
    }

    #[test]
    fn punc_norm_keeps_existing_terminators() {
        for ender in [".", "!", "?", "-", ","] {
            let t = format!("done{ender}");
            let capitalized = format!("Done{ender}");
            assert_eq!(punc_norm(&t), capitalized);
        }
    }
}
