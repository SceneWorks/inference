//! Prompt formatting + tokenization for MOSS-SoundEffect v2.0 (sc-12841).
//!
//! Mirrors the reference front-end exactly where it matters for the shipped model:
//!
//! - **Duration suffix** — the pipeline appends `" duration: {seconds:.1}s"` to every positive
//!   prompt (the training-time convention that makes `target_duration` controllable). The
//!   negative prompt is passed through without a suffix.
//! - **`whitespace` clean** — the `WanPrompter` tokenizer runs `whitespace_clean(basic_clean(t))`:
//!   HTML-entity unescape (applied twice, as the reference does), whitespace collapse, strip.
//!   The reference additionally runs `ftfy.fix_text` (mojibake repair); that step is an identity
//!   on well-formed UTF-8 input and is deliberately not ported — a Rust ftfy would be a large
//!   dependency for a transformation that only fires on already-corrupted text.
//! - **Tokenization** — the snapshot's Qwen tokenizer (`tokenizer/tokenizer.json`), truncated to
//!   [`TEXT_LEN`] tokens. The reference pads to `max_length=512` and later **zeroes** every
//!   embedding row at or beyond the sequence's valid length, so padding token ids never reach the
//!   DiT — this port therefore encodes the unpadded ids (causal attention makes the valid rows
//!   identical) and zero-pads the *embedding* to 512 rows in the pipeline.

use candle_audio::{AudioError, Result};
use tokenizers::Tokenizer;

/// The DiT's text-context length (the `WanPrompter` `text_len` — every context is exactly this
/// many rows, valid rows first, zero rows after).
pub const TEXT_LEN: usize = 512;

/// Round a duration to one decimal (the reference `round(float(seconds), 1)`).
pub fn round_seconds(seconds: f32) -> f32 {
    (seconds * 10.0).round() / 10.0
}

/// The positive-prompt duration suffix: `"{prompt} duration: {seconds:.1}s"`.
pub fn with_duration_suffix(prompt: &str, seconds: f32) -> String {
    format!("{} duration: {seconds:.1}s", prompt.trim())
}

/// Unescape the common HTML entities (`html.unescape` for the named + numeric forms that occur
/// in practice; the reference applies it twice, which this reproduces at the call site).
fn html_unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'&' {
            if let Some(semi) = text[i..].find(';').map(|p| i + p) {
                let entity = &text[i + 1..semi];
                let replacement = match entity {
                    "amp" => Some('&'),
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    "nbsp" => Some('\u{a0}'),
                    _ => entity
                        .strip_prefix('#')
                        .and_then(|num| {
                            if let Some(hex) = num.strip_prefix('x').or(num.strip_prefix('X')) {
                                u32::from_str_radix(hex, 16).ok()
                            } else {
                                num.parse::<u32>().ok()
                            }
                        })
                        .and_then(char::from_u32),
                };
                if let Some(c) = replacement {
                    out.push(c);
                    i = semi + 1;
                    continue;
                }
            }
        }
        let c = text[i..].chars().next().unwrap();
        out.push(c);
        i += c.len_utf8();
    }
    out
}

/// The reference `whitespace_clean(basic_clean(text))`: double HTML unescape, collapse every
/// whitespace run to one space, strip.
pub fn clean_prompt(text: &str) -> String {
    let unescaped = html_unescape(&html_unescape(text));
    unescaped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Tokenize a cleaned prompt to at most [`TEXT_LEN`] ids (reference truncation). An empty
/// cleaned prompt yields no ids — the caller maps that to the all-zero context, exactly like the
/// reference's zeroed-tail path with `seq_len = 0`.
pub fn tokenize(tokenizer: &Tokenizer, cleaned: &str) -> Result<Vec<u32>> {
    if cleaned.is_empty() {
        return Ok(Vec::new());
    }
    let encoding = tokenizer
        .encode(cleaned, true)
        .map_err(|e| AudioError::Msg(format!("moss-sfx tokenize: {e}")))?;
    let mut ids = encoding.get_ids().to_vec();
    ids.truncate(TEXT_LEN);
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_suffix_matches_the_training_convention() {
        assert_eq!(
            with_duration_suffix("glass shattering", 4.0),
            "glass shattering duration: 4.0s"
        );
        assert_eq!(
            with_duration_suffix("  rain on a tin roof \n", round_seconds(12.34)),
            "rain on a tin roof duration: 12.3s"
        );
    }

    #[test]
    fn clean_collapses_whitespace_and_unescapes_entities() {
        assert_eq!(clean_prompt("a   dog\n\tbarks"), "a dog barks");
        assert_eq!(clean_prompt("thunder &amp; rain"), "thunder & rain");
        // Double unescape (reference applies html.unescape twice).
        assert_eq!(clean_prompt("&amp;amp;"), "&");
        assert_eq!(clean_prompt("&#65;&#x42;"), "AB");
        assert_eq!(clean_prompt("   "), "");
        // Un-terminated / unknown entities pass through unchanged.
        assert_eq!(clean_prompt("AT&T &unknown; x"), "AT&T &unknown; x");
    }
}
