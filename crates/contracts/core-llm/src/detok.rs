//! Incremental-detokenization delta guard (host policy, sc-12452).
//!
//! Both engines stream text the same way: after each sampled token they re-decode the *running*
//! token sequence and emit the new suffix beyond what was already shown. The HF `tokenizers`
//! ByteLevel decoder is **lossy** — a multi-byte UTF-8 character split across BPE tokens first
//! decodes as U+FFFD (REPLACEMENT CHARACTER), then re-decodes as the real character once the
//! completing token arrives. Tracking "already shown" as a raw byte length over that unstable text
//! is therefore wrong in three ways:
//!
//! - **4-byte char (emoji)**: the U+FFFD placeholder is 3 bytes, the finished char is 4 —
//!   `text[shown..]` indexes mid-character and panics ("byte index is not a char boundary").
//! - **3-byte char (CJK)**: placeholder and char are both 3 bytes — `text.len() > shown` never
//!   fires, nothing is emitted, and the stream permanently shows U+FFFD in place of the real char.
//! - **2-byte char**: the placeholder is *longer* than the finished char — characters are dropped,
//!   or the next delta panics.
//!
//! [`IncrementalDetok`] is the one shared fix (the sites in `mlx-llm` and `candle-llm` all drive
//! it): feed it each full re-decode with [`push`](IncrementalDetok::push) and it returns the newly
//! *stable* suffix — it holds back a trailing U+FFFD run (a possibly-incomplete character) until a
//! later decode resolves it, and only ever advances its shown-prefix marker to a char boundary of
//! stable text. A U+FFFD that turns out to be permanent (genuinely invalid bytes mid-stream) is
//! released as soon as any text follows it. A trailing incomplete character at end-of-generation is
//! never emitted at all — dropping the placeholder is the correct "clean" behavior for a stream cut
//! off mid-character.
//!
//! This is pure `&str` policy — no tokenizer types — so it lives here in the tensor-neutral
//! contract crate and serves both engines across their (deliberately different) `tokenizers`
//! versions.

/// Streaming guard turning full re-decodes into safe, stable text deltas.
///
/// See the [module docs](self) for the failure modes this prevents. Usage:
///
/// ```
/// # use core_llm::IncrementalDetok;
/// let mut detok = IncrementalDetok::new();
/// // decode #1: "Hi" — all stable, emitted.
/// assert_eq!(detok.push("Hi"), Some("Hi"));
/// // decode #2: a split emoji decoded lossily — held back, nothing emitted.
/// assert_eq!(detok.push("Hi\u{FFFD}"), None);
/// // decode #3: the completing token arrived — the real char is emitted.
/// assert_eq!(detok.push("Hi🌍"), Some("🌍"));
/// ```
#[derive(Clone, Debug, Default)]
pub struct IncrementalDetok {
    /// Byte length of the stable prefix already emitted. Always a char boundary of every future
    /// decode's stable prefix (stable text is append-only under byte-level decoding).
    shown: usize,
}

impl IncrementalDetok {
    /// A guard with nothing shown yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the full re-decoded text so far; returns the newly stable delta to emit, if any.
    ///
    /// A trailing U+FFFD run is held back (it may be a placeholder for an incomplete character);
    /// everything before it that was not already shown is returned. Returns `None` when nothing new
    /// is stable yet.
    pub fn push<'a>(&mut self, decoded: &'a str) -> Option<&'a str> {
        // The stable prefix: everything up to a trailing REPLACEMENT CHARACTER run. A trailing
        // U+FFFD may still resolve into a real character on the next decode; a *non*-trailing
        // U+FFFD is permanent (its bytes can no longer combine with the tail) and flows through
        // here as soon as any text follows it.
        let stable = decoded.trim_end_matches(char::REPLACEMENT_CHARACTER);
        if stable.len() <= self.shown {
            return None;
        }
        // Defensive: `shown` is by construction a char boundary of the stable prefix, which
        // byte-level decoding only ever appends to. If a decoder ever rewrote earlier text, hold
        // back rather than panic on a mid-character slice.
        if !stable.is_char_boundary(self.shown) {
            return None;
        }
        let delta = &stable[self.shown..];
        self.shown = stable.len();
        Some(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives the guard exactly like the engine sinks do, but with `String::from_utf8_lossy` as
    /// the lossy decoder — the same primitive the HF `tokenizers` ByteLevel decoder uses — over a
    /// growing byte buffer, one "token" (byte chunk) at a time. Returns the emitted deltas.
    fn stream(chunks: &[&[u8]]) -> Vec<String> {
        let mut detok = IncrementalDetok::new();
        let mut acc: Vec<u8> = Vec::new();
        let mut out = Vec::new();
        for chunk in chunks {
            acc.extend_from_slice(chunk);
            let text = String::from_utf8_lossy(&acc).into_owned();
            if let Some(delta) = detok.push(&text) {
                out.push(delta.to_string());
            }
        }
        out
    }

    #[test]
    fn ascii_streams_unchanged() {
        let deltas = stream(&[b"He", b"llo", b" world"]);
        assert_eq!(deltas, ["He", "llo", " world"]);
    }

    #[test]
    fn four_byte_emoji_split_across_tokens_streams_intact() {
        // "a🌍b": the emoji (F0 9F 8C 8D) split 2+2 across byte-level tokens. The old
        // `text[shown..]` pattern emitted the U+FFFD placeholder (3 bytes) and then panicked
        // slicing the finished 4-byte char at byte 3+1: "byte index is not a char boundary".
        let s = "a🌍b".as_bytes(); // 61 F0 9F 8C 8D 62
        let deltas = stream(&[&s[..1], &s[1..3], &s[3..5], &s[5..]]);
        assert_eq!(deltas, ["a", "🌍", "b"]);
        assert_eq!(deltas.concat(), "a🌍b");
    }

    #[test]
    fn four_byte_emoji_one_byte_per_token_streams_intact() {
        // Worst case: every byte its own token — three successive U+FFFD decodes, then the char.
        let s = "🌍".as_bytes();
        let deltas = stream(&[&s[..1], &s[1..2], &s[2..3], &s[3..]]);
        assert_eq!(deltas, ["🌍"]);
    }

    #[test]
    fn three_byte_cjk_split_streams_intact_without_replacement_residue() {
        // "中" (E4 B8 AD) split 1+2: placeholder and finished char are both 3 bytes, so the old
        // `text.len() > shown` check never fired — the stream permanently showed U+FFFD.
        let s = "中".as_bytes();
        let deltas = stream(&[&s[..1], &s[1..]]);
        assert_eq!(deltas, ["中"]);
        assert!(deltas.concat().chars().all(|c| c != char::REPLACEMENT_CHARACTER));
    }

    #[test]
    fn cjk_run_split_at_every_byte_streams_intact() {
        // "你好世界", one byte per token: no delta ever carries U+FFFD, nothing is dropped.
        let s = "你好世界".as_bytes();
        let chunks: Vec<&[u8]> = s.chunks(1).collect();
        let deltas = stream(&chunks);
        assert_eq!(deltas.concat(), "你好世界");
        assert!(!deltas.iter().any(|d| d.contains(char::REPLACEMENT_CHARACTER)));
    }

    #[test]
    fn two_byte_char_split_streams_intact() {
        // "café!": "é" (C3 A9) split 1+1. The old pattern emitted a 3-byte placeholder for a
        // 2-byte char, then panicked or dropped characters on the following delta.
        let s = "café!".as_bytes(); // 63 61 66 C3 A9 21
        let deltas = stream(&[&s[..3], &s[3..4], &s[4..5], &s[5..]]);
        assert_eq!(deltas.concat(), "café!");
        assert!(!deltas.iter().any(|d| d.contains(char::REPLACEMENT_CHARACTER)));
    }

    #[test]
    fn mixed_text_exhaustive_split_sweep() {
        // Every possible single split point of a string mixing 1/2/3/4-byte chars must stream to
        // exactly the original text.
        let text = "ok é 中 🌍 end";
        let bytes = text.as_bytes();
        for cut in 1..bytes.len() {
            let deltas = stream(&[&bytes[..cut], &bytes[cut..]]);
            assert_eq!(deltas.concat(), text, "split at byte {cut}");
        }
    }

    #[test]
    fn permanent_replacement_char_is_released_once_text_follows() {
        // A genuinely invalid byte (a stray continuation byte) decodes as a *permanent* U+FFFD.
        // While trailing it is held back; once any text follows it can never resolve, so it flows
        // through — real (if mangled) model output is not silently dropped mid-stream.
        let deltas = stream(&[b"a", &[0xAD], b"b"]);
        assert_eq!(deltas, ["a", "\u{FFFD}b"]);
    }

    #[test]
    fn trailing_incomplete_char_at_end_of_stream_is_dropped_not_emitted() {
        // Generation cut off mid-emoji: the placeholder never resolves and is never emitted, so
        // the accumulated stream (the thinking/tools `streamed` result text) stays clean.
        let s = "hi🌍".as_bytes();
        let deltas = stream(&[&s[..2], &s[2..4]]); // "hi" + first 2 emoji bytes, then EOS
        assert_eq!(deltas, ["hi"]);
        assert!(!deltas.concat().contains(char::REPLACEMENT_CHARACTER));
    }

    #[test]
    fn equal_length_redecode_emits_nothing_twice() {
        // Idempotent on repeated identical decodes.
        let mut detok = IncrementalDetok::new();
        assert_eq!(detok.push("abc"), Some("abc"));
        assert_eq!(detok.push("abc"), None);
        assert_eq!(detok.push("abcd"), Some("d"));
    }
}
