//! Stop-string matching for generated text (host policy).
//!
//! A request may carry extra [`stop`](crate::TextLlmRequest::stop) strings beyond the model's EOS
//! tokens; a provider must halt generation at the first occurrence of any of them and **must not
//! emit the stop string itself** (OpenAI semantics). Because a stop string need not align to a
//! token boundary (e.g. `"END"` can decode as `…E` + `ND`), matching is on the *decoded text*, not
//! token ids — so it is backend-neutral host policy and lives here, fed the incremental text
//! deltas a provider's detokenizer already produces, not in any tensor backend. `mlx-llm` and
//! `candle-llm` both drive it through the same incremental-detokenization seam.
//!
//! [`StopMatcher`] is a small streaming matcher. Feed it each decoded text delta with
//! [`push`](StopMatcher::push); it returns the prefix that is safe to emit now and whether a stop
//! string was hit. It holds back the longest suffix that could still grow into a stop string, so a
//! stop string split across several deltas is matched correctly. When generation ends for any other
//! reason (EOS / length), call [`flush`](StopMatcher::flush) to recover any held-back tail that
//! turned out not to begin a stop string.

/// What feeding a text delta to a [`StopMatcher`] produced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StopChunk {
    /// Text that is now safe to emit: the accumulated input minus any matched stop string and minus
    /// a held-back tail that might still complete a stop string on a later delta. May be empty.
    pub emit: String,
    /// Whether a stop string was matched — generation should halt now.
    pub stop: bool,
}

/// A streaming matcher for request stop strings.
///
/// Construct with the request's stop strings, then feed each decoded text delta through
/// [`push`](StopMatcher::push). See the [module docs](self) for the matching contract.
#[derive(Clone, Debug, Default)]
pub struct StopMatcher {
    /// The (non-empty) stop strings to match against.
    stops: Vec<String>,
    /// Decoded text received but not yet emitted — the tail that might still begin a stop string.
    pending: String,
}

impl StopMatcher {
    /// Build a matcher for `stops`. Empty stop strings are dropped (they would match everywhere, so
    /// OpenAI ignores them); a matcher with no non-empty stops is a transparent pass-through (see
    /// [`is_empty`](StopMatcher::is_empty)).
    pub fn new(stops: impl IntoIterator<Item = String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            pending: String::new(),
        }
    }

    /// Whether this matcher has no stop strings — every [`push`](StopMatcher::push) returns its
    /// input unchanged and never signals a stop. Lets a provider skip the matching path entirely
    /// (and keep its existing detokenization output byte-identical) when no stops were requested.
    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    /// Feed the next decoded text `delta`. Returns a [`StopChunk`]: the text safe to emit now and
    /// whether a stop string was hit. After `stop` is `true` the matcher is drained (the stop
    /// string and everything after it on this delta are discarded) and should not be fed further.
    pub fn push(&mut self, delta: &str) -> StopChunk {
        if self.stops.is_empty() {
            return StopChunk {
                emit: delta.to_string(),
                stop: false,
            };
        }
        self.pending.push_str(delta);

        // Earliest byte offset at which any stop string occurs in full within the pending buffer.
        let mut cut: Option<usize> = None;
        for s in &self.stops {
            if let Some(pos) = self.pending.find(s.as_str()) {
                cut = Some(cut.map_or(pos, |c| c.min(pos)));
            }
        }
        if let Some(cut) = cut {
            // Emit up to the stop string; discard it and anything generated after it on this delta.
            let emit = self.pending[..cut].to_string();
            self.pending.clear();
            return StopChunk { emit, stop: true };
        }

        // No full match: hold back the longest suffix of `pending` that is a proper prefix of some
        // stop string — it might complete on a later delta. Emit everything before it.
        let hold = self.max_partial_suffix();
        let safe = self.pending.len() - hold;
        let emit = self.pending[..safe].to_string();
        self.pending.drain(..safe);
        StopChunk { emit, stop: false }
    }

    /// Take any held-back tail — text that looked like it might begin a stop string but that
    /// generation ended (EOS / length) before completing. Call once, when generation finishes for a
    /// reason *other* than a stop string; after a stop-string hit the buffer is already empty.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Byte length of the longest suffix of `pending` that equals a (proper) prefix of some stop
    /// string. Full matches are ruled out before this is called, so the matched prefix is always
    /// shorter than the stop string itself.
    fn max_partial_suffix(&self) -> usize {
        let mut best = 0;
        for s in &self.stops {
            // Longest possible overlap is bounded by both lengths; never search below `best`.
            let mut k = self.pending.len().min(s.len());
            while k > best {
                if s.is_char_boundary(k) && self.pending.ends_with(&s[..k]) {
                    best = k;
                    break;
                }
                k -= 1;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a matcher over a sequence of deltas, returning the concatenated emitted text, whether a
    /// stop was hit, and the flushed tail (only meaningful when no stop was hit).
    fn run(stops: &[&str], deltas: &[&str]) -> (String, bool, String) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()));
        let mut out = String::new();
        let mut stopped = false;
        for d in deltas {
            let chunk = m.push(d);
            out.push_str(&chunk.emit);
            if chunk.stop {
                stopped = true;
                break;
            }
        }
        let tail = if stopped { String::new() } else { m.flush() };
        (out, stopped, tail)
    }

    #[test]
    fn no_stops_is_passthrough() {
        let m = StopMatcher::new(Vec::<String>::new());
        assert!(m.is_empty());
        let (out, stopped, tail) = run(&[], &["hello ", "world"]);
        assert_eq!(out, "hello world");
        assert!(!stopped);
        assert_eq!(tail, "");
    }

    #[test]
    fn empty_stop_strings_are_ignored() {
        let m = StopMatcher::new(["".to_string(), "".to_string()]);
        assert!(m.is_empty(), "all-empty stops collapse to a pass-through");
        let m2 = StopMatcher::new(["".to_string(), "END".to_string()]);
        assert!(!m2.is_empty());
    }

    #[test]
    fn stop_within_a_single_delta_trims_and_halts() {
        let (out, stopped, _) = run(&["END"], &["abcENDxyz"]);
        assert_eq!(out, "abc", "text before the stop is emitted; the stop and tail are dropped");
        assert!(stopped);
    }

    #[test]
    fn stop_split_across_deltas() {
        // "END" arrives one byte per delta; nothing before it should leak, and it must still match.
        let (out, stopped, _) = run(&["END"], &["abc", "E", "N", "D", "more"]);
        assert_eq!(out, "abc");
        assert!(stopped);
    }

    #[test]
    fn partial_prefix_is_held_then_released_when_not_a_stop() {
        // "EN" looks like the start of "END" so it is held; the next delta proves it is not.
        let (out, stopped, tail) = run(&["END"], &["abEN", "Xyz"]);
        assert_eq!(out, "abENXyz");
        assert!(!stopped);
        assert_eq!(tail, "");
    }

    #[test]
    fn held_tail_is_flushed_on_natural_finish() {
        // Generation ends while a partial-prefix tail is still held back — it is real output.
        let (out, stopped, tail) = run(&["WORLD"], &["hello WOR"]);
        assert_eq!(out, "hello ");
        assert!(!stopped);
        assert_eq!(tail, "WOR", "the held partial prefix is recovered on flush");
    }

    #[test]
    fn earliest_of_several_stops_wins() {
        let (out, stopped, _) = run(&["END", "STOP"], &["aaSTOPbbENDcc"]);
        assert_eq!(out, "aa");
        assert!(stopped);
    }

    #[test]
    fn shorter_overlapping_stop_cuts_first() {
        // Both "ab" and "abc" match at the same offset; cut at the first occurrence of any stop.
        let (out, stopped, _) = run(&["abc", "ab"], &["xxabcyy"]);
        assert_eq!(out, "xx");
        assert!(stopped);
    }

    #[test]
    fn multibyte_partial_prefix_respects_char_boundaries() {
        // Stop "a→b" embeds a 3-byte char (U+2192). A held prefix must only ever be cut on a char
        // boundary of the stop ("a→", never mid-"→"), or `max_partial_suffix` would panic-slice.
        let mut m = StopMatcher::new(["a→b".to_string()]);
        let c1 = m.push("xa→"); // holds "a→" (a proper, char-boundary prefix), emits "x"
        assert_eq!(c1.emit, "x");
        assert!(!c1.stop);
        let c2 = m.push("b done"); // "a→b" now complete → stop, drop the trailing text
        assert_eq!(c2.emit, "");
        assert!(c2.stop);
    }

    #[test]
    fn multibyte_partial_prefix_held_across_chars() {
        // A one-char prefix of a multi-char unicode stop must be held back, not emitted early.
        let (out, stopped, _) = run(&["café"], &["a ca", "fé done"]);
        assert_eq!(out, "a ", "only the text before the stop is emitted");
        assert!(stopped);
    }

    #[test]
    fn stop_equal_to_entire_pending() {
        let (out, stopped, _) = run(&["DONE"], &["DONE"]);
        assert_eq!(out, "");
        assert!(stopped);
    }

    #[test]
    fn nothing_emitted_until_prefix_disambiguates() {
        // A single stop, fed a delta that is exactly a proper prefix: emit nothing yet.
        let mut m = StopMatcher::new(["STOP".to_string()]);
        let c = m.push("STO");
        assert_eq!(c.emit, "");
        assert!(!c.stop);
        let c2 = m.push("P");
        assert_eq!(c2.emit, "");
        assert!(c2.stop);
    }
}
