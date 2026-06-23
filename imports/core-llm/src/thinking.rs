//! Reasoning ("thinking") segmentation for generated text (host policy).
//!
//! A reasoning model emits its chain of thought wrapped in marker tags — `<think>…</think>` for
//! Qwen3 — before its final answer. Separating that reasoning from the answer is the dual of
//! request stop-string matching ([`crate::stop`]): like stop strings, the markers run on the
//! *decoded text*, not token ids (a marker need not align to a token boundary), so this is
//! backend-neutral host policy and lives here, fed the same incremental detokenized deltas a
//! provider already produces. `mlx-llm` and `candle-llm` both drive it through the same seam, only
//! when the provider advertises [`supports_thinking`](crate::TextLlmCapabilities::supports_thinking).
//!
//! [`ThinkingSegmenter`] is a small streaming state machine. Feed each decoded delta with
//! [`push`](ThinkingSegmenter::push); it returns the [`ThinkingSpan`]s now safe to emit, each
//! tagged [`Channel::Thinking`] or [`Channel::Content`] with the marker tags themselves **stripped**
//! (the markers are structure, not output). It holds back the longest suffix that could still grow
//! into the next marker, so a marker split across deltas is matched correctly. When generation ends,
//! call [`flush`](ThinkingSegmenter::flush) to recover any held-back tail.
//!
//! The segmenter starts in [`Channel::Content`] and toggles on each fully-seen marker: it looks for
//! the *open* marker while in content and the *close* marker while in thinking, so it handles
//! multiple interleaved reasoning blocks and ignores a stray non-matching marker.

use crate::output::Channel;

/// A run of decoded text on a single [`Channel`], with any reasoning marker tags removed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThinkingSpan {
    /// Whether this run is reasoning or final-answer text.
    pub channel: Channel,
    /// The (non-empty) decoded text of the run.
    pub text: String,
}

/// A streaming segmenter that splits generated text into reasoning vs answer by marker tags.
///
/// Construct with [`new`](ThinkingSegmenter::new) (or [`Default`] for the `<think>…</think>`
/// markers), then feed each decoded text delta through [`push`](ThinkingSegmenter::push). See the
/// [module docs](self) for the contract.
#[derive(Clone, Debug)]
pub struct ThinkingSegmenter {
    /// Marker that opens a reasoning block (e.g. `<think>`).
    open: String,
    /// Marker that closes a reasoning block (e.g. `</think>`).
    close: String,
    /// Whether we are currently inside a reasoning block.
    in_thinking: bool,
    /// Decoded text received but not yet emitted — the tail that might still begin the next marker.
    pending: String,
}

impl Default for ThinkingSegmenter {
    /// The Qwen3 / common convention: `<think>` … `</think>`.
    fn default() -> Self {
        Self::new("<think>", "</think>")
    }
}

impl ThinkingSegmenter {
    /// Build a segmenter for the given open/close marker tags. Both must be non-empty (an empty
    /// marker would match everywhere); the [`Default`] markers are `<think>` / `</think>`.
    pub fn new(open: impl Into<String>, close: impl Into<String>) -> Self {
        let open = open.into();
        let close = close.into();
        debug_assert!(
            !open.is_empty() && !close.is_empty(),
            "thinking markers must be non-empty"
        );
        Self {
            open,
            close,
            in_thinking: false,
            pending: String::new(),
        }
    }

    /// Whether the segmenter is currently inside a reasoning block (after the last fully-seen
    /// marker). The channel that subsequently-emitted text will carry.
    pub fn in_thinking(&self) -> bool {
        self.in_thinking
    }

    /// The [`Channel`] the next emitted text belongs to given the current state.
    pub fn channel(&self) -> Channel {
        if self.in_thinking {
            Channel::Thinking
        } else {
            Channel::Content
        }
    }

    /// Feed the next decoded text `delta`. Returns the [`ThinkingSpan`]s safe to emit now, in order,
    /// with marker tags stripped. A single delta may cross one or more marker boundaries (so it can
    /// produce several alternating-channel spans); a delta that only extends a partial marker
    /// produces none. Empty spans are never returned.
    pub fn push(&mut self, delta: &str) -> Vec<ThinkingSpan> {
        self.pending.push_str(delta);
        let mut spans: Vec<ThinkingSpan> = Vec::new();
        loop {
            let channel = self.channel();
            let marker = if self.in_thinking { &self.close } else { &self.open };
            if let Some(pos) = self.pending.find(marker.as_str()) {
                // Emit any text before the marker in the current channel, drop the marker, toggle.
                if pos > 0 {
                    spans.push(ThinkingSpan {
                        channel,
                        text: self.pending[..pos].to_string(),
                    });
                }
                self.pending.drain(..pos + marker.len());
                self.in_thinking = !self.in_thinking;
                continue;
            }
            // No full marker: hold back the longest suffix that is a proper prefix of the marker
            // (it might complete on a later delta); emit everything before it.
            let hold = max_partial_suffix(&self.pending, marker);
            let safe = self.pending.len() - hold;
            if safe > 0 {
                spans.push(ThinkingSpan {
                    channel,
                    text: self.pending[..safe].to_string(),
                });
                self.pending.drain(..safe);
            }
            break;
        }
        spans
    }

    /// Take any held-back tail — text that looked like it might begin a marker but that generation
    /// ended before completing. Call once, when generation finishes. Returns at most one span (in
    /// the current channel); empty if nothing was held back.
    pub fn flush(&mut self) -> Vec<ThinkingSpan> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let channel = self.channel();
        vec![ThinkingSpan {
            channel,
            text: std::mem::take(&mut self.pending),
        }]
    }
}

/// Byte length of the longest suffix of `pending` that equals a *proper* prefix of `marker`. Full
/// matches are ruled out by the caller before this is called, so the result is always shorter than
/// `marker`. The char-boundary guard on `marker` keeps a multibyte marker from panic-slicing.
fn max_partial_suffix(pending: &str, marker: &str) -> usize {
    let mut k = pending.len().min(marker.len());
    while k > 0 {
        if marker.is_char_boundary(k) && pending.ends_with(&marker[..k]) {
            return k;
        }
        k -= 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a segmenter over deltas; return the concatenated (content, thinking) text and the
    /// number of channel-tagged spans seen (markers are stripped throughout).
    fn run(deltas: &[&str]) -> (String, String, usize) {
        let mut seg = ThinkingSegmenter::default();
        let (mut content, mut thinking) = (String::new(), String::new());
        let mut count = 0;
        for d in deltas {
            for s in seg.push(d) {
                count += 1;
                match s.channel {
                    Channel::Content => content.push_str(&s.text),
                    Channel::Thinking => thinking.push_str(&s.text),
                }
            }
        }
        for s in seg.flush() {
            count += 1;
            match s.channel {
                Channel::Content => content.push_str(&s.text),
                Channel::Thinking => thinking.push_str(&s.text),
            }
        }
        (content, thinking, count)
    }

    #[test]
    fn no_markers_is_all_content() {
        let (content, thinking, _) = run(&["hello ", "world"]);
        assert_eq!(content, "hello world");
        assert_eq!(thinking, "");
    }

    #[test]
    fn single_block_in_one_delta() {
        let (content, thinking, _) = run(&["<think>reason</think>answer"]);
        assert_eq!(thinking, "reason");
        assert_eq!(content, "answer");
    }

    #[test]
    fn block_spread_across_deltas() {
        let (content, thinking, _) = run(&["<think>", "step 1 ", "step 2", "</think>", "the answer"]);
        assert_eq!(thinking, "step 1 step 2");
        assert_eq!(content, "the answer");
    }

    #[test]
    fn markers_split_byte_by_byte() {
        // Both markers arrive one byte per delta; none of their bytes may leak into the output.
        let deltas = [
            "<", "t", "h", "i", "n", "k", ">", "r", "<", "/", "t", "h", "i", "n", "k", ">", "a",
        ];
        let (content, thinking, _) = run(&deltas);
        assert_eq!(thinking, "r");
        assert_eq!(content, "a");
    }

    #[test]
    fn genuinely_empty_block_yields_no_thinking() {
        // Markers with nothing between them → no reasoning span at all, just the trailing content.
        // (In real no-think generation the `<think></think>` echo is in the *prompt*, so the
        // generated stream the segmenter sees has no markers — see `no_markers_is_all_content`.)
        let (content, thinking, count) = run(&["<think></think>hi"]);
        assert_eq!(thinking, "");
        assert_eq!(content, "hi");
        assert_eq!(count, 1, "only the single content span is emitted");
    }

    #[test]
    fn whitespace_only_block_is_reported_verbatim() {
        // Whatever sits between the markers is reasoning, even if it is only whitespace — the
        // segmenter strips the markers but does not trim content.
        let (content, thinking, _) = run(&["<think>\n\n</think>\n\nhi"]);
        assert_eq!(thinking, "\n\n");
        assert_eq!(content, "\n\nhi");
    }

    #[test]
    fn multiple_interleaved_blocks() {
        let (content, thinking, _) = run(&["a<think>r1</think>b<think>r2</think>c"]);
        assert_eq!(content, "abc");
        assert_eq!(thinking, "r1r2");
    }

    #[test]
    fn partial_open_marker_that_is_not_a_marker_is_flushed_as_content() {
        // "<thi" looks like the start of "<think>" so it is held; generation ends → it is content.
        let (content, thinking, _) = run(&["plain <thi"]);
        assert_eq!(content, "plain <thi");
        assert_eq!(thinking, "");
    }

    #[test]
    fn partial_open_marker_disambiguated_as_literal_text() {
        // "<think" then a non-">" proves it was not the marker; the held text is released to content.
        let (content, thinking, _) = run(&["<think", "x more"]);
        assert_eq!(content, "<thinkx more");
        assert_eq!(thinking, "");
    }

    #[test]
    fn close_marker_only_matched_inside_thinking() {
        // A stray "</think>" in content (not inside a block) is not a transition — it never opened.
        let (content, thinking, _) = run(&["answer </think> still answer"]);
        assert_eq!(content, "answer </think> still answer");
        assert_eq!(thinking, "");
    }

    #[test]
    fn state_accessors_track_position() {
        let mut seg = ThinkingSegmenter::default();
        assert!(!seg.in_thinking());
        assert_eq!(seg.channel(), Channel::Content);
        seg.push("<think>r");
        assert!(seg.in_thinking());
        assert_eq!(seg.channel(), Channel::Thinking);
        seg.push("</think>");
        assert!(!seg.in_thinking());
    }

    #[test]
    fn custom_markers() {
        let mut seg = ThinkingSegmenter::new("<reasoning>", "</reasoning>");
        let spans = seg.push("<reasoning>why</reasoning>ok");
        assert_eq!(
            spans,
            vec![
                ThinkingSpan { channel: Channel::Thinking, text: "why".into() },
                ThinkingSpan { channel: Channel::Content, text: "ok".into() },
            ]
        );
    }

    #[test]
    fn leading_content_before_first_block() {
        let (content, thinking, _) = run(&["intro <think>r</think> outro"]);
        assert_eq!(content, "intro  outro");
        assert_eq!(thinking, "r");
    }
}
