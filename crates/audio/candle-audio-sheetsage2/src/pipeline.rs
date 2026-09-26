//! The model-independent half of whole-song transcription (port of `pipeline_sheetsage2.Transcriber.
//! analyze` at `4f89269`, default preset): the window plan, the overlap prefix each later window is
//! conditioned on, strict/recovered decoding, stitching into one timeline, and the final sort.
//!
//! The model path drives a [`Stitcher`] window by window (the prefix of window *i* depends on the
//! events accepted from windows *< i*); replay drives the same [`Stitcher`] from persisted tokens,
//! and refuses tokens whose prefix does not match the one the stitched events imply.

use crate::events::{
    build_overlap_prefix_tokens, decode_generated_tokens, sliding_window_plan,
    stitched_window_events, Decoded, Event, TimeMap, Window, WindowRecord,
};
use crate::tokenizer::Tokenizer;
use crate::Error;

/// Default-preset overlap between consecutive windows, seconds.
pub const DEFAULT_OVERLAP_SECONDS: f64 = 200.0;
/// Default-preset right-hand look-ahead, seconds.
pub const DEFAULT_LOOKAHEAD_SECONDS: f64 = 100.0;
/// The decoder's positional capacity (`max_output_seq_len`).
pub const MAX_OUTPUT_SEQ_LEN: usize = 5120;

/// Window-by-window stitching state.
#[derive(Debug)]
pub struct Stitcher<'t> {
    tokenizer: &'t Tokenizer,
    prompts: Vec<&'static str>,
    duration: f64,
    window_seconds: f64,
    max_output_seq_len: usize,
    plan: Vec<Window>,
    stitched: Vec<Event>,
    records: Vec<WindowRecord>,
    warnings: Vec<String>,
    pending_base: Option<(usize, i64)>,
}

/// The finished transcription timeline.
#[derive(Clone, Debug)]
pub struct Stitched {
    /// Every accepted event, sorted by `(time, global_subbeat)`.
    pub decoded: Decoded,
    /// The windows, with their exact tokens.
    pub records: Vec<WindowRecord>,
    /// Decode warnings (recovered strict-decode failures, token-limit hits).
    pub warnings: Vec<String>,
    /// Song duration, seconds.
    pub duration: f64,
}

impl<'t> Stitcher<'t> {
    /// Plan a song of `duration` seconds for `prompts` (normalized; `timestamp` is required, as
    /// upstream requires it for timed annotations).
    pub fn new(
        tokenizer: &'t Tokenizer,
        prompts: &[&str],
        duration: f64,
        overlap_seconds: f64,
        lookahead_seconds: f64,
        max_output_seq_len: usize,
    ) -> Result<Self, Error> {
        let prompts = tokenizer.normalize_prompts(prompts)?;
        if !prompts.contains(&"timestamp") {
            return Err(Error::Request(
                "timestamp is required to export timed annotations".into(),
            ));
        }
        let window_seconds = tokenizer.audio_length_seconds();
        let plan =
            sliding_window_plan(duration, window_seconds, overlap_seconds, lookahead_seconds)?;
        Ok(Self {
            tokenizer,
            prompts,
            duration,
            window_seconds,
            max_output_seq_len,
            plan,
            stitched: Vec::new(),
            records: Vec::new(),
            warnings: Vec::new(),
            pending_base: None,
        })
    }

    /// The window plan.
    pub fn plan(&self) -> &[Window] {
        &self.plan
    }

    /// The normalized prompts.
    pub fn prompts(&self) -> &[&'static str] {
        &self.prompts
    }

    /// The generation prefix of window `index`: the prompt prefix for the first window, the overlap
    /// prefix (or the prompt prefix when the overlap holds no beat) for later ones. Refuses a prefix
    /// that leaves under 128 positions of the decoder context, as upstream does.
    pub fn prefix(&mut self, index: usize) -> Result<(Vec<u32>, usize), Error> {
        if index != self.records.len() {
            return Err(Error::Request(format!(
                "window {index} requested before window {} was accepted",
                self.records.len()
            )));
        }
        let window = self.plan[index];
        let mut base = 0;
        let mut prefix_len = 0;
        let mut prefix = self.tokenizer.prompt_prefix(&self.prompts)?;
        if index > 0 {
            if let Some((tokens, b)) = build_overlap_prefix_tokens(
                &self.stitched,
                self.tokenizer,
                &self.prompts,
                window.start,
                window.prefix_end,
            )? {
                if tokens.len() >= self.max_output_seq_len.saturating_sub(128) {
                    return Err(Error::Request(
                        "overlap prefix fills the context; reduce overlap_seconds".into(),
                    ));
                }
                prefix_len = tokens.len();
                prefix = tokens;
                base = b;
            }
        }
        self.pending_base = Some((index, base));
        Ok((prefix, prefix_len))
    }

    /// The window-local time at which generation of window `index` appends `<|eos|>`.
    pub fn stop_time(&self, index: usize) -> f64 {
        let window = self.plan[index];
        window
            .generation_stop
            .unwrap_or_else(|| (self.duration - window.start).min(self.window_seconds))
    }

    /// Accept the generated `tokens` (prefix included) of window `index`.
    pub fn accept(
        &mut self,
        index: usize,
        tokens: Vec<u32>,
        prefix_tokens: usize,
    ) -> Result<(), Error> {
        let base = match self.pending_base.take() {
            Some((i, base)) if i == index => base,
            _ => {
                return Err(Error::Request(format!(
                    "window {index} accepted without its prefix"
                )))
            }
        };
        if tokens.len() > self.max_output_seq_len {
            self.warnings.push(format!(
                "Window {} reached the token limit; inspect its token coverage",
                index + 1
            ));
        }
        let (decoded, warning) = decode_generated_tokens(self.tokenizer, &tokens)?;
        if let Some(w) = warning {
            self.warnings.push(w);
        }
        let map = TimeMap::new(&decoded, self.window_seconds);
        let window = self.plan[index];
        let accepted = stitched_window_events(&decoded, &map, &window, self.duration, index, base);
        self.stitched.extend(accepted);
        self.records.push(WindowRecord {
            index,
            window,
            prefix_tokens,
            tokens,
        });
        Ok(())
    }

    /// Replay persisted window records (tokens of every window, in order), checking that each
    /// later window's recorded prefix is exactly the prefix the accepted events imply.
    pub fn replay(mut self, windows: &[Vec<u32>]) -> Result<Stitched, Error> {
        if windows.len() != self.plan.len() {
            return Err(Error::Replay(format!(
                "{} windows of tokens for a {}-window plan",
                windows.len(),
                self.plan.len()
            )));
        }
        for (index, tokens) in windows.iter().enumerate() {
            let (prefix, prefix_len) = self.prefix(index)?;
            let expected = if prefix_len == 0 {
                &prefix[..]
            } else {
                &prefix[..]
            };
            if tokens.len() < expected.len() || tokens[..expected.len()] != *expected {
                return Err(Error::Replay(format!(
                    "window {index}: persisted tokens do not start with the prefix the stitched \
                     events imply"
                )));
            }
            self.accept(index, tokens.clone(), prefix_len)?;
        }
        Ok(self.finish())
    }

    /// Sort the accepted events into the song timeline.
    pub fn finish(mut self) -> Stitched {
        self.stitched.sort_by(|a, b| {
            let (sa, sb) = (a.stitch.expect("stitched"), b.stitch.expect("stitched"));
            sa.time
                .total_cmp(&sb.time)
                .then(sa.global_subbeat.cmp(&sb.global_subbeat))
        });
        Stitched {
            decoded: Decoded {
                prompts: self.prompts,
                events: self.stitched,
                has_eos: true,
            },
            records: self.records,
            warnings: self.warnings,
            duration: self.duration,
        }
    }
}
