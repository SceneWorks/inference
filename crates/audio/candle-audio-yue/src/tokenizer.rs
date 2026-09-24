//! **Seam: tokenizer / prompt builder.** Genre tags + structured lyrics (+ the ICL reference block)
//! → the per-segment stage-1 prompt token blocks.
//!
//! The production loader is currently the weights-free [`StubTokenizer`]; **sc-19376** replaces
//! [`load`] with the mm sentencepiece tokenizer (`tokenizer.model` in the stage-1 snapshot) and the
//! reference prompt layout. The stub stays as the end-to-end seam test's weights-free double.

use candle_audio::gen_core;

use crate::config::Assets;
use crate::icl::IclPromptCodes;
use crate::tokens::{SOA, XCODEC_SEP};

/// What the prompt builder is given.
#[derive(Clone, Copy, Debug)]
pub struct PromptInput<'a> {
    /// Genre / style tags.
    pub genres: &'a str,
    /// Structured lyrics.
    pub lyrics: &'a str,
    /// The encoded ICL reference block, for ICL renders.
    pub icl: Option<&'a IclPromptCodes>,
}

/// One lyric segment's stage-1 prompt block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentPrompt {
    /// The section label (`verse`, `chorus`, …) — surfaced in progress events.
    pub label: String,
    /// The token block stage 1 is fed before it generates this segment's audio. Segment 0's block
    /// carries the instruction/genre/lyrics head (and the ICL reference block, when present).
    pub ids: Vec<u32>,
}

/// The whole stage-1 prompt: one block per lyric segment, in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stage1Prompt {
    /// Per-segment prompt blocks.
    pub segments: Vec<SegmentPrompt>,
}

/// The tokenizer seam.
pub trait PromptTokenizer: Send {
    /// Build the per-segment prompt blocks. Must return at least one segment for non-empty lyrics.
    fn build(&self, input: &PromptInput<'_>) -> gen_core::Result<Stage1Prompt>;
}

/// Production loader. **Stub until sc-19376** (returns [`StubTokenizer`]).
pub fn load(assets: &Assets) -> gen_core::Result<Box<dyn PromptTokenizer>> {
    load_stub(assets)
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets) -> gen_core::Result<Box<dyn PromptTokenizer>> {
    Ok(Box::new(StubTokenizer))
}

/// **Stub tokenizer** — replaced as the production stage by **sc-19376**. Splits the lyrics on
/// `[label]` section markers and byte-encodes each block deterministically (ids below the special
/// range), ending every block with `<SOA><xcodec>`.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubTokenizer;

impl PromptTokenizer for StubTokenizer {
    fn build(&self, input: &PromptInput<'_>) -> gen_core::Result<Stage1Prompt> {
        let bytes = |s: &str| s.bytes().map(|b| b as u32 + 3).collect::<Vec<u32>>();
        let segments = split_sections(input.lyrics)
            .into_iter()
            .enumerate()
            .map(|(i, (label, text))| {
                let mut ids = Vec::new();
                if i == 0 {
                    ids.extend(bytes(input.genres));
                    if let Some(icl) = input.icl {
                        ids.extend_from_slice(&icl.ids);
                    }
                }
                ids.extend(bytes(&label));
                ids.extend(bytes(&text));
                ids.extend([SOA, XCODEC_SEP]);
                SegmentPrompt { label, ids }
            })
            .collect();
        Ok(Stage1Prompt { segments })
    }
}

/// Split `[label]`-sectioned lyrics into `(label, text)` pairs; unlabelled lyrics are one
/// `lyrics` section.
fn split_sections(lyrics: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut rest = lyrics.trim();
    if !rest.starts_with('[') {
        let end = rest.find('[').unwrap_or(rest.len());
        out.push(("lyrics".into(), rest[..end].trim().to_string()));
        rest = &rest[end..];
    }
    while let Some(close) = rest.find(']') {
        let label = rest[1..close].trim().to_string();
        let body = &rest[close + 1..];
        let end = body.find('[').unwrap_or(body.len());
        out.push((label, body[..end].trim().to_string()));
        rest = &body[end..];
    }
    out.retain(|(label, text)| !(label == "lyrics" && text.is_empty()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_builds_one_block_per_section_with_the_head_on_the_first() {
        let p = StubTokenizer
            .build(&PromptInput {
                genres: "pop",
                lyrics: "[verse]\nhello\n[chorus]\nworld",
                icl: None,
            })
            .unwrap();
        let labels: Vec<&str> = p.segments.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, ["verse", "chorus"]);
        assert!(p.segments[0].ids.len() > p.segments[1].ids.len());
        assert!(p
            .segments
            .iter()
            .all(|s| s.ids.ends_with(&[SOA, XCODEC_SEP])));
    }

    #[test]
    fn unlabelled_lyrics_are_one_section() {
        assert_eq!(
            split_sections("just words"),
            [("lyrics".into(), "just words".into())]
        );
    }
}
