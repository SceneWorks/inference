//! **Seam: tokenizer / prompt builder.** Genre tags + structured lyrics (+ the ICL reference block)
//! → the per-segment stage-1 prompt token blocks (sc-19376).
//!
//! # Tokenizer
//!
//! [`MmTokenizer`] is YuE's mm SentencePiece tokenizer, loaded from the `tokenizer.json` that ships
//! in every stage-1 tier directory: the byte-fallback BPE derived from the upstream
//! `mm_tokenizer_v0.2_hf/tokenizer.model`, with the mm special tokens (`<SOA>`, `<EOA>`,
//! `<stage_1>`, …, `<s>`, `</s>`, `<CLS>`, `<SEP>`, `<MASK>`, `<PAD>`) as added tokens at their mm
//! ids. The upstream `_MMSentencePieceTokenizer.tokenize` splits the text on those special-token
//! strings and SentencePiece-encodes each span on its own; the `tokenizers` added-token split does
//! the same (every special token ends in the only `>` it contains, so "earliest match, first
//! declared wins" and "leftmost-longest" pick the same token).
//!
//! # Prompt layout
//!
//! [`YuePromptBuilder`] ports the reference `Stage1Pipeline.get_prompt_texts`,
//! `get_first_segment_prompt` and `get_segment_prompt` (YuE-v1 `infer.py`; the same layout in
//! YuE-exllamav2's `infer_stage1.py`):
//!
//! * sections: every `\[(\w+)\](.*?)(?=\[|\Z)` match of the lyrics, formatted
//!   `"[label]\n{body.strip()}\n\n"` (text before the first label is dropped; a label that is not
//!   `\w+`, e.g. `[verse 1]`, is not a section and ends the previous body);
//! * head: `"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{sections
//!   joined by \n}"`, followed for ICL renders by
//!   `[start_of_reference] <SOA><xcodec> {codes} <EOA> [end_of_reference]`;
//! * segment 0: `head + [start_of_segment] + section_0 + <SOA><xcodec>`;
//! * segment i > 0: `[end_of_segment] + [start_of_segment] + section_i + <SOA><xcodec>`.
//!
//! The reference CLI reads genres and lyrics from text files (`open(path).read().strip()`), so the
//! builder applies the same universal-newline translation (`\r\n`, `\r` → `\n`) and Python
//! `str.strip()` before building — the golden fixture goes through that exact read.
//!
//! # Smart context
//!
//! [`shorten_context`] ports YuE-exllamav2's `shorten_input`: when the running stage-1 sequence
//! outgrows `max_context` (`cache − max_new_tokens − 1`), drop the oldest `[start_of_segment]`
//! block (keeping the head) until it fits, falling back to a plain tail cut once fewer than three
//! segment markers remain. It is a pure function over the running sequence — which only stage 1
//! holds (prompt blocks interleaved with generated audio) — so the stage-1 decode (sc-19380) calls
//! it before each segment's prefill; the marker it scans for is [`START_OF_SEGMENT`].
//!
//! The golden fixture `tests/fixtures/yue_prompt_reference.json` is produced by running the
//! reference Python itself (`scripts/reference/yue_prompt_reference.py`).
//! [`StubTokenizer`] stays as the end-to-end seam test's weights-free double.

use std::path::Path;
use std::sync::OnceLock;

use candle_audio::gen_core;
use regex::Regex;

use crate::config::Assets;
use crate::icl::IclPromptCodes;
use crate::tokens::{EOA, SOA, XCODEC_SEP};

/// The tokenizer file inside the stage-1 snapshot.
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// `tokenize("[start_of_segment]")` under the mm tokenizer — the marker [`shorten_context`] scans
/// for. The tokenizer is one fixed file shared by all six stage-1 checkpoints; the golden fixture
/// and the real-tokenizer test both pin this value.
pub const START_OF_SEGMENT: [u32; 7] = [518, 2962, 29918, 974, 29918, 28192, 29962];

/// The reference's instruction line, the head of segment 0's block.
const INSTRUCTION: &str = "Generate music from the given lyrics segment by segment.";

/// What the prompt builder is given.
#[derive(Clone, Copy, Debug)]
pub struct PromptInput<'a> {
    /// Genre / style tags.
    pub genres: &'a str,
    /// Structured lyrics.
    pub lyrics: &'a str,
    /// The encoded ICL reference block, for ICL renders: the windowed codebook-0 mm ids (the
    /// builder adds the `<SOA><xcodec>` … `<EOA>` and `[start_of_reference]` wrapping).
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

/// Text → mm token ids, the one operation the prompt builder needs from a tokenizer.
pub trait TextEncoder: Send {
    /// Encode `text` exactly as the upstream `_MMSentencePieceTokenizer.tokenize` does (special
    /// token strings map to their ids; no BOS/EOS added).
    fn encode(&self, text: &str) -> gen_core::Result<Vec<u32>>;
}

/// YuE's mm tokenizer over `tokenizer.json`.
pub struct MmTokenizer {
    inner: tokenizers::Tokenizer,
}

impl std::fmt::Debug for MmTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmTokenizer").finish_non_exhaustive()
    }
}

impl MmTokenizer {
    /// Load `tokenizer.json`.
    pub fn from_file(path: &Path) -> gen_core::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            gen_core::Error::Msg(format!(
                "candle-audio-yue: cannot load the mm tokenizer {}: {e}",
                path.display()
            ))
        })?;
        Ok(Self { inner })
    }
}

impl TextEncoder for MmTokenizer {
    fn encode(&self, text: &str) -> gen_core::Result<Vec<u32>> {
        let enc = self.inner.encode(text, false).map_err(|e| {
            gen_core::Error::Msg(format!("candle-audio-yue: mm tokenizer encode failed: {e}"))
        })?;
        Ok(enc.get_ids().to_vec())
    }
}

/// The reference stage-1 prompt builder over a [`TextEncoder`].
#[derive(Debug)]
pub struct YuePromptBuilder<E> {
    encoder: E,
    start_of_segment: Vec<u32>,
    end_of_segment: Vec<u32>,
    start_of_reference: Vec<u32>,
    end_of_reference: Vec<u32>,
}

impl<E: TextEncoder> YuePromptBuilder<E> {
    /// Wrap `encoder`, tokenizing the segment/reference markers once (as the reference does at
    /// pipeline construction).
    pub fn new(encoder: E) -> gen_core::Result<Self> {
        Ok(Self {
            start_of_segment: encoder.encode("[start_of_segment]")?,
            end_of_segment: encoder.encode("[end_of_segment]")?,
            start_of_reference: encoder.encode("[start_of_reference]")?,
            end_of_reference: encoder.encode("[end_of_reference]")?,
            encoder,
        })
    }

    /// The tokenized `[start_of_segment]` marker ([`START_OF_SEGMENT`] under the mm tokenizer).
    pub fn start_of_segment(&self) -> &[u32] {
        &self.start_of_segment
    }
}

impl<E: TextEncoder> PromptTokenizer for YuePromptBuilder<E> {
    fn build(&self, input: &PromptInput<'_>) -> gen_core::Result<Stage1Prompt> {
        let genres = read_like_file(input.genres);
        let lyrics = read_like_file(input.lyrics);
        let sections = split_lyrics(&lyrics);
        if sections.is_empty() {
            return Err(gen_core::Error::Msg(
                "candle-audio-yue: the lyrics hold no `[section]` label (YuE sings lyrics split \
                 into `[verse]`, `[chorus]`, … sections)"
                    .into(),
            ));
        }
        let full_lyrics = sections
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let mut head = self
            .encoder
            .encode(&format!("{INSTRUCTION}\n[Genre] {genres}\n{full_lyrics}"))?;
        if let Some(icl) = input.icl {
            head.extend_from_slice(&self.start_of_reference);
            head.extend([SOA, XCODEC_SEP]);
            head.extend_from_slice(&icl.ids);
            head.push(EOA);
            head.extend_from_slice(&self.end_of_reference);
        }

        let mut segments = Vec::with_capacity(sections.len());
        for (i, section) in sections.into_iter().enumerate() {
            let section_text = section
                .text
                .replace("[start_of_segment]", "")
                .replace("[end_of_segment]", "");
            let mut ids = if i == 0 {
                std::mem::take(&mut head)
            } else {
                self.end_of_segment.clone()
            };
            ids.extend_from_slice(&self.start_of_segment);
            ids.extend(self.encoder.encode(&section_text)?);
            ids.extend([SOA, XCODEC_SEP]);
            segments.push(SegmentPrompt {
                label: section.label,
                ids,
            });
        }
        Ok(Stage1Prompt { segments })
    }
}

/// Production loader: the mm tokenizer from the stage-1 snapshot's `tokenizer.json` behind the
/// reference prompt builder.
pub fn load(assets: &Assets) -> gen_core::Result<Box<dyn PromptTokenizer>> {
    let tokenizer = MmTokenizer::from_file(&assets.tokenizer_json())?;
    Ok(Box::new(YuePromptBuilder::new(tokenizer)?))
}

/// One lyric section: its label and the reference's structured text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    /// The `\w+` label inside the brackets.
    pub label: String,
    /// `"[label]\n{body.strip()}\n\n"`.
    pub text: String,
}

/// The reference `split_lyrics`: `re.findall(r"\[(\w+)\](.*?)(?=\[|\Z)", lyrics, re.DOTALL)`.
///
/// Python's Unicode `\w` is exactly `[\p{L}\p{N}_]`, and a lazy `.*?` up to the next `[` (or the
/// end) is `[^\[]*`, so the leftmost non-overlapping matches are the same.
pub fn split_lyrics(lyrics: &str) -> Vec<Section> {
    static SECTION: OnceLock<Regex> = OnceLock::new();
    let re = SECTION.get_or_init(|| {
        Regex::new(r"\[([\p{L}\p{N}_]+)\]([^\[]*)").expect("section pattern compiles")
    });
    re.captures_iter(lyrics)
        .map(|c| {
            let label = c[1].to_string();
            let text = format!("[{label}]\n{}\n\n", py_strip(&c[2]));
            Section { label, text }
        })
        .collect()
}

/// Python `str.isspace`: Unicode `White_Space` plus the four information separators
/// U+001C..=U+001F.
fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python `str.strip()`.
fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// What the reference CLI's `open(path).read().strip()` yields for `text`: universal-newline
/// translation (`\r\n` and lone `\r` → `\n`), then Python `strip()`.
fn read_like_file(text: &str) -> String {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    py_strip(&text).to_string()
}

/// The reference "smart context" (`Stage1Pipeline.shorten_input`, YuE-exllamav2): while `seq` is
/// longer than `max_context`, drop the oldest segment block — the span from the first
/// [`START_OF_SEGMENT`] marker to the second, so the head before it survives. Once fewer than three
/// markers remain (the current segment plus at least one earlier one must stay for continuity),
/// fall back to keeping the last `max_context` tokens of the sequence as shortened so far.
pub fn shorten_context(seq: &[u32], max_context: usize) -> Vec<u32> {
    let mut seq = seq.to_vec();
    while seq.len() > max_context {
        let marks: Vec<usize> = seq
            .windows(START_OF_SEGMENT.len())
            .enumerate()
            .filter(|(_, w)| *w == START_OF_SEGMENT)
            .map(|(i, _)| i)
            .take(3)
            .collect();
        if marks.len() < 3 {
            return seq.split_off(seq.len() - max_context);
        }
        seq.drain(marks[0]..marks[1]);
    }
    seq
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets) -> gen_core::Result<Box<dyn PromptTokenizer>> {
    Ok(Box::new(StubTokenizer))
}

/// **Stub tokenizer** — the end-to-end seam test's weights-free double (production is
/// [`YuePromptBuilder`] over [`MmTokenizer`]). Splits the lyrics on `[label]` section markers and
/// byte-encodes each block deterministically (ids below the special range), ending every block
/// with `<SOA><xcodec>`.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubTokenizer;

impl PromptTokenizer for StubTokenizer {
    fn build(&self, input: &PromptInput<'_>) -> gen_core::Result<Stage1Prompt> {
        let bytes = |s: &str| s.bytes().map(|b| b as u32 + 3).collect::<Vec<u32>>();
        let segments = stub_split_sections(input.lyrics)
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

/// The stub's lenient split: `[label]`-sectioned lyrics into `(label, text)` pairs; unlabelled
/// lyrics are one `lyrics` section.
fn stub_split_sections(lyrics: &str) -> Vec<(String, String)> {
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
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use serde_json::Value;

    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/yue_prompt_reference.json");

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("golden fixture parses")
    }

    fn ids(v: &Value) -> Vec<u32> {
        v.as_array()
            .expect("id array")
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect()
    }

    /// Replays the reference tokenizer: the exact `tokenize(text) → ids` calls the Python prompt
    /// builder made. Any text the Rust builder asks for that the reference never tokenized is an
    /// error, so this also proves the builder tokenizes the same strings.
    struct ReplayEncoder(HashMap<String, Vec<u32>>);

    impl TextEncoder for ReplayEncoder {
        fn encode(&self, text: &str) -> gen_core::Result<Vec<u32>> {
            self.0.get(text).cloned().ok_or_else(|| {
                gen_core::Error::Msg(format!("the reference never tokenized {text:?}"))
            })
        }
    }

    fn replay_table(fx: &Value, case: &Value) -> HashMap<String, Vec<u32>> {
        let mut table: HashMap<String, Vec<u32>> = case["encoded"]
            .as_object()
            .expect("encoded table")
            .iter()
            .map(|(k, v)| (k.clone(), ids(v)))
            .collect();
        for (marker, text) in [
            ("start_of_segment", "[start_of_segment]"),
            ("end_of_segment", "[end_of_segment]"),
            ("start_of_reference", "[start_of_reference]"),
            ("end_of_reference", "[end_of_reference]"),
        ] {
            table.insert(text.into(), ids(&fx["markers"][marker]));
        }
        table
    }

    fn icl_of(case: &Value) -> Option<IclPromptCodes> {
        (!case["icl_ids"].is_null()).then(|| IclPromptCodes {
            ids: ids(&case["icl_ids"]),
        })
    }

    /// Build `case` with `builder` and assert labels + every block equal the reference.
    fn assert_case_matches(builder: &dyn PromptTokenizer, case: &Value) {
        let name = case["name"].as_str().unwrap();
        let icl = icl_of(case);
        let got = builder
            .build(&PromptInput {
                genres: case["genres"].as_str().unwrap(),
                lyrics: case["lyrics"].as_str().unwrap(),
                icl: icl.as_ref(),
            })
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let labels: Vec<&str> = got.segments.iter().map(|s| s.label.as_str()).collect();
        let want_labels: Vec<&str> = case["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l.as_str().unwrap())
            .collect();
        assert_eq!(labels, want_labels, "{name}: section labels");
        let want = case["segments"].as_array().unwrap();
        assert_eq!(got.segments.len(), want.len(), "{name}: segment count");
        for (i, (g, w)) in got.segments.iter().zip(want).enumerate() {
            let w = ids(w);
            if g.ids != w {
                let first = g.ids.iter().zip(&w).position(|(a, b)| a != b);
                panic!(
                    "{name}: segment {i} differs from the reference (first diff at {first:?}; \
                     len {} vs {})",
                    g.ids.len(),
                    w.len()
                );
            }
        }
    }

    #[test]
    fn marker_and_special_ids_match_the_reference() {
        let fx = fixture();
        let m = &fx["markers"];
        assert_eq!(ids(&m["start_of_segment"]), START_OF_SEGMENT);
        assert_eq!(m["soa"].as_u64(), Some(SOA as u64));
        assert_eq!(m["eoa"].as_u64(), Some(EOA as u64));
        assert_eq!(ids(&m["xcodec_sep"]), [XCODEC_SEP]);
    }

    /// AC2: first-segment, later-segment and ICL-wrapped (single- and dual-track) blocks equal the
    /// Python reference token for token, for English, Chinese, Japanese/Korean and the section-split
    /// edge cases — the tokenizer replayed from the reference so this runs without weights.
    #[test]
    fn prompt_blocks_match_the_reference_prompt_builder() {
        let fx = fixture();
        let cases = fx["prompts"].as_array().unwrap();
        assert!(cases.iter().any(|c| !c["icl_ids"].is_null()));
        for case in cases {
            let builder = YuePromptBuilder::new(ReplayEncoder(replay_table(&fx, case))).unwrap();
            assert_case_matches(&builder, case);
        }
    }

    /// AC3: smart-context shortening equals the reference `shorten_input` for the untouched,
    /// block-drop (one and two blocks) and tail-truncation fallback cases.
    #[test]
    fn shorten_context_matches_the_reference() {
        let fx = fixture();
        let cases = fx["shorten"].as_array().unwrap();
        let names: Vec<&str> = cases.iter().map(|c| c["name"].as_str().unwrap()).collect();
        for want in ["drop_one_block", "drop_two_blocks", "fallback_two_markers"] {
            assert!(names.contains(&want), "fixture lacks {want}");
        }
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let max_context = case["max_context"].as_u64().unwrap() as usize;
            let got = shorten_context(&ids(&case["input"]), max_context);
            assert_eq!(got, ids(&case["output"]), "{name}");
            assert!(got.len() <= max_context, "{name}: result fits");
        }
    }

    #[test]
    fn lyrics_without_a_section_label_are_refused() {
        let builder = YuePromptBuilder::new(ReplayEncoder(
            [
                "[start_of_segment]",
                "[end_of_segment]",
                "[start_of_reference]",
                "[end_of_reference]",
            ]
            .into_iter()
            .map(|t| (t.to_string(), vec![1]))
            .collect(),
        ))
        .unwrap();
        let err = builder
            .build(&PromptInput {
                genres: "pop",
                lyrics: "just words, no [verse 1] labels",
                icl: None,
            })
            .unwrap_err();
        assert!(err.to_string().contains("no `[section]` label"), "{err}");
    }

    #[test]
    fn python_whitespace_and_word_classes() {
        assert_eq!(py_strip("\u{1c} a \u{3000}\u{85}"), "a");
        assert_eq!(read_like_file(" a\r\nb\rc \n"), "a\nb\nc");
        // `\w` admits letters, numbers (incl. CJK and other-number) and `_`, not combining marks.
        let s = split_lyrics("[副歌_2]x[a\u{0301}]y[②]z");
        let labels: Vec<&str> = s.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, ["副歌_2", "②"]);
    }

    /// A tiny word-level `tokenizer.json` with the mm special tokens at their ids — enough for the
    /// production loader to load and build without the 5 MB real tokenizer.
    pub(crate) fn write_test_tokenizer(dir: &Path) -> PathBuf {
        let special = |id: u32, content: &str| {
            format!(
                r#"{{"id":{id},"content":"{content}","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}"#
            )
        };
        let json = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{},{},{}],"normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{"<unk>":0,"[":1,"]":2,"start_of_segment":3,"end_of_segment":4,"start_of_reference":5,"end_of_reference":6,"verse":7,"chorus":8}},"unk_token":"<unk>"}}}}"#,
            special(SOA, "<SOA>"),
            special(EOA, "<EOA>"),
            special(XCODEC_SEP, "<xcodec>"),
        );
        let path = dir.join(TOKENIZER_FILE);
        std::fs::write(&path, json).unwrap();
        path
    }

    fn assets_at(stage1: &Path) -> Assets {
        Assets {
            stage1: stage1.to_path_buf(),
            stage2: "/staged/yue-s2".into(),
            xcodec: "/staged/xcodec".into(),
        }
    }

    /// The production loader (no longer `Unsupported`) reads the stage-1 snapshot's
    /// `tokenizer.json` and builds reference-layout blocks.
    #[test]
    fn production_loader_builds_from_the_snapshot_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        write_test_tokenizer(dir.path());
        let tok = load(&assets_at(dir.path())).unwrap();
        let p = tok
            .build(&PromptInput {
                genres: "pop",
                lyrics: "[verse]\nla la\n[chorus]\nhey",
                icl: None,
            })
            .unwrap();
        assert_eq!(p.segments.len(), 2);
        // [end_of_segment] then [start_of_segment] open every later block.
        assert_eq!(p.segments[1].ids[..6], [1, 4, 2, 1, 3, 2]);
        assert!(p
            .segments
            .iter()
            .all(|s| s.ids.ends_with(&[SOA, XCODEC_SEP])));
    }

    #[test]
    fn production_loader_names_a_missing_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        let err = load(&assets_at(dir.path()))
            .err()
            .expect("missing tokenizer");
        assert!(err.to_string().contains(TOKENIZER_FILE), "{err}");
    }

    /// The stage-1 tier directory holding the real `tokenizer.json` (any of the six variants — the
    /// tokenizer is one file), e.g. `…/yue-s1-7b-anneal-en-cot-candle/bf16`.
    fn real_stage1_dir() -> PathBuf {
        PathBuf::from(std::env::var("YUE_S1_SNAPSHOT").expect(
            "set YUE_S1_SNAPSHOT to a staged YuE stage-1 tier dir containing tokenizer.json",
        ))
    }

    /// AC1: the Rust tokenizer's ids equal the upstream Python SentencePiece `tokenize` on every
    /// fixture text (English, Chinese, Japanese, Korean, byte fallback, whitespace runs, embedded
    /// special tokens) and on every string the reference prompt builder tokenized.
    #[test]
    #[ignore = "needs the real mm tokenizer (YUE_S1_SNAPSHOT); dev box / real-weights lane"]
    fn real_tokenizer_matches_upstream_sentencepiece() {
        let fx = fixture();
        let tok = MmTokenizer::from_file(&real_stage1_dir().join(TOKENIZER_FILE)).unwrap();
        let mut checked = 0;
        for case in fx["tokenize"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            assert_eq!(tok.encode(text).unwrap(), ids(&case["ids"]), "{text:?}");
            checked += 1;
        }
        for case in fx["prompts"].as_array().unwrap() {
            for (text, want) in replay_table(&fx, case) {
                assert_eq!(tok.encode(&text).unwrap(), want, "{text:?}");
                checked += 1;
            }
        }
        let builder = YuePromptBuilder::new(tok).unwrap();
        assert_eq!(builder.start_of_segment(), START_OF_SEGMENT);
        eprintln!("{checked} texts match the upstream tokenizer");
    }

    /// AC2 end to end: the production loader over the real tokenizer reproduces every reference
    /// prompt block.
    #[test]
    #[ignore = "needs the real mm tokenizer (YUE_S1_SNAPSHOT); dev box / real-weights lane"]
    fn real_prompt_builder_matches_the_reference() {
        let fx = fixture();
        let builder = load(&assets_at(&real_stage1_dir())).unwrap();
        for case in fx["prompts"].as_array().unwrap() {
            assert_case_matches(builder.as_ref(), case);
        }
    }

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
    fn stub_treats_unlabelled_lyrics_as_one_section() {
        assert_eq!(
            stub_split_sections("just words"),
            [("lyrics".into(), "just words".into())]
        );
    }
}
