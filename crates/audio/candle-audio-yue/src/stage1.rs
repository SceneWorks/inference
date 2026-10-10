//! **Seam: stage 1** — the 7B Llama that turns the prompt into interleaved vocal/instrumental
//! codebook-0 tokens, one lyric segment at a time.
//!
//! The seam is **step-wise on purpose**: the engine owns the decode loop, so it checks the render's
//! cancel flag before every [`Stage1Model::step`] (cancellation lands within one decode step for
//! every implementation) and enforces the per-segment token budget. Sampling — CFG batch-of-2 with
//! the masked unconditional stream, the `[EOA] + [CODEC_OFFSET, STAGE1_ALLOW_MAX]` allow-list,
//! top-p, repetition penalty, the minimum-new-tokens floor, and the "smart context" that drops the
//! oldest segment block when the sequence outgrows the cache — lives behind `step`.
//!
//! The production loader ([`load`]) is [`Stage1Lm`]: the candle-llm `CausalLm` (Llama, GQA, KV
//! cache, bf16/q8/q4) driven decode (sc-19380). [`StubStage1`] stays as the end-to-end seam test's
//! weights-free double.

use candle_audio::gen_core;

use crate::config::{Assets, DecodeConfig, Tier};
use crate::tokens::{codec_token, CODEBOOK_SIZE};

mod lm;
#[cfg(test)]
mod parity;

pub use lm::Stage1Lm;

/// What the engine hands stage 1 at the start of a segment.
#[derive(Clone, Copy, Debug)]
pub struct SegmentStart<'a> {
    /// 0-based segment index.
    pub index: usize,
    /// This segment's prompt block (segment 0's includes the head).
    pub prompt: &'a [u32],
    /// The CFG scale for this segment, `None` when guidance is off.
    pub guidance_scale: Option<f32>,
    /// The render's decode configuration.
    pub decode: &'a DecodeConfig,
}

/// One decode step's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage1Step {
    /// A generated codebook-0 token (mm-vocabulary id).
    Token(u32),
    /// `<EOA>` was sampled: the segment is complete.
    EndOfAudio,
}

/// The stage-1 positions a render can occupy: every segment block the render runs (segment 0's
/// includes the head), plus each segment's `max_new_tokens` budget and the `<EOA>` that closes it
/// (sampled or forced). The whole sequence never outgrows this, and the smart context only ever
/// shortens it, so it bounds the KV cache (capped at the model context by the implementation).
/// Saturating.
pub fn render_positions<'a>(
    blocks: impl IntoIterator<Item = &'a [u32]>,
    max_new_tokens: u32,
) -> usize {
    blocks.into_iter().fold(0usize, |acc, block| {
        acc.saturating_add(block.len())
            .saturating_add(max_new_tokens as usize)
            .saturating_add(1)
    })
}

/// The stage-1 seam.
pub trait Stage1Model: Send {
    /// Start a render: reset context and seed the sampler (one RNG stream across all segments).
    /// `max_positions` is the render's sequence bound ([`render_positions`] over the segments the
    /// engine will run); an implementation sizes its KV cache to it and refuses a segment that
    /// could outgrow it.
    fn begin_render(&mut self, seed: u64, max_positions: usize) -> gen_core::Result<()>;
    /// Append this segment's prompt block to the context (prefill).
    fn begin_segment(&mut self, segment: &SegmentStart<'_>) -> gen_core::Result<()>;
    /// Decode one token.
    fn step(&mut self) -> gen_core::Result<Stage1Step>;
    /// Close the segment (e.g. append `<EOA>` when the budget, not the model, ended it).
    fn end_segment(&mut self) -> gen_core::Result<()>;
}

/// Production loader: the stage-1 LM from `assets.stage1` through candle-llm ([`Stage1Lm::load`]).
/// `tier` is `None` when the caller asserted no tier (load whatever tier is staged).
pub fn load(assets: &Assets, tier: Option<Tier>) -> gen_core::Result<Box<dyn Stage1Model>> {
    Ok(Box::new(Stage1Lm::load(&assets.stage1, tier)?))
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets, _tier: Option<Tier>) -> gen_core::Result<Box<dyn Stage1Model>> {
    Ok(Box::new(StubStage1::default()))
}

/// Frames (vocal + instrumental token pairs) the stub emits per segment before `<EOA>`.
pub const STUB_FRAMES_PER_SEGMENT: usize = 10;

/// **Stub stage 1** — the weights-free test double for [`Stage1Lm`]. Emits
/// [`STUB_FRAMES_PER_SEGMENT`] deterministic codebook-0 token pairs (keyed by seed, segment, step
/// and the prompt length) and then `<EOA>`.
#[derive(Clone, Debug, Default)]
pub struct StubStage1 {
    seed: u64,
    segment: usize,
    prompt_len: usize,
    emitted: usize,
}

impl Stage1Model for StubStage1 {
    fn begin_render(&mut self, seed: u64, _max_positions: usize) -> gen_core::Result<()> {
        *self = Self {
            seed,
            ..Self::default()
        };
        Ok(())
    }

    fn begin_segment(&mut self, segment: &SegmentStart<'_>) -> gen_core::Result<()> {
        self.segment = segment.index;
        self.prompt_len = segment.prompt.len();
        self.emitted = 0;
        Ok(())
    }

    fn step(&mut self) -> gen_core::Result<Stage1Step> {
        if self.emitted == STUB_FRAMES_PER_SEGMENT * 2 {
            return Ok(Stage1Step::EndOfAudio);
        }
        let key = crate::stub::hash(&[
            self.seed,
            self.segment as u64,
            self.prompt_len as u64,
            self.emitted as u64,
        ]);
        self.emitted += 1;
        Ok(Stage1Step::Token(codec_token(
            0,
            (key % CODEBOOK_SIZE as u64) as u32,
        )))
    }

    fn end_segment(&mut self) -> gen_core::Result<()> {
        Ok(())
    }
}
