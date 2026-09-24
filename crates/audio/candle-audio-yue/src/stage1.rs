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
//! The production loader is currently the weights-free [`StubStage1`]; **sc-19380** replaces
//! [`load`] with the candle-llm `CausalLm` (Llama, GQA, KV cache, bf16/q8/q4) driven decode. The
//! stub stays as the end-to-end seam test's weights-free double.

use candle_audio::gen_core;

use crate::config::{Assets, DecodeConfig, Tier};
use crate::tokens::{codec_token, CODEBOOK_SIZE};

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

/// The stage-1 seam.
pub trait Stage1Model: Send {
    /// Start a render: reset context and seed the sampler (one RNG stream across all segments).
    fn begin_render(&mut self, seed: u64) -> gen_core::Result<()>;
    /// Append this segment's prompt block to the context (prefill).
    fn begin_segment(&mut self, segment: &SegmentStart<'_>) -> gen_core::Result<()>;
    /// Decode one token.
    fn step(&mut self) -> gen_core::Result<Stage1Step>;
    /// Close the segment (e.g. append `<EOA>` when the budget, not the model, ended it).
    fn end_segment(&mut self) -> gen_core::Result<()>;
}

/// Production loader. **Stub until sc-19380** (returns [`StubStage1`]). `tier` is `None` when the
/// caller asserted no tier (detect it from the staged snapshot).
pub fn load(assets: &Assets, tier: Option<Tier>) -> gen_core::Result<Box<dyn Stage1Model>> {
    load_stub(assets, tier)
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets, _tier: Option<Tier>) -> gen_core::Result<Box<dyn Stage1Model>> {
    Ok(Box::new(StubStage1::default()))
}

/// Frames (vocal + instrumental token pairs) the stub emits per segment before `<EOA>`.
pub const STUB_FRAMES_PER_SEGMENT: usize = 10;

/// **Stub stage 1** — replaced as the production stage by **sc-19380**. Emits
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
    fn begin_render(&mut self, seed: u64) -> gen_core::Result<()> {
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
