//! **Seam: stage 2** — the 1B Llama that upsamples one track's codebook-0 codes to all
//! [`crate::tokens::NUM_CODEBOOKS`] xcodec codebooks, teacher-forced: per frame it is
//! fed the codebook-0 token and greedily decodes the seven residuals with logits sliced to
//! `[STAGE2_SLICE_MIN, STAGE2_SLICE_MAX]`, in 6 s (300-frame) chunks.
//!
//! The production loader refuses (`Unsupported`) until its story lands;
//! [`StubStage2`] is the test double; **sc-19381** replaces
//! [`load`] with the candle-llm `CausalLm` driven teacher-forced decode (bf16/q8/q4). The stub stays
//! as the end-to-end seam test's weights-free double.

use candle_audio::gen_core::{self, CancelFlag};

use crate::config::{Assets, Tier};
use crate::tokens::{CodecFrames, CODEBOOK_SIZE, NUM_CODEBOOKS};

/// The stage-2 seam.
pub trait Stage2Model: Send {
    /// Upsample one track. The returned grid's row 0 must equal `cb0` (checked by the engine).
    /// Must honor `cancel` at chunk granularity by returning [`gen_core::Error::Canceled`].
    fn upsample(&mut self, cb0: &[u32], cancel: &CancelFlag) -> gen_core::Result<CodecFrames>;
}

/// Production loader. **Refuses until sc-19381** lands the real stage-2 LM ([`load_stub`] is the
/// test double only).
pub fn load(_assets: &Assets, _tier: Option<Tier>) -> gen_core::Result<Box<dyn Stage2Model>> {
    Err(crate::stages::not_yet_implemented("stage-2 LM", "sc-19381"))
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets, _tier: Option<Tier>) -> gen_core::Result<Box<dyn Stage2Model>> {
    Ok(Box::new(StubStage2))
}

/// **Stub stage 2** — replaced as the production stage by **sc-19381**. Codebook `k` of frame `t`
/// is `(cb0[t] + 97·k) mod 1024`.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubStage2;

impl Stage2Model for StubStage2 {
    fn upsample(&mut self, cb0: &[u32], cancel: &CancelFlag) -> gen_core::Result<CodecFrames> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let codebooks = (0..NUM_CODEBOOKS as u32)
            .map(|k| cb0.iter().map(|&c| (c + 97 * k) % CODEBOOK_SIZE).collect())
            .collect();
        Ok(CodecFrames { codebooks })
    }
}
