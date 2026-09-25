//! **Seam: ICL reference encoder.** A reference clip window (single-track mix, or dual-track vocal
//! + instrumental) → the mm-vocabulary codec token block the ICL prompt embeds.
//!
//! Loaded only for ICL renders and released before stage 1 loads. The production loader refuses (`Unsupported`) until its story lands;
//! [`StubIclEncoder`] is the test double; **sc-19379** replaces [`load`] with the xcodec
//! encoder (SEANet + RVQ + HuBERT semantic branch from the `xcodec_mini_infer` snapshot). The stub
//! stays as the end-to-end seam test's weights-free double.

use candle_audio::gen_core::{self, CancelFlag};

use crate::config::{Assets, IclReference, IclTracks};
use crate::tokens::{codec_token, CODEBOOK_SIZE, FRAMES_PER_SECOND};

/// The encoded reference block: codebook-0 mm-vocabulary ids (interleaved vocal/instrumental per
/// frame for a dual-track reference).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IclPromptCodes {
    /// The token block.
    pub ids: Vec<u32>,
}

/// The ICL encoder seam.
pub trait IclEncoder: Send {
    /// Encode the reference window. Must honor `cancel` between clips/chunks by returning
    /// [`gen_core::Error::Canceled`].
    fn encode(
        &self,
        reference: &IclReference,
        cancel: &CancelFlag,
    ) -> gen_core::Result<IclPromptCodes>;
}

/// Production loader. **Refuses until sc-19379** lands the real ICL encoder ([`load_stub`] is the
/// test double only).
pub fn load(_assets: &Assets) -> gen_core::Result<Box<dyn IclEncoder>> {
    Err(crate::stages::not_yet_implemented(
        "ICL encoder",
        "sc-19379",
    ))
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets) -> gen_core::Result<Box<dyn IclEncoder>> {
    Ok(Box::new(StubIclEncoder))
}

/// **Stub ICL encoder** — replaced as the production stage by **sc-19379**. Emits one
/// deterministic codebook-0 id per 20 ms frame of the window (two, interleaved, for dual-track),
/// keyed by the clip's sample count.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubIclEncoder;

impl IclEncoder for StubIclEncoder {
    fn encode(
        &self,
        reference: &IclReference,
        cancel: &CancelFlag,
    ) -> gen_core::Result<IclPromptCodes> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let frames = ((reference.end_secs - reference.start_secs) * FRAMES_PER_SECOND as f32)
            .ceil()
            .max(1.0) as usize;
        let keys: Vec<u64> = match &reference.tracks {
            IclTracks::Single(t) => vec![t.samples.len() as u64],
            IclTracks::Dual {
                vocals,
                instrumental,
            } => vec![
                vocals.samples.len() as u64,
                instrumental.samples.len() as u64,
            ],
        };
        let mut ids = Vec::with_capacity(frames * keys.len());
        for f in 0..frames {
            for &k in &keys {
                let code = (crate::stub::hash(&[k, f as u64]) % CODEBOOK_SIZE as u64) as u32;
                ids.push(codec_token(0, code));
            }
        }
        Ok(IclPromptCodes { ids })
    }
}
