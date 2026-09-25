//! **Seam: xcodec decode** — one track's 8-codebook grid → its 16 kHz waveform plus the quantized
//! codec embedding the Vocos upsampler consumes.
//!
//! fp16 at every tier (the approved epic-R2 carve-out). The production loader refuses (`Unsupported`) until its story lands;
//! [`StubCodec`] is the test double; **sc-19377** replaces [`load`] with the xcodec (SoundStream/SEANet +
//! RVQ) decoder from the `xcodec_mini_infer` snapshot. The stub stays as the end-to-end seam test's
//! weights-free double.

use candle_audio::candle_core::{Device, Tensor};
use candle_audio::gen_core::{self, CancelFlag};

use crate::config::Assets;
use crate::tokens::{CodecFrames, CODEBOOK_SIZE, FRAMES_PER_SECOND};

/// The codec's native output rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Waveform samples per codec frame at [`SAMPLE_RATE`].
pub const SAMPLES_PER_FRAME: usize = (SAMPLE_RATE / FRAMES_PER_SECOND) as usize;

/// One decoded track.
#[derive(Clone, Debug)]
pub struct DecodedTrack {
    /// Mono waveform at [`SAMPLE_RATE`].
    pub wave: Vec<f32>,
    /// The RVQ-decoded codec embedding, `[1, dim, frames]` — the Vocos input.
    pub embedding: Tensor,
}

/// The codec-decoder seam.
pub trait CodecDecoder: Send {
    /// Decode one track's grid. Must honor `cancel` by returning [`gen_core::Error::Canceled`].
    fn decode(&self, frames: &CodecFrames, cancel: &CancelFlag) -> gen_core::Result<DecodedTrack>;
}

/// Production loader. **Refuses until sc-19377** lands the real xcodec decoder ([`load_stub`] is
/// the test double only).
pub fn load(_assets: &Assets) -> gen_core::Result<Box<dyn CodecDecoder>> {
    Err(crate::stages::not_yet_implemented(
        "codec decoder",
        "sc-19377",
    ))
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets) -> gen_core::Result<Box<dyn CodecDecoder>> {
    Ok(Box::new(StubCodec))
}

/// Embedding width the stub reports.
const STUB_DIM: usize = 4;

/// **Stub xcodec decoder** — replaced as the production stage by **sc-19377**. Renders
/// [`SAMPLES_PER_FRAME`] samples of a code-keyed tone per frame, and an embedding holding each
/// frame's first four codes scaled to `[0, 1)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubCodec;

impl CodecDecoder for StubCodec {
    fn decode(&self, frames: &CodecFrames, cancel: &CancelFlag) -> gen_core::Result<DecodedTrack> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let n = frames.frames();
        let mut wave = Vec::with_capacity(n * SAMPLES_PER_FRAME);
        for t in 0..n {
            let key = frames.codebooks.iter().map(|row| row[t] as u64).sum();
            wave.extend(crate::stub::tone(key, SAMPLES_PER_FRAME, SAMPLE_RATE));
        }
        let mut data = Vec::with_capacity(STUB_DIM * n);
        for row in frames.codebooks.iter().take(STUB_DIM) {
            data.extend(row.iter().map(|&c| c as f32 / CODEBOOK_SIZE as f32));
        }
        let embedding = Tensor::from_vec(data, (1, STUB_DIM, n), &Device::Cpu)
            .map_err(|e| gen_core::Error::Msg(format!("stub codec embedding: {e}")))?;
        Ok(DecodedTrack { wave, embedding })
    }
}
