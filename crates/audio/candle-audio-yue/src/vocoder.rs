//! **Seam: Vocos upsampler** — a track's codec embedding → its 44.1 kHz waveform, through the
//! track's own decoder checkpoint (separate vocal and instrumental decoders in
//! `xcodec_mini_infer/decoders/`).
//!
//! fp16 at every tier (the approved epic-R2 carve-out). The production loader is currently the
//! weights-free [`StubVocoder`]; **sc-19378** replaces [`load`] with the two Vocos decoders. The
//! stub stays as the end-to-end seam test's weights-free double.

use candle_audio::candle_core::Tensor;
use candle_audio::gen_core::{self, CancelFlag};

use crate::config::Assets;
use crate::tokens::FRAMES_PER_SECOND;

/// The upsampled output rate (the final mix and stem rate).
pub const SAMPLE_RATE: u32 = 44_100;
/// Waveform samples per codec frame at [`SAMPLE_RATE`].
pub const SAMPLES_PER_FRAME: usize = (SAMPLE_RATE / FRAMES_PER_SECOND) as usize;

/// Which stem a decoder renders (each has its own Vocos checkpoint).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Track {
    /// The vocal stem.
    Vocals,
    /// The instrumental stem.
    Instrumental,
}

impl Track {
    /// The [`gen_core::AudioStem::name`] this track is published under.
    pub fn stem_name(self) -> &'static str {
        match self {
            Self::Vocals => "vocals",
            Self::Instrumental => "instrumental",
        }
    }
}

/// The vocoder seam (both decoders).
pub trait Vocoder: Send {
    /// Upsample one track's `[1, dim, frames]` embedding to mono [`SAMPLE_RATE`] audio. Must honor
    /// `cancel` by returning [`gen_core::Error::Canceled`].
    fn decode(
        &self,
        track: Track,
        embedding: &Tensor,
        cancel: &CancelFlag,
    ) -> gen_core::Result<Vec<f32>>;
}

/// Production loader. **Stub until sc-19378** (returns [`StubVocoder`]).
pub fn load(assets: &Assets) -> gen_core::Result<Box<dyn Vocoder>> {
    load_stub(assets)
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets) -> gen_core::Result<Box<dyn Vocoder>> {
    Ok(Box::new(StubVocoder))
}

/// **Stub Vocos** — replaced as the production stage by **sc-19378**. Renders
/// [`SAMPLES_PER_FRAME`] samples per embedding frame of a tone keyed by the frame's embedding and
/// the track.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubVocoder;

impl Vocoder for StubVocoder {
    fn decode(
        &self,
        track: Track,
        embedding: &Tensor,
        cancel: &CancelFlag,
    ) -> gen_core::Result<Vec<f32>> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let rows = embedding
            .squeeze(0)
            .and_then(|e| e.to_vec2::<f32>())
            .map_err(|e| gen_core::Error::Msg(format!("stub vocoder: {e}")))?;
        let frames = rows.first().map_or(0, Vec::len);
        let mut out = Vec::with_capacity(frames * SAMPLES_PER_FRAME);
        for t in 0..frames {
            let key = rows.iter().map(|r| (r[t] * 1024.0) as u64).sum::<u64>() + track as u64;
            out.extend(crate::stub::tone(key, SAMPLES_PER_FRAME, SAMPLE_RATE));
        }
        Ok(out)
    }
}
