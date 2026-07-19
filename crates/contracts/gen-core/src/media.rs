//! Concrete media types that cross the public `Generator`/`Transform` boundary.
//!
//! Deliberately free of any `mlx-rs` types: a consumer can use the contract without depending
//! on MLX array types. Models decode their internal MLX tensors into these at the edge.

/// An 8-bit RGB image, row-major, with `pixels.len() == width * height * 3`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Interleaved PCM audio — the audio track of a video generation (e.g. LTX-2.3), a pure audio
/// synthesis (TTS / music), or a voice conversion.
///
/// ## Optional source-separated stems (sc-12842)
///
/// A music generator that can emit source-separated stems (vocals / drums / bass / other) carries
/// them additively in [`stems`](Self::stems) **alongside** the mixed track in `samples` — the mix
/// stays the primary payload so every existing consumer (which reads `samples` / `sample_rate` /
/// `channels`) is unaffected. `stems` is empty for every model that emits only a mix, which is the
/// common case (most text-to-music models — ACE-Step 1.5 included — render a single stereo mixdown;
/// stem separation is a distinct audio-to-audio task). This field is the additive carrier so a
/// future stem-emitting model needs no further contract change; a model must never fabricate stems
/// to populate it. Tensor-free, like every media type here.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioTrack {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    /// Source-separated stems accompanying this mix, empty unless the producing model genuinely
    /// emits them (see the type docs). Each stem shares this track's `sample_rate` / `channels`.
    pub stems: Vec<AudioStem>,
}

/// One incremental slice of interleaved PCM streamed by a realtime/streaming audio
/// [`Generator`](crate::generator::Generator) during
/// [`generate_streaming`](crate::generator::Generator::generate_streaming) (sc-12846) — the
/// low-latency counterpart of the one-shot [`AudioTrack`]. A streaming provider emits an
/// `AudioChunk` as each block of audio becomes available (e.g. per block of decoded RVQ/codec
/// frames for an autoregressive TTS model), so a consumer can begin playback long before the full
/// track finishes rendering.
///
/// **The reassembly law:** concatenating the [`samples`](Self::samples) of every chunk in `index`
/// order yields exactly the [`AudioTrack::samples`] of the
/// [`GenerationOutput::Audio`](crate::generator::GenerationOutput::Audio) the same
/// streamed call returns (and, for a deterministic provider, of the one-shot
/// [`generate`](crate::generator::Generator::generate) for the same request+seed). Every chunk
/// shares the track's `sample_rate` / `channels`. This is the invariant the
/// `gen-core-testkit` streaming conformance check enforces, so a provider that buffers the whole
/// output and emits it as one terminal chunk (defeating the point of streaming) or whose chunks do
/// not reassemble to the track is a CI failure rather than a field report.
///
/// Tensor-free, like every media type here.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioChunk {
    /// Interleaved PCM for **this** increment only (not cumulative). Concatenating every chunk's
    /// samples in `index` order reconstructs the full track (see the type-level reassembly law).
    pub samples: Vec<f32>,
    /// Sample rate (Hz) — identical across every chunk of a stream and equal to the final
    /// [`AudioTrack::sample_rate`].
    pub sample_rate: u32,
    /// Channel count — identical across every chunk and equal to the final
    /// [`AudioTrack::channels`]. `samples.len()` is a whole number of frames (a multiple of
    /// `channels`).
    pub channels: u16,
    /// 0-based position of this chunk within the stream. The first chunk is `0` and the index
    /// increments by one per chunk, with no gaps.
    pub index: usize,
}

/// One named, source-separated stem accompanying an [`AudioTrack`] mix (sc-12842) — e.g.
/// `"vocals"`, `"drums"`, `"bass"`, `"other"`. Additive and tensor-free; only present when the
/// producing model genuinely separates stems.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioStem {
    /// Stem name (free-form; conventionally `"vocals"` / `"drums"` / `"bass"` / `"other"`).
    pub name: String,
    /// Interleaved PCM for this stem, at the parent [`AudioTrack`]'s `sample_rate` and `channels`.
    pub samples: Vec<f32>,
}
