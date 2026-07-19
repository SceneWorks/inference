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
