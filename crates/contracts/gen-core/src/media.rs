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

impl Image {
    /// Interleaved 8-bit channels per pixel — always 3 (`[R, G, B]`).
    ///
    /// A constant, not a stored field: this type's invariant *is* three channels, and the
    /// four-channel case lives in [`RgbaImage`] precisely so that no consumer has to ask at
    /// runtime whether the buffer it is about to index by `3` really has three channels. It is
    /// exposed so a channel-count-driven sink (e.g. an encoder that maps 3 → RGB8 and 4 → RGBA8)
    /// can read the same accessor off either type instead of hardcoding the stride per variant.
    pub const CHANNELS: usize = 3;

    /// Interleaved channels per pixel — see [`Image::CHANNELS`].
    pub fn channels(&self) -> usize {
        Self::CHANNELS
    }

    /// Bytes per row: `width · CHANNELS`.
    pub fn row_stride(&self) -> usize {
        self.width as usize * Self::CHANNELS
    }
}

/// An 8-bit **RGBA** image, row-major interleaved, with `pixels.len() == width * height * 4`
/// (sc-24111) — the four-channel sibling of [`Image`].
///
/// ## Why a separate type rather than an `alpha` field on [`Image`]
///
/// [`Image`]'s whole contract is `width · height · 3`, and roughly a thousand construction sites
/// and every consumer index against it. A nullable fourth plane bolted on would make "does this
/// image have alpha?" a runtime question at every one of them, whose answer is `None` at all but
/// two. A distinct type makes the four-channel case unrepresentable where it is not wanted and
/// unmissable where it is: a consumer that only understands RGB cannot be handed one by accident,
/// because it arrives on its own [`GenerationOutput`](crate::generator::GenerationOutput) variant
/// and its own [`Conditioning`](crate::generator::Conditioning) variant.
///
/// ## Alpha convention — **straight (un-premultiplied)**, clamped
///
/// `pixels` are `[R, G, B, A]` per pixel with the colour channels **not** multiplied by alpha.
/// This is exactly what diffusers' `VaeImageProcessor.postprocess` produces from a four-channel
/// decode: `(x / 2 + 0.5).clamp(0, 1)` per channel *independently*, then `(v · 255).round()` to
/// `uint8` and `PIL.Image.fromarray(..., mode="RGBA")`. Nothing premultiplies, and `A = 0` does
/// **not** imply `R = G = B = 0` — a fully transparent pixel still carries whatever colour the
/// decoder painted there. A consumer that flattens must therefore composite
/// (`out = rgb·a + bg·(1 − a)`), which is what [`to_rgb_over_white`](Self::to_rgb_over_white)
/// does; it must not simply drop the fourth byte.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    /// Interleaved `[R, G, B, A]`, `width · height · 4` bytes, row-major, **straight** alpha.
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    /// Interleaved 8-bit channels per pixel — always 4 (`[R, G, B, A]`, straight alpha).
    ///
    /// The counterpart of [`Image::CHANNELS`], so a channel-count-driven sink (an encoder mapping
    /// 3 → RGB8 and 4 → RGBA8, for instance) reads one accessor off either type rather than
    /// hardcoding a stride per output variant.
    pub const CHANNELS: usize = 4;

    /// Interleaved channels per pixel — see [`RgbaImage::CHANNELS`].
    pub fn channels(&self) -> usize {
        Self::CHANNELS
    }

    /// Bytes per row: `width · CHANNELS`.
    pub fn row_stride(&self) -> usize {
        self.width as usize * Self::CHANNELS
    }

    /// Reject an image whose buffer disagrees with its declared dimensions, or whose dimensions
    /// are zero — the [`HdrFrame::validate`] contract, for the same reason: every consumer indexes
    /// by `width`/`height` and would otherwise walk off the end of a short buffer.
    pub fn validate(&self) -> crate::Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(crate::Error::Msg(format!(
                "RgbaImage: zero dimension — {}×{} (both edges must be > 0)",
                self.width, self.height
            )));
        }
        let want = self.width as usize * self.height as usize * 4;
        if self.pixels.len() != want {
            return Err(crate::Error::Msg(format!(
                "RgbaImage: buffer length {} disagrees with {}×{} RGBA (need {want})",
                self.pixels.len(),
                self.width,
                self.height
            )));
        }
        Ok(())
    }

    /// Widen an opaque RGB [`Image`] to RGBA with `A = 255` everywhere — upstream's
    /// `img.convert("RGBA")` on an opaque source.
    pub fn from_rgb(image: &Image) -> crate::Result<Self> {
        let want = image.width as usize * image.height as usize * 3;
        if image.width == 0 || image.height == 0 || image.pixels.len() != want {
            return Err(crate::Error::Msg(format!(
                "RgbaImage::from_rgb: buffer length {} disagrees with {}×{} RGB (need {want})",
                image.pixels.len(),
                image.width,
                image.height
            )));
        }
        let mut pixels = Vec::with_capacity(want / 3 * 4);
        for rgb in image.pixels.chunks_exact(3) {
            pixels.extend_from_slice(rgb);
            pixels.push(255);
        }
        Ok(Self {
            width: image.width,
            height: image.height,
            pixels,
        })
    }

    /// Composite this straight-alpha image over an opaque **white** background, yielding the RGB
    /// [`Image`] an RGB-only consumer would show it as: `out = round(rgb·a + 255·(1 − a))` with
    /// `a = A/255`.
    ///
    /// White, not black, because that is what a viewer showing a transparent PNG on a page shows,
    /// and because the model's own reference path flattens RGBA over white before the vision tower
    /// (see the provider crates' `UPSTREAM.md`). Exactly the identity when `A = 255` everywhere.
    pub fn to_rgb_over_white(&self) -> crate::Result<Image> {
        self.validate()?;
        let mut pixels = Vec::with_capacity(self.pixels.len() / 4 * 3);
        for px in self.pixels.chunks_exact(4) {
            let a = f32::from(px[3]) / 255.0;
            for &c in &px[..3] {
                pixels.push((f32::from(c) * a + 255.0 * (1.0 - a)).round() as u8);
            }
        }
        Ok(Image {
            width: self.width,
            height: self.height,
            pixels,
        })
    }

    /// Whether every pixel is fully opaque (`A == 255`) — i.e. this image carries no transparency
    /// and [`to_rgb_over_white`](Self::to_rgb_over_white) is a lossless channel drop.
    pub fn is_opaque(&self) -> bool {
        self.pixels.chunks_exact(4).all(|px| px[3] == 255)
    }
}

/// One high-dynamic-range frame: interleaved `f32` RGB, row-major, `rgb.len() == width · height · 3`.
///
/// The HDR counterpart of [`Image`] (sc-18790). Deliberately **unbounded** — scene-linear light
/// runs past `1.0` (a specular highlight is legitimately `50.0`), which is exactly the range an
/// 8-bit [`Image`] cannot carry and the reason this type exists. Depending on the request's
/// [`HdrColorSpace`](crate::hdr::HdrColorSpace) the samples are either scene-linear light or
/// ACEScct log working codes; the colour space travels alongside the frame rather than being
/// guessed from the values.
///
/// Tensor-free like every media type here.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HdrFrame {
    pub width: u32,
    pub height: u32,
    /// Interleaved RGB, `width · height · 3` samples, row-major.
    pub rgb: Vec<f32>,
}

impl HdrFrame {
    /// Reject a frame whose buffer disagrees with its declared dimensions, or whose dimensions
    /// are zero.
    ///
    /// Every colour-math entry point calls this first: the transforms index by `width`/`height`
    /// and would otherwise panic deep inside a loop (or, worse, silently transform a partial
    /// frame) when handed an inconsistent buffer.
    pub fn validate(&self) -> crate::Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(crate::Error::Msg(format!(
                "HdrFrame: zero dimension — {}×{} (both edges must be > 0)",
                self.width, self.height
            )));
        }
        let want = self.width as usize * self.height as usize * 3;
        if self.rgb.len() != want {
            return Err(crate::Error::Msg(format!(
                "HdrFrame: buffer length {} disagrees with {}×{} RGB (need {want})",
                self.rgb.len(),
                self.width,
                self.height
            )));
        }
        Ok(())
    }
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
