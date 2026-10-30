//! The reference-image resize mode a [`GenerationRequest`](crate::GenerationRequest) carries.
//! Reference normalization and the Ref2VA reference set live in `gen_core_h3::h3_reference`
//! (sc-27316).

use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// **How an image reference is resized before it is encoded** (sglang `reference_resize_mode`).
///
/// Every mode preserves the reference's own aspect ratio, never crops, and rounds each side
/// independently to the nearest multiple of `gen_core_h3::h3_reference::REFERENCE_IMAGE_MULTIPLE` (Python `round`, i.e.
/// half-to-even, floored at one multiple):
///
/// | mode | scale |
/// |---|---|
/// | [`Match`](Self::Match) | `min(1, sqrt(target_w · target_h / (w · h)))` |
/// | [`Max`](Self::Max) | `min(1, short_edge / min(w, h))` |
/// | [`Diffusers`](Self::Diffusers) | `short_edge / min(w, h)` (upscales) |
///
/// `short_edge` is the request's effective
/// [`reference_image_short_edge`](crate::GenerationRequest::reference_image_short_edge) —
/// `gen_core_h3::h3_reference::REFERENCE_IMAGE_BASE_SHORT_EDGE` unless the caller lowered it. `Match` does not read it.
///
/// Serde form: lowercase strings `"match"` / `"max"` / `"diffusers"`, upstream's own spelling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReferenceImageResizeMode {
    /// Match the reference's pixel area to the target canvas, never upscaling. The default: the
    /// policy ComfyUI ships and the ModelTC Turbo LoRA was distilled with.
    #[default]
    Match,
    /// Only downsize a reference whose short edge exceeds `short_edge`.
    Max,
    /// Always scale the short edge to `short_edge`, upscaling smaller inputs — the original
    /// diffusers / sglang behaviour.
    Diffusers,
}
impl ReferenceImageResizeMode {
    /// Every mode, in upstream's documentation order.
    pub const ALL: [Self; 3] = [Self::Match, Self::Max, Self::Diffusers];

    /// Upstream's spelling, identical to the serde form.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Max => "max",
            Self::Diffusers => "diffusers",
        }
    }
}
impl std::fmt::Display for ReferenceImageResizeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
impl std::str::FromStr for ReferenceImageResizeMode {
    type Err = Error;

    /// Case-insensitive and whitespace-trimmed, as upstream's `str(mode).strip().lower()`.
    fn from_str(s: &str) -> Result<Self> {
        let wanted = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|m| m.as_str() == wanted)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "reference_image_resize_mode must be one of diffusers, match, max, got {s:?}"
                ))
            })
    }
}
/// The **effective** resize mode of one request: the requested one, or
/// [`ReferenceImageResizeMode::Match`] when it omits one.
pub fn effective_reference_image_resize_mode(
    req: &crate::GenerationRequest,
) -> ReferenceImageResizeMode {
    req.reference_image_resize_mode.unwrap_or_default()
}
