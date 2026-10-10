//! The `Transform` contract — non-prompt image→image (restore / upscale), designed around
//! SeedVR2. See `docs/MODEL_ARCHITECTURE.md` §3.3.
//!
//! Restorers/upscalers are **not** `Generator`s: there is no prompt — the input image *is* the
//! subject. SeedVR2 is a diffusion-based single-image super-resolution model (`seed` + input +
//! target size + softness → restored image, 1-step, its own VAE+transformer, fixed text
//! embedding). Scope is image→image; a video restorer would extend this later, not now.

use crate::media::Image;
use crate::runtime::{CancelFlag, Progress};
use crate::Result;

/// A non-prompt image→image transform (super-resolution / restoration).
pub trait Transform {
    fn descriptor(&self) -> &TransformDescriptor;
    fn validate(&self, req: &TransformRequest) -> Result<()>;
    fn apply(&self, req: &TransformRequest, on_progress: &mut dyn FnMut(Progress))
        -> Result<Image>;
}

/// A transform request — `Default`-able like [`GenerationRequest`](crate::generator::GenerationRequest).
#[derive(Clone, Debug, Default)]
pub struct TransformRequest {
    pub image: Image,
    pub target: TargetSize,
    /// Diffusion restorers (SeedVR2) use this; deterministic ones ignore it.
    pub seed: Option<u64>,
    /// Model-defined restoration knob (SeedVR2 "softness", 0..1).
    pub strength: Option<f32>,
    /// SeedVR2 is 1-step; override only if the model allows it.
    pub steps: Option<u32>,
    /// How the input is sized before the transform runs (sc-25683). [`InputSizing::Budgeted`] (the
    /// default) applies the provider's declared [`TransformCapabilities::input_budget`];
    /// [`InputSizing::Original`] processes the input at its own size. A provider never switches
    /// between the two on its own.
    pub input_sizing: InputSizing,
    /// Model-defined colour correction (Iris-3B: the wavelet colour fix). `None` ⇒ the provider's
    /// declared default; a provider without the knob refuses `Some(_)`
    /// ([`TransformCapabilities::supports_color_fix`]).
    pub color_fix: Option<bool>,
    pub cancel: CancelFlag,
}

/// Whether a transform first fits its input into the provider's declared input budget
/// ([`TransformCapabilities::input_budget`]) or processes it at its original size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum InputSizing {
    /// Downscale (never upscale) the input into the provider's declared budget first. A provider
    /// whose `input_budget` is `None` has no budget, so this is its original-size path.
    #[default]
    Budgeted,
    /// Process the input at its original size (compute grows with the output area).
    Original,
}

/// A transform's declared input budget: inputs are downscaled (never upscaled) so the short side is
/// at most `short_side` and the long side at most `long_side`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputBudget {
    pub short_side: u32,
    pub long_side: u32,
}

/// How big to make the output.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum TargetSize {
    /// The provider's declared default ([`TransformCapabilities::default_scale`]) — so an omitted
    /// target is never silently a different scale than the model's own default (sc-25683).
    #[default]
    ModelDefault,
    /// ESRGAN-style factor × the min edge (SeedVR2 "2x"/"3x").
    Scale(f32),
    /// Target for `min(w, h)` (SeedVR2 `resolution: int`).
    MinEdge(u32),
    /// Explicit output resolution.
    Resolution { width: u32, height: u32 },
}

/// A transform's stable identity + advertised capabilities.
#[derive(Clone, Debug)]
pub struct TransformDescriptor {
    pub id: &'static str,
    pub family: &'static str,
    /// Tensor backend that registered this transform ("mlx" | "candle"); used by the worker's
    /// per-backend capability advertisement (sc-4906, epic 3720).
    pub backend: &'static str,
    pub capabilities: TransformCapabilities,
}

/// What target modes / knobs a transform supports.
#[derive(Clone, Debug, Default)]
pub struct TransformCapabilities {
    pub scale: bool,
    pub min_edge: bool,
    pub resolution: bool,
    pub max_scale: f32,
    /// Uses a seed (diffusion-based, e.g. SeedVR2).
    pub is_diffusion: bool,
    pub supports_strength: bool,
    pub mac_only: bool,
    /// The scale [`TargetSize::ModelDefault`] resolves to.
    pub default_scale: f32,
    /// The budget [`InputSizing::Budgeted`] applies; `None` = no budget (inputs run as given).
    pub input_budget: Option<InputBudget>,
    /// Honours [`TransformRequest::color_fix`].
    pub supports_color_fix: bool,
}
