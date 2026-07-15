//! Mochi 1 text-to-video pipeline — the CFG denoise loop + VAE decode that turns a T5-XXL
//! conditioning + seeded latents into a `Vec<Image>`.
//!
//! (A4 phase 1 lands the compiling seam; the real denoise/decode bodies are wired in phase 2.)

use mlx_gen::{CancelFlag, Error, Image, Progress, Result};
use mlx_rs::Array;

use crate::transformer::MochiTransformer3DModel;
use crate::vae::MochiVaeDecoder;

/// N-step 1st-order FlowMatchEuler **true-CFG** denoise (Mochi is not distilled). Wired in phase 2.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    _transformer: &MochiTransformer3DModel,
    _init_latents: &Array,
    _enc: &Array,
    _enc_mask: &Array,
    _steps: usize,
    _guidance: f32,
    _shift: f32,
    _cancel: &CancelFlag,
    _on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    Err(Error::Msg("mochi denoise: wired in A4 phase 2".into()))
}

/// VAE-decode the final latents into `(F, H, W, 3)` uint8 frames. Wired in phase 2.
pub fn decode_to_frames(
    _vae: &MochiVaeDecoder,
    _latents: &Array,
    _cancel: &CancelFlag,
) -> Result<Array> {
    Err(Error::Msg("mochi decode_to_frames: wired in A4 phase 2".into()))
}

/// `(F, H, W, 3)` uint8 → one [`Image`] per frame. Wired in phase 2.
pub fn frames_to_images(_frames: &Array) -> Result<Vec<Image>> {
    Err(Error::Msg("mochi frames_to_images: wired in A4 phase 2".into()))
}
