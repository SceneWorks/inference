//! Native Supra2-IMG text-to-image on MLX (epic sc-24212, story sc-24213).
//!
//! `SupraLabs/Supra2-IMG` — a 104.1M-parameter rectified-flow DiT conditioned on Flan-T5-Base and
//! decoded by SD VAE ft-MSE — at 256×256. The backend-neutral half (geometry, revision pins, the
//! install-time `.pt` → bundle converter, tokenization, schedule, noise, quantization) lives in
//! [`gen_core_checkpoint::supra2`] and is shared with the Candle port.
//!
//! Load contract: `LoadSpec::weights` is the converted bundle directory
//! ([`convert_pinned_checkpoint`](gen_core_checkpoint::supra2::convert_pinned_checkpoint) output);
//! `LoadSpec::components` carries `text_encoder` (the pinned `google/flan-t5-base` snapshot) and
//! `vae` (the pinned `stabilityai/sd-vae-ft-mse` snapshot). Nothing is downloaded.

pub mod dit;
pub mod memory_strategy;
pub mod model;
pub mod sampling;

pub use dit::SupraDit;
pub use model::{descriptor, load, Supra2Image, Supra2Output, Supra2Sources, TextConditioning};
pub use sampling::{Supra2EulerDriver, Supra2VelocityForward};

/// Compose the Supra2 generator and its memory surfaces into an explicit registry.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry.register_route(model::route_supra2())
}
