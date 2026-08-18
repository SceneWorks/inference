//! The latent→pixel decode seam (epic 7840, sc-7844 / candle dup sc-7853).
//!
//! Every candle image engine ends sampling with `vae.decode(latent)` called inline. To let a single
//! generation optionally route that final step through NVIDIA **PiD** — a pixel-diffusion decoder
//! that decodes *and* super-resolves in one pass — instead of the native VAE, without N bespoke
//! per-engine swaps, the decode step is expressed against this one trait. `candle-gen-pid` implements
//! it for PiD (the `PidDecoder`); a provider passes `Some(&pid)` at its decode call site when the
//! per-generation `req.use_pid` toggle is set, and the native VAE decode otherwise (the byte-exact
//! default). This is the candle twin of `mlx_gen::decoder::LatentDecoder` — a core-crate trait, NOT a
//! gen-core contract type (the gen-core seam is the `LoadSpec::pid` / `GenerationRequest::use_pid`
//! fields), so it lives here rather than requiring a gen-core pin bump.

use candle_core::Tensor;
use gen_core::tiling::TilingConfig;
use gen_core::CancelFlag;
use gen_core::LatentSpace;

use crate::{CandleError, Result};

/// Decodes a model's final latent into a decoded image tensor — the input a provider's
/// tensor→[`gen_core::Image`] step then turns into an image.
///
/// Contract:
/// - The input is the engine's descriptor-declared native or patch-packed **normalized sampler
///   latent** (e.g. Qwen/FLUX 16-ch `[1, C, H/8, W/8]`, SDXL 4-ch, FLUX.2 packed 128-ch). The caller
///   must preserve that declared layout; the implementor owns any de-normalization required by its
///   native VAE or PiD student. Each implementor is tied to one latent space.
/// - The output is a decoded tensor ready for the provider's image conversion. Implementors may
///   preserve their established compute dtype; callers that require `f32` must cast explicitly.
/// - The output's spatial size **may exceed** the VAE-native size: PiD decodes and super-resolves in
///   a single pass. Callers must read dimensions from the returned tensor, never assume
///   `latent · spatial_scale`.
pub trait LatentDecoder {
    /// Exact latent tensor this decoder accepts. `None` means unknown and is incompatible with every
    /// producer; implementations must not infer a space from tensor rank or channel count.
    fn input_latent_space(&self) -> Option<&LatentSpace>;

    /// Decode `latents` to a decoded image tensor.
    fn decode(&self, latents: &Tensor) -> Result<Tensor>;

    /// Decode with a caller-selected VAE tile policy. Native VAEs that support bounded decode
    /// override this method; decoders with their own policy inherit the forwarding default. The
    /// default still observes a pre-tripped cancellation flag before entering a decoder whose legacy
    /// [`Self::decode`] method has no cancellation parameter.
    fn decode_tiled(
        &self,
        latents: &Tensor,
        _tiling: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(CandleError::Canceled);
        }
        self.decode(latents)
    }
}

/// Require full latent-space compatibility before selecting an alternate decoder. Missing
/// descriptors and learned per-channel normalization without a content hash fail closed through
/// [`gen_core::latent_spaces_compatible`].
pub fn ensure_decoder_compatible(
    denoiser_output: Option<&LatentSpace>,
    decoder: &dyn LatentDecoder,
) -> Result<()> {
    if !gen_core::latent_spaces_compatible(denoiser_output, decoder.input_latent_space()) {
        return Err(CandleError::Msg(format!(
            "decoder route rejected: denoiser latent {denoiser_output:?} is not compatible with decoder input {:?}",
            decoder.input_latent_space()
        )));
    }
    Ok(())
}

/// Require the caller's declared tensor structure to match the selected decoder before routing an
/// already-normalized latent across the seam. This checks channels, compression, packed layout, and
/// temporal law, and rejects either malformed descriptor; it deliberately does **not** claim
/// normalization compatibility.
///
/// Use [`gen_core::latent_spaces_compatible`] for alternate-decoder selection. That stronger helper
/// intentionally rejects `LearnedPerChannel` normalization without a content hash, even when the
/// family identity matches. This structural guard exists for an already-selected, provider-owned
/// route such as FLUX.2 PiD, where it prevents the packed-vs-unpacked contract violation without
/// weakening that fail-closed compatibility policy.
pub fn ensure_decoder_layout(
    denoiser_output: Option<&LatentSpace>,
    decoder: &dyn LatentDecoder,
) -> Result<()> {
    let output = denoiser_output.ok_or_else(|| {
        CandleError::Msg(
            "decoder route rejected: denoiser output latent space is undeclared".into(),
        )
    })?;
    let input = decoder.input_latent_space().ok_or_else(|| {
        CandleError::Msg("decoder route rejected: decoder input latent space is undeclared".into())
    })?;
    if !output.is_well_formed() || !input.is_well_formed() {
        return Err(CandleError::Msg(format!(
            "decoder route rejected: malformed denoiser or decoder latent layout ({output:?} -> {input:?})"
        )));
    }
    if output.channels != input.channels
        || output.spatial_compression != input.spatial_compression
        || output.patch_layout != input.patch_layout
        || output.temporal_law != input.temporal_law
    {
        return Err(CandleError::Msg(format!(
            "decoder route rejected: denoiser latent layout {output:?} does not match decoder input layout {input:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    struct ForwardingDecoder;

    impl LatentDecoder for ForwardingDecoder {
        fn input_latent_space(&self) -> Option<&LatentSpace> {
            Some(&gen_core::QWEN_KREA_Z16_LATENT_SPACE)
        }

        fn decode(&self, latents: &Tensor) -> Result<Tensor> {
            Ok(latents.clone())
        }
    }

    struct PackedDecoder;

    impl LatentDecoder for PackedDecoder {
        fn input_latent_space(&self) -> Option<&LatentSpace> {
            Some(&gen_core::FLUX2_PACKED_LATENT_SPACE)
        }

        fn decode(&self, latents: &Tensor) -> Result<Tensor> {
            Ok(latents.clone())
        }
    }

    struct DescriptorDecoder(LatentSpace);

    impl LatentDecoder for DescriptorDecoder {
        fn input_latent_space(&self) -> Option<&LatentSpace> {
            Some(&self.0)
        }

        fn decode(&self, latents: &Tensor) -> Result<Tensor> {
            Ok(latents.clone())
        }
    }

    #[test]
    fn tiled_default_forwards_exactly_and_preserves_cancellation() {
        let decoder = ForwardingDecoder;
        assert!(
            ensure_decoder_compatible(Some(&gen_core::QWEN_KREA_Z16_LATENT_SPACE), &decoder)
                .is_ok()
        );
        let input = Tensor::from_slice(&[1.0f32, 2.0], (1, 2), &Device::Cpu).unwrap();
        let cfg = TilingConfig::spatial_only(512, 64);
        assert_eq!(
            decoder
                .decode_tiled(&input, &cfg, None)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            input.to_vec2::<f32>().unwrap()
        );

        let cancel = CancelFlag::default();
        cancel.cancel();
        assert!(matches!(
            decoder.decode_tiled(&input, &cfg, Some(&cancel)),
            Err(CandleError::Canceled)
        ));
    }

    #[test]
    fn packed_route_requires_the_declared_structural_layout() {
        let decoder = PackedDecoder;
        assert!(ensure_decoder_layout(None, &decoder).is_err());
        assert!(
            ensure_decoder_layout(Some(&gen_core::FLUX2_PACKED_LATENT_SPACE), &decoder).is_ok()
        );
        assert!(
            ensure_decoder_compatible(Some(&gen_core::FLUX2_PACKED_LATENT_SPACE), &decoder)
                .is_err()
        );
        let mut unpacked = gen_core::FLUX2_PACKED_LATENT_SPACE;
        unpacked.patch_layout = gen_core::LatentPatchLayout::Unpacked;
        assert!(ensure_decoder_layout(Some(&unpacked), &decoder).is_err());
        assert!(
            !gen_core::latent_spaces_compatible(
                Some(&gen_core::FLUX2_PACKED_LATENT_SPACE),
                decoder.input_latent_space(),
            ),
            "the structural route guard must not weaken learned-normalization compatibility"
        );
        assert!(
            ensure_decoder_layout(Some(&gen_core::QWEN_KREA_Z16_LATENT_SPACE), &decoder).is_err()
        );

        let mut malformed = gen_core::FLUX2_PACKED_LATENT_SPACE;
        malformed.channels = 0;
        let malformed_decoder = DescriptorDecoder(malformed);
        assert!(ensure_decoder_layout(Some(&malformed), &malformed_decoder).is_err());
        assert!(ensure_decoder_layout(
            Some(&gen_core::FLUX2_PACKED_LATENT_SPACE),
            &malformed_decoder,
        )
        .is_err());
    }
}
