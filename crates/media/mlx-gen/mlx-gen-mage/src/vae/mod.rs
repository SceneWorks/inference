//! Mage-VAE — the one-step 128-channel / 16× codec — **owned by sc-14039**.
//!
//! Port of `_vendor/mage_flow/models/modules/mage_vae.py`.
//!
//! ## Shape
//!
//! [`LATENT_CHANNELS`] 128, and [`VAE_DOWNSAMPLE_FACTOR`] 16 — fully convolutional. The CoD
//! decoder does carry attention, but it is **local** (independent 32×32 latent tiles), not global.
//!
//! State-dict prefixes: [`ENCODER_PREFIX`] `student.dconv_encoder.*` (342 tensors) and
//! [`DECODER_PREFIX`] `pipeline.*` (497), verified against the published checkpoint by
//! `tests/vae_decode_real_weights.rs`. (The scaffold originally recorded the decoder prefix as
//! `dconv_denoiser`, taken from the epic description; the shipped weights use `pipeline`, and the
//! denoiser's own submodules sit directly beneath it.)
//!
//! ## Encode is a single forward at t = 0
//!
//! `z_t = zeros[B, z_ch, H/16, W/16]`, `t = zeros`, `out = dconv_encoder.forward_pred(z_t, t, x)`;
//! `mean = out[:, :128]`, `logvar = out[:, 128:256].clamp(-20, 10)` (`:598-623`). Because the adaLN
//! modulation depends only on `t` and `t` is always 0, the reference precomputes it and frees the
//! adaLN MLPs (`_replace_adaln_with_const`, `:343-374`) — worth replicating, ~37M params.
//!
//! ## Decode
//!
//! One-step `_DConvDenoiser` + a CoD pixel-diffusion decoder → RGB in `[-1, 1]` (`:626-633`).
//!
//! ## Two traps
//!
//! 1. **No latent scale or shift** ([`LATENT_SCALE_SHIFT`](crate::config::LATENT_SCALE_SHIFT) is
//!    `None`): latents feed `img_in = Linear(128 → 3072)` raw. There is no `scaling_factor` /
//!    `shift_factor` / `latents_mean` / `latents_std` anywhere in the reference or the published
//!    configs. Do not import a FLUX/SD-style constant.
//! 2. **The published `vae/config.json sample_posterior: false` is not what runs.**
//!    `ModelConfig.vae_sample_posterior` defaults to `true` (`models/mage_flow.py:35`) and
//!    `load_from_repo` never overrides it, so the edit path *samples* the posterior, seeded off the
//!    global RNG (`pipeline.py:499`). The golden therefore gates the moments (`enc_mean` /
//!    `enc_logvar`) and the port applies its own RNG. Watermark detection uses the mean.
//!
//! ## Weight loading
//!
//! There is no shared `loader.rs` (see the decision note in `lib.rs`): this component's [`load`]
//! lives here, so the concurrent text-encoder and DiT ports never touch a file this story owns.
//!
//! ## Status — decode shipped (sc-14039), encode pending (sc-14046)
//!
//! [`MageVae::decode`] is complete and gated against the CPU torch golden. The encoder
//! (`_DConvEncoder`, `student.dconv_encoder.*`) is sc-14046's; [`load`] already accepts a
//! [`VaePart`] selector and the shared building blocks it needs —
//! [`dico::DiCoCore`] / [`dico::DiCoBlock`] / [`dico::EncoderDiCoBlock`],
//! [`timestep::TimestepEmbedder`], [`layers`] — are in place, so adding it is a new
//! `encoder.rs` plus an `encoder: Option<..>` field, with no restructuring here.

pub mod cod_decoder;
pub mod denoiser;
pub mod dico;
pub mod encoder;
pub mod layers;
pub mod timestep;

use mlx_rs::ops::zeros_dtype;
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::{LATENT_CHANNELS, VAE_DOWNSAMPLE_FACTOR};
use cod_decoder::CodDecoder;
use denoiser::{DConvDenoiser, MageVaeShape};
use encoder::DConvEncoder;
pub use encoder::EncoderMoments;

// The codec's own shape and the crate-wide latent contract are two descriptions of the same
// thing; tie them together at compile time so they cannot drift apart silently.
const _: () = {
    assert!(MageVaeShape::PUBLISHED.bottleneck == LATENT_CHANNELS);
    assert!(MageVaeShape::PUBLISHED.patch == VAE_DOWNSAMPLE_FACTOR as i32);
};

/// State-dict prefix for the decoder half (`mage_vae.py:579`).
pub const DECODER_PREFIX: &str = "pipeline";
/// State-dict prefix for the encoder half (`mage_vae.py:560`) — consumed by sc-14046.
pub const ENCODER_PREFIX: &str = "student.dconv_encoder";
/// The `pipeline.*` sub-trees that belong to the discarded FLUX.2 VAE encoder side and are
/// deliberately skipped (`mage_vae.py:588`).
///
/// In the published kl0.1 checkpoint `y_embedder.encoder.*` is **111 tensors** and
/// `y_embedder.bottleneck.*` is absent — the reference skips both defensively. This port loads by
/// explicit name rather than walking the state dict, so the skip is structural;
/// `tests/vae_decode_real_weights.rs` asserts the subtrees really are there to be skipped, so the
/// rule cannot quietly become vacuous.
pub const SKIPPED_DECODER_SUBTREES: &[&str] = &["y_embedder.encoder.", "y_embedder.bottleneck."];

/// Which half of the codec to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaePart {
    /// The decode path only — `pipeline.*`. Everything sc-14039 ships.
    Decode,
    /// Both encoder and decoder, for image-conditioning and round trips.
    Both,
}

/// The Mage-VAE one-step codec.
///
/// **Not a standard VAE.** Decoding is a single deterministic forward of a diffusion denoiser at
/// `t = 0` with zero noise, conditioned on the latent through the CoD decoder — there is no
/// iterative sampling, and [no latent scale or shift](crate::config::LATENT_SCALE_SHIFT) is
/// applied on either side.
pub struct MageVae {
    encoder: Option<DConvEncoder>,
    cod: CodDecoder,
    denoiser: DConvDenoiser,
    dtype: Dtype,
    shape: MageVaeShape,
}

impl MageVae {
    /// Quantize every live VAE Linear; convolutions and normalization remain dense like `nn.quantize`.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        if let Some(encoder) = &mut self.encoder {
            encoder.quantize(bits)?;
        }
        self.denoiser.quantize(bits)?;
        let (expected, packed) = self.quantization_count();
        if packed != expected {
            return Err(Error::Msg(format!(
                "mage_vae: quantization packed {packed}/{expected} required live Linear projections"
            )));
        }
        Ok(())
    }

    pub fn quantization_count(&self) -> (usize, usize) {
        let mut count = self.denoiser.quantization_count();
        if let Some(encoder) = &self.encoder {
            let next = encoder.quantization_count();
            count = (count.0 + next.0, count.1 + next.1);
        }
        count
    }

    /// Build the decode path from a `vae/diffusion_pytorch_model.safetensors` weight map.
    ///
    /// `fold_decode` folds the 21 DiCo blocks' adaLN MLPs at `t = 0`, the fixed DCT contribution,
    /// and the decode-only zero-RGB lanes. It is `true` in production; the parity suite builds an
    /// unfolded copy to prove the paths agree.
    pub fn from_weights(w: &Weights, dtype: Dtype, fold_decode: bool) -> Result<Self> {
        Self::from_weights_with_shape(
            w,
            dtype,
            fold_decode,
            MageVaeShape::PUBLISHED,
            DECODER_PREFIX,
        )
    }

    /// As [`MageVae::from_weights`], but for an arbitrary [`MageVaeShape`] and state-dict prefix.
    /// Only the tiny-fixture parity test uses anything but the published values.
    pub fn from_weights_with_shape(
        w: &Weights,
        dtype: Dtype,
        fold_decode: bool,
        shape: MageVaeShape,
        prefix: &str,
    ) -> Result<Self> {
        let cod =
            CodDecoder::from_weights(w, &format!("{prefix}.y_embedder.decoder"), shape.attn_tile)?;
        let mut denoiser = DConvDenoiser::from_weights(w, prefix, shape)?;
        if fold_decode {
            denoiser.fold_adaln_at_zero(dtype)?;
        }
        Ok(Self {
            encoder: None,
            cod,
            denoiser,
            dtype,
            shape,
        })
    }

    /// Build both published codec halves from their shared checkpoint.
    pub fn from_weights_full(w: &Weights, dtype: Dtype) -> Result<Self> {
        let mut codec = Self::from_weights(w, dtype, true)?;
        let mut encoder = DConvEncoder::from_weights(w, ENCODER_PREFIX, dtype)?;
        encoder.fold_adaln_at_zero()?;
        codec.encoder = Some(encoder);
        Ok(codec)
    }

    /// The architectural dimensions this instance was built with.
    pub fn shape(&self) -> MageVaeShape {
        self.shape
    }

    /// Whether the adaLN MLPs have been constant-folded away.
    pub fn is_adaln_folded(&self) -> bool {
        self.denoiser.is_adaln_folded()
    }

    /// Whether the fixed DCT contribution and zero-RGB lanes are folded out of decode.
    pub fn is_decode_folded(&self) -> bool {
        self.denoiser.is_decode_folded()
    }

    /// The dtype the codec runs in.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Encode an image to its deterministic posterior mean and clamped log-variance.
    pub fn encode_moments(&self, image_nchw: &Array) -> Result<EncoderMoments> {
        self.encoder
            .as_ref()
            .ok_or_else(|| Error::Msg("mage-vae encoder was not loaded".into()))?
            .moments(image_nchw)
    }

    /// Encode an image to the posterior mean (`sample_posterior = false`).
    pub fn encode_mean(&self, image_nchw: &Array) -> Result<Array> {
        Ok(self.encode_moments(image_nchw)?.mean)
    }

    /// Test instrumentation for the real decode x-embedder activation. This deliberately calls
    /// the same builder as [`Self::decode`]; it exists so allocator regression tests can measure
    /// the folded 32-wide versus literal 99-wide boundary before a later MLP masks the saving.
    #[doc(hidden)]
    pub fn decode_conditioning_for_memory_test(&self, z_nchw: &Array) -> Result<Array> {
        let sh = z_nchw.shape();
        if sh.len() != 4 || sh[1] != self.shape.bottleneck {
            return Err(Error::Msg(format!(
                "mage-vae x-embedder probe: expected [B, {}, h, w], got {sh:?}",
                self.shape.bottleneck
            )));
        }
        let z = z_nchw.transpose_axes(&[0, 2, 3, 1])?.as_dtype(self.dtype)?;
        self.cod.forward(&z)
    }

    #[doc(hidden)]
    pub fn decode_x_embedder_input_for_memory_test(
        &self,
        cond: &Array,
        height: i32,
        width: i32,
    ) -> Result<Array> {
        let b = cond.shape()[0];
        let noise = if self.denoiser.is_decode_folded() {
            None
        } else {
            Some(zeros_dtype(
                &[b, height, width, self.shape.in_channels],
                self.dtype,
            )?)
        };
        self.denoiser
            .decode_x_embedder_input(noise.as_ref(), height, width, cond)
    }

    /// Decode a `[B, 128, h, w]` NCHW latent to `[B, 3, h·16, w·16]` NCHW RGB in `[-1, 1]`
    /// (`mage_vae.py:625-633`).
    ///
    /// The output is **not** clamped — the reference returns the raw head output, which overshoots
    /// the nominal range slightly.
    pub fn decode(&self, z_nchw: &Array) -> Result<Array> {
        let sh = z_nchw.shape();
        if sh.len() != 4 {
            return Err(Error::Msg(format!(
                "mage-vae decode: expected an NCHW (rank 4) latent, got shape {sh:?}"
            )));
        }
        if sh[1] != self.shape.bottleneck {
            return Err(Error::Msg(format!(
                "mage-vae decode: expected {} latent channels, got {}",
                self.shape.bottleneck, sh[1]
            )));
        }
        let (b, hl, wl) = (sh[0], sh[2], sh[3]);
        let scale = self.shape.patch;

        let z = z_nchw.transpose_axes(&[0, 2, 3, 1])?.as_dtype(self.dtype)?; // NHWC
        let cond = self.cod.forward(&z)?;

        let t = zeros_dtype(&[b], self.dtype)?;
        let out = if self.denoiser.is_decode_folded() {
            self.denoiser
                .forward_zero_rgb(hl * scale, wl * scale, &t, &cond)?
        } else {
            let noise = zeros_dtype(
                &[b, hl * scale, wl * scale, self.shape.in_channels],
                self.dtype,
            )?;
            self.denoiser.forward(&noise, &t, &cond)?
        };

        Ok(out.transpose_axes(&[0, 3, 1, 2])?) // NCHW
    }
}

/// Load the Mage-VAE from a model directory (the `vae/` folder of a Mage-Flow snapshot, or the
/// snapshot root).
///
/// The published checkpoint packs both halves into one file: the encoder under
/// [`ENCODER_PREFIX`] and the decoder under [`DECODER_PREFIX`], with
/// [`SKIPPED_DECODER_SUBTREES`] belonging to a discarded FLUX.2 VAE encoder. Only the requested
/// half is built.
pub fn load(dir: impl AsRef<std::path::Path>, part: VaePart, dtype: Dtype) -> Result<MageVae> {
    let dir = dir.as_ref();
    let vae_dir = if dir.join("vae").is_dir() {
        dir.join("vae")
    } else {
        dir.to_path_buf()
    };
    let mut w = Weights::from_dir(&vae_dir)?;
    w.cast_all(dtype)?;
    let model = match part {
        VaePart::Decode => MageVae::from_weights(&w, dtype, true)?,
        VaePart::Both => MageVae::from_weights_full(&w, dtype)?,
    };
    // Remove exactly the tensors constructors proved they read. Wholesale requested-prefix removal
    // would hide a missing constructor field: with access tracking, one omitted read remains below
    // and is a hard load error.
    w.remove_accessed();
    // Published VAE files also carry subtrees that belong to the discarded FLUX.2 VAE encoder.
    // They are intentionally not modules in MageVae; release those checkpoint-only handles too.
    for prefix in SKIPPED_DECODER_SUBTREES {
        w.remove_prefix(&format!("{DECODER_PREFIX}.{prefix}"));
    }
    let remaining_requested = w
        .keys()
        .filter(|key| {
            key.starts_with(DECODER_PREFIX)
                || (matches!(part, VaePart::Both) && key.starts_with(ENCODER_PREFIX))
        })
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if !remaining_requested.is_empty() {
        return Err(Error::Msg(format!(
            "mage_vae: drain left {} requested source tensors resident (first: {:?})",
            remaining_requested.len(),
            &remaining_requested[..remaining_requested.len().min(5)]
        )));
    }
    Ok(model)
}
