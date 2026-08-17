//! Mage-VAE — the one-step 128-channel / 16× codec — **owned by sc-14039**.
//!
//! Port of `_vendor/mage_flow/models/modules/mage_vae.py`.
//!
//! ## Shape
//!
//! [`LATENT_CHANNELS`] 128, and [`VAE_DOWNSAMPLE_FACTOR`] 16 — fully convolutional. The CoD
//! decoder does carry attention, but it is **local** (independent 32×32 latent tiles), not global.
//! "Local" here means *bounded*, not *tile-invariant*: those windows are anchored at the latent
//! origin, so a decode tile boundary re-cuts them. That, plus the CoD `GroupNorm(32)`s and the DiCo
//! blocks' whole-extent squeeze-and-excite pools, is why [`MageVae::decode_tiled`] runs the whole
//! latent-resolution half once and tiles only the per-latent-pixel tail (sc-19753).
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

use mlx_rs::ops::{concatenate_axis, zeros_dtype};
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

fn tiling_geometry(shape: MageVaeShape) -> mlx_gen::tiling::VaeTiling {
    mlx_gen::tiling::VaeTiling {
        spatial_scale: shape.patch,
        temporal_scale: 1,
        causal_temporal: false,
        full_res_channels: shape.hidden_x,
    }
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

    /// Spatially tiled decode for the shared bounded-decode rung. Mage's codec maps one latent
    /// pixel to 16 output pixels; the shared tiler owns partitioning, blending, cancellation, and
    /// per-tile materialization while this closure executes the unchanged per-pixel tail.
    ///
    /// **Normalization semantics (sc-19753).** Only the *per-latent-pixel* half of the codec tiles.
    /// Everything with a cross-position dependence runs once on the whole latent:
    ///
    /// - the CoD decoder ([`cod_decoder::CodDecoder`]) — its three resnets and its `norm_out` are
    ///   `GroupNorm(32)` over the full latent, and its two `AttnBlock`s attend within fixed 32×32
    ///   latent windows that a tile boundary would re-cut;
    /// - the denoiser's conditioning stream
    ///   ([`DConvDenoiser::conditioning_stream`](denoiser::DConvDenoiser::conditioning_stream)) —
    ///   each of the 21 DiCo blocks gates on a squeeze-and-excite **average over the whole spatial
    ///   extent**.
    ///
    /// Tiling the whole `decode`, as this did before, gave every crop its own GroupNorm statistics,
    /// its own attention windows and its own SE gates — a different decode, not a blend artifact.
    /// Both halves that now run whole are latent-resolution; the tiled tail is the
    /// output-resolution one (`hidden_x` values per output pixel), so the memory bound is retained.
    pub fn decode_tiled(
        &self,
        latents: &Array,
        cfg: &mlx_gen::tiling::TilingConfig,
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(mlx_gen::CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let shape = latents.shape();
        if shape.len() != 4 {
            return Err(Error::Msg(format!(
                "mage-vae tiled decode expects NCHW latents, got {shape:?}"
            )));
        }
        // `SimpleMlpAdaLn` is the widest materialized output-resolution stage: its documented
        // shape is `[B*latent_pixels, patch^2, hidden_x]`, i.e. `hidden_x` values per output pixel.
        // The 384-wide CoD path is latent-resolution and therefore must not drive this cap.
        let vae = tiling_geometry(self.shape);
        if !cfg.needs_tiling(vae, 1, shape[2], shape[3]) {
            return self.decode(latents);
        }
        if shape[1] != self.shape.bottleneck {
            return Err(Error::Msg(format!(
                "mage-vae tiled decode: expected {} latent channels, got {}",
                self.shape.bottleneck, shape[1]
            )));
        }
        let scale = self.shape.patch;

        // --- globally-scoped halves, once on the whole latent -------------------------------
        let z = latents
            .transpose_axes(&[0, 2, 3, 1])?
            .as_dtype(self.dtype)?; // NHWC
        let cond = self.cod.forward(&z)?;
        let folded = self.denoiser.is_decode_folded();
        let noise = if folded {
            None
        } else {
            Some(zeros_dtype(
                &[
                    shape[0],
                    shape[2] * scale,
                    shape[3] * scale,
                    self.shape.in_channels,
                ],
                self.dtype,
            )?)
        };
        let t = zeros_dtype(&[shape[0]], self.dtype)?;
        let s = self
            .denoiser
            .conditioning_stream(noise.as_ref(), &t, &cond)?;

        // --- per-latent-pixel tail, tiled ---------------------------------------------------
        // `s` and `cond` share the latent grid, so they are carried through the shared tiler as one
        // channel-concatenated array and split back apart inside the closure. The tail has no
        // cross-position dependence, so each crop's result is exact and the partition-of-unity
        // blend reproduces it.
        let hidden = self.shape.hidden;
        let paired = concatenate_axis(&[s, cond], -1)?;
        let ps = paired.shape();
        let lifted = paired.reshape(&[ps[0], 1, ps[1], ps[2], ps[3]])?;
        let plan = cfg.plan(vae, 1, shape[2], shape[3]);
        let decoded =
            mlx_gen::vae_tiling::tiled_decode(&lifted, &plan, [1, 2, 3], cancel, |tile| {
                let ts = tile.shape();
                let flat = tile.reshape(&[ts[0], ts[2], ts[3], ts[4]])?;
                let parts = flat.split_axis(&[hidden], -1)?;
                let (s_tile, cond_tile) = (&parts[0], &parts[1]);
                let (height, width) = (ts[2] * scale, ts[3] * scale);
                let tile_noise = if folded {
                    None
                } else {
                    Some(zeros_dtype(
                        &[ts[0], height, width, self.shape.in_channels],
                        self.dtype,
                    )?)
                };
                let out = self.denoiser.per_pixel_tail(
                    s_tile,
                    cond_tile,
                    tile_noise.as_ref(),
                    height,
                    width,
                )?; // NHWC [B, H, W, 3]
                let os = out.shape();
                Ok(out.reshape(&[os[0], 1, os[1], os[2], os[3]])?)
            })?;
        let ds = decoded.shape();
        Ok(decoded
            .reshape(&[ds[0], ds[2], ds[3], ds[4]])?
            .transpose_axes(&[0, 3, 1, 2])?) // NCHW
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

    /// Sample the edit-path posterior with a request seed.
    pub fn encode_sample(&self, image_nchw: &Array, seed: u64) -> Result<Array> {
        let moments = self.encode_moments(image_nchw)?;
        let key = mlx_rs::random::key(seed)?;
        let noise = mlx_rs::random::normal::<f32>(moments.mean.shape(), None, None, Some(&key))?
            .as_dtype(moments.mean.dtype())?;
        let std = moments
            .logvar
            .multiply(Array::from_slice(&[0.5f32], &[1]))?
            .exp()?;
        Ok(moments.mean.add(&std.multiply(&noise)?)?)
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

#[cfg(test)]
mod tiling_tests {
    use super::*;

    #[test]
    fn bounded_decode_geometry_tracks_the_architectural_pixel_stage() {
        let geometry = tiling_geometry(MageVaeShape::PUBLISHED);
        assert_eq!(geometry.spatial_scale, MageVaeShape::PUBLISHED.patch);
        assert_eq!(geometry.full_res_channels, MageVaeShape::PUBLISHED.hidden_x);
        assert_eq!(geometry.full_res_channels, 32);

        let mut mutated = MageVaeShape::PUBLISHED;
        mutated.hidden_x = 48;
        mutated.patch = 8;
        let geometry = tiling_geometry(mutated);
        assert_eq!(geometry.full_res_channels, 48);
        assert_eq!(geometry.spatial_scale, 8);
    }
}
