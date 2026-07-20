//! VAE decoder assembly + `Vae::decode`. Port of `Decoder.__call__` / `VAE.decode`:
//! conv_in → mid-block → up-blocks → GroupNorm-out → SiLU → conv_out, with the scale/shift
//! that maps latents to the decoder's input range. NCHW throughout.

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Array;

use super::conv_layers::{ConvLayer, ConvNormOut};
use super::encoder::{Encoder, VaeEncoderConfig};
use super::mid_block::UNetMidBlock;
use super::up_decoder_block::UpDecoderBlock;
use mlx_gen::nn::silu;
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::vae_tiling::tiled_decode;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LatentDecoder, Result};

/// Per-up-block `(num_resnet_layers, add_upsample)`.
#[derive(Debug, Clone)]
pub struct VaeDecoderConfig {
    pub up_blocks: Vec<(usize, bool)>,
}

impl VaeDecoderConfig {
    /// The production Z-Image VAE decoder: 4 up-blocks of 3 resnets, upsampling on the first 3.
    pub fn default_z_image() -> Self {
        Self {
            up_blocks: vec![(3, true), (3, true), (3, true), (3, false)],
        }
    }
}

pub struct Decoder {
    conv_in: ConvLayer,
    mid_block: UNetMidBlock,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out: ConvNormOut,
    conv_out: ConvLayer,
}

impl Decoder {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &VaeDecoderConfig) -> Result<Self> {
        // Support an empty top-level prefix (sub-module prefixes are always non-empty).
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let up_blocks = cfg
            .up_blocks
            .iter()
            .enumerate()
            .map(|(i, &(layers, up))| {
                UpDecoderBlock::from_weights(w, &p(&format!("up_blocks.{i}")), layers, up)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: ConvLayer::from_weights(w, &p("conv_in"))?,
            mid_block: UNetMidBlock::from_weights(w, &p("mid_block"))?,
            up_blocks,
            conv_norm_out: ConvNormOut::from_weights(w, &p("conv_norm_out"))?,
            conv_out: ConvLayer::from_weights(w, &p("conv_out"))?,
        })
    }

    /// Quantize the decoder's only quantizable Linears — the mid-block attention (conv_in/out,
    /// up-blocks, and norms are conv/norm, not quantized).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.mid_block.quantize(bits)
    }

    /// `latents` NCHW → image NCHW (3 channels, spatial ×8).
    pub fn forward(&self, latents: &Array) -> Result<Array> {
        self.forward_upsample_tail(&self.forward_pre_upsample(latents)?)
    }

    /// The pre-upsample **head** at LATENT resolution: `conv_in → mid_block`. The mid-block carries a
    /// *global* per-image spatial self-attention over all H·W tokens, so a tiled decode (sc-13571) must
    /// run this ONCE on the full latent — tiling it would make each tile attend only within itself,
    /// changing the whole image. Cheap at latent resolution, so running it full adds no meaningful
    /// memory over the single-pass decode.
    pub fn forward_pre_upsample(&self, latents: &Array) -> Result<Array> {
        let h = self.conv_in.forward(latents)?;
        self.mid_block.forward(&h)
    }

    /// The **upsample tail**: `up_blocks → conv_norm_out → SiLU → conv_out`. Every op here is spatially
    /// LOCAL (resnet convs, nearest-2× + conv upsamplers, GroupNorm, head conv), so it tiles seam-free
    /// (overlap + trapezoidal blend absorbs the conv halo). This is where the ×8 decode memory spike
    /// lives, so tiling it is what bounds the peak (sc-13571 / GitHub #1658). `head` is the pre-upsample
    /// output at latent resolution.
    pub fn forward_upsample_tail(&self, head: &Array) -> Result<Array> {
        let mut h = head.clone();
        for up in &self.up_blocks {
            h = up.forward(&h)?;
        }
        h = self.conv_norm_out.forward(&h)?;
        h = silu(&h)?;
        self.conv_out.forward(&h)
    }
}

/// The Z-Image VAE. `decode` undoes the latent scale/shift then runs the decoder; `encode`
/// (img2img) runs the encoder and maps the predicted mean into latent space. The encoder is
/// optional so a decode-only `Vae` can still be built from decoder weights alone.
pub struct Vae {
    decoder: Decoder,
    encoder: Option<Encoder>,
    scaling_factor: f32,
    shift_factor: f32,
}

impl Vae {
    pub const SCALING_FACTOR: f32 = 0.3611;
    pub const SHIFT_FACTOR: f32 = 0.1159;

    pub fn from_weights(w: &Weights, prefix: &str, cfg: &VaeDecoderConfig) -> Result<Self> {
        Ok(Self {
            decoder: Decoder::from_weights(w, prefix, cfg)?,
            encoder: None,
            scaling_factor: Self::SCALING_FACTOR,
            shift_factor: Self::SHIFT_FACTOR,
        })
    }

    /// Like [`Vae::from_weights`] but with caller-supplied latent `scaling_factor` / `shift_factor`.
    /// The AutoencoderKL structure (16-ch, GroupNorm-32, the decoder's scale/shift de-norm math) is
    /// shared across diffusers 16-ch VAEs; only the two latent-normalization constants differ between
    /// families. Z-Image uses the [`Vae::SCALING_FACTOR`] / [`Vae::SHIFT_FACTOR`] defaults; SD3.5
    /// reuses this same module with its own `1.5305` / `0.0609` factors (mlx-gen-sd3, sc-7863).
    pub fn from_weights_with_factors(
        w: &Weights,
        prefix: &str,
        cfg: &VaeDecoderConfig,
        scaling_factor: f32,
        shift_factor: f32,
    ) -> Result<Self> {
        Ok(Self {
            decoder: Decoder::from_weights(w, prefix, cfg)?,
            encoder: None,
            scaling_factor,
            shift_factor,
        })
    }

    /// The latent `scaling_factor` this VAE de-normalizes with (`decode`: `z/scale + shift`).
    pub fn scaling_factor(&self) -> f32 {
        self.scaling_factor
    }

    /// The latent `shift_factor` this VAE de-normalizes with.
    pub fn shift_factor(&self) -> f32 {
        self.shift_factor
    }

    /// Attach the img2img encoder, loaded from `prefix` (the diffusers `encoder.*` tree, remapped
    /// to the crate's internal naming by [`crate::loader::remap_vae_encoder`]).
    pub fn with_encoder(
        mut self,
        w: &Weights,
        prefix: &str,
        cfg: &VaeEncoderConfig,
    ) -> Result<Self> {
        self.encoder = Some(Encoder::from_weights(w, prefix, cfg)?);
        Ok(self)
    }

    /// Quantize the VAE's quantizable Linears (the decoder's — and, if loaded, the encoder's —
    /// mid-block spatial attention) to Q4/Q8. The VAE is otherwise all conv, so this is the full
    /// set the fork's `nn.quantize(vae, …)` hits. Output is pixel-unchanged in practice (the VAE
    /// quant is measurably 0% px on the decode), so this is for memory/`nn.quantize` faithfulness.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.decoder.quantize(bits)?;
        if let Some(encoder) = self.encoder.as_mut() {
            encoder.quantize(bits)?;
        }
        Ok(())
    }

    /// Image NCHW `[1,3,H,W]` (or `[1,3,1,H,W]`) → latent `[1,16,H/8,W/8]`. Port of the fork's
    /// `VAE.encode` composed with `VAEUtil.encode`'s 5-D→4-D fixup: run the encoder, take the
    /// distribution **mean** (first half of the channels), then map to latent space as
    /// `(mean - shift) * scaling`.
    pub fn encode(&self, image: &Array) -> Result<Array> {
        let encoder = self.encoder.as_ref().ok_or_else(|| {
            Error::Msg("z_image VAE encoder not loaded (img2img unavailable)".into())
        })?;
        let sh = image.shape();
        let image4 = if sh.len() == 5 {
            image.reshape(&[sh[0], sh[1], sh[3], sh[4]])?
        } else {
            image.clone()
        };
        let h = encoder.forward(&image4)?; // [1, 2C, H/8, W/8]
        if h.shape()[1] % 2 != 0 {
            return Err(Error::Msg(format!(
                "z-image vae encode: expected an even (2C: mean|logvar) channel count, got {}",
                h.shape()[1]
            )));
        }
        let c = h.shape()[1] / 2;
        let idx = Array::from_slice(&(0..c).collect::<Vec<i32>>(), &[c]);
        let mean = h.take_axis(&idx, 1)?; // first C channels
        Ok(multiply(
            &subtract(&mean, Array::from_slice(&[self.shift_factor], &[1]))?,
            Array::from_slice(&[self.scaling_factor], &[1]),
        )?)
    }

    /// `latents`: `(B, C, F, H, W)` (F squeezed) or `(B, C, H, W)` → image `(B, 3, 1, H·8, W·8)`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let sh = latents.shape();
        let latents4 = if sh.len() == 5 {
            // squeeze the (size-1) frame axis: (B,C,1,H,W) -> (B,C,H,W)
            latents.reshape(&[sh[0], sh[1], sh[3], sh[4]])?
        } else {
            latents.clone()
        };
        let scaled = add(
            &multiply(
                &latents4,
                Array::from_slice(&[1.0 / self.scaling_factor], &[1]),
            )?,
            Array::from_slice(&[self.shift_factor], &[1]),
        )?;
        let decoded = self.decoder.forward(&scaled)?;
        let d = decoded.shape();
        Ok(decoded.reshape(&[d[0], d[1], 1, d[2], d[3]])?) // restore frame axis
    }

    /// The decode **head** at LATENT resolution: squeeze the frame axis, denormalize (`z/scale + shift`),
    /// then run the decoder's [`Decoder::forward_pre_upsample`] (`conv_in → mid-block global attention`).
    /// A tiled decode runs this ONCE on the full latent (sc-13571). Output is 4-D NCHW at latent res.
    pub fn decode_pre_upsample(&self, latents: &Array) -> Result<Array> {
        let sh = latents.shape();
        let latents4 = if sh.len() == 5 {
            latents.reshape(&[sh[0], sh[1], sh[3], sh[4]])?
        } else {
            latents.clone()
        };
        let scaled = add(
            &multiply(
                &latents4,
                Array::from_slice(&[1.0 / self.scaling_factor], &[1]),
            )?,
            Array::from_slice(&[self.shift_factor], &[1]),
        )?;
        self.decoder.forward_pre_upsample(&scaled)
    }

    /// **Tiled** decode (sc-13571, GitHub #1658) for a memory-bounded large-image decode. The single-pass
    /// [`decode`](Self::decode) materializes the whole ×8 output transient in one shot (~14 GiB at 1024²),
    /// which OOMs / corrupts to a flat image on an 8 GB Mac. Here the denormalize + pre-upsample head
    /// (global mid-block attention) run ONCE on the full latent, then only the spatially-local upsample
    /// tail — where the ×8 spike lives — is split into overlapping tiles and trapezoidally blended, so the
    /// decode peak scales with the tile size, not the full resolution. Reuses the shared sc-11747 facility
    /// ([`tiled_decode`]) and the [`VaeTiling::QWEN_IMAGE`] geometry (this VAE is also spatial ×8 /
    /// singleton-temporal). Falls back to the single-pass [`decode`](Self::decode) when `cfg` doesn't fire
    /// for these dims (small image / large-memory machine → zero tiling overhead, exact output). No clamp
    /// here (matching [`decode`](Self::decode); the `[-1,1]` clamp is applied later by `decoded_to_image`),
    /// so the tiled output matches the untiled one to within the blend tolerance.
    ///
    /// Shared verbatim by every crate in the Flux1 / Z-Image latent space (Z-Image, FLUX.1, Boogu,
    /// Chroma), so all of them gain the memory-bounded decode.
    pub fn decode_tiled(
        &self,
        latents: &Array,
        cfg: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let sh = latents.shape();
        let (h, w) = if sh.len() == 5 {
            (sh[3], sh[4])
        } else {
            (sh[2], sh[3])
        };
        let f = 1; // still-image VAE: singleton temporal axis
        if !cfg.needs_tiling(VaeTiling::QWEN_IMAGE, f, h, w) {
            return self.decode(latents);
        }
        // Head runs ONCE on the full latent (denormalize → conv_in → mid-block global attention),
        // identical to single-pass `decode` up to the up-blocks, so parity is exact here.
        let head4 = self.decode_pre_upsample(latents)?; // 4-D NCHW at latent res
        let hs = head4.shape();
        // Lift to 5-D NCTHW (singleton T) so the shared NCTHW `tiled_decode` (axes [2,3,4]) can slice it.
        let head5 = head4.reshape(&[hs[0], hs[1], 1, hs[2], hs[3]])?;
        let plan = cfg.plan(VaeTiling::QWEN_IMAGE, f, h, w);
        tiled_decode(&head5, &plan, [2, 3, 4], cancel, |tile5| {
            // tile5 [B,C,1,h,w] → squeeze T → local upsample tail (4-D) → restore T → [B,3,1,h·8,w·8].
            let ts = tile5.shape();
            let tile4 = tile5.reshape(&[ts[0], ts[1], ts[3], ts[4]])?;
            let out4 = self.decoder.forward_upsample_tail(&tile4)?;
            let os = out4.shape();
            Ok(out4.reshape(&[os[0], os[1], 1, os[2], os[3]])?)
        })
    }

    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }
}

/// The native decoder for the Flux1 / Z-Image latent space (the behavior-preserving default of the PiD
/// decode seam, sc-7844). This `Vae` is reused verbatim by every crate in that latent space — Z-Image,
/// FLUX.1, Boogu, Chroma — so this single impl makes all of them PiD-swappable (sc-7846). Delegates to
/// the inherent [`Vae::decode`], which accepts the 4-D `(B, C, H, W)` normalized latent the engines
/// hand the seam (it re-adds the singleton frame axis on output). A PiD decoder for this same latent
/// space (`mlx-gen-pid`, sc-7843) implements the same trait, so a generation can swap between them at
/// the decode call site.
impl LatentDecoder for Vae {
    fn decode(&self, latents: &Array) -> Result<Array> {
        Vae::decode(self, latents)
    }
}
