//! The XLabs FLUX **IP-Adapter** provider (sc-5872, epic 5480) — reference-image (identity)
//! conditioning on FLUX.1 `dev`/`schnell`, the candle (Windows/CUDA) sibling of `mlx-gen-flux`'s XLabs
//! IP path. It composes the reused FLUX text encoders / VAE / flow-match schedule with the forked DiT
//! ([`crate::ip_dit::IpFlux`], the only FLUX DiT with an IP seam) + the XLabs adapter
//! ([`crate::ip_adapter`]) + the pooled CLIP-ViT-L image encoder ([`crate::ip_image_encoder`]).
//!
//! **Single distilled forward** (no true-CFG): FLUX is guidance/timestep-distilled, so — like the
//! candle txt2img path — each denoise step is a single DiT forward (dev embeds the guidance scalar;
//! schnell ignores it), with the XLabs IP residual injected per double block. The reference's identity
//! tokens are computed **once** (constant across the denoise) and bound into a [`FluxIpInjector`] at
//! `ip_adapter_scale`; at `scale = 0` the forked DiT is byte-identical to the stock FLUX path — the
//! no-IP arm of the validation ablation (`crate::ip_validate`).
//!
//! The provider is a plain struct driven **directly** by the worker (a bespoke reference stream, like
//! `candle_gen_sdxl::IpAdapterSdxl`), not a gen-core-registered `Generator` — the
//! registered `flux1_*` descriptors stay txt2img-only.

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::flux::sampling::{get_schedule, State};
use std::path::{Path, PathBuf};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{GenerationMemory, Image, PreviewSink, Progress};
use candle_gen::preview::PreviewHook;
use candle_gen::weights::Weights;
use candle_gen::{CandleError, Result};

use crate::flux1_load;
use crate::ip_adapter::{FluxIpAdapter, FluxIpInjector};
use crate::ip_image_encoder::FluxIpImageEncoder;
use crate::pipeline::{flow_mu, BASE_SHIFT, MAX_SHIFT};
use crate::ref_backbone::{FluxRefBackbone, FluxRefHeavy};
use crate::Variant;

/// FLUX runs at bf16.
const DTYPE: DType = DType::BF16;
/// FLUX latent channel count (the raw VAE latent / initial noise; the DiT packs it 2×2 to 64).
const LATENT_CHANNELS: usize = 16;
// FLUX dev's flow-match time-shift endpoints (`BASE_SHIFT`/`MAX_SHIFT`) are shared from
// `crate::pipeline` (sc-11249 / F-140) rather than re-declared here.

/// Default `ip_adapter_scale` for the XLabs FLUX IP-Adapter (mlx-gen-flux `DEFAULT_IP_SCALE`).
pub const DEFAULT_IP_SCALE: f32 = 0.7;

/// FLUX packs the /8 VAE latent 2×2, so both render dims must be multiples of 16 (matching the flux1
/// txt2img `validate` and the control/edit siblings). Single source of truth = the crate-root
/// [`crate::SIZE_MULTIPLE`] (sc-12612).
use crate::SIZE_MULTIPLE;

/// Paths to the FLUX IP-Adapter checkpoints.
pub struct IpAdapterFluxPaths {
    /// The black-forest-labs FLUX.1 snapshot dir (`flux1-{dev,schnell}.safetensors`, `ae.safetensors`,
    /// `text_encoder/`, `text_encoder_2/`, `tokenizer_2/`). The variant is detected from which DiT file
    /// is present.
    pub flux_base: PathBuf,
    /// The XLabs adapter (`XLabs-AI/flux-ip-adapter` `ip_adapter.safetensors`: `ip_adapter_proj_model.*`
    /// + `double_blocks.{0..18}.processor.ip_adapter_double_stream_{k,v}_proj.*`).
    pub ip_adapter: PathBuf,
    /// The CLIP ViT-L/14 image encoder (`openai/clip-vit-large-patch14`) — a dir (`model.safetensors`)
    /// or the file directly.
    pub image_encoder: PathBuf,
}

/// One FLUX IP-Adapter generation request.
#[derive(Clone)]
pub struct IpAdapterFluxRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Guidance scale — embedded by the dev DiT, inert on schnell.
    pub guidance: f32,
    /// IP-Adapter scale (the decoupled-cross-attn weight on the reference image tokens).
    pub ip_adapter_scale: f32,
    pub seed: u64,
    /// Shared FLUX memory ladder selected by worker admission.
    pub memory: GenerationMemory,
    /// Per-step latent-preview sink (epic 16948, sc-16956) — the bespoke-request twin of
    /// [`gen_core::GenerationRequest::preview`](candle_gen::gen_core::GenerationRequest::preview),
    /// which this provider cannot carry because it is worker-invoked by name rather than through a
    /// registered descriptor. Default is inert, and an inert sink makes a seeded render byte-identical
    /// to one with no preview at all.
    pub preview: PreviewSink,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for IpAdapterFluxRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 25,
            guidance: 3.5,
            ip_adapter_scale: DEFAULT_IP_SCALE,
            seed: 0,
            memory: GenerationMemory::default(),
            preview: PreviewSink::default(),
            cancel: CancelFlag::default(),
        }
    }
}

/// Resolve the CLIP image-encoder weight file from a dir-or-file path (a file is used directly; a dir
/// resolves `model.safetensors` then `model.fp16.safetensors`).
fn resolve_image_encoder(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    for name in ["model.safetensors", "model.fp16.safetensors"] {
        let p = path.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(CandleError::Msg(format!(
        "flux ip-adapter: CLIP image encoder not found under {} (expected a model.safetensors or a \
         direct .safetensors file)",
        path.display()
    )))
}

/// Detect the FLUX variant from the snapshot by which DiT checkpoint is present (dev preferred if both).
fn detect_variant(flux_base: &Path) -> Result<Variant> {
    if flux_base.join(Variant::Dev.transformer_file()).is_file() {
        Ok(Variant::Dev)
    } else if flux_base
        .join(Variant::Schnell.transformer_file())
        .is_file()
    {
        Ok(Variant::Schnell)
    } else if flux_base.join("transformer/config.json").is_file() {
        let path = flux_base.join("transformer/config.json");
        let bytes = std::fs::read(&path).map_err(|error| {
            CandleError::Msg(format!(
                "flux ip-adapter: read packed transformer config {}: {error}",
                path.display()
            ))
        })?;
        let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
            CandleError::Msg(format!(
                "flux ip-adapter: parse packed transformer config {}: {error}",
                path.display()
            ))
        })?;
        Ok(
            if config
                .get("guidance_embeds")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                Variant::Dev
            } else {
                Variant::Schnell
            },
        )
    } else {
        Err(CandleError::Msg(format!(
            "flux ip-adapter: no BFL checkpoint or packed transformer/config.json in {}",
            flux_base.display()
        )))
    }
}

/// The loaded FLUX IP-Adapter model: the reused FLUX text encoders + VAE, the forked IP DiT, the XLabs
/// adapter, and the CLIP ViT-L image encoder.
pub struct IpAdapterFlux {
    variant: Variant,
    device: Device,
    dtype: DType,
    backbone: FluxRefBackbone,
    ip_encoder: FluxIpImageEncoder,
    adapter: FluxIpAdapter,
}

impl IpAdapterFlux {
    /// Load the FLUX backbone (text encoders + forked DiT + VAE) + the XLabs adapter + the CLIP ViT-L
    /// image encoder from a FLUX snapshot, the XLabs `ip_adapter.safetensors`, and a CLIP image encoder.
    pub fn load(paths: &IpAdapterFluxPaths) -> Result<Self> {
        Self::load_with_memory(paths, GenerationMemory::default())
    }

    pub fn load_with_memory(paths: &IpAdapterFluxPaths, memory: GenerationMemory) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let dtype = DTYPE;
        let root = paths.flux_base.clone();
        let variant = detect_variant(&root)?;

        // CLIP-L + T5-XXL text encoders (shared FLUX.1 backbone load, sc-9003).
        let backbone = FluxRefBackbone::load_with_memory(&root, variant, &device, dtype, memory)?;

        // The forked FLUX DiT (the IP seam) from the root BFL checkpoint — the genuine per-provider drift
        // (IpFlux, not the stock Flux), so the wrapper choice stays here over the shared mmap.
        // XLabs adapter weights (`ip_adapter.safetensors`).
        let ipa = Weights::from_file(&paths.ip_adapter, &device, dtype).map_err(|e| {
            CandleError::Msg(format!(
                "flux ip-adapter: load adapter {:?}: {e}",
                paths.ip_adapter
            ))
        })?;
        let adapter = FluxIpAdapter::from_weights(&ipa)?;
        if adapter.num_blocks() != 19 {
            return Err(CandleError::Msg(format!(
                "flux ip-adapter: adapter has {} double-block pairs but FLUX.1 has 19 double blocks",
                adapter.num_blocks(),
            )));
        }

        // CLIP ViT-L/14 image encoder (`vision_model.*` + `visual_projection.*`).
        let enc_path = resolve_image_encoder(&paths.image_encoder)?;
        let enc_w = Weights::from_file(&enc_path, &device, dtype).map_err(|e| {
            CandleError::Msg(format!(
                "flux ip-adapter: load CLIP image encoder {enc_path:?}: {e}"
            ))
        })?;
        let ip_encoder = FluxIpImageEncoder::from_weights(&enc_w)?;

        Ok(Self {
            variant,
            device,
            dtype,
            backbone,
            ip_encoder,
            adapter,
        })
    }

    /// Reference-image T2I: condition the FLUX generation on `reference`'s CLIP-ViT-L identity tokens at
    /// `req.ip_adapter_scale` (a single distilled forward per step — no true-CFG).
    pub fn generate(
        &self,
        req: &IpAdapterFluxRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        validate_request(req)?;

        // Conditioning: text (T5 seq + CLIP pooled) and the reference image tokens (computed once).
        self.backbone
            .validate_native_vae_request(false, &req.cancel)?;
        let (t5_emb, clip_emb) = self
            .backbone
            .encode_text_with_memory(&req.prompt, &req.cancel)?;
        candle_gen::check_cancel(&req.cancel)?;
        let embeds = self
            .ip_encoder
            .image_embeds(reference)?
            .to_dtype(self.dtype)?;
        candle_gen::check_cancel(&req.cancel)?;
        let tokens = self.adapter.tokens(&embeds)?;
        let injector = FluxIpInjector::new(&self.adapter, tokens, req.ip_adapter_scale as f64);
        let heavy = self.backbone.load_heavy(&req.cancel)?;

        // candle's get_noise geometry: latent is /8 of a multiple-of-16 request. sc-3673 parity:
        // deterministic, launch-portable CPU-seeded initial noise (shared FLUX.1 helper, sc-9003).
        let lat_h = (req.height as usize).div_ceil(16) * 2;
        let lat_w = (req.width as usize).div_ceil(16) * 2;
        let noise = flux1_load::seeded_noise(
            req.seed,
            LATENT_CHANNELS,
            lat_h,
            lat_w,
            &self.device,
            self.dtype,
        )?;

        let state = State::new(&t5_emb, &clip_emb, &noise)?;
        let timesteps = if self.variant.is_dev() {
            get_schedule(req.steps, Some((state.img.dim(1)?, BASE_SHIFT, MAX_SHIFT)))
        } else {
            get_schedule(req.steps, None)
        };
        let guidance: f64 = if self.variant.supports_guidance() {
            req.guidance as f64
        } else {
            0.0
        };

        // Per-step latent preview (epic 16948, sc-16956), bound to the SAME `(width, height)` the
        // decode below is given. The XLabs reference tokens are injected inside the DiT forward and
        // never join the running latent, so the strip shows the target image alone.
        let preview = crate::preview::hook(&req.preview, req.width, req.height);
        let latents = self.denoise(
            &state,
            &timesteps,
            guidance,
            &injector,
            heavy.as_ref(),
            req.seed,
            &preview,
            &req.cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        // IP-Adapter lane does not carry a PiD decoder (base txt2img is the shipping PiD path, epic 7840
        // / sc-7853); native FLUX VAE decode.
        self.backbone.decode_with_memory(
            heavy.as_ref(),
            &latents,
            req.height as usize,
            req.width as usize,
            None,
            &req.cancel,
        )
    }

    /// The flow-match denoise with the XLabs IP injector, routed through the unified curated
    /// sampler/scheduler driver (epic 7114 P4, sc-7123) — the IP twin of the txt2img
    /// [`crate::pipeline`] `denoise`. The `scheduler` axis re-strides FLUX's native `get_schedule(..)`
    /// over the time-shift `mu`; the `sampler` axis picks the integrator. The forked
    /// [`IpFlux::forward`] (`Some(injector)`) replaces the stock FLUX forward, and the XLabs IP residual
    /// injection stays INSIDE the `predict` closure so a multi-eval solver re-runs the whole step. The
    /// DEFAULT (`euler` over the native schedule) is the N1 no-op for the legacy inline flow-match Euler
    /// loop `img += pred·(σ_{i+1} − σ_i)`. FLUX feeds the raw timestep (`Sigma` convention: `t == σ`);
    /// guidance is a per-batch tensor only embedded by the dev DiT. The provider request carries no
    /// sampler/scheduler knob (the worker drives this stream directly), so it defaults to the native
    /// flow-match Euler path. Cancellation + progress are owned by the driver.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        state: &State,
        timesteps: &[f64],
        guidance: f64,
        injector: &FluxIpInjector,
        heavy: Option<&FluxRefHeavy>,
        seed: u64,
        preview: &PreviewHook<'_>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Tensor> {
        let b_sz = state.img.dim(0)?;
        let guidance_t = Tensor::full(guidance as f32, b_sz, &self.device)?;
        // Native schedule = candle's verbatim `get_schedule(..)` (f32 descending, trailing 0.0); the
        // IP request exposes no curated knob, so this defaults to the byte-exact native flow path.
        let native: Vec<f32> = timesteps.iter().map(|&t| t as f32).collect();
        let mu = flow_mu(self.variant, state.img.dim(1)?);
        let steps = native.len().saturating_sub(1);
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, steps, &native);
        candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            state.img.clone(),
            seed,
            cancel,
            on_progress,
            Some(preview),
            |img, t| -> Result<Tensor> {
                // The forked DiT forward returns a `candle_core::Result`; `?` bridges it into the
                // driver's `CandleError`. The XLabs IP residual injection lives inside this closure.
                let t_vec = Tensor::full(t, b_sz, &self.device)?;
                self.backbone.forward_ip_with_memory(
                    heavy,
                    img,
                    &state.img_ids,
                    &state.txt,
                    &state.txt_ids,
                    &t_vec,
                    &state.vec,
                    Some(&guidance_t),
                    injector,
                    cancel,
                )
            },
        )
    }
}

/// Validate the seed-independent request knobs before any tensor work. The empty-prompt guard
/// (sc-9171, the sc-8646 bug class) mirrors the flux1 **control** provider
/// ([`crate::control_provider`]) and the registered flux1 txt2img `validate`. flux1-IP conditions on
/// **T5 + CLIP** (`encode_text`): an empty prompt reaches the T5 encoder as an all-pad sequence and
/// CLIP as bare BOS/EOS, i.e. degenerate identity-free conditioning rather than the intended
/// text-plus-reference blend. Rejecting it up front turns a silent quality collapse (or, on the CLIP
/// side, the deeper `"empty CLIP tokenization"` error in `encode_text`) into a clean, actionable
/// validation error. The provider is a single distilled forward (no true-CFG), so there is no
/// negative/uncond prompt — the exposure is solely the positive prompt.
fn validate_request(req: &IpAdapterFluxRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(CandleError::Msg(
            "flux ip-adapter: prompt is required".into(),
        ));
    }
    // Bring this lane up to parity with its three siblings (flux1 txt2img/control/edit): a
    // multiple-of-16 size floor and a `steps == 0` reject. Without them `get_schedule(0, …)` builds an
    // empty flow-match trajectory (zero sampler steps ⇒ pure seeded noise decoded and returned as
    // success), burning GPU time for garbage rather than the fast typed error the siblings give
    // (sc-9016/F-032 established the guard; sc-11182/F-102 sweeps it here).
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(CandleError::Msg(format!(
            "flux ip-adapter: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.steps == 0 {
        return Err(CandleError::Msg(
            "flux ip-adapter: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The empty-prompt guard (sc-9171, the sc-8646 bug class): an empty or whitespace-only prompt is
    /// a clean validation error — never a degenerate all-pad T5 / bare CLIP encode — while a real
    /// prompt passes. flux1-IP conditions on T5+CLIP, so this is a genuinely degenerate site (unlike
    /// the CLIP-fixed-pad SDXL edit/IP or ChatGLM-framed Kolors-IP siblings audited in sc-9171).
    #[test]
    fn validate_request_rejects_empty_prompt() {
        let empty = IpAdapterFluxRequest::default();
        assert!(empty.prompt.is_empty());
        let err = validate_request(&empty).unwrap_err();
        assert!(err.to_string().contains("prompt is required"), "{err}");

        let whitespace = IpAdapterFluxRequest {
            prompt: " \t\n".into(),
            ..Default::default()
        };
        let err = validate_request(&whitespace).unwrap_err();
        assert!(err.to_string().contains("prompt is required"), "{err}");

        let ok = IpAdapterFluxRequest {
            prompt: "a portrait".into(),
            ..Default::default()
        };
        assert!(validate_request(&ok).is_ok());
    }

    /// Parity with the three flux1 siblings (sc-11182, F-102): `steps == 0` and a non-multiple-of-16
    /// size are fast typed errors (never a decoded pure-noise "success"); the defaults pass.
    #[test]
    fn validate_request_floors_steps_and_size() {
        let base = IpAdapterFluxRequest {
            prompt: "a portrait".into(),
            ..Default::default()
        };

        let zero_steps = IpAdapterFluxRequest {
            steps: 0,
            ..base.clone()
        };
        let err = validate_request(&zero_steps).unwrap_err();
        assert!(err.to_string().contains("steps must be >= 1"), "{err}");

        let bad_size = IpAdapterFluxRequest {
            width: 1000, // not a multiple of 16
            ..base.clone()
        };
        let err = validate_request(&bad_size).unwrap_err();
        assert!(err.to_string().contains("multiples of 16"), "{err}");

        assert!(validate_request(&base).is_ok());
    }

    /// The request defaults match the FLUX dev IP-Adapter knobs (1024², 25 steps, guidance 3.5, ip 0.7).
    #[test]
    fn request_defaults() {
        let r = IpAdapterFluxRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 25);
        assert_eq!(r.guidance, 3.5);
        assert_eq!(r.ip_adapter_scale, DEFAULT_IP_SCALE);
        assert!(!r.cancel.is_cancelled());
    }

    /// `resolve_image_encoder`: a directory resolves `model.safetensors`; a missing dir errors loudly;
    /// a direct file is used as-is.
    #[test]
    fn image_encoder_resolution() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        assert!(resolve_image_encoder(&dir).is_err());
        let f = dir.join("model.safetensors");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(resolve_image_encoder(&dir).unwrap(), f);
        assert_eq!(resolve_image_encoder(&f).unwrap(), f);
    }

    /// `detect_variant` keys off the DiT checkpoint filename and errors when neither is present.
    #[test]
    fn variant_detection() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        assert!(detect_variant(&dir).is_err());
        std::fs::write(dir.join(Variant::Schnell.transformer_file()), b"x").unwrap();
        assert_eq!(detect_variant(&dir).unwrap(), Variant::Schnell);
        std::fs::write(dir.join(Variant::Dev.transformer_file()), b"x").unwrap();
        assert_eq!(detect_variant(&dir).unwrap(), Variant::Dev); // dev preferred if both

        // Clear the checkpoint files so the next leg exercises the transformer/config.json
        // route rather than re-reading the filenames above. A reset, not cleanup.
        std::fs::remove_file(dir.join(Variant::Schnell.transformer_file())).unwrap();
        std::fs::remove_file(dir.join(Variant::Dev.transformer_file())).unwrap();
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        std::fs::write(
            dir.join("transformer/config.json"),
            br#"{"guidance_embeds":false}"#,
        )
        .unwrap();
        assert_eq!(detect_variant(&dir).unwrap(), Variant::Schnell);
        std::fs::write(
            dir.join("transformer/config.json"),
            br#"{"guidance_embeds":true}"#,
        )
        .unwrap();
        assert_eq!(detect_variant(&dir).unwrap(), Variant::Dev);
    }
}
