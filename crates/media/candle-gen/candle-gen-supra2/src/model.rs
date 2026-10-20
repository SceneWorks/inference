//! The registered Supra2-IMG generator on Candle: load the converted bundle + pinned Flan-T5 + SD
//! VAE, then run upstream's `generate()` — tokenize/truncate/mask, encode, stored-or-fallback
//! unconditional context, CFG-or-plain Euler flow integration, VAE decode, `save_image`
//! quantization. The request/load surface is the MLX twin's (`mlx-gen-supra2`), field for field:
//! both read [`candle_gen::gen_core::supra2`].

use std::path::Path;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::supra2::{
    self as s2, image_seed, initial_noise, quantize_rgb, BundleCheck, Supra2Bundle, Supra2Config,
    Supra2Resolved, Supra2Sources, Supra2Tokenizer, Supra2Tokens, PROVIDER_ID,
    TEXT_ENCODER_COMPONENT, VAE_COMPONENT,
};
use candle_gen::gen_core::{
    self, CancelFlag, Capabilities, GenerationOutput, GenerationRequest, Generator, Image,
    LoadShape, LoadSpec, MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, Modality,
    ModelDescriptor, OffloadPolicy, PerComponentBytes, Precision, Progress, SizeFloor,
};
use candle_gen::{CandleError as Error, Result};
use candle_gen_sdxl::vae_decoder::SdxlVaeDecoder;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig;
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};

use crate::dit::SupraDit;
use crate::memory_strategy;
use crate::sampling::{self, ContextBatch};

/// The weights-free descriptor (the MLX twin's, with `backend = "candle"` and `mac_only = false`).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        // The SD VAE's 4-channel f8 latent, scaled by 0.18215 — the same space SVD's VAE consumes.
        denoiser_output_latent_space: Some(&gen_core::SVD_LATENT_SPACE),
        control_kinds: None,
        required_components: &[TEXT_ENCODER_COMPONENT, VAE_COMPONENT],
        id: PROVIDER_ID,
        family: s2::FAMILY,
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: true,
            samplers: vec![s2::SAMPLER],
            schedulers: vec![s2::SCHEDULER],
            supported_guidance_methods: vec!["cfg"],
            min_size: s2::IMAGE_SIZE,
            max_size: s2::IMAGE_SIZE,
            max_count: s2::MAX_IMAGES,
            size_floor: SizeFloor::RangeChecked,
            mac_only: false,
            ..Default::default()
        },
    }
}

/// The SD VAE ft-MSE AutoencoderKL geometry ([`gen_core::supra2`]'s pinned constants).
pub fn vae_config() -> AutoEncoderKLConfig {
    AutoEncoderKLConfig {
        block_out_channels: s2::VAE_BLOCK_OUT_CHANNELS.to_vec(),
        layers_per_block: s2::VAE_LAYERS_PER_BLOCK,
        latent_channels: Supra2Config::published().latent_channels,
        norm_num_groups: s2::VAE_NORM_GROUPS,
        use_quant_conv: true,
        use_post_quant_conv: true,
    }
}

/// Load the SD VAE decoder (f32) from an sd-vae-ft-mse snapshot. The shared SDXL decoder reads both
/// the legacy `query`/`key`/`value`/`proj_attn` mid-block attention keys this checkpoint carries and
/// diffusers' renamed `to_q`/… layout.
pub fn load_vae(sources: &Supra2Sources, device: &Device) -> Result<SdxlVaeDecoder> {
    SdxlVaeDecoder::from_file(&sources.vae_weights(), device, DType::F32, &vae_config())
}

/// Load the Flan-T5-Base encoder (12 blocks × 12 heads, f32) — candle's T5 v1.1 encoder, which reads
/// `shared.*` + `encoder.*` and never touches the decoder half of the checkpoint.
pub fn load_text_encoder(sources: &Supra2Sources, device: &Device) -> Result<T5EncoderModel> {
    let cfg_path = sources.text_encoder.join("config.json");
    let cfg: T5Config =
        serde_json::from_slice(&std::fs::read(&cfg_path).map_err(|e| {
            Error::Unsupported(format!("supra2: read {}: {e}", cfg_path.display()))
        })?)
        .map_err(|e| Error::Unsupported(format!("supra2: invalid {}: {e}", cfg_path.display())))?;
    let vb = candle_gen::mmap_var_builder(&[sources.text_encoder_weights()], DType::F32, device)?;
    Ok(T5EncoderModel::load(vb, &cfg)?)
}

/// Load the converted denoiser, verifying the exact tensor inventory.
pub fn load_denoiser(bundle: &Supra2Bundle, device: &Device) -> Result<SupraDit> {
    SupraDit::load(&bundle.denoiser_path(), &bundle.manifest.config, device)
}

/// Text conditioning for one prompt: `[1, ctx_len, ctx_dim]` f32 on the device + its 0/1 mask.
#[derive(Clone)]
pub struct TextConditioning {
    pub context: Tensor,
    pub mask: Vec<f32>,
}

/// The denoised result of one request.
pub struct Supra2Output {
    /// Final latents `[count, C, S, S]` (row-major f32, before VAE scaling).
    pub latents: Vec<f32>,
    pub images: Vec<Image>,
    /// Untruncated prompt token count and whether truncation to `ctx_len` happened.
    pub prompt_tokens: usize,
    pub truncated: bool,
    /// Whether CFG used the checkpoint-stored unconditional embedding (`false` ⇒ fallback or no CFG).
    pub used_stored_unconditional: bool,
}

fn check(cancel: &CancelFlag) -> Result<()> {
    if cancel.is_cancelled() {
        Err(Error::Canceled)
    } else {
        Ok(())
    }
}

/// Run the T5 encoder over one tokenized prompt and lay the result out as upstream's padded
/// `[1, ctx_len, ctx_dim]` context.
///
/// Upstream runs the HF encoder over all `ctx_len` positions with the attention mask. Candle's
/// T5 encoder takes no key-padding mask, so the **real** tokens (`mask == 1`, always a prefix — the
/// tokenizer right-pads) are encoded at their natural length: with every padded key masked, each
/// real token attends to exactly the same keys with the same relative-position buckets, so these
/// rows are upstream's rows. The padded rows are written as zeros. Upstream's padded rows hold
/// finite values that nothing reads: the DiT's cross-attention masks those keys with `-inf`, so
/// their weight is exactly zero (`tests/` asserts that invariance and holds the real rows to the
/// torch reference).
pub fn encode_tokens(
    t5: &mut T5EncoderModel,
    tokens: &Supra2Tokens,
    ctx_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let ctx_len = tokens.ids.len();
    let real = tokens.mask.iter().take_while(|&&m| m == 1).count();
    if real == 0 || tokens.mask[real..].iter().any(|&m| m != 0) {
        return Err(Error::Msg(
            "supra2: the token mask must be a non-empty prefix of real tokens".into(),
        ));
    }
    let ids: Vec<u32> = tokens.ids[..real].iter().map(|&i| i as u32).collect();
    let ids = Tensor::from_vec(ids, (1, real), device)?;
    let hidden = t5.forward(&ids)?.to_dtype(DType::F32)?;
    if hidden.dims() != [1, real, ctx_dim] {
        return Err(Error::Msg(format!(
            "supra2: text encoder produced {:?}, expected [1, {real}, {ctx_dim}]",
            hidden.dims()
        )));
    }
    if real == ctx_len {
        return Ok(hidden);
    }
    let pad = Tensor::zeros((1, ctx_len - real, ctx_dim), DType::F32, device)?;
    Ok(Tensor::cat(&[&hidden, &pad], 1)?)
}

/// A loaded Supra2-IMG generator (every component resident, f32).
pub struct Supra2Image {
    descriptor: ModelDescriptor,
    spec: LoadSpec,
    sources: Supra2Sources,
    device: Device,
    tokenizer: Supra2Tokenizer,
    /// `T5EncoderModel::forward` takes `&mut self`; generation is serialized through this lock.
    text_encoder: Mutex<T5EncoderModel>,
    dit: SupraDit,
    vae: SdxlVaeDecoder,
    stored_unconditional: Option<TextConditioning>,
    memory: MemoryProviderContract,
}

/// Registry load entry point, on the process-default device (`cuda:0` on a CUDA build).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(Supra2Image::load_on_device(
        spec,
        candle_gen::default_device()?,
    )?))
}

impl Supra2Image {
    /// Load every component resident (f32) on `device`. Supra2 is small (~1.5 GB of f32 weights in
    /// total), so only the resident, eager, default-precision shape is offered — as on MLX.
    pub fn load_on_device(spec: &LoadSpec, device: Device) -> Result<Self> {
        if spec.precision != Precision::Bf16
            || spec.offload_policy != OffloadPolicy::Resident
            || spec.load_shape != LoadShape::EagerMaterialization
            || spec.quantize.is_some()
            || !spec.adapters.is_empty()
            || spec.control.is_some()
            || spec.pid.is_some()
        {
            return Err(Error::Unsupported(format!(
                "{PROVIDER_ID}: only the resident f32 checkpoint load is supported (no quantization, \
                 adapters, controls, PiD, staging or deferred materialization)"
            )));
        }
        let sources = Supra2Sources::resolve(spec)?;
        let memory = memory_strategy::contract_for(&sources)?;
        let tokenizer = Supra2Tokenizer::from_dir(&sources.text_encoder)?;
        let text_encoder = load_text_encoder(&sources, &device)?;
        let dit = load_denoiser(&sources.bundle, &device)?;
        let vae = load_vae(&sources, &device)?;
        let stored_unconditional = match sources.bundle.conditioning_path() {
            Some(path) => Some(load_stored_unconditional(
                &path,
                &sources.bundle.manifest.config,
                &device,
            )?),
            None => None,
        };
        Ok(Self {
            descriptor: descriptor(),
            spec: spec.clone(),
            sources,
            device,
            tokenizer,
            text_encoder: Mutex::new(text_encoder),
            dit,
            vae,
            stored_unconditional,
            memory,
        })
    }

    pub fn config(&self) -> &Supra2Config {
        &self.sources.bundle.manifest.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn denoiser(&self) -> &SupraDit {
        &self.dit
    }

    pub fn vae(&self) -> &SdxlVaeDecoder {
        &self.vae
    }

    /// The checkpoint-stored unconditional conditioning, when the bundle has one.
    pub fn stored_unconditional(&self) -> Option<&TextConditioning> {
        self.stored_unconditional.as_ref()
    }

    /// Tokenize (truncate to `ctx_len`, pad, mask) and encode one prompt with Flan-T5.
    pub fn encode_prompt(&self, prompt: &str) -> Result<(TextConditioning, Supra2Tokens)> {
        let cfg = *self.config();
        let tokens = self.tokenizer.encode(prompt, cfg.ctx_len)?;
        let context = {
            let mut t5 = candle_gen::lock_recover(&self.text_encoder);
            encode_tokens(&mut t5, &tokens, cfg.ctx_dim, &self.device)?
        };
        Ok((
            TextConditioning {
                context,
                mask: tokens.mask.iter().map(|&m| m as f32).collect(),
            },
            tokens,
        ))
    }

    /// Unconditional conditioning: the checkpoint's stored embedding when present, else the
    /// empty-string encoding (upstream's fallback). Returns whether the stored one was used.
    pub fn unconditional(&self) -> Result<(TextConditioning, bool)> {
        match &self.stored_unconditional {
            Some(stored) => Ok((stored.clone(), true)),
            None => Ok((self.encode_prompt("")?.0, false)),
        }
    }

    /// Decode one unscaled latent `[1, C, S, S]` (the VAE applies `/ 0.18215` here, as upstream's
    /// `vae.decode(z / VAE_SCALE)`) → `[1, 3, 8S, 8S]` f32 in ~[-1, 1].
    pub fn decode_latent(&self, z: &Tensor) -> Result<Tensor> {
        self.vae.decode(&(z / s2::VAE_SCALE as f64)?)
    }

    /// Full generation. `noise` overrides the seeded initial latents (`count × C×S×S` f32) so a
    /// reference run can feed both sides the identical tensor.
    pub fn generate_with(
        &self,
        request: &GenerationRequest,
        noise: Option<&[f32]>,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Supra2Output> {
        self.validate_request(request)?;
        let r = Supra2Resolved::resolve(request)?;
        let cfg = *self.config();
        let n = r.count as usize;
        let per_image = cfg.latent_numel();
        check(&request.cancel)?;

        let (cond, tokens) = self.encode_prompt(&request.prompt)?;
        let (uncond, used_stored) = if r.use_cfg {
            let (u, stored) = self.unconditional()?;
            (Some(u), stored)
        } else {
            (None, false)
        };
        check(&request.cancel)?;

        let z0: Vec<f32> = match noise {
            Some(values) => {
                if values.len() != n * per_image {
                    return Err(Error::Msg(format!(
                        "supra2: supplied noise has {} values, expected {}",
                        values.len(),
                        n * per_image
                    )));
                }
                values.to_vec()
            }
            None => (0..r.count)
                .flat_map(|i| initial_noise(image_seed(r.seed, i), per_image))
                .collect(),
        };
        let s = cfg.latent_size;
        let z0 = Tensor::from_vec(z0, (n, cfg.latent_channels, s, s), &self.device)?;
        let (cond_ctx, cond_mask) = sampling::broadcast(&cond.context, &cond.mask, n)?;
        let uncond = uncond
            .map(|u| sampling::broadcast(&u.context, &u.mask, n))
            .transpose()?;
        let z = sampling::denoise(
            &self.dit,
            z0,
            r.steps,
            r.guidance,
            ContextBatch {
                context: &cond_ctx,
                mask: &cond_mask,
            },
            uncond
                .as_ref()
                .map(|(context, mask)| ContextBatch { context, mask }),
            &request.cancel,
            progress,
        )?;
        check(&request.cancel)?;
        let latents = z.flatten_all()?.to_vec1::<f32>()?;
        if latents.iter().any(|v| !v.is_finite()) {
            return Err(Error::Msg("supra2: non-finite latents".into()));
        }

        progress(Progress::Decoding);
        let mut images = Vec::with_capacity(n);
        for i in 0..n {
            check(&request.cancel)?;
            let decoded = self.decode_latent(&z.narrow(0, i, 1)?)?;
            let (_, _, h, w) = decoded.dims4()?;
            let hwc = decoded
                .squeeze(0)?
                .permute((1, 2, 0))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            images.push(Image {
                width: w as u32,
                height: h as u32,
                pixels: quantize_rgb(&hwc),
            });
        }
        check(&request.cancel)?;
        Ok(Supra2Output {
            latents,
            images,
            prompt_tokens: tokens.prompt_tokens,
            truncated: tokens.truncated,
            used_stored_unconditional: used_stored,
        })
    }

    fn validate_request(&self, request: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(PROVIDER_ID, request)?;
        if request
            .memory
            .is_some_and(|m| m != gen_core::GenerationMemory::default())
        {
            return Err(Error::Unsupported(format!(
                "{PROVIDER_ID}: only resident execution is implemented"
            )));
        }
        if request.use_pid {
            return Err(Error::Unsupported(format!(
                "{PROVIDER_ID}: PiD decoding is not available for the SD VAE latent space"
            )));
        }
        Supra2Resolved::resolve(request)?;
        Ok(())
    }
}

/// Read the bundle's stored `uncond_text` `[ctx_len, ctx_dim]` + `uncond_mask` `[ctx_len]`.
fn load_stored_unconditional(
    path: &Path,
    cfg: &Supra2Config,
    device: &Device,
) -> Result<TextConditioning> {
    let w = candle_gen::candle_core::safetensors::load(path, &Device::Cpu)?;
    let get = |name: &str| -> Result<Tensor> {
        w.get(name)
            .cloned()
            .ok_or_else(|| Error::Msg(format!("supra2: conditioning is missing `{name}`")))
    };
    let text = get("uncond_text")?;
    let mask = get("uncond_mask")?;
    if text.dims() != [cfg.ctx_len, cfg.ctx_dim] || mask.dims() != [cfg.ctx_len] {
        return Err(Error::Msg(format!(
            "supra2: stored conditioning has shapes {:?}/{:?}",
            text.dims(),
            mask.dims()
        )));
    }
    Ok(TextConditioning {
        context: text
            .to_dtype(DType::F32)?
            .reshape((1, cfg.ctx_len, cfg.ctx_dim))?
            .to_device(device)?,
        mask: mask.to_dtype(DType::F32)?.to_vec1::<f32>()?,
    })
}

impl Generator for Supra2Image {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        self.validate_request(request).map_err(Into::into)
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        let out = self.generate_with(request, None, on_progress)?;
        Ok(GenerationOutput::Images(out.images))
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        Some(&self.memory)
    }

    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        memory_strategy::safety(&self.spec, &self.memory, context)
    }
}

fn footprint(spec: &LoadSpec) -> gen_core::Result<PerComponentBytes> {
    // Footprint queries never re-hash the 417 MB denoiser; the load that follows does.
    let f = s2::memory_asset_facts(&Supra2Sources::resolve_checked(
        spec,
        BundleCheck::SizesOnly,
    )?)?;
    Ok(PerComponentBytes {
        text_encoder: f.conditioning_bytes,
        dit: f.transformer_bytes,
        vae: f.decoder_bytes,
    })
}

candle_gen::register_generators! {pub(crate) const REGISTRATION = descriptor => load; footprint = footprint}

pub(crate) const MEMORY: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: PROVIDER_ID,
    contract: memory_strategy::contract,
    safety_check: memory_strategy::safety,
};

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::WeightsSource;

    #[test]
    fn descriptor_mirrors_the_mlx_twin_on_candle() {
        let d = descriptor();
        assert_eq!(d.id, "supra2_img");
        assert_eq!(d.family, "supra2");
        assert_eq!(d.backend, "candle");
        assert!(!d.capabilities.mac_only);
        assert_eq!(d.required_components, ["text_encoder", "vae"]);
        assert_eq!(d.capabilities.samplers, ["euler"]);
        assert_eq!(d.capabilities.schedulers, ["supra2_uniform_flow"]);
        assert_eq!(
            (d.capabilities.min_size, d.capabilities.max_size),
            (256, 256)
        );
        assert_eq!(d.capabilities.max_count, 64);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_guidance);
    }

    #[test]
    fn off_contract_loads_are_typed_refusals() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/supra2-not-on-disk".into()));
        spec.quantize = Some(gen_core::Quant::Q4);
        match load(&spec) {
            Err(gen_core::Error::Unsupported(m)) => assert!(m.contains("resident f32"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
        let spec = LoadSpec::new(WeightsSource::Dir("/supra2-not-on-disk".into()));
        match load(&spec) {
            Err(gen_core::Error::Unsupported(m)) => assert!(m.contains("bundle"), "{m}"),
            other => panic!("expected a missing-bundle refusal, got {:?}", other.err()),
        }
    }
}
