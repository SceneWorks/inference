//! The registered Supra2-IMG generator: load the converted bundle + pinned Flan-T5 + SD VAE, then run
//! upstream's `generate()` — tokenize/truncate/mask, encode, stored-or-fallback unconditional
//! context, CFG-or-plain Euler flow integration, VAE decode, `save_image` quantization.

use std::path::Path;

use crate::sampling::{self, ContextBatch};
use mlx_gen::gen_core::supra2::{
    self as s2, image_seed, initial_noise, quantize_rgb, BundleCheck, Supra2Bundle, Supra2Config,
    Supra2Resolved, Supra2Tokenizer, PROVIDER_ID, TEXT_ENCODER_COMPONENT, VAE_COMPONENT,
};
use mlx_gen::gen_core::{
    self, LoadShape, MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, Modality,
    PerComponentBytes, SizeFloor,
};
use mlx_gen::weights::Weights;
use mlx_gen::{
    CancelFlag, Capabilities, Error, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, ModelDescriptor, OffloadPolicy, Precision, Progress, Result,
};
use mlx_gen_flux::text_encoder::{T5Geometry, T5TextEncoder};
use mlx_gen_sdxl::config::VaeConfig;
use mlx_gen_sdxl::vae::Autoencoder;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::dit::SupraDit;
use crate::memory_strategy;

/// The weights-free descriptor.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        // The SD VAE's 4-channel f8 latent, scaled by 0.18215 — the same space SVD's VAE consumes.
        denoiser_output_latent_space: Some(&gen_core::SVD_LATENT_SPACE),
        control_kinds: None,
        required_components: &[TEXT_ENCODER_COMPONENT, VAE_COMPONENT],
        id: PROVIDER_ID,
        family: s2::FAMILY,
        backend: "mlx",
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
            mac_only: true,
            ..Default::default()
        },
    }
}

/// The resolved component directories of one load (backend-neutral; shared with the Candle port).
pub use mlx_gen::gen_core::supra2::Supra2Sources;

fn vae_config() -> VaeConfig {
    VaeConfig {
        scaling_factor: s2::VAE_SCALE,
        ..VaeConfig::sdxl_base()
    }
}

/// sd-vae-ft-mse predates diffusers' attention rename: map its legacy mid-block attention keys
/// (`query`/`key`/`value`/`proj_attn`) onto the `to_q`/`to_k`/`to_v`/`to_out.0` layout the shared
/// AutoencoderKL reads. Newer-layout files pass through unchanged.
pub(crate) fn rename_legacy_vae_attention(w: &mut Weights) {
    let renames: Vec<(String, String)> = w
        .keys()
        .filter(|k| k.contains(".attentions."))
        .filter_map(|k| {
            [
                (".query.", ".to_q."),
                (".key.", ".to_k."),
                (".value.", ".to_v."),
                (".proj_attn.", ".to_out.0."),
            ]
            .iter()
            .find(|(from, _)| k.contains(from))
            .map(|(from, to)| (k.to_string(), k.replace(from, to)))
        })
        .collect();
    for (from, to) in renames {
        if let Some(t) = w.remove(&from) {
            w.insert(to, t);
        }
    }
}

/// Load the shared AutoencoderKL from an sd-vae-ft-mse snapshot, f32.
pub fn load_vae(dir: &Path) -> Result<Autoencoder> {
    let mut w = Weights::from_file(dir.join("diffusion_pytorch_model.safetensors"))?;
    rename_legacy_vae_attention(&mut w);
    w.cast_all(mlx_rs::Dtype::Float32)?;
    let vae = Autoencoder::from_weights(&w, &vae_config())?;
    w.materialize_accessed()?;
    let unused = w.unused_keys();
    if !unused.is_empty() {
        return Err(Error::Unsupported(format!(
            "supra2: unconsumed VAE tensors {unused:?}"
        )));
    }
    Ok(vae)
}

/// Load the Flan-T5-Base encoder (12 blocks × 12 heads), f32.
pub fn load_text_encoder(dir: &Path) -> Result<T5TextEncoder> {
    let w = Weights::from_file(dir.join("model.safetensors"))?;
    let t5 = T5TextEncoder::from_weights_with_geometry(
        &w,
        "",
        // Affine group size is only read for packed checkpoints; Flan-T5 loads dense f32.
        64,
        T5Geometry {
            blocks: s2::T5_LAYERS,
            heads: s2::T5_HEADS as i32,
        },
    )?;
    w.materialize_accessed()?;
    Ok(t5)
}

/// Load the converted denoiser, verifying the exact tensor inventory.
pub fn load_denoiser(bundle: &Supra2Bundle) -> Result<SupraDit> {
    let w = Weights::from_file(bundle.denoiser_path())?;
    let dit = SupraDit::from_weights(&w, &bundle.manifest.config)?;
    w.materialize_accessed()?;
    Ok(dit)
}

/// Host text conditioning for one prompt: `[1, ctx_len, ctx_dim]` f32 + its 0/1 mask.
#[derive(Clone)]
pub struct TextConditioning {
    pub context: Array,
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

/// Native Supra2-IMG generator.
pub struct Supra2Image {
    descriptor: ModelDescriptor,
    spec: LoadSpec,
    sources: Supra2Sources,
    tokenizer: Supra2Tokenizer,
    text_encoder: T5TextEncoder,
    dit: SupraDit,
    vae: Autoencoder,
    stored_unconditional: Option<TextConditioning>,
    memory: MemoryProviderContract,
}

/// Registry load entry point.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(Supra2Image::load(spec)?))
}

impl Supra2Image {
    /// Load every component resident (f32). Supra2 is small (~1.5 GB of f32 weights in total), so
    /// only the resident, eager, default-precision shape is offered.
    pub fn load(spec: &LoadSpec) -> Result<Self> {
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
        let text_encoder = load_text_encoder(&sources.text_encoder)?;
        let dit = load_denoiser(&sources.bundle)?;
        let vae = load_vae(&sources.vae)?;
        let stored_unconditional = match sources.bundle.conditioning_path() {
            Some(path) => {
                let w = Weights::from_file(&path)?;
                let cfg = &sources.bundle.manifest.config;
                let text = w.require("uncond_text")?.clone();
                let mask = mlx_gen::array::contiguous(
                    &w.require("uncond_mask")?.as_dtype(mlx_rs::Dtype::Float32)?,
                )?;
                eval([&text, &mask])?;
                Some(TextConditioning {
                    context: text.reshape(&[1, cfg.ctx_len as i32, cfg.ctx_dim as i32])?,
                    mask: mask.as_slice::<f32>().to_vec(),
                })
            }
            None => None,
        };
        Ok(Self {
            descriptor: descriptor(),
            spec: spec.clone(),
            sources,
            tokenizer,
            text_encoder,
            dit,
            vae,
            stored_unconditional,
            memory,
        })
    }

    pub fn config(&self) -> &Supra2Config {
        &self.sources.bundle.manifest.config
    }

    pub fn denoiser(&self) -> &SupraDit {
        &self.dit
    }

    pub fn vae(&self) -> &Autoencoder {
        &self.vae
    }

    /// The checkpoint-stored unconditional conditioning, when the bundle has one.
    pub fn stored_unconditional(&self) -> Option<&TextConditioning> {
        self.stored_unconditional.as_ref()
    }

    /// Tokenize (truncate to `ctx_len`, pad, mask) and encode one prompt with Flan-T5.
    pub fn encode_prompt(&self, prompt: &str) -> Result<(TextConditioning, s2::Supra2Tokens)> {
        let ctx_len = self.config().ctx_len;
        let tokens = self.tokenizer.encode(prompt, ctx_len)?;
        let ids = Array::from_slice(&tokens.ids, &[1, ctx_len as i32]);
        let additive: Vec<f32> = tokens
            .mask
            .iter()
            .map(|&m| if m == 1 { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        let mask = Array::from_slice(&additive, &[1, 1, 1, ctx_len as i32]);
        let context = self.text_encoder.forward_masked(&ids, Some(&mask))?;
        eval([&context])?;
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

    fn broadcast(c: &TextConditioning, n: usize) -> Result<(Array, Array)> {
        sampling::broadcast(&c.context, &c.mask, n)
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
        let (cond_ctx, cond_mask) = Self::broadcast(&cond, n)?;
        let (uncond, used_stored) = if r.use_cfg {
            let (u, stored) = self.unconditional()?;
            (Some(Self::broadcast(&u, n)?), stored)
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
        let s = cfg.latent_size as i32;
        let z0 = Array::from_slice(&z0, &[n as i32, cfg.latent_channels as i32, s, s]);
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
        let flat = mlx_gen::array::contiguous(&z)?;
        eval([&flat])?;
        let latents = flat.as_slice::<f32>().to_vec();
        if latents.iter().any(|v| !v.is_finite()) {
            return Err(Error::Msg("supra2: non-finite latents".into()));
        }

        progress(Progress::Decoding);
        let mut images = Vec::with_capacity(n);
        for i in 0..n {
            check(&request.cancel)?;
            let zi = mlx_gen::attention::slice_axis(&z, 0, i as i32, 1)?
                .transpose_axes(&[0, 2, 3, 1])?;
            let decoded = mlx_gen::array::contiguous(
                &self.vae.decode(&zi)?.as_dtype(mlx_rs::Dtype::Float32)?,
            )?;
            eval([&decoded])?;
            let sh = decoded.shape();
            images.push(Image {
                width: sh[2] as u32,
                height: sh[1] as u32,
                pixels: quantize_rgb(decoded.as_slice::<f32>()),
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
    let sources = Supra2Sources::resolve_checked(spec, BundleCheck::SizesOnly)?;
    let f = memory_strategy::facts(&sources)?;
    Ok(PerComponentBytes {
        text_encoder: f.conditioning_bytes,
        dit: f.transformer_bytes,
        vae: f.decoder_bytes,
    })
}

mlx_gen::register_generators! {pub(crate) const REGISTRATION = descriptor => load; footprint = footprint}

pub(crate) const MEMORY: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: PROVIDER_ID,
    contract: memory_strategy::contract,
    safety_check: memory_strategy::safety,
};

#[cfg(test)]
mod cpu_tests {
    use super::*;

    /// sd-vae-ft-mse's legacy mid-block attention keys map onto the AutoencoderKL layout; keys
    /// already in the new layout, and non-attention keys, are left alone. Tensor-free (no eval).
    #[test]
    fn legacy_vae_attention_keys_are_renamed_and_new_keys_left_alone() {
        let t = || Array::from_slice(&[0f32], &[1]);
        let mut map = std::collections::HashMap::new();
        for side in ["encoder", "decoder"] {
            for leaf in ["query", "key", "value", "proj_attn"] {
                for kind in ["weight", "bias"] {
                    map.insert(format!("{side}.mid_block.attentions.0.{leaf}.{kind}"), t());
                }
            }
        }
        map.insert(
            "decoder.mid_block.attentions.0.group_norm.weight".into(),
            t(),
        );
        map.insert("other.attentions.0.to_q.weight".into(), t());
        map.insert("decoder.conv_in.weight".into(), t());
        let mut w = Weights::from_map(map);
        rename_legacy_vae_attention(&mut w);
        let mut keys: Vec<&str> = w.keys().collect();
        keys.sort_unstable();
        let mut want = vec![
            "decoder.conv_in.weight".to_string(),
            "decoder.mid_block.attentions.0.group_norm.weight".into(),
            "other.attentions.0.to_q.weight".into(),
        ];
        for side in ["encoder", "decoder"] {
            for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                for kind in ["weight", "bias"] {
                    want.push(format!("{side}.mid_block.attentions.0.{leaf}.{kind}"));
                }
            }
        }
        want.sort_unstable();
        assert_eq!(keys, want);
    }
}
