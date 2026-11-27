//! The registered Supra2-IMG generator: load the converted bundle + pinned Flan-T5 + SD VAE, then run
//! upstream's `generate()` — tokenize/truncate/mask, encode, stored-or-fallback unconditional
//! context, CFG-or-plain Euler flow integration, VAE decode, `save_image` quantization.

use std::path::Path;

use crate::sampling::{self, ContextBatch};
use gen_core_checkpoint::{
    supra2 as s2,
    supra2::{
        image_seed, initial_noise, quantize_rgb, BundleCheck, Supra2Bundle, Supra2Config,
        Supra2Resolved, Supra2Tokenizer, PROVIDER_ID, TEXT_ENCODER_COMPONENT, VAE_COMPONENT,
    },
};
use mlx_gen::gen_core::{
    self, GenerationReport, LoadShape, MemoryAssetFacts, MemoryProviderContract, MemoryRunContext,
    MemorySafetyDecision, Modality, PerComponentBytes, SizeFloor,
};
use mlx_gen::weights::Weights;
use mlx_gen::{
    CancelFlag, Error, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, ModelDescriptor, OffloadPolicy, Precision, Progress, Result, WeightsSource,
};
use mlx_gen_flux::text_encoder::{T5Geometry, T5TextEncoder};
use mlx_gen_sdxl::config::VaeConfig;
use mlx_gen_sdxl::vae::Autoencoder;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::dit::SupraDit;
use crate::memory_strategy;

/// Primary typed route declaration for Supra2-IMG (`REG-1`, `AP5`).
pub(crate) fn route_supra2() -> gen_core::RouteRecord {
    gen_core::RouteRecord {
        route_id: PROVIDER_ID,
        family: s2::FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        mac_only: true,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![gen_core::TextEncoderDeclaration::new(
                    "flan_t5_base",
                    Some(s2::MAX_PROMPT_TOKENS),
                )],
                max_prompt_tokens: Some(s2::MAX_PROMPT_TOKENS),
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                kinds: Vec::new(),
                control_kinds: None,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: false,
                modules: Vec::new(),
            },
            sampling: gen_core::SamplingSurface {
                samplers: vec![s2::SAMPLER],
                schedulers: vec![s2::SCHEDULER],
                guidance_methods: vec!["cfg"],
                supports_guidance: true,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: gen_core::ModelBinding::family_model(gen_core::FamilyModelSpec::new(
                "supra2_dit",
            )),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("sd_vae_ft_mse"),
                // The SD VAE's 4-channel f8 latent, scaled by 0.18215 — the same space SVD's VAE consumes.
                output_latent_space: Some(&gen_core::SVD_LATENT_SPACE),
                pixel_stride: Some(8),
                token_width: None,
                supports_diffusion_decoder: false,
            },
            upsamplers: gen_core::UpsamplerSurface::default(),
            adapters: gen_core::AdapterSurface::unsupported(),
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::PromptEncode,
                        ["flan_t5_base"],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["supra2_dit"],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["sd_vae_ft_mse"],
                        false,
                    ),
                ],
                supports_sequential_offload: false,
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::SnapshotDirectory,
                supported_quants: &[],
                component_precision_floors: &[],
                required_components: &[TEXT_ENCODER_COMPONENT, VAE_COMPONENT],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
            },
            output: gen_core::OutputSurface {
                preview: gen_core::PreviewSupport::Unsupported,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: s2::IMAGE_SIZE,
            max_size: s2::IMAGE_SIZE,
            size_floor: SizeFloor::RangeChecked,
            supported_steps: gen_core::StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::DeterministicOffset,
            max_count: s2::MAX_IMAGES,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(s2::DEFAULT_STEPS),
            guidance: Some(s2::DEFAULT_GUIDANCE),
            sampler: Some(s2::SAMPLER),
            schedule: Some(s2::SCHEDULER),
            shift: None,
            strength: None,
            dimensions: Some((s2::IMAGE_SIZE, s2::IMAGE_SIZE)),
            ..Default::default()
        },
        bindings: gen_core::RouteBindings::generator(descriptor, load)
        .with_footprint(footprint)
        .with_memory_registration(MEMORY)
        .with_memory_fixture(memory_strategy::surface_specs, memory_strategy::fixture),
    }
}

/// The weights-free descriptor.
pub fn descriptor() -> ModelDescriptor {
    route_supra2().compute_descriptor()
}

/// The resolved component directories of one load (backend-neutral; shared with the Candle port).
pub use gen_core_checkpoint::supra2::Supra2Sources;

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
    config: Supra2Config,
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
            config: sources.bundle.manifest.config,
            tokenizer,
            text_encoder,
            dit,
            vae,
            stored_unconditional,
            memory,
        })
    }

    /// **Test seam** — assemble a generator from already-built parts on a non-published
    /// `config` (the crate's tiny random-weight `tests/`). The registered provider never reaches
    /// it: [`load`] builds only through [`Supra2Image::load`], which reads a pinned bundle. The
    /// memory contract carries the geometry and no byte facts.
    #[doc(hidden)]
    pub fn from_parts(
        config: Supra2Config,
        tokenizer: Supra2Tokenizer,
        text_encoder: T5TextEncoder,
        dit: SupraDit,
        vae: Autoencoder,
        stored_unconditional: Option<TextConditioning>,
    ) -> Self {
        Self {
            descriptor: descriptor(),
            spec: LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::new())),
            config,
            tokenizer,
            text_encoder,
            dit,
            vae,
            stored_unconditional,
            memory: memory_strategy::contract_with_facts(&config, MemoryAssetFacts::default()),
        }
    }

    /// **Test seam** — drop the checkpoint-stored unconditional conditioning so CFG takes
    /// upstream's empty-string fallback on the real checkpoint (`tests/real_weights.rs`). Not
    /// reachable from the registered provider, whose pinned bundle always stores one.
    #[doc(hidden)]
    pub fn without_stored_unconditional(mut self) -> Self {
        self.stored_unconditional = None;
        self
    }

    pub fn config(&self) -> &Supra2Config {
        &self.config
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

        let (_cond, cond_ctx, cond_mask, uncond, used_stored, tokens) =
            mlx_gen::gen_core::with_prompt_encoding(
                progress,
                || -> std::result::Result<_, mlx_gen::Error> {
                    let (cond, tokens) = self.encode_prompt(&request.prompt)?;
                    let (cond_ctx, cond_mask) = Self::broadcast(&cond, n)?;
                    let (uncond, used_stored) = if r.use_cfg {
                        let (u, stored) = self.unconditional()?;
                        (Some(Self::broadcast(&u, n)?), stored)
                    } else {
                        (None, false)
                    };
                    Ok((cond, cond_ctx, cond_mask, uncond, used_stored, tokens))
                },
            )?;
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
        mlx_gen::image_execution::scalar("supra2.guidance", r.guidance);
        mlx_gen::image_execution::policy_debug("supra2.use_cfg", &r.use_cfg);
        mlx_gen::image_execution::policy_debug("supra2.stored_unconditional", &used_stored);
        mlx_gen::image_execution::policy_debug(
            "supra2.decoder",
            &"resident-sd-vae-rgb8-round-half-up-v1",
        );
        if noise.is_some() {
            mlx_gen::image_execution::unavailable(
                "supra2_external_noise",
                "explicit initial noise bypassed the seeded producer and has no recorded digest",
                "initial_noise",
            );
        } else {
            let consumed = s2::euler_timesteps(r.steps);
            for i in 0..r.count {
                mlx_gen::image_execution::sampled_native_with_rng(
                    "supra2-euler-v1",
                    "clean-fraction",
                    z.dtype(),
                    &consumed,
                    image_seed(r.seed, i),
                    "splitmix64-box-muller-f64-to-f32-v1;initial=output-seed;no-step-draws",
                );
            }
        }
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
        mlx_gen::image_execution::without_capture(|| self.generate_observed(request, on_progress))
    }

    /// The render plus a [`s2::PROMPT_TRUNCATED_WARNING`] when the prompt exceeded the
    /// [`s2::MAX_PROMPT_TOKENS`]-token context (upstream truncates it and logs a warning).
    fn generate_with_report(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        mlx_gen::image_execution::without_capture(|| {
            self.generate_with_report_observed(request, on_progress)
        })
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        Some(&self.memory)
    }

    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        memory_strategy::safety(&self.spec, &self.memory, context)
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        mlx_gen::image_execution::complete_image_report_attempt(
            self.descriptor(),
            &req.cancel,
            || {
                mlx_gen::image_execution::policy_debug(
                    "Supra2Image.load_precision_policy",
                    &self.spec.precision,
                );
                mlx_gen::image_execution::policy_debug(
                    "Supra2Image.load_quantization_policy",
                    &self.spec.quantize,
                );
                let mut progress =
                    |event| mlx_gen::image_execution::without_capture(|| on_progress(event));
                self.generate_with_report_observed(req, &mut progress)
            },
        )
    }
}

impl Supra2Image {
    fn generate_observed(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_with_report(request, on_progress)?
            .output
            .ok_or_else(|| gen_core::Error::Msg("supra2: the render produced no output".into()))
    }
    fn generate_with_report_observed(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        let out = self.generate_with(request, None, on_progress)?;
        let mut report = GenerationReport::from_output(GenerationOutput::Images(out.images));
        report.warnings.extend(s2::prompt_truncated_warning(
            out.prompt_tokens,
            self.config.ctx_len,
        ));
        Ok(report)
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

pub(crate) const MEMORY: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: PROVIDER_ID,
    contract: memory_strategy::contract,
    safety_check: memory_strategy::safety,
};

#[cfg(test)]
mod cpu_tests {
    use super::*;

    /// The 256² geometry and the 128-token prompt limit are advertised weights-free.
    #[test]
    fn descriptor_advertises_the_native_geometry_and_prompt_limit() {
        let d = descriptor();
        assert_eq!(d.backend, "mlx");
        assert_eq!(
            (d.capabilities.min_size, d.capabilities.max_size),
            (256, 256)
        );
        assert_eq!(d.capabilities.max_prompt_tokens, Some(128));
    }

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
