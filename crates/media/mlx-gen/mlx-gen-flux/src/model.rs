//! FLUX.1 provider registration and txt2img generation path.

use mlx_gen::array::scalar;
use mlx_gen::gen_core;
use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    Conditioning, Error, GenerationOutput, GenerationRequest, Generator, Image, LatentDecoder,
    LoadSpec, ModelDescriptor, OffloadPolicy, Precision, Progress, Residency, Result, SamplerRun,
    TimestepConvention, WeightsSource,
};
use mlx_gen_pid::decoder::PidDecoder;
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use mlx_gen_z_image::vae::Vae;
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{Array, Dtype};
use std::path::Path;

use crate::config::{FluxVariant, DEFAULT_SAMPLER, HYPER_SAMPLER};
use crate::image_encoder::FluxIpImageEncoder;
use crate::ip_adapter::{FluxIpAdapter, FluxIpInjector};
use crate::loader;
use crate::pipeline::{build_sigmas_with, create_noise, unpack_latents};
use crate::text_encoder::FluxTextEncoders;
use crate::transformer::FluxTransformer;

/// Default `ip_adapter_scale` when a `Conditioning::Reference` omits its `strength` (epic 3621).
/// SceneWorks maps `ipAdapterScale` (default 0.7) → `strength`; this is the engine-side fallback.
const DEFAULT_IP_SCALE: f32 = 0.7;
/// `true_cfg` clamp range for the IP-Adapter dev path (SceneWorks default 4.0).
const TRUE_CFG_MIN: f32 = 1.0;
const TRUE_CFG_MAX: f32 = 10.0;

/// FLUX packs the /8 VAE latent 2×2, so both request dims must be multiples of 16. Exposed as the
/// pinned-engine stride SceneWorks ties each advertised FLUX image bucket to (sc-12612), mirroring
/// `wan::config::SIZE_MULTIPLE_14B`; both the txt2img `validate_request` and the control lane enforce
/// exactly this value, so the const cannot drift from the check.
pub const SIZE_MULTIPLE: u32 = 16;

/// PiD backbone (latent-space) tag for FLUX.1 (epic 7840, sc-7846). Resolves to the `flux` 16-ch
/// student in `mlx_gen_pid::registry`; used only at load time to build the [`PidEngine`]. Shared by
/// both FLUX.1 variants (dev + schnell) — they share the FLUX.1 VAE latent space.
pub const PID_BACKBONE: &str = "flux";

/// A completed FLUX.1 render has strict-applied every load-time adapter in this order.
pub(crate) fn applied_adapter_controls(
    spec: &LoadSpec,
) -> Vec<(usize, mlx_gen::runtime::AdapterKind, f32)> {
    spec.adapters
        .iter()
        .enumerate()
        .map(|(index, adapter)| (index, adapter.kind, adapter.scale))
        .collect()
}

pub fn descriptor_schnell() -> ModelDescriptor {
    descriptor_for(FluxVariant::Schnell)
}

pub fn descriptor_dev() -> ModelDescriptor {
    descriptor_for(FluxVariant::Dev)
}

pub fn descriptor_for(variant: FluxVariant) -> ModelDescriptor {
    variant.descriptor()
}

pub fn load_schnell(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(FluxVariant::Schnell, spec)
}

pub fn load_dev(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(FluxVariant::Dev, spec)
}

fn load_variant(variant: FluxVariant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_flux1(variant, spec)?))
}

/// Load a fully-weighted [`Flux1`] (the concrete type, not boxed) — the entry point the PuLID-FLUX
/// provider (`mlx-gen-pulid`, sc-3074) wraps so it can drive the denoise through
/// [`Flux1::generate_with_injector`].
///
/// Honors [`LoadSpec::offload_policy`] (sc-10840): `Resident` (default) builds the T5 + CLIP text
/// encoders, the DiT, and the VAE now and holds them warm; `Sequential` keeps only the per-phase
/// loader closures and re-loads per generate in phase order (encode → drop the T5 + CLIP encoders →
/// denoise/decode) to bound peak unified memory to `max(T5+CLIP, DiT+VAE)`. The XLabs IP-Adapter (a
/// small CLIP-ViT image encoder + adapter modules) stays warm-resident either way: it participates
/// only on the reference path, is small relative to the T5-XXL win, and its presence is checked
/// up front in `generate` before any residency work.
pub fn load_flux1(variant: FluxVariant, spec: &LoadSpec) -> Result<Flux1> {
    // Precision + snapshot-dir guard up front for BOTH policies (fail fast).
    resolve_root(variant, spec)?;
    crate::memory_strategy::validate_load_contract(variant.id(), spec)?;
    let stream_inventory = crate::artifact_inventory::verified_stream_inventory(variant.id(), spec);
    // F-181: a `Sequential` + `spec.quantize` load over a *dense* snapshot re-quantizes the whole
    // model on every generate; `Resident` quantizes once. Warn for that combination only.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential) && stream_inventory.is_none() {
            mlx_gen::residency::warn_sequential_requantize(variant.id(), q.bits());
        }
    }

    // Optional XLabs IP-Adapter (epic 3621), loaded from `LoadSpec::ip_adapter`, held warm-resident.
    // The plain txt2img path is unaffected when absent.
    let ip_adapter = if let Some(WeightsSource::Dir(path)) = &spec.ip_adapter {
        Some(loader::load_flux_ip_adapter(path)?)
    } else {
        None
    };

    let memory_strategy =
        crate::memory_strategy::validated_memory_strategy_contract_with_inventory(
            variant.id(),
            spec,
            stream_inventory.as_ref(),
        )?;
    let residency = build_residency(variant, spec, stream_inventory.clone())?;
    Ok(Flux1 {
        descriptor: descriptor_for(variant),
        variant,
        residency,
        ip_adapter,
        memory_strategy,
        loaded_spec: spec.clone(),
        stream_inventory,
    })
}

/// The phase-A text components (T5 + CLIP tokenizers + the T5-XXL / CLIP-L text encoders) dropped
/// first under `Sequential`. Shared by the base [`Flux1`] and the control variant
/// ([`crate::model_control::Flux1DevControl`]), whose text path is identical.
pub(crate) struct FluxTextOwned {
    pub(crate) t5_tokenizer: TextTokenizer,
    pub(crate) clip_tokenizer: TextTokenizer,
    pub(crate) text_encoders: FluxTextEncoders,
}

impl FluxTextOwned {
    /// Tokenize + T5/CLIP-encode a prompt into `(prompt_embeds, pooled_prompt_embeds)`.
    pub(crate) fn encode(&self, prompt: &str) -> Result<(Array, Array)> {
        let (t5_ids, _) = mlx_gen::tokenizer::to_arrays(&self.t5_tokenizer.tokenize(prompt)?);
        let (clip_ids, _) = mlx_gen::tokenizer::to_arrays(&self.clip_tokenizer.tokenize(prompt)?);
        self.text_encoders.encode(&t5_ids, &clip_ids)
    }

    /// [`encode`](Self::encode), then evaluate both embeddings before returning. Every render route
    /// calls this inside its Conditioning phase, so the T5/CLIP forward (its allocations and any
    /// failure) belongs to Conditioning under both `Sequential` and `Resident` offload rather than
    /// running later in the staged-residency materialization or the first denoise step. Evaluating
    /// earlier copies nothing and leaves the values unchanged.
    pub(crate) fn encode_materialized(&self, prompt: &str) -> Result<(Array, Array)> {
        let (prompt_embeds, pooled_prompt_embeds) = self.encode(prompt)?;
        mlx_rs::transforms::eval([&prompt_embeds, &pooled_prompt_embeds])?;
        Ok((prompt_embeds, pooled_prompt_embeds))
    }
}

/// The heavy render-phase components (the DiT transformer, the VAE, and the optional PiD decoder) —
/// everything but the text encoders and the (small, warm-resident) IP-Adapter. Owned by the
/// `Resident` components or by a `Sequential` generate.
pub(crate) struct FluxHeavyOwned {
    transformer: FluxTransformer,
    vae: Vae,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7846). `Some` only when the
    /// `LoadSpec` carried `pid` AND this generate uses it (`req.use_pid`); swaps the native FLUX.1
    /// VAE for a 4× PiD decode at the `run_denoise` decode call site.
    pid: Option<PidEngine>,
}

/// Precision guard (only dense bf16 is wired) + snapshot-dir resolution (rejecting a single-file
/// source), shared by [`load_flux1`] and [`build_residency`]'s per-phase loaders.
pub(crate) fn resolve_root(variant: FluxVariant, spec: &LoadSpec) -> Result<&Path> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{}: only dense bf16 is wired for the FLUX.1 port plan",
            variant.id()
        )));
    }
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(format!(
            "{} expects a FLUX.1 snapshot directory (tokenizer/ tokenizer_2/ text_encoder/ \
             text_encoder_2/ transformer/ vae/), not a single .safetensors file",
            variant.id()
        ))),
    }
}

/// Load the T5 + CLIP tokenizers and the T5-XXL / CLIP-L text encoders (+ optional whole-encoder
/// Q4/Q8) — the phase-A components dropped first under `Sequential`. Factored so the `Resident` and
/// `Sequential` paths build byte-identical encoders.
pub(crate) fn load_flux_text(
    variant: FluxVariant,
    spec: &LoadSpec,
    stream_inventory: Option<&crate::artifact_inventory::PackedArtifactInventory>,
) -> Result<FluxTextOwned> {
    let root = resolve_root(variant, spec)?;
    if let Some(inventory) = stream_inventory {
        inventory
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
    }
    let t5_tokenizer = match stream_inventory {
        Some(inventory) => loader::load_t5_tokenizer_from_file(
            inventory.t5_tokenizer_source().loader_path(),
            variant,
        )?,
        None => loader::load_t5_tokenizer(root, variant)?,
    };
    let clip_tokenizer = loader::load_clip_tokenizer()?;
    let mut text_encoders = FluxTextEncoders {
        t5: match stream_inventory {
            Some(inventory) => {
                loader::load_t5_encoder_from_file(inventory.t5_encoder_source().loader_path())?
            }
            None => loader::load_t5_encoder(root)?,
        },
        clip: match stream_inventory {
            Some(inventory) => {
                loader::load_clip_encoder_from_file(inventory.clip_encoder_source().loader_path())?
            }
            None => loader::load_clip_encoder(root)?,
        },
    };
    if let Some(q) = spec.quantize {
        text_encoders.quantize(q.bits())?;
    }
    if let Some(inventory) = stream_inventory {
        inventory
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
    }
    Ok(FluxTextOwned {
        t5_tokenizer,
        clip_tokenizer,
        text_encoders,
    })
}

/// Load the heavy render-phase components — the DiT transformer (+ Q4/Q8 + LoRA/LoKr residuals), the
/// VAE (+ Q4/Q8), and the optional PiD overlay — everything but the text encoders. Factored so the
/// `Sequential` path loads these AFTER the encoders are dropped (bounding peak to
/// `max(T5+CLIP, DiT+VAE)`). Quantize-then-adapters order matches the pre-seam composition; the
/// components are independent of the text encoders (separate weight files, deterministic RNG-free
/// quant), so the `Resident` composition is byte-identical.
fn load_flux_heavy(
    variant: FluxVariant,
    spec: &LoadSpec,
    load_pid: bool,
    stream_inventory: Option<&crate::artifact_inventory::PackedArtifactInventory>,
) -> Result<FluxHeavyOwned> {
    let root = resolve_root(variant, spec)?;
    let mut transformer = match stream_inventory {
        Some(inventory) => {
            inventory
                .ensure_unchanged()
                .map_err(|error| Error::Msg(error.to_string()))?;
            let transformer = loader::load_transformer_from_file(
                inventory.transformer_source().loader_path(),
                variant,
            )?;
            inventory
                .ensure_unchanged()
                .map_err(|error| Error::Msg(error.to_string()))?;
            transformer
                .with_block_stream(inventory.clone(), spec.quantize.map(mlx_gen::Quant::bits))
        }
        None => loader::load_transformer(root, variant)?,
    };
    let mut vae = match stream_inventory {
        Some(inventory) => {
            let vae = loader::load_vae_from_file(inventory.vae_source().loader_path())?;
            inventory
                .ensure_unchanged()
                .map_err(|error| Error::Msg(error.to_string()))?;
            vae
        }
        None => loader::load_vae(root)?,
    };
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        if stream_inventory.is_some() {
            crate::block_stream::ensure_prepacked(&mut transformer, "transformer")?;
        }
        transformer.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // Install LoRA/LoKr adapters AFTER quantization (the fork merges/applies post-quantize too; a
    // forward-time residual over the now-quantized base, never a fused merge). No-op when empty; a
    // non-empty spec list that matches nothing — or any unmatched target — errors loudly (sc-2534).
    crate::adapters::apply_flux_adapters(&mut transformer, &spec.adapters)?;
    transformer.finalize_block_stream()?;
    if stream_inventory.is_some() && transformer.resident_block_counts() != (0, 0) {
        return Err(Error::Msg(
            "flux1: deferred DiT retained joint or single transformer blocks".to_owned(),
        ));
    }
    // Optional PiD decoder overlay (epic 7840, sc-7846): the FLUX.1 16-ch VAE latent space has a PiD
    // student, so the final decode can route through `mlx_gen_pid` when `req.use_pid` is set. Loaded
    // only when the spec carries `pid` AND this generate uses it (`load_pid`, F-177): the Resident path
    // passes `true` (loaded once, reused), the Sequential path passes `req.use_pid` so a non-PiD
    // generate skips the student + its caption encoder entirely.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    Ok(FluxHeavyOwned {
        transformer,
        vae,
        pid,
    })
}

/// The policy→[`Residency`] dispatch the FLUX.1 base variants share (sc-10840), routed through the
/// single [`Residency::from_policy`] seam. `Resident` eager-loads the text encoders + heavy bundle now
/// (any PiD overlay loaded once, reused); `Sequential` captures the two per-phase loaders and loads
/// nothing now, deferring each to [`Residency::run`]. Both use the same [`load_flux_text`] /
/// [`load_flux_heavy`], so the `Resident` composition is byte-identical to the pre-seam one.
fn build_residency(
    variant: FluxVariant,
    spec: &LoadSpec,
    stream_inventory: Option<crate::artifact_inventory::PackedArtifactInventory>,
) -> Result<Residency<FluxTextOwned, FluxHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    match spec.offload_policy {
        OffloadPolicy::Resident => Residency::from_policy(
            OffloadPolicy::Resident,
            move || load_flux_text(variant, &spec_text, None),
            move |use_pid| load_flux_heavy(variant, &spec_heavy, use_pid, None),
        ),
        OffloadPolicy::Sequential => {
            let text_inventory = stream_inventory.clone();
            Ok(Residency::request_scoped(
                move |streamable| {
                    let inventory = if streamable {
                        Some(text_inventory.as_ref().ok_or_else(|| {
                            Error::Unsupported(
                                "flux1: block streaming requested without a verified inventory"
                                    .to_owned(),
                            )
                        })?)
                    } else {
                        None
                    };
                    load_flux_text(variant, &spec_text, inventory)
                },
                move |use_pid, streamable| {
                    let inventory = if streamable {
                        Some(stream_inventory.as_ref().ok_or_else(|| {
                            Error::Unsupported(
                                "flux1: block streaming requested without a verified inventory"
                                    .to_owned(),
                            )
                        })?)
                    } else {
                        None
                    };
                    load_flux_heavy(variant, &spec_heavy, use_pid, inventory)
                },
            ))
        }
    }
}

pub struct Flux1 {
    descriptor: ModelDescriptor,
    variant: FluxVariant,
    /// Component-residency strategy (sc-10840), selected from [`LoadSpec::offload_policy`]. `Resident`
    /// (default) holds the T5 + CLIP text encoders + DiT + VAE warm; `Sequential` holds only the
    /// per-phase loader closures and re-loads per generation in phase order (encode → **drop the T5 +
    /// CLIP encoders** → denoise/decode). The [`Residency`] seam owns the eval/drop/clear discipline,
    /// the stage-boundary cancel checks, and the error-safe cache flush.
    residency: Residency<FluxTextOwned, FluxHeavyOwned>,
    /// XLabs IP-Adapter (epic 3621): the CLIP-ViT-L/14 image encoder (sc-3622) + the adapter modules
    /// (sc-3623), held warm-resident (small; used only on the reference path). `Some` only when a
    /// `LoadSpec::ip_adapter` was supplied. A `Conditioning::Reference` request errors loudly when
    /// this is `None` — checked up front in `generate`, before any residency work.
    ip_adapter: Option<(FluxIpImageEncoder, FluxIpAdapter)>,
    /// Loaded-contract copy used by the runtime selector and its defense-in-depth admission check.
    memory_strategy: gen_core::MemoryProviderContract,
    /// Exact load axes used to build [`Self::memory_strategy`]. Overlay-bearing specs deliberately
    /// keep bounded attention unavailable until those extra attention paths have their own proof.
    loaded_spec: LoadSpec,
    /// Exact packed component + T5-tokenizer inventory retained by the runtime streaming path. The
    /// DiT source is the pinned `transformer/model.safetensors`, never a rediscovered shard/glob.
    stream_inventory: Option<crate::artifact_inventory::PackedArtifactInventory>,
}

impl Flux1 {
    pub fn new_for_tests(variant: FluxVariant) -> Self {
        let loaded_spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        let memory_strategy = crate::memory_strategy::weights_free_memory_strategy_contract(
            variant.id(),
            &loaded_spec,
        )
        .expect("test-only FLUX contract");
        Self {
            descriptor: descriptor_for(variant),
            variant,
            // Weightless: loader closures that error if invoked. The validation tests never `run` the
            // residency; the `generate_reference_without_ip_adapter_errors` test hits the up-front
            // IP-adapter check in `generate` before the residency, so this is never driven either.
            residency: Residency::sequential(
                || {
                    Err(Error::Msg(
                        "flux1: text encoder not loadable in a test-only instance".into(),
                    ))
                },
                |_use_pid| {
                    Err(Error::Msg(
                        "flux1: heavy bundle not loadable in a test-only instance".into(),
                    ))
                },
            ),
            ip_adapter: None,
            memory_strategy,
            loaded_spec,
            stream_inventory: None,
        }
    }

    fn requested_transformer_window(
        &self,
        req: &GenerationRequest,
        _injector: Option<&dyn crate::transformer::DitImageInjector>,
    ) -> Result<Option<usize>> {
        let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
            return Ok(None);
        };
        if req.use_pid
            || !memory.stage_residency
            || !self
                .memory_strategy
                .lifecycle
                .transformer_window_materialization
            || !crate::artifact_inventory::structurally_streamable(
                self.variant.id(),
                &self.loaded_spec,
            )
        {
            return Err(Error::Unsupported(
                "flux1: transformer streaming requires the clean-base Sequential + DeferredMaterialization route"
                    .to_owned(),
            ));
        }
        if memory
            .transformer_window_component
            .unwrap_or(gen_core::TransformerComponent::Dit)
            != gen_core::TransformerComponent::Dit
        {
            return Err(Error::Unsupported(
                "flux1: transformer streaming is DiT-only".to_owned(),
            ));
        }
        let size = memory
            .transformer_window_size
            .unwrap_or(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE);
        if size != crate::memory_strategy::TRANSFORMER_WINDOW_SIZE {
            return Err(Error::Unsupported(format!(
                "flux1: transformer window {size} is outside the published production domain"
            )));
        }
        self.verify_stream_inventory()?;
        Ok(Some(size as usize))
    }

    fn verify_stream_inventory(&self) -> Result<()> {
        self.stream_inventory
            .as_ref()
            .ok_or_else(|| {
                Error::Unsupported(
                    "flux1: transformer streaming has no verified packed inventory".to_owned(),
                )
            })?
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))
    }

    fn finish_streamed_generation<T>(&self, streamed: bool, result: Result<T>) -> Result<T> {
        if matches!(&result, Err(Error::ProcessPoisoned(_))) {
            return result;
        }
        if streamed {
            // Run after the residency driver has released the heavy phase, including on a block
            // fault, cancellation, or ordinary forward error. A source mutation outranks those
            // errors; explicit process poison must retain its typed retirement signal.
            self.verify_stream_inventory()?;
        }
        result
    }
}

impl Flux1 {
    /// Load-time selections bound to every reported attempt.
    fn observe_loaded(&self) {
        mlx_gen::image_execution::policy_debug(
            "Flux1.loaded_sequential_residency",
            &self.residency.is_sequential(),
        );
        mlx_gen::image_execution::policy_debug(
            "Flux1.load_precision_policy",
            &self.loaded_spec.precision,
        );
        mlx_gen::image_execution::policy_debug(
            "Flux1.load_quantization_policy",
            &self.loaded_spec.quantize,
        );
        mlx_gen::image_execution::policy_debug(
            "Flux1.applied_adapter_controls",
            &applied_adapter_controls(&self.loaded_spec),
        );
    }
}

impl Generator for Flux1 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        mlx_gen::image_execution::without_capture(|| self.generate_observed(req, on_progress))
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&gen_core::RenderLoadBinding::from_spec(&self.loaded_spec))?;
        // Refusal precedes the XLabs reference encode and either prompt encode.
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled.into());
        }
        let run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        let output = if let Some((image, strength)) = single_reference(req)? {
            let (encoder, adapter) = self.ip_adapter.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "{}: a reference image needs an IP-adapter — load it via LoadSpec::with_ip_adapter",
                    self.descriptor.id
                ))
            })?;
            let scale = strength.unwrap_or(DEFAULT_IP_SCALE).clamp(0.0, 1.0);
            mlx_gen::image_execution::scalar("flux.ip_scale", scale);
            match req.true_cfg {
                Some(cfg) if self.variant.supports_guidance() => {
                    // The negative IP branch uses a black image, not a missing injector.
                    let neg_prompt = req.negative_prompt.as_deref().unwrap_or("");
                    self.generate_with_prepared_injector_cfg_planned(
                        req,
                        neg_prompt,
                        cfg.clamp(TRUE_CFG_MIN, TRUE_CFG_MAX),
                        0,
                        &run,
                        || {
                            let embeds = encoder.encode(image)?;
                            let black_embeds = encoder.encode_black()?;
                            mlx_rs::transforms::eval([&embeds, &black_embeds])?;
                            let pos = FluxIpInjector::new(adapter, &embeds, scale)?;
                            let neg = FluxIpInjector::new(adapter, &black_embeds, scale)?;
                            pos.materialize_tokens()?;
                            neg.materialize_tokens()?;
                            Ok((pos, neg))
                        },
                        on_progress,
                    )?
                }
                _ => self.generate_with_prepared_injector_planned(
                    req,
                    &run,
                    || {
                        let embeds = encoder.encode(image)?;
                        mlx_rs::transforms::eval([&embeds])?;
                        let injector = FluxIpInjector::new(adapter, &embeds, scale)?;
                        injector.materialize_tokens()?;
                        Ok(injector)
                    },
                    on_progress,
                )?,
            }
        } else {
            self.run_with_injector(req, None, &run, false, on_progress)?
        };
        let pag_report = self
            .descriptor
            .capabilities
            .resolve_pag(self.descriptor.id, req)?
            .map(|p| p.execution_report(plan.denoise_sigmas().len().saturating_sub(1) as u32));
        let slot_config = if self.loaded_spec.adapters.is_empty() {
            "base_dit".to_owned()
        } else {
            format!("base_dit+adapters:{}", self.loaded_spec.adapters.len())
        };
        let low_res_report = plan.low_res_start_execution_report("f32", 4, &slot_config)?;
        let mut report = gen_core::GenerationReport::from_output(output);
        report.pag = pag_report;
        report.low_res_start = low_res_report;
        Ok(report)
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        mlx_gen::image_execution::without_capture(|| {
            self.generate_with_report_observed(req, on_progress)
        })
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.loaded_spec, &self.memory_strategy, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        if self.memory_strategy.engages(
            context.selection.strategy,
            gen_core::MemoryStrategy::BoundedTransformerResidency,
        ) {
            let inventory = self.stream_inventory.as_ref().ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "{}: transformer streaming has no verified packed inventory",
                    self.descriptor.id
                ))
            })?;
            inventory.ensure_unchanged()?;
            // The block loader opens this exact pin, never rediscovering a file by glob.
            inventory.transformer_source().ensure_unchanged()?;
        }
        crate::memory_strategy::begin_request(
            self.variant.id(),
            &self.loaded_spec,
            &self.memory_strategy,
            context,
        )
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
                self.observe_loaded();
                let mut progress =
                    |event| mlx_gen::image_execution::without_capture(|| on_progress(event));
                self.generate_with_report_observed(req, &mut progress)
            },
        )
    }

    /// The admitted runner's entry: the same planned body, captured as one attempt that covers
    /// the work inside its Conditioning/Denoise/Decode scopes.
    fn generate_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        mlx_gen::image_execution::complete_planned_image_report(
            self,
            req,
            plan,
            scope,
            on_progress,
            || self.observe_loaded(),
        )
    }
}

impl Flux1 {
    fn generate_with_report_observed(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        // The plan freezes the trajectory before any reference or prompt encode; every encode then
        // runs inside the plan's Conditioning phase.
        self.validate(req)?;
        let plan = compile_trajectory(self.variant, req)?.bind_load(&self.loaded_spec);
        self.generate_planned(req, &plan, None, on_progress)
    }

    fn generate_observed(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_with_report_observed(req, on_progress)?
            .output
            .ok_or_else(|| gen_core::Error::Msg("FLUX.1 render produced no images".into()))
    }
}

/// Extract the single reference image + its optional `strength` (`ip_adapter_scale`) from a request,
/// or `None` for plain txt2img. Errors on `MultiReference` or more than one reference (FLUX.1's
/// XLabs IP-Adapter conditions on exactly one image).
fn single_reference(req: &GenerationRequest) -> Result<Option<(&Image, Option<f32>)>> {
    match req.conditioning.as_slice() {
        [] => Ok(None),
        [Conditioning::Reference { image, strength }] => Ok(Some((image, *strength))),
        _ => Err(Error::Msg(
            "flux: reference conditioning supports exactly one Reference image".into(),
        )),
    }
}

impl Flux1 {
    /// As [`Generator::generate`], but threading an optional per-block image-stream residual
    /// injector (the PuLID id cross-attn, sc-3072/3074) into every denoise step's transformer
    /// forward. `injector = None` is byte-identical to [`Generator::generate`] (it IS that path).
    pub fn generate_with_injector(
        &self,
        req: &GenerationRequest,
        injector: Option<&dyn crate::transformer::DitImageInjector>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let plan = compile_trajectory(self.variant, req)?.bind_load(&self.loaded_spec);
        let run = gen_core::RenderRun::new(&plan, &req.cancel, None);
        self.run_with_injector(req, injector, &run, false, on_progress)
    }

    /// PuLID validates its own descriptor and source-bound plan, then delegates the same FLUX
    /// trajectory to this backbone with its reference consumed into an injector.
    pub fn generate_with_injector_planned(
        &self,
        req: &GenerationRequest,
        injector: &dyn crate::transformer::DitImageInjector,
        run: &gen_core::RenderRun<'_, '_>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        self.run_with_injector(req, Some(injector), run, false, on_progress)
    }

    fn run_with_injector(
        &self,
        req: &GenerationRequest,
        injector: Option<&dyn crate::transformer::DitImageInjector>,
        run: &gen_core::RenderRun<'_, '_>,
        conditioning_started: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.run_with_prepared_injector(
            req,
            run,
            conditioning_started,
            || Ok(injector),
            |prepared| *prepared,
            on_progress,
        )
    }

    /// PuLID prepares its owned identity injector only after staged text weights have dropped.
    /// The same Conditioning phase remains open across that residency handoff.
    pub fn generate_with_prepared_injector_planned<I: crate::transformer::DitImageInjector>(
        &self,
        req: &GenerationRequest,
        run: &gen_core::RenderRun<'_, '_>,
        prepare: impl FnOnce() -> Result<I>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        run.begin_conditioning()?;
        self.run_with_prepared_injector(
            req,
            run,
            true,
            prepare,
            |prepared| Some(prepared),
            on_progress,
        )
    }

    fn run_with_prepared_injector<I>(
        &self,
        req: &GenerationRequest,
        run: &gen_core::RenderRun<'_, '_>,
        conditioning_started: bool,
        prepare: impl FnOnce() -> Result<I>,
        borrow: impl for<'a> Fn(&'a I) -> Option<&'a dyn crate::transformer::DitImageInjector>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let transformer_window = self.requested_transformer_window(req, None)?;
        mlx_gen::image_execution::policy_debug("flux.transformer_window", &transformer_window);
        let streamable = transformer_window.is_some();
        let stage_residency = matches!(self.loaded_spec.offload_policy, OffloadPolicy::Sequential);
        mlx_gen::image_execution::policy_debug("flux.stage_residency", &stage_residency);
        // Staged residency lifecycle (sc-10840): under `Sequential` the seam loads the T5 + CLIP
        // encoders, encodes the prompt, materializes, then DROPS them + `clear_cache()` before the
        // DiT/VAE load — the peak-bounding win. Under `Resident` it borrows the warm encoders and runs
        // the identical encode/denoise/decode with no eval/clear.
        // PiD's Gemma caption encode is text conditioning too, and its engine arrives with the heavy
        // bundle: a PiD request keeps Conditioning open across the residency handoff until the
        // decoder (and its caption) is prepared; an ordinary single-source img2img request also
        // keeps Conditioning open until its init image is VAE-encoded at phase-0 geometry.
        let hold_conditioning = conditioning_started || req.use_pid || req.source_image.is_some();
        let result = self.residency.run_request_scoped(
            stage_residency,
            streamable,
            &req.cancel,
            req.use_pid,
            on_progress,
            |text: &FluxTextOwned| {
                if hold_conditioning {
                    if !conditioning_started {
                        run.begin_conditioning()?;
                    }
                    text.encode_materialized(&req.prompt)
                } else {
                    run.encode(|| text.encode_materialized(&req.prompt))
                }
            },
            // Materialize the embeds while the encoders are still alive (Sequential only) — MLX is lazy,
            // so an un-evaluated embed keeps the encoders referenced and the drop would free nothing.
            |encoded| {
                let Some((prompt_embeds, pooled_prompt_embeds)) = encoded else {
                    return Ok(());
                };
                mlx_rs::transforms::eval([prompt_embeds, pooled_prompt_embeds])?;
                Ok(())
            },
            |heavy, (prompt_embeds, pooled_prompt_embeds), on_progress| {
                let prepared = prepare()?;
                let pid_decoder = self.conditioned_pid_decoder(run, req, heavy.pid.as_ref())?;
                let clean_init = match &req.source_image {
                    Some(source_image) => {
                        let (init_w, init_h) = run.plan().phase_output_size(0);
                        let image_nchw =
                            mlx_gen::img2img::preprocess_init_image(source_image, init_w, init_h)?;
                        let encoded = heavy.vae.encode(&image_nchw)?;
                        let packed = crate::pipeline::pack_latents(&encoded, init_w, init_h)?;
                        mlx_rs::transforms::eval([&packed])?;
                        Some(packed)
                    }
                    None => None,
                };
                if hold_conditioning {
                    run.end_conditioning()?;
                }
                let injector = borrow(&prepared);
                let transformer = &heavy.transformer;
                let block_window = transformer.block_window(
                    transformer_window,
                    &req.cancel,
                    req.memory.is_some_and(|memory| {
                        memory.calibration_fault_harness_authorized
                            && memory.calibration_error_phase
                                == Some(gen_core::MemoryPhase::Denoise)
                    }),
                )?;
                // The shared plan covers the FLUX trunk. Identity providers independently bound
                // their injector attention before advertising this request configuration.
                let attention = if self.memory_strategy.lifecycle.attention_chunking {
                    crate::memory_strategy::attention_plan(req)
                } else {
                    mlx_gen::attention::AttentionPlan::UNBOUNDED
                };
                let pag = self
                    .descriptor
                    .capabilities
                    .resolve_pag(self.descriptor.id, req)?;
                mlx_gen::image_execution::policy_debug("flux.pag", &pag);
                let total_steps = run.plan().denoise_sigmas().len().saturating_sub(1);
                self.run_denoise_with(
                    run,
                    req,
                    &heavy.vae,
                    pid_decoder,
                    clean_init,
                    on_progress,
                    |x_in, step_idx, timestep, guidance, phase_w, phase_h| {
                        let pos = transformer.forward_injected_memory_windowed_pag(
                            x_in,
                            &prompt_embeds,
                            &pooled_prompt_embeds,
                            timestep,
                            guidance,
                            phase_w,
                            phase_h,
                            injector,
                            attention,
                            block_window,
                            None,
                        )?;
                        let step_pag = pag
                            .as_ref()
                            .map_or(0.0, |p| p.step_scale(step_idx, total_steps));
                        if step_pag > 0.0 {
                            mlx_rs::transforms::eval([&pos])?;
                            let layers = pag
                                .as_ref()
                                .map_or(&[][..], |p| p.step_layers(step_idx, total_steps));
                            let pert = transformer.forward_injected_memory_windowed_pag(
                                x_in,
                                &prompt_embeds,
                                &pooled_prompt_embeds,
                                timestep,
                                guidance,
                                phase_w,
                                phase_h,
                                injector,
                                attention,
                                block_window,
                                Some(layers),
                            )?;
                            Ok(add(
                                &pos,
                                &multiply(&subtract(&pos, &pert)?, scalar(step_pag))?,
                            )?)
                        } else {
                            Ok(pos)
                        }
                    },
                )
            },
        );
        self.finish_streamed_generation(streamable, result)
    }

    /// Dual-branch real-CFG denoise — the seam PuLID-FLUX `true_cfg > 1.0` (sc-3075) uses. Each step
    /// runs a positive forward (`req.prompt` + `pos_injector`) and, once `t >= timestep_to_start_cfg`,
    /// a negative forward (`neg_prompt` + `neg_injector`), combined as `neg + true_cfg·(pos − neg)`.
    /// The distilled `guidance` is still applied on both branches (the upstream PuLID recipe). The
    /// injectors are generic, so this carries no PuLID-specific code; with two no-op injectors it
    /// reduces to classifier-free guidance between two prompts.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_injector_cfg(
        &self,
        req: &GenerationRequest,
        pos_injector: &dyn crate::transformer::DitImageInjector,
        neg_injector: &dyn crate::transformer::DitImageInjector,
        neg_prompt: &str,
        true_cfg: f32,
        timestep_to_start_cfg: usize,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let plan = compile_trajectory(self.variant, req)?.bind_load(&self.loaded_spec);
        let run = gen_core::RenderRun::new(&plan, &req.cancel, None);
        self.run_with_injector_cfg(
            req,
            pos_injector,
            neg_injector,
            neg_prompt,
            true_cfg,
            timestep_to_start_cfg,
            &run,
            false,
            on_progress,
        )
    }

    /// PuLID's real-CFG variant of [`Self::generate_with_injector_planned`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_injector_cfg_planned(
        &self,
        req: &GenerationRequest,
        pos_injector: &dyn crate::transformer::DitImageInjector,
        neg_injector: &dyn crate::transformer::DitImageInjector,
        neg_prompt: &str,
        true_cfg: f32,
        timestep_to_start_cfg: usize,
        run: &gen_core::RenderRun<'_, '_>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        self.run_with_injector_cfg(
            req,
            pos_injector,
            neg_injector,
            neg_prompt,
            true_cfg,
            timestep_to_start_cfg,
            run,
            false,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_with_injector_cfg(
        &self,
        req: &GenerationRequest,
        pos_injector: &dyn crate::transformer::DitImageInjector,
        neg_injector: &dyn crate::transformer::DitImageInjector,
        neg_prompt: &str,
        true_cfg: f32,
        timestep_to_start_cfg: usize,
        run: &gen_core::RenderRun<'_, '_>,
        conditioning_started: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.run_with_prepared_injector_cfg(
            req,
            neg_prompt,
            true_cfg,
            timestep_to_start_cfg,
            run,
            conditioning_started,
            || Ok((pos_injector, neg_injector)),
            |prepared| (prepared.0, prepared.1),
            on_progress,
        )
    }

    /// PuLID builds both positive and unconditional identity branches after text weights drop.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_prepared_injector_cfg_planned<I: crate::transformer::DitImageInjector>(
        &self,
        req: &GenerationRequest,
        neg_prompt: &str,
        true_cfg: f32,
        timestep_to_start_cfg: usize,
        run: &gen_core::RenderRun<'_, '_>,
        prepare: impl FnOnce() -> Result<(I, I)>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        run.begin_conditioning()?;
        self.run_with_prepared_injector_cfg(
            req,
            neg_prompt,
            true_cfg,
            timestep_to_start_cfg,
            run,
            true,
            prepare,
            |prepared| (&prepared.0, &prepared.1),
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_with_prepared_injector_cfg<I>(
        &self,
        req: &GenerationRequest,
        neg_prompt: &str,
        true_cfg: f32,
        timestep_to_start_cfg: usize,
        run: &gen_core::RenderRun<'_, '_>,
        conditioning_started: bool,
        prepare: impl FnOnce() -> Result<I>,
        borrow: impl for<'a> Fn(
            &'a I,
        ) -> (
            &'a dyn crate::transformer::DitImageInjector,
            &'a dyn crate::transformer::DitImageInjector,
        ),
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let transformer_window = self.requested_transformer_window(req, None)?;
        mlx_gen::image_execution::scalar("flux.true_cfg", true_cfg);
        mlx_gen::image_execution::policy_debug("flux.true_cfg_start", &timestep_to_start_cfg);
        mlx_gen::image_execution::policy_debug("flux.transformer_window", &transformer_window);
        let streamable = transformer_window.is_some();
        // Staged residency lifecycle (sc-10840): the phase-A encode covers BOTH the positive and
        // negative prompts, so both text encoders drop together before the DiT load under `Sequential`.
        // As in the single-branch path, a PiD request holds Conditioning through decoder preparation.
        let hold_conditioning = conditioning_started || req.use_pid;
        let result = self.residency.run_request_scoped(
            matches!(self.loaded_spec.offload_policy, OffloadPolicy::Sequential),
            streamable,
            &req.cancel,
            req.use_pid,
            on_progress,
            |text: &FluxTextOwned| {
                let encode_text = || {
                    let (pos_embeds, pos_pooled) = text.encode_materialized(&req.prompt)?;
                    let (neg_embeds, neg_pooled) = text.encode_materialized(neg_prompt)?;
                    Ok((pos_embeds, pos_pooled, neg_embeds, neg_pooled))
                };
                if hold_conditioning {
                    if !conditioning_started {
                        run.begin_conditioning()?;
                    }
                    encode_text()
                } else {
                    run.encode(encode_text)
                }
            },
            |encoded| {
                let Some((pos_embeds, pos_pooled, neg_embeds, neg_pooled)) = encoded else {
                    return Ok(());
                };
                mlx_rs::transforms::eval([pos_embeds, pos_pooled, neg_embeds, neg_pooled])?;
                Ok(())
            },
            |heavy, (pos_embeds, pos_pooled, neg_embeds, neg_pooled), on_progress| {
                let prepared = prepare()?;
                let pid_decoder = self.conditioned_pid_decoder(run, req, heavy.pid.as_ref())?;
                if hold_conditioning {
                    run.end_conditioning()?;
                }
                let (pos_injector, neg_injector) = borrow(&prepared);
                let transformer = &heavy.transformer;
                let block_window = transformer.block_window(
                    transformer_window,
                    &req.cancel,
                    req.memory.is_some_and(|memory| {
                        memory.calibration_fault_harness_authorized
                            && memory.calibration_error_phase
                                == Some(gen_core::MemoryPhase::Denoise)
                    }),
                )?;
                let attention = if self.memory_strategy.lifecycle.attention_chunking {
                    crate::memory_strategy::attention_plan(req)
                } else {
                    mlx_gen::attention::AttentionPlan::UNBOUNDED
                };
                let pag = self
                    .descriptor
                    .capabilities
                    .resolve_pag(self.descriptor.id, req)?;
                mlx_gen::image_execution::policy_debug("flux.pag", &pag);
                let total_steps = run.plan().denoise_sigmas().len().saturating_sub(1);
                self.run_denoise_with(
                    run,
                    req,
                    &heavy.vae,
                    pid_decoder,
                    None,
                    on_progress,
                    |x_in, t, timestep, guidance, phase_w, phase_h| {
                        let pos = transformer.forward_injected_memory_windowed_pag(
                            x_in,
                            &pos_embeds,
                            &pos_pooled,
                            timestep,
                            guidance,
                            phase_w,
                            phase_h,
                            Some(pos_injector),
                            attention,
                            block_window,
                            None,
                        )?;
                        let step_pag = pag.as_ref().map_or(0.0, |p| p.step_scale(t, total_steps));
                        let base = if t >= timestep_to_start_cfg {
                            mlx_rs::transforms::eval([&pos])?;
                            let neg = transformer.forward_injected_memory_windowed_pag(
                                x_in,
                                &neg_embeds,
                                &neg_pooled,
                                timestep,
                                guidance,
                                phase_w,
                                phase_h,
                                Some(neg_injector),
                                attention,
                                block_window,
                                None,
                            )?;
                            // neg + true_cfg · (pos − neg)
                            add(&neg, &multiply(&subtract(&pos, &neg)?, scalar(true_cfg))?)?
                        } else {
                            pos.clone()
                        };
                        if step_pag > 0.0 {
                            mlx_rs::transforms::eval([&base, &pos])?;
                            let layers = pag
                                .as_ref()
                                .map_or(&[][..], |p| p.step_layers(t, total_steps));
                            let pert = transformer.forward_injected_memory_windowed_pag(
                                x_in,
                                &pos_embeds,
                                &pos_pooled,
                                timestep,
                                guidance,
                                phase_w,
                                phase_h,
                                Some(pos_injector),
                                attention,
                                block_window,
                                Some(layers),
                            )?;
                            Ok(add(
                                &base,
                                &multiply(&subtract(&pos, &pert)?, scalar(step_pag))?,
                            )?)
                        } else {
                            Ok(base)
                        }
                    },
                )
            },
        );
        self.finish_streamed_generation(streamable, result)
    }

    /// Shared denoise scaffold for the injector generators (F-108): resolve seed / sampler / steps /
    /// guidance, build the flow-match sigmas + sampler, enable the compile-glue, and run the
    /// per-count → per-step loop (cancel check → `scale_model_input` → `velocity_fn` → `step` →
    /// progress → unpack/decode). `velocity_fn(x_in, step_idx, timestep, guidance)` returns the step's
    /// velocity — the only thing that differs between the single-branch and CFG paths. Bit-identical
    /// to the prior inline loops (the base render stays e2e-parity-exact).
    ///
    /// Takes the heavy render components (`vae`, `pid`) explicitly, so the shared [`Residency`] seam's
    /// render closure passes either the warm-resident bundle or the just-loaded `Sequential` one
    /// (sc-10840). `validate` is done by the public callers before the residency seam runs, so it is
    /// not re-checked here.
    /// Mint the request's PiD decoder (or `None` for the native VAE) and evaluate its Gemma
    /// caption embeddings. Called inside the Conditioning phase: the caption encode is text
    /// conditioning, not decode work.
    fn conditioned_pid_decoder(
        &self,
        run: &gen_core::RenderRun<'_, '_>,
        req: &GenerationRequest,
        pid: Option<&PidEngine>,
    ) -> Result<Option<PidDecoder>> {
        let plan = run.plan();
        let decoder = resolve_pid_decoder_at_sigma(
            pid,
            req,
            plan.base_seed(),
            self.descriptor.id,
            plan.capture_sigma(),
        )?;
        if let Some(decoder) = &decoder {
            decoder.materialize_caption()?;
        }
        Ok(decoder)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_denoise_with(
        &self,
        run: &gen_core::RenderRun<'_, '_>,
        req: &GenerationRequest,
        vae: &Vae,
        pid_decoder: Option<PidDecoder>,
        clean_init: Option<Array>,
        on_progress: &mut dyn FnMut(Progress),
        mut velocity_fn: impl FnMut(&Array, usize, f32, f32, u32, u32) -> Result<Array>,
    ) -> Result<GenerationOutput> {
        let plan = run.plan();
        let guidance = plan.guidance();
        mlx_gen::image_execution::scalar("flux.embedded_guidance", guidance);
        // Native VAE tiling is a clean-base mechanism. PiD keeps its own checked decode route and
        // never receives a native VAE tile geometry; overlay-bearing contracts leave rung 2 Missing.
        let native_tiling = match &pid_decoder {
            None if self.memory_strategy.lifecycle.decode_tiling => {
                crate::memory_strategy::decode_tiling(req)?
            }
            _ => None,
        };
        let denoise_sigmas = plan.denoise_sigmas();
        let (init_w, init_h) = plan.phase_output_size(0);
        // Route the flow-match denoise through the unified curated-sampler framework (epic 7114 P3):
        // the default `flow_match` / `hyper` profile maps to Euler (the legacy `x + v·Δσ` step, within
        // the N1 parity tolerance), and the curated menu (heun / dpmpp_2m / uni_pc / …) becomes
        // selectable. FLUX feeds the raw schedule sigma as the transformer timestep (Sigma convention).
        // sc-2963: run the MMDiT's fusable elementwise glue (adaLN affine, gated residual, tanh-GELU
        // FFN, RoPE rotation) through `mx.compile` — bit-exact and a per-step win. Scoped to this
        // render by the RAII guard (F-007): the render thread's prior setting is restored on drop.
        let _compile_glue = crate::transformer::CompileGlueGuard::enable();

        let images = run.images_with_denoise_phases(
            on_progress,
            |seed| {
                let noise = create_noise(seed, init_w, init_h)?;
                match &clean_init {
                    Some(clean) => mlx_gen::img2img::add_noise_by_interpolation(
                        clean,
                        &noise,
                        denoise_sigmas[0],
                    ),
                    None => Ok(noise),
                }
            },
            |seed, latents, on_progress| {
                execute_flux_denoise_trajectory(
                    run,
                    req,
                    seed,
                    latents,
                    on_progress,
                    &mut velocity_fn,
                )
            },
            |_, final_latents, on_progress| {
                on_progress(Progress::Decoding);
                let unpacked = unpack_latents(&final_latents, req.width, req.height)?;
                let decoded = decode_latents_via_seam(
                    vae,
                    pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder),
                    self.descriptor.denoiser_output_latent_space,
                    &unpacked,
                    native_tiling.as_ref(),
                    &req.cancel,
                )?;
                // A fault probe must observe the selected lazy decode before returning its synthetic
                // error. Normal untiled requests retain their historical materialization path.
                if req.memory.is_some_and(|memory| {
                    memory.calibration_fault_harness_authorized
                        && memory.calibration_error_phase == Some(gen_core::MemoryPhase::Decode)
                }) {
                    decoded.eval()?;
                }
                crate::memory_strategy::calibration_fault(
                    req,
                    gen_core::MemoryPhase::Decode,
                    self.descriptor.id,
                )?;
                decoded_to_image(&decoded)
            },
        )?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Execute the typed fp32 boundary transition for MLX FLUX.1 at `sigma_k` (`SW-6939`: `PIPE-4`, `FEAT-1`, `§6.1`).
///
/// Refuses [`gen_core::BoundaryState::XSigma`] with a typed [`Error::Unsupported`] so a transition
/// can never double-noise a noisy latent.
#[allow(clippy::too_many_arguments)]
pub fn transition_mlx_flux_boundary_state(
    boundary_state: gen_core::BoundaryState<Array>,
    low_res_plan: &gen_core::ResolvedLowResStartPlan,
    sigma_k: f32,
    ms: &dyn gen_core::sampling::ModelSampling,
    request_seed: u64,
    output_index: u32,
    pass_index: usize,
    phase_index: usize,
) -> Result<Array> {
    let (x0_low, source) = boundary_state.into_clean_estimate()?;
    let target_dtype = x0_low.dtype();
    let x0_f32 = mlx_gen::array::contiguous(&x0_low.as_dtype(Dtype::Float32)?)?;
    mlx_rs::transforms::eval([&x0_f32])?;
    let packed_slice = x0_f32.as_slice::<f32>();
    let repacked = gen_core::transition_packed_clean_estimate_f32(
        gen_core::BoundaryState::CleanEstimate {
            latent: packed_slice,
            source,
        },
        1,
        16,
        2,
        8,
        low_res_plan,
        sigma_k,
        ms,
        request_seed,
        output_index,
        pass_index,
        phase_index,
    )?;
    let full_tokens = ((low_res_plan.full_height / SIZE_MULTIPLE)
        * (low_res_plan.full_width / SIZE_MULTIPLE)) as i32;
    let out = Array::from_slice(
        &repacked,
        &[1, full_tokens, crate::pipeline::PACKED_TOKEN_WIDTH],
    );
    if target_dtype == Dtype::Float32 {
        Ok(out)
    } else {
        Ok(out.as_dtype(target_dtype)?)
    }
}

/// Unified MLX FLUX.1 denoise trajectory runner supporting both single-phase (`low_res_start: None`)
/// and two-phase (`low_res_start: Some(..)`) execution over the same sampler implementation (`SW-6939`: `HOOK-5`, `E2`).
pub fn execute_flux_denoise_trajectory(
    run: &gen_core::RenderRun<'_, '_>,
    req: &GenerationRequest,
    seed: u64,
    latents: Array,
    on_progress: &mut dyn FnMut(Progress),
    mut velocity_fn: impl FnMut(&Array, usize, f32, f32, u32, u32) -> Result<Array>,
) -> Result<Array> {
    use std::cell::{Cell, RefCell};

    let plan = run.plan();
    let sampler_name = plan.sampler();
    let guidance = plan.guidance();
    let denoise_sigmas = plan.denoise_sigmas();
    let denoise_spans = plan.denoise_spans();
    let previews = mlx_gen::preview::PreviewCounter::new(denoise_sigmas);
    let canonical_solver =
        mlx_gen::gen_core::sampling::canonical_solver_or_euler(Some(sampler_name));
    let total_steps = denoise_sigmas.len().saturating_sub(1);
    let mut eval_idx = 0usize;

    let Some(low_res_plan) = plan.low_res_start().copied() else {
        return mlx_gen::run_flow_sampler_with_phase_transition_spans(
            SamplerRun {
                sampler_name: Some(sampler_name),
                sigmas: denoise_sigmas,
                seed,
                cancel: &req.cancel,
                on_progress,
            },
            TimestepConvention::Sigma,
            latents,
            &denoise_spans,
            Some(run),
            |latents, sigma| {
                crate::preview::emit_preview(
                    &req.preview,
                    &previews,
                    denoise_sigmas,
                    sigma,
                    latents,
                    req.width,
                    req.height,
                );
            },
            |x_in, timestep| {
                let current_eval = eval_idx;
                eval_idx += 1;
                let step_idx = mlx_gen::gen_core::sampling::step_index_for_eval(
                    canonical_solver,
                    current_eval,
                    total_steps,
                );
                velocity_fn(x_in, step_idx, timestep, guidance, req.width, req.height)
            },
            None,
        );
    };

    let current_dims = Cell::new((low_res_plan.low_width, low_res_plan.low_height));
    let velocity_cell = RefCell::new(&mut velocity_fn);
    let output_index = seed.wrapping_sub(plan.base_seed()) as u32;
    let pass_index = plan.pass_index();
    let base_seed = plan.base_seed();
    if let Some(low_res_report) = plan.low_res_start_execution_report("f32", 4, "base_dit")? {
        mlx_gen::image_execution::policy(
            "flux.low_res_start",
            serde_json::to_value(&low_res_report).unwrap_or(serde_json::Value::Null),
        );
    }

    let mut transition_cb =
        |_phase: usize,
         boundary_step: usize,
         ops: &mlx_gen::MlxLatentOps,
         ms: &dyn gen_core::sampling::ModelSampling,
         x_sigma_k: &Array,
         state: &mut gen_core::sampling::SolverRunState<mlx_gen::MlxLatentOps>|
         -> Result<Array> {
            if boundary_step != low_res_plan.k {
                return Ok(x_sigma_k.clone());
            }
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let sigma_k = denoise_sigmas[boundary_step];
            let boundary_state = match low_res_plan.estimate_source {
                gen_core::EstimateSource::EndpointEvaluation => {
                    let x0_low =
                        gen_core::sampling::denoise(ops, ms, x_sigma_k, sigma_k, |x_in, t| {
                            (velocity_cell.borrow_mut())(
                                x_in,
                                boundary_step,
                                t,
                                guidance,
                                low_res_plan.low_width,
                                low_res_plan.low_height,
                            )
                            .map_err(gen_core::Error::from)
                        })?;
                    gen_core::BoundaryState::CleanEstimate {
                        latent: x0_low,
                        source: gen_core::EstimateSource::EndpointEvaluation,
                    }
                }
                gen_core::EstimateSource::PreviousEvaluation => {
                    let captured = state.take_captured_primary_estimate(boundary_step - 1)?;
                    gen_core::BoundaryState::CleanEstimate {
                        latent: captured.latent,
                        source: gen_core::EstimateSource::PreviousEvaluation,
                    }
                }
            };
            let transitioned = transition_mlx_flux_boundary_state(
                boundary_state,
                &low_res_plan,
                sigma_k,
                ms,
                base_seed,
                output_index,
                pass_index,
                0,
            )?;
            state.note_geometry_change(
                (low_res_plan.low_width, low_res_plan.low_height),
                (low_res_plan.full_width, low_res_plan.full_height),
            );
            state.note_renoise_transition();
            state.apply_phase_boundary(gen_core::sampling::PhaseBoundary::Reset, true)?;
            current_dims.set((low_res_plan.full_width, low_res_plan.full_height));
            Ok(transitioned)
        };

    mlx_gen::run_flow_sampler_with_phase_transition_spans(
        SamplerRun {
            sampler_name: Some(sampler_name),
            sigmas: denoise_sigmas,
            seed,
            cancel: &req.cancel,
            on_progress,
        },
        TimestepConvention::Sigma,
        latents,
        &denoise_spans,
        Some(run),
        |latents, sigma| {
            let (pw, ph) = current_dims.get();
            crate::preview::emit_preview(
                &req.preview,
                &previews,
                denoise_sigmas,
                sigma,
                latents,
                pw,
                ph,
            );
        },
        |x_in, timestep| {
            let current_eval = eval_idx;
            eval_idx += 1;
            let step_idx = mlx_gen::gen_core::sampling::step_index_for_eval(
                canonical_solver,
                current_eval,
                total_steps,
            );
            let (pw, ph) = current_dims.get();
            (velocity_cell.borrow_mut())(x_in, step_idx, timestep, guidance, pw, ph)
        },
        Some(&mut transition_cb),
    )
}

/// Freeze the provider-native schedule before any encoder, transformer or VAE moves.
fn compile_trajectory(
    variant: FluxVariant,
    req: &GenerationRequest,
) -> Result<gen_core::RenderPlan> {
    compile_trajectory_for(variant, &descriptor_for(variant), req)
}

/// Compile the FLUX.1 backbone trajectory for a provider that owns its own reference injector.
/// The enclosing descriptor and request remain in the plan so its source cannot be swapped.
pub fn compile_trajectory_for(
    variant: FluxVariant,
    descriptor: &ModelDescriptor,
    req: &GenerationRequest,
) -> Result<gen_core::RenderPlan> {
    let sampler_name = req.sampler.as_deref().unwrap_or(DEFAULT_SAMPLER);
    let (def_steps, def_guidance) = profile_defaults(variant, sampler_name);
    let steps = req.steps.unwrap_or(def_steps) as usize;
    let guidance = if variant.supports_guidance() {
        req.guidance.unwrap_or(def_guidance)
    } else {
        0.0
    };
    // The FLUX diffusion path is MIXED precision, matching the mflux reference (sc-2787): the
    // latents (`create_noise` → f32) and the main residual stream stay f32; only the CLIP pooled
    // embedding + the time/text/guidance conditioning run bf16 (in the encoders / `TimeTextEmbed`).
    // So latents are NOT cast to bf16 — that would diverge from the fork.
    // Native `LinearScheduler` schedule is the byte-exact default (epic 7114 N1); a curated
    // `req.scheduler` re-shapes σ over FLUX.1's own resolution-dependent mu (schnell: unshifted).
    let sigmas = build_sigmas_with(
        steps,
        req.width,
        req.height,
        variant.requires_sigma_shift(),
        req.scheduler.as_deref(),
    )?;
    let mut sigmas = mlx_gen::resolve_flow_schedule_with_shift(
        req.scheduler.as_deref(),
        req.scheduler_shift,
        crate::pipeline::linear_sigma_mu(req.width, req.height, variant.requires_sigma_shift()),
        steps,
        &sigmas,
    )?;
    if req.source_image.is_some() {
        let (_, truncated) = gen_core::resolve_img2img_truncated_schedule(&sigmas, req.strength)?;
        sigmas = truncated;
    }
    let (capture_sigma, keep) = flow_capture_for_request(req, &sigmas, 0);
    Ok(gen_core::RenderPlan::packed_image_trajectory(
        descriptor,
        req,
        gen_core::render::PackedImagePass {
            pixel_stride: SIZE_MULTIPLE,
            token_width: crate::pipeline::PACKED_TOKEN_WIDTH as u32,
            sampler: sampler_name,
            guidance,
            sigmas,
            executed_steps: keep - 1,
            capture_sigma,
        },
    )?)
}

pub(crate) fn route_schnell() -> gen_core::RouteRecord {
    let mut route = FluxVariant::Schnell.route_record();
    route.bindings = gen_core::RouteBindings::generator(descriptor_schnell, |spec| {
        load_schnell(spec).map_err(Into::into)
    })
    .with_footprint(component_footprint)
    .with_render_plan(|spec, req| compile_base(FluxVariant::Schnell, spec, req))
    .with_memory_registration(SCHNELL_MEMORY_REGISTRATION)
    .with_memory_fixture(gen_core::mlx_memory_contract_surface_specs, |spec| {
        crate::memory_strategy::weights_free_memory_strategy_contract(crate::FLUX1_SCHNELL_ID, spec)
    })
    .with_memory_surface_resolver(|surface| {
        crate::memory_strategy::weights_free_memory_surface_contract(
            crate::FLUX1_SCHNELL_ID,
            surface,
        )
    })
    .with_memory_behavior_registration(SCHNELL_MEMORY_BEHAVIOR);
    route
}

pub(crate) fn route_dev() -> gen_core::RouteRecord {
    let mut route = FluxVariant::Dev.route_record();
    route.bindings = gen_core::RouteBindings::generator(descriptor_dev, |spec| {
        load_dev(spec).map_err(Into::into)
    })
    .with_footprint(component_footprint)
    .with_render_plan(|spec, req| compile_base(FluxVariant::Dev, spec, req))
    .with_memory_registration(DEV_MEMORY_REGISTRATION)
    .with_memory_fixture(gen_core::mlx_memory_contract_surface_specs, |spec| {
        crate::memory_strategy::weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, spec)
    })
    .with_memory_surface_resolver(|surface| {
        crate::memory_strategy::weights_free_memory_surface_contract(crate::FLUX1_DEV_ID, surface)
    })
    .with_memory_behavior_registration(DEV_MEMORY_BEHAVIOR)
    .with_activation_memory(crate::DEV_ACTIVATION_MEMORY_REGISTRATION.anchor);
    route
}

pub(crate) fn route_schnell_ip_adapter() -> gen_core::RouteRecord {
    let mut route = FluxVariant::Schnell.route_record();
    route.route_id = "flux1_schnell_ip_adapter";
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresReference;
    route.structure.weight_plan.requires_ip_adapter_overlay = true;
    route.defaults.strength = Some(DEFAULT_IP_SCALE);
    route.bindings = gen_core::RouteBindings::feature_route(
        Some(crate::FLUX1_SCHNELL_ID),
        descriptor_schnell_ip_adapter,
    )
    .with_render_plan(|spec, req| compile_ip_adapter(FluxVariant::Schnell, spec, req));
    route
}

pub(crate) fn route_dev_ip_adapter() -> gen_core::RouteRecord {
    let mut route = FluxVariant::Dev.route_record();
    route.route_id = "flux1_dev_ip_adapter";
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresReference;
    route.structure.weight_plan.requires_ip_adapter_overlay = true;
    route.defaults.strength = Some(DEFAULT_IP_SCALE);
    route.bindings = gen_core::RouteBindings::feature_route(
        Some(crate::FLUX1_DEV_ID),
        descriptor_dev_ip_adapter,
    )
    .with_render_plan(|spec, req| compile_ip_adapter(FluxVariant::Dev, spec, req));
    route
}

pub(crate) fn descriptor_schnell_ip_adapter() -> ModelDescriptor {
    route_schnell_ip_adapter().compute_descriptor()
}

pub(crate) fn descriptor_dev_ip_adapter() -> ModelDescriptor {
    route_dev_ip_adapter().compute_descriptor()
}

pub(crate) fn compile_ip_adapter(
    variant: FluxVariant,
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    if single_reference(req)?.is_none() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}_ip_adapter: a reference image (Conditioning::Reference) is required",
            variant.id()
        )));
    }
    if req.low_res_start.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}_ip_adapter: low_res_start is not supported by this model route",
            variant.id()
        )));
    }
    compile_base(variant, spec, req)
}

pub(crate) fn compile_base(
    variant: FluxVariant,
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    validate_request(&descriptor_for(variant), req)?;
    crate::memory_strategy::validate_load_contract(variant.id(), spec)?;
    if single_reference(req)?.is_some() && spec.ip_adapter.is_none() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: reference image needs an IP-adapter",
            variant.id()
        )));
    }
    if req.cancel.is_cancelled() {
        return Err(gen_core::Error::Canceled);
    }
    if req.use_pid && spec.pid.is_none() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: PiD decode needs a loaded overlay",
            variant.id()
        )));
    }
    Ok(Some(compile_trajectory(variant, req)?.bind_load(spec)))
}

/// Decode the final unpacked FLUX latent through the native VAE or a request-selected PiD
/// override. Kept as one production helper so the default/no-override byte contract can be tested
/// without constructing the transformer and text encoders.
fn decode_latents_via_seam(
    native: &dyn LatentDecoder,
    decoder_override: Option<&dyn LatentDecoder>,
    expected_latent_space: Option<&gen_core::LatentSpace>,
    latents: &Array,
    native_tiling: Option<&mlx_gen::TilingConfig>,
    cancel: &mlx_gen::CancelFlag,
) -> Result<Array> {
    mlx_gen::image_execution::policy_debug("flux.native_decode_tiling", &native_tiling);
    mlx_gen::image_execution::policy_debug("flux.decode_override", &decoder_override.is_some());
    let decoder = decoder_override.unwrap_or(native);
    mlx_gen::ensure_decoder_compatible(expected_latent_space, decoder)?;
    let decoded = match native_tiling {
        Some(tiling) => decoder.decode_tiled(latents, tiling, Some(cancel))?,
        None => decoder.decode(latents)?,
    };
    Ok(decoded.as_dtype(Dtype::Float32)?)
}

/// Few-step profile defaults `(steps, guidance)` applied when the request omits them (sc-2908). The
/// base flow-match path uses the variant's own defaults; the `hyper` profile (Hyper-FLUX.1-dev) is 8
/// steps at guidance 3.5 — paired with the ByteDance Hyper-FLUX 8-step LoRA loaded at `scale≈0.125`
/// (the documented `lora_scale`) via `spec.adapters`. `hyper` is dev-only (it is a FLUX.1-dev LoRA)
/// and schnell never advertises it, so it never reaches here for schnell.
fn profile_defaults(variant: FluxVariant, sampler: &str) -> (u32, f32) {
    match sampler {
        HYPER_SAMPLER => (8, crate::config::DEFAULT_GUIDANCE),
        _ => (variant.default_steps(), crate::config::DEFAULT_GUIDANCE),
    }
}

fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{}: prompt is required", desc.id)));
    }
    // Reject a sampler the variant does not advertise (e.g. `hyper` on schnell, or any typo) rather
    // than silently falling back to the base flow-match path.
    if let Some(s) = &req.sampler {
        if !desc.capabilities.samplers.contains(&s.as_str()) {
            return Err(Error::Msg(format!(
                "{}: unsupported sampler {s:?} (supported: {:?})",
                desc.id, desc.capabilities.samplers
            )));
        }
    }
    // Likewise reject an unadvertised scheduler (F-100: previously silently accepted). flux1's
    // negative/true_cfg checks keep their IP-Adapter reference carve-out below, so this validate
    // stays bespoke rather than fully delegating to `Capabilities::validate_request`.
    if let Some(s) = &req.scheduler {
        if !desc.capabilities.schedulers.contains(&s.as_str()) {
            return Err(Error::Msg(format!(
                "{}: unsupported scheduler {s:?} (supported: {:?})",
                desc.id, desc.capabilities.schedulers
            )));
        }
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "{}: width and height must be multiples of {SIZE_MULTIPLE}, got {}x{}",
            desc.id, req.width, req.height
        )));
    }
    let caps = &desc.capabilities;
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(Error::Msg(format!(
            "{}: size {}x{} outside supported range {}..={}",
            desc.id, req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    if req.count == 0 || req.count > caps.max_count {
        return Err(Error::Msg(format!(
            "{}: count must be 1..={}",
            desc.id, caps.max_count
        )));
    }
    // Floor checks the bespoke validator had drifted below (F-105): the base flux1 variant kept a
    // hand-rolled validate for the IP-Adapter reference carve-out below, so it never absorbed these.
    // `steps: Some(0)` otherwise silently clamps to 1; a non-finite guidance/true_cfg flows into
    // `time_text_embed` / survives `.clamp` (Rust clamp returns NaN) and NaN-poisons the render; an
    // unadvertised `guidance_method` was silently ignored. (Finiteness is centralized on the request
    // so every `Option<f32>` knob inherits the guard — F-001.)
    if req.steps == Some(0) {
        return Err(Error::Msg(format!(
            "{}: steps must be >= 1 (an explicit 0 renders undenoised noise)",
            desc.id
        )));
    }
    if req.phases.is_some() {
        return Err(Error::Unsupported(format!(
            "{}: multi-phase denoise (phases) is not supported",
            desc.id
        )));
    }
    req.ensure_finite_floats()?;
    if let Some(m) = &req.guidance_method {
        if !caps.supported_guidance_methods.contains(&m.as_str()) {
            return Err(Error::Unsupported(format!(
                "{}: unsupported guidance_method {m:?} (supported: {:?})",
                desc.id, caps.supported_guidance_methods
            )));
        }
    }
    // Reference conditioning (XLabs IP-Adapter, epic 3621): exactly one `Reference`. The negative
    // prompt + true_cfg knobs ride the IP-Adapter dev path (real CFG), so they are permitted
    // alongside a reference on a guidance-capable (dev) variant even though base txt2img advertises
    // neither — but never on the plain (no-reference) path, which stays bit-identical.
    let has_reference = match req.conditioning.as_slice() {
        [] => false,
        [Conditioning::Reference { .. }] => true,
        conditioning
            if conditioning
                .iter()
                .any(|value| matches!(value, Conditioning::Reference { .. }))
                && conditioning
                    .iter()
                    .any(|value| !matches!(value, Conditioning::Reference { .. })) =>
        {
            return Err(Error::Unsupported(format!(
                "{}: FLUX.1 IP-Adapter Reference conditioning cannot be combined with pose/control or other conditioning",
                desc.id
            )));
        }
        _ => {
            return Err(Error::Msg(format!(
                "{}: only a single Reference image is supported (no MultiReference / multiple references)",
                desc.id
            )))
        }
    };
    let ref_cfg_ok = has_reference && caps.supports_guidance;
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt && !ref_cfg_ok {
        return Err(Error::Msg(format!(
            "{}: negative prompts are not supported",
            desc.id
        )));
    }
    if req.guidance.is_some() && !caps.supports_guidance {
        return Err(Error::Msg(format!(
            "{}: guidance is not supported by this distilled variant",
            desc.id
        )));
    }
    if req.true_cfg.is_some() && !caps.supports_true_cfg && !ref_cfg_ok {
        return Err(Error::Msg(format!(
            "{}: true_cfg is not supported",
            desc.id
        )));
    }
    caps.resolve_pag(desc.id, req)?;
    if let Some(source) = &req.source_image {
        if !desc.capabilities.supports_low_res_start {
            return Err(Error::Unsupported(format!(
                "{}: source_image is not supported by this model route",
                desc.id
            )));
        }
        if has_reference || !req.conditioning.is_empty() {
            return Err(Error::Unsupported(format!(
                "{}: source_image cannot be combined with pass-0 reference/control conditioning",
                desc.id
            )));
        }
        let expected = u64::from(source.width)
            .checked_mul(u64::from(source.height))
            .and_then(|p| p.checked_mul(3))
            .ok_or_else(|| {
                Error::Unsupported(format!("{}: source_image geometry overflows", desc.id))
            })?;
        if source.width == 0 || source.height == 0 || source.pixels.len() as u64 != expected {
            return Err(Error::Unsupported(format!(
                "{}: source_image buffer len {} does not match {}x{}x3",
                desc.id,
                source.pixels.len(),
                source.width,
                source.height
            )));
        }
        if let Some(s) = req.strength {
            if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                return Err(Error::Unsupported(format!(
                    "{}: img2img strength {s} must lie in 0.0..=1.0",
                    desc.id
                )));
            }
        }
    }
    if req.low_res_start.is_some() {
        if !desc.capabilities.supports_low_res_start {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start is not supported by this model route",
                desc.id
            )));
        }
        if has_reference || !req.conditioning.is_empty() {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start conflicts with pass-0 edit/reference/control conditioning",
                desc.id
            )));
        }
        let variant = if desc.id.contains("schnell") {
            FluxVariant::Schnell
        } else {
            FluxVariant::Dev
        };
        compile_trajectory_for(variant, desc, req)?;
    }
    Ok(())
}

// The registration constants bridge the crate's rich `Result` into backend-neutral
// `gen_core::Result`. The `impl Generator`
// above stays hand-written because `generate` is bespoke (IP-adapter / true-CFG branching), not the
// plain delegation `impl_generator!` expresses.
/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate's staged-residency split — the CLIP
/// and T5 text encoders (`text_encoder/`, `text_encoder_2/`), the DiT (`transformer/`), and the VAE
/// (`vae/`), summed from the exact snapshot subdirs [`crate::loader`] loads. Shared by schnell / dev /
/// dev-control (the control checkpoint is folded by the worker).
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder", "text_encoder_2"],
        &["transformer"],
        &["vae"],
    )
}

pub const SCHNELL_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: crate::FLUX1_SCHNELL_ID,
        contract: |spec| {
            crate::memory_strategy::memory_strategy_contract(crate::FLUX1_SCHNELL_ID, spec)
        },
        safety_check: crate::memory_strategy::registered_safety_check,
    };

pub const DEV_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: crate::FLUX1_DEV_ID,
    contract: |spec| crate::memory_strategy::memory_strategy_contract(crate::FLUX1_DEV_ID, spec),
    safety_check: crate::memory_strategy::registered_safety_check,
};

pub const SCHNELL_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::FLUX1_SCHNELL_ID,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(
                crate::FLUX1_SCHNELL_ID,
                spec,
                contract,
                context,
            )
        },
    };

pub const DEV_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::FLUX1_DEV_ID,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(
                crate::FLUX1_DEV_ID,
                spec,
                contract,
                context,
            )
        },
    };

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUX1_DEV_ID, FLUX1_SCHNELL_ID};
    use std::cell::Cell;

    #[test]
    fn applied_adapter_controls_distinguish_scale_and_order() {
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let first = AdapterSpec::new("/fixture/one".into(), 0.5, AdapterKind::Lora);
        let second = AdapterSpec::new("/fixture/two".into(), 1.0, AdapterKind::Lokr);
        let source = LoadSpec::new(WeightsSource::Dir("/fixture/base".into()))
            .with_adapters(vec![first.clone(), second.clone()]);
        let scaled = LoadSpec::new(WeightsSource::Dir("/fixture/base".into())).with_adapters(vec![
            AdapterSpec::new(first.path, 0.75, first.kind),
            second.clone(),
        ]);
        let reversed =
            LoadSpec::new(WeightsSource::Dir("/fixture/base".into())).with_adapters(vec![
                second,
                AdapterSpec::new("/fixture/one".into(), 0.5, AdapterKind::Lora),
            ]);
        assert_ne!(
            applied_adapter_controls(&source),
            applied_adapter_controls(&scaled)
        );
        assert_ne!(
            applied_adapter_controls(&source),
            applied_adapter_controls(&reversed)
        );
    }

    #[test]
    fn loaded_generator_accepts_native_and_pid_requests_when_pid_is_available() {
        let spec = LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/flux1-loaded-pid-contract".into(),
        ))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_pid(
            WeightsSource::File("/nonexistent/pid.safetensors".into()),
            WeightsSource::Dir("/nonexistent/gemma".into()),
        );
        let model = load_flux1(FluxVariant::Dev, &spec)
            .expect("Sequential load must construct the loaded generator without touching weights");
        let contract = model
            .memory_strategy_contract()
            .expect("loaded generator must expose its exact contract");
        let fixture_contract =
            crate::memory_strategy::weights_free_memory_strategy_contract(FLUX1_DEV_ID, &spec)
                .unwrap();
        let fixture = crate::memory_strategy::registered_valid_fixture(
            &spec,
            &fixture_contract,
            gen_core::MemoryStrategy::StagedResidency,
        )
        .unwrap()
        .remove(0);
        assert_eq!(contract.provider_id, FLUX1_DEV_ID);

        for use_pid in [false, true] {
            let mut context = fixture.context.clone();
            context.use_pid = use_pid;
            context.optimization_authority = gen_core::MemoryOptimizationAuthority::Estimated;
            assert_eq!(
                model.memory_strategy_safety_check(&context),
                gen_core::MemorySafetyDecision::Accept
            );
            let mut scope = model
                .begin_memory_strategy_request(&context)
                .unwrap()
                .expect("loaded generator must open its registered request scope");
            let mut request = fixture.request.clone();
            request.use_pid = use_pid;
            scope.configure_request(&mut request).unwrap();
            assert!(request.memory.unwrap().stage_residency);
        }
    }

    #[test]
    fn production_load_rejects_the_same_unsupported_components_as_registry_admission() {
        let mut spec = LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/flux1-invalid-component".into(),
        ))
        .with_offload_policy(OffloadPolicy::Sequential);
        spec.components.insert(
            "unexpected".into(),
            WeightsSource::File("/nonexistent/unexpected.safetensors".into()),
        );
        assert!(crate::memory_strategy::memory_strategy_contract(FLUX1_DEV_ID, &spec).is_err());
        assert!(load_flux1(FluxVariant::Dev, &spec).is_err());
    }

    struct DecodeSpy {
        output: Array,
        calls: Cell<usize>,
    }

    impl DecodeSpy {
        fn new(output: Array) -> Self {
            Self {
                output,
                calls: Cell::new(0),
            }
        }
    }

    impl LatentDecoder for DecodeSpy {
        fn input_latent_space(&self) -> Option<&gen_core::LatentSpace> {
            Some(&gen_core::FLUX1_LATENT_SPACE)
        }

        fn decode(&self, _latents: &Array) -> Result<Array> {
            self.calls.set(self.calls.get() + 1);
            Ok(self.output.clone())
        }
    }

    fn legacy_decode_one(decoder: &dyn LatentDecoder, latents: &Array) -> Image {
        let decoded = decoder
            .decode(latents)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        decoded_to_image(&decoded).unwrap()
    }

    /// SC-18309 N1: compare the helper used by the FLUX generation loop against the exact
    /// pre-seam decode expression. The asymmetric NCTHW fixture makes dtype conversion,
    /// singleton-frame removal, NCHW-to-RGB layout, denormalization, and output bytes observable.
    /// The override arm proves a PiD decoder retains precedence and its dynamic output geometry.
    #[test]
    fn decode_engine_keeps_native_and_pid_dispatch_byte_exact() {
        let latents = Array::from_slice(&[0.0f32; 16 * 4 * 6], &[1, 16, 4, 6]);
        let native_output = Array::from_slice(
            &[
                -1.0f32, -0.5, 0.0, 0.5, 1.0, 0.25, // R
                1.0, 0.5, 0.0, -0.5, -1.0, -0.25, // G
                -0.75, -0.25, 0.25, 0.75, -1.0, 1.0, // B
            ],
            &[1, 3, 1, 2, 3],
        );
        let legacy_native = DecodeSpy::new(native_output.clone());
        let expected = legacy_decode_one(&legacy_native, &latents);
        let native = DecodeSpy::new(native_output);
        let cancel = mlx_gen::CancelFlag::new();
        let decoded = decode_latents_via_seam(
            &native,
            None,
            Some(&gen_core::FLUX1_LATENT_SPACE),
            &latents,
            None,
            &cancel,
        )
        .unwrap();
        let got = decoded_to_image(&decoded).unwrap();
        assert_eq!(got, expected);
        assert_eq!(native.calls.get(), 1);

        let tiled_output = Array::from_slice(
            &[
                -1.0f32, -0.5, 0.0, 0.5, 1.0, 0.25, 1.0, 0.5, 0.0, -0.5, -1.0, -0.25, -0.75, -0.25,
                0.25, 0.75, -1.0, 1.0,
            ],
            &[1, 3, 1, 2, 3],
        );
        let tiling = mlx_gen::TilingConfig::spatial_only(16, 4);
        let legacy_tiled = DecodeSpy::new(tiled_output.clone());
        let legacy_tiled_decoded = legacy_tiled
            .decode_tiled(&latents, &tiling, Some(&cancel))
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let expected_tiled = decoded_to_image(&legacy_tiled_decoded).unwrap();
        let native_tiled = DecodeSpy::new(tiled_output);
        let decoded = decode_latents_via_seam(
            &native_tiled,
            None,
            Some(&gen_core::FLUX1_LATENT_SPACE),
            &latents,
            Some(&tiling),
            &cancel,
        )
        .unwrap();
        assert_eq!(decoded_to_image(&decoded).unwrap(), expected_tiled);
        assert_eq!(native_tiled.calls.get(), 1);

        let native = DecodeSpy::new(Array::zeros::<f32>(&[1, 3, 1, 2, 3]).unwrap());
        let pid_output = Array::from_slice(&vec![1.0f32; 3 * 4 * 5], &[1, 3, 4, 5]);
        let legacy_pid = DecodeSpy::new(pid_output.clone());
        let expected_pid = legacy_decode_one(&legacy_pid, &latents);
        let pid = DecodeSpy::new(pid_output);
        let decoded = decode_latents_via_seam(
            &native,
            Some(&pid),
            Some(&gen_core::FLUX1_LATENT_SPACE),
            &latents,
            None,
            &cancel,
        )
        .unwrap();
        let got = decoded_to_image(&decoded).unwrap();
        assert_eq!(got, expected_pid);
        assert_eq!(native.calls.get(), 0);
        assert_eq!(pid.calls.get(), 1);
    }

    #[test]
    fn validates_base_txt2img_request() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn schnell_rejects_guidance() {
        let model = Flux1::new_for_tests(FluxVariant::Schnell);
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("guidance is not supported"));
    }

    #[test]
    fn rejects_non_multiple_of_16() {
        // sc-12612: pin the exported stride so it cannot drift from the check SceneWorks ties to.
        assert_eq!(SIZE_MULTIPLE, 16);
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            width: 1025,
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("multiples of 16"));
        // A multiple of 8 that is NOT a multiple of SIZE_MULTIPLE (16) is still rejected.
        let off8 = GenerationRequest {
            prompt: "a red fox".into(),
            width: 1000, // 125×8
            ..Default::default()
        };
        assert!(model
            .validate(&off8)
            .unwrap_err()
            .to_string()
            .contains("multiples of 16"));
        // An on-stride in-range size (default 1024×1024) is accepted.
        let on = GenerationRequest {
            prompt: "a red fox".into(),
            guidance: Some(3.5),
            width: 1024,
            ..Default::default()
        };
        assert!(model.validate(&on).is_ok());
    }

    #[test]
    fn constants_match_expected_ids() {
        assert_eq!(FluxVariant::Schnell.id(), FLUX1_SCHNELL_ID);
        assert_eq!(FluxVariant::Dev.id(), FLUX1_DEV_ID);
    }

    // ---- epic 3621: XLabs IP-Adapter reference conditioning -----------------------------------

    fn tiny_image() -> Image {
        Image {
            width: 8,
            height: 8,
            pixels: vec![128u8; 8 * 8 * 3],
        }
    }

    fn reference(strength: Option<f32>) -> Conditioning {
        Conditioning::Reference {
            image: tiny_image(),
            strength,
        }
    }

    #[test]
    fn both_variants_advertise_reference_conditioning() {
        for v in [FluxVariant::Dev, FluxVariant::Schnell] {
            assert!(
                descriptor_for(v)
                    .capabilities
                    .conditioning
                    .contains(&mlx_gen::ConditioningKind::Reference),
                "{} must advertise Reference conditioning",
                v.id()
            );
        }
    }

    #[test]
    fn validate_accepts_single_reference_with_ip_knobs_on_dev() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            guidance: Some(3.5),
            true_cfg: Some(4.0),
            negative_prompt: Some("blurry".into()),
            conditioning: vec![reference(Some(0.7))],
            ..Default::default()
        };
        // true_cfg + negative_prompt are permitted alongside a reference on dev even though base
        // txt2img advertises neither.
        model.validate(&req).unwrap();
    }

    #[test]
    fn validate_rejects_true_cfg_without_reference() {
        // The IP knobs are scoped to the reference path; the plain dev path stays unchanged.
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            true_cfg: Some(4.0),
            ..Default::default()
        };
        assert!(model
            .validate(&req)
            .unwrap_err()
            .to_string()
            .contains("true_cfg is not supported"));
    }

    #[test]
    fn validate_rejects_true_cfg_on_schnell_even_with_reference() {
        // schnell is not guidance/CFG-capable, so true_cfg is rejected regardless of a reference.
        let model = Flux1::new_for_tests(FluxVariant::Schnell);
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            true_cfg: Some(4.0),
            conditioning: vec![reference(None)],
            ..Default::default()
        };
        assert!(model
            .validate(&req)
            .unwrap_err()
            .to_string()
            .contains("true_cfg is not supported"));
    }

    #[test]
    fn validate_rejects_multiple_references() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            conditioning: vec![reference(None), reference(None)],
            ..Default::default()
        };
        assert!(model
            .validate(&req)
            .unwrap_err()
            .to_string()
            .contains("single Reference"));
    }

    #[test]
    fn validate_rejects_reference_plus_pose_as_an_unsupported_combination() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a posed portrait".into(),
            conditioning: vec![
                reference(Some(0.7)),
                Conditioning::Control {
                    image: tiny_image(),
                    kind: mlx_gen::ControlKind::Pose,
                    scale: Some(1.0),
                },
            ],
            ..Default::default()
        };
        assert!(
            matches!(model.validate(&req), Err(gen_core::Error::Unsupported(message)) if message.contains("cannot be combined"))
        );
    }

    #[test]
    fn registry_freezes_the_native_mlx_schedule_without_opening_weights() {
        mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
            let registry = crate::provider_registry().unwrap();
            let spec = LoadSpec::new(WeightsSource::Dir("/missing/flux".into()));
            for variant in [FluxVariant::Schnell, FluxVariant::Dev] {
                let req = GenerationRequest {
                    prompt: "base".into(),
                    seed: Some(7),
                    count: 2,
                    steps: Some(4),
                    width: 640,
                    height: 480,
                    ..Default::default()
                };
                let plan = registry
                    .compile_render_plan(variant.id(), &spec, &req)
                    .unwrap()
                    .expect("both base variants compile before load");
                let native = crate::pipeline::build_linear_sigmas(
                    4,
                    640,
                    480,
                    variant.requires_sigma_shift(),
                )
                .unwrap();
                assert_eq!(plan.denoise_sigmas(), native);
                assert_eq!(plan.packed_tokens(), 1200);
                assert_eq!(plan.base_seed(), 7);
                plan.check_request(&descriptor_for(variant), &req).unwrap();
                plan.check_load(&gen_core::RenderLoadBinding::from_spec(&spec))
                    .unwrap();
            }
        });
    }

    /// Every FLUX.1 render route (base txt2img and Hires.fix passes, XLabs IP with and without
    /// real CFG, PuLID through the prepared-injector entry points, and the ControlNet route)
    /// encodes text through `encode_materialized`, which evaluates the T5/CLIP embeddings before
    /// returning, so the encode runs inside the Conditioning phase under both offload policies.
    /// A lazy `text.encode(` at a route would let the encoder run after Conditioning closes.
    /// Pinned against the shipped sources; nothing executes.
    #[test]
    fn every_route_evaluates_text_embeddings_inside_conditioning() {
        let code = |source: &str| -> String {
            let shipped = source.split("#[cfg(test)]\nmod tests").next().unwrap();
            shipped
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let model = code(include_str!("model.rs"));
        let control = code(include_str!("model_control.rs"));
        for (file, source, routes) in [("model.rs", &model, 4), ("model_control.rs", &control, 1)] {
            assert!(
                !source.contains("text.encode("),
                "{file}: a route encodes text lazily; use encode_materialized inside Conditioning"
            );
            assert_eq!(
                source.matches("text.encode_materialized(").count(),
                routes,
                "{file}: every residency encode closure must materialize its embeddings"
            );
        }
        // The non-started routes open and close Conditioning around the materializing encode.
        assert!(model.contains("run.encode(|| text.encode_materialized(&req.prompt))"));
        assert!(model.contains("run.encode(encode_text)"));
        let helper = &model[model.find("fn encode_materialized(").unwrap()..];
        let helper = &helper[..helper.find("\n    }").unwrap()];
        assert!(
            helper.contains("mlx_rs::transforms::eval([&prompt_embeds, &pooled_prompt_embeds])"),
            "encode_materialized must evaluate both embeddings before returning: {helper}"
        );

        // PiD's Gemma caption encode is text conditioning as well. The decoder is minted only by
        // `conditioned_pid_decoder`, which evaluates the caption, and each heavy closure mints it
        // before Conditioning closes; a PiD request holds Conditioning open across the handoff.
        assert_eq!(model.matches("resolve_pid_decoder_at_sigma(").count(), 1);
        let pid = &model[model.find("fn conditioned_pid_decoder(").unwrap()..];
        let pid = &pid[..pid.find("\n    }").unwrap()];
        assert!(
            pid.contains("resolve_pid_decoder_at_sigma(") && pid.contains("materialize_caption()")
        );
        assert_eq!(
            model
                .matches("let hold_conditioning = conditioning_started || req.use_pid || req.source_image.is_some();")
                .count(),
            1
        );
        assert_eq!(
            model
                .matches("let hold_conditioning = conditioning_started || req.use_pid;")
                .count(),
            1
        );
        let mut closures = 0;
        for (at, _) in
            model.match_indices("self.conditioned_pid_decoder(run, req, heavy.pid.as_ref())?;")
        {
            let rest = &model[at..];
            let close = rest
                .find("run.end_conditioning()?;")
                .expect("Conditioning closes after PiD");
            let denoise = rest
                .find("self.run_denoise_with(")
                .expect("denoise follows");
            assert!(
                close < denoise,
                "PiD decoder must be prepared before Conditioning closes"
            );
            closures += 1;
        }
        assert_eq!(
            closures, 2,
            "both prepared-injector heavy closures prepare PiD in Conditioning"
        );
    }

    #[test]
    fn pre_load_plan_refuses_invalid_inputs_without_opening_any_weights() {
        let registry = crate::provider_registry().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir("/missing/flux".into()));
        let mut req = GenerationRequest {
            prompt: "base".into(),
            width: 321,
            ..Default::default()
        };
        for variant in [FluxVariant::Dev, FluxVariant::Schnell] {
            let error = registry
                .compile_render_plan(variant.id(), &spec, &req)
                .unwrap_err();
            assert!(error.to_string().contains("multiples of 16"), "{error}");
        }
        req.width = 512;
        let invalid_overlay = spec
            .clone()
            .with_control(WeightsSource::Dir("/missing/control".into()));
        assert!(registry
            .compile_render_plan(FluxVariant::Dev.id(), &invalid_overlay, &req)
            .is_err());
        let file_overlay = spec
            .clone()
            .with_ip_adapter(WeightsSource::File("/missing/ip".into()));
        assert!(registry
            .compile_render_plan(FluxVariant::Dev.id(), &file_overlay, &req)
            .is_err());
        req.conditioning = vec![reference(Some(0.5))];
        let error = registry
            .compile_render_plan(FluxVariant::Dev.id(), &spec, &req)
            .unwrap_err();
        assert!(error.to_string().contains("IP-adapter"), "{error}");
        // A complete XLabs route now compiles before load and freezes its reference bytes.
        let spec = spec.with_ip_adapter(WeightsSource::Dir("/missing/ip".into()));
        let plan = registry
            .compile_render_plan(FluxVariant::Dev.id(), &spec, &req)
            .unwrap()
            .expect("XLabs reference route needs a pre-load plan");
        plan.check_request(&descriptor_for(FluxVariant::Dev), &req)
            .unwrap();
        if let Conditioning::Reference { image, .. } = &mut req.conditioning[0] {
            image.pixels[0] ^= 1;
        }
        assert!(plan
            .check_request(&descriptor_for(FluxVariant::Dev), &req)
            .is_err());
        req.conditioning.push(reference(None));
        assert!(registry
            .compile_render_plan(FluxVariant::Dev.id(), &spec, &req)
            .is_err());
    }

    /// The complete XLabs IP route refuses zero steps and an off-range size in the pre-load
    /// compiler at a missing snapshot and adapter, so no weight is ever opened.
    #[test]
    fn xlabs_ip_plan_refuses_zero_steps_and_invalid_sizes_before_load() {
        let registry = crate::provider_registry().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir("/missing/flux".into()))
            .with_ip_adapter(WeightsSource::Dir("/missing/ip".into()));
        let valid = GenerationRequest {
            prompt: "a portrait".into(),
            width: 512,
            height: 512,
            steps: Some(4),
            conditioning: vec![reference(Some(0.5))],
            ..Default::default()
        };
        registry
            .compile_render_plan(FluxVariant::Dev.id(), &spec, &valid)
            .unwrap()
            .expect("XLabs reference route needs a pre-load plan");
        for (width, steps, expected) in [
            (512, Some(0), "steps must be >= 1"),
            (321, Some(4), "multiples of 16"),
            (128, Some(4), "outside supported range"),
        ] {
            let req = GenerationRequest {
                width,
                steps,
                ..valid.clone()
            };
            let direct = compile_base(FluxVariant::Dev, &spec, &req)
                .unwrap_err()
                .to_string();
            assert!(direct.contains(expected), "{width}: {direct}");
            let registered = registry
                .compile_render_plan(FluxVariant::Dev.id(), &spec, &req)
                .unwrap_err()
                .to_string();
            assert!(registered.contains(expected), "{width}: {registered}");
        }
    }

    /// Hires.fix with the XLabs IP route: pass 1 renders at the base size, pass 2 renders from
    /// noise at the upscaled size conditioned on pass 1's stretched output at strength
    /// 1 − denoise. Both passes compile before load; pass 2's reference is bound by its declared
    /// size and strength because its pixels exist only after pass 1.
    #[test]
    fn xlabs_ip_hires_fix_compiles_two_passes_before_load() {
        let registry = crate::provider_registry().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir("/missing/flux".into()))
            .with_ip_adapter(WeightsSource::Dir("/missing/ip".into()));
        let first = GenerationRequest {
            prompt: "a portrait".into(),
            width: 512,
            height: 512,
            steps: Some(4),
            seed: Some(7),
            ..Default::default()
        };
        let refine = GenerationRequest {
            width: 768,
            height: 768,
            steps: Some(3),
            ..first.clone()
        };
        let hires = gen_core::ProducedReference {
            from_pass: 0,
            width: 768,
            height: 768,
            strength: Some(1.0 - 0.45),
        };
        let compile = |refine: &GenerationRequest, produced| {
            registry.compile_multi_pass_render_plan(
                FluxVariant::Dev.id(),
                &spec,
                &[
                    gen_core::PlannedPassRequest {
                        request: &first,
                        produced_reference: None,
                    },
                    gen_core::PlannedPassRequest {
                        request: refine,
                        produced_reference: Some(produced),
                    },
                ],
            )
        };
        let plan = compile(&refine, hires)
            .unwrap()
            .expect("the XLabs Hires.fix route needs a pre-load plan");
        let [base, refined] = plan.passes() else {
            panic!("two passes");
        };
        base.check_request(&descriptor_for(FluxVariant::Dev), &first)
            .unwrap();
        assert_eq!(base.output_size(), (512, 512));
        assert_eq!(refined.output_size(), (768, 768));
        assert_eq!(refined.base_seed(), 7);
        assert_eq!(plan.total_executed_steps(), 7);
        let native = build_sigmas_with(3, 768, 768, true, None).unwrap();
        assert_eq!(refined.denoise_sigmas(), native);
        plan.check_load(&gen_core::RenderLoadBinding::from_spec(&spec))
            .unwrap();

        // The stretched pass-1 output (any pixels) of the declared size binds; others refuse.
        let with_reference = |size: u32, strength: Option<f32>| GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: size,
                    height: size,
                    pixels: vec![17; (size * size * 3) as usize],
                },
                strength,
            }],
            ..refine.clone()
        };
        refined
            .check_request(
                &descriptor_for(FluxVariant::Dev),
                &with_reference(768, hires.strength),
            )
            .unwrap();
        for wrong in [
            with_reference(512, hires.strength),
            with_reference(768, Some(0.45)),
            refine.clone(),
        ] {
            assert!(refined
                .check_request(&descriptor_for(FluxVariant::Dev), &wrong)
                .is_err());
        }

        // Pass-2 refusals happen before load: off-grid, below-minimum and zero steps.
        for (width, steps, expected) in [
            (770, Some(3), "multiples of 16"),
            (128, Some(3), "outside supported range"),
            (768, Some(0), "steps must be >= 1"),
        ] {
            let bad = GenerationRequest {
                width,
                steps,
                ..refine.clone()
            };
            let error = compile(&bad, gen_core::ProducedReference { width, ..hires })
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{width}: {error}");
        }
        // The IP pass needs its adapter in the shared load contract.
        let no_adapter = LoadSpec::new(WeightsSource::Dir("/missing/flux".into()));
        let error = registry
            .compile_multi_pass_render_plan(
                FluxVariant::Dev.id(),
                &no_adapter,
                &[
                    gen_core::PlannedPassRequest {
                        request: &first,
                        produced_reference: None,
                    },
                    gen_core::PlannedPassRequest {
                        request: &refine,
                        produced_reference: Some(hires),
                    },
                ],
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("IP-adapter"), "{error}");

        // Single-pass plans are unchanged: pass 0, digest-bound reference.
        let single = registry
            .compile_render_plan(
                FluxVariant::Dev.id(),
                &spec,
                &with_reference(768, hires.strength),
            )
            .unwrap()
            .unwrap();
        assert_eq!(single.pass_index(), 0);
        assert_eq!(single.produced_reference(), None);
        let mut changed = with_reference(768, hires.strength);
        if let Conditioning::Reference { image, .. } = &mut changed.conditioning[0] {
            image.pixels[0] ^= 1;
        }
        assert!(single
            .check_request(&descriptor_for(FluxVariant::Dev), &changed)
            .is_err());
    }

    #[test]
    fn generate_reference_without_ip_adapter_errors() {
        // No `LoadSpec::ip_adapter` → a reference request errors loudly before touching the (absent)
        // transformer, so the descriptor never advertises a path that silently no-ops.
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            conditioning: vec![reference(Some(0.7))],
            ..Default::default()
        };
        let err = model.generate(&req, &mut |_| {}).unwrap_err().to_string();
        assert!(err.contains("needs an IP-adapter"), "got: {err}");
    }

    // ---- sc-2908: sampler capability surface + few-step profile -----------------------------

    #[test]
    fn dev_advertises_hyper_schnell_does_not() {
        // Both expose the curated unified-framework integrator menu (epic 7114) + the base
        // `flow_match` profile; Hyper-FLUX is a FLUX.1-dev LoRA, so only dev adds the `hyper` profile.
        let dev = descriptor_for(FluxVariant::Dev).capabilities.samplers;
        let schnell = descriptor_for(FluxVariant::Schnell).capabilities.samplers;
        assert!(dev.contains(&HYPER_SAMPLER), "dev exposes hyper");
        assert!(
            !schnell.contains(&HYPER_SAMPLER),
            "schnell does not expose the dev-only hyper profile"
        );
        for s in [DEFAULT_SAMPLER, "euler", "dpmpp_2m", "uni_pc"] {
            assert!(dev.contains(&s), "dev exposes {s}");
            assert!(schnell.contains(&s), "schnell exposes {s}");
        }
    }

    #[test]
    fn validate_accepts_base_and_hyper_on_dev() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        for s in [DEFAULT_SAMPLER, HYPER_SAMPLER] {
            let req = GenerationRequest {
                prompt: "a red fox".into(),
                guidance: Some(3.5),
                sampler: Some(s.into()),
                ..Default::default()
            };
            assert!(
                model.validate(&req).is_ok(),
                "sampler {s:?} should be accepted on dev"
            );
        }
        // An unset sampler is the base flow-match path.
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        assert!(model.validate(&req).is_ok());
    }

    #[test]
    fn validate_rejects_unadvertised_scheduler() {
        // F-100: an unsupported scheduler was silently accepted before. epic 7114 scheduler axis: the
        // curated names ("karras", …) + the "linear" native alias now pass; an unknown name is rejected.
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let err = model
            .validate(&GenerationRequest {
                prompt: "a red fox".into(),
                guidance: Some(3.5),
                scheduler: Some("not_a_real_scheduler".into()),
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported scheduler"), "got: {err}");
        for ok in ["linear", "karras", "sgm_uniform"] {
            model
                .validate(&GenerationRequest {
                    prompt: "a red fox".into(),
                    guidance: Some(3.5),
                    scheduler: Some(ok.into()),
                    ..Default::default()
                })
                .unwrap_or_else(|e| panic!("{ok} should validate: {e}"));
        }
    }

    #[test]
    fn validate_rejects_hyper_on_schnell_and_unknown_samplers() {
        // `hyper` is dev-only — schnell does not advertise it, so it is rejected, not downgraded.
        let schnell = Flux1::new_for_tests(FluxVariant::Schnell);
        let err = schnell
            .validate(&GenerationRequest {
                prompt: "a red fox".into(),
                sampler: Some(HYPER_SAMPLER.into()),
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported sampler"), "got: {err}");
        // A sampler outside the advertised menu is rejected on dev too. (`lcm`/`euler` are now in the
        // curated menu (epic 7114), so the rejected names are the qwen-only `lightning` profile, a
        // non-curated solver name `tcd`, and outright garbage.)
        let dev = Flux1::new_for_tests(FluxVariant::Dev);
        for bad in ["lightning", "tcd", "nonsense"] {
            let err = dev
                .validate(&GenerationRequest {
                    prompt: "a red fox".into(),
                    guidance: Some(3.5),
                    sampler: Some(bad.into()),
                    ..Default::default()
                })
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("unsupported sampler"),
                "sampler {bad:?}: {err}"
            );
        }
    }

    #[test]
    fn hyper_profile_defaults_are_eight_steps_guidance_3_5() {
        // The few-step profile: 8 steps at guidance 3.5 (the Hyper-FLUX.1-dev recommendation).
        assert_eq!(profile_defaults(FluxVariant::Dev, HYPER_SAMPLER), (8, 3.5));
        // The base path keeps the variant's own defaults (dev 25, schnell 4).
        assert_eq!(
            profile_defaults(FluxVariant::Dev, DEFAULT_SAMPLER),
            (
                FluxVariant::Dev.default_steps(),
                crate::config::DEFAULT_GUIDANCE
            )
        );
        assert_eq!(
            profile_defaults(FluxVariant::Schnell, DEFAULT_SAMPLER),
            (
                FluxVariant::Schnell.default_steps(),
                crate::config::DEFAULT_GUIDANCE
            )
        );
    }

    // ── sc-10840: weight-free, default-run proof that FLUX.1's dispatch HONORS `offload_policy`.
    // `build_residency` points at a non-existent snapshot *directory* (so the up-front precision/
    // single-file guard passes) and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the T5 + CLIP encoders from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The real-weight A/B is deferred (weights not on disk).
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/flux1-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let res = build_residency(
            FluxVariant::Dev,
            &missing_snapshot_spec(OffloadPolicy::Sequential),
            None,
        )
        .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        drop(res);
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let err = build_residency(
            FluxVariant::Dev,
            &missing_snapshot_spec(OffloadPolicy::Resident),
            None,
        )
        .err()
        .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("port plan"),
            "expected an eager-load failure, not the up-front guard: {msg}"
        );
    }

    fn exact_runtime_inventory(
        tmp: &tempfile::TempDir,
    ) -> (
        std::path::PathBuf,
        LoadSpec,
        crate::artifact_inventory::PackedArtifactInventory,
    ) {
        let root = tmp.path().join(format!(
            "flux-runtime-stream-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        crate::artifact_inventory::write_test_snapshot(&root, None);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        let inventory =
            crate::artifact_inventory::verified_stream_inventory(crate::FLUX1_DEV_ID, &spec)
                .unwrap();
        (root, spec, inventory)
    }

    #[test]
    fn runtime_window_gate_is_exact_and_generation_boundary_rechecks_inventory() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec, inventory) = exact_runtime_inventory(&tmp);
        let mut model = Flux1::new_for_tests(FluxVariant::Dev);
        model.memory_strategy = crate::memory_strategy::memory_strategy_contract_with_inventory(
            crate::FLUX1_DEV_ID,
            &spec,
            Some(&inventory),
        )
        .unwrap();
        model.loaded_spec = spec;
        model.stream_inventory = Some(inventory);

        let selected = gen_core::GenerationMemory {
            stage_residency: true,
            stream_transformer_blocks: true,
            transformer_window_size: Some(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE),
            transformer_window_component: Some(gen_core::TransformerComponent::Dit),
            ..Default::default()
        };
        let request = GenerationRequest {
            prompt: "stream".into(),
            memory: Some(selected),
            ..Default::default()
        };
        assert_eq!(
            model.requested_transformer_window(&request, None).unwrap(),
            Some(1)
        );

        let mut wrong_stage = request.clone();
        wrong_stage.memory.as_mut().unwrap().stage_residency = false;
        assert!(model
            .requested_transformer_window(&wrong_stage, None)
            .is_err());
        let mut wrong_window = request.clone();
        wrong_window
            .memory
            .as_mut()
            .unwrap()
            .transformer_window_size = Some(2);
        assert!(model
            .requested_transformer_window(&wrong_window, None)
            .is_err());
        let mut wrong_component = request.clone();
        wrong_component
            .memory
            .as_mut()
            .unwrap()
            .transformer_window_component = Some(gen_core::TransformerComponent::TextEncoder);
        assert!(model
            .requested_transformer_window(&wrong_component, None)
            .is_err());
        let mut pid = request.clone();
        pid.use_pid = true;
        assert!(model.requested_transformer_window(&pid, None).is_err());

        std::fs::write(root.join("tokenizer_2/tokenizer.json"), b"tokenizer-v2").unwrap();
        assert!(model.finish_streamed_generation(true, Ok(())).is_err());
        assert!(matches!(
            model.finish_streamed_generation::<()>(
                true,
                Err(Error::ProcessPoisoned("device lost".into()))
            ),
            Err(Error::ProcessPoisoned(ref detail)) if detail == "device lost"
        ));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn sw6939_mlx_flux_inactive_hook_is_byte_identical_and_two_phase_low_res_executes_conformance()
    {
        use gen_core::render::RenderPhaseIdentity;
        use gen_core::sampling::{FlowModelSampling, TimestepConvention};
        use gen_core::{
            BoundaryState, EstimateSource, LowResStartSpec, MemoryGeometry, MemoryPhase,
            MemoryRequestScope, MemoryRunOutcome,
        };
        use mlx_gen::request_scope::{MlxRequestScopeConfig, MlxRequestScopeCore, MlxScopeCleanup};
        use std::cell::RefCell;

        let base_req = GenerationRequest {
            prompt: "a lighthouse in a storm".into(),
            width: 512,
            height: 512,
            count: 1,
            steps: Some(4),
            seed: Some(42),
            sampler: Some("dpmpp_2m".into()),
            ..Default::default()
        };
        let active_req = GenerationRequest {
            low_res_start: Some(LowResStartSpec::new(2, Some(0.5))),
            ..base_req.clone()
        };

        // 1. Schedule & mu invariance across both flux1_dev and flux1_schnell.
        for variant in [FluxVariant::Dev, FluxVariant::Schnell] {
            let plan_off = compile_trajectory(variant, &base_req).unwrap();
            let plan_on = compile_trajectory(variant, &active_req).unwrap();
            assert_eq!(
                plan_off.denoise_sigmas(),
                plan_on.denoise_sigmas(),
                "{variant:?}: full-pass sigma schedule and mu must stay invariant under low_res_start"
            );
            assert_eq!(plan_off.denoise_spans(), vec![0..4]);
            assert_eq!(plan_on.denoise_spans(), vec![0..2, 2..4]);
            assert_eq!(plan_on.phase_output_size(0), (256, 256));
            assert_eq!(plan_on.phase_output_size(1), (512, 512));
        }

        // 2. Ordinary single-source img2img strength truncation resolves M = executed_steps before checking K.
        let src_img = Image {
            width: 64,
            height: 64,
            pixels: vec![128; 64 * 64 * 3],
        };
        // 10 full steps with strength 0.6 -> start_step = 6 -> M = 4 executed steps.
        let img2img_ok = GenerationRequest {
            steps: Some(10),
            strength: Some(0.6),
            source_image: Some(src_img.clone()),
            low_res_start: Some(LowResStartSpec::new(3, Some(0.5))),
            ..base_req.clone()
        };
        let img2img_plan = compile_trajectory(FluxVariant::Dev, &img2img_ok).unwrap();
        assert_eq!(img2img_plan.executed_steps(), 4);
        assert_eq!(img2img_plan.denoise_spans(), vec![0..3, 3..4]);

        // K = 4 >= M = 4 after strength truncation is refused with a typed error!
        let img2img_bad_k = GenerationRequest {
            low_res_start: Some(LowResStartSpec::new(4, Some(0.5))),
            ..img2img_ok
        };
        assert!(validate_request(&descriptor_dev(), &img2img_bad_k).is_err());

        // 3. BoundaryState::XSigma is refused by transition_mlx_flux_boundary_state (cannot double-noise).
        let plan_on = compile_trajectory(FluxVariant::Dev, &active_req).unwrap();
        let low_res_plan = *plan_on.low_res_start().unwrap();
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let noisy_arr = create_noise(42, 256, 256).unwrap();
        assert!(transition_mlx_flux_boundary_state(
            BoundaryState::XSigma(noisy_arr),
            &low_res_plan,
            plan_on.denoise_sigmas()[2],
            &ms,
            42,
            0,
            0,
            0,
        )
        .is_err());

        // 4. Inactive hook byte-identity: execute_flux_denoise_trajectory with low_res_start: None
        // produces byte-identical latents, evaluation trace, and progress to run_flow_sampler_with_latent_hook_spans.
        let plan_off = compile_trajectory(FluxVariant::Dev, &base_req).unwrap();
        let init_full = create_noise(42, 512, 512).unwrap();
        let trace_off = RefCell::new(Vec::<(usize, u32, u32, usize, u32)>::new());
        let mut progress_off = Vec::new();
        let run_off = gen_core::RenderRun::new(&plan_off, &base_req.cancel, None);
        run_off.encode(|| Ok::<_, gen_core::Error>(())).unwrap();
        let out_off = run_off
            .images_with_denoise_phases(
                &mut |p| progress_off.push(p),
                |_| Ok::<_, Error>(init_full.clone()),
                |seed, lat, prog| {
                    execute_flux_denoise_trajectory(
                        &run_off,
                        &base_req,
                        seed,
                        lat,
                        prog,
                        |x_in, step_idx, timestep, _guidance, pw, ph| {
                            trace_off.borrow_mut().push((
                                step_idx,
                                pw,
                                ph,
                                x_in.shape()[1] as usize,
                                timestep.to_bits(),
                            ));
                            Ok(multiply(x_in, scalar(0.15))?)
                        },
                    )
                },
                |_, lat, _| Ok(lat),
            )
            .unwrap()
            .remove(0);

        let trace_ref = RefCell::new(Vec::<(usize, u32, u32, usize, u32)>::new());
        let mut progress_ref = Vec::new();
        let run_ref = gen_core::RenderRun::new(&plan_off, &base_req.cancel, None);
        run_ref.encode(|| Ok::<_, gen_core::Error>(())).unwrap();
        let out_ref = run_ref
            .images_with_denoise_phases(
                &mut |p| progress_ref.push(p),
                |_| Ok::<_, Error>(init_full.clone()),
                |seed, lat, prog| {
                    let mut eval_idx = 0usize;
                    mlx_gen::run_flow_sampler_with_latent_hook_spans(
                        SamplerRun {
                            sampler_name: Some(plan_off.sampler()),
                            sigmas: plan_off.denoise_sigmas(),
                            seed,
                            cancel: &base_req.cancel,
                            on_progress: prog,
                        },
                        TimestepConvention::Sigma,
                        lat,
                        &plan_off.denoise_spans(),
                        Some(&run_ref),
                        |_, _| {},
                        |x_in, timestep| {
                            let step_idx = eval_idx;
                            eval_idx += 1;
                            trace_ref.borrow_mut().push((
                                step_idx,
                                512,
                                512,
                                x_in.shape()[1] as usize,
                                timestep.to_bits(),
                            ));
                            Ok(multiply(x_in, scalar(0.15))?)
                        },
                    )
                },
                |_, lat, _| Ok(lat),
            )
            .unwrap()
            .remove(0);

        mlx_rs::transforms::eval([&out_off, &out_ref]).unwrap();
        assert_eq!(out_off.as_slice::<f32>(), out_ref.as_slice::<f32>());
        assert_eq!(*trace_off.borrow(), *trace_ref.borrow());
        assert_eq!(progress_off, progress_ref);

        // 5. Active two-phase execution with MlxRequestScopeCore:
        // Phase 0 (steps 0..2) runs at 256x256 (256 tokens), boundary K=2 performs read-only endpoint
        // evaluation at 256x256, transitions in fp32 to 512x512 (1024 tokens), resets Dpmpp2m history,
        // and Phase 1 (steps 2..4) runs at 512x512 (1024 tokens).
        struct RecordingMlxScope {
            core: MlxRequestScopeCore,
            phase_identities: Vec<(MemoryPhase, RenderPhaseIdentity)>,
        }
        impl MemoryRequestScope for RecordingMlxScope {
            fn configure_request(&mut self, req: &mut GenerationRequest) -> gen_core::Result<()> {
                self.core.configure_request(req)
            }
            fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
                self.core.enter_phase(phase)
            }
            fn enter_render_phase(
                &mut self,
                phase: MemoryPhase,
                identity: &RenderPhaseIdentity,
            ) -> gen_core::Result<()> {
                self.core.enter_render_phase(phase, identity)?;
                self.phase_identities.push((phase, *identity));
                Ok(())
            }
            fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
                self.core.leave_phase(phase)
            }
            fn configure_decode(
                &mut self,
                tile_edge: u32,
                overlap: u32,
                geometry: MemoryGeometry,
            ) -> gen_core::Result<()> {
                self.core.configure_decode(tile_edge, overlap, geometry)
            }
            fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
                self.core.configure_attention(chunk_size)
            }
            fn materialize_transformer_window(
                &mut self,
                first_block: u32,
                block_count: u32,
            ) -> gen_core::Result<()> {
                self.core
                    .materialize_transformer_window(first_block, block_count)
            }
            fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
                self.core.finish(outcome)
            }
        }

        let scope_cfg = MlxRequestScopeConfig::new(
            "flux1_dev",
            MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            None,
            false,
            57,
            |_, _, _| Ok(()),
        )
        .unwrap();
        let mut scope = RecordingMlxScope {
            core: MlxRequestScopeCore::with_cleanup(scope_cfg, MlxScopeCleanup::None),
            phase_identities: Vec::new(),
        };

        let trace_on = RefCell::new(Vec::<(usize, u32, u32, usize)>::new());
        let mut progress_on = Vec::new();
        let init_low = create_noise(42, 256, 256).unwrap();
        let run_on = gen_core::RenderRun::new(&plan_on, &active_req.cancel, Some(&mut scope));
        run_on.encode(|| Ok::<_, gen_core::Error>(())).unwrap();
        let out_on = run_on
            .images_with_denoise_phases(
                &mut |p| progress_on.push(p),
                |_| Ok::<_, Error>(init_low.clone()),
                |seed, lat, prog| {
                    execute_flux_denoise_trajectory(
                        &run_on,
                        &active_req,
                        seed,
                        lat,
                        prog,
                        |x_in, step_idx, _timestep, _guidance, pw, ph| {
                            // Verify per-phase RoPE derivation matches the phase geometry and token count.
                            let rope = crate::transformer::FluxRope::new()
                                .forward(512, (ph / 16) as usize, (pw / 16) as usize)
                                .unwrap();
                            let expected_tokens = ((ph / 16) * (pw / 16)) as usize;
                            assert_eq!(x_in.shape()[1] as usize, expected_tokens);
                            assert_eq!(rope.cos.shape()[0] as usize, 512 + expected_tokens);
                            trace_on.borrow_mut().push((
                                step_idx,
                                pw,
                                ph,
                                x_in.shape()[1] as usize,
                            ));
                            Ok(multiply(x_in, scalar(0.15))?)
                        },
                    )
                },
                |_, lat, _| Ok(lat),
            )
            .unwrap()
            .remove(0);
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        mlx_rs::transforms::eval([&out_on]).unwrap();
        assert_eq!(out_on.shape(), &[1, 1024, 64]);
        // Progress emitted exactly 4 outer steps (1..=4); the read-only boundary evaluation did not emit progress.
        assert_eq!(
            progress_on,
            vec![
                Progress::Step {
                    current: 1,
                    total: 4
                },
                Progress::Step {
                    current: 2,
                    total: 4
                },
                Progress::Step {
                    current: 3,
                    total: 4
                },
                Progress::Step {
                    current: 4,
                    total: 4
                },
            ]
        );
        // Trace: 2 steps at 256x256 (256 tokens) + 1 read-only endpoint eval at step K=2 (256x256) + 2 steps at 512x512 (1024 tokens).
        assert_eq!(
            *trace_on.borrow(),
            vec![
                (0, 256, 256, 256),
                (1, 256, 256, 256),
                (2, 256, 256, 256),
                (2, 512, 512, 1024),
                (3, 512, 512, 1024),
            ]
        );
        // Scope recorded Conditioning(512x512), Denoise phase 0 (256x256), Denoise phase 1 (512x512), Decode(512x512).
        let denoise_identities: Vec<RenderPhaseIdentity> = scope
            .phase_identities
            .iter()
            .filter_map(|(phase, id)| (*phase == MemoryPhase::Denoise).then_some(*id))
            .collect();
        assert_eq!(denoise_identities.len(), 2);
        assert_eq!(
            (
                denoise_identities[0].pass,
                denoise_identities[0].phase,
                denoise_identities[0].geometry.width,
                denoise_identities[0].geometry.height
            ),
            (0, 0, 256, 256)
        );
        assert_eq!(
            (
                denoise_identities[1].pass,
                denoise_identities[1].phase,
                denoise_identities[1].geometry.width,
                denoise_identities[1].geometry.height
            ),
            (0, 1, 512, 512)
        );
        let _ = EstimateSource::EndpointEvaluation;
    }

    #[test]
    fn sw6940_mlx_flux_endpoint_and_previous_evaluation_end_to_end_and_provenance_report() {
        use gen_core::{EstimateSource, EvaluationKind, LowResStartSpec};
        use std::cell::RefCell;

        for (sampler_name, expected_p0_evals, expected_p1_evals) in [
            ("euler", 2usize, 2usize),
            ("dpmpp_2m", 2usize, 2usize),
            ("heun", 4usize, 3usize),
            ("dpmpp_sde", 4usize, 3usize),
        ] {
            let make_req = |source: EstimateSource| GenerationRequest {
                prompt: "a lighthouse in a storm".into(),
                width: 512,
                height: 512,
                count: 1,
                steps: Some(4),
                seed: Some(42),
                sampler: Some(sampler_name.into()),
                low_res_start: Some(
                    LowResStartSpec::new(2, Some(0.5)).with_estimate_source(source),
                ),
                ..Default::default()
            };

            let req_endpoint = make_req(EstimateSource::EndpointEvaluation);
            let req_previous = make_req(EstimateSource::PreviousEvaluation);
            let plan_endpoint = compile_trajectory(FluxVariant::Dev, &req_endpoint).unwrap();
            let plan_previous = compile_trajectory(FluxVariant::Dev, &req_previous).unwrap();

            let report_endpoint = plan_endpoint
                .low_res_start_execution_report("f32", 4, "base_dit")
                .unwrap()
                .unwrap();
            let report_previous = plan_previous
                .low_res_start_execution_report("f32", 4, "base_dit")
                .unwrap()
                .unwrap();

            assert_eq!(report_endpoint.estimator, "endpoint_evaluation_v1");
            assert_eq!(
                report_endpoint.estimate_provenance.evaluation_kind,
                EvaluationKind::BoundaryEstimate
            );
            assert_eq!(report_endpoint.estimate_provenance.step_index, 2);
            assert_eq!(
                report_endpoint.estimate_provenance.sigma.to_bits(),
                plan_endpoint.denoise_sigmas()[2].to_bits()
            );
            assert_eq!(report_endpoint.boundary_evaluations, 1);
            assert_eq!(
                report_endpoint.phase0_step_evaluations,
                expected_p0_evals as u32
            );
            assert_eq!(
                report_endpoint.phase1_step_evaluations,
                expected_p1_evals as u32
            );
            assert_eq!(
                report_endpoint.total_model_evaluations,
                (expected_p0_evals + 1 + expected_p1_evals) as u32
            );

            assert_eq!(report_previous.estimator, "previous_evaluation_v1");
            assert_eq!(
                report_previous.estimate_provenance.evaluation_kind,
                EvaluationKind::GuidedPrimaryX0
            );
            assert_eq!(report_previous.estimate_provenance.step_index, 1);
            assert_eq!(
                report_previous.estimate_provenance.sigma.to_bits(),
                plan_previous.denoise_sigmas()[1].to_bits()
            );
            assert_eq!(report_previous.boundary_evaluations, 0);
            assert_eq!(
                report_previous.total_model_evaluations,
                (expected_p0_evals + expected_p1_evals) as u32
            );
            assert_eq!(
                report_previous
                    .estimate_provenance
                    .retained_primary_estimate_bytes,
                256 * 64 * 4
            );

            let run_policy = |req: &GenerationRequest, plan: &gen_core::RenderPlan| {
                let trace = RefCell::new(Vec::<(usize, u32, u32, usize, u32)>::new());
                let mut progress = Vec::new();
                let init_low = create_noise(42, 256, 256).unwrap();
                let run = gen_core::RenderRun::new(plan, &req.cancel, None);
                run.encode(|| Ok::<_, gen_core::Error>(())).unwrap();
                let out = run
                    .images_with_denoise_phases(
                        &mut |p| progress.push(p),
                        |_| Ok::<_, Error>(init_low.clone()),
                        |seed, lat, prog| {
                            execute_flux_denoise_trajectory(
                                &run,
                                req,
                                seed,
                                lat,
                                prog,
                                |x_in, step_idx, timestep, _guidance, pw, ph| {
                                    trace.borrow_mut().push((
                                        step_idx,
                                        pw,
                                        ph,
                                        x_in.shape()[1] as usize,
                                        timestep.to_bits(),
                                    ));
                                    Ok(multiply(x_in, scalar(0.15))?)
                                },
                            )
                        },
                        |_, lat, _| Ok(lat),
                    )
                    .unwrap()
                    .remove(0);
                mlx_rs::transforms::eval([&out]).unwrap();
                (out.as_slice::<f32>().to_vec(), trace.into_inner(), progress)
            };

            let (out_ep, trace_ep, prog_ep) = run_policy(&req_endpoint, &plan_endpoint);
            let (out_pr, trace_pr, prog_pr) = run_policy(&req_previous, &plan_previous);

            assert_eq!(out_ep.len(), 1024 * 64);
            assert_eq!(out_pr.len(), 1024 * 64);
            assert!(out_ep.iter().all(|v| v.is_finite()));
            assert!(out_pr.iter().all(|v| v.is_finite()));
            assert_ne!(
                out_ep, out_pr,
                "{sampler_name}: EndpointEvaluation (sigma_K) and PreviousEvaluation (sigma_{{K-1}}) must produce distinct trajectories"
            );
            // Identical outer progress and identical Phase-0 evaluation trace before boundary K.
            assert_eq!(prog_ep, prog_pr);
            assert_eq!(
                &trace_ep[..expected_p0_evals],
                &trace_pr[..expected_p0_evals],
                "{sampler_name}: Phase-0 evaluations must be bit-identical between EndpointEvaluation and PreviousEvaluation"
            );
            assert_eq!(trace_ep.len(), expected_p0_evals + 1 + expected_p1_evals);
            assert_eq!(trace_pr.len(), expected_p0_evals + expected_p1_evals);
        }
    }

    #[test]
    fn flux1_rejects_reference_conditioning_with_low_res_start_on_base_and_ip_adapter() {
        use mlx_gen::gen_core::{Conditioning, Image, LowResStartSpec};

        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_ip_adapter(WeightsSource::File("/nonexistent/ip.safetensors".into()));
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            width: 1024,
            height: 1024,
            steps: Some(4),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 64,
                    height: 64,
                    pixels: vec![0; 64 * 64 * 3],
                },
                strength: Some(0.75),
            }],
            low_res_start: Some(LowResStartSpec::new(2, Some(0.5))),
            ..Default::default()
        };

        for (variant, base_desc, ip_desc) in [
            (
                FluxVariant::Dev,
                descriptor_dev(),
                descriptor_dev_ip_adapter(),
            ),
            (
                FluxVariant::Schnell,
                descriptor_schnell(),
                descriptor_schnell_ip_adapter(),
            ),
        ] {
            assert!(base_desc
                .capabilities
                .validate_request(base_desc.id, &req)
                .is_err());
            assert!(ip_desc
                .capabilities
                .validate_request(ip_desc.id, &req)
                .is_err());
            assert!(validate_request(&base_desc, &req).is_err());
            assert!(validate_request(&ip_desc, &req).is_err());
            assert!(compile_base(variant, &spec, &req).is_err());
            assert!(compile_ip_adapter(variant, &spec, &req).is_err());
        }
    }
}
