//! Native MOSS-Image Base/SFT, including both explicit published DMD precision policies.
//! Caller-provisioned exact assets only. No acquisition or Python production execution.
#[macro_use]
#[path = "../../../../llm/test-support/model_proof.rs"]
mod model_proof;
const _: fn() = || {
    let _: std::result::Result<candle_gen::candle_core::Tensor, ()> =
        __model_proof_output!(std::result::Result::<candle_gen::candle_core::Tensor, ()>::Err(()));
};

mod edit;
mod loader;
pub mod memory_strategy;
mod pipeline;
mod release;
mod text_encoder;
mod transformer;

/// The exact canvas grid of the frozen packed-latent model.
pub const SIZE_MULTIPLE: u32 = 16;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{self, moss_image::*, moss_image_edit::*, *};
use candle_gen_flux2::vae::Flux2Vae;
use candle_llm::models::qwen35::Qwen35Model;
use loader::Sources;
use release::release_then_sync;
use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Default)]
struct State {
    text: Option<Qwen35Model>,
    renderer: Option<pipeline::Renderer>,
    vae: Option<Flux2Vae>,
    report: Option<MossImageExecutionReport>,
    edit_report: Option<MossImageEditExecutionReport>,
}
impl State {
    fn release_for_request(
        &mut self,
        mode: MossImageAcceleration,
        edit: bool,
        synchronize: impl FnOnce() -> candle_gen::Result<()>,
    ) -> candle_gen::Result<bool> {
        let release = self
            .renderer
            .as_ref()
            .is_some_and(|renderer| edit || renderer.mode != mode);
        if release {
            release_then_sync(self.renderer.take(), synchronize)?;
        }
        Ok(release)
    }
    fn select_renderer(
        &mut self,
        mode: MossImageAcceleration,
        synchronize: impl FnOnce() -> candle_gen::Result<()>,
        load: impl FnOnce() -> candle_gen::Result<pipeline::Renderer>,
    ) -> candle_gen::Result<()> {
        if self.renderer.as_ref().is_none_or(|r| r.mode != mode) {
            release_then_sync(self.renderer.take(), synchronize)?;
            self.renderer = Some(load()?);
        }
        Ok(())
    }
}

/// A serialized generator retaining immutable mode-specific components. Switching precision drops
/// the previous renderer before constructing the next; the ordinary backbone is never modified.
pub struct MossImageGenerator {
    descriptor: ModelDescriptor,
    sources: Sources,
    device: Device,
    state: Mutex<State>,
    contract: MemoryProviderContract,
    admission: Mutex<Option<MemoryRunContext>>,
}
impl MossImageGenerator {
    /// Construct on the caller's admitted device. CUDA logical-to-physical ownership belongs to
    /// the process allocator; no physical GPU index is inferred inside this provider.
    pub fn load_on_device(
        variant: MossImageVariant,
        spec: &LoadSpec,
        device: Device,
    ) -> gen_core::Result<Self> {
        let sources = Sources::new(variant, spec)?;
        let contract = memory_strategy::contract_sources(variant, &sources)?;
        Ok(Self {
            descriptor: descriptor(variant),
            sources,
            device,
            state: Mutex::new(State::default()),
            contract,
            admission: Mutex::new(None),
        })
    }
    /// Structural estimate for the actual expanded advanced batch and explicit mode.
    pub fn estimated_peak_bytes(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
    ) -> gen_core::Result<u64> {
        let r = MossImageResolved::resolve(self.variant(), request)?;
        let images = inputs.validate(request, &r)?;
        memory_strategy::estimate(
            self.selected_sources(r.acceleration, &request.cancel)?
                .as_ref(),
            &r,
            request,
            images,
            self.staged(request),
        )
    }
    /// Incremental structural demand above the caller's committed budget snapshot. Only this
    /// generator's actual retained tensor weights receive credit; unrelated memory never does.
    pub fn estimated_incremental_peak_bytes(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        budget: MemoryBudget,
    ) -> gen_core::Result<u64> {
        let state = candle_gen::lock_recover(&self.state);
        let peak = self.estimated_peak_bytes(request, inputs)?;
        Ok(peak.saturating_sub(
            self.cached_resident_bytes(&state)?
                .min(budget.committed_bytes),
        ))
    }
    fn cached_resident_bytes(&self, state: &State) -> gen_core::Result<u64> {
        let text = state
            .text
            .as_ref()
            .map_or(0, |t| t.weight_census().total().resident_bytes);
        let renderer = state.renderer.as_ref().map_or(0, |r| {
            r.transformer.resident_bytes()
                + r.dataspace.as_ref().map_or(0, |(q, _)| {
                    (q.elem_count() * q.dtype().size_in_bytes()) as u64
                })
        });
        let mut vae = 0u64;
        if state.vae.is_some() {
            for asset in self.sources.ordinary.iter().filter(|a| {
                a.role == MossImageAssetRole::Vae && a.relative.ends_with(".safetensors")
            }) {
                for h in self
                    .sources
                    .headers
                    .get(asset.pin.content_sha256())
                    .ok_or_else(|| {
                        gen_core::Error::Unsupported(
                            "MOSS missing authenticated VAE residency facts".into(),
                        )
                    })?
                {
                    if h.name.starts_with("decoder.")
                        || h.name.starts_with("post_quant_conv.")
                        || h.name.starts_with("bn.running_")
                    {
                        vae = vae.saturating_add(h.element_count()?.saturating_mul(4));
                    }
                }
            }
        }
        Ok(text.saturating_add(renderer).saturating_add(vae))
    }
    fn selected_sources(
        &self,
        mode: MossImageAcceleration,
        cancel: &CancelFlag,
    ) -> gen_core::Result<std::borrow::Cow<'_, Sources>> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        if self.sources.selection == Some(mode) {
            Ok(std::borrow::Cow::Borrowed(&self.sources))
        } else {
            Sources::selected_with_cancel(self.variant(), &self.sources.spec, mode, cancel)
                .map(std::borrow::Cow::Owned)
        }
    }
    fn variant(&self) -> MossImageVariant {
        MossImageVariant::from_provider_id(self.descriptor.id).expect("static descriptor")
    }
    fn staged(&self, r: &GenerationRequest) -> bool {
        r.memory.map_or(
            self.sources.spec.offload_policy == OffloadPolicy::Sequential,
            |m| m.stage_residency,
        )
    }
    fn run(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<MossImageAdvancedOutput> {
        self.run_with_edit(request, inputs, None, on_progress)
    }
    fn run_with_edit(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        edit: Option<MossImageEditInputs<'_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<MossImageAdvancedOutput> {
        let mut state = candle_gen::lock_recover(&self.state);
        state.report = None;
        state.edit_report = None;
        let admission = candle_gen::lock_recover(&self.admission).take();
        let result = (|| {
            let normalized;
            let request = if let Some(edit) = edit {
                edit.validate()?;
                edit::check_precision(&self.device)?;
                normalized = edit_request(request)?;
                &normalized
            } else {
                request
            };
            self.run_locked(&mut state, request, inputs, edit, on_progress, admission)
        })();
        // Edit renderers and reference K/V never survive a request. CUDA drops enqueue frees:
        // synchronize after release so the next admission sees the recovered driver budget.
        if result.is_err() || edit.is_some() {
            let sync = release_then_sync(
                (state.text.take(), state.renderer.take(), state.vae.take()),
                || self.device.synchronize(),
            );
            if result.is_err() || sync.is_err() {
                state.report = None;
                state.edit_report = None;
            }
            sync.map_err(candle_gen::CandleError::from)?;
        }
        result
    }
    fn run_locked(
        &self,
        state: &mut State,
        req: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        edit: Option<MossImageEditInputs<'_>>,
        on_progress: &mut dyn FnMut(Progress),
        admission: Option<MemoryRunContext>,
    ) -> gen_core::Result<MossImageAdvancedOutput> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        let r = MossImageResolved::resolve(self.variant(), req)?;
        let images = inputs.validate(req, &r)?;
        checked_host_elements(req, images)?;
        memory_strategy::check_request(req)?;
        transformer::check_cancel(&req.cancel)?;
        let sources = self.selected_sources(r.acceleration, &req.cancel)?;
        let assets = sources.assets(r.acceleration)?;
        let staged = self.staged(req);
        let edit_assets = edit
            .map(|_| edit::EditAssets::prepare(&self.sources.spec))
            .transpose()?;
        let peak = memory_strategy::estimate(&sources, &r, req, images, staged)?
            .checked_add(
                edit.map(|e| edit::extra_peak(e, req))
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or_else(|| gen_core::Error::Unsupported("MOSS edit peak overflow".into()))?;
        if let Some(context) = admission {
            let demand = peak.saturating_sub(
                self.cached_resident_bytes(state)?
                    .min(context.budget.committed_bytes),
            );
            if context.geometry.width != req.width
                || context.geometry.height != req.height
                || context.geometry.batch as usize != images
                || context.geometry.reference_count as usize
                    != edit.map_or(0, |e| e.references.len())
                || !context.budget.fits(demand)
            {
                return Err(gen_core::Error::Unsupported(format!("MOSS actual expanded request needs structural upper bound {peak} bytes; admitted geometry/budget does not cover it")));
            }
        }
        // Release a differently keyed renderer before conditioning, so its old residency cannot
        // overlap the new route's conditioning workspace after warm-budget credit is applied.
        state.release_for_request(r.acceleration, edit.is_some(), || {
            self.device
                .synchronize()
                .map_err(candle_gen::CandleError::from)
        })?;
        if staged {
            release_then_sync(
                (state.text.take(), state.renderer.take(), state.vae.take()),
                || self.device.synchronize(),
            )
            .map_err(candle_gen::CandleError::from)?;
        }
        let prompts = if req.prompt_batch.is_empty() {
            vec![req.prompt.clone()]
        } else {
            req.prompt_batch.clone()
        };
        let need_text = inputs.positive.is_none()
            || (matches!(r.acceleration, MossImageAcceleration::Off)
                && r.guidance != 1.0
                && inputs.negative.is_none());
        if need_text && state.text.is_none() {
            if staged {
                on_progress(Progress::Loading(LoadPhase::TextEncoder));
            }
            transformer::check_cancel(&req.cancel)?;
            state.text = Some(self.sources.text(&self.device)?);
        }
        let positive = if let Some(e) = inputs.positive {
            text_encoder::supplied(e, &self.device)?
        } else {
            text_encoder::strings(
                &self.sources,
                state.text.as_ref().expect("text loaded"),
                &prompts,
                r.max_sequence_length,
                &self.device,
                &req.cancel,
            )?
        };
        let negative = if matches!(r.acceleration, MossImageAcceleration::Off) && r.guidance != 1.0
        {
            Some(if let Some(e) = inputs.negative {
                text_encoder::supplied(e, &self.device)?
            } else {
                let texts = inputs
                    .negative_prompts
                    .map(<[String]>::to_vec)
                    .unwrap_or_else(|| {
                        vec![req.negative_prompt.clone().unwrap_or_default(); positive.rows.len()]
                    });
                text_encoder::strings(
                    &self.sources,
                    state.text.as_ref().expect("text loaded"),
                    &texts,
                    r.max_sequence_length,
                    &self.device,
                    &req.cancel,
                )?
            })
        } else {
            None
        };
        transformer::check_cancel(&req.cancel)?;
        if staged {
            release_then_sync(state.text.take(), || self.device.synchronize())
                .map_err(candle_gen::CandleError::from)?;
        }
        let edit_references = edit
            .map(|e| edit::encode_references(&sources, e, &self.device, &req.cancel))
            .transpose()?;
        if edit.is_some() {
            on_progress(Progress::ConditioningComplete);
        }
        if state
            .renderer
            .as_ref()
            .is_none_or(|p| p.mode != r.acceleration)
        {
            if staged {
                on_progress(Progress::Loading(LoadPhase::Renderer));
            }
            transformer::check_cancel(&req.cancel)?;
            state.select_renderer(
                r.acceleration,
                || Ok(self.device.synchronize()?),
                || pipeline::Renderer::load(&sources, r.acceleration, &self.device),
            )?;
        }
        if let Some(assets) = &edit_assets {
            assets.install(
                &mut state
                    .renderer
                    .as_mut()
                    .expect("renderer loaded")
                    .transformer,
                &req.cancel,
            )?;
        }
        let sampled = if let (Some(edit), Some((references, _))) = (edit, edit_references.as_ref())
        {
            pipeline::sample_edit(
                state.renderer.as_ref().expect("renderer loaded"),
                req,
                &r,
                edit,
                (&positive, negative.as_ref()),
                references,
                on_progress,
            )?
        } else {
            pipeline::sample(
                state.renderer.as_ref().expect("renderer loaded"),
                req,
                &r,
                inputs,
                &positive,
                negative.as_ref(),
                on_progress,
            )?
        };
        let adapter_reports = state
            .renderer
            .as_ref()
            .expect("renderer loaded")
            .adapter_reports
            .clone();
        if staged {
            release_then_sync(state.renderer.take(), || self.device.synchronize())
                .map_err(candle_gen::CandleError::from)?;
        }
        transformer::check_cancel(&req.cancel)?;
        let output = match inputs.output {
            MossImageOutputKind::Latents => MossImageAdvancedOutput::Latents {
                values: loader::tensor_f32(&sampled.latents)?,
                shape: [
                    images,
                    128,
                    req.height as usize / 16,
                    req.width as usize / 16,
                ],
            },
            MossImageOutputKind::Images => {
                if state.vae.is_none() {
                    if staged {
                        on_progress(Progress::Loading(LoadPhase::Renderer));
                    }
                    transformer::check_cancel(&req.cancel)?;
                    state.vae = Some(self.sources.vae(&self.device)?);
                }
                MossImageAdvancedOutput::Images(pipeline::decode(
                    state.vae.as_ref().expect("VAE loaded"),
                    &sampled.latents,
                    req,
                    on_progress,
                )?)
            }
        };
        if staged {
            state.vae = None;
        }
        self.device
            .synchronize()
            .map_err(candle_gen::CandleError::from)?;
        transformer::check_cancel(&req.cancel)?;
        let conditioning = positive
            .receipts
            .into_iter()
            .enumerate()
            .map(|(i, positive)| MossImageConditioningReceipt {
                positive,
                negative: negative.as_ref().map(|n| n.receipts[i].clone()),
            })
            .collect();
        let report = MossImageExecutionReport::new(MossImageExecutionData {
            execution_device: if self.device.is_cpu() {
                MossImageExecutionDevice::Cpu
            } else if self.device.is_cuda() {
                MossImageExecutionDevice::Cuda
            } else {
                MossImageExecutionDevice::Metal
            },
            resolved: r.clone(),
            assets: pipeline::receipts(&assets)?,
            sampling: sampled.sampling,
            samples: sampled.samples,
            conditioning,
            images_per_prompt: req.count,
            latent_shape: [
                images,
                128,
                req.height as usize / 16,
                req.width as usize / 16,
            ],
            output_kind: inputs.output,
            component_dtypes: component_dtypes(r.acceleration, self.device.is_cpu()),
            adapter_reports,
        })?;
        transformer::check_cancel(&req.cancel)?;
        if let (Some(edit), Some(assets), Some((_, references))) =
            (edit, edit_assets, edit_references)
        {
            assets.ensure_unchanged()?;
            let receipt = MossImageEditExecutionReport {
                base: report,
                adapter_path: assets.path().to_path_buf(),
                edit_contract: EDIT_CONTRACT.into(),
                adapter_repository: EDIT_REPOSITORY.into(),
                adapter_revision: EDIT_REVISION.into(),
                adapter_sha256: EDIT_WEIGHTS_SHA256.into(),
                config_sha256: EDIT_CONFIG_SHA256.into(),
                applied_projections: 252,
                references,
                reference_max_edge: edit.reference_max_edge,
                use_kv_cache: edit.use_kv_cache,
            };
            receipt.validate()?;
            transformer::check_cancel(&req.cancel)?;
            state.edit_report = Some(receipt);
        } else {
            state.report = Some(report);
        }
        Ok(output)
    }
}
impl MossImageAdvancedGenerator for MossImageGenerator {
    fn generate_moss_advanced(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<MossImageAdvancedOutput> {
        self.run(request, inputs, on_progress)
    }
}
impl Generator for MossImageGenerator {
    fn generate_moss_image_edit(
        &self,
        request: &GenerationRequest,
        inputs: MossImageEditInputs<'_>,
        progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<MossImageAdvancedOutput> {
        self.run_with_edit(request, inputs.advanced, Some(inputs), progress)
    }
    fn moss_image_edit_request_peak(
        &self,
        request: &GenerationRequest,
        inputs: MossImageEditInputs<'_>,
    ) -> gen_core::Result<u64> {
        memory_strategy::edit_request_peak(self.variant(), &self.sources.spec, request, inputs)
    }
    fn moss_image_edit_execution_report(&self) -> Option<MossImageEditExecutionReport> {
        candle_gen::lock_recover(&self.state).edit_report.clone()
    }
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
    fn moss_image_execution_report(&self) -> Option<MossImageExecutionReport> {
        candle_gen::lock_recover(&self.state).report.clone()
    }
    fn moss_image_owned_weight_bytes(&self) -> Option<u64> {
        self.state
            .lock()
            .ok()
            .and_then(|state| self.cached_resident_bytes(&state).ok())
    }
    fn adapter_apply_reports(&self) -> Vec<AdapterApplyReport> {
        if let Some(edit) = self.moss_image_edit_execution_report() {
            return vec![AdapterApplyReport {
                adapter_path: edit.adapter_path,
                applied: edit.applied_projections,
                skipped: vec![],
            }];
        }
        self.moss_image_execution_report()
            .map_or_else(Vec::new, |r| r.data().adapter_reports.clone())
    }
    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        Some(&self.contract)
    }
    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        let decision = memory_strategy::safety_check(&self.sources.spec, &self.contract, context);
        *candle_gen::lock_recover(&self.admission) = if decision == MemorySafetyDecision::Accept {
            Some(context.clone())
        } else {
            None
        };
        decision
    }
    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        let r = MossImageResolved::resolve(self.variant(), req)?;
        MossImageAdvancedInputs::default().validate(req, &r)?;
        checked_host_elements(
            req,
            req.prompt_batch
                .len()
                .max(1)
                .checked_mul(req.count as usize)
                .ok_or_else(|| {
                    gen_core::Error::Unsupported("MOSS expanded batch overflow".into())
                })?,
        )?;
        memory_strategy::check_request(req)?;
        self.selected_sources(r.acceleration, &req.cancel)?
            .assets(r.acceleration)?;
        Ok(())
    }
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        match self.run(req, MossImageAdvancedInputs::default(), on_progress)? {
            MossImageAdvancedOutput::Images(images) => Ok(GenerationOutput::Images(images)),
            MossImageAdvancedOutput::Latents { .. } => unreachable!("default image output"),
        }
    }
}

fn checked_host_elements(req: &GenerationRequest, images: usize) -> gen_core::Result<usize> {
    images
        .checked_mul(128)
        .and_then(|n| n.checked_mul(req.width as usize / 16))
        .and_then(|n| n.checked_mul(req.height as usize / 16))
        .filter(|n| *n <= isize::MAX as usize / 4)
        .ok_or_else(|| {
            gen_core::Error::Unsupported(
                "MOSS expanded FP32 latent/noise allocation overflows host address space".into(),
            )
        })
}
fn component_dtypes(mode: MossImageAcceleration, cpu: bool) -> BTreeMap<String, MossImageDtype> {
    let mut d: BTreeMap<_, _> = [
        ("transformer", MossImageDtype::Bf16),
        (
            "text_encoder",
            if cpu {
                MossImageDtype::F32
            } else {
                MossImageDtype::Bf16
            },
        ),
        ("vae", MossImageDtype::F32),
        ("latents", MossImageDtype::F32),
        ("time_embedding", MossImageDtype::Bf16),
        ("shared_block_modulation", MossImageDtype::Bf16),
        ("video_head_modulation", MossImageDtype::Bf16),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v))
    .collect();
    if let MossImageAcceleration::Dmd(p) = mode {
        for k in [
            "time_embedding",
            "shared_block_modulation",
            "video_head_modulation",
        ] {
            d.insert(k.into(), MossImageDtype::F32);
        }
        if p == MossDmdPrecision::FullFp32 {
            d.insert("transformer".into(), MossImageDtype::F32);
        } else {
            d.insert("lora_residual".into(), MossImageDtype::F32);
        }
    }
    d
}
fn descriptor(v: MossImageVariant) -> ModelDescriptor {
    ModelDescriptor {
        id: v.provider_id(),
        family: "moss_image",
        backend: "candle",
        modality: Modality::Image,
        encoder_contract: None,
        denoiser_output_latent_space: Some(&gen_core::FLUX2_PACKED_LATENT_SPACE),
        control_kinds: None,
        required_components: if v == MossImageVariant::Base {
            &[MOSS_SHARED_SNAPSHOT_COMPONENT]
        } else {
            &[]
        },
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_prompt_batch: true,
            supports_sequential_offload: true,
            samplers: vec!["euler"],
            schedulers: vec!["moss_training_grid"],
            supported_guidance_methods: vec!["cfg"],
            min_size: SIZE_MULTIPLE,
            max_size: u32::MAX,
            max_count: u32::MAX,
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
            ..Default::default()
        },
    }
}
/// Frozen Base descriptor.
pub fn descriptor_base() -> ModelDescriptor {
    descriptor(MossImageVariant::Base)
}
/// Frozen SFT descriptor. Eight-step execution remains explicitly opt-in per request.
pub fn descriptor_sft() -> ModelDescriptor {
    descriptor(MossImageVariant::Sft)
}
/// Registry Base loader using the process allocator's admitted logical device.
pub fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(MossImageGenerator::load_on_device(
        MossImageVariant::Base,
        spec,
        candle_gen::default_device()?,
    )?))
}
/// Registry SFT loader using the process allocator's admitted logical device.
pub fn load_sft(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(MossImageGenerator::load_on_device(
        MossImageVariant::Sft,
        spec,
        candle_gen::default_device()?,
    )?))
}
fn footprint_base(spec: &LoadSpec) -> gen_core::Result<PerComponentBytes> {
    memory_strategy::footprint_sources(&Sources::new(MossImageVariant::Base, spec)?)
}
fn footprint_sft(spec: &LoadSpec) -> gen_core::Result<PerComponentBytes> {
    memory_strategy::footprint_sources(&Sources::new(MossImageVariant::Sft, spec)?)
}
fn contract_base(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    memory_strategy::contract_sources(
        MossImageVariant::Base,
        &Sources::new(MossImageVariant::Base, spec)?,
    )
}
fn contract_sft(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    memory_strategy::contract_sources(
        MossImageVariant::Sft,
        &Sources::new(MossImageVariant::Sft, spec)?,
    )
}
candle_gen::register_generators! {pub(crate) const BASE_REGISTRATION=descriptor_base=>load_base;footprint=footprint_base}
candle_gen::register_generators! {pub(crate) const SFT_REGISTRATION=descriptor_sft=>load_sft;footprint=footprint_sft}
/// Compose both frozen generator, footprint and memory registrations.
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    registry
        .register_generator(BASE_REGISTRATION)
        .register_generator(SFT_REGISTRATION)
        .register_memory_strategy(MemoryRegistration {
            provider_id: "moss_image_base",
            contract: contract_base,
            safety_check: memory_strategy::safety_check,
        })
        .register_memory_strategy(MemoryRegistration {
            provider_id: "moss_image_sft",
            contract: contract_sft,
            safety_check: memory_strategy::safety_check,
        })
        .register_resident_only_memory_contract(ResidentOnlyMemoryContractRegistration {
            provider_id: "moss_image_base",
            contract: witness_base,
            surface_specs: witness_specs,
        })
        .register_resident_only_memory_contract(ResidentOnlyMemoryContractRegistration {
            provider_id: "moss_image_sft",
            contract: witness_sft,
            surface_specs: witness_specs,
        })
}

fn witness_specs() -> Vec<MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|s| s.selector.tier == MemoryContractSurfaceTier::Bf16)
        .collect()
}
fn witness(variant: MossImageVariant, spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    if spec.quantize.is_some() || spec.precision != Precision::Bf16 {
        return Err(gen_core::Error::Unsupported("MOSS has no generic quantization/precision override; full_fp32 is an explicit published DMD request".into()));
    }
    let mut c = MemoryProviderContract::compatibility_default(
        variant.provider_id(),
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    c.load_shape = spec.load_shape;
    Ok(c)
}
fn witness_base(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    witness(MossImageVariant::Base, spec)
}
fn witness_sft(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    witness(MossImageVariant::Sft, spec)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
fn verify_numerical_renderer_transitions(
    device: &Device,
    mut build: impl FnMut(MossImageAcceleration) -> transformer::Transformer,
    mut predict: impl FnMut(&transformer::Transformer) -> Vec<f32>,
    mut edit: impl FnMut() -> Vec<f32>,
) {
    let mut state = State::default();
    let expected_edit = edit();
    let mut ordinary = None;
    for mode in [
        MossImageAcceleration::Off,
        MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
        MossImageAcceleration::Off,
    ] {
        state
            .release_for_request(mode, false, || Ok(device.synchronize()?))
            .unwrap();
        state
            .select_renderer(
                mode,
                || Ok(device.synchronize()?),
                || {
                    Ok(pipeline::Renderer {
                        mode,
                        transformer: build(mode),
                        dataspace: None,
                        adapter_reports: vec![],
                    })
                },
            )
            .unwrap();
        let actual = predict(&state.renderer.as_ref().unwrap().transformer);
        assert_eq!(actual, predict(&build(mode)));
        if mode == MossImageAcceleration::Off {
            if let Some(baseline) = &ordinary {
                assert_eq!(&actual, baseline);
            } else {
                ordinary = Some(actual);
            }
        } else {
            assert_ne!(&actual, ordinary.as_ref().unwrap());
        }
        assert!(state
            .release_for_request(MossImageAcceleration::Off, true, || Ok(
                device.synchronize()?
            ))
            .unwrap());
        assert!(state.renderer.is_none());
        assert_eq!(edit(), expected_edit);
        assert!(
            state.renderer.is_none(),
            "request-local edit must not become ordinary residency"
        );
    }
}
